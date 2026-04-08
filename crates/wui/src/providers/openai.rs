// ============================================================================
// OpenAI Provider
//
// Implements the Provider trait against OpenAI's Chat Completions API.
// Supports: streaming, tool use (function calling).
//
// Compatible with: gpt-4o, gpt-4o-mini, gpt-4-turbo, and any
// OpenAI-compatible endpoint (Azure OpenAI, Together, Groq, etc.).
//
// Note: o-series reasoning models (o1, o3, o4) use different parameters
// (`max_completion_tokens`, `reasoning_effort`) and are not covered by this
// provider out of the box. For those, use `with_default_model` and configure
// the request body via a custom Provider implementation.
//
// ── SSE parser design ─────────────────────────────────────────────────────────
//
// OpenAI streams tool calls differently from Anthropic: the tool call `id`
// and `name` appear only on the *first* delta chunk for a given index;
// subsequent chunks carry only argument fragments. We track
// `index → (id, name)` to attribute argument deltas correctly.
//
// When `finish_reason: "tool_calls"` arrives, we emit `ToolUseEnd` for all
// in-flight tool calls (in index order), then wait for the usage chunk to
// emit `MessageEnd`. Usage is guaranteed when `stream_options.include_usage`
// is set — which this provider always requests.
//
// `parse_all` returns `Vec<StreamEvent>` so a single SSE chunk can produce
// multiple output events (e.g. one `ToolUseEnd` per concurrent tool call).
// The provider uses `flat_map` instead of `map` to expand them into the stream.
// ============================================================================

use std::collections::HashMap;
use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use futures::{Stream, StreamExt as _};
use serde_json::{json, Value};

use wui_core::event::{StopReason, StreamEvent, TokenUsage};
use wui_core::message::{ContentBlock, ImageSource, Message, Role};
use wui_core::provider::{ChatRequest, Provider, ProviderError, ToolDef};
use wui_core::types::ToolCallId;

// ── OpenAi ────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct OpenAi {
    client: reqwest::Client,
    api_key: String,
    api_url: String,
    default_model: String,
}

impl OpenAi {
    /// Create a provider using the official OpenAI API endpoint.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, "https://api.openai.com")
    }

    /// Create a provider with a custom base URL (proxy, Azure, local server, etc.).
    ///
    /// The chat completions path is appended automatically:
    /// - `"https://my-proxy.example.com"` → `".../v1/chat/completions"`
    /// - `"https://my-proxy.example.com/v1"` → `".../v1/chat/completions"`
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("failed to build reqwest client"),
            api_key: api_key.into(),
            api_url: chat_completions_url(&base_url.into()),
            default_model: "gpt-4o".to_string(),
        }
    }

    /// Override the provider's default model.
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }
}

fn chat_completions_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with("/chat/completions") {
        return base.to_string();
    }
    if base.ends_with("/v1") {
        format!("{base}/chat/completions")
    } else {
        format!("{base}/v1/chat/completions")
    }
}

#[async_trait]
impl Provider for OpenAi {
    async fn stream(
        &self,
        req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        let body = build_request_body(&req, &self.default_model);

        let response = self
            .client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Stream(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            if status.as_u16() == 429 {
                return Err(ProviderError::RateLimit {
                    retry_after_ms: 5_000,
                });
            }
            if status.as_u16() == 401 {
                return Err(ProviderError::Auth(text));
            }
            return Err(ProviderError::ServerError {
                status: status.as_u16(),
                message: text,
            });
        }

        let mut parser = SseParser::default();

        use eventsource_stream::Eventsource as _;
        let stream = response
            .bytes_stream()
            .eventsource()
            .flat_map(move |result| {
                // One SSE chunk may produce zero, one, or many StreamEvents.
                // `flat_map` expands the Vec into individual stream items.
                let events: Vec<Result<StreamEvent, ProviderError>> = match result {
                    Ok(event) if !event.data.is_empty() && event.data != "[DONE]" => {
                        match parser.parse_all(&event.data) {
                            Ok(evs) => evs.into_iter().map(Ok).collect(),
                            Err(e) => vec![Err(e)],
                        }
                    }
                    Ok(_) => vec![],
                    Err(e) => vec![Err(ProviderError::Stream(e.to_string()))],
                };
                futures::stream::iter(events)
            });

        Ok(Box::pin(stream))
    }
}

// ── Stateful SSE parser ───────────────────────────────────────────────────────

/// Parses OpenAI SSE events for one stream lifetime.
///
/// Tracks `index → (id, name)` so argument delta chunks can be attributed to
/// the correct tool call even without repeating the id on every chunk.
///
/// Holds the stop reason between the `finish_reason` chunk and the usage chunk
/// so that `MessageEnd` is emitted with real token counts, not zeros.
#[derive(Default)]
struct SseParser {
    /// Maps tool call index to (id, name). Index is present on every delta;
    /// id and name only appear on the first chunk for that index.
    tool_calls: HashMap<u32, (String, String)>,
    /// Stop reason from the last `finish_reason` field. Held until the usage
    /// chunk arrives, since usage is reported in a separate SSE event.
    pending_stop: Option<StopReason>,
}

impl SseParser {
    fn parse_all(&mut self, data: &str) -> Result<Vec<StreamEvent>, ProviderError> {
        let v: Value = serde_json::from_str(data)
            .map_err(|e| ProviderError::Stream(format!("json parse: {e}")))?;

        let mut events = Vec::new();

        // ── Choice-level events ────────────────────────────────────────────────
        if let Some(choice) = v["choices"].as_array().and_then(|c| c.first()) {
            let delta = &choice["delta"];

            // Text content delta.
            if let Some(text) = delta["content"].as_str() {
                if !text.is_empty() {
                    events.push(StreamEvent::TextDelta {
                        text: text.to_string(),
                    });
                }
            }

            // Tool call deltas — attributed by index.
            if let Some(tcs) = delta["tool_calls"].as_array() {
                for tc in tcs {
                    let index = tc["index"].as_u64().unwrap_or(0) as u32;

                    // First chunk for this index carries id and function.name.
                    if let Some(id) = tc["id"].as_str() {
                        let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                        self.tool_calls
                            .insert(index, (id.to_string(), name.clone()));
                        events.push(StreamEvent::ToolUseStart {
                            id: ToolCallId::from(id),
                            name,
                        });
                    }

                    // Subsequent chunks carry argument fragments.
                    if let Some(chunk) = tc["function"]["arguments"].as_str() {
                        if !chunk.is_empty() {
                            if let Some((id, _)) = self.tool_calls.get(&index) {
                                events.push(StreamEvent::ToolInputDelta {
                                    id: ToolCallId::from(id.as_str()),
                                    chunk: chunk.to_string(),
                                });
                            }
                        }
                    }
                }
            }

            // Finish reason — note the stop kind, close out any open tool calls.
            match choice["finish_reason"].as_str() {
                Some("tool_calls") => {
                    self.pending_stop = Some(StopReason::ToolUse);
                    // Emit ToolUseEnd for all in-flight tool calls in index order.
                    // Index order matches the ToolUse blocks the LLM produced,
                    // which is the order the engine expects results in.
                    let mut indices: Vec<u32> = self.tool_calls.keys().cloned().collect();
                    indices.sort_unstable();
                    for idx in indices {
                        if let Some((id, _)) = self.tool_calls.remove(&idx) {
                            events.push(StreamEvent::ToolUseEnd { id: id.into() });
                        }
                    }
                }
                Some("stop") => self.pending_stop = Some(StopReason::EndTurn),
                Some("length") => self.pending_stop = Some(StopReason::MaxTokens),
                _ => {}
            }
        }

        // ── Usage chunk ────────────────────────────────────────────────────────
        //
        // Emitted as a separate SSE event (choices: []) when
        // `stream_options.include_usage` is set. `prompt_tokens` being present
        // and non-null is the reliable sentinel that this is the real usage chunk.
        if v["usage"]["prompt_tokens"].as_u64().is_some() {
            let input_tokens = v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
            let output_tokens = v["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32;
            let stop_reason = self.pending_stop.take().unwrap_or(StopReason::EndTurn);
            events.push(StreamEvent::MessageEnd {
                usage: TokenUsage {
                    input_tokens,
                    output_tokens,
                    ..Default::default()
                },
                stop_reason,
            });
        }

        Ok(events)
    }
}

// ── Request building ──────────────────────────────────────────────────────────

fn build_request_body(req: &ChatRequest, default_model: &str) -> Value {
    let mut body = json!({
        "model":          req.model.clone().unwrap_or_else(|| default_model.to_string()),
        "max_tokens":     req.max_tokens,
        "stream":         true,
        "stream_options": { "include_usage": true },
        "messages":       messages_to_json(&req.system, &req.messages),
        "tools":          tools_to_json(&req.tools),
    });

    if let Some(t) = req.temperature {
        body["temperature"] = json!(t);
    }

    body
}

/// Serialise tool definitions in OpenAI format.
///
/// OpenAI wraps each tool in `{"type":"function","function":{...}}` whereas
/// Anthropic uses the schema at the top level.
fn tools_to_json(tools: &[ToolDef]) -> Value {
    Value::Array(
        tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name":        t.name,
                        "description": t.description,
                        "parameters":  t.input_schema,
                    },
                })
            })
            .collect(),
    )
}

/// Serialise messages in OpenAI format.
///
/// Key differences from Anthropic:
///
/// - **System prompt** is the first message with `"role":"system"` (not a
///   separate top-level field).
/// - **Tool results** use `"role":"tool"` with `"tool_call_id"` — one message
///   per result. In wui's internal format they are content blocks inside a
///   `Role::User` message; we expand them here.
/// - **Tool call arguments** must be a JSON string, not an object.
/// - **Thinking blocks** are Anthropic-specific and are skipped entirely.
/// - **Framework messages** (`Role::System` in mid-conversation) are sent as
///   `"role":"user"` since OpenAI expects system prompts at the start only.
fn messages_to_json(system: &str, messages: &[Message]) -> Value {
    let mut out: Vec<Value> = Vec::new();

    if !system.is_empty() {
        out.push(json!({"role": "system", "content": system}));
    }

    for msg in messages {
        match msg.role {
            // Framework-injected mid-conversation messages (hook rejections,
            // permission decisions, compression summaries, reminders).
            // OpenAI has no mid-conversation system role, so surface as user.
            Role::System => {
                let text = collect_text(&msg.content);
                if !text.is_empty() {
                    out.push(json!({"role": "user", "content": text}));
                }
            }

            Role::User => {
                // Tool results and regular text/images may coexist in the same
                // wui message; OpenAI requires them in separate typed messages.
                let mut tool_results: Vec<(&str, &str)> = Vec::new();
                let mut content_items: Vec<Value> = Vec::new();
                let mut has_images = false;

                for block in &msg.content {
                    match block {
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            tool_results.push((tool_use_id.as_str(), content.as_str()));
                        }
                        ContentBlock::Image { .. } => {
                            has_images = true;
                            content_items.push(block_to_content_item(block));
                        }
                        _ => {
                            let t = block_to_text(block);
                            if !t.is_empty() {
                                content_items.push(json!({ "type": "text", "text": t }));
                            }
                        }
                    }
                }

                // One "tool" message per tool result, in submission order.
                for (tool_call_id, content) in tool_results {
                    out.push(json!({
                        "role":        "tool",
                        "tool_call_id": tool_call_id,
                        "content":     content,
                    }));
                }

                // Remaining content: use array form when images are present
                // (required by OpenAI), plain string otherwise (more compatible).
                if !content_items.is_empty() {
                    if has_images {
                        out.push(json!({"role": "user", "content": content_items}));
                    } else {
                        // Extract text from items and join as a plain string.
                        let combined: String = content_items
                            .iter()
                            .filter_map(|item| item["text"].as_str().map(str::to_string))
                            .collect::<Vec<_>>()
                            .join("\n\n");
                        if !combined.is_empty() {
                            out.push(json!({"role": "user", "content": combined}));
                        }
                    }
                }
            }

            Role::Assistant => {
                let mut text_parts: Vec<String> = Vec::new();
                let mut tool_calls: Vec<Value> = Vec::new();

                for block in &msg.content {
                    match block {
                        ContentBlock::Text { text } => {
                            text_parts.push(text.clone());
                        }
                        ContentBlock::ToolUse {
                            id, name, input, ..
                        } => {
                            // OpenAI expects arguments as a JSON-encoded string.
                            let arguments =
                                serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string());
                            tool_calls.push(json!({
                                "id":   id,
                                "type": "function",
                                "function": { "name": name, "arguments": arguments },
                            }));
                        }
                        // Thinking is Anthropic-specific — skip to avoid confusing
                        // OpenAI models with proprietary content format.
                        ContentBlock::Thinking { .. } => {}
                        // Compression markers → plain text so context survives replay.
                        _ => {
                            let t = block_to_text(block);
                            if !t.is_empty() {
                                text_parts.push(t);
                            }
                        }
                    }
                }

                let combined = text_parts.join("\n\n");

                // Skip assistant messages that would be empty (e.g. thinking-only
                // turns from a previous Anthropic session). Sending an empty
                // assistant message causes a 400 from the OpenAI API.
                if combined.is_empty() && tool_calls.is_empty() {
                    continue;
                }

                let mut obj = json!({"role": "assistant"});
                obj["content"] = if combined.is_empty() {
                    json!(null)
                } else {
                    json!(combined)
                };
                if !tool_calls.is_empty() {
                    obj["tool_calls"] = json!(tool_calls);
                }
                out.push(obj);
            }
        }
    }

    json!(out)
}

fn collect_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter(|b| !matches!(b, ContentBlock::Image { .. }))
        .map(block_to_text)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn block_to_text(block: &ContentBlock) -> String {
    match block {
        ContentBlock::Text { text } => text.clone(),
        ContentBlock::Thinking { text } => text.clone(),
        ContentBlock::ToolUse { name, input, .. } => format!("[Tool: {name}] {input}"),
        ContentBlock::ToolResult { content, .. } => content.clone(),
        ContentBlock::Collapsed { summary, .. } => summary.clone(),
        ContentBlock::CompactBoundary { summary } => summary.clone(),
        // Image and Document blocks have no natural text representation.
        ContentBlock::Image { .. } => "[image]".to_string(),
        ContentBlock::Document { title, .. } => {
            format!("[document: {}]", title.as_deref().unwrap_or("untitled"))
        }
    }
}

/// Convert a single content block to the OpenAI "content item" JSON format.
///
/// OpenAI's multi-turn API allows content to be either a plain string or an
/// array of content items (`{"type":"text","text":...}` /
/// `{"type":"image_url","image_url":{...}}`).
///
/// This function serialises image blocks to the image_url format; all other
/// blocks fall back to text via `block_to_text`.
fn block_to_content_item(block: &ContentBlock) -> Value {
    match block {
        ContentBlock::Image { source } => {
            let url = match source {
                ImageSource::Base64 { media_type, data } => {
                    format!("data:{media_type};base64,{data}")
                }
                ImageSource::Url(url) => url.clone(),
            };
            json!({ "type": "image_url", "image_url": { "url": url } })
        }
        other => json!({ "type": "text", "text": block_to_text(other) }),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_derivation() {
        assert_eq!(
            chat_completions_url("https://api.openai.com"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("https://api.openai.com/v1/chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("https://proxy.example.com/"),
            "https://proxy.example.com/v1/chat/completions"
        );
    }

    #[test]
    fn text_delta_parsed() {
        let mut p = SseParser::default();
        let data = r#"{"choices":[{"delta":{"content":"Hello"},"finish_reason":null,"index":0}]}"#;
        let events = p.parse_all(data).unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], StreamEvent::TextDelta { text } if text == "Hello"));
    }

    #[test]
    fn tool_call_start_and_delta() {
        let mut p = SseParser::default();

        // First chunk: id + name
        let start = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"search","arguments":""}}]},"finish_reason":null,"index":0}]}"#;
        let events = p.parse_all(start).unwrap();
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], StreamEvent::ToolUseStart { id, name } if id == "call_abc" && name == "search")
        );

        // Argument delta
        let delta = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]},"finish_reason":null,"index":0}]}"#;
        let events = p.parse_all(delta).unwrap();
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], StreamEvent::ToolInputDelta { id, chunk } if id == "call_abc" && chunk == "{\"q\":")
        );
    }

    #[test]
    fn finish_reason_tool_calls_emits_tool_use_end() {
        let mut p = SseParser::default();

        // Start two tool calls
        p.parse_all(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"id0","type":"function","function":{"name":"t0","arguments":""}}]},"finish_reason":null,"index":0}]}"#).unwrap();
        p.parse_all(r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"id1","type":"function","function":{"name":"t1","arguments":""}}]},"finish_reason":null,"index":0}]}"#).unwrap();

        // finish_reason: tool_calls
        let finish = r#"{"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}"#;
        let events = p.parse_all(finish).unwrap();

        // Should emit two ToolUseEnd events in index order
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::ToolUseEnd { id } if id == "id0"));
        assert!(matches!(&events[1], StreamEvent::ToolUseEnd { id } if id == "id1"));
    }

    #[test]
    fn usage_chunk_emits_message_end() {
        let mut p = SseParser {
            pending_stop: Some(StopReason::EndTurn),
            ..Default::default()
        };

        let usage = r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#;
        let events = p.parse_all(usage).unwrap();
        assert_eq!(events.len(), 1);
        if let StreamEvent::MessageEnd { usage, stop_reason } = &events[0] {
            assert_eq!(usage.input_tokens, 10);
            assert_eq!(usage.output_tokens, 5);
            assert_eq!(*stop_reason, StopReason::EndTurn);
        } else {
            panic!("expected MessageEnd");
        }
    }

    #[test]
    fn system_message_converted_to_user() {
        let msgs = vec![wui_core::message::Message::system(
            "reminder: do not delete",
        )];
        let v = messages_to_json("", &msgs);
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["role"], "user");
        assert_eq!(arr[0]["content"], "reminder: do not delete");
    }

    #[test]
    fn tool_results_expanded_to_separate_messages() {
        use wui_core::message::{ContentBlock, Message, Role};
        let msg = Message::with_id(
            "id1".to_string(),
            Role::User,
            vec![
                ContentBlock::ToolResult {
                    tool_use_id: "call_1".into(),
                    content: "result A".into(),
                    is_error: false,
                },
                ContentBlock::ToolResult {
                    tool_use_id: "call_2".into(),
                    content: "result B".into(),
                    is_error: false,
                },
            ],
        );
        let v = messages_to_json("", &[msg]);
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["role"], "tool");
        assert_eq!(arr[0]["tool_call_id"], "call_1");
        assert_eq!(arr[1]["role"], "tool");
        assert_eq!(arr[1]["tool_call_id"], "call_2");
    }

    #[test]
    fn thinking_only_assistant_message_skipped() {
        use wui_core::message::{ContentBlock, Message, Role};
        let msg = Message::with_id(
            "id1".to_string(),
            Role::Assistant,
            vec![ContentBlock::Thinking {
                text: "internal reasoning".into(),
            }],
        );
        let v = messages_to_json("", &[msg]);
        assert_eq!(
            v.as_array().unwrap().len(),
            0,
            "thinking-only message should be skipped"
        );
    }

    #[test]
    fn image_base64_serializes_as_image_url() {
        use wui_core::message::{ContentBlock, ImageSource, Message, Role};
        let msg = Message::with_id(
            "id1".to_string(),
            Role::User,
            vec![ContentBlock::Image {
                source: ImageSource::Base64 {
                    media_type: "image/png".into(),
                    data: "abc123".into(),
                },
            }],
        );
        let v = messages_to_json("", &[msg]);
        let arr = v.as_array().unwrap();
        assert_eq!(arr[0]["role"], "user");
        let content = arr[0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "image_url");
        let url = content[0]["image_url"]["url"].as_str().unwrap();
        assert!(
            url.starts_with("data:image/png;base64,"),
            "expected data URI, got {url}"
        );
        assert!(url.ends_with("abc123"));
    }

    #[test]
    fn tool_call_arguments_encoded_as_json_string() {
        use wui_core::message::{ContentBlock, Message, Role};
        let input = serde_json::json!({"query": "rust lang"});
        let msg = Message::with_id(
            "id1".to_string(),
            Role::Assistant,
            vec![ContentBlock::ToolUse {
                id: "call_abc".into(),
                name: "search".into(),
                input: input.clone(),
                summary: None,
            }],
        );
        let v = messages_to_json("", &[msg]);
        let arr = v.as_array().unwrap();
        assert_eq!(arr[0]["role"], "assistant");
        let tc = &arr[0]["tool_calls"].as_array().unwrap()[0];
        assert_eq!(tc["id"], "call_abc");
        assert_eq!(tc["type"], "function");
        // arguments must be a JSON string (not an object)
        let args_str = tc["function"]["arguments"].as_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(args_str).unwrap();
        assert_eq!(parsed, input);
    }
}
