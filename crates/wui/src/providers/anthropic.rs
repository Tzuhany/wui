// ============================================================================
// Anthropic Provider
//
// Implements the Provider trait against Anthropic's Messages API.
// Supports: streaming, prompt caching, extended thinking, tool use.
//
// ── SSE parser design ─────────────────────────────────────────────────────────
//
// The Anthropic streaming protocol assigns content blocks by `index`. A
// `content_block_start` event at index N tells us the block's type and id;
// subsequent `content_block_delta` events at index N carry partial content.
//
// The stateless design (`parse_sse_event` as a pure function) breaks when
// multiple tool calls are in flight simultaneously: we cannot tell which
// `tool_use_id` corresponds to a given delta without tracking the
// index → id mapping across events.
//
// Solution: `SseParser` is a small stateful struct that wraps the pure
// parsing logic and maintains this mapping for the lifetime of one stream.
// ============================================================================

use std::collections::HashMap;
use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde_json::{json, Value};

use wui_core::event::{StopReason, StreamEvent, TokenUsage};
use wui_core::message::{ContentBlock, DocumentSource, ImageSource, Message, Role};
use wui_core::provider::{ChatRequest, Provider, ProviderError, ToolDef};
use wui_core::types::ToolCallId;

const ANTHROPIC_VERSION: &str = "2023-06-01";

#[derive(Clone)]
pub struct Anthropic {
    client: reqwest::Client,
    api_key: String,
    api_url: String,
    default_model: String,
    beta_headers: Vec<String>,
    thinking_budget: Option<u32>,
    /// When true, `cache_control: {type: ephemeral}` markers are injected at
    /// the end of the system prompt and after the last tool definition so
    /// Anthropic's prompt-caching layer can reuse them across turns.
    cache_enabled: bool,
}

impl Anthropic {
    /// Create a provider using the official Anthropic API endpoint.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, "https://api.anthropic.com")
    }

    /// Create a provider with a custom base URL (proxy, local mock, etc.).
    ///
    /// The messages path (`/v1/messages`) is appended automatically:
    /// - `"https://my-proxy.example.com"` → `".../v1/messages"`
    /// - `"https://my-proxy.example.com/v1"` → `".../v1/messages"`
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("failed to build reqwest client"),
            api_key: api_key.into(),
            api_url: messages_url(&base_url.into()),
            default_model: "claude-opus-4-6".to_string(),
            beta_headers: Vec::new(),
            thinking_budget: None,
            cache_enabled: false,
        }
    }

    /// Override the provider's default model.
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// Enable one Anthropic beta header.
    pub fn with_beta(mut self, beta: impl Into<String>) -> Self {
        self.beta_headers.push(beta.into());
        self
    }

    /// Enable Anthropic prompt caching.
    ///
    /// Adds the required beta header **and** injects `cache_control` markers
    /// at the two highest-value breakpoints:
    ///
    /// 1. **System prompt** — constant across all turns; highest cache hit rate.
    /// 2. **Last tool definition** — constant while tools don't change.
    ///
    /// Cache hits reduce input token cost to ~10% (write: 1.25×, read: 0.1×).
    /// Enable this whenever you make repeated calls with the same system prompt
    /// and tool set.
    pub fn with_prompt_caching(mut self) -> Self {
        self.beta_headers
            .push("prompt-caching-2024-07-31".to_string());
        self.cache_enabled = true;
        self
    }

    /// Enable extended thinking with a budget in tokens.
    pub fn with_thinking_budget(mut self, budget_tokens: u32) -> Self {
        self.thinking_budget = Some(budget_tokens);
        self
    }
}

/// Derive the messages endpoint from a base URL.
fn messages_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with("/messages") {
        return base.to_string();
    }
    if base.ends_with("/v1") {
        format!("{base}/messages")
    } else {
        format!("{base}/v1/messages")
    }
}

#[async_trait]
impl Provider for Anthropic {
    async fn stream(
        &self,
        req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        // Request-level thinking_budget overrides the provider-level default.
        let thinking_budget = req.thinking_budget.or(self.thinking_budget);
        let body = build_request_body(
            &req,
            &self.default_model,
            thinking_budget,
            self.cache_enabled,
        );

        let mut request = self
            .client
            .post(&self.api_url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json");

        if !self.beta_headers.is_empty() {
            request = request.header("anthropic-beta", self.beta_headers.join(","));
        }

        let response = request
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

        // Stateful parser — one per stream, tracks content-block index → tool_use_id.
        let mut parser = SseParser::default();

        use eventsource_stream::Eventsource as _;
        let stream = response
            .bytes_stream()
            .eventsource()
            .map(move |result| match result {
                Ok(event) if !event.data.is_empty() => parser.parse(&event.event, &event.data),
                Ok(_) => Ok(None),
                Err(e) => Err(ProviderError::Stream(e.to_string())),
            })
            .filter_map(|r| async move {
                match r {
                    Ok(Some(event)) => Some(Ok(event)),
                    Ok(None) => None,
                    Err(e) => Some(Err(e)),
                }
            });

        Ok(Box::pin(stream))
    }
}

// ── Stateful SSE parser ───────────────────────────────────────────────────────

/// Parses Anthropic SSE events for one stream lifetime.
///
/// Maintains a map of content-block `index → tool_use_id` so that
/// `input_json_delta` events can be correctly attributed to their tool call,
/// even when multiple tool calls are in-flight simultaneously.
///
/// Also captures `input_tokens` from `message_start` — the only event where
/// Anthropic reports prompt token consumption.
#[derive(Default)]
struct SseParser {
    /// Maps content-block index to its tool_use_id.
    tool_index: HashMap<u64, String>,
    /// Input tokens reported in `message_start`, carried into `MessageEnd`.
    input_tokens: u32,
}

/// Extract a string from a JSON value, returning an empty string if absent.
fn jstr(v: &Value) -> String {
    v.as_str().unwrap_or("").to_string()
}

/// Extract a u32 from a JSON value, returning 0 if absent.
fn ju32(v: &Value) -> u32 {
    v.as_u64().unwrap_or(0) as u32
}

impl SseParser {
    fn parse(
        &mut self,
        event_type: &str,
        data: &str,
    ) -> Result<Option<StreamEvent>, ProviderError> {
        let v: Value = serde_json::from_str(data)
            .map_err(|e| ProviderError::Stream(format!("json parse: {e}")))?;

        match event_type {
            // Input tokens arrive in message_start, not message_delta.
            // Capture them here; they are merged into MessageEnd below.
            "message_start" => {
                self.input_tokens = ju32(&v["message"]["usage"]["input_tokens"]);
                Ok(None)
            }

            "content_block_start" => {
                let index = v["index"].as_u64().unwrap_or(0);
                let block_type = v["content_block"]["type"].as_str().unwrap_or("");
                let id = jstr(&v["content_block"]["id"]);
                let name = jstr(&v["content_block"]["name"]);

                if block_type == "tool_use" {
                    self.tool_index.insert(index, id.clone());
                    return Ok(Some(StreamEvent::ToolUseStart {
                        id: id.into(),
                        name,
                    }));
                }
                Ok(None)
            }

            "content_block_delta" => {
                let index = v["index"].as_u64().unwrap_or(0);
                match v["delta"]["type"].as_str().unwrap_or("") {
                    "text_delta" => Ok(Some(StreamEvent::TextDelta {
                        text: jstr(&v["delta"]["text"]),
                    })),
                    "thinking_delta" => Ok(Some(StreamEvent::ThinkingDelta {
                        text: jstr(&v["delta"]["thinking"]),
                    })),
                    "input_json_delta" => {
                        let id: ToolCallId = self
                            .tool_index
                            .get(&index)
                            .map(|id| ToolCallId::from(id.as_str()))
                            .unwrap_or_else(|| {
                                tracing::error!(
                                    index,
                                    "input_json_delta for unknown content-block index"
                                );
                                format!("unknown_tool_{index}").into()
                            });
                        Ok(Some(StreamEvent::ToolInputDelta {
                            id,
                            chunk: jstr(&v["delta"]["partial_json"]),
                        }))
                    }
                    _ => Ok(None),
                }
            }

            "content_block_stop" => {
                // If this block was a tool_use, signal the engine that the
                // tool's full input has arrived and it can be dispatched.
                let index = v["index"].as_u64().unwrap_or(u64::MAX);
                if let Some(id) = self.tool_index.remove(&index) {
                    return Ok(Some(StreamEvent::ToolUseEnd { id: id.into() }));
                }
                Ok(None)
            }

            "message_delta" => {
                let stop_reason = match v["delta"]["stop_reason"].as_str() {
                    Some("tool_use") => StopReason::ToolUse,
                    Some("max_tokens") => StopReason::MaxTokens,
                    _ => StopReason::EndTurn,
                };
                let usage = TokenUsage {
                    input_tokens: self.input_tokens,
                    output_tokens: ju32(&v["usage"]["output_tokens"]),
                    cache_read_tokens: ju32(&v["usage"]["cache_read_input_tokens"]),
                    cache_write_tokens: ju32(&v["usage"]["cache_creation_input_tokens"]),
                };
                Ok(Some(StreamEvent::MessageEnd { usage, stop_reason }))
            }

            _ => Ok(None),
        }
    }
}

// ── Request building ──────────────────────────────────────────────────────────

fn build_request_body(
    req: &ChatRequest,
    default_model: &str,
    thinking_budget: Option<u32>,
    cache_enabled: bool,
) -> Value {
    let mut body = json!({
        "model":      req.model.clone().unwrap_or_else(|| default_model.to_string()),
        "max_tokens": req.max_tokens,
        "system":     system_to_json(&req.system, cache_enabled),
        "messages":   messages_to_json(&req.messages),
        "tools":      tools_to_json(&req.tools, cache_enabled),
        "stream":     true,
    });

    if let Some(t) = req.temperature {
        body["temperature"] = json!(t);
    }

    if let Some(budget_tokens) = thinking_budget {
        body["thinking"] = json!({
            "type": "enabled",
            "budget_tokens": budget_tokens,
        });
    }

    body
}

/// Serialise the system prompt.
///
/// When caching is enabled the system text is wrapped in a content-array with
/// a trailing `cache_control` marker, creating a cache breakpoint after the
/// system prompt. Anthropic caches everything up to (and including) the last
/// marked block; subsequent turns that send the same system text get a cache
/// hit and are billed at the reduced cache-read rate.
fn system_to_json(system: &str, cache_enabled: bool) -> Value {
    if !cache_enabled || system.is_empty() {
        return json!(system);
    }
    json!([{
        "type":          "text",
        "text":          system,
        "cache_control": { "type": "ephemeral" },
    }])
}

fn messages_to_json(messages: &[Message]) -> Value {
    // Anthropic's API uses only "user" and "assistant" roles in the messages array.
    // System-role messages (injected by the engine for permission decisions, hook
    // rejections, and compression notices) are translated to "user" so the LLM
    // sees them in context. This is the correct provider-boundary translation.
    Value::Array(
        messages
            .iter()
            .map(|m| {
                json!({
                    "role":    match m.role { Role::Assistant => "assistant", _ => "user" },
                    "content": blocks_to_json(&m.content),
                })
            })
            .collect(),
    )
}

fn blocks_to_json(blocks: &[ContentBlock]) -> Value {
    Value::Array(
        blocks
            .iter()
            .map(|b| match b {
                ContentBlock::Text { text } => json!({"type":"text","text":text}),
                ContentBlock::Thinking { text } => json!({"type":"thinking","thinking":text}),
                ContentBlock::ToolUse {
                    id, name, input, ..
                } => json!({"type":"tool_use","id":id,"name":name,"input":input}),
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => json!({
                    "type":        "tool_result",
                    "tool_use_id": tool_use_id,
                    "content":     content,
                    "is_error":    is_error,
                }),
                ContentBlock::Image { source } => match source {
                    ImageSource::Base64 { media_type, data } => json!({
                        "type": "image",
                        "source": { "type": "base64", "media_type": media_type, "data": data },
                    }),
                    ImageSource::Url(url) => json!({
                        "type": "image",
                        "source": { "type": "url", "url": url },
                    }),
                },
                ContentBlock::Document { source, title } => {
                    let mut obj = match source {
                        DocumentSource::Base64 { media_type, data } => json!({
                            "type": "document",
                            "source": { "type": "base64", "media_type": media_type, "data": data },
                        }),
                        DocumentSource::Url(url) => json!({
                            "type": "document",
                            "source": { "type": "url", "url": url },
                        }),
                    };
                    if let Some(t) = title {
                        obj["title"] = json!(t);
                    }
                    obj
                }
                // Framework compression markers are translated to plain text so the LLM
                // sees the summary as part of the conversation context.
                ContentBlock::Collapsed { summary, .. } => json!({"type":"text","text":summary}),
                ContentBlock::CompactBoundary { summary } => json!({"type":"text","text":summary}),
            })
            .collect(),
    )
}

/// Serialise tool definitions.
///
/// When caching is enabled a `cache_control` marker is added to the last tool
/// definition. Anthropic caches everything up to the last marked item, so all
/// tool definitions (which rarely change within a run) are covered by a single
/// breakpoint at the end of the list.
fn tools_to_json(tools: &[ToolDef], cache_enabled: bool) -> Value {
    let last = tools.len().saturating_sub(1);
    Value::Array(
        tools
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let mut obj = json!({
                    "name":         t.name,
                    "description":  t.description,
                    "input_schema": t.input_schema,
                });
                if cache_enabled && i == last && !tools.is_empty() {
                    obj["cache_control"] = json!({ "type": "ephemeral" });
                }
                obj
            })
            .collect(),
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use wui_core::message::{ContentBlock, ImageSource, Message, Role};

    fn msg(role: Role, blocks: Vec<ContentBlock>) -> Message {
        Message {
            id: "test".into(),
            role,
            content: blocks,
        }
    }

    #[test]
    fn user_text_message_serializes_correctly() {
        let messages = vec![msg(
            Role::User,
            vec![ContentBlock::Text {
                text: "Hello world".into(),
            }],
        )];
        let v = messages_to_json(&messages);
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["role"], "user");
        let content = arr[0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "Hello world");
    }

    #[test]
    fn assistant_tool_use_serializes_correctly() {
        let input = serde_json::json!({"query": "rust"});
        let messages = vec![msg(
            Role::Assistant,
            vec![ContentBlock::ToolUse {
                id: "tu_1".into(),
                name: "search".into(),
                input: input.clone(),
                summary: None,
            }],
        )];
        let v = messages_to_json(&messages);
        let arr = v.as_array().unwrap();
        assert_eq!(arr[0]["role"], "assistant");
        let block = &arr[0]["content"].as_array().unwrap()[0];
        assert_eq!(block["type"], "tool_use");
        assert_eq!(block["id"], "tu_1");
        assert_eq!(block["name"], "search");
        assert_eq!(block["input"], input);
    }

    #[test]
    fn tool_result_message_serializes_correctly() {
        let messages = vec![msg(
            Role::User,
            vec![ContentBlock::ToolResult {
                tool_use_id: "tu_1".into(),
                content: "42 results".into(),
                is_error: false,
            }],
        )];
        let v = messages_to_json(&messages);
        let arr = v.as_array().unwrap();
        assert_eq!(arr[0]["role"], "user");
        let block = &arr[0]["content"].as_array().unwrap()[0];
        assert_eq!(block["type"], "tool_result");
        assert_eq!(block["tool_use_id"], "tu_1");
        assert_eq!(block["content"], "42 results");
        assert_eq!(block["is_error"], false);
    }

    #[test]
    fn image_base64_serializes_correctly() {
        let messages = vec![msg(
            Role::User,
            vec![ContentBlock::Image {
                source: ImageSource::Base64 {
                    media_type: "image/png".into(),
                    data: "abc123".into(),
                },
            }],
        )];
        let v = messages_to_json(&messages);
        let block = &v.as_array().unwrap()[0]["content"].as_array().unwrap()[0];
        assert_eq!(block["type"], "image");
        assert_eq!(block["source"]["type"], "base64");
        assert_eq!(block["source"]["media_type"], "image/png");
        assert_eq!(block["source"]["data"], "abc123");
    }

    #[test]
    fn thinking_block_serialized_as_thinking_type() {
        let messages = vec![msg(
            Role::Assistant,
            vec![ContentBlock::Thinking {
                text: "Let me think...".into(),
            }],
        )];
        let v = messages_to_json(&messages);
        let block = &v.as_array().unwrap()[0]["content"].as_array().unwrap()[0];
        // Thinking blocks are serialized with type "thinking" for Anthropic API
        assert_eq!(block["type"], "thinking");
        assert_eq!(block["thinking"], "Let me think...");
    }

    #[test]
    fn system_role_message_becomes_user() {
        // System-role mid-conversation messages must be translated to "user"
        // so the Anthropic API sees them in context.
        let messages = vec![msg(
            Role::System,
            vec![ContentBlock::Text {
                text: "You must be polite.".into(),
            }],
        )];
        let v = messages_to_json(&messages);
        let arr = v.as_array().unwrap();
        assert_eq!(arr[0]["role"], "user");
    }

    #[test]
    fn url_derivation() {
        assert_eq!(
            messages_url("https://api.anthropic.com"),
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(
            messages_url("https://api.anthropic.com/v1"),
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(
            messages_url("https://api.anthropic.com/v1/messages"),
            "https://api.anthropic.com/v1/messages"
        );
    }

    #[test]
    fn sse_parser_tool_use_attributed_correctly() {
        let mut parser = SseParser::default();

        // content_block_start for a tool_use block at index 1
        let start =
            r#"{"index":1,"content_block":{"type":"tool_use","id":"tu_abc","name":"search"}}"#;
        let result = parser.parse("content_block_start", start).unwrap();
        assert!(
            matches!(result, Some(StreamEvent::ToolUseStart { id, name }) if id == "tu_abc" && name == "search")
        );

        // input_json_delta attributed to index 1
        let delta = r#"{"index":1,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}"#;
        let result = parser.parse("content_block_delta", delta).unwrap();
        assert!(
            matches!(result, Some(StreamEvent::ToolInputDelta { id, chunk }) if id == "tu_abc" && chunk == "{\"q\":")
        );

        // content_block_stop for index 1 → ToolUseEnd
        let stop = r#"{"index":1}"#;
        let result = parser.parse("content_block_stop", stop).unwrap();
        assert!(matches!(result, Some(StreamEvent::ToolUseEnd { id }) if id == "tu_abc"));
    }

    // ── Property tests ───────────────────────────────────────────────────────

    mod fuzz {
        use super::*;
        use proptest::prelude::*;

        /// SSE parser must never panic on arbitrary event types and data.
        #[test]
        fn parse_never_panics_on_arbitrary_input() {
            proptest!(ProptestConfig::with_cases(1000), |(
                event_type in ".*",
                data in ".*",
            )| {
                let mut parser = SseParser::default();
                // We only care that it doesn't panic — Ok or Err are both fine.
                let _ = parser.parse(&event_type, &data);
            });
        }

        /// SSE parser must never panic on known event types with arbitrary JSON.
        #[test]
        fn parse_never_panics_on_known_events_with_arbitrary_json() {
            let event_types = [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
            ];
            proptest!(ProptestConfig::with_cases(2000), |(
                event_idx in 0..event_types.len(),
                json_str in prop::string::string_regex(
                    r#"\{[a-z":{},0-9 ]*\}"#
                ).unwrap(),
            )| {
                let mut parser = SseParser::default();
                let _ = parser.parse(event_types[event_idx], &json_str);
            });
        }

        /// Valid JSON with unexpected structure must not panic.
        #[test]
        fn parse_handles_valid_json_with_wrong_shape() {
            proptest!(ProptestConfig::with_cases(500), |(
                event_type in prop::sample::select(vec![
                    "message_start",
                    "content_block_start",
                    "content_block_delta",
                    "content_block_stop",
                    "message_delta",
                ]),
            )| {
                let mut parser = SseParser::default();
                // Empty object
                let _ = parser.parse(event_type, "{}");
                // Null
                let _ = parser.parse(event_type, "null");
                // Array
                let _ = parser.parse(event_type, "[]");
                // Deeply nested
                let _ = parser.parse(event_type, r#"{"a":{"b":{"c":{"d":1}}}}"#);
                // Wrong types for expected fields
                let _ = parser.parse(event_type, r#"{"index":"not_a_number","delta":42}"#);
            });
        }

        /// A sequence of well-formed events should produce a coherent stream.
        #[test]
        fn full_sequence_produces_valid_stream() {
            let mut parser = SseParser::default();

            // message_start
            let data = r#"{"message":{"usage":{"input_tokens":100}}}"#;
            let r = parser.parse("message_start", data).unwrap();
            assert!(r.is_none());

            // text content
            let data = r#"{"index":0,"delta":{"type":"text_delta","text":"hello"}}"#;
            let r = parser.parse("content_block_delta", data).unwrap();
            assert!(matches!(r, Some(StreamEvent::TextDelta { text }) if text == "hello"));

            // message_delta (end)
            let data = r#"{"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":50}}"#;
            let r = parser.parse("message_delta", data).unwrap();
            assert!(
                matches!(r, Some(StreamEvent::MessageEnd { usage, .. }) if usage.input_tokens == 100 && usage.output_tokens == 50)
            );
        }
    }
}
