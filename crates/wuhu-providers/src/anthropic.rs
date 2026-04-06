// ============================================================================
// Anthropic Provider
//
// Implements the Provider trait against Anthropic's Messages API.
// Supports: streaming, prompt caching, extended thinking, tool use.
//
// Extensions recognised in ChatRequest.extensions:
//   "thinking"  → {"type":"enabled","budget_tokens":N}
//   "betas"     → ["prompt-caching-2024-07-31", ...]
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

use wuhu_core::event::{StopReason, StreamEvent, TokenUsage};
use wuhu_core::message::{ContentBlock, Message, Role};
use wuhu_core::provider::{ChatRequest, Provider, ProviderError, ToolDef};

const ANTHROPIC_VERSION: &str = "2023-06-01";

#[derive(Clone)]
pub struct Anthropic {
    client:  reqwest::Client,
    api_key: String,
    api_url: String,
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
        }
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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError> {
        let body = build_request_body(&req);

        let mut request = self.client
            .post(&self.api_url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json");

        // Inject beta headers if requested.
        if let Some(betas) = req.extensions.get("betas").and_then(|v| v.as_array()) {
            let beta_str: Vec<&str> = betas.iter()
                .filter_map(|v| v.as_str())
                .collect();
            if !beta_str.is_empty() {
                request = request.header("anthropic-beta", beta_str.join(","));
            }
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
                return Err(ProviderError::RateLimit { retry_after_ms: 5_000 });
            }
            if status.as_u16() == 401 {
                return Err(ProviderError::Auth(text));
            }
            return Err(ProviderError::ServerError { status: status.as_u16(), message: text });
        }

        // Stateful parser — one per stream, tracks content-block index → tool_use_id.
        let mut parser = SseParser::default();

        use eventsource_stream::Eventsource as _;
        let stream = response.bytes_stream()
            .eventsource()
            .map(move |result| match result {
                Ok(event) if !event.data.is_empty() => parser.parse(&event.event, &event.data),
                Ok(_)  => Ok(None),
                Err(e) => Err(ProviderError::Stream(e.to_string())),
            })
            .filter_map(|r| async move {
                match r {
                    Ok(Some(event)) => Some(Ok(event)),
                    Ok(None)        => None,
                    Err(e)          => Some(Err(e)),
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

impl SseParser {
    fn parse(&mut self, event_type: &str, data: &str) -> Result<Option<StreamEvent>, ProviderError> {
        let v: Value = serde_json::from_str(data)
            .map_err(|e| ProviderError::Stream(format!("json parse: {e}")))?;

        match event_type {
            // Input tokens arrive in message_start, not message_delta.
            // Capture them here; they are merged into MessageEnd below.
            "message_start" => {
                self.input_tokens = v["message"]["usage"]["input_tokens"]
                    .as_u64().unwrap_or(0) as u32;
                Ok(None)
            }

            "content_block_start" => {
                let index      = v["index"].as_u64().unwrap_or(0);
                let block_type = v["content_block"]["type"].as_str().unwrap_or("");
                let id         = v["content_block"]["id"].as_str().unwrap_or("").to_string();
                let name       = v["content_block"]["name"].as_str().unwrap_or("").to_string();

                if block_type == "tool_use" {
                    self.tool_index.insert(index, id.clone());
                    return Ok(Some(StreamEvent::ToolUseStart { id, name }));
                }
                Ok(None)
            }

            "content_block_delta" => {
                let index      = v["index"].as_u64().unwrap_or(0);
                let delta_type = v["delta"]["type"].as_str().unwrap_or("");

                match delta_type {
                    "text_delta" => {
                        let text = v["delta"]["text"].as_str().unwrap_or("").to_string();
                        Ok(Some(StreamEvent::TextDelta { text }))
                    }
                    "thinking_delta" => {
                        let text = v["delta"]["thinking"].as_str().unwrap_or("").to_string();
                        Ok(Some(StreamEvent::ThinkingDelta { text }))
                    }
                    "input_json_delta" => {
                        // Resolve the tool_use_id by content-block index.
                        let id = self.tool_index.get(&index)
                            .cloned()
                            .unwrap_or_else(|| index.to_string()); // fallback: should not happen
                        let chunk = v["delta"]["partial_json"].as_str().unwrap_or("").to_string();
                        Ok(Some(StreamEvent::ToolInputDelta { id, chunk }))
                    }
                    _ => Ok(None),
                }
            }

            "content_block_stop" => {
                // If this block was a tool_use, signal the engine that the
                // tool's full input has arrived and it can be dispatched.
                let index = v["index"].as_u64().unwrap_or(u64::MAX);
                if let Some(id) = self.tool_index.remove(&index) {
                    return Ok(Some(StreamEvent::ToolUseEnd { id }));
                }
                Ok(None)
            }

            "message_delta" => {
                let stop_reason = match v["delta"]["stop_reason"].as_str() {
                    Some("tool_use")   => StopReason::ToolUse,
                    Some("max_tokens") => StopReason::MaxTokens,
                    _                  => StopReason::EndTurn,
                };
                let usage = TokenUsage {
                    input_tokens:        self.input_tokens,
                    output_tokens:       v["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32,
                    cache_read_tokens:   v["usage"]["cache_read_input_tokens"].as_u64().unwrap_or(0) as u32,
                    cache_write_tokens:  v["usage"]["cache_creation_input_tokens"].as_u64().unwrap_or(0) as u32,
                };
                Ok(Some(StreamEvent::MessageEnd { usage, stop_reason }))
            }

            _ => Ok(None),
        }
    }
}

// ── Request building ──────────────────────────────────────────────────────────

fn build_request_body(req: &ChatRequest) -> Value {
    let mut body = json!({
        "model":      req.model,
        "max_tokens": req.max_tokens,
        "system":     req.system,
        "messages":   messages_to_json(&req.messages),
        "tools":      tools_to_json(&req.tools),
        "stream":     true,
    });

    if let Some(t) = req.temperature {
        body["temperature"] = json!(t);
    }

    // Extended thinking.
    if let Some(thinking) = req.extensions.get("thinking") {
        body["thinking"] = thinking.clone();
    }

    // Tool choice — force tool use when specified.
    // Example: .extension("tool_choice", json!({"type": "any"}))
    if let Some(tc) = req.extensions.get("tool_choice") {
        body["tool_choice"] = tc.clone();
    }

    body
}

fn messages_to_json(messages: &[Message]) -> Value {
    // Anthropic's API uses only "user" and "assistant" roles in the messages array.
    // System-role messages (injected by the engine for permission decisions, hook
    // rejections, and compression notices) are translated to "user" so the LLM
    // sees them in context. This is the correct provider-boundary translation.
    Value::Array(
        messages.iter()
            .map(|m| json!({
                "role":    match m.role { Role::Assistant => "assistant", _ => "user" },
                "content": blocks_to_json(&m.content),
            }))
            .collect()
    )
}

fn blocks_to_json(blocks: &[ContentBlock]) -> Value {
    Value::Array(blocks.iter().filter_map(|b| match b {
        ContentBlock::Text       { text }                               => Some(json!({"type":"text","text":text})),
        ContentBlock::Thinking   { text }                               => Some(json!({"type":"thinking","thinking":text})),
        ContentBlock::ToolUse    { id, name, input }                    => Some(json!({"type":"tool_use","id":id,"name":name,"input":input})),
        ContentBlock::ToolResult { tool_use_id, content, is_error }     => Some(json!({
            "type":        "tool_result",
            "tool_use_id": tool_use_id,
            "content":     content,
            "is_error":    is_error,
        })),
        // Framework compression markers are translated to plain text so the LLM
        // sees the summary as part of the conversation context.
        ContentBlock::Collapsed       { summary, .. } => Some(json!({"type":"text","text":summary})),
        ContentBlock::CompactBoundary { summary }     => Some(json!({"type":"text","text":summary})),
    }).collect())
}

fn tools_to_json(tools: &[ToolDef]) -> Value {
    Value::Array(tools.iter().map(|t| json!({
        "name":         t.name,
        "description":  t.description,
        "input_schema": t.input_schema,
    })).collect())
}
