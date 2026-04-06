// ============================================================================
// Anthropic Provider
//
// Implements the Provider trait against Anthropic's Messages API.
// Supports: streaming, prompt caching, extended thinking, tool use.
//
// Extensions recognised in ChatRequest.extensions:
//   "thinking"  → {"type":"enabled","budget_tokens":N}
//   "betas"     → ["prompt-caching-2024-07-31", ...]
// ============================================================================

use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde_json::{json, Value};

use wuhu_core::event::{StopReason, StreamEvent, TokenUsage};
use wuhu_core::message::{ContentBlock, Message, Role};
use wuhu_core::provider::{ChatRequest, Provider, ProviderError, ToolDef};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

pub struct Anthropic {
    client:  reqwest::Client,
    api_key: String,
}

impl Anthropic {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("failed to build reqwest client"),
            api_key: api_key.into(),
        }
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
            .post(API_URL)
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

        let stream = eventsource_stream::Eventsource::new(response.bytes_stream())
            .map(|result| match result {
                Ok(event) if event.event == "content_block_delta"
                    || event.event == "message_delta"
                    || event.event == "content_block_start"
                    || event.event == "message_stop" =>
                    parse_sse_event(&event.event, &event.data),
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

// ── Request building ──────────────────────────────────────────────────────────

fn build_request_body(req: &ChatRequest) -> Value {
    let mut body = json!({
        "model":       req.model,
        "max_tokens":  req.max_tokens,
        "system":      req.system,
        "messages":    messages_to_json(&req.messages),
        "tools":       tools_to_json(&req.tools),
        "stream":      true,
    });

    if let Some(t) = req.temperature {
        body["temperature"] = json!(t);
    }

    // Extended thinking.
    if let Some(thinking) = req.extensions.get("thinking") {
        body["thinking"] = thinking.clone();
    }

    body
}

fn messages_to_json(messages: &[Message]) -> Value {
    Value::Array(
        messages.iter()
            .filter(|m| m.role != Role::System) // System is a top-level field in Anthropic's API.
            .map(|m| json!({
                "role":    match m.role { Role::User => "user", Role::Assistant => "assistant", Role::System => "user" },
                "content": blocks_to_json(&m.content),
            }))
            .collect()
    )
}

fn blocks_to_json(blocks: &[ContentBlock]) -> Value {
    Value::Array(blocks.iter().filter_map(|b| match b {
        ContentBlock::Text     { text }                    => Some(json!({"type":"text","text":text})),
        ContentBlock::Thinking { text }                    => Some(json!({"type":"thinking","thinking":text})),
        ContentBlock::ToolUse  { id, name, input }         => Some(json!({"type":"tool_use","id":id,"name":name,"input":input})),
        ContentBlock::ToolResult { tool_use_id, content, is_error } => Some(json!({
            "type":        "tool_result",
            "tool_use_id": tool_use_id,
            "content":     content,
            "is_error":    is_error,
        })),
        ContentBlock::Compressed { summary, .. } => Some(json!({"type":"text","text":summary})),
    }).collect())
}

fn tools_to_json(tools: &[ToolDef]) -> Value {
    Value::Array(tools.iter().map(|t| json!({
        "name":         t.name,
        "description":  t.description,
        "input_schema": t.input_schema,
    })).collect())
}

// ── SSE event parsing ─────────────────────────────────────────────────────────

fn parse_sse_event(event_type: &str, data: &str) -> Result<Option<StreamEvent>, ProviderError> {
    let v: Value = serde_json::from_str(data)
        .map_err(|e| ProviderError::Stream(format!("json parse: {e}")))?;

    match event_type {
        "content_block_start" => {
            let block_type = v["content_block"]["type"].as_str().unwrap_or("");
            let id   = v["content_block"]["id"].as_str().unwrap_or("").to_string();
            let name = v["content_block"]["name"].as_str().unwrap_or("").to_string();
            if block_type == "tool_use" {
                return Ok(Some(StreamEvent::ToolUseStart { id, name }));
            }
            Ok(None)
        }

        "content_block_delta" => {
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
                    let chunk = v["delta"]["partial_json"].as_str().unwrap_or("").to_string();
                    // index doubles as a pseudo-id for tool input deltas
                    let index = v["index"].as_u64().unwrap_or(0).to_string();
                    Ok(Some(StreamEvent::ToolInputDelta { id: index, chunk }))
                }
                _ => Ok(None),
            }
        }

        "message_delta" => {
            let stop_reason = match v["delta"]["stop_reason"].as_str() {
                Some("tool_use") => StopReason::ToolUse,
                Some("max_tokens") => StopReason::MaxTokens,
                _ => StopReason::EndTurn,
            };
            let usage = TokenUsage {
                input_tokens:        0,
                output_tokens:       v["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32,
                cache_read_tokens:   v["usage"]["cache_read_input_tokens"].as_u64().unwrap_or(0) as u32,
                cache_write_tokens:  v["usage"]["cache_creation_input_tokens"].as_u64().unwrap_or(0) as u32,
            };
            Ok(Some(StreamEvent::MessageEnd { usage, stop_reason }))
        }

        _ => Ok(None),
    }
}
