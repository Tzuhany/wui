// ============================================================================
// OpenAI Request Serialization
//
// Converts wui's internal message/tool types to OpenAI's JSON format.
// ============================================================================

use serde_json::{json, Value};

use wui_core::message::{ContentBlock, ImageSource, Message, Role};
use wui_core::provider::{ChatRequest, ResponseFormat, ToolDef};

pub(crate) fn build_request_body(req: &ChatRequest, default_model: &str) -> Value {
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

    if let Some(ref fmt) = req.response_format {
        match fmt {
            ResponseFormat::JsonSchema { name, schema } => {
                body["response_format"] = json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "schema": schema,
                        "strict": true,
                    },
                });
            }
        }
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

    if let Some(sys) = system_message_to_json(system) {
        out.push(sys);
    }

    for msg in messages {
        match msg.role {
            Role::System => {
                if let Some(v) = framework_message_to_json(msg) {
                    out.push(v);
                }
            }
            Role::User => out.extend(user_message_to_json(msg)),
            Role::Assistant => {
                if let Some(v) = assistant_message_to_json(msg) {
                    out.push(v);
                }
            }
        }
    }

    json!(out)
}

/// Serialise the top-level system prompt (if non-empty).
fn system_message_to_json(system: &str) -> Option<Value> {
    if system.is_empty() {
        return None;
    }
    Some(json!({"role": "system", "content": system}))
}

/// Serialise a framework-injected mid-conversation system message as a user
/// message (OpenAI has no mid-conversation system role).
fn framework_message_to_json(msg: &Message) -> Option<Value> {
    let text = collect_text(&msg.content);
    if text.is_empty() {
        return None;
    }
    Some(json!({"role": "user", "content": text}))
}

/// Serialise a user message into one or more JSON values.
///
/// Tool results and regular text/images may coexist in the same wui message;
/// OpenAI requires them in separate typed messages. Returns a Vec because a
/// single wui user message can expand into multiple OpenAI messages (tool
/// results + content).
fn user_message_to_json(msg: &Message) -> Vec<Value> {
    let mut out = Vec::new();
    let mut content_items: Vec<Value> = Vec::new();
    let mut has_images = false;

    for block in &msg.content {
        match block {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                out.push(json!({
                    "role":         "tool",
                    "tool_call_id": tool_use_id,
                    "content":      content,
                }));
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

    // Remaining content: use array form when images are present
    // (required by OpenAI), plain string otherwise (more compatible).
    if !content_items.is_empty() {
        if has_images {
            out.push(json!({"role": "user", "content": content_items}));
        } else {
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

    out
}

/// Serialise an assistant message. Returns `None` when empty (e.g. thinking-only
/// turns from a previous Anthropic session — sending an empty assistant message
/// causes a 400 from the OpenAI API).
fn assistant_message_to_json(msg: &Message) -> Option<Value> {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();

    for block in &msg.content {
        match block {
            ContentBlock::Text { text } => text_parts.push(text.clone()),
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                let arguments = serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string());
                tool_calls.push(json!({
                    "id":   id,
                    "type": "function",
                    "function": { "name": name, "arguments": arguments },
                }));
            }
            // Thinking is Anthropic-specific — skip.
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
    if combined.is_empty() && tool_calls.is_empty() {
        return None;
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
    Some(obj)
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

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
    fn user_text_message_serializes_correctly() {
        let msg = Message::user("Hello, world!");
        let v = messages_to_json("", &[msg]);
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["role"], "user");
        assert_eq!(arr[0]["content"], "Hello, world!");
    }

    #[test]
    fn tool_call_produces_function_wrapper() {
        use wui_core::provider::ToolDef;

        let req = ChatRequest {
            model: None,
            max_tokens: 100,
            temperature: None,
            system: String::new(),
            messages: vec![],
            tools: vec![ToolDef {
                name: "search".to_string(),
                description: "Search the web".to_string(),
                input_schema: serde_json::json!({"type": "object", "properties": {"q": {"type": "string"}}}),
            }],
            thinking_budget: None,
            cache_boundary: None,
            response_format: None,
        };

        let body = build_request_body(&req, "gpt-4");
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "search");
        assert_eq!(tools[0]["function"]["description"], "Search the web");
        assert!(tools[0]["function"]["parameters"].is_object());
    }

    #[test]
    fn system_prompt_becomes_system_role() {
        let v = messages_to_json("You are a helpful assistant.", &[]);
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["role"], "system");
        assert_eq!(arr[0]["content"], "You are a helpful assistant.");
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
