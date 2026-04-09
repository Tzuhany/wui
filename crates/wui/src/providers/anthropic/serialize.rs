// ============================================================================
// Anthropic Request Serialization
//
// Converts wui's internal message/tool types to Anthropic's JSON format.
// ============================================================================

use serde_json::{json, Value};

use wui_core::message::{ContentBlock, DocumentSource, ImageSource, Message, Role};
use wui_core::provider::{ChatRequest, ToolDef};

pub(crate) fn build_request_body(
    req: &ChatRequest,
    default_model: &str,
    thinking_budget: Option<u32>,
    cache_enabled: bool,
) -> Value {
    let mut body = json!({
        "model":      req.model.clone().unwrap_or_else(|| default_model.to_string()),
        "max_tokens": req.max_tokens,
        "system":     system_to_json(&req.system, cache_enabled, req.cache_boundary),
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
fn system_to_json(system: &str, cache_enabled: bool, cache_boundary: Option<usize>) -> Value {
    if !cache_enabled || system.is_empty() {
        return json!(system);
    }

    // When a boundary is set, split into stable (cached) + dynamic (uncached).
    if let Some(idx) = cache_boundary {
        let stable = &system[..idx];
        let dynamic = system[idx..].trim_start();
        let mut blocks = vec![json!({
            "type":          "text",
            "text":          stable,
            "cache_control": { "type": "ephemeral" },
        })];
        if !dynamic.is_empty() {
            blocks.push(json!({
                "type": "text",
                "text": dynamic,
            }));
        }
        return Value::Array(blocks);
    }

    // No boundary — single block with cache_control on the whole thing.
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

// ── Tests ────────────────────────────────────────────────────────────────────

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
}
