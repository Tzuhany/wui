// ============================================================================
// Anthropic SSE Parser
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

use serde_json::Value;

use wui_core::event::{StopReason, StreamEvent, TokenUsage};
use wui_core::provider::ProviderError;
use wui_core::tool::ToolCallId;

/// Parses Anthropic SSE events for one stream lifetime.
///
/// Maintains a map of content-block `index → tool_use_id` so that
/// `input_json_delta` events can be correctly attributed to their tool call,
/// even when multiple tool calls are in-flight simultaneously.
///
/// Also captures `input_tokens` from `message_start` — the only event where
/// Anthropic reports prompt token consumption.
#[derive(Default)]
pub(crate) struct SseParser {
    /// Maps content-block index to its tool_use_id.
    tool_index: HashMap<u64, String>,
    /// Input tokens reported in `message_start`, carried into `MessageEnd`.
    input_tokens: u32,
}

/// Extract a string from a JSON value, returning an empty string if absent.
fn json_str(v: &Value) -> String {
    v.as_str().unwrap_or("").to_string()
}

/// Extract a u32 from a JSON value, returning 0 if absent.
fn json_u32(v: &Value) -> u32 {
    v.as_u64().unwrap_or(0) as u32
}

impl SseParser {
    pub(crate) fn parse(
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
                self.input_tokens = json_u32(&v["message"]["usage"]["input_tokens"]);
                Ok(None)
            }

            "content_block_start" => {
                let index = v["index"].as_u64().unwrap_or(0);
                let block_type = v["content_block"]["type"].as_str().unwrap_or("");
                let id = json_str(&v["content_block"]["id"]);
                let name = json_str(&v["content_block"]["name"]);

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
                        text: json_str(&v["delta"]["text"]),
                    })),
                    "thinking_delta" => Ok(Some(StreamEvent::ThinkingDelta {
                        text: json_str(&v["delta"]["thinking"]),
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
                            chunk: json_str(&v["delta"]["partial_json"]),
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
                    output_tokens: json_u32(&v["usage"]["output_tokens"]),
                    cache_read_tokens: json_u32(&v["usage"]["cache_read_input_tokens"]),
                    cache_write_tokens: json_u32(&v["usage"]["cache_creation_input_tokens"]),
                };
                Ok(Some(StreamEvent::MessageEnd { usage, stop_reason }))
            }

            _ => Ok(None),
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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

    // ── Property tests ──────────────────────────────────────────────────────

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
