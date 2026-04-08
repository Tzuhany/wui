// ============================================================================
// OpenAI SSE Parser
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

use serde_json::Value;

use wui_core::event::{StopReason, StreamEvent, TokenUsage};
use wui_core::provider::ProviderError;
use wui_core::types::ToolCallId;

/// Parses OpenAI SSE events for one stream lifetime.
///
/// Tracks `index → (id, name)` so argument delta chunks can be attributed to
/// the correct tool call even without repeating the id on every chunk.
///
/// Holds the stop reason between the `finish_reason` chunk and the usage chunk
/// so that `MessageEnd` is emitted with real token counts, not zeros.
#[derive(Default)]
pub(crate) struct SseParser {
    /// Maps tool call index to (id, name). Index is present on every delta;
    /// id and name only appear on the first chunk for that index.
    tool_calls: HashMap<u32, (String, String)>,
    /// Stop reason from the last `finish_reason` field. Held until the usage
    /// chunk arrives, since usage is reported in a separate SSE event.
    pending_stop: Option<StopReason>,
}

impl SseParser {
    pub(crate) fn parse_all(&mut self, data: &str) -> Result<Vec<StreamEvent>, ProviderError> {
        let v: Value = serde_json::from_str(data)
            .map_err(|e| ProviderError::Stream(format!("json parse: {e}")))?;

        let mut events = Vec::new();

        // ── Choice-level events ───────────────────────────────────────────────
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

        // ── Usage chunk ───────────────────────────────────────────────────────
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

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
}
