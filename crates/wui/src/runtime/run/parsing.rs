// ── Stream event parsing ────────────────────────────────────────────────────

use futures::StreamExt;
use tokio::sync::mpsc;

use wui_core::event::{AgentError, AgentEvent};
use wui_core::provider::ProviderError;

use super::registry::ToolRegistry;
use super::tool_batch::emit_tool_event;
use super::CompletedTool;
use super::IterationCtx;

/// Parse all stream events, accumulating tool calls, text, and thinking into
/// the iteration context. Returns `Ok(true)` when `MessageEnd` was received,
/// `Ok(false)` when the stream ended without it, `Err` on fatal errors.
pub(super) async fn parse_stream(
    stream: &mut (impl futures::Stream<Item = Result<wui_core::event::StreamEvent, ProviderError>>
              + Unpin),
    ctx: &mut IterationCtx,
    active_registry: &ToolRegistry,
    tx: &mpsc::Sender<AgentEvent>,
) -> Result<bool, AgentError> {
    use wui_core::event::StreamEvent::*;

    let mut text_chunk_count: u32 = 0;

    while let Some(event) = stream.next().await {
        let event = match event {
            Ok(e) => e,
            Err(e) => {
                if e.is_retryable() {
                    continue;
                }
                return Err(AgentError::fatal(e.to_string()));
            }
        };

        match event {
            TextDelta { text } => {
                text_chunk_count += 1;
                ctx.text_buf.push_str(&text);
                tx.send(AgentEvent::TextDelta(text)).await.ok();
            }

            ThinkingDelta { text } => {
                ctx.thinking_buf.push_str(&text);
                tx.send(AgentEvent::ThinkingDelta(text)).await.ok();
            }

            ToolUseStart { id, name } => {
                ctx.pending_inputs.insert(id, (name, String::new()));
            }

            ToolInputDelta { id, chunk } => {
                if let Some((_, json)) = ctx.pending_inputs.get_mut(&id) {
                    json.push_str(&chunk);
                }
            }

            ToolUseEnd { id } => {
                let Some((name, json)) = ctx.pending_inputs.remove(&id) else {
                    continue;
                };

                let input: serde_json::Value = match serde_json::from_str(&json) {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!(tool = %name, error = %e, "malformed tool input JSON from provider");
                        ctx.record_tool_use(
                            id.clone(),
                            name.clone(),
                            serde_json::Value::Null,
                            None,
                        );
                        let denied = CompletedTool::immediate(
                            id.clone(),
                            name,
                            wui_core::tool::ToolOutput::invalid_input(format!(
                                "malformed tool input JSON: {e}"
                            )),
                        );
                        ctx.emission_guard.first_time(&denied.id);
                        emit_tool_event(&denied, tx).await;
                        ctx.completed_map.insert(denied.id.clone(), denied);
                        continue;
                    }
                };

                let tool_summary = active_registry
                    .get(&name)
                    .and_then(|t| t.executor_hints(&input).summary);
                ctx.record_tool_use(id.clone(), name.clone(), input.clone(), tool_summary);

                // Queue for authorization after MessageEnd.
                ctx.pending_auths.push((id, name, input));
            }

            MessageEnd {
                usage: u,
                stop_reason: sr,
            } => {
                ctx.usage = u;
                ctx.stop_reason = sr;
                let span = tracing::Span::current();
                span.record("text_chunks_count", text_chunk_count);
                span.record("total_text_len", ctx.text_buf.len() as u64);
                return Ok(true);
            }

            Error { message, retryable } => {
                return Err(AgentError {
                    message,
                    retryable,
                    detail: None,
                    permission_denied: false,
                });
            }
        }
    }
    Ok(false)
}
