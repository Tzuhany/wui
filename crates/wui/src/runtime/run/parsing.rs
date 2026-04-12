// ── Stream parsing with eager tool dispatch ─────────────────────────────────
//
// Processes the provider's stream event by event. When a tool's input is
// complete (ToolUseEnd), it is prepared immediately — hooks run, permissions
// are evaluated. Tools that can proceed without human approval are submitted
// to the executor right away, while the stream continues. Tools that need
// HITL approval are queued for after MessageEnd.
//
// This implements the "Streaming Before Orchestration" principle: work
// starts as soon as it is describable, not after the LLM finishes speaking.
//
// ── Timeline ────────────────────────────────────────────────────────────────
//
//   stream: [text...] [tool₁ input complete → hook → auto-approve → execute]
//                      [text...] [tool₂ input complete → hook → needs HITL → queue]
//           [MessageEnd]
//   post:   [tool₂ HITL prompt → approve/deny → execute/skip]
//           [collect all results]

use std::sync::Arc;

use futures::StreamExt;
use tokio::sync::mpsc;

use wui_core::event::{AgentError, AgentEvent};
use wui_core::message::Message;
use wui_core::provider::ProviderError;
use wui_core::tool::ToolCallId;

use super::auth::{self, AuthOutcome, AuthRequest, PrepareOutcome};
use super::registry::ToolRegistry;
use super::tool_batch::emit_tool_event;
use super::RunConfig;
use super::{CompletedTool, IterationCtx, PendingTool, ToolExecutor};

/// A tool that passed preparation but needs interactive approval.
pub(super) struct DeferredApproval {
    pub id: ToolCallId,
    pub name: String,
    pub input: serde_json::Value,
    pub session_pattern: String,
    pub ctrl_req: wui_core::event::ControlRequest,
}

/// Parse the provider stream, eagerly dispatching tools that don't need
/// human approval. Returns `Ok(true)` on `MessageEnd`, `Ok(false)` on
/// premature stream end.
///
/// Tools needing HITL are collected in `deferred` for the caller to
/// handle after the stream completes.
pub(super) async fn stream_and_dispatch(
    stream: &mut (impl futures::Stream<Item = Result<wui_core::event::StreamEvent, ProviderError>>
              + Unpin),
    ctx: &mut IterationCtx,
    config: &Arc<RunConfig>,
    active_registry: &Arc<ToolRegistry>,
    executor: &mut ToolExecutor,
    messages: &[Message],
    deferred: &mut Vec<DeferredApproval>,
    tx: &mpsc::Sender<AgentEvent>,
) -> Result<bool, AgentError> {
    use wui_core::event::StreamEvent::*;

    let history_snapshot: Arc<[Message]> = Arc::from(messages.to_vec());
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
                        tracing::warn!(tool = %name, error = %e, "malformed tool input JSON");
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

                // ── Eager dispatch: prepare immediately, dispatch if ready ──
                let req = AuthRequest {
                    id: id.clone(),
                    name: name.clone(),
                    input,
                };
                match auth::prepare_tool(config, active_registry, req).await {
                    PrepareOutcome::Ready(outcome) => {
                        handle_ready_outcome(ctx, executor, &history_snapshot, outcome, tx).await;
                    }
                    PrepareOutcome::NeedsApproval {
                        id,
                        name,
                        input,
                        session_pattern,
                        ctrl_req,
                        ..
                    } => {
                        deferred.push(DeferredApproval {
                            id,
                            name,
                            input,
                            session_pattern,
                            ctrl_req,
                        });
                    }
                }
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

/// Process tools deferred for HITL approval (after the LLM finishes speaking).
///
/// All deferred tools are authorized concurrently via a `JoinSet`. Each tool's
/// HITL prompt fires independently, and approved tools start executing as soon
/// as their own approval resolves — no head-of-line blocking.
pub(super) async fn approve_deferred(
    deferred: Vec<DeferredApproval>,
    ctx: &mut IterationCtx,
    config: &Arc<RunConfig>,
    executor: &mut ToolExecutor,
    messages: &[Message],
    tx: &mpsc::Sender<AgentEvent>,
) {
    let history_snapshot: Arc<[Message]> = Arc::from(messages.to_vec());

    let mut auth_tasks: tokio::task::JoinSet<AuthOutcome> = tokio::task::JoinSet::new();

    for tool in deferred {
        let config_c = Arc::clone(config);
        let tx_c = tx.clone();
        auth_tasks.spawn(async move {
            auth::approve_tool(
                &config_c,
                tool.id,
                tool.name,
                tool.input,
                tool.session_pattern,
                tool.ctrl_req,
                &tx_c,
            )
            .await
        });
    }

    while let Some(outcome) = auth_tasks.join_next().await {
        let outcome = match outcome {
            Ok(o) => o,
            Err(e) => {
                tracing::error!(error = %e, "deferred auth task panicked — skipping tool");
                continue;
            }
        };
        handle_ready_outcome(ctx, executor, &history_snapshot, outcome, tx).await;
    }
}

/// Handle an authorization outcome: dispatch allowed tools, record denied ones.
async fn handle_ready_outcome(
    ctx: &mut IterationCtx,
    executor: &mut ToolExecutor,
    history_snapshot: &Arc<[Message]>,
    outcome: AuthOutcome,
    tx: &mpsc::Sender<AgentEvent>,
) {
    match outcome {
        AuthOutcome::Allowed {
            id,
            name,
            input,
            injections,
        } => {
            ctx.auth_injections.extend(injections);
            ctx.remember_tool_input(&id, input.clone());
            tracing::debug!(tool = %name, "tool dispatched");
            tx.send(AgentEvent::ToolStart {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
            })
            .await
            .ok();
            executor.submit(PendingTool {
                id,
                name,
                input,
                messages: Arc::clone(history_snapshot),
            });
        }
        AuthOutcome::Denied { tool, injections } => {
            ctx.auth_injections.extend(injections);
            ctx.emission_guard.first_time(&tool.id);
            emit_tool_event(&tool, tx).await;
            ctx.completed_map.insert(tool.id.clone(), tool);
        }
    }
}
