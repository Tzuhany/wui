// ── Tool authorization, hooks, and event emission ───────────────────────────

use std::sync::Arc;

use tokio::sync::mpsc;

use wui_core::event::AgentEvent;
use wui_core::hook::HookDecision;
use wui_core::message::{ContentBlock, Message};

use super::auth::{self, AuthOutcome};
use super::history::system_reminder_msg;
use super::registry::ToolRegistry;
use super::RunConfig;
use super::{CompletedTool, IterationCtx, PendingTool, ToolExecutor};

/// Authorize all collected tools concurrently and dispatch approved ones to
/// the executor. Denied tools are recorded immediately with their events.
pub(super) async fn authorize_and_dispatch(
    ctx: &mut IterationCtx,
    config: &Arc<RunConfig>,
    active_registry: &Arc<ToolRegistry>,
    executor: &mut ToolExecutor,
    messages: &[Message],
    tx: &mpsc::Sender<AgentEvent>,
) {
    let mut auth_tasks: tokio::task::JoinSet<AuthOutcome> = tokio::task::JoinSet::new();

    for (id, name, input) in std::mem::take(&mut ctx.pending_auths) {
        let config_c = Arc::clone(config);
        let registry_c = Arc::clone(active_registry);
        let tx_c = tx.clone();
        auth_tasks.spawn(async move {
            let (result, injections) = auth::authorize_tool(
                &config_c,
                &registry_c,
                id.clone(),
                name.clone(),
                input,
                &tx_c,
            )
            .await;
            match result {
                Ok(effective_input) => AuthOutcome::Allowed {
                    id,
                    name,
                    input: effective_input,
                    injections,
                },
                Err(denied) => AuthOutcome::Denied {
                    tool: denied,
                    injections,
                },
            }
        });
    }

    while let Some(outcome) = auth_tasks.join_next().await {
        let outcome = match outcome {
            Ok(o) => o,
            Err(e) => {
                tracing::error!(error = %e, "auth task panicked — skipping tool");
                continue;
            }
        };
        match outcome {
            AuthOutcome::Allowed {
                id,
                name,
                input,
                injections,
            } => {
                ctx.auth_injections.extend(injections);
                let history_snap = Arc::from(messages);
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
                    messages: history_snap,
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
}

/// Collect remaining executor results and run post-tool hooks.
/// Emits tool events in submission order, after hook mutations are applied.
pub(super) async fn run_post_hooks(
    ctx: &mut IterationCtx,
    executor: ToolExecutor,
    config: &RunConfig,
    messages: &mut Vec<Message>,
    tx: &mpsc::Sender<AgentEvent>,
) {
    for done in executor.collect_remaining().await {
        ctx.completed_map.insert(done.id.clone(), done);
    }

    for id in &ctx.submission_order {
        let Some(done) = ctx.completed_map.get(id) else {
            continue;
        };
        let tool_name = done.name.clone();
        let is_error = done.output.is_error();

        let decision = if is_error {
            let input = ctx.assistant_blocks.iter().find_map(|b| {
                if let ContentBlock::ToolUse { id: bid, input, .. } = b {
                    (bid == id).then_some(input)
                } else {
                    None
                }
            });
            let input = input.unwrap_or(&serde_json::Value::Null);
            config
                .hooks
                .post_tool_failure(&tool_name, input, &done.output)
                .await
        } else {
            config.hooks.post_tool_use(&tool_name, &done.output).await
        };

        match decision {
            HookDecision::Block { reason } => {
                tracing::debug!(tool = %tool_name, %reason, "post-tool hook blocked output");
                messages.push(system_reminder_msg(&format!(
                    "The output of tool '{tool_name}' was blocked by policy: {reason}. \
                     Do not use this output.",
                )));
            }
            HookDecision::MutateOutput { content } => {
                tracing::debug!(tool = %tool_name, "post-tool hook mutated output");
                if let Some(done_mut) = ctx.completed_map.get_mut(id) {
                    done_mut.output.content = content;
                }
            }
            _ => {}
        }

        if ctx.emission_guard.first_time(id) {
            if let Some(done) = ctx.completed_map.get(id) {
                emit_tool_event(done, tx).await;
            }
        }
    }
}

/// Emit the appropriate AgentEvent for a completed tool.
///
/// Called in two distinct contexts:
/// - **Instant failures** (malformed input, denied tools): emitted immediately,
///   before hooks run, because these outcomes cannot be mutated.
/// - **Successful / failed executor tools**: emitted *after* post-tool hooks,
///   in submission order, so that `MutateOutput` decisions are reflected in
///   the event the caller receives.
pub(super) async fn emit_tool_event(done: &CompletedTool, tx: &mpsc::Sender<AgentEvent>) {
    let event = if let Some(kind) = &done.output.failure {
        tracing::debug!(tool = %done.name, kind = ?kind, ms = done.ms, "tool failed");
        AgentEvent::ToolError {
            id: done.id.clone(),
            name: done.name.clone(),
            error: done.output.content.clone(),
            kind: kind.clone(),
            ms: done.ms,
        }
    } else {
        tracing::debug!(tool = %done.name, ms = done.ms, "tool succeeded");
        AgentEvent::ToolDone {
            id: done.id.clone(),
            name: done.name.clone(),
            output: done.output.content.clone(),
            ms: done.ms,
            attempts: done.attempts,
            structured: done.output.structured.clone(),
        }
    };
    tx.send(event).await.ok();
}

/// Guards against double-emission of tool events.
pub(super) struct EmissionGuard {
    emitted: std::collections::HashSet<String>,
}

impl EmissionGuard {
    pub(super) fn new() -> Self {
        Self {
            emitted: std::collections::HashSet::new(),
        }
    }

    /// Returns `true` if this is the first time — caller should emit.
    pub(super) fn first_time(&mut self, id: &str) -> bool {
        self.emitted.insert(id.to_owned())
    }
}
