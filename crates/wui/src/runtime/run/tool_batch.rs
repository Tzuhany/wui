// ── Post-tool hooks and event emission ──────────────────────────────────────

use tokio::sync::mpsc;

use wui_core::event::AgentEvent;
use wui_core::hook::HookDecision;
use wui_core::message::Message;

use super::history::system_reminder_msg;
use super::RunConfig;
use super::{CompletedTool, IterationCtx, ToolExecutor};

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
            let null_input = serde_json::Value::Null;
            let input = ctx.tool_input(id).unwrap_or(&null_input);
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
