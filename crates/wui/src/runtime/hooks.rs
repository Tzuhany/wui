// Hook runner — applies all registered hooks in order.
//
// Blocks short-circuit immediately. Supported mutations are threaded forward
// so later hooks see the updated input/output/response.

use std::sync::Arc;

use wui_core::event::{RunStopReason, RunSummary};
use wui_core::hook::SessionId;
use wui_core::hook::{Hook, HookDecision, HookEvent};
use wui_core::message::Message;
use wui_core::tool::ToolOutput;

pub struct HookRunner {
    hooks: Vec<Arc<dyn Hook>>,
}

impl HookRunner {
    pub fn new(hooks: Vec<Arc<dyn Hook>>) -> Self {
        Self { hooks }
    }

    #[must_use]
    pub async fn pre_tool_use(&self, name: &str, input: &serde_json::Value) -> HookDecision {
        let mut current_input = None;

        for hook in &self.hooks {
            let input_ref = current_input.as_ref().unwrap_or(input);
            let event = HookEvent::PreToolUse {
                name,
                input: input_ref,
            };
            if !hook.handles(&event) {
                continue;
            }

            match hook.evaluate(&event).await {
                HookDecision::Allow | HookDecision::MutateOutput { .. } => {}
                HookDecision::Mutate { input } => current_input = Some(input),
                HookDecision::Block { reason } => return HookDecision::Block { reason },
            }
        }

        match current_input {
            Some(input) => HookDecision::Mutate { input },
            None => HookDecision::Allow,
        }
    }

    #[must_use]
    pub async fn post_tool_use(&self, name: &str, output: &ToolOutput) -> HookDecision {
        let mut current_output = None;

        for hook in &self.hooks {
            let output_ref = current_output.as_ref().unwrap_or(output);
            let event = HookEvent::PostToolUse {
                name,
                output: output_ref,
            };
            if !hook.handles(&event) {
                continue;
            }

            match hook.evaluate(&event).await {
                HookDecision::Allow | HookDecision::Mutate { .. } => {}
                HookDecision::MutateOutput { content } => {
                    let out = current_output.get_or_insert_with(|| output.clone());
                    out.content = content;
                }
                HookDecision::Block { reason } => return HookDecision::Block { reason },
            }
        }

        match current_output {
            Some(output) => HookDecision::MutateOutput {
                content: output.content,
            },
            None => HookDecision::Allow,
        }
    }

    #[must_use]
    pub async fn post_tool_failure(
        &self,
        name: &str,
        input: &serde_json::Value,
        output: &ToolOutput,
    ) -> HookDecision {
        let event = HookEvent::PostToolFailure {
            name,
            input,
            output,
        };
        self.run_blocking_only(&event).await
    }

    /// Fire the `PreCompact` hook before context compression.
    ///
    /// Returns `Block { reason }` when the hook wants to inject preservation
    /// context into the conversation before the summariser runs. The caller
    /// is responsible for inserting the reason as a system message.
    #[must_use]
    pub async fn pre_compact(&self, messages: &[Message]) -> HookDecision {
        let event = HookEvent::PreCompact { messages };
        self.run_blocking_only(&event).await
    }

    /// Fire the `PreStop` hook before any run termination.
    ///
    /// `stop_reason` tells the hook why the run is ending.
    /// `stop_hook_active` is `true` when this hook already blocked the current
    /// stop attempt once — a signal that the hook should return `Allow` to
    /// avoid an infinite loop.
    #[must_use]
    pub async fn pre_stop(
        &self,
        response: &str,
        stop_reason: RunStopReason,
        stop_hook_active: bool,
    ) -> HookDecision {
        let mut current_response = None;

        for hook in &self.hooks {
            let response_ref = current_response.as_deref().unwrap_or(response);
            let event = HookEvent::PreStop {
                response: response_ref,
                stop_reason: stop_reason.clone(),
                stop_hook_active,
            };
            if !hook.handles(&event) {
                continue;
            }

            match hook.evaluate(&event).await {
                HookDecision::Allow | HookDecision::Mutate { .. } => {}
                HookDecision::MutateOutput { content } => current_response = Some(content),
                HookDecision::Block { reason } => return HookDecision::Block { reason },
            }
        }

        match current_response {
            Some(content) => HookDecision::MutateOutput { content },
            None => HookDecision::Allow,
        }
    }

    // ── Lifecycle notifications (fire-and-forget) ─────────────────────

    pub async fn notify_session_start(&self, session_id: &SessionId) {
        self.notify(&HookEvent::SessionStart { session_id }).await;
    }

    pub async fn notify_turn_start(&self, messages: &[Message]) {
        self.notify(&HookEvent::TurnStart { messages }).await;
    }

    pub async fn notify_turn_end(&self, summary: &RunSummary) {
        self.notify(&HookEvent::TurnEnd { summary }).await;
    }

    pub async fn notify_subagent_start(&self, name: &str, prompt: &str) {
        self.notify(&HookEvent::SubagentStart { name, prompt })
            .await;
    }

    pub async fn notify_subagent_end(&self, name: &str, result: Result<&str, &str>) {
        self.notify(&HookEvent::SubagentEnd { name, result }).await;
    }

    /// Fire a lifecycle notification — decision is ignored.
    async fn notify(&self, event: &HookEvent<'_>) {
        for hook in &self.hooks {
            if hook.handles(event) {
                let _ = hook.evaluate(event).await;
            }
        }
    }

    async fn run_blocking_only(&self, event: &HookEvent<'_>) -> HookDecision {
        for hook in &self.hooks {
            if !hook.handles(event) {
                continue;
            }
            let decision = hook.evaluate(event).await;
            if let HookDecision::Block { reason } = decision {
                return HookDecision::Block { reason };
            }
        }
        HookDecision::Allow
    }
}
