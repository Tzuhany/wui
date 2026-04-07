// Hook runner — applies all registered hooks in order, stopping at the first
// non-Allow decision.

use std::sync::Arc;

use wui_core::event::RunStopReason;
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

    pub async fn pre_tool_use(&self, name: &str, input: &serde_json::Value) -> HookDecision {
        let event = HookEvent::PreToolUse { name, input };
        self.run(&event).await
    }

    pub async fn post_tool_use(&self, name: &str, output: &ToolOutput) -> HookDecision {
        let event = HookEvent::PostToolUse { name, output };
        self.run(&event).await
    }

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
        self.run(&event).await
    }

    /// Fire the `PreCompact` hook before context compression.
    ///
    /// Returns `Block { reason }` when the hook wants to inject preservation
    /// context into the conversation before the summariser runs. The caller
    /// is responsible for inserting the reason as a system message.
    pub async fn pre_compact(&self, messages: &[Message]) -> HookDecision {
        let event = HookEvent::PreCompact { messages };
        self.run(&event).await
    }

    /// Fire the `PreStop` hook before any run termination.
    ///
    /// `stop_reason` tells the hook why the run is ending.
    /// `stop_hook_active` is `true` when this hook already blocked the current
    /// stop attempt once — a signal that the hook should return `Allow` to
    /// avoid an infinite loop.
    pub async fn pre_stop(
        &self,
        response: &str,
        stop_reason: RunStopReason,
        stop_hook_active: bool,
    ) -> HookDecision {
        let event = HookEvent::PreStop {
            response,
            stop_reason,
            stop_hook_active,
        };
        self.run(&event).await
    }

    async fn run(&self, event: &HookEvent<'_>) -> HookDecision {
        for hook in &self.hooks {
            if !hook.handles(event) {
                continue;
            }
            let decision = hook.evaluate(event).await;
            // Stop on any non-Allow decision: Block terminates immediately,
            // Mutate/MutateOutput are applied and the chain stops there
            // (subsequent hooks would not have seen the mutated value anyway).
            if !matches!(decision, HookDecision::Allow) {
                return decision;
            }
        }
        HookDecision::Allow
    }
}
