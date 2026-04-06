// Hook runner — applies all registered hooks in order, stopping at the first Block.

use std::sync::Arc;

use wuhu_core::hook::{Hook, HookDecision, HookEvent};
use wuhu_core::tool::ToolOutput;

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

    pub async fn pre_complete(&self, response: &str) -> HookDecision {
        let event = HookEvent::PreComplete { response };
        self.run(&event).await
    }

    async fn run(&self, event: &HookEvent<'_>) -> HookDecision {
        for hook in &self.hooks {
            if !hook.handles(event) { continue; }
            let decision = hook.evaluate(event).await;
            if decision.is_blocked() { return decision; }
        }
        HookDecision::Allow
    }
}
