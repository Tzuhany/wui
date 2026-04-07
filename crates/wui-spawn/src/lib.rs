// ============================================================================
// wui-spawn — background agent delegation for Wui supervisors.
//
// This crate is a companion, not the product. It provides one specific pattern:
// a supervisor agent that spawns sub-agents in the background and queries their
// status across turns. Whether that pattern is right for your application is
// your decision.
//
// What this crate is:
//   - A background job registry (AgentRegistry) built on tokio tasks.
//   - Four delegation tools that surface job control to the LLM.
//   - A concrete implementation of supervisor/worker agent patterns.
//
// What this crate is NOT:
//   - A general agent orchestration framework.
//   - A declaration that background delegation is the right model for all agents.
//
// Tool surface:
//
//   delegate(prompt)          → job_id (spawns sub-agent, returns immediately)
//   agent_status(job_id)      → "running" | "done: <result>" | "failed: <err>"
//   agent_await(job_id)       → blocks until done, returns result
//   agent_cancel(job_id)      → cancels the job
//
// Usage:
//
//   let registry = AgentRegistry::new();
//   let tools    = registry.delegation_tools("researcher", "...", researcher_agent);
//
//   let supervisor = Agent::builder(provider)
//       .tools(tools)
//       .build();
// ============================================================================

mod registry;
mod tools;

use std::sync::Arc;

pub use registry::{AgentRegistry, JobStatus};
pub use tools::{AgentAwait, AgentCancel, AgentStatus, DelegateAgent};

impl AgentRegistry {
    /// Create a set of four tools for delegating to a named sub-agent.
    ///
    /// Returns `DelegateAgent`, `AgentStatus`, `AgentAwait`, and `AgentCancel`
    /// tools, all sharing the same registry.
    pub fn delegation_tools(
        &self,
        name: impl Into<String>,
        description: impl Into<String>,
        agent: wui::Agent,
    ) -> Vec<Arc<dyn wui_core::tool::Tool>> {
        let registry = Arc::new(self.clone());
        vec![
            Arc::new(DelegateAgent::new(
                name,
                description,
                agent,
                Arc::clone(&registry),
            )),
            Arc::new(AgentStatus::new(Arc::clone(&registry))),
            Arc::new(AgentAwait::new(Arc::clone(&registry))),
            Arc::new(AgentCancel::new(Arc::clone(&registry))),
        ]
    }
}

// ── Smoke tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use wui::PermissionMode;
    use wui_eval::MockProvider;

    use super::*;

    /// `delegation_tools` returns 4 tools: the delegate tool (user-named) plus
    /// the three fixed control tools (agent_status, agent_await, agent_cancel).
    #[test]
    fn delegation_tools_count_and_names() {
        let provider = MockProvider::new(vec![]);
        let sub_agent = wui::Agent::builder(provider)
            .permission(PermissionMode::Auto)
            .build();

        let registry = AgentRegistry::new();
        let tools = registry.delegation_tools("worker", "Does work.", sub_agent);

        assert_eq!(tools.len(), 4, "expected 4 delegation tools");

        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        // The delegate tool uses the caller-supplied name.
        assert!(names.contains(&"worker"), "missing delegate tool 'worker'");
        // Control tools always use fixed names.
        assert!(names.contains(&"agent_status"), "missing agent_status");
        assert!(names.contains(&"agent_await"), "missing agent_await");
        assert!(names.contains(&"agent_cancel"), "missing agent_cancel");
    }
}
