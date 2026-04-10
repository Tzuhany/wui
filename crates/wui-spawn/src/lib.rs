// ============================================================================
// wui-spawn — background agent delegation for Wui supervisors.
//
// Use this crate when a sub-task is long-running or when the supervisor needs
// to continue doing other work while waiting for the result. The supervisor
// spawns a sub-agent as a background job (non-blocking, returns a job ID
// immediately) and polls status or awaits the result in a later turn.
//
// ── wui-spawn vs SubAgent ────────────────────────────────────────────────────
//
// `SubAgent` (wui crate, AgentBuilder::sub_agent): wraps one Agent as a
// synchronous Tool. The supervisor calls it, blocks until the sub-agent
// finishes, and the result appears in the same turn. Use for short tasks
// where the result is needed immediately.
//
// `wui-spawn` (this crate): background job registry. The supervisor fires and
// returns a job ID, then checks status or awaits the result across separate
// turns. Use for long-running tasks or when the supervisor should do parallel
// work while waiting. Supports pluggable transports for cross-process
// delegation.
//
// ── Delegation patterns ──────────────────────────────────────────────────────
//
// 1. **In-process** (direct): spawn sub-agents as background tokio tasks.
//    Use `AgentRegistry::delegation_tools()` for the simplest setup.
//
// 2. **Transport-backed** (pluggable): delegate via an `AgentTransport`
//    trait. `LocalTransport` wraps AgentRegistry for in-process use;
//    future transports can use HTTP, message queues, or the A2A protocol
//    for cross-process / cross-network delegation.
//
// In-process tool surface (delegation_tools):
//
//   delegate(prompt)          -> job_id
//   agent_status(job_id)      -> "running" | "done" | "failed" | "not_found"
//   agent_await(job_id)       -> blocks until done, returns result
//   agent_cancel(job_id)      -> cancels the job
//
// Transport-backed tool surface (remote_tools):
//
//   delegate_remote(agent_name, prompt) -> job_id
//   remote_status(job_id)               -> status
//   remote_await(job_id)                -> blocks until done
//   remote_cancel(job_id)               -> cancels
// ============================================================================

mod registry;
pub mod remote_tools;
mod tools;
pub mod transport;

use std::sync::Arc;

pub use registry::{AgentRegistry, JobStatus};
pub use tools::{AgentAwait, AgentCancel, AgentStatus, DelegateAgent};

pub use remote_tools::{remote_tools, RemoteAwait, RemoteCancel, RemoteDelegate, RemoteStatus};
pub use transport::{
    AgentTransport, LocalTransport, RemoteAgentResult, RemoteAgentStatus, RemoteJobHandle,
    TransportError,
};

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

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

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

    #[test]
    fn remote_tools_count_and_names() {
        let transport = Arc::new(LocalTransport::new());
        let tools = remote_tools("delegate", "Delegate work", transport);

        assert_eq!(tools.len(), 4);

        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"delegate"));
        assert!(names.contains(&"remote_status"));
        assert!(names.contains(&"remote_await"));
        assert!(names.contains(&"remote_cancel"));
    }

    #[tokio::test]
    async fn local_transport_send_and_await() {
        let provider = MockProvider::new(vec![MockProvider::text("result from worker")]);
        let agent = wui::Agent::builder(provider)
            .permission(PermissionMode::Auto)
            .build();

        let mut transport = LocalTransport::new();
        transport.register("worker", agent);
        let transport = Arc::new(transport);

        let handle = transport.send("worker", "do work".into()).await.unwrap();
        assert_eq!(handle.agent_name, "worker");

        let result = transport.result(&handle).await.unwrap();
        assert!(result.output.unwrap().contains("result from worker"));
    }

    #[tokio::test]
    async fn local_transport_not_found() {
        let transport = LocalTransport::new();
        let result = transport.send("nonexistent", "hello".into()).await;
        assert!(result.is_err());
    }
}
