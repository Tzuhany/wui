// ============================================================================
// AgentTransport — pluggable transport for remote agent delegation.
//
// The transport abstraction decouples the delegation tools from the
// communication mechanism. In-process (LocalTransport using AgentRegistry),
// HTTP, message queues, or the A2A protocol can all implement this trait.
// ============================================================================

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Error ────────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("connection failed: {0}")]
    Connection(String),

    #[error("timeout waiting for remote agent")]
    Timeout,

    #[error("remote agent not found: {0}")]
    NotFound(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

// ── Types ────────────────────────────────────────────────────────────────────

/// Handle to a delegated remote job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteJobHandle {
    pub id: String,
    pub agent_name: String,
}

/// Status of a remote agent job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RemoteAgentStatus {
    Running,
    Done,
    Failed { error: String },
    Cancelled,
}

/// Result of a completed remote agent job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteAgentResult {
    pub status: RemoteAgentStatus,
    pub output: Option<String>,
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Transport layer for delegating work to remote (or local) agents.
///
/// Implementations decide how the prompt reaches the agent and how results
/// come back. The four methods mirror the delegation tool surface:
/// send, status, result, cancel.
#[async_trait]
pub trait AgentTransport: Send + Sync + 'static {
    /// Dispatch a prompt to a named agent. Returns a handle for tracking.
    async fn send(
        &self,
        agent_name: &str,
        prompt: String,
    ) -> Result<RemoteJobHandle, TransportError>;

    /// Check the current status of a delegated job.
    async fn status(&self, handle: &RemoteJobHandle) -> Result<RemoteAgentStatus, TransportError>;

    /// Block until the job completes and return the result.
    async fn result(&self, handle: &RemoteJobHandle) -> Result<RemoteAgentResult, TransportError>;

    /// Cancel a running job.
    async fn cancel(&self, handle: &RemoteJobHandle) -> Result<(), TransportError>;
}

// ── LocalTransport ───────────────────────────────────────────────────────────

/// In-process transport backed by [`AgentRegistry`](crate::AgentRegistry).
///
/// Maps named agents to in-process `wui::Agent` instances and delegates
/// via the existing background job infrastructure. Use this as the default
/// transport when all agents run in the same process.
///
/// ```rust,ignore
/// let mut local = LocalTransport::new();
/// local.register("researcher", researcher_agent);
/// local.register("coder", coder_agent);
///
/// let tools = remote_tools("delegate", "Delegate work", Arc::new(local));
/// ```
pub struct LocalTransport {
    registry: crate::AgentRegistry,
    agents: HashMap<String, wui::Agent>,
}

impl LocalTransport {
    pub fn new() -> Self {
        Self {
            registry: crate::AgentRegistry::new(),
            agents: HashMap::new(),
        }
    }

    /// Register a named agent for in-process delegation.
    pub fn register(&mut self, name: impl Into<String>, agent: wui::Agent) -> &mut Self {
        self.agents.insert(name.into(), agent);
        self
    }
}

impl Default for LocalTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AgentTransport for LocalTransport {
    async fn send(
        &self,
        agent_name: &str,
        prompt: String,
    ) -> Result<RemoteJobHandle, TransportError> {
        let agent = self
            .agents
            .get(agent_name)
            .ok_or_else(|| TransportError::NotFound(agent_name.to_string()))?;
        let id = self.registry.spawn(agent, prompt).await;
        Ok(RemoteJobHandle {
            id: id.to_string(),
            agent_name: agent_name.to_string(),
        })
    }

    async fn status(&self, handle: &RemoteJobHandle) -> Result<RemoteAgentStatus, TransportError> {
        let id = parse_uuid(&handle.id)?;
        let status = self.registry.status(id).await;
        Ok(match status {
            crate::JobStatus::Running => RemoteAgentStatus::Running,
            crate::JobStatus::Done(_) => RemoteAgentStatus::Done,
            crate::JobStatus::Failed(e) => RemoteAgentStatus::Failed { error: e },
            crate::JobStatus::NotFound => {
                return Err(TransportError::NotFound(handle.id.clone()))
            }
        })
    }

    async fn result(&self, handle: &RemoteJobHandle) -> Result<RemoteAgentResult, TransportError> {
        let id = parse_uuid(&handle.id)?;
        let status = self.registry.wait(id).await;
        Ok(match status {
            crate::JobStatus::Done(text) => RemoteAgentResult {
                status: RemoteAgentStatus::Done,
                output: Some(text),
            },
            crate::JobStatus::Failed(e) => RemoteAgentResult {
                status: RemoteAgentStatus::Failed {
                    error: e.clone(),
                },
                output: Some(e),
            },
            crate::JobStatus::NotFound => {
                return Err(TransportError::NotFound(handle.id.clone()))
            }
            crate::JobStatus::Running => RemoteAgentResult {
                status: RemoteAgentStatus::Running,
                output: None,
            },
        })
    }

    async fn cancel(&self, handle: &RemoteJobHandle) -> Result<(), TransportError> {
        let id = parse_uuid(&handle.id)?;
        self.registry.cancel(id).await;
        Ok(())
    }
}

fn parse_uuid(s: &str) -> Result<Uuid, TransportError> {
    Uuid::parse_str(s).map_err(|e| TransportError::Connection(format!("invalid job id: {e}")))
}
