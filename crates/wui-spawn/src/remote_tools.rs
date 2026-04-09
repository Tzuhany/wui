// ============================================================================
// Remote delegation tools — transport-backed agent delegation.
//
// Four tools mirror the in-process delegation pattern (DelegateAgent, etc.)
// but route through an AgentTransport instead of direct spawning.
// ============================================================================

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use wui_core::tool::{Tool, ToolCtx, ToolInput, ToolMeta, ToolOutput};

use crate::transport::{AgentTransport, RemoteAgentStatus, RemoteJobHandle};

// ── RemoteDelegate ───────────────────────────────────────────────────────────

/// Dispatch a prompt to a named remote agent.
pub struct RemoteDelegate {
    tool_name: String,
    tool_desc: String,
    transport: Arc<dyn AgentTransport>,
}

impl RemoteDelegate {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        transport: Arc<dyn AgentTransport>,
    ) -> Self {
        Self {
            tool_name: name.into(),
            tool_desc: description.into(),
            transport,
        }
    }
}

#[async_trait]
impl Tool for RemoteDelegate {
    fn name(&self) -> &str {
        &self.tool_name
    }
    fn description(&self) -> &str {
        &self.tool_desc
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Name of the remote agent to delegate to"
                },
                "prompt": {
                    "type": "string",
                    "description": "The task to delegate"
                }
            },
            "required": ["agent_name", "prompt"]
        })
    }

    async fn call(&self, input: Value, _ctx: &ToolCtx) -> ToolOutput {
        let inp = ToolInput(&input);
        let agent_name = match inp.required_str("agent_name") {
            Ok(s) => s,
            Err(e) => return ToolOutput::invalid_input(e),
        };
        let prompt = match inp.required_str("prompt") {
            Ok(s) => s.to_string(),
            Err(e) => return ToolOutput::invalid_input(e),
        };
        match self.transport.send(agent_name, prompt).await {
            Ok(handle) => ToolOutput::success(format!(
                "Remote agent '{}' started. Job ID: {}",
                handle.agent_name, handle.id
            ))
            .with_structured(json!({
                "job_id": handle.id,
                "agent_name": handle.agent_name,
            })),
            Err(e) => ToolOutput::error(e.to_string()),
        }
    }
}

// ── RemoteStatus ─────────────────────────────────────────────────────────────

/// Check the status of a remote agent job.
pub struct RemoteStatus {
    transport: Arc<dyn AgentTransport>,
}

impl RemoteStatus {
    pub fn new(transport: Arc<dyn AgentTransport>) -> Self {
        Self { transport }
    }
}

#[async_trait]
impl Tool for RemoteStatus {
    fn name(&self) -> &str {
        "remote_status"
    }
    fn description(&self) -> &str {
        "Check the status of a remote agent job by job ID"
    }
    fn meta(&self, _input: &Value) -> ToolMeta {
        ToolMeta {
            readonly: true,
            concurrent: true,
            ..ToolMeta::default()
        }
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "job_id": { "type": "string", "description": "The job ID to check" }
            },
            "required": ["job_id"]
        })
    }

    async fn call(&self, input: Value, _ctx: &ToolCtx) -> ToolOutput {
        let inp = ToolInput(&input);
        let job_id = match inp.required_str("job_id") {
            Ok(s) => s,
            Err(e) => return ToolOutput::invalid_input(e),
        };
        let handle = RemoteJobHandle {
            id: job_id.to_string(),
            agent_name: String::new(),
        };
        match self.transport.status(&handle).await {
            Ok(status) => {
                let msg = match &status {
                    RemoteAgentStatus::Running => "running".to_string(),
                    RemoteAgentStatus::Done => "done".to_string(),
                    RemoteAgentStatus::Failed { error } => format!("failed: {error}"),
                    RemoteAgentStatus::Cancelled => "cancelled".to_string(),
                };
                ToolOutput::success(msg)
            }
            Err(e) => ToolOutput::error(e.to_string()),
        }
    }
}

// ── RemoteAwait ──────────────────────────────────────────────────────────────

/// Block until a remote agent job completes and return the result.
pub struct RemoteAwait {
    transport: Arc<dyn AgentTransport>,
}

impl RemoteAwait {
    pub fn new(transport: Arc<dyn AgentTransport>) -> Self {
        Self { transport }
    }
}

#[async_trait]
impl Tool for RemoteAwait {
    fn name(&self) -> &str {
        "remote_await"
    }
    fn description(&self) -> &str {
        "Wait for a remote agent job to complete and return its result"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "job_id": { "type": "string", "description": "The job ID to await" }
            },
            "required": ["job_id"]
        })
    }

    async fn call(&self, input: Value, _ctx: &ToolCtx) -> ToolOutput {
        let inp = ToolInput(&input);
        let job_id = match inp.required_str("job_id") {
            Ok(s) => s,
            Err(e) => return ToolOutput::invalid_input(e),
        };
        let handle = RemoteJobHandle {
            id: job_id.to_string(),
            agent_name: String::new(),
        };
        match self.transport.result(&handle).await {
            Ok(result) => match result.output {
                Some(text) => ToolOutput::success(text),
                None => ToolOutput::success("completed with no output"),
            },
            Err(e) => ToolOutput::error(e.to_string()),
        }
    }
}

// ── RemoteCancel ─────────────────────────────────────────────────────────────

/// Cancel a running remote agent job.
pub struct RemoteCancel {
    transport: Arc<dyn AgentTransport>,
}

impl RemoteCancel {
    pub fn new(transport: Arc<dyn AgentTransport>) -> Self {
        Self { transport }
    }
}

#[async_trait]
impl Tool for RemoteCancel {
    fn name(&self) -> &str {
        "remote_cancel"
    }
    fn description(&self) -> &str {
        "Cancel a running remote agent job"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "job_id": { "type": "string", "description": "The job ID to cancel" }
            },
            "required": ["job_id"]
        })
    }

    async fn call(&self, input: Value, _ctx: &ToolCtx) -> ToolOutput {
        let inp = ToolInput(&input);
        let job_id = match inp.required_str("job_id") {
            Ok(s) => s,
            Err(e) => return ToolOutput::invalid_input(e),
        };
        let handle = RemoteJobHandle {
            id: job_id.to_string(),
            agent_name: String::new(),
        };
        match self.transport.cancel(&handle).await {
            Ok(()) => ToolOutput::success("cancelled"),
            Err(e) => ToolOutput::error(e.to_string()),
        }
    }
}

// ── Factory ──────────────────────────────────────────────────────────────────

/// Create a set of four remote delegation tools sharing one transport.
///
/// Returns `RemoteDelegate` (with the caller-supplied name), `RemoteStatus`,
/// `RemoteAwait`, and `RemoteCancel`.
pub fn remote_tools(
    name: impl Into<String>,
    description: impl Into<String>,
    transport: Arc<dyn AgentTransport>,
) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(RemoteDelegate::new(
            name,
            description,
            Arc::clone(&transport),
        )),
        Arc::new(RemoteStatus::new(Arc::clone(&transport))),
        Arc::new(RemoteAwait::new(Arc::clone(&transport))),
        Arc::new(RemoteCancel::new(transport)),
    ]
}
