use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use uuid::Uuid;

use wui_core::runner::AgentRunner;
use wui_core::tool::{Tool, ToolCtx, ToolInput, ToolMeta, ToolOutput};

use crate::registry::{AgentRegistry, JobStatus};

// ── DelegateAgent ─────────────────────────────────────────────────────────────

pub struct DelegateAgent {
    tool_name: String,
    tool_desc: String,
    agent: Arc<dyn AgentRunner>,
    registry: Arc<AgentRegistry>,
}

impl DelegateAgent {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        agent: impl AgentRunner,
        registry: Arc<AgentRegistry>,
    ) -> Self {
        Self {
            tool_name: name.into(),
            tool_desc: description.into(),
            agent: Arc::new(agent),
            registry,
        }
    }
}

#[async_trait]
impl Tool for DelegateAgent {
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
                "prompt": {
                    "type": "string",
                    "description": "The task to delegate to this sub-agent"
                }
            },
            "required": ["prompt"]
        })
    }

    async fn call(&self, input: Value, _ctx: &ToolCtx) -> ToolOutput {
        let inp = ToolInput(&input);
        let prompt = match inp.required_str("prompt") {
            Ok(p) => p.to_string(),
            Err(e) => return ToolOutput::invalid_input(e),
        };
        let id = self.registry.spawn(Arc::clone(&self.agent), prompt).await;
        ToolOutput::success(format!("Sub-agent started. Job ID: {id}"))
    }
}

// ── AgentStatus ───────────────────────────────────────────────────────────────

pub struct AgentStatus {
    registry: Arc<AgentRegistry>,
}

impl AgentStatus {
    pub fn new(registry: Arc<AgentRegistry>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for AgentStatus {
    fn name(&self) -> &str {
        "agent_status"
    }
    fn description(&self) -> &str {
        "Check the status of a running sub-agent by job ID"
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
                "job_id": { "type": "string", "description": "The job ID returned by the delegation tool" }
            },
            "required": ["job_id"]
        })
    }

    async fn call(&self, input: Value, _ctx: &ToolCtx) -> ToolOutput {
        let inp = ToolInput(&input);
        let id_str = match inp.required_str("job_id") {
            Ok(s) => s,
            Err(e) => return ToolOutput::invalid_input(e),
        };
        let id = match Uuid::parse_str(id_str) {
            Ok(id) => id,
            Err(e) => return ToolOutput::invalid_input(format!("invalid job_id: {e}")),
        };
        let status = self.registry.status(id).await;
        match status {
            JobStatus::Running => ToolOutput::success("running"),
            JobStatus::Done(r) => ToolOutput::success(format!("done: {r}")),
            JobStatus::Failed(e) => ToolOutput::success(format!("failed: {e}")),
            JobStatus::NotFound => ToolOutput::success("not_found"),
        }
    }
}

// ── AgentAwait ────────────────────────────────────────────────────────────────

pub struct AgentAwait {
    registry: Arc<AgentRegistry>,
}

impl AgentAwait {
    pub fn new(registry: Arc<AgentRegistry>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for AgentAwait {
    fn name(&self) -> &str {
        "agent_await"
    }
    fn description(&self) -> &str {
        "Block until a sub-agent job completes, then return its result"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "job_id": { "type": "string", "description": "The job ID returned by the delegation tool" }
            },
            "required": ["job_id"]
        })
    }

    async fn call(&self, input: Value, _ctx: &ToolCtx) -> ToolOutput {
        let inp = ToolInput(&input);
        let id_str = match inp.required_str("job_id") {
            Ok(s) => s,
            Err(e) => return ToolOutput::invalid_input(e),
        };
        let id = match Uuid::parse_str(id_str) {
            Ok(id) => id,
            Err(e) => return ToolOutput::invalid_input(format!("invalid job_id: {e}")),
        };
        let status = self.registry.wait(id).await;
        match status {
            JobStatus::Done(r) => ToolOutput::success(r),
            JobStatus::Failed(e) => ToolOutput::error(e),
            JobStatus::NotFound => ToolOutput::error("job not found"),
            JobStatus::Running => ToolOutput::error("still running (unexpected)"),
        }
    }
}

// ── AgentCancel ───────────────────────────────────────────────────────────────

pub struct AgentCancel {
    registry: Arc<AgentRegistry>,
}

impl AgentCancel {
    pub fn new(registry: Arc<AgentRegistry>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for AgentCancel {
    fn name(&self) -> &str {
        "agent_cancel"
    }
    fn description(&self) -> &str {
        "Cancel a running sub-agent job"
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
        let id_str = match inp.required_str("job_id") {
            Ok(s) => s,
            Err(e) => return ToolOutput::invalid_input(e),
        };
        let id = match Uuid::parse_str(id_str) {
            Ok(id) => id,
            Err(e) => return ToolOutput::invalid_input(format!("invalid job_id: {e}")),
        };
        let cancelled = self.registry.cancel(id).await;
        if cancelled {
            ToolOutput::success("cancelled")
        } else {
            ToolOutput::success("not_found")
        }
    }
}
