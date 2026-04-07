//! Custom tool with `ToolMeta`.
//!
//! Shows implementing the `Tool` trait with `fn meta()` returning a
//! `ToolMeta { readonly: true, ..Default::default() }`, registering the
//! tool via `.tool()`, and running the agent with `PermissionMode::Auto`.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-... cargo run --example 02_tools -p wui --features anthropic

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};
use wui::providers::Anthropic;
use wui::{Agent, AgentEvent, PermissionMode, Tool, ToolCtx, ToolMeta, ToolOutput};

// ── A simple read-only tool ───────────────────────────────────────────────────

struct CurrentTimeTool;

#[async_trait]
impl Tool for CurrentTimeTool {
    fn name(&self) -> &str {
        "current_time"
    }

    fn description(&self) -> &str {
        "Returns the current UTC time as an ISO 8601 string."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    fn meta(&self, _input: &Value) -> ToolMeta {
        // Read-only: no external state is modified. Safe to run concurrently.
        ToolMeta {
            readonly: true,
            ..ToolMeta::default()
        }
    }

    async fn call(&self, _input: Value, _ctx: &ToolCtx) -> ToolOutput {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        ToolOutput::success(format!("Current UNIX timestamp: {now}"))
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;

    let agent = Agent::builder(Anthropic::new(api_key))
        .system("You are a helpful assistant with access to real-time tools.")
        .tool(CurrentTimeTool)
        .permission(PermissionMode::Auto)
        .build();

    let mut stream = agent.stream("What is the current UNIX timestamp?");

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TextDelta(text) => print!("{text}"),
            AgentEvent::ToolStart { name, .. } => eprintln!("\n[tool call: {name}]"),
            AgentEvent::ToolDone { name, output, .. } => {
                eprintln!("[tool done: {name}] → {output}")
            }
            AgentEvent::Done(_) => println!(),
            AgentEvent::Error(e) => return Err(e.into()),
            _ => {}
        }
    }

    Ok(())
}
