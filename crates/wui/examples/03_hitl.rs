//! Human-in-the-loop (HITL) approval flow.
//!
//! Shows `PermissionMode::Ask`, handling `AgentEvent::Control`, and
//! responding with `handle.approve()` or `handle.deny()`.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-... cargo run --example 03_hitl -p wui --features anthropic

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};
use wui::providers::Anthropic;
use wui::{Agent, AgentEvent, PermissionMode, Tool, ToolCtx, ToolMeta, ToolOutput};

// ── A tool that always asks for permission ────────────────────────────────────

struct DeleteFileTool;

#[async_trait]
impl Tool for DeleteFileTool {
    fn name(&self) -> &str {
        "delete_file"
    }

    fn description(&self) -> &str {
        "Delete a file at the given path."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to delete." }
            },
            "required": ["path"]
        })
    }

    fn meta(&self, _input: &Value) -> ToolMeta {
        // Destructive: the runtime shows this context in the HITL prompt.
        ToolMeta {
            destructive: true,
            ..ToolMeta::default()
        }
    }

    async fn call(&self, input: Value, _ctx: &ToolCtx) -> ToolOutput {
        let path = input["path"].as_str().unwrap_or("(unknown)");
        // Simulated — not actually deleting anything.
        ToolOutput::success(format!("Deleted {path}"))
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;

    let agent = Agent::builder(Anthropic::new(api_key))
        .system("You are a file manager. When asked to delete a file, call delete_file.")
        .tool(DeleteFileTool)
        // Ask mode: every tool call surfaces a ControlHandle for the caller to approve.
        .permission(PermissionMode::Ask)
        .build();

    let mut stream = agent.stream("Please delete /tmp/old-report.txt");

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TextDelta(text) => print!("{text}"),

            // The engine paused and is waiting for a human decision.
            AgentEvent::Control(handle) => {
                let tool_name = handle.request.tool_name().unwrap_or("unknown");
                let description = handle.request.description();
                println!("\n[HITL] Tool '{tool_name}' is requesting permission.");
                println!("[HITL] Description: {description}");

                // In a real application: show a UI prompt and await user input.
                // Here we simulate approval with a simple heuristic.
                // For destructive tools we deny; for others we approve.
                let is_destructive = tool_name == "delete_file";

                if is_destructive {
                    println!("[HITL] Denied — destructive tools require manual review.");
                    handle.deny("User declined the destructive operation.");
                } else {
                    println!("[HITL] Approved.");
                    handle.approve();
                }
            }

            AgentEvent::Done(_) => println!(),
            AgentEvent::Error(e) => return Err(e.into()),
            _ => {}
        }
    }

    Ok(())
}
