//! Human-in-the-loop (HITL) approval flow.
//!
//! Shows `PermissionMode::Ask`, handling `AgentEvent::Control`, and
//! responding with `handle.approve()` or `handle.deny("reason")`.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-... cargo run --example 03_hitl -p wui --features anthropic

use async_trait::async_trait;
use futures::StreamExt;
use wui::providers::Anthropic;
use wui::{Agent, AgentEvent, PermissionMode, ToolCtx, ToolMeta, ToolOutput, TypedTool};
use wui_macros::ToolInput;

// ── Tool definition ───────────────────────────────────────────────────────────

#[derive(ToolInput)]
struct DeleteFileInput {
    /// Absolute path to the file to delete.
    path: String,
}

/// Simulates deleting a file — requires human approval on every call.
struct DeleteFileTool;

#[async_trait]
impl TypedTool for DeleteFileTool {
    type Input = DeleteFileInput;

    fn name(&self) -> &str {
        "delete_file"
    }

    fn description(&self) -> &str {
        "Delete a file at the given path."
    }

    fn meta(&self, _: &DeleteFileInput) -> ToolMeta {
        // destructive: true — shown in the HITL approval prompt.
        ToolMeta {
            destructive: true,
            ..ToolMeta::default()
        }
    }

    async fn call_typed(&self, input: DeleteFileInput, _ctx: &ToolCtx) -> ToolOutput {
        // Simulated — not actually deleting anything.
        ToolOutput::success(format!("Deleted {}", input.path))
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
                println!("\n[HITL] '{tool_name}' is requesting permission.");
                println!("[HITL] {description}");

                // In a real application: show a UI prompt and await user input.
                // Here we deny destructive operations and approve everything else.
                if tool_name == "delete_file" {
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
