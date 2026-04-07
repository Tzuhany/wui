//! Concurrent vs sequential tool execution.
//!
//! Shows two tools: one with `concurrent: true` (the default) and one with
//! `concurrent: false`. When the LLM calls both, the concurrent tool starts
//! immediately while the sequential tool waits its turn.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-... cargo run --example 06_concurrent -p wui --features anthropic

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};
use wui::providers::Anthropic;
use wui::{Agent, AgentEvent, PermissionMode, Tool, ToolCtx, ToolMeta, ToolOutput};

// ── A fast, safe, concurrent tool ────────────────────────────────────────────

/// Simulates a read-only database lookup. Safe to run alongside other tools.
struct FastLookupTool;

#[async_trait]
impl Tool for FastLookupTool {
    fn name(&self) -> &str {
        "fast_lookup"
    }

    fn description(&self) -> &str {
        "Look up a value from the read-only cache. Fast and safe to run concurrently."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "key": { "type": "string" }
            },
            "required": ["key"]
        })
    }

    fn meta(&self, _input: &Value) -> ToolMeta {
        // concurrent: true (already the default) — may run alongside other tools.
        ToolMeta {
            concurrent: true,
            readonly: true,
            ..ToolMeta::default()
        }
    }

    async fn call(&self, input: Value, _ctx: &ToolCtx) -> ToolOutput {
        let key = input["key"].as_str().unwrap_or("?");
        ToolOutput::success(format!("cache[{key}] = 42"))
    }
}

// ── A slow, stateful, sequential tool ────────────────────────────────────────

/// Simulates writing to a shared resource. Must not overlap with other writers.
struct WriteLogTool;

#[async_trait]
impl Tool for WriteLogTool {
    fn name(&self) -> &str {
        "write_log"
    }

    fn description(&self) -> &str {
        "Append an entry to the shared audit log. Must run sequentially."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message": { "type": "string" }
            },
            "required": ["message"]
        })
    }

    fn meta(&self, _input: &Value) -> ToolMeta {
        // concurrent: false — the executor places this in the serial queue.
        // It will not start until all currently running concurrent tools finish.
        ToolMeta {
            concurrent: false,
            ..ToolMeta::default()
        }
    }

    async fn call(&self, input: Value, _ctx: &ToolCtx) -> ToolOutput {
        let msg = input["message"].as_str().unwrap_or("(empty)");
        ToolOutput::success(format!("Logged: {msg}"))
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;

    let agent = Agent::builder(Anthropic::new(api_key))
        .system(
            "You have two tools: fast_lookup and write_log. \
             When asked, call both in one response.",
        )
        .tool(FastLookupTool)
        .tool(WriteLogTool)
        .permission(PermissionMode::Auto)
        .build();

    let mut stream =
        agent.stream("Look up key 'user_count' and write 'session started' to the log.");

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TextDelta(text) => print!("{text}"),
            AgentEvent::ToolStart { name, .. } => eprintln!("\n[tool start: {name}]"),
            AgentEvent::ToolDone { name, output, .. } => {
                eprintln!("[tool done:  {name}] → {output}")
            }
            AgentEvent::Done(_) => println!(),
            AgentEvent::Error(e) => return Err(e.into()),
            _ => {}
        }
    }

    Ok(())
}
