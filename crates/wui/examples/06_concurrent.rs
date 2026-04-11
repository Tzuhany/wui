//! Concurrent vs sequential tool execution with `TypedTool`.
//!
//! Shows two tools: one with `concurrent: true` (the default) and one with
//! `concurrent: false`. When the LLM calls both in one turn, the concurrent
//! tool starts immediately while the sequential tool waits its turn.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-... cargo run --example 06_concurrent -p wui --features anthropic

use async_trait::async_trait;
use futures::StreamExt;
use wui::providers::Anthropic;
use wui::{Agent, AgentEvent, PermissionMode, ToolCtx, ToolMeta, ToolOutput, TypedTool};
use wui_macros::ToolInput;

// ── A fast, safe, concurrent tool ────────────────────────────────────────────

#[derive(ToolInput)]
struct LookupInput {
    /// Cache key to look up.
    key: String,
}

/// Simulates a read-only cache lookup. Safe to run alongside other tools.
struct FastLookupTool;

#[async_trait]
impl TypedTool for FastLookupTool {
    type Input = LookupInput;

    fn name(&self) -> &str {
        "fast_lookup"
    }

    fn description(&self) -> &str {
        "Look up a value from the read-only cache. Fast and safe to run concurrently."
    }

    fn meta(&self, _: &LookupInput) -> ToolMeta {
        // concurrent: true (already the default) — may run alongside other tools.
        ToolMeta {
            concurrent: true,
            readonly: true,
            ..ToolMeta::default()
        }
    }

    async fn call_typed(&self, input: LookupInput, _ctx: &ToolCtx) -> ToolOutput {
        ToolOutput::success(format!("cache[{}] = 42", input.key))
    }
}

// ── A slow, stateful, sequential tool ────────────────────────────────────────

#[derive(ToolInput)]
struct WriteLogInput {
    /// Message to append to the audit log.
    message: String,
}

/// Simulates writing to a shared resource. Must not overlap with other writers.
struct WriteLogTool;

#[async_trait]
impl TypedTool for WriteLogTool {
    type Input = WriteLogInput;

    fn name(&self) -> &str {
        "write_log"
    }

    fn description(&self) -> &str {
        "Append an entry to the shared audit log. Must run sequentially."
    }

    fn meta(&self, _: &WriteLogInput) -> ToolMeta {
        // concurrent: false — the executor places this in the serial queue.
        // It will not start until all currently running concurrent tools finish.
        ToolMeta {
            concurrent: false,
            ..ToolMeta::default()
        }
    }

    async fn call_typed(&self, input: WriteLogInput, _ctx: &ToolCtx) -> ToolOutput {
        ToolOutput::success(format!("Logged: {}", input.message))
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

    // Stream manually here so we can observe tool start/done events.
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
