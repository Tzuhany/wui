//! Checkpoint and resume.
//!
//! Shows `InMemoryCheckpointStore`, `.checkpoint(store, "run-id")` on the
//! builder, running an agent, then creating a new agent with the same store
//! to resume from where it left off.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-... cargo run --example 07_checkpoint -p wui --features anthropic

use futures::StreamExt;
use wui::providers::Anthropic;
use wui::{Agent, AgentEvent, InMemoryCheckpointStore, PermissionMode};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;

    // A shared, cloneable checkpoint store.
    // InMemoryCheckpointStore is Clone — the underlying data is Arc-wrapped.
    let store = InMemoryCheckpointStore::new();
    let run_id = "demo-run-001";

    // ── First run ─────────────────────────────────────────────────────────────
    println!("=== First run ===");
    {
        let agent = Agent::builder(Anthropic::new(api_key.clone()))
            .system("You are a helpful assistant.")
            .permission(PermissionMode::Auto)
            // Save a checkpoint after every tool-use iteration.
            // On restart with the same run_id, the run continues from here.
            .checkpoint(store.clone(), run_id)
            .build();

        let mut stream = agent.stream("Count to 3, one number per sentence.");
        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::TextDelta(text) => print!("{text}"),
                AgentEvent::Done(summary) => {
                    println!();
                    println!("First run complete: {:?}", summary.stop_reason);
                }
                AgentEvent::Error(e) => return Err(e.into()),
                _ => {}
            }
        }
    }

    // ── Resume run ────────────────────────────────────────────────────────────
    // In practice, a resumed run would use different initial prompt text —
    // the real value is recovering a long multi-tool run after a crash.
    // Here we show the mechanics: same store + same run_id = continuation.
    println!("\n=== Resumed run (same store, same run_id) ===");
    {
        let agent = Agent::builder(Anthropic::new(api_key))
            .system("You are a helpful assistant.")
            .permission(PermissionMode::Auto)
            .checkpoint(store, run_id)
            .build();

        let mut stream = agent.stream("Now count from 4 to 6.");
        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::TextDelta(text) => print!("{text}"),
                AgentEvent::Done(summary) => {
                    println!();
                    println!("Resume run complete: {:?}", summary.stop_reason);
                }
                AgentEvent::Error(e) => return Err(e.into()),
                _ => {}
            }
        }
    }

    Ok(())
}
