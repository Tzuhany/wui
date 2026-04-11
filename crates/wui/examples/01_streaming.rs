//! Basic streaming run.
//!
//! Shows the three entry points on `Agent`: `run()` for fire-and-forget text,
//! `stream().print_text()` for streaming output, and the raw event loop when
//! you need full control over every event.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-... cargo run --example 01_streaming -p wui --features anthropic

use futures::StreamExt;
use wui::providers::Anthropic;
use wui::{Agent, AgentEvent};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;

    let agent = Agent::builder(Anthropic::new(api_key))
        .system("You are a concise, helpful assistant.")
        .build();

    // ── Option 1: fire-and-forget — just give me the text ────────────────────
    let text = agent
        .run("What is the capital of France? Answer in one sentence.")
        .await?;
    println!("{text}");

    // ── Option 2: stream and print as tokens arrive ───────────────────────────
    agent
        .stream("Name three programming languages in one sentence.")
        .print_text()
        .await?;

    // ── Option 3: full event loop — when you need every event ─────────────────
    let mut stream = agent.stream("What is 2 + 2?");
    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TextDelta(text) => print!("{text}"),
            AgentEvent::Done(_) => println!(),
            AgentEvent::Error(e) => return Err(e.into()),
            _ => {}
        }
    }

    Ok(())
}
