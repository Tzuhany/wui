//! Multi-turn session conversation.
//!
//! Shows `agent.session()`, `session.send()`, consuming events, and calling
//! `send` again with the next message — the session maintains full context.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-... cargo run --example 04_session -p wui --features anthropic

use futures::StreamExt;
use wui::providers::Anthropic;
use wui::{Agent, AgentEvent, PermissionMode};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;

    let agent = Agent::builder(Anthropic::new(api_key))
        .system("You are a helpful assistant. Keep answers brief.")
        .permission(PermissionMode::Auto)
        .build();

    // A session owns message history across turns.
    let session = agent.session("demo-session").await;

    // ── Turn 1 ────────────────────────────────────────────────────────────────
    println!("User: My favourite colour is blue.");
    let mut stream = session.send("My favourite colour is blue. Just acknowledge.").await;

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TextDelta(text) => print!("{text}"),
            AgentEvent::Done(_) => println!(),
            AgentEvent::Error(e) => return Err(e.into()),
            _ => {}
        }
    }

    // ── Turn 2 ────────────────────────────────────────────────────────────────
    // The session remembers the first turn automatically.
    println!("\nUser: What is my favourite colour?");
    let mut stream = session.send("What is my favourite colour?").await;

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
