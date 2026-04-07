//! Basic streaming run.
//!
//! Shows `Agent::builder`, `.system()`, `.build()`, `agent.stream()`,
//! and consuming `AgentEvent::TextDelta` events as they arrive.
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

    let mut stream = agent.stream("What is the capital of France? Answer in one sentence.");

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
