//! Multi-turn session conversation.
//!
//! Shows `agent.session()`, `session.send()`, and how the session maintains
//! full context across turns automatically.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-... cargo run --example 04_session -p wui --features anthropic

use futures::{Stream, StreamExt};
use wui::providers::Anthropic;
use wui::{Agent, AgentError, AgentEvent, PermissionMode};

/// Drain a session turn stream, printing text and returning on Done or Error.
async fn print_turn(mut stream: impl Stream<Item = AgentEvent> + Unpin) -> Result<(), AgentError> {
    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TextDelta(text) => print!("{text}"),
            AgentEvent::Done(_) => {
                println!();
                return Ok(());
            }
            AgentEvent::Error(e) => return Err(e),
            _ => {}
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;

    let agent = Agent::builder(Anthropic::new(api_key))
        .system("You are a helpful assistant. Keep answers brief.")
        .permission(PermissionMode::Auto)
        .build();

    // A session owns message history across turns.
    let session = agent.session("demo-session").await;

    println!("User: My favourite colour is blue.");
    print_turn(
        session
            .send("My favourite colour is blue. Just acknowledge.")
            .await,
    )
    .await?;

    // The session remembers the first turn automatically.
    println!("\nUser: What is my favourite colour?");
    print_turn(session.send("What is my favourite colour?").await).await?;

    Ok(())
}
