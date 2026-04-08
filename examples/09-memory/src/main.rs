//! 09 — Long-term memory with wui-memory.
//!
//! Demonstrates the recall/remember/forget capability:
//!   - `memory_remember` stores facts across turns
//!   - `memory_recall` surfaces them by query
//!   - `InMemoryStore` is the reference backend (swap for pgvector/Redis/etc.)
//!
//! The agent is asked to remember a fact in one message, then recall it in
//! another. Because memory persists in the store (not in the context window),
//! this works even after the original message has been compressed away.
//!
//!   cargo run --example 09_memory --features anthropic

use std::sync::Arc;

use futures::StreamExt;
use wui::prelude::*;
use wui::providers::Anthropic;
use wui_memory::{all_memory_tools, InMemoryStore, RecallBackend};

fn anthropic() -> Anthropic {
    let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
    match std::env::var("ANTHROPIC_BASE_URL") {
        Ok(url) => Anthropic::with_base_url(key, url),
        Err(_) => Anthropic::new(key),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== 09: Memory (recall / remember) ===");

    let store = Arc::new(InMemoryStore::new());
    let tools = all_memory_tools(store.clone());

    let agent = Agent::builder(anthropic())
        .system(
            "You are a helpful assistant with long-term memory. \
             When asked to remember something, call memory_remember. \
             When asked to recall something, call memory_recall.",
        )
        .model("claude-haiku-4-5-20251001")
        .tools(tools)
        .permission(PermissionMode::Auto)
        .build();

    let session = agent.session("example-09").await;

    // Turn 1 — ask the agent to store a fact.
    println!("  [turn 1] asking agent to remember a fact ...");
    let mut stream = session.send("Please remember this: the project codename is Phoenix.").await;
    let mut remembered = false;

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::ToolStart { name, .. } if name == "memory_remember" => {
                println!("    → memory_remember called");
                remembered = true;
            }
            AgentEvent::Done(_) => break,
            AgentEvent::Error(e) => panic!("error: {e}"),
            _ => {}
        }
    }
    assert!(remembered, "expected agent to call memory_remember");

    // Turn 2 — ask the agent to recall the fact.
    println!("  [turn 2] asking agent to recall the codename ...");
    let mut stream = session.send("What is the project codename you were asked to remember?").await;
    let mut recalled = false;
    let mut response = String::new();

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::ToolStart { name, .. } if name == "memory_recall" => {
                println!("    → memory_recall called");
                recalled = true;
            }
            AgentEvent::TextDelta(t) => response.push_str(&t),
            AgentEvent::Done(_) => break,
            AgentEvent::Error(e) => panic!("error: {e}"),
            _ => {}
        }
    }
    assert!(recalled, "expected agent to call memory_recall");
    assert!(
        response.to_lowercase().contains("phoenix"),
        "expected 'Phoenix' in response, got: {response}"
    );
    println!("    response: {response}");

    // Verify the store directly — the fact is there regardless of what the LLM said.
    let hits: Vec<_> = store.recall("Phoenix", None).await?;
    assert!(!hits.is_empty(), "store should contain at least one entry");
    println!(
        "    store has {} matching entr{}",
        hits.len(),
        if hits.len() == 1 { "y" } else { "ies" }
    );

    println!("PASS");
    Ok(())
}
