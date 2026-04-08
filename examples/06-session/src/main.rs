//! 06 — Multi-turn session with an in-memory session store.
//!
//! Verifies:
//!   - Session retains message history across turns
//!   - Second turn can refer to first-turn content
//!   - InMemory session store saves and loads correctly (session resumption)
//!   - Session::messages() returns accumulated messages
//!
//!   cargo run --example 06_session --features anthropic

use futures::StreamExt;
use wui::prelude::*;
use wui::providers::Anthropic;

fn anthropic() -> Anthropic {
    let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
    match std::env::var("ANTHROPIC_BASE_URL") {
        Ok(url) => Anthropic::with_base_url(key, url),
        Err(_) => Anthropic::new(key),
    }
}

// ── Test: multi-turn history ──────────────────────────────────────────────────

async fn test_history(provider: Anthropic) {
    println!("  [multi-turn] history across turns ...");

    let agent = Agent::builder(provider)
        .system("You are a helpful assistant with a good memory.")
        .model("claude-haiku-4-5-20251001")
        .build();

    let session = agent.session("history-test").await;

    // Turn 1: introduce a fact.
    {
        let mut stream = session
            .send("My secret code word is BANANA42. Acknowledge.")
            .await;
        let mut text = String::new();
        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::TextDelta(t) => text.push_str(&t),
                AgentEvent::Done(_) => break,
                AgentEvent::Error(e) => panic!("turn1 error: {e}"),
                _ => {}
            }
        }
        println!("    turn 1 response: {text}");
        assert!(!text.is_empty());
    }

    // Turn 2: recall the fact.
    {
        let mut stream = session.send("What was my secret code word?").await;
        let mut text = String::new();
        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::TextDelta(t) => text.push_str(&t),
                AgentEvent::Done(_) => break,
                AgentEvent::Error(e) => panic!("turn2 error: {e}"),
                _ => {}
            }
        }
        println!("    turn 2 response: {text}");
        assert!(
            text.contains("BANANA42"),
            "LLM forgot the code word; response: {text}"
        );
    }

    // Verify history has at least 4 messages (user + assistant × 2).
    let history = session.messages();
    println!("    history length: {}", history.len());
    assert!(
        history.len() >= 4,
        "expected ≥4 messages, got {}",
        history.len()
    );
    println!("    ✓ multi-turn history works");
}

// ── Test: session-store save + load ───────────────────────────────────────────

async fn test_session_store(provider: Anthropic) {
    println!("  [session-store] InMemory save/load ...");

    // InMemorySessionStore is Clone — clones share the same underlying store.
    let store = InMemorySessionStore::new();

    // Agent A writes.
    let agent_a = Agent::builder(provider.clone())
        .system("You are a concise assistant. Confirm what you're told.")
        .model("claude-haiku-4-5-20251001")
        .session_store(store.clone())
        .build();

    {
        let session = agent_a.session("cp-test").await;
        let mut stream = session.send("Remember: the magic number is 777.").await;
        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::Done(_) => break,
                AgentEvent::Error(e) => panic!("error: {e}"),
                _ => {}
            }
        }
        println!("    session A: wrote magic number");
    }

    // Verify the session store has a snapshot by loading from a clone of the same store.
    let snapshot = store
        .load("cp-test")
        .await
        .expect("session store load failed")
        .expect("no snapshot saved");
    assert!(!snapshot.messages.is_empty(), "snapshot has no messages");
    println!("    snapshot messages: {}", snapshot.messages.len());

    // Agent B resumes from the same session store (another clone — still same store).
    let agent_b = Agent::builder(provider)
        .system("You are a concise assistant. Answer from conversation history.")
        .model("claude-haiku-4-5-20251001")
        .session_store(store.clone())
        .build();

    {
        let session = agent_b.session("cp-test").await;
        let mut stream = session
            .send("What was the magic number I told you about?")
            .await;
        let mut text = String::new();
        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::TextDelta(t) => text.push_str(&t),
                AgentEvent::Done(_) => break,
                AgentEvent::Error(e) => panic!("error: {e}"),
                _ => {}
            }
        }
        println!("    session B response: {text}");
        assert!(
            text.contains("777"),
            "resumed session forgot the magic number; got: {text}"
        );
    }

    println!("    ✓ session-store save/load works");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== 06: Session + session store ===");
    test_history(anthropic()).await;
    test_session_store(anthropic()).await;
    println!("PASS");
    Ok(())
}
