//! 02 — One-shot `Agent::run()`.
//!
//! Verifies: run() collects text from a streaming response and returns it
//! as a plain String. No event handling required from the caller.
//!
//!   cargo run --example 02_run --features anthropic

use wui::prelude::*;
use wui::providers::Anthropic;

fn anthropic() -> Anthropic {
    let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
    match std::env::var("ANTHROPIC_BASE_URL") {
        Ok(url) => Anthropic::with_base_url(key, url),
        Err(_) => Anthropic::new(key),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== 02: One-shot run() ===");

    let agent = Agent::builder(anthropic())
        .system("Reply with exactly: 'pong'")
        .model("claude-haiku-4-5-20251001")
        .build();

    let result = agent.run("ping").await?;

    println!("  result: {:?}", result);
    assert!(
        result.to_lowercase().contains("pong"),
        "expected 'pong', got: {result}"
    );

    println!("PASS");
    Ok(())
}
