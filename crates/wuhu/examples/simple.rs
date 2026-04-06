/// Minimal example: single-turn, streaming text output.
///
/// Run with:
///   ANTHROPIC_API_KEY=sk-... cargo run --example simple --features anthropic

use futures::StreamExt;
use wuhu::prelude::*;
use wuhu_providers::Anthropic;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let key      = std::env::var("ANTHROPIC_API_KEY")?;
    let provider = match std::env::var("ANTHROPIC_BASE_URL") {
        Ok(url) => Anthropic::with_base_url(key, url),
        Err(_)  => Anthropic::new(key),
    };
    let agent = Agent::builder(provider)
        .system("You are a concise assistant. Answer in one sentence.")
        .model("claude-haiku-4-5-20251001")
        .build();

    let mut stream = agent.stream("What is the speed of light?").await;

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TextDelta(text) => print!("{text}"),
            AgentEvent::Done(_)         => { println!(); break; }
            AgentEvent::Error(e)        => { eprintln!("error: {e}"); break; }
            _                           => {}
        }
    }

    Ok(())
}
