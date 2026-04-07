//! 01 — Streaming text output.
//!
//! Verifies: TextDelta events accumulate into coherent text, Done fires with
//! a RunSummary that includes non-zero token counts.
//!
//!   ANTHROPIC_API_KEY=sk-... \
//!   ANTHROPIC_BASE_URL=https://... \
//!   cargo run --example 01_streaming --features anthropic

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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== 01: Streaming text ===");

    let agent = Agent::builder(anthropic())
        .system("You are a concise assistant. Answer in exactly one sentence.")
        .model("claude-haiku-4-5-20251001")
        .build();

    let mut stream = agent.stream("What is the capital of France?");
    let mut text = String::new();
    let mut got_done = false;

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TextDelta(t) => {
                print!("{t}");
                text.push_str(&t);
            }
            AgentEvent::Done(summary) => {
                println!();
                println!("  stop_reason  : {:?}", summary.stop_reason);
                println!("  iterations   : {}", summary.iterations);
                println!("  input_tokens : {}", summary.usage.input_tokens);
                println!("  output_tokens: {}", summary.usage.output_tokens);

                assert!(!text.is_empty(), "expected text response");
                assert!(
                    text.to_lowercase().contains("paris"),
                    "expected Paris in response, got: {text}"
                );
                assert_eq!(summary.stop_reason, RunStopReason::Completed);
                assert!(summary.iterations == 1, "expected 1 iteration");
                // Note: some proxies report input_tokens=0 in message_start.
                // We assert output_tokens which is always reported in message_delta.
                assert!(
                    summary.usage.output_tokens > 0,
                    "expected non-zero output tokens"
                );

                got_done = true;
                break;
            }
            AgentEvent::Error(e) => {
                panic!("unexpected error: {e}");
            }
            _ => {}
        }
    }

    assert!(got_done, "stream ended without Done event");
    println!("PASS");
    Ok(())
}
