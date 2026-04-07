//! XML structured output extraction.
//!
//! Shows `agent.run_structured("...")`, `.extract("tag_name").await`, and
//! `.extract_as::<MyStruct>("tag").await` for typed deserialization.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-... cargo run --example 05_structured -p wui --features anthropic

use serde::Deserialize;
use wui::providers::Anthropic;
use wui::{Agent, PermissionMode};

// The agent will respond with JSON inside an <analysis> tag.
#[derive(Debug, Deserialize)]
struct SentimentAnalysis {
    sentiment: String,
    score: f32,
    reasoning: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;

    let agent = Agent::builder(Anthropic::new(api_key))
        .system(
            "You are a sentiment analyser. \
             Always wrap your answer in an <answer> tag. \
             When asked for JSON analysis, wrap it in an <analysis> tag.",
        )
        .permission(PermissionMode::Auto)
        .build();

    // ── Simple tag extraction ─────────────────────────────────────────────────
    let answer = agent
        .run_structured("What is 2 + 2? Wrap your answer in <answer> tags.")
        .extract("answer")
        .await?;

    println!("Simple extraction: {answer}");

    // ── Typed JSON extraction ─────────────────────────────────────────────────
    let analysis: SentimentAnalysis = agent
        .run_structured(
            "Analyse the sentiment of: 'I love working with Rust, it is fantastic!' \
             Respond with a JSON object inside <analysis> tags. \
             The JSON must have: sentiment (string), score (number 0-1), reasoning (string).",
        )
        .extract_as::<SentimentAnalysis>("analysis")
        .await?;

    println!("Sentiment: {}", analysis.sentiment);
    println!("Score:     {:.2}", analysis.score);
    println!("Reasoning: {}", analysis.reasoning);

    Ok(())
}
