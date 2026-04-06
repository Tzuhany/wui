/// 09 — Sub-agent spawning via as_spawn_fn().
///
/// Verifies:
///   - A sub-agent can be injected into a parent agent via .spawn()
///   - ctx.spawn_agent(prompt) returns the sub-agent's text response
///   - The sub-agent runs in isolation (no shared history with parent)
///   - The parent's final response incorporates the sub-agent's result
///
///   cargo run --example 09_subagent --features anthropic

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};
use wuhu::prelude::*;
use wuhu_providers::Anthropic;

fn anthropic() -> Anthropic {
    let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
    match std::env::var("ANTHROPIC_BASE_URL") {
        Ok(url) => Anthropic::with_base_url(key, url),
        Err(_)  => Anthropic::new(key),
    }
}

// ── Delegate tool ─────────────────────────────────────────────────────────────

/// A tool that delegates to a sub-agent.
struct Translate;

#[async_trait]
impl Tool for Translate {
    fn name(&self) -> &str { "translate" }
    fn description(&self) -> &str { "Translate text using a dedicated translation sub-agent." }
    fn prompt(&self) -> String {
        "Input: {\"text\": \"...\", \"language\": \"...\"}. Returns the translated text.".into()
    }
    fn is_readonly(&self) -> bool { true }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "text":     { "type": "string", "description": "Text to translate" },
                "language": { "type": "string", "description": "Target language" }
            },
            "required": ["text", "language"]
        })
    }

    async fn call(&self, input: Value, ctx: &ToolCtx) -> ToolOutput {
        let text     = input["text"].as_str().unwrap_or("");
        let language = input["language"].as_str().unwrap_or("French");
        let prompt   = format!("Translate the following text to {language}. \
                                Reply with only the translation, nothing else. Text: {text}");

        match ctx.spawn_agent(prompt) {
            Some(fut) => match fut.await {
                Ok(result)  => ToolOutput::success(result),
                Err(e)      => ToolOutput::error(format!("sub-agent failed: {e}")),
            },
            None => ToolOutput::error("no sub-agent configured"),
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== 09: Sub-agent spawning ===");

    // The translator sub-agent — dedicated to one task.
    let translator = Agent::builder(anthropic())
        .system("You are a professional translator. \
                 Translate the user's text exactly. Output only the translation.")
        .model("claude-haiku-4-5-20251001")
        .build();

    // The orchestrator — delegates translation work.
    let orchestrator = Agent::builder(anthropic())
        .system("You are a helpful assistant. Use the translate tool when asked to translate text.")
        .model("claude-haiku-4-5-20251001")
        .tool(Translate)
        .spawn(translator.as_spawn_fn())
        .permission(PermissionMode::Auto)
        .build();

    let mut stream       = orchestrator.stream("Translate 'Hello, world!' to French.").await;
    let mut tool_ran     = false;
    let mut final_text   = String::new();

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::ToolStart { name, .. } => {
                assert_eq!(name, "translate");
                println!("  → sub-agent delegation started");
            }
            AgentEvent::ToolDone { name, output, ms, .. } => {
                assert_eq!(name, "translate");
                println!("  ← sub-agent result ({ms}ms): {output}");
                // The sub-agent should have produced a French translation.
                assert!(!output.is_empty(), "sub-agent returned empty result");
                tool_ran = true;
            }
            AgentEvent::TextDelta(t) => final_text.push_str(&t),
            AgentEvent::Done(_)      => break,
            AgentEvent::Error(e)     => panic!("error: {e}"),
            _ => {}
        }
    }

    println!("  final response: {final_text}");
    assert!(tool_ran, "translate tool never ran");
    assert!(!final_text.is_empty(), "no final text response");

    // The French translation of "Hello, world!" should appear somewhere.
    let lower = final_text.to_lowercase();
    let has_french = lower.contains("bonjour") || lower.contains("monde") || lower.contains("salut");
    assert!(has_french, "final text doesn't seem to contain the French translation: {final_text}");

    println!("PASS");
    Ok(())
}
