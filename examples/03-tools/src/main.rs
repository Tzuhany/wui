//! 03 — Tool use with PermissionMode::Auto.
//!
//! Verifies:
//!   - ToolStart fires before the tool executes
//!   - ToolDone fires after with output and elapsed ms
//!   - ToolOutput::success result is used in the LLM's final response
//!   - ToolOutput::error is surfaced as ToolError with FailureKind::Execution
//!
//!   cargo run --example 03_tools --features anthropic

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};
use wui::prelude::*;
use wui::providers::Anthropic;
use wui::FailureKind;

fn anthropic() -> Anthropic {
    let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
    match std::env::var("ANTHROPIC_BASE_URL") {
        Ok(url) => Anthropic::with_base_url(key, url),
        Err(_) => Anthropic::new(key),
    }
}

// ── Tools ─────────────────────────────────────────────────────────────────────

/// A tool that returns a secret code the LLM cannot know without calling it.
/// This guarantees the LLM must call the tool to answer the question.
struct SecretCode;

#[async_trait]
impl Tool for SecretCode {
    fn name(&self) -> &str {
        "get_secret_code"
    }
    fn description(&self) -> &str {
        "Retrieve the current secret access code."
    }
    fn meta(&self, _input: &Value) -> ToolMeta {
        ToolMeta {
            readonly: true,
            concurrent: true,
            ..ToolMeta::default()
        }
    }

    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }

    async fn call(&self, _input: Value, _ctx: &ToolCtx) -> ToolOutput {
        ToolOutput::success("WUI-ALPHA-9274")
    }
}

/// A tool that always fails with an execution error.
struct AlwaysFails;

#[async_trait]
impl Tool for AlwaysFails {
    fn name(&self) -> &str {
        "always_fails"
    }
    fn description(&self) -> &str {
        "A tool that always returns an error."
    }

    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }

    async fn call(&self, _input: Value, _ctx: &ToolCtx) -> ToolOutput {
        ToolOutput::error("something went wrong intentionally")
    }
}

// ── Test: successful tool ─────────────────────────────────────────────────────

async fn test_success(provider: Anthropic) {
    println!("  [success] echo tool ...");

    // The LLM cannot know the secret code — it MUST call get_secret_code.
    let agent = Agent::builder(provider)
        .system("Answer the user's question by using the available tools.")
        .model("claude-haiku-4-5-20251001")
        .tool(SecretCode)
        .permission(PermissionMode::Auto)
        .build();

    let mut stream = agent.stream("What is the current secret access code?");
    let mut tool_start = false;
    let mut tool_done = false;
    let mut text = String::new();

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::ToolStart { name, .. } => {
                assert_eq!(name, "get_secret_code");
                tool_start = true;
                println!("    → tool start: {name}");
            }
            AgentEvent::ToolDone {
                name, output, ms, ..
            } => {
                assert_eq!(name, "get_secret_code");
                assert_eq!(output, "WUI-ALPHA-9274", "unexpected output: {output}");
                assert!(ms < 5000, "tool took too long: {ms}ms");
                tool_done = true;
                println!("    ← tool done: {name} ({ms}ms) output={output}");
            }
            AgentEvent::TextDelta(t) => text.push_str(&t),
            AgentEvent::Done(_) => break,
            AgentEvent::Error(e) => panic!("error: {e}"),
            _ => {}
        }
    }

    assert!(tool_start, "ToolStart never fired");
    assert!(tool_done, "ToolDone never fired");
    assert!(!text.is_empty(), "no final text");
    assert!(
        text.contains("WUI-ALPHA-9274"),
        "response doesn't mention the code: {text}"
    );
    println!("    final text: {text}");
}

// ── Test: failing tool ────────────────────────────────────────────────────────

async fn test_failure(provider: Anthropic) {
    println!("  [failure] always_fails tool ...");

    let agent = Agent::builder(provider)
        .system("When asked to get the status, call the always_fails tool. Report the error you receive.")
        .model("claude-haiku-4-5-20251001")
        .tool(AlwaysFails)
        .permission(PermissionMode::Auto)
        .build();

    let mut stream = agent.stream("Get the current system status.");
    let mut got_error = false;

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::ToolError {
                name,
                error,
                kind,
                ms,
                ..
            } => {
                assert_eq!(name, "always_fails");
                assert!(
                    error.contains("something went wrong"),
                    "unexpected error: {error}"
                );
                assert_eq!(kind, FailureKind::Execution);
                assert!(ms < 5000);
                got_error = true;
                println!("    ✓ ToolError: kind={kind:?} error={error}");
            }
            AgentEvent::Done(_) => break,
            AgentEvent::Error(e) => panic!("error: {e}"),
            _ => {}
        }
    }

    assert!(got_error, "ToolError never fired");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== 03: Tool use ===");
    test_success(anthropic()).await;
    test_failure(anthropic()).await;
    println!("PASS");
    Ok(())
}
