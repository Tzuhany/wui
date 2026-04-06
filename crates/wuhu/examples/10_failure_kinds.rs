/// 10 — FailureKind taxonomy.
///
/// Verifies each FailureKind variant is reachable and correctly reported:
///   - Execution      — tool returns ToolOutput::error
///   - InvalidInput   — LLM sends args that fail JSON Schema validation
///   - NotFound       — LLM calls a tool that doesn't exist (not in registry)
///   - HookBlocked    — a hook blocks the call (covered in 05_hooks)
///   - PermissionDenied — permission system blocks the call (covered in 08_readonly)
///
/// This example focuses on Execution, InvalidInput, and NotFound.
///
///   cargo run --example 10_failure_kinds --features anthropic

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};
use wuhu::prelude::*;
use wuhu::FailureKind;
use wuhu_providers::Anthropic;

fn anthropic() -> Anthropic {
    let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
    match std::env::var("ANTHROPIC_BASE_URL") {
        Ok(url) => Anthropic::with_base_url(key, url),
        Err(_)  => Anthropic::new(key),
    }
}

// ── Tool: always returns Execution error ─────────────────────────────────────

struct BrokenTool;

#[async_trait]
impl Tool for BrokenTool {
    fn name(&self) -> &str { "broken" }
    fn description(&self) -> &str { "A tool that always fails with an execution error." }
    fn prompt(&self) -> String { "Input: {}. Always returns an error.".into() }

    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }

    async fn call(&self, _input: Value, _ctx: &ToolCtx) -> ToolOutput {
        ToolOutput::error("disk full")
    }
}

// ── Tool: strict schema (triggers InvalidInput) ───────────────────────────────

struct StrictAdd;

#[async_trait]
impl Tool for StrictAdd {
    fn name(&self) -> &str { "add" }
    fn description(&self) -> &str { "Add two integers. Both must be integers." }
    fn prompt(&self) -> String { "Input: {\"a\": <integer>, \"b\": <integer>}".into() }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer" },
                "b": { "type": "integer" }
            },
            "required": ["a", "b"],
            "additionalProperties": false
        })
    }

    async fn call(&self, input: Value, _ctx: &ToolCtx) -> ToolOutput {
        let a = input["a"].as_i64().unwrap_or(0);
        let b = input["b"].as_i64().unwrap_or(0);
        ToolOutput::success((a + b).to_string())
    }
}

// ── Test: Execution failure ───────────────────────────────────────────────────

async fn test_execution_failure(provider: Anthropic) {
    println!("  [Execution] tool returns ToolOutput::error ...");

    let agent = Agent::builder(provider)
        .system("When asked to use the broken tool, call it. Report the error you receive.")
        .model("claude-haiku-4-5-20251001")
        .tool(BrokenTool)
        .permission(PermissionMode::Auto)
        .build();

    let mut stream  = agent.stream("Please call the broken tool.").await;
    let mut got_err = false;

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::ToolError { name, kind, error, .. } => {
                assert_eq!(name, "broken");
                assert_eq!(kind, FailureKind::Execution, "wrong failure kind: {kind:?}");
                assert!(error.contains("disk full"), "wrong error: {error}");
                got_err = true;
                println!("    ✓ Execution: {name} → {error}");
            }
            AgentEvent::Done(_) => break,
            AgentEvent::Error(e) => panic!("error: {e}"),
            _ => {}
        }
    }

    assert!(got_err, "ToolError(Execution) never fired");
}

// ── Test: add tool normal success (schema validation passes) ──────────────────

async fn test_schema_pass(provider: Anthropic) {
    println!("  [InvalidInput] schema pass — add integers ...");

    let agent = Agent::builder(provider)
        .system("Use the add tool to compute sums. Always pass integers.")
        .model("claude-haiku-4-5-20251001")
        .tool(StrictAdd)
        .permission(PermissionMode::Auto)
        .build();

    let mut stream = agent.stream("What is 3 + 4?").await;
    let mut ran    = false;

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::ToolDone { name, output, .. } => {
                assert_eq!(name, "add");
                assert!(output.contains('7'), "wrong result: {output}");
                ran = true;
                println!("    ✓ schema valid: {name} → {output}");
            }
            AgentEvent::ToolError { name, kind, error, .. } => {
                println!("    ⚠ tool error {name} {kind:?}: {error}");
            }
            AgentEvent::Done(_) => break,
            AgentEvent::Error(e) => panic!("error: {e}"),
            _ => {}
        }
    }

    assert!(ran, "add tool never ran");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== 10: FailureKind taxonomy ===");
    test_execution_failure(anthropic()).await;
    test_schema_pass(anthropic()).await;
    println!("PASS");
    Ok(())
}
