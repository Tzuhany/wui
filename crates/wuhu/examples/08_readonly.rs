/// 08 — PermissionMode::Readonly.
///
/// Verifies:
///   - Tools with is_readonly() = true run without prompting
///   - Tools with is_readonly() = false are blocked with FailureKind::PermissionDenied
///   - PermissionMode::Auto bypasses both (control group)
///
///   cargo run --example 08_readonly --features anthropic

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

/// Read-only tool: is_readonly() = true.
struct Lookup;

#[async_trait]
impl Tool for Lookup {
    fn name(&self) -> &str { "lookup" }
    fn description(&self) -> &str { "Look up a value by key (read-only)." }
    fn prompt(&self) -> String { "Input: {\"key\": \"...\"}. Returns a value string.".into() }
    fn is_readonly(&self) -> bool { true }

    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": { "key": { "type": "string" } }, "required": ["key"] })
    }

    async fn call(&self, input: Value, _ctx: &ToolCtx) -> ToolOutput {
        let key = input["key"].as_str().unwrap_or("?");
        ToolOutput::success(format!("value_for_{key}"))
    }
}

/// Mutating tool: is_readonly() = false (default).
struct Delete;

#[async_trait]
impl Tool for Delete {
    fn name(&self) -> &str { "delete" }
    fn description(&self) -> &str { "Delete a record (mutating)." }
    fn prompt(&self) -> String { "Input: {\"id\": \"...\"}. Deletes a record.".into() }

    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": { "id": { "type": "string" } }, "required": ["id"] })
    }

    async fn call(&self, input: Value, _ctx: &ToolCtx) -> ToolOutput {
        let id = input["id"].as_str().unwrap_or("?");
        ToolOutput::success(format!("deleted:{id}"))
    }
}

// ── Test: readonly tool passes ────────────────────────────────────────────────

async fn test_readonly_allowed(provider: Anthropic) {
    println!("  [Readonly] read-only tool should pass ...");

    let agent = Agent::builder(provider)
        .system("When asked to look something up, use the lookup tool.")
        .model("claude-haiku-4-5-20251001")
        .tool(Lookup)
        .permission(PermissionMode::Readonly)
        .build();

    let mut stream  = agent.stream("Look up the value for key 'x'.").await;
    let mut ran     = false;

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::ToolDone { name, output, .. } => {
                assert_eq!(name, "lookup");
                assert!(output.contains("value_for_x"), "unexpected output: {output}");
                ran = true;
                println!("    ✓ readonly tool ran: {output}");
            }
            AgentEvent::ToolError { name, kind, .. } => {
                panic!("read-only tool blocked unexpectedly: {name} {kind:?}");
            }
            AgentEvent::Done(_) => break,
            AgentEvent::Error(e) => panic!("error: {e}"),
            _ => {}
        }
    }

    assert!(ran, "readonly tool never ran");
}

// ── Test: mutating tool blocked ───────────────────────────────────────────────

async fn test_mutating_blocked(provider: Anthropic) {
    println!("  [Readonly] mutating tool should be blocked ...");

    let agent = Agent::builder(provider)
        .system("When asked to delete, use the delete tool.")
        .model("claude-haiku-4-5-20251001")
        .tool(Delete)
        .permission(PermissionMode::Readonly)
        .build();

    let mut stream   = agent.stream("Delete record id '42'.").await;
    let mut blocked  = false;

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::ToolError { name, kind, .. } => {
                assert_eq!(name, "delete");
                assert_eq!(kind, FailureKind::PermissionDenied);
                blocked = true;
                println!("    ✓ mutating tool blocked: {name} kind={kind:?}");
            }
            AgentEvent::ToolDone { name, .. } => {
                panic!("mutating tool ran in Readonly mode: {name}");
            }
            AgentEvent::Done(_) => break,
            AgentEvent::Error(e) => panic!("error: {e}"),
            _ => {}
        }
    }

    assert!(blocked, "mutating tool was not blocked in Readonly mode");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== 08: PermissionMode::Readonly ===");
    test_readonly_allowed(anthropic()).await;
    test_mutating_blocked(anthropic()).await;
    println!("PASS");
    Ok(())
}
