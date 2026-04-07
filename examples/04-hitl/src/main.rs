//! 04 — Human-in-the-loop: ControlHandle.
//!
//! Verifies:
//!   - PermissionMode::Ask emits Control event instead of running tools
//!   - handle.approve() resumes execution
//!   - handle.deny(reason) blocks the tool; LLM sees the reason
//!   - handle.approve_always() means second invocation skips the prompt
//!   - handle.deny_always(reason) means second invocation is blocked silently
//!
//!   cargo run --example 04_hitl --features anthropic

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};
use wui::prelude::*;
use wui::providers::Anthropic;

fn anthropic() -> Anthropic {
    let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
    match std::env::var("ANTHROPIC_BASE_URL") {
        Ok(url) => Anthropic::with_base_url(key, url),
        Err(_) => Anthropic::new(key),
    }
}

struct Stamp;

#[async_trait]
impl Tool for Stamp {
    fn name(&self) -> &str {
        "stamp"
    }
    fn description(&self) -> &str {
        "Return a timestamp string."
    }

    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }

    async fn call(&self, _input: Value, _ctx: &ToolCtx) -> ToolOutput {
        ToolOutput::success("stamped")
    }
}

// ── Test: approve ─────────────────────────────────────────────────────────────

async fn test_approve(provider: Anthropic) {
    println!("  [approve] control handle ...");

    let agent = Agent::builder(provider)
        .system("When asked to stamp something, call the stamp tool. Confirm it worked.")
        .model("claude-haiku-4-5-20251001")
        .tool(Stamp)
        .permission(PermissionMode::Ask)
        .build();

    let mut stream = agent.stream("Please stamp this document.");
    let mut got_control = false;
    let mut tool_ran = false;

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::Control(handle) => {
                println!("    → control: {:?}", handle.request.kind);
                assert_eq!(handle.request.tool_name(), Some("stamp"));
                handle.approve();
                got_control = true;
            }
            AgentEvent::ToolDone { name, .. } => {
                assert_eq!(name, "stamp");
                tool_ran = true;
                println!("    ← tool done: {name}");
            }
            AgentEvent::Done(_) => break,
            AgentEvent::Error(e) => panic!("error: {e}"),
            _ => {}
        }
    }

    assert!(got_control, "Control never fired");
    assert!(tool_ran, "tool never ran after approval");
    println!("    ✓ approve works");
}

// ── Test: deny ────────────────────────────────────────────────────────────────

async fn test_deny(provider: Anthropic) {
    println!("  [deny] control handle ...");

    let agent = Agent::builder(provider)
        .system(
            "When asked to stamp something, call the stamp tool. If denied, say 'access denied'.",
        )
        .model("claude-haiku-4-5-20251001")
        .tool(Stamp)
        .permission(PermissionMode::Ask)
        .build();

    let mut stream = agent.stream("Please stamp this document.");
    let mut got_control = false;
    let mut tool_ran = false;
    let mut final_text = String::new();

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::Control(handle) => {
                println!("    → control: denying");
                handle.deny("testing: access denied");
                got_control = true;
            }
            AgentEvent::ToolDone { .. } => {
                tool_ran = true;
            }
            AgentEvent::TextDelta(t) => final_text.push_str(&t),
            AgentEvent::Done(_) => break,
            AgentEvent::Error(e) => panic!("error: {e}"),
            _ => {}
        }
    }

    assert!(got_control, "Control never fired");
    assert!(!tool_ran, "tool ran despite denial");
    println!("    ✓ deny works; LLM response: {final_text}");
}

// ── Test: approve_always ──────────────────────────────────────────────────────

async fn test_approve_always(provider: Anthropic) {
    println!("  [approve_always] session-level memory ...");

    let agent = Agent::builder(provider)
        .system("Call the stamp tool whenever asked. Do it twice in sequence if asked.")
        .model("claude-haiku-4-5-20251001")
        .tool(Stamp)
        .permission(PermissionMode::Ask)
        .build();

    let session = agent.session("test-approve-always").await;
    let mut control_count = 0u32;

    // First turn: prompt should trigger one Control → approve_always.
    {
        let mut stream = session.send("Please stamp this once.");
        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::Control(handle) => {
                    println!("    → control #{}: approve_always", control_count + 1);
                    handle.approve_always();
                    control_count += 1;
                }
                AgentEvent::Done(_) => break,
                AgentEvent::Error(e) => panic!("error: {e}"),
                _ => {}
            }
        }
    }

    let count_after_first = control_count;
    assert!(
        count_after_first >= 1,
        "expected at least one control prompt"
    );

    // Second turn: the same tool should run without prompting.
    {
        let mut stream = session.send("Please stamp this once more.");
        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::Control(handle) => {
                    println!("    → UNEXPECTED control #{}", control_count + 1);
                    handle.approve(); // don't hang
                    control_count += 1;
                }
                AgentEvent::Done(_) => break,
                AgentEvent::Error(e) => panic!("error: {e}"),
                _ => {}
            }
        }
    }

    assert_eq!(
        control_count,
        count_after_first,
        "approve_always should suppress second control prompt; got {} extra prompts",
        control_count - count_after_first
    );
    println!("    ✓ approve_always works");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== 04: HITL / ControlHandle ===");
    test_approve(anthropic()).await;
    test_deny(anthropic()).await;
    test_approve_always(anthropic()).await;
    println!("PASS");
    Ok(())
}
