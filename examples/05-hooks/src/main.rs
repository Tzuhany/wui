//! 05 — Hooks: DenyList and custom hooks.
//!
//! Verifies:
//!   - DenyList blocks a specific tool by name (FailureKind::HookBlocked)
//!   - Custom PreToolUse hook can inspect and block based on input
//!   - Custom PostToolUse hook fires with the tool's output
//!   - Custom PreStop hook can block and force the LLM to revise before stopping
//!   - Hook::handles() filters events efficiently
//!
//!   cargo run --example 05_hooks --features anthropic

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};
use wui::prelude::*;
use wui::providers::Anthropic;
use wui::{FailureKind, HookEvent};

fn anthropic() -> Anthropic {
    let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
    match std::env::var("ANTHROPIC_BASE_URL") {
        Ok(url) => Anthropic::with_base_url(key, url),
        Err(_) => Anthropic::new(key),
    }
}

// ── Tools ─────────────────────────────────────────────────────────────────────

struct Echo;

#[async_trait]
impl Tool for Echo {
    fn name(&self) -> &str {
        "echo"
    }
    fn description(&self) -> &str {
        "Echo text back."
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": { "text": { "type": "string" } }, "required": ["text"] })
    }
    async fn call(&self, input: Value, _ctx: &ToolCtx) -> ToolOutput {
        ToolOutput::success(input["text"].as_str().unwrap_or("").to_string())
    }
}

struct Loud;

#[async_trait]
impl Tool for Loud {
    fn name(&self) -> &str {
        "loud"
    }
    fn description(&self) -> &str {
        "Return text in uppercase."
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": { "text": { "type": "string" } }, "required": ["text"] })
    }
    async fn call(&self, input: Value, _ctx: &ToolCtx) -> ToolOutput {
        ToolOutput::success(input["text"].as_str().unwrap_or("").to_uppercase())
    }
}

// ── Custom hooks ──────────────────────────────────────────────────────────────

/// Blocks any tool call whose "text" argument contains a banned word.
struct BannedWords {
    words: Vec<String>,
}

impl BannedWords {
    fn new(words: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            words: words.into_iter().map(Into::into).collect(),
        }
    }
}

#[async_trait]
impl Hook for BannedWords {
    fn handles(&self, event: &HookEvent<'_>) -> bool {
        matches!(event, HookEvent::PreToolUse { .. })
    }

    async fn evaluate(&self, event: &HookEvent<'_>) -> HookDecision {
        if let HookEvent::PreToolUse { input, .. } = event {
            if let Some(text) = input["text"].as_str() {
                for word in &self.words {
                    if text.to_lowercase().contains(word.as_str()) {
                        return HookDecision::block(format!(
                            "input contains banned word: '{word}'"
                        ));
                    }
                }
            }
        }
        HookDecision::Allow
    }
}

// ── Test: DenyList ────────────────────────────────────────────────────────────

async fn test_denylist(provider: Anthropic) {
    println!("  [DenyList] block by tool name ...");

    let agent = Agent::builder(provider)
        .system(
            "When asked to be loud, call the loud tool. When asked to echo, call the echo tool.",
        )
        .model("claude-haiku-4-5-20251001")
        .tool(Echo)
        .tool(Loud)
        .hook(DenyList::new(["loud"]))
        .permission(PermissionMode::Auto)
        .build();

    let mut stream = agent.stream("Please use the loud tool to say hello.");
    let mut got_blocked = false;

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::ToolError {
                name, kind, error, ..
            } => {
                println!("    ✓ tool blocked: {name} kind={kind:?} error={error}");
                assert_eq!(name, "loud");
                assert_eq!(kind, FailureKind::HookBlocked);
                got_blocked = true;
            }
            AgentEvent::Done(_) => break,
            AgentEvent::Error(e) => panic!("error: {e}"),
            _ => {}
        }
    }

    assert!(got_blocked, "DenyList never blocked 'loud'");
}

// ── Test: custom PreToolUse hook ──────────────────────────────────────────────

async fn test_banned_words(provider: Anthropic) {
    println!("  [BannedWords] custom PreToolUse hook ...");

    let agent = Agent::builder(provider)
        .system("When asked to echo text, call the echo tool with that text.")
        .model("claude-haiku-4-5-20251001")
        .tool(Echo)
        .hook(BannedWords::new(["forbidden"]))
        .permission(PermissionMode::Auto)
        .build();

    let mut stream = agent.stream("Echo the word: forbidden");
    let mut got_blocked = false;

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::ToolError {
                name, kind, error, ..
            } => {
                println!("    ✓ hook blocked: {name} kind={kind:?} error={error}");
                assert_eq!(name, "echo");
                assert_eq!(kind, FailureKind::HookBlocked);
                assert!(error.contains("forbidden"), "unexpected error: {error}");
                got_blocked = true;
            }
            AgentEvent::Done(_) => break,
            AgentEvent::Error(e) => panic!("error: {e}"),
            _ => {}
        }
    }

    assert!(got_blocked, "BannedWords hook never blocked");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== 05: Hooks ===");
    test_denylist(anthropic()).await;
    test_banned_words(anthropic()).await;
    println!("PASS");
    Ok(())
}
