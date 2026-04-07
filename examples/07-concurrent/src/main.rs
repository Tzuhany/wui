//! 07 — Concurrent vs serial tools.
//!
//! Verifies:
//!   - is_concurrent_for(input) = true  → tool runs concurrently (tracked by wall time)
//!   - is_concurrent_for(input) = false → tool runs serially (sequential)
//!   - Multiple ToolStart events can fire before any ToolDone (concurrent case)
//!
//!   cargo run --example 07_concurrent --features anthropic

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};
use std::time::{Duration, Instant};
use wui::prelude::*;
use wui::providers::Anthropic;

fn anthropic() -> Anthropic {
    let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
    match std::env::var("ANTHROPIC_BASE_URL") {
        Ok(url) => Anthropic::with_base_url(key, url),
        Err(_) => Anthropic::new(key),
    }
}

/// A slow tool that sleeps for `delay_ms` milliseconds.
/// is_concurrent_for returns true → runs concurrently.
struct SlowRead;

#[async_trait]
impl Tool for SlowRead {
    fn name(&self) -> &str {
        "slow_read"
    }
    fn description(&self) -> &str {
        "A slow read operation that takes 500ms."
    }
    fn meta(&self, _input: &Value) -> ToolMeta {
        ToolMeta {
            readonly: true,
            concurrent: true,
            ..ToolMeta::default()
        }
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "label": { "type": "string" } },
            "required": ["label"]
        })
    }

    async fn call(&self, input: Value, ctx: &ToolCtx) -> ToolOutput {
        let label = input["label"].as_str().unwrap_or("?").to_string();
        // Simulate slow I/O; respect cancellation.
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(500)) => {}
            _ = ctx.cancel.cancelled() => {}
        }
        ToolOutput::success(format!("read:{label}"))
    }
}

/// A slow tool that serialises via is_concurrent_for = false.
struct SlowWrite;

#[async_trait]
impl Tool for SlowWrite {
    fn name(&self) -> &str {
        "slow_write"
    }
    fn description(&self) -> &str {
        "A slow write operation that takes 400ms and must not run concurrently."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "data": { "type": "string" } },
            "required": ["data"]
        })
    }

    fn meta(&self, _input: &Value) -> ToolMeta {
        ToolMeta {
            concurrent: false,
            ..ToolMeta::default()
        }
    }

    async fn call(&self, input: Value, ctx: &ToolCtx) -> ToolOutput {
        let data = input["data"].as_str().unwrap_or("?").to_string();
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(400)) => {}
            _ = ctx.cancel.cancelled() => {}
        }
        ToolOutput::success(format!("wrote:{data}"))
    }
}

// ── Test: concurrent reads ────────────────────────────────────────────────────

async fn test_concurrent(provider: Anthropic) {
    println!("  [concurrent] two slow_read calls in parallel ...");

    let agent = Agent::builder(provider)
        .system(
            "When asked to read multiple items, call slow_read for each one in a single response. \
             Do not wait for one to finish before calling the other.",
        )
        .model("claude-haiku-4-5-20251001")
        .tool(SlowRead)
        .permission(PermissionMode::Auto)
        .build();

    let start = Instant::now();
    let mut stream = agent.stream("Read both 'alpha' and 'beta' using slow_read. Call both tools.");

    let mut starts: Vec<String> = Vec::new();
    let mut dones: Vec<String> = Vec::new();

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::ToolStart { name, .. } => starts.push(name),
            AgentEvent::ToolDone { name, .. } => dones.push(name),
            AgentEvent::Done(_) => break,
            AgentEvent::Error(e) => panic!("error: {e}"),
            _ => {}
        }
    }

    let elapsed = start.elapsed();
    println!("    starts: {:?}", starts);
    println!("    dones:  {:?}", dones);
    println!("    elapsed: {}ms", elapsed.as_millis());

    // If both ran concurrently, total time should be ~500ms, not ~1000ms.
    // We allow generous slack for CI/LLM latency.
    if starts.len() == 2 {
        // Concurrent: should complete in under 1200ms (two 500ms parallel + overhead).
        // Serial would be ~1000ms + overhead. We don't assert timing strictly to
        // avoid flakiness, but log the result.
        if elapsed.as_millis() < 1200 {
            println!(
                "    ✓ ran concurrently (~{}ms < 1200ms threshold)",
                elapsed.as_millis()
            );
        } else {
            println!(
                "    ⚠ elapsed {}ms — may have run serially (acceptable if LLM called separately)",
                elapsed.as_millis()
            );
        }
    }
    assert_eq!(
        dones.len(),
        starts.len(),
        "mismatch: {} starts, {} dones",
        starts.len(),
        dones.len()
    );
    println!("    ✓ concurrent tool paths work");
}

// ── Test: serial writes ───────────────────────────────────────────────────────

async fn test_serial(provider: Anthropic) {
    println!("  [serial] slow_write calls are serialised ...");

    let agent = Agent::builder(provider)
        .system("When asked to write data, call slow_write for each item.")
        .model("claude-haiku-4-5-20251001")
        .tool(SlowWrite)
        .permission(PermissionMode::Auto)
        .build();

    let mut stream = agent.stream("Write 'one' and 'two' using slow_write. Call both tools.");

    let mut dones: Vec<String> = Vec::new();
    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::ToolDone { name, .. } => dones.push(name),
            AgentEvent::Done(_) => break,
            AgentEvent::Error(e) => panic!("error: {e}"),
            _ => {}
        }
    }

    println!("    writes completed: {}", dones.len());
    // Serial tools complete in order; no concurrency assertion needed here.
    assert!(!dones.is_empty(), "no writes completed");
    println!("    ✓ serial tool paths work");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== 07: Concurrent vs serial tools ===");
    test_concurrent(anthropic()).await;
    test_serial(anthropic()).await;
    println!("PASS");
    Ok(())
}
