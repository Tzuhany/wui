//! Custom tool with `TypedTool` and `#[derive(ToolInput)]`.
//!
//! Shows the preferred way to write Rust tools: derive the input schema and
//! parser automatically, implement `TypedTool` with strongly-typed input, and
//! register via `.tool()`. No manual schema JSON or `ToolInput` calls needed.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-... cargo run --example 02_tools -p wui --features anthropic

use async_trait::async_trait;
use wui::providers::Anthropic;
use wui::{Agent, PermissionMode, ToolCtx, ToolMeta, ToolOutput, TypedTool};
use wui_macros::ToolInput;

// ── Input type ────────────────────────────────────────────────────────────────

/// #[derive(ToolInput)] generates the JSON schema and parser automatically.
/// Doc comments on fields become `"description"` entries in the schema.
/// `Option<T>` fields are optional; all others are required.
#[derive(ToolInput)]
struct CurrentTimeInput {
    /// Optional timezone offset in hours from UTC (e.g. 9 for JST). Defaults to UTC.
    offset_hours: Option<i64>,
}

// ── Tool implementation ───────────────────────────────────────────────────────

struct CurrentTimeTool;

#[async_trait]
impl TypedTool for CurrentTimeTool {
    type Input = CurrentTimeInput;

    fn name(&self) -> &str {
        "current_time"
    }

    fn description(&self) -> &str {
        "Returns the current UTC time as a UNIX timestamp, with optional timezone offset."
    }

    fn meta(&self, _input: &CurrentTimeInput) -> ToolMeta {
        // Read-only: no external state is modified. Safe to run concurrently.
        ToolMeta {
            readonly: true,
            concurrent: true,
            ..ToolMeta::default()
        }
    }

    async fn call_typed(&self, input: CurrentTimeInput, _ctx: &ToolCtx) -> ToolOutput {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let offset_secs = input.offset_hours.unwrap_or(0) * 3600;
        let adjusted = now as i64 + offset_secs;
        ToolOutput::success(format!("Current UNIX timestamp (UTC{:+}h): {adjusted}", input.offset_hours.unwrap_or(0)))
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;

    let agent = Agent::builder(Anthropic::new(api_key))
        .system("You are a helpful assistant with access to real-time tools.")
        .tool(CurrentTimeTool)
        .permission(PermissionMode::Auto)
        .build();

    // collect_text() replaces the manual while-let event loop for the common case.
    let response = agent
        .stream("What is the current time in Tokyo (UTC+9)?")
        .collect_text()
        .await?;

    println!("{response}");
    Ok(())
}
