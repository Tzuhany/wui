/// Example: agent with a custom tool.
///
/// Run with:
///   ANTHROPIC_API_KEY=sk-... cargo run --example with_tools --features anthropic

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};
use wuhu::prelude::*;
use wuhu_providers::Anthropic;

// ── Custom tool ───────────────────────────────────────────────────────────────

struct Calculator;

#[async_trait]
impl Tool for Calculator {
    fn name(&self) -> &str { "calculator" }

    fn description(&self) -> &str {
        "Evaluate a mathematical expression and return the result."
    }

    fn prompt(&self) -> String {
        "Use this tool to evaluate mathematical expressions.\n\
         Input: { \"expression\": \"2 + 2 * 3\" }\n\
         Returns the numeric result as a string.".to_string()
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        })
    }

    async fn call(&self, input: Value, _ctx: &ToolCtx) -> ToolOutput {
        let expr = match input["expression"].as_str() {
            Some(e) => e,
            None    => return ToolOutput::error("missing expression"),
        };

        // Toy evaluator: only handles simple integer arithmetic.
        match eval(expr) {
            Some(result) => ToolOutput::success(result.to_string()),
            None         => ToolOutput::error(format!("could not evaluate: {expr}")),
        }
    }
}

fn eval(expr: &str) -> Option<i64> {
    // Very simplified — real implementations would use a proper parser.
    let parts: Vec<&str> = expr.split('+').collect();
    if parts.len() == 2 {
        let a: i64 = parts[0].trim().parse().ok()?;
        let b: i64 = parts[1].trim().parse().ok()?;
        return Some(a + b);
    }
    None
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let key   = std::env::var("ANTHROPIC_API_KEY")?;
    let agent = Agent::builder(Anthropic::new(key))
        .system("You are a helpful math assistant.")
        .tool(Calculator)
        .permission(PermissionMode::Auto)
        .build();

    let mut stream = agent.stream("What is 42 + 58?").await;

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TextDelta(t)                => print!("{t}"),
            AgentEvent::ToolStart { name, .. }      => eprintln!("\n[→ {name}]"),
            AgentEvent::ToolDone  { name, ms, .. }  => eprintln!("[← {name} {ms}ms]"),
            AgentEvent::Done(_)                     => { println!(); break; }
            AgentEvent::Error(e)                    => { eprintln!("error: {e}"); break; }
            _                                       => {}
        }
    }

    Ok(())
}
