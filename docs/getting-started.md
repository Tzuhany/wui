# Getting Started

Build your first LLM agent in four steps — each one adds a capability.

## Prerequisites

Add `wui` to your project with your preferred provider:

```toml
[dependencies]
wui = { path = "../wui", features = ["anthropic"] }
tokio = { version = "1", features = ["full"] }
futures = "0.3"
anyhow = "1"
```

Set your API key:

```sh
export ANTHROPIC_API_KEY=sk-...
```

## Step 1: Streaming text (30 seconds)

The simplest agent: send a prompt, stream the response.

```rust
use futures::StreamExt;
use wui::{Agent, AgentEvent};
use wui::providers::Anthropic;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = Agent::builder(Anthropic::new(std::env::var("ANTHROPIC_API_KEY")?))
        .system("You are a concise, helpful assistant.")
        .build();

    // Fire-and-forget — just give me the text:
    let text = agent.run("What is the capital of France?").await?;
    println!("{text}");

    // Or stream token by token:
    let mut stream = agent.stream("Name three programming languages.");
    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TextDelta(t) => print!("{t}"),
            AgentEvent::Done(_) => { println!(); break; }
            AgentEvent::Error(e) => return Err(e.into()),
            _ => {}
        }
    }

    Ok(())
}
```

Three entry points, one agent:
- `run()` — returns the final text (blocks until done)
- `stream()` — returns an event stream (tokens arrive as they're generated)
- `session()` — multi-turn conversation (Step 4)

## Step 2: Add a tool

Tools let the LLM take actions. Implement the `TypedTool` trait:

```rust
use serde::Deserialize;
use wui::{Agent, AgentEvent, PermissionMode, ToolOutput, ToolCtx, ToolMeta};
use wui::{TypedTool, ToolArgs, ToolInput};
use wui::providers::Anthropic;
use futures::StreamExt;

// 1. Define the input schema
#[derive(Deserialize, ToolInput)]
struct CalculatorInput {
    /// Mathematical expression to evaluate
    expression: String,
}

// 2. Implement the tool
struct Calculator;

impl TypedTool for Calculator {
    type Input = CalculatorInput;

    fn name() -> &'static str { "calculator" }
    fn description() -> &'static str { "Evaluate a mathematical expression" }

    fn meta(_input: &Self::Input) -> ToolMeta {
        ToolMeta { readonly: true, concurrent: true, ..Default::default() }
    }

    async fn call_typed(input: Self::Input, _ctx: &ToolCtx) -> ToolOutput {
        // In a real tool you'd parse and evaluate — here we just echo.
        ToolOutput::success(format!("Result of '{}' = 42", input.expression))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = Agent::builder(Anthropic::new(std::env::var("ANTHROPIC_API_KEY")?))
        .system("Use the calculator tool when asked to compute.")
        .tool(Calculator)
        .permission(PermissionMode::Auto)  // tools run without asking
        .build();

    let mut stream = agent.stream("What is 6 * 7?");
    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TextDelta(t) => print!("{t}"),
            AgentEvent::ToolStart { name, .. } => println!("[tool: {name}]"),
            AgentEvent::ToolDone { name, output, .. } => println!("[{name} → {output}]"),
            AgentEvent::Done(_) => { println!(); break; }
            AgentEvent::Error(e) => return Err(e.into()),
            _ => {}
        }
    }

    Ok(())
}
```

Key concepts:
- `ToolInput` derive macro generates JSON Schema from your struct
- `ToolMeta` tells the executor *how* to run the tool (read-only? concurrent? destructive?)
- `PermissionMode::Auto` lets tools run without human approval

## Step 3: Human-in-the-loop permissions

For tools that modify state, you want human approval. Switch to `PermissionMode::Ask` (the default) and handle `AgentEvent::Control`:

```rust
use futures::StreamExt;
use wui::{Agent, AgentEvent, PermissionMode};
use wui::providers::Anthropic;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = Agent::builder(Anthropic::new(std::env::var("ANTHROPIC_API_KEY")?))
        .system("You are a helpful assistant with file tools.")
        .tool(Calculator)  // from Step 2
        .permission(PermissionMode::Ask)
        .build();

    let mut stream = agent.stream("Calculate 6 * 7");
    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TextDelta(t) => print!("{t}"),

            // The agent wants to use a tool — you decide.
            AgentEvent::Control(handle) => {
                println!("Agent wants to: {}", handle.request.description());

                // Options:
                handle.approve();              // allow this once
                // handle.approve_always();     // allow all future calls to this tool
                // handle.deny("not allowed");  // deny this once
                // handle.deny_always("never"); // deny all future calls
            }

            AgentEvent::ToolDone { name, output, .. } => {
                println!("[{name}] {output}");
            }

            AgentEvent::Done(_) => break,
            AgentEvent::Error(e) => return Err(e.into()),
            _ => {}
        }
    }

    Ok(())
}
```

You can also use static rules for fine-grained control:

```rust
Agent::builder(provider)
    .tool(Calculator)
    .tool(FileWriter)
    .allow_tool("calculator")        // always allow calculator
    .deny_tool("bash(rm -rf")        // never allow dangerous bash patterns
    .permission(PermissionMode::Ask) // ask for everything else
    .build();
```

## Step 4: Multi-turn sessions

Sessions preserve conversation history across turns:

```rust
use futures::StreamExt;
use wui::{Agent, AgentEvent, PermissionMode};
use wui::providers::Anthropic;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = Agent::builder(Anthropic::new(std::env::var("ANTHROPIC_API_KEY")?))
        .system("You are a helpful assistant.")
        .permission(PermissionMode::Auto)
        .build();

    let session = agent.session("my-conversation").await;

    // Turn 1
    let mut s1 = session.send("My name is Alice.").await;
    while let Some(event) = s1.next().await {
        match event {
            AgentEvent::TextDelta(t) => print!("{t}"),
            AgentEvent::Done(_) => { println!(); break; }
            _ => {}
        }
    }

    // Turn 2 — the agent remembers turn 1
    let mut s2 = session.send("What's my name?").await;
    while let Some(event) = s2.next().await {
        match event {
            AgentEvent::TextDelta(t) => print!("{t}"),
            AgentEvent::Done(_) => { println!(); break; }
            _ => {}
        }
    }

    Ok(())
}
```

For persistent sessions across restarts, add a `SessionStore`:

```rust
use wui::InMemorySessionStore; // or implement your own

let agent = Agent::builder(provider)
    .session_store(InMemorySessionStore::new())
    .build();
```

## What's next?

- **Hooks** — intercept tool calls, inject context before compression, gate the stop condition. See `examples/05_hooks.rs`.
- **Structured output** — get typed Rust values from the LLM with `agent.run_as::<MyStruct>()`.
- **Sub-agents** — delegate tasks to child agents with `.sub_agent("name", "desc", child_agent)`.
- **Context compression** — automatic; configure thresholds via `CompressPipeline`.
- **Other providers** — see [docs/providers.md](providers.md) for Ollama, vLLM, Azure, and more.

Read `docs/philosophy.md` for design principles, `docs/architecture.md` for the crate structure, and `docs/tool-authoring.md` for the full tool API.
