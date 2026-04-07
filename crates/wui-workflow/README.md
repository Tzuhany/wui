# wui-workflow

Deterministic orchestration overlay on top of `wui` agents — not a competing runtime.

Deterministic pipeline orchestration on top of the Wui runtime. When a task is better expressed as explicit control flow than open-ended LLM reasoning, use this crate: sequential stages, conditional branches, and parallel fan-out — all composable.

## Install

```toml
[dependencies]
wui-workflow = "0.1"
```

## Usage

### Sequential pipeline

```rust
use wui_workflow::{Pipeline, AgentStep};

let pipeline = Pipeline::new("research")
    .step(AgentStep::new("fetch",     fetcher_agent))
    .step(AgentStep::new("summarise", summariser_agent));

let result = pipeline.run("quantum computing").await?;
println!("{}", result.output);
println!("total: {}ms", result.elapsed.as_millis());
```

### Conditional branch

```rust
use wui_workflow::Branch;

Pipeline::new("triage")
    .step(Branch::new(
        "route",
        |input: &str| input.to_lowercase().contains("urgent"),
        AgentStep::new("urgent_path", urgent_agent),
        AgentStep::new("normal_path", normal_agent),
    ));
```

### Parallel fan-out

```rust
use wui_workflow::Parallel;
use std::sync::Arc;

Pipeline::new("multi-perspective")
    .step(Parallel::new(
        "analyse",
        vec![
            Arc::new(AgentStep::new("technical", tech_agent)),
            Arc::new(AgentStep::new("business",  biz_agent)),
        ],
        |results| results.join("\n\n---\n\n"),
    ));
```

## Primitives

| Type | What it does |
|------|-------------|
| `Pipeline` | Sequential `TextStep` chain; short-circuits on error |
| `AgentStep` | Runs an `Agent` with the step's input as prompt |
| `MapStep` | Applies a synchronous `String → String` function |
| `Branch` | Routes to one of two steps based on a predicate |
| `Parallel` | Runs multiple steps concurrently; joins their outputs |

`TextStep` is the composition unit — implement it for custom steps.

Full docs: https://github.com/Tzuhany/wui
