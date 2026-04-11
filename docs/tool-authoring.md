# Tool Authoring Guide

The preferred way to write a tool is `TypedTool` + `#[derive(ToolInput)]`. You get
compile-time schema generation, automatic JSON deserialization, and clean typed input
— with no boilerplate.

## TypedTool — the default path

```rust
use async_trait::async_trait;
use wui::{ToolArgs, ToolCtx, ToolOutput, TypedTool};
use wui_macros::ToolInput;

#[derive(ToolInput)]
struct SearchInput {
    /// Search query string.
    query: String,
    /// Maximum number of results to return.
    limit: Option<u32>,
}

struct SearchTool;

#[async_trait]
impl TypedTool for SearchTool {
    type Input = SearchInput;

    fn name(&self) -> &str { "search" }
    fn description(&self) -> &str { "Search the knowledge base." }

    async fn call_typed(&self, input: SearchInput, _ctx: &ToolCtx) -> ToolOutput {
        let limit = input.limit.unwrap_or(10);
        ToolOutput::success(format!("Top {} results for '{}'", limit, input.query))
    }
}
```

`#[derive(ToolInput)]` generates the JSON Schema and deserialization from doc comments and
field types. `Option<T>` fields are optional; all others are required. The macro satisfies
the `ToolArgs` bound automatically — no hand-written schema, no raw `Value` parsing.

Register with the builder:

```rust
let agent = Agent::builder(provider)
    .tool(SearchTool)
    .build();
```

## ToolMeta — semantic flags

Override `meta()` when your tool has non-default semantics:

```rust
fn meta(&self, _: &SearchInput) -> ToolMeta {
    ToolMeta {
        readonly: true,       // no side effects → runs in Readonly mode
        concurrent: true,     // safe to run alongside other tools (default)
        destructive: false,   // shown to user in HITL prompt
        requires_interaction: false,
        permission_key: None,
        ..ToolMeta::default()
    }
}
```

**Rules of thumb:**
- `readonly: true` if the tool only reads state (search, grep, file read). Lets it run in `PermissionMode::Readonly`.
- `concurrent: false` if the tool modifies shared state. Two concurrent writes to the same path would corrupt each other.
- `destructive: true` if the action cannot be undone (delete, send email, deploy). Shown to the user in the HITL prompt.
- `requires_interaction: true` if the tool needs a human in the loop (browser, GUI). Denied automatically in `PermissionMode::Auto`.

## ExecutorHints — runtime tuning

Override `executor_hints()` for timeout, retries, and output limits:

```rust
fn executor_hints(&self, _: &SearchInput) -> ExecutorHints {
    ExecutorHints {
        timeout: Some(Duration::from_secs(30)),
        max_retries: 2,
        max_output_chars: Some(50_000),
        summary: Some("searched knowledge base".into()),
        ..ExecutorHints::default()
    }
}
```

- `timeout` — kills the tool after this duration. Use for network calls.
- `max_retries` — only `FailureKind::Execution` is retried. Schema errors, permission denials, and hook blocks are never retried.
- `max_output_chars` — truncates output before it reaches the LLM context. If a `ResultStore` is configured, the full output is persisted first.
- `summary` — one-line string preserved in L2 collapse placeholders so the LLM retains a trace of what this tool did.

## Permission key — sub-tool granularity

For tools with a wide input surface (like a shell), use `permission_key` to enable fine-grained rules:

```rust
fn meta(&self, input: &ShellInput) -> ToolMeta {
    ToolMeta {
        permission_key: Some(input.command.clone()),
        ..ToolMeta::default()
    }
}
```

This lets the builder write rules like:

```rust
.allow_tool("bash(ls)")
.deny_tool("bash(rm -rf)")
```

For even more control, override `permission_matcher()` to return a wildcard matching closure.

## InterruptBehavior

Override when the default (`Block`) is wrong:

```rust
fn interrupt_behavior(&self) -> InterruptBehavior {
    InterruptBehavior::Cancel  // safe to abort: search, read, grep
}
```

Use `Cancel` for tools with no side effects. Keep `Block` for anything that writes state.

## ToolOutput patterns

```rust
// Success
ToolOutput::success("Found 42 files.")

// Success with structured data for callers
ToolOutput::success("Found 42 files.")
    .with_structured(json!({"count": 42}))

// Success with artifacts
ToolOutput::success("Chart generated.")
    .with_artifacts([Artifact::text("chart.csv", csv_data)])

// Success with context injection
ToolOutput::success("Skill loaded.")
    .inject(ContextInjection::new("Remember: always use snake_case."))

// Error (retryable by the executor)
ToolOutput::error("network timeout")

// Invalid input (not retried)
ToolOutput::invalid_input("'url' must start with https://")
```

## Cancellation and progress

Check `ctx.cancel` periodically in long-running tools, and call `ctx.report()` to
emit real-time progress as `AgentEvent::ToolProgress`:

```rust
async fn call_typed(&self, input: SearchInput, ctx: &ToolCtx) -> ToolOutput {
    for (i, chunk) in chunks.iter().enumerate() {
        if ctx.cancel.is_cancelled() {
            return ToolOutput::error("cancelled");
        }
        process(chunk).await;
        ctx.report(format!("processed {}/{}", i + 1, chunks.len()));
    }
    ToolOutput::success("done")
}
```

## Side effects and idempotency

The framework does NOT guarantee exactly-once execution. If a run is interrupted and
resumed from a checkpoint, tools that already ran will NOT be replayed — but the
checkpoint boundary is per-iteration, not per-tool.

**If your tool has side effects:**
- Make it idempotent where possible (use PUT not POST, write to deterministic paths).
- Use `destructive: true` so HITL prompts the user.
- Check for prior completion before acting (e.g., check if a file exists before creating it).

**The framework guarantees:**
- `call_typed()` is only invoked after schema validation passes.
- The tool's output always reaches the LLM, even if the tool panics (as an error message).
- A timeout produces a structured error, not a hang.

## Low-level: implementing `Tool` directly

Use the raw `Tool` trait when you need a dynamic schema (e.g., generated at runtime from
user configuration) or when you are wrapping an external tool definition that already
provides its own JSON Schema.

```rust
#[async_trait]
impl Tool for MyDynamicTool {
    fn name(&self) -> &str { "my_tool" }
    fn description(&self) -> &str { "..." }

    fn input_schema(&self) -> serde_json::Value {
        // Your schema, however you build it.
        json!({ "type": "object", "properties": { ... } })
    }

    async fn call(&self, input: serde_json::Value, ctx: &ToolCtx) -> ToolOutput {
        // Parse manually.
        let x = input["field"].as_str().unwrap_or_default();
        ToolOutput::success(format!("got {x}"))
    }
}
```

All `ToolMeta`, `ExecutorHints`, and `InterruptBehavior` overrides work the same way on
raw `Tool` as on `TypedTool`.
