# Tool Authoring Guide

Best practices for implementing `Tool` in Wui.

## The minimum

Four methods are required:

```rust
fn name(&self) -> &str;
fn description(&self) -> &str;
fn input_schema(&self) -> Value;
async fn call(&self, input: Value, ctx: &ToolCtx) -> ToolOutput;
```

Everything else has sensible defaults. Override only what differs.

## ToolMeta — semantic flags

Override `meta()` when your tool has non-default semantics:

```rust
fn meta(&self, _input: &Value) -> ToolMeta {
    ToolMeta {
        readonly: true,       // no side effects → runs in Readonly mode
        concurrent: true,     // safe to run alongside other tools (default)
        destructive: false,   // shown to user in HITL prompt
        requires_interaction: false,
        permission_key: None, // sub-tool granularity (see below)
        ..ToolMeta::default()
    }
}
```

**Rules of thumb:**
- `readonly: true` if the tool only reads state (search, grep, file read). This lets it run in `PermissionMode::Readonly`.
- `concurrent: false` if the tool modifies shared state. Two concurrent file writes to the same path would corrupt each other.
- `destructive: true` if the action cannot be undone (delete, send email, deploy). The HITL prompt shows this to the user.
- `requires_interaction: true` if the tool needs a human in the loop (browser, GUI). Denied automatically in `PermissionMode::Auto`.

## ExecutorHints — runtime tuning

Override `executor_hints()` for timeout, retries, and output limits:

```rust
fn executor_hints(&self, _input: &Value) -> ExecutorHints {
    ExecutorHints {
        timeout: Some(Duration::from_secs(30)),
        max_retries: 2,              // retry on Execution errors only
        max_output_chars: Some(50_000),
        summary: Some("searched files".into()),
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
fn meta(&self, input: &Value) -> ToolMeta {
    let cmd = input["command"].as_str().unwrap_or("");
    ToolMeta {
        permission_key: Some(cmd.to_string()),
        ..ToolMeta::default()
    }
}
```

This lets the builder write rules like:

```rust
.allow_tool("bash(ls)")     // allow ls
.deny_tool("bash(rm -rf)")  // deny rm -rf
```

For even more control, override `permission_matcher()` to return a wildcard matching closure:

```rust
fn permission_matcher(&self, input: &Value) -> Option<PermissionMatcher> {
    let cmd = input["command"].as_str()?.to_string();
    Some(Box::new(move |pattern| {
        cmd == pattern || cmd.starts_with(&format!("{pattern} "))
    }))
}
```

## InterruptBehavior

Override when the default (`Block`) is wrong:

```rust
fn interrupt_behavior(&self) -> InterruptBehavior {
    InterruptBehavior::Cancel  // safe to abort: search, read, grep
}
```

Use `Cancel` for tools with no side effects. Keep `Block` for anything that writes state.

## Side effects and idempotency

The framework does NOT guarantee exactly-once execution. If a run is
interrupted and resumed from a checkpoint, tools that already ran will NOT
be replayed — but the checkpoint boundary is per-iteration, not per-tool.

**If your tool has side effects:**
- Make it idempotent where possible (use PUT not POST, write to deterministic paths).
- Use `destructive: true` so HITL prompts the user.
- Consider checking for prior completion before acting (e.g., check if a file exists before creating it).

**The framework guarantees:**
- `call()` is only invoked after schema validation passes.
- The tool's output always reaches the LLM, even if the tool panics (as an error message).
- A timeout produces a structured error, not a hang.

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

## Cancellation

Check `ctx.cancel` periodically in long-running tools:

```rust
async fn call(&self, input: Value, ctx: &ToolCtx) -> ToolOutput {
    for chunk in chunks {
        if ctx.cancel.is_cancelled() {
            return ToolOutput::error("cancelled");
        }
        process(chunk).await;
        ctx.report(format!("processed {}/{}", i, total));
    }
    ToolOutput::success("done")
}
```

`ctx.report()` emits `AgentEvent::ToolProgress` for real-time UI updates.
