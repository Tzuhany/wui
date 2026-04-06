# Wuhu Architecture

## Crate Map

```
wuhu/
├── crates/
│   ├── wuhu-core      # Vocabulary. Traits + types. Zero runtime deps.
│   ├── wuhu-engine    # The loop. Streaming execution + HITL.
│   ├── wuhu-compress  # Context compression pipeline (L1/L2/L3).
│   ├── wuhu-providers # LLM adapters: Anthropic, OpenAI.
│   └── wuhu           # Facade. The public API users import.
└── docs/
```

Dependency flow is strictly one-directional:

```
wuhu (facade)
  └── wuhu-engine
        ├── wuhu-compress
        └── wuhu-core  ←── wuhu-providers
                       ←── user-defined Tools / Hooks / Checkpoints
```

---

## `wuhu-core` — Vocabulary

Zero runtime logic. Defines what everything *is*, not how it behaves.

```
src/
├── lib.rs          re-exports everything
├── message.rs      Message, Role, ContentBlock
├── event.rs        StreamEvent (internal), AgentEvent (external)
├── tool.rs         Tool trait, ToolCtx, ToolOutput
├── provider.rs     Provider trait, ChatRequest
├── hook.rs         Hook trait, HookEvent, HookDecision
├── checkpoint.rs   Checkpoint trait, SessionSnapshot
├── memory.rs       Memory trait, MemoryEntry
└── error.rs        WuhuError
```

### Two Kinds of Events

`StreamEvent` is the raw LLM stream: text deltas, tool use markers, message end. It lives inside the engine — users never see it.

`AgentEvent` is what the user receives: higher-level, semantically meaningful, exhaustive. One `AgentEvent` may be synthesized from multiple `StreamEvent`s.

```
LLM SSE stream
  → [StreamEvent, StreamEvent, StreamEvent, ...]
      → Engine processes
          → [AgentEvent, AgentEvent, ...]
              → User's stream
```

### The Tool Trait

The most important design decision in the tool interface:

```rust
fn is_concurrent_for(&self, input: &Value) -> bool { true }
```

Concurrency is decided **per invocation**, not per tool type. A `Bash` tool can allow `ls` to run concurrently while forcing `rm` to run serially — the same tool, different behavior based on the actual arguments. This is not possible with a static `is_concurrent() -> bool`.

---

## `wuhu-engine` — The Heart

```
src/
├── lib.rs         public API: run() → impl Stream<Item=AgentEvent>
├── run.rs         the main loop
├── executor.rs    concurrent tool executor (JoinSet-based)
└── permission.rs  HITL pause/resume via oneshot channels
```

### The Loop

```
loop {
    1. check_context_pressure() → compress if needed
    2. build ChatRequest
    3. provider.stream(request)
    4. StreamProcessor:
         TextDelta/ThinkingDelta   → emit AgentEvent
         ToolUseEnd                → executor.submit() ← starts NOW
         poll executor             → emit completed tool events
         MessageEnd                → break inner loop
    5. executor.collect_remaining()
    6. run hooks (PreComplete)
    7. check stop condition
    8. append tool results → continue
}
```

### Streaming Concurrent Execution

The engine's signature capability. Tools submit as soon as the LLM finishes describing their arguments — not after the LLM's full response.

```
Naive:    LLM ──────────────── done ──→ ToolA ──→ ToolB ──→
Wuhu:     LLM ──ToolA──────ToolB── done ──→
                 │                        ↑ already done
                 └──────────────────────→ ↑ already done
```

The executor uses `tokio::task::JoinSet` for concurrent tools and a `VecDeque` for tools that declare themselves non-concurrent via `is_concurrent_for()`. The JoinSet is polled non-blocking during LLM streaming, so completed tools are harvested eagerly.

### HITL (Human-in-the-Loop)

When `PermissionMode::Ask` and a tool is called:

1. Engine emits `AgentEvent::Control(ControlRequest)`
2. Engine creates a `oneshot::channel` and suspends at `rx.await`
3. User calls `session.respond(ControlResponse)` → sends on the channel
4. Engine resumes with the response injected as a system message
5. Loop continues

No thread is blocked. No polling. Pure async suspension.

---

## `wuhu-compress` — Compression Pipeline

Three tiers, applied in order until pressure is relieved:

```
L1 · Budget Trim
   ↓ (if still over)
L2 · Collapse
   ↓ (if still over)
L3 · Summarize
```

**L1 — Budget Trim** (free, always runs)
Individual tool results that exceed `budget_per_result` tokens are truncated. The LLM is told the full result was saved. No LLM call required.

**L2 — Collapse** (cheap, reversible)
Messages older than the active window are replaced with `ContentBlock::Compressed` placeholders that record `folded_count` and a brief rule-generated summary. The original messages are retained in the `SessionSnapshot` and can be surfaced again. No LLM call required.

**L3 — Summarize** (expensive, irreversible)
The LLM itself is asked to summarize the oldest message batch. The summary replaces the originals permanently in the working context. An `AgentEvent::Compressed` is emitted so the user knows this happened.

The pipeline stops at the first tier that relieves sufficient pressure.

---

## `wuhu-providers` — LLM Adapters

```
src/
├── lib.rs        feature re-exports
├── anthropic.rs  Anthropic SSE streaming (prompt cache, extended thinking)
└── openai.rs     OpenAI SSE streaming
```

Providers are feature-gated:

```toml
wuhu = { version = "0.1", features = ["anthropic"] }
```

`ChatRequest` has an `extensions: HashMap<String, Value>` field for provider-specific capabilities (Anthropic's `betas`, `thinking`, OpenAI's `store`, etc.) that don't belong in the universal interface.

---

## `wuhu` — Facade

The only crate most users will import directly.

```
src/
├── lib.rs      re-exports + prelude
├── agent.rs    Agent struct
├── builder.rs  AgentBuilder (the fluent API)
└── session.rs  Session (multi-turn state + respond())
```

### Three API Levels

**Level 1 — One line:**
```rust
let output = wuhu::run(Anthropic::new(key), "You are...", "Hello").await?;
```

**Level 2 — Streaming:**
```rust
let agent = Agent::builder()
    .provider(Anthropic::new(key))
    .tool(WebSearch::new())
    .build();

let mut stream = agent.stream("Search for...").await?;
while let Some(event) = stream.next().await { ... }
```

**Level 3 — Full control:**
```rust
let agent = Agent::builder()
    .provider(Anthropic::new(key))
    .tool(WebSearch::new())
    .hook(AuditLog::new())
    .checkpoint(MyStore::new())
    .memory(MyMemory::new())
    .permission(PermissionMode::Ask)
    .on_control(|req| async move { ControlResponse::approve() })
    .build();

let mut session = agent.session("id");
session.send("Hello").await?;
session.respond(ControlResponse::approve()).await?;
```

---

## Extension Points

| Want to... | Implement |
|-----------|-----------|
| Use a different LLM | `Provider` in `wuhu-core` |
| Add a new capability | `Tool` in `wuhu-core` |
| Audit or block behavior | `Hook` in `wuhu-core` |
| Persist sessions | `Checkpoint` in `wuhu-core` |
| Add memory recall | `Memory` in `wuhu-core` |
| Change compression | `CompressPipeline` config in `wuhu-compress` |
| Change permission behavior | `PermissionMode` + `on_control` callback |
