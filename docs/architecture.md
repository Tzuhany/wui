# Wui Architecture

## Crate Map

```
wui/
├── crates/
│   ├── wui-core      # Vocabulary. Traits + types. No runtime logic.
│   ├── wui           # Runtime, compression pipeline, providers, public API.
│   ├── wui-mcp       # MCP protocol adapter + McpCatalog (lazy tool discovery).
│   ├── wui-memory    # Optional memory extension (recall/remember/forget/semantic).
│   ├── wui-observe   # Timeline collection and OpenTelemetry span emission.
│   ├── wui-spawn     # Background agent delegation registry.
│   ├── wui-skills    # File-based skill discovery (SKILL.md → ToolCatalog).
│   └── wui-eval      # Testing infrastructure: MockProvider + AgentHarness.
├── examples/
│   └── NN-name/       # Standalone binary crates (cargo run -p example-NN-name)
└── docs/
```

Dependency flow (strictly one-directional):

```
                        ┌─ wui-memory
                        ├─ wui-observe
wui ←── wui-core  ←──├─ wui-mcp
                        ├─ wui-spawn
                        ├─ wui-skills
                        └─ wui-eval
```

User-defined `Tool` / `Hook` / `Provider` implementations depend on `wui-core`
only — they carry no runtime dependency.

---

## `wui-core` — Vocabulary

Defines what everything *is*. No runtime logic, no HTTP clients.
Dependencies are minimal: `serde`, `serde_json`, `async-trait`, `tokio` (traits
only), `uuid`, `tracing`.

```
src/
├── lib.rs          re-exports + prelude
├── message.rs      Message, Role, ContentBlock
├── event.rs        StreamEvent (internal), AgentEvent (external)
├── tool.rs         Tool trait, ToolCtx, ToolOutput
├── provider.rs     Provider trait, ChatRequest
├── hook.rs         Hook trait, HookEvent, HookDecision
└── fmt.rs          system_reminder(), kv() formatting helpers
```

### Two Kinds of Events

`StreamEvent` is the raw LLM stream: text deltas, tool use markers, message end.
It lives inside the engine — users never see it.

`AgentEvent` is what the user receives: higher-level, semantically meaningful,
exhaustive. One `AgentEvent` may be synthesized from multiple `StreamEvent`s.

```
LLM SSE stream
  → [StreamEvent, StreamEvent, ...]
      → Engine processes
          → [AgentEvent, AgentEvent, ...]
              → User's stream
```

### The Tool Trait

Four methods are required: `name`, `description`, `input_schema`, and `call`.
Everything else has safe defaults and is opt-in.

```rust
pub trait Tool: Send + Sync + 'static {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> Value;
    async fn call(&self, input: Value, ctx: &ToolCtx) -> ToolOutput;

    // Semantic cross-runtime properties (any executor understands these).
    fn meta(&self, input: &Value) -> ToolMeta { ToolMeta::default() }

    // Wui-executor-specific tuning (other runtimes may ignore this).
    fn executor_hints(&self, input: &Value) -> ExecutorHints { ExecutorHints::default() }
}
```

#### ToolMeta — cross-runtime semantic properties

```rust
pub struct ToolMeta {
    pub concurrent:           bool,            // default: true
    pub readonly:             bool,            // default: false
    pub destructive:          bool,            // default: false
    pub requires_interaction: bool,            // default: false
    pub permission_key:       Option<String>,  // default: None
}
```

`concurrent` is decided **per invocation**, not per tool type. The same `Bash`
tool can allow `ls` concurrently while forcing `rm` to run serially — different
behavior based on actual arguments, not the tool's identity.

`requires_interaction` causes structural denial under `PermissionMode::Auto`,
preventing headless runs from hanging indefinitely.

`permission_key` is a suffix appended to the tool name for fine-grained
permission rules: `"bash(rm -rf"` can be denied without blocking all `bash` calls.

#### ExecutorHints — Wui-executor tuning

```rust
pub struct ExecutorHints {
    pub summary:          Option<String>,           // display hint for tool history
    pub timeout:          Option<Duration>,         // per-call timeout override
    pub max_output_chars: Option<usize>,            // output truncation limit
    pub max_retries:      u32,                      // auto-retry on error (default: 0)
}
```

`ExecutorHints` fields are **not** part of the universal tool vocabulary. They
are Wui executor implementation details. Runtimes built on `wui-core` without
Wui's executor may ignore these entirely. A tool that only sets `ToolMeta` runs
correctly on any conforming executor.

---

## `wui` — Runtime + Facade

The public API crate. Contains the engine, compression pipeline, and provider
adapters as private modules. Users import only `wui`.

```
src/
├── lib.rs              re-exports + prelude
├── facade/
│   ├── agent.rs        Agent struct (run / stream / session)
│   ├── builder.rs      AgentBuilder (fluent API)
│   ├── session.rs      Session (multi-turn state; HITL via ControlHandle in stream)
│   └── sub_agent.rs    SubAgent synchronous delegation tool
├── runtime/
│   ├── run/            main loop split into stream/parsing/history/tool-batch phases
│   ├── executor.rs     concurrent tool executor (JoinSet-based)
│   ├── permission/     PermissionMode + static/session decision pipeline
│   ├── hooks.rs        HookRunner
│   ├── registry.rs     ToolRegistry
│   ├── session_store.rs SessionStore trait + InMemorySessionStore
│   └── tool_search.rs  deferred tool discovery tool
├── compress/           compression strategies + pipeline
└── providers/
    ├── anthropic/      Anthropic SSE streaming + serialization
    └── openai/         OpenAI-compatible streaming + serialization
```

### The Loop

```
loop {
    1. check_context_pressure() → compress if needed
    2. build ChatRequest
    3. provider.stream(request) with retry
    4. process stream (eager dispatch):
         TextDelta / ThinkingDelta  → emit AgentEvent immediately
         ToolUseEnd                 → run pre-tool hook + permission verdict:
                                        - ready now (allow/deny) → dispatch/record immediately
                                        - needs approval         → queue for deferred HITL
         MessageEnd                 → break inner loop
    5. approve deferred tools concurrently (JoinSet):
         - only tools requiring interactive approval reach this phase
         - HITL prompts fire after MessageEnd
         - each approved tool begins executing when its own auth resolves
    6. executor.collect_remaining() — await all still-running tools
    7. run PostToolUse hooks — may inject system notices for blocked outputs
    8. append tool results to message history (in submission order)
    9. append context injections as <system-reminder> messages
   10. run PreStop hook
   11. evaluate stop condition (EndTurn / MaxTokens / ToolUse)
   12. continue or return RunSummary
}
```

### HITL timing: after the LLM finishes speaking

HITL prompts fire only after `MessageEnd`, so the user reads the assistant's
full intent before deciding. Tools that do not require interactive approval
(static/session allow, Auto/Readonly/Callback allow, immediate deny paths)
can be dispatched while streaming is still in progress.

When multiple deferred tools need HITL in one turn, their approval tasks run
concurrently in a `JoinSet`. One slow decision does not block others from
prompting or starting execution.

### Concurrent Execution

The executor uses `tokio::task::JoinSet` for concurrent tools and a `VecDeque`
for tools that declare `meta.concurrent = false`. Sequential tools wait until
all concurrent work in the batch completes, then run in submission order.

```
JoinSet: ToolA ─────────────────────────┤
         ToolB ──────────────┤           │
         ToolC ────────────────────┤     │
                                   ↓     ↓
VecDeque: ToolD (sequential) ──────────────────────┤
```

### HITL (Human-in-the-Loop)

When `PermissionMode::Ask` fires for a tool call:

1. Engine creates a `oneshot::channel`, wraps request + sender in a `ControlHandle`
2. Engine emits `AgentEvent::Control(handle)` and suspends at `rx.await`
3. Caller calls `handle.approve()`, `handle.deny()`, `handle.approve_always()`, etc.
4. Engine resumes, injects the decision as a `<system-reminder>` the LLM sees
5. If approved, the tool begins executing immediately; if denied, a `ToolOutput::permission_denied` result is written

The `ControlHandle` is the entire API surface for HITL. No `session.respond()`. The handle is self-contained.

No thread is blocked. No polling. Pure async suspension.

### Static Permission Rules

`PermissionRules` (set via `AgentBuilder::allow_tool` / `deny_tool`) is a
static layer evaluated **before** the session's runtime memory and before
`PermissionMode`. Useful for always-safe and never-safe tool calls:

```rust
Agent::builder(provider)
    .allow_tool("memory_recall")      // always allow — read-only
    .allow_tool("bash(ls")            // allow bash only when key starts with "ls"
    .deny_tool("bash(rm")             // always deny rm
    .build()
```

Pattern format: `"tool_name"` matches the tool by name; `"tool_name(prefix"`
additionally requires `meta.permission_key` to start with `prefix`.

Evaluation order for a tool call:
1. Pre-tool hook (can block via `HookDecision::Block`)
2. `meta.requires_interaction` + `PermissionMode::Auto` → structural denial
3. Static deny rules → hard block
4. Session always-denied → hard block
5. Static allow rules → fast pass
6. Session always-allowed → fast pass
7. `PermissionMode` check (readonly gate, HITL prompt, etc.)

### Session Store

Turn-level persistence lives in `wui::runtime`, not in `wui-core`.

`SessionStore` stores and restores **message history** at turn boundaries.
Session permissions (`ApproveAlways` / `DenyAlways`) are ephemeral and reset
on each new session, even when history is restored from a snapshot.

### Compression Pipeline

Three tiers, applied in order until pressure is relieved:

```
L1 · Budget Trim  (free, always runs first)
   ↓ (if still over threshold)
L2 · Collapse     (cheap, no LLM call)
   ↓ (if still over threshold)
L3 · Summarize    (expensive, irreversible — LLM call)
```

**L1** truncates oversized tool results individually.
**L2** folds old messages into a `Collapsed` placeholder.
**L3** asks the LLM to summarize the oldest batch; the summary replaces the originals.

---

## `wui-mcp` — MCP Adapter

Connects any MCP (Model Context Protocol) server as a source of `Arc<dyn Tool>`.

```rust
let tools = McpClient::stdio("uvx", ["mcp-server-filesystem", "/tmp"])
    .await?
    .into_tools()
    .await?;

Agent::builder(provider).tools(tools).build()
```

---

## `wui-memory` — Optional Memory Extension

Three capability traits, each mapping to one agent tool:

| Trait | Tool | What the agent can do |
|-------|------|----------------------|
| `RecallBackend` | `memory_recall` | Search stored memories by query |
| `RememberBackend` | `memory_remember` | Write a new memory |
| `ForgetBackend` | `memory_forget` | Signal "this should stop influencing reasoning" |

`ForgetBackend` is an **intent signal**, not a deletion contract. Backends may
hard-delete, soft-delete, tombstone, or blacklist — the only guarantee is that
forgotten ids will not surface in future recall results.

`InMemoryStore` is a reference backend for development and testing — substring
matching, importance-weighted ranking, vec + RwLock. Not a prescription for
production memory schemas; swap it for pgvector, Redis, or any remote API by
implementing `RecallBackend` + `RememberBackend`.

For semantic search, `wui-memory` also exposes `VectorStore` and
`SemanticMemoryTool`. Plug in any embedding function and get cosine-similarity
recall out of the box:

```rust
// String recall:
let store = Arc::new(InMemoryStore::new());
let tools = all_memory_tools(store);

// Semantic recall (requires an embedding function):
let vec_store = Arc::new(InMemoryVectorStore::new());
let tool = SemanticMemoryTool::new("memory_search", vec_store, embed_fn);
```

---

## `wui-eval` — Testing Infrastructure

`wui-eval` provides the primitives for deterministic agent testing without live
API calls.

```
src/
├── lib.rs
├── mock_provider.rs    MockProvider — queued text/tool/error responses
├── harness.rs          AgentHarness — run + collect events + assert_*
└── scenario.rs         ScenarioRunner — data-driven test tables
```

**`MockProvider`** drains a queue of `MockResponse` values — `Text`, `ToolCall`,
`Error` — in order. Tests compose a scenario by pushing responses onto the queue
before running the agent. The queue is exhausted once, then returns an error, so
lingering calls are immediately visible.

**`AgentHarness`** wraps an agent, drives a run, collects all `AgentEvent`s, and
exposes typed assertions:

```rust
let harness = AgentHarness::run(&agent, "prompt").await;
harness.assert_text_contains("expected");
harness.assert_tool_called("my_tool");
harness.assert_stop_reason(RunStopReason::Completed);
```

**`ScenarioRunner`** runs a table of `Scenario` structs and reports failures
collectively — a lightweight property-test harness for agent behavior.

---

## Crate Maturity

| Crate | Maturity | What "stable" means here |
|-------|----------|--------------------------|
| `wui-core` | **stable** | Trait signatures + type shapes will not break without a semver major. `ExecutorHints` fields are frozen (see freeze policy in `tool.rs`). |
| `wui` | **stable** | Public API surface (builder, stream, session, run, HITL) is settled. Internal runtime modules are `pub(crate)`. |
| `wui-mcp` | **beta** | Useful and tested; transport API may evolve as MCP spec matures. |
| `wui-memory` | **beta** | Trait shapes are stable; builder API and `InMemoryStore` helpers may change. |
| `wui-spawn` | **beta** | Registry and delegation tools work; naming and cancellation API may refine. |
| `wui-observe` | **stabilizing** | Timeline + OTel span API is close to final; minor adjustments may still land. |
| `wui-skills` | **beta** | Frontmatter schema and catalog API may refine. |
| `wui-eval` | **mixed** | `MockProvider` and `AgentHarness` are stable for testing; `ScenarioRunner` API may evolve. |

**Beta → Stable promotion criteria:** at least one smoke test, a usage snippet in the crate's doc comment, CI coverage, and no unresolved API questions in the open issues.

---

## Extension Points

| Want to… | Implement |
|-----------|-----------|
| Use a different LLM | `Provider` trait in `wui-core` |
| Add a new capability | `Tool` trait in `wui-core` |
| Audit or block behavior | `Hook` trait in `wui-core` |
| Persist sessions | `SessionStore` in `wui` |
| Checkpoint/resume runs | `CheckpointStore` in `wui` |
| Add string-match memory | `RecallBackend` + `RememberBackend` in `wui-memory` |
| Add semantic search memory | `VectorStore` + `SemanticMemoryTool` in `wui-memory` |
| Change compression | `CompressStrategy` in `wui::compress` |
| Change permission behavior | `PermissionMode` in `wui` |
| Test deterministically | `MockProvider` + `AgentHarness` in `wui-eval` |
