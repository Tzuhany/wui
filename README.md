# Wui

> The framework is an executor, not a thinker.

Wui is a Rust framework for building LLM agents. It handles the hard parts — streaming concurrent tool execution, automatic context compression, and native human-in-the-loop — so you can focus on what the agent should *do*, not how it runs.

```rust
use wui::{Agent, AgentEvent};
use wui::providers::Anthropic;
use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = Agent::builder(Anthropic::new(std::env::var("ANTHROPIC_API_KEY")?))
        .system("You are a helpful assistant.")
        .build();

    let mut stream = agent.stream("What is the capital of France?");
    while let Some(event) = stream.next().await {
        if let AgentEvent::TextDelta(text) = event {
            print!("{text}");
        }
    }
    Ok(())
}
```

## Why Wui

**Streaming concurrent tools.** Tools start executing the moment the LLM describes them — not after it finishes. Two tool calls in one response run in parallel, automatically.

**Three-tier context compression.** Every long-running agent eventually runs out of context. Wui handles this gracefully: trim tool outputs, collapse old messages, summarize with the LLM — in that order, stopping as soon as pressure is relieved.

**Native HITL.** Suspend execution, ask the human, resume. It's a first-class API, not an afterthought.

**Runtime-first persistence.** Bring your own database. Wui exposes a turn-level `SessionStore` in the runtime, so persistence stays optional without pretending it is part of the core vocabulary.

**Provider-owned tuning.** Provider-specific knobs belong on the provider itself, not in the generic agent builder. Configure them where they live.

## What is stable today

| What | Status | Notes |
|------|--------|-------|
| `wui-core` traits (`Tool`, `Hook`, `Provider`) | **stable** | Will not break without a semver major |
| `wui` runtime (run loop, HITL, compression, permission flow) | **stable** | |
| `wui` Anthropic provider | **stable** | |
| `wui` OpenAI provider | **stable** | |
| `wui::Session` multi-turn API | **stable** | |
| `wui::StructuredRun` | **stable** | |
| All beta extension crates | **beta** | APIs may change between minor versions |

**stable** means: adding the crate at a pinned version will not break between patch releases.
**beta** means: useful and tested, but method signatures may evolve.

## Recommended learning path

Start narrow. Add surface only as you need it.

1. **`examples/01_streaming`** — stream text from the LLM; the minimal end-to-end loop
2. **`examples/02_run`** — `agent.run()` for when you don't need streaming
3. **`examples/03_tools`** — add a tool; see how the loop calls it and feeds the result back
4. **`examples/04_hitl`** — suspend on a tool call, ask the user, resume
5. **`examples/05_hooks`** — observe and block tool calls from outside the agent
6. **`examples/06_session`** — multi-turn conversation with memory of the prior turn
7. **`examples/07_concurrent`** — two tools in one response running in parallel
8. **`examples/08_readonly`** — permission mode that only permits read-only tools automatically
9. **`examples/10_failure_kinds`** — distinguish schema errors from execution errors
10. Extension crates (`wui-memory`, `wui-mcp`, `wui-observe`, …) once the above are familiar

## Where to start

New to wui? Start with `wui`. The extension crates are companion libraries — useful for specific needs, not required to build real agents.

## Crate map

| Crate | Role | Maturity |
|-------|------|----------|
| `wui-core` | Vocabulary: Provider, Tool, Hook, Message, events | stable |
| `wui` | Runtime executor + builder facade | stable |
| `wui-mcp` | Bridge MCP ecosystem tools into Wui catalogs | beta |
| `wui-memory` | Reference memory capabilities (string + vector) | beta |
| `wui-spawn` | Background sub-agent delegation on top of Wui runs | beta |
| `wui-observe` | Structured timeline + OpenTelemetry spans | beta |
| `wui-workflow` | Deterministic pipeline orchestration overlay | beta |
| `wui-skills` | File-based discoverable instruction catalogs | beta |
| `wui-eval` | Testing infrastructure: MockProvider + AgentHarness | beta |

**stable** — API is settled; breaking changes follow semver.
**beta** — Useful and tested, but API may evolve between minor versions.
**experimental** — Proof of concept; expect breaking changes.

## Provider features

```toml
wui = { version = "0.1", features = ["anthropic"] }  # Anthropic
wui = { version = "0.1", features = ["full"] }        # all production-ready providers
```

## Extensions

### Memory — recall and remember across turns

```toml
wui-memory = "0.1"
```

```rust
use std::sync::Arc;
use wui_memory::{InMemoryStore, all_memory_tools};

let store = Arc::new(InMemoryStore::new());

let agent = Agent::builder(provider)
    .tools(all_memory_tools(store))
    .permission(PermissionMode::Auto)
    .build();
```

`wui-memory` defines three capability traits (`RecallBackend`, `RememberBackend`,
`ForgetBackend`) and ships one reference backend (`InMemoryStore`) for development
and testing. The traits are the point — not the implementation. Every application
has its own idea of what memory means, so the backend is always yours to choose:
pgvector, Redis, SQLite, a remote API. `InMemoryStore` makes it easy to build and
test before you've decided.

### MCP — any MCP server as a tool source

```toml
wui-mcp = "0.1"
```

**Eager** — connect at build time, tools always in the prompt:

```rust
use wui_mcp::McpClient;

let tools = McpClient::stdio("uvx", ["mcp-server-filesystem", "/tmp"])
    .await?
    .into_tools();

Agent::builder(provider).tools(tools).build()
```

**Lazy catalog** — connect on first use, tools discovered via `tool_search`:

```rust
use wui_mcp::McpCatalog;

Agent::builder(provider)
    .catalog(McpCatalog::stdio("uvx", &["mcp-server-filesystem", "/tmp"]).namespace("fs"))
    .build()
```

Use catalogs when you have many MCP servers — they connect lazily and never
appear in the initial prompt, so token cost grows only with what the agent
actually uses.

### Observability — structured timeline + OTel spans

```toml
wui-observe = "0.1"
```

```rust
use wui_observe::observe;

let (stream, timeline) = observe(agent.stream("prompt")).await;
// drive stream...
let tl = timeline.await;
println!("{}", tl.summary());
```

### Delegation — background agents

```toml
wui-spawn = "0.1"
```

```rust
use wui_spawn::AgentRegistry;

let registry = AgentRegistry::new();
let supervisor = Agent::builder(provider)
    .tools(registry.delegation_tools("analyst", "Analyse data.", analyst))
    .build();
// supervisor can now call: spawn(prompt) → job_id
//                          agent_status(job_id) / agent_await(job_id) / agent_cancel(job_id)
```

> **`SubAgent` vs `wui-spawn`**
> `wui::SubAgent` wraps one agent as a synchronous tool — supervisor calls it
> and waits for the result within the same turn. `wui-spawn` manages a
> registry of background agents that run across turns: spawn now, check status
> later, await when ready.

### Skills — file-based instruction sets

```toml
wui-skills = "0.1"
```

```rust
use wui_skills::SkillsCatalog;

Agent::builder(provider)
    .catalog(SkillsCatalog::new("./skills"))
    .build()
// Each skills/git-commit.md becomes a discoverable tool that injects
// its content as a <system-reminder> when the agent activates it.
```

## License

MIT OR Apache-2.0
