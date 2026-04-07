# wui-spawn

Background delegation built on top of `wui` runs.

Background sub-agent delegation for Wui. Lets a supervisor agent spawn sub-agents asynchronously, poll their status across turns, await their results, or cancel them — without blocking the supervisor's own loop.

## Install

```toml
[dependencies]
wui-spawn = "0.1"
```

## Usage

```rust
use wui_spawn::AgentRegistry;

let registry = AgentRegistry::new();

let analyst = Agent::builder(provider.clone())
    .system("You are a data analyst.")
    .build();

// delegation_tools returns four tools: delegate, agent_status,
// agent_await, and agent_cancel — all sharing the registry.
let supervisor = Agent::builder(provider)
    .tools(registry.delegation_tools("analyst", "Analyse data.", analyst))
    .build();
```

The supervisor's LLM can now:

| Tool call | What happens |
|-----------|-------------|
| `delegate(prompt)` | Spawns the sub-agent; returns `job_id` immediately |
| `agent_status(job_id)` | Returns `"running"`, `"done: <result>"`, or `"failed: <err>"` |
| `agent_await(job_id)` | Blocks until the sub-agent finishes; returns the result |
| `agent_cancel(job_id)` | Cancels the running job |

## `SubAgent` vs `wui-spawn`

`wui::SubAgent` wraps one agent as a synchronous tool — the supervisor calls it and waits for the result within the same turn. Use it when the delegation is simple and the wait is short.

`wui-spawn` manages a registry of background agents that run across turns: spawn now, check status later, await when ready. Use it for long-running tasks where the supervisor should continue doing other work while waiting.

Full docs: https://github.com/Tzuhany/wui
