# Wuhu 🌀

> The framework is an executor, not a thinker.

Wuhu is a Rust framework for building LLM agents. It handles the hard parts — streaming concurrent tool execution, automatic context compression, and native human-in-the-loop — so you can focus on what the agent should *do*, not how it runs.

```rust
use wuhu::{Agent, AgentEvent};
use wuhu_providers::Anthropic;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = Agent::builder()
        .provider(Anthropic::new(std::env::var("ANTHROPIC_API_KEY")?))
        .system("You are a helpful assistant.")
        .build();

    let mut stream = agent.stream("What is the capital of France?").await?;
    while let Some(event) = stream.next().await {
        if let AgentEvent::TextDelta(text) = event {
            print!("{text}");
        }
    }
    Ok(())
}
```

## Why Wuhu

**Streaming concurrent tools.** Tools start executing the moment the LLM describes them — not after it finishes. Two tool calls in one response run in parallel, automatically.

**Three-tier context compression.** Every long-running agent eventually runs out of context. Wuhu handles this gracefully: trim tool outputs, collapse old messages, summarize with the LLM — in that order, stopping as soon as pressure is relieved.

**Native HITL.** Suspend execution, ask the human, resume. It's a first-class API, not an afterthought.

**Zero storage dependencies.** Bring your own database. Wuhu defines a `Checkpoint` trait; you wire it to whatever backend you have.

## Crates

| Crate | Description |
|-------|-------------|
| `wuhu` | The facade. Start here. |
| `wuhu-core` | Traits and types. Zero runtime deps. |
| `wuhu-engine` | The execution loop. |
| `wuhu-compress` | Context compression pipeline. |
| `wuhu-providers` | LLM adapters (Anthropic, OpenAI). |

## License

MIT OR Apache-2.0
