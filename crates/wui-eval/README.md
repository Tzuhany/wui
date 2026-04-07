# wui-eval

Built on the `wui` runtime for testing purposes — not a runtime itself.

Testing infrastructure for Wui agents. Provides a scripted mock provider, an assertion harness, and a scenario runner — so you can write fast, deterministic agent tests without hitting a real LLM.

## Install

```toml
[dev-dependencies]
wui-eval = "0.1"
```

## Usage

```rust
use wui::{Agent, PermissionMode};
use wui_eval::{AgentHarness, MockProvider, ScenarioRunner, Scenario, Check};

// Script the provider's responses in order.
let provider = MockProvider::new(vec![
    MockProvider::text("The capital of France is Paris."),
]);

let agent = Agent::builder(provider)
    .permission(PermissionMode::Auto)
    .build();

// Run and assert.
let h = AgentHarness::run(&agent, "What is the capital of France?").await;
h.assert_text_contains("Paris");

// Or batch-test with named scenarios.
let runner = ScenarioRunner::new(agent);
let results = runner.run_all(&[
    Scenario {
        name:   "capital".into(),
        prompt: "Capital of France?".into(),
        checks: vec![Check::TextContains("Paris".into())],
    },
]).await;
```

`MockProvider` replays one `MockResponse` per LLM call in order — a panic if the queue is empty signals a test that made more calls than expected.

## Components

| Type | What it does |
|------|-------------|
| `MockProvider` | Deterministic `Provider` that replays scripted responses |
| `AgentHarness` | Runs an agent and exposes assertion helpers |
| `ScenarioRunner` | Runs named scenarios against a fixed agent |
| `Check` | Expected outcome: `TextContains`, `ToolCalled`, `StopReason`, ... |

Full docs: https://github.com/Tzuhany/wui
