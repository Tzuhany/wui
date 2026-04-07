# wui-observe

Observability layer wrapped around `wui` event streams.

Observability for Wui agent runs. Wraps any `RunStream` with structured timeline collection and OpenTelemetry-compatible `tracing` span emission — zero changes to the rest of your agent code.

## Install

```toml
[dependencies]
wui-observe = "0.1"
```

## Usage

```rust
use wui_observe::observe;
use futures::StreamExt;

let mut obs = observe(agent.stream("Write a haiku."));

while let Some(event) = obs.next().await {
    if let AgentEvent::TextDelta(text) = event {
        print!("{text}");
    }
}

let timeline = obs.into_timeline();
println!("{}", timeline.summary());
// elapsed: 1243ms | tokens: 312in 87out | tools: search(240ms)
```

## OpenTelemetry / Langfuse

Wire in `tracing-opentelemetry` and an OTLP exporter — every run produces a span tree automatically:

```
wui.agent.run
  gen_ai.usage.input_tokens  = 312
  gen_ai.usage.output_tokens = 87
  wui.iterations            = 2

  └── wui.tool.call
        gen_ai.tool.name      = "search"
        wui.tool.duration_ms = 240
```

## API

| Type | What it does |
|------|-------------|
| `observe(stream)` | Wraps a stream; returns `ObservedStream` |
| `ObservedStream` | Transparent pass-through + span emission |
| `Timeline` | Collected events, usage, and elapsed time |
| `TimelineEvent` | One structured event with `elapsed_ms` and `kind` |

Full docs: https://github.com/Tzuhany/wui
