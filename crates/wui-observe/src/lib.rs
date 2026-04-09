// ============================================================================
// wui-observe — observability for Wui agent runs.
//
// Wraps a RunStream and collects a structured Timeline while emitting proper
// tracing::Span objects for OpenTelemetry integration.
//
// # Timeline usage
//
//   use wui_observe::observe;
//   use futures::StreamExt;
//
//   let mut obs = observe(agent.stream("Write a haiku."));
//   while let Some(event) = obs.next().await { /* handle events */ }
//   let timeline = obs.into_timeline();
//   println!("{}", timeline.summary());
//
// # OpenTelemetry / Langfuse usage
//
//   Wire in `tracing-opentelemetry` + an OTLP exporter and every run will
//   produce a span tree in Jaeger, Honeycomb, Grafana, or Langfuse:
//
//   ```ignore
//   use opentelemetry_otlp::WithExportConfig;
//   use opentelemetry::KeyValue;
//   use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
//
//   let exporter = opentelemetry_otlp::new_exporter()
//       .http()
//       .with_endpoint("https://cloud.langfuse.com/api/public/otel/v1/traces")
//       .with_header("Authorization", "Basic <base64(pk:sk)>");
//
//   let tracer = opentelemetry_otlp::new_pipeline()
//       .tracing()
//       .with_exporter(exporter)
//       .install_batch(opentelemetry_sdk::runtime::Tokio)
//       .unwrap();
//
//   tracing_subscriber::registry()
//       .with(tracing_opentelemetry::layer().with_tracer(tracer))
//       .init();
//   ```
//
// # Span structure
//
//   wui.agent.run                   (root span per run)
//     gen_ai.operation.name = "chat"
//     gen_ai.system         = "wui"
//     gen_ai.usage.input_tokens  = N
//     gen_ai.usage.output_tokens = N
//     wui.iterations        = N
//
//     └── wui.tool.call             (child span per tool invocation)
//           gen_ai.tool.name         = "search"
//           wui.tool.id             = "t-abc123"
//           wui.tool.duration_ms    = 42
//           wui.tool.error          = "…"   (only on error)
//
// ============================================================================

mod stream;
mod timeline;

pub use stream::{observe, ObservedStream};
pub use timeline::{Timeline, TimelineEvent, TimelineEventKind};

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use futures::{Stream, StreamExt as _};
    use wui_core::event::AgentEvent;

    fn fake_stream(events: Vec<AgentEvent>) -> impl Stream<Item = AgentEvent> + Unpin {
        stream::iter(events)
    }

    fn fake_done() -> AgentEvent {
        use wui_core::event::{RunStopReason, RunSummary, TokenUsage};
        AgentEvent::Done(RunSummary {
            stop_reason: RunStopReason::Completed,
            iterations: 2,
            usage: TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                ..Default::default()
            },
            messages: vec![],
        })
    }

    #[tokio::test]
    async fn collects_timeline_events() {
        let events = vec![
            AgentEvent::ToolStart {
                id: "t1".into(),
                name: "search".into(),
                input: serde_json::json!({"q": "rust"}),
            },
            AgentEvent::ToolDone {
                id: "t1".into(),
                name: "search".into(),
                output: "results".into(),
                ms: 42,
                attempts: 1,
                structured: None,
            },
            fake_done(),
        ];

        let mut obs = observe(fake_stream(events));
        while obs.next().await.is_some() {}
        let tl = obs.into_timeline();

        // RunStart + ToolStart + ToolDone + RunDone = 4
        assert_eq!(tl.events().len(), 4);
        assert!(tl.run_summary().is_some());
        assert_eq!(tl.usage().input_tokens, 100);
    }

    #[tokio::test]
    async fn summary_includes_tool_names() {
        let events = vec![
            AgentEvent::ToolDone {
                id: "1".into(),
                name: "fetch".into(),
                output: "ok".into(),
                ms: 10,
                attempts: 1,
                structured: None,
            },
            fake_done(),
        ];

        let mut obs = observe(fake_stream(events));
        while obs.next().await.is_some() {}
        let tl = obs.into_timeline();

        assert!(tl.summary().contains("fetch"));
    }

    #[test]
    fn timeline_records_text_delta() {
        // TextDelta events are not recorded as timeline entries (they are high-frequency
        // streaming events). Verify the timeline still collects surrounding events correctly.
        let events = vec![
            AgentEvent::TextDelta("Hello ".into()),
            AgentEvent::TextDelta("world.".into()),
            fake_done(),
        ];

        // Drive synchronously by collecting into a Vec.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let tl = rt.block_on(async {
            let mut obs = observe(fake_stream(events));
            while obs.next().await.is_some() {}
            obs.into_timeline()
        });

        // RunStart + RunDone — TextDelta doesn't produce timeline entries.
        assert_eq!(tl.events().len(), 2);
        assert!(matches!(tl.events()[0].kind, TimelineEventKind::RunStart));
        assert!(matches!(
            tl.events()[1].kind,
            TimelineEventKind::RunDone { .. }
        ));
        assert_eq!(tl.usage().output_tokens, 50);
    }

    #[tokio::test]
    async fn tool_spans_cleaned_up_on_error() {
        let events = vec![
            AgentEvent::ToolStart {
                id: "t2".into(),
                name: "risky".into(),
                input: serde_json::json!({}),
            },
            AgentEvent::ToolError {
                id: "t2".into(),
                name: "risky".into(),
                error: "boom".into(),
                kind: wui_core::prelude::FailureKind::Execution,
                ms: 5,
            },
            fake_done(),
        ];

        let mut obs = observe(fake_stream(events));
        while obs.next().await.is_some() {}
        let tl = obs.into_timeline();

        // Should have: RunStart, ToolStart, ToolError, RunDone
        assert_eq!(tl.events().len(), 4);
    }
}
