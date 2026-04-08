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

use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use futures::{Stream, StreamExt as _};
use serde::{Deserialize, Serialize};
use tracing::Span;

use wui_core::event::{AgentEvent, RunSummary, TokenUsage};
use wui_core::types::ToolCallId;

// ── TimelineEvent ─────────────────────────────────────────────────────────────

/// A single event captured in a run timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    /// Milliseconds since the run started.
    pub elapsed_ms: u64,
    /// What happened.
    pub kind: TimelineEventKind,
}

/// The kind of a `TimelineEvent`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TimelineEventKind {
    /// The run started.
    RunStart,

    /// A tool call began.
    ToolStart {
        id: ToolCallId,
        name: String,
        input: serde_json::Value,
    },

    /// A tool call completed successfully.
    ToolDone {
        id: ToolCallId,
        name: String,
        output: String,
        ms: u64,
    },

    /// A tool call failed.
    ToolError {
        id: ToolCallId,
        name: String,
        error: String,
        ms: u64,
    },

    /// The context was compressed.
    Compressed {
        method: String,
        chars_removed: Option<usize>,
    },

    /// The provider returned a retryable error.
    Retrying {
        attempt: u32,
        delay_ms: u64,
        reason: String,
    },

    /// The run completed.
    RunDone {
        iterations: u32,
        input_tokens: u32,
        output_tokens: u32,
    },
}

// ── Timeline ──────────────────────────────────────────────────────────────────

/// The complete history of a run, collected by [`ObservedStream`].
#[derive(Debug, Clone, Default)]
pub struct Timeline {
    events: Vec<TimelineEvent>,
    summary: Option<RunSummary>,
    elapsed: Duration,
}

impl Timeline {
    /// All events in chronological order.
    pub fn events(&self) -> &[TimelineEvent] {
        &self.events
    }

    /// The `RunSummary` if the run completed normally (i.e. `AgentEvent::Done` was received).
    pub fn run_summary(&self) -> Option<&RunSummary> {
        self.summary.as_ref()
    }

    /// Total elapsed time from run start to first `Done` event.
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }

    /// Token usage across the run, or zeroes if the run did not complete.
    pub fn usage(&self) -> TokenUsage {
        self.summary
            .as_ref()
            .map(|s| s.usage.clone())
            .unwrap_or_default()
    }

    /// A human-readable one-line summary.
    pub fn summary(&self) -> String {
        let tools: Vec<_> = self
            .events
            .iter()
            .filter_map(|e| {
                if let TimelineEventKind::ToolDone { name, ms, .. } = &e.kind {
                    Some(format!("{name}({ms}ms)"))
                } else {
                    None
                }
            })
            .collect();

        let usage = self.usage();
        let tool_info = if tools.is_empty() {
            String::new()
        } else {
            format!(" | tools: {}", tools.join(", "))
        };

        format!(
            "elapsed: {}ms | tokens: {}in {}out{tool_info}",
            self.elapsed.as_millis(),
            usage.input_tokens,
            usage.output_tokens,
        )
    }

    /// Serialize the timeline to a JSON value.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "elapsed_ms": self.elapsed.as_millis(),
            "events":     self.events,
            "usage": {
                "input_tokens":       self.usage().input_tokens,
                "output_tokens":      self.usage().output_tokens,
                "cache_read_tokens":  self.usage().cache_read_tokens,
                "cache_write_tokens": self.usage().cache_write_tokens,
            },
        })
    }
}

// ── ObservedStream ────────────────────────────────────────────────────────────

/// A stream wrapper that records a [`Timeline`] and emits OTel-compatible
/// tracing spans while forwarding `AgentEvent`s.
///
/// Implements `Stream<Item = AgentEvent>` — use it anywhere a `RunStream` is used.
/// Call [`ObservedStream::into_timeline`] after the stream ends.
///
/// ```rust,ignore
/// let mut obs = observe(agent.stream("..."));
/// while let Some(event) = obs.next().await { /* handle events */ }
/// println!("{}", obs.into_timeline().summary());
/// ```
pub struct ObservedStream<S> {
    inner: S,
    timeline: Vec<TimelineEvent>,
    start: Instant,
    summary: Option<RunSummary>,
    /// Root span for the whole run — kept alive until `into_timeline()`.
    run_span: Span,
    /// In-flight tool spans, keyed by tool call id.
    tool_spans: HashMap<ToolCallId, Span>,
}

impl<S> ObservedStream<S> {
    /// Consume the stream and return the collected timeline.
    ///
    /// This drops the root run span, which closes it in the OTel exporter.
    /// Call this after the stream has ended (i.e. after `None` is returned or
    /// after you have received `AgentEvent::Done`).
    pub fn into_timeline(self) -> Timeline {
        let elapsed = self.start.elapsed();
        // run_span and any remaining tool_spans are dropped here, closing them.
        Timeline {
            events: self.timeline,
            summary: self.summary,
            elapsed,
        }
    }
}

impl<S> ObservedStream<S> {
    /// Close and record the tool span when a tool finishes (Done or Error).
    fn close_tool_span(&mut self, id: &ToolCallId, ms: u64, error: Option<&str>) {
        if let Some(span) = self.tool_spans.remove(id) {
            span.record("wui.tool.duration_ms", ms);
            if let Some(err) = error {
                span.record("wui.tool.error", err);
            }
        }
    }
}

impl<S: Stream<Item = AgentEvent> + Unpin> Stream for ObservedStream<S> {
    type Item = AgentEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<AgentEvent>> {
        let event = match self.inner.poll_next_unpin(cx) {
            Poll::Ready(Some(e)) => e,
            Poll::Ready(None) => return Poll::Ready(None),
            Poll::Pending => return Poll::Pending,
        };

        let elapsed_ms = self.start.elapsed().as_millis() as u64;

        let kind = match &event {
            AgentEvent::ToolStart { id, name, input } => {
                let span = tracing::info_span!(
                    parent: &self.run_span,
                    "wui.tool.call",
                    "gen_ai.tool.name"  = %name,
                    "wui.tool.id"      = %id,
                );
                self.tool_spans.insert(id.clone(), span);
                Some(TimelineEventKind::ToolStart {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                })
            }

            AgentEvent::ToolDone {
                id,
                name,
                output,
                ms,
                ..
            } => {
                self.close_tool_span(id, *ms, None);
                Some(TimelineEventKind::ToolDone {
                    id: id.clone(),
                    name: name.clone(),
                    output: output.clone(),
                    ms: *ms,
                })
            }

            AgentEvent::ToolError {
                id,
                name,
                error,
                ms,
                ..
            } => {
                self.close_tool_span(id, *ms, Some(error));
                Some(TimelineEventKind::ToolError {
                    id: id.clone(),
                    name: name.clone(),
                    error: error.clone(),
                    ms: *ms,
                })
            }

            AgentEvent::Compressed { method, .. } => Some(TimelineEventKind::Compressed {
                method: format!("{method:?}"),
                chars_removed: None,
            }),

            AgentEvent::Retrying {
                attempt,
                delay_ms,
                reason,
            } => {
                tracing::warn!(
                    parent: &self.run_span, attempt, delay_ms,
                    reason = %reason, "retrying",
                );
                Some(TimelineEventKind::Retrying {
                    attempt: *attempt,
                    delay_ms: *delay_ms,
                    reason: reason.clone(),
                })
            }

            AgentEvent::Done(summary) => {
                self.run_span
                    .record("gen_ai.usage.input_tokens", summary.usage.input_tokens);
                self.run_span
                    .record("gen_ai.usage.output_tokens", summary.usage.output_tokens);
                self.run_span.record("wui.iterations", summary.iterations);
                self.summary = Some(summary.clone());
                Some(TimelineEventKind::RunDone {
                    iterations: summary.iterations,
                    input_tokens: summary.usage.input_tokens,
                    output_tokens: summary.usage.output_tokens,
                })
            }

            _ => None,
        };

        if let Some(k) = kind {
            self.timeline.push(TimelineEvent {
                elapsed_ms,
                kind: k,
            });
        }

        Poll::Ready(Some(event))
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Wrap a stream of `AgentEvent`s with timeline collection and OTel span emission.
///
/// The returned stream is a transparent pass-through — it forwards every event
/// unchanged while recording a [`Timeline`] and emitting tracing spans compatible
/// with `tracing-opentelemetry`.
///
/// ```rust,ignore
/// let mut obs = wui_observe::observe(agent.stream("What is 2+2?"));
/// while let Some(event) = obs.next().await {
///     if let AgentEvent::TextDelta(text) = event { print!("{text}"); }
/// }
/// let timeline = obs.into_timeline();
/// println!("{}", timeline.summary());
/// ```
pub fn observe<S>(stream: S) -> ObservedStream<S>
where
    S: Stream<Item = AgentEvent> + Unpin,
{
    // Root span — fields recorded dynamically once Done fires.
    let run_span = tracing::info_span!(
        "wui.agent.run",
        "gen_ai.operation.name" = "chat",
        "gen_ai.system" = "wui",
        // These are empty at start; filled in when Done fires.
        "gen_ai.usage.input_tokens" = tracing::field::Empty,
        "gen_ai.usage.output_tokens" = tracing::field::Empty,
        "wui.iterations" = tracing::field::Empty,
    );

    let mut obs = ObservedStream {
        inner: stream,
        timeline: Vec::new(),
        start: Instant::now(),
        summary: None,
        run_span,
        tool_spans: HashMap::new(),
    };
    obs.timeline.push(TimelineEvent {
        elapsed_ms: 0,
        kind: TimelineEventKind::RunStart,
    });
    obs
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

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
