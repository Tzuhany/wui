use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use futures::{Stream, StreamExt as _};
use tracing::Span;

use wui_core::event::{AgentEvent, RunSummary};
use wui_core::tool::ToolCallId;

use crate::timeline::{Timeline, TimelineEvent, TimelineEventKind};

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
