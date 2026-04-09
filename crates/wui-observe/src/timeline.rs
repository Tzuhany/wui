use std::time::Duration;

use serde::{Deserialize, Serialize};

use wui_core::event::{RunSummary, TokenUsage};
use wui_core::tool::ToolCallId;

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

/// The complete history of a run, collected by [`super::ObservedStream`].
#[derive(Debug, Clone, Default)]
pub struct Timeline {
    pub(crate) events: Vec<TimelineEvent>,
    pub(crate) summary: Option<RunSummary>,
    pub(crate) elapsed: Duration,
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
