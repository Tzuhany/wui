use futures::StreamExt;
use wui::Agent;
use wui_core::event::{AgentEvent, RunStopReason, RunSummary};

// ── AgentHarness ──────────────────────────────────────────────────────────────

/// Runs an agent against a single prompt and collects all events.
///
/// Use the assertion methods to verify agent behaviour in tests.
///
/// ```rust,ignore
/// let h = AgentHarness::run(&agent, "Hello").await;
/// h.assert_text_contains("Hello")
///  .assert_stop_reason(RunStopReason::Completed);
/// ```
pub struct AgentHarness {
    pub(crate) events: Vec<AgentEvent>,
    pub(crate) summary: Option<RunSummary>,
}

impl AgentHarness {
    /// Run `agent` with `prompt` and collect all events.
    pub async fn run(agent: &Agent, prompt: impl Into<String>) -> Self {
        let mut stream = agent.stream(prompt.into());
        let mut events = Vec::new();
        let mut summary = None;

        while let Some(event) = stream.next().await {
            if let AgentEvent::Done(ref s) = event {
                summary = Some(s.clone());
            }
            events.push(event);
        }

        Self { events, summary }
    }

    // ── Assertions ────────────────────────────────────────────────────────────

    /// Assert that a tool named `name` was called at least once.
    ///
    /// Panics with a descriptive message if no `ToolStart` for this name is found.
    pub fn assert_tool_called(&self, name: &str) -> &Self {
        let found = self
            .events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolStart { name: n, .. } if n == name));
        assert!(
            found,
            "AgentHarness: expected tool '{name}' to be called, but it was not.\nAll events: {:?}",
            self.event_summary()
        );
        self
    }

    /// Assert that a tool named `name` was NOT called.
    ///
    /// Panics with a descriptive message if a `ToolStart` for this name is found.
    pub fn assert_tool_not_called(&self, name: &str) -> &Self {
        let found = self
            .events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolStart { name: n, .. } if n == name));
        assert!(
            !found,
            "AgentHarness: expected tool '{name}' NOT to be called, but it was.\nAll events: {:?}",
            self.event_summary()
        );
        self
    }

    /// Assert that the concatenated text output contains `s`.
    ///
    /// Panics with the full text if not found.
    pub fn assert_text_contains(&self, s: &str) -> &Self {
        let text = self.full_text();
        assert!(
            text.contains(s),
            "AgentHarness: expected text to contain {s:?}\nFull text: {text}"
        );
        self
    }

    /// Assert that the run ended with a specific `RunStopReason`.
    pub fn assert_stop_reason(&self, reason: RunStopReason) -> &Self {
        let actual = self.summary().stop_reason.clone();
        assert_eq!(
            actual, reason,
            "AgentHarness: expected stop reason {reason:?}, got {actual:?}"
        );
        self
    }

    /// Assert that the run completed in exactly `n` iterations.
    pub fn assert_iterations(&self, n: u32) -> &Self {
        let actual = self.summary().iterations;
        assert_eq!(
            actual, n,
            "AgentHarness: expected {n} iterations, got {actual}"
        );
        self
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Return all `ToolStart` events for calls to tool `name`.
    pub fn tool_calls(&self, name: &str) -> Vec<&AgentEvent> {
        self.events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolStart { name: n, .. } if n == name))
            .collect()
    }

    /// Return the concatenated text output from all `TextDelta` events.
    pub fn full_text(&self) -> String {
        self.events
            .iter()
            .filter_map(|e| {
                if let AgentEvent::TextDelta(t) = e {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return the `RunSummary` from the final `Done` event.
    ///
    /// Panics if the run did not complete (no `Done` event was received, e.g.
    /// because it ended with `Error`).
    pub fn summary(&self) -> &RunSummary {
        self.summary.as_ref().expect(
            "AgentHarness: run did not complete — no Done event received. \
             Did the agent return an Error?",
        )
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    fn event_summary(&self) -> Vec<String> {
        self.events
            .iter()
            .map(|e| match e {
                AgentEvent::TextDelta(t) => format!("TextDelta({t:?})"),
                AgentEvent::ToolStart { name, .. } => format!("ToolStart({name})"),
                AgentEvent::ToolDone { name, .. } => format!("ToolDone({name})"),
                AgentEvent::ToolError { name, .. } => format!("ToolError({name})"),
                AgentEvent::Done(s) => format!("Done({:?})", s.stop_reason),
                AgentEvent::Error(e) => format!("Error({e})"),
                other => format!("{other:?}"),
            })
            .collect()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use serde_json::{json, Value};
    use wui::{Agent, PermissionMode, RunStopReason, Tool, ToolCtx, ToolOutput};

    use crate::{AgentHarness, MockProvider};

    #[tokio::test]
    async fn mock_provider_tool_call_response() {
        struct EchoTool;
        #[async_trait]
        impl Tool for EchoTool {
            fn name(&self) -> &str {
                "echo"
            }
            fn description(&self) -> &str {
                "Echo."
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object", "properties": {} })
            }
            async fn call(&self, _input: Value, _ctx: &ToolCtx) -> ToolOutput {
                ToolOutput::success("echoed")
            }
        }

        let provider = MockProvider::new(vec![
            MockProvider::tool_call("echo", json!({})),
            MockProvider::text("done"),
        ]);
        let agent = Agent::builder(provider)
            .tool(EchoTool)
            .permission(PermissionMode::Auto)
            .build();

        let h = AgentHarness::run(&agent, "echo").await;
        h.assert_tool_called("echo")
            .assert_text_contains("done")
            .assert_stop_reason(RunStopReason::Completed);
    }
}
