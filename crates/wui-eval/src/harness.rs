// ============================================================================
// AgentHarness — the complete test surface for agent runs.
//
// Collects every event from an agent run and exposes two kinds of API:
//
//   Queries   — extract data: full_text(), tool_outputs(), thinking(), etc.
//   Assertions — verify behaviour: assert_tool_called(), assert_no_error(), etc.
//
// All assertions return `&Self` for chaining. All panics include diagnostic
// context (event timeline, full text, etc.) so failures are self-explanatory.
// ============================================================================

use std::time::{Duration, Instant};

use futures::StreamExt;
use wui::Agent;
use wui_core::event::{AgentError, AgentEvent, RunStopReason, RunSummary, TokenUsage};
use wui_core::tool::FailureKind;

// ── AgentHarness ────────────────────────────────────────────────────────────

/// Runs an agent against a single prompt and collects all events.
///
/// Use the query methods to extract data and the assertion methods to
/// verify agent behaviour in tests. All assertions are chainable.
///
/// ```rust,ignore
/// let h = AgentHarness::run(&agent, "Hello").await;
/// h.assert_text_contains("Hello")
///  .assert_tool_called("search")
///  .assert_tool_called_times("search", 1)
///  .assert_no_error()
///  .assert_stop_reason(RunStopReason::Completed);
/// ```
pub struct AgentHarness {
    /// All events emitted during the run, in order.
    pub events: Vec<AgentEvent>,
    /// The final `RunSummary`, if the run completed with `Done`.
    pub summary: Option<RunSummary>,
    /// The terminal error, if the run ended with `Error`.
    pub error: Option<AgentError>,
    /// Wall-clock duration of the run.
    pub elapsed: Duration,
}

impl AgentHarness {
    /// Run `agent` with `prompt` and collect all events.
    pub async fn run(agent: &Agent, prompt: impl Into<String>) -> Self {
        let start = Instant::now();
        let mut stream = agent.stream(prompt.into());
        let mut events = Vec::new();
        let mut summary = None;
        let mut error = None;

        while let Some(event) = stream.next().await {
            match &event {
                AgentEvent::Done(s) => summary = Some(s.clone()),
                AgentEvent::Error(e) => error = Some(e.clone()),
                _ => {}
            }
            events.push(event);
        }

        Self {
            events,
            summary,
            error,
            elapsed: start.elapsed(),
        }
    }

    // ── Queries ──────────────────────────────────────────────────────────────

    /// Concatenated text output from all `TextDelta` events.
    pub fn full_text(&self) -> String {
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::TextDelta(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Concatenated thinking output from all `ThinkingDelta` events.
    pub fn thinking(&self) -> String {
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ThinkingDelta(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    /// All `ToolStart` events for tool `name`.
    pub fn tool_calls(&self, name: &str) -> Vec<&AgentEvent> {
        self.events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolStart { name: n, .. } if n == name))
            .collect()
    }

    /// The input values passed to each call of tool `name`.
    pub fn tool_inputs(&self, name: &str) -> Vec<&serde_json::Value> {
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolStart { name: n, input, .. } if n == name => Some(input),
                _ => None,
            })
            .collect()
    }

    /// The output strings from each successful call to tool `name`.
    pub fn tool_outputs(&self, name: &str) -> Vec<&str> {
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolDone {
                    name: n, output, ..
                } if n == name => Some(output.as_str()),
                _ => None,
            })
            .collect()
    }

    /// All `ToolError` events across all tools.
    pub fn tool_errors(&self) -> Vec<(&str, &str, &FailureKind)> {
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolError {
                    name, error, kind, ..
                } => Some((name.as_str(), error.as_str(), kind)),
                _ => None,
            })
            .collect()
    }

    /// How many times tool `name` was called (started).
    pub fn tool_call_count(&self, name: &str) -> usize {
        self.tool_calls(name).len()
    }

    /// Total tokens consumed (input + output), from the `RunSummary`.
    pub fn total_tokens(&self) -> u32 {
        self.summary.as_ref().map(|s| s.usage.total()).unwrap_or(0)
    }

    /// Token usage breakdown, from the `RunSummary`.
    pub fn usage(&self) -> &TokenUsage {
        static DEFAULT: TokenUsage = TokenUsage {
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
        };
        self.summary.as_ref().map(|s| &s.usage).unwrap_or(&DEFAULT)
    }

    /// Number of iterations the run executed.
    pub fn iterations(&self) -> u32 {
        self.summary.as_ref().map(|s| s.iterations).unwrap_or(0)
    }

    /// The `RunSummary` from the final `Done` event.
    ///
    /// Panics if the run did not complete.
    pub fn summary(&self) -> &RunSummary {
        self.summary.as_ref().expect(
            "AgentHarness: run did not complete — no Done event received. \
             Did the agent return an Error?",
        )
    }

    // ── Assertions ───────────────────────────────────────────────────────────

    /// Assert that a tool named `name` was called at least once.
    pub fn assert_tool_called(&self, name: &str) -> &Self {
        assert!(
            self.tool_call_count(name) > 0,
            "expected tool '{name}' to be called, but it was not.\n{}",
            self.diagnostic()
        );
        self
    }

    /// Assert that a tool named `name` was NOT called.
    pub fn assert_tool_not_called(&self, name: &str) -> &Self {
        assert!(
            self.tool_call_count(name) == 0,
            "expected tool '{name}' NOT to be called, but it was.\n{}",
            self.diagnostic()
        );
        self
    }

    /// Assert that tool `name` was called exactly `n` times.
    pub fn assert_tool_called_times(&self, name: &str, n: usize) -> &Self {
        let actual = self.tool_call_count(name);
        assert_eq!(
            actual,
            n,
            "expected tool '{name}' to be called {n} times, got {actual}.\n{}",
            self.diagnostic()
        );
        self
    }

    /// Assert that tool `name` completed successfully (at least once).
    pub fn assert_tool_succeeded(&self, name: &str) -> &Self {
        let found = self
            .events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolDone { name: n, .. } if n == name));
        assert!(
            found,
            "expected tool '{name}' to succeed, but no ToolDone event found.\n{}",
            self.diagnostic()
        );
        self
    }

    /// Assert that the concatenated text output contains `s`.
    pub fn assert_text_contains(&self, s: &str) -> &Self {
        let text = self.full_text();
        assert!(
            text.contains(s),
            "expected text to contain {s:?}\nFull text: {text}"
        );
        self
    }

    /// Assert that the thinking output contains `s`.
    pub fn assert_thinking_contains(&self, s: &str) -> &Self {
        let thinking = self.thinking();
        assert!(
            thinking.contains(s),
            "expected thinking to contain {s:?}\nFull thinking: {thinking}"
        );
        self
    }

    /// Assert that the run ended with a specific `RunStopReason`.
    pub fn assert_stop_reason(&self, reason: RunStopReason) -> &Self {
        let actual = self.summary().stop_reason.clone();
        assert_eq!(
            actual, reason,
            "expected stop reason {reason:?}, got {actual:?}"
        );
        self
    }

    /// Assert that the run completed in exactly `n` iterations.
    pub fn assert_iterations(&self, n: u32) -> &Self {
        let actual = self.summary().iterations;
        assert_eq!(actual, n, "expected {n} iterations, got {actual}");
        self
    }

    /// Assert that the run completed in fewer than `n` iterations.
    pub fn assert_iterations_under(&self, n: u32) -> &Self {
        let actual = self.summary().iterations;
        assert!(
            actual < n,
            "expected fewer than {n} iterations, got {actual}"
        );
        self
    }

    /// Assert that total token usage is below `n`.
    pub fn assert_tokens_under(&self, n: u32) -> &Self {
        let actual = self.total_tokens();
        assert!(actual < n, "expected fewer than {n} tokens, used {actual}");
        self
    }

    /// Assert that the run did NOT end with an error.
    pub fn assert_no_error(&self) -> &Self {
        if let Some(e) = &self.error {
            panic!("expected no error, but got: {e}\n{}", self.diagnostic());
        }
        self
    }

    /// Assert that the run ended with an error.
    pub fn assert_error(&self) -> &Self {
        assert!(
            self.error.is_some(),
            "expected an error, but run completed successfully.\n{}",
            self.diagnostic()
        );
        self
    }

    /// Assert that the run completed within `duration`.
    pub fn assert_elapsed_under(&self, duration: Duration) -> &Self {
        assert!(
            self.elapsed < duration,
            "expected run to complete within {duration:?}, took {:?}",
            self.elapsed
        );
        self
    }

    /// Assert that tool `a` was called before tool `b`.
    ///
    /// Compares the position of the first `ToolStart` event for each tool.
    pub fn assert_tool_called_before(&self, a: &str, b: &str) -> &Self {
        let pos_a = self
            .events
            .iter()
            .position(|e| matches!(e, AgentEvent::ToolStart { name, .. } if name == a));
        let pos_b = self
            .events
            .iter()
            .position(|e| matches!(e, AgentEvent::ToolStart { name, .. } if name == b));

        match (pos_a, pos_b) {
            (Some(pa), Some(pb)) => {
                assert!(
                    pa < pb,
                    "expected tool '{a}' to be called before '{b}', \
                     but '{a}' was at position {pa} and '{b}' at {pb}.\n{}",
                    self.diagnostic()
                );
            }
            (None, _) => panic!("tool '{a}' was never called.\n{}", self.diagnostic()),
            (_, None) => panic!("tool '{b}' was never called.\n{}", self.diagnostic()),
        }
        self
    }

    /// Assert that no tool errors occurred during the run.
    pub fn assert_no_tool_errors(&self) -> &Self {
        let errors = self.tool_errors();
        assert!(
            errors.is_empty(),
            "expected no tool errors, but got {} error(s): {:?}",
            errors.len(),
            errors
                .iter()
                .map(|(name, err, _)| format!("{name}: {err}"))
                .collect::<Vec<_>>()
        );
        self
    }

    // ── Diagnostic ───────────────────────────────────────────────────────────

    /// Compact event timeline for error messages.
    fn diagnostic(&self) -> String {
        let timeline: Vec<String> = self
            .events
            .iter()
            .map(|e| match e {
                AgentEvent::TextDelta(t) => {
                    let preview = if t.len() > 40 {
                        format!("{}…", &t[..40])
                    } else {
                        t.clone()
                    };
                    format!("  TextDelta({preview:?})")
                }
                AgentEvent::ThinkingDelta(_) => "  ThinkingDelta(…)".to_string(),
                AgentEvent::ToolStart { name, .. } => format!("  ToolStart({name})"),
                AgentEvent::ToolDone {
                    name, ms, output, ..
                } => {
                    let preview = if output.len() > 40 {
                        format!("{}…", &output[..40])
                    } else {
                        output.clone()
                    };
                    format!("  ToolDone({name}, {ms}ms, {preview:?})")
                }
                AgentEvent::ToolError {
                    name, error, kind, ..
                } => format!("  ToolError({name}, {kind:?}, {error:?})"),
                AgentEvent::Control(_) => "  Control(…)".to_string(),
                AgentEvent::Compressed { method, freed, .. } => {
                    format!("  Compressed({method:?}, freed={freed})")
                }
                AgentEvent::Retrying {
                    attempt, reason, ..
                } => format!("  Retrying(#{attempt}, {reason:?})"),
                AgentEvent::Done(s) => format!("  Done({:?}, {}iter)", s.stop_reason, s.iterations),
                AgentEvent::Error(e) => format!("  Error({e})"),
                other => format!("  {other:?}"),
            })
            .collect();

        format!(
            "Event timeline ({} events, {:?}):\n{}",
            self.events.len(),
            self.elapsed,
            timeline.join("\n")
        )
    }
}

// ── SessionHarness ──────────────────────────────────────────────────────────

/// Multi-turn test harness that drives a `Session`.
///
/// Each `send()` returns an `AgentHarness` for that turn, while the session
/// maintains history across turns.
///
/// ```rust,ignore
/// let sh = SessionHarness::new(&agent, "test-session").await;
/// let t1 = sh.send("Hello").await;
/// t1.assert_text_contains("Hi");
/// let t2 = sh.send("What did I say?").await;
/// t2.assert_text_contains("Hello");
/// ```
pub struct SessionHarness {
    session: wui::Session,
}

impl SessionHarness {
    /// Create a new session harness.
    pub async fn new(agent: &Agent, session_id: impl Into<String>) -> Self {
        Self {
            session: agent.session(session_id).await,
        }
    }

    /// Send a message and collect all events for this turn.
    pub async fn send(&self, input: impl Into<wui_core::message::Message>) -> AgentHarness {
        let start = Instant::now();
        let mut stream = self.session.send(input).await;
        let mut events = Vec::new();
        let mut summary = None;
        let mut error = None;

        while let Some(event) = stream.next().await {
            match &event {
                AgentEvent::Done(s) => summary = Some(s.clone()),
                AgentEvent::Error(e) => error = Some(e.clone()),
                _ => {}
            }
            events.push(event);
        }

        AgentHarness {
            events,
            summary,
            error,
            elapsed: start.elapsed(),
        }
    }

    /// The current message history.
    pub fn messages(&self) -> Vec<wui_core::message::Message> {
        self.session.messages()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use serde_json::{json, Value};
    use wui::{Agent, PermissionMode, RunStopReason, Tool, ToolCtx, ToolOutput};

    use crate::{AgentHarness, MockProvider, SessionHarness};

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

    #[tokio::test]
    async fn harness_queries_and_assertions() {
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
            .assert_tool_not_called("nonexistent")
            .assert_tool_called_times("echo", 1)
            .assert_tool_succeeded("echo")
            .assert_text_contains("done")
            .assert_no_error()
            .assert_no_tool_errors()
            .assert_stop_reason(RunStopReason::Completed);

        assert_eq!(h.tool_outputs("echo"), vec!["echoed"]);
        assert_eq!(h.tool_call_count("echo"), 1);
        assert!(h.elapsed.as_millis() < 5000);
        assert!(h.iterations() >= 1);
    }

    #[tokio::test]
    async fn harness_captures_error() {
        let provider = MockProvider::new(vec![MockProvider::error("test error", false)]);
        let agent = Agent::builder(provider)
            .permission(PermissionMode::Auto)
            .build();

        let h = AgentHarness::run(&agent, "fail").await;
        h.assert_error();
        assert!(h.error.is_some());
        assert!(h.summary.is_none());
    }

    #[tokio::test]
    async fn session_harness_preserves_history() {
        let provider = MockProvider::new(vec![
            MockProvider::text("turn one"),
            MockProvider::text("turn two"),
        ]);
        let agent = Agent::builder(provider)
            .permission(PermissionMode::Auto)
            .build();

        let sh = SessionHarness::new(&agent, "test").await;
        let t1 = sh.send("first").await;
        t1.assert_text_contains("turn one");

        let t2 = sh.send("second").await;
        t2.assert_text_contains("turn two");

        // Session should have history from both turns.
        assert!(sh.messages().len() >= 3);
    }
}
