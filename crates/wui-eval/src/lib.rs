// ============================================================================
// wui-eval — evaluation and testing infrastructure for Wui agents.
//
// Three components:
//
//   MockProvider    — a deterministic Provider that replays scripted responses.
//   AgentHarness    — runs an agent and exposes assertion helpers.
//   ScenarioRunner  — loads and executes named test scenarios.
//
// Usage:
//
//   let provider = MockProvider::new(vec![
//       MockProvider::text("Paris"),
//   ]);
//   let agent = Agent::builder(provider)
//       .permission(PermissionMode::Auto)
//       .build();
//
//   let h = AgentHarness::run(&agent, "What is the capital of France?").await;
//   h.assert_text_contains("Paris");
// ============================================================================

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde_json::Value;
use wui::Agent;
use wui_core::event::{AgentEvent, RunStopReason, RunSummary, StopReason, StreamEvent, TokenUsage};
use wui_core::provider::{ChatRequest, Provider, ProviderError};
use wui_core::types::ToolCallId;

// ── MockProvider ──────────────────────────────────────────────────────────────

/// A scripted response for `MockProvider`.
pub enum MockResponse {
    /// The provider emits this text and stops.
    Text(String),
    /// The provider emits a single tool call and stops.
    ToolCall {
        name: String,
        /// A stable id for the tool call (propagated as `tool_use_id`).
        id: String,
        input: Value,
    },
    /// The provider emits a retryable error.
    Error { message: String, retryable: bool },
}

/// A deterministic `Provider` that replays scripted responses in order.
///
/// Each call to `stream()` pops the next `MockResponse` from the queue and
/// returns a stream that yields the corresponding `StreamEvent`s. Panics when
/// the queue is exhausted — this indicates a test bug (the agent made more LLM
/// calls than expected).
///
/// ```rust,ignore
/// let provider = MockProvider::new(vec![
///     MockProvider::text("Hello!"),
///     MockProvider::tool_call("bash", json!({"command":"ls"})),
///     MockProvider::text("Done."),
/// ]);
/// ```
pub struct MockProvider {
    responses: Arc<Mutex<VecDeque<MockResponse>>>,
}

impl MockProvider {
    /// Create a provider that will serve `responses` in order.
    pub fn new(responses: Vec<MockResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(VecDeque::from(responses))),
        }
    }

    /// Construct a `MockResponse::Text` conveniently.
    pub fn text(s: impl Into<String>) -> MockResponse {
        MockResponse::Text(s.into())
    }

    /// Construct a `MockResponse::ToolCall` conveniently.
    ///
    /// A random-ish id is generated from the name. For deterministic ids,
    /// construct `MockResponse::ToolCall` directly.
    pub fn tool_call(name: impl Into<String>, input: Value) -> MockResponse {
        let name = name.into();
        let id = format!("mock_{}_{}", name, uuid_short());
        MockResponse::ToolCall { name, id, input }
    }

    /// Construct a `MockResponse::Error` conveniently.
    pub fn error(message: impl Into<String>, retryable: bool) -> MockResponse {
        MockResponse::Error {
            message: message.into(),
            retryable,
        }
    }
}

fn uuid_short() -> String {
    // Simple deterministic suffix using a counter-like approach.
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed).to_string()
}

#[async_trait]
impl Provider for MockProvider {
    async fn stream(
        &self,
        _req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        let response = {
            let mut queue = self.responses.lock().expect("MockProvider lock poisoned");
            queue.pop_front().expect(
                "MockProvider: no more scripted responses — the agent made more \
                 LLM calls than expected. Add more MockResponse entries to the queue.",
            )
        };

        let events: Vec<Result<StreamEvent, ProviderError>> = match response {
            MockResponse::Text(text) => {
                let chars = text.len() as u32;
                vec![
                    Ok(StreamEvent::TextDelta { text }),
                    Ok(StreamEvent::MessageEnd {
                        usage: TokenUsage {
                            input_tokens: 10,
                            output_tokens: chars,
                            ..Default::default()
                        },
                        stop_reason: StopReason::EndTurn,
                    }),
                ]
            }

            MockResponse::ToolCall { name, id, input } => {
                let input_json = serde_json::to_string(&input).unwrap_or_default();
                let id = ToolCallId::from(id);
                vec![
                    Ok(StreamEvent::ToolUseStart {
                        id: id.clone(),
                        name,
                    }),
                    Ok(StreamEvent::ToolInputDelta {
                        id: id.clone(),
                        chunk: input_json,
                    }),
                    Ok(StreamEvent::ToolUseEnd { id }),
                    Ok(StreamEvent::MessageEnd {
                        usage: TokenUsage {
                            input_tokens: 10,
                            output_tokens: 10,
                            ..Default::default()
                        },
                        stop_reason: StopReason::ToolUse,
                    }),
                ]
            }

            MockResponse::Error { message, retryable } => {
                vec![Ok(StreamEvent::Error { message, retryable })]
            }
        };

        let stream = futures::stream::iter(events);
        Ok(Box::pin(stream))
    }
}

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
    events: Vec<AgentEvent>,
    summary: Option<RunSummary>,
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

// ── ScenarioRunner ────────────────────────────────────────────────────────────

/// A check applied to an agent's output in a `Scenario`.
pub enum Check {
    /// The concatenated text output must contain this string.
    TextContains(String),
    /// The named tool must have been called at least once.
    ToolCalled(String),
    /// The named tool must NOT have been called.
    ToolNotCalled(String),
    /// The run must have ended with this stop reason.
    StopReason(RunStopReason),
}

/// A named test scenario — a prompt plus a list of expected outcomes.
pub struct Scenario {
    pub name: String,
    pub prompt: String,
    pub checks: Vec<Check>,
}

/// The result of running one scenario.
pub struct ScenarioResult {
    pub name: String,
    pub passed: bool,
    /// Human-readable description of each failed check.
    pub failures: Vec<String>,
}

/// Runs one or many `Scenario`s against a fixed `Agent`.
///
/// The same `Agent` instance is reused across scenarios (each scenario starts
/// a fresh run, not a fresh session).
///
/// ```rust,ignore
/// let runner = ScenarioRunner::new(agent);
/// let results = runner.run_all(&scenarios).await;
/// for r in &results {
///     println!("{}: {}", r.name, if r.passed { "PASS" } else { "FAIL" });
///     for f in &r.failures { println!("  - {f}"); }
/// }
/// ```
pub struct ScenarioRunner {
    agent: Agent,
}

impl ScenarioRunner {
    /// Create a runner backed by `agent`.
    pub fn new(agent: Agent) -> Self {
        Self { agent }
    }

    /// Run a single scenario and return its result.
    pub async fn run_scenario(&self, scenario: &Scenario) -> ScenarioResult {
        let harness = AgentHarness::run(&self.agent, scenario.prompt.clone()).await;
        let mut failures = Vec::new();

        for check in &scenario.checks {
            match check {
                Check::TextContains(s) => {
                    let text = harness.full_text();
                    if !text.contains(s.as_str()) {
                        failures.push(format!(
                            "TextContains: expected text to contain {s:?}, got: {text:?}"
                        ));
                    }
                }
                Check::ToolCalled(name) | Check::ToolNotCalled(name) => {
                    let called = harness
                        .events
                        .iter()
                        .any(|e| matches!(e, AgentEvent::ToolStart { name: n, .. } if n == name));
                    let want_called = matches!(check, Check::ToolCalled(_));
                    if called != want_called {
                        let msg = if want_called {
                            format!("ToolCalled: tool '{name}' was not called")
                        } else {
                            format!(
                                "ToolNotCalled: tool '{name}' was called but should not have been"
                            )
                        };
                        failures.push(msg);
                    }
                }
                Check::StopReason(expected) => {
                    if let Some(summary) = &harness.summary {
                        if &summary.stop_reason != expected {
                            failures.push(format!(
                                "StopReason: expected {expected:?}, got {:?}",
                                summary.stop_reason
                            ));
                        }
                    } else {
                        failures
                            .push("StopReason: run did not complete (no Done event)".to_string());
                    }
                }
            }
        }

        ScenarioResult {
            name: scenario.name.clone(),
            passed: failures.is_empty(),
            failures,
        }
    }

    /// Run all scenarios and return their results.
    pub async fn run_all(&self, scenarios: &[Scenario]) -> Vec<ScenarioResult> {
        let mut results = Vec::new();
        for scenario in scenarios {
            results.push(self.run_scenario(scenario).await);
        }
        results
    }
}

// ── Smoke tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use serde_json::json;

    use wui::{Agent, PermissionMode, RunStopReason};

    use super::{AgentHarness, Check, MockProvider, Scenario, ScenarioRunner};

    #[tokio::test]
    async fn mock_provider_text_response() {
        let provider = MockProvider::new(vec![MockProvider::text("hello world")]);
        let agent = Agent::builder(provider)
            .permission(PermissionMode::Auto)
            .build();

        let h = AgentHarness::run(&agent, "say hello").await;
        h.assert_text_contains("hello world")
            .assert_stop_reason(RunStopReason::Completed);
    }

    #[tokio::test]
    async fn mock_provider_tool_call_response() {
        use async_trait::async_trait;
        use serde_json::Value;
        use wui::{Tool, ToolCtx, ToolOutput};

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

    #[tokio::test]
    async fn scenario_runner_pass() {
        let provider = MockProvider::new(vec![MockProvider::text("Paris")]);
        let agent = Agent::builder(provider)
            .permission(PermissionMode::Auto)
            .build();

        let scenarios = vec![Scenario {
            name: "capital-of-france".to_string(),
            prompt: "What is the capital of France?".to_string(),
            checks: vec![
                Check::TextContains("Paris".to_string()),
                Check::StopReason(RunStopReason::Completed),
            ],
        }];

        let runner = ScenarioRunner::new(agent);
        let results = runner.run_all(&scenarios).await;
        assert_eq!(results.len(), 1);
        assert!(
            results[0].passed,
            "scenario failed: {:?}",
            results[0].failures
        );
    }
}
