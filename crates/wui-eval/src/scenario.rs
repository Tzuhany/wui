use std::collections::HashMap;
use std::sync::Arc;

use wui::Agent;
use wui_core::event::{AgentEvent, RunStopReason};

use crate::AgentHarness;

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
    /// Custom scoring function. Returns a value in `[0.0, 1.0]`.
    ///
    /// Scores are recorded in `ScenarioResult::scores` but do not
    /// affect the pass/fail outcome — use alongside other checks
    /// for quantitative evaluation.
    ///
    /// ```rust,ignore
    /// Check::Score {
    ///     name: "brevity".into(),
    ///     func: Arc::new(|h| {
    ///         let len = h.full_text().len();
    ///         if len < 200 { 1.0 } else { 200.0 / len as f64 }
    ///     }),
    /// }
    /// ```
    Score {
        name: String,
        func: Arc<dyn Fn(&AgentHarness) -> f64 + Send + Sync>,
    },
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
    /// Scores from `Check::Score` evaluations. Key = check name.
    pub scores: HashMap<String, f64>,
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
///     for (k, v) in &r.scores { println!("  {k}: {v:.2}"); }
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
        let mut scores = HashMap::new();

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
                Check::Score { name, func } => {
                    let score = func(&harness);
                    scores.insert(name.clone(), score);
                }
            }
        }

        ScenarioResult {
            name: scenario.name.clone(),
            passed: failures.is_empty(),
            failures,
            scores,
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

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use wui::{Agent, PermissionMode, RunStopReason};

    use crate::{Check, MockProvider, Scenario, ScenarioRunner};

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

    #[tokio::test]
    async fn scenario_runner_score() {
        let provider = MockProvider::new(vec![MockProvider::text("short")]);
        let agent = Agent::builder(provider)
            .permission(PermissionMode::Auto)
            .build();

        let scenarios = vec![Scenario {
            name: "brevity".to_string(),
            prompt: "Be brief.".to_string(),
            checks: vec![Check::Score {
                name: "length_score".into(),
                func: std::sync::Arc::new(|h| {
                    let len = h.full_text().len();
                    if len < 100 {
                        1.0
                    } else {
                        100.0 / len as f64
                    }
                }),
            }],
        }];

        let runner = ScenarioRunner::new(agent);
        let results = runner.run_all(&scenarios).await;
        assert!(results[0].passed, "score checks should not cause failure");
        assert!(
            results[0].scores.contains_key("length_score"),
            "score should be recorded"
        );
        assert!(
            results[0].scores["length_score"] > 0.0,
            "score should be positive"
        );
    }
}
