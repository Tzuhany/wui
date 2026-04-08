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

mod harness;
mod mock_provider;
mod scenario;

pub use harness::AgentHarness;
pub use mock_provider::{MockProvider, MockResponse};
pub use scenario::{Check, Scenario, ScenarioResult, ScenarioRunner};
