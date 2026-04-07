// ============================================================================
// Integration tests for the wui agent framework.
//
// Uses MockProvider from wui-eval to run the full agent loop without making
// real network calls. Each test exercises a specific runtime behaviour.
// ============================================================================

use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

use async_trait::async_trait;
use serde_json::{json, Value};

use futures::StreamExt as _;

use wui::{
    Agent, AgentEvent, ExecutorHints, InMemoryCheckpointStore, PermissionMode, RunStopReason, Tool,
    ToolCtx, ToolMeta, ToolOutput,
};
use wui_eval::{AgentHarness, MockProvider};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// A minimal tool that always returns a fixed output.
struct ConstTool {
    name: &'static str,
    output: &'static str,
}

#[async_trait]
impl Tool for ConstTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "A simple test tool."
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    fn meta(&self, _input: &Value) -> ToolMeta {
        ToolMeta {
            concurrent: true,
            ..ToolMeta::default()
        }
    }
    async fn call(&self, _input: Value, _ctx: &ToolCtx) -> ToolOutput {
        ToolOutput::success(self.output)
    }
}

// ── Test 1: agent_produces_text_response ─────────────────────────────────────

#[tokio::test]
async fn agent_produces_text_response() {
    let provider = MockProvider::new(vec![MockProvider::text("Hello from the agent!")]);

    let agent = Agent::builder(provider)
        .permission(PermissionMode::Auto)
        .build();

    let h = AgentHarness::run(&agent, "Say hello.").await;
    h.assert_text_contains("Hello from the agent!")
        .assert_stop_reason(RunStopReason::Completed);
}

// ── Test 2: agent_calls_tool_and_uses_result ──────────────────────────────────

#[tokio::test]
async fn agent_calls_tool_and_uses_result() {
    let provider = MockProvider::new(vec![
        MockProvider::tool_call("test_tool", json!({})),
        MockProvider::text("done"),
    ]);

    let agent = Agent::builder(provider)
        .tool(ConstTool {
            name: "test_tool",
            output: "tool_result_value",
        })
        .permission(PermissionMode::Auto)
        .build();

    let h = AgentHarness::run(&agent, "Call the tool.").await;
    h.assert_tool_called("test_tool")
        .assert_text_contains("done")
        .assert_stop_reason(RunStopReason::Completed);
}

// ── Test 3: tool_denied_by_permission_rules ───────────────────────────────────

/// A tool that increments a counter when called.
struct CountingTool {
    name: &'static str,
    counter: Arc<AtomicU32>,
}

#[async_trait]
impl Tool for CountingTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "Counting tool."
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    async fn call(&self, _input: Value, _ctx: &ToolCtx) -> ToolOutput {
        self.counter.fetch_add(1, Ordering::SeqCst);
        ToolOutput::success("counted")
    }
}

#[tokio::test]
async fn tool_denied_by_permission_rules() {
    let counter = Arc::new(AtomicU32::new(0));

    let provider = MockProvider::new(vec![
        MockProvider::tool_call("denied_tool", json!({})),
        // After the tool is denied the LLM gets a permission-denied result
        // and produces a text response.
        MockProvider::text("I could not run the tool."),
    ]);

    let agent = Agent::builder(provider)
        .tool(CountingTool {
            name: "denied_tool",
            counter: Arc::clone(&counter),
        })
        .permission(PermissionMode::Auto)
        .deny_tool("denied_tool")
        .build();

    let h = AgentHarness::run(&agent, "Run the denied tool.").await;

    // The tool implementation must NOT have been invoked.
    assert_eq!(
        counter.load(Ordering::SeqCst),
        0,
        "denied tool should not be executed"
    );
    h.assert_tool_not_called("denied_tool")
        .assert_stop_reason(RunStopReason::Completed);
}

// ── Test 4: max_iter_stops_loop ───────────────────────────────────────────────

#[tokio::test]
async fn max_iter_stops_loop() {
    // Provider always returns a tool call, creating an infinite loop if unchecked.
    // Build a queue large enough so MockProvider never exhausts before max_iter hits.
    let responses: Vec<_> = (0..10)
        .map(|_| MockProvider::tool_call("loop_tool", json!({})))
        .collect();

    let provider = MockProvider::new(responses);

    let agent = Agent::builder(provider)
        .tool(ConstTool {
            name: "loop_tool",
            output: "looping",
        })
        .permission(PermissionMode::Auto)
        .max_iter(2)
        .build();

    let h = AgentHarness::run(&agent, "Loop forever.").await;
    h.assert_stop_reason(RunStopReason::MaxIterations);
}

// ── Test 5: checkpoint_saves_and_restores ─────────────────────────────────────

#[tokio::test]
async fn checkpoint_saves_and_restores() {
    let store = InMemoryCheckpointStore::new();
    let run_id = "test-run-checkpoint-001";

    // First run: one tool call turn then text.
    {
        let provider = MockProvider::new(vec![
            MockProvider::tool_call("cp_tool", json!({})),
            MockProvider::text("first run done"),
        ]);

        let agent = Agent::builder(provider)
            .tool(ConstTool {
                name: "cp_tool",
                output: "cp_result",
            })
            .permission(PermissionMode::Auto)
            .checkpoint(store.clone(), run_id)
            .build();

        let h = AgentHarness::run(&agent, "Do something.").await;
        h.assert_stop_reason(RunStopReason::Completed);
    }

    // Second run: checkpoint restores previous messages.
    // The MockProvider only needs to handle the NEW prompt — the history
    // from the first run is already in the checkpoint.
    {
        let provider = MockProvider::new(vec![MockProvider::text("second run done")]);

        let agent = Agent::builder(provider)
            .tool(ConstTool {
                name: "cp_tool",
                output: "cp_result",
            })
            .permission(PermissionMode::Auto)
            .checkpoint(store.clone(), run_id)
            .build();

        let h = AgentHarness::run(&agent, "Continue.").await;
        h.assert_stop_reason(RunStopReason::Completed);

        // The summary messages should include history from the first run.
        let summary = h.summary();
        assert!(
            summary.messages.len() >= 4,
            "expected checkpoint history to be restored (>= 4 messages), got {}",
            summary.messages.len(),
        );
    }
}

// ── Test 6: tool_retry_on_error ───────────────────────────────────────────────

/// A tool that fails the first `fail_times` calls, then succeeds.
struct FlakyTool {
    counter: Arc<AtomicU32>,
    fail_times: u32,
}

#[async_trait]
impl Tool for FlakyTool {
    fn name(&self) -> &str {
        "flaky_tool"
    }
    fn description(&self) -> &str {
        "Fails N times then succeeds."
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    fn executor_hints(&self, _input: &Value) -> ExecutorHints {
        ExecutorHints {
            max_retries: 2,
            ..ExecutorHints::default()
        }
    }
    async fn call(&self, _input: Value, _ctx: &ToolCtx) -> ToolOutput {
        let attempt = self.counter.fetch_add(1, Ordering::SeqCst) + 1;
        if attempt <= self.fail_times {
            ToolOutput::error(format!("attempt {attempt} failed"))
        } else {
            ToolOutput::success("finally succeeded")
        }
    }
}

#[tokio::test]
async fn tool_retry_on_error() {
    let counter = Arc::new(AtomicU32::new(0));

    let provider = MockProvider::new(vec![
        MockProvider::tool_call("flaky_tool", json!({})),
        MockProvider::text("done after retries"),
    ]);

    let agent = Agent::builder(provider)
        .tool(FlakyTool {
            counter: Arc::clone(&counter),
            fail_times: 2,
        })
        .permission(PermissionMode::Auto)
        .build();

    let h = AgentHarness::run(&agent, "Call the flaky tool.").await;
    h.assert_tool_called("flaky_tool")
        .assert_stop_reason(RunStopReason::Completed);

    // With max_retries: 2, the tool is called up to 3 times (1 initial + 2 retries).
    // It fails on attempts 1 and 2, succeeds on attempt 3.
    assert_eq!(
        counter.load(Ordering::SeqCst),
        3,
        "tool should have been called 3 times (1 initial + 2 retries)"
    );
}

// ── Test 7: hitl_approve_runs_tool ────────────────────────────────────────────

/// Drive an agent stream manually, approving every Control event.
async fn run_approve_all(agent: &Agent, prompt: impl Into<String>) -> Vec<AgentEvent> {
    let mut stream = agent.stream(prompt.into());
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        if let AgentEvent::Control(ref handle) = event {
            handle.approve();
        }
        events.push(event);
    }
    events
}

#[tokio::test]
async fn hitl_approve_runs_tool() {
    let counter = Arc::new(AtomicU32::new(0));

    let provider = MockProvider::new(vec![
        MockProvider::tool_call("ask_tool", json!({})),
        MockProvider::text("all done"),
    ]);

    let agent = Agent::builder(provider)
        .tool(CountingTool {
            name: "ask_tool",
            counter: Arc::clone(&counter),
        })
        // Ask mode — every tool call goes through HITL.
        .permission(PermissionMode::Ask)
        .build();

    let events = run_approve_all(&agent, "Call ask_tool.").await;

    // The tool must have executed once.
    assert_eq!(
        counter.load(Ordering::SeqCst),
        1,
        "approved tool should execute exactly once"
    );

    // A ToolStart event must be present.
    let tool_started = events
        .iter()
        .any(|e| matches!(e, AgentEvent::ToolStart { name, .. } if name == "ask_tool"));
    assert!(tool_started, "expected ToolStart for ask_tool");
}

// ── Test 8: hitl_deny_skips_tool ─────────────────────────────────────────────

/// Drive an agent stream manually, denying every Control event.
async fn run_deny_all(agent: &Agent, prompt: impl Into<String>) -> Vec<AgentEvent> {
    let mut stream = agent.stream(prompt.into());
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        if let AgentEvent::Control(ref handle) = event {
            handle.deny("denied by test");
        }
        events.push(event);
    }
    events
}

#[tokio::test]
async fn hitl_deny_skips_tool() {
    let counter = Arc::new(AtomicU32::new(0));

    let provider = MockProvider::new(vec![
        MockProvider::tool_call("deny_ask_tool", json!({})),
        // After denial the LLM receives a permission-denied result and responds.
        MockProvider::text("tool was denied"),
    ]);

    let agent = Agent::builder(provider)
        .tool(CountingTool {
            name: "deny_ask_tool",
            counter: Arc::clone(&counter),
        })
        .permission(PermissionMode::Ask)
        .build();

    run_deny_all(&agent, "Call deny_ask_tool.").await;

    // The tool implementation must NOT have been invoked.
    assert_eq!(
        counter.load(Ordering::SeqCst),
        0,
        "denied tool must not execute"
    );
}

// ── Test 9: session_preserves_history ────────────────────────────────────────

#[tokio::test]
async fn session_preserves_history() {
    use wui::InMemorySessionStore;

    let store = InMemorySessionStore::new();
    let session_id = "test-session-multi-turn";

    // Build a shared provider that queues two independent turns.
    // Each new session run will draw the next response.
    let provider = MockProvider::new(vec![
        MockProvider::text("turn one response"),
        MockProvider::text("turn two response"),
    ]);

    let agent = Agent::builder(provider)
        .permission(PermissionMode::Auto)
        .session_store(store)
        .build();

    let session = agent.session(session_id).await;

    // First turn.
    let mut stream = session.send("first turn");
    let mut text1 = String::new();
    while let Some(event) = stream.next().await {
        if let AgentEvent::TextDelta(t) = event {
            text1.push_str(&t);
        }
    }
    assert!(text1.contains("turn one response"), "first turn: {text1}");

    // Second turn — the session should include history from turn one.
    let mut stream = session.send("second turn");
    let mut text2 = String::new();
    let mut summary2: Option<wui::RunSummary> = None;
    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TextDelta(t) => text2.push_str(&t),
            AgentEvent::Done(s) => summary2 = Some(s),
            _ => {}
        }
    }
    assert!(text2.contains("turn two response"), "second turn: {text2}");

    // History from turn one must be present in the second turn's messages.
    let summary = summary2.expect("second turn did not complete");
    assert!(
        summary.messages.len() >= 3,
        "expected at least 3 messages after two turns, got {}",
        summary.messages.len()
    );
}

// ── Test 10: dynamic_tool_expose ──────────────────────────────────────────────

/// A tool that exposes a second tool when called.
struct BootstrapTool {
    exposed_tool: Arc<dyn Tool>,
}

#[async_trait]
impl Tool for BootstrapTool {
    fn name(&self) -> &str {
        "bootstrap"
    }
    fn description(&self) -> &str {
        "Exposes the exposed_tool when called."
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    async fn call(&self, _input: Value, _ctx: &ToolCtx) -> ToolOutput {
        ToolOutput::success("bootstrap done").expose([Arc::clone(&self.exposed_tool)])
    }
}

#[tokio::test]
async fn dynamic_tool_expose() {
    let counter = Arc::new(AtomicU32::new(0));
    let exposed = Arc::new(CountingTool {
        name: "exposed_tool",
        counter: Arc::clone(&counter),
    });

    let provider = MockProvider::new(vec![
        // First call: bootstrap tool (exposes exposed_tool into the registry).
        MockProvider::tool_call("bootstrap", json!({})),
        // Second call: the now-visible exposed_tool.
        MockProvider::tool_call("exposed_tool", json!({})),
        MockProvider::text("finished"),
    ]);

    let agent = Agent::builder(provider)
        .tool(BootstrapTool {
            exposed_tool: exposed as Arc<dyn Tool>,
        })
        .permission(PermissionMode::Auto)
        .build();

    let h = AgentHarness::run(&agent, "Bootstrap then call exposed_tool.").await;
    h.assert_tool_called("bootstrap")
        .assert_stop_reason(RunStopReason::Completed);

    assert_eq!(
        counter.load(Ordering::SeqCst),
        1,
        "exposed_tool should have been called once after bootstrap exposed it"
    );
}
