// ============================================================================
// SubAgent — run an Agent as a Tool (synchronous delegation).
//
// Delegation pattern: one agent (the "supervisor") calls another agent
// (the "sub-agent") as a tool call. The supervisor describes the task in
// natural language; the sub-agent resolves it with its own tools and returns
// a plain-text result.
//
// This is not multi-agent "orchestration" in the sense of a shared graph or
// message bus — each agent loop is completely independent. The supervisor sees
// only the sub-agent's final text output. Inner tool calls are forwarded to
// the outer stream as `ToolProgress` events for observability.
//
// The final `ToolDone` event carries a compact `SubAgentSummary` in the
// `structured` field so callers can inspect usage, tool calls, and stop reason
// without parsing prose.
//
// Sub-agents MUST be configured with `PermissionMode::Auto` or a suitable
// allow-list — `Agent::run()` cannot handle interactive approval requests.
//
// ── SubAgent vs wui-spawn ────────────────────────────────────────────────────
//
// `SubAgent` (this file): wraps one Agent as a synchronous Tool. The supervisor
// calls it, waits for the result, and the turn ends. One-shot, blocking,
// result appears in the same turn it was requested.
//
// `wui-spawn` (separate crate): background agent registry. The supervisor
// spawns an Agent as a background job (non-blocking, returns a job ID
// immediately), and can check status, await the result, or cancel across
// separate turns. Use wui-spawn when the sub-task is long-running or when
// the supervisor needs to do other work while waiting.
//
// Usage via AgentBuilder:
//
//   let researcher = Agent::builder(provider.clone())
//       .tool(WebSearch)
//       .permission(PermissionMode::Auto)
//       .build();
//
//   let supervisor = Agent::builder(provider)
//       .sub_agent("research", "Search the web and summarise findings.", researcher)
//       .build();
//
// Or directly:
//
//   let agent = Agent::builder(provider)
//       .tool(SubAgent::new("analyst", "Analyse CSV data.", analyst_agent))
//       .build();
// ============================================================================

use async_trait::async_trait;
use futures::StreamExt as _;
use serde::Serialize;
use serde_json::{json, Value};

use wui_core::event::{AgentEvent, RunStopReason, TokenUsage};
use wui_core::tool::{Tool, ToolCtx, ToolInput, ToolMeta, ToolOutput};

use super::agent::Agent;
use crate::runtime::HookRunner;

// ── SubAgentSummary ───────────────────────────────────────────────────────────

/// Compact summary of a sub-agent run.
///
/// Returned in `AgentEvent::ToolDone { structured }` so callers can inspect
/// the inner run without parsing the final text response.
#[derive(Debug, Serialize)]
pub struct SubAgentSummary {
    pub stop_reason: RunStopReason,
    pub iterations: u32,
    pub usage: TokenUsage,
    /// Tool calls made by the sub-agent, in execution order.
    pub tool_calls: Vec<SubAgentToolCall>,
}

/// A single tool invocation recorded in a [`SubAgentSummary`].
#[derive(Debug, Serialize)]
pub struct SubAgentToolCall {
    pub name: String,
    pub ms: u64,
    pub success: bool,
}

// ── SubAgent ──────────────────────────────────────────────────────────────────

/// Wraps an [`Agent`] as a [`Tool`], enabling supervisor → sub-agent delegation.
///
/// The sub-agent is invoked with a single `prompt` string. Its final text
/// response becomes the tool output. Inner tool calls are forwarded to the
/// outer stream as `ToolProgress` events for real-time observability, and a
/// compact [`SubAgentSummary`] is returned in `ToolDone::structured`.
///
/// Sub-agents run concurrently by default (`is_concurrent_for` returns `true`).
/// The inner agent MUST be configured with `PermissionMode::Auto` or an
/// allow-list — interactive approval is not supported in delegation mode.
pub struct SubAgent {
    tool_name: String,
    tool_desc: String,
    agent: Agent,
}

impl SubAgent {
    /// Create a new sub-agent tool.
    ///
    /// - `name`: the tool name the supervisor's LLM will use to invoke it
    /// - `description`: what the sub-agent does (shown in the tool listing)
    /// - `agent`: a fully configured `Agent` — must use `PermissionMode::Auto`
    ///   or a suitable allow-list for headless delegation
    pub fn new(name: impl Into<String>, description: impl Into<String>, agent: Agent) -> Self {
        Self {
            tool_name: name.into(),
            tool_desc: description.into(),
            agent,
        }
    }
}

#[async_trait]
impl Tool for SubAgent {
    fn name(&self) -> &str {
        &self.tool_name
    }
    fn description(&self) -> &str {
        &self.tool_desc
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task description for the sub-agent."
                }
            },
            "required": ["prompt"]
        })
    }

    /// Sub-agents are safe to run concurrently — each has its own isolated
    /// loop and message history.
    fn meta(&self, _input: &Value) -> ToolMeta {
        ToolMeta {
            concurrent: true,
            ..ToolMeta::default()
        }
    }

    async fn call(&self, input: Value, ctx: &ToolCtx) -> ToolOutput {
        let inp = ToolInput(&input);
        let prompt = match inp.required_str("prompt") {
            Ok(v) => v,
            Err(e) => return ToolOutput::invalid_input(e),
        };

        // Enforce sub-agent nesting depth limit.
        let max = self.agent.config.max_spawn_depth;
        if ctx.spawn_depth >= max {
            return ToolOutput::error(format!("sub-agent spawn depth limit reached (max: {max})"));
        }

        let hooks = HookRunner::new(self.agent.config.hooks.clone());
        hooks.notify_subagent_start(&self.tool_name, prompt).await;

        ctx.report(format!("delegating to sub-agent '{}'", self.tool_name));

        // Drive the inner agent via stream() so we can:
        //   1. Forward cancellation from the outer ctx.
        //   2. Report inner tool calls as ToolProgress for outer observability.
        //   3. Collect the full RunSummary for structured output.
        let mut stream = self
            .agent
            .stream_with_spawn_depth(prompt, ctx.spawn_depth + 1);

        let mut final_text = String::new();
        let mut tool_calls = Vec::<SubAgentToolCall>::new();

        loop {
            let event = tokio::select! {
                e = stream.next() => match e {
                    Some(e) => e,
                    None    => break,   // stream ended without Done — treat as empty
                },
                _ = ctx.cancel.cancelled() => {
                    let message = "sub-agent cancelled";
                    hooks
                        .notify_subagent_end(&self.tool_name, Err(message))
                        .await;
                    return ToolOutput::error(message);
                }
            };

            match event {
                AgentEvent::TextDelta(t) => {
                    final_text.push_str(&t);
                }

                AgentEvent::ToolStart { ref name, .. } => {
                    ctx.report(format!("[{}] → {name}", self.tool_name));
                }

                AgentEvent::ToolDone { ref name, ms, .. } => {
                    ctx.report(format!("[{}] ✓ {name} ({ms}ms)", self.tool_name));
                    tool_calls.push(SubAgentToolCall {
                        name: name.clone(),
                        ms,
                        success: true,
                    });
                }

                AgentEvent::ToolError {
                    ref name,
                    ms,
                    ref error,
                    ..
                } => {
                    ctx.report(format!("[{}] ✗ {name}: {error}", self.tool_name));
                    tool_calls.push(SubAgentToolCall {
                        name: name.clone(),
                        ms,
                        success: false,
                    });
                }

                AgentEvent::Done(summary) => {
                    let structured = SubAgentSummary {
                        stop_reason: summary.stop_reason,
                        iterations: summary.iterations,
                        usage: summary.usage,
                        tool_calls,
                    };
                    hooks
                        .notify_subagent_end(&self.tool_name, Ok(&final_text))
                        .await;
                    return ToolOutput::success(final_text)
                        .with_structured(serde_json::to_value(structured).unwrap_or_default());
                }

                AgentEvent::Error(e) if e.permission_denied => {
                    let message = format!(
                        "sub-agent '{}' stopped because it requires interactive tool approval, \
                         which is not supported in delegation mode. \
                         Build the sub-agent with PermissionMode::Auto or add an allow rule \
                         for the tool that triggered the request. Underlying error: {e}",
                        self.tool_name,
                    );
                    hooks
                        .notify_subagent_end(&self.tool_name, Err(&message))
                        .await;
                    return ToolOutput::error(message);
                }

                AgentEvent::Error(e) => {
                    let message = e.to_string();
                    hooks
                        .notify_subagent_end(&self.tool_name, Err(&message))
                        .await;
                    return ToolOutput::error(message);
                }

                _ => {}
            }
        }

        // Stream ended without AgentEvent::Done — return whatever text we have.
        hooks
            .notify_subagent_end(&self.tool_name, Ok(&final_text))
            .await;
        ToolOutput::success(final_text)
    }
}
