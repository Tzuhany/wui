use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use tokio::sync::Mutex;

use super::{run, RetryPolicy, RunConfig};
use crate::compress::CompressPipeline;
use crate::runtime::hooks::HookRunner;
use crate::runtime::permission::{PermissionMode, PermissionRules, SessionPermissions};
use crate::runtime::registry::ToolRegistry;
use wui_core::event::{AgentEvent, StopReason, StreamEvent, TokenUsage};
use wui_core::message::{ContentBlock, Message, Role};
use wui_core::provider::{ChatRequest, Provider, ProviderError};
use wui_core::tool::{Tool, ToolCtx, ToolMeta, ToolOutput};

// ── Test provider ────────────────────────────────────────────────────────────

#[derive(Clone, Default)]
struct SequenceProvider {
    responses: Arc<Mutex<VecDeque<Vec<StreamEvent>>>>,
    requests: Arc<Mutex<Vec<ChatRequest>>>,
}

impl SequenceProvider {
    fn new(responses: Vec<Vec<StreamEvent>>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses.into())),
            requests: Arc::new(Mutex::new(Vec::new())),
        }
    }

    async fn requests(&self) -> Vec<ChatRequest> {
        self.requests.lock().await.clone()
    }
}

#[async_trait]
impl Provider for SequenceProvider {
    async fn stream(
        &self,
        req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        self.requests.lock().await.push(req);
        let next = self
            .responses
            .lock()
            .await
            .pop_front()
            .expect("provider called more times than expected");
        Ok(Box::pin(futures::stream::iter(next.into_iter().map(Ok))))
    }
}

// ── Test tool ────────────────────────────────────────────────────────────────

struct SleepTool {
    name: &'static str,
    delay_ms: u64,
    output: &'static str,
    readonly: bool,
}

#[async_trait]
impl Tool for SleepTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "test tool"
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    fn meta(&self, _input: &Value) -> ToolMeta {
        ToolMeta {
            readonly: self.readonly,
            concurrent: self.readonly,
            ..ToolMeta::default()
        }
    }
    async fn call(&self, _input: Value, _ctx: &ToolCtx) -> ToolOutput {
        tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
        ToolOutput::success(self.output)
    }
}

// ── Config builder ───────────────────────────────────────────────────────────

fn test_config(
    provider: Arc<dyn Provider>,
    tools: Vec<Arc<dyn Tool>>,
    permission: PermissionMode,
) -> Arc<RunConfig> {
    Arc::new(RunConfig {
        provider,
        tools: Arc::new(ToolRegistry::new(tools, vec![])),
        hooks: Arc::new(HookRunner::new(Vec::new())),
        compress: Arc::new(CompressPipeline {
            window_tokens: 1_000_000,
            ..CompressPipeline::default()
        }),
        permission,
        rules: PermissionRules::default(),
        session_perms: Arc::new(SessionPermissions::new()),
        system: "You are a test agent.".to_string(),
        model: None,
        max_tokens: 1024,
        temperature: None,
        max_iter: 8,
        retry: RetryPolicy {
            max_attempts: 0,
            ..RetryPolicy::default()
        },
        tool_timeout: None,
        ignore_diminishing_returns: true,
        token_budget: None,
        thinking_budget: None,
        checkpoint_store: None,
        checkpoint_run_id: None,
        result_store: None,
    })
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn tool_results_are_written_in_submission_order_even_if_completion_order_differs() {
    let provider = SequenceProvider::new(vec![
        vec![
            StreamEvent::ToolUseStart {
                id: "slow-id".into(),
                name: "slow_tool".into(),
            },
            StreamEvent::ToolInputDelta {
                id: "slow-id".into(),
                chunk: "{}".into(),
            },
            StreamEvent::ToolUseEnd {
                id: "slow-id".into(),
            },
            StreamEvent::ToolUseStart {
                id: "fast-id".into(),
                name: "fast_tool".into(),
            },
            StreamEvent::ToolInputDelta {
                id: "fast-id".into(),
                chunk: "{}".into(),
            },
            StreamEvent::ToolUseEnd {
                id: "fast-id".into(),
            },
            StreamEvent::MessageEnd {
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 10,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                },
                stop_reason: StopReason::ToolUse,
            },
        ],
        vec![
            StreamEvent::TextDelta {
                text: "done".into(),
            },
            StreamEvent::MessageEnd {
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 10,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                },
                stop_reason: StopReason::EndTurn,
            },
        ],
    ]);

    let provider_ref = provider.clone();
    let config = test_config(
        Arc::new(provider),
        vec![
            Arc::new(SleepTool {
                name: "slow_tool",
                delay_ms: 60,
                output: "slow result",
                readonly: true,
            }),
            Arc::new(SleepTool {
                name: "fast_tool",
                delay_ms: 5,
                output: "fast result",
                readonly: true,
            }),
        ],
        PermissionMode::Auto,
    );

    let mut stream = run(config, vec![Message::user("run tools")]);
    let mut tool_done_order = Vec::new();
    let mut summary = None;

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::ToolDone { name, .. } => tool_done_order.push(name),
            AgentEvent::Done(done) => {
                summary = Some(done);
                break;
            }
            AgentEvent::Error(e) => panic!("unexpected error: {e}"),
            _ => {}
        }
    }

    assert_eq!(tool_done_order, vec!["slow_tool", "fast_tool"]);

    let summary = summary.expect("run should complete");
    let tool_result_message = summary
        .messages
        .iter()
        .find(|m| {
            m.role == Role::User
                && matches!(m.content.first(), Some(ContentBlock::ToolResult { .. }))
        })
        .expect("tool result message should exist");

    let result_order: Vec<(&str, &str)> = tool_result_message
        .content
        .iter()
        .map(|block| match block {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => (tool_use_id.as_str(), content.as_str()),
            other => panic!("expected only tool results, got {other:?}"),
        })
        .collect();

    assert_eq!(
        result_order,
        vec![("slow-id", "slow result"), ("fast-id", "fast result")]
    );
    assert_eq!(provider_ref.requests().await.len(), 2);
}

#[tokio::test]
async fn ask_mode_emits_control_and_resumes_after_approval() {
    let provider = SequenceProvider::new(vec![
        vec![
            StreamEvent::ToolUseStart {
                id: "approve-id".into(),
                name: "needs_approval".into(),
            },
            StreamEvent::ToolInputDelta {
                id: "approve-id".into(),
                chunk: "{}".into(),
            },
            StreamEvent::ToolUseEnd {
                id: "approve-id".into(),
            },
            StreamEvent::MessageEnd {
                usage: TokenUsage {
                    input_tokens: 5,
                    output_tokens: 5,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                },
                stop_reason: StopReason::ToolUse,
            },
        ],
        vec![
            StreamEvent::TextDelta {
                text: "approved path".into(),
            },
            StreamEvent::MessageEnd {
                usage: TokenUsage {
                    input_tokens: 5,
                    output_tokens: 5,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                },
                stop_reason: StopReason::EndTurn,
            },
        ],
    ]);

    let provider_ref = provider.clone();
    let config = test_config(
        Arc::new(provider),
        vec![Arc::new(SleepTool {
            name: "needs_approval",
            delay_ms: 1,
            output: "approved tool ran",
            readonly: false,
        })],
        PermissionMode::Ask,
    );

    let mut stream = run(config, vec![Message::user("please do the thing")]);
    let mut saw_control = false;
    let mut saw_tool_done = false;
    let mut done_summary = None;

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::Control(handle) => {
                saw_control = true;
                handle.approve();
            }
            AgentEvent::ToolDone { name, output, .. } => {
                assert_eq!(name, "needs_approval");
                assert_eq!(output, "approved tool ran");
                saw_tool_done = true;
            }
            AgentEvent::Done(s) => {
                done_summary = Some(s);
                break;
            }
            AgentEvent::Error(e) => panic!("unexpected error: {e}"),
            _ => {}
        }
    }

    assert!(saw_control, "ask mode should emit a control request");
    assert!(saw_tool_done, "tool should complete after approval");

    let summary = done_summary.expect("run should complete");
    assert!(summary.messages.iter().any(|msg| {
        msg.role == Role::System
            && msg.content.iter().any(|block| match block {
                ContentBlock::Text { text } => text.contains("approved your request"),
                _ => false,
            })
    }));
    assert_eq!(provider_ref.requests().await.len(), 2);
}
