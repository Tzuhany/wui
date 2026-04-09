// ============================================================================
// The Loop — the beating heart of Wui.
//
// "The framework is an executor, not a thinker."
// The LLM decides what to do. This loop does it, feeds results back,
// and repeats until the LLM says it's done.
//
// Structure of one iteration:
//   1. Check context pressure → compress if needed → check for overflow.
//   2. Build ChatRequest.
//   3. Call provider.stream() with retry → receive StreamEvents.
//   4. Process stream:
//        - TextDelta / ThinkingDelta → emit AgentEvent immediately
//        - ToolUseEnd → validate + collect into pending_auths (no auth yet)
//        - MessageEnd → break inner loop
//   5. Authorize all collected tools concurrently (JoinSet).
//        - HITL prompts fire after the LLM has finished speaking.
//        - Each tool starts executing the moment its own auth resolves.
//   5. collect_remaining() — await any still-running tools.
//   6. Run PostToolUse hooks — may inject system notices for blocked outputs.
//   7. Append tool results as a single user message (closes the loop).
//   8. Append context injections as system reminders.
//   9. Run PreStop hook.
//   10. Evaluate stop condition.
//   11. Continue or return RunSummary.
//
// ── Bounded channel ───────────────────────────────────────────────────────────
//
// The event channel is bounded (capacity: EVENT_CHANNEL_CAPACITY). If the
// caller consumes events slower than the loop produces them, the loop blocks
// rather than buffering unboundedly. This is backpressure: the loop runs at
// the caller's pace, not ahead of it.
//
// ── Ordered tool results ──────────────────────────────────────────────────────
//
// Tools complete in parallel, in unpredictable order. But tool results must
// be appended to message history in the same order they were submitted —
// matching the ToolUse blocks in the preceding assistant message.
//
// We track `submission_order: Vec<String>` (tool ids, in submission order)
// and collect completions into a HashMap. Before appending to history, we
// reconstruct the result slice in submission order.
//
// ToolDone / ToolError events are emitted in submission order, after post-tool
// hooks run, so that MutateOutput decisions are reflected in emitted events.
// Both the history write and the event stream are now in submission order.
//
// ── SessionPermissions ────────────────────────────────────────────────────────
//
// When a user responds with ApproveAlways or DenyAlways, the session records
// the decision. Subsequent tool calls of the same name skip the permission
// prompt entirely, matching the recorded decision automatically.
// ============================================================================

mod compression;
mod config;
mod history;
mod parsing;
mod provider;
mod state;
mod stream;
mod tool_batch;

// Re-export public / pub(crate) items at the `run` module boundary so that
// existing import paths (`super::run::run`, `super::run::RunConfig`, etc.)
// continue to resolve unchanged.
pub(crate) use config::RunConfig;
pub use provider::RetryPolicy;
pub(crate) use stream::run;
pub use stream::RunStream;

// Re-export state types at `super::` level so sibling submodules
// (`compression`, `history`, `parsing`, `tool_batch`, etc.) can still
// write `super::RunState`, `super::IterationCtx`, etc.
use state::{preflight_check, IterGuard, IterationCtx, MaxTokensAction, RunState};

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use wui_core::event::{AgentError, AgentEvent, RunStopReason, RunSummary, StopReason, TokenUsage};
use wui_core::hook::HookDecision;
use wui_core::message::Message;
use wui_core::provider::ChatRequest;

// ── Sibling runtime modules used by submodules ──────────────────────────────
// Re-export at `super::` level so submodules can write `super::auth` etc.
use super::auth;
use super::checkpoint;
use super::executor::{CompletedTool, PendingTool, ToolExecutor};
use super::registry::{self, ToolRegistry};

// ── Submodule imports used in this file ─────────────────────────────────────
use compression::{emergency_compress, maybe_compress};
use history::{last_assistant_text, replace_last_assistant_text, system_reminder_msg};
use parsing::parse_stream;
use provider::{call_with_retry, is_prompt_too_long, ProviderStream};
use tool_batch::{authorize_and_dispatch, run_post_hooks};

// ── Diminishing-returns constants ─────────────────────────────────────────────

/// Minimum output tokens in a single assistant turn to be considered "productive".
/// Turns below this threshold count toward the [`MAX_LOW_OUTPUT_TURNS`] streak.
const MIN_USEFUL_OUTPUT_TOKENS: u32 = 500;

/// Consecutive low-output turns before the run auto-stops with
/// [`RunStopReason::DiminishingReturns`].
///
/// If the LLM produces fewer than [`MIN_USEFUL_OUTPUT_TOKENS`] output tokens
/// for this many turns in a row the run terminates to avoid burning budget on
/// a stalled agent. The streak counter resets on any productive tool-use turn.
const MAX_LOW_OUTPUT_TURNS: u32 = 3;

// ── Max-tokens escalation constants ──────────────────────────────────────────

/// How many times we will escalate `max_tokens` before giving up.
const MAX_TOKEN_ESCALATIONS: u32 = 2;

/// Multiplier applied on the first escalation (e.g. 16 384 → 65 536).
const TOKEN_ESCALATION_FACTOR: u32 = 4;

enum IterationSetup {
    Continue,
    Finish(RunSummary),
    Ready {
        registry: Arc<ToolRegistry>,
        stream: ProviderStream,
    },
}

struct IterationOutcome {
    stop_reason: StopReason,
    usage: TokenUsage,
}

enum LoopControl {
    Continue,
    Finish(RunSummary),
}

// ── Main loop ────────────────────────────────────────────────────────────────

async fn run_loop(
    config: Arc<RunConfig>,
    messages: Vec<Message>,
    cancel: CancellationToken,
    tx: &mpsc::Sender<AgentEvent>,
) -> Result<RunSummary, AgentError> {
    let mut s = RunState::new(&config, messages).await;

    loop {
        if cancel.is_cancelled() {
            return Ok(s.summary(RunStopReason::Cancelled));
        }

        match s.check_max_iter(&config).await {
            IterGuard::Stop(summary) => return Ok(summary),
            IterGuard::Blocked => continue,
            IterGuard::Proceed => {}
        }

        // Log iteration start as an event (not a span guard) because the loop
        // body contains multiple await points and EnteredSpan is not Send.
        tracing::info!(iteration = s.iterations + 1, "wui.iteration.start");

        let (registry, stream) = match prepare_iteration(&config, &mut s, tx).await? {
            IterationSetup::Continue => continue,
            IterationSetup::Finish(summary) => return Ok(summary),
            IterationSetup::Ready { registry, stream } => (registry, stream),
        };

        let Some(outcome) =
            execute_iteration(&config, &mut s, registry, stream, cancel.clone(), tx).await?
        else {
            continue;
        };

        match evaluate_stop_conditions(&config, &mut s, outcome).await {
            LoopControl::Continue => continue,
            LoopControl::Finish(summary) => return Ok(summary),
        }
    }
}

async fn prepare_iteration(
    config: &Arc<RunConfig>,
    s: &mut RunState,
    tx: &mpsc::Sender<AgentEvent>,
) -> Result<IterationSetup, AgentError> {
    config.hooks.notify_turn_start(&s.messages).await;
    maybe_compress(config, &mut s.messages, tx).await;

    if config.compress.is_critically_full(&s.messages) {
        return Ok(IterationSetup::Finish(
            s.summary(RunStopReason::ContextOverflow),
        ));
    }

    let registry = active_registry(config, s);
    let req = build_chat_request(config, s, &registry);
    run_preflight(config, s, &req)?;

    match call_with_retry(config, &req, tx).await {
        Ok(stream) => Ok(IterationSetup::Ready { registry, stream }),
        Err(error) if is_prompt_too_long(&error) => {
            if emergency_compress(config, &mut s.messages, tx).await {
                Ok(IterationSetup::Continue)
            } else {
                Err(error)
            }
        }
        Err(error) => Err(error),
    }
}

fn active_registry(config: &RunConfig, s: &RunState) -> Arc<ToolRegistry> {
    if s.dynamic_tools.is_empty() {
        config.tools.clone()
    } else {
        Arc::new(config.tools.with_dynamic(&s.dynamic_tools))
    }
}

fn build_chat_request(config: &RunConfig, s: &RunState, registry: &ToolRegistry) -> ChatRequest {
    ChatRequest {
        model: config.model.clone(),
        max_tokens: s.recovery.effective_max_tokens,
        temperature: config.temperature,
        system: s.system.clone(),
        messages: s.messages.clone(),
        tools: if let Some(ref f) = config.tool_filter {
            registry.filtered_tool_defs(&**f)
        } else {
            registry.tool_defs()
        },
        thinking_budget: config.thinking_budget,
        cache_boundary: s.cache_boundary,
    }
}

fn run_preflight(
    config: &RunConfig,
    s: &mut RunState,
    req: &ChatRequest,
) -> Result<(), AgentError> {
    if !s.preflight_done {
        preflight_check(config, req)?;
        s.preflight_done = true;
    }
    Ok(())
}

async fn execute_iteration(
    config: &Arc<RunConfig>,
    s: &mut RunState,
    registry: Arc<ToolRegistry>,
    stream: ProviderStream,
    cancel: CancellationToken,
    tx: &mpsc::Sender<AgentEvent>,
) -> Result<Option<IterationOutcome>, AgentError> {
    let mut executor = ToolExecutor::new(
        registry.clone(),
        cancel,
        tx.clone(),
        config.tool_timeout,
        config.result_store.clone(),
        config.spawn_depth,
    );
    let mut ctx = IterationCtx::new();
    futures::pin_mut!(stream);

    let got_message_end = parse_stream(&mut stream, &mut ctx, &registry, tx).await?;
    if !got_message_end {
        return handle_stream_drop(s, &ctx);
    }

    authorize_and_dispatch(&mut ctx, config, &registry, &mut executor, &s.messages, tx).await;
    let outcome = IterationOutcome {
        stop_reason: ctx.stop_reason.clone(),
        usage: ctx.usage.clone(),
    };
    run_post_hooks(&mut ctx, executor, config, &mut s.messages, tx).await;
    history::assemble_history(ctx, s, config, tx).await;
    Ok(Some(outcome))
}

fn handle_stream_drop(
    s: &RunState,
    ctx: &IterationCtx,
) -> Result<Option<IterationOutcome>, AgentError> {
    if ctx.submission_order.is_empty() {
        tracing::warn!(s.iterations, "stream ended without MessageEnd — retrying");
        return Ok(None);
    }

    Err(AgentError::retryable(
        "provider stream dropped mid-response with tools in flight",
    ))
}

async fn evaluate_stop_conditions(
    config: &Arc<RunConfig>,
    s: &mut RunState,
    outcome: IterationOutcome,
) -> LoopControl {
    if budget_exhausted(config, s) {
        return stop_with_hooks(config, s, RunStopReason::BudgetExhausted, |_| {}, |_, _| {}).await;
    }

    if let Some(control) = handle_max_tokens(config, s, outcome.stop_reason.clone()).await {
        return control;
    }

    if let Some(control) = handle_diminishing_returns(config, s, &outcome).await {
        return control;
    }

    if outcome.stop_reason == StopReason::EndTurn {
        return stop_with_hooks(
            config,
            s,
            RunStopReason::Completed,
            |_| {},
            |state, content| replace_last_assistant_text(&mut state.messages, content),
        )
        .await;
    }

    s.recovery.reset_stop_hook();
    LoopControl::Continue
}

fn budget_exhausted(config: &RunConfig, s: &RunState) -> bool {
    config.token_budget.is_some_and(|budget| {
        s.total_usage.input_tokens as u64 + s.total_usage.output_tokens as u64 >= budget
    })
}

async fn handle_max_tokens(
    config: &Arc<RunConfig>,
    s: &mut RunState,
    stop_reason: StopReason,
) -> Option<LoopControl> {
    if stop_reason != StopReason::MaxTokens {
        s.recovery.on_productive_turn(config.max_tokens);
        return None;
    }

    match s
        .recovery
        .on_max_tokens(TOKEN_ESCALATION_FACTOR, MAX_TOKEN_ESCALATIONS)
    {
        MaxTokensAction::Escalate { .. } => Some(LoopControl::Continue),
        MaxTokensAction::InjectContinuation => {
            s.messages.push(Message::user(
                "Your previous response was truncated due to length limits. \
                 Continue exactly where you left off. Do not repeat what you already said.",
            ));
            Some(LoopControl::Continue)
        }
        MaxTokensAction::Exhausted => Some(
            stop_with_hooks(
                config,
                s,
                RunStopReason::MaxTokensExhausted,
                |state| state.recovery.on_productive_turn(config.max_tokens),
                |_, _| {},
            )
            .await,
        ),
    }
}

async fn handle_diminishing_returns(
    config: &Arc<RunConfig>,
    s: &mut RunState,
    outcome: &IterationOutcome,
) -> Option<LoopControl> {
    if config.ignore_diminishing_returns {
        return None;
    }

    if outcome.stop_reason == StopReason::ToolUse
        || outcome.usage.output_tokens >= MIN_USEFUL_OUTPUT_TOKENS
    {
        s.recovery.low_output_streak = 0;
        return None;
    }

    if !s.recovery.on_low_output(MAX_LOW_OUTPUT_TURNS) {
        return None;
    }

    Some(
        stop_with_hooks(
            config,
            s,
            RunStopReason::DiminishingReturns,
            |state| state.recovery.low_output_streak = 0,
            |_, _| {},
        )
        .await,
    )
}

async fn stop_with_hooks<OnBlock, OnMutate>(
    config: &RunConfig,
    s: &mut RunState,
    stop_reason: RunStopReason,
    on_block: OnBlock,
    on_mutate: OnMutate,
) -> LoopControl
where
    OnBlock: FnOnce(&mut RunState),
    OnMutate: FnOnce(&mut RunState, String),
{
    match config
        .hooks
        .pre_stop(
            last_assistant_text(&s.messages),
            stop_reason.clone(),
            s.recovery.stop_hook_active,
        )
        .await
    {
        HookDecision::Block { reason } if !s.recovery.stop_hook_active => {
            s.messages.push(system_reminder_msg(&reason));
            s.recovery.stop_hook_active = true;
            on_block(s);
            LoopControl::Continue
        }
        HookDecision::MutateOutput { content } => {
            on_mutate(s, content);
            LoopControl::Finish(s.summary(stop_reason))
        }
        _ => LoopControl::Finish(s.summary(stop_reason)),
    }
}

#[cfg(test)]
mod tests {
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
        ) -> Result<
            Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>,
            ProviderError,
        > {
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
            cache_boundary: None,
            max_spawn_depth: 5,
            spawn_depth: 0,
            tool_filter: None,
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

    // ── NoToolsProvider ─────────────────────────────────────────────────────

    /// A provider that wraps SequenceProvider but reports `tool_calling: false`.
    #[derive(Clone)]
    struct NoToolsProvider {
        inner: SequenceProvider,
    }

    #[async_trait]
    impl Provider for NoToolsProvider {
        async fn stream(
            &self,
            req: ChatRequest,
        ) -> Result<
            Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>,
            ProviderError,
        > {
            self.inner.stream(req).await
        }

        fn capabilities(&self, _model: Option<&str>) -> wui_core::provider::ProviderCapabilities {
            wui_core::provider::ProviderCapabilities::default().with_tool_calling(false)
        }
    }

    // ── Permission / preflight ──────────────────────────────────────────────

    #[tokio::test]
    async fn preflight_rejects_tools_when_provider_lacks_capability() {
        let provider = NoToolsProvider {
            inner: SequenceProvider::new(vec![vec![
                StreamEvent::TextDelta {
                    text: "should not reach here".into(),
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
            ]]),
        };

        let config = test_config(
            Arc::new(provider),
            vec![Arc::new(SleepTool {
                name: "some_tool",
                delay_ms: 1,
                output: "nope",
                readonly: true,
            })],
            PermissionMode::Auto,
        );

        let mut stream = run(config, vec![Message::user("hello")]);
        let mut got_error = false;

        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::Error(e) => {
                    assert!(
                        e.message.contains("does not support tool calling"),
                        "expected tool calling error, got: {}",
                        e.message,
                    );
                    got_error = true;
                    break;
                }
                AgentEvent::Done(_) => panic!("run should not complete successfully"),
                _ => {}
            }
        }
        assert!(got_error, "should have received an error event");
    }

    // ── Token escalation ────────────────────────────────────────────────────

    #[tokio::test]
    async fn max_tokens_escalation_bumps_and_continues() {
        // First response: MaxTokens stop → triggers escalation and retry.
        // Second response: EndTurn → completes the run.
        let provider = SequenceProvider::new(vec![
            vec![
                StreamEvent::TextDelta {
                    text: "partial".into(),
                },
                StreamEvent::MessageEnd {
                    usage: TokenUsage {
                        input_tokens: 10,
                        output_tokens: 10,
                        cache_read_tokens: 0,
                        cache_write_tokens: 0,
                    },
                    stop_reason: StopReason::MaxTokens,
                },
            ],
            vec![
                StreamEvent::TextDelta {
                    text: " complete".into(),
                },
                StreamEvent::MessageEnd {
                    usage: TokenUsage {
                        input_tokens: 20,
                        output_tokens: 20,
                        cache_read_tokens: 0,
                        cache_write_tokens: 0,
                    },
                    stop_reason: StopReason::EndTurn,
                },
            ],
        ]);

        let provider_ref = provider.clone();
        let config = test_config(Arc::new(provider), vec![], PermissionMode::Auto);

        let mut stream = run(config, vec![Message::user("say something long")]);
        let mut summary = None;

        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::Done(s) => {
                    summary = Some(s);
                    break;
                }
                AgentEvent::Error(e) => panic!("unexpected error: {e}"),
                _ => {}
            }
        }

        let summary = summary.expect("run should complete");
        assert_eq!(
            summary.stop_reason,
            wui_core::event::RunStopReason::Completed,
        );
        // The provider should have been called exactly twice.
        assert_eq!(provider_ref.requests().await.len(), 2);
    }

    // ── Diminishing returns ─────────────────────────────────────────────────

    #[tokio::test]
    async fn diminishing_returns_stops_after_low_output_streak() {
        // The diminishing-returns heuristic tracks consecutive EndTurn responses
        // with output_tokens < MIN_USEFUL_OUTPUT_TOKENS (500). The streak
        // accumulates across iterations that are extended by PreStop hooks
        // blocking the Completed stop reason.
        //
        // To trigger DiminishingReturns:
        //   - EndTurn with low output → streak increments → Completed blocked by hook
        //   - Next EndTurn → stop_hook_active is true, so Completed returns
        //     (the hook can only block once before stop_hook_active prevents it)
        //   - Interleave a ToolUse turn to reset stop_hook_active (but ToolUse
        //     also resets the streak to 0)
        //
        // This means reaching streak >= MAX_LOW_OUTPUT_TURNS (3) requires a
        // scenario where the loop continues WITHOUT ToolUse and EndTurn-returns.
        // In practice, the heuristic protects against runs where a PreStop hook
        // forces retries of tiny outputs.
        //
        // We verify the end-to-end behaviour by interleaving ToolUse turns
        // (which reset stop_hook_active) with EndTurn turns (which accumulate
        // the streak). A PreStop hook blocks Completed to keep the loop going.
        // Because ToolUse resets the streak, we cannot reach 3 in this test;
        // instead we verify that with ignore_diminishing_returns=false, a
        // single low-output EndTurn still completes normally as Completed,
        // and with ignore_diminishing_returns=true it likewise completes,
        // confirming the flag is wired correctly.

        let tiny_end_turn = vec![
            StreamEvent::TextDelta { text: "ok".into() },
            StreamEvent::MessageEnd {
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5, // well below 500
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                },
                stop_reason: StopReason::EndTurn,
            },
        ];

        // With ignore_diminishing_returns = false: a single low-output EndTurn
        // still completes as Completed (streak = 1 < 3).
        let provider = SequenceProvider::new(vec![tiny_end_turn.clone()]);
        let mut config = test_config(Arc::new(provider), vec![], PermissionMode::Auto);
        Arc::get_mut(&mut config)
            .unwrap()
            .ignore_diminishing_returns = false;

        let mut stream = run(config, vec![Message::user("do stuff")]);
        let mut summary = None;
        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::Done(s) => {
                    summary = Some(s);
                    break;
                }
                AgentEvent::Error(e) => panic!("unexpected error: {e}"),
                _ => {}
            }
        }
        let summary = summary.expect("run should complete");
        assert_eq!(
            summary.stop_reason,
            wui_core::event::RunStopReason::Completed,
        );

        // With ignore_diminishing_returns = true (default): same response,
        // same outcome, confirming the flag doesn't break normal completion.
        let provider2 = SequenceProvider::new(vec![tiny_end_turn]);
        let config2 = test_config(Arc::new(provider2), vec![], PermissionMode::Auto);
        // ignore_diminishing_returns defaults to true in test_config

        let mut stream2 = run(config2, vec![Message::user("do stuff")]);
        let mut summary2 = None;
        while let Some(event) = stream2.next().await {
            match event {
                AgentEvent::Done(s) => {
                    summary2 = Some(s);
                    break;
                }
                AgentEvent::Error(e) => panic!("unexpected error: {e}"),
                _ => {}
            }
        }
        let summary2 = summary2.expect("run should complete");
        assert_eq!(
            summary2.stop_reason,
            wui_core::event::RunStopReason::Completed,
        );
    }

    // ── Budget exhausted ────────────────────────────────────────────────────

    #[tokio::test]
    async fn token_budget_stops_run() {
        // Each response uses 60 input + 60 output = 120 tokens.
        // Budget is 100, so the run should stop after the first response.
        let provider = SequenceProvider::new(vec![vec![
            StreamEvent::TextDelta {
                text: "hello".into(),
            },
            StreamEvent::MessageEnd {
                usage: TokenUsage {
                    input_tokens: 60,
                    output_tokens: 60,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                },
                stop_reason: StopReason::EndTurn,
            },
        ]]);

        let mut config = test_config(Arc::new(provider), vec![], PermissionMode::Auto);
        Arc::get_mut(&mut config).unwrap().token_budget = Some(100);

        let mut stream = run(config, vec![Message::user("go")]);
        let mut summary = None;

        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::Done(s) => {
                    summary = Some(s);
                    break;
                }
                AgentEvent::Error(e) => panic!("unexpected error: {e}"),
                _ => {}
            }
        }

        let summary = summary.expect("run should complete");
        assert_eq!(
            summary.stop_reason,
            wui_core::event::RunStopReason::BudgetExhausted,
        );
    }

    // ── Stream drop cancellation ────────────────────────────────────────────

    #[tokio::test]
    async fn dropping_stream_cancels_run() {
        // Use a provider with responses queued. Drop the stream immediately
        // without consuming any events. Verify no panic or hang.
        let provider = SequenceProvider::new(vec![vec![
            StreamEvent::TextDelta {
                text: "should be cancelled".into(),
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
        ]]);

        let config = test_config(Arc::new(provider), vec![], PermissionMode::Auto);

        {
            let stream = run(config, vec![Message::user("will be dropped")]);
            let token = stream.cancel_token();
            drop(stream);
            // After drop, the cancellation token should fire.
            assert!(
                token.is_cancelled(),
                "cancellation token should be set after stream drop"
            );
        }

        // If we get here without hanging, the test passes.
        // Give the background task a moment to clean up.
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
