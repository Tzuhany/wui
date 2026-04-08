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
//   9. Run PreComplete hook.
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

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use super::auth::{self, AuthOutcome};

// ── handle_stop! ──────────────────────────────────────────────────────────────
//
// Shared stop-condition logic used by BudgetExhausted, MaxTokensExhausted,
// DiminishingReturns, and Completed.
//
// Each invocation:
//   1. Runs the PreStop hook (unless stop_hook_active is set).
//   2. On Block: injects the reason as a system reminder, sets the flag, and
//      `continue`s the outer loop.
//   3. On non-Block (or when already active): returns Ok(RunSummary{...}).
//
// The Completed variant also handles MutateOutput — pass the optional
// `$on_mutate` expression (evaluated when MutateOutput fires) as a block.
//
// `$extra_on_block` runs extra reset logic (e.g., resetting counters) before
// the `continue`. Pass `{}` when not needed.
macro_rules! handle_stop {
    (
        hooks        = $hooks:expr,
        messages     = $messages:expr,
        iterations   = $iterations:expr,
        total_usage  = $total_usage:expr,
        stop_active  = $stop_hook_active:expr,
        reason       = $reason:expr,
        extra_on_block = $extra_on_block:block,
        on_mutate    = |$mutated:ident| $on_mutate:block $(,)?
    ) => {{
        match $hooks
            .pre_stop(
                last_assistant_text(&$messages),
                $reason,
                $stop_hook_active,
            )
            .await
        {
            HookDecision::Block { reason } if !$stop_hook_active => {
                $messages.push(system_reminder_msg(&reason));
                $stop_hook_active = true;
                $extra_on_block
                continue;
            }
            HookDecision::MutateOutput { content: $mutated } => $on_mutate,
            _ => {}
        }
        return Ok(RunSummary {
            stop_reason: $reason,
            iterations: $iterations,
            usage: $total_usage,
            messages: $messages,
        });
    }};
}

use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tracing::Instrument as _;

use wui_core::event::{AgentError, AgentEvent, RunStopReason, RunSummary, StopReason, TokenUsage};
use wui_core::hook::HookDecision;
use wui_core::message::{ContentBlock, Message, Role};
use wui_core::provider::{ChatRequest, Provider, ProviderError};
use wui_core::types::ToolCallId;

/// Convenience alias for the pinned stream returned by `Provider::stream`.
type ProviderStream = std::pin::Pin<
    Box<dyn futures::Stream<Item = Result<wui_core::event::StreamEvent, ProviderError>> + Send>,
>;

use crate::compress::{CompressResult, CompressStrategy};

use super::checkpoint::{CheckpointStore, RunCheckpoint};
use super::executor::{CompletedTool, PendingTool, ToolExecutor};
use super::hooks::HookRunner;
use super::permission::{PermissionMode, PermissionRules, SessionPermissions};
use super::registry::ToolRegistry;

/// Maximum number of events buffered between the loop and the caller.
///
/// If the caller stops consuming, the loop pauses here. Tune this for your
/// use case — larger values trade memory for smoother streaming under jitter.
const EVENT_CHANNEL_CAPACITY: usize = 256;

// ── Retry Policy ──────────────────────────────────────────────────────────────

/// Exponential back-off for transient provider errors.
///
/// Applied when the provider returns `is_retryable: true` — network hiccups,
/// rate-limit 429s, intermittent 5xx. Non-retryable errors bypass this entirely.
///
/// The default (`RetryPolicy::default()`) retries up to 3 times with 500 ms
/// initial delay doubling on each attempt, capped at 10 s, with jitter enabled.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts. `0` disables retrying.
    pub max_attempts: u32,
    /// Wait before the first retry (milliseconds).
    pub initial_delay_ms: u64,
    /// Multiplier applied to the delay after each failure (exponential back-off).
    /// `2.0` doubles the delay each time: 500 ms → 1 s → 2 s → …, capped at `max_delay_ms`.
    pub multiplier: f64,
    /// Hard cap on the delay between retries (milliseconds).
    pub max_delay_ms: u64,
    /// Add equal jitter to each back-off delay.
    ///
    /// Uses the equal-jitter formula: `delay = exp/2 + rand(0, exp/2)`.
    /// This guarantees at least half the computed delay (no starvation) while
    /// spreading load across the full range (no thundering herd). The jitter
    /// range grows with the exponential delay, so later retries are spread
    /// proportionally further apart than earlier ones.
    /// Enabled by default.
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 500,
            multiplier: 2.0,
            max_delay_ms: 10_000,
            jitter: true,
        }
    }
}

impl RetryPolicy {
    /// Compute the back-off delay for a given attempt (1-indexed).
    ///
    /// Uses `retry_after_ms` from the provider when available (e.g. a 429
    /// `Retry-After` header). Falls back to exponential back-off + optional jitter.
    fn delay_ms(&self, attempt: u32, retry_after_ms: Option<u64>) -> u64 {
        if let Some(ms) = retry_after_ms {
            return ms;
        }
        let exp = (self.initial_delay_ms as f64 * self.multiplier.powi((attempt - 1) as i32))
            .min(self.max_delay_ms as f64) as u64;
        if self.jitter {
            use rand::Rng;
            // Equal jitter: half the delay is guaranteed (prevents starvation),
            // the other half is random (prevents thundering herd). Both halves
            // scale with `exp`, so spread grows proportionally across retries.
            let half = exp / 2;
            half + rand::thread_rng().gen_range(0..=half)
        } else {
            exp
        }
    }
}

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

// ── Run Stream ────────────────────────────────────────────────────────────────

/// The event stream for a running agent.
///
/// Implements `Stream<Item = AgentEvent>`. Dropping the stream cancels the
/// underlying run immediately — no orphaned tasks, no wasted tokens.
///
/// ```rust,ignore
/// let mut stream = agent.stream("Hello").await;
///
/// // Cancel explicitly:
/// stream.cancel();
///
/// // Share the cancel signal with another task:
/// let token = stream.cancel_token();
/// tokio::spawn(async move { /* call token.cancel() when needed */ });
/// ```
#[must_use = "RunStream does nothing unless polled; use .next().await or collect the events"]
pub struct RunStream {
    inner: ReceiverStream<AgentEvent>,
    cancel: CancellationToken,
}

impl std::fmt::Debug for RunStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunStream")
            .field("cancelled", &self.cancel.is_cancelled())
            .finish_non_exhaustive()
    }
}

impl RunStream {
    /// Cancel the run immediately.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    /// Clone the `CancellationToken` for this run.
    ///
    /// Use when you need to cancel the run from a separate task or store
    /// the handle for later (e.g., a cancel button in a UI).
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel.clone()
    }
}

impl futures::Stream for RunStream {
    type Item = AgentEvent;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl Drop for RunStream {
    fn drop(&mut self) {
        // Cancel the spawned task when the stream is abandoned.
        // Without this, the run loop continues burning tokens even though
        // nobody is consuming its output.
        self.cancel.cancel();
    }
}

// ── Run Config ────────────────────────────────────────────────────────────────

/// Static configuration for one run. `Arc`-wrapped so it is cheap to share
/// across the spawned task and helper services.
pub(crate) struct RunConfig {
    pub(crate) provider: Arc<dyn Provider>,
    pub(crate) tools: Arc<ToolRegistry>,
    pub(crate) hooks: Arc<HookRunner>,
    pub(crate) compress: Arc<dyn CompressStrategy>,
    pub(crate) permission: PermissionMode,
    /// Static allow/deny rules evaluated before the permission mode.
    /// Deny rules are hard constraints; allow rules bypass user prompting.
    pub(crate) rules: PermissionRules,
    pub(crate) session_perms: Arc<SessionPermissions>,
    pub(crate) system: String,
    pub(crate) model: Option<String>,
    pub(crate) max_tokens: u32,
    pub(crate) temperature: Option<f32>,
    pub(crate) max_iter: u32,

    /// Back-off policy for transient provider errors.
    pub(crate) retry: RetryPolicy,

    /// Default execution timeout applied to every tool that doesn't declare
    /// its own `Tool::timeout()`. `None` means tools may run indefinitely.
    pub(crate) tool_timeout: Option<Duration>,

    /// Disable the diminishing-returns auto-stop heuristic.
    ///
    /// When `false` (default), the engine stops a run after
    /// `MAX_LOW_OUTPUT_TURNS` consecutive turns with fewer than
    /// `MIN_USEFUL_OUTPUT_TOKENS` output tokens. Set this to `true` for
    /// long-running tasks where many short intermediate steps are expected
    /// before a large final result (e.g. research tasks, file-heavy writes).
    pub(crate) ignore_diminishing_returns: bool,

    /// Hard ceiling on cumulative tokens (input + output) for this run.
    ///
    /// When the total crosses this budget, the run stops immediately with
    /// `RunStopReason::BudgetExhausted`. More predictable than `max_iter`
    /// for cost control: you know exactly how many tokens you'll spend.
    ///
    /// `None` means no budget limit (default).
    pub(crate) token_budget: Option<u64>,

    /// Extended thinking budget (tokens) forwarded to the provider on every
    /// LLM call. `None` = no thinking (provider default).
    pub(crate) thinking_budget: Option<u32>,

    /// Checkpoint store for save/resume. `None` disables checkpointing.
    pub(crate) checkpoint_store: Option<Arc<dyn CheckpointStore>>,
    /// The run ID used to save/load checkpoints.
    pub(crate) checkpoint_run_id: Option<String>,

    /// Optional store for persisting large tool results before truncation.
    pub(crate) result_store: Option<Arc<dyn super::executor::ResultStore>>,
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Run the agent loop and return an event stream.
///
/// Spawns an internal task. The stream yields events until `AgentEvent::Done`
/// or `AgentEvent::Error` is received.
///
/// Dropping the returned `RunStream` cancels the run immediately — safe
/// to drop at any point without leaking the background task. Call
/// `stream.cancel()` or `stream.cancel_token()` for explicit control.
pub(crate) fn run(config: Arc<RunConfig>, messages: Vec<Message>) -> RunStream {
    let cancel = CancellationToken::new();
    let (tx, rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
    tokio::spawn(run_task(config, messages, cancel.clone(), tx));
    RunStream {
        inner: ReceiverStream::new(rx),
        cancel,
    }
}

// ── Internal task ─────────────────────────────────────────────────────────────

async fn run_task(
    config: Arc<RunConfig>,
    messages: Vec<Message>,
    cancel: CancellationToken,
    tx: mpsc::Sender<AgentEvent>,
) {
    let span = tracing::info_span!(
        "agent.run",
        model      = %config.model.as_deref().unwrap_or("(provider-default)"),
        max_iter   = config.max_iter,
        tools      = config.tools.len(),
    );
    let result = run_loop(config.clone(), messages, cancel, &tx)
        .instrument(span)
        .await;
    match result {
        Ok(summary) => {
            config.hooks.notify_turn_end(&summary).await;
            let _ = tx.send(AgentEvent::Done(summary)).await;
        }
        Err(e) => {
            let _ = tx.send(AgentEvent::Error(e)).await;
        }
    }
}

// ── Main loop ─────────────────────────────────────────────────────────────────

/// Mutable state carried across iterations of the run loop.
struct RunState {
    messages: Vec<Message>,
    total_usage: TokenUsage,
    iterations: u32,
    /// Augmented system prompt (base + deferred tool listings).
    system: String,
    /// How many times max_tokens has been bumped after MaxTokens stops.
    token_escalations: u32,
    /// Consecutive turns below MIN_USEFUL_OUTPUT_TOKENS.
    low_output_streak: u32,
    /// Effective max_tokens for the current call; may be escalated mid-run.
    effective_max_tokens: u32,
    /// True when a PreStop hook already blocked — prevents infinite loops.
    stop_hook_active: bool,
    /// Tools injected at runtime by ToolOutput::expose (e.g., via tool_search).
    dynamic_tools: HashMap<String, Arc<dyn wui_core::tool::Tool>>,
}

/// Result of checking max-iterations at the top of each loop iteration.
enum IterGuard {
    /// Below the limit — proceed normally.
    Proceed,
    /// At the limit, hook blocked — the loop should `continue`.
    Blocked,
    /// At the limit, no block — return this summary.
    Stop(RunSummary),
}

impl RunState {
    fn summary(&self, stop_reason: RunStopReason) -> RunSummary {
        RunSummary {
            stop_reason,
            iterations: self.iterations,
            usage: self.total_usage.clone(),
            messages: self.messages.clone(),
        }
    }

    /// Check whether we've hit `max_iter` and consult the PreStop hook.
    async fn check_max_iter(&mut self, config: &RunConfig) -> IterGuard {
        if self.iterations < config.max_iter {
            return IterGuard::Proceed;
        }
        if !self.stop_hook_active {
            if let HookDecision::Block { reason } = config
                .hooks
                .pre_stop(
                    last_assistant_text(&self.messages),
                    RunStopReason::MaxIterations,
                    false,
                )
                .await
            {
                self.messages.push(system_reminder_msg(&reason));
                self.stop_hook_active = true;
                return IterGuard::Blocked;
            }
        }
        IterGuard::Stop(self.summary(RunStopReason::MaxIterations))
    }
}

// ── Per-iteration mutable state ──────────────────────────────────────────────

/// Bundles all mutable state accumulated during a single loop iteration.
/// Extracted phases operate on this struct, keeping `run_loop` slim.
struct IterationCtx {
    pending_inputs: HashMap<ToolCallId, (String, String)>,
    assistant_blocks: Vec<ContentBlock>,
    submission_order: Vec<ToolCallId>,
    completed_map: HashMap<ToolCallId, CompletedTool>,
    text_buf: String,
    thinking_buf: String,
    stop_reason: StopReason,
    usage: TokenUsage,
    pending_auths: Vec<(ToolCallId, String, serde_json::Value)>,
    emission_guard: EmissionGuard,
    auth_injections: Vec<Message>,
}

impl IterationCtx {
    fn new() -> Self {
        Self {
            pending_inputs: HashMap::new(),
            assistant_blocks: Vec::new(),
            submission_order: Vec::new(),
            completed_map: HashMap::new(),
            text_buf: String::new(),
            thinking_buf: String::new(),
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            pending_auths: Vec::new(),
            emission_guard: EmissionGuard::new(),
            auth_injections: Vec::new(),
        }
    }
}

/// Parse all stream events, accumulating tool calls, text, and thinking into
/// the iteration context. Returns `Ok(true)` when `MessageEnd` was received,
/// `Ok(false)` when the stream ended without it, `Err` on fatal errors.
async fn parse_stream(
    stream: &mut (impl futures::Stream<Item = Result<wui_core::event::StreamEvent, ProviderError>>
              + Unpin),
    ctx: &mut IterationCtx,
    active_registry: &ToolRegistry,
    tx: &mpsc::Sender<AgentEvent>,
) -> Result<bool, AgentError> {
    use wui_core::event::StreamEvent::*;

    while let Some(event) = stream.next().await {
        let event = match event {
            Ok(e) => e,
            Err(e) => {
                if e.is_retryable() {
                    continue;
                }
                return Err(AgentError::fatal(e.to_string()));
            }
        };

        match event {
            TextDelta { text } => {
                ctx.text_buf.push_str(&text);
                tx.send(AgentEvent::TextDelta(text)).await.ok();
            }

            ThinkingDelta { text } => {
                ctx.thinking_buf.push_str(&text);
                tx.send(AgentEvent::ThinkingDelta(text)).await.ok();
            }

            ToolUseStart { id, name } => {
                ctx.pending_inputs.insert(id, (name, String::new()));
            }

            ToolInputDelta { id, chunk } => {
                if let Some((_, json)) = ctx.pending_inputs.get_mut(&id) {
                    json.push_str(&chunk);
                }
            }

            ToolUseEnd { id } => {
                let Some((name, json)) = ctx.pending_inputs.remove(&id) else {
                    continue;
                };

                // Record submission order before any early-exit path so the
                // provider always sees a matching ToolResult regardless of
                // what happens next.
                ctx.submission_order.push(id.clone());

                let input: serde_json::Value = match serde_json::from_str(&json) {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!(tool = %name, error = %e, "malformed tool input JSON from provider");
                        ctx.assistant_blocks.push(ContentBlock::ToolUse {
                            id: id.clone(),
                            name: name.clone(),
                            input: serde_json::Value::Null,
                            summary: None,
                        });
                        let denied = CompletedTool::immediate(
                            id.clone(),
                            name,
                            wui_core::tool::ToolOutput::invalid_input(format!(
                                "malformed tool input JSON: {e}"
                            )),
                        );
                        ctx.emission_guard.first_time(&denied.id);
                        emit_tool_event(&denied, tx).await;
                        ctx.completed_map.insert(denied.id.clone(), denied);
                        continue;
                    }
                };

                let tool_summary = active_registry
                    .get(&name)
                    .and_then(|t| t.executor_hints(&input).summary);
                ctx.assistant_blocks.push(ContentBlock::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                    summary: tool_summary,
                });

                // Queue for authorization after MessageEnd.
                ctx.pending_auths.push((id, name, input));
            }

            MessageEnd {
                usage: u,
                stop_reason: sr,
            } => {
                ctx.usage = u;
                ctx.stop_reason = sr;
                return Ok(true);
            }

            Error { message, retryable } => {
                return Err(AgentError {
                    message,
                    retryable,
                    detail: None,
                    permission_denied: false,
                });
            }
        }
    }
    Ok(false)
}

/// Authorize all collected tools concurrently and dispatch approved ones to
/// the executor. Denied tools are recorded immediately with their events.
async fn authorize_and_dispatch(
    ctx: &mut IterationCtx,
    config: &Arc<RunConfig>,
    active_registry: &Arc<ToolRegistry>,
    executor: &mut ToolExecutor,
    messages: &[Message],
    tx: &mpsc::Sender<AgentEvent>,
) {
    let mut auth_tasks: tokio::task::JoinSet<AuthOutcome> = tokio::task::JoinSet::new();

    for (id, name, input) in std::mem::take(&mut ctx.pending_auths) {
        let config_c = Arc::clone(config);
        let registry_c = Arc::clone(active_registry);
        let tx_c = tx.clone();
        auth_tasks.spawn(async move {
            let (result, injections) = auth::authorize_tool(
                &config_c,
                &registry_c,
                id.clone(),
                name.clone(),
                input,
                &tx_c,
            )
            .await;
            match result {
                Ok(effective_input) => AuthOutcome::Allowed {
                    id,
                    name,
                    input: effective_input,
                    injections,
                },
                Err(denied) => AuthOutcome::Denied {
                    tool: denied,
                    injections,
                },
            }
        });
    }

    while let Some(outcome) = auth_tasks.join_next().await {
        let outcome = match outcome {
            Ok(o) => o,
            Err(e) => {
                tracing::error!(error = %e, "auth task panicked — skipping tool");
                continue;
            }
        };
        match outcome {
            AuthOutcome::Allowed {
                id,
                name,
                input,
                injections,
            } => {
                ctx.auth_injections.extend(injections);
                let history_snap = Arc::from(messages);
                tracing::debug!(tool = %name, "tool dispatched");
                tx.send(AgentEvent::ToolStart {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                })
                .await
                .ok();
                executor.submit(PendingTool {
                    id,
                    name,
                    input,
                    messages: history_snap,
                });
            }
            AuthOutcome::Denied { tool, injections } => {
                ctx.auth_injections.extend(injections);
                ctx.emission_guard.first_time(&tool.id);
                emit_tool_event(&tool, tx).await;
                ctx.completed_map.insert(tool.id.clone(), tool);
            }
        }
    }
}

/// Collect remaining executor results and run post-tool hooks.
/// Emits tool events in submission order, after hook mutations are applied.
async fn run_post_hooks(
    ctx: &mut IterationCtx,
    executor: ToolExecutor,
    config: &RunConfig,
    messages: &mut Vec<Message>,
    tx: &mpsc::Sender<AgentEvent>,
) {
    for done in executor.collect_remaining().await {
        ctx.completed_map.insert(done.id.clone(), done);
    }

    for id in &ctx.submission_order {
        let Some(done) = ctx.completed_map.get(id) else {
            continue;
        };
        let tool_name = done.name.clone();
        let is_error = done.output.is_error();

        let decision = if is_error {
            let input = ctx.assistant_blocks.iter().find_map(|b| {
                if let ContentBlock::ToolUse { id: bid, input, .. } = b {
                    (bid == id).then_some(input)
                } else {
                    None
                }
            });
            let input = input.unwrap_or(&serde_json::Value::Null);
            config
                .hooks
                .post_tool_failure(&tool_name, input, &done.output)
                .await
        } else {
            config.hooks.post_tool_use(&tool_name, &done.output).await
        };

        match decision {
            HookDecision::Block { reason } => {
                tracing::debug!(tool = %tool_name, %reason, "post-tool hook blocked output");
                messages.push(system_reminder_msg(&format!(
                    "The output of tool '{tool_name}' was blocked by policy: {reason}. \
                     Do not use this output.",
                )));
            }
            HookDecision::MutateOutput { content } => {
                tracing::debug!(tool = %tool_name, "post-tool hook mutated output");
                if let Some(done_mut) = ctx.completed_map.get_mut(id) {
                    done_mut.output.content = content;
                }
            }
            _ => {}
        }

        if ctx.emission_guard.first_time(id) {
            if let Some(done) = ctx.completed_map.get(id) {
                emit_tool_event(done, tx).await;
            }
        }
    }
}

/// Build the assistant message and tool results, append them to history,
/// update usage/iterations, and save a checkpoint if applicable.
async fn assemble_history(
    ctx: IterationCtx,
    s: &mut RunState,
    config: &RunConfig,
    tx: &mpsc::Sender<AgentEvent>,
) {
    // Apply auth injections.
    s.messages.extend(ctx.auth_injections);

    // Build assistant message: thinking, text, then tool calls.
    let mut blocks = ctx.assistant_blocks;
    if !ctx.text_buf.is_empty() {
        blocks.insert(0, ContentBlock::Text { text: ctx.text_buf });
    }
    if !ctx.thinking_buf.is_empty() {
        blocks.insert(
            0,
            ContentBlock::Thinking {
                text: ctx.thinking_buf,
            },
        );
    }
    if !blocks.is_empty() {
        s.messages.push(Message::assistant(blocks));
    }

    // Append tool results in submission order.
    if !ctx.submission_order.is_empty() {
        let mut completed_map = ctx.completed_map;
        let done_tools: Vec<CompletedTool> = ctx
            .submission_order
            .iter()
            .filter_map(|id| completed_map.remove(id))
            .collect();

        for done in &done_tools {
            for tool in &done.output.expose_tools {
                s.dynamic_tools
                    .insert(tool.name().to_string(), Arc::clone(tool));
            }
        }

        for done in &done_tools {
            for artifact in &done.output.artifacts {
                tx.send(AgentEvent::Artifact {
                    tool_id: done.id.clone(),
                    tool_name: done.name.clone(),
                    artifact: artifact.clone(),
                })
                .await
                .ok();
            }
        }

        let result_blocks: Vec<ContentBlock> = done_tools
            .iter()
            .map(|done| ContentBlock::ToolResult {
                tool_use_id: done.id.clone(),
                content: done.output.content.clone(),
                is_error: done.output.is_error(),
            })
            .collect();

        if !result_blocks.is_empty() {
            s.messages.push(Message::with_id(
                uuid::Uuid::new_v4().to_string(),
                Role::User,
                result_blocks,
            ));
        }

        for done in &done_tools {
            for injection in &done.output.injections {
                s.messages.push(system_reminder_msg(&injection.text));
            }
        }
    }

    tracing::debug!(
        stop_reason    = ?ctx.stop_reason,
        input_tokens   = ctx.usage.input_tokens,
        output_tokens  = ctx.usage.output_tokens,
        cache_read     = ctx.usage.cache_read_tokens,
        cache_write    = ctx.usage.cache_write_tokens,
        "llm call complete"
    );

    s.total_usage += ctx.usage;
    s.iterations += 1;

    // Checkpoint save (only after tool-use iterations).
    if !ctx.submission_order.is_empty() {
        if let (Some(store), Some(run_id)) = (&config.checkpoint_store, &config.checkpoint_run_id) {
            let cp = RunCheckpoint {
                run_id: run_id.clone(),
                messages: s.messages.clone(),
                iteration: s.iterations,
                total_usage: s.total_usage.clone(),
            };
            if let Err(e) = store.save(run_id, &cp).await {
                tracing::warn!(run_id, error = %e, "checkpoint save failed");
            }
        }
    }
}

// ── Initialization helpers ───────────────────────────────────────────────────

/// Create the initial `RunState` and restore any saved checkpoint.
async fn init_run_state(config: &RunConfig, messages: Vec<Message>) -> RunState {
    let mut s = RunState {
        system: augment_system(&config.system, &config.tools),
        total_usage: TokenUsage::default(),
        iterations: 0,
        token_escalations: 0,
        low_output_streak: 0,
        effective_max_tokens: config.max_tokens,
        stop_hook_active: false,
        dynamic_tools: HashMap::new(),
        messages,
    };

    if let (Some(store), Some(run_id)) = (&config.checkpoint_store, &config.checkpoint_run_id) {
        match store.load(run_id).await {
            Ok(Some(cp)) => {
                tracing::info!(
                    run_id,
                    iteration = cp.iteration,
                    "checkpoint found — resuming"
                );
                s.messages = cp.messages;
                s.iterations = cp.iteration;
                s.total_usage = cp.total_usage;
            }
            Ok(None) => tracing::debug!(run_id, "no checkpoint found — starting fresh"),
            Err(e) => tracing::warn!(run_id, error = %e, "checkpoint load failed — starting fresh"),
        }
    }
    s
}

// ── Main loop ────────────────────────────────────────────────────────────────

async fn run_loop(
    config: Arc<RunConfig>,
    messages: Vec<Message>,
    cancel: CancellationToken,
    tx: &mpsc::Sender<AgentEvent>,
) -> Result<RunSummary, AgentError> {
    let mut s = init_run_state(&config, messages).await;

    loop {
        if cancel.is_cancelled() {
            return Ok(s.summary(RunStopReason::Cancelled));
        }

        match s.check_max_iter(&config).await {
            IterGuard::Stop(summary) => return Ok(summary),
            IterGuard::Blocked => continue,
            IterGuard::Proceed => {}
        }

        config.hooks.notify_turn_start(&s.messages).await;
        maybe_compress(&config, &mut s.messages, tx).await;
        if config.compress.is_critically_full(&s.messages) {
            return Ok(s.summary(RunStopReason::ContextOverflow));
        }
        let registry: Arc<ToolRegistry> = if s.dynamic_tools.is_empty() {
            config.tools.clone()
        } else {
            Arc::new(config.tools.with_dynamic(&s.dynamic_tools))
        };
        let req = ChatRequest {
            model: config.model.clone(),
            max_tokens: s.effective_max_tokens,
            temperature: config.temperature,
            system: s.system.clone(),
            messages: s.messages.clone(),
            tools: registry.tool_defs(),
            thinking_budget: config.thinking_budget,
        };
        let stream = match call_with_retry(&config, &req, tx).await {
            Ok(stream) => stream,
            Err(e) if is_prompt_too_long(&e) => {
                if emergency_compress(&config, &mut s.messages, tx).await {
                    continue;
                }
                return Err(e);
            }
            Err(e) => return Err(e),
        };

        let mut executor = ToolExecutor::new(
            registry.clone(),
            cancel.clone(),
            tx.clone(),
            config.tool_timeout,
            config.result_store.clone(),
        );
        let mut ctx = IterationCtx::new();
        futures::pin_mut!(stream);

        let got_message_end = parse_stream(&mut stream, &mut ctx, &registry, tx).await?;

        // Stream-drop recovery (needs continue/return).
        if !got_message_end {
            if ctx.submission_order.is_empty() {
                tracing::warn!(s.iterations, "stream ended without MessageEnd — retrying");
                continue;
            }
            return Err(AgentError::retryable(
                "provider stream dropped mid-response with tools in flight",
            ));
        }

        authorize_and_dispatch(&mut ctx, &config, &registry, &mut executor, &s.messages, tx).await;
        let stop_reason = ctx.stop_reason.clone();
        let usage = ctx.usage.clone();
        run_post_hooks(&mut ctx, executor, &config, &mut s.messages, tx).await;
        assemble_history(ctx, &mut s, &config, tx).await;

        // ── Stop evaluation (handle_stop! needs continue/return) ─────────

        if let Some(budget) = config.token_budget {
            if s.total_usage.input_tokens as u64 + s.total_usage.output_tokens as u64 >= budget {
                handle_stop! {
                    hooks = config.hooks, messages = s.messages,
                    iterations = s.iterations, total_usage = s.total_usage,
                    stop_active = s.stop_hook_active,
                    reason = RunStopReason::BudgetExhausted,
                    extra_on_block = {}, on_mutate = |_content| {},
                };
            }
        }

        if stop_reason == StopReason::MaxTokens {
            s.token_escalations += 1;
            if s.token_escalations == 1 {
                s.effective_max_tokens =
                    (s.effective_max_tokens * TOKEN_ESCALATION_FACTOR).min(131_072);
                continue;
            } else if s.token_escalations <= MAX_TOKEN_ESCALATIONS {
                s.messages.push(Message::user(
                    "Your previous response was truncated due to length limits. \
                     Continue exactly where you left off. Do not repeat what you already said.",
                ));
                continue;
            } else {
                handle_stop! {
                    hooks = config.hooks, messages = s.messages,
                    iterations = s.iterations, total_usage = s.total_usage,
                    stop_active = s.stop_hook_active,
                    reason = RunStopReason::MaxTokensExhausted,
                    extra_on_block = {
                        s.token_escalations = 0;
                        s.effective_max_tokens = config.max_tokens;
                    },
                    on_mutate = |_content| {},
                };
            }
        }

        s.token_escalations = 0;
        s.effective_max_tokens = config.max_tokens;

        if !config.ignore_diminishing_returns {
            if stop_reason == StopReason::ToolUse || usage.output_tokens >= MIN_USEFUL_OUTPUT_TOKENS
            {
                s.low_output_streak = 0;
            } else {
                s.low_output_streak += 1;
                if s.low_output_streak >= MAX_LOW_OUTPUT_TURNS {
                    handle_stop! {
                        hooks = config.hooks, messages = s.messages,
                        iterations = s.iterations, total_usage = s.total_usage,
                        stop_active = s.stop_hook_active,
                        reason = RunStopReason::DiminishingReturns,
                        extra_on_block = { s.low_output_streak = 0; },
                        on_mutate = |_content| {},
                    };
                }
            }
        }

        if stop_reason == StopReason::EndTurn {
            handle_stop! {
                hooks = config.hooks, messages = s.messages,
                iterations = s.iterations, total_usage = s.total_usage,
                stop_active = s.stop_hook_active,
                reason = RunStopReason::Completed,
                extra_on_block = {},
                on_mutate = |content| {
                    replace_last_assistant_text(&mut s.messages, content);
                },
            };
        }

        s.stop_hook_active = false;
    }
}

// ── Compression helpers ──────────────────────────────────────────────────────

/// Run context compression if pressure exceeds the threshold.
///
/// Fires the PreCompact hook first (may inject preservation context),
/// then runs the compression pipeline. Mutates `messages` in place.
async fn maybe_compress(
    config: &RunConfig,
    messages: &mut Vec<Message>,
    tx: &mpsc::Sender<AgentEvent>,
) {
    if config.compress.pressure(messages) >= config.compress.threshold() {
        if let HookDecision::Block { reason } = config.hooks.pre_compact(messages).await {
            tracing::debug!("pre_compact hook injecting preservation context");
            messages.push(system_reminder_msg(&reason));
        }
    }

    if config.compress.pressure(messages) < config.compress.threshold() {
        return;
    }

    let pressure_before = config.compress.pressure(messages);
    let result = config
        .compress
        .compress(
            messages.clone(),
            config.provider.clone(),
            config.model.as_deref(),
        )
        .await;
    match result {
        Ok(CompressResult {
            method: Some(method),
            freed,
            messages: new_msgs,
        }) => {
            let pressure_after = config.compress.pressure(&new_msgs);
            tracing::debug!(?method, freed, %pressure_before, %pressure_after, "context compressed");
            *messages = new_msgs;
            if method == wui_core::event::CompressMethod::L3Failed {
                tx.send(AgentEvent::CompressFallback { freed }).await.ok();
            }
            tx.send(AgentEvent::Compressed {
                method,
                freed,
                pressure_before,
                pressure_after,
            })
            .await
            .ok();
        }
        Ok(_) => {}
        Err(e) => {
            tracing::warn!(error = %e, "compression failed, continuing without");
        }
    }
}

/// Attempt emergency compression after a prompt-too-long rejection.
///
/// Returns `true` if compression succeeded and the caller should retry.
async fn emergency_compress(
    config: &RunConfig,
    messages: &mut Vec<Message>,
    tx: &mpsc::Sender<AgentEvent>,
) -> bool {
    tracing::warn!("provider rejected prompt as too long — attempting emergency compression");
    let pressure_before = config.compress.pressure(messages);
    let result = config
        .compress
        .compress(
            messages.clone(),
            config.provider.clone(),
            config.model.as_deref(),
        )
        .await;
    match result {
        Ok(CompressResult {
            method: Some(method),
            freed,
            messages: new_msgs,
        }) => {
            let pressure_after = config.compress.pressure(&new_msgs);
            tracing::info!(?method, freed, %pressure_before, %pressure_after, "emergency compression succeeded");
            *messages = new_msgs;
            tx.send(AgentEvent::Compressed {
                method,
                freed,
                pressure_before,
                pressure_after,
            })
            .await
            .ok();
            true
        }
        _ => {
            tracing::error!("emergency compression failed or had no effect");
            false
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Call the provider, retrying transient errors with exponential back-off.
async fn call_with_retry(
    config: &RunConfig,
    req: &ChatRequest,
    tx: &mpsc::Sender<AgentEvent>,
) -> Result<ProviderStream, AgentError> {
    let mut attempt = 0u32;
    loop {
        match config.provider.stream(req.clone()).await {
            Ok(stream) => return Ok(stream),

            Err(e) if e.is_retryable() && attempt < config.retry.max_attempts => {
                attempt += 1;

                // Use the provider's Retry-After hint for rate-limit errors;
                // fall back to the policy's formula for everything else.
                let retry_after_ms = if let ProviderError::RateLimit { retry_after_ms } = &e {
                    Some(*retry_after_ms)
                } else {
                    None
                };
                let delay_ms = config.retry.delay_ms(attempt, retry_after_ms);

                tracing::warn!(
                    attempt, max = config.retry.max_attempts,
                    error = %e, delay_ms, "provider error — retrying"
                );
                tx.send(AgentEvent::Retrying {
                    attempt,
                    delay_ms,
                    reason: e.to_string(),
                })
                .await
                .ok();
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }

            Err(e) => return Err(AgentError::fatal(e.to_string())),
        }
    }
}

/// Emit the appropriate AgentEvent for a completed tool.
///
/// Called in two distinct contexts:
/// - **Instant failures** (malformed input, denied tools): emitted immediately,
///   before hooks run, because these outcomes cannot be mutated.
/// - **Successful / failed executor tools**: emitted *after* post-tool hooks,
///   in submission order, so that `MutateOutput` decisions are reflected in
///   the event the caller receives.
async fn emit_tool_event(done: &CompletedTool, tx: &mpsc::Sender<AgentEvent>) {
    let event = if let Some(kind) = &done.output.failure {
        tracing::debug!(tool = %done.name, kind = ?kind, ms = done.ms, "tool failed");
        AgentEvent::ToolError {
            id: done.id.clone(),
            name: done.name.clone(),
            error: done.output.content.clone(),
            kind: kind.clone(),
            ms: done.ms,
        }
    } else {
        tracing::debug!(tool = %done.name, ms = done.ms, "tool succeeded");
        AgentEvent::ToolDone {
            id: done.id.clone(),
            name: done.name.clone(),
            output: done.output.content.clone(),
            ms: done.ms,
            attempts: done.attempts,
            structured: done.output.structured.clone(),
        }
    };
    tx.send(event).await.ok();
}

// ── Run-loop helpers ─────────────────────────────────────────────────────────

/// Guards against double-emission of tool events.
struct EmissionGuard {
    emitted: std::collections::HashSet<String>,
}

impl EmissionGuard {
    fn new() -> Self {
        Self {
            emitted: std::collections::HashSet::new(),
        }
    }

    /// Returns `true` if this is the first time — caller should emit.
    fn first_time(&mut self, id: &str) -> bool {
        self.emitted.insert(id.to_owned())
    }
}

/// Build a system-reminder `Message` from plain text.
fn system_reminder_msg(text: &str) -> Message {
    Message {
        id: uuid::Uuid::new_v4().to_string(),
        role: Role::System,
        content: vec![ContentBlock::Text {
            text: wui_core::fmt::system_reminder(text),
        }],
    }
}

/// Extract the text content of the most recent assistant message.
fn last_assistant_text(messages: &[Message]) -> &str {
    messages
        .iter()
        .rev()
        .find_map(|m| {
            if m.role != Role::Assistant {
                return None;
            }
            m.content.iter().find_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
        })
        .unwrap_or("")
}

/// Replace the text content of the most recent assistant message.
fn replace_last_assistant_text(messages: &mut [Message], new_text: String) {
    for msg in messages.iter_mut().rev() {
        if msg.role != Role::Assistant {
            continue;
        }
        for block in msg.content.iter_mut() {
            if let ContentBlock::Text { text } = block {
                *text = new_text;
                return;
            }
        }
    }
}

/// Append a deferred-tools listing to the system prompt when needed.
fn augment_system(base: &str, registry: &ToolRegistry) -> String {
    let deferred = registry.deferred_entries();
    if deferred.is_empty() {
        return base.to_string();
    }

    let listing = deferred
        .iter()
        .map(|e| format!("- **{}**: {}", e.name, e.description))
        .collect::<Vec<_>>()
        .join("\n");

    let section = format!(
        "## Additional tools\n\
        These tools are available but require loading. \
        Call `ToolSearch` with the tool name or a keyword before using them:\n\n\
        {listing}"
    );

    format!("{base}\n\n{}", wui_core::fmt::system_reminder(&section))
}

/// Detect whether an `AgentError` indicates a prompt-too-long rejection.
fn is_prompt_too_long(e: &AgentError) -> bool {
    let msg = e.message.to_lowercase();
    msg.contains("prompt is too long")
        || msg.contains("too long")
        || msg.contains("maximum context length")
        || msg.contains("context_length_exceeded")
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
            json!({
                "type": "object",
                "properties": {},
            })
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

        // Events are emitted in submission order (after hooks), not completion order.
        // slow_tool was submitted first, so it appears first even though fast_tool finishes first.
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

        let requests = provider_ref.requests().await;
        assert_eq!(requests.len(), 2, "provider should be called twice");
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
        let mut saw_tool_start = false;
        let mut saw_tool_done = false;
        let mut done_summary = None;

        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::Control(handle) => {
                    saw_control = true;
                    handle.approve();
                }
                AgentEvent::ToolStart { name, .. } => {
                    assert_eq!(name, "needs_approval");
                    saw_tool_start = true;
                }
                AgentEvent::ToolDone { name, output, .. } => {
                    assert_eq!(name, "needs_approval");
                    assert_eq!(output, "approved tool ran");
                    saw_tool_done = true;
                }
                AgentEvent::Done(summary) => {
                    done_summary = Some(summary);
                    break;
                }
                AgentEvent::Error(e) => panic!("unexpected error: {e}"),
                _ => {}
            }
        }

        assert!(saw_control, "ask mode should emit a control request");
        assert!(saw_tool_start, "tool should start after approval");
        assert!(saw_tool_done, "tool should complete after approval");

        let summary = done_summary.expect("run should complete");
        assert!(summary.messages.iter().any(|msg| {
            msg.role == Role::System
                && msg.content.iter().any(|block| match block {
                    ContentBlock::Text { text } => text.contains("approved your request"),
                    _ => false,
                })
        }));

        let requests = provider_ref.requests().await;
        assert_eq!(
            requests.len(),
            2,
            "provider should resume for a second turn after approval"
        );
    }
}
