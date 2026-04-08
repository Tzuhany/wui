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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tracing::Instrument as _;

use wui_core::event::{
    AgentError, AgentEvent, ControlDecision, ControlHandle, ControlKind, ControlRequest,
    ControlResponse, RunStopReason, RunSummary, StopReason, TokenUsage,
};
use wui_core::hook::HookDecision;
use wui_core::message::{ContentBlock, Message, Role};
use wui_core::provider::{ChatRequest, Provider, ProviderError};

/// Convenience alias for the pinned stream returned by `Provider::stream`.
type ProviderStream = std::pin::Pin<
    Box<dyn futures::Stream<Item = Result<wui_core::event::StreamEvent, ProviderError>> + Send>,
>;

use crate::compress::{CompressResult, CompressStrategy};

use super::checkpoint::{CheckpointStore, RunCheckpoint};
use super::executor::{CompletedTool, PendingTool, ToolExecutor};
use super::hooks::HookRunner;
use super::permission::{
    self, PermissionMode, PermissionOutcome, PermissionRules, SessionPermissions,
};
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
    let result = run_loop(config, messages, cancel, &tx)
        .instrument(span)
        .await;
    match result {
        Ok(summary) => {
            let _ = tx.send(AgentEvent::Done(summary)).await;
        }
        Err(e) => {
            let _ = tx.send(AgentEvent::Error(e)).await;
        }
    }
}

// ── Main loop ─────────────────────────────────────────────────────────────────

async fn run_loop(
    config: Arc<RunConfig>,
    mut messages: Vec<Message>,
    cancel: CancellationToken,
    tx: &mpsc::Sender<AgentEvent>,
) -> Result<RunSummary, AgentError> {
    // Augment the system prompt with deferred tool listings once, before the loop.
    // This tells the LLM which additional tools exist so it can call ToolSearch.
    let system = augment_system(&config.system, &config.tools);

    let mut total_usage = TokenUsage::default();
    let mut iterations = 0u32;

    // ── Checkpoint restore ────────────────────────────────────────────────────
    // If a checkpoint store and run ID are configured, attempt to restore from
    // a previously saved state. This lets a run resume mid-task after a crash,
    // process restart, or deliberate interruption.
    if let (Some(store), Some(run_id)) = (&config.checkpoint_store, &config.checkpoint_run_id) {
        match store.load(run_id).await {
            Ok(Some(cp)) => {
                tracing::info!(
                    run_id,
                    iteration = cp.iteration,
                    "checkpoint found — resuming from saved state"
                );
                messages = cp.messages;
                iterations = cp.iteration;
                total_usage = cp.total_usage;
            }
            Ok(None) => {
                tracing::debug!(run_id, "no checkpoint found — starting fresh");
            }
            Err(e) => {
                tracing::warn!(run_id, error = %e, "checkpoint load failed — starting fresh");
            }
        }
    }
    // ── Error-recovery state ──────────────────────────────────────────────────
    // Carried across iterations; reset only on non-error turns.
    let mut token_escalations = 0u32; // how many times we've bumped max_tokens
    let mut low_output_streak = 0u32; // consecutive turns below MIN_USEFUL_OUTPUT_TOKENS
                                      // The effective max_tokens for the current call; may be escalated mid-run.
    let mut effective_max_tokens = config.max_tokens;
    // ── PreStop hook state ────────────────────────────────────────────────────
    // Tracks whether the PreStop hook already blocked the current stop attempt.
    // When true, Block decisions are ignored (stop proceeds) to prevent infinite
    // loops. Reset to false after any productive tool-use iteration.
    let mut stop_hook_active = false;
    // ── Dynamic tools ─────────────────────────────────────────────────────────
    // Tools injected at runtime by ToolOutput::expose (e.g., via tool_search).
    // Merged with the static registry each iteration.
    let mut dynamic_tools: HashMap<String, Arc<dyn wui_core::tool::Tool>> = HashMap::new();

    loop {
        if cancel.is_cancelled() {
            tracing::info!(iterations, "agent run cancelled");
            return Ok(RunSummary {
                stop_reason: RunStopReason::Cancelled,
                iterations,
                usage: total_usage,
                messages,
            });
        }

        if iterations >= config.max_iter {
            tracing::warn!(
                iterations,
                max_iter = config.max_iter,
                "agent hit max iterations"
            );
            // When stop_hook_active is true the hook already blocked once —
            // stop unconditionally to prevent an infinite loop.
            if !stop_hook_active {
                match config
                    .hooks
                    .pre_stop(
                        last_assistant_text(&messages),
                        RunStopReason::MaxIterations,
                        false,
                    )
                    .await
                {
                    HookDecision::Block { reason } => {
                        tracing::debug!(
                            "pre_stop hook blocked MaxIterations — granting one extra iteration"
                        );
                        messages.push(system_reminder_msg(&reason));
                        stop_hook_active = true;
                        // fall through: run one more LLM iteration
                    }
                    _ => {
                        return Ok(RunSummary {
                            stop_reason: RunStopReason::MaxIterations,
                            iterations,
                            usage: total_usage,
                            messages,
                        })
                    }
                }
            } else {
                return Ok(RunSummary {
                    stop_reason: RunStopReason::MaxIterations,
                    iterations,
                    usage: total_usage,
                    messages,
                });
            }
        }

        tracing::debug!(
            iteration = iterations,
            messages = messages.len(),
            "iteration start"
        );

        // ── Context compression ────────────────────────────────────────────────
        //
        // Only fire PreCompact when the context is actually approaching the
        // compression threshold — not on every iteration. This avoids unnecessary
        // async overhead on short conversations.
        //
        // A Block decision injects context the hook wants preserved. Because it is
        // appended last, the injection lands in the "recent" tail that survives
        // L2/L3 compression, so the summariser always sees it.
        if config.compress.pressure(&messages) >= config.compress.threshold() {
            if let HookDecision::Block { reason } = config.hooks.pre_compact(&messages).await {
                tracing::debug!("pre_compact hook injecting preservation context");
                messages.push(system_reminder_msg(&reason));
            }
        }

        {
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
                    tracing::debug!(method = ?method, freed_tokens = freed, "context compressed");
                    messages = new_msgs;
                    tx.send(AgentEvent::Compressed { method, freed }).await.ok();
                }
                Ok(_) => {} // no compression needed
                Err(e) => {
                    tracing::warn!(error = %e, "compression failed, continuing without");
                }
            }
        }

        // Even after compression, check whether the context is still above the
        // hard ceiling. If we cannot make room for a new LLM call, stop gracefully.
        if config.compress.is_critically_full(&messages) {
            tracing::warn!(iterations, "context overflow: cannot relieve pressure");
            return Ok(RunSummary {
                stop_reason: RunStopReason::ContextOverflow,
                iterations,
                usage: total_usage,
                messages,
            });
        }

        // Merge static registry with dynamic tools discovered this run.
        let active_registry = if dynamic_tools.is_empty() {
            config.tools.clone()
        } else {
            Arc::new(config.tools.with_dynamic(&dynamic_tools))
        };

        let req = ChatRequest {
            model: config.model.clone(),
            max_tokens: effective_max_tokens,
            temperature: config.temperature,
            system: system.clone(),
            messages: messages.clone(),
            tools: active_registry.tool_defs(),
            thinking_budget: config.thinking_budget,
        };

        // ── Call provider with retry ───────────────────────────────────────────
        let stream = call_with_retry(&config, &req, tx).await?;

        let mut executor = ToolExecutor::new(
            active_registry.clone(),
            cancel.clone(),
            tx.clone(),
            config.tool_timeout,
        );

        // Accumulate tool input JSON chunks keyed by tool_use_id.
        let mut pending_inputs: HashMap<String, (String, String)> = HashMap::new();
        let mut assistant_blocks: Vec<ContentBlock> = Vec::new();
        // Track submission order for ordered history reconstruction.
        let mut submission_order: Vec<String> = Vec::new();
        let mut completed_map: HashMap<String, CompletedTool> = HashMap::new();
        let mut text_buf = String::new();
        let mut thinking_buf = String::new();
        let mut stop_reason = StopReason::EndTurn;
        let mut usage = TokenUsage::default();
        let mut got_message_end = false;

        futures::pin_mut!(stream);

        // Track tools whose events were already emitted (instant failures and
        // denied tools).  All other completions are held until after post-tool
        // hooks run so that MutateOutput is reflected in the emitted event.
        let mut pre_emitted: HashSet<String> = HashSet::new();
        // Validated tool calls collected during streaming. Authorization and
        // execution happen after MessageEnd — once the LLM has finished speaking
        // — so HITL prompts are never shown mid-stream.
        let mut pending_auths: Vec<(String, String, serde_json::Value)> = Vec::new();
        let mut auth_injections: Vec<Message> = Vec::new();

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
                    text_buf.push_str(&text);
                    tx.send(AgentEvent::TextDelta(text)).await.ok();
                }

                ThinkingDelta { text } => {
                    thinking_buf.push_str(&text);
                    tx.send(AgentEvent::ThinkingDelta(text)).await.ok();
                }

                ToolUseStart { id, name } => {
                    pending_inputs.insert(id, (name, String::new()));
                }

                ToolInputDelta { id, chunk } => {
                    if let Some((_, json)) = pending_inputs.get_mut(&id) {
                        json.push_str(&chunk);
                    }
                }

                ToolUseEnd { id } => {
                    let Some((name, json)) = pending_inputs.remove(&id) else {
                        continue;
                    };

                    // Record submission order before any early-exit path so the
                    // provider always sees a matching ToolResult regardless of
                    // what happens next.
                    submission_order.push(id.clone());

                    // Malformed JSON is a provider/protocol error — surface it
                    // explicitly rather than silently degrading to an empty object.
                    // The silent `{}` path would let a protocol bug masquerade as
                    // a valid (but empty) tool call, scattering the corruption
                    // across subsequent reasoning turns.
                    let input: serde_json::Value = match serde_json::from_str(&json) {
                        Ok(v) => v,
                        Err(e) => {
                            tracing::warn!(tool = %name, error = %e, "malformed tool input JSON from provider");
                            assistant_blocks.push(ContentBlock::ToolUse {
                                id: id.clone(),
                                name: name.clone(),
                                input: serde_json::Value::Null,
                                summary: None,
                            });
                            let denied = instant_failure(
                                id.clone(),
                                name,
                                wui_core::tool::ToolOutput::invalid_input(format!(
                                    "malformed tool input JSON: {e}"
                                )),
                            );
                            pre_emitted.insert(denied.id.clone());
                            emit_tool_event(&denied, tx).await;
                            completed_map.insert(denied.id.clone(), denied);
                            continue;
                        }
                    };

                    // The Anthropic API (and most providers) require every
                    // ToolUse in the assistant message to have a matching
                    // ToolResult in the following user message — even when
                    // the tool is denied, blocked, or fails validation.
                    // Add the ToolUse block NOW, before any deny paths.
                    //
                    // Compute the tool summary at submission time while we
                    // still have access to the registry and the input value.
                    let tool_summary = active_registry
                        .get(&name)
                        .and_then(|t| t.executor_hints(&input).summary);
                    assistant_blocks.push(ContentBlock::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                        summary: tool_summary,
                    });

                    // Queue for authorization after MessageEnd.
                    pending_auths.push((id, name, input));
                }

                MessageEnd {
                    usage: u,
                    stop_reason: sr,
                } => {
                    usage = u;
                    stop_reason = sr;
                    got_message_end = true;
                    break;
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

        // ── Stream-drop recovery ───────────────────────────────────────────────
        //
        // If we exited without a MessageEnd (network drop mid-stream), we have
        // two cases:
        //
        //   Clean (no tools seen yet): no state is locked in. Retry this
        //   iteration transparently — `continue` restarts from `call_with_retry`.
        //
        //   Dirty (ToolUse blocks already built): we've committed ToolUse blocks
        //   to the assistant message. Restarting would produce duplicate blocks.
        //   Surface a retryable error so the caller can restart the full run.
        if !got_message_end {
            if submission_order.is_empty() {
                tracing::warn!(
                    iterations,
                    "stream ended without MessageEnd, no tools in flight — retrying"
                );
                continue;
            }
            tracing::error!(
                iterations,
                "stream dropped mid-response with tools in flight"
            );
            return Err(AgentError::retryable(
                "provider stream dropped mid-response with tools in flight",
            ));
        }

        // ── Authorize and dispatch all tools ──────────────────────────────────
        //
        // Authorization runs after MessageEnd — the LLM has finished speaking
        // before the user is prompted. Multiple tools are authorized concurrently
        // via a JoinSet; each tool starts executing the moment its auth resolves.
        {
            let mut auth_tasks: tokio::task::JoinSet<AuthOutcome> = tokio::task::JoinSet::new();

            for (id, name, input) in pending_auths {
                let config_c = Arc::clone(&config);
                let registry_c = Arc::clone(&active_registry);
                let tx_c = tx.clone();
                auth_tasks.spawn(async move {
                    let (result, injections) = authorize_tool(
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
                match outcome.expect("auth task panicked") {
                    AuthOutcome::Allowed {
                        id,
                        name,
                        input,
                        injections,
                    } => {
                        auth_injections.extend(injections);
                        let history_snap = Arc::from(messages.as_slice());
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
                        auth_injections.extend(injections);
                        pre_emitted.insert(tool.id.clone());
                        emit_tool_event(&tool, tx).await;
                        completed_map.insert(tool.id.clone(), tool);
                    }
                }
            }
        }

        // ── Collect remaining tools ────────────────────────────────────────────
        // No events emitted here — hooks may mutate outputs. Events are emitted
        // after the hook pass below, once the final content is settled.
        for done in executor.collect_remaining().await {
            completed_map.insert(done.id.clone(), done);
        }

        // ── Post-tool hooks ────────────────────────────────────────────────────
        //
        // Successful tools → PostToolUse. Failed tools → PostToolFailure.
        // The two paths are separate so hooks can specialise cleanly:
        // audit-on-failure, transform-on-success, without filtering inside
        // a single handler.
        //
        // A Block decision injects a system notice so the LLM knows to
        // disregard the tool's output on the next turn.
        for id in &submission_order {
            // Extract name and error flag before borrowing mutably below.
            let Some(done) = completed_map.get(id) else {
                continue;
            };
            let tool_name = done.name.clone();
            let is_error = done.output.is_error();

            let decision = if is_error {
                // Recover the original input from the assistant blocks
                // we built earlier this iteration (they're still in scope).
                let input = assistant_blocks.iter().find_map(|b| {
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
                    if let Some(done_mut) = completed_map.get_mut(id) {
                        done_mut.output.content = content;
                    }
                }
                _ => {}
            }

            // Emit the tool event now — after any mutation — unless it was
            // already emitted as an instant failure (malformed JSON / denied).
            if !pre_emitted.contains(id) {
                if let Some(done) = completed_map.get(id) {
                    emit_tool_event(done, tx).await;
                }
            }
        }

        // ── Apply auth injections ──────────────────────────────────────────────
        // HITL permission messages collected during authorization. These go
        // before the assistant message so the LLM sees them as context for
        // why certain tools were approved or denied on the next call.
        messages.extend(auth_injections);

        // ── Build assistant message ────────────────────────────────────────────
        // Thinking first (so the LLM can reference its reasoning), then text,
        // then tool calls — matching the order providers expect.
        //
        // Insert in reverse order so earlier items end up at lower indices:
        // text inserted at 0 → [Text, ...tools]
        // thinking inserted at 0 → [Thinking, Text, ...tools] ✓
        if !text_buf.is_empty() {
            assistant_blocks.insert(0, ContentBlock::Text { text: text_buf });
        }
        if !thinking_buf.is_empty() {
            assistant_blocks.insert(0, ContentBlock::Thinking { text: thinking_buf });
        }
        if !assistant_blocks.is_empty() {
            messages.push(Message::assistant(assistant_blocks));
        }

        // ── Append tool results in submission order ────────────────────────────
        //
        // Results MUST appear in the same order as the ToolUse blocks in the
        // preceding assistant message. Providers validate this; a mismatch
        // causes an API error. We reconstruct submission order from
        // `submission_order` + `completed_map` regardless of completion order.
        if !submission_order.is_empty() {
            let done_tools: Vec<CompletedTool> = submission_order
                .iter()
                .filter_map(|id| completed_map.remove(id))
                .collect();

            // Collect dynamic tools exposed by tool results.
            for done in &done_tools {
                for tool in &done.output.expose_tools {
                    dynamic_tools.insert(tool.name().to_string(), Arc::clone(tool));
                }
            }

            // Emit artifacts before ToolResult so callers receive them
            // while the tool result is still being processed.
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
                messages.push(Message::with_id(
                    uuid::Uuid::new_v4().to_string(),
                    Role::User,
                    result_blocks,
                ));
            }

            // Inject context after the tool results. These are system-level
            // reminders the tool wanted the LLM to see on the next turn.
            for done in &done_tools {
                for injection in &done.output.injections {
                    messages.push(system_reminder_msg(&injection.text));
                }
            }
        }

        tracing::debug!(
            stop_reason    = ?stop_reason,
            input_tokens   = usage.input_tokens,
            output_tokens  = usage.output_tokens,
            cache_read     = usage.cache_read_tokens,
            cache_write    = usage.cache_write_tokens,
            "llm call complete"
        );

        total_usage.input_tokens += usage.input_tokens;
        total_usage.output_tokens += usage.output_tokens;
        total_usage.cache_read_tokens += usage.cache_read_tokens;
        total_usage.cache_write_tokens += usage.cache_write_tokens;
        iterations += 1;

        // ── Checkpoint save ────────────────────────────────────────────────────
        // Save after every tool-use iteration so a resume can pick up from the
        // most recent complete tool exchange. We only checkpoint after tool use
        // (not after text-only turns) because a text-only turn with no new tool
        // results doesn't add resumable state to the history.
        if !submission_order.is_empty() {
            if let (Some(store), Some(run_id)) =
                (&config.checkpoint_store, &config.checkpoint_run_id)
            {
                let cp = RunCheckpoint {
                    run_id: run_id.clone(),
                    messages: messages.clone(),
                    iteration: iterations,
                    total_usage: total_usage.clone(),
                };
                if let Err(e) = store.save(run_id, &cp).await {
                    tracing::warn!(run_id, error = %e, "checkpoint save failed — continuing");
                } else {
                    tracing::debug!(run_id, iteration = iterations, "checkpoint saved");
                }
            }
        }

        // ── Token budget ───────────────────────────────────────────────────────
        if let Some(budget) = config.token_budget {
            let spent = total_usage.input_tokens as u64 + total_usage.output_tokens as u64;
            if spent >= budget {
                tracing::info!(iterations, spent, budget, "token budget exhausted");
                match config
                    .hooks
                    .pre_stop(
                        last_assistant_text(&messages),
                        RunStopReason::BudgetExhausted,
                        stop_hook_active,
                    )
                    .await
                {
                    HookDecision::Block { reason } if !stop_hook_active => {
                        tracing::debug!("pre_stop hook blocked BudgetExhausted");
                        messages.push(system_reminder_msg(&reason));
                        stop_hook_active = true;
                        continue;
                    }
                    _ => {}
                }
                return Ok(RunSummary {
                    stop_reason: RunStopReason::BudgetExhausted,
                    iterations,
                    usage: total_usage,
                    messages,
                });
            }
        }

        // ── Stop condition ─────────────────────────────────────────────────────

        // ── MaxTokens escalation ───────────────────────────────────────────────
        //
        // When the LLM hits max_tokens mid-response, we try two recovery steps
        // before giving up:
        //   1. Escalate max_tokens (×TOKEN_ESCALATION_FACTOR) and retry.
        //   2. Inject a continuation prompt so the LLM picks up where it left off.
        //   3. If both fail, return MaxTokensExhausted.
        if stop_reason == StopReason::MaxTokens {
            token_escalations += 1;

            if token_escalations == 1 {
                // First hit: escalate and retry the same turn.
                let escalated = (effective_max_tokens * TOKEN_ESCALATION_FACTOR).min(131_072); // hard ceiling
                tracing::warn!(
                    iterations,
                    current = effective_max_tokens,
                    escalated,
                    "MaxTokens hit — escalating max_tokens"
                );
                effective_max_tokens = escalated;
                // Don't advance the iteration counter; retry immediately.
                continue;
            } else if token_escalations <= MAX_TOKEN_ESCALATIONS {
                // Subsequent hits: inject a continuation prompt.
                tracing::warn!(
                    iterations,
                    escalation = token_escalations,
                    "MaxTokens hit again — injecting continuation prompt"
                );
                messages.push(Message::user(
                    "Your previous response was truncated due to length limits. \
                     Continue exactly where you left off. Do not repeat what you already said.",
                ));
                continue;
            } else {
                tracing::error!(
                    iterations,
                    escalations = token_escalations,
                    "MaxTokens exhausted after escalation and continuation — aborting"
                );
                match config
                    .hooks
                    .pre_stop(
                        last_assistant_text(&messages),
                        RunStopReason::MaxTokensExhausted,
                        stop_hook_active,
                    )
                    .await
                {
                    HookDecision::Block { reason } if !stop_hook_active => {
                        tracing::debug!("pre_stop hook blocked MaxTokensExhausted");
                        messages.push(system_reminder_msg(&reason));
                        stop_hook_active = true;
                        token_escalations = 0;
                        effective_max_tokens = config.max_tokens;
                        continue;
                    }
                    _ => {}
                }
                return Ok(RunSummary {
                    stop_reason: RunStopReason::MaxTokensExhausted,
                    iterations,
                    usage: total_usage,
                    messages,
                });
            }
        }

        // Reset escalation state on a normal (non-MaxTokens) turn.
        token_escalations = 0;
        effective_max_tokens = config.max_tokens;

        // ── Diminishing returns ────────────────────────────────────────────────
        //
        // Stop if the model has produced negligible output for too many turns in
        // a row. This catches runaway loops where the agent is technically alive
        // but not doing useful work.
        //
        // Tool-use turns are always treated as productive (the tool itself is the
        // work). We skip the streak increment — but still reset, so a streak that
        // began before a tool call doesn't persist through the tool-using turn.
        if !config.ignore_diminishing_returns {
            if stop_reason == StopReason::ToolUse || usage.output_tokens >= MIN_USEFUL_OUTPUT_TOKENS
            {
                low_output_streak = 0;
            } else {
                low_output_streak += 1;
                if low_output_streak >= MAX_LOW_OUTPUT_TURNS {
                    tracing::warn!(
                        iterations,
                        streak = low_output_streak,
                        "diminishing returns: stopping after {} low-output turns",
                        low_output_streak
                    );
                    match config
                        .hooks
                        .pre_stop(
                            last_assistant_text(&messages),
                            RunStopReason::DiminishingReturns,
                            stop_hook_active,
                        )
                        .await
                    {
                        HookDecision::Block { reason } if !stop_hook_active => {
                            tracing::debug!("pre_stop hook blocked DiminishingReturns");
                            messages.push(system_reminder_msg(&reason));
                            stop_hook_active = true;
                            low_output_streak = 0; // reset streak so the hook's injection gets a fair turn
                            continue;
                        }
                        _ => {}
                    }
                    return Ok(RunSummary {
                        stop_reason: RunStopReason::DiminishingReturns,
                        iterations,
                        usage: total_usage,
                        messages,
                    });
                }
            }
        }

        if stop_reason == StopReason::EndTurn {
            let last_text = last_assistant_text(&messages);

            match config
                .hooks
                .pre_stop(last_text, RunStopReason::Completed, stop_hook_active)
                .await
            {
                HookDecision::Block { reason } if !stop_hook_active => {
                    tracing::debug!("pre_stop hook blocked Completed — retrying");
                    messages.push(system_reminder_msg(&reason));
                    stop_hook_active = true;
                    continue;
                }
                HookDecision::MutateOutput { content } => {
                    tracing::debug!("pre_stop hook mutated final response");
                    replace_last_assistant_text(&mut messages, content);
                }
                _ => {}
            }

            tracing::info!(
                iterations,
                input_tokens = total_usage.input_tokens,
                output_tokens = total_usage.output_tokens,
                "agent run completed"
            );
            return Ok(RunSummary {
                stop_reason: RunStopReason::Completed,
                iterations,
                usage: total_usage,
                messages,
            });
        }

        // stop_reason == ToolUse → loop with tool results appended.
        // A productive tool-use turn means the agent is making progress —
        // reset the stop-hook flag so PreStop fires fresh on the next stop.
        stop_hook_active = false;
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Call the provider, retrying transient errors with exponential back-off.
///
/// Extracted from the main loop to keep `run_loop` readable. This function
/// contains all the retry state; `run_loop` just awaits a clean result or
/// a terminal error.
///
/// Retry behaviour:
/// - `RateLimit { retry_after_ms }` — uses the provider's hint when set,
///   otherwise falls back to the policy's exponential + jitter formula.
/// - `ServerError` / `Timeout` — always uses the policy formula.
/// - Non-retryable errors are returned immediately.
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
            structured: done.output.structured.clone(),
        }
    };
    tx.send(event).await.ok();
}

/// Create an immediately-completed failed tool (no execution, no timer).
fn instant_failure(id: String, name: String, output: wui_core::tool::ToolOutput) -> CompletedTool {
    CompletedTool {
        id,
        name,
        output,
        ms: 0,
        attempts: 1,
    }
}

/// Build a system-reminder `Message` from plain text.
///
/// Used to inject framework-level notices into the conversation so the LLM
/// can see them without mistaking them for user input.
fn system_reminder_msg(text: &str) -> Message {
    Message {
        id: uuid::Uuid::new_v4().to_string(),
        role: Role::System,
        content: vec![ContentBlock::Text {
            text: wui_core::fmt::system_reminder(text),
        }],
    }
}

/// Outcome of a concurrent authorization task.
///
/// Both variants carry `injections` — system messages produced by the HITL
/// flow (e.g., "tool X was approved"). These are collected and applied to
/// `messages` before the assistant turn is committed.
enum AuthOutcome {
    Allowed {
        id: String,
        name: String,
        input: serde_json::Value,
        injections: Vec<Message>,
    },
    Denied {
        tool: CompletedTool,
        injections: Vec<Message>,
    },
}

/// Verify that a tool call is permitted to run.
///
/// Checks in this order — first match wins:
///   1. Pre-tool hook (may block or mutate input).
///   2. Static deny rules — hard constraint, no override possible.
///   3. Session always-denied — user's runtime decision to block.
///   4. Static allow rules — developer pre-approval, skips prompting.
///   5. Session always-allowed — user's runtime decision to allow.
///   6. Mode-based check (Auto → allow, Readonly → check is_readonly, Ask → HITL).
///
/// Returns `(Ok(effective_input), injections)` when cleared. The input may
/// differ from the original if a `PreToolUse` hook mutated it. `injections`
/// contains HITL system messages to be applied to the conversation history.
/// Returns `(Err(CompletedTool), injections)` when blocked or denied.
async fn authorize_tool(
    config: &RunConfig,
    registry: &ToolRegistry,
    id: String,
    name: String,
    input: serde_json::Value,
    tx: &mpsc::Sender<AgentEvent>,
) -> (Result<serde_json::Value, CompletedTool>, Vec<Message>) {
    // 1. Pre-tool hook — may allow, mutate, or block.
    let input = match config.hooks.pre_tool_use(&name, &input).await {
        HookDecision::Block { reason } => {
            return (
                Err(instant_failure(
                    id,
                    name,
                    wui_core::tool::ToolOutput::hook_blocked(reason),
                )),
                vec![],
            )
        }
        HookDecision::Mutate { input: new } => new,
        // MutateOutput is not meaningful for PreToolUse — treat as Allow.
        HookDecision::Allow | HookDecision::MutateOutput { .. } => input,
    };

    // Compute per-invocation tool metadata once — reused across checks below.
    let tool_impl = registry.get(&name);
    let tool_meta = tool_impl
        .as_deref()
        .map(|t| t.meta(&input))
        .unwrap_or_default();
    let perm_key = tool_meta.permission_key.clone();
    let perm_key_ref = perm_key.as_deref();
    let is_destructive = tool_meta.destructive;
    let needs_interaction = tool_meta.requires_interaction;

    // 2. Structural check: tools that require user interaction cannot run
    //    headlessly. This is not a permission decision — the loop literally
    //    has no mechanism to fulfil the tool's interaction requirement in
    //    Auto mode. Deny immediately rather than hanging indefinitely.
    if needs_interaction && config.permission.is_auto() {
        return (
            Err(instant_failure(
                id,
                name.clone(),
                wui_core::tool::ToolOutput::permission_denied(format!(
                    "tool '{name}' requires user interaction and cannot run in Auto mode; \
                 switch to PermissionMode::Ask or disable this tool for headless runs"
                )),
            )),
            vec![],
        );
    }

    // Evaluate static rules once — used at steps 3 and 5 below.
    // `None` = no rule matched; `Some(false)` = deny; `Some(true)` = allow.
    let static_rule = config.rules.evaluate(&name, perm_key_ref);

    // 3. Static deny rules — developer hard constraint.
    if static_rule == Some(false) {
        return (
            Err(instant_failure(
                id,
                name.clone(),
                wui_core::tool::ToolOutput::permission_denied(format!(
                    "tool '{name}' is in the configured deny list"
                )),
            )),
            vec![],
        );
    }

    // 4. Session always-denied — user's runtime decision (strengthens security).
    if config.session_perms.is_always_denied(&name).await {
        return (
            Err(instant_failure(
                id,
                name.clone(),
                wui_core::tool::ToolOutput::permission_denied(format!(
                    "tool '{name}' was previously denied for this session"
                )),
            )),
            vec![],
        );
    }

    // 5. Static allow rules — developer pre-approval (bypasses prompting).
    if static_rule == Some(true) {
        return (Ok(input), vec![]);
    }

    // 6. Session always-allowed — user's runtime decision to allow.
    if config.session_perms.is_always_allowed(&name).await {
        return (Ok(input), vec![]);
    }

    // 7. Mode-based permission check.
    //    Enrich the description with key suffix and destructive flag so the
    //    human sees enough context to make a meaningful decision.
    let tool_is_readonly = tool_meta.readonly;
    let description = {
        let base = match perm_key_ref {
            Some(key) => format!("call {name}({key})"),
            None => format!("call {name}"),
        };
        if is_destructive {
            format!("{base} [destructive — cannot be undone]")
        } else {
            base
        }
    };
    let ctrl_req = ControlRequest {
        id: uuid::Uuid::new_v4().to_string(),
        kind: ControlKind::PermissionRequest {
            tool_name: name.clone(),
            description,
        },
    };

    match permission::check(&config.permission, &ctrl_req, tool_is_readonly, &input) {
        PermissionOutcome::Allowed => (Ok(input), vec![]),
        PermissionOutcome::Denied { reason } => (
            Err(instant_failure(
                id,
                name,
                wui_core::tool::ToolOutput::permission_denied(reason),
            )),
            vec![],
        ),
        PermissionOutcome::NeedsApproval => {
            await_approval(config, id, name, input, ctrl_req, tx).await
        }
    }
}

/// Emit a `ControlHandle`, wait for the human's decision, and return the
/// outcome along with any injection messages produced by the exchange.
///
/// Extracted from `authorize_tool` so each function has a single clear job.
/// The returned `Vec<Message>` contains the HITL permission system message
/// to be applied to conversation history after all tool auth tasks settle.
async fn await_approval(
    config: &RunConfig,
    id: String,
    name: String,
    input: serde_json::Value,
    ctrl_req: ControlRequest,
    tx: &mpsc::Sender<AgentEvent>,
) -> (Result<serde_json::Value, CompletedTool>, Vec<Message>) {
    let request_id = ctrl_req.id.clone();
    let (handle, rx) = ControlHandle::new(ctrl_req);
    tx.send(AgentEvent::Control(handle)).await.ok();

    let response = rx.await.unwrap_or_else(|_| {
        ControlResponse::deny(
            request_id,
            "control handle dropped before response was sent",
        )
    });

    // Collect the human's decision as a system message. It will be injected
    // into `messages` once all auth tasks for this turn have settled, placing
    // it before the assistant message so the LLM sees it on the next call.
    let injection = Message::system(permission::response_to_system_message(&response));

    let result = match response.decision {
        ControlDecision::Deny { reason } => Err(instant_failure(
            id,
            name,
            wui_core::tool::ToolOutput::permission_denied(reason),
        )),

        ControlDecision::DenyAlways { reason } => {
            config.session_perms.set_always_deny(name.clone()).await;
            Err(instant_failure(
                id,
                name,
                wui_core::tool::ToolOutput::permission_denied(reason),
            ))
        }

        ControlDecision::ApproveAlways => {
            config.session_perms.set_always_allow(name).await;
            Ok(input)
        }

        // The modification note (if any) is already in the system message above.
        ControlDecision::Approve { .. } => Ok(input),
    };

    (result, vec![injection])
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
            m.content.iter().find_map(|b| {
                if let ContentBlock::Text { text } = b {
                    Some(text.as_str())
                } else {
                    None
                }
            })
        })
        .unwrap_or("")
}

/// Replace the text content of the most recent assistant message.
///
/// Called when a `PreStop` hook returns `MutateOutput` — the hook has
/// rewritten the response, and the message history must reflect the mutation
/// before the run completes.
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
///
/// If the registry has no deferred tools, returns the base unchanged.
/// Otherwise appends an "## Additional tools" section so the LLM knows
/// these tools exist and should call ToolSearch to get their schemas.
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
