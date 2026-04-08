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

mod compression;
mod history;
mod parsing;
mod provider;
mod stream;
mod tool_batch;

// Re-export public / pub(crate) items at the `run` module boundary so that
// existing import paths (`super::run::run`, `super::run::RunConfig`, etc.)
// continue to resolve unchanged.
pub use provider::RetryPolicy;
pub(crate) use stream::run;
pub use stream::RunStream;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use wui_core::event::{AgentError, AgentEvent, RunStopReason, RunSummary, StopReason, TokenUsage};
use wui_core::hook::HookDecision;
use wui_core::message::{ContentBlock, Message};
use wui_core::provider::ChatRequest;
use wui_core::types::ToolCallId;

use crate::compress::CompressStrategy;

// ── Sibling runtime modules used by submodules ──────────────────────────────
// Re-export at `super::` level so submodules can write `super::auth` etc.
use super::auth;
use super::checkpoint::{self, CheckpointStore};
use super::executor::{CompletedTool, PendingTool, ToolExecutor};
use super::hooks::HookRunner;
use super::permission::{PermissionMode, PermissionRules, SessionPermissions};
use super::registry::{self, ToolRegistry};

// ── Submodule imports used in this file ─────────────────────────────────────
use compression::{emergency_compress, maybe_compress};
use history::{
    augment_system, last_assistant_text, replace_last_assistant_text, system_reminder_msg,
};
use parsing::parse_stream;
use provider::{call_with_retry, is_prompt_too_long};
use tool_batch::{authorize_and_dispatch, run_post_hooks, EmissionGuard};

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

// ── Run Config ────────────────────────────────────────────────────────────────

/// Static configuration for one run. `Arc`-wrapped so it is cheap to share
/// across the spawned task and helper services.
pub(crate) struct RunConfig {
    pub(crate) provider: Arc<dyn wui_core::provider::Provider>,
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
    pub(crate) retry: provider::RetryPolicy,

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

// ── RunState ────────────────────────────────────────────────────────────────

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
        history::assemble_history(ctx, &mut s, &config, tx).await;

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

#[cfg(test)]
mod tests;
