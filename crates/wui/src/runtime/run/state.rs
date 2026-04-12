use std::collections::HashMap;
use std::sync::Arc;

use wui_core::event::{AgentError, RunStopReason, RunSummary, StopReason, TokenUsage};
use wui_core::hook::HookDecision;
use wui_core::message::{ContentBlock, Message};
use wui_core::tool::ToolCallId;

use serde::{Deserialize, Serialize};

use super::history::{last_assistant_text, system_reminder_msg};
use super::tool_batch::EmissionGuard;
use super::RunConfig;
use crate::runtime::executor::CompletedTool;
use crate::runtime::registry::ToolRegistry;

/// Consecutive compression failures before the circuit breaker opens.
const MAX_COMPRESS_FAILURES: u32 = 3;

// ── IterationTransition ─────────────────────────────────────────────────────

/// Why the run loop continued after a given iteration.
///
/// Recorded in `RecoveryState::last_transition` for observability. Test
/// harnesses and logging can inspect this to understand the loop's decisions
/// without parsing tracing output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IterationTransition {
    /// Normal continuation — the LLM returned tool calls.
    ToolUse,
    /// Context was compressed to relieve pressure.
    Compressed,
    /// `max_tokens` was escalated after a truncated response.
    MaxTokensEscalated,
    /// A "continue where you left off" message was injected.
    ContinuationInjected,
    /// A PreStop hook blocked the stop and forced another iteration.
    HookRetry,
    /// Emergency compression after a prompt-too-long rejection.
    EmergencyCompress,
}

// ── MaxTokensAction ─────────────────────────────────────────────────────────

/// What to do when the LLM stops with `MaxTokens`.
pub(super) enum MaxTokensAction {
    /// Bump `effective_max_tokens` and retry immediately.
    Escalate,
    /// Inject a "continue where you left off" message.
    InjectContinuation,
    /// All escalation and continuation attempts are exhausted.
    Exhausted,
}

// ── RecoveryState ───────────────────────────────────────────────────────────

/// Cross-iteration error recovery state.
///
/// Tracks escalation counters and heuristic signals that the run loop
/// uses to decide whether to continue, escalate, or stop.
pub(super) struct RecoveryState {
    /// How many times max_tokens has been bumped after MaxTokens stops.
    pub(super) token_escalations: u32,
    /// Consecutive turns below MIN_USEFUL_OUTPUT_TOKENS.
    pub(super) low_output_streak: u32,
    /// Effective max_tokens for the current call; may be escalated mid-run.
    pub(super) effective_max_tokens: u32,
    /// True when a PreStop hook already blocked — prevents infinite loops.
    pub(super) stop_hook_active: bool,
    /// Consecutive compression attempts that failed to relieve pressure.
    /// After `MAX_COMPRESS_FAILURES` the circuit opens and compression is
    /// skipped, letting `ContextOverflow` fire instead of looping.
    pub(super) compress_failures: u32,
    /// Why the loop continued at the end of the last iteration.
    /// Recorded for observability — accessible from `RunSummary`.
    pub(super) last_transition: Option<IterationTransition>,
}

impl RecoveryState {
    /// Create a new recovery state with the given base max_tokens.
    pub(super) fn new(base_max_tokens: u32) -> Self {
        Self {
            token_escalations: 0,
            low_output_streak: 0,
            effective_max_tokens: base_max_tokens,
            stop_hook_active: false,
            compress_failures: 0,
            last_transition: None,
        }
    }

    /// Record a compression that relieved pressure — resets the circuit breaker.
    pub(super) fn on_compress_success(&mut self) {
        self.compress_failures = 0;
    }

    /// Record a compression that failed to relieve pressure.
    /// Returns `true` when the circuit breaker has opened.
    pub(super) fn on_compress_failure(&mut self) -> bool {
        self.compress_failures += 1;
        self.compress_failures >= MAX_COMPRESS_FAILURES
    }

    /// Handle a `MaxTokens` stop reason. Returns the action the run loop
    /// should take.
    pub(super) fn on_max_tokens(
        &mut self,
        escalation_factor: u32,
        max_escalations: u32,
    ) -> MaxTokensAction {
        self.token_escalations += 1;
        if self.token_escalations == 1 {
            self.effective_max_tokens =
                (self.effective_max_tokens * escalation_factor).min(131_072);
            return MaxTokensAction::Escalate;
        }
        if self.token_escalations <= max_escalations {
            return MaxTokensAction::InjectContinuation;
        }
        MaxTokensAction::Exhausted
    }

    /// Record a productive turn — resets the low-output streak and
    /// token escalation state.
    pub(super) fn on_productive_turn(&mut self, base_max_tokens: u32) {
        self.token_escalations = 0;
        self.effective_max_tokens = base_max_tokens;
    }

    /// Record a low-output turn. Returns `true` if the streak has reached
    /// the given `threshold` and the run should stop.
    pub(super) fn on_low_output(&mut self, threshold: u32) -> bool {
        self.low_output_streak += 1;
        self.low_output_streak >= threshold
    }

    /// Reset the stop-hook guard so the next stop reason can be blocked.
    pub(super) fn reset_stop_hook(&mut self) {
        self.stop_hook_active = false;
    }
}

// ── RunState ────────────────────────────────────────────────────────────────

/// Mutable state carried across iterations of the run loop.
pub(super) struct RunState {
    pub(super) messages: Vec<Message>,
    pub(super) total_usage: TokenUsage,
    pub(super) iterations: u32,
    /// Augmented system prompt (base + deferred tool listings).
    pub(super) system: String,
    /// Cross-iteration error recovery state.
    pub(super) recovery: RecoveryState,
    /// Tools injected at runtime by ToolOutput::expose (e.g., via tool_search).
    pub(super) dynamic_tools: HashMap<String, Arc<dyn wui_core::tool::Tool>>,
    /// True once provider capability preflight has run (first iteration only).
    pub(super) preflight_done: bool,
    /// Byte index in `system` where the cache boundary falls.
    /// Passed through to `ChatRequest` so providers can optimize caching.
    pub(super) cache_boundary: Option<usize>,
}

/// Result of checking max-iterations at the top of each loop iteration.
pub(super) enum IterGuard {
    /// Below the limit — proceed normally.
    Proceed,
    /// At the limit, hook blocked — the loop should `continue`.
    Blocked,
    /// At the limit, no block — return this summary.
    Stop(RunSummary),
}

impl RunState {
    pub(super) fn summary(&self, stop_reason: RunStopReason) -> RunSummary {
        RunSummary {
            stop_reason,
            iterations: self.iterations,
            usage: self.total_usage.clone(),
            messages: self.messages.clone(),
            last_transition: self
                .recovery
                .last_transition
                .as_ref()
                .and_then(|t| serde_json::to_value(t).ok())
                .and_then(|v| v.as_str().map(String::from)),
        }
    }

    /// Check whether we've hit `max_iter` and consult the PreStop hook.
    pub(super) async fn check_max_iter(&mut self, config: &RunConfig) -> IterGuard {
        if self.iterations < config.max_iter {
            return IterGuard::Proceed;
        }
        if !self.recovery.stop_hook_active {
            if let HookDecision::Block { reason } = config
                .hooks
                .pre_stop(
                    last_assistant_text(&self.messages),
                    RunStopReason::MaxIterations,
                    false,
                    &self.messages,
                )
                .await
            {
                self.messages.push(system_reminder_msg(&reason));
                self.recovery.stop_hook_active = true;
                return IterGuard::Blocked;
            }
        }
        IterGuard::Stop(self.summary(RunStopReason::MaxIterations))
    }
}

// ── Per-iteration mutable state ──────────────────────────────────────────────

/// Bundles all mutable state accumulated during a single loop iteration.
/// Extracted phases operate on this struct, keeping `run_loop` slim.
pub(super) struct IterationCtx {
    pub(super) pending_inputs: HashMap<ToolCallId, (String, String)>,
    pub(super) tool_inputs: HashMap<ToolCallId, serde_json::Value>,
    pub(super) assistant_blocks: Vec<ContentBlock>,
    pub(super) submission_order: Vec<ToolCallId>,
    pub(super) completed_map: HashMap<ToolCallId, CompletedTool>,
    pub(super) text_buf: String,
    pub(super) thinking_buf: String,
    pub(super) stop_reason: StopReason,
    pub(super) usage: TokenUsage,
    pub(super) emission_guard: EmissionGuard,
    pub(super) auth_injections: Vec<Message>,
}

impl IterationCtx {
    pub(super) fn new() -> Self {
        Self {
            pending_inputs: HashMap::new(),
            tool_inputs: HashMap::new(),
            assistant_blocks: Vec::new(),
            submission_order: Vec::new(),
            completed_map: HashMap::new(),
            text_buf: String::new(),
            thinking_buf: String::new(),
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            emission_guard: EmissionGuard::new(),
            auth_injections: Vec::new(),
        }
    }

    pub(super) fn record_tool_use(
        &mut self,
        id: ToolCallId,
        name: String,
        input: serde_json::Value,
        summary: Option<String>,
    ) {
        self.submission_order.push(id.clone());
        self.tool_inputs.insert(id.clone(), input.clone());
        self.assistant_blocks.push(ContentBlock::ToolUse {
            id,
            name,
            input,
            summary,
        });
    }

    pub(super) fn remember_tool_input(&mut self, id: &ToolCallId, input: serde_json::Value) {
        self.tool_inputs.insert(id.clone(), input);
    }

    pub(super) fn tool_input(&self, id: &ToolCallId) -> Option<&serde_json::Value> {
        self.tool_inputs.get(id)
    }
}

// ── RunState construction ───────────────────────────────────────────────────

impl RunState {
    /// Create the initial run state and restore any saved checkpoint.
    pub(super) async fn new(config: &RunConfig, messages: Vec<Message>) -> Self {
        let mut s = Self {
            system: augment_system(&config.system, &config.tools),
            total_usage: TokenUsage::default(),
            iterations: 0,
            recovery: RecoveryState::new(config.max_tokens),
            dynamic_tools: HashMap::new(),
            preflight_done: false,
            cache_boundary: config.cache_boundary,
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
                Err(e) => {
                    tracing::warn!(run_id, error = %e, "checkpoint load failed — starting fresh")
                }
            }
        }
        s
    }
}

// ── System prompt augmentation ──────────────────────────────────────────────

/// Append a deferred-tools listing to the system prompt when needed.
///
/// Lives here (not in history.rs) because it's about initial prompt
/// construction, not about assembling message history after a turn.
pub(super) fn augment_system(base: &str, registry: &ToolRegistry) -> String {
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
        Call `tool_search` with the tool name or a keyword before using them:\n\n\
        {listing}"
    );

    format!("{base}\n\n{}", wui_core::fmt::system_reminder(&section))
}

// ── Preflight ────────────────────────────────────────────────────────────────

/// Verify that the provider supports the features the request requires.
///
/// Runs once on the first iteration. Produces an explicit fatal error
/// instead of letting the API return a confusing rejection.
pub(super) fn preflight_check(
    config: &RunConfig,
    req: &wui_core::provider::ChatRequest,
) -> Result<(), AgentError> {
    let caps = config.provider.capabilities(config.model.as_deref());

    if !req.tools.is_empty() && !caps.tool_calling {
        return Err(AgentError::fatal(
            "provider does not support tool calling, but tools were registered",
        ));
    }
    if req.thinking_budget.is_some() && !caps.thinking {
        return Err(AgentError::fatal(
            "provider does not support extended thinking, but a thinking budget was set",
        ));
    }
    let has_image = req.messages.iter().any(|m| {
        m.content
            .iter()
            .any(|b| matches!(b, ContentBlock::Image { .. }))
    });
    if has_image && !caps.image_input {
        return Err(AgentError::fatal(
            "provider does not support image input, but messages contain images",
        ));
    }
    let has_doc = req.messages.iter().any(|m| {
        m.content
            .iter()
            .any(|b| matches!(b, ContentBlock::Document { .. }))
    });
    if has_doc && !caps.document_input {
        return Err(AgentError::fatal(
            "provider does not support document input, but messages contain documents",
        ));
    }
    Ok(())
}
