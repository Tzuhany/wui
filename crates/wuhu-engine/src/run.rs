// ============================================================================
// The Loop — the beating heart of Wuhu.
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
//        - ToolUseEnd → authorize_tool() → executor.submit() starts the tool NOW
//        - poll executor during stream → harvest completed tools early
//        - MessageEnd → break inner loop
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
// Real-time events (ToolDone, ToolError) are still emitted in completion
// order for immediate UI feedback. Only the history write is ordered.
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

use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tracing::Instrument as _;

use wuhu_core::event::{
    AgentError, AgentEvent, ControlDecision, ControlHandle, ControlKind, ControlRequest,
    ControlResponse, RunStopReason, RunSummary, StopReason, TokenUsage,
};
use wuhu_core::hook::HookDecision;
use wuhu_core::message::{ContentBlock, Message, Role};
use wuhu_core::provider::{ChatRequest, Provider};
use wuhu_core::tool::SpawnFn;

use wuhu_compress::CompressPipeline;

use crate::executor::{CompletedTool, PendingTool, ToolExecutor};
use crate::hooks::HookRunner;
use crate::permission::{self, PermissionMode, PermissionOutcome, SessionPermissions};
use crate::query_chain::QueryChain;
use crate::registry::ToolRegistry;

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
    pub max_attempts:     u32,
    /// Wait before the first retry (milliseconds).
    pub initial_delay_ms: u64,
    /// Multiplier applied to the delay after each failure (exponential back-off).
    /// `2.0` doubles the delay each time: 500 ms → 1 s → 2 s → …, capped at `max_delay_ms`.
    pub multiplier:       f64,
    /// Hard cap on the delay between retries (milliseconds).
    pub max_delay_ms:     u64,
    /// Add equal jitter to each back-off delay.
    ///
    /// Uses the equal-jitter formula: `delay = exp/2 + rand(0, exp/2)`.
    /// This guarantees at least half the computed delay (no starvation) while
    /// spreading load across the full range (no thundering herd). The jitter
    /// range grows with the exponential delay, so later retries are spread
    /// proportionally further apart than earlier ones.
    /// Enabled by default.
    pub jitter:           bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts:     3,
            initial_delay_ms: 500,
            multiplier:       2.0,
            max_delay_ms:     10_000,
            jitter:           true,
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
        let exp = (self.initial_delay_ms as f64
            * self.multiplier.powi((attempt - 1) as i32))
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

/// Minimum useful output tokens per turn. Below this = "not making progress."
const MIN_USEFUL_OUTPUT_TOKENS: u32 = 500;

/// Consecutive low-output turns before the engine auto-stops.
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
    inner:  ReceiverStream<AgentEvent>,
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
/// across the spawned task and any sub-agent closures.
pub struct RunConfig {
    pub provider:      Arc<dyn Provider>,
    pub tools:         Arc<ToolRegistry>,
    pub hooks:         Arc<HookRunner>,
    pub compress:      CompressPipeline,
    pub permission:    PermissionMode,
    pub session_perms: Arc<SessionPermissions>,
    pub system:        String,
    pub model:         String,
    pub max_tokens:    u32,
    pub temperature:   Option<f32>,
    pub max_iter:      u32,
    /// Extensions applied to every LLM call — e.g. `thinking`, `betas`.
    pub extensions:         HashMap<String, serde_json::Value>,

    /// Extensions applied only on the first LLM call (iteration 0).
    ///
    /// Use for parameters that should guide the opening response but must
    /// not persist into continuation turns. Primary use case: `tool_choice`
    /// — forcing the first turn to use a tool while leaving subsequent turns
    /// free to respond with text.
    ///
    /// On iteration 0, these are merged over `extensions` (initial wins on
    /// conflict). On all later iterations they are absent from the request.
    pub initial_extensions: HashMap<String, serde_json::Value>,

    pub spawn: Option<SpawnFn>,

    /// Sub-agent depth tracker. `None` for top-level runs.
    ///
    /// When set, the engine logs the chain ID and depth for every iteration
    /// and rejects runs where `chain.depth > chain.max_depth`. Pass
    /// `chain.child()?` when spawning sub-agents to enforce the ceiling.
    pub query_chain: Option<QueryChain>,

    /// Back-off policy for transient provider errors.
    pub retry: RetryPolicy,

    /// Default execution timeout applied to every tool that doesn't declare
    /// its own `Tool::timeout()`. `None` means tools may run indefinitely.
    pub tool_timeout: Option<Duration>,

    /// Disable the diminishing-returns auto-stop heuristic.
    ///
    /// When `false` (default), the engine stops a run after
    /// `MAX_LOW_OUTPUT_TURNS` consecutive turns with fewer than
    /// `MIN_USEFUL_OUTPUT_TOKENS` output tokens. Set this to `true` for
    /// long-running tasks where many short intermediate steps are expected
    /// before a large final result (e.g. research tasks, file-heavy writes).
    pub ignore_diminishing_returns: bool,
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
pub fn run(
    config:   Arc<RunConfig>,
    messages: Vec<Message>,
) -> RunStream {
    let cancel = CancellationToken::new();
    let (tx, rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
    tokio::spawn(run_task(config, messages, cancel.clone(), tx));
    RunStream { inner: ReceiverStream::new(rx), cancel }
}

// ── Internal task ─────────────────────────────────────────────────────────────

async fn run_task(
    config:   Arc<RunConfig>,
    messages: Vec<Message>,
    cancel:   CancellationToken,
    tx:       mpsc::Sender<AgentEvent>,
) {
    let span = tracing::info_span!(
        "agent.run",
        model      = %config.model,
        max_iter   = config.max_iter,
        tools      = config.tools.len(),
        chain_id   = config.query_chain.as_ref().map(|c| c.chain_id.as_str()).unwrap_or(""),
        depth      = config.query_chain.as_ref().map(|c| c.depth).unwrap_or(0),
    );
    let result = run_loop(config, messages, cancel, &tx).instrument(span).await;
    match result {
        Ok(summary) => { let _ = tx.send(AgentEvent::Done(summary)).await; }
        Err(e)      => { let _ = tx.send(AgentEvent::Error(e)).await; }
    }
}

// ── Main loop ─────────────────────────────────────────────────────────────────

async fn run_loop(
    config:       Arc<RunConfig>,
    mut messages: Vec<Message>,
    cancel:       CancellationToken,
    tx:           &mpsc::Sender<AgentEvent>,
) -> Result<RunSummary, AgentError> {
    // Enforce sub-agent depth ceiling before doing any work.
    if let Some(chain) = &config.query_chain {
        if chain.depth > chain.max_depth {
            return Err(AgentError {
                message:   format!(
                    "sub-agent depth limit {} reached (chain {})",
                    chain.max_depth, chain.chain_id,
                ),
                retryable: false,
            });
        }
    }

    // Augment the system prompt with deferred tool listings once, before the loop.
    // This tells the LLM which additional tools exist so it can call ToolSearch.
    let system = augment_system(&config.system, &config.tools);

    let mut total_usage         = TokenUsage::default();
    let mut iterations          = 0u32;
    // ── Error-recovery state ──────────────────────────────────────────────────
    // Carried across iterations; reset only on non-error turns.
    let mut token_escalations   = 0u32;  // how many times we've bumped max_tokens
    let mut low_output_streak   = 0u32;  // consecutive turns below MIN_USEFUL_OUTPUT_TOKENS
    // The effective max_tokens for the current call; may be escalated mid-run.
    let mut effective_max_tokens = config.max_tokens;

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
            tracing::warn!(iterations, max_iter = config.max_iter, "agent hit max iterations");
            return Ok(RunSummary {
                stop_reason: RunStopReason::MaxIterations,
                iterations,
                usage: total_usage,
                messages,
            });
        }

        tracing::debug!(iteration = iterations, messages = messages.len(), "iteration start");

        // ── Context compression ────────────────────────────────────────────────
        if let Some((compressed, method, freed)) = config.compress.maybe_compress(
            &messages,
            config.provider.as_ref(),
            &config.model,
        ).await {
            tracing::debug!(method = ?method, freed_tokens = freed, "context compressed");
            messages = compressed;
            tx.send(AgentEvent::Compressed { method, freed }).await.ok();
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

        // ── Build request ──────────────────────────────────────────────────────
        // Merge initial_extensions into extensions on the first turn only.
        let extensions = if iterations == 0 && !config.initial_extensions.is_empty() {
            let mut ext = config.extensions.clone();
            ext.extend(config.initial_extensions.clone());
            ext
        } else {
            config.extensions.clone()
        };

        let req = ChatRequest {
            model:       config.model.clone(),
            max_tokens:  effective_max_tokens,
            temperature: config.temperature,
            system:      system.clone(),
            messages:    messages.clone(),
            tools:       config.tools.tool_defs(),
            extensions,
        };

        // ── Call provider with retry ───────────────────────────────────────────
        let stream = call_with_retry(&config, &req, tx).await?;

        let mut executor = ToolExecutor::new(
            config.tools.clone(),
            cancel.clone(),
            config.spawn.clone(),
            config.query_chain.clone(),
            tx.clone(),
            config.tool_timeout,
        );

        // Accumulate tool input JSON chunks keyed by tool_use_id.
        let mut pending_inputs:   HashMap<String, (String, String)> = HashMap::new();
        let mut assistant_blocks: Vec<ContentBlock>                  = Vec::new();
        // Track submission order for ordered history reconstruction.
        let mut submission_order: Vec<String>                        = Vec::new();
        let mut completed_map:    HashMap<String, CompletedTool>     = HashMap::new();
        let mut text_buf          = String::new();
        let mut thinking_buf      = String::new();
        let mut stop_reason       = StopReason::EndTurn;
        let mut usage             = TokenUsage::default();
        let mut got_message_end   = false;

        futures::pin_mut!(stream);

        while let Some(event) = stream.next().await {
            // Non-blocking harvest: collect tools that finished while LLM streamed.
            for done in executor.poll_completed() {
                emit_tool_event(&done, tx).await;
                completed_map.insert(done.id.clone(), done);
            }

            let event = match event {
                Ok(e)  => e,
                Err(e) => {
                    if e.is_retryable() { continue; }
                    return Err(AgentError { message: e.to_string(), retryable: false });
                }
            };

            use wuhu_core::event::StreamEvent::*;
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
                    let Some((name, json)) = pending_inputs.remove(&id) else { continue };

                    let input: serde_json::Value = serde_json::from_str(&json)
                        .unwrap_or(serde_json::Value::Object(Default::default()));

                    // Record submission order before any early-exit paths.
                    submission_order.push(id.clone());

                    // The Anthropic API (and most providers) require every
                    // ToolUse in the assistant message to have a matching
                    // ToolResult in the following user message — even when
                    // the tool is denied, blocked, or fails validation.
                    // Add the ToolUse block NOW, before any deny paths.
                    assistant_blocks.push(ContentBlock::ToolUse {
                        id:    id.clone(),
                        name:  name.clone(),
                        input: input.clone(),
                    });

                    match authorize_tool(&config, id.clone(), name.clone(), input, tx, &mut messages).await {
                        Err(denied) => {
                            emit_tool_event(&denied, tx).await;
                            completed_map.insert(denied.id.clone(), denied);
                        }
                        Ok(effective_input) => {
                            let history_snap = Arc::from(messages.as_slice());
                            tracing::debug!(tool = %name, "tool dispatched");
                            tx.send(AgentEvent::ToolStart {
                                id:    id.clone(),
                                name:  name.clone(),
                                input: effective_input.clone(),
                            }).await.ok();
                            executor.submit(PendingTool {
                                id, name, input: effective_input, messages: history_snap,
                            });
                        }
                    }
                }

                MessageEnd { usage: u, stop_reason: sr } => {
                    usage           = u;
                    stop_reason     = sr;
                    got_message_end = true;
                    break;
                }

                wuhu_core::event::StreamEvent::Error { message, retryable } => {
                    return Err(AgentError { message, retryable });
                }
            }
        }

        // ── Stream-drop recovery ───────────────────────────────────────────────
        //
        // If we exited without a MessageEnd (network drop mid-stream), we have
        // two cases:
        //
        //   Clean (no tools submitted yet): no state is locked in. Retry this
        //   iteration transparently — `continue` restarts from `call_with_retry`.
        //
        //   Dirty (tools already submitted): we've already built ToolUse blocks
        //   and dispatched concurrent tasks. Restarting would double-execute.
        //   Surface a retryable error so the caller can restart the full run.
        if !got_message_end {
            if submission_order.is_empty() {
                tracing::warn!(iterations, "stream ended without MessageEnd, no tools in flight — retrying");
                continue;
            }
            tracing::error!(iterations, "stream dropped mid-response with tools in flight");
            return Err(AgentError {
                message:   "provider stream dropped mid-response with tools in flight".to_string(),
                retryable: true,
            });
        }

        // ── Collect remaining tools ────────────────────────────────────────────
        for done in executor.collect_remaining().await {
            emit_tool_event(&done, tx).await;
            completed_map.insert(done.id.clone(), done);
        }

        // ── Post-tool hooks ────────────────────────────────────────────────────
        //
        // Run after all tools complete. A Block decision injects a system notice
        // so the LLM knows to disregard the tool's output on the next turn.
        for id in &submission_order {
            let Some(done) = completed_map.get(id) else { continue };
            let decision = config.hooks.post_tool_use(&done.name, &done.output).await;
            if let HookDecision::Block { reason } = decision {
                tracing::debug!(tool = %done.name, %reason, "post-tool hook blocked output");
                messages.push(system_reminder_msg(&format!(
                    "The output of tool '{}' was blocked by policy: {reason}. Do not use this output.",
                    done.name,
                )));
            }
        }

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
            let done_tools: Vec<CompletedTool> = submission_order.iter()
                .filter_map(|id| completed_map.remove(id))
                .collect();

            // Emit artifacts before ToolResult so callers receive them
            // while the tool result is still being processed.
            for done in &done_tools {
                for artifact in &done.output.artifacts {
                    tx.send(AgentEvent::Artifact {
                        tool_id:   done.id.clone(),
                        tool_name: done.name.clone(),
                        artifact:  artifact.clone(),
                    }).await.ok();
                }
            }

            let result_blocks: Vec<ContentBlock> = done_tools.iter()
                .map(|done| ContentBlock::ToolResult {
                    tool_use_id: done.id.clone(),
                    content:     done.output.content.clone(),
                    is_error:    done.output.is_error(),
                })
                .collect();

            if !result_blocks.is_empty() {
                messages.push(Message {
                    id:      uuid::Uuid::new_v4().to_string(),
                    role:    Role::User,
                    content: result_blocks,
                });
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

        total_usage.input_tokens       += usage.input_tokens;
        total_usage.output_tokens      += usage.output_tokens;
        total_usage.cache_read_tokens  += usage.cache_read_tokens;
        total_usage.cache_write_tokens += usage.cache_write_tokens;
        iterations += 1;

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
                let escalated = (effective_max_tokens * TOKEN_ESCALATION_FACTOR)
                    .min(131_072); // hard ceiling
                tracing::warn!(
                    iterations, current = effective_max_tokens, escalated,
                    "MaxTokens hit — escalating max_tokens"
                );
                effective_max_tokens = escalated;
                // Don't advance the iteration counter; retry immediately.
                continue;
            } else if token_escalations <= MAX_TOKEN_ESCALATIONS {
                // Subsequent hits: inject a continuation prompt.
                tracing::warn!(
                    iterations, escalation = token_escalations,
                    "MaxTokens hit again — injecting continuation prompt"
                );
                messages.push(Message {
                    id:      uuid::Uuid::new_v4().to_string(),
                    role:    Role::User,
                    content: vec![ContentBlock::Text {
                        text: "Your previous response was truncated due to length limits. \
                               Continue exactly where you left off. Do not repeat what you already said."
                            .to_string(),
                    }],
                });
                continue;
            } else {
                tracing::error!(
                    iterations, escalations = token_escalations,
                    "MaxTokens exhausted after escalation and continuation — aborting"
                );
                return Ok(RunSummary {
                    stop_reason: RunStopReason::MaxTokensExhausted,
                    iterations,
                    usage: total_usage,
                    messages,
                });
            }
        }

        // Reset escalation state on a normal (non-MaxTokens) turn.
        token_escalations    = 0;
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
            if stop_reason == StopReason::ToolUse || usage.output_tokens >= MIN_USEFUL_OUTPUT_TOKENS {
                low_output_streak = 0;
            } else {
                low_output_streak += 1;
                if low_output_streak >= MAX_LOW_OUTPUT_TURNS {
                    tracing::warn!(
                        iterations, streak = low_output_streak,
                        "diminishing returns: stopping after {} low-output turns", low_output_streak
                    );
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

            if let HookDecision::Block { reason } = config.hooks.pre_complete(last_text).await {
                messages.push(system_reminder_msg(&reason));
                continue;
            }

            tracing::info!(
                iterations,
                input_tokens  = total_usage.input_tokens,
                output_tokens = total_usage.output_tokens,
                "agent run completed"
            );
            return Ok(RunSummary {
                stop_reason: RunStopReason::Completed,
                iterations,
                usage:    total_usage,
                messages,
            });
        }
        // stop_reason == ToolUse → loop with tool results appended.
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
    req:    &ChatRequest,
    tx:     &mpsc::Sender<AgentEvent>,
) -> Result<
    std::pin::Pin<Box<dyn futures::Stream<Item = Result<wuhu_core::event::StreamEvent, wuhu_core::provider::ProviderError>> + Send>>,
    AgentError,
> {
    use wuhu_core::provider::ProviderError;
    let mut attempt = 0u32;
    loop {
        match config.provider.stream(req.clone()).await {
            Ok(stream) => return Ok(stream),

            Err(e) if e.is_retryable() && attempt < config.retry.max_attempts => {
                attempt += 1;

                // Use the provider's Retry-After hint for rate-limit errors;
                // fall back to the policy's formula for everything else.
                let retry_after_ms = match &e {
                    ProviderError::RateLimit { retry_after_ms } => Some(*retry_after_ms),
                    _ => None,
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
                }).await.ok();
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }

            Err(e) => return Err(AgentError { message: e.to_string(), retryable: false }),
        }
    }
}

/// Emit the appropriate AgentEvent for a completed tool.
///
/// Events are emitted in completion order (real-time UI feedback).
/// The history write uses submission order — see the ordering section above.
async fn emit_tool_event(done: &CompletedTool, tx: &mpsc::Sender<AgentEvent>) {
    let event = if let Some(kind) = &done.output.failure {
        tracing::debug!(tool = %done.name, kind = ?kind, ms = done.ms, "tool failed");
        AgentEvent::ToolError {
            id:    done.id.clone(),
            name:  done.name.clone(),
            error: done.output.content.clone(),
            kind:  kind.clone(),
            ms:    done.ms,
        }
    } else {
        tracing::debug!(tool = %done.name, ms = done.ms, "tool succeeded");
        AgentEvent::ToolDone {
            id:     done.id.clone(),
            name:   done.name.clone(),
            output: done.output.content.clone(),
            ms:     done.ms,
        }
    };
    tx.send(event).await.ok();
}

/// Create an immediately-completed failed tool (no execution, no timer).
fn instant_failure(id: String, name: String, output: wuhu_core::tool::ToolOutput) -> CompletedTool {
    CompletedTool { id, name, output, ms: 0 }
}

/// Build a system-reminder `Message` from plain text.
///
/// Used to inject framework-level notices into the conversation so the LLM
/// can see them without mistaking them for user input.
fn system_reminder_msg(text: &str) -> Message {
    Message {
        id:      uuid::Uuid::new_v4().to_string(),
        role:    Role::System,
        content: vec![ContentBlock::Text { text: wuhu_core::fmt::system_reminder(text) }],
    }
}

/// Verify that a tool call is permitted to run.
///
/// Handles pre-tool hooks (which may mutate the input), session permission
/// memory, and the full permission dance (possibly suspending for human
/// approval).
///
/// Returns `Ok(effective_input)` when the tool is cleared to execute — the
/// input may differ from the original if a `PreToolUse` hook mutated it.
/// Returns `Err(CompletedTool)` with an instant failure when blocked or denied.
async fn authorize_tool(
    config:   &RunConfig,
    id:       String,
    name:     String,
    input:    serde_json::Value,
    tx:       &mpsc::Sender<AgentEvent>,
    messages: &mut Vec<Message>,
) -> Result<serde_json::Value, CompletedTool> {
    // 1. Pre-tool hook — may allow, mutate, or block.
    let input = match config.hooks.pre_tool_use(&name, &input).await {
        HookDecision::Block { reason } =>
            return Err(instant_failure(id, name, wuhu_core::tool::ToolOutput::hook_blocked(reason))),
        HookDecision::Mutate { input: new } => new,
        HookDecision::Allow => input,
    };

    // 2. Session memory — skip the full dance for known decisions.
    if config.session_perms.is_always_denied(&name).await {
        return Err(instant_failure(
            id, name.clone(),
            wuhu_core::tool::ToolOutput::permission_denied(
                format!("tool '{name}' was previously denied for this session")
            ),
        ));
    }
    if config.session_perms.is_always_allowed(&name).await {
        return Ok(input);
    }

    // 3. Evaluate permission mode.
    let tool_is_readonly = config.tools.get(&name).is_some_and(|t| t.is_readonly());
    let ctrl_req = ControlRequest {
        id:   uuid::Uuid::new_v4().to_string(),
        kind: ControlKind::PermissionRequest {
            tool_name:   name.clone(),
            description: format!("call {name}"),
        },
    };

    match permission::check(&config.permission, &ctrl_req, tool_is_readonly) {
        PermissionOutcome::Allowed           => Ok(input),
        PermissionOutcome::Denied { reason } =>
            Err(instant_failure(id, name, wuhu_core::tool::ToolOutput::permission_denied(reason))),
        PermissionOutcome::NeedsApproval     =>
            await_approval(config, id, name, input, ctrl_req, tx, messages).await,
    }
}

/// Suspend the loop, emit a `ControlHandle`, and wait for the human's decision.
///
/// Extracted from `authorize_tool` so each function has a single clear job.
async fn await_approval(
    config:   &RunConfig,
    id:       String,
    name:     String,
    input:    serde_json::Value,
    ctrl_req: ControlRequest,
    tx:       &mpsc::Sender<AgentEvent>,
    messages: &mut Vec<Message>,
) -> Result<serde_json::Value, CompletedTool> {
    let request_id   = ctrl_req.id.clone();
    let (handle, rx) = ControlHandle::new(ctrl_req);
    tx.send(AgentEvent::Control(handle)).await.ok();

    let response = rx.await.unwrap_or_else(|_| {
        ControlResponse::deny(request_id, "control handle dropped before response was sent")
    });

    // Inject the human's decision as a system message — the LLM sees it.
    // `response_to_system_message` already formats the modification note
    // (if any) as part of the approval text; no extra handling needed.
    messages.push(system_reminder_msg(&permission::response_to_system_message(&response)));

    match response.decision {
        ControlDecision::Deny { reason } =>
            Err(instant_failure(id, name, wuhu_core::tool::ToolOutput::permission_denied(reason))),

        ControlDecision::DenyAlways { reason } => {
            config.session_perms.set_always_deny(name.clone()).await;
            Err(instant_failure(id, name, wuhu_core::tool::ToolOutput::permission_denied(reason)))
        }

        ControlDecision::ApproveAlways => {
            config.session_perms.set_always_allow(name).await;
            Ok(input)
        }

        // The modification note (if any) is already in the system message above.
        ControlDecision::Approve { .. } => Ok(input),
    }
}

/// Extract the text content of the most recent assistant message.
fn last_assistant_text(messages: &[Message]) -> &str {
    messages.iter().rev()
        .find_map(|m| {
            if m.role != Role::Assistant { return None; }
            m.content.iter().find_map(|b| {
                if let ContentBlock::Text { text } = b { Some(text.as_str()) } else { None }
            })
        })
        .unwrap_or("")
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

    let listing = deferred.iter()
        .map(|e| match &e.hint {
            Some(h) => format!("- **{}**: {} ({})", e.name, e.description, h),
            None    => format!("- **{}**: {}",       e.name, e.description),
        })
        .collect::<Vec<_>>()
        .join("\n");

    let section = format!(
        "## Additional tools\n\
        These tools are available but require loading. \
        Call `ToolSearch` with the tool name or a keyword before using them:\n\n\
        {listing}"
    );

    format!("{base}\n\n{}", wuhu_core::fmt::system_reminder(&section))
}
