// ============================================================================
// The Loop — the beating heart of Wuhu.
//
// "The framework is an executor, not a thinker."
// The LLM decides what to do. This loop does it, feeds results back,
// and repeats until the LLM says it's done.
//
// Structure of one iteration:
//   1. Check context pressure → compress if needed.
//   2. Build ChatRequest.
//   3. Call provider.stream() → receive StreamEvents.
//   4. Process stream:
//        - TextDelta / ThinkingDelta → emit AgentEvent immediately
//        - ToolUseEnd → executor.submit() starts the tool NOW
//        - poll executor during stream → harvest completed tools early
//        - MessageEnd → break inner loop
//   5. collect_remaining() — await any still-running tools.
//   6. Append tool results as a single user message (closes the loop).
//   7. Run hooks (PreComplete).
//   8. Evaluate stop condition.
//   9. Continue or return RunSummary.
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
    RunStopReason, RunSummary, StopReason, TokenUsage,
};
use wuhu_core::hook::HookDecision;
use wuhu_core::message::{ContentBlock, Message, Role};
use wuhu_core::provider::{ChatRequest, Provider};
use wuhu_core::tool::{SpawnFn, ToolOutput};

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
/// initial delay doubling on each attempt, capped at 10 s.
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
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts:     3,
            initial_delay_ms: 500,
            multiplier:       2.0,
            max_delay_ms:     10_000,
        }
    }
}

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

    let mut total_usage = TokenUsage::default();
    let mut iterations  = 0u32;

    loop {
        // ── Cancellation ───────────────────────────────────────────
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

        // ── Context compression ────────────────────────────────────
        if let Some((compressed, method, freed)) = config.compress.maybe_compress(
            &messages,
            config.provider.as_ref(),
            &config.model,
        ).await {
            tracing::debug!(iteration = iterations, method = ?method, freed_tokens = freed, "context compressed");
            messages = compressed;
            tx.send(AgentEvent::Compressed { method, freed }).await.ok();
        }

        // ── Build request ──────────────────────────────────────────
        // Merge initial_extensions into extensions on the first turn only.
        // This lets callers force tool_choice (or similar) on turn 0 without
        // inadvertently looping on every continuation turn.
        let extensions = if iterations == 0 && !config.initial_extensions.is_empty() {
            let mut ext = config.extensions.clone();
            ext.extend(config.initial_extensions.clone());
            ext
        } else {
            config.extensions.clone()
        };

        let req = ChatRequest {
            model:       config.model.clone(),
            max_tokens:  config.max_tokens,
            temperature: config.temperature,
            system:      system.clone(),
            messages:    messages.clone(),
            tools:       config.tools.tool_defs(),
            extensions,
        };

        // ── Call provider with retry ───────────────────────────────
        let stream = {
            let mut delay_ms = config.retry.initial_delay_ms;
            let mut attempt  = 0u32;
            loop {
                match config.provider.stream(req.clone()).await {
                    Ok(s) => break s,
                    Err(e) if e.is_retryable() && attempt < config.retry.max_attempts => {
                        attempt += 1;
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
                        delay_ms = (delay_ms as f64 * config.retry.multiplier)
                            .min(config.retry.max_delay_ms as f64) as u64;
                    }
                    Err(e) => return Err(AgentError { message: e.to_string(), retryable: false }),
                }
            }
        };

        let mut executor = ToolExecutor::new(
            config.tools.clone(),
            cancel.clone(),
            config.spawn.clone(),
            tx.clone(),
            config.tool_timeout,
        );

        // Accumulate tool input JSON chunks keyed by tool_use_id.
        let mut pending_inputs:   HashMap<String, (String, String)> = HashMap::new();
        let mut assistant_blocks: Vec<ContentBlock>                  = Vec::new();
        // Track submission order for ordered history reconstruction.
        let mut submission_order: Vec<String>                        = Vec::new();
        let mut completed_map:    HashMap<String, CompletedTool>     = HashMap::new();
        let mut text_buf     = String::new();
        let mut thinking_buf = String::new();
        let mut stop_reason  = StopReason::EndTurn;
        let mut usage        = TokenUsage::default();

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
                    if let Some((name, json)) = pending_inputs.remove(&id) {
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

                        match authorize_tool(&config, id.clone(), name.clone(), &input, tx, &mut messages).await {
                            Err(denied) => {
                                emit_tool_event(&denied, tx).await;
                                completed_map.insert(denied.id.clone(), denied);
                            }
                            Ok(()) => {
                                let history_snap: Arc<[Message]> = messages.as_slice().into();
                                tracing::debug!(tool = %name, "tool dispatched");
                                tx.send(AgentEvent::ToolStart {
                                    id:    id.clone(),
                                    name:  name.clone(),
                                    input: input.clone(),
                                }).await.ok();
                                executor.submit(PendingTool { id, name, input, messages: history_snap });
                            }
                        }
                    }
                }

                MessageEnd { usage: u, stop_reason: sr } => {
                    usage       = u;
                    stop_reason = sr;
                    break;
                }

                wuhu_core::event::StreamEvent::Error { message, retryable } => {
                    return Err(AgentError { message, retryable });
                }
            }
        }

        // ── Collect remaining tools ────────────────────────────────
        for done in executor.collect_remaining().await {
            emit_tool_event(&done, tx).await;
            completed_map.insert(done.id.clone(), done);
        }

        // ── Post-tool hooks ────────────────────────────────────────
        for id in &submission_order {
            if let Some(done) = completed_map.get(id) {
                let _ = config.hooks.post_tool_use(&done.name, &done.output).await;
            }
        }

        // ── Build assistant message ────────────────────────────────
        // Thinking first (so the LLM can reference its reasoning),
        // then text, then tool calls.
        if !thinking_buf.is_empty() {
            assistant_blocks.insert(0, ContentBlock::Thinking { text: thinking_buf });
        }
        if !text_buf.is_empty() {
            assistant_blocks.insert(0, ContentBlock::Text { text: text_buf });
        }
        if !assistant_blocks.is_empty() {
            messages.push(Message::assistant(assistant_blocks));
        }

        // ── Append tool results in submission order ────────────────
        //
        // Results MUST appear in the same order as the ToolUse blocks in the
        // preceding assistant message. Providers validate this; a mismatch
        // causes an API error. We reconstruct submission order from
        // `submission_order` + `completed_map` regardless of completion order.
        if !submission_order.is_empty() {
            // Collect in submission order — drain the map.
            let done_tools: Vec<CompletedTool> = submission_order.iter()
                .filter_map(|id| completed_map.remove(id))
                .collect();

            // Emit artifacts before the ToolResult so callers receive them
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

            // Inject extra messages after the tool results.
            // These are additional context the tools want the LLM to see
            // on the next turn (e.g. injected documentation, memory recall).
            for done in done_tools {
                messages.extend(done.output.extra_messages);
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

        // ── Stop condition ─────────────────────────────────────────
        if stop_reason == StopReason::EndTurn {
            let last_text = last_assistant_text(&messages);

            if let HookDecision::Block { reason } = config.hooks.pre_complete(last_text).await {
                messages.push(Message {
                    id:      uuid::Uuid::new_v4().to_string(),
                    role:    Role::System,
                    content: vec![ContentBlock::Text {
                        text: wuhu_core::fmt::system_reminder(&reason),
                    }],
                });
                continue;
            }

            tracing::info!(
                iterations   = iterations,
                input_tokens = total_usage.input_tokens,
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

/// Emit the appropriate AgentEvent for a completed tool.
///
/// Events are emitted in completion order (real-time UI feedback).
/// The history write uses submission order — see the ordering section above.
async fn emit_tool_event(done: &CompletedTool, tx: &mpsc::Sender<AgentEvent>) {
    let event = if let Some(kind) = &done.output.failure {
        tracing::debug!(
            tool    = %done.name,
            kind    = ?kind,
            ms      = done.ms,
            "tool failed"
        );
        AgentEvent::ToolError {
            id:    done.id.clone(),
            name:  done.name.clone(),
            error: done.output.content.clone(),
            kind:  kind.clone(),
            ms:    done.ms,
        }
    } else {
        tracing::debug!(
            tool = %done.name,
            ms   = done.ms,
            "tool succeeded"
        );
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
fn instant_failure(id: String, name: String, output: ToolOutput) -> CompletedTool {
    CompletedTool { id, name, output, ms: 0 }
}

/// Verify that a tool call is permitted to run.
///
/// Runs pre-tool hooks, session permission checks, and the full permission
/// dance (possibly suspending for human approval). Returns `Ok(())` when
/// the tool is cleared to execute, or `Err(CompletedTool)` with an instant
/// failure record when it is blocked or denied.
///
/// The caller is responsible for emitting the `CompletedTool` event and
/// inserting it into `completed_map` on the `Err` path.
async fn authorize_tool(
    config:   &RunConfig,
    id:       String,
    name:     String,
    input:    &serde_json::Value,
    tx:       &mpsc::Sender<AgentEvent>,
    messages: &mut Vec<Message>,
) -> Result<(), CompletedTool> {
    // Pre-tool hook.
    if let HookDecision::Block { reason } = config.hooks.pre_tool_use(&name, input).await {
        return Err(instant_failure(id, name, ToolOutput::hook_blocked(reason)));
    }

    // Session memory check — short-circuit before the full permission dance.
    if config.session_perms.is_always_denied(&name).await {
        return Err(instant_failure(
            id, name.clone(),
            ToolOutput::permission_denied(format!(
                "tool '{name}' was previously denied for this session"
            )),
        ));
    }

    // If always-allowed, skip the permission check entirely.
    if config.session_perms.is_always_allowed(&name).await {
        return Ok(());
    }

    let tool_is_readonly = config.tools.get(&name)
        .map(|t| t.is_readonly())
        .unwrap_or(false);

    let ctrl_req = ControlRequest {
        id:   uuid::Uuid::new_v4().to_string(),
        kind: ControlKind::PermissionRequest {
            tool_name:   name.clone(),
            description: format!("call {name}"),
        },
    };

    match permission::check(&config.permission, &ctrl_req, tool_is_readonly) {
        PermissionOutcome::Allowed => {}

        PermissionOutcome::Denied { reason } => {
            return Err(instant_failure(id, name, ToolOutput::permission_denied(reason)));
        }

        PermissionOutcome::NeedsApproval => {
            let request_id   = ctrl_req.id.clone();
            let (handle, rx) = ControlHandle::new(ctrl_req);
            tx.send(AgentEvent::Control(handle)).await.ok();

            // Suspend until the human responds.
            let response = rx.await.unwrap_or_else(|_| {
                wuhu_core::event::ControlResponse::deny(
                    request_id,
                    "control handle dropped before response was sent",
                )
            });

            // Inject decision as a system message visible to the LLM.
            let sys = permission::response_to_system_message(&response);
            messages.push(Message {
                id:      uuid::Uuid::new_v4().to_string(),
                role:    Role::System,
                content: vec![ContentBlock::Text { text: sys }],
            });

            match &response.decision {
                ControlDecision::Deny { reason } => {
                    return Err(instant_failure(
                        id, name, ToolOutput::permission_denied(reason.clone()),
                    ));
                }
                ControlDecision::DenyAlways { reason } => {
                    config.session_perms.set_always_deny(name.clone()).await;
                    return Err(instant_failure(
                        id, name, ToolOutput::permission_denied(reason.clone()),
                    ));
                }
                ControlDecision::ApproveAlways => {
                    config.session_perms.set_always_allow(name.clone()).await;
                }
                ControlDecision::Approve { .. } => {}
            }
        }
    }

    Ok(())
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
