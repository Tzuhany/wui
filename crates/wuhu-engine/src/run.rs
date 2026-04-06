// ============================================================================
// The Loop — the beating heart of Wuhu.
//
// Philosophy:
//   "The framework is an executor, not a thinker."
//   The LLM decides what to do. This loop does it, feeds results back,
//   and repeats until the LLM says it's done.
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
//   6. Run hooks (PreComplete).
//   7. Evaluate stop condition.
//   8. If continuing → append tool results, loop.
//
// The loop is a free function that takes a config and cancel token and
// returns a stream of AgentEvents. It spawns a task internally.
// ============================================================================

use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_util::sync::CancellationToken;

use wuhu_core::event::{
    AgentError, AgentEvent, CompressMethod, ControlKind, ControlRequest,
    RunStopReason, RunSummary, StopReason, TokenUsage,
};
use wuhu_core::hook::{HookDecision, HookEvent};
use wuhu_core::message::{ContentBlock, Message, Role};
use wuhu_core::provider::{ChatRequest, Provider};
use wuhu_core::tool::SpawnFn;

use wuhu_compress::CompressPipeline;

use crate::executor::{PendingTool, ToolExecutor};
use crate::hooks::HookRunner;
use crate::permission::{self, PermissionMode, PermissionOutcome};
use crate::registry::ToolRegistry;

// ── Run Config ────────────────────────────────────────────────────────────────

/// Static configuration for one run. `Arc`-wrapped so it is cheap to share
/// across the spawned task and any sub-agent closures.
pub struct RunConfig {
    pub provider:    Arc<dyn Provider>,
    pub tools:       Arc<ToolRegistry>,
    pub hooks:       Arc<HookRunner>,
    pub compress:    CompressPipeline,
    pub permission:  PermissionMode,
    pub system:      String,
    pub model:       String,
    pub max_tokens:  u32,
    pub temperature: Option<f32>,
    pub max_iter:    u32,
    pub extensions:  HashMap<String, serde_json::Value>,
    pub spawn:       Option<SpawnFn>,
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Run the agent loop and return a stream of events.
///
/// Spawns an internal task. The stream yields events until `AgentEvent::Done`
/// or `AgentEvent::Error` is received.
pub fn run(
    config:   Arc<RunConfig>,
    messages: Vec<Message>,
    cancel:   CancellationToken,
) -> impl futures::Stream<Item = AgentEvent> {
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(run_task(config, messages, cancel, tx));
    UnboundedReceiverStream::new(rx)
}

// ── Internal task ─────────────────────────────────────────────────────────────

async fn run_task(
    config:   Arc<RunConfig>,
    messages: Vec<Message>,
    cancel:   CancellationToken,
    tx:       mpsc::UnboundedSender<AgentEvent>,
) {
    let emit = |e: AgentEvent| { let _ = tx.send(e); };

    let result = run_loop(config, messages, cancel.clone(), &emit).await;

    match result {
        Ok(summary)  => emit(AgentEvent::Done(summary)),
        Err(e)       => emit(AgentEvent::Error(e)),
    }
}

// ── Main loop ─────────────────────────────────────────────────────────────────

async fn run_loop(
    config:   Arc<RunConfig>,
    mut messages: Vec<Message>,
    cancel:   CancellationToken,
    emit:     &impl Fn(AgentEvent),
) -> Result<RunSummary, AgentError> {
    let mut total_usage = TokenUsage::default();
    let mut iterations  = 0u32;

    loop {
        // ── Cancellation check ─────────────────────────────────────
        if cancel.is_cancelled() {
            return Ok(RunSummary {
                stop_reason: RunStopReason::Cancelled,
                iterations,
                usage: total_usage,
                messages: messages.iter()
                    .flat_map(|m| m.content.clone())
                    .collect(),
            });
        }

        if iterations >= config.max_iter {
            return Ok(RunSummary {
                stop_reason: RunStopReason::MaxIterations,
                iterations,
                usage: total_usage,
                messages: messages.iter()
                    .flat_map(|m| m.content.clone())
                    .collect(),
            });
        }

        // ── Context compression ────────────────────────────────────
        if let Some((compressed, method, freed)) = config.compress.maybe_compress(
            &messages,
            config.provider.as_ref(),
            &config.model,
        ).await {
            messages = compressed;
            emit(AgentEvent::Compressed { method, freed });
        }

        // ── Build request ──────────────────────────────────────────
        let req = ChatRequest {
            model:       config.model.clone(),
            max_tokens:  config.max_tokens,
            temperature: config.temperature,
            system:      config.system.clone(),
            messages:    messages.clone(),
            tools:       config.tools.tool_defs(),
            extensions:  config.extensions.clone(),
        };

        // ── Stream ────────────────────────────────────────────────
        let stream = config.provider.stream(req).await
            .map_err(|e| AgentError { message: e.to_string(), retryable: e.is_retryable() })?;

        let mut executor = ToolExecutor::new(
            config.tools.clone(),
            cancel.clone(),
            config.spawn.clone(),
        );

        // Accumulate tool input JSON chunks keyed by tool_use_id.
        let mut pending_inputs: HashMap<String, (String, String)> = HashMap::new(); // id → (name, json)
        let mut assistant_blocks: Vec<ContentBlock> = Vec::new();
        let mut text_buf = String::new();
        let mut thinking_buf = String::new();
        let mut stop_reason = StopReason::EndTurn;
        let mut usage = TokenUsage::default();

        futures::pin_mut!(stream);

        while let Some(event) = stream.next().await {
            // Non-blocking harvest: collect tools that finished while LLM streamed.
            for done in executor.poll_completed() {
                handle_completed_tool(done, &config, &messages, emit).await;
            }

            let event = match event {
                Ok(e)  => e,
                Err(e) => {
                    if e.is_retryable() { continue; } // engine will handle retries
                    return Err(AgentError { message: e.to_string(), retryable: false });
                }
            };

            use wuhu_core::event::StreamEvent::*;
            match event {
                TextDelta { text } => {
                    text_buf.push_str(&text);
                    emit(AgentEvent::TextDelta(text));
                }

                ThinkingDelta { text } => {
                    thinking_buf.push_str(&text);
                    emit(AgentEvent::ThinkingDelta(text));
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

                        // Pre-tool hook.
                        let decision = config.hooks.pre_tool_use(&name, &input).await;
                        if let HookDecision::Block { reason } = decision {
                            let blocked = instant_error(id.clone(), name.clone(), reason);
                            handle_completed_tool(blocked, &config, &messages, emit).await;
                            continue;
                        }

                        // Permission check.
                        let req = ControlRequest {
                            id:   uuid::Uuid::new_v4().to_string(),
                            kind: ControlKind::PermissionRequest {
                                tool_name:   name.clone(),
                                description: format!("call {name}"),
                            },
                        };
                        match permission::check(&config.permission, req) {
                            PermissionOutcome::Allowed => {}
                            PermissionOutcome::Denied { reason } => {
                                let blocked = instant_error(id.clone(), name.clone(), reason);
                                handle_completed_tool(blocked, &config, &messages, emit).await;
                                continue;
                            }
                            PermissionOutcome::NeedsApproval(approval, rx) => {
                                emit(AgentEvent::Control(approval.request.clone()));
                                // Suspend until the human responds.
                                let response = rx.await.unwrap_or_else(|_| {
                                    wuhu_core::event::ControlResponse::deny(
                                        approval.request.id.clone(),
                                        "session dropped",
                                    )
                                });
                                approval.respond(response.clone());
                                // Inject the decision as a system message visible to the LLM.
                                let sys = permission::response_to_system_message(&response);
                                messages.push(Message {
                                    id:      uuid::Uuid::new_v4().to_string(),
                                    role:    Role::System,
                                    content: vec![ContentBlock::Text { text: sys }],
                                });
                                // On denial, generate an instant error result.
                                if matches!(response.decision, wuhu_core::event::ControlDecision::Deny { .. }) {
                                    let blocked = instant_error(id.clone(), name.clone(), "denied by user");
                                    handle_completed_tool(blocked, &config, &messages, emit).await;
                                    continue;
                                }
                            }
                        }

                        assistant_blocks.push(ContentBlock::ToolUse {
                            id:    id.clone(),
                            name:  name.clone(),
                            input: input.clone(),
                        });
                        emit(AgentEvent::ToolStart { id: id.clone(), name: name.clone(), input: input.clone() });
                        executor.submit(PendingTool { id, name, input, messages: messages.clone() });
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
        let remaining = executor.collect_remaining().await;
        for done in remaining {
            handle_completed_tool(done, &config, &messages, emit).await;
        }

        // ── Build assistant message ────────────────────────────────
        if !text_buf.is_empty() {
            assistant_blocks.insert(0, ContentBlock::Text { text: text_buf });
        }
        if !thinking_buf.is_empty() {
            assistant_blocks.insert(0, ContentBlock::Thinking { text: thinking_buf });
        }
        if !assistant_blocks.is_empty() {
            messages.push(Message::assistant(assistant_blocks));
        }

        total_usage.input_tokens  += usage.input_tokens;
        total_usage.output_tokens += usage.output_tokens;
        total_usage.cache_read_tokens  += usage.cache_read_tokens;
        total_usage.cache_write_tokens += usage.cache_write_tokens;
        iterations += 1;

        // ── Stop condition ─────────────────────────────────────────
        if stop_reason == StopReason::EndTurn {
            // Pre-complete hook.
            let last_text = messages.iter().rev()
                .find(|m| m.role == Role::Assistant)
                .and_then(|m| m.content.iter().find_map(|b| {
                    if let ContentBlock::Text { text } = b { Some(text.as_str()) } else { None }
                }))
                .unwrap_or("");

            if let HookDecision::Block { reason } = config.hooks.pre_complete(last_text).await {
                // Hook wants a revision. Inject reason and continue the loop.
                messages.push(Message {
                    id:      uuid::Uuid::new_v4().to_string(),
                    role:    Role::System,
                    content: vec![ContentBlock::Text { text: reason }],
                });
                continue;
            }

            return Ok(RunSummary {
                stop_reason: RunStopReason::Completed,
                iterations,
                usage: total_usage,
                messages: messages.iter()
                    .flat_map(|m| m.content.clone())
                    .collect(),
            });
        }
        // stop_reason == ToolUse → continue the loop with tool results appended.
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async fn handle_completed_tool(
    done:     crate::executor::CompletedTool,
    config:   &Arc<RunConfig>,
    messages: &[Message],
    emit:     &impl Fn(AgentEvent),
) {
    // Post-tool hook.
    let _ = config.hooks.post_tool_use(&done.name, &done.output).await;

    if done.output.is_error {
        emit(AgentEvent::ToolError {
            id:    done.id,
            name:  done.name,
            error: done.output.content,
            ms:    done.ms,
        });
    } else {
        emit(AgentEvent::ToolDone {
            id:     done.id,
            name:   done.name,
            output: done.output.content,
            ms:     done.ms,
        });
    }
    let _ = messages; // used by future artifact/tool-result appending
}

fn instant_error(id: String, name: String, reason: impl Into<String>) -> crate::executor::CompletedTool {
    crate::executor::CompletedTool {
        id,
        name,
        output: wuhu_core::tool::ToolOutput::error(reason),
        ms: 0,
    }
}
