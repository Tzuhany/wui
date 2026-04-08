// ============================================================================
// Streaming Concurrent Tool Executor
//
// The executor's job: start tools as early as possible and harvest results
// as eagerly as possible — all while the LLM is still streaming.
//
// Two improvements over the naive "collect all tool calls then execute" approach:
//
// 1. STREAMING EXECUTION
//    submit() is called the moment the LLM finishes describing a tool call
//    (ToolUseEnd event). The tool starts running immediately — in parallel
//    with the LLM's remaining output and other tool calls.
//
// 2. PER-CALL CONCURRENCY
//    is_concurrent_for(input) is consulted for each invocation. A tool can
//    declare itself concurrent for some inputs and sequential for others.
//    Sequential tools wait in a VecDeque until all concurrent work finishes.
//
// Timeline:
//
//   LLM streaming: ──token──ToolA start──token──ToolB start──token──done──
//   ToolA:                  ├──────────────────────┤
//   ToolB:                                 ├───────────────┤
//   poll_completed():                      ↑harvest        ↑harvest
//   collect_remaining():                                           ├─done─┤
//
//   vs naive:
//   LLM streaming: ──token──ToolA start──token──ToolB start──token──done──
//   ToolA:                                                          ├──────┤
//   ToolB:                                                                 ├──┤
//
// ── Arc<[Message]> history ────────────────────────────────────────────────────
//
// Tool invocations receive the conversation history via `Arc<[Message]>`.
// Every tool in a concurrent batch shares the same Arc — no copying, one
// allocation. The history is read-only for tools; they cannot modify it.
// ============================================================================

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::FutureExt as _;
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use wui_core::event::AgentEvent;
use wui_core::message::Message;
use wui_core::tool::{Tool, ToolCtx, ToolOutput};

pub use wui_core::tool::ExecutorHints;

use super::registry::ToolRegistry;

// ── Input types ───────────────────────────────────────────────────────────────

/// A tool call received from the LLM, ready for validation and execution.
pub struct PendingTool {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
    /// Snapshot of the conversation at submission time.
    /// `Arc` so the snapshot is shared across concurrent tools without copying.
    pub messages: Arc<[Message]>,
}

/// A tool call that has finished executing.
pub struct CompletedTool {
    pub id: String,
    pub name: String,
    pub output: ToolOutput,
    pub ms: u64,
    /// Number of invocations that were made.
    ///
    /// `1` means the first attempt succeeded. `2` means the tool was retried
    /// once, etc. See `Tool::max_retries()`.
    // Exposed as public API for hooks and test harnesses; not read internally.
    #[allow(dead_code)]
    pub attempts: u32,
}

/// A validated tool waiting in the sequential queue.
struct QueuedTool {
    id: String,
    name: String,
    input: serde_json::Value,
    messages: Arc<[Message]>,
    impl_: Arc<dyn Tool>,
}

// ── Executor ──────────────────────────────────────────────────────────────────

/// Executes tools concurrently while the LLM streams.
pub struct ToolExecutor {
    registry: Arc<ToolRegistry>,
    cancel: CancellationToken,
    /// Channel to the caller's event stream — used for `ToolProgress` events.
    tx: mpsc::Sender<AgentEvent>,
    /// Global tool timeout, applied when a tool doesn't declare its own.
    default_timeout: Option<Duration>,
    pending: JoinSet<(String, String, ToolOutput, u64, u32)>,
    sequential: VecDeque<QueuedTool>,
}

impl ToolExecutor {
    pub fn new(
        registry: Arc<ToolRegistry>,
        cancel: CancellationToken,
        tx: mpsc::Sender<AgentEvent>,
        default_timeout: Option<Duration>,
    ) -> Self {
        Self {
            registry,
            cancel,
            tx,
            default_timeout,
            pending: JoinSet::new(),
            sequential: VecDeque::new(),
        }
    }

    /// Submit a tool for execution.
    ///
    /// Called immediately when `ToolUseEnd` is received — the LLM may still
    /// be streaming. Steps:
    ///   1. Look up the tool in the registry.
    ///   2. Validate input against the tool's JSON Schema.
    ///   3. Check `is_concurrent_for(input)`.
    ///   4. Spawn immediately (concurrent) or enqueue (sequential).
    pub fn submit(&mut self, pending: PendingTool) {
        let PendingTool {
            id,
            name,
            input,
            messages,
        } = pending;

        let Some(impl_) = self.registry.get(&name) else {
            let out = ToolOutput::not_found(format!("unknown tool: {name}"));
            self.pending
                .spawn(async move { (id, name, out, 0u64, 1u32) });
            return;
        };

        if let Err(errors) = validate_input(&input, &impl_.input_schema()) {
            let msg = format!("invalid input for '{name}': {}", errors.join("; "));
            tracing::warn!(tool = %name, %msg, "input validation failed");
            let out = ToolOutput::invalid_input(msg);
            self.pending
                .spawn(async move { (id, name, out, 0u64, 1u32) });
            return;
        }

        if impl_.meta(&input).concurrent {
            self.spawn_concurrent(id, name, input, messages, impl_);
        } else {
            self.sequential.push_back(QueuedTool {
                id,
                name,
                input,
                messages,
                impl_,
            });
        }
    }

    /// Wait for all remaining concurrent tools, then run sequential tools.
    ///
    /// Called after all tools have been authorized and submitted. Every
    /// submitted tool is accounted for before returning.
    pub async fn collect_remaining(mut self) -> Vec<CompletedTool> {
        let mut results = Vec::new();

        // Drain concurrent tasks first.
        while let Some(res) = self.pending.join_next().await {
            match res {
                Ok((id, name, output, ms, attempts)) => {
                    results.push(CompletedTool {
                        id,
                        name,
                        output,
                        ms,
                        attempts,
                    });
                }
                Err(e) => tracing::error!(error = %e, "tool task panicked"),
            }
        }

        // Run sequential tools in submission order.
        while let Some(tool) = self.sequential.pop_front() {
            let start = Instant::now();
            let hints = tool.impl_.executor_hints(&tool.input);
            let timeout = hints.timeout.or(self.default_timeout);
            let ctx = make_ctx(
                self.cancel.clone(),
                tool.messages,
                self.tx.clone(),
                tool.id.clone(),
                tool.name.clone(),
            );
            let (out, attempts) = run_with_retries(
                tool.impl_,
                tool.input,
                &ctx,
                timeout,
                hints.max_output_chars,
                hints.max_retries,
            )
            .await;
            let ms = start.elapsed().as_millis() as u64;
            results.push(CompletedTool {
                id: tool.id,
                name: tool.name,
                output: out,
                ms,
                attempts,
            });
        }

        results
    }

    fn spawn_concurrent(
        &mut self,
        id: String,
        name: String,
        input: serde_json::Value,
        messages: Arc<[Message]>,
        impl_: Arc<dyn Tool>,
    ) {
        let cancel = self.cancel.clone();
        let tx = self.tx.clone();
        let hints = impl_.executor_hints(&input);
        let timeout = hints.timeout.or(self.default_timeout);
        self.pending.spawn(async move {
            let start = Instant::now();
            let ctx = make_ctx(cancel, messages, tx, id.clone(), name.clone());
            let (out, attempts) = run_with_retries(
                impl_,
                input,
                &ctx,
                timeout,
                hints.max_output_chars,
                hints.max_retries,
            )
            .await;
            let ms = start.elapsed().as_millis() as u64;
            (id, name, out, ms, attempts)
        });
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Run a tool with automatic retries.
///
/// Executes `run_tool_safely` up to `1 + max_retries` times. A retry is
/// attempted when the output is an error and retries remain. A 50 ms fixed
/// delay is inserted between attempts.
///
/// Returns the final `ToolOutput` and the number of attempts made.
async fn run_with_retries(
    impl_: Arc<dyn Tool>,
    input: serde_json::Value,
    ctx: &ToolCtx,
    timeout: Option<Duration>,
    max_chars: Option<usize>,
    max_retries: u32,
) -> (ToolOutput, u32) {
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        let out = apply_output_limit(
            run_tool_safely(Arc::clone(&impl_), input.clone(), ctx, timeout).await,
            max_chars,
        );
        let retries_left = max_retries.saturating_sub(attempt - 1);
        if retries_left == 0 || !out.is_error() {
            return (out, attempt);
        }
        tracing::debug!(
            tool     = %impl_.name(),
            attempt,
            max      = 1 + max_retries,
            "tool output retryable — waiting 50 ms before retry"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Execute a tool, converting a panic or timeout into an error result.
///
/// A panicking tool must not leave a missing ToolResult in the history —
/// that would cause a 400 on the next API call. `catch_unwind` ensures
/// every submitted tool produces *some* output, even if it crashes.
///
/// When `timeout` is set and the tool exceeds it, a `"tool timed out"` error
/// is returned immediately and the underlying future is dropped.
async fn run_tool_safely(
    impl_: Arc<dyn Tool>,
    input: serde_json::Value,
    ctx: &ToolCtx,
    timeout: Option<Duration>,
) -> ToolOutput {
    let fut = std::panic::AssertUnwindSafe(impl_.call(input, ctx)).catch_unwind();
    let result = if let Some(d) = timeout {
        tokio::time::timeout(d, fut)
            .await
            .unwrap_or_else(|_| Ok(ToolOutput::error("tool timed out")))
    } else {
        fut.await
    };
    result.unwrap_or_else(|_| ToolOutput::error("tool panicked unexpectedly"))
}

/// Truncate tool output that exceeds the tool's declared character limit.
///
/// When a tool exceeds `max_output_chars()`, the content is cut at the
/// nearest UTF-8 character boundary and a truncation notice is appended.
/// This prevents a single large result from consuming disproportionate context.
///
/// The truncation is always visible to the LLM — it sees the notice and can
/// decide whether to request a narrower result rather than silently receiving
/// an incomplete one.
fn apply_output_limit(mut output: ToolOutput, limit: Option<usize>) -> ToolOutput {
    let Some(max) = limit else { return output };
    if output.content.len() <= max {
        return output;
    }

    // Find the last valid UTF-8 boundary at or before `max` bytes.
    let cut = (0..=max)
        .rev()
        .find(|&i| output.content.is_char_boundary(i))
        .unwrap_or(0);
    output.content.truncate(cut);
    output.content.push_str(&format!(
        "\n[output truncated: exceeded {max} character limit declared by this tool]"
    ));
    output.truncated = true;
    output
}

fn make_ctx(
    cancel: CancellationToken,
    messages: Arc<[Message]>,
    tx: mpsc::Sender<AgentEvent>,
    tool_id: String,
    tool_name: String,
) -> ToolCtx {
    // Clone once here so the closure captures owned Strings rather than
    // cloning on every ctx.report() call.
    let id_cap = tool_id.clone();
    let name_cap = tool_name.clone();
    ToolCtx {
        cancel,
        messages,
        on_progress: Box::new(move |text: String| {
            tracing::debug!(tool = %name_cap, progress = %text);
            // Non-blocking send — a full channel drops the event rather than
            // blocking the tool. Progress is best-effort.
            let _ = tx.try_send(AgentEvent::ToolProgress {
                tool_id: id_cap.clone(),
                tool_name: name_cap.clone(),
                text,
            });
        }),
    }
}

fn validate_input(
    input: &serde_json::Value,
    schema: &serde_json::Value,
) -> Result<(), Vec<String>> {
    let validator = match jsonschema::validator_for(schema) {
        Ok(v) => v,
        Err(e) => {
            // The tool's own schema is malformed — this is a tool author bug.
            // Surface it as a validation failure so the LLM and caller both
            // learn about it, rather than silently skipping validation and
            // letting mismatched inputs reach `call()`.
            tracing::error!(error = %e, "tool has invalid JSON Schema — rejecting call");
            return Err(vec![format!("tool schema is invalid: {e}")]);
        }
    };

    let errors: Vec<String> = validator
        .iter_errors(input)
        .map(|e| {
            let path = e.instance_path.to_string();
            if path.is_empty() {
                e.to_string()
            } else {
                format!("{path}: {e}")
            }
        })
        .collect();

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
