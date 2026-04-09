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
use tracing::Instrument as _;

use wui_core::event::AgentEvent;
use wui_core::message::Message;
use wui_core::tool::ToolCallId;
use wui_core::tool::{Tool, ToolCtx, ToolOutput};

pub use wui_core::tool::ExecutorHints;

use super::registry::ToolRegistry;

// ── Input types ───────────────────────────────────────────────────────────────

/// A tool call received from the LLM, ready for validation and execution.
pub struct PendingTool {
    /// Unique identifier assigned by the LLM for this tool call.
    pub id: ToolCallId,
    /// Tool name as declared in the schema.
    pub name: String,
    /// JSON arguments produced by the LLM.
    pub input: serde_json::Value,
    /// Snapshot of the conversation at submission time.
    /// `Arc` so the snapshot is shared across concurrent tools without copying.
    pub messages: Arc<[Message]>,
}

/// A tool call that has finished executing.
pub struct CompletedTool {
    /// Unique identifier matching the original `ToolUse` block.
    pub id: ToolCallId,
    /// Tool name as declared in the schema.
    pub name: String,
    /// The tool's output (content, artifacts, injections, etc.).
    pub output: ToolOutput,
    /// Wall-clock execution time in milliseconds.
    pub ms: u64,
    /// Number of invocations that were made.
    ///
    /// `1` means the first attempt succeeded. `2` means the tool was retried
    /// once, etc. See `Tool::max_retries()`.
    pub attempts: u32,
}

impl CompletedTool {
    /// Create an immediately-completed tool (zero duration, single attempt).
    ///
    /// Used for permission denials, hook blocks, and malformed-input rejections
    /// where no actual execution takes place.
    pub(crate) fn immediate(id: ToolCallId, name: String, output: ToolOutput) -> Self {
        Self {
            id,
            name,
            output,
            ms: 0,
            attempts: 1,
        }
    }
}

/// A validated tool waiting in the sequential queue.
struct QueuedTool {
    id: ToolCallId,
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
    /// Child of `cancel` — fires when a concurrent tool errors to abort
    /// sibling tasks without cancelling the entire run.
    sibling_cancel: CancellationToken,
    /// Channel to the caller's event stream — used for `ToolProgress` events.
    tx: mpsc::Sender<AgentEvent>,
    /// Global tool timeout, applied when a tool doesn't declare its own.
    default_timeout: Option<Duration>,
    /// Optional store for persisting large tool results before truncation.
    result_store: Option<Arc<dyn ResultStore>>,
    pending: JoinSet<(ToolCallId, String, ToolOutput, u64, u32)>,
    sequential: VecDeque<QueuedTool>,
}

impl ToolExecutor {
    pub fn new(
        registry: Arc<ToolRegistry>,
        cancel: CancellationToken,
        tx: mpsc::Sender<AgentEvent>,
        default_timeout: Option<Duration>,
        result_store: Option<Arc<dyn ResultStore>>,
    ) -> Self {
        let sibling_cancel = cancel.child_token();
        Self {
            registry,
            cancel,
            sibling_cancel,
            tx,
            default_timeout,
            result_store,
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
        let mut results = self.drain_concurrent().await;
        results.extend(self.run_sequential().await);
        results
    }

    /// Drain the JoinSet of concurrent tasks, handling panics and triggering
    /// sibling cancellation when a tool fails with a retryable error (e.g., if
    /// `bash "npm test"` fails, don't also run `bash "npm build"`).
    async fn drain_concurrent(&mut self) -> Vec<CompletedTool> {
        let mut results = Vec::new();
        while let Some(res) = self.pending.join_next().await {
            match res {
                Ok((id, name, output, ms, attempts)) => {
                    if output.is_retryable() {
                        tracing::debug!(tool = %name, "concurrent tool failed — cancelling siblings");
                        self.sibling_cancel.cancel();
                    }
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
        results
    }

    /// Run sequential tools in submission order, one at a time.
    async fn run_sequential(&mut self) -> Vec<CompletedTool> {
        let mut results = Vec::new();
        while let Some(tool) = self.sequential.pop_front() {
            tracing::info!(
                tool.id = %tool.id,
                tool.name = %tool.name,
                tool.concurrent = false,
                "wui.tool.exec.start"
            );
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
            let out_cfg = OutputConfig {
                tool_id: &tool.id,
                max_chars: hints.max_output_chars,
                max_retries: hints.max_retries,
                result_store: self.result_store.as_deref(),
            };
            let (out, attempts) =
                run_with_retries(tool.impl_, tool.input, &ctx, timeout, &out_cfg).await;
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
        id: ToolCallId,
        name: String,
        input: serde_json::Value,
        messages: Arc<[Message]>,
        impl_: Arc<dyn Tool>,
    ) {
        let cancel = self.sibling_cancel.clone();
        let tx = self.tx.clone();
        let hints = impl_.executor_hints(&input);
        let timeout = hints.timeout.or(self.default_timeout);
        let result_store = self.result_store.clone();
        let span = tracing::info_span!(
            "wui.tool.exec",
            tool.id = %id,
            tool.name = %name,
            tool.concurrent = true,
        );
        self.pending.spawn(
            async move {
                let start = Instant::now();
                let ctx = make_ctx(cancel, messages, tx, id.clone(), name.clone());
                let out_cfg = OutputConfig {
                    tool_id: &id,
                    max_chars: hints.max_output_chars,
                    max_retries: hints.max_retries,
                    result_store: result_store.as_deref(),
                };
                let (out, attempts) = run_with_retries(impl_, input, &ctx, timeout, &out_cfg).await;
                let ms = start.elapsed().as_millis() as u64;
                (id, name, out, ms, attempts)
            }
            .instrument(span),
        );
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Per-invocation output-handling context for retries and truncation.
struct OutputConfig<'a> {
    tool_id: &'a ToolCallId,
    max_chars: Option<usize>,
    max_retries: u32,
    result_store: Option<&'a dyn ResultStore>,
}

/// Run a tool with automatic retries.
///
/// Executes `run_tool_safely` up to `1 + max_retries` times. A retry is
/// attempted when the output is an error and retries remain. An exponential
/// backoff delay is inserted between attempts: 50 ms, 100 ms, 200 ms, etc.
///
/// Returns the final `ToolOutput` and the number of attempts made.
async fn run_with_retries(
    impl_: Arc<dyn Tool>,
    input: serde_json::Value,
    ctx: &ToolCtx,
    timeout: Option<Duration>,
    out_cfg: &OutputConfig<'_>,
) -> (ToolOutput, u32) {
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        // call() takes ownership; clone is required for retry.
        let out = apply_output_limit(
            run_tool_safely(Arc::clone(&impl_), input.clone(), ctx, timeout).await,
            out_cfg.max_chars,
            out_cfg.tool_id,
            out_cfg.result_store,
        )
        .await;
        let retries_left = out_cfg.max_retries.saturating_sub(attempt - 1);
        if retries_left == 0 || !out.is_retryable() {
            return (out, attempt);
        }
        let delay_ms = 50u64 * (1u64 << attempt.saturating_sub(1));
        tracing::debug!(
            tool     = %impl_.name(),
            attempt,
            max      = 1 + out_cfg.max_retries,
            delay_ms,
            "tool output retryable — waiting before retry"
        );
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
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

// ── Result Store ─────────────────────────────────────────────────────────────

/// Persists large tool outputs so the LLM receives a compact preview
/// instead of the full content.
///
/// When a tool result exceeds `max_output_chars`, the executor saves the
/// complete content via `ResultStore::save` and replaces the LLM-facing
/// output with a preview + the returned reference (path, URI, etc.).
///
/// # Example
///
/// ```rust,ignore
/// struct FileResultStore { dir: PathBuf }
///
/// #[async_trait::async_trait]
/// impl ResultStore for FileResultStore {
///     async fn save(&self, tool_id: &ToolCallId, content: &str) -> Option<String> {
///         let path = self.dir.join(format!("{tool_id}.txt"));
///         tokio::fs::write(&path, content).await.ok()?;
///         Some(path.display().to_string())
///     }
/// }
/// ```
#[async_trait::async_trait]
pub trait ResultStore: Send + Sync + 'static {
    /// Save the full tool output and return a reference string (path, URI, key).
    ///
    /// Returns `None` if the save fails — the executor falls back to simple
    /// truncation. The returned reference is included in the truncation notice
    /// so the LLM (or a subsequent tool) can retrieve the full content.
    async fn save(&self, tool_id: &ToolCallId, content: &str) -> Option<String>;
}

/// Truncate tool output that exceeds the tool's declared character limit.
///
/// When a `ResultStore` is configured, the full content is persisted before
/// truncation and the notice includes the store reference. Without a store,
/// content is simply cut and a truncation notice is appended.
async fn apply_output_limit(
    mut output: ToolOutput,
    limit: Option<usize>,
    tool_id: &ToolCallId,
    result_store: Option<&dyn ResultStore>,
) -> ToolOutput {
    let Some(max) = limit else { return output };
    if output.content.len() <= max {
        return output;
    }

    // Persist the full content if a store is available.
    let reference = if let Some(store) = result_store {
        store.save(tool_id, &output.content).await
    } else {
        None
    };

    // Find the last valid UTF-8 boundary at or before `max` bytes.
    let cut = (0..=max)
        .rev()
        .find(|&i| output.content.is_char_boundary(i))
        .unwrap_or(0);
    output.content.truncate(cut);

    match reference {
        Some(path) => {
            output.content.push_str(&format!(
                "\n[output truncated: exceeded {max} character limit — full output saved to: {path}]"
            ));
        }
        None => {
            output.content.push_str(&format!(
                "\n[output truncated: exceeded {max} character limit declared by this tool]"
            ));
        }
    }

    output.truncated = true;
    output
}

fn make_ctx(
    cancel: CancellationToken,
    messages: Arc<[Message]>,
    tx: mpsc::Sender<AgentEvent>,
    tool_id: ToolCallId,
    tool_name: String,
) -> ToolCtx {
    // Clone once here so the closure captures owned values rather than
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
