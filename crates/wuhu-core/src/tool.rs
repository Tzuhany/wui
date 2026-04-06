// ============================================================================
// Tool — the agent's hands.
//
// Tools are stateless. They are Send + Sync + 'static so the executor can
// spawn them freely across tokio tasks without lifetime friction.
//
// State lives in ToolCtx, not in the tool. A tool is pure logic: given an
// input and a context, produce an output. That is all.
//
// ── is_concurrent_for(input) ──────────────────────────────────────────────────
//
// Concurrency is a per-invocation decision, not a per-tool-type decision.
// A shell tool, for example, can allow read-only commands to run in parallel
// while serialising writes. The input is inspected at submission time; the
// executor routes accordingly.
//
// ── ToolOutput and FailureKind ────────────────────────────────────────────────
//
// When a tool fails, the kind of failure matters. A schema validation error
// suggests the LLM should retry with corrected arguments. A permission denial
// means it should stop and explain. An execution error might be retried after
// fixing the underlying condition.
//
// FailureKind gives the harness — and the LLM — structured information to
// act on, rather than a generic `is_error: bool`.
// ============================================================================

use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::Message;

/// The interface every tool implements.
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    /// Unique identifier. This is how the LLM names the tool in its output.
    fn name(&self) -> &str;

    /// One-line description shown to the LLM in the tool listing.
    fn description(&self) -> &str;

    /// Full usage instructions. May be loaded lazily via a ToolSearch tool
    /// if `defer_loading()` returns true — only injected into context when
    /// the LLM explicitly requests details.
    fn prompt(&self) -> String;

    /// JSON Schema describing the tool's input parameters.
    ///
    /// The executor validates every invocation against this schema before
    /// calling `call()`. Invalid input produces an immediate error result
    /// that the LLM sees and can self-correct from — `call()` is never
    /// invoked with malformed arguments.
    fn input_schema(&self) -> Value;

    /// Whether to defer this tool's full description until requested.
    ///
    /// When `true`, only `name()` and `description()` appear in the initial
    /// system prompt. The LLM must call `ToolSearch` to fetch `prompt()` and
    /// `input_schema()` before it can use the tool. Useful for large tool
    /// libraries where injecting every schema would consume too many tokens.
    fn defer_loading(&self) -> bool {
        false
    }

    /// A short hint for the ToolSearch tool (3–10 words).
    ///
    /// Only relevant when `defer_loading()` returns `true`.
    fn search_hint(&self) -> Option<&str> {
        None
    }

    /// Whether this tool has no observable side effects.
    ///
    /// Defaults to `false` (assume the tool mutates state). Override to
    /// return `true` for pure read operations: web search, file read, etc.
    ///
    /// Used by `PermissionMode::Readonly`: only tools returning `true` are
    /// permitted; all others are blocked without prompting the user.
    fn is_readonly(&self) -> bool {
        false
    }

    /// Whether this specific invocation can run concurrently with others.
    ///
    /// Defaults to `true`. Override to inspect `input` when the safety
    /// decision depends on arguments — e.g. a shell tool that runs read-only
    /// commands concurrently but serialises writes:
    ///
    /// ```rust,ignore
    /// fn is_concurrent_for(&self, input: &Value) -> bool {
    ///     is_readonly_command(input["command"].as_str().unwrap_or(""))
    /// }
    /// ```
    ///
    /// Tools with unconditional side effects should return `false` without
    /// inspecting `input`.
    #[allow(unused_variables)]
    fn is_concurrent_for(&self, input: &Value) -> bool {
        true
    }

    /// Execute the tool.
    ///
    /// Called only after input has been validated against `input_schema()`.
    /// `ctx` provides the conversation history, a cancellation token, and
    /// a progress reporter.
    async fn call(&self, input: Value, ctx: &ToolCtx) -> ToolOutput;
}

// ── Tool Context ─────────────────────────────────────────────────────────────

/// Runtime context injected into every tool invocation.
pub struct ToolCtx {
    /// Cancellation signal. Check periodically in long-running tools.
    pub cancel: tokio_util::sync::CancellationToken,

    /// The conversation history at the time of this invocation (read-only).
    ///
    /// `Arc<[Message]>` rather than `Vec<Message>`: the history is shared
    /// across all concurrent tool invocations without copying. Clone the Arc
    /// to pass it around cheaply; call `.to_vec()` only if you need mutation.
    pub messages: Arc<[Message]>,

    /// Report incremental progress to the stream.
    ///
    /// Each call emits a log line; the framework forwards it as a trace event.
    pub on_progress: Box<dyn Fn(String) + Send + Sync>,

    /// Sub-agent spawn capability, if configured.
    pub spawn: Option<SpawnFn>,
}

impl ToolCtx {
    pub fn report(&self, msg: impl Into<String>) {
        (self.on_progress)(msg.into());
    }

    /// Spawn a sub-agent. Returns `None` if the capability was not configured.
    pub fn spawn_agent(
        &self,
        prompt: impl Into<String>,
    ) -> Option<BoxFuture<'static, anyhow::Result<String>>> {
        self.spawn.as_ref().map(|f| f(prompt.into()))
    }
}

/// The closure type used to spawn sub-agents.
///
/// Captured by the engine after `RunConfig` is constructed and injected
/// into `ToolCtx` at execution time. The `Tool` trait never imports engine
/// types — it depends only on `wuhu-core`.
pub type SpawnFn =
    Arc<dyn Fn(String) -> BoxFuture<'static, anyhow::Result<String>> + Send + Sync>;

// ── Tool Output ───────────────────────────────────────────────────────────────

/// The result of a tool invocation.
///
/// When the tool succeeds, `failure` is `None`.
/// When it fails, `failure` carries the reason — not just a boolean — so the
/// harness and the LLM can respond appropriately to each failure kind.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    /// The content returned to the LLM (description, result, or error message).
    pub content: String,
    /// `None` on success. `Some(kind)` on failure.
    pub failure: Option<FailureKind>,
}

impl ToolOutput {
    // ── Public constructors ────────────────────────────────────────────────

    /// Successful tool execution.
    pub fn success(content: impl Into<String>) -> Self {
        Self { content: content.into(), failure: None }
    }

    /// Failed execution — the tool ran but produced an error.
    ///
    /// This is the constructor tool implementors use. More specific failures
    /// (schema violations, permission denials) are created by the engine.
    pub fn error(content: impl Into<String>) -> Self {
        Self { content: content.into(), failure: Some(FailureKind::Execution) }
    }

    // ── Engine-internal constructors ───────────────────────────────────────

    /// The tool's input did not satisfy its JSON Schema.
    pub fn invalid_input(content: impl Into<String>) -> Self {
        Self { content: content.into(), failure: Some(FailureKind::InvalidInput) }
    }

    /// The tool was not found in the registry.
    pub fn not_found(content: impl Into<String>) -> Self {
        Self { content: content.into(), failure: Some(FailureKind::NotFound) }
    }

    /// The tool was blocked by a hook before execution.
    pub fn hook_blocked(content: impl Into<String>) -> Self {
        Self { content: content.into(), failure: Some(FailureKind::HookBlocked) }
    }

    /// The tool was denied by the permission system.
    pub fn permission_denied(content: impl Into<String>) -> Self {
        Self { content: content.into(), failure: Some(FailureKind::PermissionDenied) }
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    pub fn is_ok(&self) -> bool {
        self.failure.is_none()
    }

    pub fn is_error(&self) -> bool {
        self.failure.is_some()
    }
}

// ── Failure Kind ─────────────────────────────────────────────────────────────

/// Why a tool invocation failed.
///
/// The harness and the LLM use this to decide how to recover:
///
/// | Kind              | Typical recovery                                     |
/// |-------------------|------------------------------------------------------|
/// | `Execution`       | Surface the error; LLM decides whether to retry     |
/// | `InvalidInput`    | Inject schema hints; LLM retries with correct args  |
/// | `NotFound`        | Inform the LLM; do not retry                         |
/// | `HookBlocked`     | Inject reason; LLM may seek an alternative approach |
/// | `PermissionDenied`| Inject reason; LLM must not retry this tool         |
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    /// The tool executed but produced an error result.
    Execution,
    /// The tool's input failed JSON Schema validation.
    InvalidInput,
    /// The tool name was not found in the registry.
    NotFound,
    /// A hook blocked the tool call before execution.
    HookBlocked,
    /// The permission system denied the tool call.
    PermissionDenied,
}
