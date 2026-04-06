// ============================================================================
// Tool — the agent's hands.
//
// Tools are stateless. They are Send + Sync + 'static so the executor can
// spawn them freely across tokio tasks without lifetime friction.
//
// State lives in ToolCtx, not in the tool. A tool is pure logic: given an
// input and a context, produce an output. That is all.
//
// The is_concurrent_for(input) method is the key design decision that sets
// this trait apart from simpler frameworks: concurrency is a per-invocation
// decision, not a per-tool-type decision. A shell tool, for example, can
// allow read-only commands to run in parallel while serialising writes.
// ============================================================================

use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
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

    /// A short hint for the ToolSearch tool (3-10 words).
    ///
    /// Only relevant when `defer_loading()` is true.
    fn search_hint(&self) -> Option<&str> {
        None
    }

    /// Whether this specific invocation can run concurrently with others.
    ///
    /// The default is `true`. Override to inspect `input` when the safety
    /// decision depends on arguments — e.g. a shell tool that runs read-only
    /// commands concurrently but serialises writes:
    ///
    /// ```rust
    /// fn is_concurrent_for(&self, input: &Value) -> bool {
    ///     let cmd = input["command"].as_str().unwrap_or("");
    ///     is_readonly(cmd)
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
    /// `ctx` provides cancellation, the current message history, and
    /// progress reporting.
    async fn call(&self, input: Value, ctx: &ToolCtx) -> ToolOutput;
}

// ── Tool Context ─────────────────────────────────────────────────────────────

/// Runtime context injected into every tool invocation.
pub struct ToolCtx {
    /// Cancellation signal. Check periodically in long-running tools.
    pub cancel: tokio_util::sync::CancellationToken,

    /// The conversation history at the time of this invocation (read-only).
    /// Useful for tools that need context awareness (e.g. sub-agents).
    pub messages: Vec<Message>,

    /// Report incremental progress to the stream. Each call emits a log
    /// line; the framework forwards these to the caller as trace events.
    pub on_progress: Box<dyn Fn(String) + Send + Sync>,

    /// The sub-agent spawn capability, if configured.
    ///
    /// Present when the framework was built with a sub-agent provider.
    /// `None` for tools that don't need to spawn agents.
    pub spawn: Option<SpawnFn>,
}

impl ToolCtx {
    pub fn report(&self, msg: impl Into<String>) {
        (self.on_progress)(msg.into());
    }

    /// Spawn a sub-agent. Returns `None` if sub-agent capability was not
    /// configured on the enclosing `Agent`.
    pub fn spawn_agent(&self, prompt: impl Into<String>) -> Option<BoxFuture<'static, anyhow::Result<String>>> {
        self.spawn.as_ref().map(|f| f(prompt.into()))
    }
}

/// The closure type used to spawn sub-agents.
///
/// Captured by the engine after `LoopConfig` is constructed and injected
/// into `ToolCtx` at execution time. The `Tool` trait itself never imports
/// engine types — it depends only on `wuhu-core`.
pub type SpawnFn = Arc<dyn Fn(String) -> BoxFuture<'static, anyhow::Result<String>> + Send + Sync>;

// ── Tool Output ───────────────────────────────────────────────────────────────

/// The result of a tool invocation.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    /// The content returned to the LLM.
    pub content:  String,
    /// Whether the tool failed. The LLM sees this and can decide how to
    /// handle the failure — retry, inform the user, or give up.
    pub is_error: bool,
}

impl ToolOutput {
    pub fn success(content: impl Into<String>) -> Self {
        Self { content: content.into(), is_error: false }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self { content: content.into(), is_error: true }
    }

    pub fn is_ok(&self) -> bool {
        !self.is_error
    }
}
