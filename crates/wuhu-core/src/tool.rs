// ============================================================================
// Tool — the agent's hands.
//
// Tools are stateless, Send + Sync + 'static. The executor spawns them freely
// across tokio tasks without lifetime friction. State lives in ToolCtx.
//
// ── is_concurrent_for(input) ──────────────────────────────────────────────────
//
// Concurrency is a per-invocation decision. A shell tool can run read-only
// commands in parallel while serialising writes. The input is inspected at
// submission time; the executor routes accordingly.
//
// ── FailureKind ───────────────────────────────────────────────────────────────
//
// A schema error suggests the LLM should retry with corrected arguments.
// A permission denial means it should stop and explain.
// An execution error might be retried after fixing the underlying condition.
//
// FailureKind gives the harness — and the LLM — structured information to
// act on, rather than a generic boolean.
//
// ── Artifact ──────────────────────────────────────────────────────────────────
//
// Tools can produce more than text. An artifact is any discrete output that
// warrants separate delivery: a generated file, a rendered chart, a binary
// blob. Artifacts travel alongside ToolOutput but are emitted as their own
// AgentEvent so callers can route them independently.
//
// ── ToolInput ────────────────────────────────────────────────────────────────
//
// Every tool's call() receives a raw serde_json::Value. ToolInput wraps it
// with typed accessors so each extraction is one clear line instead of
// nested match arms and unwrap chains.
// ============================================================================

use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::Message;
use crate::query_chain::QueryChain;

// ── Tool trait ────────────────────────────────────────────────────────────────

/// The interface every tool implements.
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    /// Unique identifier. This is how the LLM names the tool in its output.
    fn name(&self) -> &str;

    /// One-line description shown to the LLM in the tool listing.
    fn description(&self) -> &str;

    /// Full usage instructions. May be loaded lazily via `ToolSearch` when
    /// `defer_loading()` returns `true` — injected into context only when
    /// the LLM explicitly requests details.
    fn prompt(&self) -> String;

    /// JSON Schema describing the tool's input parameters.
    ///
    /// The executor validates every invocation against this schema before
    /// calling `call()`. Invalid input produces an immediate error result
    /// that the LLM sees and can self-correct from.
    fn input_schema(&self) -> Value;

    /// Whether to defer this tool's full description until requested.
    ///
    /// When `true`, only `name()` and `description()` appear in the initial
    /// system prompt. The LLM calls `ToolSearch` to load the full schema.
    /// Use this for large tool libraries where every schema would consume
    /// too many tokens.
    fn defer_loading(&self) -> bool {
        false
    }

    /// A short hint for `ToolSearch` (3–10 words describing what this tool
    /// does). Only used when `defer_loading()` returns `true`.
    fn search_hint(&self) -> Option<&str> {
        None
    }

    /// Whether this tool has no observable side effects.
    ///
    /// Defaults to `false`. Override to return `true` for pure read
    /// operations: web search, file read, etc.
    ///
    /// Used by `PermissionMode::Readonly`: only tools returning `true` are
    /// permitted; all others are blocked without prompting the user.
    fn is_readonly(&self) -> bool {
        false
    }

    /// How long the executor waits before cancelling this tool.
    ///
    /// `None` means no timeout. Override to protect against hung tools:
    ///
    /// ```rust,ignore
    /// fn timeout(&self) -> Option<std::time::Duration> {
    ///     Some(std::time::Duration::from_secs(30))
    /// }
    /// ```
    ///
    /// Per-tool timeouts take precedence over the global `tool_timeout`
    /// set on `AgentBuilder`. Both are `None` by default.
    fn timeout(&self) -> Option<std::time::Duration> {
        None
    }

    /// Whether this specific invocation can run concurrently with others.
    ///
    /// Defaults to `true`. Override to inspect `input` when the safety
    /// decision depends on arguments:
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
    /// `ctx` provides the conversation history, cancellation, and progress.
    async fn call(&self, input: Value, ctx: &ToolCtx) -> ToolOutput;
}

// ── Tool Context ─────────────────────────────────────────────────────────────

/// Runtime context injected into every tool invocation.
pub struct ToolCtx {
    /// Cancellation signal. Check periodically in long-running tools.
    pub cancel: tokio_util::sync::CancellationToken,

    /// The conversation history at the time of this invocation (read-only).
    ///
    /// `Arc<[Message]>` — shared across concurrent tools without copying.
    pub messages: Arc<[Message]>,

    /// Report incremental progress to the stream.
    pub on_progress: Box<dyn Fn(String) + Send + Sync>,

    /// Sub-agent spawn capability, if configured.
    pub spawn: Option<SpawnFn>,

    /// Position in the sub-agent delegation tree, if this tool is running
    /// inside a sub-agent. `None` for tools running in a top-level agent.
    ///
    /// Tools can inspect `chain.depth` and `chain.remaining()` before
    /// spawning further sub-agents to avoid runaway delegation.
    pub chain: Option<QueryChain>,
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
/// Injected into `ToolCtx` at execution time. The `Tool` trait depends only
/// on `wuhu-core` — it never imports engine types directly.
pub type SpawnFn =
    Arc<dyn Fn(String) -> BoxFuture<'static, anyhow::Result<String>> + Send + Sync>;

// ── Tool Output ───────────────────────────────────────────────────────────────

/// The result of a tool invocation.
///
/// When the tool succeeds, `failure` is `None`.
/// When it fails, `failure` carries the reason — not just a boolean — so the
/// harness and the LLM can respond appropriately to each failure kind.
///
/// Tools may attach `artifacts` (files, images, charts) and `injections`
/// (system-level context) alongside the primary text `content`.
/// These are processed after all tool results are collected.
#[derive(Debug, Clone, Default)]
pub struct ToolOutput {
    /// The content returned to the LLM (description, result, or error message).
    pub content: String,
    /// `None` on success. `Some(kind)` on failure.
    pub failure: Option<FailureKind>,
    /// Artifacts produced by this tool: files, images, structured data.
    ///
    /// Emitted as `AgentEvent::Artifact` events so callers can handle them
    /// independently from the text response (save to disk, render in UI, etc.).
    pub artifacts: Vec<Artifact>,
    /// System-level context to inject after this tool's result.
    ///
    /// Use sparingly. Typical use: a tool that fetches documentation injects
    /// the doc content so the LLM sees it on the next turn without the user
    /// having to ask again.
    ///
    /// Injections are formatted as `<system-reminder>` blocks — they are
    /// explicitly system-level, not forgeable User or Assistant turns.
    pub injections: Vec<ContextInjection>,
}

impl ToolOutput {
    // ── Constructors ──────────────────────────────────────────────────────

    /// Successful tool execution.
    pub fn success(content: impl Into<String>) -> Self {
        Self { content: content.into(), ..Default::default() }
    }

    /// Failed execution — the tool ran but produced an error.
    pub fn error(content: impl Into<String>) -> Self {
        Self { content: content.into(), failure: Some(FailureKind::Execution), ..Default::default() }
    }

    /// The tool's input did not satisfy its JSON Schema.
    pub fn invalid_input(content: impl Into<String>) -> Self {
        Self { content: content.into(), failure: Some(FailureKind::InvalidInput), ..Default::default() }
    }

    /// The tool was not found in the registry.
    pub fn not_found(content: impl Into<String>) -> Self {
        Self { content: content.into(), failure: Some(FailureKind::NotFound), ..Default::default() }
    }

    /// The tool was blocked by a hook before execution.
    pub fn hook_blocked(content: impl Into<String>) -> Self {
        Self { content: content.into(), failure: Some(FailureKind::HookBlocked), ..Default::default() }
    }

    /// The tool was denied by the permission system.
    pub fn permission_denied(content: impl Into<String>) -> Self {
        Self { content: content.into(), failure: Some(FailureKind::PermissionDenied), ..Default::default() }
    }

    // ── Builders ──────────────────────────────────────────────────────────

    /// Attach artifacts to a successful result.
    pub fn with_artifacts(mut self, artifacts: impl IntoIterator<Item = Artifact>) -> Self {
        self.artifacts = artifacts.into_iter().collect();
        self
    }

    /// Attach system-level context injections.
    pub fn with_injections(mut self, items: impl IntoIterator<Item = ContextInjection>) -> Self {
        self.injections = items.into_iter().collect();
        self
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
/// | Kind              | Typical recovery                                      |
/// |-------------------|-------------------------------------------------------|
/// | `Execution`       | Surface the error; LLM decides whether to retry      |
/// | `InvalidInput`    | Inject schema hints; LLM retries with correct args   |
/// | `NotFound`        | Inform the LLM; do not retry                          |
/// | `HookBlocked`     | Inject reason; LLM may seek an alternative approach  |
/// | `PermissionDenied`| Inject reason; LLM must not retry this tool          |
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

// ── Artifact ──────────────────────────────────────────────────────────────────

/// A discrete output produced by a tool — beyond the primary text content.
///
/// Artifacts are emitted as `AgentEvent::Artifact` events, separate from
/// `ToolDone`, so callers can route them to the right destination:
/// save files to disk, render images in a UI, index structured data, etc.
///
/// The `kind` field is open-ended — the framework does not interpret it.
/// Producers and consumers agree on their own convention:
///   `"file"`, `"image"`, `"chart"`, `"json"`, `"diff"`, ...
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Open-ended type tag. The framework does not interpret this.
    pub kind: String,
    /// Human-readable title for display.
    pub title: String,
    /// MIME type, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// The actual content: inline bytes or a URI reference.
    pub content: ArtifactContent,
}

/// The data carried by an `Artifact`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ArtifactContent {
    /// Raw bytes embedded directly.
    Inline { data: Vec<u8> },
    /// A reference to external storage (URI, path, object key, etc.).
    Reference { uri: String },
}

impl Artifact {
    /// Construct a text artifact (UTF-8 content as inline bytes).
    pub fn text(kind: impl Into<String>, title: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            kind:      kind.into(),
            title:     title.into(),
            mime_type: Some("text/plain".to_string()),
            content:   ArtifactContent::Inline { data: text.into().into_bytes() },
        }
    }

    /// Construct a file artifact from raw bytes.
    pub fn bytes(
        kind:      impl Into<String>,
        title:     impl Into<String>,
        mime_type: impl Into<String>,
        data:      Vec<u8>,
    ) -> Self {
        Self {
            kind:      kind.into(),
            title:     title.into(),
            mime_type: Some(mime_type.into()),
            content:   ArtifactContent::Inline { data },
        }
    }

    /// Construct a reference artifact (the content lives elsewhere).
    pub fn reference(
        kind:  impl Into<String>,
        title: impl Into<String>,
        uri:   impl Into<String>,
    ) -> Self {
        Self {
            kind:      kind.into(),
            title:     title.into(),
            mime_type: None,
            content:   ArtifactContent::Reference { uri: uri.into() },
        }
    }
}

// ── Context Injection ─────────────────────────────────────────────────────────

/// System-level context a tool wants the LLM to see on the next turn.
///
/// Injected as a `<system-reminder>` block — clearly a framework message,
/// not a forged User or Assistant turn. This is the only injection surface
/// tools have: they cannot write arbitrary roles into the conversation history.
///
/// ```rust,ignore
/// ToolOutput::success("Done.")
///     .with_injections([ContextInjection::new("The file was written to /tmp/out.json")])
/// ```
#[derive(Debug, Clone)]
pub struct ContextInjection {
    /// The text to inject. Plain prose — the engine wraps it in
    /// `<system-reminder>` tags before appending to the conversation.
    pub text: String,
}

impl ContextInjection {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

// ── ToolInput ─────────────────────────────────────────────────────────────────

/// Ergonomic wrapper for extracting typed fields from a JSON tool input.
///
/// Reduces boilerplate in `Tool::call()` implementations:
///
/// ```rust,ignore
/// async fn call(&self, input: Value, _ctx: &ToolCtx) -> ToolOutput {
///     let inp = ToolInput(&input);
///     let url   = match inp.required_str("url")   { Ok(v) => v, Err(e) => return ToolOutput::error(e) };
///     let limit = inp.optional_u64("limit").unwrap_or(100);
///     ...
/// }
/// ```
#[derive(Copy, Clone)]
pub struct ToolInput<'a>(pub &'a Value);

impl<'a> ToolInput<'a> {
    // ── String ────────────────────────────────────────────────────────

    /// Extract a required, non-empty string field.
    pub fn required_str(&self, key: &str) -> Result<&'a str, String> {
        match self.0[key].as_str().filter(|s| !s.is_empty()) {
            Some(s) => Ok(s),
            None    => Err(format!("'{key}' is required")),
        }
    }

    /// Extract an optional string field. Returns `None` when absent or null.
    pub fn optional_str(&self, key: &str) -> Option<&'a str> {
        self.0[key].as_str()
    }

    // ── Boolean ───────────────────────────────────────────────────────

    /// Extract a required boolean field.
    pub fn required_bool(&self, key: &str) -> Result<bool, String> {
        self.0[key].as_bool().ok_or_else(|| format!("'{key}' is required (bool)"))
    }

    /// Extract an optional boolean field.
    pub fn optional_bool(&self, key: &str) -> Option<bool> {
        self.0[key].as_bool()
    }

    // ── Integer ───────────────────────────────────────────────────────

    /// Extract a required unsigned integer field.
    pub fn required_u64(&self, key: &str) -> Result<u64, String> {
        self.0[key].as_u64().ok_or_else(|| format!("'{key}' is required (integer)"))
    }

    /// Extract an optional unsigned integer field.
    pub fn optional_u64(&self, key: &str) -> Option<u64> {
        self.0[key].as_u64()
    }

    /// Extract a required signed integer field.
    pub fn required_i64(&self, key: &str) -> Result<i64, String> {
        self.0[key].as_i64().ok_or_else(|| format!("'{key}' is required (integer)"))
    }

    /// Extract an optional signed integer field.
    pub fn optional_i64(&self, key: &str) -> Option<i64> {
        self.0[key].as_i64()
    }

    // ── Float ─────────────────────────────────────────────────────────

    /// Extract an optional float field.
    pub fn optional_f64(&self, key: &str) -> Option<f64> {
        self.0[key].as_f64()
    }

    // ── Array ─────────────────────────────────────────────────────────

    /// Extract a required array field.
    pub fn required_array(&self, key: &str) -> Result<&'a Vec<Value>, String> {
        self.0[key].as_array().ok_or_else(|| format!("'{key}' is required (array)"))
    }

    /// Extract an optional array field.
    pub fn optional_array(&self, key: &str) -> Option<&'a Vec<Value>> {
        self.0[key].as_array()
    }

    // ── Object ────────────────────────────────────────────────────────

    /// Extract a required object field.
    pub fn required_object(&self, key: &str) -> Result<&'a serde_json::Map<String, Value>, String> {
        self.0[key].as_object().ok_or_else(|| format!("'{key}' is required (object)"))
    }

    /// Extract an optional object field.
    pub fn optional_object(&self, key: &str) -> Option<&'a serde_json::Map<String, Value>> {
        self.0[key].as_object()
    }
}
