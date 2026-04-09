// ============================================================================
// Tool — the agent's hands.
//
// Tools are stateless, Send + Sync + 'static. The executor spawns them freely
// across tokio tasks without lifetime friction. State lives in ToolCtx.
// ============================================================================

use std::borrow::Borrow;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::Message;

// ── ToolCallId ───────────────────────────────────────────────────────────────

/// A unique identifier for a tool call within a single LLM turn.
///
/// Assigned by the provider (e.g., `toolu_abc123` for Anthropic).
/// Used to correlate `ToolStart`, `ToolDone`, and `ToolResult` events.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolCallId(String);

impl ToolCallId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_inner(self) -> String {
        self.0
    }
}

impl std::ops::Deref for ToolCallId {
    type Target = str;
    fn deref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ToolCallId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for ToolCallId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for ToolCallId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

impl PartialEq<str> for ToolCallId {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<String> for ToolCallId {
    fn eq(&self, other: &String) -> bool {
        self.0 == *other
    }
}

impl PartialEq<&str> for ToolCallId {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

impl Borrow<str> for ToolCallId {
    fn borrow(&self) -> &str {
        &self.0
    }
}

// ── ToolMeta ─────────────────────────────────────────────────────────────────

/// Per-invocation semantic metadata for a tool call.
///
/// `ToolMeta` contains only cross-runtime properties: semantic flags that any
/// executor built on `wui-core` needs to reason about execution ordering,
/// safety, and permission decisions. Wui-executor-specific tuning knobs
/// (timeout, retries, output limits) live in `ExecutorHints` in `wui`.
///
/// Return from [`Tool::meta`] to communicate these properties to the runtime.
/// All fields have safe defaults — override only what differs.
///
/// # Example
/// ```rust,ignore
/// fn meta(&self, _input: &serde_json::Value) -> ToolMeta {
///     ToolMeta { readonly: true, ..ToolMeta::default() }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ToolMeta {
    /// Allow concurrent execution alongside other tools. Default: `true`.
    pub concurrent: bool,
    /// Tool does not modify external state. Default: `false`.
    pub readonly: bool,
    /// Tool cannot be undone (shown to user in HITL prompt). Default: `false`.
    pub destructive: bool,
    /// Tool requires live user interaction and cannot run headlessly. Default: `false`.
    pub requires_interaction: bool,
    /// Suffix appended to the tool name for fine-grained permission rules.
    /// E.g. `Some("rm -rf /")` for a bash tool.
    pub permission_key: Option<String>,
}

impl Default for ToolMeta {
    fn default() -> Self {
        Self {
            concurrent: true,
            readonly: false,
            destructive: false,
            requires_interaction: false,
            permission_key: None,
        }
    }
}

// ── ExecutorHints ────────────────────────────────────────────────────────────

/// Wui-executor-specific execution hints returned by [`Tool::executor_hints`].
///
/// These are NOT part of the universal tool vocabulary — they are tuning knobs
/// specific to Wui's executor implementation. Any runtime built on `wui-core`
/// that doesn't use these hints may ignore them entirely.
///
/// Tools return these from `executor_hints()` to customise per-invocation
/// behaviour. The defaults (no timeout, no retries, no output limit, no summary)
/// are safe for any tool.
///
/// # Freeze policy
///
/// `ExecutorHints` is a **closed set**. Fields are added only when a new
/// executor behaviour cannot be expressed by composing the existing ones and
/// has proven necessary in real-world use. Before adding a field, ask:
///
/// 1. Can this be expressed by combining existing fields?
/// 2. Is this Wui-runtime-specific or truly universal (belongs in `ToolMeta`)?
/// 3. Is there at least one concrete use-case that cannot be solved without it?
///
/// Routine convenience is not sufficient justification. Keeping this type small
/// is the mechanism by which `wui-core` stays honest.
#[derive(Debug, Clone, Default)]
pub struct ExecutorHints {
    /// One-line summary for display in tool history. Default: `None`.
    pub summary: Option<String>,
    /// Per-call execution timeout. Default: use the executor's global timeout.
    pub timeout: Option<std::time::Duration>,
    /// Truncate output to this many characters. Default: no limit.
    pub max_output_chars: Option<usize>,
    /// Retry up to this many times on error output. Default: `0`.
    pub max_retries: u32,
}

// ── InterruptBehavior ────────────────────────────────────────────────────────

/// What should happen when the user submits a new message while this tool runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptBehavior {
    /// Stop the tool and discard its result.
    Cancel,
    /// Keep running; the new message waits until this tool completes.
    Block,
}

/// A permission-matching closure returned by [`Tool::permission_matcher`].
pub type PermissionMatcher = Box<dyn Fn(&str) -> bool + Send + Sync>;

// ── Tool Trait ───────────────────────────────────────────────────────────────

/// The interface every tool implements.
///
/// Four methods are required: `name`, `description`, `input_schema`, and `call`.
/// Override `meta` to communicate per-invocation semantic hints (concurrency,
/// readonly, destructive, requires_interaction) to the runtime.
/// Override `executor_hints` to communicate Wui-executor-specific tuning
/// (timeout, retries, output limits, display summary) without affecting other
/// runtimes.
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    /// Unique identifier. This is how the LLM names the tool in its output.
    fn name(&self) -> &str;

    /// One-line description shown to the LLM in the tool listing.
    fn description(&self) -> &str;

    /// JSON Schema describing the tool's input parameters.
    ///
    /// The executor validates every invocation against this schema before
    /// calling `call()`. Invalid input produces an immediate error result
    /// that the LLM sees and can self-correct from.
    fn input_schema(&self) -> Value;

    /// Semantic metadata for this tool invocation.
    ///
    /// Called once per invocation with the resolved input. Return a customised
    /// [`ToolMeta`] to communicate cross-runtime execution properties (concurrency,
    /// readonly, destructive, requires_interaction, permission_key). Most tools
    /// return `ToolMeta::default()` and override one or two fields.
    fn meta(&self, _input: &Value) -> ToolMeta {
        ToolMeta::default()
    }

    /// Wui-executor-specific execution hints for this tool invocation.
    ///
    /// Called once per invocation with the resolved input. Return a customised
    /// [`ExecutorHints`] to provide timeout, retry, output-limit, or display
    /// summary values to the Wui executor. Other runtimes that don't use Wui's
    /// executor may ignore these hints entirely.
    ///
    /// ```rust,ignore
    /// fn executor_hints(&self, _input: &Value) -> ExecutorHints {
    ///     ExecutorHints {
    ///         timeout:     Some(std::time::Duration::from_secs(30)),
    ///         max_retries: 2,
    ///         ..ExecutorHints::default()
    ///     }
    /// }
    /// ```
    fn executor_hints(&self, _input: &Value) -> ExecutorHints {
        ExecutorHints::default()
    }

    /// What happens when the user interrupts while this tool is running.
    ///
    /// - `Cancel` — abort the tool and discard its result.
    /// - `Block` (default) — keep running; the interruption waits.
    ///
    /// Override for tools whose results are disposable on interruption
    /// (searches, reads). Keep the default for tools with side effects
    /// (file writes, API calls) that should finish cleanly.
    fn interrupt_behavior(&self) -> InterruptBehavior {
        InterruptBehavior::Block
    }

    /// Prepare a matcher for wildcard permission rules.
    ///
    /// Called once per tool invocation during the permission check. Returns a
    /// closure that tests whether a permission rule pattern matches this
    /// specific invocation. When `None` (default), the framework falls back
    /// to prefix matching on [`ToolMeta::permission_key`].
    ///
    /// Use this for tools with complex input structures (like a shell tool)
    /// where the permission pattern needs to match against parsed subcommands:
    ///
    /// ```rust,ignore
    /// fn permission_matcher(&self, input: &Value) -> Option<Box<dyn Fn(&str) -> bool + Send>> {
    ///     let cmd = input["command"].as_str()?.to_string();
    ///     Some(Box::new(move |pattern| {
    ///         cmd == pattern || cmd.starts_with(&format!("{pattern} "))
    ///     }))
    /// }
    /// ```
    fn permission_matcher(&self, _input: &Value) -> Option<PermissionMatcher> {
        None
    }

    /// Execute the tool.
    ///
    /// Called only after input has been validated against `input_schema()`.
    /// `ctx` provides the conversation history, cancellation, and progress.
    async fn call(&self, input: Value, ctx: &ToolCtx) -> ToolOutput;
}

// ── ToolCtx ──────────────────────────────────────────────────────────────────

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
}

impl ToolCtx {
    pub fn report(&self, msg: impl Into<String>) {
        (self.on_progress)(msg.into());
    }
}

// ── ToolOutput ───────────────────────────────────────────────────────────────

/// The result of a tool invocation.
///
/// When the tool succeeds, `failure` is `None`.
/// When it fails, `failure` carries the reason — not just a boolean — so the
/// harness and the LLM can respond appropriately to each failure kind.
///
/// Tools may attach `artifacts` (files, images, charts) and `injections`
/// (system-level context) alongside the primary text `content`.
/// These are processed after all tool results are collected.
#[derive(Clone, Default)]
pub struct ToolOutput {
    /// The content returned to the LLM (description, result, or error message).
    pub content: String,
    /// `true` when the executor truncated this output because it exceeded the
    /// configured `max_output_chars` limit. Callers can inspect this flag to
    /// decide whether to retry with a smaller scope or retrieve remaining output
    /// via a follow-up tool call.
    pub truncated: bool,
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

    /// Machine-readable result for the caller, separate from the LLM-facing text.
    ///
    /// The LLM receives `content` (a string). Callers can read `structured`
    /// from `AgentEvent::ToolDone` to extract typed data without parsing prose.
    ///
    /// ```rust,ignore
    /// ToolOutput::success("Found 42 results.")
    ///     .with_structured(json!({"count": 42, "items": [...]}))
    /// ```
    pub structured: Option<serde_json::Value>,

    /// Additional tools to expose to the agent after this tool completes.
    ///
    /// The run loop adds these tools to the active tool set for the remainder
    /// of the current run. Use this in `tool_search` implementations to
    /// dynamically inject discovered tools.
    pub expose_tools: Vec<Arc<dyn Tool>>,
}

impl std::fmt::Debug for ToolOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolOutput")
            .field("content", &self.content)
            .field("failure", &self.failure)
            .field("artifacts", &self.artifacts)
            .field("structured", &self.structured)
            .field("expose_tools_count", &self.expose_tools.len())
            .finish()
    }
}

impl ToolOutput {
    // ── Constructors ──────────────────────────────────────────────────────

    /// Successful tool execution.
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            expose_tools: vec![],
            ..Default::default()
        }
    }

    /// Failed execution — the tool ran but produced an error.
    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            failure: Some(FailureKind::Execution),
            expose_tools: vec![],
            ..Default::default()
        }
    }

    /// The tool's input did not satisfy its JSON Schema.
    pub fn invalid_input(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            failure: Some(FailureKind::InvalidInput),
            expose_tools: vec![],
            ..Default::default()
        }
    }

    /// The tool was not found in the registry.
    pub fn not_found(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            failure: Some(FailureKind::NotFound),
            expose_tools: vec![],
            ..Default::default()
        }
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

    /// Attach a machine-readable structured result for the caller.
    pub fn with_structured(mut self, value: impl Into<serde_json::Value>) -> Self {
        self.structured = Some(value.into());
        self
    }

    /// Expose additional tools to the agent after this tool completes.
    ///
    /// The run loop adds these tools to the active tool set for the remainder
    /// of the current run. Use this in `tool_search` implementations to
    /// dynamically inject discovered tools.
    pub fn expose(mut self, tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> Self {
        self.expose_tools = tools.into_iter().collect();
        self
    }

    /// Attach a single context injection.
    pub fn inject(mut self, injection: ContextInjection) -> Self {
        self.injections.push(injection);
        self
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    pub fn is_ok(&self) -> bool {
        self.failure.is_none()
    }

    pub fn is_error(&self) -> bool {
        self.failure.is_some()
    }

    /// Whether this output represents a retryable failure.
    ///
    /// Returns `true` only for `FailureKind::Execution`. All other failure
    /// kinds (invalid input, not found, permission denied, hook blocked) are
    /// deterministic — retrying would produce the same result.
    pub fn is_retryable(&self) -> bool {
        self.failure.as_ref().is_some_and(|k| k.is_retryable())
    }
}

// ── FailureKind ──────────────────────────────────────────────────────────────

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

impl FailureKind {
    /// Whether this failure kind is worth retrying.
    ///
    /// Only `Execution` errors are retryable — the tool ran but hit a
    /// transient or recoverable problem. All other kinds are deterministic:
    /// retrying with the same input, registry, hooks, and permissions will
    /// produce the same outcome.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Execution)
    }
}

// ── ContextInjection ─────────────────────────────────────────────────────────

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

// ── ToolInput ────────────────────────────────────────────────────────────────

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
            None => Err(format!("'{key}' is required")),
        }
    }

    /// Extract an optional string field. Returns `None` when absent or null.
    pub fn optional_str(&self, key: &str) -> Option<&'a str> {
        self.0[key].as_str()
    }

    // ── Boolean ───────────────────────────────────────────────────────

    /// Extract a required boolean field.
    pub fn required_bool(&self, key: &str) -> Result<bool, String> {
        self.0[key]
            .as_bool()
            .ok_or_else(|| format!("'{key}' is required (bool)"))
    }

    /// Extract an optional boolean field.
    pub fn optional_bool(&self, key: &str) -> Option<bool> {
        self.0[key].as_bool()
    }

    // ── Integer ───────────────────────────────────────────────────────

    /// Extract a required unsigned integer field.
    pub fn required_u64(&self, key: &str) -> Result<u64, String> {
        self.0[key]
            .as_u64()
            .ok_or_else(|| format!("'{key}' is required (integer)"))
    }

    /// Extract an optional unsigned integer field.
    pub fn optional_u64(&self, key: &str) -> Option<u64> {
        self.0[key].as_u64()
    }

    /// Extract a required signed integer field.
    pub fn required_i64(&self, key: &str) -> Result<i64, String> {
        self.0[key]
            .as_i64()
            .ok_or_else(|| format!("'{key}' is required (integer)"))
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
        self.0[key]
            .as_array()
            .ok_or_else(|| format!("'{key}' is required (array)"))
    }

    /// Extract an optional array field.
    pub fn optional_array(&self, key: &str) -> Option<&'a Vec<Value>> {
        self.0[key].as_array()
    }

    // ── Object ────────────────────────────────────────────────────────

    /// Extract a required object field.
    pub fn required_object(&self, key: &str) -> Result<&'a serde_json::Map<String, Value>, String> {
        self.0[key]
            .as_object()
            .ok_or_else(|| format!("'{key}' is required (object)"))
    }

    /// Extract an optional object field.
    pub fn optional_object(&self, key: &str) -> Option<&'a serde_json::Map<String, Value>> {
        self.0[key].as_object()
    }
}

// ── Artifact ─────────────────────────────────────────────────────────────────

/// The semantic kind of an [`Artifact`].
///
/// Use [`ArtifactKind::Custom`] for application-specific kinds not covered
/// by the standard variants.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    /// A file artifact (source code, documents, data files, etc.)
    File,
    /// An image artifact (PNG, JPEG, SVG, etc.)
    Image,
    /// A structured chart or graph artifact.
    Chart,
    /// A JSON data artifact.
    Json,
    /// An application-specific artifact kind.
    Custom(String),
}

impl ArtifactKind {
    /// Returns the string representation of this kind.
    pub fn as_str(&self) -> &str {
        match self {
            ArtifactKind::File => "file",
            ArtifactKind::Image => "image",
            ArtifactKind::Chart => "chart",
            ArtifactKind::Json => "json",
            ArtifactKind::Custom(s) => s.as_str(),
        }
    }
}

impl fmt::Display for ArtifactKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A discrete output produced by a tool — beyond the primary text content.
///
/// Artifacts are emitted as `AgentEvent::Artifact` events, separate from
/// `ToolDone`, so callers can route them to the right destination:
/// save files to disk, render images in a UI, index structured data, etc.
///
/// The `kind` field uses [`ArtifactKind`] — use [`ArtifactKind::Custom`] for
/// application-specific kinds not covered by the standard variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Semantic type of this artifact.
    pub kind: ArtifactKind,
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
    pub fn text(title: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            kind: ArtifactKind::File,
            title: title.into(),
            mime_type: Some("text/plain".to_string()),
            content: ArtifactContent::Inline {
                data: text.into().into_bytes(),
            },
        }
    }

    /// Construct a binary artifact from raw bytes with an explicit kind.
    pub fn bytes(
        title: impl Into<String>,
        kind: ArtifactKind,
        mime_type: Option<impl Into<String>>,
        data: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            kind,
            title: title.into(),
            mime_type: mime_type.map(|m| m.into()),
            content: ArtifactContent::Inline { data: data.into() },
        }
    }

    /// Construct a reference artifact (the content lives elsewhere).
    pub fn reference(title: impl Into<String>, kind: ArtifactKind, uri: impl Into<String>) -> Self {
        Self {
            kind,
            title: title.into(),
            mime_type: None,
            content: ArtifactContent::Reference { uri: uri.into() },
        }
    }
}

// ── ToolArgs + ToolInputError + TypedTool ────────────────────────────────────

/// Structured error produced when parsing typed tool input fails.
///
/// Includes enough detail for the LLM to self-correct: which field was
/// wrong, what was expected, and (optionally) what was actually received.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolInputError {
    /// The field that caused the error, if identifiable.
    pub field: Option<String>,
    /// What the tool expected (e.g. `"string"`, `"integer > 0"`).
    pub expected: String,
    /// What the tool actually received, as a string representation.
    pub got: Option<String>,
}

impl ToolInputError {
    /// Create a general parse error with just an expected description.
    pub fn new(expected: impl Into<String>) -> Self {
        Self {
            field: None,
            expected: expected.into(),
            got: None,
        }
    }

    /// Create a field-specific parse error.
    pub fn field(name: impl Into<String>, expected: impl Into<String>) -> Self {
        Self {
            field: Some(name.into()),
            expected: expected.into(),
            got: None,
        }
    }

    /// Attach what was actually received.
    pub fn with_got(mut self, got: impl Into<String>) -> Self {
        self.got = Some(got.into());
        self
    }
}

impl fmt::Display for ToolInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.field, &self.got) {
            (Some(field), Some(got)) => {
                write!(f, "field '{field}': expected {}, got {got}", self.expected)
            }
            (Some(field), None) => {
                write!(f, "field '{field}': expected {}", self.expected)
            }
            (None, Some(got)) => {
                write!(f, "expected {}, got {got}", self.expected)
            }
            (None, None) => {
                write!(f, "expected {}", self.expected)
            }
        }
    }
}

impl std::error::Error for ToolInputError {}

/// Trait for types that can serve as typed tool arguments.
///
/// Typically derived via `#[derive(ToolInput)]`, which generates both the
/// JSON Schema and the parser. Manual implementation is also supported for
/// tools with complex validation logic.
pub trait ToolArgs: Sized + Send + Sync {
    /// Returns the JSON Schema describing the input parameters.
    fn schema() -> Value;

    /// Parse a JSON value into this type.
    fn parse(value: Value) -> Result<Self, ToolInputError>;
}

/// A tool that receives parsed, strongly-typed input.
///
/// Implement this instead of `Tool` when writing Rust tools. The blanket
/// `impl<T: TypedTool> Tool for T` bridges TypedTool into the runtime
/// automatically — no registration changes needed.
///
/// ```rust,ignore
/// #[derive(ToolInput)]
/// struct SearchInput {
///     /// The search query.
///     query: String,
///     /// Maximum results.
///     max_results: Option<u64>,
/// }
///
/// struct SearchTool;
///
/// #[async_trait]
/// impl TypedTool for SearchTool {
///     type Input = SearchInput;
///     fn name(&self) -> &str { "search" }
///     fn description(&self) -> &str { "Search the web." }
///     async fn call_typed(&self, input: SearchInput, ctx: &ToolCtx) -> ToolOutput {
///         ToolOutput::success(format!("found results for: {}", input.query))
///     }
/// }
/// ```
#[async_trait]
pub trait TypedTool: Send + Sync + 'static {
    /// The strongly-typed input for this tool.
    type Input: ToolArgs;

    /// Unique identifier. This is how the LLM names the tool.
    fn name(&self) -> &str;

    /// One-line description shown to the LLM.
    fn description(&self) -> &str;

    /// Per-invocation semantic metadata. See [`ToolMeta`].
    fn meta(&self, _input: &Self::Input) -> ToolMeta {
        ToolMeta::default()
    }

    /// Wui-executor-specific execution hints. See [`ExecutorHints`].
    fn executor_hints(&self, _input: &Self::Input) -> ExecutorHints {
        ExecutorHints::default()
    }

    /// Execute the tool with parsed input.
    async fn call_typed(&self, input: Self::Input, ctx: &ToolCtx) -> ToolOutput;
}

// ── Blanket impl: TypedTool -> Tool ──────────────────────────────────────────

#[async_trait]
impl<T: TypedTool> Tool for T {
    fn name(&self) -> &str {
        TypedTool::name(self)
    }

    fn description(&self) -> &str {
        TypedTool::description(self)
    }

    fn input_schema(&self) -> Value {
        T::Input::schema()
    }

    fn meta(&self, input: &Value) -> ToolMeta {
        match T::Input::parse(input.clone()) {
            Ok(parsed) => TypedTool::meta(self, &parsed),
            Err(_) => ToolMeta::default(),
        }
    }

    fn executor_hints(&self, input: &Value) -> ExecutorHints {
        match T::Input::parse(input.clone()) {
            Ok(parsed) => TypedTool::executor_hints(self, &parsed),
            Err(_) => ExecutorHints::default(),
        }
    }

    fn interrupt_behavior(&self) -> InterruptBehavior {
        InterruptBehavior::Block
    }

    fn permission_matcher(&self, _input: &Value) -> Option<PermissionMatcher> {
        None
    }

    async fn call(&self, input: Value, ctx: &ToolCtx) -> ToolOutput {
        match T::Input::parse(input) {
            Ok(parsed) => self.call_typed(parsed, ctx).await,
            Err(e) => ToolOutput::invalid_input(e.to_string()),
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_id_basics() {
        let id = ToolCallId::new("toolu_abc123");
        assert_eq!(id.as_str(), "toolu_abc123");
        assert_eq!(&*id, "toolu_abc123");
        assert_eq!(id, *"toolu_abc123");
        assert_eq!(format!("{id}"), "toolu_abc123");
    }

    #[test]
    fn tool_call_id_serde_roundtrip() {
        let id = ToolCallId::new("abc");
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, r#""abc""#);
        let back: ToolCallId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, id);
    }
}
