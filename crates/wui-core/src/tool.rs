// ============================================================================
// Tool — the agent's hands.
//
// Tools are stateless, Send + Sync + 'static. The executor spawns them freely
// across tokio tasks without lifetime friction. State lives in ToolCtx.
//
// ── ToolMeta / meta(input) ────────────────────────────────────────────────────
//
// Semantic tool properties live in ToolMeta, returned by Tool::meta(). These
// are cross-runtime properties: any executor needs to know about concurrency,
// readonly status, destructiveness, and interaction requirements. Executor-
// specific tuning (timeout, retries, output limits) lives in ExecutorHints
// inside the wui crate, not here.
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
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::Message;

// ── ToolMeta ──────────────────────────────────────────────────────────────────

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

// ── ExecutorHints ─────────────────────────────────────────────────────────────

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

// ── Tool trait ────────────────────────────────────────────────────────────────

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
    #[allow(unused_variables)]
    fn executor_hints(&self, input: &Value) -> ExecutorHints {
        ExecutorHints::default()
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
}

impl ToolCtx {
    pub fn report(&self, msg: impl Into<String>) {
        (self.on_progress)(msg.into());
    }
}

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

    /// The tool was blocked by a hook before execution.
    pub fn hook_blocked(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            failure: Some(FailureKind::HookBlocked),
            expose_tools: vec![],
            ..Default::default()
        }
    }

    /// The tool was denied by the permission system.
    pub fn permission_denied(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            failure: Some(FailureKind::PermissionDenied),
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
    pub fn text(
        kind: impl Into<String>,
        title: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            kind: kind.into(),
            title: title.into(),
            mime_type: Some("text/plain".to_string()),
            content: ArtifactContent::Inline {
                data: text.into().into_bytes(),
            },
        }
    }

    /// Construct a file artifact from raw bytes.
    pub fn bytes(
        kind: impl Into<String>,
        title: impl Into<String>,
        mime_type: impl Into<String>,
        data: Vec<u8>,
    ) -> Self {
        Self {
            kind: kind.into(),
            title: title.into(),
            mime_type: Some(mime_type.into()),
            content: ArtifactContent::Inline { data },
        }
    }

    /// Construct a reference artifact (the content lives elsewhere).
    pub fn reference(
        kind: impl Into<String>,
        title: impl Into<String>,
        uri: impl Into<String>,
    ) -> Self {
        Self {
            kind: kind.into(),
            title: title.into(),
            mime_type: None,
            content: ArtifactContent::Reference { uri: uri.into() },
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

    /// Alias for [`Self::new`] — conveys intent when injecting system-level context.
    pub fn system(text: impl Into<String>) -> Self {
        Self::new(text)
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
