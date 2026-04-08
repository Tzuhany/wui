// ── Tool Output ───────────────────────────────────────────────────────────────
//
// FailureKind gives the harness — and the LLM — structured information to
// act on, rather than a generic boolean.
//
// A schema error suggests the LLM should retry with corrected arguments.
// A permission denial means it should stop and explain.
// An execution error might be retried after fixing the underlying condition.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::{Artifact, ContextInjection, Tool};
use crate::message::Message;

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
