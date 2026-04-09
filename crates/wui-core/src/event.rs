// ============================================================================
// Events — two kinds, two audiences.
//
// StreamEvent: raw LLM stream output. Internal to the engine. Users never
//   see these directly. They are the low-level protocol between a Provider
//   and the execution loop.
//
// AgentEvent: processed, meaningful events. External. These are what a user's
//   stream receives. One AgentEvent may be synthesised from many StreamEvents.
//
// The separation matters: it lets the engine present a clean, stable API to
// users while remaining free to evolve how it interprets raw provider output.
//
// ── ControlHandle — capability, not just notification ─────────────────────────
//
// Most frameworks send a ControlRequest (data) and require the caller to track
// a separate response channel. ControlHandle bundles both: the request data
// AND the capability to respond. The caller receives one object and calls
// handle.approve() or handle.deny("reason") — no side-channel bookkeeping.
//
// ── ApproveAlways / DenyAlways — permission memory ───────────────────────────
//
// A session remembers "always allow / always deny" decisions across turns.
// Once a tool is always-approved, subsequent calls skip the permission prompt.
// This eliminates the friction of re-approving the same tool call repeatedly
// without sacrificing safety: the user makes the decision once, explicitly.
// ============================================================================

use std::fmt;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::message::Message;
use crate::tool::{Artifact, FailureKind, ToolCallId};

// ── Token Usage ──────────────────────────────────────────────────────────────

/// Token consumption for a single LLM call.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_tokens: u32,
    pub cache_write_tokens: u32,
}

impl TokenUsage {
    /// Total non-cache tokens consumed.
    pub fn total(&self) -> u32 {
        self.input_tokens.saturating_add(self.output_tokens)
    }

    /// All tokens consumed including cache activity.
    pub fn total_with_cache(&self) -> u32 {
        self.total()
            .saturating_add(self.cache_read_tokens)
            .saturating_add(self.cache_write_tokens)
    }
}

impl std::ops::Add for TokenUsage {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            input_tokens: self.input_tokens.saturating_add(rhs.input_tokens),
            output_tokens: self.output_tokens.saturating_add(rhs.output_tokens),
            cache_read_tokens: self.cache_read_tokens.saturating_add(rhs.cache_read_tokens),
            cache_write_tokens: self
                .cache_write_tokens
                .saturating_add(rhs.cache_write_tokens),
        }
    }
}

impl std::ops::AddAssign for TokenUsage {
    fn add_assign(&mut self, rhs: Self) {
        self.input_tokens = self.input_tokens.saturating_add(rhs.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(rhs.output_tokens);
        self.cache_read_tokens = self.cache_read_tokens.saturating_add(rhs.cache_read_tokens);
        self.cache_write_tokens = self
            .cache_write_tokens
            .saturating_add(rhs.cache_write_tokens);
    }
}

// ── Stop Reason ──────────────────────────────────────────────────────────────

/// Why the LLM stopped generating.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Natural completion.
    EndTurn,
    /// The LLM emitted one or more tool calls.
    ToolUse,
    /// Output was truncated at `max_tokens`.
    MaxTokens,
}

// ── StreamEvent (internal: LLM -> Engine) ────────────────────────────────────

/// Raw events emitted by a Provider's stream.
///
/// The engine consumes these directly. They should never appear in user code.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextDelta {
        text: String,
    },
    ThinkingDelta {
        text: String,
    },

    /// The LLM has started describing a tool call.
    ToolUseStart {
        id: ToolCallId,
        name: String,
    },

    /// A chunk of JSON input for an in-progress tool call.
    ToolInputDelta {
        id: ToolCallId,
        chunk: String,
    },

    /// The LLM has finished describing the tool call. The engine submits it
    /// to the executor immediately upon receiving this event.
    ToolUseEnd {
        id: ToolCallId,
    },

    MessageEnd {
        usage: TokenUsage,
        stop_reason: StopReason,
    },

    /// A retryable or fatal error from the provider.
    Error {
        message: String,
        retryable: bool,
    },
}

// ── Run Summary + Stop Reasons ───────────────────────────────────────────────

/// Summary emitted when a run completes successfully.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub stop_reason: RunStopReason,
    pub iterations: u32,
    pub usage: TokenUsage,
    /// The full conversation at the time the run ended (user + assistant turns).
    pub messages: Vec<Message>,
}

/// Why the run ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStopReason {
    /// The LLM returned `EndTurn` with no tool calls.
    Completed,
    /// The run was cancelled by the caller.
    Cancelled,
    /// The maximum iteration limit was reached.
    MaxIterations,
    /// Context pressure could not be relieved even with full compression.
    ContextOverflow,
    /// Output tokens per turn fell below the useful threshold for too many
    /// consecutive turns. The agent is no longer making progress.
    DiminishingReturns,
    /// Output was truncated at `max_tokens` even after escalation and
    /// continuation injection.
    MaxTokensExhausted,
    /// The cumulative token budget set via `AgentBuilder::token_budget` was
    /// exhausted. The run stopped before `max_iter` to stay within cost limits.
    BudgetExhausted,
}

// ── AgentError ───────────────────────────────────────────────────────────────

/// A terminal error emitted as the last event in the stream.
///
/// `message` is the user-facing description — safe to surface in any UI.
///
/// `detail` carries developer-facing context (stack traces, internal state,
/// raw provider responses) that may contain file paths, code fragments, or
/// other information not appropriate for end-user display. Log it; do not
/// show it verbatim to end users.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct AgentError {
    pub message: String,
    pub retryable: bool,
    /// Developer-facing technical context. May contain paths, code, or
    /// raw provider output — not for end-user display.
    pub detail: Option<String>,
    /// True when the run stopped because a tool required interactive human
    /// approval that this calling context cannot provide.
    ///
    /// Callers that do not support `AgentEvent::Control` (e.g. `Agent::run()`
    /// or `SubAgent`) set this flag so callers higher up can detect the
    /// misconfiguration and provide an actionable error rather than a generic
    /// failure message.
    pub permission_denied: bool,
}

impl AgentError {
    /// A non-retryable terminal error.
    pub fn fatal(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: false,
            detail: None,
            permission_denied: false,
        }
    }

    /// A retryable error — the caller may restart the run.
    pub fn retryable(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: true,
            detail: None,
            permission_denied: false,
        }
    }

    /// The run stopped because a tool required interactive approval that the
    /// current calling context cannot provide.
    ///
    /// `tool_name` is the name of the tool that triggered the approval request,
    /// or `None` if it could not be determined.
    pub fn permission_required(tool_name: Option<&str>) -> Self {
        let message = match tool_name {
            Some(name) => format!(
                "tool '{name}' requires interactive approval; \
                 configure the agent with PermissionMode::Auto or add \
                 an allow rule for this tool to use it in a headless context"
            ),
            None => "a tool requires interactive approval; \
                     configure the agent with PermissionMode::Auto for headless use"
                .to_string(),
        };
        Self {
            message,
            retryable: false,
            detail: None,
            permission_denied: true,
        }
    }

    /// Attach developer-facing technical context to any error.
    ///
    /// The detail string may contain file paths, raw API responses, or other
    /// information that is useful for debugging but should not be shown
    /// verbatim to end users.
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}

// ── AgentEvent (external: Engine -> User) ────────────────────────────────────

/// Events emitted to the caller's stream.
///
/// The enum is exhaustive — every possible agent milestone is represented.
/// Match on it to build any kind of consumer: TUI, websocket, log file, test.
///
/// Only `Control` requires caller action. All other variants are informational.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    // ── Streaming text ────────────────────────────────────────────────
    TextDelta(String),
    ThinkingDelta(String),

    // ── Tool lifecycle ────────────────────────────────────────────────
    ToolStart {
        id: ToolCallId,
        name: String,
        input: serde_json::Value,
    },
    ToolDone {
        id: ToolCallId,
        name: String,
        output: String,
        ms: u64,
        attempts: u32,
        /// Machine-readable result alongside the LLM-facing text, if the
        /// tool provided one. The LLM sees `output`; callers can extract
        /// typed data from `structured` without parsing human-readable prose.
        structured: Option<serde_json::Value>,
    },
    ToolError {
        id: ToolCallId,
        name: String,
        error: String,
        /// Structured reason for the failure. Use this to tailor UI and
        /// recovery logic — a `PermissionDenied` warrants different
        /// treatment than an `Execution` error.
        kind: FailureKind,
        ms: u64,
    },

    // ── HITL ──────────────────────────────────────────────────────────
    /// The agent has paused and is waiting for a human decision.
    ///
    /// Call `handle.approve()`, `handle.approve_always()`,
    /// `handle.deny("reason")`, or `handle.deny_always("reason")` to resume.
    Control(ControlHandle),

    // ── Artifacts ─────────────────────────────────────────────────────
    /// A discrete output produced by a tool — a file, image, chart, etc.
    ///
    /// Emitted before `ToolDone` so callers receive the artifact while the
    /// tool result is still being processed. Route it to wherever it belongs:
    /// disk, object storage, a UI renderer.
    Artifact {
        /// ID of the tool call that produced this artifact.
        tool_id: ToolCallId,
        tool_name: String,
        artifact: Artifact,
    },

    // ── Tool progress ─────────────────────────────────────────────────
    /// Incremental progress reported by a running tool via `ctx.report()`.
    ///
    /// Emitted in real time — use this to stream tool status to a UI
    /// without waiting for the tool to complete.
    ToolProgress {
        tool_id: ToolCallId,
        tool_name: String,
        text: String,
    },

    // ── Context management ────────────────────────────────────────────
    /// Context compression was applied.
    Compressed {
        method: CompressMethod,
        /// Approximate tokens freed.
        freed: usize,
        /// Context pressure before compression (0.0–1.0).
        pressure_before: f64,
        /// Context pressure after compression (0.0–1.0).
        pressure_after: f64,
    },

    /// L3 (LLM summarization) was attempted but failed and the pipeline fell
    /// back to L2 collapse. The `freed` field shows how many tokens were
    /// reclaimed by the fallback L2 pass.
    CompressFallback {
        freed: usize,
    },

    // ── Retry ─────────────────────────────────────────────────────────
    /// The provider returned a transient error; the engine is about to retry.
    ///
    /// `attempt` is 1-indexed. `delay_ms` is the back-off wait before this
    /// attempt. Use to show a "retrying…" indicator rather than a hard error.
    Retrying {
        attempt: u32,
        delay_ms: u64,
        reason: String,
    },

    // ── Session lifecycle ─────────────────────────────────────────────
    /// Emitted by `Session::send()` after each completed turn.
    ///
    /// Carries the turn index (1-based) and cumulative token usage for this turn.
    /// This is a session-level lifecycle event — it is NOT emitted by `Agent::stream()`.
    TurnDone {
        turn: u32,
        usage: TokenUsage,
    },

    // ── Terminal ──────────────────────────────────────────────────────
    Done(RunSummary),
    Error(AgentError),
}

/// How compression was applied.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressMethod {
    /// L1: oversized tool results were truncated.
    BudgetTrim,
    /// L2: old messages were collapsed into placeholders.
    Collapse,
    /// L3: the LLM was asked to summarise old messages.
    Summarize,
    /// L3 summarization was attempted but failed; fell back to L2 collapse.
    L3Failed,
}

// ── ControlHandle ────────────────────────────────────────────────────────────

/// A pause point bundled with the capability to resume it.
///
/// Received inside `AgentEvent::Control`. The handle is cheaply cloneable;
/// only the first response is delivered — subsequent calls are silent no-ops.
///
/// **Must be responded to.** Dropping a handle without calling any method
/// auto-denies the request — the engine treats a dropped sender as a denial
/// so the agent loop can always make progress rather than hanging forever.
///
/// ```rust,ignore
/// AgentEvent::Control(handle) => {
///     println!("Agent wants to: {}", handle.request.description());
///     handle.approve();            // once
///     handle.approve_always();     // for all future calls to this tool
///     handle.deny("not allowed");  // once
///     handle.deny_always("never"); // for all future calls to this tool
/// }
/// ```
#[must_use = "ControlHandle must be responded to; dropping it auto-denies the request"]
#[derive(Clone)]
pub struct ControlHandle {
    /// The details of what the agent is requesting.
    pub request: ControlRequest,
    // Arc<Mutex<Option<...>>> ensures ControlHandle is cheaply Clone-able (e.g.
    // for storage in multiple UI components) while guaranteeing that only the
    // *first* response call (approve/deny/etc.) wins. Subsequent calls are
    // silent no-ops because `take()` consumes the inner sender.
    tx: Arc<Mutex<Option<tokio::sync::oneshot::Sender<ControlResponse>>>>,
}

impl ControlHandle {
    /// Create a handle and its matching receiver (used by the engine).
    pub fn new(request: ControlRequest) -> (Self, tokio::sync::oneshot::Receiver<ControlResponse>) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let handle = Self {
            request,
            tx: Arc::new(Mutex::new(Some(tx))),
        };
        (handle, rx)
    }

    /// Approve this request once.
    pub fn approve(&self) {
        self.respond(ControlDecision::Approve { modification: None });
    }

    /// Approve with a modification note the LLM will see.
    pub fn approve_with(&self, modification: impl Into<String>) {
        self.respond(ControlDecision::Approve {
            modification: Some(modification.into()),
        });
    }

    /// Approve this tool for all future calls in this session.
    ///
    /// The session remembers this decision — subsequent calls to the same
    /// tool will be allowed without prompting.
    pub fn approve_always(&self) {
        self.respond(ControlDecision::ApproveAlways);
    }

    /// Deny this request once. The LLM sees `reason` and will not retry.
    pub fn deny(&self, reason: impl Into<String>) {
        self.respond(ControlDecision::Deny {
            reason: reason.into(),
        });
    }

    /// Deny this tool for all future calls in this session.
    ///
    /// The session remembers this decision — subsequent calls to the same
    /// tool will be blocked without prompting.
    pub fn deny_always(&self, reason: impl Into<String>) {
        self.respond(ControlDecision::DenyAlways {
            reason: reason.into(),
        });
    }

    fn respond(&self, decision: ControlDecision) {
        let response = ControlResponse {
            request_id: self.request.id.clone(),
            decision,
        };
        if let Ok(mut guard) = self.tx.lock() {
            if let Some(tx) = guard.take() {
                let _ = tx.send(response);
            }
        }
    }
}

impl fmt::Debug for ControlHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ControlHandle")
            .field("request", &self.request)
            .finish_non_exhaustive()
    }
}

// ── Control Request / Response ───────────────────────────────────────────────

/// A pause point where the human must respond before the agent continues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlRequest {
    pub id: String,
    pub kind: ControlKind,
}

impl ControlRequest {
    /// A short human-readable description of what is being requested.
    pub fn description(&self) -> &str {
        match &self.kind {
            ControlKind::PermissionRequest { description, .. } => description,
            ControlKind::PlanReview { plan } => plan,
        }
    }

    /// The tool name, if this is a permission request.
    pub fn tool_name(&self) -> Option<&str> {
        match &self.kind {
            ControlKind::PermissionRequest { tool_name, .. } => Some(tool_name),
            _ => None,
        }
    }
}

/// The nature of the control request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ControlKind {
    /// The agent wants to execute a tool and is asking for permission.
    PermissionRequest {
        tool_name: String,
        description: String,
    },
    /// The agent has produced a plan and is asking for review before acting.
    PlanReview { plan: String },
}

/// The human's response to a `ControlRequest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlResponse {
    pub request_id: String,
    pub decision: ControlDecision,
}

impl ControlResponse {
    /// Approve the request identified by `request_id`.
    pub fn approve(request_id: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            decision: ControlDecision::Approve { modification: None },
        }
    }
    /// Deny the request identified by `request_id` with the given reason.
    pub fn deny(request_id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            decision: ControlDecision::Deny {
                reason: reason.into(),
            },
        }
    }
}

/// How the human responded to the control request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ControlDecision {
    /// Allow this specific invocation.
    Approve { modification: Option<String> },
    /// Allow this tool for all future invocations in this session.
    ApproveAlways,
    /// Deny this specific invocation.
    Deny { reason: String },
    /// Deny this tool for all future invocations in this session.
    DenyAlways { reason: String },
}
