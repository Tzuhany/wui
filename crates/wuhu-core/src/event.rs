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
use crate::tool::FailureKind;

// ── Internal: LLM → Engine ───────────────────────────────────────────────────

/// Raw events emitted by a Provider's stream.
///
/// The engine consumes these directly. They should never appear in user code.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextDelta      { text: String },
    ThinkingDelta  { text: String },

    /// The LLM has started describing a tool call.
    ToolUseStart   { id: String, name: String },

    /// A chunk of JSON input for an in-progress tool call.
    ToolInputDelta { id: String, chunk: String },

    /// The LLM has finished describing the tool call. The engine submits it
    /// to the executor immediately upon receiving this event.
    ToolUseEnd     { id: String },

    MessageEnd     { usage: TokenUsage, stop_reason: StopReason },

    /// A retryable or fatal error from the provider.
    Error          { message: String, retryable: bool },
}

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

/// Token consumption for a single LLM call.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens:       u32,
    pub output_tokens:      u32,
    pub cache_read_tokens:  u32,
    pub cache_write_tokens: u32,
}

impl TokenUsage {
    /// Total non-cache tokens consumed.
    pub fn total(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }

    /// All tokens consumed including cache activity.
    pub fn total_with_cache(&self) -> u32 {
        self.total() + self.cache_read_tokens + self.cache_write_tokens
    }
}

// ── External: Engine → User ──────────────────────────────────────────────────

/// Events emitted to the caller's stream.
///
/// The enum is exhaustive — every possible agent milestone is represented.
/// Match on it to build any kind of consumer: TUI, websocket, log file, test.
///
/// Only `Control` requires caller action. All other variants are informational.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    // ── Streaming text ────────────────────────────────────────────────
    TextDelta    (String),
    ThinkingDelta(String),

    // ── Tool lifecycle ────────────────────────────────────────────────
    ToolStart {
        id:    String,
        name:  String,
        input: serde_json::Value,
    },
    ToolDone {
        id:     String,
        name:   String,
        output: String,
        ms:     u64,
    },
    ToolError {
        id:    String,
        name:  String,
        error: String,
        /// Structured reason for the failure. Use this to tailor UI and
        /// recovery logic — a `PermissionDenied` warrants different
        /// treatment than an `Execution` error.
        kind:  FailureKind,
        ms:    u64,
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
        tool_id:   String,
        tool_name: String,
        artifact:  crate::tool::Artifact,
    },

    // ── Tool progress ─────────────────────────────────────────────────
    /// Incremental progress reported by a running tool via `ctx.report()`.
    ///
    /// Emitted in real time — use this to stream tool status to a UI
    /// without waiting for the tool to complete.
    ToolProgress {
        tool_id:   String,
        tool_name: String,
        text:      String,
    },

    // ── Context management ────────────────────────────────────────────
    /// Context compression was applied.
    Compressed {
        method: CompressMethod,
        /// Approximate tokens freed.
        freed:  usize,
    },

    // ── Retry ─────────────────────────────────────────────────────────
    /// The provider returned a transient error; the engine is about to retry.
    ///
    /// `attempt` is 1-indexed. `delay_ms` is the back-off wait before this
    /// attempt. Use to show a "retrying…" indicator rather than a hard error.
    Retrying {
        attempt:  u32,
        delay_ms: u64,
        reason:   String,
    },

    // ── Terminal ──────────────────────────────────────────────────────
    Done  (RunSummary),
    Error (AgentError),
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
}

// ── ControlHandle ─────────────────────────────────────────────────────────────

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
    tx: Arc<Mutex<Option<tokio::sync::oneshot::Sender<ControlResponse>>>>,
}

impl ControlHandle {
    /// Create a handle and its matching receiver (used by the engine).
    pub fn new(
        request: ControlRequest,
    ) -> (Self, tokio::sync::oneshot::Receiver<ControlResponse>) {
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
        self.respond(ControlDecision::Deny { reason: reason.into() });
    }

    /// Deny this tool for all future calls in this session.
    ///
    /// The session remembers this decision — subsequent calls to the same
    /// tool will be blocked without prompting.
    pub fn deny_always(&self, reason: impl Into<String>) {
        self.respond(ControlDecision::DenyAlways { reason: reason.into() });
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

// ── Control request / response ────────────────────────────────────────────────

/// A pause point where the human must respond before the agent continues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlRequest {
    pub id:   String,
    pub kind: ControlKind,
}

impl ControlRequest {
    /// A short human-readable description of what is being requested.
    pub fn description(&self) -> &str {
        match &self.kind {
            ControlKind::PermissionRequest { description, .. } => description,
            ControlKind::PlanReview        { plan }            => plan,
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
    PermissionRequest { tool_name: String, description: String },
    /// The agent has produced a plan and is asking for review before acting.
    PlanReview        { plan: String },
}

/// The human's response to a `ControlRequest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlResponse {
    pub request_id: String,
    pub decision:   ControlDecision,
}

impl ControlResponse {
    pub fn approve(request_id: impl Into<String>) -> Self {
        Self { request_id: request_id.into(), decision: ControlDecision::Approve { modification: None } }
    }
    pub fn deny(request_id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self { request_id: request_id.into(), decision: ControlDecision::Deny { reason: reason.into() } }
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

// ── Run summary ───────────────────────────────────────────────────────────────

/// Summary emitted when a run completes successfully.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub stop_reason: RunStopReason,
    pub iterations:  u32,
    pub usage:       TokenUsage,
    /// The full conversation at the time the run ended (user + assistant turns).
    pub messages:    Vec<Message>,
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
}

/// A terminal error emitted as the last event in the stream.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct AgentError {
    pub message:   String,
    pub retryable: bool,
}
