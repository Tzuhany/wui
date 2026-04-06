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
// ============================================================================

use serde::{Deserialize, Serialize};

use crate::message::ContentBlock;

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
    pub input_tokens:        u32,
    pub output_tokens:       u32,
    pub cache_read_tokens:   u32,
    pub cache_write_tokens:  u32,
}

impl TokenUsage {
    pub fn total(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

// ── External: Engine → User ──────────────────────────────────────────────────

/// Events emitted to the caller's stream.
///
/// The enum is exhaustive — every possible agent milestone is represented.
/// Match on it to build any kind of consumer: TUI, websocket, log file, test.
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
        ms:    u64,
    },

    // ── HITL ──────────────────────────────────────────────────────────
    /// The agent has paused and is waiting for a human decision.
    Control(ControlRequest),

    // ── Context management ────────────────────────────────────────────
    /// Context compression was applied. Emitted so the caller can log it
    /// or surface it in UI.
    Compressed {
        method: CompressMethod,
        /// Approximate tokens freed.
        freed:  usize,
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

/// A pause point where the human must respond before the agent continues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlRequest {
    pub id:   String,
    pub kind: ControlKind,
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
    pub request_id:   String,
    pub decision:     ControlDecision,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ControlDecision {
    Approve { modification: Option<String> },
    Deny    { reason: String },
}

impl ControlResponse {
    pub fn approve(request_id: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            decision:   ControlDecision::Approve { modification: None },
        }
    }

    pub fn approve_with(request_id: impl Into<String>, modification: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            decision:   ControlDecision::Approve { modification: Some(modification.into()) },
        }
    }

    pub fn deny(request_id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            decision:   ControlDecision::Deny { reason: reason.into() },
        }
    }
}

/// Summary emitted when a run completes successfully.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub stop_reason:  RunStopReason,
    pub iterations:   u32,
    pub usage:        TokenUsage,
    /// All messages produced during this run (user + assistant).
    pub messages:     Vec<ContentBlock>,
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
}

/// A terminal error emitted as the last event in the stream.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct AgentError {
    pub message:   String,
    pub retryable: bool,
}
