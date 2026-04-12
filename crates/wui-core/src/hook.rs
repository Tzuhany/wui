// ============================================================================
// Hook — the agent's conscience.
//
// Hooks intercept the agent loop at five points and can block or allow
// each action. When a hook blocks, the reason is injected into the
// conversation as a system message — the LLM sees it and self-corrects.
// This creates a natural feedback loop without requiring any special
// "error handling mode".
//
// ── Six hook phases ───────────────────────────────────────────────────────────
//
//   PreToolUse      — before a tool runs. Can mutate input or block entirely.
//   PostToolUse     — after a tool succeeds. Can block the output from reaching
//                     the LLM (inject a system notice instead).
//   PostToolFailure — after a tool fails. Receives the input alongside the
//                     failure output. Separate from PostToolUse so hooks can
//                     cleanly specialise: audit on failure, transform on success.
//   PreCompact      — before the context is compressed (L1/L2/L3). Block to
//                     inject content that should survive summarisation. The
//                     Block reason is inserted into the conversation before
//                     compression runs, so the summariser sees it.
//   PreStop         — before the run stops for ANY reason (completed, budget
//                     exhausted, max iterations, etc.). Block to force a retry;
//                     use `stop_hook_active` to avoid infinite loops.
//
// Hooks are composable: the engine runs all registered hooks in order.
// Blocks short-circuit immediately; mutations are threaded forward so later
// hooks see the updated value. Register multiple hooks freely.
// ============================================================================

use std::borrow::Borrow;
use std::collections::HashSet;
use std::fmt;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::event::{RunStopReason, RunSummary};
use crate::message::Message;
use crate::tool::ToolOutput;

// ── SessionId ────────────────────────────────────────────────────────────────

/// A unique identifier for a multi-turn session.
///
/// Used to load/save session state from a `SessionStore`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(String);

impl SessionId {
    /// Create a new `SessionId`.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Return the inner string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume and return the inner `String`.
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl std::ops::Deref for SessionId {
    type Target = str;
    fn deref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for SessionId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for SessionId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

impl PartialEq<str> for SessionId {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<String> for SessionId {
    fn eq(&self, other: &String) -> bool {
        self.0 == *other
    }
}

impl PartialEq<&str> for SessionId {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

impl Borrow<str> for SessionId {
    fn borrow(&self) -> &str {
        &self.0
    }
}

/// A hook that intercepts the agent loop at key moments.
///
/// Hooks are `Send + Sync + 'static` for the same reason tools are:
/// the engine may invoke them from spawned tasks.
///
/// Implement `evaluate()`. Override `handles()` to skip events you don't care
/// about — this avoids an unnecessary async dispatch for irrelevant events.
#[async_trait]
pub trait Hook: Send + Sync + 'static {
    /// Whether this hook wants to see the given event.
    ///
    /// Defaults to `true` (handle all events). Override to filter:
    /// ```rust,ignore
    /// fn handles(&self, event: &HookEvent<'_>) -> bool {
    ///     matches!(event, HookEvent::PreToolUse { .. })
    /// }
    /// ```
    fn handles(&self, _event: &HookEvent<'_>) -> bool {
        true
    }

    /// Evaluate the event and decide whether to allow or block it.
    async fn evaluate(&self, event: &HookEvent<'_>) -> HookDecision;
}

// ── Hook Event ────────────────────────────────────────────────────────────────

/// The six moments where hooks can intervene.
#[derive(Debug)]
pub enum HookEvent<'a> {
    /// The LLM has decided to call a tool. The hook can inspect the name
    /// and arguments and block the call before any execution happens.
    PreToolUse { name: &'a str, input: &'a Value },

    /// A tool has returned a successful result. The hook can inspect the
    /// output and block the loop from continuing (e.g. to enforce output
    /// policies). Receives no input — use `PreToolUse` if you need both.
    PostToolUse {
        name: &'a str,
        output: &'a ToolOutput,
    },

    /// A tool invocation ended in failure. Receives both the input that was
    /// attempted and the failure output. Separate from `PostToolUse` so hooks
    /// can handle success and failure paths independently — an audit hook
    /// that logs every failure need not filter inside `PostToolUse`.
    ///
    /// A `Block` decision here injects a system notice so the LLM understands
    /// the failure context was intentionally suppressed.
    PostToolFailure {
        name: &'a str,
        input: &'a Value,
        output: &'a ToolOutput,
    },

    /// The context window is about to be compressed.
    ///
    /// Fires before every compression attempt — L1 budget trim, L2 collapse,
    /// and L3 LLM summarisation. This is your last chance to inject context
    /// that must survive the summariser:
    ///
    /// ```rust,ignore
    /// async fn evaluate(&self, event: &HookEvent<'_>) -> HookDecision {
    ///     if let HookEvent::PreCompact { .. } = event {
    ///         return HookDecision::block(
    ///             "The user's current task is: refactor the auth module. \
    ///              Preserve all decisions made so far.",
    ///         );
    ///     }
    ///     HookDecision::Allow
    /// }
    /// ```
    ///
    /// The Block reason is inserted into the conversation as a system message
    /// immediately before compression runs. Allow lets compression proceed
    /// without any injection.
    PreCompact {
        /// The current message history, in conversation order.
        messages: &'a [Message],
    },

    // ── Lifecycle events (informational, decision is ignored) ──────────
    /// A session has started (or resumed from a store).
    SessionStart { session_id: &'a SessionId },

    /// A session turn is about to begin. Fires after history is loaded
    /// but before the LLM is called.
    TurnStart { messages: &'a [Message] },

    /// A session turn has completed.
    TurnEnd { summary: &'a RunSummary },

    /// A sub-agent is about to start.
    SubagentStart { name: &'a str, prompt: &'a str },

    /// A sub-agent has finished.
    SubagentEnd {
        name: &'a str,
        result: Result<&'a str, &'a str>,
    },

    // ── Decision events ─────────────────────────────────────────────
    /// The run is about to stop for any reason. The hook runs before the
    /// `RunSummary` is returned to the caller.
    ///
    /// - `response`: the last assistant text (empty string if the run ends
    ///   before any text is produced, e.g. at the very first MaxIterations).
    /// - `stop_reason`: why the run is stopping.
    /// - `stop_hook_active`: `true` when this hook already blocked this stop
    ///   attempt once. A well-written hook should return `Allow` when this
    ///   flag is set to prevent an infinite loop.
    /// - `messages`: the current message history. Hooks can read this to
    ///   make informed decisions. For `ContextOverflow`, a hook that returns
    ///   `Block` may implement a custom degradation strategy (e.g. dropping
    ///   old tool results) — the runtime will re-check pressure after the
    ///   hook runs and continue if it was relieved.
    ///
    /// Block to force an additional iteration. MutateOutput to rewrite the
    /// final assistant response before it is delivered (honoured for
    /// `Completed` only). Allow to proceed normally.
    PreStop {
        response: &'a str,
        stop_reason: RunStopReason,
        stop_hook_active: bool,
        /// The current message history at the time of stopping.
        messages: &'a [Message],
    },
}

// ── Hook Decision ─────────────────────────────────────────────────────────────

/// What a hook decides to do with an event.
///
/// Not all decisions are honoured for all events. The matrix:
///
/// | Event             | Allow | Block | Mutate (input) | MutateOutput |
/// |-------------------|-------|-------|----------------|--------------|
/// | PreToolUse        | yes   | yes   | **yes**        | ignored      |
/// | PostToolUse       | yes   | yes   | ignored        | **yes**      |
/// | PostToolFailure   | yes   | yes   | ignored        | ignored      |
/// | PreCompact        | yes   | **yes** (inject) | ignored | ignored |
/// | PreStop           | yes   | **yes** (retry)  | ignored | **yes** (Completed only) |
/// | SessionStart      | ignored | ignored | ignored   | ignored      |
/// | TurnStart         | ignored | ignored | ignored   | ignored      |
/// | TurnEnd           | ignored | ignored | ignored   | ignored      |
/// | SubagentStart     | ignored | ignored | ignored   | ignored      |
/// | SubagentEnd       | ignored | ignored | ignored   | ignored      |
///
/// **Lifecycle events** (Session/Turn/Subagent) are fire-and-forget
/// notifications. The decision is always ignored — hooks observe but
/// cannot alter behaviour. Use these for logging, metrics, and audit.
#[derive(Debug, Clone)]
pub enum HookDecision {
    /// Let the action proceed unchanged.
    Allow,

    /// Let the action proceed, but replace the tool's input JSON.
    ///
    /// Honoured for `PreToolUse` only. Use for input sanitisation, argument
    /// normalisation, or injecting caller-side context before the tool runs:
    ///
    /// ```rust,ignore
    /// fn evaluate(&self, event: &HookEvent<'_>) -> HookDecision {
    ///     if let HookEvent::PreToolUse { name: "bash", input } = event {
    ///         let mut safe = input.clone();
    ///         safe["command"] = sanitize(input["command"].as_str().unwrap_or(""));
    ///         return HookDecision::mutate(safe);
    ///     }
    ///     HookDecision::Allow
    /// }
    /// ```
    Mutate { input: serde_json::Value },

    /// Let the action proceed, but replace the text content with a new string.
    ///
    /// Honoured for `PostToolUse` and `PreStop`. Use to redact PII, strip
    /// secrets, normalise output formatting, or rewrite a final response:
    ///
    /// - In `PostToolUse`: replaces the tool output the LLM sees on the next turn.
    /// - In `PreStop`: rewrites the assistant's final response before it is
    ///   returned to the caller (honoured for `Completed` stop reason only).
    ///
    /// ```rust,ignore
    /// async fn evaluate(&self, event: &HookEvent<'_>) -> HookDecision {
    ///     if let HookEvent::PostToolUse { name: "db_query", output } = event {
    ///         let redacted = redact_pii(&output.content);
    ///         return HookDecision::mutate_output(redacted);
    ///     }
    ///     HookDecision::Allow
    /// }
    /// ```
    MutateOutput { content: String },

    /// Prevent the action. The reason is injected into the conversation
    /// as a system-level notice so the LLM can see it and self-correct.
    Block { reason: String },
}

impl HookDecision {
    /// Allow the action to proceed.
    pub fn allow() -> Self {
        Self::Allow
    }

    /// Allow the action but replace the tool input.
    pub fn mutate(input: serde_json::Value) -> Self {
        Self::Mutate { input }
    }

    /// Replace the text content in `PostToolUse` or `PreStop` contexts.
    pub fn mutate_output(content: impl Into<String>) -> Self {
        Self::MutateOutput {
            content: content.into(),
        }
    }

    /// Block the action with the given reason.
    pub fn block(reason: impl Into<String>) -> Self {
        Self::Block {
            reason: reason.into(),
        }
    }

    /// `true` when this decision blocks the action.
    pub fn is_blocked(&self) -> bool {
        matches!(self, Self::Block { .. })
    }
}

// ── Built-in Hooks ────────────────────────────────────────────────────────────

/// Blocks any tool whose name is in the deny list.
///
/// ```rust,ignore
/// Agent::builder()
///     .hook(DenyList::new(["bash", "file_write"]))
///     .build()
/// ```
pub struct DenyList {
    denied: HashSet<String>,
}

impl DenyList {
    /// Create a deny list from the given tool names.
    pub fn new(tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            denied: tools.into_iter().map(Into::into).collect(),
        }
    }
}

#[async_trait]
impl Hook for DenyList {
    fn handles(&self, event: &HookEvent<'_>) -> bool {
        matches!(event, HookEvent::PreToolUse { .. })
    }

    async fn evaluate(&self, event: &HookEvent<'_>) -> HookDecision {
        if let HookEvent::PreToolUse { name, .. } = event {
            if self.denied.contains(*name) {
                return HookDecision::block(format!(
                    "tool '{name}' is not allowed in this context"
                ));
            }
        }
        HookDecision::Allow
    }
}
