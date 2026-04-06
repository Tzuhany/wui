// ============================================================================
// Permission System — earned trust.
//
// Three modes cover the full spectrum from fully automated to fully cautious.
// The key design decision: `Ask` mode suspends the loop via ControlHandle —
// the caller receives both the request and the capability to respond in one
// object. No polling, no threads blocked, pure async suspension.
//
// ── SessionPermissions — memory within a session ──────────────────────────────
//
// When a user responds with `ApproveAlways` or `DenyAlways`, the session
// remembers that decision for the rest of the conversation. Subsequent
// invocations of the same tool skip the permission prompt entirely.
//
// This eliminates friction without sacrificing safety: the user makes the
// decision once, explicitly, and it persists until the session ends.
// Starting a new session always starts from a clean permission slate.
// ============================================================================

use std::collections::HashSet;
use std::sync::Arc;

use tokio::sync::RwLock;

use wuhu_core::event::{ControlDecision, ControlRequest, ControlResponse};

// ── Permission Mode ───────────────────────────────────────────────────────────

/// How the engine handles tool permission checks.
#[derive(Debug, Clone, Default)]
pub enum PermissionMode {
    /// All tools are allowed without asking. Use in trusted, automated
    /// environments where every tool in the registry is pre-approved.
    Auto,

    /// Tools require human approval before execution. The loop pauses,
    /// emits `AgentEvent::Control(handle)`, and resumes when the handle
    /// is responded to.
    ///
    /// `ApproveAlways` and `DenyAlways` responses are remembered for
    /// the lifetime of the session.
    #[default]
    Ask,

    /// Only tools that declare `is_readonly() == true` are allowed.
    /// All others are denied immediately without prompting the user.
    /// Use for read-only agents: research, analysis, retrieval.
    Readonly,
}

impl PermissionMode {
    pub fn is_auto(&self) -> bool     { matches!(self, Self::Auto)     }
    pub fn is_ask(&self) -> bool      { matches!(self, Self::Ask)      }
    pub fn is_readonly(&self) -> bool { matches!(self, Self::Readonly) }
}

// ── Session Permissions ───────────────────────────────────────────────────────

/// Per-session permission memory.
///
/// Tracks which tools have been always-approved or always-denied by the user.
/// Shared across all turns in a session; discarded when the session ends.
///
/// Thread-safe: uses a `RwLock` so concurrent tool lookups are lock-free.
#[derive(Debug, Default, Clone)]
pub struct SessionPermissions {
    inner: Arc<RwLock<PermissionsInner>>,
}

#[derive(Debug, Default)]
struct PermissionsInner {
    always_allow: HashSet<String>,
    always_deny:  HashSet<String>,
}

impl SessionPermissions {
    pub fn new() -> Self {
        Self::default()
    }

    /// `true` if this tool has been always-approved in this session.
    pub async fn is_always_allowed(&self, tool: &str) -> bool {
        self.inner.read().await.always_allow.contains(tool)
    }

    /// `true` if this tool has been always-denied in this session.
    pub async fn is_always_denied(&self, tool: &str) -> bool {
        self.inner.read().await.always_deny.contains(tool)
    }

    /// Record an always-allow decision for a tool.
    pub async fn set_always_allow(&self, tool: impl Into<String>) {
        let mut inner = self.inner.write().await;
        let tool = tool.into();
        inner.always_deny.remove(&tool);
        inner.always_allow.insert(tool);
    }

    /// Record an always-deny decision for a tool.
    pub async fn set_always_deny(&self, tool: impl Into<String>) {
        let mut inner = self.inner.write().await;
        let tool = tool.into();
        inner.always_allow.remove(&tool);
        inner.always_deny.insert(tool);
    }
}

// ── Decision ─────────────────────────────────────────────────────────────────

/// The outcome of a permission check.
#[derive(Debug)]
pub enum PermissionOutcome {
    Allowed,
    Denied { reason: String },
    /// The loop must create a ControlHandle, emit it, and await the response.
    NeedsApproval,
}

/// Evaluate a tool call against the current permission mode.
///
/// `tool_is_readonly` is the result of `Tool::is_readonly()` and is relevant
/// only in `Readonly` mode: readonly tools are allowed, all others are denied.
pub fn check(
    mode:             &PermissionMode,
    request:          &ControlRequest,
    tool_is_readonly: bool,
) -> PermissionOutcome {
    match mode {
        PermissionMode::Auto => PermissionOutcome::Allowed,

        PermissionMode::Readonly => {
            if tool_is_readonly {
                PermissionOutcome::Allowed
            } else {
                PermissionOutcome::Denied {
                    reason: format!(
                        "tool '{}' has side effects and is not permitted in read-only mode",
                        tool_name_from_request(request),
                    ),
                }
            }
        }

        PermissionMode::Ask => PermissionOutcome::NeedsApproval,
    }
}

fn tool_name_from_request(request: &ControlRequest) -> &str {
    request.tool_name().unwrap_or("(unknown)")
}

/// Convert a human's `ControlResponse` into a message the LLM will see.
pub fn response_to_system_message(response: &ControlResponse) -> String {
    match &response.decision {
        ControlDecision::Approve { modification: None } =>
            "The user approved your request. You may proceed.".to_string(),

        ControlDecision::Approve { modification: Some(m) } =>
            format!("The user approved your request with the following modification: {m}. Proceed accordingly."),

        ControlDecision::ApproveAlways =>
            "The user approved your request and will always allow this tool. You may proceed.".to_string(),

        ControlDecision::Deny { reason } =>
            format!("The user denied your request. Reason: {reason}. Do not attempt this action again."),

        ControlDecision::DenyAlways { reason } =>
            format!("The user denied your request and will never allow this tool. Reason: {reason}. Do not attempt this action again."),
    }
}
