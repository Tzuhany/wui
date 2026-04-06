// ============================================================================
// Permission System — earned trust.
//
// Three modes cover the full spectrum from fully automated to fully cautious.
// The key design decision: `Ask` mode uses a oneshot channel to suspend the
// loop — no polling, no threads blocked, pure async suspension.
//
// When the loop suspends on `rx.await`, no CPU is consumed. The session
// can sit paused for days waiting for a human to respond. When they do,
// `tx.send()` resumes it from exactly the right point.
// ============================================================================

use tokio::sync::oneshot;
use wuhu_core::event::{ControlDecision, ControlRequest, ControlResponse};

// ── Permission Mode ───────────────────────────────────────────────────────────

/// How the engine handles tool permission checks.
#[derive(Debug, Clone, Default)]
pub enum PermissionMode {
    /// All tools are allowed without asking. Use in trusted, automated
    /// environments where every tool in the registry is pre-approved.
    Auto,

    /// Tools require human approval before execution. The loop pauses,
    /// emits `AgentEvent::Control`, and resumes when `respond()` is called.
    #[default]
    Ask,

    /// Only tools that return `true` from `is_readonly()` are allowed.
    /// All others are blocked. Use for read-only agents (research, analysis).
    Readonly,
}

impl PermissionMode {
    pub fn is_auto(&self) -> bool { matches!(self, Self::Auto) }
    pub fn is_ask(&self) -> bool  { matches!(self, Self::Ask)  }
}

// ── Pending Approval ─────────────────────────────────────────────────────────

/// A suspended tool call waiting for a human decision.
///
/// Constructed by the loop when `PermissionMode::Ask` is active.
/// Handed to the session so the caller can send a response.
pub struct PendingApproval {
    pub request: ControlRequest,
    tx: oneshot::Sender<ControlResponse>,
}

impl PendingApproval {
    pub fn new(request: ControlRequest) -> (Self, oneshot::Receiver<ControlResponse>) {
        let (tx, rx) = oneshot::channel();
        (Self { request, tx }, rx)
    }

    /// Send a response, resuming the suspended loop.
    pub fn respond(self, response: ControlResponse) {
        // If the receiver is gone (session dropped), silently discard.
        let _ = self.tx.send(response);
    }
}

// ── Decision ─────────────────────────────────────────────────────────────────

/// The outcome of a permission check.
pub enum PermissionOutcome {
    Allowed,
    Denied { reason: String },
    /// The loop must suspend and wait for a human decision.
    NeedsApproval(PendingApproval, oneshot::Receiver<ControlResponse>),
}

/// Evaluate a tool call against the current permission mode.
pub fn check(
    mode:    &PermissionMode,
    request: ControlRequest,
) -> PermissionOutcome {
    match mode {
        PermissionMode::Auto     => PermissionOutcome::Allowed,
        PermissionMode::Readonly => PermissionOutcome::Denied {
            reason: format!("tool '{}' is not allowed in read-only mode", kind_tool_name(&request.kind)),
        },
        PermissionMode::Ask      => {
            let (approval, rx) = PendingApproval::new(request);
            PermissionOutcome::NeedsApproval(approval, rx)
        }
    }
}

fn kind_tool_name(kind: &wuhu_core::event::ControlKind) -> &str {
    match kind {
        wuhu_core::event::ControlKind::PermissionRequest { tool_name, .. } => tool_name,
        wuhu_core::event::ControlKind::PlanReview { .. }                   => "(plan review)",
    }
}

/// Convert a human's `ControlResponse` into a message the LLM will see.
pub fn response_to_system_message(response: &ControlResponse) -> String {
    match &response.decision {
        ControlDecision::Approve { modification: None } =>
            "The user approved your request. You may proceed.".to_string(),

        ControlDecision::Approve { modification: Some(m) } =>
            format!("The user approved your request with the following modification: {m}. Proceed accordingly."),

        ControlDecision::Deny { reason } =>
            format!("The user denied your request. Reason: {reason}. Do not attempt this action again."),
    }
}
