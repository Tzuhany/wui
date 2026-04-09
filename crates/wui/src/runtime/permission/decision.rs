// ── Decision — the permission pipeline output ───────────────────────────────

use wui_core::event::{ControlDecision, ControlRequest, ControlResponse};

use super::mode::PermissionMode;
use super::rules::PermissionRules;
use super::session::SessionPermissions;

/// The outcome of a permission check.
#[derive(Debug)]
pub enum PermissionOutcome {
    Allowed,
    Denied {
        reason: String,
    },
    /// The loop must create a ControlHandle, emit it, and await the response.
    NeedsApproval,
}

/// Per-invocation data needed by the permission pipeline.
///
/// Bundles the tool identity, semantic flags, and optional matcher so
/// callers don't have to thread 5+ arguments through every call.
pub struct PermissionCheck<'a> {
    pub tool_name: &'a str,
    pub permission_key: Option<&'a str>,
    pub is_readonly: bool,
    pub requires_interaction: bool,
    pub matcher: Option<&'a (dyn Fn(&str) -> bool + Send + Sync)>,
}

/// Which step of the permission pipeline produced the decision.
///
/// Exposed for auditing and debugging — lets callers answer "why was
/// this tool allowed/denied?" without guessing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionSource {
    StaticDeny,
    StaticAllow,
    SessionDeny,
    SessionAllow,
    ModeAuto,
    ModeReadonly,
    ModeReadonlyDenied,
    ModeAsk,
    StructuralDeny,
}

#[derive(Debug)]
pub enum PermissionVerdict {
    Allowed {
        source: PermissionSource,
    },
    Denied {
        reason: String,
        source: PermissionSource,
    },
    NeedsApproval,
}

impl PermissionVerdict {
    fn allowed(source: PermissionSource, tool_name: &str) -> Self {
        trace_decision(tool_name, &source, "allowed");
        Self::Allowed { source }
    }

    fn denied(reason: String, source: PermissionSource, tool_name: &str) -> Self {
        trace_decision(tool_name, &source, "denied");
        Self::Denied { reason, source }
    }
}

fn trace_decision(tool_name: &str, source: &PermissionSource, outcome: &str) {
    tracing::info!(
        tool.name = %tool_name,
        permission.source = ?source,
        permission.outcome = %outcome,
        "wui.permission.decision"
    );
}

// ── Verdict pipeline ────────────────────────────────────────────────────────

impl PermissionRules {
    /// Full permission check combining static rules, session memory, and mode.
    ///
    /// **Evaluation order** (first match wins):
    /// 1. Structural — interaction-requiring tools denied in Auto mode.
    /// 2. Static deny — hard developer constraint.
    /// 3. Session deny — user's runtime decision.
    /// 4. Static allow — developer pre-approval.
    /// 5. Session allow — user's runtime decision.
    /// 6. Mode-based (Auto/Readonly/Ask/Callback).
    pub async fn verdict(
        &self,
        session: &SessionPermissions,
        mode: &PermissionMode,
        check: &PermissionCheck<'_>,
    ) -> PermissionVerdict {
        let name = check.tool_name;

        // 1. Structural: interaction-requiring tools can't run headlessly.
        if check.requires_interaction && matches!(mode, PermissionMode::Auto) {
            return PermissionVerdict::denied(
                format!(
                    "tool '{name}' requires user interaction and cannot run in Auto mode; \
                         switch to PermissionMode::Ask or disable this tool for headless runs"
                ),
                PermissionSource::StructuralDeny,
                name,
            );
        }

        // Evaluate static rules once (used in steps 2 and 4).
        let static_rule = self.evaluate_with_matcher(name, check.permission_key, check.matcher);

        // 2. Static deny rules.
        if static_rule == Some(false) {
            return PermissionVerdict::denied(
                format!("tool '{name}' is in the configured deny list"),
                PermissionSource::StaticDeny,
                name,
            );
        }

        // 3. Session always-denied.
        if session.is_always_denied(name).await {
            return PermissionVerdict::denied(
                format!("tool '{name}' was previously denied for this session"),
                PermissionSource::SessionDeny,
                name,
            );
        }

        // 4. Static allow rules.
        if static_rule == Some(true) {
            return PermissionVerdict::allowed(PermissionSource::StaticAllow, name);
        }

        // 5. Session always-allowed.
        if session.is_always_allowed(name).await {
            return PermissionVerdict::allowed(PermissionSource::SessionAllow, name);
        }

        // 6. Mode-based check.
        match mode {
            PermissionMode::Auto => PermissionVerdict::allowed(PermissionSource::ModeAuto, name),
            PermissionMode::Readonly if check.is_readonly => {
                PermissionVerdict::allowed(PermissionSource::ModeReadonly, name)
            }
            PermissionMode::Readonly => PermissionVerdict::denied(
                format!("tool '{name}' has side effects and is not permitted in read-only mode"),
                PermissionSource::ModeReadonlyDenied,
                name,
            ),
            PermissionMode::Ask | PermissionMode::Callback(_) => {
                trace_decision(name, &PermissionSource::ModeAsk, "needs_approval");
                PermissionVerdict::NeedsApproval
            }
        }
    }
}

// ── Mode-level check (for HITL fallback) ────────────────────────────────────

/// Evaluate a tool call against the current permission mode.
pub fn check(
    mode: &PermissionMode,
    request: &ControlRequest,
    tool_is_readonly: bool,
    input: &serde_json::Value,
) -> PermissionOutcome {
    let name = request.tool_name().unwrap_or("(unknown)");
    match mode {
        PermissionMode::Auto => PermissionOutcome::Allowed,
        PermissionMode::Readonly if tool_is_readonly => PermissionOutcome::Allowed,
        PermissionMode::Readonly => PermissionOutcome::Denied {
            reason: format!(
                "tool '{name}' has side effects and is not permitted in read-only mode"
            ),
        },
        PermissionMode::Ask => PermissionOutcome::NeedsApproval,
        PermissionMode::Callback(f) => {
            if f(name, input) {
                PermissionOutcome::Allowed
            } else {
                PermissionOutcome::Denied {
                    reason: "denied by approval callback".to_string(),
                }
            }
        }
    }
}

// ── Response conversion ─────────────────────────────────────────────────────

/// Convert a human's `ControlResponse` into a message the LLM will see.
pub fn response_to_system_message(response: &ControlResponse) -> String {
    let body = match &response.decision {
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
    };
    wui_core::fmt::system_reminder(&body)
}
