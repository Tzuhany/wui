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
                format!("tool '{name}' requires user interaction and cannot run in this mode"),
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
        if session
            .denies(name, check.permission_key, check.matcher)
            .await
        {
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
        if session
            .allows(name, check.permission_key, check.matcher)
            .await
        {
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

/// Convert a human's `ControlResponse` into a factual message the LLM will see.
///
/// Messages state what happened — no directives about what the LLM should do.
pub fn response_to_system_message(response: &ControlResponse) -> String {
    let body = match &response.decision {
        ControlDecision::Approve { modification: None } => {
            "The user approved your request.".to_string()
        }
        ControlDecision::Approve {
            modification: Some(m),
        } => format!("The user approved your request with a modification: {m}"),
        ControlDecision::ApproveAlways => {
            "The user approved your request. This tool is now always-allowed for this session."
                .to_string()
        }
        ControlDecision::Deny { reason } => {
            format!("The user denied your request. Reason: {reason}")
        }
        ControlDecision::DenyAlways { reason } => format!(
            "The user denied your request. Reason: {reason}. \
             This tool is now always-denied for this session."
        ),
    };
    wui_core::fmt::system_reminder(&body)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_check(name: &str) -> PermissionCheck<'_> {
        PermissionCheck {
            tool_name: name,
            permission_key: None,
            is_readonly: false,
            requires_interaction: false,
            matcher: None,
        }
    }

    #[tokio::test]
    async fn auto_mode_allows_all() {
        let rules = PermissionRules::new();
        let session = SessionPermissions::new();
        let v = rules
            .verdict(&session, &PermissionMode::Auto, &simple_check("bash"))
            .await;
        assert!(matches!(
            v,
            PermissionVerdict::Allowed {
                source: PermissionSource::ModeAuto
            }
        ));
    }

    #[tokio::test]
    async fn static_deny_overrides_auto() {
        let rules = PermissionRules::new().deny("bash");
        let session = SessionPermissions::new();
        let v = rules
            .verdict(&session, &PermissionMode::Auto, &simple_check("bash"))
            .await;
        assert!(matches!(
            v,
            PermissionVerdict::Denied {
                source: PermissionSource::StaticDeny,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn static_allow_in_ask_mode() {
        let rules = PermissionRules::new().allow("fetch");
        let session = SessionPermissions::new();
        let v = rules
            .verdict(&session, &PermissionMode::Ask, &simple_check("fetch"))
            .await;
        assert!(matches!(
            v,
            PermissionVerdict::Allowed {
                source: PermissionSource::StaticAllow
            }
        ));
    }

    #[tokio::test]
    async fn ask_mode_needs_approval() {
        let rules = PermissionRules::new();
        let session = SessionPermissions::new();
        let v = rules
            .verdict(&session, &PermissionMode::Ask, &simple_check("bash"))
            .await;
        assert!(matches!(v, PermissionVerdict::NeedsApproval));
    }

    #[tokio::test]
    async fn session_deny_overrides_static_allow() {
        // Session deny is checked BEFORE static allow in the pipeline
        // (step 3 before step 4), so session deny wins.
        let rules = PermissionRules::new().allow("bash");
        let session = SessionPermissions::new();
        session.set_always_deny("bash").await;
        let v = rules
            .verdict(&session, &PermissionMode::Auto, &simple_check("bash"))
            .await;
        assert!(matches!(
            v,
            PermissionVerdict::Denied {
                source: PermissionSource::SessionDeny,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn session_allow_in_ask_mode() {
        let rules = PermissionRules::new();
        let session = SessionPermissions::new();
        session.set_always_allow("bash").await;
        let v = rules
            .verdict(&session, &PermissionMode::Ask, &simple_check("bash"))
            .await;
        assert!(matches!(
            v,
            PermissionVerdict::Allowed {
                source: PermissionSource::SessionAllow
            }
        ));
    }

    #[tokio::test]
    async fn readonly_allows_readonly_tool() {
        let rules = PermissionRules::new();
        let session = SessionPermissions::new();
        let check = PermissionCheck {
            tool_name: "read_file",
            permission_key: None,
            is_readonly: true,
            requires_interaction: false,
            matcher: None,
        };
        let v = rules
            .verdict(&session, &PermissionMode::Readonly, &check)
            .await;
        assert!(matches!(
            v,
            PermissionVerdict::Allowed {
                source: PermissionSource::ModeReadonly
            }
        ));
    }

    #[tokio::test]
    async fn readonly_denies_non_readonly_tool() {
        let rules = PermissionRules::new();
        let session = SessionPermissions::new();
        let v = rules
            .verdict(&session, &PermissionMode::Readonly, &simple_check("bash"))
            .await;
        assert!(matches!(
            v,
            PermissionVerdict::Denied {
                source: PermissionSource::ModeReadonlyDenied,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn structural_deny_for_interactive_in_auto() {
        let rules = PermissionRules::new();
        let session = SessionPermissions::new();
        let check = PermissionCheck {
            tool_name: "prompt_user",
            permission_key: None,
            is_readonly: false,
            requires_interaction: true,
            matcher: None,
        };
        let v = rules.verdict(&session, &PermissionMode::Auto, &check).await;
        assert!(matches!(
            v,
            PermissionVerdict::Denied {
                source: PermissionSource::StructuralDeny,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn callback_mode_delegates() {
        let rules = PermissionRules::new();
        let session = SessionPermissions::new();
        let mode = PermissionMode::Callback(std::sync::Arc::new(|name, _| name == "allowed_tool"));

        let v = rules
            .verdict(&session, &mode, &simple_check("allowed_tool"))
            .await;
        // Callback mode falls into NeedsApproval in the verdict pipeline,
        // the actual callback is evaluated at a higher level (auth.rs).
        assert!(matches!(v, PermissionVerdict::NeedsApproval));
    }
}
