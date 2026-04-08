// ============================================================================
// Permission System — earned trust.
//
// Three modes cover the full spectrum from fully automated to fully cautious.
// The key design decision: `Ask` mode suspends the loop via ControlHandle —
// the caller receives both the request and the capability to respond in one
// object. No polling, no threads blocked, pure async suspension.
//
// ── PermissionRules — static allow/deny ──────────────────────────────────────
//
// Static rules are developer intent: set at build time via AgentBuilder and
// applied before any user interaction. A deny rule is a hard constraint —
// the tool never runs regardless of mode or session decisions. An allow rule
// pre-approves a tool so the user is never prompted for it.
//
// Rules support sub-tool granularity via the `Tool::permission_key()` suffix:
//
//   .allow_tool("fetch")            // allow all calls to the "fetch" tool
//   .deny_tool("bash(rm -rf")       // deny bash calls whose key starts with "rm -rf"
//
// Deny rules take absolute precedence. Allow rules are checked after deny.
// Session decisions layer on top: a user can always-deny a tool that the
// builder allowed (strengthening security), but cannot always-allow a tool
// the builder denied (the hard constraint holds).
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

use wui_core::event::{ControlDecision, ControlRequest, ControlResponse};

// ── Permission Rules ──────────────────────────────────────────────────────────

/// Static allow/deny rules applied before the per-session permission mode.
///
/// Rules use tool names, optionally with a key suffix for sub-tool granularity:
///
/// | Pattern           | Matches                                               |
/// |-------------------|-------------------------------------------------------|
/// | `"fetch"`         | any call to the tool named `fetch`                    |
/// | `"bash(ls"`       | bash calls where `permission_key()` starts with `ls`  |
///
/// **Evaluation order** (first match wins):
/// 1. Deny rules — hard constraint, overrides everything.
/// 2. Allow rules — pre-approved, skips mode and session checks.
/// 3. Falls through to `PermissionMode` + session memory.
#[derive(Debug, Clone, Default)]
pub struct PermissionRules {
    always_allow: Vec<String>,
    always_deny: Vec<String>,
}

impl PermissionRules {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-approve a tool, bypassing the `PermissionMode` check.
    ///
    /// The tool will run without user prompting. Use for tools you fully trust
    /// and never want to interrupt the agent for:
    ///
    /// ```rust,ignore
    /// .allow_tool("read_file")
    /// .allow_tool("bash(ls")   // only bash calls whose key starts with "ls"
    /// ```
    pub fn allow(mut self, pattern: impl Into<String>) -> Self {
        self.always_allow.push(pattern.into());
        self
    }

    /// Hard-deny a tool, regardless of mode and session decisions.
    ///
    /// The tool will never run. Use to enforce absolute constraints:
    ///
    /// ```rust,ignore
    /// .deny_tool("delete_database")
    /// .deny_tool("bash(rm -rf")  // deny dangerous bash patterns specifically
    /// ```
    pub fn deny(mut self, pattern: impl Into<String>) -> Self {
        self.always_deny.push(pattern.into());
        self
    }

    /// Evaluate a tool call against the static rules.
    ///
    /// Returns `Some(false)` → hard denied, `Some(true)` → pre-approved,
    /// `None` → no rule matched (fall through to mode check).
    pub fn evaluate(&self, tool_name: &str, permission_key: Option<&str>) -> Option<bool> {
        if self
            .always_deny
            .iter()
            .any(|r| matches_rule(r, tool_name, permission_key))
        {
            return Some(false);
        }
        if self
            .always_allow
            .iter()
            .any(|r| matches_rule(r, tool_name, permission_key))
        {
            return Some(true);
        }
        None
    }

    pub fn is_empty(&self) -> bool {
        self.always_allow.is_empty() && self.always_deny.is_empty()
    }
}

/// Match a rule pattern against a tool name and optional permission key.
///
/// Pattern `"bash(ls"` matches tool `"bash"` whose permission key starts
/// with `"ls"`. Pattern `"bash"` matches any call to tool `"bash"`.
fn matches_rule(rule: &str, tool_name: &str, permission_key: Option<&str>) -> bool {
    if let Some(open) = rule.find('(') {
        let rule_tool = &rule[..open];
        // A well-formed key rule looks like "tool_name(key_prefix)".
        // strip_suffix removes exactly one closing paren; malformed rules (missing
        // closing paren) are skipped rather than matched incorrectly.
        let Some(rule_key) = rule[open + 1..].strip_suffix(')') else {
            return false;
        };
        rule_tool == tool_name && permission_key.is_some_and(|k| k.starts_with(rule_key))
    } else {
        rule == tool_name
    }
}

// ── Permission Mode ───────────────────────────────────────────────────────────

/// How the engine handles tool permission checks.
#[derive(Clone, Default)]
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

    /// Delegate each tool approval to a callback.
    ///
    /// The callback receives the tool name and raw input JSON.
    /// Return `true` to allow, `false` to deny.
    /// Unlike `Ask`, the callback is synchronous and cannot upgrade to
    /// `approve_always` — use `Ask` with `ControlHandle` for that.
    #[allow(clippy::type_complexity)]
    Callback(Arc<dyn Fn(&str, &serde_json::Value) -> bool + Send + Sync>),
}

impl std::fmt::Debug for PermissionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "Auto"),
            Self::Ask => write!(f, "Ask"),
            Self::Readonly => write!(f, "Readonly"),
            Self::Callback(_) => write!(f, "Callback(...)"),
        }
    }
}

impl PermissionMode {
    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }
    pub fn is_ask(&self) -> bool {
        matches!(self, Self::Ask)
    }
    pub fn is_readonly(&self) -> bool {
        matches!(self, Self::Readonly)
    }
    pub fn is_callback(&self) -> bool {
        matches!(self, Self::Callback(_))
    }
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
    always_deny: HashSet<String>,
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
    Denied {
        reason: String,
    },
    /// The loop must create a ControlHandle, emit it, and await the response.
    NeedsApproval,
}

/// Evaluate a tool call against the current permission mode.
///
/// `tool_is_readonly` is the result of `Tool::is_readonly()` and is relevant
/// only in `Readonly` mode: readonly tools are allowed, all others are denied.
/// `input` is the raw tool input JSON, used by the `Callback` mode.
pub fn check(
    mode: &PermissionMode,
    request: &ControlRequest,
    tool_is_readonly: bool,
    input: &serde_json::Value,
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

        PermissionMode::Callback(f) => {
            let tool_name = tool_name_from_request(request);
            if f(tool_name, input) {
                PermissionOutcome::Allowed
            } else {
                PermissionOutcome::Denied {
                    reason: "denied by approval callback".to_string(),
                }
            }
        }
    }
}

fn tool_name_from_request(request: &ControlRequest) -> &str {
    request.tool_name().unwrap_or("(unknown)")
}

/// Convert a human's `ControlResponse` into a message the LLM will see.
///
/// The message is wrapped in `<system-reminder>` so the LLM can clearly
/// distinguish this framework-injected decision from user conversation.
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
