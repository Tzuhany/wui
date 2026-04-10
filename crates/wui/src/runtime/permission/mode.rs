// ── Permission Mode ───────────────────────────────────────────────────────────

use std::sync::Arc;

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
    ///
    /// The callback is synchronous: it runs inline, does not suspend the event
    /// loop, and emits no `AgentEvent::Control`. There is no session memory —
    /// every tool call re-evaluates the callback from scratch.
    ///
    /// Use `Callback` for automated servers or embedded hosts that apply a
    /// fixed allow/deny policy without user interaction.
    ///
    /// Use `Ask` instead when you need any of:
    /// - `approve_always` / `deny_always` (session-scoped memory)
    /// - async decision logic (e.g. database or network lookup)
    /// - a running event consumer that handles `AgentEvent::Control`
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
