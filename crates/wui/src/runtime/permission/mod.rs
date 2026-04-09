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

mod decision;
mod mode;
mod rules;
mod session;

pub use decision::{
    check, response_to_system_message, PermissionCheck, PermissionOutcome, PermissionSource,
    PermissionVerdict,
};
pub use mode::PermissionMode;
pub use rules::PermissionRules;
pub use session::SessionPermissions;
