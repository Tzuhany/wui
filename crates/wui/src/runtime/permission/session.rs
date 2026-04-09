// ── Session Permissions ───────────────────────────────────────────────────────

use std::collections::HashSet;
use std::sync::Arc;

use tokio::sync::RwLock;

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

    /// Clear any always-allow or always-deny decision for a tool, returning
    /// it to the default per-invocation approval behaviour.
    ///
    /// No-op if the tool has no standing decision.
    pub async fn revoke(&self, tool: &str) {
        let mut inner = self.inner.write().await;
        inner.always_allow.remove(tool);
        inner.always_deny.remove(tool);
    }
}
