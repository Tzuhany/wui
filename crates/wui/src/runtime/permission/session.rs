// ── Session Permissions ───────────────────────────────────────────────────────

use std::collections::HashSet;
use std::sync::Arc;

use tokio::sync::RwLock;

use super::rules::matches_rule_with_matcher;

/// Per-session permission memory.
///
/// Tracks which invocation patterns have been always-approved or always-denied
/// by the user. Patterns use the same grammar as static `PermissionRules`
/// (`"tool"` or `"tool(prefix)"`), so session memory preserves the runtime's
/// fine-grained permission model instead of collapsing everything to a bare
/// tool name.
///
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

    /// `true` if this exact stored pattern has been always-approved.
    pub async fn is_always_allowed(&self, pattern: &str) -> bool {
        self.inner.read().await.always_allow.contains(pattern)
    }

    /// `true` if this exact stored pattern has been always-denied.
    pub async fn is_always_denied(&self, pattern: &str) -> bool {
        self.inner.read().await.always_deny.contains(pattern)
    }

    /// `true` if any always-allow pattern matches this invocation.
    pub async fn allows(
        &self,
        tool_name: &str,
        permission_key: Option<&str>,
        matcher: Option<&(dyn Fn(&str) -> bool + Send + Sync)>,
    ) -> bool {
        let inner = self.inner.read().await;
        inner
            .always_allow
            .iter()
            .any(|rule| matches_rule_with_matcher(rule, tool_name, permission_key, matcher))
    }

    /// `true` if any always-deny pattern matches this invocation.
    pub async fn denies(
        &self,
        tool_name: &str,
        permission_key: Option<&str>,
        matcher: Option<&(dyn Fn(&str) -> bool + Send + Sync)>,
    ) -> bool {
        let inner = self.inner.read().await;
        inner
            .always_deny
            .iter()
            .any(|rule| matches_rule_with_matcher(rule, tool_name, permission_key, matcher))
    }

    /// Record an always-allow decision for a pattern.
    pub async fn set_always_allow(&self, pattern: impl Into<String>) {
        let mut inner = self.inner.write().await;
        let pattern = pattern.into();
        inner.always_deny.remove(&pattern);
        inner.always_allow.insert(pattern);
    }

    /// Record an always-deny decision for a pattern.
    pub async fn set_always_deny(&self, pattern: impl Into<String>) {
        let mut inner = self.inner.write().await;
        let pattern = pattern.into();
        inner.always_allow.remove(&pattern);
        inner.always_deny.insert(pattern);
    }

    /// Clear any always-allow or always-deny decision for a tool, returning
    /// it to the default per-invocation approval behaviour.
    ///
    /// No-op if the pattern has no standing decision.
    pub async fn revoke(&self, pattern: &str) {
        let mut inner = self.inner.write().await;
        inner.always_allow.remove(pattern);
        inner.always_deny.remove(pattern);
    }
}
