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
    pub(super) always_allow: Vec<String>,
    pub(super) always_deny: Vec<String>,
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
        self.evaluate_with_matcher(tool_name, permission_key, None)
    }

    /// Evaluate with an optional tool-provided matcher for wildcard rules.
    ///
    /// When `matcher` is `Some`, sub-tool rules (e.g. `"bash(git *)"`) are
    /// tested via the matcher closure instead of simple prefix matching.
    /// This allows tools to implement arbitrarily complex pattern matching
    /// (glob, regex, parsed-command matching) for their permission rules.
    pub fn evaluate_with_matcher(
        &self,
        tool_name: &str,
        permission_key: Option<&str>,
        matcher: Option<&(dyn Fn(&str) -> bool + Send + Sync)>,
    ) -> Option<bool> {
        if self
            .always_deny
            .iter()
            .any(|r| matches_rule_with_matcher(r, tool_name, permission_key, matcher))
        {
            return Some(false);
        }
        if self
            .always_allow
            .iter()
            .any(|r| matches_rule_with_matcher(r, tool_name, permission_key, matcher))
        {
            return Some(true);
        }
        None
    }

    pub fn is_empty(&self) -> bool {
        self.always_allow.is_empty() && self.always_deny.is_empty()
    }
}

/// Encode one invocation into the same pattern grammar used by static rules.
pub(crate) fn invocation_pattern(tool_name: &str, permission_key: Option<&str>) -> String {
    match permission_key {
        Some(key) => format!("{tool_name}({key})"),
        None => tool_name.to_owned(),
    }
}

/// Match a rule with an optional tool-provided matcher.
///
/// When `matcher` is provided, sub-tool patterns are tested against the
/// matcher closure first. If the matcher rejects, falls through to the
/// default prefix matching. This lets tools implement glob, wildcard,
/// or parsed-command matching for their permission rules.
pub(super) fn matches_rule_with_matcher(
    rule: &str,
    tool_name: &str,
    permission_key: Option<&str>,
    matcher: Option<&(dyn Fn(&str) -> bool + Send + Sync)>,
) -> bool {
    let Some(open) = rule.find('(') else {
        return rule == tool_name;
    };
    let rule_tool = &rule[..open];
    let Some(rule_key) = rule[open + 1..].strip_suffix(')') else {
        return false;
    };
    if rule_tool != tool_name {
        return false;
    }
    // Try the tool-provided matcher first (supports wildcards, globs, etc.).
    if let Some(matcher) = matcher {
        if matcher(rule_key) {
            return true;
        }
    }
    // Fall back to prefix matching on permission_key.
    permission_key.is_some_and(|k| k.starts_with(rule_key))
}
