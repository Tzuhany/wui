// ============================================================================
// Hook — the agent's conscience.
//
// Hooks intercept the agent loop at three points and can block or allow
// each action. When a hook blocks, the reason is injected into the
// conversation as an error message — the LLM sees it and self-corrects.
// This creates a natural feedback loop without requiring any special
// "error handling mode".
//
// Hooks are composable: the engine runs all registered hooks in order and
// stops at the first Block decision. Register multiple hooks freely.
// ============================================================================

use async_trait::async_trait;
use serde_json::Value;

use crate::tool::ToolOutput;

/// A hook that intercepts the agent loop at key moments.
///
/// Hooks are `Send + Sync + 'static` for the same reason tools are:
/// the engine may invoke them from spawned tasks.
#[async_trait]
pub trait Hook: Send + Sync + 'static {
    /// Whether this hook wants to see the given event.
    ///
    /// Returning `false` skips `evaluate()` entirely. Use this to avoid
    /// the overhead of an async call for events you don't care about.
    fn handles(&self, event: &HookEvent<'_>) -> bool;

    /// Evaluate the event and decide whether to allow or block it.
    async fn evaluate(&self, event: &HookEvent<'_>) -> HookDecision;
}

// ── Hook Event ────────────────────────────────────────────────────────────────

/// The three moments where hooks can intervene.
#[derive(Debug)]
pub enum HookEvent<'a> {
    /// The LLM has decided to call a tool. The hook can inspect the name
    /// and arguments and block the call before any execution happens.
    PreToolUse {
        name:  &'a str,
        input: &'a Value,
    },

    /// A tool has returned a result. The hook can inspect the output and
    /// block the loop from continuing (e.g. to enforce output policies).
    PostToolUse {
        name:   &'a str,
        output: &'a ToolOutput,
    },

    /// The LLM has produced a complete response with no further tool calls.
    /// The hook runs before the response is returned to the caller.
    /// Block here to force the LLM to revise its output.
    PreComplete {
        response: &'a str,
    },
}

// ── Hook Decision ─────────────────────────────────────────────────────────────

/// What a hook decides to do with an event.
#[derive(Debug, Clone)]
pub enum HookDecision {
    /// Let the action proceed.
    Allow,

    /// Prevent the action. The reason is injected into the conversation
    /// as a system-level error so the LLM can see it and self-correct.
    Block { reason: String },
}

impl HookDecision {
    pub fn allow() -> Self {
        Self::Allow
    }

    pub fn block(reason: impl Into<String>) -> Self {
        Self::Block { reason: reason.into() }
    }

    pub fn is_blocked(&self) -> bool {
        matches!(self, Self::Block { .. })
    }
}

// ── Built-in Hooks ────────────────────────────────────────────────────────────

/// Blocks any tool whose name is in the deny list.
///
/// ```rust
/// Agent::builder()
///     .hook(DenyList::new(["bash", "file_write"]))
///     .build()
/// ```
pub struct DenyList {
    denied: Vec<String>,
}

impl DenyList {
    pub fn new(tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self { denied: tools.into_iter().map(Into::into).collect() }
    }
}

#[async_trait]
impl Hook for DenyList {
    fn handles(&self, event: &HookEvent<'_>) -> bool {
        matches!(event, HookEvent::PreToolUse { .. })
    }

    async fn evaluate(&self, event: &HookEvent<'_>) -> HookDecision {
        if let HookEvent::PreToolUse { name, .. } = event {
            if self.denied.iter().any(|d| d == name) {
                return HookDecision::block(
                    format!("tool '{name}' is not allowed in this context")
                );
            }
        }
        HookDecision::Allow
    }
}
