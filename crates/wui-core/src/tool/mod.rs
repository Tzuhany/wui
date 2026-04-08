// ============================================================================
// Tool — the agent's hands.
//
// Tools are stateless, Send + Sync + 'static. The executor spawns them freely
// across tokio tasks without lifetime friction. State lives in ToolCtx.
//
// ── Module layout ────────────────────────────────────────────────────────────
//
// mod.rs     — Tool trait, ToolMeta, ExecutorHints, InterruptBehavior,
//              PermissionMatcher. "What a tool IS."
// output.rs  — ToolOutput, ToolCtx, FailureKind. "What a tool PRODUCES."
// artifact.rs — Artifact, ArtifactKind, ArtifactContent. Discrete outputs.
// input.rs   — ToolInput, ContextInjection. "What a tool RECEIVES."
// ============================================================================

mod artifact;
mod input;
mod output;

pub use artifact::{Artifact, ArtifactContent, ArtifactKind};
pub use input::{ContextInjection, ToolInput};
pub use output::{FailureKind, ToolCtx, ToolOutput};

use async_trait::async_trait;
use serde_json::Value;

/// A permission-matching closure returned by [`Tool::permission_matcher`].
pub type PermissionMatcher = Box<dyn Fn(&str) -> bool + Send + Sync>;

// ── InterruptBehavior ─────────────────────────────────────────────────────────

/// What should happen when the user submits a new message while this tool runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptBehavior {
    /// Stop the tool and discard its result.
    Cancel,
    /// Keep running; the new message waits until this tool completes.
    Block,
}

// ── ToolMeta ──────────────────────────────────────────────────────────────────

/// Per-invocation semantic metadata for a tool call.
///
/// `ToolMeta` contains only cross-runtime properties: semantic flags that any
/// executor built on `wui-core` needs to reason about execution ordering,
/// safety, and permission decisions. Wui-executor-specific tuning knobs
/// (timeout, retries, output limits) live in `ExecutorHints` in `wui`.
///
/// Return from [`Tool::meta`] to communicate these properties to the runtime.
/// All fields have safe defaults — override only what differs.
///
/// # Example
/// ```rust,ignore
/// fn meta(&self, _input: &serde_json::Value) -> ToolMeta {
///     ToolMeta { readonly: true, ..ToolMeta::default() }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ToolMeta {
    /// Allow concurrent execution alongside other tools. Default: `true`.
    pub concurrent: bool,
    /// Tool does not modify external state. Default: `false`.
    pub readonly: bool,
    /// Tool cannot be undone (shown to user in HITL prompt). Default: `false`.
    pub destructive: bool,
    /// Tool requires live user interaction and cannot run headlessly. Default: `false`.
    pub requires_interaction: bool,
    /// Suffix appended to the tool name for fine-grained permission rules.
    /// E.g. `Some("rm -rf /")` for a bash tool.
    pub permission_key: Option<String>,
}

impl Default for ToolMeta {
    fn default() -> Self {
        Self {
            concurrent: true,
            readonly: false,
            destructive: false,
            requires_interaction: false,
            permission_key: None,
        }
    }
}

// ── ExecutorHints ─────────────────────────────────────────────────────────────

/// Wui-executor-specific execution hints returned by [`Tool::executor_hints`].
///
/// These are NOT part of the universal tool vocabulary — they are tuning knobs
/// specific to Wui's executor implementation. Any runtime built on `wui-core`
/// that doesn't use these hints may ignore them entirely.
///
/// Tools return these from `executor_hints()` to customise per-invocation
/// behaviour. The defaults (no timeout, no retries, no output limit, no summary)
/// are safe for any tool.
///
/// # Freeze policy
///
/// `ExecutorHints` is a **closed set**. Fields are added only when a new
/// executor behaviour cannot be expressed by composing the existing ones and
/// has proven necessary in real-world use. Before adding a field, ask:
///
/// 1. Can this be expressed by combining existing fields?
/// 2. Is this Wui-runtime-specific or truly universal (belongs in `ToolMeta`)?
/// 3. Is there at least one concrete use-case that cannot be solved without it?
///
/// Routine convenience is not sufficient justification. Keeping this type small
/// is the mechanism by which `wui-core` stays honest.
#[derive(Debug, Clone, Default)]
pub struct ExecutorHints {
    /// One-line summary for display in tool history. Default: `None`.
    pub summary: Option<String>,
    /// Per-call execution timeout. Default: use the executor's global timeout.
    pub timeout: Option<std::time::Duration>,
    /// Truncate output to this many characters. Default: no limit.
    pub max_output_chars: Option<usize>,
    /// Retry up to this many times on error output. Default: `0`.
    pub max_retries: u32,
}

// ── Tool trait ────────────────────────────────────────────────────────────────

/// The interface every tool implements.
///
/// Four methods are required: `name`, `description`, `input_schema`, and `call`.
/// Override `meta` to communicate per-invocation semantic hints (concurrency,
/// readonly, destructive, requires_interaction) to the runtime.
/// Override `executor_hints` to communicate Wui-executor-specific tuning
/// (timeout, retries, output limits, display summary) without affecting other
/// runtimes.
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    /// Unique identifier. This is how the LLM names the tool in its output.
    fn name(&self) -> &str;

    /// One-line description shown to the LLM in the tool listing.
    fn description(&self) -> &str;

    /// JSON Schema describing the tool's input parameters.
    ///
    /// The executor validates every invocation against this schema before
    /// calling `call()`. Invalid input produces an immediate error result
    /// that the LLM sees and can self-correct from.
    fn input_schema(&self) -> Value;

    /// Semantic metadata for this tool invocation.
    ///
    /// Called once per invocation with the resolved input. Return a customised
    /// [`ToolMeta`] to communicate cross-runtime execution properties (concurrency,
    /// readonly, destructive, requires_interaction, permission_key). Most tools
    /// return `ToolMeta::default()` and override one or two fields.
    fn meta(&self, _input: &Value) -> ToolMeta {
        ToolMeta::default()
    }

    /// Wui-executor-specific execution hints for this tool invocation.
    ///
    /// Called once per invocation with the resolved input. Return a customised
    /// [`ExecutorHints`] to provide timeout, retry, output-limit, or display
    /// summary values to the Wui executor. Other runtimes that don't use Wui's
    /// executor may ignore these hints entirely.
    ///
    /// ```rust,ignore
    /// fn executor_hints(&self, _input: &Value) -> ExecutorHints {
    ///     ExecutorHints {
    ///         timeout:     Some(std::time::Duration::from_secs(30)),
    ///         max_retries: 2,
    ///         ..ExecutorHints::default()
    ///     }
    /// }
    /// ```
    fn executor_hints(&self, _input: &Value) -> ExecutorHints {
        ExecutorHints::default()
    }

    /// What happens when the user interrupts while this tool is running.
    ///
    /// - `Cancel` — abort the tool and discard its result.
    /// - `Block` (default) — keep running; the interruption waits.
    ///
    /// Override for tools whose results are disposable on interruption
    /// (searches, reads). Keep the default for tools with side effects
    /// (file writes, API calls) that should finish cleanly.
    fn interrupt_behavior(&self) -> InterruptBehavior {
        InterruptBehavior::Block
    }

    /// Prepare a matcher for wildcard permission rules.
    ///
    /// Called once per tool invocation during the permission check. Returns a
    /// closure that tests whether a permission rule pattern matches this
    /// specific invocation. When `None` (default), the framework falls back
    /// to prefix matching on [`ToolMeta::permission_key`].
    ///
    /// Use this for tools with complex input structures (like a shell tool)
    /// where the permission pattern needs to match against parsed subcommands:
    ///
    /// ```rust,ignore
    /// fn permission_matcher(&self, input: &Value) -> Option<Box<dyn Fn(&str) -> bool + Send>> {
    ///     let cmd = input["command"].as_str()?.to_string();
    ///     Some(Box::new(move |pattern| {
    ///         cmd == pattern || cmd.starts_with(&format!("{pattern} "))
    ///     }))
    /// }
    /// ```
    fn permission_matcher(&self, _input: &Value) -> Option<PermissionMatcher> {
        None
    }

    /// Execute the tool.
    ///
    /// Called only after input has been validated against `input_schema()`.
    /// `ctx` provides the conversation history, cancellation, and progress.
    async fn call(&self, input: Value, ctx: &ToolCtx) -> ToolOutput;
}
