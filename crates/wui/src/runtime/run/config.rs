use std::sync::Arc;
use std::time::Duration;

use crate::compress::CompressStrategy;
use crate::facade::builder::ToolFilterFn;
use crate::runtime::checkpoint::CheckpointStore;
use crate::runtime::executor::ResultStore;
use crate::runtime::hooks::HookRunner;
use crate::runtime::permission::{PermissionMode, PermissionRules, SessionPermissions};
use crate::runtime::registry::ToolRegistry;

use super::provider::RetryPolicy;

// ── Run Config ────────────────────────────────────────────────────────────────

/// Static configuration for one run. `Arc`-wrapped so it is cheap to share
/// across the spawned task and helper services.
pub(crate) struct RunConfig {
    pub(crate) provider: Arc<dyn wui_core::provider::Provider>,
    pub(crate) tools: Arc<ToolRegistry>,
    pub(crate) hooks: Arc<HookRunner>,
    pub(crate) compress: Arc<dyn CompressStrategy>,
    pub(crate) permission: PermissionMode,
    /// Static allow/deny rules evaluated before the permission mode.
    /// Deny rules are hard constraints; allow rules bypass user prompting.
    pub(crate) rules: PermissionRules,
    pub(crate) session_perms: Arc<SessionPermissions>,
    pub(crate) system: String,
    pub(crate) model: Option<String>,
    pub(crate) max_tokens: u32,
    pub(crate) temperature: Option<f32>,
    pub(crate) max_iter: u32,

    /// Back-off policy for transient provider errors.
    pub(crate) retry: RetryPolicy,

    /// Default execution timeout applied to every tool that doesn't declare
    /// its own `Tool::timeout()`. `None` means tools may run indefinitely.
    pub(crate) tool_timeout: Option<Duration>,

    /// Disable the stall-detection heuristic.
    ///
    /// When `false` (default), the engine stops a run after
    /// `MAX_LOW_OUTPUT_TURNS` consecutive turns with fewer than
    /// `MIN_USEFUL_OUTPUT_TOKENS` output tokens. Set this to `true` for
    /// long-running tasks where many short intermediate steps are expected
    /// before a large final result (e.g. research tasks, file-heavy writes).
    pub(crate) expect_long_task: bool,

    /// Hard ceiling on cumulative tokens (input + output) for this run.
    ///
    /// When the total crosses this budget, the run stops immediately with
    /// `RunStopReason::BudgetExhausted`. More predictable than `max_iter`
    /// for cost control: you know exactly how many tokens you'll spend.
    ///
    /// `None` means no budget limit (default).
    pub(crate) token_budget: Option<u64>,

    /// Extended thinking budget (tokens) forwarded to the provider on every
    /// LLM call. `None` = no thinking (provider default).
    pub(crate) thinking_budget: Option<u32>,

    /// Checkpoint store for save/resume. `None` disables checkpointing.
    pub(crate) checkpoint_store: Option<Arc<dyn CheckpointStore>>,
    /// The run ID used to save/load checkpoints.
    pub(crate) checkpoint_run_id: Option<String>,

    /// Optional store for persisting large tool results before truncation.
    pub(crate) result_store: Option<Arc<dyn ResultStore>>,

    /// Byte index in `system` where the cache boundary falls.
    ///
    /// Everything before this index is the stable prefix; the provider may
    /// use this to optimize caching. `None` means no boundary.
    pub(crate) cache_boundary: Option<usize>,

    /// Current sub-agent nesting depth for this run.
    pub(crate) spawn_depth: u32,

    /// Optional predicate that filters which tools are sent to the provider.
    pub(crate) tool_filter: Option<ToolFilterFn>,

    /// Requested response format for structured output.
    ///
    /// When set, the engine includes this in every `ChatRequest` so that
    /// providers supporting native JSON mode can constrain the model's output.
    /// Providers that do not support structured output ignore it.
    pub(crate) response_format: Option<wui_core::provider::ResponseFormat>,

    /// Maximum number of tools executing concurrently. `None` = unlimited.
    pub(crate) max_concurrent_tools: Option<usize>,

    /// Callback for custom degradation when context overflows.
    pub(crate) on_context_overflow: Option<crate::facade::builder::ContextOverflowFn>,
}
