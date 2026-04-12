// ============================================================================
// AgentBuilder — the fluent configuration API.
//
// The builder pattern here is not ceremonial: each method is a deliberate
// choice the caller makes about the agent's capabilities and behaviour.
// The absence of a method means the default applies. Defaults are chosen
// to be safe (PermissionMode::Ask) and minimal (no tools, no hooks).
// ============================================================================

use std::sync::Arc;

use crate::compress::{CompressPipeline, CompressStrategy};
use crate::runtime::{
    CheckpointStore, PermissionMode, PermissionRules, ResultStore, RetryPolicy, SessionStore,
};
use wui_core::hook::Hook;
use wui_core::provider::Provider;
use wui_core::tool::Tool;

use super::agent::Agent;
use super::session::SessionHooks;
use super::sub_agent::SubAgent;

/// Predicate that decides whether a tool should be sent to the provider.
pub(crate) type ToolFilterFn = Arc<dyn Fn(&str, &wui_core::tool::ToolMeta) -> bool + Send + Sync>;

/// Callback invoked when context pressure is critically full.
pub(crate) type ContextOverflowFn = Arc<dyn Fn(&mut Vec<wui_core::message::Message>) + Send + Sync>;

// ── Effort ────────────────────────────────────────────────────────────────────

/// Reasoning effort level — controls the extended thinking budget sent to the
/// provider on every LLM call.
///
/// Maps to Anthropic's `thinking.budget_tokens` parameter; other providers may
/// interpret it differently or ignore it. Effort is a convenience wrapper —
/// you can also set `thinking_budget` on `ChatRequest` directly for fine-grained
/// control.
///
/// ```rust,ignore
/// let agent = Agent::builder(provider)
///     .effort(Effort::High)
///     .build();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effort {
    /// No extended thinking. Fastest and cheapest — good for simple tasks.
    Low,
    /// 4 096-token thinking budget. Good balance for most tasks.
    Medium,
    /// 16 384-token thinking budget. For complex multi-step reasoning.
    High,
    /// 32 768-token thinking budget. Maximum reasoning depth.
    Ultra,
}

impl Effort {
    /// Convert the effort level to a thinking budget in tokens.
    ///
    /// `Low` maps to `None` (no thinking). All other levels enable thinking
    /// with the corresponding token budget.
    pub fn thinking_budget_tokens(self) -> Option<u32> {
        match self {
            Effort::Low => None,
            Effort::Medium => Some(4_096),
            Effort::High => Some(16_384),
            Effort::Ultra => Some(32_768),
        }
    }
}

/// All configuration needed to build an `Agent`.
#[derive(Clone)]
pub struct AgentConfig {
    pub(crate) provider: Arc<dyn Provider>,
    pub(crate) tools: Vec<Arc<dyn Tool>>,
    pub(crate) hooks: Vec<Arc<dyn Hook>>,
    pub(crate) session_store: Option<Arc<dyn SessionStore>>,
    pub(crate) compress: Arc<dyn CompressStrategy>,
    pub(crate) permission: PermissionMode,
    /// Static allow/deny rules applied before the permission mode check.
    pub(crate) rules: PermissionRules,
    pub(crate) system: String,
    /// Stable system prompt sections (cached by the provider across turns).
    /// These are concatenated and placed before the cache boundary.
    pub(crate) system_stable: Vec<String>,
    /// Dynamic system prompt sections (change per turn, placed after cache boundary).
    pub(crate) system_dynamic: Vec<String>,
    pub(crate) model: Option<String>,
    pub(crate) max_tokens: u32,
    pub(crate) temperature: Option<f32>,
    pub(crate) max_iter: u32,
    pub(crate) retry: RetryPolicy,
    pub(crate) tool_timeout: Option<std::time::Duration>,
    pub(crate) expect_long_task: bool,
    /// Hard ceiling on cumulative tokens (input + output). `None` = no limit.
    pub(crate) token_budget: Option<u64>,
    /// Extended thinking budget (tokens) sent on every LLM call.
    ///
    /// Set via `.effort(Effort::High)` or directly. `None` = no thinking.
    pub(crate) thinking_budget: Option<u32>,
    /// Tools with deferred schemas — listed by name only in the initial prompt.
    /// The LLM calls the built-in `tool_search` tool to load their full schema
    /// before use.
    pub(crate) deferred_tools: Vec<Arc<dyn Tool>>,
    /// Lazily-loaded, searchable tool catalogs.
    pub(crate) catalogs: Vec<Arc<dyn crate::catalog::ToolCatalog>>,
    /// Session-level lifecycle hooks.
    pub(crate) session_hooks: Option<Arc<SessionHooks>>,
    /// Checkpoint store for save/resume support. `None` disables checkpointing.
    pub(crate) checkpoint_store: Option<Arc<dyn CheckpointStore>>,
    /// The run ID under which checkpoints are saved and loaded.
    /// Required when `checkpoint_store` is `Some`.
    pub(crate) checkpoint_run_id: Option<String>,
    /// Maximum number of results returned by `tool_search` from catalog searches.
    /// Default: 5.
    pub(crate) catalog_limit: usize,
    /// Optional store for persisting large tool results before truncation.
    pub(crate) result_store: Option<Arc<dyn ResultStore>>,
    /// Maximum sub-agent nesting depth. Default: 5.
    pub(crate) max_spawn_depth: u32,
    /// Optional predicate that filters which tools are sent to the provider.
    pub(crate) tool_filter: Option<ToolFilterFn>,
    /// Requested response format for structured output.
    ///
    /// Set automatically by `run_typed()` when the provider supports it.
    /// `None` means natural language (default).
    pub(crate) response_format: Option<wui_core::provider::ResponseFormat>,
    /// Maximum number of tools executing concurrently. `None` = unlimited.
    pub(crate) max_concurrent_tools: Option<usize>,
    /// Callback invoked when context pressure is critically full after all
    /// compression tiers have been exhausted. Receives the message history
    /// as a mutable reference — the callback can drop messages, truncate
    /// tool results, or apply any other degradation strategy.
    ///
    /// If the callback relieves enough pressure, the run continues.
    /// If pressure remains critical after the callback, the run stops
    /// with `RunStopReason::ContextOverflow`.
    pub(crate) on_context_overflow: Option<ContextOverflowFn>,
}

/// Fluent builder for `Agent`.
#[must_use = "AgentBuilder does nothing until you call .build()"]
pub struct AgentBuilder {
    config: AgentConfig,
    /// `true` when the user explicitly called `.compress()`.
    compress_explicitly_set: bool,
}

impl AgentBuilder {
    pub(crate) fn new(provider: Arc<dyn Provider>) -> Self {
        Self {
            compress_explicitly_set: false,
            config: AgentConfig {
                provider,
                tools: Vec::new(),
                hooks: Vec::new(),
                session_store: None,
                compress: Arc::new(CompressPipeline::default()),
                permission: PermissionMode::Ask,
                rules: PermissionRules::default(),
                system: String::new(),
                system_stable: Vec::new(),
                system_dynamic: Vec::new(),
                model: None,
                max_tokens: 8192,
                temperature: None,
                max_iter: 20,
                retry: RetryPolicy::default(),
                tool_timeout: None,
                expect_long_task: false,
                token_budget: None,
                thinking_budget: None,
                deferred_tools: Vec::new(),
                catalogs: Vec::new(),
                session_hooks: None,
                checkpoint_store: None,
                checkpoint_run_id: None,
                catalog_limit: 5,
                result_store: None,
                max_spawn_depth: 5,
                tool_filter: None,
                response_format: None,
                max_concurrent_tools: None,
                on_context_overflow: None,
            },
        }
    }

    /// Add a tool to the agent's toolkit.
    pub fn tool(mut self, tool: impl Tool) -> Self {
        self.config.tools.push(Arc::new(tool));
        self
    }

    /// Add a pre-built `Arc<dyn Tool>` (useful for sharing tools across agents).
    pub fn tool_arc(mut self, tool: Arc<dyn Tool>) -> Self {
        self.config.tools.push(tool);
        self
    }

    /// Add a tool with a deferred schema.
    ///
    /// Deferred tools appear only as `name + description` in the initial
    /// system prompt. The LLM calls the built-in `tool_search` tool to
    /// retrieve the full schema before using them. This saves tokens for
    /// large tool libraries where listing every schema upfront would bloat
    /// the context.
    ///
    /// The backing `ToolSearch` implementation is injected automatically when
    /// any deferred tools or catalogs are registered.
    pub fn tool_deferred(mut self, tool: impl Tool) -> Self {
        self.config.deferred_tools.push(Arc::new(tool));
        self
    }

    /// Add multiple tools at once.
    ///
    /// Designed for bulk registration, especially from external sources:
    ///
    /// ```rust,ignore
    /// let tools = McpClient::stdio("uvx", ["mcp-server-filesystem", "/tmp"])
    ///     .await?
    ///     .into_tools()
    ///     .await?;
    ///
    /// let agent = Agent::builder(provider)
    ///     .tools(tools)
    ///     .build();
    /// ```
    pub fn tools(mut self, tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> Self {
        self.config.tools.extend(tools);
        self
    }

    /// Add a hook to the agent's conscience.
    pub fn hook(mut self, hook: impl Hook) -> Self {
        self.config.hooks.push(Arc::new(hook));
        self
    }

    /// Attach a session store for turn-level session persistence.
    pub fn session_store(mut self, store: impl SessionStore) -> Self {
        self.config.session_store = Some(Arc::new(store));
        self
    }

    /// Set the system prompt (placed in the stable/cached section).
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.config.system = system.into();
        self
    }

    /// **Advanced.** Append a section to the stable (cached) portion of the system prompt.
    ///
    /// Stable sections should not change between turns. When a provider
    /// supports prompt caching, everything in the stable portion is placed
    /// before the cache boundary so it can be reused across requests.
    ///
    /// Most agents only need `.system(...)`. Use this method when you have
    /// multiple prompt sections and want fine-grained control over which ones
    /// are eligible for caching.
    pub fn system_stable(mut self, section: impl Into<String>) -> Self {
        self.config.system_stable.push(section.into());
        self
    }

    /// **Advanced.** Append a section to the dynamic portion of the system prompt.
    ///
    /// Dynamic sections are placed after the cache boundary and may change
    /// every turn. Content here does not invalidate the cache for the stable
    /// prefix.
    ///
    /// Most agents only need `.system(...)`. Use this method when injecting
    /// per-turn context (e.g. current date, user-specific state) that should
    /// not interfere with cached stable content.
    pub fn system_dynamic(mut self, section: impl Into<String>) -> Self {
        self.config.system_dynamic.push(section.into());
        self
    }

    /// Set the model name for the provider request.
    ///
    /// When omitted, the provider chooses its own default model.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.config.model = Some(model.into());
        self
    }

    /// Set the max tokens per response. Defaults to 8192.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.config.max_tokens = max_tokens;
        self
    }

    /// Set the sampling temperature.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    /// Set the maximum number of tool-use iterations. Defaults to 20.
    pub fn max_iter(mut self, max_iter: u32) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the permission mode. Defaults to `PermissionMode::Ask`.
    pub fn permission(mut self, mode: PermissionMode) -> Self {
        self.config.permission = mode;
        self
    }

    /// Pre-approve a tool, bypassing the permission mode check.
    ///
    /// The tool will run without user prompting. Supports sub-tool patterns
    /// via the `Tool::permission_key()` suffix (see `PermissionRules`):
    ///
    /// ```rust,ignore
    /// .allow_tool("fetch")        // allow all calls to the "fetch" tool
    /// .allow_tool("bash(ls")      // allow bash calls whose key starts with "ls"
    /// ```
    pub fn allow_tool(mut self, pattern: impl Into<String>) -> Self {
        self.config.rules = self.config.rules.allow(pattern);
        self
    }

    /// Hard-deny a tool regardless of permission mode and session decisions.
    ///
    /// The tool will never run. Supports sub-tool patterns:
    ///
    /// ```rust,ignore
    /// .deny_tool("delete_db")          // never allow this tool
    /// .deny_tool("bash(rm -rf")        // deny dangerous bash patterns
    /// ```
    pub fn deny_tool(mut self, pattern: impl Into<String>) -> Self {
        self.config.rules = self.config.rules.deny(pattern);
        self
    }

    /// Register an `Agent` as a synchronous sub-agent tool.
    ///
    /// The supervisor's LLM can call the sub-agent by `name` to handle a
    /// task. The sub-agent runs its own full reasoning loop and returns its
    /// final text response as the tool output (blocking until complete).
    ///
    /// For non-blocking background spawning across turns, use `wui-spawn`.
    ///
    /// ```rust,ignore
    /// let researcher = Agent::builder(provider.clone())
    ///     .tool(WebSearch)
    ///     .permission(PermissionMode::Auto)
    ///     .build();
    ///
    /// let supervisor = Agent::builder(provider)
    ///     .sub_agent("research", "Search the web and summarise findings.", researcher)
    ///     .build();
    /// ```
    pub fn sub_agent(
        self,
        name: impl Into<String>,
        description: impl Into<String>,
        agent: Agent,
    ) -> Self {
        self.tool(SubAgent::new(name, description, agent))
    }

    /// Replace the default compression strategy with a custom implementation.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// Agent::builder(provider)
    ///     .compress(MyStrategy::new())
    ///     .build()
    /// ```
    pub fn compress(mut self, strategy: impl CompressStrategy + 'static) -> Self {
        self.config.compress = Arc::new(strategy);
        self.compress_explicitly_set = true;
        self
    }

    /// Add a `ToolCatalog` — a lazily-loaded, searchable tool source.
    ///
    /// Catalog tools are NOT listed in the initial prompt. The LLM discovers
    /// them via `tool_search`. Catalogs connect lazily on first search.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// .catalog(McpCatalog::new("npx", &["-y", "@mcp/filesystem"]).namespace("fs"))
    /// ```
    pub fn catalog(mut self, catalog: impl crate::catalog::ToolCatalog + 'static) -> Self {
        self.config
            .catalogs
            .push(Arc::new(catalog) as Arc<dyn crate::catalog::ToolCatalog>);
        self
    }

    /// Set the maximum number of results returned per catalog search.
    ///
    /// Applies to the built-in `tool_search` tool when catalogs are
    /// registered via `.catalog()`.
    /// Default: 5.
    pub fn catalog_limit(mut self, n: usize) -> Self {
        self.config.catalog_limit = n;
        self
    }

    /// Session-level lifecycle hooks.
    pub fn session_hooks(mut self, hooks: SessionHooks) -> Self {
        self.config.session_hooks = Some(Arc::new(hooks));
        self
    }

    /// Enable checkpoint / resume for this agent.
    ///
    /// The run loop saves a checkpoint at the end of each tool-use iteration.
    /// On the next call with the same `run_id`, it restores the saved messages,
    /// iteration counter, and token usage before continuing.
    ///
    /// ```rust,ignore
    /// let store = InMemoryCheckpointStore::new();
    /// let agent = Agent::builder(provider)
    ///     .checkpoint(store, "my-run-123")
    ///     .build();
    /// ```
    pub fn checkpoint(mut self, store: impl CheckpointStore, run_id: impl Into<String>) -> Self {
        self.config.checkpoint_store = Some(Arc::new(store));
        self.config.checkpoint_run_id = Some(run_id.into());
        self
    }

    /// Simple tool approval callback for hosts that don't need the full HITL stream.
    ///
    /// Called before each tool invocation. Return `true` to allow, `false` to deny.
    /// For full control (approve_always, deny_always, etc.) use
    /// `PermissionMode::Ask` with `AgentEvent::Control`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// .on_tool_approval(|tool_name, _input| {
    ///     !["bash", "write_file"].contains(&tool_name)
    /// })
    /// ```
    pub fn on_tool_approval(
        mut self,
        f: impl Fn(&str, &serde_json::Value) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.config.permission = PermissionMode::Callback(Arc::new(f));
        self
    }

    /// Override the retry policy for transient provider errors.
    ///
    /// The default policy retries up to 3 times with exponential back-off
    /// starting at 500 ms, capped at 10 s. Pass `RetryPolicy { max_attempts: 0, .. }`
    /// to disable retrying entirely.
    pub fn retry(mut self, policy: RetryPolicy) -> Self {
        self.config.retry = policy;
        self
    }

    /// Set a default timeout for tool execution.
    ///
    /// Applied to every tool that doesn't declare its own `Tool::timeout()`.
    /// When a tool exceeds this duration, the executor returns a
    /// `"tool timed out"` error result and the LLM is informed.
    pub fn tool_timeout(mut self, duration: std::time::Duration) -> Self {
        self.config.tool_timeout = Some(duration);
        self
    }

    /// Disable the stall-detection heuristic.
    ///
    /// By default, the engine stops a run after several consecutive turns with
    /// negligible output (< 500 tokens), on the assumption that the agent is
    /// stuck. Call this when long tasks are expected to produce many short
    /// intermediate steps before a large final result (research, file scans,
    /// multi-file code generation).
    pub fn expect_long_task(mut self) -> Self {
        self.config.expect_long_task = true;
        self
    }

    /// Set a hard ceiling on cumulative tokens (input + output) for the run.
    ///
    /// When the total crosses this budget, the run stops with
    /// `RunStopReason::BudgetExhausted`. More predictable than `max_iter`
    /// for cost control: you know exactly how many tokens you'll spend.
    ///
    /// ```rust,ignore
    /// .token_budget(50_000)  // stop after 50k tokens
    /// ```
    pub fn token_budget(mut self, tokens: u64) -> Self {
        self.config.token_budget = Some(tokens);
        self
    }

    /// Set the reasoning effort level.
    ///
    /// Translates to an extended thinking budget sent to the provider on every
    /// LLM call. `Effort::Low` disables thinking; higher levels enable it with
    /// increasing token budgets.
    ///
    /// ```rust,ignore
    /// .effort(Effort::High)   // 16 384 thinking tokens per call
    /// .effort(Effort::Ultra)  // 32 768 thinking tokens per call
    /// ```
    pub fn effort(mut self, effort: Effort) -> Self {
        self.config.thinking_budget = effort.thinking_budget_tokens();
        self
    }

    /// Attach a store for persisting large tool results.
    ///
    /// When a tool result exceeds `max_output_chars`, the executor saves
    /// the full content via this store and gives the LLM a preview + reference.
    /// Without a store, large results are simply truncated.
    pub fn result_store(mut self, store: impl ResultStore) -> Self {
        self.config.result_store = Some(Arc::new(store));
        self
    }

    /// Set the maximum sub-agent nesting depth. Default: 5.
    pub fn max_spawn_depth(mut self, n: u32) -> Self {
        self.config.max_spawn_depth = n;
        self
    }

    /// Set a filter predicate for tools sent to the provider.
    pub fn tool_filter(
        mut self,
        f: impl Fn(&str, &wui_core::tool::ToolMeta) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.config.tool_filter = Some(Arc::new(f));
        self
    }

    /// Limit the number of tools executing concurrently.
    ///
    /// When set, tools beyond this limit wait for a slot before starting.
    /// Use this when tools call rate-limited external APIs.
    /// `None` (default) = unlimited concurrency.
    pub fn max_concurrent_tools(mut self, n: usize) -> Self {
        self.config.max_concurrent_tools = Some(n);
        self
    }

    /// Set a callback for when the context window overflows.
    ///
    /// Called after all compression tiers (L1 trim → L2 collapse → L3
    /// summarize) have been exhausted and the context is still critically
    /// full. The callback receives the message history as `&mut Vec<Message>`
    /// and can apply a custom degradation strategy — for example, dropping
    /// old tool results, truncating long messages, or removing all but the
    /// most recent N messages.
    ///
    /// If the callback relieves enough pressure, the run continues
    /// automatically. If pressure remains critical, the run stops with
    /// `RunStopReason::ContextOverflow`.
    ///
    /// ```rust,ignore
    /// Agent::builder(provider)
    ///     .on_context_overflow(|messages| {
    ///         // Keep only the last 4 messages
    ///         if messages.len() > 4 {
    ///             messages.drain(..messages.len() - 4);
    ///         }
    ///     })
    ///     .build();
    /// ```
    pub fn on_context_overflow(
        mut self,
        f: impl Fn(&mut Vec<wui_core::message::Message>) + Send + Sync + 'static,
    ) -> Self {
        self.config.on_context_overflow = Some(Arc::new(f));
        self
    }

    /// Finalise the builder and return a ready-to-run `Agent`.
    ///
    /// # Panics
    ///
    /// Panics if any tool name is registered more than once. For a fallible
    /// alternative that returns an error instead of panicking, use
    /// [`try_build`](Self::try_build).
    pub fn build(self) -> Agent {
        self.try_build().unwrap_or_else(|e| panic!("{e}"))
    }

    /// Finalise the builder and return a ready-to-run `Agent`, or a
    /// [`BuildError`] if the configuration is invalid.
    ///
    /// Prefer this over [`build`](Self::build) in library code or anywhere
    /// a misconfigured agent should be a recoverable error rather than a crash.
    ///
    /// ```rust,ignore
    /// let agent = Agent::builder(provider)
    ///     .tool(ToolA)
    ///     .tool(ToolB)
    ///     .permission(PermissionMode::Auto)
    ///     .try_build()?;
    /// ```
    pub fn try_build(mut self) -> Result<Agent, BuildError> {
        // Auto-calibrate the compression window from provider capabilities
        // when the user hasn't explicitly set a custom strategy.
        if !self.compress_explicitly_set {
            let caps = self
                .config
                .provider
                .capabilities(self.config.model.as_deref());
            if let Some(window) = caps.max_context_window {
                self.config.compress = Arc::new(CompressPipeline {
                    window_tokens: window,
                    ..CompressPipeline::default()
                });
            }
        }

        // Validate the tool set: duplicate names produce a clear error at
        // construction time instead of surfacing inside a running stream.
        super::agent::build_registry(
            &self.config.tools,
            &self.config.deferred_tools,
            &self.config.catalogs,
            self.config.catalog_limit,
        )
        .map_err(BuildError::DuplicateTool)?;

        // Warn when tools are registered under PermissionMode::Ask.
        // PermissionMode::Ask requires a caller that handles AgentEvent::Control.
        // Agent::run() cannot do this and will return an error; stream() works
        // but only if the caller explicitly handles Control events.
        if matches!(self.config.permission, crate::runtime::PermissionMode::Ask)
            && !self.config.tools.is_empty()
        {
            tracing::warn!(
                "AgentBuilder: tools are registered but PermissionMode::Ask is active. \
                 Agent::run() will error on the first tool call. \
                 Add .permission(PermissionMode::Auto) for headless use, or \
                 handle AgentEvent::Control in your stream loop for interactive use."
            );
        }

        Ok(Agent {
            config: Arc::new(self.config),
        })
    }
}

// ── BuildError ────────────────────────────────────────────────────────────────

/// Error returned by [`AgentBuilder::try_build`] when the agent configuration
/// is invalid.
#[derive(Debug)]
#[non_exhaustive]
pub enum BuildError {
    /// Two or more tools were registered with the same name.
    ///
    /// The inner string is the conflicting tool name.
    DuplicateTool(String),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::DuplicateTool(name) => {
                write!(
                    f,
                    "duplicate tool name '{name}': each tool must have a unique name"
                )
            }
        }
    }
}

impl std::error::Error for BuildError {}
