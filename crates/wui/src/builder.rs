// ============================================================================
// AgentBuilder — the fluent configuration API.
//
// The builder pattern here is not ceremonial: each method is a deliberate
// choice the caller makes about the agent's capabilities and behaviour.
// The absence of a method means the default applies. Defaults are chosen
// to be safe (PermissionMode::Ask) and minimal (no tools, no hooks).
// ============================================================================

use std::sync::Arc;

use crate::compress::{CompactionStrategy, CompressPipeline};
use crate::runtime::{CheckpointStore, PermissionMode, PermissionRules, RetryPolicy, SessionStore};
use wui_core::hook::Hook;
use wui_core::provider::Provider;
use wui_core::tool::Tool;

use crate::session::SessionHooks;
use crate::sub_agent::SubAgent;
use crate::Agent;

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
    pub provider: Arc<dyn Provider>,
    pub tools: Vec<Arc<dyn Tool>>,
    pub hooks: Vec<Arc<dyn Hook>>,
    pub session_store: Option<Arc<dyn SessionStore>>,
    pub compress: Arc<dyn CompactionStrategy>,
    pub permission: PermissionMode,
    /// Static allow/deny rules applied before the permission mode check.
    pub rules: PermissionRules,
    pub system: String,
    pub model: Option<String>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub max_iter: u32,
    pub retry: RetryPolicy,
    pub tool_timeout: Option<std::time::Duration>,
    pub ignore_diminishing_returns: bool,
    /// Hard ceiling on cumulative tokens (input + output). `None` = no limit.
    pub token_budget: Option<u64>,
    /// Extended thinking budget (tokens) sent on every LLM call.
    ///
    /// Set via `.effort(Effort::High)` or directly. `None` = no thinking.
    pub thinking_budget: Option<u32>,
    /// Tools with deferred schemas — listed by name only in the initial prompt.
    /// The LLM calls `ToolSearch` to load their full schema before use.
    pub deferred_tools: Vec<Arc<dyn Tool>>,
    /// Lazily-loaded, searchable tool catalogs.
    pub catalogs: Vec<Arc<dyn crate::catalog::ToolCatalog>>,
    /// Session-level lifecycle hooks.
    pub session_hooks: Option<Arc<SessionHooks>>,
    /// Checkpoint store for save/resume support. `None` disables checkpointing.
    pub checkpoint_store: Option<Arc<dyn CheckpointStore>>,
    /// The run ID under which checkpoints are saved and loaded.
    /// Required when `checkpoint_store` is `Some`.
    pub checkpoint_run_id: Option<String>,
}

/// Fluent builder for `Agent`.
#[must_use = "AgentBuilder does nothing until you call .build()"]
pub struct AgentBuilder {
    config: AgentConfig,
}

impl AgentBuilder {
    pub(crate) fn new(provider: Arc<dyn Provider>) -> Self {
        Self {
            config: AgentConfig {
                provider,
                tools: Vec::new(),
                hooks: Vec::new(),
                session_store: None,
                compress: Arc::new(CompressPipeline::default()),
                permission: PermissionMode::Ask,
                rules: PermissionRules::default(),
                system: String::new(),
                model: None,
                max_tokens: 8192,
                temperature: None,
                max_iter: 20,
                retry: RetryPolicy::default(),
                tool_timeout: None,
                ignore_diminishing_returns: false,
                token_budget: None,
                thinking_budget: None,
                deferred_tools: Vec::new(),
                catalogs: Vec::new(),
                session_hooks: None,
                checkpoint_store: None,
                checkpoint_run_id: None,
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
    /// system prompt. The LLM calls `ToolSearch` to retrieve the full schema
    /// before using them. This saves tokens for large tool libraries where
    /// listing every schema upfront would bloat the context.
    ///
    /// `ToolSearch` is injected automatically when any deferred tools or
    /// catalogs are registered.
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

    /// Set the system prompt.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.config.system = system.into();
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
    ///     .spawn_agent("research", "Search the web and summarise findings.", researcher)
    ///     .build();
    /// ```
    pub fn spawn_agent(
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
    ///     .compaction(MyStrategy::new())
    ///     .build()
    /// ```
    pub fn compaction(mut self, strategy: impl CompactionStrategy + 'static) -> Self {
        self.config.compress = Arc::new(strategy);
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

    /// Disable the diminishing-returns auto-stop heuristic.
    ///
    /// By default, the engine stops a run after several consecutive turns with
    /// negligible output (< 500 tokens), on the assumption that the agent is
    /// stuck. Call this when long tasks are expected to produce many short
    /// intermediate steps before a large final result (research, file scans).
    pub fn ignore_diminishing_returns(mut self) -> Self {
        self.config.ignore_diminishing_returns = true;
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

    /// Finalise the builder and return a ready-to-run `Agent`.
    pub fn build(self) -> Agent {
        Agent {
            config: Arc::new(self.config),
        }
    }
}
