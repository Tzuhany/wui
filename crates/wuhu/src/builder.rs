// ============================================================================
// AgentBuilder — the fluent configuration API.
//
// The builder pattern here is not ceremonial: each method is a deliberate
// choice the caller makes about the agent's capabilities and behaviour.
// The absence of a method means the default applies. Defaults are chosen
// to be safe (PermissionMode::Ask) and minimal (no tools, no hooks).
// ============================================================================

use std::collections::HashMap;
use std::sync::Arc;

use wuhu_core::checkpoint::Checkpoint;
use wuhu_core::hook::Hook;
use wuhu_core::provider::Provider;
use wuhu_core::tool::{SpawnFn, Tool};
use wuhu_compress::CompressPipeline;
use wuhu_engine::{PermissionMode, QueryChain, RetryPolicy};

use crate::Agent;

/// All configuration needed to build an `Agent`.
#[derive(Clone)]
pub struct AgentConfig {
    pub provider:    Arc<dyn Provider>,
    pub tools:       Vec<Arc<dyn Tool>>,
    pub hooks:       Vec<Arc<dyn Hook>>,
    pub checkpoint:  Option<Arc<dyn Checkpoint>>,
    pub compress:    CompressPipeline,
    pub permission:  PermissionMode,
    pub system:      String,
    pub model:       String,
    pub max_tokens:  u32,
    pub temperature: Option<f32>,
    pub max_iter:    u32,
    pub extensions:         HashMap<String, serde_json::Value>,
    pub initial_extensions: HashMap<String, serde_json::Value>,
    pub spawn:              Option<SpawnFn>,
    pub query_chain:        Option<QueryChain>,
    pub retry:                      RetryPolicy,
    pub tool_timeout:               Option<std::time::Duration>,
    pub ignore_diminishing_returns: bool,
}

/// Fluent builder for `Agent`.
pub struct AgentBuilder {
    config: AgentConfig,
}

impl AgentBuilder {
    pub(crate) fn new(provider: Arc<dyn Provider>) -> Self {
        Self {
            config: AgentConfig {
                provider,
                tools:       Vec::new(),
                hooks:       Vec::new(),
                checkpoint:  None,
                compress:    CompressPipeline::default(),
                permission:  PermissionMode::Ask,
                system:      String::new(),
                model:       "claude-opus-4-6".to_string(),
                max_tokens:  8192,
                temperature: None,
                max_iter:    20,
                extensions:         HashMap::new(),
                initial_extensions: HashMap::new(),
                spawn:              None,
                query_chain:        None,
                retry:                      RetryPolicy::default(),
                tool_timeout:               None,
                ignore_diminishing_returns: false,
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

    /// Add a hook to the agent's conscience.
    pub fn hook(mut self, hook: impl Hook) -> Self {
        self.config.hooks.push(Arc::new(hook));
        self
    }

    /// Attach a checkpoint backend for session persistence.
    pub fn checkpoint(mut self, cp: impl Checkpoint) -> Self {
        self.config.checkpoint = Some(Arc::new(cp));
        self
    }

    /// Attach a sub-agent spawn capability.
    ///
    /// Tools can call `ctx.spawn_agent(prompt)` to delegate work to this
    /// sub-agent. Use `another_agent.as_spawn_fn()` to create the value:
    ///
    /// ```rust,ignore
    /// let researcher = Agent::builder(provider.clone())
    ///     .system("You are a research assistant.")
    ///     .tool(WebSearch)
    ///     .build();
    ///
    /// let orchestrator = Agent::builder(provider)
    ///     .spawn(researcher.as_spawn_fn())
    ///     .build();
    /// ```
    pub fn spawn(mut self, spawn_fn: SpawnFn) -> Self {
        self.config.spawn = Some(spawn_fn);
        self
    }

    /// Set the system prompt.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.config.system = system.into();
        self
    }

    /// Set the model name. Defaults to `claude-opus-4-6`.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.config.model = model.into();
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

    /// Configure the context compression pipeline.
    pub fn compress(mut self, compress: CompressPipeline) -> Self {
        self.config.compress = compress;
        self
    }

    /// Add a provider-specific extension to every request.
    ///
    /// Example — Anthropic extended thinking:
    /// ```rust,ignore
    /// .extension("thinking", json!({"type": "enabled", "budget_tokens": 8000}))
    /// ```
    pub fn extension(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.config.extensions.insert(key.into(), value.into());
        self
    }

    /// Add a provider-specific extension applied **only on the first LLM call**.
    ///
    /// Use for parameters that should steer the opening response but must not
    /// persist into continuation turns. The canonical example is `tool_choice`:
    /// forcing the first turn to call a tool while allowing subsequent turns to
    /// respond freely. Without this, `tool_choice=any` would apply to every
    /// iteration and loop until `max_iter`.
    ///
    /// ```rust,ignore
    /// .initial_extension("tool_choice", json!({"type": "any"}))
    /// ```
    pub fn initial_extension(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.config.initial_extensions.insert(key.into(), value.into());
        self
    }

    /// Set the sub-agent chain for depth tracking.
    ///
    /// Pass `parent_chain.child()?` when building a sub-agent to inherit
    /// the chain ID and enforce the depth ceiling.
    pub fn query_chain(mut self, chain: QueryChain) -> Self {
        self.config.query_chain = Some(chain);
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

    /// Finalise the builder and return a ready-to-run `Agent`.
    pub fn build(self) -> Agent {
        Agent { config: Arc::new(self.config) }
    }
}
