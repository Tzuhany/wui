// ============================================================================
// Agent — the top-level interface.
//
// Three entry points, one for each level of commitment:
//
//   run()       — one-shot, returns final text
//   stream()    — single turn, returns event stream
//   session()   — multi-turn, owns message history
//
// All three are thin wrappers over the same engine. There is no magic here —
// just convenience shaped to the most common use cases.
// ============================================================================

use std::sync::Arc;

use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use wuhu_core::event::AgentEvent;
use wuhu_core::message::Message;
use wuhu_core::provider::Provider;
use wuhu_core::tool::SpawnFn;
use wuhu_engine::{HookRunner, RunConfig, SessionPermissions, ToolRegistry, ToolSearch, run};

use crate::builder::{AgentBuilder, AgentConfig};
use crate::session::Session;

/// A configured agent ready to run.
///
/// Create one via `Agent::builder()`, then call `run()`, `stream()`, or
/// `session()`. Cheap to clone — all configuration is behind an Arc.
#[derive(Clone)]
pub struct Agent {
    pub(crate) config: Arc<AgentConfig>,
}

impl Agent {
    /// Start building an agent.
    ///
    /// ```rust,ignore
    /// let agent = Agent::builder()
    ///     .provider(Anthropic::new(api_key))
    ///     .system("You are a helpful assistant.")
    ///     .build();
    /// ```
    pub fn builder(provider: impl Provider) -> AgentBuilder {
        AgentBuilder::new(Arc::new(provider))
    }

    // ── One-shot ──────────────────────────────────────────────────────────────

    /// Run a single prompt and return the final text response.
    ///
    /// This is a fire-and-forget convenience wrapper over `stream()`. It
    /// collects the full text response and discards all other events.
    ///
    /// **Permission handling:** if a tool requires human approval
    /// (`PermissionMode::Ask`) and no one is available to respond, `run()`
    /// automatically approves it — the caller has no way to respond to
    /// `AgentEvent::Control` in a one-shot call. For interactive permission
    /// prompts, use `stream()` or `session()` instead.
    ///
    /// For tool visibility, streaming output, or HITL, use `stream()` directly.
    pub async fn run(&self, prompt: impl Into<String>) -> Result<String, wuhu_core::event::AgentError> {
        let mut text = String::new();
        let mut stream = self.stream(prompt).await;

        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::TextDelta(t)    => text.push_str(&t),
                // Auto-approve in one-shot context — caller cannot respond.
                AgentEvent::Control(handle) => handle.approve(),
                AgentEvent::Done(_)         => break,
                AgentEvent::Error(e)        => return Err(e),
                _                           => {}
            }
        }

        Ok(text)
    }

    // ── Single-turn stream ────────────────────────────────────────────────────

    /// Run a single prompt and return a stream of `AgentEvent`s.
    ///
    /// The stream ends with `AgentEvent::Done` or `AgentEvent::Error`.
    pub async fn stream(
        &self,
        prompt: impl Into<String>,
    ) -> impl futures::Stream<Item = AgentEvent> {
        let messages = vec![Message::user(prompt.into())];
        let config   = self.make_run_config();
        let cancel   = CancellationToken::new();
        run(Arc::new(config), messages, cancel)
    }

    // ── Multi-turn ────────────────────────────────────────────────────────────

    /// Create a session for multi-turn conversation.
    ///
    /// The session owns the message history. If a `Checkpoint` is configured,
    /// the session resumes from the last saved state.
    ///
    /// ```rust,ignore
    /// let mut session = agent.session("user-42").await;
    /// let stream = session.send("Hello").await;
    /// ```
    pub async fn session(&self, id: impl Into<String>) -> Session {
        Session::new(id, self.config.clone()).await
    }

    // ── Sub-agent ─────────────────────────────────────────────────────────────

    /// Return a `SpawnFn` that spawns this agent as a sub-agent.
    ///
    /// Pass the result to another agent's `.spawn()` builder method. Tools
    /// in the outer agent can then call `ctx.spawn_agent(prompt)` to delegate
    /// work to the inner agent — inheriting its tools, system prompt, and model.
    ///
    /// ```rust,ignore
    /// let researcher = Agent::builder(provider.clone())
    ///     .system("You are a research assistant.")
    ///     .tool(WebSearch)
    ///     .build();
    ///
    /// let orchestrator = Agent::builder(provider)
    ///     .system("You are an orchestrator.")
    ///     .spawn(researcher.as_spawn_fn())
    ///     .build();
    /// ```
    ///
    /// The sub-agent runs a fresh single-turn `run()` for each invocation.
    /// It does not share conversation history with the parent — it receives
    /// only the prompt string and returns only the final text response.
    pub fn as_spawn_fn(&self) -> SpawnFn {
        let agent = Arc::new(self.clone());
        Arc::new(move |prompt: String| {
            let agent = agent.clone();
            Box::pin(async move {
                agent.run(prompt).await
                    .map_err(|e| anyhow::anyhow!("{}", e.message))
            })
        })
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    fn make_run_config(&self) -> RunConfig {
        RunConfig {
            provider:      self.config.provider.clone(),
            tools:         Arc::new(build_registry(&self.config.tools)),
            hooks:         Arc::new(HookRunner::new(self.config.hooks.clone())),
            compress:      self.config.compress.clone(),
            permission:    self.config.permission.clone(),
            session_perms: Arc::new(SessionPermissions::new()),
            system:        self.config.system.clone(),
            model:         self.config.model.clone(),
            max_tokens:    self.config.max_tokens,
            temperature:   self.config.temperature,
            max_iter:      self.config.max_iter,
            extensions:         self.config.extensions.clone(),
            initial_extensions: self.config.initial_extensions.clone(),
            spawn:              self.config.spawn.clone(),
            query_chain:        self.config.query_chain.clone(),
        }
    }
}

/// Build the tool registry, automatically injecting `ToolSearch` when
/// deferred tools are present. The LLM needs ToolSearch to discover and
/// load their schemas before calling them.
pub(crate) fn build_registry(tools: &[Arc<dyn wuhu_core::tool::Tool>]) -> ToolRegistry {
    let has_deferred = tools.iter().any(|t| t.defer_loading());
    if !has_deferred {
        return ToolRegistry::new(tools.to_vec());
    }
    let deferred: Vec<_> = tools.iter()
        .filter(|t| t.defer_loading())
        .cloned()
        .collect();
    let mut all = tools.to_vec();
    all.push(Arc::new(ToolSearch::new(deferred)));
    ToolRegistry::new(all)
}
