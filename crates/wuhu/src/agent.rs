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
use wuhu_engine::{HookRunner, PermissionMode, RunConfig, ToolRegistry, run};

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
    /// ```rust
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
    /// Convenience wrapper over `stream()`. For streaming output or tool
    /// visibility, use `stream()` directly.
    pub async fn run(&self, prompt: impl Into<String>) -> Result<String, wuhu_core::event::AgentError> {
        let mut text = String::new();
        let mut stream = self.stream(prompt).await;

        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::TextDelta(t) => text.push_str(&t),
                AgentEvent::Done(_)      => break,
                AgentEvent::Error(e)     => return Err(e),
                _                        => {}
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
    /// ```rust
    /// let mut session = agent.session("user-42").await;
    /// let stream = session.send("Hello").await;
    /// ```
    pub async fn session(&self, id: impl Into<String>) -> Session {
        Session::new(id, self.config.clone()).await
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    fn make_run_config(&self) -> RunConfig {
        RunConfig {
            provider:    self.config.provider.clone(),
            tools:       Arc::new(ToolRegistry::new(self.config.tools.clone())),
            hooks:       Arc::new(HookRunner::new(self.config.hooks.clone())),
            compress:    self.config.compress.clone(),
            permission:  self.config.permission.clone(),
            system:      self.config.system.clone(),
            model:       self.config.model.clone(),
            max_tokens:  self.config.max_tokens,
            temperature: self.config.temperature,
            max_iter:    self.config.max_iter,
            extensions:  self.config.extensions.clone(),
            spawn:       self.config.spawn.clone(),
        }
    }
}
