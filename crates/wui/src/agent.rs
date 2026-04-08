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
use schemars::JsonSchema;

use crate::runtime::{
    run, HookRunner, RunConfig, RunStream, SessionPermissions, ToolRegistry, ToolSearch,
};
use wui_core::event::AgentEvent;
use wui_core::message::Message;
use wui_core::provider::Provider;

use crate::builder::{AgentBuilder, AgentConfig};
use crate::session::Session;
use crate::structured::StructuredRun;

/// A configured agent ready to run.
///
/// Create one via `Agent::builder()`, then call `run()`, `stream()`, or
/// `session()`. Cheap to clone — all configuration is behind an Arc.
#[must_use = "an Agent does nothing until you call run(), stream(), or session()"]
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
    /// **Permission handling:** `run()` does not auto-approve interactive
    /// tool requests. If a tool requires human approval under
    /// `PermissionMode::Ask`, `run()` returns an error and the caller should
    /// switch to `stream()` or `session()`.
    ///
    /// For tool visibility, streaming output, or HITL, use `stream()` directly.
    pub async fn run(
        &self,
        prompt: impl Into<String>,
    ) -> Result<String, wui_core::event::AgentError> {
        let mut text = String::new();
        let mut stream = self.stream(prompt);

        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::TextDelta(t) => text.push_str(&t),
                AgentEvent::Control(handle) => {
                    let tool_name = handle.request.tool_name().map(str::to_owned);
                    handle.deny("Agent::run() does not support interactive approvals");
                    return Err(wui_core::event::AgentError::permission_required(
                        tool_name.as_deref(),
                    ));
                }
                AgentEvent::Done(_) => break,
                AgentEvent::Error(e) => return Err(e),
                _ => {}
            }
        }

        Ok(text)
    }

    // ── Single-turn stream ────────────────────────────────────────────────────

    /// Run a single prompt and return a `RunStream` of `AgentEvent`s.
    ///
    /// The stream ends with `AgentEvent::Done` or `AgentEvent::Error`.
    /// Dropping the stream cancels the run. Call `stream.cancel()` for
    /// explicit cancellation or `stream.cancel_token()` to share the
    /// signal with other tasks.
    pub fn stream(&self, prompt: impl Into<String>) -> RunStream {
        let messages = vec![Message::user(prompt.into())];
        run(Arc::new(self.make_run_config()), messages)
    }

    // ── Typed output ─────────────────────────────────────────────────────────

    /// Run a single prompt and deserialise the response into `T`.
    ///
    /// Instructs the LLM to respond with JSON conforming to `T`'s schema,
    /// then parses the response. The schema is derived at compile time via
    /// `schemars::JsonSchema`.
    ///
    /// ```rust,ignore
    /// #[derive(serde::Deserialize, schemars::JsonSchema)]
    /// struct Analysis {
    ///     sentiment: String,
    ///     score: f32,
    ///     keywords: Vec<String>,
    /// }
    ///
    /// let analysis: Analysis = agent
    ///     .run_typed("Analyse the sentiment of: 'Great product!'")
    ///     .await?;
    /// ```
    pub async fn run_typed<T>(
        &self,
        prompt: impl Into<String>,
    ) -> Result<T, wui_core::event::AgentError>
    where
        T: serde::de::DeserializeOwned + JsonSchema,
    {
        // Generate a JSON Schema for T so we can inject it into the prompt.
        let schema_root = schemars::gen::SchemaGenerator::default().into_root_schema_for::<T>();
        let schema_str =
            serde_json::to_string_pretty(&schema_root).unwrap_or_else(|_| "{}".to_string());

        // Create a derived agent whose system prompt includes the format instruction.
        // This does not mutate the original agent — we clone the Arc'd config and
        // add the instruction to the copy.
        let mut config = (*self.config).clone();
        let instruction = format!(
            "\n\nIMPORTANT: Your response must be a single valid JSON value with no \
             explanation, no markdown code fences, and no surrounding text. \
             The JSON must conform to this schema:\n{schema_str}"
        );
        config.system.push_str(&instruction);
        let typed_agent = Agent {
            config: Arc::new(config),
        };

        let text = typed_agent.run(prompt).await?;

        // Strip markdown fences (```json ... ```) that some models emit
        // despite being asked not to.
        let json_str = strip_json_fences(text.trim());

        serde_json::from_str(json_str).map_err(|e| {
            wui_core::event::AgentError::fatal(format!(
                "response is not valid JSON for the requested type: {e}\nRaw: {text}"
            ))
        })
    }

    // ── Structured output (XML extraction) ───────────────────────────────────

    /// Run a prompt and extract structured output from XML-tagged regions.
    ///
    /// Returns a [`StructuredRun`] builder. Chain `.extract("tag")` to pull a
    /// single tagged value or `.extract_all()` to get every top-level tag.
    ///
    /// ```rust,ignore
    /// let agent = Agent::builder(provider)
    ///     .system("Always wrap your final answer in <answer> tags.")
    ///     .build();
    ///
    /// let result = agent
    ///     .run_structured("What is 2+2?")
    ///     .extract("answer")
    ///     .await?;
    /// // result == "4"
    /// ```
    pub fn run_structured(&self, prompt: impl Into<String>) -> StructuredRun<'_> {
        StructuredRun {
            agent: self,
            prompt: prompt.into(),
        }
    }

    // ── Multi-turn ────────────────────────────────────────────────────────────

    /// Create a session for multi-turn conversation.
    ///
    /// The session owns the message history. If a `SessionStore` is configured,
    /// the session resumes from the last saved state.
    ///
    /// ```rust,ignore
    /// let mut session = agent.session("user-42").await;
    /// let stream = session.send("Hello").await;
    /// ```
    pub async fn session(&self, id: impl Into<String>) -> Session {
        Session::new(id.into(), self.config.clone()).await
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    fn make_run_config(&self) -> RunConfig {
        build_run_config(&self.config, Arc::new(SessionPermissions::new()))
    }
}

/// Strip markdown code fences from a JSON response.
///
/// Models sometimes wrap JSON in ` ```json ... ``` ` or ` ``` ... ``` ` blocks
/// despite being instructed not to. This helper strips those fences so the
/// remaining string can be parsed directly.
fn strip_json_fences(text: &str) -> &str {
    let text = text.trim();
    // Try ```json first, then bare ```.
    for prefix in &["```json", "```"] {
        if let Some(rest) = text.strip_prefix(prefix) {
            if let Some(i) = rest.rfind("```") {
                return rest[..i].trim();
            }
        }
    }
    text
}

/// Build a `RunConfig` from an `AgentConfig` with an explicit permission store.
///
/// `session_perms` is separated out so `Agent` (fresh perms each run) and
/// `Session` (shared perms across turns) can use the same construction logic.
pub(crate) fn build_run_config(
    config: &AgentConfig,
    session_perms: Arc<SessionPermissions>,
) -> RunConfig {
    RunConfig {
        provider: config.provider.clone(),
        tools: Arc::new(build_registry(
            &config.tools,
            &config.deferred_tools,
            &config.catalogs,
        )),
        hooks: Arc::new(HookRunner::new(config.hooks.clone())),
        compress: config.compress.clone(),
        permission: config.permission.clone(),
        rules: config.rules.clone(),
        session_perms,
        system: config.system.clone(),
        model: config.model.clone(),
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        max_iter: config.max_iter,
        retry: config.retry.clone(),
        tool_timeout: config.tool_timeout,
        ignore_diminishing_returns: config.ignore_diminishing_returns,
        token_budget: config.token_budget,
        thinking_budget: config.thinking_budget,
        checkpoint_store: config.checkpoint_store.clone(),
        checkpoint_run_id: config.checkpoint_run_id.clone(),
    }
}

/// Build the tool registry, automatically injecting `ToolSearch` when
/// deferred tools or catalogs are present.
///
/// `ToolSearch` is the default strategy for "lazy schema loading": the LLM
/// sees a compact listing and calls `ToolSearch` to fetch the full schema
/// before use. It is injected here, not in the registry or core, because it
/// is a prompt-economy policy — not a fundamental property of the tool system.
pub(crate) fn build_registry(
    resident: &[Arc<dyn wui_core::tool::Tool>],
    deferred: &[Arc<dyn wui_core::tool::Tool>],
    catalogs: &[Arc<dyn crate::catalog::ToolCatalog>],
) -> ToolRegistry {
    let has_deferred = !deferred.is_empty();
    let has_catalogs = !catalogs.is_empty();

    if !has_deferred && !has_catalogs {
        return ToolRegistry::new(resident.to_vec(), vec![]);
    }

    let tool_search = Arc::new(ToolSearch::new(deferred.to_vec(), catalogs.to_vec()));
    let mut all_resident = resident.to_vec();
    all_resident.push(tool_search);
    ToolRegistry::new(all_resident, deferred.to_vec())
}
