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

use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use schemars::JsonSchema;

use crate::runtime::{
    run, HookRunner, RunConfig, RunStream, SessionPermissions, ToolRegistry, ToolSearch,
};
use wui_core::event::{AgentError, AgentEvent};
use wui_core::message::Message;
use wui_core::provider::Provider;

use super::builder::{AgentBuilder, AgentConfig};
use super::session::Session;

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
        input: impl Into<Message>,
    ) -> Result<String, wui_core::event::AgentError> {
        let mut text = String::new();
        let mut stream = self.stream(input);

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
    pub fn stream(&self, input: impl Into<Message>) -> RunStream {
        let messages = vec![input.into()];
        run(Arc::new(self.make_run_config()), messages)
    }

    // ── JSON output ───────────────────────────────────────────────────────────

    /// Run a single prompt and deserialise the response into `T`.
    ///
    /// Constrains the LLM to respond with JSON matching `T`'s schema (via
    /// provider-native JSON mode when available, prompt injection otherwise),
    /// then parses and **validates** the response against the JSON Schema
    /// before deserialising. When validation fails, the LLM is retried with
    /// a corrective message (up to 2 retries by default).
    ///
    /// Use this when you want the **entire response** as a typed Rust value.
    /// For extracting a typed value from a larger natural-language response,
    /// see [`run_structured`](Agent::run_structured).
    ///
    /// For control over retry count, use [`run_as_with_retries`](Agent::run_as_with_retries).
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
    ///     .run_as("Analyse the sentiment of: 'Great product!'")
    ///     .await?;
    /// ```
    pub async fn run_as<T>(
        &self,
        input: impl Into<Message>,
    ) -> Result<T, wui_core::event::AgentError>
    where
        T: serde::de::DeserializeOwned + JsonSchema,
    {
        self.run_as_with_retries::<T>(input, 2).await
    }

    /// Like [`run_as`](Agent::run_as), but with an explicit retry count for
    /// schema validation failures.
    ///
    /// When the LLM's response fails JSON Schema validation, a corrective
    /// message is sent (via a session) describing the validation errors.
    /// The LLM gets `max_retries` additional attempts to produce valid output.
    ///
    /// Set `max_retries` to `0` to disable validation retries (validation
    /// still runs — errors are reported, just not retried).
    pub async fn run_as_with_retries<T>(
        &self,
        input: impl Into<Message>,
        max_retries: u32,
    ) -> Result<T, wui_core::event::AgentError>
    where
        T: serde::de::DeserializeOwned + JsonSchema,
    {
        // Generate a JSON Schema for T so we can inject it into the prompt.
        let schema_root = schemars::gen::SchemaGenerator::default().into_root_schema_for::<T>();
        let schema_json =
            serde_json::to_value(&schema_root).unwrap_or_else(|_| serde_json::json!({}));
        let schema_str =
            serde_json::to_string_pretty(&schema_json).unwrap_or_else(|_| "{}".to_string());

        // Check whether the provider supports native structured output.
        let caps = self
            .config
            .provider
            .capabilities(self.config.model.as_deref());
        let use_native = caps.structured_output;

        // Create a derived agent whose system prompt includes the format
        // instruction. This does not mutate the original agent — we clone the
        // Arc'd config and add the instruction to the copy.
        let mut config = (*self.config).clone();

        if use_native {
            // Provider supports JSON mode — set response_format and use a
            // lighter system instruction (the schema is enforced by the API).
            let type_name = std::any::type_name::<T>()
                .rsplit("::")
                .next()
                .unwrap_or("Response");
            config.response_format = Some(wui_core::provider::ResponseFormat::JsonSchema {
                name: type_name.to_string(),
                schema: schema_json.clone(),
            });
            config.system.push_str(
                "\n\nIMPORTANT: Respond with a single valid JSON value. \
                 No explanation, no markdown code fences, no surrounding text.",
            );
        } else {
            // Fall back to prompt-based extraction with the full schema.
            let instruction = format!(
                "\n\nIMPORTANT: Your response must be a single valid JSON value with no \
                 explanation, no markdown code fences, and no surrounding text. \
                 The JSON must conform to this schema:\n{schema_str}"
            );
            config.system.push_str(&instruction);
        }

        let typed_agent = Agent {
            config: Arc::new(config),
        };

        // Build the JSON Schema validator once, reuse across retries.
        let validator = jsonschema::validator_for(&schema_json).ok();

        let input_msg: Message = input.into();

        if max_retries == 0 {
            // No retries — single-shot run.
            let text = typed_agent.run(input_msg).await?;
            return parse_and_validate::<T>(&text, validator.as_ref());
        }

        // Use a session for multi-turn retry on validation failure.
        let session = typed_agent
            .session(format!("run_as_{}", uuid_v4_short()))
            .await;

        let text = collect_session_text(&session, input_msg).await?;

        match parse_and_validate::<T>(&text, validator.as_ref()) {
            Ok(value) => Ok(value),
            Err(first_error) => {
                // Retry loop: send corrective messages.
                let mut last_error = first_error;
                for attempt in 1..=max_retries {
                    let correction = Message::user(format!(
                        "Your previous response did not match the required JSON schema.\n\
                         Error: {}\n\n\
                         Please try again. Respond with ONLY a valid JSON value, no explanation.",
                        last_error
                    ));
                    tracing::debug!(
                        attempt,
                        max_retries,
                        error = %last_error,
                        "run_as: retrying after schema validation failure"
                    );
                    let retry_text = collect_session_text(&session, correction).await?;

                    match parse_and_validate::<T>(&retry_text, validator.as_ref()) {
                        Ok(value) => return Ok(value),
                        Err(e) => last_error = e,
                    }
                }
                Err(wui_core::event::AgentError::fatal(format!(
                    "response did not match the required schema after {max_retries} retries: \
                     {last_error}"
                )))
            }
        }
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
    pub fn run_structured(&self, input: impl Into<Message>) -> StructuredRun<'_> {
        StructuredRun {
            agent: self,
            input: input.into(),
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

    /// Create a session pre-seeded with an existing message history.
    ///
    /// Use when resuming a conversation from an external source — a database,
    /// a serialised proto, or any other store that isn't a [`crate::SessionStore`].
    /// The provided messages become the initial history; subsequent `send()`
    /// calls continue from there.
    ///
    /// ```rust,ignore
    /// let history: Vec<Message> = rows.iter().map(|r| r.into_message()).collect();
    /// let session = agent.session_from("conv-42", history).await;
    /// session.send("Continue from here").await;
    /// ```
    pub async fn session_from(&self, id: impl Into<String>, history: Vec<Message>) -> Session {
        Session::new_with_history(id, self.config.clone(), history).await
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    pub(crate) fn stream_with_spawn_depth(
        &self,
        input: impl Into<Message>,
        spawn_depth: u32,
    ) -> RunStream {
        let messages = vec![input.into()];
        let mut config = self.make_run_config();
        config.spawn_depth = spawn_depth;
        run(Arc::new(config), messages)
    }

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
    // Assemble the combined system prompt from sections.
    //
    // Layout: [system] [stable sections...] | cache boundary | [dynamic sections...]
    //
    // The `system` field (from `.system()`) is always part of the stable prefix.
    // If dynamic sections exist, we record the byte offset where they begin so
    // providers can insert a cache boundary marker.
    let mut stable_parts: Vec<&str> = Vec::new();
    if !config.system.is_empty() {
        stable_parts.push(&config.system);
    }
    for s in &config.system_stable {
        if !s.is_empty() {
            stable_parts.push(s);
        }
    }
    let stable = stable_parts.join("\n\n");

    let dynamic_parts: Vec<&str> = config
        .system_dynamic
        .iter()
        .filter(|s| !s.is_empty())
        .map(|s| s.as_str())
        .collect();
    let dynamic = dynamic_parts.join("\n\n");

    let (system, cache_boundary) = if dynamic.is_empty() {
        (stable, None)
    } else {
        let boundary = stable.len();
        (format!("{stable}\n\n{dynamic}"), Some(boundary))
    };

    RunConfig {
        provider: config.provider.clone(),
        tools: Arc::new(
            build_registry(
                &config.tools,
                &config.deferred_tools,
                &config.catalogs,
                config.catalog_limit,
            )
            .expect("tool registry invalid: this is a bug — AgentBuilder::try_build() should have caught duplicate tool names"),
        ),
        hooks: Arc::new(HookRunner::new(config.hooks.clone())),
        compress: config.compress.clone(),
        permission: config.permission.clone(),
        rules: config.rules.clone(),
        session_perms,
        system,
        model: config.model.clone(),
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        max_iter: config.max_iter,
        retry: config.retry.clone(),
        tool_timeout: config.tool_timeout,
        expect_long_task: config.expect_long_task,
        token_budget: config.token_budget,
        thinking_budget: config.thinking_budget,
        checkpoint_store: config.checkpoint_store.clone(),
        checkpoint_run_id: config.checkpoint_run_id.clone(),
        result_store: config.result_store.clone(),
        cache_boundary,
        spawn_depth: 0,
        tool_filter: config.tool_filter.clone(),
        response_format: config.response_format.clone(),
        max_concurrent_tools: config.max_concurrent_tools,
        on_context_overflow: config.on_context_overflow.clone(),
    }
}

/// Build the tool registry, automatically injecting `ToolSearch` when
/// deferred tools or catalogs are present.
///
/// The built-in `tool_search` tool is the default strategy for "lazy schema
/// loading": the LLM sees a compact listing and calls `tool_search` to fetch
/// the full schema before use. The backing `ToolSearch` implementation is
/// injected here, not in the registry or core, because it is a prompt-economy
/// policy — not a fundamental property of the tool system.
pub(crate) fn build_registry(
    resident: &[Arc<dyn wui_core::tool::Tool>],
    deferred: &[Arc<dyn wui_core::tool::Tool>],
    catalogs: &[Arc<dyn crate::catalog::ToolCatalog>],
    catalog_limit: usize,
) -> Result<ToolRegistry, String> {
    let has_deferred = !deferred.is_empty();
    let has_catalogs = !catalogs.is_empty();

    if !has_deferred && !has_catalogs {
        return ToolRegistry::new(resident.to_vec(), vec![]);
    }

    let tool_search = Arc::new(
        ToolSearch::new(deferred.to_vec(), catalogs.to_vec()).with_catalog_limit(catalog_limit),
    );
    let mut all_resident = resident.to_vec();
    all_resident.push(tool_search);
    ToolRegistry::new(all_resident, deferred.to_vec())
}

// ── StructuredRun ───────────────────────────────────────────────────────────

/// A pending structured agent run.
///
/// Create via [`Agent::run_structured`]. Use [`Self::extract`], [`Self::extract_all`], or
/// [`Self::extract_as`] to drive the run and capture the result.
pub struct StructuredRun<'a> {
    pub(crate) agent: &'a Agent,
    pub(crate) input: Message,
}

impl<'a> StructuredRun<'a> {
    /// Run the agent and return the content inside the first `<tag>...</tag>`.
    ///
    /// Returns `Err` if the tag is not found in the response.
    pub async fn extract(self, tag: &str) -> Result<String, AgentError> {
        let tag = tag.to_string();
        let text = self.collect_text().await?;
        wui_core::fmt::extract_tag(&text, &tag)
            .map(str::to_string)
            .ok_or_else(|| {
                AgentError::fatal(format!(
                    "structured output: tag <{tag}> not found in response.\nFull response:\n{text}"
                ))
            })
    }

    /// Run the agent and return all top-level XML tag contents.
    pub async fn extract_all(self) -> Result<HashMap<String, String>, AgentError> {
        let text = self.collect_text().await?;
        Ok(wui_core::fmt::extract_tags(&text))
    }

    /// Run the agent, extract the content inside `<tag>`, and deserialise it
    /// as JSON into `T`.
    ///
    /// Useful when the LLM is asked to respond with a JSON value wrapped in
    /// XML tags, combining the structural clarity of XML boundaries with the
    /// expressive power of JSON for nested data.
    pub async fn extract_as<T>(self, tag: &str) -> Result<T, AgentError>
    where
        T: serde::de::DeserializeOwned,
    {
        let tag = tag.to_string();
        let raw = self.collect_text().await?;
        let inner = wui_core::fmt::extract_tag(&raw, &tag).ok_or_else(|| {
            AgentError::fatal(format!(
                "structured output: tag <{tag}> not found in response.\nFull response:\n{raw}"
            ))
        })?;
        serde_json::from_str(inner).map_err(|e| {
            AgentError::fatal(format!(
                "structured output: failed to deserialise <{tag}> content as JSON: {e}\n\
             Content was:\n{inner}"
            ))
        })
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    async fn collect_text(self) -> Result<String, AgentError> {
        let mut text = String::new();
        let mut stream = self.agent.stream(self.input);

        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::TextDelta(t) => text.push_str(&t),
                AgentEvent::Control(handle) => {
                    let tool_name = handle.request.tool_name().map(str::to_owned);
                    handle.deny("StructuredRun does not support interactive approvals");
                    return Err(AgentError::permission_required(tool_name.as_deref()));
                }
                AgentEvent::Done(_) => break,
                AgentEvent::Error(e) => return Err(e),
                _ => {}
            }
        }

        Ok(text)
    }
}

// ── AgentRunner ───────────────────────────────────────────────────────────────

impl wui_core::runner::AgentRunner for Agent {
    fn run_stream(&self, prompt: String) -> futures::stream::BoxStream<'static, AgentEvent> {
        Box::pin(self.stream(prompt))
    }
}

// ── Helpers for run_as ──────────────────────────────────────────────────────

/// Collect streaming text from a session turn.
async fn collect_session_text(session: &Session, input: Message) -> Result<String, AgentError> {
    let mut text = String::new();
    let mut stream = session.send(input).await;
    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TextDelta(t) => text.push_str(&t),
            AgentEvent::Control(handle) => {
                let tool_name = handle.request.tool_name().map(str::to_owned);
                handle.deny("run_as does not support interactive approvals");
                return Err(AgentError::permission_required(tool_name.as_deref()));
            }
            AgentEvent::Done(_) => break,
            AgentEvent::Error(e) => return Err(e),
            _ => {}
        }
    }
    Ok(text)
}

/// Parse JSON text, optionally validate against a JSON Schema, then
/// deserialise into `T`.
fn parse_and_validate<T: serde::de::DeserializeOwned>(
    raw: &str,
    validator: Option<&jsonschema::Validator>,
) -> Result<T, AgentError> {
    let json_str = strip_json_fences(raw.trim());

    // Parse as a generic JSON value first so we can validate.
    let value: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| AgentError::fatal(format!("response is not valid JSON: {e}\nRaw: {raw}")))?;

    // Validate against the schema if a validator is available.
    if let Some(v) = validator {
        if let Err(error) = v.validate(&value) {
            return Err(AgentError::fatal(format!(
                "JSON schema validation failed: {error}\nRaw: {raw}",
            )));
        }
    }

    // Deserialise from the validated value.
    serde_json::from_value(value).map_err(|e| {
        AgentError::fatal(format!(
            "response is valid JSON but does not match the target type: {e}\nRaw: {raw}"
        ))
    })
}

/// Generate a short pseudo-random ID for internal session naming.
fn uuid_v4_short() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    format!("{nanos:08x}")
}
