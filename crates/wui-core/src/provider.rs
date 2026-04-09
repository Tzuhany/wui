// ============================================================================
// Provider — the intelligence behind the agent.
//
// The framework is agnostic about which LLM it calls. A Provider is simply:
// given a conversation, give me a stream of events. That is the entire
// contract.
//
// Provider-specific behavior belongs with the provider implementation, not in
// the universal request shape.
// ============================================================================

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use serde_json::Value;

use serde::{Deserialize, Serialize};

use crate::event::StreamEvent;
use crate::message::Message;
use crate::tool::Tool;

// ── ProviderError ────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("rate limit exceeded, retry after {retry_after_ms}ms")]
    RateLimit { retry_after_ms: u64 },

    #[error("server error ({status}): {message}")]
    ServerError { status: u16, message: String },

    #[error("authentication failed: {0}")]
    Auth(String),

    #[error("request timed out")]
    Timeout,

    #[error("stream error: {0}")]
    Stream(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl ProviderError {
    /// Whether this error is transient and worth retrying.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::RateLimit { .. } | Self::ServerError { .. } | Self::Timeout
        )
    }
}

// ── ProviderCapabilities ─────────────────────────────────────────────────────

/// Declares what a provider (and its model) can handle.
///
/// The runtime uses these capabilities for preflight checks: if the request
/// requires a feature the provider doesn't support, the run fails explicitly
/// instead of letting the API return a confusing error.
///
/// All fields default to the most permissive values so that providers that
/// don't override `capabilities()` behave the same as before.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ProviderCapabilities {
    /// Supports function/tool calling in the request.
    pub tool_calling: bool,
    /// Supports extended thinking / chain-of-thought budget.
    pub thinking: bool,
    /// Accepts image content blocks in messages.
    pub image_input: bool,
    /// Accepts document content blocks in messages.
    pub document_input: bool,
    /// Supports structured (JSON mode) output.
    pub structured_output: bool,
    /// Maximum context window size in tokens.
    ///
    /// When set, the compression pipeline can auto-calibrate its
    /// `window_tokens` instead of relying on manual configuration.
    pub max_context_window: Option<usize>,
}

impl Default for ProviderCapabilities {
    fn default() -> Self {
        Self {
            tool_calling: true,
            thinking: true,
            image_input: true,
            document_input: true,
            structured_output: false,
            max_context_window: None,
        }
    }
}

impl ProviderCapabilities {
    /// Create with all capabilities enabled and no context window limit.
    pub fn all() -> Self {
        Self::default()
    }

    /// Set whether tool calling is supported.
    pub fn with_tool_calling(mut self, v: bool) -> Self {
        self.tool_calling = v;
        self
    }
    /// Set whether extended thinking is supported.
    pub fn with_thinking(mut self, v: bool) -> Self {
        self.thinking = v;
        self
    }
    /// Set whether image input is supported.
    pub fn with_image_input(mut self, v: bool) -> Self {
        self.image_input = v;
        self
    }
    /// Set whether document input is supported.
    pub fn with_document_input(mut self, v: bool) -> Self {
        self.document_input = v;
        self
    }
    /// Set whether structured output is supported.
    pub fn with_structured_output(mut self, v: bool) -> Self {
        self.structured_output = v;
        self
    }
    /// Set the maximum context window size in tokens.
    pub fn with_max_context_window(mut self, tokens: usize) -> Self {
        self.max_context_window = Some(tokens);
        self
    }
}

// ── TokenEstimate ────────────────────────────────────────────────────────────

/// A token count estimate returned by [`Provider::count_tokens`].
///
/// The `exact` flag tells the compression pipeline whether to trust this
/// number for budget decisions or to treat it as a heuristic. Both the
/// flag and the source are recorded in telemetry so dashboards never
/// accidentally present estimates as exact counts.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TokenEstimate {
    /// Estimated input tokens for the request.
    pub input_tokens: usize,
    /// The output budget that would be sent with this request.
    pub output_tokens_budget: Option<usize>,
    /// Total prompt tokens (system + messages + tools).
    pub total_prompt_tokens: usize,
    /// `true` when the estimate comes from the provider's own tokenizer.
    pub exact: bool,
}

// ── ResponseFormat ───────────────────────────────────────────────────────────

/// Response format hint for providers that support structured output.
///
/// When set on a [`ChatRequest`], providers that support structured output
/// (see [`ProviderCapabilities::structured_output`]) will constrain the
/// model's response accordingly. Providers that do not support it silently
/// ignore the field — the caller should fall back to prompt-based extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    /// The provider should return valid JSON conforming to the given schema.
    ///
    /// Maps to OpenAI's `response_format: { type: "json_schema", json_schema: { name, schema, strict: true } }`.
    JsonSchema {
        /// A short name for the schema (e.g. the Rust type name).
        name: String,
        /// The JSON Schema that the response must conform to.
        schema: serde_json::Value,
    },
}

// ── ChatRequest + ToolDef ────────────────────────────────────────────────────

/// Everything a Provider needs to produce a response.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub system: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDef>,
    /// Request-level thinking budget (tokens). Overrides any provider-level
    /// default when set. `None` defers to the provider's own configuration.
    pub thinking_budget: Option<u32>,
    /// Byte index in `system` where the cache boundary falls.
    ///
    /// Everything before this index is stable across turns; the provider
    /// may use this to split the system prompt into a cached prefix and a
    /// dynamic suffix. `None` means no boundary (single block).
    pub cache_boundary: Option<usize>,
    /// Requested response format. `None` means natural language (default).
    ///
    /// When set, providers that support structured output will constrain the
    /// model's response to match the specified format. Providers that do not
    /// support structured output ignore this field.
    pub response_format: Option<ResponseFormat>,
}

/// A tool definition as sent to the LLM.
///
/// Derived from a `Tool` impl by the engine. Users don't construct these
/// directly.
#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

impl ToolDef {
    /// Build a `ToolDef` from a `Tool` implementation.
    pub fn from_tool(tool: &dyn Tool) -> Self {
        Self {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            input_schema: tool.input_schema(),
        }
    }
}

// ── Provider Trait ───────────────────────────────────────────────────────────

/// The LLM backend.
///
/// Implement this to connect any model to the Wui engine.
/// The single `stream()` method is intentional: agents need only
/// streaming chat completion. Adding methods for embeddings, fine-tuning,
/// or other capabilities belongs in separate, purpose-built traits.
#[async_trait]
pub trait Provider: Send + Sync + 'static {
    /// Send a chat request and return a stream of raw events.
    ///
    /// The returned stream yields `StreamEvent`s until the model signals
    /// `MessageEnd` or `Error`. The engine handles all other lifecycle
    /// concerns (retries, tool execution, compression).
    async fn stream(
        &self,
        req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>;

    /// Declare what this provider supports for the given model.
    ///
    /// The runtime calls this before the first request to run preflight
    /// checks. Override to report actual capabilities; the default
    /// assumes everything is supported (backward compatible).
    fn capabilities(&self, _model: Option<&str>) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }

    /// Estimate token usage for a request without sending it.
    ///
    /// Returns `Ok(None)` when the provider has no tokenizer available.
    /// The compression pipeline uses this to make accurate budget
    /// decisions; when unavailable it falls back to character-ratio
    /// estimation.
    fn count_tokens(&self, _req: &ChatRequest) -> Result<Option<TokenEstimate>, ProviderError> {
        Ok(None)
    }
}
