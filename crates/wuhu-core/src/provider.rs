// ============================================================================
// Provider — the intelligence behind the agent.
//
// The framework is agnostic about which LLM it calls. A Provider is simply:
// given a conversation, give me a stream of events. That is the entire
// contract.
//
// `ChatRequest` carries an `extensions` map for provider-specific features
// (Anthropic's prompt caching, extended thinking; OpenAI's reasoning effort)
// without polluting the universal interface with vendor-specific fields.
// ============================================================================

use std::collections::HashMap;
use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use serde_json::Value;

use crate::event::StreamEvent;
use crate::message::Message;
use crate::tool::Tool;

/// The LLM backend.
///
/// Implement this to connect any model to the Wuhu engine.
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
}

// ── Chat Request ──────────────────────────────────────────────────────────────

/// Everything a Provider needs to produce a response.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model:       String,
    pub max_tokens:  u32,
    pub temperature: Option<f32>,
    pub system:      String,
    pub messages:    Vec<Message>,
    pub tools:       Vec<ToolDef>,

    /// Provider-specific configuration.
    ///
    /// Examples:
    /// - `"thinking" → {"type": "enabled", "budget_tokens": 8000}` (Anthropic)
    /// - `"betas" → ["prompt-caching-2024-07-31"]` (Anthropic)
    /// - `"reasoning_effort" → "high"` (OpenAI)
    pub extensions:  HashMap<String, Value>,
}

impl ChatRequest {
    pub fn extend(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.extensions.insert(key.into(), value.into());
        self
    }
}

/// A tool definition as sent to the LLM.
///
/// Derived from a `Tool` impl by the engine. Users don't construct these
/// directly.
#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name:         String,
    pub description:  String,
    pub input_schema: Value,
}

impl ToolDef {
    pub fn from_tool(tool: &dyn Tool) -> Self {
        Self {
            name:         tool.name().to_string(),
            description:  tool.description().to_string(),
            input_schema: tool.input_schema(),
        }
    }
}

// ── Provider Error ────────────────────────────────────────────────────────────

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
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::RateLimit { .. } | Self::ServerError { .. } | Self::Timeout)
    }
}
