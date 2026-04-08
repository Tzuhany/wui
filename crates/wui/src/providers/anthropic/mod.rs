// ============================================================================
// Anthropic Provider
//
// Implements the Provider trait against Anthropic's Messages API.
// Supports: streaming, prompt caching, extended thinking, tool use.
// ============================================================================

mod serialize;
mod sse;

use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use futures::{Stream, StreamExt};

use wui_core::event::StreamEvent;
use wui_core::provider::{ChatRequest, Provider, ProviderError};

use self::serialize::build_request_body;
use self::sse::SseParser;

const ANTHROPIC_VERSION: &str = "2023-06-01";

#[derive(Clone)]
pub struct Anthropic {
    client: reqwest::Client,
    api_key: String,
    api_url: String,
    default_model: String,
    beta_headers: Vec<String>,
    thinking_budget: Option<u32>,
    /// When true, `cache_control: {type: ephemeral}` markers are injected at
    /// the end of the system prompt and after the last tool definition so
    /// Anthropic's prompt-caching layer can reuse them across turns.
    cache_enabled: bool,
}

impl Anthropic {
    /// Create a provider using the official Anthropic API endpoint.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, "https://api.anthropic.com")
    }

    /// Create a provider with a custom base URL (proxy, local mock, etc.).
    ///
    /// The messages path (`/v1/messages`) is appended automatically:
    /// - `"https://my-proxy.example.com"` → `".../v1/messages"`
    /// - `"https://my-proxy.example.com/v1"` → `".../v1/messages"`
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("failed to build reqwest client"),
            api_key: api_key.into(),
            api_url: messages_url(&base_url.into()),
            default_model: "claude-opus-4-6".to_string(),
            beta_headers: Vec::new(),
            thinking_budget: None,
            cache_enabled: false,
        }
    }

    /// Override the provider's default model.
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// Enable one Anthropic beta header.
    pub fn with_beta(mut self, beta: impl Into<String>) -> Self {
        self.beta_headers.push(beta.into());
        self
    }

    /// Enable Anthropic prompt caching.
    ///
    /// Adds the required beta header **and** injects `cache_control` markers
    /// at the two highest-value breakpoints:
    ///
    /// 1. **System prompt** — constant across all turns; highest cache hit rate.
    /// 2. **Last tool definition** — constant while tools don't change.
    ///
    /// Cache hits reduce input token cost to ~10% (write: 1.25×, read: 0.1×).
    /// Enable this whenever you make repeated calls with the same system prompt
    /// and tool set.
    pub fn with_prompt_caching(mut self) -> Self {
        self.beta_headers
            .push("prompt-caching-2024-07-31".to_string());
        self.cache_enabled = true;
        self
    }

    /// Enable extended thinking with a budget in tokens.
    pub fn with_thinking_budget(mut self, budget_tokens: u32) -> Self {
        self.thinking_budget = Some(budget_tokens);
        self
    }
}

/// Derive the messages endpoint from a base URL.
fn messages_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with("/messages") {
        return base.to_string();
    }
    if base.ends_with("/v1") {
        format!("{base}/messages")
    } else {
        format!("{base}/v1/messages")
    }
}

#[async_trait]
impl Provider for Anthropic {
    async fn stream(
        &self,
        req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        // Request-level thinking_budget overrides the provider-level default.
        let thinking_budget = req.thinking_budget.or(self.thinking_budget);
        let body = build_request_body(
            &req,
            &self.default_model,
            thinking_budget,
            self.cache_enabled,
        );

        let mut request = self
            .client
            .post(&self.api_url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json");

        if !self.beta_headers.is_empty() {
            request = request.header("anthropic-beta", self.beta_headers.join(","));
        }

        let response = request
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Stream(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            if status.as_u16() == 429 {
                return Err(ProviderError::RateLimit {
                    retry_after_ms: 5_000,
                });
            }
            if status.as_u16() == 401 {
                return Err(ProviderError::Auth(text));
            }
            return Err(ProviderError::ServerError {
                status: status.as_u16(),
                message: text,
            });
        }

        // Stateful parser — one per stream, tracks content-block index → tool_use_id.
        let mut parser = SseParser::default();

        use eventsource_stream::Eventsource as _;
        let stream = response
            .bytes_stream()
            .eventsource()
            .map(move |result| match result {
                Ok(event) if !event.data.is_empty() => parser.parse(&event.event, &event.data),
                Ok(_) => Ok(None),
                Err(e) => Err(ProviderError::Stream(e.to_string())),
            })
            .filter_map(|r| async move {
                match r {
                    Ok(Some(event)) => Some(Ok(event)),
                    Ok(None) => None,
                    Err(e) => Some(Err(e)),
                }
            });

        Ok(Box::pin(stream))
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_derivation() {
        assert_eq!(
            messages_url("https://api.anthropic.com"),
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(
            messages_url("https://api.anthropic.com/v1"),
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(
            messages_url("https://api.anthropic.com/v1/messages"),
            "https://api.anthropic.com/v1/messages"
        );
    }
}
