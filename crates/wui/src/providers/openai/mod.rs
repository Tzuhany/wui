// ============================================================================
// OpenAI Provider
//
// Implements the Provider trait against OpenAI's Chat Completions API.
// Supports: streaming, tool use (function calling).
//
// Compatible with: gpt-4o, gpt-4o-mini, gpt-4-turbo, and any
// OpenAI-compatible endpoint (Azure OpenAI, Together, Groq, etc.).
//
// Note: o-series reasoning models (o1, o3, o4) use different parameters
// (`max_completion_tokens`, `reasoning_effort`) and are not covered by this
// provider out of the box. For those, use `with_default_model` and configure
// the request body via a custom Provider implementation.
// ============================================================================

mod parser;
mod serialize;

use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use futures::{Stream, StreamExt as _};

use wui_core::event::StreamEvent;
use wui_core::provider::{ChatRequest, Provider, ProviderError};

use self::parser::SseParser;
use self::serialize::build_request_body;

// ── OpenAi ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct OpenAi {
    client: reqwest::Client,
    api_key: String,
    api_url: String,
    default_model: String,
}

impl OpenAi {
    /// Create a provider using the official OpenAI API endpoint.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, "https://api.openai.com")
    }

    /// Create a provider with a custom base URL (proxy, Azure, local server, etc.).
    ///
    /// The chat completions path is appended automatically:
    /// - `"https://my-proxy.example.com"` → `".../v1/chat/completions"`
    /// - `"https://my-proxy.example.com/v1"` → `".../v1/chat/completions"`
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("failed to build reqwest client"),
            api_key: api_key.into(),
            api_url: chat_completions_url(&base_url.into()),
            default_model: "gpt-4o".to_string(),
        }
    }

    /// Override the provider's default model.
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }
}

fn chat_completions_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with("/chat/completions") {
        return base.to_string();
    }
    if base.ends_with("/v1") {
        format!("{base}/chat/completions")
    } else {
        format!("{base}/v1/chat/completions")
    }
}

#[async_trait]
impl Provider for OpenAi {
    async fn stream(
        &self,
        req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        let body = build_request_body(&req, &self.default_model);

        let response = self
            .client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
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

        let mut parser = SseParser::default();

        use eventsource_stream::Eventsource as _;
        let stream = response
            .bytes_stream()
            .eventsource()
            .flat_map(move |result| {
                // One SSE chunk may produce zero, one, or many StreamEvents.
                // `flat_map` expands the Vec into individual stream items.
                let events: Vec<Result<StreamEvent, ProviderError>> = match result {
                    Ok(event) if !event.data.is_empty() && event.data != "[DONE]" => {
                        match parser.parse_all(&event.data) {
                            Ok(evs) => evs.into_iter().map(Ok).collect(),
                            Err(e) => vec![Err(e)],
                        }
                    }
                    Ok(_) => vec![],
                    Err(e) => vec![Err(ProviderError::Stream(e.to_string()))],
                };
                futures::stream::iter(events)
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
            chat_completions_url("https://api.openai.com"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("https://api.openai.com/v1/chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("https://proxy.example.com/"),
            "https://proxy.example.com/v1/chat/completions"
        );
    }
}
