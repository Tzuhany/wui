// OpenAI Provider — stub.
// Full implementation follows the same pattern as anthropic.rs.
// Enable with: wuhu-providers = { features = ["openai"] }

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use wuhu_core::event::StreamEvent;
use wuhu_core::provider::{ChatRequest, Provider, ProviderError};

pub struct OpenAI {
    client:  reqwest::Client,
    api_key: String,
}

impl OpenAI {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client:  reqwest::Client::new(),
            api_key: api_key.into(),
        }
    }
}

#[async_trait]
impl Provider for OpenAI {
    async fn stream(
        &self,
        _req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError> {
        // TODO: implement OpenAI SSE streaming.
        Err(ProviderError::Other(anyhow::anyhow!("OpenAI provider not yet implemented")))
    }
}
