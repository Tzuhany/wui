use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::Stream;
use serde_json::Value;
use wui_core::event::{StopReason, StreamEvent, TokenUsage};
use wui_core::provider::{ChatRequest, Provider, ProviderError};
use wui_core::tool::ToolCallId;

// ── MockProvider ──────────────────────────────────────────────────────────────

/// A scripted response for `MockProvider`.
pub enum MockResponse {
    /// The provider emits this text and stops.
    Text(String),
    /// The provider emits a single tool call and stops.
    ToolCall {
        name: String,
        /// A stable id for the tool call (propagated as `tool_use_id`).
        id: String,
        input: Value,
    },
    /// The provider emits a retryable error.
    Error { message: String, retryable: bool },
}

/// A deterministic `Provider` that replays scripted responses in order.
///
/// Each call to `stream()` pops the next `MockResponse` from the queue and
/// returns a stream that yields the corresponding `StreamEvent`s. Panics when
/// the queue is exhausted — this indicates a test bug (the agent made more LLM
/// calls than expected).
///
/// ```rust,ignore
/// let provider = MockProvider::new(vec![
///     MockProvider::text("Hello!"),
///     MockProvider::tool_call("bash", json!({"command":"ls"})),
///     MockProvider::text("Done."),
/// ]);
/// ```
pub struct MockProvider {
    responses: Arc<Mutex<VecDeque<MockResponse>>>,
}

impl MockProvider {
    /// Create a provider that will serve `responses` in order.
    pub fn new(responses: Vec<MockResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(VecDeque::from(responses))),
        }
    }

    /// Construct a `MockResponse::Text` conveniently.
    pub fn text(s: impl Into<String>) -> MockResponse {
        MockResponse::Text(s.into())
    }

    /// Construct a `MockResponse::ToolCall` conveniently.
    ///
    /// A random-ish id is generated from the name. For deterministic ids,
    /// construct `MockResponse::ToolCall` directly.
    pub fn tool_call(name: impl Into<String>, input: Value) -> MockResponse {
        let name = name.into();
        let id = format!("mock_{}_{}", name, uuid_short());
        MockResponse::ToolCall { name, id, input }
    }

    /// Construct a `MockResponse::Error` conveniently.
    pub fn error(message: impl Into<String>, retryable: bool) -> MockResponse {
        MockResponse::Error {
            message: message.into(),
            retryable,
        }
    }
}

pub(crate) fn uuid_short() -> String {
    // Simple deterministic suffix using a counter-like approach.
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed).to_string()
}

#[async_trait]
impl Provider for MockProvider {
    async fn stream(
        &self,
        _req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        let response = {
            let mut queue = self.responses.lock().expect("MockProvider lock poisoned");
            queue.pop_front().expect(
                "MockProvider: no more scripted responses — the agent made more \
                 LLM calls than expected. Add more MockResponse entries to the queue.",
            )
        };

        let events: Vec<Result<StreamEvent, ProviderError>> = match response {
            MockResponse::Text(text) => {
                let chars = text.len() as u32;
                vec![
                    Ok(StreamEvent::TextDelta { text }),
                    Ok(StreamEvent::MessageEnd {
                        usage: TokenUsage {
                            input_tokens: 10,
                            output_tokens: chars,
                            ..Default::default()
                        },
                        stop_reason: StopReason::EndTurn,
                    }),
                ]
            }

            MockResponse::ToolCall { name, id, input } => {
                let input_json = serde_json::to_string(&input).unwrap_or_default();
                let id = ToolCallId::from(id);
                vec![
                    Ok(StreamEvent::ToolUseStart {
                        id: id.clone(),
                        name,
                    }),
                    Ok(StreamEvent::ToolInputDelta {
                        id: id.clone(),
                        chunk: input_json,
                    }),
                    Ok(StreamEvent::ToolUseEnd { id }),
                    Ok(StreamEvent::MessageEnd {
                        usage: TokenUsage {
                            input_tokens: 10,
                            output_tokens: 10,
                            ..Default::default()
                        },
                        stop_reason: StopReason::ToolUse,
                    }),
                ]
            }

            MockResponse::Error { message, retryable } => {
                vec![Ok(StreamEvent::Error { message, retryable })]
            }
        };

        let stream = futures::stream::iter(events);
        Ok(Box::pin(stream))
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use wui::{Agent, PermissionMode, RunStopReason};

    use crate::{AgentHarness, MockProvider};

    #[tokio::test]
    async fn mock_provider_text_response() {
        let provider = MockProvider::new(vec![MockProvider::text("hello world")]);
        let agent = Agent::builder(provider)
            .permission(PermissionMode::Auto)
            .build();

        let h = AgentHarness::run(&agent, "say hello").await;
        h.assert_text_contains("hello world")
            .assert_stop_reason(RunStopReason::Completed);
    }
}
