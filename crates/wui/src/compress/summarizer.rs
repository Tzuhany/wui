// ── SummarizingCompressor ─────────────────────────────────────────────────────

use std::sync::Arc;

use async_trait::async_trait;

use super::estimator::{block_text, CharRatioEstimator, TokenEstimator};
use super::{CompressResult, CompressStrategy};

/// A simple compressor that keeps the most-recent turns verbatim and replaces
/// older turns with a single LLM-generated summary.
///
/// Unlike [`super::CompressPipeline`]'s multi-tier approach, `SummarizingCompressor`
/// is intentionally simple: one threshold, one LLM call, one summary. Choose
/// it when you want predictable behaviour and a single compression knob.
///
/// ```rust,ignore
/// Agent::builder(provider)
///     .compress(SummarizingCompressor::default())
///     .build()
/// ```
pub struct SummarizingCompressor {
    /// Number of most-recent turns to preserve verbatim. Default: 6.
    pub recent_turns: usize,
    /// Override model for the summarisation call (e.g. a cheaper/faster model).
    /// `None` uses the same model as the main agent.
    pub summary_model: Option<String>,
    /// System prompt sent to the summarisation model.
    pub summary_system: String,
    /// Pressure fraction `[0.0, 1.0]` above which compression triggers.
    /// Default: 0.75.
    pub threshold: f32,
    /// Token estimator used for pressure calculations.
    /// Default: [`CharRatioEstimator`] with `chars_per_token = 4`.
    pub token_estimator: Arc<dyn TokenEstimator>,
    /// Assumed context window size in tokens.
    /// Default: 200 000.
    pub window_tokens: usize,
}

impl Default for SummarizingCompressor {
    fn default() -> Self {
        Self {
            recent_turns: 6,
            summary_model: None,
            summary_system: "Summarize the following conversation concisely, preserving all \
                              key decisions, facts, tool results, and context needed to \
                              continue the task."
                .to_string(),
            threshold: 0.75,
            token_estimator: Arc::new(CharRatioEstimator::default()),
            window_tokens: 200_000,
        }
    }
}

impl SummarizingCompressor {
    fn estimate_tokens(&self, messages: &[wui_core::message::Message]) -> usize {
        messages
            .iter()
            .flat_map(|m| m.content.iter())
            .map(|b| self.token_estimator.estimate(&block_text(b)))
            .sum()
    }

    fn current_pressure(&self, messages: &[wui_core::message::Message]) -> f32 {
        self.estimate_tokens(messages) as f32 / self.window_tokens.max(1) as f32
    }

    async fn run_compress(
        &self,
        messages: Vec<wui_core::message::Message>,
        provider: Arc<dyn wui_core::provider::Provider>,
        model: Option<&str>,
    ) -> anyhow::Result<CompressResult> {
        use futures::StreamExt;
        use wui_core::event::StreamEvent;
        use wui_core::message::Role;
        use wui_core::provider::ChatRequest;

        let n = messages.len();
        let pressure = self.current_pressure(&messages);
        if (pressure as f64) < self.threshold as f64 || n == 0 {
            return Ok(CompressResult {
                messages,
                freed: 0,
                method: None,
            });
        }

        let recent_count = self.recent_turns.min(n);
        let split_at = n.saturating_sub(recent_count);
        let (old, recent) = messages.split_at(split_at);

        if old.is_empty() {
            return Ok(CompressResult {
                messages,
                freed: 0,
                method: None,
            });
        }

        // Format the old messages as a transcript.
        let transcript: String = old
            .iter()
            .flat_map(|m| m.content.iter())
            .map(block_text)
            .filter(|t| !t.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");

        if transcript.trim().is_empty() {
            return Ok(CompressResult {
                messages,
                freed: 0,
                method: None,
            });
        }

        let req = ChatRequest {
            model: model
                .map(str::to_string)
                .or_else(|| self.summary_model.clone()),
            max_tokens: 2048,
            temperature: Some(0.0),
            system: self.summary_system.clone(),
            messages: vec![wui_core::message::Message::user(transcript)],
            tools: vec![],
            thinking_budget: None,
        };

        let stream = provider.stream(req).await?;
        futures::pin_mut!(stream);

        let mut summary = String::new();
        while let Some(Ok(event)) = stream.next().await {
            if let StreamEvent::TextDelta { text } = event {
                summary.push_str(&text);
            }
        }

        if summary.trim().is_empty() {
            return Ok(CompressResult {
                messages,
                freed: 0,
                method: None,
            });
        }

        let tokens_before = self.estimate_tokens(&messages);

        let placeholder = wui_core::message::Message {
            id: uuid::Uuid::new_v4().to_string(),
            role: Role::System,
            content: vec![wui_core::message::ContentBlock::CompactBoundary { summary }],
        };

        let mut result = vec![placeholder];
        result.extend_from_slice(recent);

        let tokens_after = self.estimate_tokens(&result);
        let freed = tokens_before.saturating_sub(tokens_after);

        Ok(CompressResult {
            messages: result,
            freed,
            method: Some(wui_core::event::CompressMethod::Summarize),
        })
    }
}

#[async_trait]
impl CompressStrategy for SummarizingCompressor {
    async fn compress(
        &self,
        messages: Vec<wui_core::message::Message>,
        provider: Arc<dyn wui_core::provider::Provider>,
        model: Option<&str>,
    ) -> anyhow::Result<CompressResult> {
        self.run_compress(messages, provider, model).await
    }

    fn pressure(&self, messages: &[wui_core::message::Message]) -> f64 {
        self.current_pressure(messages) as f64
    }

    fn threshold(&self) -> f64 {
        self.threshold as f64
    }

    fn is_critically_full(&self, messages: &[wui_core::message::Message]) -> bool {
        self.current_pressure(messages) >= 1.0
    }
}
