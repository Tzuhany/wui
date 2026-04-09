// ============================================================================
// Context Compression Pipeline
//
// Context windows are finite. This pipeline makes that fact graceful.
//
// Three tiers, applied in order, stopping at the first that relieves
// sufficient pressure:
//
//   L1 · Budget Trim  — truncate oversized tool results. Free. Always runs.
//   L2 · Collapse     — fold old messages into placeholders. No LLM call.
//   L3 · Summarize    — LLM summarises the oldest batch. Expensive, irreversible.
//
// The pipeline returns `None` when no compression is needed, or
// `Some((new_messages, method, tokens_freed))` when it acted.
//
// ── Configurable heuristics ───────────────────────────────────────────────────
//
// Compression decisions (how many messages to keep, when to trigger) are
// *configurable* — the framework never silently makes these choices for you.
// `CompressPipeline::default()` provides sensible starting values, but all
// thresholds are fields you can override via struct literal or builder.
// ============================================================================

mod estimator;
mod pipeline;
mod summarizer;

use std::sync::Arc;

use async_trait::async_trait;

use wui_core::event::CompressMethod;

// ── Re-exports ───────────────────────────────────────────────────────────────

pub use estimator::{CharRatioEstimator, ContextBreakdown, TokenEstimator};
pub use pipeline::CompressPipeline;
pub use summarizer::SummarizingCompressor;

// ── CompressStrategy ──────────────────────────────────────────────────────────

/// Pluggable context compression strategy.
///
/// The default implementation is [`CompressPipeline`] (L1 trim → L2 collapse →
/// L3 summarise). Replace it to implement custom summarisation prompts, external
/// summarisation services, or alternative compression algorithms.
///
/// # Example
///
/// ```rust,ignore
/// Agent::builder(provider)
///     .compress(MyStrategy::new())
///     .build()
/// ```
#[async_trait]
pub trait CompressStrategy: Send + Sync + 'static {
    /// Compress `messages` if needed, returning the (possibly shorter) list.
    ///
    /// The implementation may call the provider's LLM for summarisation.
    /// `freed` is the number of tokens estimated to have been freed; 0 if
    /// no compression was applied.
    async fn compress(
        &self,
        messages: Vec<wui_core::message::Message>,
        provider: Arc<dyn wui_core::provider::Provider>,
        model: Option<&str>,
    ) -> anyhow::Result<CompressResult>;

    /// Pressure in `[0.0, 1.0]`: current token usage / window size.
    /// Used to decide whether compression should run at all.
    fn pressure(&self, messages: &[wui_core::message::Message]) -> f64;

    /// Threshold above which `compress()` is triggered.
    fn threshold(&self) -> f64;

    /// Whether the context is critically full (at or beyond window).
    fn is_critically_full(&self, messages: &[wui_core::message::Message]) -> bool;

    /// Classify current context pressure into a discrete level.
    ///
    /// Uses `pressure()` and `threshold()` to determine the level:
    /// - `Normal` — below 75% of the threshold
    /// - `Elevated` — 75%–100% of the threshold
    /// - `Critical` — at or above the threshold
    fn pressure_level(
        &self,
        messages: &[wui_core::message::Message],
    ) -> wui_core::event::ContextPressure {
        let p = self.pressure(messages);
        if p >= self.threshold() {
            wui_core::event::ContextPressure::Critical
        } else if p >= self.threshold() * 0.75 {
            wui_core::event::ContextPressure::Elevated
        } else {
            wui_core::event::ContextPressure::Normal
        }
    }
}

/// Result of a compression pass.
pub struct CompressResult {
    pub messages: Vec<wui_core::message::Message>,
    /// Tokens freed (0 if no compression was applied).
    pub freed: usize,
    /// Which compression method was applied, if any.
    pub method: Option<CompressMethod>,
}
