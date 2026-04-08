// ── Token Estimation & Context Breakdown ─────────────────────────────────────

use wui_core::message::ContentBlock;

// ── TokenEstimator ───────────────────────────────────────────────────────────

/// Estimates the token count of a text string.
///
/// The default implementation ([`CharRatioEstimator`]) divides `text.len()` by
/// a configurable characters-per-token ratio. Replace it with a provider-specific
/// tokenizer (tiktoken, Anthropic tokenizer, etc.) for accurate counts —
/// especially important for CJK text or code where the `len()/4` heuristic
/// can be off by 2-4x.
///
/// # Example
///
/// ```rust,ignore
/// struct TiktokenEstimator { /* ... */ }
///
/// impl TokenEstimator for TiktokenEstimator {
///     fn estimate(&self, text: &str) -> usize {
///         self.bpe.encode_ordinary(text).len()
///     }
/// }
///
/// let pipeline = CompressPipeline::default()
///     .token_estimator(TiktokenEstimator::new());
/// ```
pub trait TokenEstimator: Send + Sync + 'static {
    /// Return the estimated number of tokens in `text`.
    fn estimate(&self, text: &str) -> usize;
}

/// Default token estimator: `text.len() / chars_per_token`.
///
/// Accurate for English ASCII prose at the default of 4. For code or CJK
/// text, use `chars_per_token = 2` or supply a real tokenizer.
#[derive(Debug, Clone)]
pub struct CharRatioEstimator {
    pub chars_per_token: usize,
}

impl Default for CharRatioEstimator {
    fn default() -> Self {
        Self { chars_per_token: 4 }
    }
}

impl TokenEstimator for CharRatioEstimator {
    fn estimate(&self, text: &str) -> usize {
        text.len() / self.chars_per_token.max(1)
    }
}

// ── ContextBreakdown ─────────────────────────────────────────────────────────

/// Token usage breakdown by category.
///
/// Produced by [`CompressPipeline::breakdown`]. Exposes where tokens are
/// being spent so callers (dashboards, CLI tools, tests) can diagnose
/// context pressure without guessing.
#[derive(Debug, Clone, Default)]
pub struct ContextBreakdown {
    /// Tokens consumed by user text messages.
    pub user_text_tokens: usize,
    /// Tokens consumed by assistant text + thinking blocks.
    pub assistant_tokens: usize,
    /// Tokens consumed by tool use blocks (tool name + input JSON).
    pub tool_use_tokens: usize,
    /// Tokens consumed by tool result blocks.
    pub tool_result_tokens: usize,
    /// Tokens consumed by system/compressed/collapsed blocks.
    pub system_tokens: usize,
    /// Total tokens across all categories.
    pub total: usize,
    /// Context window size in tokens.
    pub window: usize,
    /// Current pressure (total / window).
    pub pressure: f64,
}

/// Extract a textual representation of a [`ContentBlock`] for token estimation.
pub(crate) fn block_text(block: &ContentBlock) -> String {
    match block {
        ContentBlock::Text { text } => text.clone(),
        ContentBlock::Thinking { text } => text.clone(),
        ContentBlock::ToolUse { name, input, .. } => format!("[Tool: {name}] {input}"),
        ContentBlock::ToolResult { content, .. } => content.clone(),
        ContentBlock::Collapsed { summary, .. } => summary.clone(),
        ContentBlock::CompactBoundary { summary } => summary.clone(),
        ContentBlock::Image { .. } => "[image]".to_string(),
        ContentBlock::Document { title, .. } => {
            format!("[document: {}]", title.as_deref().unwrap_or("untitled"))
        }
    }
}
