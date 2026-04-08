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

use std::sync::Arc;

use async_trait::async_trait;

use wui_core::event::CompressMethod;
use wui_core::message::{ContentBlock, Message, Role};
use wui_core::provider::Provider;

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
}

/// Result of a compression pass.
pub struct CompressResult {
    pub messages: Vec<wui_core::message::Message>,
    /// Tokens freed (0 if no compression was applied).
    pub freed: usize,
    /// Which compression method was applied, if any.
    pub method: Option<CompressMethod>,
}

/// Configuration for the compression pipeline.
///
/// All fields are public — adjust any threshold without a builder API.
/// `CompressPipeline::default()` is a good starting point for most agents.
pub struct CompressPipeline {
    /// Total context window size in tokens.
    pub window_tokens: usize,

    /// Fraction of `window_tokens` at which compression triggers.
    ///
    /// `0.80` means "begin compressing when the context is more than 80% full".
    /// Lower values trigger compression earlier, giving more headroom.
    pub compact_threshold: f64,

    /// Maximum tokens allowed in a single tool result (L1 budget trim).
    ///
    /// Tool results exceeding this are truncated before any LLM call,
    /// with a notice injected so the LLM knows the output was cut.
    pub budget_per_result: usize,

    /// Token estimator used for all token-count calculations.
    ///
    /// Default: [`CharRatioEstimator`] with `chars_per_token = 4`.
    /// Replace with a provider-specific tokenizer for accurate counts.
    pub token_estimator: Arc<dyn TokenEstimator>,

    /// Fraction of messages to keep during L2 collapse.
    ///
    /// `0.5` means "keep the most recent half of the conversation intact
    /// and fold everything older into a placeholder". Range: `(0.0, 1.0)`.
    pub collapse_keep_fraction: f64,

    /// Minimum number of messages to keep during L2 collapse, regardless
    /// of `collapse_keep_fraction`. Prevents collapsing tiny histories.
    pub collapse_keep_min: usize,

    /// Optional focus hint for L3 summarisation.
    ///
    /// When set, the L3 summariser is asked to pay special attention to this
    /// topic when compressing the conversation. Useful when the agent's task
    /// is narrow and you want summaries to preserve domain-specific detail:
    ///
    /// ```rust,ignore
    /// let pipeline = CompressPipeline {
    ///     compact_focus: Some("preserve all file paths, function names, and \
    ///                          security findings".to_string()),
    ///     ..Default::default()
    /// };
    /// ```
    pub compact_focus: Option<String>,

    /// Whether to attempt L3 (LLM summarisation) when L2 is insufficient.
    ///
    /// Set to `false` to skip L3 entirely and rely only on L1 + L2.
    /// Defaults to `true`.
    pub allow_l3: bool,

    /// Aggregate token budget for all tool results in the conversation.
    ///
    /// When the total tokens across all `ToolResult` blocks exceeds this
    /// budget, the oldest results are replaced with `"[result trimmed]"`
    /// placeholders until the total fits. This prevents accumulation of
    /// stale tool results from consuming disproportionate context in long
    /// sessions.
    ///
    /// `None` disables aggregate budgeting (default). L1 per-result trim
    /// still applies independently.
    pub result_token_budget: Option<usize>,
}

impl Default for CompressPipeline {
    fn default() -> Self {
        Self {
            window_tokens: 200_000,
            compact_threshold: 0.80,
            budget_per_result: 10_000,
            token_estimator: Arc::new(CharRatioEstimator::default()),
            collapse_keep_fraction: 0.5,
            collapse_keep_min: 4,
            compact_focus: None,
            allow_l3: true,
            result_token_budget: None,
        }
    }
}

impl CompressPipeline {
    /// Replace the token estimator with a custom implementation.
    pub fn with_token_estimator(mut self, estimator: impl TokenEstimator) -> Self {
        self.token_estimator = Arc::new(estimator);
        self
    }

    /// Produce a token breakdown of the current message history.
    ///
    /// Useful for debugging context pressure — shows exactly where tokens
    /// are being spent (user text, assistant output, tool calls, tool results,
    /// system/compression markers).
    pub fn breakdown(&self, messages: &[Message]) -> ContextBreakdown {
        let mut b = ContextBreakdown {
            window: self.window_tokens,
            ..Default::default()
        };
        for msg in messages {
            for block in &msg.content {
                let tokens = self.estimate_tokens(&block_text(block));
                match block {
                    ContentBlock::Text { .. } => match msg.role {
                        Role::User => b.user_text_tokens += tokens,
                        Role::Assistant => b.assistant_tokens += tokens,
                        Role::System => b.system_tokens += tokens,
                    },
                    ContentBlock::Thinking { .. } => b.assistant_tokens += tokens,
                    ContentBlock::ToolUse { .. } => b.tool_use_tokens += tokens,
                    ContentBlock::ToolResult { .. } => b.tool_result_tokens += tokens,
                    ContentBlock::Collapsed { .. } | ContentBlock::CompactBoundary { .. } => {
                        b.system_tokens += tokens
                    }
                    ContentBlock::Image { .. } | ContentBlock::Document { .. } => {
                        b.user_text_tokens += tokens
                    }
                }
            }
        }
        b.total = b.user_text_tokens
            + b.assistant_tokens
            + b.tool_use_tokens
            + b.tool_result_tokens
            + b.system_tokens;
        b.pressure = b.total as f64 / self.window_tokens.max(1) as f64;
        b
    }
}

impl CompressPipeline {
    /// Estimate token count using the configured token estimator.
    fn estimate_tokens(&self, text: &str) -> usize {
        self.token_estimator.estimate(text)
    }

    fn total_tokens(&self, messages: &[Message]) -> usize {
        messages
            .iter()
            .flat_map(|m| m.content.iter())
            .map(|b| self.estimate_tokens(&block_text(b)))
            .sum()
    }

    /// Fraction of `window_tokens` currently in use. `1.0` = completely full.
    pub fn pressure(&self, messages: &[Message]) -> f64 {
        self.total_tokens(messages) as f64 / self.window_tokens.max(1) as f64
    }

    /// Whether the context has filled or exceeded the hard token ceiling.
    ///
    /// Unlike `compact_threshold` (which triggers compression at e.g. 80%),
    /// this returns `true` only when the context is truly full — no further
    /// LLM call can succeed without shedding messages. The run loop uses this
    /// after compression to decide whether to emit `ContextOverflow`.
    pub fn is_critically_full(&self, messages: &[Message]) -> bool {
        self.total_tokens(messages) >= self.window_tokens
    }

    /// Returns `true` when the context pressure meets or exceeds the
    /// compression threshold and compression should run.
    fn should_compress(&self, messages: &[Message]) -> bool {
        self.pressure(messages) >= self.compact_threshold
    }

    /// How many recent messages to keep during L2/L3 compression.
    fn keep_count(&self, message_count: usize) -> usize {
        ((message_count as f64 * self.collapse_keep_fraction) as usize).max(self.collapse_keep_min)
    }

    /// Run the pipeline. Returns `None` if no compression was needed.
    pub async fn maybe_compress(
        &self,
        messages: &[Message],
        provider: &dyn Provider,
        model: Option<&str>,
    ) -> Option<(Vec<Message>, CompressMethod, usize)> {
        let before = self.total_tokens(messages);

        // L1: trim oversized tool results — always run first.
        let trimmed = self.l1_budget_trim(messages);
        let after_l1 = self.total_tokens(&trimmed);
        let l1_reduced = after_l1 < before;

        if l1_reduced && !self.should_compress(&trimmed) {
            let freed = before.saturating_sub(after_l1);
            return Some((trimmed, CompressMethod::BudgetTrim, freed));
        }

        // Aggregate budget trim: replace the oldest tool results with
        // placeholders when total tool-result tokens exceed the budget.
        let after_aggregate = if let Some(budget) = self.result_token_budget {
            let candidate = if l1_reduced { &trimmed } else { messages };
            let agg_trimmed = self.aggregate_result_trim(candidate, budget);
            if agg_trimmed.len() != candidate.len()
                || self.total_tokens(&agg_trimmed) < self.total_tokens(candidate)
            {
                Some(agg_trimmed)
            } else {
                None
            }
        } else {
            None
        };

        let working = after_aggregate.unwrap_or_else(|| {
            if l1_reduced {
                trimmed
            } else {
                messages.to_vec()
            }
        });

        if !self.should_compress(&working) {
            let after = self.total_tokens(&working);
            if after < before {
                let freed = before.saturating_sub(after);
                return Some((working, CompressMethod::BudgetTrim, freed));
            }
            return None; // No pressure.
        }

        // L2: collapse old messages into a placeholder.
        let collapsed = self.l2_collapse(&working);
        if !self.should_compress(&collapsed) {
            let freed = before.saturating_sub(self.total_tokens(&collapsed));
            return Some((collapsed, CompressMethod::Collapse, freed));
        }

        // L3: LLM summarises the oldest batch (skipped when allow_l3 is false).
        if !self.allow_l3 {
            let freed = before.saturating_sub(self.total_tokens(&collapsed));
            return Some((collapsed, CompressMethod::Collapse, freed));
        }

        match self.l3_summarize(&working, provider, model).await {
            Some(summarised) => {
                let freed = before.saturating_sub(self.total_tokens(&summarised));
                Some((summarised, CompressMethod::Summarize, freed))
            }
            None => {
                // L3 failed (network error, etc.) — fall back to L2.
                let freed = before.saturating_sub(self.total_tokens(&collapsed));
                Some((collapsed, CompressMethod::L3Failed, freed))
            }
        }
    }

    // ── L1: Budget Trim ───────────────────────────────────────────────────────

    fn l1_budget_trim(&self, messages: &[Message]) -> Vec<Message> {
        messages
            .iter()
            .map(|msg| {
                let content = msg.content.iter().map(|b| self.trim_block(b)).collect();
                Message::with_id(msg.id.clone(), msg.role.clone(), content)
            })
            .collect()
    }

    /// Truncate a single ToolResult block if it exceeds the token budget.
    /// Non-ToolResult blocks are returned as-is.
    fn trim_block(&self, block: &ContentBlock) -> ContentBlock {
        let ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } = block
        else {
            return block.clone();
        };
        let tokens = self.estimate_tokens(content);
        if tokens <= self.budget_per_result {
            return block.clone();
        }
        let truncated = self.truncate_to_budget(content, tokens);
        ContentBlock::ToolResult {
            tool_use_id: tool_use_id.clone(),
            content: format!(
                "[Result truncated: {tokens} tokens → {} token limit]\n\n{truncated}",
                self.budget_per_result,
            ),
            is_error: *is_error,
        }
    }

    /// Truncate `content` so that `estimate_tokens(result) <= budget_per_result`.
    ///
    /// Uses the ratio `(budget / current_tokens * len)` as an initial estimate,
    /// then verifies with the estimator. If the estimate overshoots (possible
    /// with non-linear tokenizers), it halves the overshoot iteratively.
    fn truncate_to_budget<'a>(&self, content: &'a str, tokens: usize) -> &'a str {
        // Initial estimate: proportional shrink.
        let mut limit = if tokens > 0 {
            content.len() * self.budget_per_result / tokens
        } else {
            return content;
        };
        limit = limit.min(content.len());

        // Find a valid UTF-8 boundary.
        limit = (0..=limit)
            .rev()
            .find(|&i| content.is_char_boundary(i))
            .unwrap_or(0);

        // Verify the estimate satisfies the budget. If not, shrink iteratively.
        // At most 3 rounds — each halves the overshoot, converging quickly.
        for _ in 0..3 {
            if self.estimate_tokens(&content[..limit]) <= self.budget_per_result {
                break;
            }
            // Shrink by the overshoot ratio.
            let actual = self.estimate_tokens(&content[..limit]);
            let shrink = if actual > 0 {
                limit * self.budget_per_result / actual
            } else {
                break;
            };
            limit = (0..=shrink.min(limit.saturating_sub(1)))
                .rev()
                .find(|&i| content.is_char_boundary(i))
                .unwrap_or(0);
        }

        &content[..limit]
    }

    // ── Aggregate Result Trim ──────────────────────────────────────────────────
    //
    // Enforces a session-level token budget across all tool results.
    // Scans messages in REVERSE order (most recent first) summing tool-result
    // tokens. Once the running total exceeds `budget`, older results are
    // replaced with a short placeholder. This preserves the most recent
    // (and likely most relevant) results while shedding stale ones.

    fn aggregate_result_trim(&self, messages: &[Message], budget: usize) -> Vec<Message> {
        // First pass (reverse): decide which tool-result blocks to keep.
        // Walk newest-first so recent results win the budget.
        let mut remaining = budget;
        let mut keep: Vec<Vec<bool>> = messages
            .iter()
            .map(|m| vec![true; m.content.len()])
            .collect();

        for mi in (0..messages.len()).rev() {
            for bi in (0..messages[mi].content.len()).rev() {
                let ContentBlock::ToolResult { content, .. } = &messages[mi].content[bi] else {
                    continue;
                };
                let tokens = self.estimate_tokens(content);
                if tokens <= remaining {
                    remaining = remaining.saturating_sub(tokens);
                } else {
                    keep[mi][bi] = false;
                }
            }
        }

        // Second pass: rebuild, replacing over-budget results with placeholders.
        messages
            .iter()
            .enumerate()
            .map(|(mi, msg)| {
                let content = msg
                    .content
                    .iter()
                    .enumerate()
                    .map(|(bi, block)| Self::maybe_stub_result(block, keep[mi][bi]))
                    .collect();
                Message::with_id(msg.id.clone(), msg.role.clone(), content)
            })
            .collect()
    }

    /// If `keep` is false and the block is a ToolResult, replace it with a
    /// stub placeholder. Otherwise return the block unchanged.
    fn maybe_stub_result(block: &ContentBlock, keep: bool) -> ContentBlock {
        if keep {
            return block.clone();
        }
        let ContentBlock::ToolResult {
            tool_use_id,
            is_error,
            ..
        } = block
        else {
            return block.clone();
        };
        ContentBlock::ToolResult {
            tool_use_id: tool_use_id.clone(),
            content: "[result trimmed: aggregate budget exceeded]".to_string(),
            is_error: *is_error,
        }
    }

    // ── L2: Collapse ──────────────────────────────────────────────────────────
    //
    // Keep the most recent `keep` messages intact; fold everything older into
    // a Collapsed placeholder. The original messages survive elsewhere —
    // this is a reversible operation (in principle).
    //
    // When tool calls in the folded segment have `summary` set (via
    // `Tool::tool_summary()`), those summaries are included in the placeholder
    // so the LLM retains a trace of what was accomplished, rather than seeing
    // an opaque "N messages folded" notice.

    fn l2_collapse(&self, messages: &[Message]) -> Vec<Message> {
        let keep = self.keep_count(messages.len());

        if messages.len() <= keep {
            return messages.to_vec();
        }

        let (old, recent) = messages.split_at(messages.len() - keep);
        let folded_count = old.len() as u32;

        // Record the ID range so storage-aware applications can re-expand.
        let first_id = old.first().map(|m| m.id.clone());
        let last_id = old.last().map(|m| m.id.clone());

        // Collect tool summaries from the folded segment — gives the LLM a
        // trace of what was accomplished without keeping the full outputs.
        let tool_actions: Vec<String> = old
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|b| {
                if let ContentBlock::ToolUse {
                    name,
                    summary: Some(s),
                    ..
                } = b
                {
                    Some(format!("{name}: {s}"))
                } else {
                    None
                }
            })
            .collect();

        let summary = if tool_actions.is_empty() {
            format!("[{folded_count} earlier messages folded to save context.]")
        } else {
            format!(
                "[{folded_count} earlier messages folded to save context. \
                 Tool calls included: {}]",
                tool_actions.join("; ")
            )
        };

        let placeholder = Message {
            id: uuid::Uuid::new_v4().to_string(),
            role: Role::System,
            content: vec![ContentBlock::Collapsed {
                summary,
                folded_count,
                first_id,
                last_id,
            }],
        };

        let mut result = vec![placeholder];
        result.extend_from_slice(recent);
        result
    }

    // ── L3: Summarize ─────────────────────────────────────────────────────────
    //
    // Ask the LLM to summarise the oldest portion of the history.

    async fn l3_summarize(
        &self,
        messages: &[Message],
        provider: &dyn Provider,
        model: Option<&str>,
    ) -> Option<Vec<Message>> {
        let keep = self.keep_count(messages.len());

        if messages.len() <= keep {
            return None;
        }

        let (old, recent) = messages.split_at(messages.len() - keep);

        // Nothing old to summarise.
        if old.is_empty() {
            return None;
        }

        let old_text: String = old
            .iter()
            .flat_map(|m| m.content.iter())
            .map(block_text)
            .collect::<Vec<_>>()
            .join("\n\n");

        // Nothing to summarise (all old messages were empty).
        if old_text.trim().is_empty() {
            return None;
        }

        let system = match &self.compact_focus {
            Some(focus) => format!(
                "Summarise the key events, decisions, and outcomes from the \
                 conversation fragment below. Be concise but complete. \
                 Preserve tool names, results, and important values. \
                 Pay special attention to: {focus}."
            ),
            None => "Summarise the key events, decisions, and outcomes from the \
                     conversation fragment below. Be concise but complete. \
                     Preserve tool names, results, and important values."
                .to_string(),
        };

        let summary_req = wui_core::provider::ChatRequest {
            model: model.map(str::to_string),
            max_tokens: 1024,
            temperature: Some(0.0),
            system,
            messages: vec![Message::user(old_text)],
            tools: vec![],
            thinking_budget: None,
        };

        let stream = provider.stream(summary_req).await.ok()?;
        futures::pin_mut!(stream);

        use futures::StreamExt;
        use wui_core::event::StreamEvent;

        let mut summary = String::new();
        while let Some(Ok(event)) = stream.next().await {
            if let StreamEvent::TextDelta { text } = event {
                summary.push_str(&text);
            }
        }

        if summary.trim().is_empty() {
            return None;
        }

        let placeholder = Message {
            id: uuid::Uuid::new_v4().to_string(),
            role: Role::System,
            content: vec![ContentBlock::CompactBoundary { summary }],
        };

        let mut result = vec![placeholder];
        result.extend_from_slice(recent);
        Some(result)
    }
}

// ── CompressStrategy impl for CompressPipeline ───────────────────────────────

#[async_trait]
impl CompressStrategy for CompressPipeline {
    async fn compress(
        &self,
        messages: Vec<wui_core::message::Message>,
        provider: Arc<dyn wui_core::provider::Provider>,
        model: Option<&str>,
    ) -> anyhow::Result<CompressResult> {
        match self
            .maybe_compress(&messages, provider.as_ref(), model)
            .await
        {
            Some((msgs, method, freed)) => Ok(CompressResult {
                messages: msgs,
                freed,
                method: Some(method),
            }),
            None => Ok(CompressResult {
                messages,
                freed: 0,
                method: None,
            }),
        }
    }

    fn pressure(&self, messages: &[wui_core::message::Message]) -> f64 {
        self.pressure(messages)
    }

    fn threshold(&self) -> f64 {
        self.compact_threshold
    }

    fn is_critically_full(&self, messages: &[wui_core::message::Message]) -> bool {
        self.is_critically_full(messages)
    }
}

// ── SummarizingCompressor ─────────────────────────────────────────────────────

/// A simple compressor that keeps the most-recent turns verbatim and replaces
/// older turns with a single LLM-generated summary.
///
/// Unlike [`CompressPipeline`]'s multi-tier approach, `SummarizingCompressor`
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

fn block_text(block: &ContentBlock) -> String {
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

#[cfg(test)]
mod tests;
