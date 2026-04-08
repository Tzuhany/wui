// ── CompressPipeline (L1 / L2 / L3) ─────────────────────────────────────────

use std::sync::Arc;

use async_trait::async_trait;

use wui_core::event::CompressMethod;
use wui_core::message::{ContentBlock, Message, Role};
use wui_core::provider::Provider;

use super::estimator::{block_text, CharRatioEstimator, ContextBreakdown, TokenEstimator};
use super::{CompressResult, CompressStrategy};

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
    pub(crate) fn estimate_tokens(&self, text: &str) -> usize {
        self.token_estimator.estimate(text)
    }

    pub(crate) fn total_tokens(&self, messages: &[Message]) -> usize {
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

    pub(crate) fn l1_budget_trim(&self, messages: &[Message]) -> Vec<Message> {
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

    pub(crate) fn l2_collapse(&self, messages: &[Message]) -> Vec<Message> {
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
