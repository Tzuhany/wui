// ── CompressPipeline (L1 / L2 / L3) ─────────────────────────────────────────

use std::sync::Arc;

use async_trait::async_trait;

use wui_core::event::CompressMethod;
use wui_core::message::{ContentBlock, Message, Role};
use wui_core::provider::Provider;

use super::estimator::{block_text, CharRatioEstimator, ContextBreakdown, TokenEstimator};
use super::{CompressResult, CompressStrategy};

// ── Config ──────────────────────────────────────────────────────────────────

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

// ── Token Accounting ────────────────────────────────────────────────────────

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
    pub(super) fn should_compress(&self, messages: &[Message]) -> bool {
        self.pressure(messages) >= self.compact_threshold
    }

    /// How many recent messages to keep during L2/L3 compression.
    pub(super) fn keep_count(&self, message_count: usize) -> usize {
        ((message_count as f64 * self.collapse_keep_fraction) as usize).max(self.collapse_keep_min)
    }
}

// ── Orchestrator ────────────────────────────────────────────────────────────

impl CompressPipeline {
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
}

// ── L1 Budget Trim ──────────────────────────────────────────────────────────

impl CompressPipeline {
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

    /// Enforce a session-level token budget across all tool results.
    ///
    /// Scans messages in REVERSE order (most recent first) summing tool-result
    /// tokens. Once the running total exceeds `budget`, older results are
    /// replaced with a short placeholder.
    fn aggregate_result_trim(&self, messages: &[Message], budget: usize) -> Vec<Message> {
        // First pass (reverse): decide which tool-result blocks to keep.
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
                    .map(|(bi, block)| maybe_stub_result(block, keep[mi][bi]))
                    .collect();
                Message::with_id(msg.id.clone(), msg.role.clone(), content)
            })
            .collect()
    }
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

// ── L2 Collapse ─────────────────────────────────────────────────────────────
//
// Keep the most recent `keep` messages intact; fold everything older into
// a Collapsed placeholder. When tool calls in the folded segment have
// `summary` set (via `Tool::tool_summary()`), those summaries are included
// so the LLM retains a trace of what was accomplished.

impl CompressPipeline {
    pub(crate) fn l2_collapse(&self, messages: &[Message]) -> Vec<Message> {
        let keep = self.keep_count(messages.len());

        if messages.len() <= keep {
            return messages.to_vec();
        }

        let (old, recent) = messages.split_at(messages.len() - keep);
        let folded_count = old.len() as u32;

        let first_id = old.first().map(|m| m.id.clone());
        let last_id = old.last().map(|m| m.id.clone());

        // Collect tool summaries from the folded segment.
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
}

// ── L3 Summarize ────────────────────────────────────────────────────────────
//
// Ask the LLM to summarise the oldest portion of the history.

impl CompressPipeline {
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

        if old.is_empty() {
            return None;
        }

        let old_text: String = old
            .iter()
            .flat_map(|m| m.content.iter())
            .map(block_text)
            .collect::<Vec<_>>()
            .join("\n\n");

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
            cache_boundary: None,
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

// ── CompressStrategy impl ───────────────────────────────────────────────────

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

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use wui_core::message::{ContentBlock, Message, Role};

    fn estimator(chars_per_token: usize) -> Arc<dyn TokenEstimator> {
        Arc::new(CharRatioEstimator { chars_per_token })
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn text_msg(role: Role, text: &str) -> Message {
        Message {
            id: uuid::Uuid::new_v4().to_string(),
            role,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
        }
    }

    fn tool_result_msg(content: &str) -> Message {
        Message {
            id: uuid::Uuid::new_v4().to_string(),
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: wui_core::tool::ToolCallId::from("tu_test"),
                content: content.to_string(),
                is_error: false,
            }],
        }
    }

    /// Generate N user messages each containing `chars_each` bytes of text.
    fn msgs(n: usize, chars_each: usize) -> Vec<Message> {
        (0..n)
            .map(|_| text_msg(Role::User, &"x".repeat(chars_each)))
            .collect()
    }

    // ── estimate_tokens ─────────────────────────────────────────────────────

    #[test]
    fn estimate_tokens_basic() {
        let p = CompressPipeline::default(); // chars_per_token_estimate = 4
        assert_eq!(p.estimate_tokens(""), 0);
        assert_eq!(p.estimate_tokens("abcd"), 1);
        assert_eq!(p.estimate_tokens("abcdefgh"), 2);
        assert_eq!(p.estimate_tokens("abc"), 0); // truncating division
    }

    #[test]
    fn estimate_tokens_custom_token_estimator() {
        let p = CompressPipeline {
            token_estimator: estimator(2),
            ..Default::default()
        };
        assert_eq!(p.estimate_tokens("abcd"), 2);
        assert_eq!(p.estimate_tokens("ab"), 1);
    }

    #[test]
    fn estimate_tokens_zero_guard() {
        // chars_per_token = 0 must not panic (guarded by .max(1))
        let p = CompressPipeline {
            token_estimator: estimator(0),
            ..Default::default()
        };
        assert_eq!(p.estimate_tokens("abcd"), 4);
    }

    // ── total_tokens / pressure ─────────────────────────────────────────────

    #[test]
    fn total_tokens_empty() {
        let p = CompressPipeline::default();
        assert_eq!(p.total_tokens(&[]), 0);
    }

    #[test]
    fn pressure_zero_window_guard() {
        // window_tokens = 0 must not panic (guarded by .max(1))
        let p = CompressPipeline {
            window_tokens: 0,
            ..Default::default()
        };
        let messages = msgs(10, 400);
        let pressure = p.pressure(&messages);
        assert!(pressure > 0.0);
    }

    // ── L1: Budget Trim ─────────────────────────────────────────────────────

    #[test]
    fn l1_does_not_trim_small_results() {
        let p = CompressPipeline {
            budget_per_result: 100,
            ..Default::default()
        };
        let small = "x".repeat(100); // 100 chars / 4 = 25 tokens < 100 limit
        let msg = tool_result_msg(&small);
        let out = p.l1_budget_trim(std::slice::from_ref(&msg));
        assert_eq!(out.len(), 1);
        if let ContentBlock::ToolResult { content, .. } = &out[0].content[0] {
            assert_eq!(content, &small, "small result should not be trimmed");
        }
    }

    #[test]
    fn l1_trims_oversized_results() {
        let p = CompressPipeline {
            budget_per_result: 10, // 10 token limit
            token_estimator: estimator(4),
            ..Default::default()
        };
        let huge = "a".repeat(400); // 400 chars / 4 = 100 tokens > 10 limit
        let msg = tool_result_msg(&huge);
        let out = p.l1_budget_trim(&[msg]);
        if let ContentBlock::ToolResult { content, .. } = &out[0].content[0] {
            assert!(
                content.contains("[Result truncated:"),
                "expected truncation notice"
            );
            assert!(content.len() <= 40 + 200, "content too long after trim"); // notice + 40 chars
        } else {
            panic!("expected ToolResult block");
        }
    }

    #[test]
    fn l1_preserves_non_tool_result_blocks() {
        let p = CompressPipeline::default();
        let msg = text_msg(Role::User, "hello");
        let out = p.l1_budget_trim(&[msg]);
        assert_eq!(out.len(), 1);
        if let ContentBlock::Text { text } = &out[0].content[0] {
            assert_eq!(text, "hello");
        } else {
            panic!("expected Text block");
        }
    }

    #[test]
    fn l1_preserves_is_error_flag() {
        let p = CompressPipeline {
            budget_per_result: 1,
            token_estimator: estimator(1),
            ..Default::default()
        };
        let mut msg = tool_result_msg(&"x".repeat(100));
        // Mark as error.
        if let ContentBlock::ToolResult { is_error, .. } = &mut msg.content[0] {
            *is_error = true;
        }
        let out = p.l1_budget_trim(&[msg]);
        if let ContentBlock::ToolResult { is_error, .. } = &out[0].content[0] {
            assert!(*is_error, "is_error flag lost during L1 trim");
        }
    }

    // ── L2: Collapse ────────────────────────────────────────────────────────

    #[test]
    fn l2_no_op_when_few_messages() {
        let p = CompressPipeline {
            collapse_keep_min: 6,
            collapse_keep_fraction: 0.5,
            ..Default::default()
        };
        let messages = msgs(4, 10); // 4 < keep_min=6 -> no-op
        let out = p.l2_collapse(&messages);
        assert_eq!(out.len(), 4, "should not collapse tiny history");
    }

    #[test]
    fn l2_collapses_old_messages() {
        let p = CompressPipeline {
            collapse_keep_min: 2,
            collapse_keep_fraction: 0.5,
            ..Default::default()
        };
        // 10 messages -> keep = max(5, 2) = 5 -> fold 5 -> placeholder + 5 = 6
        let messages = msgs(10, 10);
        let out = p.l2_collapse(&messages);
        assert_eq!(out.len(), 6, "expected placeholder + 5 recent");

        // First message should be the Compressed placeholder.
        if let ContentBlock::Collapsed {
            folded_count,
            summary,
            ..
        } = &out[0].content[0]
        {
            assert_eq!(*folded_count, 5);
            assert!(summary.contains('5'), "summary should mention folded count");
        } else {
            panic!(
                "first message should be Compressed, got {:?}",
                out[0].content[0]
            );
        }
    }

    #[test]
    fn l2_keep_fraction_respected() {
        let p = CompressPipeline {
            collapse_keep_min: 1,
            collapse_keep_fraction: 0.25, // keep 25% of 8 = 2 messages
            ..Default::default()
        };
        let messages = msgs(8, 10);
        let out = p.l2_collapse(&messages);
        // keep = max(2, 1) = 2 -> fold 6 -> placeholder + 2 = 3
        assert_eq!(out.len(), 3, "expected placeholder + 2 recent");
        if let ContentBlock::Collapsed { folded_count, .. } = &out[0].content[0] {
            assert_eq!(*folded_count, 6);
        }
    }

    #[test]
    fn l2_keep_min_floor() {
        let p = CompressPipeline {
            collapse_keep_min: 8,        // floor dominates
            collapse_keep_fraction: 0.1, // would give 1, but floor is 8
            ..Default::default()
        };
        let messages = msgs(10, 10);
        let out = p.l2_collapse(&messages);
        // keep = max(1, 8) = 8 -> fold 2 -> placeholder + 8 = 9
        assert_eq!(out.len(), 9);
        if let ContentBlock::Collapsed { folded_count, .. } = &out[0].content[0] {
            assert_eq!(*folded_count, 2);
        }
    }

    // ── maybe_compress (no provider — only L1/L2 paths) ─────────────────────

    #[tokio::test]
    async fn maybe_compress_returns_none_below_threshold() {
        let p = CompressPipeline {
            window_tokens: 1_000_000, // enormous window
            compact_threshold: 0.80,
            ..Default::default()
        };
        // Tiny messages — way below threshold.
        let messages = msgs(5, 40);
        // Use a null provider — should never be called at L1/L2.
        struct NullProvider;
        #[async_trait::async_trait]
        impl wui_core::provider::Provider for NullProvider {
            async fn stream(
                &self,
                _: wui_core::provider::ChatRequest,
            ) -> Result<
                std::pin::Pin<
                    Box<
                        dyn futures::Stream<
                                Item = Result<
                                    wui_core::event::StreamEvent,
                                    wui_core::provider::ProviderError,
                                >,
                            > + Send,
                    >,
                >,
                wui_core::provider::ProviderError,
            > {
                panic!("provider should not be called")
            }
        }
        let result = p
            .maybe_compress(&messages, &NullProvider, Some("test"))
            .await;
        assert!(result.is_none(), "expected no compression needed");
    }

    #[tokio::test]
    async fn maybe_compress_l1_triggers_on_oversized_result() {
        use wui_core::event::CompressMethod;

        let p = CompressPipeline {
            window_tokens: 1000,
            compact_threshold: 0.05, // very sensitive: any content triggers
            budget_per_result: 10,   // 10 token limit per result
            token_estimator: estimator(4),
            collapse_keep_min: 2,
            collapse_keep_fraction: 0.5,
            compact_focus: None,
            allow_l3: true,
            result_token_budget: None,
        };

        // One huge tool result: 400 chars / 4 = 100 tokens >> budget of 10.
        // After trim, tokens drop well below threshold of 1000 * 0.05 = 50.
        let big = "z".repeat(400);
        let small = "a".repeat(4); // 1 token
        let messages = vec![tool_result_msg(&big), text_msg(Role::User, &small)];

        struct NullProvider;
        #[async_trait::async_trait]
        impl wui_core::provider::Provider for NullProvider {
            async fn stream(
                &self,
                _: wui_core::provider::ChatRequest,
            ) -> Result<
                std::pin::Pin<
                    Box<
                        dyn futures::Stream<
                                Item = Result<
                                    wui_core::event::StreamEvent,
                                    wui_core::provider::ProviderError,
                                >,
                            > + Send,
                    >,
                >,
                wui_core::provider::ProviderError,
            > {
                panic!("provider should not be called for L1")
            }
        }

        let result = p
            .maybe_compress(&messages, &NullProvider, Some("test"))
            .await;
        match result {
            Some((_, method, freed)) => {
                println!("method={method:?} freed={freed}");
                // L1 or L2 should have fired; L3 (provider) should not have.
                assert!(
                    method == CompressMethod::BudgetTrim || method == CompressMethod::Collapse,
                    "unexpected method: {method:?}"
                );
                assert!(freed > 0, "expected freed > 0");
            }
            None => panic!("expected compression to trigger"),
        }
    }

    // ── Mock provider that returns a fixed summary ───────────────────────

    struct MockSummaryProvider {
        reply: String,
    }

    impl MockSummaryProvider {
        fn new(reply: &str) -> Self {
            Self {
                reply: reply.to_string(),
            }
        }
    }

    #[async_trait::async_trait]
    impl wui_core::provider::Provider for MockSummaryProvider {
        async fn stream(
            &self,
            _: wui_core::provider::ChatRequest,
        ) -> Result<
            std::pin::Pin<
                Box<
                    dyn futures::Stream<
                            Item = Result<
                                wui_core::event::StreamEvent,
                                wui_core::provider::ProviderError,
                            >,
                        > + Send,
                >,
            >,
            wui_core::provider::ProviderError,
        > {
            let text = self.reply.clone();
            Ok(Box::pin(futures::stream::once(async move {
                Ok(wui_core::event::StreamEvent::TextDelta { text })
            })))
        }
    }

    // ── L3 Summarize ────────────────────────────────────────────────────

    #[tokio::test]
    async fn l3_summarize_produces_compact_boundary() {
        // To force L3, we need L2 to NOT be enough: set collapse_keep_fraction high
        // so that L2 keeps most messages and pressure stays above threshold.
        //
        // Math (estimator = 1 char = 1 token):
        //   window=200, threshold=0.5 -> trigger above 100 tokens
        //   40 messages x 20 chars = 800 tokens -> pressure 4.0
        //   L2 keeps max(40*0.9, 1) = 36 -> ~720 tokens -> pressure 3.6 > 0.5 -> L3 fires
        //   L3 summarises the 4 oldest messages (4 x 20 = 80 tokens) into "ok" (2 tokens)
        //   Result: 2 + 36*20 = 722 tokens -> freed = 800 - 722 = 78 > 0
        let p = CompressPipeline {
            window_tokens: 200,
            compact_threshold: 0.5,
            token_estimator: estimator(1),
            collapse_keep_min: 1,
            collapse_keep_fraction: 0.9, // keep 90% -> L2 barely helps
            allow_l3: true,
            ..Default::default()
        };

        let messages = msgs(40, 20);
        // Short summary so it's smaller than the messages it replaces.
        let provider = MockSummaryProvider::new("ok");

        let result = p
            .maybe_compress(&messages, &provider, Some("test-model"))
            .await;
        match result {
            Some((out, method, freed)) => {
                assert_eq!(
                    method,
                    wui_core::event::CompressMethod::Summarize,
                    "expected L3 Summarize, got {method:?}"
                );
                assert!(freed > 0, "expected freed > 0");
                // First message should be a CompactBoundary.
                assert!(
                    matches!(&out[0].content[0], ContentBlock::CompactBoundary { summary } if summary == "ok"),
                    "expected CompactBoundary with mock summary, got {:?}",
                    out[0].content[0]
                );
            }
            None => panic!("expected L3 compression to trigger"),
        }
    }

    // ── Aggregate result trim ───────────────────────────────────────────

    #[tokio::test]
    async fn aggregate_result_trim_respects_budget() {
        let p = CompressPipeline {
            window_tokens: 10_000,
            compact_threshold: 0.5,
            budget_per_result: 10_000,     // per-result trim won't fire
            token_estimator: estimator(1), // 1 char = 1 token
            result_token_budget: Some(20), // aggregate budget: 20 tokens total
            collapse_keep_min: 1,
            collapse_keep_fraction: 0.5,
            ..Default::default()
        };

        // 5 tool results, each 10 tokens = 50 total >> budget of 20.
        // Most recent results should be kept, oldest replaced.
        let messages: Vec<Message> = (0..5)
            .map(|i| {
                Message::with_id(
                    format!("msg_{i}"),
                    Role::User,
                    vec![ContentBlock::ToolResult {
                        tool_use_id: wui_core::tool::ToolCallId::from(format!("tu_{i}").as_str()),
                        content: "x".repeat(10),
                        is_error: false,
                    }],
                )
            })
            .collect();

        let trimmed = p.aggregate_result_trim(&messages, 20);
        assert_eq!(trimmed.len(), 5, "message count should not change");

        // Count how many got replaced with placeholder.
        let stubbed: Vec<_> = trimmed
            .iter()
            .filter(|m| {
                matches!(&m.content[0], ContentBlock::ToolResult { content, .. }
                    if content.contains("[result trimmed"))
            })
            .collect();
        assert!(
            !stubbed.is_empty(),
            "expected some results to be trimmed by aggregate budget"
        );

        // The most recent result(s) should be preserved (not stubbed).
        if let ContentBlock::ToolResult { content, .. } = &trimmed[4].content[0] {
            assert!(
                !content.contains("[result trimmed"),
                "most recent result should be preserved"
            );
        }
    }

    // ── SummarizingCompressor below threshold ───────────────────────────

    #[tokio::test]
    async fn summarizing_compressor_below_threshold_no_op() {
        use super::super::summarizer::SummarizingCompressor;
        use super::super::CompressStrategy;

        let compressor = SummarizingCompressor {
            threshold: 0.75,
            window_tokens: 10_000,
            ..Default::default()
        };

        // Small messages: 5 x 40 chars / 4 chars_per_token = 50 tokens
        // Pressure = 50 / 10_000 = 0.005 << 0.75 threshold
        let messages = msgs(5, 40);

        let provider: Arc<dyn wui_core::provider::Provider> =
            Arc::new(MockSummaryProvider::new("should not be called"));
        let result = compressor
            .compress(messages.clone(), provider, Some("test"))
            .await
            .unwrap();

        assert!(
            result.method.is_none(),
            "expected no compression below threshold, got {:?}",
            result.method
        );
        assert_eq!(result.freed, 0);
        assert_eq!(result.messages.len(), messages.len());
    }

    #[tokio::test]
    async fn maybe_compress_l2_when_l1_insufficient() {
        use wui_core::event::CompressMethod;

        // L1 won't help (no tool results) but L2 will (large text messages).
        //
        // Math:
        //   window=200, threshold=0.5 -> trigger above 100 tokens
        //   20 msgs x 40 chars / 4 chars-per-token = 200 tokens -> pressure 1.0 -> compress
        //   L2 keep = max(20 * 0.2, 1) = 4 msgs -> 4 x 10 tokens = 40 tokens
        //   pressure after = 40/200 = 0.2 < 0.5 -> L2 sufficient, no L3 call
        let p = CompressPipeline {
            window_tokens: 200,
            compact_threshold: 0.5,
            budget_per_result: 10_000,
            token_estimator: estimator(4),
            collapse_keep_min: 1,
            collapse_keep_fraction: 0.2,
            compact_focus: None,
            allow_l3: true,
            result_token_budget: None,
        };

        // 20 messages x 40 chars = 800 chars / 4 = 200 tokens.
        let messages = msgs(20, 40);

        struct NullProvider;
        #[async_trait::async_trait]
        impl wui_core::provider::Provider for NullProvider {
            async fn stream(
                &self,
                _: wui_core::provider::ChatRequest,
            ) -> Result<
                std::pin::Pin<
                    Box<
                        dyn futures::Stream<
                                Item = Result<
                                    wui_core::event::StreamEvent,
                                    wui_core::provider::ProviderError,
                                >,
                            > + Send,
                    >,
                >,
                wui_core::provider::ProviderError,
            > {
                panic!("provider should not be called for L2")
            }
        }

        let result = p
            .maybe_compress(&messages, &NullProvider, Some("test"))
            .await;
        match result {
            Some((out, CompressMethod::Collapse, freed)) => {
                println!(
                    "L2 collapse: {} -> {} msgs, freed {freed} tokens",
                    messages.len(),
                    out.len()
                );
                assert!(out.len() < messages.len(), "L2 should reduce message count");
                assert!(freed > 0);
                // First message should be the placeholder.
                assert!(matches!(out[0].content[0], ContentBlock::Collapsed { .. }));
            }
            Some((_, method, _)) => panic!("expected Collapse, got {method:?}"),
            None => panic!("expected L2 compression"),
        }
    }
}
