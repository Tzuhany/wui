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

use wuhu_core::event::CompressMethod;
use wuhu_core::message::{ContentBlock, Message, Role};
use wuhu_core::provider::Provider;

/// Configuration for the compression pipeline.
///
/// All fields are public — adjust any threshold without a builder API.
/// `CompressPipeline::default()` is a good starting point for most agents.
#[derive(Debug, Clone)]
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

    /// How many characters approximate one token (`chars / N ≈ tokens`).
    ///
    /// `4` is a reasonable default for English prose. Code and JSON are
    /// denser — use `2` if your agent works heavily with structured data.
    pub chars_per_token: usize,

    /// Fraction of messages to keep during L2 collapse.
    ///
    /// `0.5` means "keep the most recent half of the conversation intact
    /// and fold everything older into a placeholder". Range: `(0.0, 1.0)`.
    pub collapse_keep_fraction: f64,

    /// Minimum number of messages to keep during L2 collapse, regardless
    /// of `collapse_keep_fraction`. Prevents collapsing tiny histories.
    pub collapse_keep_min: usize,
}

impl Default for CompressPipeline {
    fn default() -> Self {
        Self {
            window_tokens:         200_000,
            compact_threshold:     0.80,
            budget_per_result:     10_000,
            chars_per_token:       4,
            collapse_keep_fraction: 0.5,
            collapse_keep_min:     4,
        }
    }
}

impl CompressPipeline {
    /// Estimate token count from character count.
    fn estimate_tokens(&self, text: &str) -> usize {
        text.len() / self.chars_per_token.max(1)
    }

    fn total_tokens(&self, messages: &[Message]) -> usize {
        messages.iter()
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

    /// How many recent messages to keep during L2/L3 compression.
    fn keep_count(&self, message_count: usize) -> usize {
        ((message_count as f64 * self.collapse_keep_fraction) as usize)
            .max(self.collapse_keep_min)
    }

    /// Run the pipeline. Returns `None` if no compression was needed.
    pub async fn maybe_compress(
        &self,
        messages: &[Message],
        provider: &dyn Provider,
        model:    &str,
    ) -> Option<(Vec<Message>, CompressMethod, usize)> {
        let before = self.total_tokens(messages);

        // L1: trim oversized tool results — always run first.
        let trimmed    = self.l1_budget_trim(messages);
        let after_l1   = self.total_tokens(&trimmed);
        let l1_reduced = after_l1 < before;

        if l1_reduced && self.pressure(&trimmed) < self.compact_threshold {
            let freed = before.saturating_sub(after_l1);
            return Some((trimmed, CompressMethod::BudgetTrim, freed));
        }

        let working = if l1_reduced { trimmed } else { messages.to_vec() };

        if self.pressure(&working) < self.compact_threshold {
            return None; // No pressure.
        }

        // L2: collapse old messages into a placeholder.
        let collapsed = self.l2_collapse(&working);
        if self.pressure(&collapsed) < self.compact_threshold {
            let freed = before.saturating_sub(self.total_tokens(&collapsed));
            return Some((collapsed, CompressMethod::Collapse, freed));
        }

        // L3: LLM summarises the oldest batch.
        match self.l3_summarize(&working, provider, model).await {
            Some(summarised) => {
                let freed = before.saturating_sub(self.total_tokens(&summarised));
                Some((summarised, CompressMethod::Summarize, freed))
            }
            None => {
                // L3 failed (network error, etc.) — fall back to L2.
                let freed = before.saturating_sub(self.total_tokens(&collapsed));
                Some((collapsed, CompressMethod::Collapse, freed))
            }
        }
    }

    // ── L1: Budget Trim ───────────────────────────────────────────────────────

    fn l1_budget_trim(&self, messages: &[Message]) -> Vec<Message> {
        messages.iter().map(|msg| {
            let content = msg.content.iter().map(|block| {
                if let ContentBlock::ToolResult { tool_use_id, content, is_error } = block {
                    let tokens = self.estimate_tokens(content);
                    if tokens > self.budget_per_result {
                        let limit  = self.budget_per_result * self.chars_per_token;
                        let notice = format!(
                            "[Result truncated: {} tokens → {} token limit]\n\n{}",
                            tokens, self.budget_per_result,
                            &content[..limit.min(content.len())],
                        );
                        return ContentBlock::ToolResult {
                            tool_use_id: tool_use_id.clone(),
                            content:     notice,
                            is_error:    *is_error,
                        };
                    }
                }
                block.clone()
            }).collect();
            Message { id: msg.id.clone(), role: msg.role.clone(), content }
        }).collect()
    }

    // ── L2: Collapse ──────────────────────────────────────────────────────────
    //
    // Keep the most recent `keep` messages intact; fold everything older into
    // a Collapsed placeholder. The original messages survive elsewhere —
    // this is a reversible operation (in principle).

    fn l2_collapse(&self, messages: &[Message]) -> Vec<Message> {
        let keep = self.keep_count(messages.len());

        if messages.len() <= keep {
            return messages.to_vec();
        }

        let (old, recent) = messages.split_at(messages.len() - keep);
        let folded_count  = old.len() as u32;

        // Record the ID range so storage-aware applications can re-expand.
        let first_id = old.first().map(|m| m.id.clone());
        let last_id  = old.last().map(|m| m.id.clone());

        let placeholder = Message {
            id:      uuid::Uuid::new_v4().to_string(),
            role:    Role::System,
            content: vec![ContentBlock::Collapsed {
                summary: format!(
                    "[{folded_count} earlier messages folded to save context.]"
                ),
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
        model:    &str,
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

        let old_text: String = old.iter()
            .flat_map(|m| m.content.iter())
            .map(|b| block_text(b))
            .collect::<Vec<_>>()
            .join("\n\n");

        // Nothing to summarise (all old messages were empty).
        if old_text.trim().is_empty() {
            return None;
        }

        let summary_req = wuhu_core::provider::ChatRequest {
            model:       model.to_string(),
            max_tokens:  1024,
            temperature: Some(0.0),
            system:      "Summarise the key events, decisions, and outcomes from the \
                          conversation fragment below. Be concise but complete. \
                          Preserve tool names, results, and important values."
                          .to_string(),
            messages:    vec![Message::user(old_text)],
            tools:       vec![],
            extensions:  Default::default(),
        };

        let stream = provider.stream(summary_req).await.ok()?;
        futures::pin_mut!(stream);

        use futures::StreamExt;
        use wuhu_core::event::StreamEvent;

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
            id:      uuid::Uuid::new_v4().to_string(),
            role:    Role::System,
            content: vec![ContentBlock::CompactBoundary { summary }],
        };

        let mut result = vec![placeholder];
        result.extend_from_slice(recent);
        Some(result)
    }
}

fn block_text(block: &ContentBlock) -> String {
    match block {
        ContentBlock::Text             { text }          => text.clone(),
        ContentBlock::Thinking         { text }          => text.clone(),
        ContentBlock::ToolUse          { name, input, .. } => format!("[Tool: {name}] {input}"),
        ContentBlock::ToolResult       { content, .. }   => content.clone(),
        ContentBlock::Collapsed        { summary, .. }   => summary.clone(),
        ContentBlock::CompactBoundary  { summary }       => summary.clone(),
    }
}

// =============================================================================
// Unit tests — no LLM call required.
// =============================================================================

#[cfg(test)]
mod tests {
    use wuhu_core::message::{ContentBlock, Message, Role};
    use super::CompressPipeline;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn text_msg(role: Role, text: &str) -> Message {
        Message {
            id:      uuid::Uuid::new_v4().to_string(),
            role,
            content: vec![ContentBlock::Text { text: text.to_string() }],
        }
    }

    fn tool_result_msg(content: &str) -> Message {
        Message {
            id:      uuid::Uuid::new_v4().to_string(),
            role:    Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "tu_test".to_string(),
                content:     content.to_string(),
                is_error:    false,
            }],
        }
    }

    /// Generate N user messages each containing `chars_each` bytes of text.
    fn msgs(n: usize, chars_each: usize) -> Vec<Message> {
        (0..n)
            .map(|_| text_msg(Role::User, &"x".repeat(chars_each)))
            .collect()
    }

    // ── estimate_tokens ───────────────────────────────────────────────────────

    #[test]
    fn estimate_tokens_basic() {
        let p = CompressPipeline::default(); // chars_per_token = 4
        assert_eq!(p.estimate_tokens(""),           0);
        assert_eq!(p.estimate_tokens("abcd"),       1);
        assert_eq!(p.estimate_tokens("abcdefgh"),   2);
        assert_eq!(p.estimate_tokens("abc"),        0); // truncating division
    }

    #[test]
    fn estimate_tokens_custom_chars_per_token() {
        let p = CompressPipeline { chars_per_token: 2, ..Default::default() };
        assert_eq!(p.estimate_tokens("abcd"), 2);
        assert_eq!(p.estimate_tokens("ab"),   1);
    }

    #[test]
    fn estimate_tokens_zero_guard() {
        // chars_per_token = 0 must not panic (guarded by .max(1))
        let p = CompressPipeline { chars_per_token: 0, ..Default::default() };
        assert_eq!(p.estimate_tokens("abcd"), 4);
    }

    // ── total_tokens / pressure ───────────────────────────────────────────────

    #[test]
    fn total_tokens_empty() {
        let p = CompressPipeline::default();
        assert_eq!(p.total_tokens(&[]), 0);
    }

    #[test]
    fn pressure_zero_window_guard() {
        // window_tokens = 0 must not panic (guarded by .max(1))
        let p = CompressPipeline { window_tokens: 0, ..Default::default() };
        let messages = msgs(10, 400);
        let pressure = p.pressure(&messages);
        assert!(pressure > 0.0);
    }

    // ── L1: Budget Trim ───────────────────────────────────────────────────────

    #[test]
    fn l1_does_not_trim_small_results() {
        let p = CompressPipeline { budget_per_result: 100, ..Default::default() };
        let small = "x".repeat(100); // 100 chars / 4 = 25 tokens < 100 limit
        let msg   = tool_result_msg(&small);
        let out   = p.l1_budget_trim(&[msg.clone()]);
        assert_eq!(out.len(), 1);
        if let ContentBlock::ToolResult { content, .. } = &out[0].content[0] {
            assert_eq!(content, &small, "small result should not be trimmed");
        }
    }

    #[test]
    fn l1_trims_oversized_results() {
        let p = CompressPipeline {
            budget_per_result: 10,   // 10 token limit
            chars_per_token:   4,    // → 40 char limit
            ..Default::default()
        };
        let huge = "a".repeat(400); // 400 chars / 4 = 100 tokens > 10 limit
        let msg  = tool_result_msg(&huge);
        let out  = p.l1_budget_trim(&[msg]);
        if let ContentBlock::ToolResult { content, .. } = &out[0].content[0] {
            assert!(content.contains("[Result truncated:"), "expected truncation notice");
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
        let p = CompressPipeline { budget_per_result: 1, chars_per_token: 1, ..Default::default() };
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

    // ── L2: Collapse ──────────────────────────────────────────────────────────

    #[test]
    fn l2_no_op_when_few_messages() {
        let p = CompressPipeline {
            collapse_keep_min:      6,
            collapse_keep_fraction: 0.5,
            ..Default::default()
        };
        let messages = msgs(4, 10); // 4 < keep_min=6 → no-op
        let out = p.l2_collapse(&messages);
        assert_eq!(out.len(), 4, "should not collapse tiny history");
    }

    #[test]
    fn l2_collapses_old_messages() {
        let p = CompressPipeline {
            collapse_keep_min:      2,
            collapse_keep_fraction: 0.5,
            ..Default::default()
        };
        // 10 messages → keep = max(5, 2) = 5 → fold 5 → placeholder + 5 = 6
        let messages = msgs(10, 10);
        let out = p.l2_collapse(&messages);
        assert_eq!(out.len(), 6, "expected placeholder + 5 recent");

        // First message should be the Compressed placeholder.
        if let ContentBlock::Collapsed { folded_count, summary, .. } = &out[0].content[0] {
            assert_eq!(*folded_count, 5);
            assert!(summary.contains('5'), "summary should mention folded count");
        } else {
            panic!("first message should be Compressed, got {:?}", out[0].content[0]);
        }
    }

    #[test]
    fn l2_keep_fraction_respected() {
        let p = CompressPipeline {
            collapse_keep_min:      1,
            collapse_keep_fraction: 0.25, // keep 25% of 8 = 2 messages
            ..Default::default()
        };
        let messages = msgs(8, 10);
        let out = p.l2_collapse(&messages);
        // keep = max(2, 1) = 2 → fold 6 → placeholder + 2 = 3
        assert_eq!(out.len(), 3, "expected placeholder + 2 recent");
        if let ContentBlock::Collapsed { folded_count, .. } = &out[0].content[0] {
            assert_eq!(*folded_count, 6);
        }
    }

    #[test]
    fn l2_keep_min_floor() {
        let p = CompressPipeline {
            collapse_keep_min:      8,   // floor dominates
            collapse_keep_fraction: 0.1, // would give 1, but floor is 8
            ..Default::default()
        };
        let messages = msgs(10, 10);
        let out = p.l2_collapse(&messages);
        // keep = max(1, 8) = 8 → fold 2 → placeholder + 8 = 9
        assert_eq!(out.len(), 9);
        if let ContentBlock::Collapsed { folded_count, .. } = &out[0].content[0] {
            assert_eq!(*folded_count, 2);
        }
    }

    // ── maybe_compress (no provider — only L1/L2 paths) ──────────────────────

    #[tokio::test]
    async fn maybe_compress_returns_none_below_threshold() {
        let p = CompressPipeline {
            window_tokens:     1_000_000, // enormous window
            compact_threshold: 0.80,
            ..Default::default()
        };
        // Tiny messages — way below threshold.
        let messages = msgs(5, 40);
        // Use a null provider — should never be called at L1/L2.
        struct NullProvider;
        #[async_trait::async_trait]
        impl wuhu_core::provider::Provider for NullProvider {
            async fn stream(&self, _: wuhu_core::provider::ChatRequest)
                -> Result<std::pin::Pin<Box<dyn futures::Stream<Item=Result<wuhu_core::event::StreamEvent, wuhu_core::provider::ProviderError>> + Send>>, wuhu_core::provider::ProviderError> {
                panic!("provider should not be called")
            }
        }
        let result = p.maybe_compress(&messages, &NullProvider, "test").await;
        assert!(result.is_none(), "expected no compression needed");
    }

    #[tokio::test]
    async fn maybe_compress_l1_triggers_on_oversized_result() {
        use wuhu_core::event::CompressMethod;

        let p = CompressPipeline {
            window_tokens:     1000,
            compact_threshold: 0.05, // very sensitive: any content triggers
            budget_per_result: 10,   // 10 token limit per result
            chars_per_token:   4,
            collapse_keep_min:      2,
            collapse_keep_fraction: 0.5,
        };

        // One huge tool result: 400 chars / 4 = 100 tokens >> budget of 10.
        // After trim, tokens drop well below threshold of 1000 * 0.05 = 50.
        let big   = "z".repeat(400);
        let small = "a".repeat(4); // 1 token
        let messages = vec![
            tool_result_msg(&big),
            text_msg(Role::User, &small),
        ];

        struct NullProvider;
        #[async_trait::async_trait]
        impl wuhu_core::provider::Provider for NullProvider {
            async fn stream(&self, _: wuhu_core::provider::ChatRequest)
                -> Result<std::pin::Pin<Box<dyn futures::Stream<Item=Result<wuhu_core::event::StreamEvent, wuhu_core::provider::ProviderError>> + Send>>, wuhu_core::provider::ProviderError> {
                panic!("provider should not be called for L1")
            }
        }

        let result = p.maybe_compress(&messages, &NullProvider, "test").await;
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

    #[tokio::test]
    async fn maybe_compress_l2_when_l1_insufficient() {
        use wuhu_core::event::CompressMethod;

        // L1 won't help (no tool results) but L2 will (large text messages).
        //
        // Math:
        //   window=200, threshold=0.5 → trigger above 100 tokens
        //   20 msgs × 40 chars / 4 chars-per-token = 200 tokens → pressure 1.0 → compress
        //   L2 keep = max(20 * 0.2, 1) = 4 msgs → 4 × 10 tokens = 40 tokens
        //   pressure after = 40/200 = 0.2 < 0.5 → L2 sufficient, no L3 call
        let p = CompressPipeline {
            window_tokens:          200,
            compact_threshold:      0.5,
            budget_per_result:      10_000,
            chars_per_token:        4,
            collapse_keep_min:      1,
            collapse_keep_fraction: 0.2,
        };

        // 20 messages × 40 chars = 800 chars / 4 = 200 tokens.
        let messages = msgs(20, 40);

        struct NullProvider;
        #[async_trait::async_trait]
        impl wuhu_core::provider::Provider for NullProvider {
            async fn stream(&self, _: wuhu_core::provider::ChatRequest)
                -> Result<std::pin::Pin<Box<dyn futures::Stream<Item=Result<wuhu_core::event::StreamEvent, wuhu_core::provider::ProviderError>> + Send>>, wuhu_core::provider::ProviderError> {
                panic!("provider should not be called for L2")
            }
        }

        let result = p.maybe_compress(&messages, &NullProvider, "test").await;
        match result {
            Some((out, CompressMethod::Collapse, freed)) => {
                println!("L2 collapse: {} → {} msgs, freed {freed} tokens", messages.len(), out.len());
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
