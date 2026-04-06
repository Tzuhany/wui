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
// ============================================================================

use wuhu_core::event::CompressMethod;
use wuhu_core::message::{ContentBlock, Message, Role};
use wuhu_core::provider::Provider;

/// Configuration for the compression pipeline.
#[derive(Debug, Clone)]
pub struct CompressPipeline {
    /// Total context window size in tokens.
    pub window_tokens: usize,

    /// Fraction of `window_tokens` at which compression triggers.
    /// E.g. 0.80 means "compress when > 80% full".
    pub compact_threshold: f64,

    /// Maximum tokens allowed in a single tool result.
    /// Results exceeding this are truncated before any LLM call.
    pub budget_per_result: usize,

    /// How many characters to use as a rough token estimate (chars / N ≈ tokens).
    pub chars_per_token: usize,
}

impl Default for CompressPipeline {
    fn default() -> Self {
        Self {
            window_tokens:     200_000,
            compact_threshold: 0.80,
            budget_per_result: 10_000,
            chars_per_token:   4,
        }
    }
}

impl CompressPipeline {
    /// Estimate token count from character count.
    fn estimate_tokens(&self, text: &str) -> usize {
        text.len() / self.chars_per_token
    }

    fn total_tokens(&self, messages: &[Message]) -> usize {
        messages.iter()
            .flat_map(|m| m.content.iter())
            .map(|b| self.estimate_tokens(&block_text(b)))
            .sum()
    }

    fn pressure(&self, messages: &[Message]) -> f64 {
        self.total_tokens(messages) as f64 / self.window_tokens as f64
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
        let trimmed = self.l1_budget_trim(messages);
        let after_l1 = self.total_tokens(&trimmed);
        if after_l1 < before {
            // Only report if L1 alone was sufficient.
            if self.pressure(&trimmed) < self.compact_threshold {
                let freed = before.saturating_sub(after_l1);
                return Some((trimmed, CompressMethod::BudgetTrim, freed));
            }
        }

        let working = if after_l1 < before { trimmed } else { messages.to_vec() };

        if self.pressure(&working) < self.compact_threshold {
            return None; // No pressure.
        }

        // L2: collapse old messages.
        let collapsed = self.l2_collapse(&working);
        if self.pressure(&collapsed) < self.compact_threshold {
            let freed = before.saturating_sub(self.total_tokens(&collapsed));
            return Some((collapsed, CompressMethod::Collapse, freed));
        }

        // L3: LLM summarises old messages.
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
                            "[Result truncated: {} tokens → {} token limit. Full output saved.]\n\n{}",
                            tokens, self.budget_per_result, &content[..limit.min(content.len())]
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
    // Keep the most recent N messages intact; fold everything older into a
    // Compressed placeholder that records how many messages were folded.

    fn l2_collapse(&self, messages: &[Message]) -> Vec<Message> {
        let keep = (messages.len() / 2).max(4);
        if messages.len() <= keep {
            return messages.to_vec();
        }

        let (old, recent) = messages.split_at(messages.len() - keep);
        let folded_count = old.len();

        let placeholder = Message {
            id:   uuid::Uuid::new_v4().to_string(),
            role: Role::System,
            content: vec![ContentBlock::Compressed {
                summary:      format!("[{folded_count} earlier messages have been condensed to save context.]"),
                folded_count,
            }],
        };

        let mut result = vec![placeholder];
        result.extend_from_slice(recent);
        result
    }

    // ── L3: Summarize ─────────────────────────────────────────────────────────
    // Ask the LLM to summarise the oldest half of the history.

    async fn l3_summarize(
        &self,
        messages: &[Message],
        provider: &dyn Provider,
        model:    &str,
    ) -> Option<Vec<Message>> {
        let keep = (messages.len() / 2).max(4);
        if messages.len() <= keep {
            return None;
        }

        let (old, recent) = messages.split_at(messages.len() - keep);

        let old_text: String = old.iter()
            .flat_map(|m| m.content.iter())
            .map(|b| block_text(b))
            .collect::<Vec<_>>()
            .join("\n\n");

        let summary_req = wuhu_core::provider::ChatRequest {
            model:       model.to_string(),
            max_tokens:  1024,
            temperature: Some(0.0),
            system:      "You are a precise summariser. Summarise the key events, decisions, and outcomes from the conversation fragment below. Be concise but complete. Preserve tool names, results, and any important values.".to_string(),
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

        if summary.is_empty() { return None; }

        let placeholder = Message {
            id:   uuid::Uuid::new_v4().to_string(),
            role: Role::System,
            content: vec![ContentBlock::Compressed {
                summary,
                folded_count: old.len(),
            }],
        };

        let mut result = vec![placeholder];
        result.extend_from_slice(recent);
        Some(result)
    }
}

fn block_text(block: &ContentBlock) -> String {
    match block {
        ContentBlock::Text       { text }          => text.clone(),
        ContentBlock::Thinking   { text }          => text.clone(),
        ContentBlock::ToolUse    { name, input, .. } =>
            format!("[Tool: {name}] {input}"),
        ContentBlock::ToolResult { content, .. }   => content.clone(),
        ContentBlock::Compressed { summary, .. }   => summary.clone(),
    }
}
