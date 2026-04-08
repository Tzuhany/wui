use std::sync::Arc;

use super::{CharRatioEstimator, CompressPipeline, TokenEstimator};
use wui_core::message::{ContentBlock, Message, Role};

fn estimator(chars_per_token: usize) -> Arc<dyn TokenEstimator> {
    Arc::new(CharRatioEstimator { chars_per_token })
}

// ── Helpers ───────────────────────────────────────────────────────────────

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
            tool_use_id: wui_core::types::ToolCallId::from("tu_test"),
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

// ── estimate_tokens ───────────────────────────────────────────────────────

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

// ── total_tokens / pressure ───────────────────────────────────────────────

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

// ── L1: Budget Trim ───────────────────────────────────────────────────────

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

// ── L2: Collapse ──────────────────────────────────────────────────────────

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

// ── maybe_compress (no provider — only L1/L2 paths) ──────────────────────

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
