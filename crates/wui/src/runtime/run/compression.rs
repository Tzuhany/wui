// ── Compression integration ─────────────────────────────────────────────────

use tokio::sync::mpsc;

use wui_core::event::AgentEvent;
use wui_core::hook::HookDecision;
use wui_core::message::Message;

use crate::compress::CompressResult;

use super::history::system_reminder_msg;
use super::state::RecoveryState;
use super::RunConfig;

/// Run context compression if pressure exceeds the threshold.
///
/// Fires the PreCompact hook first (may inject preservation context),
/// then runs the compression pipeline. Mutates `messages` in place.
///
/// Respects the compression circuit breaker: after `MAX_COMPRESS_FAILURES`
/// consecutive failures, compression is skipped entirely. A successful
/// compression resets the counter.
pub(super) async fn maybe_compress(
    config: &RunConfig,
    messages: &mut Vec<Message>,
    recovery: &mut RecoveryState,
    tx: &mpsc::Sender<AgentEvent>,
) {
    if config.compress.pressure(messages) < config.compress.threshold() {
        return;
    }

    // Circuit breaker: stop retrying after consecutive failures.
    if recovery.compress_failures >= 3 {
        tracing::warn!(
            failures = recovery.compress_failures,
            "compression circuit breaker open — skipping compression"
        );
        return;
    }

    if let HookDecision::Block { reason } = config.hooks.pre_compact(messages).await {
        tracing::debug!("pre_compact hook injecting preservation context");
        messages.push(system_reminder_msg(&reason));
    }

    if config.compress.pressure(messages) < config.compress.threshold() {
        return;
    }

    let messages_before = messages.len();
    let pressure_before = config.compress.pressure(messages);
    let result = config
        .compress
        .compress(
            messages.clone(),
            config.provider.clone(),
            config.model.as_deref(),
        )
        .await;
    match result {
        Ok(CompressResult {
            method: Some(method),
            freed,
            messages: new_msgs,
        }) => {
            let pressure_after = config.compress.pressure(&new_msgs);
            tracing::info!(
                method = ?method,
                messages_before,
                messages_after = new_msgs.len(),
                tokens_freed = freed,
                %pressure_before,
                %pressure_after,
                "wui.compress"
            );
            *messages = new_msgs;
            tx.send(AgentEvent::Compressed {
                method,
                freed,
                pressure_before,
                pressure_after,
            })
            .await
            .ok();

            // Compression succeeded — reset circuit breaker.
            recovery.on_compress_success();

            // Post-compact hook: re-inject critical context lost in compression.
            if let HookDecision::Block { reason } = config.hooks.post_compact(messages, freed).await
            {
                tracing::debug!("post_compact hook re-injecting context after compression");
                messages.push(system_reminder_msg(&reason));
            }
        }
        Ok(_) => {
            // No compression applied (e.g. below per-result budgets).
            recovery.on_compress_failure();
        }
        Err(e) => {
            tracing::warn!(error = %e, "compression failed");
            recovery.on_compress_failure();
        }
    }
}

/// Attempt emergency compression after a prompt-too-long rejection.
///
/// When `token_gap` is `Some`, the function first tries a precise drop:
/// oldest messages are removed until the estimated token total is reduced
/// by at least that amount. This avoids over-compressing when the exact
/// overage is known from the API error.
///
/// Falls back to the full compression pipeline if precise dropping is
/// insufficient or `token_gap` is `None`.
///
/// Returns `true` if compression succeeded and the caller should retry.
pub(super) async fn emergency_compress(
    config: &RunConfig,
    messages: &mut Vec<Message>,
    recovery: &mut RecoveryState,
    token_gap: Option<usize>,
    tx: &mpsc::Sender<AgentEvent>,
) -> bool {
    tracing::warn!(
        messages_before = messages.len(),
        ?token_gap,
        "wui.compress.emergency — provider rejected prompt as too long"
    );

    // Precise drop: if we know the exact token overage, remove oldest
    // messages until we've freed enough. Keep at least 2 messages.
    if let Some(gap) = token_gap {
        let pressure_before = config.compress.pressure(messages);
        let mut freed = 0usize;
        while messages.len() > 2 {
            let oldest = &messages[0];
            let est: usize = oldest
                .content
                .iter()
                .map(|b| match b {
                    wui_core::message::ContentBlock::Text { text } => text.len() / 4,
                    wui_core::message::ContentBlock::ToolResult { content, .. } => {
                        content.len() / 4
                    }
                    _ => 50,
                })
                .sum();
            messages.remove(0);
            freed += est;
            if freed >= gap {
                break;
            }
        }
        let pressure_after = config.compress.pressure(messages);
        if freed > 0 {
            tracing::info!(
                freed_tokens_est = freed,
                token_gap = gap,
                messages_dropped = "precise",
                %pressure_before,
                %pressure_after,
                "emergency precise drop succeeded"
            );
            tx.send(AgentEvent::Compressed {
                method: wui_core::event::CompressMethod::BudgetTrim,
                freed,
                pressure_before,
                pressure_after,
            })
            .await
            .ok();
            recovery.on_compress_success();
            return true;
        }
    }

    // Fallback: full compression pipeline.
    let pressure_before = config.compress.pressure(messages);
    let result = config
        .compress
        .compress(
            messages.clone(),
            config.provider.clone(),
            config.model.as_deref(),
        )
        .await;
    match result {
        Ok(CompressResult {
            method: Some(method),
            freed,
            messages: new_msgs,
        }) => {
            let pressure_after = config.compress.pressure(&new_msgs);
            tracing::info!(?method, freed, %pressure_before, %pressure_after, "emergency compression succeeded");
            *messages = new_msgs;
            tx.send(AgentEvent::Compressed {
                method,
                freed,
                pressure_before,
                pressure_after,
            })
            .await
            .ok();
            recovery.on_compress_success();
            true
        }
        _ => {
            tracing::error!("emergency compression failed or had no effect");
            recovery.on_compress_failure();
            false
        }
    }
}
