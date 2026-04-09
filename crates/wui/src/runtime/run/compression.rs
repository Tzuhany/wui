// ── Compression integration ─────────────────────────────────────────────────

use tokio::sync::mpsc;

use wui_core::event::AgentEvent;
use wui_core::hook::HookDecision;
use wui_core::message::Message;

use crate::compress::CompressResult;

use super::history::system_reminder_msg;
use super::RunConfig;

/// Run context compression if pressure exceeds the threshold.
///
/// Fires the PreCompact hook first (may inject preservation context),
/// then runs the compression pipeline. Mutates `messages` in place.
pub(super) async fn maybe_compress(
    config: &RunConfig,
    messages: &mut Vec<Message>,
    tx: &mpsc::Sender<AgentEvent>,
) {
    if config.compress.pressure(messages) >= config.compress.threshold() {
        if let HookDecision::Block { reason } = config.hooks.pre_compact(messages).await {
            tracing::debug!("pre_compact hook injecting preservation context");
            messages.push(system_reminder_msg(&reason));
        }
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
            tracing::debug!(?method, freed, %pressure_before, %pressure_after, "context compressed");
            *messages = new_msgs;
            if method == wui_core::event::CompressMethod::L3Failed {
                tx.send(AgentEvent::CompressFallback { freed }).await.ok();
            }
            tx.send(AgentEvent::Compressed {
                method,
                freed,
                pressure_before,
                pressure_after,
            })
            .await
            .ok();
        }
        Ok(_) => {}
        Err(e) => {
            tracing::warn!(error = %e, "compression failed, continuing without");
        }
    }
}

/// Attempt emergency compression after a prompt-too-long rejection.
///
/// Returns `true` if compression succeeded and the caller should retry.
pub(super) async fn emergency_compress(
    config: &RunConfig,
    messages: &mut Vec<Message>,
    tx: &mpsc::Sender<AgentEvent>,
) -> bool {
    tracing::warn!(
        messages_before = messages.len(),
        "wui.compress.emergency — provider rejected prompt as too long"
    );
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
            true
        }
        _ => {
            tracing::error!("emergency compression failed or had no effect");
            false
        }
    }
}
