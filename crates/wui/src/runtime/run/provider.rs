// ── Retry Policy & Provider Call ─────────────────────────────────────────────

use std::time::Duration;

use tokio::sync::mpsc;

use wui_core::event::{AgentError, AgentEvent};
use wui_core::provider::{ChatRequest, ProviderError};

use super::RunConfig;

/// Convenience alias for the pinned stream returned by `Provider::stream`.
pub(crate) type ProviderStream = std::pin::Pin<
    Box<dyn futures::Stream<Item = Result<wui_core::event::StreamEvent, ProviderError>> + Send>,
>;

/// Exponential back-off for transient provider errors.
///
/// Applied when the provider returns `is_retryable: true` — network hiccups,
/// rate-limit 429s, intermittent 5xx. Non-retryable errors bypass this entirely.
///
/// The default (`RetryPolicy::default()`) retries up to 3 times with 500 ms
/// initial delay doubling on each attempt, capped at 10 s, with jitter enabled.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts. `0` disables retrying.
    pub max_attempts: u32,
    /// Wait before the first retry (milliseconds).
    pub initial_delay_ms: u64,
    /// Multiplier applied to the delay after each failure (exponential back-off).
    /// `2.0` doubles the delay each time: 500 ms → 1 s → 2 s → …, capped at `max_delay_ms`.
    pub multiplier: f64,
    /// Hard cap on the delay between retries (milliseconds).
    pub max_delay_ms: u64,
    /// Add equal jitter to each back-off delay.
    ///
    /// Uses the equal-jitter formula: `delay = exp/2 + rand(0, exp/2)`.
    /// This guarantees at least half the computed delay (no starvation) while
    /// spreading load across the full range (no thundering herd). The jitter
    /// range grows with the exponential delay, so later retries are spread
    /// proportionally further apart than earlier ones.
    /// Enabled by default.
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 500,
            multiplier: 2.0,
            max_delay_ms: 10_000,
            jitter: true,
        }
    }
}

impl RetryPolicy {
    /// Compute the back-off delay for a given attempt (1-indexed).
    ///
    /// Uses `retry_after_ms` from the provider when available (e.g. a 429
    /// `Retry-After` header). Falls back to exponential back-off + optional jitter.
    pub(super) fn delay_ms(&self, attempt: u32, retry_after_ms: Option<u64>) -> u64 {
        if let Some(ms) = retry_after_ms {
            return ms;
        }
        let exp = (self.initial_delay_ms as f64 * self.multiplier.powi((attempt - 1) as i32))
            .min(self.max_delay_ms as f64) as u64;
        if self.jitter {
            use rand::Rng;
            // Equal jitter: half the delay is guaranteed (prevents starvation),
            // the other half is random (prevents thundering herd). Both halves
            // scale with `exp`, so spread grows proportionally across retries.
            let half = exp / 2;
            half + rand::thread_rng().gen_range(0..=half)
        } else {
            exp
        }
    }
}

/// Call the provider, retrying transient errors with exponential back-off.
pub(super) async fn call_with_retry(
    config: &RunConfig,
    req: &ChatRequest,
    tx: &mpsc::Sender<AgentEvent>,
) -> Result<ProviderStream, AgentError> {
    let mut attempt = 0u32;
    loop {
        match config.provider.stream(req.clone()).await {
            Ok(stream) => return Ok(stream),

            Err(e) if e.is_retryable() && attempt < config.retry.max_attempts => {
                attempt += 1;

                // Use the provider's Retry-After hint for rate-limit errors;
                // fall back to the policy's formula for everything else.
                let retry_after_ms = if let ProviderError::RateLimit { retry_after_ms } = &e {
                    Some(*retry_after_ms)
                } else {
                    None
                };
                let delay_ms = config.retry.delay_ms(attempt, retry_after_ms);

                tracing::warn!(
                    attempt, max = config.retry.max_attempts,
                    error = %e, delay_ms, "provider error — retrying"
                );
                tx.send(AgentEvent::Retrying {
                    attempt,
                    delay_ms,
                    reason: e.to_string(),
                })
                .await
                .ok();
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }

            Err(e) => return Err(AgentError::fatal(e.to_string())),
        }
    }
}

/// Detect whether an `AgentError` indicates a prompt-too-long rejection.
pub(super) fn is_prompt_too_long(e: &AgentError) -> bool {
    let msg = e.message.to_lowercase();
    msg.contains("prompt is too long")
        || msg.contains("too long")
        || msg.contains("maximum context length")
        || msg.contains("context_length_exceeded")
}
