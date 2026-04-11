// ── Run Stream ────────────────────────────────────────────────────────────────

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tracing::Instrument as _;

use std::sync::Arc;

use wui_core::event::{AgentError, AgentEvent};
use wui_core::message::Message;

use super::run_loop;
use super::RunConfig;

/// Maximum number of events buffered between the loop and the caller.
///
/// If the caller stops consuming, the loop pauses here. Tune this for your
/// use case — larger values trade memory for smoother streaming under jitter.
pub(crate) const EVENT_CHANNEL_CAPACITY: usize = 256;

/// The event stream for a running agent.
///
/// Implements `Stream<Item = AgentEvent>`. Dropping the stream cancels the
/// underlying run immediately — no orphaned tasks, no wasted tokens.
///
/// ```rust,ignore
/// let mut stream = agent.stream("Hello");
///
/// // Cancel explicitly:
/// stream.cancel();
///
/// // Share the cancel signal with another task:
/// let token = stream.cancel_token();
/// tokio::spawn(async move { /* call token.cancel() when needed */ });
/// ```
#[must_use = "RunStream does nothing unless polled; use .next().await or collect the events"]
pub struct RunStream {
    inner: ReceiverStream<AgentEvent>,
    cancel: CancellationToken,
}

impl std::fmt::Debug for RunStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunStream")
            .field("cancelled", &self.cancel.is_cancelled())
            .finish_non_exhaustive()
    }
}

impl RunStream {
    /// Cancel the run immediately.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    /// Clone the `CancellationToken` for this run.
    ///
    /// Use when you need to cancel the run from a separate task or store
    /// the handle for later (e.g., a cancel button in a UI).
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    // ── Convenience helpers ───────────────────────────────────────────────────

    /// Consume the stream and collect the final text response.
    ///
    /// Equivalent to driving the event loop manually and accumulating
    /// `TextDelta` events. Tool events are silently ignored.
    ///
    /// ```rust,ignore
    /// let text = agent.stream("What is 2 + 2?").collect_text().await?;
    /// println!("{text}");
    /// ```
    pub async fn collect_text(mut self) -> Result<String, AgentError> {
        use futures::StreamExt as _;
        let mut text = String::new();
        while let Some(event) = self.next().await {
            match event {
                AgentEvent::TextDelta(t) => text.push_str(&t),
                AgentEvent::Done(_) => break,
                AgentEvent::Error(e) => return Err(e),
                _ => {}
            }
        }
        Ok(text)
    }

    /// Consume the stream, printing text deltas to stdout as they arrive.
    ///
    /// Prints a trailing newline after the run completes. Tool events are
    /// silently ignored.
    ///
    /// ```rust,ignore
    /// agent.stream("Write a haiku about Rust.").print_text().await?;
    /// ```
    pub async fn print_text(mut self) -> Result<(), AgentError> {
        use futures::StreamExt as _;
        while let Some(event) = self.next().await {
            match event {
                AgentEvent::TextDelta(t) => print!("{t}"),
                AgentEvent::Done(_) => {
                    println!();
                    break;
                }
                AgentEvent::Error(e) => return Err(e),
                _ => {}
            }
        }
        Ok(())
    }
}

impl futures::Stream for RunStream {
    type Item = AgentEvent;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl Drop for RunStream {
    fn drop(&mut self) {
        // Cancel the spawned task when the stream is abandoned.
        // Without this, the run loop continues burning tokens even though
        // nobody is consuming its output.
        self.cancel.cancel();
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Run the agent loop and return an event stream.
///
/// Spawns an internal task. The stream yields events until `AgentEvent::Done`
/// or `AgentEvent::Error` is received.
///
/// Dropping the returned `RunStream` cancels the run immediately — safe
/// to drop at any point without leaking the background task. Call
/// `stream.cancel()` or `stream.cancel_token()` for explicit control.
pub(crate) fn run(config: Arc<RunConfig>, messages: Vec<Message>) -> RunStream {
    let cancel = CancellationToken::new();
    let (tx, rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
    tokio::spawn(run_task(config, messages, cancel.clone(), tx));
    RunStream {
        inner: ReceiverStream::new(rx),
        cancel,
    }
}

// ── Internal task ─────────────────────────────────────────────────────────────

async fn run_task(
    config: Arc<RunConfig>,
    messages: Vec<Message>,
    cancel: CancellationToken,
    tx: mpsc::Sender<AgentEvent>,
) {
    let span = tracing::info_span!(
        "wui.run",
        run_id     = %uuid::Uuid::new_v4(),
        model      = %config.model.as_deref().unwrap_or("(provider-default)"),
        max_iter   = config.max_iter,
        tools      = config.tools.len(),
    );
    let result = run_loop(config.clone(), messages, cancel, &tx)
        .instrument(span)
        .await;
    match result {
        Ok(summary) => {
            config.hooks.notify_turn_end(&summary).await;
            let _ = tx.send(AgentEvent::Done(summary)).await;
        }
        Err(e) => {
            let _ = tx.send(AgentEvent::Error(e)).await;
        }
    }
}
