// ============================================================================
// Session — multi-turn conversation state.
//
// A Session owns the message history for one ongoing conversation.
// After each turn, the full updated history is stored so the next call
// to `send()` continues from exactly where the last one left off.
//
// ── History synchrony ─────────────────────────────────────────────────────────
//
// `messages` uses `std::sync::Mutex` (not tokio's) so the map() closure that
// fires on `AgentEvent::Done` can update history *synchronously*, without
// spawning a task. By the time `Done` is yielded to the caller, `self.messages`
// already reflects the full turn — the next `send()` immediately sees it.
//
// We never hold this lock across an await point, so std::sync::Mutex is safe.
//
// ── Checkpoint persistence ────────────────────────────────────────────────────
//
// History is committed synchronously on Done. The checkpoint I/O (disk/network)
// is spawned — it's best-effort and doesn't block the event stream. A failed
// checkpoint save is logged and the in-memory history is still correct.
//
// ── Permission memory ─────────────────────────────────────────────────────────
//
// `SessionPermissions` is shared with the run loop. `ApproveAlways` /
// `DenyAlways` responses from ControlHandle are stored here and honoured
// for subsequent tool calls within this session — no re-prompting needed.
//
// ── Cancellation ─────────────────────────────────────────────────────────────
//
// Each `send()` call stores its `CancellationToken` in `current_cancel`.
// Calling `cancel_current()` or dropping the `SessionStream` aborts the run
// immediately. A new `send()` replaces the token for that turn.
// ============================================================================

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use wuhu_core::checkpoint::{Checkpoint, SessionSnapshot};
use wuhu_core::event::AgentEvent;
use wuhu_core::message::Message;
use wuhu_engine::{RunConfig, SessionPermissions, run};

use crate::agent::build_run_config;
use crate::builder::AgentConfig;

/// An active multi-turn conversation.
pub struct Session {
    id:             String,
    config:         Arc<AgentConfig>,
    messages:       Arc<Mutex<Vec<Message>>>,
    checkpoint:     Option<Arc<dyn Checkpoint>>,
    session_perms:  Arc<SessionPermissions>,
    /// Cancel token for the currently running turn, if any.
    current_cancel: Arc<Mutex<Option<CancellationToken>>>,
}

impl Session {
    pub(crate) async fn new(id: String, config: Arc<AgentConfig>) -> Self {
        // Restore from checkpoint if available, otherwise start fresh.
        let messages = if let Some(cp) = &config.checkpoint {
            match cp.load(&id).await {
                Ok(Some(snapshot)) => snapshot.messages,
                Ok(None)           => Vec::new(),
                Err(e) => {
                    tracing::warn!(
                        session_id = %id, error = %e,
                        "checkpoint load failed, starting fresh"
                    );
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        Self {
            id,
            checkpoint: config.checkpoint.clone(),
            config,
            messages:       Arc::new(Mutex::new(messages)),
            session_perms:  Arc::new(SessionPermissions::new()),
            current_cancel: Arc::new(Mutex::new(None)),
        }
    }

    /// Send a message and return a stream of events.
    ///
    /// Consume the stream fully. When `AgentEvent::Done` is yielded,
    /// session history has already been updated in memory — the next `send()`
    /// immediately sees this turn's full context.
    ///
    /// Dropping the returned stream cancels the current run. Use
    /// `session.cancel_current()` to abort the run without dropping the stream.
    pub fn send(
        &self,
        content: impl Into<String>,
    ) -> impl futures::Stream<Item = AgentEvent> + Unpin + '_ {
        let user_msg = Message::user(content.into());
        self.messages.lock()
            .expect("session messages poisoned")
            .push(user_msg);

        let messages   = self.messages.lock()
            .expect("session messages poisoned")
            .clone();
        let run_config = self.make_run_config();
        let run_stream = run(Arc::new(run_config), messages);

        // Store the cancel token for this turn so cancel_current() can abort it.
        *self.current_cancel.lock().expect("current_cancel poisoned")
            = Some(run_stream.cancel_token());

        // Wire Done back to session state.
        let messages_ref  = self.messages.clone();
        let checkpoint    = self.checkpoint.clone();
        let session_id    = self.id.clone();
        let cancel_store  = self.current_cancel.clone();

        // Use `then` (async map) so checkpoint I/O completes before `Done` is
        // yielded to the caller. This guarantees that by the time the caller
        // sees `Done`, both in-memory history and the checkpoint are up to date.
        // Box::pin makes the result Unpin so callers can use `stream.next().await`
        // without pinning manually.
        Box::pin(run_stream.then(move |event| {
            let messages_ref  = messages_ref.clone();
            let checkpoint    = checkpoint.clone();
            let session_id    = session_id.clone();
            let cancel_store  = cancel_store.clone();
            async move {
                if let AgentEvent::Done(ref summary) = event {
                    let updated = summary.messages.clone();

                    // Update in-memory history synchronously.
                    *messages_ref.lock().expect("session messages poisoned") = updated.clone();

                    // Clear the stored cancel token — the run is done.
                    *cancel_store.lock().expect("current_cancel poisoned") = None;

                    // Persist checkpoint inline — caller sees Done only after save.
                    if let Some(cp) = checkpoint {
                        let sid      = session_id.clone();
                        let snapshot = SessionSnapshot {
                            session_id: sid.clone(),
                            messages:   updated,
                        };
                        if let Err(e) = cp.save(&sid, &snapshot).await {
                            tracing::warn!(session_id = %sid, error = %e, "checkpoint save failed");
                        }
                    }
                }
                event
            }
        }))
    }

    /// Cancel the currently running turn, if any.
    ///
    /// The run loop will emit `AgentEvent::Done` with `stop_reason: Cancelled`
    /// and exit. Has no effect if no turn is running.
    pub fn cancel_current(&self) {
        if let Some(token) = self.current_cancel
            .lock().expect("current_cancel poisoned")
            .as_ref()
        {
            token.cancel();
        }
    }

    /// The current message history for this session (read-only snapshot).
    pub fn messages(&self) -> Vec<Message> {
        self.messages.lock().expect("session messages poisoned").clone()
    }

    /// The session's permission memory.
    ///
    /// Inspect or programmatically update which tools are always-allowed or
    /// always-denied without going through the HITL flow.
    pub fn permissions(&self) -> Arc<SessionPermissions> {
        self.session_perms.clone()
    }

    fn make_run_config(&self) -> RunConfig {
        build_run_config(&self.config, self.session_perms.clone())
    }
}

