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
// ── Session-store persistence ─────────────────────────────────────────────────
//
// History is committed synchronously on Done. Session-store I/O happens inline
// before `Done` is yielded, so the caller sees a durable turn boundary: both
// in-memory history and the persisted turn state are up to date.
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

use crate::runtime::{run, RunConfig, SessionPermissions, SessionStore, StoredSession};
use wui_core::event::{AgentEvent, RunSummary};
use wui_core::message::Message;

use crate::agent::build_run_config;
use crate::builder::AgentConfig;

// ── SessionHooks ──────────────────────────────────────────────────────────────

/// Session-level lifecycle callbacks.
///
/// All callbacks are optional. Provide only what you need.
///
/// # Example
///
/// ```rust,ignore
/// let agent = Agent::builder(provider)
///     .session_hooks(SessionHooks {
///         on_before_send: Some(Arc::new(|msgs| {
///             println!("sending with {} messages", msgs.len());
///             msgs
///         })),
///         ..Default::default()
///     })
///     .build();
/// ```
#[allow(clippy::type_complexity)]
#[derive(Default, Clone)]
pub struct SessionHooks {
    /// Called with the full message list (history + new user message) before
    /// the run starts. May return a modified list (e.g., inject a preamble).
    pub on_before_send: Option<Arc<dyn Fn(Vec<Message>) -> Vec<Message> + Send + Sync>>,

    /// Called after each completed turn with the turn's `RunSummary`.
    pub on_after_turn: Option<Arc<dyn Fn(&RunSummary) + Send + Sync>>,

    /// Called when the run returns an error, before any retry attempt.
    /// Return `true` to suppress the error and allow retry; `false` to
    /// propagate immediately. `attempt` is 1-based.
    pub on_error: Option<Arc<dyn Fn(&wui_core::event::AgentError, u32) -> bool + Send + Sync>>,
}

/// An active multi-turn conversation.
pub struct Session {
    id: String,
    config: Arc<AgentConfig>,
    messages: Arc<Mutex<Vec<Message>>>,
    session_store: Option<Arc<dyn SessionStore>>,
    session_perms: Arc<SessionPermissions>,
    /// Cancel token for the currently running turn, if any.
    current_cancel: Arc<Mutex<Option<CancellationToken>>>,
}

impl Session {
    pub(crate) async fn new(id: String, config: Arc<AgentConfig>) -> Self {
        // Restore from the session store if available, otherwise start fresh.
        let messages = if let Some(store) = &config.session_store {
            match store.load(&id).await {
                Ok(Some(snapshot)) => snapshot.messages,
                Ok(None) => Vec::new(),
                Err(e) => {
                    tracing::warn!(
                        session_id = %id, error = %e,
                        "session store load failed, starting fresh"
                    );
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        Self {
            id,
            session_store: config.session_store.clone(),
            config,
            messages: Arc::new(Mutex::new(messages)),
            session_perms: Arc::new(SessionPermissions::new()),
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
        // Build the message list for this turn: existing history + the new user
        // message. We do NOT push into `self.messages` here — that is only
        // updated on `AgentEvent::Done` (via `summary.messages`). If the run
        // ends in an error, `self.messages` stays at its pre-send state so the
        // next `send()` does not see a dangling user message without a response.
        let messages = {
            let lock = self.messages.lock().expect("session messages poisoned");
            let mut messages = lock.clone();
            messages.push(Message::user(content.into()));
            messages
        };

        // Apply on_before_send hook if configured.
        let messages = if let Some(ref hooks) = self.config.session_hooks {
            if let Some(ref f) = hooks.on_before_send {
                f(messages)
            } else {
                messages
            }
        } else {
            messages
        };

        let run_config = self.make_run_config();
        let run_stream = run(Arc::new(run_config), messages);

        // Store the cancel token for this turn so cancel_current() can abort it.
        *self.current_cancel.lock().expect("current_cancel poisoned") =
            Some(run_stream.cancel_token());

        // Wire Done back to session state.
        let messages_ref = self.messages.clone();
        let session_store = self.session_store.clone();
        let session_id = self.id.clone();
        let cancel_store = self.current_cancel.clone();
        let session_hooks = self.config.session_hooks.clone();

        // Use `then` (async map) so session-store I/O completes before `Done` is
        // yielded to the caller. This guarantees that by the time the caller
        // sees `Done`, both in-memory history and the persisted session state
        // are up to date.
        //
        // The 5 Arcs are only cloned when the event is `Done` (once per run).
        // All other events (TextDelta, ToolStart, …) pass through without any
        // heap allocation in this closure.
        //
        // Box::pin makes the result Unpin so callers can use `stream.next().await`
        // without pinning manually.
        Box::pin(run_stream.then(move |event| {
            // Capture only for Done events. For everything else the closure body
            // compiles to a no-op branch and the Arcs are never touched.
            let done_state = if matches!(event, AgentEvent::Done(_)) {
                Some((
                    messages_ref.clone(),
                    session_store.clone(),
                    session_id.clone(),
                    cancel_store.clone(),
                    session_hooks.clone(),
                ))
            } else {
                None
            };
            async move {
                if let (AgentEvent::Done(ref summary), Some((messages_ref, session_store, session_id, cancel_store, session_hooks))) =
                    (&event, done_state)
                {
                    let updated = summary.messages.clone();

                    // Update in-memory history synchronously.
                    *messages_ref.lock().expect("session messages poisoned") = updated.clone();

                    // Clear the stored cancel token — the run is done.
                    *cancel_store.lock().expect("current_cancel poisoned") = None;

                    // Call on_after_turn hook if configured.
                    if let Some(ref hooks) = session_hooks {
                        if let Some(ref f) = hooks.on_after_turn {
                            f(summary);
                        }
                    }

                    // Persist session state inline — caller sees Done only after save.
                    if let Some(store) = session_store {
                        let snapshot = StoredSession {
                            session_id: session_id.clone(),
                            messages: updated,
                        };
                        if let Err(e) = store.save(&session_id, &snapshot).await {
                            tracing::warn!(session_id = %session_id, error = %e, "session store save failed");
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
        if let Some(token) = self
            .current_cancel
            .lock()
            .expect("current_cancel poisoned")
            .as_ref()
        {
            token.cancel();
        }
    }

    /// The current message history for this session (read-only snapshot).
    pub fn messages(&self) -> Vec<Message> {
        self.messages
            .lock()
            .expect("session messages poisoned")
            .clone()
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
