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
// ── Turn serialisation ───────────────────────────────────────────────────────
//
// `send()` is async and acquires a `Semaphore(1)` permit before starting.
// If another turn is already in progress, the second `send()` awaits until
// the first completes (or is dropped). This prevents concurrent turns from
// forking the same history snapshot and silently overwriting each other.
//
// ── Cancellation ─────────────────────────────────────────────────────────────
//
// Each `send()` call stores its `CancellationToken` in `current_cancel`.
// Calling `cancel_current()` or dropping the `SessionStream` aborts the run
// immediately. A new `send()` replaces the token for that turn.
// ============================================================================

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::runtime::{run, HookRunner, RunConfig, SessionPermissions, SessionStore, StoredSession};
use wui_core::event::{AgentEvent, RunSummary};
use wui_core::message::Message;
use wui_core::types::SessionId;

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
    id: SessionId,
    config: Arc<AgentConfig>,
    messages: Arc<Mutex<Vec<Message>>>,
    session_store: Option<Arc<dyn SessionStore>>,
    session_perms: Arc<SessionPermissions>,
    /// Cancel token for the currently running turn, if any.
    current_cancel: Arc<Mutex<Option<CancellationToken>>>,
    /// Ensures only one turn runs at a time. Without this, two concurrent
    /// `send()` calls would clone the same history, diverge, and the last
    /// to finish would silently overwrite the other's results.
    turn_guard: Arc<Semaphore>,
}

impl Session {
    pub(crate) async fn new(id: impl Into<String>, config: Arc<AgentConfig>) -> Self {
        let id = SessionId::from(id.into());
        // Restore from the session store if available, otherwise start fresh.
        let messages = if let Some(store) = &config.session_store {
            match store.load(id.as_str()).await {
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

        // Fire SessionStart notification before returning.
        let hook_runner = HookRunner::new(config.hooks.clone());
        hook_runner.notify_session_start(&id).await;

        Self {
            id,
            session_store: config.session_store.clone(),
            config,
            messages: Arc::new(Mutex::new(messages)),
            session_perms: Arc::new(SessionPermissions::new()),
            current_cancel: Arc::new(Mutex::new(None)),
            turn_guard: Arc::new(Semaphore::new(1)),
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
    pub async fn send(
        &self,
        content: impl Into<String>,
    ) -> impl futures::Stream<Item = AgentEvent> + Unpin + '_ {
        // Acquire the turn guard — ensures only one turn runs at a time.
        // If another turn is in progress, this awaits until it completes.
        // The permit is moved into the stream's `then` closure and held for
        // its lifetime; it is released when the stream is dropped or Done fires.
        let permit = self
            .turn_guard
            .clone()
            .acquire_owned()
            .await
            .expect("turn_guard semaphore closed");

        // Build the message list for this turn: existing history + the new user
        // message. We do NOT push into `self.messages` here — that is only
        // updated on `AgentEvent::Done` (via `summary.messages`). If the run
        // ends in an error, `self.messages` stays at its pre-send state so the
        // next `send()` does not see a dangling user message without a response.
        let messages = {
            let lock = self.messages.lock().unwrap_or_else(|e| {
                tracing::error!("session messages mutex poisoned, recovering");
                e.into_inner()
            });
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
        *self.current_cancel.lock().unwrap_or_else(|e| {
            tracing::error!("current_cancel mutex poisoned, recovering");
            e.into_inner()
        }) = Some(run_stream.cancel_token());

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
        // The semaphore permit is wrapped in Arc<Mutex<Option<_>>> so it can be
        // explicitly released when Done fires. This ensures the next send() can
        // proceed even if the caller keeps the stream alive after consuming Done.
        // If the stream is dropped without reaching Done, the permit is released
        // via the Arc's Drop — the closure owns the last Arc.
        let permit_slot: Arc<std::sync::Mutex<Option<tokio::sync::OwnedSemaphorePermit>>> =
            Arc::new(std::sync::Mutex::new(Some(permit)));
        Box::pin(run_stream.then(move |event| {
            let is_terminal = matches!(event, AgentEvent::Done(_) | AgentEvent::Error(_));

            // Capture state only for terminal events (Done or Error).
            // All other events pass through without heap allocation.
            let terminal_state = if is_terminal {
                Some((
                    messages_ref.clone(),
                    session_store.clone(),
                    session_id.clone(),
                    cancel_store.clone(),
                    session_hooks.clone(),
                    permit_slot.clone(),
                ))
            } else {
                None
            };
            async move {
                if let Some((messages_ref, session_store, session_id, cancel_store, session_hooks, permit_slot)) = terminal_state {
                    // Clear the cancel token on any terminal event.
                    match cancel_store.lock() {
                        Ok(mut guard) => *guard = None,
                        Err(e) => {
                            tracing::error!("current_cancel mutex poisoned on terminal event, recovering");
                            *e.into_inner() = None;
                        }
                    }

                    // Done-specific: update history, run hooks, persist.
                    if let AgentEvent::Done(ref summary) = event {
                        let updated = summary.messages.clone();

                        match messages_ref.lock() {
                            Ok(mut guard) => *guard = updated.clone(),
                            Err(e) => {
                                tracing::error!("session messages mutex poisoned on Done, recovering");
                                *e.into_inner() = updated.clone();
                            }
                        }

                        if let Some(ref hooks) = session_hooks {
                            if let Some(ref f) = hooks.on_after_turn {
                                f(summary);
                            }
                        }

                        if let Some(store) = session_store {
                            let snapshot = StoredSession {
                                session_id: session_id.to_string(),
                                messages: updated,
                            };
                            if let Err(e) = store.save(session_id.as_str(), &snapshot).await {
                                tracing::warn!(session_id = %session_id, error = %e, "session store save failed");
                            }
                        }
                    }

                    // Release the turn guard on ANY terminal event (Done or Error)
                    // so the next send() can proceed.
                    drop(permit_slot.lock().ok().and_then(|mut g| g.take()));
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
        let guard = self.current_cancel.lock().unwrap_or_else(|e| {
            tracing::error!("current_cancel mutex poisoned in cancel_current, recovering");
            e.into_inner()
        });
        if let Some(token) = guard.as_ref() {
            token.cancel();
        }
    }

    /// The current message history for this session (read-only snapshot).
    pub fn messages(&self) -> Vec<Message> {
        self.messages
            .lock()
            .unwrap_or_else(|e| {
                tracing::error!("session messages mutex poisoned in messages(), recovering");
                e.into_inner()
            })
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
