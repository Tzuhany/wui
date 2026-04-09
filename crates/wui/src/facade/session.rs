// ============================================================================
// Session — multi-turn conversation state.
//
// A Session owns the message history for one ongoing conversation.
// After each turn, the full updated history is stored so the next call
// to `send()` continues from exactly where the last one left off.
//
// ── History synchrony ─────────────────────────────────────────────────────────
//
// `messages` uses `std::sync::Mutex` (not tokio's) so terminal-event cleanup
// can update history *synchronously*, without spawning a task. By the time
// `Done` is yielded to the caller, `self.messages` already reflects the full
// turn — the next `send()` immediately sees it.
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

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};

use futures::{stream, StreamExt};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::runtime::{
    run, HookRunner, RunConfig, RunStream, SessionPermissions, SessionStore, StoredSession,
};
use wui_core::event::{AgentError, AgentEvent, RunSummary};
use wui_core::hook::SessionId;
use wui_core::message::{ContentBlock, Message, Role};

use super::agent::build_run_config;
use super::builder::AgentConfig;

/// Lock a mutex, recovering from poisoning with a logged warning.
fn recover_mutex<'a, T>(mutex: &'a Mutex<T>, label: &str) -> MutexGuard<'a, T> {
    mutex.lock().unwrap_or_else(|e| {
        tracing::error!("{label} mutex poisoned, recovering");
        e.into_inner()
    })
}

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

    /// Called when the run returns an error, before any session-level retry
    /// attempt. Return `true` to suppress the error and retry; `false` to
    /// propagate immediately. `attempt` is 1-based.
    ///
    /// Retries are only attempted when the failed attempt has not yet emitted
    /// user-visible output; once text, tool events, or approval prompts have
    /// been streamed, the error is surfaced rather than replaying the turn.
    pub on_error: Option<Arc<dyn Fn(&wui_core::event::AgentError, u32) -> bool + Send + Sync>>,
}

/// An active multi-turn conversation.
pub struct Session {
    id: SessionId,
    config: Arc<AgentConfig>,
    messages: Arc<Mutex<Vec<Message>>>,
    turn_count: Arc<Mutex<u32>>,
    session_store: Option<Arc<dyn SessionStore>>,
    session_perms: Arc<SessionPermissions>,
    /// Cancel token for the currently running turn, if any.
    current_cancel: Arc<Mutex<Option<CancellationToken>>>,
    /// Ensures only one turn runs at a time. Without this, two concurrent
    /// `send()` calls would clone the same history, diverge, and the last
    /// to finish would silently overwrite the other's results.
    turn_guard: Arc<Semaphore>,
}

// ── TurnCleanup ──────────────────────────────────────────────────────────────
//
// Bundles the shared state needed for terminal events (Done / Error). Keeps
// session stream cleanup local to one type instead of spreading it across the
// event loop.

struct TurnCleanup {
    messages: Arc<Mutex<Vec<Message>>>,
    turn_count: Arc<Mutex<u32>>,
    session_store: Option<Arc<dyn SessionStore>>,
    session_id: SessionId,
    cancel_store: Arc<Mutex<Option<CancellationToken>>>,
    session_hooks: Option<Arc<SessionHooks>>,
    permit: Arc<Mutex<Option<tokio::sync::OwnedSemaphorePermit>>>,
}

struct SessionSendState<'a> {
    session: &'a Session,
    messages: Vec<Message>,
    current: RunStream,
    cleanup: Arc<TurnCleanup>,
    pending: VecDeque<AgentEvent>,
    attempt: u32,
    emitted_visible_output: bool,
    finished: bool,
}

enum SessionPoll {
    Emit(AgentEvent),
    Continue,
}

impl TurnCleanup {
    /// Handle a terminal event (Done or Error): clear the cancel token,
    /// update history, fire hooks, persist, and release the turn guard.
    async fn on_terminal(&self, event: AgentEvent) -> Vec<AgentEvent> {
        *recover_mutex(&self.cancel_store, "current_cancel") = None;

        let out = match event {
            AgentEvent::Done(summary) => {
                let updated = summary.messages.clone();
                *recover_mutex(&self.messages, "session messages") = updated.clone();

                if let Some(f) = self
                    .session_hooks
                    .as_ref()
                    .and_then(|h| h.on_after_turn.as_ref())
                {
                    f(&summary);
                }

                if let Some(store) = &self.session_store {
                    let snapshot = StoredSession {
                        session_id: self.session_id.to_string(),
                        messages: updated,
                    };
                    if let Err(e) = store.save(self.session_id.as_str(), &snapshot).await {
                        tracing::warn!(
                            session_id = %self.session_id, error = %e,
                            "session store save failed"
                        );
                    }
                }

                let turn = {
                    let mut count = recover_mutex(&self.turn_count, "turn_count");
                    *count += 1;
                    *count
                };

                vec![
                    AgentEvent::TurnDone {
                        turn,
                        usage: summary.usage.clone(),
                    },
                    AgentEvent::Done(summary),
                ]
            }
            other => vec![other],
        };

        // Release the turn guard so the next send() can proceed.
        drop(recover_mutex(&self.permit, "permit_slot").take());
        out
    }
}

fn stored_turn_count(messages: &[Message]) -> u32 {
    messages
        .iter()
        .filter(|msg| {
            msg.role == Role::User
                && msg
                    .content
                    .iter()
                    .any(|block| !matches!(block, ContentBlock::ToolResult { .. }))
        })
        .count() as u32
}

fn is_visible_attempt_output(event: &AgentEvent) -> bool {
    matches!(
        event,
        AgentEvent::TextDelta(_)
            | AgentEvent::ThinkingDelta(_)
            | AgentEvent::ToolStart { .. }
            | AgentEvent::ToolDone { .. }
            | AgentEvent::ToolError { .. }
            | AgentEvent::ToolProgress { .. }
            | AgentEvent::Artifact { .. }
            | AgentEvent::Control(_)
    )
}

impl<'a> SessionSendState<'a> {
    async fn next_event(&mut self) -> Option<AgentEvent> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Some(event);
            }
            if self.finished {
                return None;
            }

            let event = self.current.next().await?;
            match self.handle_event(event).await {
                SessionPoll::Emit(event) => return Some(event),
                SessionPoll::Continue => continue,
            }
        }
    }

    async fn handle_event(&mut self, event: AgentEvent) -> SessionPoll {
        match event {
            AgentEvent::Done(summary) => {
                self.finish_with(AgentEvent::Done(summary)).await;
                SessionPoll::Continue
            }
            AgentEvent::Error(error) => self.handle_error(error).await,
            other => {
                self.mark_visible_output(&other);
                SessionPoll::Emit(other)
            }
        }
    }

    async fn handle_error(&mut self, error: AgentError) -> SessionPoll {
        if self.should_retry(&error) && !self.emitted_visible_output {
            self.restart_attempt();
            return SessionPoll::Continue;
        }

        self.finish_with(AgentEvent::Error(error)).await;
        SessionPoll::Continue
    }

    fn should_retry(&self, error: &AgentError) -> bool {
        self.session
            .config
            .session_hooks
            .as_ref()
            .and_then(|hooks| hooks.on_error.as_ref())
            .is_some_and(|callback| callback(error, self.attempt))
    }

    fn restart_attempt(&mut self) {
        self.attempt += 1;
        self.current = self.session.start_run_stream(&self.messages);
    }

    fn mark_visible_output(&mut self, event: &AgentEvent) {
        if is_visible_attempt_output(event) {
            self.emitted_visible_output = true;
        }
    }

    async fn finish_with(&mut self, event: AgentEvent) {
        self.pending.extend(self.cleanup.on_terminal(event).await);
        self.finished = true;
    }
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
            turn_count: Arc::new(Mutex::new(stored_turn_count(&messages))),
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
        // The permit is moved into TurnCleanup and held for the stream's
        // lifetime; it is released when Done/Error fires or the stream drops.
        let permit = self
            .turn_guard
            .clone()
            .acquire_owned()
            .await
            .expect("turn_guard semaphore closed");

        let messages = self.prepare_turn_messages(content.into());

        let run_stream = self.start_run_stream(&messages);

        // Terminal cleanup completes before `Done` is yielded to the caller.
        // This guarantees that by the time the caller sees `Done`, both
        // in-memory history and the persisted session state are up to date.
        //
        // TurnCleanup is cloned (Arc bumps) only for terminal events. All other
        // events pass through without heap allocation.
        //
        // Box::pin makes the result Unpin so callers can use `stream.next().await`
        // without pinning manually.
        let cleanup = self.make_turn_cleanup(permit);

        let state = SessionSendState {
            session: self,
            messages,
            current: run_stream,
            cleanup,
            pending: VecDeque::new(),
            attempt: 1,
            emitted_visible_output: false,
            finished: false,
        };

        Box::pin(stream::unfold(state, |mut state| async move {
            state.next_event().await.map(|event| (event, state))
        }))
    }

    /// Cancel the currently running turn, if any.
    ///
    /// The run loop will emit `AgentEvent::Done` with `stop_reason: Cancelled`
    /// and exit. Has no effect if no turn is running.
    pub fn cancel_current(&self) {
        let guard = recover_mutex(&self.current_cancel, "current_cancel");
        if let Some(token) = guard.as_ref() {
            token.cancel();
        }
    }

    /// The current message history for this session (read-only snapshot).
    pub fn messages(&self) -> Vec<Message> {
        recover_mutex(&self.messages, "session messages").clone()
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

    fn start_run_stream(&self, messages: &[Message]) -> RunStream {
        let run_config = self.make_run_config();
        let run_stream = run(Arc::new(run_config), messages.to_vec());
        *recover_mutex(&self.current_cancel, "current_cancel") = Some(run_stream.cancel_token());
        run_stream
    }

    fn prepare_turn_messages(&self, content: String) -> Vec<Message> {
        // Build the message list for this turn: existing history + the new
        // user message. We do NOT push into `self.messages` here — that is
        // only updated on `AgentEvent::Done` (via `summary.messages`). If the
        // run ends in an error, `self.messages` stays at its pre-send state so
        // the next `send()` does not see a dangling user message without a
        // response.
        let mut messages = recover_mutex(&self.messages, "session messages").clone();
        messages.push(Message::user(content));

        match self
            .config
            .session_hooks
            .as_ref()
            .and_then(|hooks| hooks.on_before_send.as_ref())
        {
            Some(hook) => hook(messages),
            None => messages,
        }
    }

    fn make_turn_cleanup(&self, permit: tokio::sync::OwnedSemaphorePermit) -> Arc<TurnCleanup> {
        Arc::new(TurnCleanup {
            messages: self.messages.clone(),
            turn_count: self.turn_count.clone(),
            session_store: self.session_store.clone(),
            session_id: self.id.clone(),
            cancel_store: self.current_cancel.clone(),
            session_hooks: self.config.session_hooks.clone(),
            permit: Arc::new(Mutex::new(Some(permit))),
        })
    }
}
