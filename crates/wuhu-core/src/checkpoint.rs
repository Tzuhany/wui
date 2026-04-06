// ============================================================================
// Checkpoint — optional persistence.
//
// Not implementing Checkpoint means your sessions are in-memory and lost
// when the process exits. Implementing it means sessions survive restarts,
// can resume after HITL pauses, and support multi-process deployments.
//
// The framework defines the interface; you bring the storage backend.
// Redis, PostgreSQL, SQLite, a flat file — all are valid implementations.
// ============================================================================

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::event::ControlRequest;
use crate::message::Message;

/// Pluggable session persistence.
///
/// Implement this trait to connect any storage backend. The engine calls
/// `save()` after every turn and `load()` when a session is resumed.
#[async_trait]
pub trait Checkpoint: Send + Sync + 'static {
    /// Load the most recent snapshot for a session.
    ///
    /// Returns `None` if no snapshot exists (new session).
    async fn load(&self, session_id: &str) -> Result<Option<SessionSnapshot>, CheckpointError>;

    /// Persist the current session state.
    ///
    /// Called after every completed turn. Implementations should be
    /// durable — a crash after `save()` returns must not lose the snapshot.
    async fn save(&self, session_id: &str, snapshot: &SessionSnapshot) -> Result<(), CheckpointError>;

    /// Delete all state for a session.
    ///
    /// Called when a session is explicitly terminated. Implementations may
    /// choose to archive rather than delete.
    async fn delete(&self, session_id: &str) -> Result<(), CheckpointError>;
}

// ── Session Snapshot ──────────────────────────────────────────────────────────

/// The serialisable state of a session at a point in time.
///
/// Everything needed to resume a session exactly where it left off:
/// the message history, any pending human decision, and the original
/// (pre-compression) message archive for L2 recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub session_id:  String,
    pub messages:    Vec<Message>,

    /// If the session was paused waiting for a human response, this holds
    /// the control request so it can be replayed after a restart.
    pub pending:     Option<PendingControl>,

    /// The complete uncompressed message archive.
    ///
    /// L2 (collapse) compression replaces messages with placeholders in
    /// `messages` but archives the originals here. This allows the full
    /// history to be surfaced for display or forking while keeping the
    /// working context small.
    pub archive:     Vec<Message>,
}

impl SessionSnapshot {
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            messages:   Vec::new(),
            pending:    None,
            archive:    Vec::new(),
        }
    }
}

/// A paused control request waiting for a human response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingControl {
    pub request:   ControlRequest,
    /// Index into `messages` where the pause happened.
    /// Used to resume from exactly the right position.
    pub resume_at: usize,
}

// ── Checkpoint Error ──────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("serialization failed: {0}")]
    Serialize(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

// ── Built-in: In-Memory ───────────────────────────────────────────────────────

/// A `Checkpoint` backed by a shared `Arc<RwLock<HashMap>>`.
///
/// Useful for testing and single-process deployments where persistence
/// across restarts is not required. Drop-in replacement for any persistent
/// backend during development.
///
/// `Clone` is cheap — all clones share the same underlying store. Pass the
/// same `InMemory` to multiple agents and they will all read from and write
/// to the same session map.
#[derive(Clone)]
pub struct InMemory {
    store: std::sync::Arc<tokio::sync::RwLock<std::collections::HashMap<String, SessionSnapshot>>>,
}

impl InMemory {
    pub fn new() -> Self {
        Self { store: std::sync::Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())) }
    }
}

impl Default for InMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Checkpoint for InMemory {
    async fn load(&self, session_id: &str) -> Result<Option<SessionSnapshot>, CheckpointError> {
        Ok(self.store.read().await.get(session_id).cloned())
    }

    async fn save(&self, session_id: &str, snapshot: &SessionSnapshot) -> Result<(), CheckpointError> {
        self.store.write().await.insert(session_id.to_string(), snapshot.clone());
        Ok(())
    }

    async fn delete(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.store.write().await.remove(session_id);
        Ok(())
    }
}
