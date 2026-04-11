use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::MemoryError;

/// Input to `RememberBackend::remember`.
///
/// All fields except `content` are hints to the backend — they are part of the
/// interface contract so callers can express intent, but backends are free to
/// interpret or ignore them according to their storage model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewMemory {
    pub content: String,
    /// A short, human-readable label for this memory.
    ///
    /// Intended for display and lookup, not uniqueness. Two memories may share
    /// the same name. Backends may index or ignore it.
    pub name: Option<String>,
    /// Agent-defined category tag: `"fact"`, `"preference"`, `"instruction"`, …
    pub kind: Option<String>,
    /// Suggested relevance weight (0.0–1.0). Higher entries surface first in recall.
    /// Backends may use this for ranking, indexing, or ignore it entirely.
    pub importance: Option<f32>,
    /// Hint that this memory should always be considered, independent of query relevance.
    ///
    /// What "always considered" means is the backend's decision: it might inject
    /// pinned entries unconditionally, boost their score, or ignore the flag entirely.
    /// Defaults to `false`.
    #[serde(default)]
    pub pinned: bool,
}

/// A stable reference to a stored memory, returned after a successful write.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRef {
    pub id: String,
}

/// A single search result returned by `RecallBackend::recall`.
///
/// All optional fields reflect what the backend chooses to track and return;
/// callers should not assume their presence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHit {
    pub id: String,
    pub content: String,
    /// Human-readable label, if the backend stores one.
    pub name: Option<String>,
    /// Category tag if the backend tracks one. May be `None`.
    pub kind: Option<String>,
    /// Importance weight if the backend tracks one. May be `None`.
    pub importance: Option<f32>,
    /// Backend-assigned relevance score. Higher is more relevant.
    /// `None` when the backend does not compute scores.
    pub score: Option<f32>,
    /// Whether the backend has this entry marked as pinned.
    #[serde(default)]
    pub pinned: bool,
}

/// Search memory by query.
///
/// The backend decides what "relevant" means — keyword match, vector similarity,
/// BM25, or any other retrieval strategy.
#[async_trait]
pub trait RecallBackend: Send + Sync + 'static {
    async fn recall(
        &self,
        query: &str,
        limit: Option<usize>,
    ) -> Result<Vec<MemoryHit>, MemoryError>;
}

/// Write a new memory entry.
#[async_trait]
pub trait RememberBackend: Send + Sync + 'static {
    async fn remember(&self, item: NewMemory) -> Result<MemoryRef, MemoryError>;
}

/// Request that a memory entry no longer be recalled.
///
/// This expresses **agent intent** — "I want this to stop surfacing" — not a
/// storage operation. The backend decides how to honour the request:
///   - hard delete (true erasure)
///   - soft delete / tombstone (keeps the record, marks it suppressed)
///   - blacklist (filters the id at query time)
///
/// The only contract: after a successful `forget`, the same id must not appear
/// in future `recall` results. How the data is physically stored is up to the
/// implementation.
#[async_trait]
pub trait ForgetBackend: Send + Sync + 'static {
    async fn forget(&self, id: &str) -> Result<(), MemoryError>;
}
