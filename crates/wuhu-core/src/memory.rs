// ============================================================================
// Memory — continuity across sessions.
//
// Memory is distinct from the conversation history. History is what happened
// in this session; memory is what the agent knows across all sessions.
//
// The framework defines the trait and the injection contract. The recall
// strategy (vector search, BM25, recency, hybrid) and the storage backend
// are application concerns — the framework does not prescribe them.
//
// The engine calls recall() before each turn and injects the results into
// the system prompt. The agent can call write() to persist new knowledge.
// ============================================================================

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Pluggable memory backend.
///
/// Implement this to give the agent persistent knowledge across sessions.
/// The simplest implementation is a list of strings in a JSON file.
/// A production implementation might use pgvector + BM25 hybrid retrieval.
#[async_trait]
pub trait Memory: Send + Sync + 'static {
    /// Recall entries relevant to the current turn.
    ///
    /// `query` is typically the latest user message or a summary of recent
    /// context. The implementation decides how to interpret it.
    ///
    /// The engine injects the returned entries into the system prompt before
    /// each LLM call, ordered by `importance` descending.
    async fn recall(&self, query: &str) -> Result<Vec<MemoryEntry>, MemoryError>;

    /// Persist a new memory entry.
    ///
    /// Called by the agent (via the MemoryWrite tool) when it decides
    /// something is worth remembering. The framework does not call this
    /// automatically — the LLM decides what to remember.
    async fn write(&self, entry: NewMemory) -> Result<String, MemoryError>;

    /// Search for entries matching a query (semantic or keyword).
    ///
    /// Different from `recall()` in that it is called explicitly by the
    /// agent (via the MemorySearch tool), not automatically by the engine.
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>, MemoryError>;

    /// Delete an entry by id.
    async fn forget(&self, id: &str) -> Result<(), MemoryError>;
}

// ── Memory Entry ──────────────────────────────────────────────────────────────

/// A recalled or searched memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id:         String,
    /// Short title shown in the agent's memory index.
    pub name:       String,
    /// Full content of the memory.
    pub body:       String,
    /// 0.0 - 1.0. Higher importance entries are injected first and are
    /// less likely to be evicted when the prompt is full.
    pub importance: f32,
    /// Semantic category (e.g. "feedback", "preference", "fact").
    pub kind:       String,
}

/// A memory entry to be written.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewMemory {
    pub name:       String,
    pub body:       String,
    pub kind:       String,
    /// Optional TTL in days. `None` means permanent.
    pub ttl_days:   Option<u32>,
}

// ── Memory Error ──────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("entry not found: {0}")]
    NotFound(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
