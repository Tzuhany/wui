// ============================================================================
// Checkpoint / Resume — durable run state snapshots.
//
// After every tool-use iteration the run loop saves a `RunCheckpoint` to the
// configured store. On startup, if a checkpoint exists for the `run_id`, the
// run picks up from where it left off rather than starting from scratch.
//
// Two built-in stores:
//   InMemoryCheckpointStore — fast, ephemeral; good for testing.
//   FileCheckpointStore     — persists JSON files to a directory on disk.
//
// Plug in your own store (Redis, Postgres, S3, etc.) by implementing the
// `CheckpointStore` trait.
// ============================================================================

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use wui_core::event::TokenUsage;
use wui_core::message::Message;

// ── RunCheckpoint ─────────────────────────────────────────────────────────────

/// A snapshot of a run's state at the end of one iteration.
///
/// Contains enough information to resume the run from the point it was saved:
/// the full message history, the current iteration counter, and cumulative
/// token usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunCheckpoint {
    /// Identifies this run. Matches the key passed to `CheckpointStore::save`.
    pub run_id: String,
    /// The conversation history at checkpoint time.
    pub messages: Vec<Message>,
    /// How many iterations have completed.
    pub iteration: u32,
    /// Cumulative token usage across all completed iterations.
    pub total_usage: TokenUsage,
}

// ── CheckpointStore trait ─────────────────────────────────────────────────────

/// Storage backend for run checkpoints.
///
/// Implement this to persist checkpoints in any backend: file system, Redis,
/// Postgres, S3, etc. The run loop calls `save` at the end of each iteration
/// and `load` once on startup.
#[async_trait]
pub trait CheckpointStore: Send + Sync + 'static {
    /// Persist `state` under `run_id`.
    ///
    /// Overwrites any existing checkpoint for the same `run_id`.
    async fn save(&self, run_id: &str, state: &RunCheckpoint) -> anyhow::Result<()>;

    /// Retrieve the most recent checkpoint for `run_id`.
    ///
    /// Returns `Ok(None)` when no checkpoint exists (fresh run).
    async fn load(&self, run_id: &str) -> anyhow::Result<Option<RunCheckpoint>>;

    /// Delete the checkpoint for `run_id`.
    ///
    /// A no-op when no checkpoint exists.
    async fn clear(&self, run_id: &str) -> anyhow::Result<()>;
}

// ── InMemoryCheckpointStore ───────────────────────────────────────────────────

/// An in-memory checkpoint store backed by a `HashMap`.
///
/// Data is lost when the process exits. Use this for testing or for runs
/// that only need to survive within a single process lifetime.
///
/// Thread-safe: all methods are protected by a `Mutex`.
#[derive(Default, Clone)]
pub struct InMemoryCheckpointStore {
    inner: Arc<Mutex<HashMap<String, RunCheckpoint>>>,
}

impl InMemoryCheckpointStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl CheckpointStore for InMemoryCheckpointStore {
    async fn save(&self, run_id: &str, state: &RunCheckpoint) -> anyhow::Result<()> {
        self.inner
            .lock()
            .await
            .insert(run_id.to_string(), state.clone());
        Ok(())
    }

    async fn load(&self, run_id: &str) -> anyhow::Result<Option<RunCheckpoint>> {
        Ok(self.inner.lock().await.get(run_id).cloned())
    }

    async fn clear(&self, run_id: &str) -> anyhow::Result<()> {
        self.inner.lock().await.remove(run_id);
        Ok(())
    }
}

// ── FileCheckpointStore ───────────────────────────────────────────────────────

/// A checkpoint store that persists state as JSON files on disk.
///
/// Each run is stored in a separate file named `{dir}/{run_id}.json`.
/// File I/O is performed with `tokio::fs` for async compatibility.
///
/// ```rust,ignore
/// let store = FileCheckpointStore::new("/tmp/my-agent-checkpoints");
/// let agent = Agent::builder(provider)
///     .checkpoint(store, "run-42")
///     .build();
/// ```
#[derive(Clone)]
pub struct FileCheckpointStore {
    dir: PathBuf,
}

impl FileCheckpointStore {
    /// Create a store that reads and writes files in `dir`.
    ///
    /// The directory is created if it does not already exist.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn path(&self, run_id: &str) -> PathBuf {
        // Sanitise run_id: replace path separators to avoid directory traversal.
        let safe = run_id.replace(['/', '\\', ':'], "_");
        self.dir.join(format!("{safe}.json"))
    }
}

#[async_trait]
impl CheckpointStore for FileCheckpointStore {
    async fn save(&self, run_id: &str, state: &RunCheckpoint) -> anyhow::Result<()> {
        tokio::fs::create_dir_all(&self.dir).await?;
        let json = serde_json::to_string_pretty(state)?;
        tokio::fs::write(self.path(run_id), json).await?;
        Ok(())
    }

    async fn load(&self, run_id: &str) -> anyhow::Result<Option<RunCheckpoint>> {
        let path = self.path(run_id);
        match tokio::fs::read_to_string(&path).await {
            Ok(contents) => {
                let cp: RunCheckpoint = serde_json::from_str(&contents)?;
                Ok(Some(cp))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    async fn clear(&self, run_id: &str) -> anyhow::Result<()> {
        let path = self.path(run_id);
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
}
