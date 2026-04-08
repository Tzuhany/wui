// ============================================================================
// wui-memory — recall/remember/forget capability traits for Wui agents.
//
// This crate is a companion, not the product. It defines three capability
// traits that let agents interact with memory, and ships one reference backend
// for development and testing. The traits are the point.
//
// What this crate is:
//   - A definition of what "memory capability" means in the wui vocabulary.
//   - Three agent-facing tools: recall, remember, forget.
//   - A reference backend (InMemoryStore) for development and tests.
//
// What this crate is NOT:
//   - A memory system. Every application has its own idea of what memory is,
//     how long it lives, and where it is stored. That decision belongs to the
//     application, not to this crate.
//   - A prescription. InMemoryStore shows one way to satisfy the traits;
//     it is not the intended answer for production.
//
//   RecallBackend   → memory_recall   (search by query)
//   RememberBackend → memory_remember (write a new entry)
//   ForgetBackend   → memory_forget   (request that an entry no longer be recalled)
//
// Usage (reference backend):
//
//   let store = Arc::new(InMemoryStore::new());
//   let agent = Agent::builder(provider)
//       .tools(memory_tools()
//           .with_recall(store.clone())
//           .with_remember(store.clone())
//           .build())
//       .build();
//
// Usage (your own backend):
//
//   struct MyPgVectorStore { ... }
//   impl RecallBackend for MyPgVectorStore { ... }
//   impl RememberBackend for MyPgVectorStore { ... }
//   let agent = Agent::builder(provider)
//       .tools(memory_tools().with_recall(store.clone()).with_remember(store).build())
//       .build();
// ============================================================================

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::RwLock;
use uuid::Uuid;

use wui_core::tool::{Tool, ToolCtx, ToolInput, ToolMeta, ToolOutput};

// ── Types ─────────────────────────────────────────────────────────────────────

/// Input to `RememberBackend::remember`.
///
/// `kind` and `importance` are hints to the backend — they are part of the
/// interface contract so callers can express intent, but backends are free to
/// interpret or ignore them according to their storage model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewMemory {
    pub content: String,
    /// Agent-defined category tag: `"fact"`, `"preference"`, `"instruction"`, …
    pub kind: Option<String>,
    /// Suggested relevance weight (0.0–1.0). Higher entries surface first in recall.
    /// Backends may use this for ranking, indexing, or ignore it entirely.
    pub importance: Option<f32>,
}

/// A stable reference to a stored memory, returned after a successful write.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRef {
    pub id: String,
}

/// A single search result returned by `RecallBackend::recall`.
///
/// `kind`, `importance`, and `score` are all `Option` — not every backend
/// tracks them, and callers should not assume their presence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHit {
    pub id: String,
    pub content: String,
    /// Category tag if the backend tracks one. May be `None`.
    pub kind: Option<String>,
    /// Importance weight if the backend tracks one. May be `None`.
    pub importance: Option<f32>,
    /// Backend-assigned relevance score. Higher is more relevant.
    /// `None` when the backend does not compute scores.
    pub score: Option<f32>,
}

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("entry not found: {0}")]
    NotFound(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

// ── Backend traits ────────────────────────────────────────────────────────────

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

// ── InMemoryStore ─────────────────────────────────────────────────────────────
//
// A reference backend for development and testing.
//
// This is deliberately simple: a `Vec` behind an `RwLock`, with substring
// matching for recall. It is not a recommendation about how memory should be
// structured in production. Swap it for any backend that implements the traits.

/// Internal storage entry. Not part of the public trait surface — this is an
/// implementation detail of `InMemoryStore`.
#[derive(Debug, Clone)]
struct StoredEntry {
    id: String,
    content: String,
    kind: Option<String>,
    importance: f32,
}

#[derive(Clone, Default)]
pub struct InMemoryStore {
    entries: Arc<RwLock<Vec<StoredEntry>>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl RecallBackend for InMemoryStore {
    async fn recall(
        &self,
        query: &str,
        limit: Option<usize>,
    ) -> Result<Vec<MemoryHit>, MemoryError> {
        let needle = query.trim().to_lowercase();
        if needle.is_empty() {
            return Ok(Vec::new());
        }

        let entries = self.entries.read().await;
        let mut hits: Vec<MemoryHit> = entries
            .iter()
            .filter_map(|e| {
                let occurrences = e.content.to_lowercase().matches(&needle).count();
                (occurrences > 0).then(|| MemoryHit {
                    id: e.id.clone(),
                    content: e.content.clone(),
                    kind: e.kind.clone(),
                    importance: Some(e.importance),
                    score: Some(occurrences as f32 * e.importance),
                })
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });

        if let Some(n) = limit {
            hits.truncate(n);
        }
        Ok(hits)
    }
}

#[async_trait]
impl RememberBackend for InMemoryStore {
    async fn remember(&self, item: NewMemory) -> Result<MemoryRef, MemoryError> {
        let id = Uuid::new_v4().to_string();
        self.entries.write().await.push(StoredEntry {
            id: id.clone(),
            content: item.content,
            kind: item.kind,
            importance: item.importance.unwrap_or(0.5).clamp(0.0, 1.0),
        });
        Ok(MemoryRef { id })
    }
}

#[async_trait]
impl ForgetBackend for InMemoryStore {
    async fn forget(&self, id: &str) -> Result<(), MemoryError> {
        self.entries.write().await.retain(|e| e.id != id);
        Ok(())
    }
}

// ── MemoryTools builder ───────────────────────────────────────────────────────

#[derive(Default)]
pub struct MemoryTools {
    recall: Option<Arc<dyn RecallBackend>>,
    remember: Option<Arc<dyn RememberBackend>>,
    forget: Option<Arc<dyn ForgetBackend>>,
}

impl MemoryTools {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_recall(mut self, backend: Arc<dyn RecallBackend>) -> Self {
        self.recall = Some(backend);
        self
    }

    pub fn with_remember(mut self, backend: Arc<dyn RememberBackend>) -> Self {
        self.remember = Some(backend);
        self
    }

    pub fn with_forget(mut self, backend: Arc<dyn ForgetBackend>) -> Self {
        self.forget = Some(backend);
        self
    }

    pub fn build(self) -> Vec<Arc<dyn Tool>> {
        let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
        if let Some(b) = self.recall {
            tools.push(Arc::new(RecallTool::new(b)));
        }
        if let Some(b) = self.remember {
            tools.push(Arc::new(RememberTool::new(b)));
        }
        if let Some(b) = self.forget {
            tools.push(Arc::new(ForgetTool::new(b)));
        }
        tools
    }
}

/// Start building a set of memory tools.
pub fn memory_tools() -> MemoryTools {
    MemoryTools::new()
}

/// Convenience: all three memory tools backed by one `InMemoryStore`.
///
/// Good for prototyping. For production, implement the backend traits against
/// your own store and assemble with `memory_tools()`.
///
/// ```rust,ignore
/// let store = Arc::new(InMemoryStore::new());
/// let agent = Agent::builder(provider)
///     .tools(all_memory_tools(store))
///     .build();
/// ```
pub fn all_memory_tools(store: Arc<InMemoryStore>) -> Vec<Arc<dyn Tool>> {
    memory_tools()
        .with_recall(store.clone())
        .with_remember(store.clone())
        .with_forget(store)
        .build()
}

// ── Tools ─────────────────────────────────────────────────────────────────────

struct RecallTool {
    backend: Arc<dyn RecallBackend>,
}

impl RecallTool {
    pub(crate) fn new(b: Arc<dyn RecallBackend>) -> Self {
        Self { backend: b }
    }
}

struct RememberTool {
    backend: Arc<dyn RememberBackend>,
}

impl RememberTool {
    pub(crate) fn new(b: Arc<dyn RememberBackend>) -> Self {
        Self { backend: b }
    }
}

struct ForgetTool {
    backend: Arc<dyn ForgetBackend>,
}

impl ForgetTool {
    pub(crate) fn new(b: Arc<dyn ForgetBackend>) -> Self {
        Self { backend: b }
    }
}

#[async_trait]
impl Tool for RecallTool {
    fn name(&self) -> &str {
        "memory_recall"
    }
    fn description(&self) -> &str {
        "Search long-term memory by query. Results are ranked by relevance × importance."
    }
    fn meta(&self, _input: &serde_json::Value) -> ToolMeta {
        ToolMeta {
            readonly: true,
            concurrent: true,
            ..ToolMeta::default()
        }
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "What to search for." },
                "limit": { "type": "integer", "minimum": 1, "description": "Max results (default: no limit)." }
            },
            "required": ["query"]
        })
    }
    async fn call(&self, input: serde_json::Value, ctx: &ToolCtx) -> ToolOutput {
        let inp = ToolInput(&input);
        let query = match inp.required_str("query") {
            Ok(v) => v,
            Err(e) => return ToolOutput::invalid_input(e),
        };
        let limit = inp.optional_u64("limit").map(|v| v as usize);

        ctx.report("recalling memories");
        let result = tokio::select! {
            r = self.backend.recall(query, limit) => r,
            _ = ctx.cancel.cancelled() => return ToolOutput::error("cancelled"),
        };

        match result {
            Err(e) => ToolOutput::error(e.to_string()),
            Ok(hits) if hits.is_empty() => ToolOutput::success("No relevant memories found.")
                .with_structured(json!({ "hits": [] })),
            Ok(hits) => {
                let text = hits
                    .iter()
                    .map(|h| format!("[{}] {}", h.id, h.content))
                    .collect::<Vec<_>>()
                    .join("\n");
                ToolOutput::success(text).with_structured(json!({ "hits": hits }))
            }
        }
    }
}

#[async_trait]
impl Tool for RememberTool {
    fn name(&self) -> &str {
        "memory_remember"
    }
    fn description(&self) -> &str {
        "Store a durable memory for later recall. Use `kind` to categorise (e.g. \"fact\", \"preference\") and `importance` (0.0–1.0) to prioritise."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "content":    { "type": "string", "description": "The memory to store." },
                "kind":       { "type": "string", "description": "Category tag, e.g. \"preference\" or \"fact\"." },
                "importance": { "type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Priority weight (default 0.5)." }
            },
            "required": ["content"]
        })
    }
    async fn call(&self, input: serde_json::Value, ctx: &ToolCtx) -> ToolOutput {
        let inp = ToolInput(&input);
        let content = match inp.required_str("content") {
            Ok(v) => v,
            Err(e) => return ToolOutput::invalid_input(e),
        };
        let item = NewMemory {
            content: content.to_string(),
            kind: inp.optional_str("kind").map(str::to_string),
            importance: inp.optional_f64("importance").map(|v| v as f32),
        };

        ctx.report("storing memory");
        let result = tokio::select! {
            r = self.backend.remember(item) => r,
            _ = ctx.cancel.cancelled() => return ToolOutput::error("cancelled"),
        };

        match result {
            Err(e) => ToolOutput::error(e.to_string()),
            Ok(mref) => ToolOutput::success(format!("Stored memory {}", mref.id))
                .with_structured(json!({ "memory": mref })),
        }
    }
}

#[async_trait]
impl Tool for ForgetTool {
    fn name(&self) -> &str {
        "memory_forget"
    }
    fn description(&self) -> &str {
        "Request that a memory no longer be recalled. The backend decides how to honour this — hard delete, soft delete, or suppression."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "id": { "type": "string", "description": "The memory id to forget." }
            },
            "required": ["id"]
        })
    }
    async fn call(&self, input: serde_json::Value, ctx: &ToolCtx) -> ToolOutput {
        let inp = ToolInput(&input);
        let id = match inp.required_str("id") {
            Ok(v) => v,
            Err(e) => return ToolOutput::invalid_input(e),
        };

        ctx.report("forgetting memory");
        let result = tokio::select! {
            r = self.backend.forget(id) => r,
            _ = ctx.cancel.cancelled() => return ToolOutput::error("cancelled"),
        };

        match result {
            Err(e) => ToolOutput::error(e.to_string()),
            Ok(()) => {
                ToolOutput::success(format!("Memory {id} will no longer influence reasoning."))
                    .with_structured(json!({ "forgotten_id": id }))
            }
        }
    }
}

// ── Vector store ─────────────────────────────────────────────────────────────
//
// VectorStore provides semantic similarity search over dense float vectors.
// InMemoryVectorStore is a brute-force reference implementation suitable for
// development and small datasets. SemanticMemoryTool wraps any VectorStore
// and EmbeddingFn into a Tool the agent can call.

use futures::future::BoxFuture;

/// A search result returned by `VectorStore::search`.
pub struct VectorHit {
    /// The stored entry identifier.
    pub id: String,
    /// The original text that was upserted.
    pub text: String,
    /// Cosine similarity in `[-1.0, 1.0]`. Higher = more similar.
    pub score: f32,
}

/// Semantic vector storage: upsert, search, and delete.
///
/// Implement this to back `SemanticMemoryTool` with your preferred vector
/// database (pgvector, Pinecone, Qdrant, Weaviate, etc.).
#[async_trait]
pub trait VectorStore: Send + Sync + 'static {
    /// Store or update an entry.
    ///
    /// If `id` already exists the entry is replaced in full.
    async fn upsert(&self, id: &str, text: &str, vector: Vec<f32>) -> anyhow::Result<()>;

    /// Return the `limit` entries with the highest cosine similarity to `query_vector`.
    async fn search(&self, query_vector: &[f32], limit: usize) -> anyhow::Result<Vec<VectorHit>>;

    /// Remove the entry with the given `id`.
    ///
    /// A no-op when the id does not exist.
    async fn delete(&self, id: &str) -> anyhow::Result<()>;
}

/// An in-memory vector store backed by a `Vec` with brute-force cosine search.
///
/// Suitable for development and small corpora. For production, implement
/// `VectorStore` against a dedicated vector database.
#[derive(Clone, Default)]
pub struct InMemoryVectorStore {
    entries: Arc<RwLock<Vec<VectorEntry>>>,
}

#[derive(Clone)]
struct VectorEntry {
    id: String,
    text: String,
    vector: Vec<f32>,
}

impl InMemoryVectorStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn upsert(&self, id: &str, text: &str, vector: Vec<f32>) -> anyhow::Result<()> {
        let mut entries = self.entries.write().await;
        if let Some(e) = entries.iter_mut().find(|e| e.id == id) {
            e.text = text.to_string();
            e.vector = vector;
        } else {
            entries.push(VectorEntry {
                id: id.to_string(),
                text: text.to_string(),
                vector,
            });
        }
        Ok(())
    }

    async fn search(&self, query_vector: &[f32], limit: usize) -> anyhow::Result<Vec<VectorHit>> {
        let entries = self.entries.read().await;
        let mut hits: Vec<VectorHit> = entries
            .iter()
            .map(|e| VectorHit {
                id: e.id.clone(),
                text: e.text.clone(),
                score: cosine_similarity(query_vector, &e.vector),
            })
            .collect();
        // Sort descending by score.
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(limit);
        Ok(hits)
    }

    async fn delete(&self, id: &str) -> anyhow::Result<()> {
        self.entries.write().await.retain(|e| e.id != id);
        Ok(())
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let dot: f32 = a[..len]
        .iter()
        .zip(b[..len].iter())
        .map(|(x, y)| x * y)
        .sum();
    let norm_a: f32 = a[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

// ── SemanticMemoryTool ────────────────────────────────────────────────────────

/// Type alias for an async embedding function.
///
/// Takes a `String` input and returns a `Vec<f32>` embedding vector.
pub type EmbedFn = Arc<dyn Fn(String) -> BoxFuture<'static, Vec<f32>> + Send + Sync>;

/// A tool that stores and retrieves text via semantic (vector) similarity.
///
/// The agent can call this tool to remember facts and later recall them by
/// semantic similarity rather than exact keyword match. This requires an
/// embedding provider — pass any async function that maps strings to vectors.
///
/// ```rust,ignore
/// let store  = Arc::new(InMemoryVectorStore::new());
/// let embed: EmbedFn = Arc::new(|text: String| {
///     Box::pin(async move { my_embed_api(text).await })
/// });
/// let tool = SemanticMemoryTool::new(store, embed, 5);
/// ```
pub struct SemanticMemoryTool {
    store: Arc<dyn VectorStore>,
    embed: EmbedFn,
    max_results: usize,
}

impl SemanticMemoryTool {
    /// Create a new `SemanticMemoryTool`.
    ///
    /// - `store`       — the vector storage backend.
    /// - `embed`       — async function that converts text to a float vector.
    /// - `max_results` — maximum number of hits returned per search.
    pub fn new(store: Arc<dyn VectorStore>, embed: EmbedFn, max_results: usize) -> Self {
        Self {
            store,
            embed,
            max_results,
        }
    }
}

#[async_trait]
impl Tool for SemanticMemoryTool {
    fn name(&self) -> &str {
        "semantic_memory"
    }

    fn description(&self) -> &str {
        "Store or search memories using semantic (vector) similarity. \
         Use action=\"upsert\" to store and action=\"search\" to query."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["upsert", "search", "delete"],
                    "description": "\"upsert\" to store, \"search\" to retrieve, \"delete\" to remove."
                },
                "id":   { "type": "string", "description": "Unique entry ID (required for upsert and delete)." },
                "text": { "type": "string", "description": "Text to store or use as search query." },
                "limit": {
                    "type": "integer", "minimum": 1,
                    "description": "Max results for search (default: tool max_results)."
                }
            },
            "required": ["action"]
        })
    }

    async fn call(&self, input: serde_json::Value, ctx: &ToolCtx) -> ToolOutput {
        let inp = ToolInput(&input);
        let action = match inp.required_str("action") {
            Ok(v) => v,
            Err(e) => return ToolOutput::invalid_input(e),
        };

        match action {
            "upsert" => {
                let id = match inp.required_str("id") {
                    Ok(v) => v,
                    Err(e) => return ToolOutput::invalid_input(e),
                };
                let text = match inp.required_str("text") {
                    Ok(v) => v,
                    Err(e) => return ToolOutput::invalid_input(e),
                };

                ctx.report(format!("embedding text for id={id}"));
                let vector = (self.embed)(text.to_string()).await;

                match self.store.upsert(id, text, vector).await {
                    Ok(()) => ToolOutput::success(format!("Stored memory with id={id}.")),
                    Err(e) => ToolOutput::error(format!("upsert failed: {e}")),
                }
            }
            "search" => {
                let text = match inp.required_str("text") {
                    Ok(v) => v,
                    Err(e) => return ToolOutput::invalid_input(e),
                };
                let limit = inp
                    .optional_u64("limit")
                    .map(|v| v as usize)
                    .unwrap_or(self.max_results);

                ctx.report("embedding query");
                let query_vec = (self.embed)(text.to_string()).await;

                match self.store.search(&query_vec, limit).await {
                    Err(e) => ToolOutput::error(format!("search failed: {e}")),
                    Ok(hits) if hits.is_empty() => {
                        ToolOutput::success("No semantically similar memories found.")
                    }
                    Ok(hits) => {
                        let text = hits
                            .iter()
                            .map(|h| format!("[{:.3}] [{}] {}", h.score, h.id, h.text))
                            .collect::<Vec<_>>()
                            .join("\n");
                        ToolOutput::success(text)
                    }
                }
            }
            "delete" => {
                let id = match inp.required_str("id") {
                    Ok(v) => v,
                    Err(e) => return ToolOutput::invalid_input(e),
                };
                match self.store.delete(id).await {
                    Ok(()) => ToolOutput::success(format!("Deleted memory id={id}.")),
                    Err(e) => ToolOutput::error(format!("delete failed: {e}")),
                }
            }
            other => ToolOutput::invalid_input(format!("unknown action: {other}")),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn remember_and_recall() {
        let store = InMemoryStore::new();
        let r1 = store
            .remember(NewMemory {
                content: "Alice likes matcha".into(),
                kind: Some("preference".into()),
                importance: Some(0.8),
            })
            .await
            .unwrap();
        let _r2 = store
            .remember(NewMemory {
                content: "Bob likes coffee".into(),
                kind: Some("preference".into()),
                importance: Some(0.5),
            })
            .await
            .unwrap();

        let hits = store.recall("matcha", Some(5)).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, r1.id);
        assert_eq!(hits[0].kind.as_deref(), Some("preference"));
        assert_eq!(hits[0].importance, Some(0.8));
    }

    #[tokio::test]
    async fn forget_removes_from_recall() {
        let store = InMemoryStore::new();
        let r = store
            .remember(NewMemory {
                content: "temporary note".into(),
                kind: None,
                importance: None,
            })
            .await
            .unwrap();

        store.forget(&r.id).await.unwrap();

        let hits = store.recall("temporary", None).await.unwrap();
        assert!(
            hits.is_empty(),
            "forgotten entry should not appear in recall"
        );
    }

    #[tokio::test]
    async fn recall_ranks_by_score() {
        let store = InMemoryStore::new();
        store
            .remember(NewMemory {
                content: "rust rust rust".into(),
                kind: None,
                importance: Some(0.5),
            })
            .await
            .unwrap();
        store
            .remember(NewMemory {
                content: "rust".into(),
                kind: None,
                importance: Some(0.9),
            })
            .await
            .unwrap();

        let hits = store.recall("rust", None).await.unwrap();
        // First entry has 3 occurrences × 0.5 = 1.5 score.
        // Second entry has 1 occurrence × 0.9 = 0.9 score.
        // First should rank higher.
        assert_eq!(hits[0].content, "rust rust rust");
    }

    #[tokio::test]
    async fn memory_tools_builder_respects_selection() {
        let store = Arc::new(InMemoryStore::new());
        let tools = memory_tools()
            .with_recall(store.clone())
            .with_remember(store.clone())
            .build();

        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert_eq!(names, vec!["memory_recall", "memory_remember"]);
    }

    #[tokio::test]
    async fn all_memory_tools_returns_three() {
        let store = Arc::new(InMemoryStore::new());
        let tools = all_memory_tools(store);
        assert_eq!(tools.len(), 3);
    }

    /// Smoke test: remember a fact via the tool, recall it back by keyword.
    #[tokio::test]
    async fn memory_tool_remember_and_recall() {
        use tokio_util::sync::CancellationToken;
        use wui_core::tool::ToolCtx;

        let store = Arc::new(InMemoryStore::new());
        let tools = all_memory_tools(store);

        let remember_tool = tools
            .iter()
            .find(|t| t.name() == "memory_remember")
            .unwrap()
            .clone();
        let recall_tool = tools
            .iter()
            .find(|t| t.name() == "memory_recall")
            .unwrap()
            .clone();

        let ctx = ToolCtx {
            cancel: CancellationToken::new(),
            messages: Arc::<[wui_core::message::Message]>::from(Vec::new()),
            on_progress: Box::new(|_| {}),
        };

        // Remember a fact.
        let remember_output = remember_tool
            .call(serde_json::json!({"content": "the sky is blue"}), &ctx)
            .await;
        assert!(
            remember_output.is_ok(),
            "remember failed: {}",
            remember_output.content
        );

        // Recall by keyword.
        let recall_output = recall_tool
            .call(serde_json::json!({"query": "sky"}), &ctx)
            .await;
        assert!(
            recall_output.is_ok(),
            "recall failed: {}",
            recall_output.content
        );
        assert!(
            recall_output.content.contains("blue"),
            "expected 'blue' in recall output"
        );
    }

    /// Smoke test: upsert vectors, search, assert top hit is closest.
    #[tokio::test]
    async fn in_memory_vector_store_upsert_and_search() {
        let store = InMemoryVectorStore::new();

        // Three unit vectors in different directions.
        store
            .upsert("a", "north", vec![1.0, 0.0, 0.0])
            .await
            .unwrap();
        store
            .upsert("b", "east", vec![0.0, 1.0, 0.0])
            .await
            .unwrap();
        store.upsert("c", "up", vec![0.0, 0.0, 1.0]).await.unwrap();

        // Query vector close to "north".
        let query = vec![0.99, 0.1, 0.0];
        let hits = store.search(&query, 1).await.unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].id, "a",
            "top hit should be 'north' (closest to query)"
        );
        assert!(hits[0].score > 0.9, "cosine similarity should be high");
    }
}
