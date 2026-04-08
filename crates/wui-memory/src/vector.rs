// ── Vector store ─────────────────────────────────────────────────────────────
//
// VectorStore provides semantic similarity search over dense float vectors.
// InMemoryVectorStore is a brute-force reference implementation suitable for
// development and small datasets. SemanticMemoryTool wraps any VectorStore
// and EmbeddingFn into a Tool the agent can call.

use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
use serde_json::json;
use tokio::sync::RwLock;

use wui_core::tool::{Tool, ToolCtx, ToolInput, ToolOutput};

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

#[cfg(test)]
mod tests {
    use super::*;

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
