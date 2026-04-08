// ── Vector store — semantic similarity storage abstraction ───────────────────

use async_trait::async_trait;
use tokio::sync::RwLock;

/// A search result returned by `VectorStore::search`.
pub struct VectorHit {
    pub id: String,
    pub text: String,
    pub score: f32,
}

/// Semantic vector storage: upsert, search, and delete.
///
/// Implement this to back `SemanticMemoryTool` with your preferred vector
/// database (pgvector, Pinecone, Qdrant, Weaviate, etc.).
#[async_trait]
pub trait VectorStore: Send + Sync + 'static {
    async fn upsert(&self, id: &str, text: &str, vector: Vec<f32>) -> anyhow::Result<()>;
    async fn search(&self, query_vector: &[f32], limit: usize) -> anyhow::Result<Vec<VectorHit>>;
    async fn delete(&self, id: &str) -> anyhow::Result<()>;
}

// ── InMemoryVectorStore ──────────────────────────────────────────────────────

struct Entry {
    id: String,
    text: String,
    vector: Vec<f32>,
}

/// Brute-force in-memory vector store for development and testing.
#[derive(Default)]
pub struct InMemoryVectorStore {
    entries: RwLock<Vec<Entry>>,
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
        }
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn upsert(&self, id: &str, text: &str, vector: Vec<f32>) -> anyhow::Result<()> {
        let mut entries = self.entries.write().await;
        entries.retain(|e| e.id != id);
        entries.push(Entry {
            id: id.to_string(),
            text: text.to_string(),
            vector,
        });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn in_memory_vector_store_upsert_and_search() {
        let store = InMemoryVectorStore::new();
        store
            .upsert("a", "north", vec![1.0, 0.0, 0.0])
            .await
            .unwrap();
        store
            .upsert("b", "east", vec![0.0, 1.0, 0.0])
            .await
            .unwrap();
        store.upsert("c", "up", vec![0.0, 0.0, 1.0]).await.unwrap();

        let query = vec![0.99, 0.1, 0.0];
        let hits = store.search(&query, 1).await.unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "a");
        assert!(hits[0].score > 0.9);
    }
}
