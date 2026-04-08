// ── InMemoryStore ─────────────────────────────────────────────────────────────
//
// A reference backend for development and testing.
//
// This is deliberately simple: a `Vec` behind an `RwLock`, with substring
// matching for recall. It is not a recommendation about how memory should be
// structured in production. Swap it for any backend that implements the traits.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::{
    ForgetBackend, MemoryError, MemoryHit, MemoryRef, NewMemory, RecallBackend, RememberBackend,
};

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
