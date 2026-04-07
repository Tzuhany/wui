// ============================================================================
// SessionStore — turn-level conversation persistence.
//
// Stores and restores the message history at turn boundaries. Nothing more:
// session permissions (ApproveAlways / DenyAlways) are intentionally
// ephemeral and reset when a session is re-created from a stored snapshot.
// ============================================================================

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use wui_core::message::Message;

#[async_trait]
pub trait SessionStore: Send + Sync + 'static {
    async fn load(&self, session_id: &str) -> Result<Option<StoredSession>, SessionStoreError>;

    async fn save(
        &self,
        session_id: &str,
        session: &StoredSession,
    ) -> Result<(), SessionStoreError>;
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StoredSession {
    pub session_id: String,
    pub messages: Vec<Message>,
}

impl StoredSession {
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            messages: Vec::new(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SessionStoreError {
    #[error("serialization failed: {0}")]
    Serialize(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[derive(Clone)]
pub struct InMemorySessionStore {
    store: std::sync::Arc<tokio::sync::RwLock<std::collections::HashMap<String, StoredSession>>>,
}

impl InMemorySessionStore {
    pub fn new() -> Self {
        Self {
            store: std::sync::Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
        }
    }
}

impl Default for InMemorySessionStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SessionStore for InMemorySessionStore {
    async fn load(&self, session_id: &str) -> Result<Option<StoredSession>, SessionStoreError> {
        Ok(self.store.read().await.get(session_id).cloned())
    }

    async fn save(
        &self,
        session_id: &str,
        session: &StoredSession,
    ) -> Result<(), SessionStoreError> {
        self.store
            .write()
            .await
            .insert(session_id.to_string(), session.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{InMemorySessionStore, SessionStore, StoredSession};
    use wui_core::message::Message;

    #[tokio::test]
    async fn clone_shares_underlying_store() {
        let store = InMemorySessionStore::new();
        let clone = store.clone();

        let snapshot = StoredSession {
            session_id: "session-1".to_string(),
            messages: vec![Message::user("hello")],
        };

        store
            .save("session-1", &snapshot)
            .await
            .expect("save should succeed");

        let loaded = clone
            .load("session-1")
            .await
            .expect("load should succeed")
            .expect("session should exist");

        assert_eq!(loaded.session_id, "session-1");
        assert_eq!(loaded.messages.len(), 1);
    }
}
