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

mod keyword_store;
mod tools;
mod traits;
mod vector;

// ── Error type (lives here, used by traits and tools) ────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("entry not found: {0}")]
    NotFound(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

// ── Re-exports ───────────────────────────────────────────────────────────────

pub use keyword_store::InMemoryStore;
pub use tools::{all_memory_tools, memory_tools, MemoryTools};
pub use traits::{ForgetBackend, MemoryHit, MemoryRef, NewMemory, RecallBackend, RememberBackend};
pub use vector::{EmbedFn, InMemoryVectorStore, SemanticMemoryTool, VectorHit, VectorStore};
