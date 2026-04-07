# wui-memory

Reference memory capabilities built on `wui` tools — not a replacement memory architecture.

Optional memory extension for Wui agents. Provides keyword-based and semantic (vector) recall/remember/forget tools backed by pluggable storage traits. `InMemoryStore` ships as a reference backend for development; production backends implement the traits against your own storage layer.

## Install

```toml
[dependencies]
wui-memory = "0.1"
```

## Usage

### Keyword memory (substring search)

```rust
use std::sync::Arc;
use wui_memory::{InMemoryStore, all_memory_tools, memory_tools};

// Quick start — all three tools from one store.
let store = Arc::new(InMemoryStore::new());
let agent = Agent::builder(provider)
    .tools(all_memory_tools(store))
    .build();

// Fine-grained — pick only the capabilities you need.
let tools = memory_tools()
    .with_recall(store.clone())
    .with_remember(store.clone())
    .build();
```

### Semantic memory (vector similarity)

```rust
use std::sync::Arc;
use wui_memory::{InMemoryVectorStore, SemanticMemoryTool, EmbedFn};

let store: Arc<InMemoryVectorStore> = Arc::new(InMemoryVectorStore::new());
let embed: EmbedFn = Arc::new(|text: String| {
    Box::pin(async move { my_embed_api(text).await })
});

let agent = Agent::builder(provider)
    .tool(SemanticMemoryTool::new(store, embed, 5))
    .build();
```

## Backend traits

Implement these to use your own storage layer (pgvector, Redis, SQLite, a remote API):

| Trait | Tool exposed | Operation |
|-------|-------------|-----------|
| `RecallBackend` | `memory_recall` | Search by query string |
| `RememberBackend` | `memory_remember` | Write a new entry |
| `ForgetBackend` | `memory_forget` | Signal "stop surfacing this" |
| `VectorStore` | (used by `SemanticMemoryTool`) | Upsert / search / delete vectors |

Full docs: https://github.com/Tzuhany/wui
