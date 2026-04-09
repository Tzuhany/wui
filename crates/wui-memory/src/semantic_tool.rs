// ── SemanticMemoryTool — agent-facing adapter for vector stores ──────────────

use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
use serde_json::json;

use wui_core::tool::{Tool, ToolCtx, ToolInput, ToolOutput};

use super::vector_store::VectorStore;

/// Type alias for an async embedding function.
pub type EmbedFn = Arc<dyn Fn(String) -> BoxFuture<'static, Vec<f32>> + Send + Sync>;

/// A tool that stores and retrieves text via semantic (vector) similarity.
///
/// ```rust,ignore
/// let tool = SemanticMemoryTool::new(store, embed, 5);
/// ```
pub struct SemanticMemoryTool {
    store: Arc<dyn VectorStore>,
    embed: EmbedFn,
    max_results: usize,
}

impl SemanticMemoryTool {
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
            "upsert" => self.handle_upsert(&inp, ctx).await,
            "search" => self.handle_search(&inp, ctx).await,
            "delete" => self.handle_delete(&inp).await,
            other => ToolOutput::invalid_input(format!("unknown action: {other}")),
        }
    }
}

impl SemanticMemoryTool {
    async fn handle_upsert(&self, inp: &ToolInput<'_>, ctx: &ToolCtx) -> ToolOutput {
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

    async fn handle_search(&self, inp: &ToolInput<'_>, ctx: &ToolCtx) -> ToolOutput {
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

    async fn handle_delete(&self, inp: &ToolInput<'_>) -> ToolOutput {
        let id = match inp.required_str("id") {
            Ok(v) => v,
            Err(e) => return ToolOutput::invalid_input(e),
        };
        match self.store.delete(id).await {
            Ok(()) => ToolOutput::success(format!("Deleted memory id={id}.")),
            Err(e) => ToolOutput::error(format!("delete failed: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::vector_store::InMemoryVectorStore;

    /// Trivial embedding: hash the string into a fixed-length vector.
    fn trivial_embed(text: String) -> BoxFuture<'static, Vec<f32>> {
        Box::pin(async move {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            text.hash(&mut hasher);
            let h = hasher.finish();

            // Spread the hash bits into a 4-dimensional vector.
            vec![
                (h & 0xFFFF) as f32,
                ((h >> 16) & 0xFFFF) as f32,
                ((h >> 32) & 0xFFFF) as f32,
                ((h >> 48) & 0xFFFF) as f32,
            ]
        })
    }

    fn make_ctx() -> ToolCtx {
        ToolCtx {
            cancel: tokio_util::sync::CancellationToken::new(),
            messages: Arc::from(vec![]),
            spawn_depth: 0,
            on_progress: Box::new(|_| {}),
        }
    }

    #[tokio::test]
    async fn semantic_tool_upsert_and_search() {
        let store = Arc::new(InMemoryVectorStore::new());
        let embed: EmbedFn = Arc::new(trivial_embed);
        let tool = SemanticMemoryTool::new(store, embed, 5);
        let ctx = make_ctx();

        // Upsert a few items.
        let result = tool
            .call(
                json!({"action": "upsert", "id": "item1", "text": "Rust programming language"}),
                &ctx,
            )
            .await;
        assert!(
            result.content.contains("Stored"),
            "upsert should succeed, got: {}",
            result.content
        );

        let result = tool
            .call(
                json!({"action": "upsert", "id": "item2", "text": "Python programming language"}),
                &ctx,
            )
            .await;
        assert!(result.content.contains("Stored"));

        let result = tool
            .call(
                json!({"action": "upsert", "id": "item3", "text": "Go programming language"}),
                &ctx,
            )
            .await;
        assert!(result.content.contains("Stored"));

        // Search — the trivial embed is deterministic, so the same text
        // should return itself as the top hit.
        let result = tool
            .call(
                json!({"action": "search", "text": "Rust programming language", "limit": 2}),
                &ctx,
            )
            .await;
        assert!(
            result.content.contains("item1"),
            "expected item1 in search results, got: {}",
            result.content
        );

        // Delete and verify it's gone.
        let result = tool
            .call(json!({"action": "delete", "id": "item1"}), &ctx)
            .await;
        assert!(result.content.contains("Deleted"));

        // Search again — item1 should no longer appear.
        let result = tool
            .call(
                json!({"action": "search", "text": "Rust programming language", "limit": 5}),
                &ctx,
            )
            .await;
        assert!(
            !result.content.contains("item1"),
            "item1 should have been deleted, got: {}",
            result.content
        );
    }
}
