use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use wui_core::tool::{Tool, ToolCtx, ToolInput, ToolMeta, ToolOutput};

use crate::{ForgetBackend, InMemoryStore, NewMemory, RecallBackend, RememberBackend};

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

macro_rules! memory_tool {
    ($name:ident, $backend:ident) => {
        struct $name {
            backend: Arc<dyn $backend>,
        }

        impl $name {
            pub(crate) fn new(b: Arc<dyn $backend>) -> Self {
                Self { backend: b }
            }
        }
    };
}

memory_tool!(RecallTool, RecallBackend);
memory_tool!(RememberTool, RememberBackend);
memory_tool!(ForgetTool, ForgetBackend);

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
