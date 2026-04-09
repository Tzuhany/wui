// ============================================================================
// ToolSearch — implementation of the built-in `tool_search` discovery tool.
//
// When an agent has deferred tools, their full schemas are not in the initial
// system prompt. The LLM knows they exist (from the system prompt listing)
// but needs to call `tool_search` to get their schema before it can use them.
//
// This is the "目录常驻，正文按需" principle: the table of contents is
// always visible; the full content is fetched on demand.
//
// `ToolSearch` is automatically injected by AgentBuilder when deferred tools
// or catalogs are present. You never need to add it manually.
// ============================================================================

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use wui_core::tool::{Tool, ToolCtx, ToolInput, ToolMeta, ToolOutput};

use crate::catalog::{search_catalogs, ToolCatalog};

/// Built-in tool that lets the LLM discover and load deferred tool schemas.
pub struct ToolSearch {
    deferred: Vec<Arc<dyn Tool>>,
    catalogs: Vec<Arc<dyn ToolCatalog>>,
    catalog_results_limit: usize,
}

impl ToolSearch {
    pub fn new(deferred: Vec<Arc<dyn Tool>>, catalogs: Vec<Arc<dyn ToolCatalog>>) -> Self {
        Self {
            deferred,
            catalogs,
            catalog_results_limit: 5,
        }
    }

    /// Set the maximum number of results returned from catalog searches.
    pub fn with_catalog_limit(mut self, n: usize) -> Self {
        self.catalog_results_limit = n;
        self
    }
}

#[async_trait]
impl Tool for ToolSearch {
    fn name(&self) -> &str {
        "tool_search"
    }

    fn description(&self) -> &str {
        "Load the full schema for a deferred tool or search for tools in connected catalogs"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Tool name or keyword to search for"
                }
            },
            "required": ["query"]
        })
    }

    fn meta(&self, _input: &Value) -> ToolMeta {
        ToolMeta {
            readonly: true,
            concurrent: true,
            ..ToolMeta::default()
        }
    }

    async fn call(&self, input: Value, _ctx: &ToolCtx) -> ToolOutput {
        let inp = ToolInput(&input);
        let query = match inp.required_str("query") {
            Ok(q) => q.to_lowercase(),
            Err(e) => return ToolOutput::error(e),
        };

        // Search deferred (registered) tools by name and description.
        let deferred_matches: Vec<&Arc<dyn Tool>> = self
            .deferred
            .iter()
            .filter(|t| {
                t.name().to_lowercase().contains(&query)
                    || t.description().to_lowercase().contains(&query)
            })
            .collect();

        // Search external catalogs.
        let catalog_hits =
            search_catalogs(&self.catalogs, &query, self.catalog_results_limit).await;

        if deferred_matches.is_empty() && catalog_hits.is_empty() {
            let available = self
                .deferred
                .iter()
                .map(|t| t.name())
                .collect::<Vec<_>>()
                .join(", ");
            let catalog_names: Vec<_> = self.catalogs.iter().map(|c| c.name()).collect();
            let catalog_info = if catalog_names.is_empty() {
                String::new()
            } else {
                format!("\n\nSearched catalogs: {}", catalog_names.join(", "))
            };
            return ToolOutput::success(format!(
                "No tools found matching '{query}'.\n\nAvailable deferred tools: {available}{catalog_info}"
            ));
        }

        let mut sections: Vec<String> = Vec::new();
        let mut exposed_tools: Vec<Arc<dyn Tool>> = deferred_matches
            .iter()
            .map(|tool| Arc::clone(tool))
            .collect();

        // Format deferred tool results.
        for t in &deferred_matches {
            let schema = serde_json::to_string_pretty(&t.input_schema()).unwrap_or_default();
            sections.push(format!(
                "## {name}\n{desc}\n\n**Input schema:**\n```json\n{schema}\n```",
                name = t.name(),
                desc = t.description(),
            ));
        }

        // Format catalog results and collect tools to expose.
        for hit in &catalog_hits {
            let t = &hit.tool;
            let schema = serde_json::to_string_pretty(&t.input_schema()).unwrap_or_default();
            sections.push(format!(
                "## {name} (from catalog, score: {score:.2})\n{desc}\n\n**Input schema:**\n```json\n{schema}\n```",
                name  = t.name(),
                desc  = t.description(),
                score = hit.score,
            ));
            exposed_tools.push(Arc::clone(&hit.tool));
        }

        let result = sections.join("\n\n---\n\n");

        ToolOutput::success(result).expose(exposed_tools)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use serde_json::{json, Value};
    use tokio_util::sync::CancellationToken;

    use super::ToolSearch;
    use wui_core::tool::{Tool, ToolCtx};

    struct BareTool;

    #[async_trait]
    impl Tool for BareTool {
        fn name(&self) -> &str {
            "bare_tool"
        }
        fn description(&self) -> &str {
            "A tool without extra docs."
        }
        fn input_schema(&self) -> Value {
            json!({"type":"object","properties":{}})
        }
        async fn call(&self, _input: Value, _ctx: &ToolCtx) -> wui_core::tool::ToolOutput {
            wui_core::tool::ToolOutput::success("ok")
        }
    }

    #[tokio::test]
    async fn deferred_tool_schema_is_shown() {
        let search = ToolSearch::new(vec![Arc::new(BareTool)], vec![]);
        let ctx = ToolCtx {
            cancel: CancellationToken::new(),
            messages: Arc::<[wui_core::message::Message]>::from(Vec::new()),
            on_progress: Box::new(|_| {}),
        };

        let output = search.call(json!({"query":"bare"}), &ctx).await;

        assert!(output.content.contains("## bare_tool"));
        assert!(output.content.contains("**Input schema:**"));
    }
}
