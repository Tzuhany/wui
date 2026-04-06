// ============================================================================
// ToolSearch — the deferred tool discovery tool.
//
// When an agent has deferred tools (tools with defer_loading() = true),
// their full schemas are not in the initial system prompt. The LLM knows
// they exist (from the system prompt listing) but needs to call ToolSearch
// to get their schema before it can use them.
//
// This is the "目录常驻，正文按需" principle: the table of contents is
// always visible; the full content is fetched on demand.
//
// ToolSearch is automatically injected by AgentBuilder when deferred tools
// are present. You never need to add it manually.
// ============================================================================

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use wuhu_core::tool::{Tool, ToolCtx, ToolInput, ToolOutput};

/// Built-in tool that lets the LLM discover and load deferred tool schemas.
pub struct ToolSearch {
    deferred: Vec<Arc<dyn Tool>>,
}

impl ToolSearch {
    pub fn new(deferred: Vec<Arc<dyn Tool>>) -> Self {
        Self { deferred }
    }
}

#[async_trait]
impl Tool for ToolSearch {
    fn name(&self) -> &str { "ToolSearch" }

    fn description(&self) -> &str {
        "Load the full schema and instructions for a deferred tool"
    }

    fn prompt(&self) -> String {
        "Search for and load a deferred tool's complete schema and usage instructions. \
        Call this before using any tool listed in the 'Additional tools' section of \
        the system prompt. Provide the tool name or a keyword describing what you need."
        .to_string()
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

    fn is_readonly(&self) -> bool { true }

    async fn call(&self, input: Value, _ctx: &ToolCtx) -> ToolOutput {
        let inp   = ToolInput(&input);
        let query = match inp.required_str("query") {
            Ok(q)  => q.to_lowercase(),
            Err(e) => return ToolOutput::error(e),
        };

        let matches: Vec<&Arc<dyn Tool>> = self.deferred.iter()
            .filter(|t| {
                t.name().to_lowercase().contains(&query)
                    || t.description().to_lowercase().contains(&query)
                    || t.search_hint()
                        .map(|h| h.to_lowercase().contains(&query))
                        .unwrap_or(false)
            })
            .collect();

        if matches.is_empty() {
            let available = self.deferred.iter()
                .map(|t| t.name())
                .collect::<Vec<_>>()
                .join(", ");
            return ToolOutput::success(format!(
                "No tools found matching '{query}'.\n\nAvailable deferred tools: {available}"
            ));
        }

        let result = matches.iter()
            .map(|t| {
                let schema = serde_json::to_string_pretty(&t.input_schema())
                    .unwrap_or_default();
                format!(
                    "## {name}\n{desc}\n\n{prompt}\n\n**Input schema:**\n```json\n{schema}\n```",
                    name   = t.name(),
                    desc   = t.description(),
                    prompt = t.prompt(),
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");

        ToolOutput::success(result)
    }
}
