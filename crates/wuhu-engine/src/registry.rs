// ============================================================================
// Tool Registry — the agent's available capabilities.
//
// A flat map from tool name to implementation. The resident/deferred split
// is expressed through Tool::defer_loading(), not through separate maps.
//
// Resident tools: full schema in the LLM's system prompt. Used for the tools
// the LLM will call most often. The schema cost is paid upfront.
//
// Deferred tools: only name + description in the prompt. The LLM calls
// ToolSearch to retrieve the full schema before using them. Saves token cost
// for large tool libraries where 30+ schemas would bloat the context.
//
// All tools (resident + deferred) are stored in the same map for uniform
// execution lookup. The distinction only matters when building the system prompt.
// ============================================================================

use std::collections::HashMap;
use std::sync::Arc;

use wuhu_core::provider::ToolDef;
use wuhu_core::tool::Tool;

pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

/// A brief entry for a deferred tool — enough to appear in the system prompt.
#[derive(Debug, Clone)]
pub struct DeferredEntry {
    pub name:        String,
    pub description: String,
    /// Optional 3–10 word hint for ToolSearch.
    pub hint:        Option<String>,
}

impl ToolRegistry {
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Self {
        Self {
            tools: tools.into_iter().map(|t| (t.name().to_string(), t)).collect(),
        }
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// `ToolDef`s for all resident (non-deferred) tools, sorted by name.
    ///
    /// Sorted so the LLM sees a deterministic tool order regardless of
    /// registration order or HashMap iteration. Non-deterministic ordering
    /// can subtly affect which tool the LLM gravitates toward.
    pub fn tool_defs(&self) -> Vec<ToolDef> {
        let mut defs: Vec<ToolDef> = self.tools
            .values()
            .filter(|t| !t.defer_loading())
            .map(|t| ToolDef::from_tool(t.as_ref()))
            .collect();
        defs.sort_by(|a, b| a.name.cmp(&b.name));
        defs
    }

    /// A `ToolDef` for a specific tool by name (resident or deferred).
    pub fn tool_def(&self, name: &str) -> Option<ToolDef> {
        self.tools.get(name).map(|t| ToolDef::from_tool(t.as_ref()))
    }

    /// Brief entries for all deferred tools, sorted by name.
    ///
    /// Used to build the "Additional tools" section of the system prompt
    /// so the LLM knows these tools exist and can call ToolSearch to learn more.
    pub fn deferred_entries(&self) -> Vec<DeferredEntry> {
        let mut entries: Vec<DeferredEntry> = self.tools
            .values()
            .filter(|t| t.defer_loading())
            .map(|t| DeferredEntry {
                name:        t.name().to_string(),
                description: t.description().to_string(),
                hint:        t.search_hint().map(|s| s.to_string()),
            })
            .collect();
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        entries
    }

    /// Whether any deferred tools are registered.
    pub fn has_deferred(&self) -> bool {
        self.tools.values().any(|t| t.defer_loading())
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.keys().map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}
