// ============================================================================
// Tool Registry — the agent's available capabilities.
//
// A flat map from tool name to implementation. Intentionally simple:
// the registry has no concept of scoping, versioning, or access control —
// those concerns live in the permission system and in Hook implementations.
// ============================================================================

use std::collections::HashMap;
use std::sync::Arc;

use wuhu_core::provider::ToolDef;
use wuhu_core::tool::Tool;

pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
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

    /// Produce `ToolDef`s for all non-deferred tools, sorted by name.
    ///
    /// Sorted so the LLM always sees the same tool order regardless of
    /// registration order or HashMap iteration order. Non-deterministic
    /// ordering can subtly affect which tool the LLM prefers to call.
    ///
    /// Deferred tools appear in the prompt by name/description only.
    /// Their full schema is fetched on demand via `ToolSearch`.
    pub fn tool_defs(&self) -> Vec<ToolDef> {
        let mut defs: Vec<ToolDef> = self.tools
            .values()
            .filter(|t| !t.defer_loading())
            .map(|t| ToolDef::from_tool(t.as_ref()))
            .collect();
        defs.sort_by(|a, b| a.name.cmp(&b.name));
        defs
    }

    /// Produce a `ToolDef` for a specific tool by name.
    pub fn tool_def(&self, name: &str) -> Option<ToolDef> {
        self.tools.get(name).map(|t| ToolDef::from_tool(t.as_ref()))
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
