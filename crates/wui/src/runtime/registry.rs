// ============================================================================
// Tool Registry — the agent's available capabilities.
//
// Resident tools: full schema in the LLM's system prompt. Used for the tools
// the LLM will call most often. The schema cost is paid upfront.
//
// Deferred tools: only name + description in the prompt. The LLM calls the
// built-in `tool_search` tool to retrieve the full schema before using them.
// Saves token cost for large tool libraries where 30+ schemas would bloat the
// context.
//
// All tools (resident + deferred) are stored in the same map for uniform
// execution lookup. The distinction only matters when building the system prompt.
// ============================================================================

use std::collections::{hash_map::Entry, HashMap, HashSet};
use std::sync::Arc;

use wui_core::provider::ToolDef;
use wui_core::tool::Tool;

pub(crate) struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
    deferred_names: HashSet<String>,
}

/// A brief entry for a deferred tool — enough to appear in the system prompt.
#[derive(Debug, Clone)]
pub(crate) struct DeferredEntry {
    pub name: String,
    pub description: String,
}

impl ToolRegistry {
    /// Build a registry from separate resident and deferred tool lists.
    ///
    /// Resident tools appear in the initial prompt with full schemas.
    /// Deferred tools appear only as name + description; the LLM calls
    /// `tool_search` to fetch the full schema before using them.
    ///
    /// Both sets are stored in the same lookup map for execution.
    pub fn new(resident: Vec<Arc<dyn Tool>>, deferred: Vec<Arc<dyn Tool>>) -> Self {
        let mut map: HashMap<String, Arc<dyn Tool>> = HashMap::new();
        let mut deferred_names: HashSet<String> = HashSet::new();

        for t in resident {
            insert_unique(&mut map, t, "resident");
        }
        for t in deferred {
            let name = t.name().to_string();
            deferred_names.insert(name.clone());
            insert_unique(&mut map, t, "deferred");
        }

        Self {
            tools: map,
            deferred_names,
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
        let mut defs: Vec<ToolDef> = self
            .tools
            .iter()
            .filter(|(name, _)| !self.deferred_names.contains(*name))
            .map(|(_, t)| ToolDef::from_tool(t.as_ref()))
            .collect();
        defs.sort_by(|a, b| a.name.cmp(&b.name));
        defs
    }

    /// Brief entries for all deferred tools, sorted by name.
    ///
    /// Used to build the "Additional tools" section of the system prompt so
    /// the LLM knows these tools exist and can call `tool_search` to learn
    /// more.
    pub(crate) fn deferred_entries(&self) -> Vec<DeferredEntry> {
        let mut entries: Vec<DeferredEntry> = self
            .deferred_names
            .iter()
            .filter_map(|name| {
                self.tools.get(name).map(|t| DeferredEntry {
                    name: name.clone(),
                    description: t.description().to_string(),
                })
            })
            .collect();
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        entries
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Create a new registry that merges this one with additional dynamic tools.
    ///
    /// Dynamic tools are added at runtime (e.g., discovered via `tool_search`).
    /// They do not appear in the initial system prompt but can be called after
    /// being returned by `ToolOutput::expose`.
    pub fn with_dynamic(&self, extra: &HashMap<String, Arc<dyn Tool>>) -> Self {
        if extra.is_empty() {
            return Self {
                tools: self.tools.clone(),
                deferred_names: self.deferred_names.clone(),
            };
        }
        let mut merged = self.tools.clone();
        let mut deferred_names = self.deferred_names.clone();
        for (name, tool) in extra {
            deferred_names.remove(name);
            merged
                .entry(name.clone())
                .or_insert_with(|| Arc::clone(tool));
        }
        Self {
            tools: merged,
            deferred_names,
        }
    }
}

fn insert_unique(map: &mut HashMap<String, Arc<dyn Tool>>, tool: Arc<dyn Tool>, source: &str) {
    let name = tool.name().to_string();
    match map.entry(name.clone()) {
        Entry::Vacant(entry) => {
            entry.insert(tool);
        }
        Entry::Occupied(_) => {
            panic!("duplicate tool name '{name}' encountered while building {source} tools")
        }
    }
}
