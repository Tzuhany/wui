// ============================================================================
// Catalog system — built-in ToolCatalog implementations.
//
// The core traits (ToolCatalog, CatalogHit, SearchStrategy,
// TokenOverlapStrategy) live in wui-core so that extension crates
// (wui-skills, wui-mcp, ...) can implement them without depending on wui.
//
// This module provides the convenience implementations that need the full
// wui runtime: StaticCatalog and the multi-catalog search utility.
// ============================================================================

use std::sync::Arc;

use wui_core::tool::Tool;

// Re-export core catalog types so existing `use wui::catalog::…` paths
// continue to work without any changes.
pub use wui_core::catalog::{CatalogHit, SearchStrategy, TokenOverlapStrategy, ToolCatalog};

// ── StaticCatalog ─────────────────────────────────────────────────────────────

/// A [`ToolCatalog`] backed by a fixed, in-memory list of tools.
///
/// Useful for testing and for registering a known set of tools that should
/// be discoverable via `tool_search` without paying the upfront schema cost.
///
/// ```rust,ignore
/// Agent::builder(provider)
///     .catalog(StaticCatalog::new("my-tools", vec![
///         Arc::new(MyTool),
///         Arc::new(AnotherTool),
///     ]))
///     .build()
/// ```
pub struct StaticCatalog {
    name: String,
    tools: Vec<Arc<dyn Tool>>,
    searcher: Arc<dyn SearchStrategy>,
}

impl StaticCatalog {
    pub fn new(name: impl Into<String>, tools: Vec<Arc<dyn Tool>>) -> Self {
        Self {
            name: name.into(),
            tools,
            searcher: Arc::new(TokenOverlapStrategy),
        }
    }

    pub fn with_searcher(mut self, s: impl SearchStrategy + 'static) -> Self {
        self.searcher = Arc::new(s);
        self
    }
}

#[async_trait::async_trait]
impl ToolCatalog for StaticCatalog {
    fn name(&self) -> &str {
        &self.name
    }

    async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<CatalogHit>> {
        let candidates: Vec<(&str, &str)> = self
            .tools
            .iter()
            .map(|t| (t.name(), t.description()))
            .collect();

        let ranked = self.searcher.rank(query, &candidates);

        Ok(ranked
            .into_iter()
            .take(limit)
            .map(|(idx, score)| CatalogHit {
                tool: Arc::clone(&self.tools[idx]),
                score,
            })
            .collect())
    }
}

// ── Multi-catalog search ──────────────────────────────────────────────────────

/// Search all `catalogs` in parallel for `query`, merge results, deduplicate
/// by tool name, and return the top `limit` hits.
pub async fn search_catalogs(
    catalogs: &[Arc<dyn ToolCatalog>],
    query: &str,
    limit: usize,
) -> Vec<CatalogHit> {
    use futures::future::join_all;

    // Share the query string across all catalog futures without per-future clones.
    let q: Arc<str> = query.into();

    let futures: Vec<_> = catalogs
        .iter()
        .map(|c| {
            let c = Arc::clone(c);
            let q = Arc::clone(&q);
            async move { c.search(&q, limit).await.unwrap_or_default() }
        })
        .collect();

    let results = join_all(futures).await;

    // Merge, deduplicate by tool name (first occurrence wins — highest-scored
    // catalog wins since we iterate in parallel result order), sort by score.
    let mut seen = std::collections::HashSet::new();
    let mut merged: Vec<CatalogHit> = results
        .into_iter()
        .flatten()
        .filter(|hit| seen.insert(hit.tool.name().to_string()))
        .collect();

    // Scores are Jaccard values in [0.0, 1.0] — never NaN.
    merged.sort_by(|a, b| b.score.total_cmp(&a.score));
    merged.truncate(limit);
    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::{json, Value};
    use wui_core::tool::{ToolCtx, ToolOutput};

    struct DummyTool {
        name: &'static str,
        desc: &'static str,
    }

    #[async_trait]
    impl Tool for DummyTool {
        fn name(&self) -> &str {
            self.name
        }
        fn description(&self) -> &str {
            self.desc
        }
        fn input_schema(&self) -> Value {
            json!({"type": "object"})
        }
        async fn call(&self, _input: Value, _ctx: &ToolCtx) -> ToolOutput {
            ToolOutput::success("ok")
        }
    }

    fn dummy(name: &'static str, desc: &'static str) -> Arc<dyn Tool> {
        Arc::new(DummyTool { name, desc })
    }

    #[tokio::test]
    async fn static_catalog_search_returns_results() {
        let cat = StaticCatalog::new(
            "test",
            vec![
                dummy("read_file", "Read a file from disk"),
                dummy("write_file", "Write content to a file"),
                dummy("search", "Search for text in files"),
            ],
        );
        let results = cat.search("file", 10).await.unwrap();
        assert!(!results.is_empty());
        // "read_file" and "write_file" both mention "file"
        let names: Vec<&str> = results.iter().map(|h| h.tool.name()).collect();
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"write_file"));
    }

    #[tokio::test]
    async fn static_catalog_search_respects_limit() {
        let cat = StaticCatalog::new(
            "test",
            vec![
                dummy("a", "tool a"),
                dummy("b", "tool b"),
                dummy("c", "tool c"),
            ],
        );
        let results = cat.search("tool", 2).await.unwrap();
        assert!(results.len() <= 2);
    }

    #[tokio::test]
    async fn multi_catalog_deduplicates() {
        let cat1 = Arc::new(StaticCatalog::new(
            "cat1",
            vec![dummy("shared", "shared tool")],
        )) as Arc<dyn ToolCatalog>;
        let cat2 = Arc::new(StaticCatalog::new(
            "cat2",
            vec![dummy("shared", "shared tool copy")],
        )) as Arc<dyn ToolCatalog>;

        let results = search_catalogs(&[cat1, cat2], "shared", 10).await;
        let names: Vec<&str> = results.iter().map(|h| h.tool.name()).collect();
        // "shared" should appear only once
        assert_eq!(names.iter().filter(|n| **n == "shared").count(), 1);
    }

    #[tokio::test]
    async fn multi_catalog_empty() {
        let results = search_catalogs(&[], "anything", 10).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn catalog_name() {
        let cat = StaticCatalog::new("my-catalog", vec![]);
        assert_eq!(cat.name(), "my-catalog");
    }
}
