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
