// ============================================================================
// Catalog system — ToolCatalog trait + built-in implementations.
//
// ToolCatalog is a tool-discovery abstraction that lives here (in wui) rather
// than in wui-core, because it is a product/discovery concern rather than a
// vocabulary type. wui-core defines what tools *are*; wui defines how they
// are discovered and loaded.
// ============================================================================

use std::sync::Arc;

use wui_core::tool::Tool;

// ── ToolCatalog ───────────────────────────────────────────────────────────────

/// A lazily-loaded, searchable collection of tools.
///
/// Catalogs are not listed in the agent's initial prompt. The LLM discovers
/// them by calling `ToolSearch`, which queries all registered catalogs and
/// dynamically injects matching tools into the active tool set.
///
/// # Example
///
/// ```rust,ignore
/// Agent::builder(provider)
///     .catalog(McpCatalog::stdio("npx", &["-y", "@mcp/filesystem"]).namespace("fs"))
///     .build()
/// ```
#[async_trait::async_trait]
pub trait ToolCatalog: Send + Sync + 'static {
    /// Unique name for this catalog source (used in diagnostics and namespacing).
    fn name(&self) -> &str;

    /// Search for tools matching a natural language query.
    ///
    /// Implementations connect and load lazily on the first call.
    /// Results are ranked by relevance; at most `limit` are returned.
    async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<CatalogHit>>;
}

/// A tool returned from a [`ToolCatalog`] search, with a relevance score.
pub struct CatalogHit {
    /// The tool that matched the query.
    pub tool: Arc<dyn Tool>,
    /// Relevance score in `[0.0, 1.0]`. Higher is more relevant.
    pub score: f32,
}

// ── StaticCatalog ─────────────────────────────────────────────────────────────

/// A [`ToolCatalog`] backed by a fixed, in-memory list of tools.
///
/// Useful for testing and for registering a known set of tools that should
/// be discoverable via `ToolSearch` without paying the upfront schema cost.
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

// ── SearchStrategy ────────────────────────────────────────────────────────────

/// Ranks tool candidates against a natural language query.
///
/// The built-in [`TokenOverlapStrategy`] works well for up to ~500 tools
/// with no external dependencies. For larger catalogs, implement this trait
/// with an embedding-based approach.
pub trait SearchStrategy: Send + Sync + 'static {
    /// Score each `(name, description)` pair against `query`.
    ///
    /// Returns `(index, score)` pairs **sorted by score descending**.
    /// Indices refer to positions in `candidates`.
    fn rank(&self, query: &str, candidates: &[(&str, &str)]) -> Vec<(usize, f32)>;
}

// ── TokenOverlapStrategy ──────────────────────────────────────────────────────

/// Simple token-overlap search strategy. Zero dependencies.
///
/// Tokenises both query and candidate text by splitting on whitespace and
/// punctuation, then scores by Jaccard similarity of the token sets.
/// Sufficient for catalogs up to ~500 tools.
#[derive(Clone, Default)]
pub struct TokenOverlapStrategy;

impl SearchStrategy for TokenOverlapStrategy {
    fn rank(&self, query: &str, candidates: &[(&str, &str)]) -> Vec<(usize, f32)> {
        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
            return vec![];
        }

        let mut scored: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .filter_map(|(i, (name, desc))| {
                // Tokenize name and desc separately to avoid allocating a joined
                // String. The union of both token sets equals tokenize("name desc").
                let mut tokens = tokenize(name);
                tokens.extend(tokenize(desc));
                let score = jaccard(&query_tokens, &tokens);
                if score > 0.0 {
                    Some((i, score))
                } else {
                    None
                }
            })
            .collect();

        // Jaccard scores are always in [0.0, 1.0] — never NaN.
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored
    }
}

fn tokenize(s: &str) -> std::collections::HashSet<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 1)
        .map(|t| t.to_lowercase())
        .collect()
}

fn jaccard(a: &std::collections::HashSet<String>, b: &std::collections::HashSet<String>) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    // Use inclusion-exclusion to avoid materialising the union iterator:
    //   |A ∪ B| = |A| + |B| - |A ∩ B|
    let intersection = a.intersection(b).count();
    let union = a.len() + b.len() - intersection;
    if union == 0 {
        return 0.0;
    }
    intersection as f32 / union as f32
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
