// ── ToolCatalog ───────────────────────────────────────────────────────────────

use std::sync::Arc;

use crate::tool::Tool;

/// A lazily-loaded, searchable collection of tools.
///
/// Catalogs are not listed in the agent's initial prompt. The LLM discovers
/// them by calling the built-in `tool_search` tool, which queries all
/// registered catalogs and dynamically injects matching tools into the active
/// tool set.
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
    let intersection = a.intersection(b).count();
    let union = a.len() + b.len() - intersection;
    if union == 0 {
        return 0.0;
    }
    intersection as f32 / union as f32
}
