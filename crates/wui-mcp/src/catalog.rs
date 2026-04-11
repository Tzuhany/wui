// ============================================================================
// McpCatalog — lazily-connected MCP server as a ToolCatalog.
//
// The MCP server is NOT connected at construction time. The first call to
// `search()` establishes the connection, fetches the full tool list, and
// caches it. Subsequent searches reuse the cached tools.
//
// Tool names are optionally prefixed with a namespace to prevent collisions
// when multiple MCP servers expose tools with the same name.
// ============================================================================

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex;

use wui_core::catalog::{CatalogHit, SearchStrategy, ToolCatalog};
use wui_core::tool::{Tool, ToolCtx, ToolMeta, ToolOutput};

use crate::{McpClient, McpError};

// ── State ─────────────────────────────────────────────────────────────────────

enum CatalogState {
    Idle {
        command: String,
        args: Vec<String>,
        transport: TransportKind,
    },
    Ready {
        tools: Vec<Arc<dyn Tool>>,
    },
}

enum TransportKind {
    Stdio,
    Http(String),
}

// ── McpCatalog ────────────────────────────────────────────────────────────────

/// A [`ToolCatalog`] backed by an MCP server.
///
/// The server process is not started until the first `search()` call
/// (lazy initialisation). The connection is then cached for subsequent
/// searches.
///
/// # Example
///
/// ```rust,ignore
/// Agent::builder(provider)
///     .catalog(McpCatalog::stdio("npx", &["-y", "@mcp/filesystem"]).namespace("fs"))
///     .build()
/// ```
pub struct McpCatalog {
    catalog_name: String,
    namespace: Option<String>,
    state: Mutex<CatalogState>,
    searcher: Arc<dyn SearchStrategy>,
}

impl McpCatalog {
    /// Create a catalog backed by an MCP server spawned via stdio.
    pub fn stdio(command: impl Into<String>, args: &[impl AsRef<str>]) -> Self {
        let command = command.into();
        let args = args.iter().map(|a| a.as_ref().to_string()).collect();
        Self {
            catalog_name: command.clone(),
            namespace: None,
            state: Mutex::new(CatalogState::Idle {
                command,
                args,
                transport: TransportKind::Stdio,
            }),
            searcher: Arc::new(wui_core::catalog::TokenOverlapStrategy),
        }
    }

    /// Create a catalog backed by an MCP server accessible over HTTP.
    pub fn http(url: impl Into<String>) -> Self {
        let url = url.into();
        Self {
            catalog_name: url.clone(),
            namespace: None,
            state: Mutex::new(CatalogState::Idle {
                command: url.clone(),
                args: vec![],
                transport: TransportKind::Http(url),
            }),
            searcher: Arc::new(wui_core::catalog::TokenOverlapStrategy),
        }
    }

    /// Prefix all tool names from this catalog with `ns__`.
    ///
    /// Prevents name collisions when multiple catalogs expose tools with
    /// the same name (e.g., two MCP servers both have `read_file`).
    pub fn namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = Some(ns.into());
        self
    }

    /// Override the search strategy (default: `TokenOverlapStrategy`).
    pub fn with_searcher(mut self, s: impl SearchStrategy + 'static) -> Self {
        self.searcher = Arc::new(s);
        self
    }

    fn apply_namespace(&self, tools: Vec<Arc<dyn Tool>>) -> Vec<Arc<dyn Tool>> {
        match &self.namespace {
            None => tools,
            Some(ns) => tools
                .into_iter()
                .map(|t| {
                    let namespaced_name = format!("{}_{}", ns, t.name());
                    Arc::new(NamespacedTool {
                        namespaced_name,
                        inner: t,
                    }) as Arc<dyn Tool>
                })
                .collect(),
        }
    }

    async fn ensure_ready(&self) -> anyhow::Result<()> {
        let mut state = self.state.lock().await;

        if matches!(&*state, CatalogState::Ready { .. }) {
            return Ok(());
        }

        let (command, args, transport) = match &*state {
            CatalogState::Idle {
                command,
                args,
                transport,
            } => (
                command.clone(),
                args.clone(),
                match transport {
                    TransportKind::Stdio => TransportKind::Stdio,
                    TransportKind::Http(u) => TransportKind::Http(u.clone()),
                },
            ),
            CatalogState::Ready { .. } => unreachable!(),
        };

        let client = match &transport {
            TransportKind::Stdio => McpClient::stdio(&command, &args)
                .await
                .map_err(|e: McpError| anyhow::anyhow!("{e}"))?,
            TransportKind::Http(u) => McpClient::http(u.as_str())
                .await
                .map_err(|e: McpError| anyhow::anyhow!("{e}"))?,
        };

        let raw_tools = client
            .into_tools()
            .await
            .map_err(|e: McpError| anyhow::anyhow!("{e}"))?;
        let tools = self.apply_namespace(raw_tools);
        *state = CatalogState::Ready { tools };

        Ok(())
    }
}

#[async_trait]
impl ToolCatalog for McpCatalog {
    fn name(&self) -> &str {
        &self.catalog_name
    }

    async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<CatalogHit>> {
        self.ensure_ready().await?;

        let state = self.state.lock().await;
        let CatalogState::Ready { tools } = &*state else {
            unreachable!()
        };

        let candidates: Vec<(&str, &str)> =
            tools.iter().map(|t| (t.name(), t.description())).collect();

        let ranked = self.searcher.rank(query, &candidates);

        Ok(ranked
            .into_iter()
            .take(limit)
            .map(|(idx, score)| CatalogHit {
                tool: Arc::clone(&tools[idx]),
                score,
            })
            .collect())
    }
}

// ── NamespacedTool ────────────────────────────────────────────────────────────

struct NamespacedTool {
    namespaced_name: String,
    inner: Arc<dyn Tool>,
}

#[async_trait]
impl Tool for NamespacedTool {
    fn name(&self) -> &str {
        &self.namespaced_name
    }
    fn description(&self) -> &str {
        self.inner.description()
    }
    fn input_schema(&self) -> Value {
        self.inner.input_schema()
    }
    fn meta(&self, input: &Value) -> ToolMeta {
        self.inner.meta(input)
    }

    async fn call(&self, input: Value, ctx: &ToolCtx) -> ToolOutput {
        self.inner.call(input, ctx).await
    }
}
