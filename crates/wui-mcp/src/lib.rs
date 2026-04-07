// ============================================================================
// wui-mcp — bridge any MCP server into the Wui tool ecosystem.
//
// This crate is a companion, not the product. MCP is an external protocol;
// this crate translates between it and wui's Tool interface. The bridge is
// transparent — McpTool works exactly like any other Tool from the runtime's
// perspective. Whether you need MCP is your choice.
//
// What this crate is NOT:
//   - A claim that MCP is the right tool source for all agents.
//   - A replacement for writing native wui Tools.
//
// ── wui-mcp — MCP ecosystem, meet wui.
//
// Any tool from the Model Context Protocol ecosystem becomes a first-class
// wui Tool in two lines:
//
//   let tools = McpClient::stdio("uvx", ["mcp-server-filesystem", "/tmp"])
//       .await?
//       .into_tools()
//       .await?;
//
//   let agent = Agent::builder(provider).tools(tools).build();
//
// The bridge is transparent: McpTool delegates every call to the MCP server
// over the configured transport, respects wui cancellation, and surfaces
// MCP structured content alongside the LLM-facing text — all with zero
// changes to the rest of the wui stack.
//
// The MCP connection stays alive as long as any tool does — it is reference-
// counted behind an Arc. Drop the last tool; the process exits and the
// connection closes. No explicit teardown required.
//
// ── Transports ───────────────────────────────────────────────────────────────
//
//   McpClient::stdio(cmd, args)   — spawn a subprocess (uvx, npx, deno, …)
//   McpClient::http(url)          — connect to a remote Streamable HTTP server
//
// Both normalise to the same wui interface; the caller never needs to know
// which transport is in use.
// ============================================================================

pub mod catalog;
pub use catalog::McpCatalog;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use rmcp::model::{CallToolRequestParams, RawContent};
use rmcp::service::{ClientInitializeError, RunningService};
use rmcp::{RoleClient, ServiceError, ServiceExt};

use wui_core::tool::{Tool, ToolCtx, ToolOutput};

// ── Error ─────────────────────────────────────────────────────────────────────

/// Errors that can occur when connecting to or calling an MCP server.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    /// The MCP protocol handshake failed.
    ///
    /// Boxed: `ClientInitializeError` is large enough to inflate the enum
    /// significantly; boxing keeps all variants the same pointer size.
    #[error("MCP init: {0}")]
    Init(Box<ClientInitializeError>),
    /// A tool call or tool listing request failed.
    #[error("MCP service: {0}")]
    Service(#[from] ServiceError),
    /// The transport could not be created (e.g. the subprocess failed to start).
    #[error("transport: {0}")]
    Transport(String),
}

impl From<ClientInitializeError> for McpError {
    fn from(e: ClientInitializeError) -> Self {
        Self::Init(Box::new(e))
    }
}

// ── Internal alias ────────────────────────────────────────────────────────────

// The live rmcp connection. Arc-wrapped so all McpTools from the same server
// share ownership — the connection stays open as long as any tool is alive.
type SharedService = Arc<RunningService<RoleClient, ()>>;

// ── Client ────────────────────────────────────────────────────────────────────

/// A live connection to an MCP server.
///
/// Create via [`McpClient::stdio`] or [`McpClient::http`], then call
/// [`McpClient::into_tools`] to receive wui [`Tool`] objects ready for
/// `AgentBuilder`.
pub struct McpClient {
    service: RunningService<RoleClient, ()>,
}

impl McpClient {
    /// Connect to an MCP server by spawning a subprocess (stdio transport).
    ///
    /// The process is killed when the last tool from this client is dropped.
    ///
    /// ```rust,ignore
    /// let tools = McpClient::stdio("uvx", ["mcp-server-git", "--repository", "."])
    ///     .await?
    ///     .into_tools()
    ///     .await?;
    /// ```
    pub async fn stdio(
        command: impl AsRef<str>,
        args: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<Self, McpError> {
        use rmcp::transport::TokioChildProcess;
        use tokio::process::Command;

        let mut cmd = Command::new(command.as_ref());
        for arg in args {
            cmd.arg(arg.as_ref());
        }
        let transport =
            TokioChildProcess::new(cmd).map_err(|e| McpError::Transport(e.to_string()))?;
        let service = ().serve(transport).await?;
        Ok(Self { service })
    }

    /// Connect to a remote MCP server via Streamable HTTP.
    ///
    /// ```rust,ignore
    /// let tools = McpClient::http("http://localhost:8080/mcp")
    ///     .await?
    ///     .into_tools()
    ///     .await?;
    /// ```
    pub async fn http(url: impl Into<Arc<str>>) -> Result<Self, McpError> {
        use rmcp::transport::StreamableHttpClientTransport;

        // `from_uri` uses the default reqwest client bundled with rmcp —
        // no version conflict with the workspace reqwest.
        let transport = StreamableHttpClientTransport::from_uri(url);
        let service = ().serve(transport).await?;
        Ok(Self { service })
    }

    /// Discover all tools exposed by the MCP server and return them as wui
    /// [`Tool`] objects.
    ///
    /// The underlying connection is reference-counted across the returned
    /// tools — it stays open as long as at least one tool is alive.
    pub async fn into_tools(self) -> Result<Vec<Arc<dyn Tool>>, McpError> {
        let specs = self.service.list_all_tools().await?;
        let service = Arc::new(self.service);

        let tools = specs
            .into_iter()
            .map(|spec| {
                Arc::new(McpTool {
                    name: spec.name.to_string(),
                    description: spec.description.map(|d| d.to_string()).unwrap_or_default(),
                    schema: Value::Object((*spec.input_schema).clone()),
                    service: service.clone(),
                }) as Arc<dyn Tool>
            })
            .collect();

        Ok(tools)
    }
}

// ── Tool bridge ───────────────────────────────────────────────────────────────

// Not public: callers interact with the wui Tool trait, never with McpTool
// directly. The MCP protocol is an implementation detail of the bridge.
struct McpTool {
    name: String,
    description: String,
    schema: Value,
    service: SharedService,
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn input_schema(&self) -> Value {
        self.schema.clone()
    }

    async fn call(&self, input: Value, ctx: &ToolCtx) -> ToolOutput {
        let arguments = match input.as_object().cloned() {
            Some(map) => map,
            None => return ToolOutput::invalid_input("MCP tool input must be a JSON object"),
        };

        let params = CallToolRequestParams::new(self.name.clone()).with_arguments(arguments);

        let result = tokio::select! {
            r = self.service.call_tool(params) => r,
            _ = ctx.cancel.cancelled()         => return ToolOutput::error("cancelled"),
        };

        match result {
            Err(e) => ToolOutput::error(e.to_string()),
            Ok(result) => {
                let text = extract_text(&result.content);
                let base = if result.is_error.unwrap_or(false) {
                    ToolOutput::error(text)
                } else {
                    ToolOutput::success(text)
                };
                // MCP structured_content flows directly into wui's structured
                // output field — no adaptation needed on either side.
                match result.structured_content {
                    Some(s) => base.with_structured(s),
                    None => base,
                }
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Concatenate all text blocks from an MCP content list.
///
/// Non-text content (images, audio, resources) is skipped — tools that need
/// to surface binary outputs should return them as wui Artifacts instead.
fn extract_text(content: &[rmcp::model::Content]) -> String {
    content
        .iter()
        .filter_map(|c| match &c.raw {
            RawContent::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ── Smoke tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// `McpError` variants are `Debug` and display correctly.
    /// This test does not require a live MCP server.
    #[test]
    fn mcp_error_display() {
        let e = McpError::Transport("subprocess failed: no such file".to_string());
        let s = e.to_string();
        assert!(s.contains("transport"), "expected 'transport' in: {s}");
        assert!(s.contains("subprocess failed"), "expected message in: {s}");
    }

    /// `extract_text` concatenates multiple text blocks and skips non-text content.
    #[test]
    fn extract_text_joins_text_blocks() {
        use rmcp::model::{Annotated, RawContent, RawTextContent};

        let make_text = |s: &str| Annotated {
            raw: RawContent::Text(RawTextContent {
                text: s.to_string(),
                meta: None,
            }),
            annotations: None,
        };

        let content = vec![make_text("hello"), make_text("world")];
        let result = extract_text(&content);
        assert_eq!(result, "hello\nworld");
    }
}
