# wui-mcp

Bridges the MCP ecosystem into `wui`'s Tool/Catalog abstraction.

MCP (Model Context Protocol) adapter for Wui. Any MCP server becomes a set of first-class `wui::Tool` objects in two lines. Supports stdio (subprocess) and HTTP (Streamable HTTP) transports, and a `McpCatalog` for lazy, on-demand tool discovery.

## Install

```toml
[dependencies]
wui-mcp = "0.1"
```

## Usage

### Eager — tools always in the prompt

```rust
use wui_mcp::McpClient;

let tools = McpClient::stdio("uvx", ["mcp-server-filesystem", "/tmp"])
    .await?
    .into_tools()
    .await?;

let agent = Agent::builder(provider).tools(tools).build();
```

### Lazy catalog — tools discovered on first use

```rust
use wui_mcp::McpCatalog;

let agent = Agent::builder(provider)
    .catalog(
        McpCatalog::stdio("uvx", &["mcp-server-filesystem", "/tmp"])
            .namespace("fs")
    )
    .build();
```

Use catalogs when you have many MCP servers — they connect on first search and never appear in the initial prompt, so token cost grows only with what the agent actually uses.

### HTTP transport

```rust
let tools = McpClient::http("http://localhost:8080/mcp")
    .await?
    .into_tools()
    .await?;
```

The MCP connection is reference-counted across all tools from the same server. Drop the last tool and the connection closes automatically.

Full docs: https://github.com/Tzuhany/wui
