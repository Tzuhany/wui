//! 01 — MCP server as a tool source.
//!
//! Demonstrates connecting a live MCP server to a wui agent. The MCP tools
//! become first-class wui Tools — the agent uses them identically to any
//! other tool. Transport and protocol details stay behind the bridge.
//!
//! ⚠  This example requires a running MCP server. Swap the command for any
//!    server you have installed, e.g.:
//!
//!      uvx mcp-server-filesystem /tmp          (Python, filesystem)
//!      npx @modelcontextprotocol/server-github  (Node.js, GitHub)
//!      cargo run --bin my-mcp-server            (your own Rust server)
//!
//!   cargo run --example 01_mcp
//!
//! Without a server, the program exits immediately after printing the tool
//! listing — the assert at the end is what fails without a real connection.

use std::sync::Arc;

use futures::StreamExt;
use wui::prelude::*;
use wui::providers::Anthropic;
use wui_mcp::McpClient;

fn anthropic() -> Anthropic {
    let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
    match std::env::var("ANTHROPIC_BASE_URL") {
        Ok(url) => Anthropic::with_base_url(key, url),
        Err(_) => Anthropic::new(key),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== 01: MCP tool source ===");

    // ── Connect to an MCP server ──────────────────────────────────────────────
    //
    // `McpClient::stdio` spawns the process and completes the MCP handshake.
    // The process stays alive for as long as any tool derived from this client
    // is alive — no explicit teardown required.
    //
    // Swap the command for the MCP server you want to use.
    let client = McpClient::stdio("uvx", ["mcp-server-filesystem", "/tmp"]).await?;

    // Discover all tools the server exposes.
    let tools: Vec<Arc<dyn Tool>> = client.into_tools().await?;

    println!(
        "  MCP server exposes {} tool{}:",
        tools.len(),
        if tools.len() == 1 { "" } else { "s" }
    );
    for t in &tools {
        println!("    - {} : {}", t.name(), t.description());
    }

    assert!(
        !tools.is_empty(),
        "expected at least one tool from the MCP server"
    );

    // ── Use MCP tools in an agent ─────────────────────────────────────────────
    //
    // From here the agent has no idea these tools come from MCP — they are
    // plain `Arc<dyn Tool>` objects from its perspective.
    let agent = Agent::builder(anthropic())
        .system("You are a helpful assistant. Use the available tools to answer questions.")
        .model("claude-haiku-4-5-20251001")
        .tools(tools)
        .permission(PermissionMode::Auto)
        .build();

    let mut stream = agent.stream("List the files in /tmp.");
    let mut response = String::new();
    let mut used_mcp = false;

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::ToolStart { name, .. } => {
                println!("    → tool called: {name}");
                used_mcp = true;
            }
            AgentEvent::TextDelta(t) => response.push_str(&t),
            AgentEvent::Done(_) => break,
            AgentEvent::Error(e) => return Err(anyhow::anyhow!("agent error: {e}")),
            _ => {}
        }
    }

    assert!(used_mcp, "expected agent to call at least one MCP tool");
    assert!(!response.is_empty(), "expected a text response");
    println!("  response: {response}");

    println!("PASS");
    Ok(())
}
