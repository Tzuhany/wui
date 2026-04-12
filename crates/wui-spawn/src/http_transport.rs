// ============================================================================
// HttpTransport — cross-process agent delegation over HTTP.
//
// Reference implementation of `AgentTransport` that communicates with a
// remote agent server over HTTP/JSON. The server must expose four endpoints:
//
//   POST   /send     { agent_name, prompt }     -> { id, agent_name }
//   GET    /status   ?id=...&agent_name=...     -> { status, ... }
//   GET    /result   ?id=...&agent_name=...     -> { status, output }
//   POST   /cancel   { id, agent_name }         -> 200 OK
//
// All request/response bodies use the same serde types as the transport
// layer (RemoteJobHandle, RemoteAgentStatus, RemoteAgentResult), so a
// server can deserialize directly into the framework's types.
//
// ── Usage ────────────────────────────────────────────────────────────────────
//
//   use wui_spawn::HttpTransport;
//
//   let transport = HttpTransport::new("http://agent-server:8080");
//   let tools = remote_tools("delegate", "Delegate work", Arc::new(transport));
//
// ── Building a compatible server ─────────────────────────────────────────────
//
// Any HTTP server that accepts the four endpoints above and returns JSON
// in the expected format will work. See the `RemoteJobHandle`,
// `RemoteAgentStatus`, and `RemoteAgentResult` types in `transport.rs` for
// the exact JSON shapes.
// ============================================================================

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::transport::{
    AgentTransport, RemoteAgentResult, RemoteAgentStatus, RemoteJobHandle, TransportError,
};

/// HTTP-based transport for cross-process agent delegation.
///
/// Connects to a remote agent server that manages agent lifecycle over
/// a simple REST API. All communication is JSON over HTTP.
///
/// # Example
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use wui_spawn::{HttpTransport, remote_tools};
///
/// let transport = HttpTransport::new("http://localhost:8080");
/// let tools = remote_tools("delegate", "Delegate work", Arc::new(transport));
///
/// // Register tools with an agent:
/// let agent = Agent::builder(provider)
///     .tools(tools)
///     .permission(PermissionMode::Auto)
///     .build();
/// ```
pub struct HttpTransport {
    client: Client,
    base_url: String,
}

#[derive(Serialize)]
struct SendRequest<'a> {
    agent_name: &'a str,
    prompt: String,
}

#[derive(Serialize)]
struct CancelRequest<'a> {
    id: &'a str,
    agent_name: &'a str,
}

#[derive(Deserialize)]
struct ErrorResponse {
    error: String,
}

impl HttpTransport {
    /// Create a new HTTP transport pointing at the given base URL.
    ///
    /// The base URL should not end with a trailing slash.
    /// Example: `"http://localhost:8080"` or `"https://agents.example.com/api"`.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
        }
    }

    /// Create an HTTP transport with a custom `reqwest::Client`.
    ///
    /// Use this to configure timeouts, TLS, headers, or connection pooling.
    pub fn with_client(client: Client, base_url: impl Into<String>) -> Self {
        Self {
            client,
            base_url: base_url.into().trim_end_matches('/').to_string(),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.base_url, path.trim_start_matches('/'))
    }
}

#[async_trait]
impl AgentTransport for HttpTransport {
    async fn send(
        &self,
        agent_name: &str,
        prompt: String,
    ) -> Result<RemoteJobHandle, TransportError> {
        let resp = self
            .client
            .post(self.url("/send"))
            .json(&SendRequest { agent_name, prompt })
            .send()
            .await
            .map_err(|e| TransportError::Connection(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(TransportError::Connection(format!(
                "POST /send returned {status}: {body}"
            )));
        }

        resp.json::<RemoteJobHandle>()
            .await
            .map_err(|e| TransportError::Connection(format!("invalid response from /send: {e}")))
    }

    async fn status(&self, handle: &RemoteJobHandle) -> Result<RemoteAgentStatus, TransportError> {
        let resp = self
            .client
            .get(self.url("/status"))
            .query(&[("id", &handle.id), ("agent_name", &handle.agent_name)])
            .send()
            .await
            .map_err(|e| TransportError::Connection(e.to_string()))?;

        if resp.status().as_u16() == 404 {
            return Err(TransportError::NotFound(handle.id.clone()));
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(TransportError::Connection(format!(
                "GET /status returned {status}: {body}"
            )));
        }

        resp.json::<RemoteAgentStatus>()
            .await
            .map_err(|e| TransportError::Connection(format!("invalid response from /status: {e}")))
    }

    async fn result(&self, handle: &RemoteJobHandle) -> Result<RemoteAgentResult, TransportError> {
        let resp = self
            .client
            .get(self.url("/result"))
            .query(&[("id", &handle.id), ("agent_name", &handle.agent_name)])
            .send()
            .await
            .map_err(|e| TransportError::Connection(e.to_string()))?;

        if resp.status().as_u16() == 404 {
            return Err(TransportError::NotFound(handle.id.clone()));
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(TransportError::Connection(format!(
                "GET /result returned {status}: {body}"
            )));
        }

        resp.json::<RemoteAgentResult>()
            .await
            .map_err(|e| TransportError::Connection(format!("invalid response from /result: {e}")))
    }

    async fn cancel(&self, handle: &RemoteJobHandle) -> Result<(), TransportError> {
        let resp = self
            .client
            .post(self.url("/cancel"))
            .json(&CancelRequest {
                id: &handle.id,
                agent_name: &handle.agent_name,
            })
            .send()
            .await
            .map_err(|e| TransportError::Connection(e.to_string()))?;

        if resp.status().as_u16() == 404 {
            return Err(TransportError::NotFound(handle.id.clone()));
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(TransportError::Connection(format!(
                "POST /cancel returned {status}: {body}"
            )));
        }

        Ok(())
    }
}
