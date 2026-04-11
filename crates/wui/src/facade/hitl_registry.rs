// ============================================================================
// HitlRegistry — auto-registering store for in-flight HITL handles.
//
// Applications that serve Control events over HTTP / WebSocket / message queues
// face a bookkeeping problem: `ControlHandle` is emitted inside the agent
// event stream, but responses arrive on a separate channel (REST endpoint,
// WebSocket frame, etc.) identified only by request ID.
//
// The typical workaround is a hand-rolled `DashMap<String, ControlHandle>`:
// insert on Control, look up on response, remove after reply. HitlRegistry
// is that pattern, done once, correctly.
//
// Usage:
//
//   let registry = HitlRegistry::new();
//   let mut stream = registry.attach(session.send("…").await);
//
//   // consume stream events, surfacing Control.request fields to your UI
//   while let Some(event) = stream.next().await { … }
//
//   // from a WebSocket handler / REST endpoint:
//   registry.approve("req_abc123");
//
// The handle is cloned into the registry as it passes through `attach()`.
// The original event is still yielded to the caller — nothing is swallowed.
// The first response wins (via ControlHandle's Arc<Mutex<Option<Sender>>>):
// responding through the registry and directly via the event handle are
// both safe — only one will fire.
// ============================================================================

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures::Stream;
use futures::StreamExt;

use wui_core::event::{AgentEvent, ControlHandle};

/// Auto-registering store for in-flight HITL control handles.
///
/// Wrap a `Session::send()` stream with [`attach()`][HitlRegistry::attach]
/// and every `AgentEvent::Control` handle is automatically registered as it
/// flows through — indexed by `ControlRequest::id`. Respond from any async
/// context by ID, without threading handles through your call stack.
///
/// All `approve`/`deny` methods remove the handle from the registry after
/// responding, so `pending_ids()` reflects only handles still awaiting a
/// decision.
///
/// # Example
///
/// ```rust,ignore
/// use wui::{HitlRegistry, Session};
///
/// let registry = HitlRegistry::new();
///
/// let mut stream = registry.attach(session.send("delete all files").await);
/// while let Some(event) = stream.next().await {
///     if let AgentEvent::Control(ref h) = event {
///         // surface to UI: h.request.description(), h.request.tool_name()
///         println!("pending: {}", h.request.id);
///     }
/// }
///
/// // From a WebSocket handler or REST endpoint:
/// registry.approve("req_abc123");
/// registry.deny("req_xyz789", "not allowed");
/// ```
pub struct HitlRegistry {
    handles: Arc<Mutex<HashMap<String, ControlHandle>>>,
}

impl Default for HitlRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl HitlRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            handles: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Wrap `stream`, auto-registering every `Control` handle as it flows through.
    ///
    /// The `Control` event is still yielded to the caller — this is a
    /// transparent intercept. Attach to the stream returned by
    /// [`Session::send()`] or [`Session::send_with_cancel()`].
    pub fn attach<S>(&self, stream: S) -> impl Stream<Item = AgentEvent> + Unpin
    where
        S: Stream<Item = AgentEvent> + Unpin,
    {
        let handles = self.handles.clone();
        stream.map(move |event| {
            if let AgentEvent::Control(ref handle) = event {
                if let Ok(mut map) = handles.lock() {
                    map.insert(handle.request.id.clone(), handle.clone());
                }
            }
            event
        })
    }

    /// IDs of all registered handles not yet responded to.
    pub fn pending_ids(&self) -> Vec<String> {
        self.handles
            .lock()
            .map(|map| map.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Remove and return the handle for `request_id`.
    ///
    /// Prefer the typed `approve`/`deny` helpers unless you need to call
    /// multiple response methods conditionally.
    pub fn take(&self, request_id: &str) -> Option<ControlHandle> {
        self.handles.lock().ok()?.remove(request_id)
    }

    /// Approve the request identified by `request_id`. Returns `false` if not found.
    pub fn approve(&self, request_id: &str) -> bool {
        self.take(request_id).is_some_and(|h| { h.approve(); true })
    }

    /// Approve with a modification note the LLM will see. Returns `false` if not found.
    pub fn approve_with(&self, request_id: &str, modification: impl Into<String>) -> bool {
        self.take(request_id)
            .is_some_and(|h| { h.approve_with(modification); true })
    }

    /// Approve this tool for all future calls in this session. Returns `false` if not found.
    pub fn approve_always(&self, request_id: &str) -> bool {
        self.take(request_id).is_some_and(|h| { h.approve_always(); true })
    }

    /// Deny the request identified by `request_id`. Returns `false` if not found.
    pub fn deny(&self, request_id: &str, reason: impl Into<String>) -> bool {
        self.take(request_id)
            .is_some_and(|h| { h.deny(reason); true })
    }

    /// Deny this tool for all future calls in this session. Returns `false` if not found.
    pub fn deny_always(&self, request_id: &str, reason: impl Into<String>) -> bool {
        self.take(request_id)
            .is_some_and(|h| { h.deny_always(reason); true })
    }
}
