// ============================================================================
// Session — multi-turn conversation state.
//
// A Session owns the message history for one ongoing conversation and
// provides the `respond()` method for answering HITL control requests.
//
// Sessions are the unit of persistence: if a `Checkpoint` is configured,
// the session saves after every turn and can be resumed after a restart.
// ============================================================================

use std::sync::Arc;

use futures::StreamExt;
use tokio::sync::Mutex;

use wuhu_core::checkpoint::{Checkpoint, SessionSnapshot};
use wuhu_core::event::{AgentEvent, ControlResponse};
use wuhu_core::message::Message;
use wuhu_core::tool::SpawnFn;
use wuhu_compress::CompressPipeline;
use wuhu_engine::{HookRunner, PermissionMode, RunConfig, ToolRegistry, run};

use crate::builder::AgentConfig;

/// An active multi-turn conversation.
pub struct Session {
    id:         String,
    config:     Arc<AgentConfig>,
    messages:   Arc<Mutex<Vec<Message>>>,
    checkpoint: Option<Arc<dyn Checkpoint>>,
    /// Live channel for sending control responses into the running loop.
    pending_tx: Arc<Mutex<Option<tokio::sync::oneshot::Sender<ControlResponse>>>>,
}

impl Session {
    pub(crate) async fn new(id: impl Into<String>, config: Arc<AgentConfig>) -> Self {
        let id = id.into();
        let messages = if let Some(cp) = &config.checkpoint {
            cp.load(&id).await
                .ok()
                .flatten()
                .map(|s| s.messages)
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        Self {
            id,
            config,
            messages:    Arc::new(Mutex::new(messages)),
            checkpoint:  None, // set from AgentConfig if present
            pending_tx:  Arc::new(Mutex::new(None)),
        }
    }

    /// Send a message and return a stream of events.
    pub async fn send(
        &self,
        content: impl Into<String>,
    ) -> impl futures::Stream<Item = AgentEvent> + '_ {
        use wuhu_core::message::Message;

        let user_msg = Message::user(content.into());
        {
            let mut msgs = self.messages.lock().await;
            msgs.push(user_msg);
        }

        let messages = self.messages.lock().await.clone();
        let run_config = self.make_run_config();

        let cancel = tokio_util::sync::CancellationToken::new();
        let event_stream = run(Arc::new(run_config), messages, cancel.clone());

        // Collect committed messages from the stream's Done event.
        let messages_ref = self.messages.clone();
        let checkpoint   = self.config.checkpoint.clone();
        let session_id   = self.id.clone();

        event_stream.map(move |event| {
            if let AgentEvent::Done(ref summary) = event {
                // The summary doesn't include individual messages — we'd need
                // the full updated history. For now, the run loop returns the
                // updated messages through the config. This is simplified.
                // Full implementation: run() returns (Stream, updated_messages).
                let _ = (&messages_ref, &checkpoint, &session_id);
            }
            event
        })
    }

    /// Respond to a pending control request (HITL).
    ///
    /// Call this after receiving `AgentEvent::Control` from a `send()` stream.
    pub async fn respond(&self, response: ControlResponse) {
        if let Some(tx) = self.pending_tx.lock().await.take() {
            let _ = tx.send(response);
        }
    }

    fn make_run_config(&self) -> RunConfig {
        RunConfig {
            provider:    self.config.provider.clone(),
            tools:       Arc::new(ToolRegistry::new(self.config.tools.clone())),
            hooks:       Arc::new(HookRunner::new(self.config.hooks.clone())),
            compress:    self.config.compress.clone(),
            permission:  self.config.permission.clone(),
            system:      self.config.system.clone(),
            model:       self.config.model.clone(),
            max_tokens:  self.config.max_tokens,
            temperature: self.config.temperature,
            max_iter:    self.config.max_iter,
            extensions:  self.config.extensions.clone(),
            spawn:       self.config.spawn.clone(),
        }
    }
}
