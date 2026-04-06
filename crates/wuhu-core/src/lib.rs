// ============================================================================
// wuhu-core — the vocabulary of the Wuhu agent framework.
//
// This crate defines what everything *is*. It contains no runtime logic,
// no HTTP clients, no database connections. Its only dependencies are serde,
// serde_json, and async-trait.
//
// Every other crate in the framework depends on this one. User-defined
// tools, hooks, checkpoints, and memory backends depend on this one.
// If you are building an extension for Wuhu, this is the only crate you
// need to import.
// ============================================================================

pub mod checkpoint;
pub mod event;
pub mod hook;
pub mod memory;
pub mod message;
pub mod provider;
pub mod tool;

// ── Prelude ───────────────────────────────────────────────────────────────────
//
// The types and traits you reach for every time you work with Wuhu.
// `use wuhu_core::prelude::*` to bring them all into scope.

pub mod prelude {
    pub use crate::checkpoint::{Checkpoint, CheckpointError, InMemory, SessionSnapshot};
    pub use crate::event::{
        AgentEvent, AgentError, CompressMethod, ControlDecision, ControlHandle,
        ControlKind, ControlRequest, ControlResponse, RunStopReason, RunSummary,
        StopReason, StreamEvent, TokenUsage,
    };
    pub use crate::hook::{DenyList, Hook, HookDecision, HookEvent};
    pub use crate::memory::{Memory, MemoryEntry, MemoryError, NewMemory};
    pub use crate::message::{ContentBlock, Message, Role};
    pub use crate::provider::{ChatRequest, Provider, ProviderError, ToolDef};
    pub use crate::tool::{FailureKind, SpawnFn, Tool, ToolCtx, ToolOutput};
}
