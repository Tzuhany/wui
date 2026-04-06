// ============================================================================
// wuhu — A Rust framework for building LLM agents.
//
// The entry point for most users. Import this crate and start building:
//
//   use wuhu::{Agent, AgentEvent};
//   use wuhu_providers::Anthropic;
//
// See docs/philosophy.md for the design principles behind this framework.
// See docs/architecture.md for the internal structure.
// ============================================================================

mod agent;
mod builder;
mod session;

pub use agent::Agent;
pub use builder::AgentBuilder;
pub use session::Session;

// Re-export the types users need most often.
pub use wuhu_core::event::{
    AgentError, AgentEvent, CompressMethod, ControlDecision, ControlHandle,
    ControlKind, ControlRequest, ControlResponse, RunStopReason, RunSummary,
    StopReason, TokenUsage,
};
pub use wuhu_core::hook::{DenyList, Hook, HookDecision, HookEvent};
pub use wuhu_core::message::{ContentBlock, Message, Role};
pub use wuhu_core::provider::Provider;
pub use wuhu_core::fmt;
pub use wuhu_core::tool::{
    Artifact, ArtifactContent, FailureKind, SpawnFn,
    Tool, ToolCtx, ToolInput, ToolOutput,
};
pub use wuhu_core::checkpoint::{Checkpoint, InMemory, SessionSnapshot};
pub use wuhu_core::memory::{Memory, MemoryEntry, NewMemory};
pub use wuhu_engine::{PermissionMode, QueryChain};
pub use wuhu_compress::CompressPipeline;

/// The prelude. `use wuhu::prelude::*` to bring the most-used types into scope.
pub mod prelude {
    pub use super::{
        Agent, AgentEvent, AgentError, AgentBuilder,
        Artifact, ArtifactContent,
        ControlResponse, CompressPipeline,
        DenyList, Hook, HookDecision,
        InMemory, Message, PermissionMode, QueryChain,
        Provider, RunStopReason, Session,
        Tool, ToolCtx, ToolInput, ToolOutput,
    };
    pub use futures::StreamExt;
}
