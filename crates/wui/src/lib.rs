// ============================================================================
// wui — A Rust framework for building LLM agents.
//
// Import this crate and start building:
//
//   use wui::prelude::*;
//
// Optional extensions (separate crates, opt-in):
//   wui-observe  — timeline collection and OTel span emission
//   wui-workflow — deterministic pipeline / branch / parallel primitives
//   wui-memory   — built-in remember/recall/forget tools
//   wui-mcp      — MCP protocol adapter
//
// Provider features:
//   wui = { version = "...", features = ["anthropic"] }
//   wui = { version = "...", features = ["openai"] }
//   wui = { version = "...", features = ["full"] }   // all providers
//
// `wui-core` is an internal vocabulary crate (traits + event types, no
// runtime deps). You do not need to depend on it directly — everything you
// need to implement `Tool`, `Hook`, or `Provider` is re-exported here.
// ============================================================================

mod agent;
mod builder;
mod session;
mod structured;
mod sub_agent;

pub(crate) mod compress;
pub(crate) mod runtime;

#[cfg(any(feature = "anthropic", feature = "openai"))]
pub mod providers;

pub use agent::Agent;
pub use builder::{AgentBuilder, Effort};
pub use session::Session;
pub use sub_agent::{SubAgent, SubAgentSummary, SubAgentToolCall};

// ── Core types ────────────────────────────────────────────────────────────────

pub use wui_core::event::{
    AgentError, AgentEvent, CompressMethod, ControlDecision, ControlHandle, ControlKind,
    ControlRequest, ControlResponse, RunStopReason, RunSummary, StopReason, TokenUsage,
};
pub use wui_core::fmt;
pub use wui_core::hook::{DenyList, Hook, HookDecision, HookEvent};
pub use wui_core::message::{ContentBlock, DocumentSource, ImageSource, Message, Role};
pub use wui_core::provider::Provider;
pub use wui_core::tool::{
    Artifact, ArtifactContent, ArtifactKind, ContextInjection, FailureKind, Tool, ToolCtx,
    ToolInput, ToolMeta, ToolOutput,
};

pub use structured::StructuredRun;

// ── Runtime types ─────────────────────────────────────────────────────────────

pub use runtime::{
    CheckpointStore, ExecutorHints, FileCheckpointStore, InMemoryCheckpointStore,
    InMemorySessionStore, PermissionMode, PermissionRules, PermissionVerdict, RetryPolicy,
    RunCheckpoint, RunStream, SessionStore, SessionStoreError, StoredSession,
};

// ── Compress ──────────────────────────────────────────────────────────────────

pub use compress::{CompressPipeline, CompressResult, CompressStrategy, SummarizingCompressor};

// ── Catalog ───────────────────────────────────────────────────────────────────

pub mod catalog;
pub use catalog::{CatalogHit, SearchStrategy, StaticCatalog, TokenOverlapStrategy, ToolCatalog};

// ── Prelude ───────────────────────────────────────────────────────────────────

/// `use wui::prelude::*` to bring the most-used types into scope.
pub mod prelude {
    pub use super::{
        Agent, AgentBuilder, AgentError, AgentEvent, Artifact, ArtifactContent, CompressPipeline,
        ContextInjection, ControlResponse, DenyList, Effort, Hook, HookDecision, HookEvent,
        InMemorySessionStore, Message, PermissionMode, PermissionRules, Provider, RetryPolicy,
        RunStopReason, RunStream, Session, SessionStore, SubAgent, SubAgentSummary,
        SubAgentToolCall, Tool, ToolCtx, ToolInput, ToolMeta, ToolOutput,
    };
    pub use futures::StreamExt;
}
