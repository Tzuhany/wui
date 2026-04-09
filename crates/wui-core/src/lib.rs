// ============================================================================
// wui-core — the vocabulary of the Wui agent framework.
//
// This crate defines what everything *is*: traits, types, events. It contains
// no runtime logic, no HTTP clients, no database connections. Dependencies
// are limited to what is needed to express async traits and serializable
// types: serde, async-trait, tokio (traits only), tokio-util, uuid, tracing.
//
// Every other crate in the framework depends on this one. User-defined
// tools, hooks, and providers depend on this one only. If you are building
// an extension for Wui, this is the only crate you need.
// ============================================================================

pub mod event;
pub mod fmt;
pub mod hook;
pub mod message;
pub mod provider;
pub mod tool;

pub use wui_macros::ToolInput;

// ── Prelude ───────────────────────────────────────────────────────────────────
//
// The types and traits you reach for every time you work with Wui.
// `use wui_core::prelude::*` to bring them all into scope.

pub mod prelude {
    pub use crate::event::{
        AgentError, AgentEvent, CompressMethod, ControlDecision, ControlHandle, ControlKind,
        ControlRequest, ControlResponse, RunStopReason, RunSummary, StopReason, StreamEvent,
        TokenUsage,
    };
    pub use crate::hook::{DenyList, Hook, HookDecision, HookEvent};
    pub use crate::message::{ContentBlock, DocumentSource, ImageSource, Message, Role};
    pub use crate::provider::{
        ChatRequest, Provider, ProviderCapabilities, ProviderError, TokenEstimate, ToolDef,
    };
    pub use crate::tool::{
        Artifact, ArtifactContent, ArtifactKind, ContextInjection, FailureKind, Tool, ToolArgs,
        ToolCtx, ToolInput, ToolInputError, ToolOutput, TypedTool,
    };
    pub use crate::hook::SessionId;
    pub use crate::tool::ToolCallId;
}
