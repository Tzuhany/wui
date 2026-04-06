// ============================================================================
// wuhu-engine — the execution loop.
//
// Public surface: RunConfig, run(), and the supporting types callers
// need to configure and drive the loop.
// ============================================================================

mod executor;
mod hooks;
mod permission;
mod query_chain;
mod registry;
mod run;
mod tool_search;

pub use hooks::HookRunner;
pub use permission::{PermissionMode, SessionPermissions};
pub use query_chain::{DepthExceeded, QueryChain};
pub use registry::{DeferredEntry, ToolRegistry};
pub use run::{run, RetryPolicy, RunConfig, RunStream};
pub use tool_search::ToolSearch;
