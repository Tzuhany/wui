// ============================================================================
// wuhu-engine — the execution loop.
//
// Public surface: `RunConfig`, `run()`, and the supporting types callers
// need to configure and drive the loop.
// ============================================================================

mod executor;
mod hooks;
mod permission;
mod registry;
mod run;

pub use hooks::HookRunner;
pub use permission::{PermissionMode, SessionPermissions};
pub use registry::ToolRegistry;
pub use run::{run, RunConfig};
