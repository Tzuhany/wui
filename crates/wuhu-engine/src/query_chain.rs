// QueryChain lives in wuhu-core so that Tool implementations can read their
// own position in the delegation tree via ToolCtx::chain without depending
// on wuhu-engine. Re-exported here so the engine's public surface is unchanged.
pub use wuhu_core::query_chain::{DepthExceeded, QueryChain};
