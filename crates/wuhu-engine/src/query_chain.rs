// ============================================================================
// QueryChain — sub-agent depth tracking.
//
// When an agent spawns sub-agents recursively, QueryChain threads a shared
// `chain_id` through the entire tree and enforces a `max_depth` ceiling.
//
// Without depth tracking, a misbehaving prompt can cause a sub-agent to spawn
// further sub-agents indefinitely — burning tokens and credits without bound.
// QueryChain makes runaway delegation a compile-time-catchable error: `child()`
// returns `Err(DepthExceeded)` rather than silently continuing.
//
// Usage:
//   let chain = QueryChain::root();        // depth = 0, fresh chain_id
//   let child = chain.child()?;            // depth = 1, same chain_id
//   let deep  = child.child()?;            // depth = 2
//   let err   = deep.child_with_max(2);    // Err(DepthExceeded { max_depth: 2 })
// ============================================================================

use serde::{Deserialize, Serialize};

/// A position in a sub-agent invocation tree.
///
/// Pass via `RunConfig::query_chain` to have the engine enforce depth limits
/// and propagate chain identity through nested spawns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryChain {
    /// Shared identifier for the entire tree rooted at one top-level run.
    /// Useful for correlating logs and traces across a delegation hierarchy.
    pub chain_id: String,
    /// Current depth. `0` = the top-level agent.
    pub depth: u32,
    /// Hard ceiling. `child()` returns `Err` when `depth == max_depth`.
    pub max_depth: u32,
}

/// Returned when `QueryChain::child()` would exceed `max_depth`.
#[derive(Debug, thiserror::Error)]
#[error("sub-agent depth limit {max_depth} reached (chain {chain_id})")]
pub struct DepthExceeded {
    pub chain_id:  String,
    pub max_depth: u32,
}

impl QueryChain {
    /// Start a new root chain at depth 0 with a default max depth of 5.
    pub fn root() -> Self {
        Self::root_with_max(5)
    }

    /// Start a new root chain at depth 0 with a custom max depth.
    pub fn root_with_max(max_depth: u32) -> Self {
        Self {
            chain_id:  uuid::Uuid::new_v4().to_string(),
            depth:     0,
            max_depth,
        }
    }

    /// Produce a child position one level deeper in the same chain.
    ///
    /// Returns `Err(DepthExceeded)` if `depth == max_depth`.
    pub fn child(&self) -> Result<Self, DepthExceeded> {
        if self.depth >= self.max_depth {
            return Err(DepthExceeded {
                chain_id:  self.chain_id.clone(),
                max_depth: self.max_depth,
            });
        }
        Ok(Self {
            chain_id:  self.chain_id.clone(),
            depth:     self.depth + 1,
            max_depth: self.max_depth,
        })
    }

    /// Whether this is the root of the chain (depth == 0).
    pub fn is_root(&self) -> bool {
        self.depth == 0
    }

    /// Remaining depth budget before the ceiling is hit.
    pub fn remaining(&self) -> u32 {
        self.max_depth.saturating_sub(self.depth)
    }
}
