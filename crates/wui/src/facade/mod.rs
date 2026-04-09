// ============================================================================
// facade — user-facing public API types.
//
// This module groups the primary entry points (Agent, AgentBuilder, Session,
// StructuredRun, SubAgent) separately from internal implementation modules
// (runtime/, compress/, providers/). Everything here is re-exported from the
// crate root so downstream code is unaffected.
// ============================================================================

pub(crate) mod agent;
pub(crate) mod builder;
pub(crate) mod session;
pub(crate) mod sub_agent;
