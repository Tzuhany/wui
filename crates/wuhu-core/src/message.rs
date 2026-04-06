// ============================================================================
// Message — the unit of conversation.
//
// A message is a role paired with a sequence of content blocks. The block
// model mirrors the Anthropic API's content structure and generalises cleanly
// to other providers: text, tool calls, tool results.
//
// Two framework-level compression variants survive in internal history but
// are never sent to a provider as-is:
//
//   Collapsed       — L2: a reversible placeholder for folded messages.
//                     The original messages survive in external storage;
//                     this is a lightweight stand-in. first_id/last_id
//                     track the range so the segment can be re-expanded.
//
//   CompactBoundary — L3: an irreversible summary produced by the LLM.
//                     Everything before this boundary has been replaced.
//                     The LLM reads the summary and calibrates accordingly.
//
// The two-variant design makes the distinction explicit in the type system:
// a Collapsed segment *can* be expanded; a CompactBoundary *cannot*.
// ============================================================================

use serde::{Deserialize, Serialize};

/// A single turn in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id:      String,
    pub role:    Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            id:      uuid::Uuid::new_v4().to_string(),
            role:    Role::User,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    pub fn assistant(blocks: Vec<ContentBlock>) -> Self {
        Self {
            id:      uuid::Uuid::new_v4().to_string(),
            role:    Role::Assistant,
            content: blocks,
        }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self {
            id:      uuid::Uuid::new_v4().to_string(),
            role:    Role::System,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    /// Construct a message with an explicit id.
    ///
    /// Use when reconstructing messages from storage (checkpoints, databases)
    /// where the original id must be preserved for continuity or auditing.
    /// For new messages, prefer the role-specific constructors.
    pub fn with_id(id: impl Into<String>, role: Role, content: Vec<ContentBlock>) -> Self {
        Self { id: id.into(), role, content }
    }
}

/// Who produced this message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
}

/// A piece of content within a message.
///
/// The first four variants map to LLM API primitives.
/// The last two are framework-level compression markers: they live in the
/// internal history but are translated (not passed verbatim) to providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text — the most common block type.
    Text { text: String },

    /// Extended thinking / chain-of-thought (Anthropic extended thinking).
    /// The framework preserves this in history; providers decide whether
    /// to include it in API requests.
    Thinking { text: String },

    /// A tool invocation requested by the LLM.
    ToolUse {
        id:    String,
        name:  String,
        input: serde_json::Value,
    },

    /// The result of executing a tool.
    ToolResult {
        tool_use_id: String,
        content:     String,
        is_error:    bool,
    },

    /// Left behind when L2 (collapse) folds old messages into a placeholder.
    ///
    /// The original messages survive in external storage; this is a lightweight
    /// stand-in. `first_id` and `last_id` record the range so a storage-aware
    /// application can re-expand the segment if needed.
    ///
    /// **Reversible** (in principle): the originals are not deleted.
    Collapsed {
        summary:      String,
        folded_count: u32,
        /// ID of the first folded message (`None` when IDs are unavailable).
        first_id: Option<String>,
        /// ID of the last folded message (`None` when IDs are unavailable).
        last_id:  Option<String>,
    },

    /// Left behind when L3 (LLM) compression summarises old messages.
    ///
    /// **Irreversible**: the original messages are gone from context.
    /// The LLM reads this summary and calibrates its behaviour accordingly.
    CompactBoundary {
        summary: String,
    },
}
