// ============================================================================
// Message — the unit of conversation.
//
// A message is a role paired with a sequence of content blocks. The block
// model mirrors the Anthropic API's content structure and generalises cleanly
// to other providers: every LLM ultimately produces text, calls tools, and
// returns tool results. The extra variants (Thinking, Compressed) are
// framework-level extensions — providers must strip or translate them before
// building API requests.
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

    /// True if this message has no content blocks (or only empty text blocks).
    pub fn is_empty(&self) -> bool {
        self.content.iter().all(|b| match b {
            ContentBlock::Text { text } => text.is_empty(),
            _ => false,
        })
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
/// The variants map directly to LLM primitives (text, tool calls, results)
/// plus two framework-level variants that survive in the internal history
/// but are never sent to a provider as-is.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text — the most common block type.
    Text { text: String },

    /// Extended thinking / chain-of-thought (Anthropic only for now).
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

    /// Left behind when L3 (LLM) compression summarises old messages.
    ///
    /// The agent sees this in its context and knows history was condensed.
    /// The summary is injected into the system prompt — the agent can
    /// calibrate its behaviour accordingly.
    Compressed {
        summary:      String,
        folded_count: usize,
    },
}
