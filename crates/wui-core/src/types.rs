// ============================================================================
// Domain types — strongly-typed identifiers.
//
// These newtypes replace raw Strings in the public API to prevent
// accidentally swapping a tool_call_id with a session_id, a tool name,
// or any other string. They Deref to &str for ergonomic read access
// and implement Display, From<String>, From<&str>, Serialize, and
// Deserialize.
//
// Adoption is incremental: new code should use these types; existing
// String-based APIs are migrated gradually.
// ============================================================================

use std::borrow::Borrow;
use std::fmt;

use serde::{Deserialize, Serialize};

macro_rules! define_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(s: impl Into<String>) -> Self {
                Self(s.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            pub fn into_inner(self) -> String {
                self.0
            }
        }

        impl std::ops::Deref for $name {
            type Target = str;
            fn deref(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self {
                Self(s)
            }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(s.to_owned())
            }
        }

        impl PartialEq<str> for $name {
            fn eq(&self, other: &str) -> bool {
                self.0 == other
            }
        }

        impl PartialEq<String> for $name {
            fn eq(&self, other: &String) -> bool {
                self.0 == *other
            }
        }

        impl PartialEq<&str> for $name {
            fn eq(&self, other: &&str) -> bool {
                self.0 == *other
            }
        }

        impl Borrow<str> for $name {
            fn borrow(&self) -> &str {
                &self.0
            }
        }
    };
}

define_id! {
    /// A unique identifier for a tool call within a single LLM turn.
    ///
    /// Assigned by the provider (e.g., `toolu_abc123` for Anthropic).
    /// Used to correlate `ToolStart`, `ToolDone`, and `ToolResult` events.
    ToolCallId
}

define_id! {
    /// A unique identifier for a multi-turn session.
    ///
    /// Used to load/save session state from a `SessionStore`.
    SessionId
}

define_id! {
    /// A unique identifier for a checkpoint run.
    ///
    /// Used to save/restore run state from a `CheckpointStore`.
    CheckpointRunId
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_id_basics() {
        let id = ToolCallId::new("toolu_abc123");
        assert_eq!(id.as_str(), "toolu_abc123");
        assert_eq!(&*id, "toolu_abc123");
        assert_eq!(id, *"toolu_abc123");
        assert_eq!(format!("{id}"), "toolu_abc123");
    }

    #[test]
    fn session_id_from_string() {
        let id: SessionId = "my-session".into();
        assert_eq!(id, *"my-session");
    }

    #[test]
    fn serde_roundtrip() {
        let id = ToolCallId::new("abc");
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, r#""abc""#);
        let back: ToolCallId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, id);
    }
}
