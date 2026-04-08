// ── Artifact ──────────────────────────────────────────────────────────────────
//
// Tools can produce more than text. An artifact is any discrete output that
// warrants separate delivery: a generated file, a rendered chart, a binary
// blob. Artifacts travel alongside ToolOutput but are emitted as their own
// AgentEvent so callers can route them independently.

use serde::{Deserialize, Serialize};

/// The semantic kind of an [`Artifact`].
///
/// Use [`ArtifactKind::Custom`] for application-specific kinds not covered
/// by the standard variants.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    /// A file artifact (source code, documents, data files, etc.)
    File,
    /// An image artifact (PNG, JPEG, SVG, etc.)
    Image,
    /// A structured chart or graph artifact.
    Chart,
    /// A JSON data artifact.
    Json,
    /// An application-specific artifact kind.
    Custom(String),
}

impl ArtifactKind {
    /// Returns the string representation of this kind.
    pub fn as_str(&self) -> &str {
        match self {
            ArtifactKind::File => "file",
            ArtifactKind::Image => "image",
            ArtifactKind::Chart => "chart",
            ArtifactKind::Json => "json",
            ArtifactKind::Custom(s) => s.as_str(),
        }
    }
}

impl std::fmt::Display for ArtifactKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A discrete output produced by a tool — beyond the primary text content.
///
/// Artifacts are emitted as `AgentEvent::Artifact` events, separate from
/// `ToolDone`, so callers can route them to the right destination:
/// save files to disk, render images in a UI, index structured data, etc.
///
/// The `kind` field uses [`ArtifactKind`] — use [`ArtifactKind::Custom`] for
/// application-specific kinds not covered by the standard variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Semantic type of this artifact.
    pub kind: ArtifactKind,
    /// Human-readable title for display.
    pub title: String,
    /// MIME type, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// The actual content: inline bytes or a URI reference.
    pub content: ArtifactContent,
}

/// The data carried by an `Artifact`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ArtifactContent {
    /// Raw bytes embedded directly.
    Inline { data: Vec<u8> },
    /// A reference to external storage (URI, path, object key, etc.).
    Reference { uri: String },
}

impl Artifact {
    /// Construct a text artifact (UTF-8 content as inline bytes).
    pub fn text(title: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            kind: ArtifactKind::File,
            title: title.into(),
            mime_type: Some("text/plain".to_string()),
            content: ArtifactContent::Inline {
                data: text.into().into_bytes(),
            },
        }
    }

    /// Construct a binary artifact from raw bytes with an explicit kind.
    pub fn bytes(
        title: impl Into<String>,
        kind: ArtifactKind,
        mime_type: Option<impl Into<String>>,
        data: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            kind,
            title: title.into(),
            mime_type: mime_type.map(|m| m.into()),
            content: ArtifactContent::Inline { data: data.into() },
        }
    }

    /// Construct a reference artifact (the content lives elsewhere).
    pub fn reference(title: impl Into<String>, kind: ArtifactKind, uri: impl Into<String>) -> Self {
        Self {
            kind,
            title: title.into(),
            mime_type: None,
            content: ArtifactContent::Reference { uri: uri.into() },
        }
    }
}
