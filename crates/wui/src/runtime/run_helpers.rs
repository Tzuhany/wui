// ============================================================================
// Run-loop helpers — small utilities used by run.rs.
//
// Extracted for readability. These are internal to the runtime; nothing here
// is part of the public API.
// ============================================================================

use std::collections::HashSet;

use wui_core::event::AgentError;
use wui_core::message::{ContentBlock, Message, Role};
use wui_core::types::ToolCallId;

use super::executor::CompletedTool;
use super::registry::ToolRegistry;

// ── Framework-internal ToolOutput constructors ────────────────────────────────

pub(super) fn output_hook_blocked(reason: impl Into<String>) -> wui_core::tool::ToolOutput {
    wui_core::tool::ToolOutput {
        content: reason.into(),
        failure: Some(wui_core::tool::FailureKind::HookBlocked),
        ..Default::default()
    }
}

pub(super) fn output_permission_denied(reason: impl Into<String>) -> wui_core::tool::ToolOutput {
    wui_core::tool::ToolOutput {
        content: reason.into(),
        failure: Some(wui_core::tool::FailureKind::PermissionDenied),
        ..Default::default()
    }
}

// ── EmissionGuard ────────────────────────────────────────────────────────────

/// Guards against double-emission of tool events.
pub(super) struct EmissionGuard {
    emitted: HashSet<String>,
}

impl EmissionGuard {
    pub(super) fn new() -> Self {
        Self {
            emitted: HashSet::new(),
        }
    }

    /// Returns `true` if this is the first time — caller should emit.
    pub(super) fn first_time(&mut self, id: &str) -> bool {
        self.emitted.insert(id.to_owned())
    }
}

// ── Small helpers ────────────────────────────────────────────────────────────

/// Create an immediately-completed failed tool.
pub(super) fn instant_failure(
    id: ToolCallId,
    name: String,
    output: wui_core::tool::ToolOutput,
) -> CompletedTool {
    CompletedTool {
        id,
        name,
        output,
        ms: 0,
        attempts: 1,
    }
}

/// Detect whether an `AgentError` indicates a prompt-too-long rejection.
pub(super) fn is_prompt_too_long(e: &AgentError) -> bool {
    let msg = e.message.to_lowercase();
    msg.contains("prompt is too long")
        || msg.contains("too long")
        || msg.contains("maximum context length")
        || msg.contains("context_length_exceeded")
}

/// Build a system-reminder `Message` from plain text.
pub(super) fn system_reminder_msg(text: &str) -> Message {
    Message {
        id: uuid::Uuid::new_v4().to_string(),
        role: Role::System,
        content: vec![ContentBlock::Text {
            text: wui_core::fmt::system_reminder(text),
        }],
    }
}

/// Extract the text content of the most recent assistant message.
pub(super) fn last_assistant_text(messages: &[Message]) -> &str {
    messages
        .iter()
        .rev()
        .find_map(|m| {
            if m.role != Role::Assistant {
                return None;
            }
            m.content.iter().find_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
        })
        .unwrap_or("")
}

/// Replace the text content of the most recent assistant message.
pub(super) fn replace_last_assistant_text(messages: &mut [Message], new_text: String) {
    for msg in messages.iter_mut().rev() {
        if msg.role != Role::Assistant {
            continue;
        }
        for block in msg.content.iter_mut() {
            if let ContentBlock::Text { text } = block {
                *text = new_text;
                return;
            }
        }
    }
}

/// Append a deferred-tools listing to the system prompt when needed.
pub(super) fn augment_system(base: &str, registry: &ToolRegistry) -> String {
    let deferred = registry.deferred_entries();
    if deferred.is_empty() {
        return base.to_string();
    }

    let listing = deferred
        .iter()
        .map(|e| format!("- **{}**: {}", e.name, e.description))
        .collect::<Vec<_>>()
        .join("\n");

    let section = format!(
        "## Additional tools\n\
        These tools are available but require loading. \
        Call `ToolSearch` with the tool name or a keyword before using them:\n\n\
        {listing}"
    );

    format!("{base}\n\n{}", wui_core::fmt::system_reminder(&section))
}
