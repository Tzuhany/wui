// ── Message history construction ─────────────────────────────────────────────

use std::sync::Arc;

use tokio::sync::mpsc;

use wui_core::event::AgentEvent;
use wui_core::message::{ContentBlock, ImageSource, Message, Role};
use wui_core::provider::ProviderCapabilities;
use wui_core::tool::{Artifact, ArtifactContent, ArtifactKind, ToolCallId};

use super::checkpoint::RunCheckpoint;
use super::{CompletedTool, IterationCtx, RunConfig, RunState};

struct ToolHistoryUpdate {
    result_blocks: Vec<ContentBlock>,
    injections: Vec<Message>,
}

/// Build the assistant message and tool results, append them to history,
/// update usage/iterations, and save a checkpoint if applicable.
pub(super) async fn assemble_history(
    ctx: IterationCtx,
    s: &mut RunState,
    config: &RunConfig,
    tx: &mpsc::Sender<AgentEvent>,
) {
    let IterationCtx {
        assistant_blocks,
        submission_order,
        completed_map,
        text_buf,
        thinking_buf,
        stop_reason,
        usage,
        auth_injections,
        ..
    } = ctx;

    let had_tool_results = !submission_order.is_empty();

    s.messages.extend(auth_injections);
    append_assistant_message(&mut s.messages, text_buf, thinking_buf, assistant_blocks);

    let caps = config.provider.capabilities(config.model.as_deref());
    let tool_update = collect_tool_history(
        ordered_completed_tools(submission_order, completed_map),
        s,
        tx,
        &caps,
    )
    .await;
    append_tool_history(&mut s.messages, tool_update);

    tracing::debug!(
        stop_reason    = ?stop_reason,
        input_tokens   = usage.input_tokens,
        output_tokens  = usage.output_tokens,
        cache_read     = usage.cache_read_tokens,
        cache_write    = usage.cache_write_tokens,
        "llm call complete"
    );

    s.total_usage += usage;
    s.iterations += 1;

    // Checkpoint save (only after tool-use iterations).
    if had_tool_results {
        if let (Some(store), Some(run_id)) = (&config.checkpoint_store, &config.checkpoint_run_id) {
            let cp = RunCheckpoint {
                run_id: run_id.clone(),
                messages: s.messages.clone(),
                iteration: s.iterations,
                total_usage: s.total_usage.clone(),
            };
            if let Err(e) = store.save(run_id, &cp).await {
                tracing::warn!(run_id, error = %e, "checkpoint save failed");
            }
        }
    }
}

fn append_assistant_message(
    messages: &mut Vec<Message>,
    text_buf: String,
    thinking_buf: String,
    mut assistant_blocks: Vec<ContentBlock>,
) {
    if !text_buf.is_empty() {
        assistant_blocks.insert(0, ContentBlock::Text { text: text_buf });
    }
    if !thinking_buf.is_empty() {
        assistant_blocks.insert(0, ContentBlock::Thinking { text: thinking_buf });
    }
    if !assistant_blocks.is_empty() {
        messages.push(Message::assistant(assistant_blocks));
    }
}

fn ordered_completed_tools(
    submission_order: Vec<ToolCallId>,
    mut completed_map: std::collections::HashMap<ToolCallId, CompletedTool>,
) -> Vec<CompletedTool> {
    submission_order
        .iter()
        .filter_map(|id| completed_map.remove(id))
        .collect()
}

async fn collect_tool_history(
    done_tools: Vec<CompletedTool>,
    s: &mut RunState,
    tx: &mpsc::Sender<AgentEvent>,
    caps: &ProviderCapabilities,
) -> ToolHistoryUpdate {
    let mut update = ToolHistoryUpdate {
        result_blocks: Vec::new(),
        injections: Vec::new(),
    };

    for done in done_tools {
        for tool in &done.output.expose_tools {
            s.dynamic_tools
                .insert(tool.name().to_string(), Arc::clone(tool));
        }
        for artifact in &done.output.artifacts {
            tx.send(AgentEvent::Artifact {
                tool_id: done.id.clone(),
                tool_name: done.name.clone(),
                artifact: artifact.clone(),
            })
            .await
            .ok();
        }
        update.result_blocks.push(ContentBlock::ToolResult {
            tool_use_id: done.id,
            content: done.output.content.clone(),
            is_error: done.output.is_error(),
        });

        // Inject image artifacts as content blocks so the LLM can see them
        // in the next turn. Only when the provider accepts image input.
        if caps.image_input {
            for block in artifact_image_blocks(&done.output.artifacts) {
                update.result_blocks.push(block);
            }
        }

        update.injections.extend(
            done.output
                .injections
                .iter()
                .map(|injection| system_reminder_msg(&injection.text)),
        );
    }

    update
}

/// Convert image artifacts into `ContentBlock::Image` blocks suitable for
/// inclusion in the message history.
///
/// Only artifacts that can be meaningfully represented as vision input are
/// converted:
/// - `ArtifactKind::Image` with inline data → base-64 image block.
/// - `ArtifactKind::Image` with a reference URI → URL image block.
/// - File artifacts whose MIME type starts with `image/` → same logic.
///
/// Non-image artifacts (JSON, charts without image data, etc.) are skipped.
fn artifact_image_blocks(artifacts: &[Artifact]) -> Vec<ContentBlock> {
    let mut blocks = Vec::new();

    for artifact in artifacts {
        let is_image_kind = matches!(artifact.kind, ArtifactKind::Image);
        let is_image_mime = artifact
            .mime_type
            .as_deref()
            .is_some_and(|m| m.starts_with("image/"));

        if !is_image_kind && !is_image_mime {
            continue;
        }

        match &artifact.content {
            ArtifactContent::Inline { data } => {
                // Determine MIME type: explicit > inferred from kind.
                let media_type = artifact
                    .mime_type
                    .clone()
                    .unwrap_or_else(|| "image/png".to_string());

                use base64::Engine;
                let encoded = base64::engine::general_purpose::STANDARD.encode(data);

                blocks.push(ContentBlock::Image {
                    source: ImageSource::Base64 {
                        media_type,
                        data: encoded,
                    },
                });
            }
            ArtifactContent::Reference { uri } => {
                blocks.push(ContentBlock::Image {
                    source: ImageSource::Url(uri.clone()),
                });
            }
        }
    }

    blocks
}

fn append_tool_history(messages: &mut Vec<Message>, update: ToolHistoryUpdate) {
    if !update.result_blocks.is_empty() {
        messages.push(Message::with_id(
            uuid::Uuid::new_v4().to_string(),
            Role::User,
            update.result_blocks,
        ));
    }
    messages.extend(update.injections);
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
