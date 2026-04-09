// ── Message history construction ─────────────────────────────────────────────

use std::sync::Arc;

use tokio::sync::mpsc;

use wui_core::event::AgentEvent;
use wui_core::message::{ContentBlock, Message, Role};

use super::checkpoint::RunCheckpoint;
use super::{CompletedTool, IterationCtx, RunConfig, RunState};

/// Build the assistant message and tool results, append them to history,
/// update usage/iterations, and save a checkpoint if applicable.
pub(super) async fn assemble_history(
    ctx: IterationCtx,
    s: &mut RunState,
    config: &RunConfig,
    tx: &mpsc::Sender<AgentEvent>,
) {
    // Apply auth injections.
    s.messages.extend(ctx.auth_injections);

    // Build assistant message: thinking, text, then tool calls.
    let mut blocks = ctx.assistant_blocks;
    if !ctx.text_buf.is_empty() {
        blocks.insert(0, ContentBlock::Text { text: ctx.text_buf });
    }
    if !ctx.thinking_buf.is_empty() {
        blocks.insert(
            0,
            ContentBlock::Thinking {
                text: ctx.thinking_buf,
            },
        );
    }
    if !blocks.is_empty() {
        s.messages.push(Message::assistant(blocks));
    }

    // Append tool results in submission order.
    if !ctx.submission_order.is_empty() {
        let mut completed_map = ctx.completed_map;
        let done_tools: Vec<CompletedTool> = ctx
            .submission_order
            .iter()
            .filter_map(|id| completed_map.remove(id))
            .collect();

        for done in &done_tools {
            for tool in &done.output.expose_tools {
                s.dynamic_tools
                    .insert(tool.name().to_string(), Arc::clone(tool));
            }
        }

        for done in &done_tools {
            for artifact in &done.output.artifacts {
                tx.send(AgentEvent::Artifact {
                    tool_id: done.id.clone(),
                    tool_name: done.name.clone(),
                    artifact: artifact.clone(),
                })
                .await
                .ok();
            }
        }

        let result_blocks: Vec<ContentBlock> = done_tools
            .iter()
            .map(|done| ContentBlock::ToolResult {
                tool_use_id: done.id.clone(),
                content: done.output.content.clone(),
                is_error: done.output.is_error(),
            })
            .collect();

        if !result_blocks.is_empty() {
            s.messages.push(Message::with_id(
                uuid::Uuid::new_v4().to_string(),
                Role::User,
                result_blocks,
            ));
        }

        for done in &done_tools {
            for injection in &done.output.injections {
                s.messages.push(system_reminder_msg(&injection.text));
            }
        }
    }

    tracing::debug!(
        stop_reason    = ?ctx.stop_reason,
        input_tokens   = ctx.usage.input_tokens,
        output_tokens  = ctx.usage.output_tokens,
        cache_read     = ctx.usage.cache_read_tokens,
        cache_write    = ctx.usage.cache_write_tokens,
        "llm call complete"
    );

    s.total_usage += ctx.usage;
    s.iterations += 1;

    // Checkpoint save (only after tool-use iterations).
    if !ctx.submission_order.is_empty() {
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
