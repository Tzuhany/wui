// ============================================================================
// Tool authorization — permission checks and HITL approval.
//
// The authorization pipeline:
//   1. Pre-tool hook (may block or mutate input).
//   2. Compute per-invocation metadata (ToolMeta, permission matcher).
//   3. Evaluate verdict (static rules → session → mode).
//   4. HITL approval via ControlHandle (Ask mode only).
// ============================================================================

use tokio::sync::mpsc;

use wui_core::event::{
    AgentEvent, ControlDecision, ControlHandle, ControlKind, ControlRequest, ControlResponse,
};
use wui_core::hook::HookDecision;
use wui_core::message::Message;
use wui_core::tool::ToolCallId;

use super::executor::CompletedTool;
use super::permission::{self, PermissionOutcome, PermissionVerdict};
use super::registry::ToolRegistry;
use super::run::RunConfig;

fn output_hook_blocked(reason: impl Into<String>) -> wui_core::tool::ToolOutput {
    wui_core::tool::ToolOutput {
        content: reason.into(),
        failure: Some(wui_core::tool::FailureKind::HookBlocked),
        ..Default::default()
    }
}

fn output_permission_denied(reason: impl Into<String>) -> wui_core::tool::ToolOutput {
    wui_core::tool::ToolOutput {
        content: reason.into(),
        failure: Some(wui_core::tool::FailureKind::PermissionDenied),
        ..Default::default()
    }
}

/// Outcome of a concurrent authorization task.
pub(super) enum AuthOutcome {
    Allowed {
        id: ToolCallId,
        name: String,
        input: serde_json::Value,
        injections: Vec<Message>,
    },
    Denied {
        tool: CompletedTool,
        injections: Vec<Message>,
    },
}

/// Verify that a tool call is permitted to run.
pub(super) async fn authorize_tool(
    config: &RunConfig,
    registry: &ToolRegistry,
    id: ToolCallId,
    name: String,
    input: serde_json::Value,
    tx: &mpsc::Sender<AgentEvent>,
) -> (Result<serde_json::Value, CompletedTool>, Vec<Message>) {
    tracing::info!(tool.id = %id, tool.name = %name, "wui.tool.auth.start");

    // 1. Pre-tool hook — may allow, mutate, or block.
    let input = match config.hooks.pre_tool_use(&name, &input).await {
        HookDecision::Block { reason } => {
            return (
                Err(CompletedTool::immediate(
                    id,
                    name,
                    output_hook_blocked(reason),
                )),
                vec![],
            )
        }
        HookDecision::Mutate { input: new } => new,
        HookDecision::Allow | HookDecision::MutateOutput { .. } => input,
    };

    // 2. Gather metadata and evaluate verdict.
    let (verdict, is_destructive, tool_is_readonly, perm_key) =
        evaluate_permission(config, registry, &name, &input).await;

    match verdict {
        PermissionVerdict::Allowed { source } => {
            tracing::debug!(tool = %name, ?source, "permission granted");
            return (Ok(input), vec![]);
        }
        PermissionVerdict::Denied { reason, source } => {
            tracing::debug!(tool = %name, ?source, %reason, "permission denied");
            return (
                Err(CompletedTool::immediate(
                    id,
                    name,
                    output_permission_denied(reason),
                )),
                vec![],
            );
        }
        PermissionVerdict::NeedsApproval => {}
    }

    // 3. HITL approval.
    let ctrl_req = build_control_request(&name, perm_key.as_deref(), is_destructive);

    match permission::check(&config.permission, &ctrl_req, tool_is_readonly, &input) {
        PermissionOutcome::Allowed => (Ok(input), vec![]),
        PermissionOutcome::Denied { reason } => (
            Err(CompletedTool::immediate(
                id,
                name,
                output_permission_denied(reason),
            )),
            vec![],
        ),
        PermissionOutcome::NeedsApproval => {
            await_approval(config, id, name, input, ctrl_req, tx).await
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Gather per-invocation tool metadata and run the permission verdict pipeline.
async fn evaluate_permission(
    config: &RunConfig,
    registry: &ToolRegistry,
    name: &str,
    input: &serde_json::Value,
) -> (PermissionVerdict, bool, bool, Option<String>) {
    let tool_impl = registry.get(name);
    let tool_meta = tool_impl
        .as_deref()
        .map(|t| t.meta(input))
        .unwrap_or_default();

    let perm_key = tool_meta.permission_key.clone();
    let is_destructive = tool_meta.destructive;
    let tool_is_readonly = tool_meta.readonly;

    let perm_matcher = tool_impl
        .as_deref()
        .and_then(|t| t.permission_matcher(input));
    let matcher_ref: Option<&(dyn Fn(&str) -> bool + Send + Sync)> =
        perm_matcher.as_ref().map(|b| b.as_ref() as _);

    let check = permission::PermissionCheck {
        tool_name: name,
        permission_key: perm_key.as_deref(),
        is_readonly: tool_is_readonly,
        requires_interaction: tool_meta.requires_interaction,
        matcher: matcher_ref,
    };

    let verdict = config
        .rules
        .verdict(&config.session_perms, &config.permission, &check)
        .await;

    (verdict, is_destructive, tool_is_readonly, perm_key)
}

/// Build the `ControlRequest` for a HITL permission prompt.
fn build_control_request(
    name: &str,
    perm_key: Option<&str>,
    is_destructive: bool,
) -> ControlRequest {
    let base = match perm_key {
        Some(key) => format!("call {name}({key})"),
        None => format!("call {name}"),
    };
    let description = if is_destructive {
        format!("{base} [destructive — cannot be undone]")
    } else {
        base
    };
    ControlRequest {
        id: uuid::Uuid::new_v4().to_string(),
        kind: ControlKind::PermissionRequest {
            tool_name: name.to_string(),
            description,
        },
    }
}

/// Emit a `ControlHandle`, wait for the human's decision, and return the outcome.
async fn await_approval(
    config: &RunConfig,
    id: ToolCallId,
    name: String,
    input: serde_json::Value,
    ctrl_req: ControlRequest,
    tx: &mpsc::Sender<AgentEvent>,
) -> (Result<serde_json::Value, CompletedTool>, Vec<Message>) {
    let request_id = ctrl_req.id.clone();
    let (handle, rx) = ControlHandle::new(ctrl_req);
    tx.send(AgentEvent::Control(handle)).await.ok();

    let response = rx.await.unwrap_or_else(|_| {
        ControlResponse::deny(
            request_id,
            "control handle dropped before response was sent",
        )
    });

    let injection = Message::system(permission::response_to_system_message(&response));

    let result = match response.decision {
        ControlDecision::Deny { reason } => Err(CompletedTool::immediate(
            id,
            name,
            output_permission_denied(reason),
        )),
        ControlDecision::DenyAlways { reason } => {
            config.session_perms.set_always_deny(name.clone()).await;
            Err(CompletedTool::immediate(
                id,
                name,
                output_permission_denied(reason),
            ))
        }
        ControlDecision::ApproveAlways => {
            config.session_perms.set_always_allow(name).await;
            Ok(input)
        }
        ControlDecision::Approve { .. } => Ok(input),
    };

    (result, vec![injection])
}
