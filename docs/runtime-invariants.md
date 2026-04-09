# Runtime Invariants

Guarantees that hold across all runs. Code references point to where each
invariant is enforced. If you find a violation, it is a bug.

## Tool execution

1. **ToolResult ordering** — tool results are appended to message history in
   the same order the corresponding ToolUse blocks appear in the assistant
   message, regardless of completion order. (`runtime/run/history.rs`
   submission_order + completed_map reconstruction)

2. **ToolUse / ToolResult pairing** — every ToolUse block in the assistant
   message gets exactly one ToolResult in the next user message. Even if
   validation fails, the tool panics, or permission is denied, a result is
   always produced. (`runtime/executor.rs` catch_unwind,
   `runtime/run/parsing.rs` instant failure path)

3. **Schema validation before execution** — `Tool::call()` is never invoked
   with input that fails the tool's `input_schema()`. Invalid input produces
   an immediate `FailureKind::InvalidInput` result. (`executor.rs`
   validate_input)

4. **Retry selectivity** — only `FailureKind::Execution` errors are retried.
   `InvalidInput`, `NotFound`, `PermissionDenied`, and `HookBlocked` are
   deterministic and never retried. (`ToolOutput::is_retryable`)

5. **Sibling cancellation scope** — when a concurrent tool errors, the
   sibling cancellation token fires. This cancels other tools in the same
   batch but does NOT cancel the run itself. (`executor.rs` sibling_cancel
   is a child token of the run's cancel token)

## Event ordering

6. **Terminal event exclusivity** — a run emits exactly one of
   `AgentEvent::Done` or `AgentEvent::Error`, never both, always last.
   (`runtime/run/stream.rs` run_task match)

7. **ToolStart before ToolDone/ToolError** — `ToolStart` is emitted before
   the tool begins executing. `ToolDone` or `ToolError` is emitted after
   post-tool hooks run, in submission order. (`runtime/run/tool_batch.rs`
   emission_guard)

8. **TextDelta ordering** — text deltas are forwarded in the order the
   provider streams them. No buffering, no reordering.

9. **Hook events in submission order** — `PostToolUse` and `PostToolFailure`
   hooks fire in submission order, not completion order.
   (`runtime/run/tool_batch.rs` `for id in &submission_order`)

## Session

10. **Turn serialisation** — `Session::send()` acquires a semaphore permit.
    At most one turn runs at a time. The permit is released on Done or
    Error. (`facade/session.rs` turn_guard + permit slot)

11. **TurnDone precedes Done** — completed session turns emit
    `AgentEvent::TurnDone` immediately before `AgentEvent::Done`. Erroring
    turns emit no `TurnDone`. (`facade/session.rs` TurnCleanup::on_terminal)

12. **History atomicity on Done** — when `AgentEvent::Done` is yielded to
    the caller, both in-memory history and the session store (if configured)
    have already been updated. The next `send()` sees the full turn.
    (`facade/session.rs` TurnCleanup::on_terminal, synchronous lock + async
    store.save)

13. **Error preserves history** — when a run ends with `AgentEvent::Error`,
    `session.messages` stays at its pre-send state. The dangling user
    message is not added to history. (`facade/session.rs` — history is only
    updated inside the Done branch)

## Permission

14. **Deny rules are absolute** — a static deny rule cannot be overridden by
    session decisions, mode, or hooks. (`permission.rs` evaluation order:
    deny first)

15. **Session decisions cannot weaken static denials** — a user's
    `ApproveAlways` cannot override a builder-level `.deny_tool()`.
    (`permission.rs` — deny checked before session memory)

16. **HITL after LLM finishes** — permission prompts are never shown while
    the LLM is still streaming. Tool calls are collected during streaming
    and authorized after MessageEnd. (`runtime/run/tool_batch.rs`
    pending_auths + auth_tasks)

## Compression

17. **L1 before L2 before L3** — compression tiers are applied in order,
    stopping at the first that relieves sufficient pressure. L3 (LLM
    summarisation) is never called if L2 (collapse) was enough.
    (`compress/mod.rs` maybe_compress)

18. **Compression never loses the most recent messages** — L2 collapse and
    L3 summarise only fold the oldest portion of history. The most recent
    `keep_count` messages are always preserved verbatim.
    (`compress/mod.rs` keep_count, split_at)

19. **Emergency compression is bounded** — prompt-too-long triggers at most
    one emergency compression per iteration. If compression doesn't help,
    the error propagates. (`runtime/run/mod.rs` — `continue` restarts the
    iteration, but the next attempt won't re-trigger emergency compression
    for the same request)

## Checkpoint

20. **Checkpoint is per-iteration** — a checkpoint is saved at the end of
    each tool-use iteration, after tool results are appended and injections
    are applied. Resuming from a checkpoint replays from the last completed
    iteration, not mid-tool. (`runtime/run/history.rs` checkpoint save site)
