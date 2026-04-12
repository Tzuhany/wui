# Architecture Guide (English)

## 1. Positioning and Scope

Wui is a runtime for executing agent behavior, not a framework for prescribing agent strategy.
That distinction drives most architectural choices in this repository.

The runtime is responsible for:
- streaming model output
- scheduling tool calls
- enforcing permission and human approval flow
- handling context pressure
- preserving session and checkpoint state

The runtime is not responsible for:
- evaluating whether the model made a good plan
- encoding product-specific workflows into the core
- hiding provider limitations behind generic abstractions

In practice, Wui aims for a narrow, explicit execution surface with strong runtime semantics.

## 2. Layered Crate Topology

The workspace is organized as a layered system with one-way dependencies:

- `wui-core`: stable vocabulary (traits, messages, events, tool outputs)
- `wui`: executor implementation and public builder/runtime API
- extension crates (`wui-memory`, `wui-mcp`, `wui-observe`, `wui-spawn`, `wui-skills`, `wui-eval`): optional capabilities

Design consequence:
- user-defined tools/providers/hooks can depend on `wui-core` only
- extension capabilities can evolve without forcing core vocabulary changes
- runtime policy remains in `wui`, not leaked into trait contracts

This separation is the main reason the codebase stays understandable as features grow.

## 3. Runtime Loop: The Execution Spine

Most behavior is concentrated in the run loop (`crates/wui/src/runtime/run`).

A single iteration is structured as:

1. pre-turn checks: context pressure and optional compression
2. request build: `ChatRequest` from system + history + tool catalog
3. provider stream call (with retry policy)
4. stream parse with eager dispatch:
- `TextDelta` / `ThinkingDelta` forwarded immediately
- `ToolUseEnd` triggers pre-tool hook + permission verdict immediately
- non-interactive outcomes (allow/deny) are handled immediately
- interactive outcomes are queued for deferred HITL approval
- `MessageEnd` closes the model-stream phase
5. deferred HITL authorization (concurrent):
- only deferred tools are processed here, after model output is complete
- deferred approvals run concurrently; one slow decision does not block others
- each tool starts as soon as its own authorization resolves
6. collection and post-tool hooks
7. ordered tool result write-back to history
8. context injections (system reminders)
9. stop-condition evaluation

Two details matter for system behavior:
- backpressure is explicit through bounded channels
- tool completion can be concurrent, but history write-back is in submission order

That combination preserves responsiveness without losing determinism.

## 4. Event Semantics and Data Contracts

Wui distinguishes two event layers:

- `StreamEvent`: raw provider stream protocol
- `AgentEvent`: runtime-level event protocol exposed to callers

Why the split exists:
- providers differ in streaming details, chunking, and transport semantics
- applications need stable, semantically meaningful events

The runtime translates many low-level stream fragments into fewer, higher-level agent events,
while preserving ordering guarantees for text and tool lifecycle notifications.

This makes provider adapters interchangeable without destabilizing application code.

## 5. Tool Model and Scheduling Semantics

The tool contract (`Tool` trait) keeps invocation shape simple:
- name, description, input schema, async call

Behavioral semantics are declared per invocation via metadata:
- `ToolMeta`: concurrency, readonly/destructive flags, interaction requirements, permission scope key
- `ExecutorHints`: timeout/output/retry/display hints for the Wui executor

Important scheduling rule:
- concurrency is decided from the actual call input, not just tool type

That allows one tool implementation (for example, a shell tool) to run safe operations concurrently
while forcing risky invocations into stricter execution paths.

The runtime also guarantees that tool results are appended to conversation history
in the original submission order, even when execution completes out of order.

## 6. Permission Pipeline and HITL

Permission is not an afterthought; it is part of the execution pipeline.

Evaluation order is designed to preserve hard safety boundaries:
1. pre-tool hook decision
2. structural checks (for example interaction-required tools under headless mode)
3. static deny rules
4. session deny memory
5. static allow rules
6. session allow memory
7. mode-specific behavior (`Ask`, `Auto`, readonly gating)

Human approval (`AgentEvent::Control`) is emitted after the model has finished speaking for the turn,
but only for calls that actually require interactive consent. Calls that can be decided without HITL
may proceed during streaming.

The control handle then resumes execution asynchronously with no polling and no blocked threads.

## 7. Context Pressure, Compression, and Overflow Handling

Long-running sessions are treated as a normal operating condition.

The compression pipeline is tiered and bounded:
- lightweight trimming
- structural collapse of older history
- summarization when needed

Escalation is progressive and stops once pressure is relieved.
If pressure remains critical, the loop can terminate with an explicit overflow reason
or invoke a user-supplied overflow callback for custom recovery.

Operationally, this avoids two common failure modes:
- silent token budget burn with low informational gain
- abrupt prompt-too-long failures without recovery hooks

## 8. Session and Checkpoint Semantics

Wui separates turn history persistence from per-session permission memory.

- session store: persists message history across turns
- session allow/deny memory: runtime-ephemeral behavior, reset with new session lifecycle

Checkpointing is iteration-boundary based, not mid-tool-call snapshotting.
That keeps recovery semantics predictable: resume from the last completed boundary,
not from an arbitrary in-flight execution point.

## 9. Extension Architecture

Extension crates represent optional capability domains:
- `wui-memory`: reference memory interfaces and tools
- `wui-mcp`: MCP client and lazy tool catalog
- `wui-observe`: timeline and OpenTelemetry integration
- `wui-spawn`: sub-agent delegation infrastructure
- `wui-skills`: file-backed skill discovery
- `wui-eval`: mock provider and harness for deterministic testing

The important architectural property is optionality.
The runtime does not force any single memory model, orchestration style,
or observability stack into the base dependency path.

## 10. Engineering Trade-offs

Current strengths:
- boundary discipline between vocabulary, runtime machinery, and extensions
- explicit runtime invariants documented in `docs/runtime-invariants.md`
- testability through `wui-eval` and integration-level tests

Current trade-offs:
- extension APIs marked beta/stabilizing may evolve
- advanced features increase conceptual load for first-time adopters
- production integrations still require application-level policy design

This is a deliberate bias: stricter execution semantics over “magic” automation.

## 11. Suggested Reading Path for Contributors

For a fast architecture onboarding pass:

1. `README.md` (scope and crate map)
2. `docs/philosophy.md` (design laws and boundary rules)
3. `docs/runtime-invariants.md` (behavioral guarantees)
4. `crates/wui/src/runtime/run/` (loop implementation)
5. one end-to-end example from `examples/`

That sequence mirrors how the project is designed: principles first, invariants second, code third.
