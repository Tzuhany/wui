# The Wui Philosophy

## The Central Idea

There is a temptation, when building an agent framework, to make it smart.

To encode routing logic. To build decision trees. To design graph structures
that tell the model what to do next. To create role systems, crews,
supervisors, and other abstractions that quietly replace model judgment with
framework judgment.

Wui refuses this temptation.

> **The framework is an executor, not a thinker.**

The model decides. The framework executes. It streams, schedules, validates,
pauses, resumes, compresses, retries, and records. It does not plan on the
model's behalf.

This is not austerity for its own sake. It is how we get reliability. The
less the framework pretends to think, the more honestly it can execute.

---

## The Design Laws

### I. Loop is Truth

There is one primary runtime loop. It receives model output, starts tools as
soon as they are describable, collects results, manages context pressure, and
continues until the run is complete.

Everything else exists to serve this loop.

If a feature cannot be explained as part of the loop, or as a clean extension
around the loop, it probably does not belong in the runtime core.

### II. Streaming Before Orchestration

The correct unit of execution is not a graph node. It is a live stream.

The model should be able to emit text, begin a tool call, continue reasoning,
and finish while tools are already running. Work starts as soon as there is
enough information to start it.

Streaming tool execution is not an optimization. It is the natural execution
model for agents.

### III. Concurrency is a Per-Invocation Decision

Concurrency belongs to the actual call, not the tool type in the abstract.

A shell tool may allow `ls` concurrently while forcing `rm` to run
sequentially. The runtime must decide based on the invocation, not a static
"this tool is concurrent" flag.

This keeps scheduling honest and keeps dangerous tools from acquiring
accidental blanket permissions.

### IV. Context is Finite, Compression is Grace

Every long-running agent will hit context pressure. This is physics, not a
bug.

Wui treats compression as a first-class lifecycle event. Trimming,
collapsing, and summarizing are runtime responsibilities. The loop should
continue gracefully rather than pretending history is infinite or pushing the
problem onto the caller.

### V. Trust Must Be Earned

The safe default is `Ask`, not `Auto`.

An agent that can always act is merely a powerful tool. An agent that can
pause, present a decision, and continue from a human response is a
collaborator.

Human-in-the-loop is not a UI garnish. It is a core runtime behavior.

### VI. Keep Core Honest

`wui-core` should remain brutally small.

Core exists to define the irreducible vocabulary:

- `Provider`
- `Tool`
- `Hook`
- message and event types

Everything else must justify itself.

If a concept is required for the loop to run, it belongs in the runtime. If it
is one useful way to extend or compose agents, it belongs in an extension
crate. If it mainly reflects an application's worldview, it does not belong in
the framework at all.

### VII. Runtime Over Product

Wui is a runtime, not a pre-baked agent product.

Memory, delegation, MCP, persistence, and discovery may all exist as optional
capabilities, but they should not drag the framework into silently imposing one
product worldview on every application.

The framework may provide a reference implementation. It must not confuse a
reference implementation with the universal shape of the problem.

---

## Core Boundary Rules

These rules are stricter than "this seems convenient."

### What belongs in `wui-core`

- concepts every runtime implementation must agree on
- contracts implemented by users or providers
- data structures that cross crate boundaries as stable vocabulary

### What belongs in `wui`

- the query loop
- tool scheduling and ordering
- permission flow
- retries and recovery
- compression
- session persistence
- provider adapters bundled for convenience

### What belongs in extension crates

- memory helpers
- MCP bridges
- delegation metadata
- provider-specific convenience layers that are not universal

### What must not leak into core

- a specific memory product shape
- a specific delegation model
- provider-specific tuning knobs
- management-oriented abstractions disguised as agent capabilities

When in doubt, prefer moving a concept *out* of core.

---

## Code Aesthetics

Code beauty is not decoration. It is how correctness remains legible.

### 1. Names must tell the truth

Every public name should describe what the thing really is, not what we wish
it might become later.

- If it only stores completed turn history, do not name or document it like a
  paused-run snapshot.
- If a type is provider-specific, do not hide it behind generic language.
- If a capability is optional, do not make it feel fundamental.

### 2. Public API is a promise

Every `pub` is a commitment. Every example and every README snippet is part of
the API surface.

A feature is not "supported" because a type exists. It is supported when:

- the code compiles
- the public path is stable
- the docs match reality
- the examples compile

Broken examples are broken promises.

### 3. Runtime errors must fail honestly

Silent fallback is almost always uglier than explicit failure.

- malformed tool input should not quietly become `{}`
- invalid schemas should not quietly disable validation
- provider capability gaps should not masquerade as generic support

When the framework knows something is wrong, it should say so clearly and as
early as possible.

### 4. Comments explain intent, not insecurity

Comments should illuminate why a design exists, not compensate for muddy code.

Good comments explain:

- a provider protocol quirk
- an ordering guarantee
- a recovery invariant
- why a boundary exists

Bad comments restate obvious code or make promises the code does not keep.

### 5. Small surfaces, deep behavior

We prefer:

- smaller APIs with stronger guarantees
- fewer concepts with clearer boundaries
- one strong path instead of three half-supported paths

Convenience is welcome. Conceptual sprawl is not.

### 6. Reference implementations must stay humble

Example stores, in-memory backends, and helper crates should read like
reference implementations, not declarations of ontology.

They may be useful and opinionated. They must not pretend their shape is the
shape of the entire domain.

---

## Review Standard

When we review Wui, we judge code against these questions:

1. Does this make the loop more correct, or merely more featureful?
2. Does this keep `wui-core` smaller and more honest, or does it leak runtime
   or product concerns into vocabulary?
3. Does this API tell the truth about what is implemented today?
4. Does this failure mode surface clearly, or is it being silently blurred?
5. Does this extension feel like a capability, or like the framework imposing
   an application worldview?
6. Would a new contributor understand the boundary from the code alone?

If the answer is "not really," the code is not finished.

---

## The Four Questions

Before adding any new abstraction — a new type, field, trait, or crate — ask these four questions in order:

1. **Is it vocabulary?** — does every agent runtime need to agree on this concept? If yes: `wui-core`.
2. **Is it runtime machinery?** — is it required for the main loop to run correctly? If yes: `wui`.
3. **Is it a companion capability?** — useful for *some* agents, not all? If yes: an extension crate, opt-in.
4. **Is it a product opinion?** — does it encode one application's worldview? If yes: it belongs in the application, not the framework.

If you can't confidently answer "yes" to exactly one question, the abstraction is probably not ready. Either the concept needs more clarity, or it doesn't belong in the framework at all.

This applies to everything: fields on `ToolMeta`, methods on `Tool`, new types in `wui-core`, new methods on `AgentBuilder`, new extension crates.

---

## On Scope

Wui does not aim to include:

- a database driver
- a vector store
- an HTTP server
- a graph execution engine
- a prompt templating language
- an agent role system

Applications may bring these. Wui provides the runtime that lets them act.

---

## The Name

*Wui* (呜呼) is a classical Chinese exclamation: motion, force, release.

A loop starting. A tool running. A decision becoming action.

Wui.

---

## Layer Boundaries

Three layers. One direction of flow. No feedback from the outer layers into the inner ones.

### 1. Core runtime — `wui-core` and `wui`

The runtime is a pure executor.

It streams text and tool calls, schedules concurrent execution, manages context pressure, enforces permissions, and records what happened. It does not decide what to remember, how to compress a specific domain's history, or what counts as a "good" result.

The runtime does not think. It executes what the LLM decides.

`wui-core` defines the irreducible vocabulary: `Provider`, `Tool`, `Hook`, message types, event types. No runtime logic, no HTTP clients, no product opinions.

`wui` adds the execution machinery: the query loop, tool scheduling, the compression pipeline, the permission flow, session persistence, and bundled provider adapters.

Neither crate imports from any extension crate. This is enforced structurally — extension crates depend on core, never the reverse.

### 2. Extension layer — `wui-memory`, `wui-skills`, `wui-spawn`, `wui-eval`, `wui-mcp`, etc.

Extensions are product-level components built on the runtime.

They are explicitly allowed to be opinionated. `SummarizingCompressor` makes an LLM call. `SemanticMemoryTool` embeds text before storing it. `SkillsCatalog` reads from the filesystem. `McpCatalog` lazily connects to external MCP servers.

This is not a violation of the philosophy. Extensions are *expected* to add product-level opinions — that is their job. The constraint is that these opinions live in extension crates, not in the runtime core. An application that does not use memory should not pay for memory's opinions; an application that does not use workflows should not have workflow assumptions baked into its loop.

A reference implementation is a useful starting point, not a declaration that its shape is the universal answer.

### 3. The boundary rule

> **If it must be true for *any* agent to work correctly → core.**
> **If it is useful for *some* agents → extension.**

Concretely:

- Tool execution, context management, and permission enforcement must work for every agent → they live in the runtime.
- Remembering facts across sessions is useful for *some* agents, not all → `wui-memory`.
- Connecting MCP servers is useful for *some* agents, not all → `wui-mcp`.
- Summarising old messages is one valid compression strategy, not the only one → `SummarizingCompressor` in `wui`, pluggable via `CompressStrategy`.

The runtime should never import from extension crates. If a concept seems "fundamental enough" to belong in core but only makes sense in the context of a specific product feature, it does not belong in core — it belongs in the extension that owns that feature.
