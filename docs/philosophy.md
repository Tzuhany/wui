# The Wuhu Philosophy

## The Central Idea

There is a temptation, when building an agent framework, to make it smart.

To encode routing logic. To build decision trees. To design elaborate graph structures that guide the agent from one state to the next. To create role systems, crew systems, supervisor agents. To, in short, *think for the LLM*.

Wuhu refuses this temptation.

> **The framework is an executor, not a thinker.**

The LLM decides what to do and how to do it. The framework runs it — perfectly, concurrently, gracefully — and feeds the result back. Nothing more.

This is not a limitation. It is a strength. When the framework stops trying to be intelligent, it becomes something more valuable: *reliable*.

---

## Five Laws

### I. Loop is Truth

There is only one loop. It streams, it executes tools, it compresses context when necessary, it continues. Everything else in the framework exists to serve this loop.

The loop is dumb by design. It does not understand the conversation. It does not plan. It does not decide when to stop — the LLM does, by returning `EndTurn`. The loop just runs until something says stop.

This simplicity is the source of the framework's correctness. A simple loop has simple failure modes.

### II. Concurrency is the Natural State

When a human asks you to do two things, you don't finish the first sentence before starting the first task. You begin acting as soon as you have enough information.

Wuhu applies this to tools.

When the LLM says "I'll search for X" — the search begins *immediately*, while the LLM continues streaming the rest of its response. If the LLM also calls tool Y, Y starts concurrently with X. By the time the LLM finishes its response, the tools are already running — or done.

This is not an optimization. It is the correct model of how work happens.

### III. Context is Finite, Compression is Grace

Every conversation will eventually approach the context window. This is not a bug. It is physics.

Most frameworks treat this as an error to be avoided. Wuhu treats it as a first-class lifecycle event. The three-tier compression pipeline — budget trimming, message collapsing, LLM summarization — runs silently, automatically, and gracefully. The conversation continues. The agent maintains its intent across compressions.

When compression happens, both the user and the agent are informed. There is no pretending the history is infinite. There is only graceful acknowledgment and continuation.

### IV. Trust Must Be Earned

The default permission mode is `Ask`. Not `Auto`.

This reflects a belief: an agent that can act without asking is a tool; an agent that knows *when* to ask is a collaborator. The framework makes it easy to pause, present a decision to the human, and resume from their response.

Human-in-the-loop is not a safety constraint bolted on after the fact. It is a first-class design principle expressed as a first-class API.

### V. Four Traits, Not Four Hundred

`Provider`. `Tool`. `Hook`. `Checkpoint`.

These four traits cover the entire surface of variability in an agent system:
- *What intelligence powers it?* → `Provider`
- *What can it do?* → `Tool`
- *What should stop it?* → `Hook`
- *Where does it remember?* → `Checkpoint`

If you want to change how the LLM is called — implement `Provider`. If you want a new capability — implement `Tool`. If you want to audit or block behavior — implement `Hook`. If you want persistence — implement `Checkpoint`.

Nothing else needs to be a framework concern.

---

## On Elegance

> Code is read far more than it is written.

Wuhu's code is written for the reader. Every type name is chosen for clarity. Every module boundary is a conceptual boundary. Every `pub` is a deliberate promise.

We prefer trait objects over generics at composition points, because `Arc<dyn Tool>` communicates intent where `T: Tool + Send + Sync + 'static` obscures it.

We prefer exhaustive enums over stringly-typed errors, because the compiler should enforce completeness.

We prefer `Result` over `unwrap`, because every failure mode is a design decision.

We prefer a small API surface with deep behavior over a large API surface with shallow behavior.

---

## On Scope

Wuhu does not include:

- A database driver
- An HTTP server
- A vector store
- A prompt templating language
- A graph execution engine
- An agent "role" system

These are all application concerns. An application that needs them should bring them. Wuhu provides the loop; the application provides the world.

---

## The Name

*Wuhu* (呜呼) is a classical Chinese exclamation — the sound of something happening, energy released, motion beginning.

A loop starting. A tool running. A thought completed.

Wuhu.
