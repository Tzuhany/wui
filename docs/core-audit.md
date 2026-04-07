# wui-core Vocabulary Audit

`wui-core` contains only concepts that are true for **any** agent runtime built on this vocabulary — not Wui-specific implementation choices.

## What belongs in wui-core

| Concept | Reason |
|---------|--------|
| `Provider` | Any LLM runtime needs a provider interface |
| `Tool` | Any agent runtime needs callable tools |
| `Tool::meta()` / `ToolMeta` | Semantic tool properties (concurrent, readonly, destructive, requires_interaction, permission_key) are universal execution-ordering and safety signals |
| `Tool::executor_hints()` / `ExecutorHints` | Wui-executor tuning knobs (timeout, retries, output limits, summary) live here as an *optional* default method, so any tool can express them without depending on `wui`. Runtimes that don't use Wui's executor ignore these fields entirely. |
| `Hook` | Cross-cutting lifecycle interception is a universal pattern |
| `Message` / `ContentBlock` | Conversation representation is vocabulary |
| `StreamEvent` | Provider output protocol |
| `AgentEvent` | Runtime output protocol |
| `ToolOutput` | Tool result vocabulary |
| `fmt` utilities | XML boundary formatting for LLM context |

## What does NOT belong in wui-core

| Concept | Where it lives | Reason |
|---------|---------------|--------|
| `ToolCatalog` | `wui::catalog` | Discovery mechanism, not vocabulary |
| Session persistence | `wui::runtime` | Runtime product feature |
| Compression strategy | `wui::compress` | Runtime product feature |
| Permission system | `wui::runtime` | Wui-specific trust model |
| `RetryPolicy` (provider retry) | `wui::runtime` | Wui executor policy |

## The split between ToolMeta and ExecutorHints

`ToolMeta` fields are **semantic properties** — they describe what a tool *is*, not how any specific executor should run it:

- `concurrent` — execution ordering hint any executor must respect
- `readonly` — semantic property used by permission systems
- `destructive` — semantic property shown to users in HITL prompts
- `requires_interaction` — semantic property: tool cannot run headlessly
- `permission_key` — fine-grained authorization hint

`ExecutorHints` fields are **Wui executor tuning knobs**:

- `summary` — Wui-specific display hint for tool history
- `timeout` — Wui executor policy
- `max_output_chars` — Wui executor policy (output truncation)
- `max_retries` — Wui executor policy (retry on error)

`ExecutorHints` lives in `wui-core` as an *optional* default method on `Tool` (not as a required field of `ToolMeta`) so that tools can express them without depending on `wui` directly. Any runtime that doesn't use Wui's executor simply ignores this method.

## The test

Before adding anything to `wui-core`, ask:

> "If someone wrote a completely different Wui runtime — different executor, different permission model, different scheduling — would they still need this concept?"

If yes → core. If maybe → extension. If no → keep it in `wui`.
