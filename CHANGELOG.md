# Changelog

All notable changes are documented here. Wui follows [Semantic Versioning](https://semver.org/).

Format: `Added` / `Changed` / `Fixed` / `Removed` per section.
Breaking changes are marked **[breaking]**.

---

## [Unreleased]

### Added
- `ExecutorHints` freeze policy comment — documents the closed-set rule for future maintainers.
- `docs/architecture.md` — Crate Maturity table with beta→stable promotion criteria.
- `docs/philosophy.md` — "The Four Questions" abstraction criteria.
- Integration tests for HITL approve/deny, session history preservation, and dynamic tool exposure.
- `CHANGELOG.md` — this file; establishes release discipline going forward.

### Changed
- CI: added `cargo fmt --all --check`, `cargo doc --workspace --no-deps -D warnings`.
- `wui-memory` crate header: clearer "companion, not the product" framing.
- `wui-spawn`, `wui-workflow`, `wui-mcp` crate headers: added explicit scope statement.
- `README.md`: added "What is stable today" table and "Recommended learning path".
- `README.md`: `wui-memory` description now leads with capability traits, not implementation.

---

## [0.1.6] — 2026-03-XX

### Added
- `wui-eval`: `MockProvider`, `AgentHarness`, `ScenarioRunner` for deterministic testing.
- `wui-memory`: `VectorStore` trait + `InMemoryVectorStore` + `SemanticMemoryTool`.
- `wui`: `SummarizingCompressor` — LLM-call-based L3 compression strategy.
- `wui`: `CheckpointStore` trait + `InMemoryCheckpointStore` + `FileCheckpointStore`.
- `wui-core`: `StructuredRun` — XML-based structured output with `extract` / `extract_as`.
- `wui-core`: `ImageSource`, `DocumentSource`, `ContentBlock::Image/Document` for multi-modal input.
- `wui-core`: `extract_tag` / `extract_tags` in `fmt`.
- `wui-core`: `ToolMeta` / `ExecutorHints` split — semantic vs executor-specific tool properties.
- `wui-core`: `ToolInput` typed accessor wrapper.
- `wui`: `SubAgent` — wraps an agent as a synchronous tool.
- Integration test suite (6 tests) covering text response, tool call, deny rules, max_iter, checkpoint, retry.
- `docs/architecture.md` fully updated to current design.
- `docs/core-audit.md` — boundary audit for wui-core.

### Changed
- **[breaking]** Tool trait: replaced 10 individual hint methods with `fn meta(&self, input) -> ToolMeta` and `fn executor_hints(&self, input) -> ExecutorHints`.
- Run loop: tool authorization now fires after `MessageEnd` (not mid-stream). HITL prompts appear after the LLM finishes speaking.
- `README.md`: added crate map with maturity labels, "Where to start" section.

---

## [0.1.5] — 2026-02-XX

### Added
- `RetryPolicy` with exponential back-off and equal jitter for provider errors.
- `RunStream` wrapper with map/filter helpers.
- `ToolProgress` events for mid-execution status reporting.
- Per-tool execution timeout support.

### Changed
- Streaming concurrent tool executor: `JoinSet`-based, polls during LLM stream.

---

## [0.1.4] and earlier

See git log for historical changes prior to structured release tracking.
