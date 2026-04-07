# wui-skills

File-based instruction discovery built on `wui`'s catalog abstraction.

File-based skill discovery for Wui agents. Each Markdown file in a directory becomes a discoverable tool: when the agent activates a skill, its content is injected as a `<system-reminder>` into the conversation context.

## Install

```toml
[dependencies]
wui-skills = "0.1"
```

## Usage

```rust
use wui_skills::SkillsCatalog;

let agent = Agent::builder(provider)
    .catalog(SkillsCatalog::new("./skills"))
    .build();
```

Skills are discovered lazily on first search — no directory I/O at build time.

## Skill file format

Each `.md` file must begin with YAML-like frontmatter:

```markdown
---
name: git-commit
description: Write conventional commit messages following project style.
---

Always use the imperative mood in the subject line.
Format: `<type>(<scope>): <subject>`
Types: feat, fix, docs, chore, refactor, test
```

The `name` becomes the tool name the LLM calls. The `description` appears in `tool_search` results. The body is injected verbatim as a `<system-reminder>` when the skill is activated.

## How it works

`SkillsCatalog` implements `ToolCatalog` — the LLM discovers skills via the built-in `tool_search` tool rather than having every skill's schema in the initial prompt. This keeps the context window compact for large skill libraries.

Full docs: https://github.com/Tzuhany/wui
