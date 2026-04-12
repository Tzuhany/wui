// ============================================================================
// wui-skills — file-based skill discovery.
//
// A SkillsCatalog scans a directory for skills. Each skill becomes a Tool.
// Invoking the tool injects the skill's content into the agent's context
// as a <system-reminder>, or runs it as a fork sub-agent.
//
// Two formats are supported:
//
// 1. Single-file skill — a plain .md file with YAML-like frontmatter:
//
//      skills/
//      └── git-commit.md
//
// 2. Directory skill — a subdirectory containing SKILL.md as the entry
//    point, plus any supporting files (referenced docs, templates, etc.):
//
//      skills/
//      └── release/
//          ├── SKILL.md        ← required entry point
//          ├── checklist.md    ← referenced in SKILL.md, loaded on demand
//          └── template.md
//
//    Supporting files are NOT loaded automatically — SKILL.md should
//    reference them by path so Claude reads them when needed.
//    Use ${SKILL_DIR} in SKILL.md to refer to the skill's own directory:
//
//      See the checklist at ${SKILL_DIR}/checklist.md before proceeding.
//
// Frontmatter format (same for both):
//   ---
//   name: git-commit
//   description: Write conventional commit messages following project style.
//   when_to_use: select:git-commit,commit  # richer search tokens
//   arguments: message                     # exposes $message / $ARGUMENTS
//   context: fork                          # run as sub-agent (default: inline)
//   allowed_tools: Bash, Read, Glob
//   effort: high
//   model: claude-sonnet-4-5-20250514
//   ---
//   ... skill content ...
//
// Content substitutions (applied at invocation time):
//   ${SKILL_DIR}  → absolute path to the skill's directory (dir skills only)
//   $ARGUMENTS    → the full arguments string provided by the caller
//   $1, $2, ...   → positional tokens from the arguments string
//   $name         → named parameter declared in `arguments:`
//
// Shell injection (executed at invocation time):
//   !`git rev-parse --short HEAD`  → replaced with command stdout
//
// The directory is scanned lazily on first search().
// ============================================================================

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::Value;
use tokio::sync::OnceCell;

use wui_core::catalog::{CatalogHit, ToolCatalog};
use wui_core::runner::AgentRunner;
use wui_core::tool::{ContextInjection, Tool, ToolCtx, ToolMeta, ToolOutput};

// ── SkillContext ──────────────────────────────────────────────────────────────

/// How the skill is dispatched when invoked.
#[derive(Debug, Clone, Default, PartialEq)]
enum SkillContext {
    /// Inject content as a `<system-reminder>` into the current agent context.
    #[default]
    Inline,
    /// Run the skill prompt as an independent sub-agent and return its output.
    Fork,
}

// ── SkillManifest ─────────────────────────────────────────────────────────────

/// Structured skill configuration parsed from YAML-like frontmatter.
///
/// # Frontmatter format
///
/// ```yaml
/// ---
/// name: git-commit
/// description: Write conventional commit messages.
/// when_to_use: commit message conventional changelog
/// arguments: message scope
/// context: fork
/// allowed_tools: Bash, Read, Glob
/// effort: high
/// model: claude-sonnet-4-5-20250514
/// ---
/// ```
#[derive(Debug, Clone, Default)]
pub struct SkillManifest {
    /// Extra text included in search scoring but not shown in description.
    /// Useful for synonyms, trigger keywords, and slash-command aliases.
    pub when_to_use: Option<String>,
    /// Declared parameter names. When non-empty the tool exposes an
    /// `arguments` input field. Values are substituted as `$name`.
    pub arguments: Vec<String>,
    /// Dispatch mode: `inline` (default) or `fork`.
    pub(crate) context: SkillContext,
    /// Tools that should be additionally allowed when this skill is active.
    pub allowed_tools: Vec<String>,
    /// Override the agent's effort level when this skill is active.
    pub effort: Option<String>,
    /// Override the agent's model when this skill is active.
    pub model: Option<String>,
}

// ── SkillsCatalog ─────────────────────────────────────────────────────────────

/// A [`ToolCatalog`] that discovers skills from Markdown files in a directory.
///
/// Each `.md` file (or `SKILL.md` directory) with valid frontmatter becomes a
/// callable tool. Invoking the tool either injects the skill's Markdown content
/// as a `<system-reminder>` (default) or runs it as a sub-agent (`context: fork`).
///
/// The directory is scanned lazily on first `search()` call.
///
/// # Example
///
/// ```rust,ignore
/// Agent::builder(provider)
///     .catalog(SkillsCatalog::new("./skills"))
///     .build()
///
/// // With fork support:
/// Agent::builder(provider)
///     .catalog(SkillsCatalog::new("./skills").with_fork_runner(agent.clone()))
///     .build()
/// ```
pub struct SkillsCatalog {
    dir: PathBuf,
    fork_runner: Option<Arc<dyn AgentRunner>>,
    skills: OnceCell<Vec<Arc<SkillTool>>>,
}

impl SkillsCatalog {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            fork_runner: None,
            skills: OnceCell::new(),
        }
    }

    /// Provide a runner for skills that declare `context: fork`.
    ///
    /// Without a runner, `context: fork` skills fall back to inline mode.
    pub fn with_fork_runner(mut self, runner: impl AgentRunner) -> Self {
        self.fork_runner = Some(Arc::new(runner));
        self
    }

    async fn load(&self) -> anyhow::Result<&Vec<Arc<SkillTool>>> {
        self.skills
            .get_or_try_init(|| async {
                let mut skills = Vec::new();
                let mut entries = tokio::fs::read_dir(&self.dir).await?;

                while let Some(entry) = entries.next_entry().await? {
                    let path = entry.path();
                    if path.is_file() && path.extension().is_some_and(|e| e == "md") {
                        match parse_skill_file(&path).await {
                            Ok(skill) => skills.push(Arc::new(skill)),
                            Err(e) => tracing::warn!(
                                path = %path.display(),
                                error = %e,
                                "skipping skill file with invalid frontmatter"
                            ),
                        }
                    } else if path.is_dir() {
                        let skill_md = path.join("SKILL.md");
                        if skill_md.exists() {
                            match parse_skill_dir(&path).await {
                                Ok(skill) => skills.push(Arc::new(skill)),
                                Err(e) => tracing::warn!(
                                    path = %path.display(),
                                    error = %e,
                                    "skipping skill directory with invalid SKILL.md"
                                ),
                            }
                        }
                    }
                }

                Ok(skills)
            })
            .await
    }
}

#[async_trait]
impl ToolCatalog for SkillsCatalog {
    fn name(&self) -> &str {
        "skills"
    }

    async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<CatalogHit>> {
        let skills = self.load().await?;

        let query_lower = query.to_lowercase();
        let mut scored: Vec<(f32, Arc<SkillTool>)> = skills
            .iter()
            .filter_map(|s| {
                // Include when_to_use in the searchable corpus.
                let when_to_use = s
                    .manifest
                    .when_to_use
                    .as_deref()
                    .unwrap_or("")
                    .to_lowercase();
                let text =
                    format!("{} {} {}", s.skill_name, s.description, when_to_use).to_lowercase();
                let score = text_overlap(&query_lower, &text);
                if score > 0.0 {
                    Some((score, Arc::clone(s)))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let fork_runner = self.fork_runner.clone();
        Ok(scored
            .into_iter()
            .take(limit)
            .map(|(score, skill)| {
                // Attach the fork runner at search time so callers don't need
                // to know about SkillsCatalog internals.
                let tool: Arc<dyn Tool> = if skill.manifest.context == SkillContext::Fork {
                    Arc::new(ForkableSkillTool {
                        inner: Arc::clone(&skill),
                        runner: fork_runner.clone(),
                    })
                } else {
                    skill as Arc<dyn Tool>
                };
                CatalogHit { tool, score }
            })
            .collect())
    }
}

fn text_overlap(query: &str, text: &str) -> f32 {
    let q: std::collections::HashSet<&str> = query.split_whitespace().collect();
    let t: std::collections::HashSet<&str> = text.split_whitespace().collect();
    if q.is_empty() {
        return 0.0;
    }
    q.intersection(&t).count() as f32 / q.len() as f32
}

// ── SkillTool ─────────────────────────────────────────────────────────────────

struct SkillTool {
    skill_name: String,
    description: String,
    content: String,
    manifest: SkillManifest,
    /// For directory-based skills: the skill's own directory.
    skill_dir: Option<PathBuf>,
}

impl SkillTool {
    /// Build the injected content: apply ${SKILL_DIR}, shell injections,
    /// argument substitution, and manifest hints.
    async fn build_content(&self, args: &str) -> anyhow::Result<String> {
        // 1. ${SKILL_DIR} substitution.
        let mut text = match &self.skill_dir {
            Some(dir) => self
                .content
                .replace("${SKILL_DIR}", &dir.display().to_string()),
            None => self.content.clone(),
        };

        // 2. Shell injection: replace !`cmd` with command stdout.
        let work_dir = self.skill_dir.clone();
        text = execute_shell_injections(&text, work_dir.as_deref()).await?;

        // 3. Argument substitution.
        text = substitute_arguments(&text, args, &self.manifest.arguments);

        // 4. Append allowed_tools hint.
        if !self.manifest.allowed_tools.is_empty() {
            text.push_str(&format!(
                "\n\nThis skill works best with these tools: {}",
                self.manifest.allowed_tools.join(", "),
            ));
        }

        Ok(text)
    }

    fn structured_manifest(&self) -> Option<Value> {
        if !self.manifest.allowed_tools.is_empty()
            || self.manifest.effort.is_some()
            || self.manifest.model.is_some()
        {
            Some(serde_json::json!({
                "skill_manifest": {
                    "allowed_tools": self.manifest.allowed_tools,
                    "effort": self.manifest.effort,
                    "model": self.manifest.model,
                }
            }))
        } else {
            None
        }
    }

    /// JSON Schema for this skill's tool input.
    fn schema(&self) -> Value {
        if self.manifest.arguments.is_empty() {
            serde_json::json!({ "type": "object", "properties": {}, "required": [] })
        } else {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "arguments": {
                        "type": "string",
                        "description": format!(
                            "Arguments for the skill. Parameters: {}",
                            self.manifest.arguments.join(", ")
                        )
                    }
                },
                "required": []
            })
        }
    }
}

#[async_trait]
impl Tool for SkillTool {
    fn name(&self) -> &str {
        &self.skill_name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn input_schema(&self) -> Value {
        self.schema()
    }

    async fn call(&self, input: Value, _ctx: &ToolCtx) -> ToolOutput {
        let args = input
            .get("arguments")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let injection = match self.build_content(args).await {
            Ok(c) => c,
            Err(e) => return ToolOutput::error(format!("skill error: {e}")),
        };

        let mut output =
            ToolOutput::success(format!("Skill '{}' loaded into context.", self.skill_name))
                .inject(ContextInjection::new(injection));

        if let Some(structured) = self.structured_manifest() {
            output = output.with_structured(structured);
        }

        output
    }
}

// ── ForkableSkillTool ─────────────────────────────────────────────────────────

/// Wrapper that runs a skill as a sub-agent when `context: fork`.
///
/// Falls back to inline injection when no `AgentRunner` is available.
struct ForkableSkillTool {
    inner: Arc<SkillTool>,
    runner: Option<Arc<dyn AgentRunner>>,
}

#[async_trait]
impl Tool for ForkableSkillTool {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn description(&self) -> &str {
        self.inner.description()
    }
    fn input_schema(&self) -> Value {
        self.inner.schema()
    }
    fn meta(&self, input: &Value) -> ToolMeta {
        self.inner.meta(input)
    }

    async fn call(&self, input: Value, ctx: &ToolCtx) -> ToolOutput {
        let args = input
            .get("arguments")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let prompt = match self.inner.build_content(args).await {
            Ok(c) => c,
            Err(e) => return ToolOutput::error(format!("skill error: {e}")),
        };

        let Some(runner) = &self.runner else {
            // No runner configured — fall back to inline injection.
            tracing::debug!(
                skill = self.inner.skill_name,
                "context:fork skill has no runner, falling back to inline"
            );
            let mut output = ToolOutput::success(format!(
                "Skill '{}' loaded into context.",
                self.inner.skill_name
            ))
            .inject(ContextInjection::new(prompt));
            if let Some(s) = self.inner.structured_manifest() {
                output = output.with_structured(s);
            }
            return output;
        };

        // Run as sub-agent, collect text output.
        let mut stream = runner.run_stream(prompt);
        let mut collected = String::new();
        let mut error: Option<String> = None;

        loop {
            tokio::select! {
                _ = ctx.cancel.cancelled() => {
                    return ToolOutput::error("skill fork cancelled");
                }
                event = stream.next() => {
                    match event {
                        None => break,
                        Some(wui_core::event::AgentEvent::TextDelta(t)) => collected.push_str(&t),
                        Some(wui_core::event::AgentEvent::Error(e)) => {
                            error = Some(e.message.clone());
                            break;
                        }
                        Some(_) => {}
                    }
                }
            }
        }

        if let Some(err) = error {
            return ToolOutput::error(format!("skill fork error: {err}"));
        }

        let mut output = ToolOutput::success(collected);
        if let Some(s) = self.inner.structured_manifest() {
            output = output.with_structured(s);
        }
        output
    }
}

// ── Shell injection ───────────────────────────────────────────────────────────

/// Replace `` !`cmd` `` patterns with the stdout of `cmd`.
///
/// Commands are run with `sh -c` and must complete within 10 seconds.
/// Stderr is discarded; stdout is trimmed and substituted inline.
async fn execute_shell_injections(text: &str, work_dir: Option<&Path>) -> anyhow::Result<String> {
    // Find all !`...` patterns.
    let mut result = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        // Look for the literal sequence  !`
        if i + 1 < bytes.len() && bytes[i] == b'!' && bytes[i + 1] == b'`' {
            // Find the closing backtick.
            if let Some(end) = text[i + 2..].find('`') {
                let cmd = &text[i + 2..i + 2 + end];
                let output = run_shell(cmd, work_dir).await?;
                result.push_str(&output);
                i += 2 + end + 1; // skip !`...`
                continue;
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }

    Ok(result)
}

async fn run_shell(cmd: &str, work_dir: Option<&Path>) -> anyhow::Result<String> {
    let mut builder = tokio::process::Command::new("sh");
    builder.arg("-c").arg(cmd);
    builder.stdout(std::process::Stdio::piped());
    builder.stderr(std::process::Stdio::null());
    if let Some(dir) = work_dir {
        builder.current_dir(dir);
    }

    let child = builder.spawn()?;
    let output = tokio::time::timeout(Duration::from_secs(10), child.wait_with_output())
        .await
        .map_err(|_| anyhow::anyhow!("shell injection timed out: {cmd}"))?
        .map_err(|e| anyhow::anyhow!("shell injection failed: {e}"))?;

    if !output.status.success() {
        anyhow::bail!(
            "shell injection exited with {}: {cmd}",
            output.status.code().unwrap_or(-1)
        );
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

// ── Argument substitution ─────────────────────────────────────────────────────

/// Substitute `$ARGUMENTS`, `$1`/`$2`, and `$name` placeholders.
///
/// - `$ARGUMENTS` → the full arguments string.
/// - `$1`, `$2`, ... → whitespace-delimited positional tokens.
/// - `$name` → the token at the position declared in `arguments:`.
fn substitute_arguments(text: &str, args: &str, declared: &[String]) -> String {
    let mut out = text.replace("$ARGUMENTS", args);

    let tokens: Vec<&str> = args.split_whitespace().collect();

    // Positional: $1, $2, ...
    for (pos, token) in tokens.iter().enumerate() {
        out = out.replace(&format!("${}", pos + 1), token);
    }

    // Named: $name → token at declaration position.
    for (idx, name) in declared.iter().enumerate() {
        let placeholder = format!("${name}");
        let value = tokens.get(idx).copied().unwrap_or("");
        out = out.replace(&placeholder, value);
    }

    out
}

// ── Frontmatter parser ────────────────────────────────────────────────────────

async fn parse_skill_file(path: &Path) -> anyhow::Result<SkillTool> {
    let raw = tokio::fs::read_to_string(path).await?;
    parse_skill_markdown(&raw, None)
}

async fn parse_skill_dir(dir: &Path) -> anyhow::Result<SkillTool> {
    let raw = tokio::fs::read_to_string(dir.join("SKILL.md")).await?;
    parse_skill_markdown(&raw, Some(dir.to_path_buf()))
}

fn parse_skill_markdown(content: &str, skill_dir: Option<PathBuf>) -> anyhow::Result<SkillTool> {
    let rest = content
        .strip_prefix("---\n")
        .ok_or_else(|| anyhow::anyhow!("no frontmatter"))?;

    let end = rest
        .find("\n---\n")
        .ok_or_else(|| anyhow::anyhow!("unclosed frontmatter"))?;

    let frontmatter = &rest[..end];
    let body = rest[end + 5..].trim().to_string();

    let mut name = None;
    let mut desc = None;
    let mut manifest = SkillManifest::default();

    for line in frontmatter.lines() {
        if let Some(v) = line.strip_prefix("name:") {
            name = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("description:") {
            desc = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("when_to_use:") {
            manifest.when_to_use = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("arguments:") {
            manifest.arguments = v
                .split_whitespace()
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect();
        } else if let Some(v) = line.strip_prefix("context:") {
            manifest.context = match v.trim() {
                "fork" => SkillContext::Fork,
                _ => SkillContext::Inline,
            };
        } else if let Some(v) = line.strip_prefix("allowed_tools:") {
            manifest.allowed_tools = v
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        } else if let Some(v) = line.strip_prefix("effort:") {
            manifest.effort = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("model:") {
            manifest.model = Some(v.trim().to_string());
        }
    }

    Ok(SkillTool {
        skill_name: name.ok_or_else(|| anyhow::anyhow!("missing name"))?,
        description: desc.ok_or_else(|| anyhow::anyhow!("missing description"))?,
        content: body,
        manifest,
        skill_dir,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio_util::sync::CancellationToken;

    fn make_ctx() -> ToolCtx {
        ToolCtx {
            cancel: CancellationToken::new(),
            messages: Arc::from(vec![]),
            spawn_depth: 0,
            on_progress: Box::new(|_| {}),
        }
    }

    async fn write_skill(dir: &TempDir, filename: &str, content: &str) {
        let path = dir.path().join(filename);
        std::fs::write(path, content).unwrap();
    }

    fn write_skill_dir(dir: &TempDir, name: &str, skill_md: &str, extras: &[(&str, &str)]) {
        let skill_dir = dir.path().join(name);
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(skill_dir.join("SKILL.md"), skill_md).unwrap();
        for (filename, content) in extras {
            std::fs::write(skill_dir.join(filename), content).unwrap();
        }
    }

    #[tokio::test]
    async fn skills_catalog_loads_valid_skill_file() {
        let dir = TempDir::new().unwrap();

        write_skill(
            &dir,
            "commit.md",
            indoc::indoc! {"
            ---
            name: git-commit
            description: Write conventional commit messages.
            ---
            Always use imperative mood: \"Add feature\", not \"Added feature\".
        "},
        )
        .await;

        let catalog = SkillsCatalog::new(dir.path());
        let hits = catalog.search("commit", 10).await.unwrap();

        assert_eq!(hits.len(), 1, "expected one skill hit, got: {}", hits.len());
        assert_eq!(hits[0].tool.name(), "git-commit");
        assert_eq!(
            hits[0].tool.description(),
            "Write conventional commit messages."
        );
    }

    #[tokio::test]
    async fn skills_catalog_ignores_missing_frontmatter() {
        let dir = TempDir::new().unwrap();

        write_skill(&dir, "bad.md", "No frontmatter here.").await;
        write_skill(
            &dir,
            "good.md",
            indoc::indoc! {"
            ---
            name: valid-skill
            description: A valid skill.
            ---
            Skill body.
        "},
        )
        .await;

        let catalog = SkillsCatalog::new(dir.path());
        let hits = catalog.search("valid", 10).await.unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].tool.name(), "valid-skill");
    }

    #[tokio::test]
    async fn skills_catalog_loads_directory_skill() {
        let dir = TempDir::new().unwrap();

        write_skill_dir(
            &dir,
            "release",
            indoc::indoc! {"
                ---
                name: release
                description: Run the release workflow.
                ---
                Follow the checklist at ${SKILL_DIR}/checklist.md.
            "},
            &[("checklist.md", "- bump version\n- publish\n")],
        );

        let catalog = SkillsCatalog::new(dir.path());
        let hits = catalog.search("release", 10).await.unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].tool.name(), "release");

        let ctx = make_ctx();
        let output = hits[0].tool.call(serde_json::json!({}), &ctx).await;
        assert!(
            !output.injections.is_empty(),
            "should have context injection"
        );
        let injection_text = &output.injections[0].text;
        assert!(
            !injection_text.contains("${SKILL_DIR}"),
            "placeholder should be replaced, got: {injection_text}"
        );
        let expected_path = dir.path().join("release");
        assert!(
            injection_text.contains(&expected_path.display().to_string()),
            "injection should contain skill dir path"
        );
    }

    #[tokio::test]
    async fn skills_catalog_ignores_dir_without_skill_md() {
        let dir = TempDir::new().unwrap();

        let stray = dir.path().join("stray");
        std::fs::create_dir_all(&stray).unwrap();
        std::fs::write(stray.join("notes.md"), "not a skill").unwrap();

        write_skill(
            &dir,
            "good.md",
            indoc::indoc! {"
                ---
                name: valid-skill
                description: A valid skill.
                ---
                Body.
            "},
        )
        .await;

        let catalog = SkillsCatalog::new(dir.path());
        let hits = catalog.search("valid", 10).await.unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].tool.name(), "valid-skill");
    }

    #[tokio::test]
    async fn skills_catalog_returns_empty_for_no_match() {
        let dir = TempDir::new().unwrap();

        write_skill(
            &dir,
            "git.md",
            indoc::indoc! {"
            ---
            name: git-commit
            description: Write commit messages.
            ---
            Body.
        "},
        )
        .await;

        let catalog = SkillsCatalog::new(dir.path());
        let hits = catalog.search("totally-unrelated-xyz", 10).await.unwrap();
        assert!(hits.is_empty());
    }

    // ── when_to_use ───────────────────────────────────────────────────────────

    #[tokio::test]
    async fn when_to_use_improves_search_recall() {
        let dir = TempDir::new().unwrap();

        write_skill(
            &dir,
            "deploy.md",
            indoc::indoc! {"
                ---
                name: deploy
                description: Push code to production.
                when_to_use: ship release publish prod production
                ---
                Run `cargo release` then push.
            "},
        )
        .await;

        let catalog = SkillsCatalog::new(dir.path());

        // "ship" only appears in when_to_use, not name/description.
        let hits = catalog.search("ship", 10).await.unwrap();
        assert_eq!(hits.len(), 1, "when_to_use tokens should be searchable");
        assert_eq!(hits[0].tool.name(), "deploy");
    }

    // ── $ARGUMENTS substitution ───────────────────────────────────────────────

    #[tokio::test]
    async fn arguments_substitution_positional() {
        let dir = TempDir::new().unwrap();

        write_skill(
            &dir,
            "greet.md",
            indoc::indoc! {"
                ---
                name: greet
                description: Greet a user.
                arguments: name
                ---
                Hello, $name! (arg1=$1, all=$ARGUMENTS)
            "},
        )
        .await;

        let catalog = SkillsCatalog::new(dir.path());
        let hits = catalog.search("greet", 10).await.unwrap();
        assert_eq!(hits.len(), 1);

        // The schema should now expose an `arguments` field.
        let schema = hits[0].tool.input_schema();
        assert!(
            schema["properties"]["arguments"].is_object(),
            "schema should have arguments property"
        );

        let ctx = make_ctx();
        let output = hits[0]
            .tool
            .call(serde_json::json!({ "arguments": "Alice" }), &ctx)
            .await;

        let text = &output.injections[0].text;
        assert!(text.contains("Hello, Alice!"), "named sub: {text}");
        assert!(text.contains("arg1=Alice"), "positional sub: {text}");
        assert!(text.contains("all=Alice"), "$ARGUMENTS sub: {text}");
    }

    #[tokio::test]
    async fn arguments_substitution_multiple() {
        let dir = TempDir::new().unwrap();

        write_skill(
            &dir,
            "echo.md",
            indoc::indoc! {"
                ---
                name: echo
                description: Echo arguments.
                arguments: first second
                ---
                $first then $second, all: $ARGUMENTS
            "},
        )
        .await;

        let catalog = SkillsCatalog::new(dir.path());
        let hits = catalog.search("echo", 10).await.unwrap();

        let ctx = make_ctx();
        let output = hits[0]
            .tool
            .call(serde_json::json!({ "arguments": "foo bar" }), &ctx)
            .await;

        let text = &output.injections[0].text;
        assert!(text.contains("foo then bar"), "named sub: {text}");
        assert!(text.contains("all: foo bar"), "$ARGUMENTS sub: {text}");
    }

    // ── shell injection ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn shell_injection_replaced() {
        let dir = TempDir::new().unwrap();

        write_skill(
            &dir,
            "shell.md",
            indoc::indoc! {r#"
                ---
                name: shell-skill
                description: Test shell injection.
                ---
                Version: !`echo "v1.0"`
            "#},
        )
        .await;

        let catalog = SkillsCatalog::new(dir.path());
        let hits = catalog.search("shell", 10).await.unwrap();

        let ctx = make_ctx();
        let output = hits[0].tool.call(serde_json::json!({}), &ctx).await;
        let text = &output.injections[0].text;
        assert!(
            text.contains("Version: v1.0"),
            "shell injection should be replaced: {text}"
        );
        assert!(
            !text.contains("!`"),
            "raw !` pattern should be gone: {text}"
        );
    }

    #[tokio::test]
    async fn shell_injection_multiple() {
        let dir = TempDir::new().unwrap();

        write_skill(
            &dir,
            "multi.md",
            indoc::indoc! {r#"
                ---
                name: multi-shell
                description: Multiple shell injections.
                ---
                A: !`echo "alpha"` B: !`echo "beta"`
            "#},
        )
        .await;

        let catalog = SkillsCatalog::new(dir.path());
        let hits = catalog
            .search("multiple shell injections", 10)
            .await
            .unwrap();

        let ctx = make_ctx();
        let output = hits[0].tool.call(serde_json::json!({}), &ctx).await;
        let text = &output.injections[0].text;
        assert!(text.contains("A: alpha"), "{text}");
        assert!(text.contains("B: beta"), "{text}");
    }

    // ── substitute_arguments unit tests ──────────────────────────────────────

    #[test]
    fn substitute_arguments_replaces_all_forms() {
        let declared = vec!["name".to_string(), "age".to_string()];
        let result = substitute_arguments(
            "Hello $name, you are $age ($1, $2). Full: $ARGUMENTS",
            "Alice 30",
            &declared,
        );
        assert_eq!(
            result,
            "Hello Alice, you are 30 (Alice, 30). Full: Alice 30"
        );
    }

    #[test]
    fn substitute_arguments_missing_positional_becomes_empty() {
        let declared = vec!["a".to_string(), "b".to_string()];
        let result = substitute_arguments("$a-$b", "only_one", &declared);
        assert_eq!(result, "only_one-");
    }

    #[test]
    fn substitute_arguments_no_args_leaves_placeholders_for_declared() {
        // When called with an empty string, $name → "" and $ARGUMENTS → "".
        let declared = vec!["name".to_string()];
        let result = substitute_arguments("Hello $name! Full: [$ARGUMENTS]", "", &declared);
        assert_eq!(result, "Hello ! Full: []");
    }

    #[test]
    fn substitute_arguments_unknown_dollar_left_intact() {
        // $unknown_var is not in declared and has no positional match; untouched.
        let declared = vec!["known".to_string()];
        let result = substitute_arguments("$known $unknown_var", "alpha", &declared);
        assert_eq!(result, "alpha $unknown_var");
    }

    // ── Frontmatter parsing ───────────────────────────────────────────────────

    #[test]
    fn parse_all_frontmatter_fields() {
        let md = indoc::indoc! {"
            ---
            name: full-skill
            description: A fully configured skill.
            when_to_use: synonym trigger alias
            arguments: target env
            context: fork
            allowed_tools: Bash, Read
            effort: high
            model: claude-opus-4-6
            ---
            Body text.
        "};
        let skill = parse_skill_markdown(md, None).unwrap();
        assert_eq!(skill.skill_name, "full-skill");
        assert_eq!(skill.description, "A fully configured skill.");
        assert_eq!(
            skill.manifest.when_to_use.as_deref(),
            Some("synonym trigger alias")
        );
        assert_eq!(
            skill.manifest.arguments,
            vec!["target".to_string(), "env".to_string()]
        );
        assert_eq!(skill.manifest.context, SkillContext::Fork);
        assert_eq!(
            skill.manifest.allowed_tools,
            vec!["Bash".to_string(), "Read".to_string()]
        );
        assert_eq!(skill.manifest.effort.as_deref(), Some("high"));
        assert_eq!(skill.manifest.model.as_deref(), Some("claude-opus-4-6"));
        assert_eq!(skill.content, "Body text.");
    }

    #[test]
    fn parse_context_inline_explicit() {
        let md = indoc::indoc! {"
            ---
            name: s
            description: d
            context: inline
            ---
            Body.
        "};
        let skill = parse_skill_markdown(md, None).unwrap();
        assert_eq!(skill.manifest.context, SkillContext::Inline);
    }

    #[test]
    fn parse_context_unknown_defaults_to_inline() {
        let md = indoc::indoc! {"
            ---
            name: s
            description: d
            context: something-else
            ---
            Body.
        "};
        let skill = parse_skill_markdown(md, None).unwrap();
        assert_eq!(skill.manifest.context, SkillContext::Inline);
    }

    #[test]
    fn parse_missing_name_errors() {
        let md = indoc::indoc! {"
            ---
            description: No name here.
            ---
            Body.
        "};
        assert!(parse_skill_markdown(md, None).is_err());
    }

    #[test]
    fn parse_missing_description_errors() {
        let md = indoc::indoc! {"
            ---
            name: no-desc
            ---
            Body.
        "};
        assert!(parse_skill_markdown(md, None).is_err());
    }

    #[test]
    fn parse_unknown_frontmatter_fields_ignored() {
        let md = indoc::indoc! {"
            ---
            name: s
            description: d
            future_field: some value
            another: 42
            ---
            Body.
        "};
        let skill = parse_skill_markdown(md, None).unwrap();
        assert_eq!(skill.skill_name, "s");
        assert_eq!(skill.content, "Body.");
    }

    // ── Input schema ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn schema_without_arguments_is_empty_object() {
        let dir = TempDir::new().unwrap();
        write_skill(
            &dir,
            "simple.md",
            indoc::indoc! {"
                ---
                name: simple
                description: No arguments.
                ---
                Body.
            "},
        )
        .await;

        let catalog = SkillsCatalog::new(dir.path());
        let hits = catalog.search("simple", 10).await.unwrap();
        assert_eq!(hits.len(), 1);

        let schema = hits[0].tool.input_schema();
        assert_eq!(schema["type"], "object");
        assert!(
            schema["properties"].as_object().unwrap().is_empty(),
            "no-argument skill should have empty properties"
        );
    }

    #[tokio::test]
    async fn schema_with_arguments_exposes_field() {
        let dir = TempDir::new().unwrap();
        write_skill(
            &dir,
            "arg.md",
            indoc::indoc! {"
                ---
                name: arg-skill
                description: Parametrised skill with dynamic inputs.
                arguments: target
                ---
                Deploy $target.
            "},
        )
        .await;

        let catalog = SkillsCatalog::new(dir.path());
        let hits = catalog
            .search("parametrised skill dynamic", 10)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);

        let schema = hits[0].tool.input_schema();
        assert!(
            schema["properties"]["arguments"].is_object(),
            "schema must have 'arguments' property: {schema}"
        );
    }

    // ── Structured output & manifest hints ────────────────────────────────────

    #[tokio::test]
    async fn allowed_tools_appended_to_injection_text() {
        let dir = TempDir::new().unwrap();
        write_skill(
            &dir,
            "deploy.md",
            indoc::indoc! {"
                ---
                name: deploy
                description: Deploy the app.
                allowed_tools: Bash, Read
                ---
                Run the deploy script.
            "},
        )
        .await;

        let catalog = SkillsCatalog::new(dir.path());
        let hits = catalog.search("deploy", 10).await.unwrap();
        let ctx = make_ctx();
        let output = hits[0].tool.call(serde_json::json!({}), &ctx).await;

        let text = &output.injections[0].text;
        assert!(
            text.contains("Bash") && text.contains("Read"),
            "allowed_tools hint missing from injection: {text}"
        );
    }

    #[tokio::test]
    async fn manifest_fields_appear_in_structured_output() {
        let dir = TempDir::new().unwrap();
        write_skill(
            &dir,
            "heavy.md",
            indoc::indoc! {"
                ---
                name: heavy
                description: Heavy skill.
                allowed_tools: Bash
                effort: high
                model: claude-opus-4-6
                ---
                Body.
            "},
        )
        .await;

        let catalog = SkillsCatalog::new(dir.path());
        let hits = catalog.search("heavy", 10).await.unwrap();
        let ctx = make_ctx();
        let output = hits[0].tool.call(serde_json::json!({}), &ctx).await;

        let s = output.structured.expect("should have structured output");
        let m = &s["skill_manifest"];
        assert_eq!(m["effort"], "high");
        assert_eq!(m["model"], "claude-opus-4-6");
        assert!(m["allowed_tools"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("Bash")));
    }

    #[tokio::test]
    async fn no_manifest_extras_means_no_structured_output() {
        let dir = TempDir::new().unwrap();
        write_skill(
            &dir,
            "bare.md",
            indoc::indoc! {"
                ---
                name: bare
                description: Minimal skill.
                ---
                Body.
            "},
        )
        .await;

        let catalog = SkillsCatalog::new(dir.path());
        let hits = catalog.search("bare", 10).await.unwrap();
        let ctx = make_ctx();
        let output = hits[0].tool.call(serde_json::json!({}), &ctx).await;

        assert!(
            output.structured.is_none(),
            "bare skill should produce no structured output"
        );
    }

    // ── context:fork ─────────────────────────────────────────────────────────

    /// A minimal AgentRunner that always emits a fixed text response.
    struct FixedRunner {
        text: String,
    }

    impl AgentRunner for FixedRunner {
        fn run_stream(
            &self,
            _prompt: String,
        ) -> futures::stream::BoxStream<'static, wui_core::event::AgentEvent> {
            use wui_core::event::{AgentEvent, RunStopReason, RunSummary, TokenUsage};
            let text = self.text.clone();
            Box::pin(futures::stream::iter(vec![
                AgentEvent::TextDelta(text),
                AgentEvent::Done(RunSummary {
                    stop_reason: RunStopReason::Completed,
                    iterations: 1,
                    usage: TokenUsage::default(),
                    messages: vec![],
                    last_transition: None,
                }),
            ]))
        }
    }

    /// A runner that captures the prompt it received and echoes it back.
    struct CaptureRunner {
        captured: Arc<std::sync::Mutex<String>>,
    }

    impl AgentRunner for CaptureRunner {
        fn run_stream(
            &self,
            prompt: String,
        ) -> futures::stream::BoxStream<'static, wui_core::event::AgentEvent> {
            use wui_core::event::{AgentEvent, RunStopReason, RunSummary, TokenUsage};
            *self.captured.lock().unwrap() = prompt.clone();
            Box::pin(futures::stream::iter(vec![
                AgentEvent::TextDelta(format!("got:{prompt}")),
                AgentEvent::Done(RunSummary {
                    stop_reason: RunStopReason::Completed,
                    iterations: 1,
                    usage: TokenUsage::default(),
                    messages: vec![],
                    last_transition: None,
                }),
            ]))
        }
    }

    #[tokio::test]
    async fn fork_skill_without_runner_falls_back_to_inline() {
        let dir = TempDir::new().unwrap();
        write_skill(
            &dir,
            "fork.md",
            indoc::indoc! {"
                ---
                name: fork-skill
                description: Dispatch as independent sub-agent context.
                context: fork
                ---
                Do the thing.
            "},
        )
        .await;

        // No runner configured — expect inline injection fallback.
        let catalog = SkillsCatalog::new(dir.path());
        let hits = catalog
            .search("dispatch independent sub-agent", 10)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);

        let ctx = make_ctx();
        let output = hits[0].tool.call(serde_json::json!({}), &ctx).await;

        // Inline fallback produces a context injection, not an agent run.
        assert!(
            !output.injections.is_empty(),
            "fallback should inject content"
        );
        assert!(
            output.injections[0].text.contains("Do the thing"),
            "injection should contain skill body"
        );
    }

    #[tokio::test]
    async fn fork_skill_with_runner_returns_agent_text() {
        let dir = TempDir::new().unwrap();
        write_skill(
            &dir,
            "fork2.md",
            indoc::indoc! {"
                ---
                name: fork-skill2
                description: Fork skill with runner.
                context: fork
                ---
                Prompt for agent.
            "},
        )
        .await;

        let runner = FixedRunner {
            text: "agent completed".to_string(),
        };
        let catalog = SkillsCatalog::new(dir.path()).with_fork_runner(runner);
        let hits = catalog.search("fork runner", 10).await.unwrap();
        assert_eq!(hits.len(), 1);

        let ctx = make_ctx();
        let output = hits[0].tool.call(serde_json::json!({}), &ctx).await;

        // Fork dispatch: no injections, result is the agent's text output.
        assert!(
            output.injections.is_empty(),
            "fork should not produce context injections"
        );
        assert_eq!(output.content, "agent completed");
    }

    #[tokio::test]
    async fn fork_skill_receives_fully_substituted_prompt() {
        let dir = TempDir::new().unwrap();
        write_skill(
            &dir,
            "fork3.md",
            indoc::indoc! {"
                ---
                name: fork-skill3
                description: Substitution in fork.
                context: fork
                arguments: target
                ---
                Deploy $target to production.
            "},
        )
        .await;

        let captured = Arc::new(std::sync::Mutex::new(String::new()));
        let runner = CaptureRunner {
            captured: Arc::clone(&captured),
        };
        let catalog = SkillsCatalog::new(dir.path()).with_fork_runner(runner);
        let hits = catalog.search("substitution fork", 10).await.unwrap();
        assert_eq!(hits.len(), 1);

        let ctx = make_ctx();
        hits[0]
            .tool
            .call(serde_json::json!({ "arguments": "staging" }), &ctx)
            .await;

        let prompt = captured.lock().unwrap().clone();
        assert!(
            prompt.contains("Deploy staging to production"),
            "runner should receive substituted prompt, got: {prompt}"
        );
    }

    // ── Shell injection edge cases ────────────────────────────────────────────

    #[tokio::test]
    async fn shell_injection_absent_leaves_content_unchanged() {
        let dir = TempDir::new().unwrap();
        write_skill(
            &dir,
            "plain.md",
            indoc::indoc! {"
                ---
                name: plain
                description: No shell injection.
                ---
                Just static content.
            "},
        )
        .await;

        let catalog = SkillsCatalog::new(dir.path());
        let hits = catalog.search("plain static", 10).await.unwrap();
        let ctx = make_ctx();
        let output = hits[0].tool.call(serde_json::json!({}), &ctx).await;
        assert_eq!(output.injections[0].text, "Just static content.");
    }

    #[tokio::test]
    async fn shell_injection_failure_returns_error_output() {
        let dir = TempDir::new().unwrap();
        write_skill(
            &dir,
            "bad-shell.md",
            // exit 1 makes the shell command fail.
            indoc::indoc! {r#"
                ---
                name: bad-shell
                description: Shell command that fails.
                ---
                Result: !`exit 1`
            "#},
        )
        .await;

        let catalog = SkillsCatalog::new(dir.path());
        let hits = catalog.search("shell fails", 10).await.unwrap();
        let ctx = make_ctx();
        let output = hits[0].tool.call(serde_json::json!({}), &ctx).await;

        assert!(
            output.failure.is_some(),
            "failed shell injection should produce a failure output"
        );
    }

    #[tokio::test]
    async fn shell_injection_in_dir_skill_uses_skill_dir_as_cwd() {
        let dir = TempDir::new().unwrap();
        // pwd prints the current working directory.
        write_skill_dir(
            &dir,
            "pwdskill",
            indoc::indoc! {r#"
                ---
                name: pwd-skill
                description: Prints working directory.
                ---
                CWD: !`pwd`
            "#},
            &[],
        );

        let catalog = SkillsCatalog::new(dir.path());
        let hits = catalog.search("working directory", 10).await.unwrap();
        assert_eq!(hits.len(), 1);

        let ctx = make_ctx();
        let output = hits[0].tool.call(serde_json::json!({}), &ctx).await;
        let text = &output.injections[0].text;

        let expected_dir = dir.path().join("pwdskill");
        assert!(
            text.contains(&expected_dir.display().to_string()),
            "shell injection should run inside skill dir, got: {text}"
        );
    }

    // ── Search ranking ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn search_limit_is_respected() {
        let dir = TempDir::new().unwrap();
        for i in 0..5 {
            write_skill(
                &dir,
                &format!("skill{i}.md"),
                &format!(
                    "---\nname: skill-{i}\ndescription: A test skill numbered {i}.\n---\nBody.\n"
                ),
            )
            .await;
        }

        let catalog = SkillsCatalog::new(dir.path());
        let hits = catalog.search("test skill", 3).await.unwrap();
        assert!(hits.len() <= 3, "limit not respected: {} hits", hits.len());
    }

    #[tokio::test]
    async fn search_returns_hits_sorted_by_relevance() {
        let dir = TempDir::new().unwrap();

        // Skill A: name contains "deploy", description contains "release" → matches both → score 2/2 = 1.0
        write_skill(
            &dir,
            "a.md",
            indoc::indoc! {"
                ---
                name: deploy
                description: deploy and release services
                ---
                Body.
            "},
        )
        .await;

        // Skill B: only description contains "release" → score 1/2 = 0.5
        write_skill(
            &dir,
            "b.md",
            indoc::indoc! {"
                ---
                name: build
                description: build and release binaries
                ---
                Body.
            "},
        )
        .await;

        let catalog = SkillsCatalog::new(dir.path());
        // "deploy release": skill A matches both tokens (score 1.0), skill B matches "release" only (score 0.5).
        let hits = catalog.search("deploy release", 10).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(
            hits[0].tool.name(),
            "deploy",
            "higher-scoring skill should be first"
        );
        assert!(
            hits[0].score >= hits[1].score,
            "hits must be sorted descending by score"
        );
    }

    #[tokio::test]
    async fn when_to_use_tokens_increase_score() {
        let dir = TempDir::new().unwrap();

        // Skill A: "ship" only in when_to_use.
        write_skill(
            &dir,
            "a.md",
            indoc::indoc! {"
                ---
                name: deploy-a
                description: Push the code out.
                when_to_use: ship release publish
                ---
                Body.
            "},
        )
        .await;

        // Skill B: "ship" nowhere.
        write_skill(
            &dir,
            "b.md",
            indoc::indoc! {"
                ---
                name: build
                description: Compile and test.
                ---
                Body.
            "},
        )
        .await;

        let catalog = SkillsCatalog::new(dir.path());

        // "ship" should only match skill A via when_to_use.
        let hits = catalog.search("ship", 10).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].tool.name(), "deploy-a");

        // "build" matches only skill B.
        let hits2 = catalog.search("build", 10).await.unwrap();
        assert_eq!(hits2.len(), 1);
        assert_eq!(hits2[0].tool.name(), "build");
    }

    // ── Combined substitutions ────────────────────────────────────────────────

    #[tokio::test]
    async fn skill_dir_and_arguments_substituted_together() {
        let dir = TempDir::new().unwrap();
        write_skill_dir(
            &dir,
            "combo",
            indoc::indoc! {"
                ---
                name: combo
                description: Combined substitutions.
                arguments: env
                ---
                Dir=${SKILL_DIR} Env=$env Full=$ARGUMENTS
            "},
            &[],
        );

        let catalog = SkillsCatalog::new(dir.path());
        let hits = catalog.search("combined substitutions", 10).await.unwrap();
        assert_eq!(hits.len(), 1);

        let ctx = make_ctx();
        let output = hits[0]
            .tool
            .call(serde_json::json!({ "arguments": "staging" }), &ctx)
            .await;

        let text = &output.injections[0].text;
        let skill_dir = dir.path().join("combo");
        assert!(
            text.contains(&skill_dir.display().to_string()),
            "${{SKILL_DIR}} not substituted: {text}"
        );
        assert!(text.contains("Env=staging"), "$env not substituted: {text}");
        assert!(
            text.contains("Full=staging"),
            "$ARGUMENTS not substituted: {text}"
        );
    }
}
