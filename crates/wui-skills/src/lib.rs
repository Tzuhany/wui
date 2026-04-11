// ============================================================================
// wui-skills — file-based skill discovery.
//
// A SkillsCatalog scans a directory for skills. Each skill becomes a Tool.
// Invoking the tool injects the skill's content into the agent's context
// as a <system-reminder>.
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
//   ---
//   ... skill content ...
//
// The directory is scanned lazily on first search().
// ============================================================================

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::OnceCell;

use wui_core::catalog::{CatalogHit, ToolCatalog};
use wui_core::tool::{ContextInjection, Tool, ToolCtx, ToolOutput};

// ── SkillsCatalog ─────────────────────────────────────────────────────────────

/// A [`ToolCatalog`] that discovers skills from Markdown files in a directory.
///
/// Each `.md` file with valid frontmatter (`name`, `description`) becomes a
/// callable tool. Invoking the tool injects the skill's Markdown content as a
/// `<system-reminder>` into the agent's context.
///
/// The directory is scanned lazily on first `search()` call.
///
/// # Example
///
/// ```rust,ignore
/// Agent::builder(provider)
///     .catalog(SkillsCatalog::new("./skills"))
///     .build()
/// ```
pub struct SkillsCatalog {
    dir: PathBuf,
    skills: OnceCell<Vec<Arc<SkillTool>>>,
}

impl SkillsCatalog {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            skills: OnceCell::new(),
        }
    }

    async fn load(&self) -> anyhow::Result<&Vec<Arc<SkillTool>>> {
        self.skills.get_or_try_init(|| async {
            let mut skills = Vec::new();
            let mut entries = tokio::fs::read_dir(&self.dir).await?;

            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                if path.is_file() && path.extension().is_some_and(|e| e == "md") {
                    match parse_skill_file(&path).await {
                        Ok(skill) => skills.push(Arc::new(skill)),
                        Err(e) => tracing::warn!(path = %path.display(), error = %e, "skipping skill file with invalid frontmatter"),
                    }
                } else if path.is_dir() {
                    let skill_md = path.join("SKILL.md");
                    if skill_md.exists() {
                        match parse_skill_dir(&path).await {
                            Ok(skill) => skills.push(Arc::new(skill)),
                            Err(e) => tracing::warn!(path = %path.display(), error = %e, "skipping skill directory with invalid SKILL.md"),
                        }
                    }
                }
            }

            Ok(skills)
        }).await
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
                let text = format!("{} {}", s.skill_name, s.description).to_lowercase();
                let score = text_overlap(&query_lower, &text);
                if score > 0.0 {
                    Some((score, Arc::clone(s)))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored
            .into_iter()
            .take(limit)
            .map(|(score, skill)| CatalogHit {
                tool: skill as Arc<dyn Tool>,
                score,
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
    /// Structured configuration parsed from frontmatter.
    manifest: SkillManifest,
    /// For directory-based skills: the skill's own directory.
    /// Substituted for `${SKILL_DIR}` in content at invocation time.
    skill_dir: Option<PathBuf>,
}

/// Structured skill configuration parsed from YAML-like frontmatter.
///
/// These fields let a skill declare its operational requirements —
/// turning a skill from a passive text blob into an active capability unit.
///
/// # Frontmatter format
///
/// ```yaml
/// ---
/// name: git-commit
/// description: Write conventional commit messages.
/// allowed_tools: Bash, Read, Glob
/// effort: high
/// model: claude-sonnet-4-5-20250514
/// ---
/// ```
#[derive(Debug, Clone, Default)]
pub struct SkillManifest {
    /// Tools that should be additionally allowed when this skill is active.
    /// Useful for skills that need specific tools (e.g., a deploy skill that
    /// needs Bash). Empty means no additional tool requirements.
    pub allowed_tools: Vec<String>,
    /// Override the agent's effort level when this skill is active.
    pub effort: Option<String>,
    /// Override the agent's model when this skill is active.
    pub model: Option<String>,
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
        serde_json::json!({ "type": "object", "properties": {}, "required": [] })
    }

    async fn call(&self, _input: Value, _ctx: &ToolCtx) -> ToolOutput {
        // Build the injected context: skill content + manifest hints.
        // Substitute ${SKILL_DIR} so directory skills can reference assets.
        let mut injection = match &self.skill_dir {
            Some(dir) => self
                .content
                .replace("${SKILL_DIR}", &dir.display().to_string()),
            None => self.content.clone(),
        };
        if !self.manifest.allowed_tools.is_empty() {
            injection.push_str(&format!(
                "\n\nThis skill works best with these tools: {}",
                self.manifest.allowed_tools.join(", "),
            ));
        }

        let mut output =
            ToolOutput::success(format!("Skill '{}' loaded into context.", self.skill_name))
                .inject(ContextInjection::new(injection));

        // Expose manifest as structured data for hooks, callers, or runtime
        // integrations that want to programmatically react to skill activation
        // (e.g., a hook that adjusts effort or switches model).
        if !self.manifest.allowed_tools.is_empty()
            || self.manifest.effort.is_some()
            || self.manifest.model.is_some()
        {
            output = output.with_structured(serde_json::json!({
                "skill_manifest": {
                    "allowed_tools": self.manifest.allowed_tools,
                    "effort": self.manifest.effort,
                    "model": self.manifest.model,
                }
            }));
        }

        output
    }
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

    async fn write_skill(dir: &TempDir, filename: &str, content: &str) {
        let path = dir.path().join(filename);
        std::fs::write(path, content).unwrap();
    }

    /// Create a directory skill: `skills/<name>/SKILL.md` plus optional extra files.
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

        assert_eq!(
            hits.len(),
            1,
            "expected one skill hit, got: {:?}",
            hits.len()
        );
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

        // Invoke and check that ${SKILL_DIR} was substituted.
        let ctx = ToolCtx {
            cancel: tokio_util::sync::CancellationToken::new(),
            messages: std::sync::Arc::from(vec![]),
            spawn_depth: 0,
            on_progress: Box::new(|_| {}),
        };
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

        // Directory with no SKILL.md — should be silently ignored.
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
}
