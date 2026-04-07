// ============================================================================
// wui-skills — file-based skill discovery.
//
// A SkillsCatalog scans a directory for Markdown files with YAML-like
// frontmatter. Each file becomes a Tool. Invoking the tool injects the
// skill's content into the agent's context as a <system-reminder>.
//
// Frontmatter format:
//   ---
//   name: git-commit
//   description: Write conventional commit messages following project style.
//   ---
//   ... skill content ...
//
// The directory is scanned lazily on first search().
// ============================================================================

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::OnceCell;

use wui::catalog::{CatalogHit, ToolCatalog};
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
                if path.extension().is_some_and(|e| e == "md") {
                    match parse_skill_file(&path).await {
                        Ok(skill) => skills.push(Arc::new(skill)),
                        Err(e)    => tracing::warn!(path = %path.display(), error = %e, "skipping skill file with invalid frontmatter"),
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
        ToolOutput::success(format!("Skill '{}' loaded into context.", self.skill_name))
            .inject(ContextInjection::system(self.content.clone()))
    }
}

// ── Frontmatter parser ────────────────────────────────────────────────────────

async fn parse_skill_file(path: &PathBuf) -> anyhow::Result<SkillTool> {
    let content = tokio::fs::read_to_string(path).await?;

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
    for line in frontmatter.lines() {
        if let Some(v) = line.strip_prefix("name: ") {
            name = Some(v.trim().to_string());
        }
        if let Some(v) = line.strip_prefix("description: ") {
            desc = Some(v.trim().to_string());
        }
    }

    Ok(SkillTool {
        skill_name: name.ok_or_else(|| anyhow::anyhow!("missing name"))?,
        description: desc.ok_or_else(|| anyhow::anyhow!("missing description"))?,
        content: body,
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
