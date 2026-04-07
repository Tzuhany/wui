// ============================================================================
// Message Formatting — structured context boundaries.
//
// LLMs are trained on vast amounts of XML/HTML and parse structure reliably.
// XML tags create unambiguous boundaries between content sources:
//
//   User content:      arrives as a user message — no wrapping
//   Framework content: wrapped in <system-reminder> — clearly "the framework said this"
//   Tool progress:     wrapped in <progress> — clearly mid-execution status
//
// This separation matters. Without it, a long conversation blurs the line
// between what the user requested and what the framework injected.
// The LLM calibrates its behaviour based on source — "the user told me to
// do X" warrants different weight than "the framework noted that Y".
//
// Convention:
//   - Tags are kebab-case: <system-reminder>, <progress>
//   - Framework-injected content always uses <system-reminder>
//   - User content is NEVER wrapped — the absence of tags means "user said this"
//   - Nesting is allowed for structured payloads
// ============================================================================

/// Wrap content in a `<system-reminder>` tag.
///
/// Use this for all content injected by the framework or by tools on behalf
/// of the framework — permission decisions, memory recalls, date/environment
/// context, deferred tool listings, etc.
///
/// The LLM treats content inside `<system-reminder>` as authoritative context
/// from the runtime, distinct from user instructions.
///
/// ```rust,ignore
/// let msg = system_reminder("Today is 2026-04-06. The user's timezone is UTC+8.");
/// // → "<system-reminder>\nToday is 2026-04-06...\n</system-reminder>"
/// ```
pub fn system_reminder(content: &str) -> String {
    format!("<system-reminder>\n{content}\n</system-reminder>")
}

/// Wrap tool progress text in a `<progress>` tag.
///
/// Use inside `ToolCtx::report()` when a tool emits incremental status.
/// The LLM understands these as mid-execution updates, not final results.
pub fn progress(content: &str) -> String {
    format!("<progress>{content}</progress>")
}

/// Format a key-value pair as a self-closing XML attribute-style tag.
///
/// Useful when injecting structured metadata into a system reminder:
/// ```rust,ignore
/// let body = [
///     kv("depth", "2"),
///     kv("chain", "abc-123"),
/// ].join("\n");
/// system_reminder(&body)
/// ```
pub fn kv(key: &str, value: &str) -> String {
    format!("<{key}>{value}</{key}>")
}

/// Extract content between the first matching XML tag pair.
///
/// Returns the trimmed inner content, or `None` if the opening or closing tag
/// is not found in `content`.
///
/// ```rust
/// # use wui_core::fmt::extract_tag;
/// let text = "Here is my answer: <answer>42</answer> done.";
/// assert_eq!(extract_tag(text, "answer"), Some("42"));
/// ```
pub fn extract_tag<'a>(content: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = content.find(&open)?;
    let inner_start = start + open.len();
    let end = content[inner_start..].find(&close)?;
    Some(content[inner_start..inner_start + end].trim())
}

/// Extract all top-level XML tag contents from a string.
///
/// Scans `content` for `<tag>...</tag>` patterns and returns a map from tag
/// name to trimmed inner content. Duplicate tags return the last occurrence.
///
/// ```rust
/// # use wui_core::fmt::extract_tags;
/// let text = "<name>Alice</name><score>99</score>";
/// let tags = extract_tags(text);
/// assert_eq!(tags.get("name").map(String::as_str), Some("Alice"));
/// assert_eq!(tags.get("score").map(String::as_str), Some("99"));
/// ```
pub fn extract_tags(content: &str) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    let mut rest = content;
    while let Some(open_start) = rest.find('<') {
        let after_open = &rest[open_start + 1..];
        // Must be an opening tag (not a closing tag).
        if after_open.starts_with('/') {
            // Skip past this '<' and continue.
            rest = &rest[open_start + 1..];
            continue;
        }
        // Find the end of the tag name (space, '>', or '/' for self-closing).
        let tag_end = after_open.find(['>', '/', ' ']).unwrap_or(after_open.len());
        let tag_name = &after_open[..tag_end];
        if tag_name.is_empty() {
            rest = &rest[open_start + 1..];
            continue;
        }
        let open_tag = format!("<{tag_name}>");
        let close_tag = format!("</{tag_name}>");
        if let Some(inner_start) = rest.find(&open_tag) {
            let content_start = inner_start + open_tag.len();
            if let Some(close_pos) = rest[content_start..].find(&close_tag) {
                let inner = rest[content_start..content_start + close_pos].trim();
                map.insert(tag_name.to_string(), inner.to_string());
                // Advance past the closing tag.
                rest = &rest[content_start + close_pos + close_tag.len()..];
                continue;
            }
        }
        rest = &rest[open_start + 1..];
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_reminder_wraps_correctly() {
        let out = system_reminder("hello");
        assert_eq!(out, "<system-reminder>\nhello\n</system-reminder>");
    }

    #[test]
    fn progress_wraps_correctly() {
        let out = progress("downloading...");
        assert_eq!(out, "<progress>downloading...</progress>");
    }

    #[test]
    fn kv_formats_correctly() {
        assert_eq!(kv("depth", "2"), "<depth>2</depth>");
    }
}
