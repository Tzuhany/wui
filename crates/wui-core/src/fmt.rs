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
/// Uses a single linear pass over the string.
///
/// ```rust
/// # use wui_core::fmt::extract_tags;
/// let text = "<name>Alice</name><score>99</score>";
/// let tags = extract_tags(text);
/// assert_eq!(tags.get("name").map(String::as_str), Some("Alice"));
/// assert_eq!(tags.get("score").map(String::as_str), Some("99"));
/// ```
pub fn extract_tags(text: &str) -> std::collections::HashMap<String, String> {
    let mut result = std::collections::HashMap::new();
    let mut rest = text;
    while let Some(open_start) = rest.find('<') {
        let after_open = &rest[open_start + 1..];
        // Find tag name end (space or >)
        let tag_end = after_open.find(['>', ' ', '/']);
        let Some(tag_end) = tag_end else { break };
        let tag_name = &after_open[..tag_end];
        if tag_name.is_empty() || tag_name.starts_with('/') {
            rest = &rest[open_start + 1..];
            continue;
        }
        let close_tag = format!("</{tag_name}>");
        let content_start = match after_open[tag_end..].find('>') {
            Some(i) => open_start + 1 + tag_end + i + 1,
            None => break,
        };
        if let Some(close_pos) = rest[content_start..].find(&close_tag) {
            let content = rest[content_start..content_start + close_pos].trim();
            result.insert(tag_name.to_string(), content.to_string());
            rest = &rest[content_start + close_pos + close_tag.len()..];
        } else {
            rest = &rest[open_start + 1..];
        }
    }
    result
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

    #[test]
    fn extract_tags_multiple() {
        let text = "<name>Alice</name><score>99</score><city>Paris</city>";
        let tags = extract_tags(text);
        assert_eq!(tags.get("name").map(String::as_str), Some("Alice"));
        assert_eq!(tags.get("score").map(String::as_str), Some("99"));
        assert_eq!(tags.get("city").map(String::as_str), Some("Paris"));
    }

    #[test]
    fn extract_tags_with_surrounding_text() {
        let text = "some text <answer>42</answer> more text <reason>because</reason> end";
        let tags = extract_tags(text);
        assert_eq!(tags.get("answer").map(String::as_str), Some("42"));
        assert_eq!(tags.get("reason").map(String::as_str), Some("because"));
    }

    #[test]
    fn extract_tags_skips_closing_tags() {
        let text = "</broken><valid>yes</valid>";
        let tags = extract_tags(text);
        assert_eq!(tags.get("valid").map(String::as_str), Some("yes"));
        assert!(!tags.contains_key("broken"));
    }
}
