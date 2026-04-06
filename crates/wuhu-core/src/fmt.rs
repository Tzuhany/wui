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
