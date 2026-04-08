// ── ToolInput ────────────────────────────────────────────────────────────────
//
// Every tool's call() receives a raw serde_json::Value. ToolInput wraps it
// with typed accessors so each extraction is one clear line instead of
// nested match arms and unwrap chains.

use serde_json::Value;

// ── Context Injection ─────────────────────────────────────────────────────────

/// System-level context a tool wants the LLM to see on the next turn.
///
/// Injected as a `<system-reminder>` block — clearly a framework message,
/// not a forged User or Assistant turn. This is the only injection surface
/// tools have: they cannot write arbitrary roles into the conversation history.
///
/// ```rust,ignore
/// ToolOutput::success("Done.")
///     .with_injections([ContextInjection::new("The file was written to /tmp/out.json")])
/// ```
#[derive(Debug, Clone)]
pub struct ContextInjection {
    /// The text to inject. Plain prose — the engine wraps it in
    /// `<system-reminder>` tags before appending to the conversation.
    pub text: String,
}

impl ContextInjection {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

// ── ToolInput ─────────────────────────────────────────────────────────────────

/// Ergonomic wrapper for extracting typed fields from a JSON tool input.
///
/// Reduces boilerplate in `Tool::call()` implementations:
///
/// ```rust,ignore
/// async fn call(&self, input: Value, _ctx: &ToolCtx) -> ToolOutput {
///     let inp = ToolInput(&input);
///     let url   = match inp.required_str("url")   { Ok(v) => v, Err(e) => return ToolOutput::error(e) };
///     let limit = inp.optional_u64("limit").unwrap_or(100);
///     ...
/// }
/// ```
#[derive(Copy, Clone)]
pub struct ToolInput<'a>(pub &'a Value);

impl<'a> ToolInput<'a> {
    // ── String ────────────────────────────────────────────────────────

    /// Extract a required, non-empty string field.
    pub fn required_str(&self, key: &str) -> Result<&'a str, String> {
        match self.0[key].as_str().filter(|s| !s.is_empty()) {
            Some(s) => Ok(s),
            None => Err(format!("'{key}' is required")),
        }
    }

    /// Extract an optional string field. Returns `None` when absent or null.
    pub fn optional_str(&self, key: &str) -> Option<&'a str> {
        self.0[key].as_str()
    }

    // ── Boolean ───────────────────────────────────────────────────────

    /// Extract a required boolean field.
    pub fn required_bool(&self, key: &str) -> Result<bool, String> {
        self.0[key]
            .as_bool()
            .ok_or_else(|| format!("'{key}' is required (bool)"))
    }

    /// Extract an optional boolean field.
    pub fn optional_bool(&self, key: &str) -> Option<bool> {
        self.0[key].as_bool()
    }

    // ── Integer ───────────────────────────────────────────────────────

    /// Extract a required unsigned integer field.
    pub fn required_u64(&self, key: &str) -> Result<u64, String> {
        self.0[key]
            .as_u64()
            .ok_or_else(|| format!("'{key}' is required (integer)"))
    }

    /// Extract an optional unsigned integer field.
    pub fn optional_u64(&self, key: &str) -> Option<u64> {
        self.0[key].as_u64()
    }

    /// Extract a required signed integer field.
    pub fn required_i64(&self, key: &str) -> Result<i64, String> {
        self.0[key]
            .as_i64()
            .ok_or_else(|| format!("'{key}' is required (integer)"))
    }

    /// Extract an optional signed integer field.
    pub fn optional_i64(&self, key: &str) -> Option<i64> {
        self.0[key].as_i64()
    }

    // ── Float ─────────────────────────────────────────────────────────

    /// Extract an optional float field.
    pub fn optional_f64(&self, key: &str) -> Option<f64> {
        self.0[key].as_f64()
    }

    // ── Array ─────────────────────────────────────────────────────────

    /// Extract a required array field.
    pub fn required_array(&self, key: &str) -> Result<&'a Vec<Value>, String> {
        self.0[key]
            .as_array()
            .ok_or_else(|| format!("'{key}' is required (array)"))
    }

    /// Extract an optional array field.
    pub fn optional_array(&self, key: &str) -> Option<&'a Vec<Value>> {
        self.0[key].as_array()
    }

    // ── Object ────────────────────────────────────────────────────────

    /// Extract a required object field.
    pub fn required_object(&self, key: &str) -> Result<&'a serde_json::Map<String, Value>, String> {
        self.0[key]
            .as_object()
            .ok_or_else(|| format!("'{key}' is required (object)"))
    }

    /// Extract an optional object field.
    pub fn optional_object(&self, key: &str) -> Option<&'a serde_json::Map<String, Value>> {
        self.0[key].as_object()
    }
}
