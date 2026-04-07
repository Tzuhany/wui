// ============================================================================
// Structured output via XML extraction.
//
// XML is a natural fit for LLM structured output: no format fragility, no
// escaping, and models are trained extensively on it. This module wraps the
// normal agent run with a fluent API for extracting tagged regions from the
// response.
//
// Usage:
//
//   let result = agent
//       .run_structured("What is 2+2?")
//       .extract("answer")
//       .await?;
//
//   let all = agent
//       .run_structured("Classify: ...")
//       .extract_all()
//       .await?;
// ============================================================================

use std::collections::HashMap;

use futures::StreamExt;

use wui_core::event::{AgentError, AgentEvent};

use crate::Agent;

/// A pending structured agent run.
///
/// Create via [`Agent::run_structured`]. Use [`extract`], [`extract_all`], or
/// [`extract_as`] to drive the run and capture the result.
pub struct StructuredRun<'a> {
    pub(crate) agent: &'a Agent,
    pub(crate) prompt: String,
}

impl<'a> StructuredRun<'a> {
    /// Run the agent and return the content inside the first `<tag>...</tag>`.
    ///
    /// Returns `Err` if the tag is not found in the response.
    pub async fn extract(self, tag: &str) -> Result<String, AgentError> {
        let tag = tag.to_string();
        let text = self.collect_text().await?;
        wui_core::fmt::extract_tag(&text, &tag)
            .map(str::to_string)
            .ok_or_else(|| {
                AgentError::fatal(format!(
                    "structured output: tag <{tag}> not found in response.\nFull response:\n{text}"
                ))
            })
    }

    /// Run the agent and return all top-level XML tag contents.
    pub async fn extract_all(self) -> Result<HashMap<String, String>, AgentError> {
        let text = self.collect_text().await?;
        Ok(wui_core::fmt::extract_tags(&text))
    }

    /// Run the agent, extract the content inside `<tag>`, and deserialise it
    /// as JSON into `T`.
    ///
    /// Useful when the LLM is asked to respond with a JSON value wrapped in
    /// XML tags, combining the structural clarity of XML boundaries with the
    /// expressive power of JSON for nested data.
    pub async fn extract_as<T>(self, tag: &str) -> Result<T, AgentError>
    where
        T: serde::de::DeserializeOwned,
    {
        let tag = tag.to_string();
        let raw = self.collect_text().await?;
        let inner = wui_core::fmt::extract_tag(&raw, &tag).ok_or_else(|| {
            AgentError::fatal(format!(
                "structured output: tag <{tag}> not found in response.\nFull response:\n{raw}"
            ))
        })?;
        serde_json::from_str(inner).map_err(|e| {
            AgentError::fatal(format!(
                "structured output: failed to deserialise <{tag}> content as JSON: {e}\n\
             Content was:\n{inner}"
            ))
        })
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    async fn collect_text(self) -> Result<String, AgentError> {
        let mut text = String::new();
        let mut stream = self.agent.stream(self.prompt);

        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::TextDelta(t) => text.push_str(&t),
                AgentEvent::Control(handle) => {
                    let tool_name = handle.request.tool_name().map(str::to_owned);
                    handle.deny("StructuredRun does not support interactive approvals");
                    return Err(AgentError::permission_required(tool_name.as_deref()));
                }
                AgentEvent::Done(_) => break,
                AgentEvent::Error(e) => return Err(e),
                _ => {}
            }
        }

        Ok(text)
    }
}
