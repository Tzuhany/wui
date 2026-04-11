// ── AgentRunner ───────────────────────────────────────────────────────────────

use futures::stream::BoxStream;

use crate::event::AgentEvent;

/// Anything that can run an agent prompt and stream back events.
///
/// This trait is the minimal interface that background job systems (`wui-spawn`
/// and similar crates) need in order to run agents without depending on the
/// full `wui` crate.
///
/// `wui::Agent` implements this trait. Custom runners can implement it for
/// test doubles, adapters, or alternative execution environments.
pub trait AgentRunner: Send + Sync + 'static {
    /// Start a run and return a stream of events.
    ///
    /// The stream ends with [`AgentEvent::Done`] on success or
    /// [`AgentEvent::Error`] on failure.
    fn run_stream(&self, prompt: String) -> BoxStream<'static, AgentEvent>;
}
