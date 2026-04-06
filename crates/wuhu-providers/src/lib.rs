// ============================================================================
// wuhu-providers — LLM adapters.
//
// Each provider is feature-gated. Enable only what you use:
//
//   wuhu-providers = { version = "0.1", features = ["anthropic"] }
//
// All providers implement `wuhu_core::provider::Provider`. The engine
// never imports this crate — it only knows the trait.
// ============================================================================

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub use anthropic::Anthropic;

#[cfg(feature = "openai")]
pub use openai::OpenAI;
