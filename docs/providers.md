# Using Wui with Different Providers

Wui ships with two built-in provider adapters: **Anthropic** and **OpenAI**. The OpenAI adapter speaks the OpenAI-compatible API format, which means it works with any service that implements that protocol — not just OpenAI itself.

## Built-in Providers

### Anthropic

```rust
use wui::providers::Anthropic;

let provider = Anthropic::new(api_key);

// With options:
let provider = Anthropic::new(api_key)
    .with_default_model("claude-sonnet-4-20250514")
    .with_prompt_caching()
    .with_thinking_budget(16_384);
```

Enable with `features = ["anthropic"]`.

### OpenAI

```rust
use wui::providers::OpenAI;

let provider = OpenAI::new(api_key);

// With options:
let provider = OpenAI::new(api_key)
    .with_default_model("gpt-4o");
```

Enable with `features = ["openai"]`.

## OpenAI-Compatible Services

The `OpenAI` provider supports `with_base_url()`, which makes it work with any service that implements the OpenAI chat completions API. No additional code needed.

### Ollama (local)

```rust
let provider = OpenAI::builder()
    .with_base_url("unused", "http://localhost:11434/v1")
    .with_default_model("llama3")
    .build();
```

The API key is ignored by Ollama but the field is required by the builder — pass any string.

### vLLM

```rust
let provider = OpenAI::builder()
    .with_base_url("unused", "http://localhost:8000/v1")
    .with_default_model("meta-llama/Llama-3-8b-chat-hf")
    .build();
```

### Together AI

```rust
let provider = OpenAI::builder()
    .with_base_url(together_api_key, "https://api.together.xyz/v1")
    .with_default_model("meta-llama/Llama-3-70b-chat-hf")
    .build();
```

### Groq

```rust
let provider = OpenAI::builder()
    .with_base_url(groq_api_key, "https://api.groq.com/openai/v1")
    .with_default_model("llama3-70b-8192")
    .build();
```

### Azure OpenAI

```rust
// Azure uses a different URL pattern — point to your deployment:
let provider = OpenAI::builder()
    .with_base_url(
        azure_api_key,
        "https://YOUR-RESOURCE.openai.azure.com/openai/deployments/YOUR-DEPLOYMENT/v1"
    )
    .with_default_model("gpt-4o")
    .build();
```

Note: Azure may require additional headers for API version. If so, implement a custom `Provider` (see below).

### OpenRouter

```rust
let provider = OpenAI::builder()
    .with_base_url(openrouter_api_key, "https://openrouter.ai/api/v1")
    .with_default_model("anthropic/claude-sonnet-4-20250514")
    .build();
```

## Writing a Custom Provider

For services that don't follow the OpenAI protocol (e.g. Google Gemini, AWS Bedrock), implement the `Provider` trait directly. It has a single required method:

```rust
use async_trait::async_trait;
use std::pin::Pin;
use futures::Stream;
use wui::{Provider};
use wui_core::provider::{ChatRequest, ProviderError, ProviderCapabilities};
use wui_core::event::StreamEvent;

struct MyProvider { /* ... */ }

#[async_trait]
impl Provider for MyProvider {
    async fn stream(
        &self,
        req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        // 1. Convert ChatRequest to your provider's format
        // 2. Make the HTTP request
        // 3. Return a stream of StreamEvent variants:
        //    - StreamEvent::TextDelta { text }
        //    - StreamEvent::ThinkingDelta { text }
        //    - StreamEvent::ToolUseStart { id, name }
        //    - StreamEvent::ToolInputDelta { id, chunk }
        //    - StreamEvent::ToolUseEnd { id }
        //    - StreamEvent::MessageEnd { usage, stop_reason }
        todo!()
    }

    // Optional: declare what your provider supports
    fn capabilities(&self, _model: Option<&str>) -> ProviderCapabilities {
        ProviderCapabilities::default()
            .with_tool_calling(true)
            .with_image_input(true)
    }
}
```

The `ChatRequest` you receive contains:
- `model` — model name
- `system` — system prompt
- `messages` — conversation history (user, assistant, tool results)
- `tools` — tool definitions (JSON Schema format)
- `max_tokens`, `temperature` — generation parameters
- `thinking_budget` — extended thinking budget (Anthropic-specific; ignore if unsupported)

Map these to your provider's native API format and return `StreamEvent`s as they arrive.

## Provider Capabilities

Override `capabilities()` to tell the runtime what your provider supports:

```rust
fn capabilities(&self, model: Option<&str>) -> ProviderCapabilities {
    ProviderCapabilities::default()
        .with_tool_calling(true)          // can the model call tools?
        .with_thinking(false)             // extended thinking support?
        .with_image_input(true)           // can it see images?
        .with_document_input(false)       // can it read PDFs?
        .with_structured_output(true)     // native JSON mode?
        .with_max_context_window(128_000) // auto-calibrate compression
}
```

The `max_context_window` value is used to auto-calibrate the compression pipeline — if you set it, users don't need to manually configure `CompressPipeline::window_tokens`.
