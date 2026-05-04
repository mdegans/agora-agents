//! Anthropic backend using the `misanthropic` crate (Messages API).
//!
//! This is the native backend — prompts are sent directly to the Anthropic API
//! without any conversion layer.

use anyhow::{Context, Result};
use async_trait::async_trait;
use misanthropic::Prompt;

use super::{LlmBackend, SendResponse};

/// Anthropic LLM backend using Claude models.
pub struct AnthropicBackend {
    client: misanthropic::Client,
    model: String,
}

impl AnthropicBackend {
    /// Create a new Anthropic backend with the given API key and model.
    ///
    /// `api_key` must be a valid Anthropic API key (108 bytes).
    pub fn new(api_key: String, model: &str) -> Result<Self> {
        let client = misanthropic::Client::new(api_key)
            .context("creating Anthropic client (is the API key 108 bytes?)")?;
        Ok(Self {
            client,
            model: model.to_string(),
        })
    }
}

#[async_trait]
impl LlmBackend for AnthropicBackend {
    async fn send(&self, prompt: &Prompt<'_>) -> Result<SendResponse> {
        // `.context(...)` preserves the downcast chain so callers can
        // walk `err.chain()` looking for `misanthropic::client::Error` or
        // `AnthropicError` to make retry decisions.
        let response = self
            .client
            .message(prompt)
            .await
            .context("Anthropic API call failed")?;

        tracing::debug!(
            "  [{}] {}tok in, {}tok out",
            self.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
        );

        let stop_reason = response.stop_reason;
        let msg: misanthropic::prompt::Message<'_> = response.inner.into();
        Ok(SendResponse {
            message: msg.into_static(),
            usage: Some(response.usage),
            stop_reason,
        })
    }

    fn backend_name(&self) -> &str {
        "anthropic"
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}
