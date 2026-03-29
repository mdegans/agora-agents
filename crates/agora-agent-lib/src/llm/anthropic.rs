//! Anthropic backend using the `misanthropic` crate (Messages API).
//!
//! This is the native backend — prompts are sent directly to the Anthropic API
//! without any conversion layer.

use anyhow::{Context, Result};
use async_trait::async_trait;
use misanthropic::prompt::Message as MMessage;
use misanthropic::Prompt;

use super::LlmBackend;

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
    async fn send(&self, prompt: &Prompt<'_>) -> Result<MMessage<'static>> {
        let response = self
            .client
            .message(prompt)
            .await
            .map_err(|e| anyhow::anyhow!("Anthropic API call failed: {e}"))?;

        tracing::debug!(
            "  [{}] {}tok in, {}tok out",
            self.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
        );

        // response::Message -> prompt::Message via Into<Message> on AssistantMessage
        let msg: misanthropic::prompt::Message<'_> = response.inner.into();
        Ok(msg.into_static())
    }

    fn backend_name(&self) -> &str {
        "anthropic"
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}
