//! Anthropic backend using the `misanthropic` crate (Messages API).

use anyhow::{Context, Result};
use async_trait::async_trait;

use super::{LlmBackend, Message, Role};

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
    async fn complete(
        &self,
        system_prompt: &str,
        messages: &[Message],
        max_tokens: u32,
    ) -> Result<String> {
        use misanthropic::prompt::message::{Content, Role as MRole};
        use std::num::NonZeroU32;

        let model_id: misanthropic::model::Id<'_> = self.model.as_str().into();

        let mut prompt = misanthropic::Prompt {
            model: model_id,
            max_tokens: NonZeroU32::new(max_tokens).unwrap_or(NonZeroU32::new(1024).unwrap()),
            system: Some(Content::text(system_prompt)),
            ..Default::default()
        };

        for msg in messages {
            let role = match msg.role {
                Role::User => MRole::User,
                Role::Assistant => MRole::Assistant,
            };
            prompt = prompt
                .add_message((role, msg.content.as_str()))
                .context("adding message to prompt")?;
        }

        let response = self
            .client
            .message(&prompt)
            .await
            .context("Anthropic API call failed")?;

        // response::Message has Display via its inner AssistantMessage
        Ok(response.to_string())
    }

    fn backend_name(&self) -> &str {
        "anthropic"
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}
