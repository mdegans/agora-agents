//! LLM backend abstraction for agent reasoning.
//!
//! Backends implement [`LlmBackend::send`] which takes a [`misanthropic::Prompt`]
//! and returns a [`misanthropic::prompt::Message`]. The convenience method
//! [`LlmBackend::complete`] wraps text-in/text-out for callers that don't need
//! the full prompt/message types.

pub mod anthropic;
pub mod ollama;

use anyhow::Result;
use async_trait::async_trait;

// Re-export misanthropic prompt types for callers that want the full API.
pub use misanthropic::prompt::message::Content as MContent;
pub use misanthropic::prompt::message::Role as MRole;
pub use misanthropic::prompt::Message as MMessage;
pub use misanthropic::Prompt;

/// A message in a conversation (simple text-only representation).
///
/// Used by [`LlmBackend::complete`] for backward compatibility. New code
/// should build a [`Prompt`] directly and call [`LlmBackend::send`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// Message role.
#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// Trait for LLM backends that can generate completions.
#[async_trait]
pub trait LlmBackend: Send + Sync {
    /// Send a full [`Prompt`] and get the response [`Message`](MMessage).
    ///
    /// This is the primary method backends must implement. The returned message
    /// may contain text, tool calls, or both.
    async fn send(&self, prompt: &Prompt<'_>) -> Result<MMessage<'static>>;

    /// Name of the backend for logging.
    fn backend_name(&self) -> &str;

    /// Model identifier.
    fn model_id(&self) -> &str;

    /// Convenience: text-in/text-out. Builds a [`Prompt`] and calls [`send`](Self::send).
    ///
    /// Extracts the text content from the response, ignoring tool calls and
    /// other block types. This preserves backward compatibility with callers
    /// that pass raw strings.
    async fn complete(
        &self,
        system_prompt: &str,
        messages: &[Message],
        max_tokens: u32,
    ) -> Result<String> {
        use std::num::NonZeroU32;

        let model_id: misanthropic::model::Id<'_> = self.model_id().into();

        let mut prompt = Prompt {
            model: model_id,
            max_tokens: NonZeroU32::new(max_tokens).unwrap_or(NonZeroU32::new(1024).unwrap()),
            system: Some(MContent::text(system_prompt)),
            ..Default::default()
        };

        for msg in messages {
            let role = match msg.role {
                Role::User => MRole::User,
                Role::Assistant => MRole::Assistant,
            };
            prompt
                .push_message((role, msg.content.as_str()))
                .map_err(|e| anyhow::anyhow!("turn order error: {e}"))?;
        }

        let response = self.send(&prompt).await?;
        Ok(response.content.to_string())
    }
}

// Allow calling LlmBackend methods on Box<dyn LlmBackend>.
#[async_trait]
impl LlmBackend for Box<dyn LlmBackend> {
    async fn send(&self, prompt: &Prompt<'_>) -> Result<MMessage<'static>> {
        (**self).send(prompt).await
    }

    fn backend_name(&self) -> &str {
        (**self).backend_name()
    }

    fn model_id(&self) -> &str {
        (**self).model_id()
    }
}
