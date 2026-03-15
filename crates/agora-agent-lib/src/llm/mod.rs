//! LLM backend abstraction for agent reasoning.

pub mod anthropic;
pub mod ollama;

use anyhow::Result;
use async_trait::async_trait;

/// A message in a conversation.
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
    /// Send a conversation and get a text response.
    async fn complete(
        &self,
        system_prompt: &str,
        messages: &[Message],
        max_tokens: u32,
    ) -> Result<String>;

    /// Name of the backend for logging.
    fn backend_name(&self) -> &str;

    /// Model identifier.
    fn model_id(&self) -> &str;
}

// Allow calling LlmBackend methods on Box<dyn LlmBackend>.
#[async_trait]
impl LlmBackend for Box<dyn LlmBackend> {
    async fn complete(
        &self,
        system_prompt: &str,
        messages: &[Message],
        max_tokens: u32,
    ) -> Result<String> {
        (**self).complete(system_prompt, messages, max_tokens).await
    }

    fn backend_name(&self) -> &str {
        (**self).backend_name()
    }

    fn model_id(&self) -> &str {
        (**self).model_id()
    }
}
