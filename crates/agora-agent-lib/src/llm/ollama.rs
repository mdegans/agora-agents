//! Ollama backend using the OpenAI-compatible API.

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{LlmBackend, Message, Role};

/// Ollama LLM backend using local models.
pub struct OllamaBackend {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl OllamaBackend {
    /// Create a new Ollama backend.
    ///
    /// `base_url` defaults to `http://localhost:11434` if not specified.
    pub fn new(base_url: Option<&str>, model: &str) -> Self {
        let base_url = base_url
            .unwrap_or("http://localhost:11434")
            .trim_end_matches('/')
            .to_string();
        Self {
            client: reqwest::Client::new(),
            base_url,
            model: model.to_string(),
        }
    }
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    think: bool,
    options: Option<ChatOptions>,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatOptions {
    num_predict: u32,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: Option<ResponseMessage>,
    prompt_eval_count: Option<u64>,
    eval_count: Option<u64>,
    eval_duration: Option<u64>,
    total_duration: Option<u64>,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}

#[async_trait]
impl LlmBackend for OllamaBackend {
    async fn complete(
        &self,
        system_prompt: &str,
        messages: &[Message],
        max_tokens: u32,
    ) -> Result<String> {
        // Use Ollama's native /api/chat endpoint (more reliable than OpenAI compat)
        let url = format!("{}/api/chat", self.base_url);

        let mut chat_messages = vec![ChatMessage {
            role: "system".to_string(),
            content: system_prompt.to_string(),
        }];

        for msg in messages {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            chat_messages.push(ChatMessage {
                role: role.to_string(),
                content: msg.content.clone(),
            });
        }

        let request = ChatRequest {
            model: self.model.clone(),
            messages: chat_messages,
            stream: false,
            think: false,
            options: Some(ChatOptions {
                num_predict: max_tokens,
            }),
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Ollama request failed")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Ollama returned {status}: {body}");
        }

        let chat_response: ChatResponse = response
            .json()
            .await
            .context("parsing Ollama response")?;

        // Log inference stats
        if let (Some(prompt_tokens), Some(eval_tokens)) =
            (chat_response.prompt_eval_count, chat_response.eval_count)
        {
            let tok_per_sec = chat_response
                .eval_duration
                .filter(|&d| d > 0)
                .map(|d| eval_tokens as f64 / (d as f64 / 1_000_000_000.0));
            let total_secs = chat_response
                .total_duration
                .map(|d| d as f64 / 1_000_000_000.0);

            if prompt_tokens > 32_000 {
                tracing::warn!(
                    "  [{}] LARGE CONTEXT: {}tok prompt, {}tok response, {:.1} tok/s, {:.1}s total",
                    self.model,
                    prompt_tokens,
                    eval_tokens,
                    tok_per_sec.unwrap_or(0.0),
                    total_secs.unwrap_or(0.0),
                );
            } else {
                tracing::debug!(
                    "  [{}] {}tok prompt, {}tok response, {:.1} tok/s, {:.1}s total",
                    self.model,
                    prompt_tokens,
                    eval_tokens,
                    tok_per_sec.unwrap_or(0.0),
                    total_secs.unwrap_or(0.0),
                );
            }
        }

        chat_response
            .message
            .map(|m| m.content)
            .ok_or_else(|| anyhow::anyhow!("Ollama response contained no message"))
    }

    fn backend_name(&self) -> &str {
        "ollama"
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}
