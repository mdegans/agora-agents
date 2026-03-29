//! Ollama backend using the OpenAI-compatible `/v1/chat/completions` endpoint.
//!
//! Builds prompts with [`misanthropic::Prompt`] and serializes them via the
//! OpenAI compatibility layer. This gives us tool calling support for free —
//! Ollama constrains grammar at generation time, guaranteeing valid JSON.

use anyhow::{Context, Result};
use async_trait::async_trait;
use misanthropic::openai::{ChatCompletionRequest, ChatCompletionResponse};
use misanthropic::prompt::Message as MMessage;
use misanthropic::Prompt;

use super::LlmBackend;

/// Ollama LLM backend using local models via the OpenAI-compatible API.
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

#[async_trait]
impl LlmBackend for OllamaBackend {
    async fn send(&self, prompt: &Prompt<'_>) -> Result<MMessage<'static>> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let start = std::time::Instant::now();

        // Convert the Prompt to an OpenAI-compatible request.
        let mut request = ChatCompletionRequest::from(prompt);
        // Override model — the Prompt may have a default/Anthropic model ID,
        // but we want the Ollama model name (e.g. "cogito:14b").
        request.model = self.model.clone();

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

        let chat_response: ChatCompletionResponse =
            response.json().await.context("parsing Ollama response")?;

        // Log inference stats
        let elapsed = start.elapsed();
        if let Some(usage) = &chat_response.usage {
            let prompt_tokens = usage.prompt_tokens;
            let completion_tokens = usage.completion_tokens;
            let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
                completion_tokens as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            };

            let slow = tok_per_sec > 0.0 && tok_per_sec < 20.0;
            let large_ctx = prompt_tokens > 32_000;

            if large_ctx || slow {
                tracing::warn!(
                    "  [{}]{}{} {}tok prompt, {}tok response, {:.1} tok/s, {:.1}s total",
                    self.model,
                    if large_ctx { " LARGE_CTX" } else { "" },
                    if slow { " SLOW" } else { "" },
                    prompt_tokens,
                    completion_tokens,
                    tok_per_sec,
                    elapsed.as_secs_f64(),
                );
            } else {
                tracing::info!(
                    "  [{}] {}tok prompt, {}tok response, {:.1} tok/s, {:.1}s total",
                    self.model,
                    prompt_tokens,
                    completion_tokens,
                    tok_per_sec,
                    elapsed.as_secs_f64(),
                );
            }
        }

        chat_response
            .into_message()
            .ok_or_else(|| anyhow::anyhow!("Ollama response contained no message"))
    }

    fn backend_name(&self) -> &str {
        "ollama"
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}
