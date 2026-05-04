//! Ollama backend using the Anthropic-compatible `/v1/messages` endpoint.
//!
//! Uses [`misanthropic::Client`] with a custom base URL pointed at Ollama.
//! Includes a nudge-retry loop for when the model returns thinking/text
//! without tool calls (Ollama doesn't enforce `tool_choice`).

use anyhow::{Context, Result};
use async_trait::async_trait;
use misanthropic::Prompt;
use misanthropic::prompt::Message as MMessage;
use misanthropic::prompt::message::{Block, Content, Role};

use super::{LlmBackend, SendResponse};

/// Maximum number of nudge retries when the model doesn't produce tool calls.
const MAX_NUDGES: usize = 1;

/// Message sent to nudge the model into using a tool.
const NUDGE_MESSAGE: &str =
    "You must use one of the provided tools to take an action. Do not respond with text.";

/// Dummy API key for Ollama (108 bytes, required by misanthropic but ignored
/// by Ollama's endpoint).
const DUMMY_KEY: &str = "sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";

/// Check if a message contains any `Block::ToolUse`.
pub fn has_tool_use(msg: &MMessage<'_>) -> bool {
    match &msg.content {
        Content::MultiPart(blocks) => blocks.iter().any(|b| matches!(b, Block::ToolUse { .. })),
        Content::SinglePart(_) => false,
    }
}

/// Send a prompt to an Ollama Anthropic-compat endpoint, nudging up to
/// [`MAX_NUDGES`] times if the model returns no tool calls.
///
/// Messages are preserved as-is between retries to maintain Ollama's KV
/// prefix cache coherence.
pub async fn send_with_nudge(
    client: &misanthropic::Client,
    prompt: &Prompt<'_>,
) -> Result<SendResponse> {
    let response = client
        .message(prompt)
        .await
        .context("Ollama request failed")?;
    let mut total_usage = response.usage;
    let mut last_stop = response.stop_reason;
    let msg: MMessage<'_> = response.inner.into();

    // Only nudge if the prompt has tools defined — reflect/survey prompts
    // don't use tools and plain text responses are expected.
    let has_tools = prompt.functions.as_ref().is_some_and(|f| !f.is_empty());
    if !has_tools || has_tool_use(&msg) {
        return Ok(SendResponse {
            message: msg.into_static(),
            usage: Some(total_usage),
            stop_reason: last_stop,
        });
    }

    // No tool calls — nudge the model
    let mut retry_prompt = prompt.clone().into_static();
    let mut last_msg = msg;

    for attempt in 0..MAX_NUDGES {
        retry_prompt
            .push_message(last_msg.into_static())
            .map_err(|e| anyhow::anyhow!("turn order error on nudge: {e}"))?;
        retry_prompt
            .push_message((Role::User, NUDGE_MESSAGE))
            .map_err(|e| anyhow::anyhow!("turn order error on nudge: {e}"))?;

        let response = client
            .message(&retry_prompt)
            .await
            .context("Ollama nudge request failed")?;
        total_usage += response.usage;
        last_stop = response.stop_reason;
        last_msg = response.inner.into();

        if has_tool_use(&last_msg) {
            return Ok(SendResponse {
                message: last_msg.into_static(),
                usage: Some(total_usage),
                stop_reason: last_stop,
            });
        }

        tracing::warn!(
            "Nudge {}/{MAX_NUDGES}: model still didn't produce tool calls",
            attempt + 1,
        );
    }

    // Give up — return whatever we got
    Ok(SendResponse {
        message: last_msg.into_static(),
        usage: Some(total_usage),
        stop_reason: last_stop,
    })
}

/// Create a [`misanthropic::Client`] pointed at an Ollama endpoint.
pub fn create_ollama_client(base_url: &str) -> Result<misanthropic::Client> {
    misanthropic::Client::new(DUMMY_KEY.to_string())
        .expect("dummy key is valid length")
        .with_base_url(base_url)
        .map_err(|e| anyhow::anyhow!("invalid Ollama URL '{base_url}': {e}"))
}

/// Ollama LLM backend using the Anthropic-compatible API.
pub struct OllamaBackend {
    client: misanthropic::Client,
    model: String,
}

impl OllamaBackend {
    /// Create a new Ollama backend.
    ///
    /// `base_url` defaults to `http://localhost:11434` if not specified.
    pub fn new(base_url: Option<&str>, model: &str) -> Result<Self> {
        let base = base_url.unwrap_or("http://localhost:11434");
        let client = create_ollama_client(base)?;
        Ok(Self {
            client,
            model: model.to_string(),
        })
    }
}

#[async_trait]
impl LlmBackend for OllamaBackend {
    async fn send(&self, prompt: &Prompt<'_>) -> Result<SendResponse> {
        let start = std::time::Instant::now();

        let resp = send_with_nudge(&self.client, prompt)
            .await
            .context("Ollama send")?;

        let elapsed = start.elapsed();
        if let Some(ref usage) = resp.usage {
            tracing::debug!(
                "  [{}] {:.1}s, {}tok in, {}tok out",
                self.model,
                elapsed.as_secs_f64(),
                usage.input_tokens,
                usage.output_tokens,
            );
        } else {
            tracing::info!("  [{}] {:.1}s total", self.model, elapsed.as_secs_f64());
        }

        Ok(resp)
    }

    fn backend_name(&self) -> &str {
        "ollama"
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}

/// [`LlmBackend`] adapter that binds an [`OllamaEndpoint`] to a specific
/// model per call. Lets the scheduler's sequential path reuse the
/// [`exchange`](super::exchange) / [`exchange_bare`](super::exchange_bare)
/// helpers without spinning up a fresh [`OllamaBackend`] (and therefore a
/// fresh `misanthropic::Client`) for every agent.
///
/// [`OllamaEndpoint`]: crate::batch::ollama::OllamaEndpoint
pub struct OllamaPerModel<'a> {
    pub endpoint: &'a crate::batch::ollama::OllamaEndpoint,
    pub model: &'a str,
}

#[async_trait]
impl<'a> LlmBackend for OllamaPerModel<'a> {
    async fn send(&self, prompt: &Prompt<'_>) -> Result<SendResponse> {
        self.endpoint.send_response(prompt, self.model).await
    }

    fn backend_name(&self) -> &str {
        "ollama-per-model"
    }

    fn model_id(&self) -> &str {
        self.model
    }
}
