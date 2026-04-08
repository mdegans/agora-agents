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

use super::LlmBackend;

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
) -> Result<MMessage<'static>> {
    let response = client
        .message(prompt)
        .await
        .map_err(|e| anyhow::anyhow!("Ollama request failed: {e}"))?;
    let msg: MMessage<'_> = response.inner.into();

    // Only nudge if the prompt has tools defined — reflect/survey prompts
    // don't use tools and plain text responses are expected.
    let has_tools = prompt.functions.as_ref().is_some_and(|f| !f.is_empty());
    if !has_tools || has_tool_use(&msg) {
        return Ok(msg.into_static());
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
            .map_err(|e| anyhow::anyhow!("Ollama nudge request failed: {e}"))?;
        last_msg = response.inner.into();

        if has_tool_use(&last_msg) {
            return Ok(last_msg.into_static());
        }

        tracing::warn!(
            "Nudge {}/{MAX_NUDGES}: model still didn't produce tool calls",
            attempt + 1,
        );
    }

    // Give up — return whatever we got
    Ok(last_msg.into_static())
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
    async fn send(&self, prompt: &Prompt<'_>) -> Result<MMessage<'static>> {
        let start = std::time::Instant::now();

        let msg = send_with_nudge(&self.client, prompt)
            .await
            .context("Ollama send")?;

        let elapsed = start.elapsed();
        tracing::info!("  [{}] {:.1}s total", self.model, elapsed.as_secs_f64(),);

        Ok(msg)
    }

    fn backend_name(&self) -> &str {
        "ollama"
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}
