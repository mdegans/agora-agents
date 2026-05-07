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
use url::Url;

use crate::log::log_usage;

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
// FIXME: this actually busts ollama cache. Issue is we're cloning the prompt,
// appending to it multiple times, returning another message, gets appended to
// an older version of the prompt, submit, boom, cache busted. We actually need
// to mutate the prompt which means that signatures of caller functions need to
// change to take a &mut CachedPrompt to append multiple messages, including the
// failures. Or we can swallow the cost with ollama. For Anthropic this
// behavior is fine. The agent "got it right the first time" and the breakpoint
// system ensures we can append another message out of sequence.
pub async fn send_with_nudge(
    client: &misanthropic::Client,
    // FIXME: Always use CachedPrompt instead of Prompt. Purge the codebase of
    // any bare Prompt.
    prompt: &Prompt<'_>,
) -> Result<SendResponse> {
    let mut response = client
        .message(prompt)
        .await
        .context("Ollama request failed")?
        .into_static();
    let mut total_usage = response.usage;

    // Only nudge if the prompt has tools defined — reflect/survey prompts
    // don't use tools and plain text responses are expected.
    let has_tools = prompt.functions.as_ref().is_some_and(|f| !f.is_empty());
    // AssistantMessage derefs to the inner Message
    if !has_tools || has_tool_use(&response.inner) {
        return Ok(response);
    }

    // No tool calls — nudge the model
    let mut retry_prompt = prompt.clone().into_static();

    for attempt in 0..MAX_NUDGES {
        retry_prompt
            .push_message(response.inner.clone().into_static())
            .map_err(|e| anyhow::anyhow!("turn order error on nudge: {e}"))?;
        retry_prompt
            .push_message((Role::User, NUDGE_MESSAGE))
            .map_err(|e| anyhow::anyhow!("turn order error on nudge: {e}"))?;

        let nudge_response = client
            .message(&retry_prompt)
            .await
            .context("Ollama nudge request failed")?
            .into_static();
        total_usage += nudge_response.usage;
        response = nudge_response;

        if has_tool_use(&response.inner) {
            break;
        }

        tracing::warn!(
            "Nudge {}/{MAX_NUDGES}: model still didn't produce tool calls",
            attempt + 1,
        );
    }

    // Give up — return whatever we got
    response.usage = total_usage;
    Ok(response)
}

/// Create a [`misanthropic::Client`] pointed at an Ollama endpoint.
pub fn create_ollama_client(base_url: &Url) -> Result<misanthropic::Client> {
    misanthropic::Client::new(DUMMY_KEY.to_string())
        .expect("dummy key is valid length")
        .with_base_url(base_url.as_ref())
        .map_err(|e| anyhow::anyhow!("invalid Ollama URL '{base_url}': {e}"))
}

/// Ollama LLM backend using the Anthropic-compatible API.
pub struct OllamaBackend {
    client: misanthropic::Client,
    model: String,
}

impl OllamaBackend {
    /// Create a new Ollama backend.
    pub fn new(base: &Url, model: &str) -> Result<Self> {
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
        log_usage(elapsed, resp.usage, &prompt.model.to_string());

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
