//! Messages-API client and `LlmBackend` adapter.
//!
//! [`MessagesClient`] wraps a [`misanthropic::Client`] pointed at an
//! Anthropic-compatible `/v1/messages` endpoint. The same shape covers
//! Anthropic itself, Ollama (via `/v1/messages`), and Blallama
//! (`drama_llama/bin`); the [`Backend`] variant on the client tells the
//! caller which dialect of tool-choice / output-config to apply.
//!
//! [`MessagesPerModel`] adapts a `MessagesClient` + a model name to the
//! [`LlmBackend`] trait. It owns an `Arc<MessagesClient>` so the same
//! adapter shape works for both the seed runner's pool (where many
//! adapters share one connection per endpoint) and one-shot consumers
//! like `agora-generate` (which needs a `'static` `Box<dyn LlmBackend>`).

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use misanthropic::Prompt;
use misanthropic::prompt::Message as MMessage;
use misanthropic::prompt::message::{Block, Content, Role};
use serde::{Deserialize, Serialize};
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

/// Backend variant for a [`MessagesClient`] / `Endpoint`. Produced by the
/// `Endpoint::from_str` parser in `agora-seed::config` from the URI scheme of
/// `--messages-api` / `--batch-api` values.
///
/// Different backends support caching and tool use slightly differently.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    /// Anthropic. Selected via `anthropic://...` URIs.
    ///
    /// - `tool_choice=Any` on think_act (forces *some* tool to be used)
    /// - `None` elsewhere
    /// - `output_config` supported but not used since changing it per turn
    ///   invalidates cache, unlike Blallama.
    Anthropic,
    /// Ollama `/v1/messages` impl
    ///
    /// - `tool_choice=Auto` on think_act. `Any` is unsupported by ollama. As of
    ///   writing, `Auto` means "must use tools" with ollama (Anthropic `Any`).
    /// - `None` on the reflect/mutate/etc phases (or else tools are called when
    ///   json in text blocks is what is desired).
    /// - `output_format` is ignored by ollama (structured generation).
    Ollama,
    /// Blallama (`drama_llama/bin`) `/v1/messages` impl
    ///
    /// - `tool_choice=Any` on think_act (forces *some* tool to be used)
    /// - `None` elsewhere
    /// -  `output_config` set per phase for structured-output guarantees.
    ///
    /// This is Anthropic conformant behavior with one exception:
    ///
    /// - Cache survives `output_config` changes, so the per-phase swap is
    ///   cheap. With Anthropic, changing the `output_config` invalidates cache
    ///   even though it's a generation constraint and it shouln't strictly need
    ///   to.
    Blallama,
}

impl Backend {
    /// Short lower-case identifier used in `LlmBackend::backend_name` and logs.
    pub fn name(self) -> &'static str {
        match self {
            Backend::Anthropic => "anthropic",
            Backend::Ollama => "ollama",
            Backend::Blallama => "blallama",
        }
    }
}

/// Response from Ollama's `GET /api/tags` endpoint.
#[derive(Debug, serde::Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModelInfo>,
}

/// A single model entry from `/api/tags`.
#[derive(Debug, serde::Deserialize)]
struct OllamaModelInfo {
    name: String,
}

/// Check if a message contains any `Block::ToolUse`.
pub fn has_tool_use(msg: &MMessage<'_>) -> bool {
    match &msg.content {
        Content::MultiPart(blocks) => blocks.iter().any(|b| matches!(b, Block::ToolUse { .. })),
        Content::SinglePart(_) => false,
    }
}

/// Send a prompt to a `/v1/messages` endpoint, nudging up to
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
        .context("messages-API request failed")?
        .into_static();
    let mut total_usage = response.usage;

    // No-nudge cases:
    //   - prompt has no tools → caller is expecting plain text
    //     (reflect/survey on ollama where tools are stripped)
    //   - prompt has `output_config` → response is grammar-constrained
    //     to JSON, not tool_use, by design (reflect on blallama leaves
    //     tools attached because blallama respects tool_choice; the
    //     output_config wins at the grammar layer). Nudging here burns
    //     a second 4-minute cogito-32b inference for nothing.
    //   - response already has tool_use → success
    let has_tools = prompt.functions.as_ref().is_some_and(|f| !f.is_empty());
    if !has_tools || prompt.output_config.is_some() || has_tool_use(&response.inner) {
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
            .context("messages-API nudge request failed")?
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

/// Create a [`misanthropic::Client`] pointed at a `/v1/messages` endpoint.
///
/// Uses [`DUMMY_KEY`] — Ollama and Blallama ignore the key but `misanthropic`
/// requires a key of valid length. For real Anthropic the caller should
/// build the client directly with the user-provided key.
pub fn create_messages_client(base_url: &Url) -> Result<misanthropic::Client> {
    misanthropic::Client::new(DUMMY_KEY.to_string())
        .expect("dummy key is valid length")
        .with_base_url(base_url.as_ref())
        .map_err(|e| anyhow::anyhow!("invalid messages-API URL '{base_url}': {e}"))
}

/// A single `/v1/messages` endpoint with its connection, discovered models,
/// and backend variant. Used by both the seed runner's pool (multi-model,
/// many endpoints, shared via `Arc`) and one-shot consumers via the
/// [`MessagesPerModel`] adapter.
#[derive(Clone)]
pub struct MessagesClient {
    /// Base URL (e.g. `http://localhost:11434`).
    pub url: Url,
    /// Models available on this endpoint (populated by [`Self::discover`];
    /// empty for clients constructed via [`Self::new`]).
    pub models: HashSet<String>,
    /// Anthropic-compat client pointed at this endpoint.
    pub client: misanthropic::Client,
    /// Backend dialect (Anthropic / Ollama / Blallama).
    pub backend: Backend,
}

impl std::fmt::Debug for MessagesClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MessagesClient")
            .field("url", &self.url)
            .field("backend", &self.backend)
            .field("models", &self.models)
            .finish_non_exhaustive()
    }
}

impl MessagesClient {
    /// Create a client with no discovered models. Use [`Self::discover`]
    /// instead if you want the model set populated.
    pub fn new(url: Url, backend: Backend) -> Result<Self> {
        let client = create_messages_client(&url)?;
        Ok(Self {
            url,
            models: HashSet::new(),
            client,
            backend,
        })
    }

    /// Discover available models by querying `GET /api/tags` (Ollama and
    /// Blallama expose this; Anthropic does not — callers building an
    /// Anthropic client should use [`Self::new`]).
    pub async fn discover(http: &reqwest::Client, url: Url, backend: Backend) -> Result<Self> {
        let tags_url = url
            .join("api/tags")
            .map_err(|e| anyhow::anyhow!("building /api/tags URL from {url}: {e}"))?;
        let resp: OllamaTagsResponse = http
            .get(tags_url.clone())
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("connecting to {tags_url}: {e}"))?
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("parsing /api/tags from {url}: {e}"))?;

        let models: HashSet<String> = resp.models.into_iter().map(|m| m.name).collect();

        tracing::info!("Endpoint {url}: {} model(s) [{}]", models.len(), {
            let mut sorted: Vec<_> = models.iter().cloned().collect();
            sorted.sort();
            sorted.join(", ")
        },);

        let client = create_messages_client(&url)?;

        Ok(Self {
            url,
            models,
            client,
            backend,
        })
    }

    /// Send a prompt for a specific model. Returns just the response message,
    /// discarding usage statistics. Thin wrapper around [`Self::send_response`]
    /// for callers that don't track usage.
    // FIXME: rename `send` functions to `complete`. The former is too generic
    // (send where?) while complete is more consistent with most completion APIs
    pub async fn send(
        &self,
        prompt: &Prompt<'_>,
        model: &str,
    ) -> Result<misanthropic::prompt::AssistantMessage<'static>> {
        Ok(self.send_response(prompt, model).await?.inner)
    }

    /// Send a prompt for a specific model and return the full
    /// [`SendResponse`] (assistant message + usage). Logs usage as a side
    /// effect via [`log_usage`].
    pub async fn send_response(&self, prompt: &Prompt<'_>, model: &str) -> Result<SendResponse> {
        let start = std::time::Instant::now();
        let resp = send_with_nudge(&self.client, prompt).await?;
        let elapsed = start.elapsed();

        log_usage(elapsed, resp.usage, model);

        Ok(resp)
    }
}

/// [`LlmBackend`] adapter binding a [`MessagesClient`] to a specific model.
///
/// Owns an `Arc<MessagesClient>` so the same adapter shape covers both
/// borrowed-pool use (seed runner clones the `Arc` per agent) and `'static`
/// owned use (`Box<dyn LlmBackend>` consumers like `agora-generate`).
pub struct MessagesPerModel {
    pub client: Arc<MessagesClient>,
    pub model: String,
}

#[async_trait]
impl LlmBackend for MessagesPerModel {
    async fn send(&self, prompt: &Prompt<'_>) -> Result<SendResponse> {
        self.client.send_response(prompt, &self.model).await
    }

    fn backend_name(&self) -> &str {
        self.client.backend.name()
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}
