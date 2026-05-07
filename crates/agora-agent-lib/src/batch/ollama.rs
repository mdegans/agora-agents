//! Ollama endpoint for sequential per-agent LLM calls.
//!
//! Ollama doesn't have a batch API. The seed runner drives requests
//! sequentially per agent for KV prefix cache reuse — each subsequent
//! request shares the same prefix and gets automatic cache hits.
//!
//! Uses Ollama's Anthropic-compatible `/v1/messages` endpoint via
//! [`misanthropic::Client`] with a custom base URL.

use std::collections::HashSet;

use misanthropic::{Prompt, prompt::AssistantMessage};
use url::Url;

use crate::{
    llm::ollama::{create_ollama_client, send_with_nudge},
    log::log_usage,
};

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

/// An Ollama endpoint with its URL, discovered models, and API client.
#[derive(Clone)]
pub struct OllamaEndpoint {
    /// Base URL (e.g. `http://localhost:11434`).
    pub url: Url,
    /// Models available on this endpoint (populated by [`discover`]).
    pub models: HashSet<String>,
    /// Anthropic-compat client pointed at this endpoint.
    pub client: misanthropic::Client,
}

impl std::fmt::Debug for OllamaEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaEndpoint")
            .field("url", &self.url)
            .field("models", &self.models)
            .finish_non_exhaustive()
    }
}

impl OllamaEndpoint {
    /// Create an endpoint with no discovered models.
    pub fn new(url: Url) -> anyhow::Result<Self> {
        let client = create_ollama_client(&url)?;
        Ok(Self {
            url,
            models: HashSet::new(),
            client,
        })
    }

    /// Discover available models by querying `GET /api/tags`.
    pub async fn discover(http: &reqwest::Client, url: Url) -> anyhow::Result<Self> {
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

        let client = create_ollama_client(&url)?;

        Ok(Self {
            url,
            models,
            client,
        })
    }

    /// Send a single prompt to this endpoint and return just the response
    /// message, discarding usage statistics. Thin wrapper around
    /// [`OllamaEndpoint::send_response`] for callers that don't track usage.
    // FIXME: rename `send` functions to `complete`. The former is too generic
    // (send where?) while complete is more consistent with most completion APIs
    pub async fn send(
        &self,
        prompt: &Prompt<'_>,
        model: &str,
    ) -> anyhow::Result<AssistantMessage<'static>> {
        Ok(self.send_response(prompt, model).await?.inner)
    }

    /// Send a single prompt and return the full [`SendResponse`] including
    /// usage statistics. Preferred over [`Self::send`] when usage matters
    /// (cycle accounting, cost logging).
    pub async fn send_response(
        &self,
        prompt: &Prompt<'_>,
        model: &str,
    ) -> anyhow::Result<crate::llm::SendResponse> {
        let start = std::time::Instant::now();
        let resp = send_with_nudge(&self.client, prompt).await?;
        let elapsed = start.elapsed();

        log_usage(elapsed, resp.usage, model);

        Ok(resp)
    }
}

impl Default for OllamaEndpoint {
    fn default() -> Self {
        Self::new(
            "http://localhost:11434"
                .parse()
                .expect("default Ollama URL should be valid"),
        )
        .expect("default Ollama URL should be valid")
    }
}
