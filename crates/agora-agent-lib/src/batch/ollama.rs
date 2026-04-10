//! Ollama endpoint for sequential per-agent LLM calls.
//!
//! Ollama doesn't have a batch API. The seed runner drives requests
//! sequentially per agent for KV prefix cache reuse — each subsequent
//! request shares the same prefix and gets automatic cache hits.
//!
//! Uses Ollama's Anthropic-compatible `/v1/messages` endpoint via
//! [`misanthropic::Client`] with a custom base URL.

use std::collections::HashSet;

use misanthropic::Prompt;
use misanthropic::prompt::Message as MMessage;

use crate::llm::ollama::{create_ollama_client, send_with_nudge};

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
    pub url: String,
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
    pub fn new(url: impl Into<String>) -> anyhow::Result<Self> {
        let url = url.into();
        let url = url.trim_end_matches('/').to_string();
        let client = create_ollama_client(&url)?;
        Ok(Self {
            url,
            models: HashSet::new(),
            client,
        })
    }

    /// Discover available models by querying `GET /api/tags`.
    pub async fn discover(http: &reqwest::Client, url: &str) -> anyhow::Result<Self> {
        let url = url.trim_end_matches('/').to_string();
        let tags_url = format!("{url}/api/tags");
        let resp: OllamaTagsResponse = http
            .get(&tags_url)
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
    pub async fn send(
        &self,
        prompt: &Prompt<'_>,
        model: &str,
    ) -> anyhow::Result<MMessage<'static>> {
        Ok(self.send_response(prompt, model).await?.message)
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
        if let Some(ref usage) = resp.usage {
            tracing::debug!(
                "  [{model}@{}] {:.1}s, {}tok in, {}tok out",
                self.url,
                elapsed.as_secs_f64(),
                usage.input_tokens,
                usage.output_tokens,
            );
        } else {
            tracing::debug!("  [{model}@{}] {:.1}s", self.url, elapsed.as_secs_f64());
        }

        Ok(resp)
    }
}

impl Default for OllamaEndpoint {
    fn default() -> Self {
        Self::new("http://localhost:11434").expect("default Ollama URL should be valid")
    }
}
