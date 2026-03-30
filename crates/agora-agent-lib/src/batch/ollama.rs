//! Ollama "batch" backend — sequential per model, parallel across models.
//!
//! Ollama doesn't have a batch API. Instead, we manage batching ourselves:
//! - Items are grouped by model (done by the scheduler's grouping algorithm)
//! - Within a model group, requests run **sequentially** to maximize Ollama's
//!   KV prefix cache reuse (consecutive requests with the same prefix get
//!   automatic cache hits)
//! - Different model groups run **in parallel** across endpoints/GPUs
//!
//! This avoids the wave problem where all agents of one model post at once,
//! and minimizes model loading and memory reallocation.
//!
//! [`MultiOllamaBatch`] extends this to multiple endpoints: it discovers
//! available models via `/api/tags` at startup and routes work items to the
//! right endpoint. Different endpoints process concurrently.

use std::collections::{HashMap, HashSet};

use agora_agentkit::scheduler::{
    BatchBackend, BatchError, BatchState, WorkItem, WorkResult,
};
use misanthropic::openai::{ChatCompletionRequest, ChatCompletionResponse};
use misanthropic::prompt::Message as MMessage;
use misanthropic::Prompt;

/// Handle for a pending Ollama "batch".
///
/// Since Ollama processes synchronously, the handle contains the
/// already-completed results. `poll()` immediately returns `Ready`.
pub struct OllamaPendingHandle {
    results: Vec<WorkResult<MMessage<'static>>>,
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

/// An Ollama endpoint with its URL and discovered models.
#[derive(Debug, Clone)]
pub struct OllamaEndpoint {
    /// Base URL (e.g. `http://localhost:11434`).
    pub url: String,
    /// Models available on this endpoint (populated by [`discover`]).
    pub models: HashSet<String>,
}

impl OllamaEndpoint {
    /// Create an endpoint with no discovered models (for backward compat).
    pub fn new(url: impl Into<String>) -> Self {
        let url = url.into();
        let url = url.trim_end_matches('/').to_string();
        Self {
            url,
            models: HashSet::new(),
        }
    }

    /// Discover available models by querying `GET /api/tags`.
    pub async fn discover(
        http: &reqwest::Client,
        url: &str,
    ) -> anyhow::Result<Self> {
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

        let models: HashSet<String> =
            resp.models.into_iter().map(|m| m.name).collect();

        tracing::info!(
            "Endpoint {url}: {} model(s) [{}]",
            models.len(),
            {
                let mut sorted: Vec<_> = models.iter().cloned().collect();
                sorted.sort();
                sorted.join(", ")
            },
        );

        Ok(Self { url, models })
    }
}

impl Default for OllamaEndpoint {
    fn default() -> Self {
        Self::new("http://localhost:11434")
    }
}

/// Send a single prompt to an Ollama endpoint and return the response.
async fn send_one(
    http: &reqwest::Client,
    endpoint_url: &str,
    prompt: &Prompt<'_>,
    model: &str,
) -> anyhow::Result<MMessage<'static>> {
    let url = format!("{endpoint_url}/v1/chat/completions");
    let start = std::time::Instant::now();

    let mut request = ChatCompletionRequest::from(prompt);
    request.model = model.to_string();

    let response = http
        .post(&url)
        .json(&request)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("Ollama request to {endpoint_url} failed: {e}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("Ollama {endpoint_url} returned {status}: {body}");
    }

    let chat_response: ChatCompletionResponse = response
        .json()
        .await
        .map_err(|e| anyhow::anyhow!("parsing Ollama response: {e}"))?;

    let elapsed = start.elapsed();
    if let Some(usage) = &chat_response.usage {
        tracing::debug!(
            "  [{model}@{endpoint_url}] {}tok prompt, {}tok response, {:.1}s",
            usage.prompt_tokens,
            usage.completion_tokens,
            elapsed.as_secs_f64(),
        );
    }

    chat_response
        .into_message()
        .ok_or_else(|| anyhow::anyhow!("Ollama response contained no message"))
}

/// Process a list of work items against a single endpoint.
///
/// Items are grouped by model and processed sequentially within each
/// group for KV prefix cache reuse.
async fn process_endpoint_items(
    http: &reqwest::Client,
    endpoint_url: &str,
    items: Vec<WorkItem<Prompt<'static>>>,
) -> Vec<WorkResult<MMessage<'static>>> {
    let mut by_model: HashMap<String, Vec<WorkItem<Prompt<'static>>>> =
        HashMap::new();
    for item in items {
        by_model.entry(item.model.clone()).or_default().push(item);
    }

    let mut results = Vec::new();

    for (model, items) in by_model {
        tracing::info!(
            "  [{endpoint_url}] Processing {} items for model '{model}'",
            items.len(),
        );

        for item in items {
            let agent_id = item.agent_id;
            let step = item.step;

            let response = match send_one(http, endpoint_url, &item.prompt, &model).await {
                Ok(msg) => Ok(msg),
                Err(e) => {
                    tracing::warn!(
                        "Ollama request failed for agent {agent_id} at {endpoint_url}: {e}"
                    );
                    Err(BatchError::Transport(e.to_string()))
                }
            };

            results.push(WorkResult {
                agent_id,
                step,
                response,
            });
        }
    }

    results
}

/// Ollama batch backend (single endpoint).
///
/// Processes items sequentially per model for KV prefix cache reuse.
/// For multi-endpoint support, use [`MultiOllamaBatch`] instead.
pub struct OllamaBatch {
    http: reqwest::Client,
    endpoint: OllamaEndpoint,
}

impl OllamaBatch {
    /// Create a new Ollama batch backend targeting a single endpoint.
    pub fn new(endpoint: OllamaEndpoint) -> Self {
        Self {
            http: reqwest::Client::new(),
            endpoint,
        }
    }
}

impl BatchBackend<Prompt<'static>, MMessage<'static>> for OllamaBatch {
    type Handle = OllamaPendingHandle;

    async fn submit(
        &self,
        items: Vec<WorkItem<Prompt<'static>>>,
    ) -> anyhow::Result<Self::Handle> {
        tracing::info!(
            "Ollama batch: {} items to {}",
            items.len(),
            self.endpoint.url,
        );

        let results =
            process_endpoint_items(&self.http, &self.endpoint.url, items).await;

        Ok(OllamaPendingHandle { results })
    }

    async fn poll(
        &self,
        handle: Self::Handle,
    ) -> anyhow::Result<BatchState<MMessage<'static>, Self::Handle>> {
        Ok(BatchState::Ready(handle.results))
    }

    async fn count_tokens(
        &self,
        _prompt: &Prompt<'static>,
    ) -> anyhow::Result<Option<u32>> {
        Ok(None)
    }

    fn backend_name(&self) -> &str {
        "ollama-batch"
    }
}

// ---------------------------------------------------------------------------
// MultiOllamaBatch — routes work across multiple Ollama endpoints
// ---------------------------------------------------------------------------

/// Multi-endpoint Ollama batch backend.
///
/// Routes work items to endpoints based on model availability (discovered
/// via `/api/tags` at startup). Different endpoints process concurrently;
/// same-model items within an endpoint run sequentially for KV cache reuse.
///
/// When a model is available on multiple endpoints, items are routed to the
/// endpoint with the fewest items already assigned in the current batch
/// (greedy load balancing).
pub struct MultiOllamaBatch {
    http: reqwest::Client,
    endpoints: Vec<OllamaEndpoint>,
}

impl MultiOllamaBatch {
    /// Create a new multi-endpoint Ollama batch backend.
    pub fn new(endpoints: Vec<OllamaEndpoint>) -> Self {
        Self {
            http: reqwest::Client::new(),
            endpoints,
        }
    }

    /// Return the URL of an endpoint that has `model`, or `None`.
    pub fn url_for_model(&self, model: &str) -> Option<&str> {
        self.endpoints
            .iter()
            .find(|ep| ep.models.contains(model))
            .map(|ep| ep.url.as_str())
    }

    /// Return a reference to the discovered endpoints.
    pub fn endpoints(&self) -> &[OllamaEndpoint] {
        &self.endpoints
    }
}

impl BatchBackend<Prompt<'static>, MMessage<'static>> for MultiOllamaBatch {
    type Handle = OllamaPendingHandle;

    async fn submit(
        &self,
        items: Vec<WorkItem<Prompt<'static>>>,
    ) -> anyhow::Result<Self::Handle> {
        if self.endpoints.is_empty() {
            anyhow::bail!("No Ollama endpoints configured");
        }

        // Fast path: single endpoint — skip routing overhead.
        if self.endpoints.len() == 1 {
            tracing::info!(
                "Multi-Ollama batch: {} items → {}",
                items.len(),
                self.endpoints[0].url,
            );
            let results = process_endpoint_items(
                &self.http,
                &self.endpoints[0].url,
                items,
            )
            .await;
            return Ok(OllamaPendingHandle { results });
        }

        // Group items by model.
        let mut by_model: HashMap<String, Vec<WorkItem<Prompt<'static>>>> =
            HashMap::new();
        for item in items {
            by_model.entry(item.model.clone()).or_default().push(item);
        }

        // Route each model group to an endpoint.
        // Track how many items each endpoint has been assigned for load balancing.
        let mut endpoint_load: Vec<usize> = vec![0; self.endpoints.len()];
        let mut items_by_endpoint: HashMap<usize, Vec<WorkItem<Prompt<'static>>>> =
            HashMap::new();
        let mut not_found_results: Vec<WorkResult<MMessage<'static>>> = Vec::new();

        for (model, model_items) in by_model {
            // Find all endpoints that have this model.
            let candidates: Vec<usize> = self
                .endpoints
                .iter()
                .enumerate()
                .filter(|(_, ep)| ep.models.contains(&model))
                .map(|(idx, _)| idx)
                .collect();

            if candidates.is_empty() {
                // No endpoint has this model — error each item individually.
                tracing::warn!(
                    "Model '{model}' not available on any endpoint ({} items affected)",
                    model_items.len(),
                );
                for item in model_items {
                    not_found_results.push(WorkResult {
                        agent_id: item.agent_id,
                        step: item.step,
                        response: Err(BatchError::Api {
                            message: format!(
                                "model '{model}' not found on any endpoint"
                            ),
                        }),
                    });
                }
                continue;
            }

            // Pick the candidate with the fewest assigned items.
            let best = *candidates
                .iter()
                .min_by_key(|&&idx| endpoint_load[idx])
                .unwrap();

            endpoint_load[best] += model_items.len();
            items_by_endpoint
                .entry(best)
                .or_default()
                .extend(model_items);
        }

        // Log routing summary.
        for (idx, count) in endpoint_load.iter().enumerate() {
            if *count > 0 {
                tracing::info!(
                    "  {} → {} items",
                    self.endpoints[idx].url,
                    count,
                );
            }
        }

        // Dispatch to endpoints in parallel.
        let mut join_set = tokio::task::JoinSet::new();

        for (idx, items) in items_by_endpoint {
            let http = self.http.clone();
            let url = self.endpoints[idx].url.clone();
            join_set.spawn(async move {
                process_endpoint_items(&http, &url, items).await
            });
        }

        let mut all_results = not_found_results;

        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(results) => all_results.extend(results),
                Err(e) => {
                    tracing::error!("Endpoint task panicked: {e}");
                }
            }
        }

        Ok(OllamaPendingHandle {
            results: all_results,
        })
    }

    async fn poll(
        &self,
        handle: Self::Handle,
    ) -> anyhow::Result<BatchState<MMessage<'static>, Self::Handle>> {
        Ok(BatchState::Ready(handle.results))
    }

    async fn count_tokens(
        &self,
        _prompt: &Prompt<'static>,
    ) -> anyhow::Result<Option<u32>> {
        Ok(None)
    }

    fn backend_name(&self) -> &str {
        "multi-ollama-batch"
    }
}
