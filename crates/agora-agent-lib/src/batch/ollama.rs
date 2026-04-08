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
//! Uses Ollama's Anthropic-compatible `/v1/messages` endpoint via
//! [`misanthropic::Client`] with a custom base URL.
//!
//! [`MultiOllamaBatch`] extends this to multiple endpoints: it discovers
//! available models via `/api/tags` at startup and routes work items to the
//! right endpoint. Different endpoints process concurrently.

use std::collections::{HashMap, HashSet};

use agora_agentkit::scheduler::{BatchBackend, BatchError, BatchState, WorkItem, WorkResult};
use misanthropic::Prompt;
use misanthropic::prompt::Message as MMessage;

use crate::llm::ollama::{create_ollama_client, send_with_nudge};

/// Handle for a pending Ollama "batch".
///
/// Since Ollama processes synchronously, the handle contains the
/// already-completed results. `poll()` immediately returns `Ready`.
pub struct OllamaPendingHandle {
    results: Vec<WorkResult<MMessage<'static>>>,
}

impl OllamaPendingHandle {
    /// Create an empty handle (no items to process).
    pub fn empty() -> Self {
        Self {
            results: Vec::new(),
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
}

impl OllamaEndpoint {
    /// Send a single prompt to this endpoint and return the response.
    ///
    /// This is the public API for the sequential per-agent pipeline where
    /// the scheduler drives one request at a time for KV cache reuse.
    pub async fn send(
        &self,
        prompt: &Prompt<'_>,
        model: &str,
    ) -> anyhow::Result<MMessage<'static>> {
        send_one(&self.client, &self.url, prompt, model).await
    }
}

impl Default for OllamaEndpoint {
    fn default() -> Self {
        Self::new("http://localhost:11434").expect("default Ollama URL should be valid")
    }
}

/// Send a single prompt to an Ollama endpoint and return the response.
async fn send_one(
    client: &misanthropic::Client,
    endpoint_url: &str,
    prompt: &Prompt<'_>,
    model: &str,
) -> anyhow::Result<MMessage<'static>> {
    let start = std::time::Instant::now();

    let resp = send_with_nudge(client, prompt).await?;

    let elapsed = start.elapsed();
    if let Some(ref usage) = resp.usage {
        tracing::debug!(
            "  [{model}@{endpoint_url}] {:.1}s, {}tok in, {}tok out",
            elapsed.as_secs_f64(),
            usage.input_tokens,
            usage.output_tokens,
        );
    } else {
        tracing::debug!("  [{model}@{endpoint_url}] {:.1}s", elapsed.as_secs_f64());
    }

    Ok(resp.message)
}

/// Process a list of work items against a single endpoint.
///
/// Items run sequentially — the scheduler handles model interleaving at
/// the batch level. Within a batch, all items are the same model for KV
/// prefix cache reuse.
async fn process_endpoint_items(
    client: &misanthropic::Client,
    endpoint_url: &str,
    items: Vec<WorkItem<Prompt<'static>>>,
) -> Vec<WorkResult<MMessage<'static>>> {
    let mut results = Vec::new();

    for item in &items {
        let response = match send_one(client, endpoint_url, &item.prompt, &item.model).await {
            Ok(msg) => Ok(msg),
            Err(e) => {
                tracing::warn!(
                    "Ollama request failed for agent {} at {endpoint_url}: {e}",
                    item.agent_id,
                );
                Err(BatchError::Transport(e.to_string()))
            }
        };

        results.push(WorkResult {
            agent_id: item.agent_id,
            step: item.step,
            response,
        });
    }

    results
}

/// Ollama batch backend (single endpoint).
///
/// Processes items sequentially per model for KV prefix cache reuse.
/// For multi-endpoint support, use [`MultiOllamaBatch`] instead.
pub struct OllamaBatch {
    endpoint: OllamaEndpoint,
}

impl OllamaBatch {
    /// Create a new Ollama batch backend targeting a single endpoint.
    pub fn new(endpoint: OllamaEndpoint) -> Self {
        Self { endpoint }
    }
}

impl BatchBackend<Prompt<'static>, MMessage<'static>> for OllamaBatch {
    type Handle = OllamaPendingHandle;

    async fn submit(&self, items: Vec<WorkItem<Prompt<'static>>>) -> anyhow::Result<Self::Handle> {
        tracing::info!(
            "Ollama batch: {} items to {}",
            items.len(),
            self.endpoint.url,
        );

        let results =
            process_endpoint_items(&self.endpoint.client, &self.endpoint.url, items).await;

        Ok(OllamaPendingHandle { results })
    }

    async fn poll(
        &self,
        handle: Self::Handle,
    ) -> anyhow::Result<BatchState<MMessage<'static>, Self::Handle>> {
        Ok(BatchState::Ready(handle.results))
    }

    async fn count_tokens(&self, _prompt: &Prompt<'static>) -> anyhow::Result<Option<u32>> {
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
    endpoints: Vec<OllamaEndpoint>,
}

impl MultiOllamaBatch {
    /// Create a new multi-endpoint Ollama batch backend.
    pub fn new(endpoints: Vec<OllamaEndpoint>) -> Self {
        Self { endpoints }
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

    async fn submit(&self, items: Vec<WorkItem<Prompt<'static>>>) -> anyhow::Result<Self::Handle> {
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
            let results =
                process_endpoint_items(&self.endpoints[0].client, &self.endpoints[0].url, items)
                    .await;
            return Ok(OllamaPendingHandle { results });
        }

        // Group items by model.
        let mut by_model: HashMap<String, Vec<WorkItem<Prompt<'static>>>> = HashMap::new();
        for item in items {
            by_model.entry(item.model.clone()).or_default().push(item);
        }

        // Route each model group to an endpoint.
        // Track how many items each endpoint has been assigned for load balancing.
        let mut endpoint_load: Vec<usize> = vec![0; self.endpoints.len()];
        let mut items_by_endpoint: HashMap<usize, Vec<WorkItem<Prompt<'static>>>> = HashMap::new();
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
                            message: format!("model '{model}' not found on any endpoint"),
                        }),
                    });
                }
                continue;
            }

            // Distribute items across all candidate endpoints round-robin,
            // starting from the least-loaded. This keeps both GPUs busy
            // when a model is available on multiple endpoints.
            if candidates.len() == 1 {
                let best = candidates[0];
                endpoint_load[best] += model_items.len();
                items_by_endpoint
                    .entry(best)
                    .or_default()
                    .extend(model_items);
            } else {
                // Sort candidates by current load (least loaded first)
                let mut sorted_candidates = candidates.clone();
                sorted_candidates.sort_by_key(|&idx| endpoint_load[idx]);

                for (i, item) in model_items.into_iter().enumerate() {
                    let ep = sorted_candidates[i % sorted_candidates.len()];
                    endpoint_load[ep] += 1;
                    items_by_endpoint.entry(ep).or_default().push(item);
                }
            }
        }

        // Log routing summary.
        for (idx, count) in endpoint_load.iter().enumerate() {
            if *count > 0 {
                tracing::info!("  {} → {} items", self.endpoints[idx].url, count,);
            }
        }

        // Dispatch to endpoints in parallel.
        let mut join_set = tokio::task::JoinSet::new();

        for (idx, items) in items_by_endpoint {
            let client = self.endpoints[idx].client.clone();
            let url = self.endpoints[idx].url.clone();
            join_set.spawn(async move { process_endpoint_items(&client, &url, items).await });
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

    async fn count_tokens(&self, _prompt: &Prompt<'static>) -> anyhow::Result<Option<u32>> {
        Ok(None)
    }

    fn backend_name(&self) -> &str {
        "multi-ollama-batch"
    }
}
