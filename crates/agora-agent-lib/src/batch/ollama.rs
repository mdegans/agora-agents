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

use std::collections::HashMap;

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

/// An Ollama endpoint with its URL and available models.
#[derive(Debug, Clone)]
pub struct OllamaEndpoint {
    /// Base URL (e.g. `http://localhost:11434`).
    pub url: String,
}

impl OllamaEndpoint {
    pub fn new(url: impl Into<String>) -> Self {
        let url = url.into();
        let url = url.trim_end_matches('/').to_string();
        Self { url }
    }
}

impl Default for OllamaEndpoint {
    fn default() -> Self {
        Self::new("http://localhost:11434")
    }
}

/// Ollama batch backend.
///
/// Processes items sequentially per model for KV prefix cache reuse,
/// parallel across different model groups via tokio tasks.
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

    /// Send a single prompt to Ollama and return the response message.
    async fn send_one(
        &self,
        prompt: &Prompt<'_>,
        model: &str,
    ) -> anyhow::Result<MMessage<'static>> {
        let url = format!("{}/v1/chat/completions", self.endpoint.url);
        let start = std::time::Instant::now();

        let mut request = ChatCompletionRequest::from(prompt);
        request.model = model.to_string();

        let response = self
            .http
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Ollama request failed: {e}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Ollama returned {status}: {body}");
        }

        let chat_response: ChatCompletionResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("parsing Ollama response: {e}"))?;

        let elapsed = start.elapsed();
        if let Some(usage) = &chat_response.usage {
            tracing::debug!(
                "  [{model}] {}tok prompt, {}tok response, {:.1}s",
                usage.prompt_tokens,
                usage.completion_tokens,
                elapsed.as_secs_f64(),
            );
        }

        chat_response
            .into_message()
            .ok_or_else(|| anyhow::anyhow!("Ollama response contained no message"))
    }
}

impl BatchBackend<Prompt<'static>, MMessage<'static>> for OllamaBatch {
    type Handle = OllamaPendingHandle;

    async fn submit(
        &self,
        items: Vec<WorkItem<Prompt<'static>>>,
    ) -> anyhow::Result<Self::Handle> {
        // Group items by model for sequential processing within each group.
        // The scheduler already groups by model, but a single submit call
        // might still contain multiple models if starvation-promoted items
        // were mixed in.
        let mut by_model: HashMap<String, Vec<WorkItem<Prompt<'static>>>> =
            HashMap::new();
        for item in items {
            by_model
                .entry(item.model.clone())
                .or_default()
                .push(item);
        }

        tracing::info!(
            "Ollama batch: {} model group(s), {} total items",
            by_model.len(),
            by_model.values().map(|v| v.len()).sum::<usize>(),
        );

        // Process each model group. Within a group, requests are sequential
        // (for KV cache reuse). Different groups could run in parallel if
        // we had multiple endpoints, but for a single endpoint we run them
        // sequentially too to avoid contention.
        let mut all_results = Vec::new();

        for (model, items) in by_model {
            tracing::info!(
                "  Processing {} items for model '{model}'",
                items.len(),
            );

            for item in items {
                let agent_id = item.agent_id;
                let step = item.step;

                let response = match self.send_one(&item.prompt, &model).await {
                    Ok(msg) => Ok(msg),
                    Err(e) => {
                        tracing::warn!(
                            "Ollama request failed for agent {agent_id}: {e}"
                        );
                        Err(BatchError::Transport(e.to_string()))
                    }
                };

                all_results.push(WorkResult {
                    agent_id,
                    step,
                    response,
                });
            }
        }

        // Ollama is synchronous — results are immediately ready.
        Ok(OllamaPendingHandle {
            results: all_results,
        })
    }

    async fn poll(
        &self,
        handle: Self::Handle,
    ) -> anyhow::Result<BatchState<MMessage<'static>, Self::Handle>> {
        // Ollama batches complete synchronously during submit(),
        // so poll always returns Ready.
        Ok(BatchState::Ready(handle.results))
    }

    async fn count_tokens(
        &self,
        _prompt: &Prompt<'static>,
    ) -> anyhow::Result<Option<u32>> {
        // TODO: Ollama has a token count API at POST /api/generate with
        // `"stream": false, "raw": true` or similar. For now, return None.
        Ok(None)
    }

    fn backend_name(&self) -> &str {
        "ollama-batch"
    }
}
