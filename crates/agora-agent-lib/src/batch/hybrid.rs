//! Hybrid batch backend that routes work items across Ollama and Anthropic.
//!
//! Models available on any Ollama endpoint are routed there; all others
//! are routed to the Anthropic Batch API. This enables mixed seed runs
//! where local open-source models and Anthropic models coexist.

use std::collections::HashSet;

use agora_agentkit::scheduler::{BatchBackend, BatchState, WorkItem, WorkResult};
use misanthropic::Prompt;
use misanthropic::prompt::Message as MMessage;

use super::anthropic::{AnthropicBatch, AnthropicPendingHandle};
use super::ollama::{MultiOllamaBatch, OllamaPendingHandle};

/// Pending handle for a hybrid batch.
///
/// Holds optional sub-handles for each backend (a batch may only contain
/// items for one backend) plus any results already collected.
pub struct HybridPendingHandle {
    /// Anthropic sub-handle, if any items were routed there.
    anthropic: Option<AnthropicPendingHandle>,
    /// Ollama results (ready immediately after submit).
    ollama_results: Vec<WorkResult<MMessage<'static>>>,
    /// Anthropic results collected so far.
    anthropic_results: Vec<WorkResult<MMessage<'static>>>,
}

/// Hybrid batch backend routing between Ollama and Anthropic.
///
/// Models present on any Ollama endpoint are routed to
/// [`MultiOllamaBatch`]; all others go to [`AnthropicBatch`].
pub struct HybridBatch {
    anthropic: AnthropicBatch,
    ollama: MultiOllamaBatch,
    /// Set of model names available on Ollama endpoints.
    ollama_models: HashSet<String>,
}

impl HybridBatch {
    /// Create a hybrid backend.
    ///
    /// `ollama_models` is the union of all models discovered across Ollama
    /// endpoints. Any model NOT in this set routes to Anthropic.
    pub fn new(
        anthropic: AnthropicBatch,
        ollama: MultiOllamaBatch,
        ollama_models: HashSet<String>,
    ) -> Self {
        Self {
            anthropic,
            ollama,
            ollama_models,
        }
    }

    /// Returns true if the given model routes to Ollama.
    pub fn is_ollama_model(&self, model: &str) -> bool {
        self.ollama_models.contains(model)
    }
}

impl BatchBackend<Prompt<'static>, MMessage<'static>> for HybridBatch {
    type Handle = HybridPendingHandle;

    async fn submit(&self, items: Vec<WorkItem<Prompt<'static>>>) -> anyhow::Result<Self::Handle> {
        // Partition items by backend.
        let mut ollama_items = Vec::new();
        let mut anthropic_items = Vec::new();

        for item in items {
            if self.ollama_models.contains(&item.model) {
                ollama_items.push(item);
            } else {
                anthropic_items.push(item);
            }
        }

        let ollama_count = ollama_items.len();
        let anthropic_count = anthropic_items.len();

        if ollama_count > 0 && anthropic_count > 0 {
            tracing::info!("Hybrid batch: {ollama_count} → ollama, {anthropic_count} → anthropic");
        }

        // Submit to both backends in parallel.
        let (ollama_result, anthropic_result) = tokio::join!(
            async {
                if ollama_items.is_empty() {
                    Ok(OllamaPendingHandle::empty())
                } else {
                    self.ollama.submit(ollama_items).await
                }
            },
            async {
                if anthropic_items.is_empty() {
                    Ok(None)
                } else {
                    self.anthropic.submit(anthropic_items).await.map(Some)
                }
            },
        );

        // Ollama results are ready immediately.
        let ollama_handle = ollama_result?;
        let ollama_results = match self.ollama.poll(ollama_handle).await? {
            BatchState::Ready(results) => results,
            BatchState::Pending(_) => unreachable!("OllamaBatch poll is always Ready"),
        };

        let anthropic_handle = anthropic_result?;

        Ok(HybridPendingHandle {
            anthropic: anthropic_handle,
            ollama_results,
            anthropic_results: Vec::new(),
        })
    }

    async fn poll(
        &self,
        handle: Self::Handle,
    ) -> anyhow::Result<BatchState<MMessage<'static>, Self::Handle>> {
        let HybridPendingHandle {
            anthropic,
            ollama_results,
            mut anthropic_results,
        } = handle;

        match anthropic {
            // No Anthropic items — everything is ready.
            None => {
                let mut all = ollama_results;
                all.extend(anthropic_results);
                Ok(BatchState::Ready(all))
            }
            Some(anthropic_handle) => {
                // Poll the Anthropic batch.
                match self.anthropic.poll(anthropic_handle).await? {
                    BatchState::Ready(results) => {
                        anthropic_results.extend(results);
                        let mut all = ollama_results;
                        all.extend(anthropic_results);
                        Ok(BatchState::Ready(all))
                    }
                    BatchState::Pending(next) => Ok(BatchState::Pending(HybridPendingHandle {
                        anthropic: Some(next),
                        ollama_results,
                        anthropic_results,
                    })),
                }
            }
        }
    }

    async fn count_tokens(&self, _prompt: &Prompt<'static>) -> anyhow::Result<Option<u32>> {
        Ok(None)
    }

    fn backend_name(&self) -> &str {
        "hybrid-batch"
    }
}
