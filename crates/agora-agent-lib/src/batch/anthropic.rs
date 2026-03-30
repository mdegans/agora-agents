//! Anthropic Messages Batch API backend.
//!
//! Implements [`BatchBackend`] using the Anthropic Batch API via
//! [`misanthropic::Client`]. Supports prompt caching — tool definitions
//! and system prefix are cached across agents in the same batch.
//!
//! # Cost optimization
//!
//! The Batch API provides a 50% discount on input tokens. Combined with
//! prompt caching (0.1x cost for cache hits), agents sharing the same
//! system prefix in a batch can see up to 95% cost reduction on cached
//! tokens.

use std::collections::HashMap;

use agora_agentkit::ids::AgentId;
use agora_agentkit::scheduler::{
    BatchBackend, BatchError, BatchState, CycleStep, WorkItem, WorkResult,
};
use misanthropic::batch;
use misanthropic::prompt::Message as MMessage;
use misanthropic::Prompt;

/// Handle for a pending Anthropic batch.
///
/// Holds the misanthropic pending batch state and a mapping from
/// batch IDs back to agent IDs and cycle steps.
pub struct AnthropicPendingHandle {
    pending: batch::Pending<'static>,
    /// Maps batch::Id → (AgentId, CycleStep) for result routing.
    id_map: HashMap<batch::Id, (AgentId, CycleStep)>,
}

/// Anthropic Batch API backend.
///
/// Wraps a [`misanthropic::Client`] and submits work items as a batch.
/// Prompt caching is handled at the prompt construction level (cache_control
/// on tool definitions and system prefix blocks).
pub struct AnthropicBatch {
    client: misanthropic::Client,
}

impl AnthropicBatch {
    /// Create a new Anthropic batch backend from an existing client.
    pub fn new(client: misanthropic::Client) -> Self {
        Self { client }
    }

    /// Create from an API key string.
    pub fn from_key(api_key: String) -> anyhow::Result<Self> {
        let client = misanthropic::Client::new(api_key)
            .map_err(|e| anyhow::anyhow!("invalid API key: {e}"))?;
        Ok(Self { client })
    }
}

impl BatchBackend<Prompt<'static>, MMessage<'static>> for AnthropicBatch {
    type Handle = AnthropicPendingHandle;

    async fn submit(
        &self,
        items: Vec<WorkItem<Prompt<'static>>>,
    ) -> anyhow::Result<Self::Handle> {
        // Build the tagged prompts: (batch::Id, Prompt) pairs
        // and track the mapping back to agent IDs
        let mut id_map = HashMap::new();
        let mut tagged_prompts = Vec::with_capacity(items.len());

        for item in items {
            let batch_id = batch::Id::default();
            id_map.insert(batch_id, (item.agent_id, item.step));
            tagged_prompts.push((batch_id, item.prompt));
        }

        tracing::info!(
            "Submitting Anthropic batch with {} prompts",
            tagged_prompts.len()
        );

        let pending = self.client.tagged_batch(tagged_prompts).await?;

        tracing::info!(
            "Batch submitted: id={}",
            pending.meta().id,
        );

        Ok(AnthropicPendingHandle { pending, id_map })
    }

    async fn poll(
        &self,
        handle: Self::Handle,
    ) -> anyhow::Result<BatchState<MMessage<'static>, Self::Handle>> {
        let AnthropicPendingHandle { pending, id_map } = handle;

        let batch_id = pending.meta().id.clone();
        let batch_result = self.client.batch_poll(pending).await?;

        match batch_result {
            batch::Batch::Pending(pending) => {
                let stats = &pending.meta().stats;
                tracing::debug!(
                    "Batch {batch_id} still processing: {} succeeded, {} processing, {} errored",
                    stats.succeeded,
                    stats.processing,
                    stats.errored,
                );
                Ok(BatchState::Pending(AnthropicPendingHandle {
                    pending,
                    id_map,
                }))
            }
            batch::Batch::Ready(ready) => {
                tracing::info!("Batch {batch_id} complete");

                let mut results = Vec::new();

                // Decompose to get owned results
                let (_pending, result_map) = ready.decompose();

                for (bid, result) in result_map {
                    let Some(&(agent_id, step)) = id_map.get(&bid) else {
                        tracing::warn!("Unknown batch ID in results: {bid}");
                        continue;
                    };

                    let response = match result {
                        batch::BatchResult::Ok(response_message) => {
                            // response::Message.inner -> prompt::AssistantMessage
                            // -> Into<prompt::Message>
                            let msg: MMessage<'_> = response_message.inner.into();
                            Ok(msg.into_static())
                        }
                        batch::BatchResult::Error(err) => {
                            tracing::warn!(
                                "Batch item error for agent {agent_id}: {err}"
                            );
                            Err(BatchError::Api {
                                message: err.to_string(),
                            })
                        }
                        batch::BatchResult::Canceled => {
                            tracing::warn!("Batch item canceled for agent {agent_id}");
                            Err(BatchError::Canceled)
                        }
                        batch::BatchResult::Expired => {
                            tracing::warn!("Batch item expired for agent {agent_id}");
                            Err(BatchError::Expired)
                        }
                    };

                    results.push(WorkResult {
                        agent_id,
                        step,
                        response,
                    });
                }

                Ok(BatchState::Ready(results))
            }
        }
    }

    async fn count_tokens(
        &self,
        _prompt: &Prompt<'static>,
    ) -> anyhow::Result<Option<u32>> {
        // TODO: Use misanthropic::Client::count_tokens once it's on the dev branch.
        // For now, return None — the scheduler will skip token-based grouping.
        Ok(None)
    }

    fn backend_name(&self) -> &str {
        "anthropic-batch"
    }
}
