//! Anthropic Messages Batch API backend.
//!
//! Implements [`BatchBackend`] using the Anthropic Batch API via
//! [`misanthropic::Client`]. Supports prompt caching — tool definitions
//! and system prefix are cached across agents in the same batch.
//!
//! # Prompt caching
//!
//! The batch API processes items on separate workers that may not share
//! cache. To maximize cache hits:
//! 1. Prime the cache before the first batch by sending one request via
//!    the Messages API with the shared prefix (tools + constitution).
//! 2. Use `CachedPrompt` to ensure the prefix is never mutated — any
//!    change to tools, tool_choice, or system content invalidates the
//!    entire cache prefix.
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
use misanthropic::CachedPrompt;
use misanthropic::batch;
use misanthropic::prompt::Message as MMessage;

/// Handle for a pending Anthropic batch.
///
/// Holds the misanthropic pending batch state and a mapping from
/// batch IDs back to agent IDs and cycle steps.
pub struct AnthropicPendingHandle {
    pending: batch::Pending<CachedPrompt<'static>>,
    /// Maps batch::Id → (AgentId, CycleStep) for result routing.
    id_map: HashMap<batch::Id, (AgentId, CycleStep)>,
}

/// Anthropic Batch API backend.
///
/// Wraps a [`misanthropic::Client`] and submits work items as a batch.
/// Prompt caching is handled at the prompt construction level (cache_control
/// on tool definitions and system prefix blocks).
///
/// Cache priming is done lazily: the first `submit()` call for a given
/// prefix hash sends one message through the non-batch API to warm the
/// prompt cache; subsequent batches sharing the same prefix skip the
/// prime. This matters because each prime call adds ~5s of latency, and
/// re-priming between every phase would push consecutive batches past
/// the 5-minute ephemeral cache TTL and cause the cache to expire right
/// before the next batch runs. The prefix_hash is stable per model
/// across all phases (think / reflect / evolve / survey) so a single
/// prime covers a whole cycle.
pub struct AnthropicBatch {
    client: misanthropic::Client,
    /// Prefix hashes we've already primed this session. Checked on every
    /// `submit()` call — if the batch's prefix_hash is in here, we skip
    /// priming.
    primed_prefixes: std::sync::Mutex<std::collections::HashSet<u64>>,
}

impl AnthropicBatch {
    /// Create a new Anthropic batch backend from an existing client.
    pub fn new(client: misanthropic::Client) -> Self {
        Self {
            client,
            primed_prefixes: std::sync::Mutex::new(std::collections::HashSet::new()),
        }
    }

    /// Create from an API key string.
    pub fn from_key(api_key: String) -> anyhow::Result<Self> {
        let client = misanthropic::Client::new(api_key)
            .map_err(|e| anyhow::anyhow!("invalid API key: {e}"))?;
        Ok(Self::new(client))
    }

    /// Warm Anthropic's prompt cache by submitting a single-item batch
    /// with the given prompt and polling until it completes.
    ///
    /// This is the [Anthropic-recommended prompt caching pattern][docs]:
    /// submit one batch request carrying the shared prefix, wait for it
    /// to finish, then submit the rest. Cache entries written by the
    /// batch API are read by subsequent batch requests with the same
    /// prefix.
    ///
    /// [docs]: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
    async fn prime_via_batch(
        &self,
        prompt: &CachedPrompt<'static>,
    ) -> anyhow::Result<()> {
        let prime_prompt = prompt.clone();
        let mut pending = self
            .client
            .batch([prime_prompt])
            .await
            .map_err(|e| anyhow::anyhow!("prime batch submit failed: {e}"))?;

        let prime_id = pending.meta().id.clone();
        tracing::debug!("Prime batch submitted: id={prime_id}");

        // Poll to completion. Single-item batches complete in a few
        // seconds; we use a short interval to minimize the delay before
        // the main batch can start reading from the cache.
        let prime_timeout = std::time::Duration::from_secs(120);
        let poll_interval = std::time::Duration::from_millis(500);
        let prime_start = std::time::Instant::now();

        let ready = loop {
            match self
                .client
                .batch_poll(pending)
                .await
                .map_err(|e| anyhow::anyhow!("prime batch poll failed: {e}"))?
            {
                batch::Batch::Ready(ready) => break ready,
                batch::Batch::Pending(p) => {
                    if prime_start.elapsed() > prime_timeout {
                        anyhow::bail!(
                            "prime batch {} did not complete within {}s — aborting \
                             prime and letting main batch surface errors",
                            prime_id,
                            prime_timeout.as_secs()
                        );
                    }
                    pending = p;
                    tokio::time::sleep(poll_interval).await;
                }
            }
        };

        // Inspect the single result to surface any item-level error
        // loudly — an error here means the prompt itself is malformed
        // (e.g. empty text block) and the main batch would hit the
        // same error anyway. Fail fast.
        let (_pending, results) = ready.decompose();
        let (_id, result) = results
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("prime batch returned no results"))?;

        match result {
            batch::BatchResult::Ok(msg) => {
                let usage = &msg.usage;
                tracing::info!(
                    "Cache primed via batch API ({:.1}s): {} creation tokens, \
                     {} read tokens, {} input tokens",
                    prime_start.elapsed().as_secs_f64(),
                    usage.cache_creation_input_tokens.unwrap_or(0),
                    usage.cache_read_input_tokens.unwrap_or(0),
                    usage.input_tokens,
                );
                Ok(())
            }
            batch::BatchResult::Error(e) => {
                anyhow::bail!("prime batch item errored: {e}")
            }
            batch::BatchResult::Canceled => {
                anyhow::bail!("prime batch canceled")
            }
            batch::BatchResult::Expired => {
                anyhow::bail!("prime batch expired")
            }
        }
    }
}

impl BatchBackend<CachedPrompt<'static>, MMessage<'static>> for AnthropicBatch {
    type Handle = AnthropicPendingHandle;

    async fn submit(
        &self,
        items: Vec<WorkItem<CachedPrompt<'static>>>,
    ) -> anyhow::Result<Self::Handle> {
        // Prime the prompt cache before submitting the batch — but only
        // once per prefix_hash per process lifetime.
        //
        // Batch items share an identical cacheable prefix (tools +
        // constitution + system) but are processed across different
        // workers that may not share cache. By sending one request first
        // via the Messages API, we write the prefix to Anthropic's cache.
        // Batch items can then read from it at 0.1x cost (stacking with
        // the 0.5x batch discount for 0.05x effective cost on cached
        // tokens).
        //
        // The prefix is stable across all phases and rounds for a given
        // agent model, so we only need to prime once per model per
        // cycle. Priming before every batch would add ~5s of latency
        // each time, pushing consecutive batches past the 5-minute
        // ephemeral cache TTL — exactly the opposite of what we want.
        let prefix_hash = items.first().map(|it| it.prefix_hash);
        let needs_prime = match prefix_hash {
            Some(hash) => {
                let mut set = self
                    .primed_prefixes
                    .lock()
                    .expect("primed_prefixes mutex poisoned");
                set.insert(hash) // insert returns true if the value was new
            }
            None => false,
        };

        if needs_prime && items.len() > 1 {
            tracing::info!(
                "Priming prompt cache for prefix {:016x} (first batch for this prefix; \
                 {} items)",
                prefix_hash.unwrap(),
                items.len(),
            );
            // Submit a single-item batch with the first prompt and poll
            // until it completes, per Anthropic's documented cache-warming
            // pattern: "Send a batch request with just a single request
            // that has this shared prefix and a cache block. As soon as
            // this is complete, submit the rest of the requests."
            //
            // We used to prime via `client.message()` (the non-batch
            // Messages API). That appeared to work — we saw creation
            // tokens on the prime and non-zero hit rates on subsequent
            // batches — but it was relying on undocumented cache sharing
            // between the Messages and Batch APIs. Switching to a batch
            // prime removes that implementation-detail dependency.
            //
            // No 3-item fallback: if the first item errors (e.g. empty
            // text block), the main batch would hit the same error on
            // that item anyway. Surfacing the error loudly and aborting
            // the prime is better than silently paving over a prompt-
            // assembly bug. We do NOT abort the main batch; we roll back
            // the prefix-primed set and continue, letting the real error
            // surface through the normal submit path.
            match self.prime_via_batch(&items[0].prompt).await {
                Ok(()) => {
                    // Success path already logged stats from the polled
                    // result.
                }
                Err(e) => {
                    tracing::error!(
                        "Cache priming via batch API failed: {e}. Continuing \
                         with main batch submission — the real error will \
                         surface through the normal submit path."
                    );
                    if let Some(hash) = prefix_hash {
                        self.primed_prefixes
                            .lock()
                            .expect("primed_prefixes mutex poisoned")
                            .remove(&hash);
                    }
                }
            }
        } else if !needs_prime {
            tracing::debug!(
                "Skipping cache prime for prefix {:016x} — already primed this session",
                prefix_hash.unwrap_or(0),
            );
        }

        // Build the tagged prompts: (batch::Id, CachedPrompt) pairs
        // and track the mapping back to agent IDs.
        //
        // CachedPrompt implements Serialize (delegates to inner Prompt),
        // so tagged_batch() accepts it directly — no into_inner() needed.
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

        tracing::info!("Batch submitted: id={}", pending.meta().id,);

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
                            tracing::warn!("Batch item error for agent {agent_id}: {err}");
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

    async fn count_tokens(&self, _prompt: &CachedPrompt<'static>) -> anyhow::Result<Option<u32>> {
        // TODO: Use misanthropic::Client::count_tokens once available.
        // CachedPrompt derefs to &Prompt which count_tokens accepts.
        Ok(None)
    }

    fn backend_name(&self) -> &str {
        "anthropic-batch"
    }
}
