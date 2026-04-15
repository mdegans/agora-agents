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

/// Maximum consecutive prime-batch failures before we give up priming
/// a prefix and let the main batch submit cold.
///
/// Rationale: if we've timed out waiting for this many primes in a row,
/// the prime batches themselves have almost certainly landed on
/// Anthropic's side — we just didn't see their completion within our
/// poll window. Submitting the main batch at that point is still likely
/// to hit the cache, and even in the worst case (cold cache writes) we
/// bound the wasted time. Without this cap, a persistent prime timeout
/// would retry forever on every subsequent `submit()` call.
const MAX_PRIME_ATTEMPTS: u32 = 3;

/// Per-prefix cache priming state tracked by [`AnthropicBatch`].
#[derive(Debug, Clone, Copy)]
enum PrimeState {
    /// Successfully primed this session. Skip future primes for this prefix.
    Primed,
    /// `n` consecutive prime attempts have failed. Future `submit()` calls
    /// retry as long as `n < MAX_PRIME_ATTEMPTS`.
    Failed(u32),
    /// Hit the failure cap. Skip priming; submit the main batch directly.
    Abandoned,
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
    /// Per-prefix priming state. See [`PrimeState`] for semantics. Keyed
    /// by prefix hash so each distinct `(tools, system)` prefix is tracked
    /// independently — the seed runner has one entry per Anthropic model.
    prime_state: std::sync::Mutex<HashMap<u64, PrimeState>>,
}

impl AnthropicBatch {
    /// Create a new Anthropic batch backend from an existing client.
    pub fn new(client: misanthropic::Client) -> Self {
        Self {
            client,
            prime_state: std::sync::Mutex::new(HashMap::new()),
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
    async fn prime_via_batch(&self, prompt: &CachedPrompt<'static>) -> anyhow::Result<()> {
        let mut pending = self
            .client
            .batch([prompt])
            .await
            .map_err(|e| anyhow::anyhow!("prime batch submit failed: {e}"))?;

        let prime_id = pending.meta().id.clone();
        tracing::debug!("Prime batch submitted: id={prime_id}");

        // Poll to completion. Single-item batches usually finish in a
        // few seconds but there's no SLA — Anthropic guarantees only a
        // 24h ceiling. Cap at 45 minutes: long enough to give Anthropic's
        // typical 5min–1h batch window a fair shot, short enough that a
        // stuck prime doesn't stall the cycle indefinitely. The caller's
        // retry cap (`MAX_PRIME_ATTEMPTS`) bounds total wall-clock spent
        // priming a given prefix across successive `submit()` calls.
        let prime_timeout = std::time::Duration::from_secs(45 * 60);
        let poll_interval = std::time::Duration::from_secs(5);
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
        // cycle. Priming before every batch would add several minutes of
        // latency each time, pushing consecutive batches past the 5-minute
        // ephemeral cache TTL — exactly the opposite of what we want.
        //
        // If a prime batch has failed `MAX_PRIME_ATTEMPTS` times in a row
        // for this prefix, we transition to `Abandoned` and submit the
        // main batch without priming. See [`PrimeState`] for the full
        // state machine.
        let prefix_hash = items.first().map(|it| it.prefix_hash);
        let should_prime = if let Some(hash) = prefix_hash {
            let mut state = self.prime_state.lock().expect("prime_state mutex poisoned");
            match state.get(&hash).copied() {
                None => true,
                Some(PrimeState::Failed(n)) if n < MAX_PRIME_ATTEMPTS => true,
                Some(PrimeState::Failed(n)) => {
                    // Hit the cap on this call. Transition to Abandoned
                    // and log loudly — subsequent batches will see
                    // `Abandoned` and skip priming silently.
                    tracing::warn!(
                        "Cache priming abandoned for prefix {hash:016x} after {n} \
                         consecutive failures. Submitting main batch without prime; \
                         if earlier prime batches eventually landed on Anthropic's \
                         side, the cache may still serve hits."
                    );
                    state.insert(hash, PrimeState::Abandoned);
                    false
                }
                Some(PrimeState::Primed) => {
                    tracing::debug!(
                        "Skipping cache prime for prefix {hash:016x} — already primed this session"
                    );
                    false
                }
                Some(PrimeState::Abandoned) => {
                    tracing::debug!(
                        "Skipping cache prime for prefix {hash:016x} — abandoned after \
                         {MAX_PRIME_ATTEMPTS} consecutive failures"
                    );
                    false
                }
            }
        } else {
            false
        };

        if should_prime && items.len() > 1 {
            let hash = prefix_hash.expect("should_prime implies prefix_hash is Some");
            tracing::info!(
                "Priming prompt cache for prefix {hash:016x} ({} items in this batch)",
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
            match self.prime_via_batch(&items[0].prompt).await {
                Ok(()) => {
                    let mut state = self.prime_state.lock().expect("prime_state mutex poisoned");
                    state.insert(hash, PrimeState::Primed);
                }
                Err(e) => {
                    tracing::error!("Cache priming via batch API failed: {e}.");
                    // Increment the failure counter. If a concurrent
                    // submit() happened to land a successful prime for
                    // the same prefix while this one was failing, trust
                    // the success and do not overwrite `Primed`.
                    let mut state = self.prime_state.lock().expect("prime_state mutex poisoned");
                    let next = match state.get(&hash).copied() {
                        Some(PrimeState::Primed) => None,
                        Some(PrimeState::Failed(n)) => Some(PrimeState::Failed(n + 1)),
                        _ => Some(PrimeState::Failed(1)),
                    };
                    if let Some(new_state) = next {
                        state.insert(hash, new_state);
                    }
                    return Err(e); // Fail fast — scheduler will retry.
                }
            }
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
                // Aggregate usage across all succeeded items so we can
                // compute and log the cache hit rate for this batch.
                // `Usage: Default + AddAssign + Copy` so summation is trivial.
                let mut total_usage = misanthropic::response::Usage::default();
                let mut ok_count = 0usize;

                // Decompose to get owned results
                let (_pending, result_map) = ready.decompose();

                for (bid, result) in result_map {
                    let Some(&(agent_id, step)) = id_map.get(&bid) else {
                        tracing::warn!("Unknown batch ID in results: {bid}");
                        continue;
                    };

                    let response = match result {
                        batch::BatchResult::Ok(response_message) => {
                            total_usage += response_message.usage;
                            ok_count += 1;
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

                // Cache hit rate = read / (uncached-input + created + read),
                // where the denominator is the total input tokens that
                // went through the pipeline (cacheable or not). A high
                // hit rate means the batch is mostly reading prefix
                // blocks written by a prior prime or earlier batch.
                //
                // Uses u128 to avoid any chance of overflow when a batch
                // happens to carry very large contexts — Usage fields
                // are already u64 but the sum of 30 items' contributions
                // can creep up.
                let read = total_usage.cache_read_input_tokens.unwrap_or(0) as u128;
                let created = total_usage.cache_creation_input_tokens.unwrap_or(0) as u128;
                let uncached = total_usage.input_tokens as u128;
                let total_input = read + created + uncached;
                if total_input > 0 {
                    let hit_pct = (read as f64 / total_input as f64) * 100.0;
                    tracing::info!(
                        "Batch {batch_id} cache: {hit_pct:.1}% hit ({ok_count} items, \
                         {read} read, {created} created, {uncached} uncached, \
                         {out} out)",
                        out = total_usage.output_tokens,
                    );
                } else {
                    tracing::debug!(
                        "Batch {batch_id} complete but no input tokens reported \
                         ({ok_count} items)"
                    );
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
