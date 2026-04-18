//! Anthropic Messages Batch API backend.
//!
//! Implements [`BatchBackend`] using the Anthropic Batch API via
//! [`misanthropic::Client`]. Supports prompt caching — tool definitions
//! and system prefix are cached across agents in the same batch and
//! across subsequent batches within the same session via eager priming.
//!
//! # Eager prompt caching
//!
//! The batch API processes items on separate workers that may not share
//! cache. To maximize cache hits the seed runner primes Anthropic's
//! prompt cache **at the start of each cycle** per unique model, before
//! that cycle's real batch submissions, via
//! [`AnthropicBatch::prime_prefixes`]. Each prime is a single-item batch
//! carrying only the cacheable `tools + system` prefix (built by
//! `agora_seed::prompt::build_base_prompt`) with a 1h cache breakpoint.
//! Subsequent real batches for the same model read the warmed cache at
//! 0.1x base input cost. Per-cycle invocation is cheap: a fresh entry
//! within [`PRIME_FRESHNESS`] short-circuits without hitting the API,
//! so re-calling every cycle is a no-op unless the cache has actually
//! aged out (cycles can take hours) or a prior prime was abandoned and
//! the infra may have recovered.
//!
//! Key properties of this design:
//!
//! - **No in-submit priming.** [`BatchBackend::submit`] is purely
//!   mechanical and never touches [`PrimeState`]. This decouples real-
//!   batch submission from priming: a stuck prime cannot leave a real-
//!   batch prompt in a half-mutated state, which was the failure mode
//!   that motivated this design (see agora-agents PR for eager priming).
//! - **Bounded retry.** [`AnthropicBatch::prime_prefix`] retries a
//!   failed prime up to [`MAX_PRIME_ATTEMPTS`] times within one call.
//!   On the cap it transitions the prefix to [`PrimeState::Abandoned`]
//!   and returns. Real batches for an abandoned prefix still submit —
//!   they just pay uncached input cost, which is the desired soft-
//!   degrade.
//! - **Freshness-based re-priming.** `Primed` and `Abandoned` entries
//!   carry an `Instant` and age out after [`PRIME_FRESHNESS`]. Stale
//!   entries fall through on the next `prime_prefix` call: stale
//!   `Primed` re-warms the cache before the 1h TTL expires; stale
//!   `Abandoned` retries prime in case the infra hiccup that caused
//!   the original abandonment has resolved.
//! - **1h TTL everywhere.** The seed-runner prompt builder places 1h
//!   cache breakpoints so a single prime stays warm across every phase
//!   (perceive → think_act → reflect → evolve → survey) within a cycle.
//!   `PRIME_FRESHNESS` is sized under that TTL with a safety margin.
//!
//! # Cost optimization
//!
//! The Batch API provides a 50% discount on input tokens. Combined with
//! prompt caching (0.1x cost for cache hits), agents sharing the same
//! system prefix in a batch can see up to 95% cost reduction on cached
//! tokens. 1h cache writes cost 2x base input (vs 1.25x for 5min), but
//! at observed hit rates the read savings swamp the higher write cost
//! as long as any reads land.

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

/// How long a terminal prime outcome (Primed or Abandoned) stays
/// "fresh" before [`AnthropicBatch::prime_prefix`] will redo the work.
///
/// We place 1h `cache_1h` breakpoints on our prompts, so Anthropic's
/// cache TTL for these entries is 1h. The freshness window is set
/// under the TTL with a 10-min safety margin so the next real batch
/// after a re-prime still reads a warm cache even if the re-prime
/// itself takes a few minutes to land.
///
/// The same window applies to `Abandoned`: a batch-API infra hiccup
/// severe enough to burn through 3 × 45min timeouts is almost always
/// transient (observed 2026-04-15: Anthropic was broadly slow that
/// afternoon, back to normal the next day). Retrying on the next
/// cycle after the window elapses lets a long multi-cycle run
/// recover the cache-hit benefit once infra is healthy again.
const PRIME_FRESHNESS: std::time::Duration = std::time::Duration::from_secs(50 * 60);

/// Per-prefix cache priming state tracked by [`AnthropicBatch`].
#[derive(Debug, Clone, Copy)]
enum PrimeState {
    /// Successfully primed at this instant. Short-circuit future
    /// `prime_prefix` calls until [`PRIME_FRESHNESS`] has elapsed, after
    /// which the cache is considered potentially stale and we re-prime.
    Primed { at: std::time::Instant },
    /// `n` consecutive prime attempts have failed within a single
    /// `prime_prefix` call. Transient — never observed between calls
    /// because the retry loop either transitions to `Primed` or
    /// `Abandoned` before returning.
    Failed(u32),
    /// Hit the failure cap at this instant. Short-circuit future calls
    /// until [`PRIME_FRESHNESS`] has elapsed, after which we retry in
    /// case the underlying infra issue has resolved.
    Abandoned { at: std::time::Instant },
}

/// Anthropic Batch API backend.
///
/// Wraps a [`misanthropic::Client`] and submits work items as a batch.
/// Prompt caching is handled at the prompt construction level
/// (`cache_1h()` on system blocks and tool definitions — see
/// `agora_seed::prompt::build_base_prompt`).
///
/// Cache priming is eager: the caller invokes [`Self::prime_prefixes`]
/// at the start of each cycle before any real batch runs, and by the
/// time [`BatchBackend::submit`] is called the prefix is either primed
/// or [`PrimeState::Abandoned`]. The prefix hash is stable per model
/// across every phase (think / reflect / evolve / survey) within a
/// cycle, and stale entries past [`PRIME_FRESHNESS`] are automatically
/// re-primed on the next cycle to ride the 1h cache TTL.
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
    /// with the given prompt and polling until it completes. Returns the
    /// `Usage` record from the batch result so callers can log
    /// `cache_creation_input_tokens` on success.
    ///
    /// This is the [Anthropic-recommended prompt caching pattern][docs]:
    /// submit one batch request carrying the shared prefix, wait for it
    /// to finish, then submit the rest. Cache entries written by the
    /// batch API are read by subsequent batch requests with the same
    /// prefix.
    ///
    /// This function is the single-attempt inner core — retries and
    /// state tracking live in [`Self::prime_prefix`].
    ///
    /// [docs]: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
    async fn prime_via_batch(
        &self,
        model: &str,
        prompt: &CachedPrompt<'static>,
    ) -> anyhow::Result<misanthropic::response::Usage> {
        let mut pending = self
            .client
            .batch([prompt])
            .await
            .map_err(|e| anyhow::anyhow!("prime batch submit failed for model {model}: {e}"))?;

        let prime_id = pending.meta().id.clone();
        tracing::debug!(%model, batch_id = %prime_id, "prime batch submitted");

        // Poll to completion. Single-item batches usually finish in a
        // few seconds but there's no SLA — Anthropic guarantees only a
        // 24h ceiling. Cap at 45 minutes: long enough to give Anthropic's
        // typical 5min–1h batch window a fair shot, short enough that a
        // stuck prime doesn't stall session startup indefinitely. The
        // caller's retry cap (`MAX_PRIME_ATTEMPTS`) bounds total wall-
        // clock spent priming a given prefix.
        let prime_timeout = std::time::Duration::from_secs(45 * 60);
        let poll_interval = std::time::Duration::from_secs(5);
        let prime_start = std::time::Instant::now();

        let ready =
            loop {
                match self.client.batch_poll(pending).await.map_err(|e| {
                    anyhow::anyhow!("prime batch poll failed for model {model}: {e}")
                })? {
                    batch::Batch::Ready(ready) => break ready,
                    batch::Batch::Pending(p) => {
                        if prime_start.elapsed() > prime_timeout {
                            anyhow::bail!(
                                "prime batch {} for model {} did not complete within {}s",
                                prime_id,
                                model,
                                prime_timeout.as_secs()
                            );
                        }
                        pending = p;
                        tokio::time::sleep(poll_interval).await;
                    }
                }
            };

        let (_pending, results) = ready.decompose();
        let (_id, result) = results
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("prime batch for model {model} returned no results"))?;

        match result {
            batch::BatchResult::Ok(msg) => Ok(msg.usage),
            batch::BatchResult::Error(e) => {
                anyhow::bail!("prime batch item for model {model} errored: {e}")
            }
            batch::BatchResult::Canceled => {
                anyhow::bail!("prime batch for model {model} canceled")
            }
            batch::BatchResult::Expired => {
                anyhow::bail!("prime batch for model {model} expired")
            }
        }
    }

    /// Prime the cache for a single prefix, retrying up to
    /// [`MAX_PRIME_ATTEMPTS`] times on failure.
    ///
    /// - Returns `true` if the prefix is primed (either by this call or
    ///   a previous successful call within [`PRIME_FRESHNESS`]).
    /// - Returns `false` if the prefix is in the `Abandoned` state and
    ///   was abandoned within [`PRIME_FRESHNESS`] (recent failure, don't
    ///   retry yet), or if this call just exhausted its retry cap. In
    ///   either case the caller should proceed without a prime — real
    ///   batches still submit, they just pay uncached input cost.
    ///
    /// A stale `Primed` or `Abandoned` entry (older than
    /// [`PRIME_FRESHNESS`]) falls through into the retry loop and is
    /// re-primed. This lets long multi-cycle runs re-warm the cache
    /// after the 1h TTL expires, and lets transient Anthropic infra
    /// hiccups resolve without permanently locking the prefix out.
    ///
    /// Never panics and never returns an error. Soft failure only.
    ///
    /// Log messages emitted:
    /// - info on successful prime (with creation-token count from the Usage)
    /// - info when a stale Primed/Abandoned entry falls through to re-prime
    /// - warn per failed attempt with attempt count
    /// - warn on terminal abandonment after the cap
    /// - debug on "already primed" / "already abandoned" short-circuit
    pub async fn prime_prefix(
        &self,
        prefix_hash: u64,
        model: &str,
        prompt: &CachedPrompt<'static>,
    ) -> bool {
        // Short-circuit if the prefix is in a fresh terminal state.
        // Stale entries (older than PRIME_FRESHNESS) fall through and
        // re-enter the retry loop below.
        {
            let state = self.prime_state.lock().expect("prime_state mutex poisoned");
            match state.get(&prefix_hash).copied() {
                Some(PrimeState::Primed { at }) if at.elapsed() < PRIME_FRESHNESS => {
                    tracing::debug!(
                        %model,
                        age_s = at.elapsed().as_secs(),
                        "prefix primed recently, skipping"
                    );
                    return true;
                }
                Some(PrimeState::Abandoned { at }) if at.elapsed() < PRIME_FRESHNESS => {
                    tracing::debug!(
                        %model,
                        age_s = at.elapsed().as_secs(),
                        "prefix recently abandoned after {MAX_PRIME_ATTEMPTS} failures, skipping"
                    );
                    return false;
                }
                Some(PrimeState::Primed { at }) => {
                    tracing::info!(
                        %model,
                        age_s = at.elapsed().as_secs(),
                        "primed entry is stale (past cache TTL), re-priming"
                    );
                }
                Some(PrimeState::Abandoned { at }) => {
                    tracing::info!(
                        %model,
                        age_s = at.elapsed().as_secs(),
                        "previously abandoned prefix is stale, retrying prime (infra may have recovered)"
                    );
                }
                _ => {}
            }
        }

        // Retry loop. Each iteration is one `prime_via_batch` attempt.
        // On success: record `Primed`, return true.
        // On failure: increment the failure counter, loop if below the
        // cap, else transition to `Abandoned` and return false.
        loop {
            let attempt_started = std::time::Instant::now();
            match self.prime_via_batch(model, prompt).await {
                Ok(usage) => {
                    let mut state = self.prime_state.lock().expect("prime_state mutex poisoned");
                    state.insert(
                        prefix_hash,
                        PrimeState::Primed {
                            at: std::time::Instant::now(),
                        },
                    );
                    tracing::info!(
                        %model,
                        elapsed_s = attempt_started.elapsed().as_secs_f64(),
                        cache_creation = usage.cache_creation_input_tokens.unwrap_or(0),
                        cache_read = usage.cache_read_input_tokens.unwrap_or(0),
                        input_tokens = usage.input_tokens,
                        "cache primed via eager batch"
                    );
                    return true;
                }
                Err(e) => {
                    let mut state = self.prime_state.lock().expect("prime_state mutex poisoned");
                    let next_count = match state.get(&prefix_hash).copied() {
                        Some(PrimeState::Failed(n)) => n + 1,
                        Some(PrimeState::Primed { .. }) => {
                            // A concurrent caller succeeded while we
                            // were still attempting. Trust their success.
                            tracing::debug!(
                                %model,
                                "prime attempt failed but concurrent caller already primed this prefix"
                            );
                            return true;
                        }
                        _ => 1,
                    };

                    if next_count >= MAX_PRIME_ATTEMPTS {
                        state.insert(
                            prefix_hash,
                            PrimeState::Abandoned {
                                at: std::time::Instant::now(),
                            },
                        );
                        drop(state);
                        tracing::warn!(
                            %model,
                            attempts = next_count,
                            error = %e,
                            "cache prime abandoned after {MAX_PRIME_ATTEMPTS} consecutive failures \
                             — real batches for this model will submit without caching; if prior \
                             prime batches eventually landed on Anthropic's side, the cache may \
                             still serve hits"
                        );
                        return false;
                    }

                    state.insert(prefix_hash, PrimeState::Failed(next_count));
                    drop(state);
                    tracing::warn!(
                        %model,
                        attempt = next_count,
                        max_attempts = MAX_PRIME_ATTEMPTS,
                        error = %e,
                        "cache prime attempt failed, retrying"
                    );
                    // Continue the loop — another iteration will call
                    // `prime_via_batch` again.
                }
            }
        }
    }

    /// Prime every prefix in `entries` sequentially. Entries are
    /// `(prefix_hash, model, prompt)` tuples, typically one per unique
    /// Anthropic model in the current run. Returns `(ok_count,
    /// total_count)` so the caller can log an aggregate.
    ///
    /// Sequential rather than concurrent: the common case is 1 unique
    /// Anthropic model (one Haiku variant) and parallelism would only
    /// help in the rare >1-model case. Keeping this simple avoids adding
    /// a `futures` dependency to the agents workspace for a benefit we
    /// don't actually observe in practice.
    pub async fn prime_prefixes(
        &self,
        entries: Vec<(u64, String, CachedPrompt<'static>)>,
    ) -> (usize, usize) {
        let total = entries.len();
        let mut ok = 0usize;
        for (hash, model, prompt) in entries {
            if self.prime_prefix(hash, &model, &prompt).await {
                ok += 1;
            }
        }
        (ok, total)
    }
}

impl BatchBackend<CachedPrompt<'static>, MMessage<'static>> for AnthropicBatch {
    type Handle = AnthropicPendingHandle;

    async fn submit(
        &self,
        items: Vec<WorkItem<CachedPrompt<'static>>>,
    ) -> anyhow::Result<Self::Handle> {
        // No priming happens here. Cache priming is eager: the caller
        // invokes [`AnthropicBatch::prime_prefixes`] at the start of
        // each cycle, before that cycle's real batches, so by the time
        // `submit` is called the cache is either already warm or the
        // prefix has been `Abandoned` (in which case this batch just
        // pays uncached input cost — that's the desired soft-degrade).
        //
        // `submit` is therefore purely mechanical: build the id_map,
        // build the tagged prompts, hand off to `client.tagged_batch`,
        // return the pending handle.
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
