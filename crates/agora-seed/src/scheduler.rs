//! Pipeline scheduler for batch agent execution.
//!
//! Replaces the old wave-based scheduler with a pipeline that:
//! - Batches THINK requests for cache efficiency
//! - Interleaves batches so later agents see earlier agents' actions
//! - Supports both Anthropic Batch API and Ollama backends
//!
//! The cycle for each agent follows: PERCEIVE → THINK → ACT → REFLECT
//! where THINK and REFLECT are batched, while PERCEIVE and ACT are per-agent.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use agora_agent_lib::agora_agentkit::ids::AgentId;
use agora_agent_lib::agora_agentkit::scheduler::{
    BatchBackend, BatchState, CycleStep, WorkItem,
};
use agora_agent_lib::batch::anthropic::AnthropicBatch;
use agora_agent_lib::batch::ollama::OllamaEndpoint;
use agora_agent_lib::llm::{LlmBackend, Message, Role};
use agora_agent_lib::tools;
use anyhow::Result;
use misanthropic::prompt::Message as MMessage;
use misanthropic::{CachedPrompt, Prompt};
use rand::seq::SliceRandom;
use rand::Rng;
use serde::Serialize;

use crate::agent::Agent;
use crate::client::AgoraClient;
use crate::config::{Backend, Cli};
use crate::prompt;

/// Check if a model name is compatible with the Anthropic API.
///
/// Anthropic models contain a family name (haiku, sonnet, opus) or use
/// the "claude-" prefix. Ollama models (cogito, qwen, gpt-oss, etc.)
/// and aliases like "seed-runner" won't work — they need to be resolved
/// to real Anthropic model IDs first.
fn is_anthropic_model(model: &str) -> bool {
    let m = model.to_lowercase();
    m.contains("haiku")
        || m.contains("sonnet")
        || m.contains("opus")
        || m.starts_with("claude")
}

/// Check if a model name looks like a real model (not a placeholder or alias).
///
/// Valid models are either Anthropic models or Ollama model IDs (which contain
/// a colon separating name from size, e.g. "cogito:14b").
fn is_valid_model(model: &str) -> bool {
    is_anthropic_model(model) || model.contains(':')
}

/// End-of-run statistics report.
#[derive(Debug, Default, Serialize)]
pub struct RunReport {
    /// Total agents that participated.
    pub agents: usize,
    /// Number of cycles executed.
    pub cycles: usize,
    /// Wall-clock duration in seconds.
    pub duration_secs: f64,
    /// Action counts by type.
    pub actions: ActionCounts,
    /// Per-model breakdown of actions.
    pub by_model: HashMap<String, ActionCounts>,
    /// Error/skip counts.
    pub skipped: SkipCounts,
    /// Soul evolution stats.
    pub evolution: EvolutionCounts,
    /// Survey stats.
    pub surveys: SurveyCounts,
}

#[derive(Debug, Default, Serialize)]
pub struct ActionCounts {
    pub posts: u32,
    pub comments: u32,
    pub votes: u32,
    pub flags: u32,
    pub observations: u32,
}

#[derive(Debug, Default, Serialize)]
pub struct SkipCounts {
    pub duplicate_comments: u32,
    pub repetitive_titles: u32,
    pub perceive_failures: u32,
    pub think_failures: u32,
    pub reflect_failures: u32,
    pub post_failures: u32,
    pub comment_failures: u32,
    pub vote_failures: u32,
}

#[derive(Debug, Default, Serialize)]
pub struct EvolutionCounts {
    pub deep_mutations: u32,
    pub evolution_entries: u32,
    pub mutation_failures: u32,
}

#[derive(Debug, Default, Serialize)]
pub struct SurveyCounts {
    pub submitted: u32,
    pub skipped_empty: u32,
    pub failures: u32,
}

impl RunReport {
    fn model_actions(&mut self, model: &str) -> &mut ActionCounts {
        self.by_model
            .entry(model.to_string())
            .or_default()
    }
}

/// Shared pool of batches that workers pull from on demand.
///
/// Replaces the old per-endpoint channel design where all work was
/// pre-assigned at cycle start. Now workers pull their next batch when
/// ready, so faster endpoints naturally consume more work.
struct BatchPool {
    batches: std::sync::Mutex<Vec<(String, Vec<Agent>)>>,
    /// Models available on exactly one endpoint. These get a scoring
    /// bonus so the endpoint that exclusively serves them picks them up
    /// first, leaving shared models for whichever endpoint is free.
    exclusive_models: std::collections::HashSet<String>,
}

impl BatchPool {
    fn new(batches: Vec<(String, Vec<Agent>)>, endpoints: &[OllamaEndpoint]) -> Self {
        // Count how many endpoints can serve each model.
        let mut model_counts: HashMap<String, usize> = HashMap::new();
        for ep in endpoints {
            for model in &ep.models {
                *model_counts.entry(model.clone()).or_default() += 1;
            }
        }
        let exclusive_models: std::collections::HashSet<String> = model_counts
            .into_iter()
            .filter(|(_, count)| *count == 1)
            .map(|(model, _)| model)
            .collect();

        if !exclusive_models.is_empty() {
            tracing::info!(
                "Exclusive models (prioritized on their endpoint): [{}]",
                {
                    let mut sorted: Vec<_> = exclusive_models.iter().cloned().collect();
                    sorted.sort();
                    sorted.join(", ")
                },
            );
        }

        Self {
            batches: std::sync::Mutex::new(batches),
            exclusive_models,
        }
    }

    /// Pull the next batch this endpoint can handle.
    ///
    /// Scores each eligible batch and samples weighted by score, rather
    /// than deterministically picking the "best" one. This prevents
    /// behavioral waves from consecutive same-model batches while still
    /// favoring cache locality and pool ordering.
    ///
    /// Scoring (multiplied together):
    /// - **Position**: earlier in pool = higher weight (preserves the
    ///   largest-first + round-robin interleaving from `create_batches`).
    ///   Weight: `1 / (rank + 1)`.
    /// - **Cache hit**: same as `last_model` gets a 1.5x bonus (KV cache
    ///   still loaded, avoids expensive model swap).
    /// - **Exclusivity**: 3x bonus for models only this endpoint can run.
    ///   Ensures endpoints prioritize work nobody else can handle before
    ///   picking up shared models.
    fn next_for(
        &self,
        endpoint: &OllamaEndpoint,
        last_model: Option<&str>,
    ) -> Option<(String, Vec<Agent>)> {
        let mut pool = self.batches.lock().unwrap();

        // Collect (pool_index, weight) for all batches this endpoint supports.
        let candidates: Vec<(usize, f64)> = pool
            .iter()
            .enumerate()
            .filter(|(_, (m, _))| endpoint.models.contains(m))
            .enumerate() // rank among eligible (0 = first eligible)
            .map(|(rank, (pool_idx, (model, _)))| {
                // Position weight: earlier in pool = higher weight.
                // 1/(rank+1) → first=1.0, second=0.5, third=0.33...
                let position_weight = 1.0 / (rank as f64 + 1.0);

                // Cache bonus: prefer keeping the same model loaded.
                let cache_bonus = match last_model {
                    Some(last) if last == model => 1.5,
                    _ => 1.0,
                };

                // Exclusivity bonus: strongly prefer models only we can run.
                let exclusive_bonus = if self.exclusive_models.contains(model) {
                    3.0
                } else {
                    1.0
                };

                (pool_idx, position_weight * cache_bonus * exclusive_bonus)
            })
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Weighted random selection.
        let total: f64 = candidates.iter().map(|(_, w)| w).sum();
        let mut roll = rand::thread_rng().r#gen::<f64>() * total;
        let mut chosen = candidates[0].0;
        for &(idx, weight) in &candidates {
            roll -= weight;
            if roll <= 0.0 {
                chosen = idx;
                break;
            }
        }

        Some(pool.remove(chosen))
    }

    fn remaining(&self) -> usize {
        self.batches.lock().unwrap().len()
    }
}

/// Run all agents using the pipeline scheduler.
pub async fn run_all(
    agents: &mut Vec<Agent>,
    client: &AgoraClient,
    config: &Cli,
    constitution: &str,
) -> Result<()> {
    let start = Instant::now();

    // Filter agents
    if let Some(ref filter) = config.agent_filter {
        agents.retain(|a| a.name.contains(filter.as_str()));
    }

    // Remove unregistered agents
    agents.retain(|a| {
        if a.agent_id.is_none() {
            tracing::warn!("Skipping unregistered agent: {}", a.name);
            false
        } else {
            true
        }
    });

    // Validate model assignments — fail fast on bad data.
    let invalid: Vec<_> = agents.iter()
        .filter(|a| !is_valid_model(&a.model))
        .map(|a| format!("{} (model_info='{}')", a.name, a.model))
        .collect();
    if !invalid.is_empty() {
        anyhow::bail!(
            "{} agent(s) have invalid model assignments (not an Anthropic model or Ollama model:size):\n  {}",
            invalid.len(),
            invalid.join("\n  ")
        );
    }

    if agents.is_empty() {
        tracing::warn!("No registered agents to run");
        return Ok(());
    }

    let mut report = RunReport {
        agents: agents.len(),
        cycles: config.cycles,
        ..Default::default()
    };

    tracing::info!(
        "Pipeline scheduler: {} agents, {} cycles",
        agents.len(),
        config.cycles,
    );

    // Select and run with the appropriate backend
    match config.backend {
        Backend::Ollama => {
            // Discover models on each endpoint.
            let http = reqwest::Client::new();
            let urls = config.effective_ollama_urls();
            let mut endpoints = Vec::with_capacity(urls.len());
            for url in &urls {
                match OllamaEndpoint::discover(&http, url).await {
                    Ok(ep) => endpoints.push(ep),
                    Err(e) => {
                        tracing::error!("Failed to discover models at {url}: {e}");
                        anyhow::bail!("Cannot reach Ollama endpoint {url}: {e}");
                    }
                }
            }

            // Collect all models available on Ollama.
            let ollama_models: std::collections::HashSet<String> = endpoints
                .iter()
                .flat_map(|ep| ep.models.iter().cloned())
                .collect();

            // Find agents whose model isn't on any Ollama endpoint.
            let mut anthropic_missing: HashMap<String, usize> = HashMap::new();
            let mut unsupported: HashMap<String, usize> = HashMap::new();
            for agent in agents.iter() {
                if !ollama_models.contains(&agent.model) {
                    if is_anthropic_model(&agent.model) {
                        *anthropic_missing.entry(agent.model.clone()).or_default() += 1;
                    } else {
                        *unsupported.entry(agent.model.clone()).or_default() += 1;
                    }
                }
            }

            // Drop agents with models that aren't on Ollama or Anthropic.
            if !unsupported.is_empty() {
                for (model, count) in &unsupported {
                    tracing::warn!(
                        "Model '{model}' not on any Ollama endpoint and not an Anthropic model — skipping {count} agents"
                    );
                }
                agents.retain(|a| ollama_models.contains(&a.model) || is_anthropic_model(&a.model));
                report.agents = agents.len();
            }

            let ollama_endpoints = endpoints;

            if !anthropic_missing.is_empty() && config.anthropic_key_file.is_some() {
                let key_file = config.anthropic_key_file.as_ref().unwrap();
                let api_key = tokio::fs::read_to_string(key_file).await
                    .map_err(|e| anyhow::anyhow!("reading Anthropic key from {}: {e}", key_file.display()))?;
                let anthropic = AnthropicBatch::from_key(api_key.trim().to_string())?;

                for (model, count) in &anthropic_missing {
                    tracing::info!("Model '{model}' → anthropic ({count} agents)");
                }

                run_cycles(&ollama_endpoints, Some(&anthropic), agents, client, config, constitution, &ollama_models, &mut report).await?;
            } else {
                for (model, count) in &anthropic_missing {
                    tracing::warn!(
                        "Model '{model}' not on any endpoint ({count} agents affected)"
                    );
                }

                run_cycles(&ollama_endpoints, None, agents, client, config, constitution, &HashSet::new(), &mut report).await?;
            }
        }
        Backend::Anthropic => {
            let key_file = config.anthropic_key_file.as_ref()
                .ok_or_else(|| anyhow::anyhow!("--anthropic-key-file is required when --backend=anthropic"))?;
            let api_key = tokio::fs::read_to_string(key_file).await
                .map_err(|e| anyhow::anyhow!("reading Anthropic key from {}: {e}", key_file.display()))?;
            let backend = AnthropicBatch::from_key(api_key.trim().to_string())?;
            run_cycles(&[], Some(&backend), agents, client, config, constitution, &std::collections::HashSet::new(), &mut report).await?;
        }
    }

    report.duration_secs = start.elapsed().as_secs_f64();

    // Print JSON report
    tracing::info!("Pipeline scheduler complete!");
    match serde_json::to_string_pretty(&report) {
        Ok(json) => {
            tracing::info!("=== RUN REPORT ===\n{json}");
            // Write to file if reports/ directory exists
            let reports_dir = std::path::Path::new("reports");
            if reports_dir.is_dir() {
                let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S");
                let path = reports_dir.join(format!("run_{ts}.json"));
                if let Err(e) = tokio::fs::write(&path, &json).await {
                    tracing::warn!("Failed to write report to {}: {e}", path.display());
                } else {
                    tracing::info!("Report written to {}", path.display());
                }
            }
        }
        Err(e) => tracing::warn!("Failed to serialize run report: {e}"),
    }

    Ok(())
}

/// Run all cycles using a pull-based pool scheduler.
///
/// - **Producer**: groups Ollama agents by model into same-model batches
/// - **Pool**: batches go into a shared `BatchPool`
/// - **Workers**: one per endpoint, pull from the pool when ready — faster
///   GPUs naturally consume more batches
/// - **Anthropic**: runs as a separate consumer concurrently
async fn run_cycles(
    ollama_endpoints: &[OllamaEndpoint],
    anthropic: Option<&AnthropicBatch>,
    agents: &mut Vec<Agent>,
    client: &AgoraClient,
    config: &Cli,
    constitution: &str,
    ollama_models: &std::collections::HashSet<String>,
    report: &mut RunReport,
) -> Result<()> {
    let batch_size = config.batch_size.unwrap_or(50);

    let all_endpoints: Vec<OllamaEndpoint> = ollama_endpoints.to_vec();

    for cycle in 0..config.cycles {
        tracing::info!("=== Cycle {}/{} ===", cycle + 1, config.cycles);

        agents.shuffle(&mut rand::thread_rng());

        // Split Anthropic agents from Ollama agents.
        let ollama_count = if !ollama_models.is_empty() && anthropic.is_some() {
            agents.sort_by_key(|a| if ollama_models.contains(&a.model) { 0 } else { 1 });
            agents.iter().position(|a| !ollama_models.contains(&a.model))
                .unwrap_or(agents.len())
        } else if anthropic.is_some() && ollama_endpoints.is_empty() {
            0
        } else {
            agents.len()
        };

        if !ollama_endpoints.is_empty() && ollama_count > 0 {
            // --- Producer: create interleaved same-model batches ---
            let all_ollama: Vec<Agent> = agents.drain(..ollama_count).collect();
            // agents now contains only Anthropic agents (if any).
            let anthropic_agents = agents.as_mut_slice();

            let batches = create_batches(all_ollama, batch_size);
            let model_count = batches.iter()
                .map(|(m, _)| m.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len();

            // --- Pool: shared batch pool that workers pull from ---
            let pool = BatchPool::new(batches, ollama_endpoints);
            tracing::info!(
                "Work pool: {} batches, {} models, {} endpoints",
                pool.remaining(), model_count, ollama_endpoints.len(),
            );
            if !anthropic_agents.is_empty() {
                tracing::info!(
                    "Anthropic: {} agents in 1 batch (concurrent)",
                    anthropic_agents.len(),
                );
            }

            // --- Workers: one per endpoint + Anthropic ---
            let mut worker_reports: Vec<RunReport> = ollama_endpoints
                .iter().map(|_| RunReport::default()).collect();
            let mut anthropic_report = RunReport::default();
            let all_eps = &all_endpoints;

            let (results_tx, mut results_rx) =
                tokio::sync::mpsc::unbounded_channel::<Vec<Agent>>();

            let anthropic_fut = async {
                if let Some(backend) = anthropic {
                    if !anthropic_agents.is_empty() {
                        tracing::info!("--- Anthropic batch ({} agents) ---", anthropic_agents.len());
                        run_batch(
                            backend, anthropic_agents, client, config, constitution,
                            Some(all_eps.as_slice()), &mut anthropic_report, cycle,
                        ).await?;
                    }
                }
                Ok::<_, anyhow::Error>(())
            };

            let (ollama_result, anthropic_result) = tokio::join!(
                async {
                    match ollama_endpoints.len() {
                        0 => Ok::<_, anyhow::Error>(()),
                        1 => {
                            run_worker(
                                &ollama_endpoints[0], &pool,
                                results_tx.clone(),
                                client, config, constitution,
                                &mut worker_reports[0], cycle,
                            ).await
                        }
                        2 => {
                            let (r0, r1) = worker_reports.split_at_mut(1);
                            let (a, b) = tokio::join!(
                                run_worker(
                                    &ollama_endpoints[0], &pool,
                                    results_tx.clone(),
                                    client, config, constitution,
                                    &mut r0[0], cycle,
                                ),
                                run_worker(
                                    &ollama_endpoints[1], &pool,
                                    results_tx.clone(),
                                    client, config, constitution,
                                    &mut r1[0], cycle,
                                ),
                            );
                            a.and(b)
                        }
                        _ => {
                            let (r0, rest) = worker_reports.split_at_mut(1);
                            let (r1, r2) = rest.split_at_mut(1);
                            let (a, b, c) = tokio::join!(
                                run_worker(
                                    &ollama_endpoints[0], &pool,
                                    results_tx.clone(),
                                    client, config, constitution,
                                    &mut r0[0], cycle,
                                ),
                                run_worker(
                                    &ollama_endpoints[1], &pool,
                                    results_tx.clone(),
                                    client, config, constitution,
                                    &mut r1[0], cycle,
                                ),
                                run_worker(
                                    &ollama_endpoints[2], &pool,
                                    results_tx.clone(),
                                    client, config, constitution,
                                    &mut r2[0], cycle,
                                ),
                            );
                            a.and(b).and(c)
                        }
                    }
                },
                anthropic_fut,
            );
            drop(results_tx);

            if let Err(e) = &ollama_result {
                tracing::error!("Ollama pipeline error: {e:#}");
            }
            if let Err(e) = &anthropic_result {
                tracing::error!("Anthropic pipeline error: {e:#}");
            }
            for wr in &worker_reports {
                merge_reports(report, wr);
            }
            merge_reports(report, &anthropic_report);

            // Collect processed agents from results channel.
            let mut processed = Vec::with_capacity(ollama_count);
            while let Ok(batch_agents) = results_rx.try_recv() {
                processed.extend(batch_agents);
            }
            processed.append(agents);
            *agents = processed;

            ollama_result?;
            anthropic_result?;

        } else {
            // No Ollama endpoints — Anthropic only.
            if let Some(backend) = anthropic {
                let anthropic_agents = agents.as_mut_slice();
                if !anthropic_agents.is_empty() {
                    let mut anthropic_report = RunReport::default();
                    tracing::info!("--- Anthropic batch ({} agents) ---", anthropic_agents.len());
                    run_batch(
                        backend, anthropic_agents, client, config, constitution,
                        None, &mut anthropic_report, cycle,
                    ).await?;
                    merge_reports(report, &anthropic_report);
                }
            }
        }
    }

    Ok(())
}

/// Extract parameter size from a model name like "cogito:70b" → 70, "qwen3.5:35b" → 35.
/// Returns 0 for names without a recognizable size suffix (e.g. Anthropic model IDs).
fn extract_model_size(model: &str) -> u64 {
    // Look for the part after ':' (e.g. "70b", "14b", "24b", "e4b")
    let tag = model.rsplit(':').next().unwrap_or("");
    // Strip trailing 'b' and parse the number
    let num_str = tag.trim_end_matches('b');
    num_str.parse::<u64>().unwrap_or(0)
}

/// Group a sorted list of agents into same-model batches of up to `batch_size`.
/// Create same-model batches, interleaved round-robin across models.
///
/// Each batch contains up to `batch_size` agents of the same model (needed
/// for KV prefix cache reuse). Batches are ordered by round-robin across
/// models so that different models alternate — this prevents behavioral
/// waves where all agents of one model (e.g. posters) act before agents
/// of another model (e.g. commenters).
fn create_batches(agents: Vec<Agent>, batch_size: usize) -> Vec<(String, Vec<Agent>)> {
    // Group agents by model, preserving insertion order.
    let mut by_model: Vec<(String, Vec<Agent>)> = Vec::new();
    let mut model_index: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for agent in agents {
        if let Some(&idx) = model_index.get(&agent.model) {
            by_model[idx].1.push(agent);
        } else {
            model_index.insert(agent.model.clone(), by_model.len());
            by_model.push((agent.model.clone(), vec![agent]));
        }
    }

    // Sort models so the largest (slowest) come first in the round-robin.
    // This ensures slower endpoints (e.g. Mac) pick up big models immediately
    // while faster endpoints (e.g. 3090) handle small models in parallel,
    // so both finish closer together.
    by_model.sort_by(|(a, _), (b, _)| {
        extract_model_size(b).cmp(&extract_model_size(a))
    });

    // Round-robin: take one batch_size chunk from each model in turn.
    let mut batches = Vec::new();
    let total: usize = by_model.iter().map(|(_, agents)| agents.len()).sum();
    let mut emitted = 0;

    while emitted < total {
        for (model, agents) in by_model.iter_mut() {
            if agents.is_empty() {
                continue;
            }
            let take = batch_size.min(agents.len());
            let batch: Vec<Agent> = agents.drain(..take).collect();
            emitted += batch.len();
            batches.push((model.clone(), batch));
        }
    }

    batches
}

/// Endpoint worker: pulls batches from the shared pool, processes each
/// through the full pipeline, sends processed agents to the results channel.
///
/// The pool's weighted sampling favors cache-hot models (1.5x bonus) and
/// earlier pool positions (largest-first ordering), while keeping all
/// eligible batches reachable to prevent behavioral waves.
async fn run_worker(
    endpoint: &OllamaEndpoint,
    pool: &BatchPool,
    results_tx: tokio::sync::mpsc::UnboundedSender<Vec<Agent>>,
    client: &AgoraClient,
    config: &Cli,
    constitution: &str,
    report: &mut RunReport,
    cycle: usize,
) -> Result<()> {
    let mut batches_done = 0usize;
    let mut last_model: Option<String> = None;
    while let Some((model, mut batch_agents)) = pool.next_for(endpoint, last_model.as_deref()) {
        batches_done += 1;
        tracing::info!(
            "--- {} batch {} ({} × {}) [{} remaining] ---",
            endpoint.url, batches_done, batch_agents.len(), model,
            pool.remaining(),
        );
        run_batch_sequential(
            endpoint, &mut batch_agents, client, config, constitution,
            report, cycle,
        ).await?;
        let _ = results_tx.send(batch_agents);
        last_model = Some(model);
    }
    tracing::info!("{} finished: {} batches processed", endpoint.url, batches_done);
    Ok(())
}

/// Run a single batch of agents through the full pipeline:
/// PERCEIVE → THINK → ACT → REFLECT → EVOLVE → SURVEY.
async fn run_batch<B>(
    backend: &B,
    batch_agents: &mut [Agent],
    client: &AgoraClient,
    config: &Cli,
    constitution: &str,
    ollama_endpoints: Option<&[OllamaEndpoint]>,
    report: &mut RunReport,
    cycle: usize,
) -> Result<()>
where
    B: BatchBackend<Prompt<'static>, MMessage<'static>>,
{
    // Phase 1: PERCEIVE
    let mut agent_contexts: Vec<AgentCycleContext> = Vec::new();

    for (idx, agent) in batch_agents.iter_mut().enumerate() {
        let agent_id = agent.agent_id.unwrap();
        match perceive(agent, agent_id, client, config).await {
            Ok(mut ctx) => {
                ctx.batch_index = idx;
                agent_contexts.push(ctx);
            }
            Err(e) => {
                tracing::warn!("Perceive failed for {}: {e:#}", agent.name);
                report.skipped.perceive_failures += 1;
            }
        }
    }

    if agent_contexts.is_empty() {
        return Ok(());
    }

    // Phase 2: THINK
    let mut work_items: Vec<WorkItem<Prompt<'static>>> = Vec::new();

    for ctx in &agent_contexts {
        let agent = &batch_agents[ctx.batch_index];
        let think_prompt = prompt::build_think_prompt(
            &agent.model,
            &agent.soul.as_system_prompt(),
            &agent.memory.content,
            &ctx.recent_activity,
            &ctx.pending_replies_text,
            constitution,
            &ctx.perception_text,
        );

        let prefix_hash = {
            let mut hasher = DefaultHasher::new();
            agent.model.hash(&mut hasher);
            hasher.finish()
        };

        work_items.push(WorkItem {
            agent_id: agent.agent_id.unwrap(),
            prompt: think_prompt,
            step: CycleStep::Think,
            prefix_hash,
            model: agent.model.clone(),
            queued_at: Instant::now(),
            token_count: 0,
        });
    }

    let think_results = submit_and_poll(backend, work_items).await?;

    // Phase 3: ACT
    let mut action_summaries_map: HashMap<AgentId, Vec<String>> = HashMap::new();
    let mut think_response_map: HashMap<AgentId, String> = HashMap::new();

    for result in &think_results {
        let response = match &result.response {
            Ok(msg) => msg,
            Err(e) => {
                tracing::warn!("Think failed for agent {}: {e}", result.agent_id);
                report.skipped.think_failures += 1;
                continue;
            }
        };

        let Some(ctx) = agent_contexts.iter().find(|c| {
            batch_agents[c.batch_index].agent_id == Some(result.agent_id)
        }) else {
            continue;
        };
        let agent = &mut batch_agents[ctx.batch_index];

        think_response_map.insert(result.agent_id, response.content.to_string());

        let actions = tools::extract_actions(response);
        tracing::info!(
            "[{}/{}] {} — act ({} actions)",
            cycle + 1, config.cycles, agent.name, actions.len(),
        );

        let summaries = execute_actions(
            agent, &actions, client, &ctx.feeds, &ctx.comment_replies, report,
        ).await;

        action_summaries_map.insert(result.agent_id, summaries);
    }

    // Phase 4: REFLECT
    let mut reflect_items: Vec<WorkItem<Prompt<'static>>> = Vec::new();

    for ctx in &agent_contexts {
        let agent = &batch_agents[ctx.batch_index];
        let agent_id = agent.agent_id.unwrap();
        let summaries = action_summaries_map
            .get(&agent_id)
            .cloned()
            .unwrap_or_default();

        let reflect_text = prompt::build_memory_rewrite_prompt(
            &agent.name, &agent.memory.content, &summaries,
        );

        let reflect_prompt = build_text_prompt(
            &agent.model,
            "You are a memory manager. Rewrite the agent's personal notes concisely.",
            &reflect_text,
            512,
        );

        reflect_items.push(WorkItem {
            agent_id,
            prompt: reflect_prompt,
            step: CycleStep::Reflect,
            prefix_hash: 0,
            model: agent.model.clone(),
            queued_at: Instant::now(),
            token_count: 0,
        });
    }

    let reflect_results = submit_and_poll(backend, reflect_items).await?;

    for result in &reflect_results {
        let response_text = match &result.response {
            Ok(msg) => msg.content.to_string(),
            Err(e) => {
                tracing::warn!("Reflect failed for agent {}: {e}", result.agent_id);
                report.skipped.reflect_failures += 1;
                continue;
            }
        };

        let Some(ctx) = agent_contexts.iter().find(|c| {
            batch_agents[c.batch_index].agent_id == Some(result.agent_id)
        }) else {
            continue;
        };
        let agent = &mut batch_agents[ctx.batch_index];

        let memory_content = prompt::parse_memory_rewrite(&response_text)
            .unwrap_or(response_text);
        agent.memory.update(memory_content);
        if let Err(e) = agent.save_memory().await {
            tracing::warn!("Failed to save memory for {}: {e}", agent.name);
        }
        agent.last_cycle_at = Some(chrono::Utc::now());
    }

    // Phase 5: EVOLVE
    for ctx in &agent_contexts {
        let agent = &mut batch_agents[ctx.batch_index];
        let agent_id = agent.agent_id.unwrap();
        let summaries = action_summaries_map
            .get(&agent_id)
            .cloned()
            .unwrap_or_default();

        let on_ollama = ollama_endpoints
            .and_then(|eps| {
                eps.iter()
                    .find(|ep| ep.models.contains(&agent.model))
                    .map(|ep| ep.url.as_str())
            });

        let single_backend: Box<dyn LlmBackend> = match config.backend {
            Backend::Anthropic => {
                let key_file = config.anthropic_key_file.as_ref().unwrap();
                let api_key = tokio::fs::read_to_string(key_file).await
                    .unwrap_or_default();
                match agora_agent_lib::llm::anthropic::AnthropicBackend::new(
                    api_key.trim().to_string(), &agent.model,
                ) {
                    Ok(b) => Box::new(b),
                    Err(e) => {
                        tracing::warn!("Failed to create Anthropic backend for evolution: {e}");
                        continue;
                    }
                }
            }
            Backend::Ollama => {
                if let Some(url) = on_ollama {
                    match agora_agent_lib::llm::ollama::OllamaBackend::new(
                        Some(url), &agent.model,
                    ) {
                        Ok(b) => Box::new(b),
                        Err(e) => {
                            tracing::warn!("Failed to create Ollama backend for {}: {e}", agent.name);
                            continue;
                        }
                    }
                } else if let Some(ref key_file) = config.anthropic_key_file {
                    let api_key = tokio::fs::read_to_string(key_file).await
                        .unwrap_or_default();
                    match agora_agent_lib::llm::anthropic::AnthropicBackend::new(
                        api_key.trim().to_string(), &agent.model,
                    ) {
                        Ok(b) => Box::new(b),
                        Err(e) => {
                            tracing::warn!("Failed to create Anthropic backend for {}: {e}", agent.name);
                            continue;
                        }
                    }
                } else {
                    match agora_agent_lib::llm::ollama::OllamaBackend::new(
                        config.ollama_url.as_deref(), &agent.model,
                    ) {
                        Ok(b) => Box::new(b),
                        Err(e) => {
                            tracing::warn!("Failed to create Ollama backend for {}: {e}", agent.name);
                            continue;
                        }
                    }
                }
            }
        };

        run_evolution(
            agent, single_backend.as_ref(), config.mutation_chance,
            &summaries, report,
        ).await;
    }

    // Phase 6: SURVEY
    let mut survey_items: Vec<WorkItem<Prompt<'static>>> = Vec::new();

    for ctx in &agent_contexts {
        let agent = &batch_agents[ctx.batch_index];
        let agent_id = agent.agent_id.unwrap();

        if !config.force_survey && rand::random::<f64>() >= 0.10 {
            continue;
        }

        let summaries = action_summaries_map
            .get(&agent_id)
            .cloned()
            .unwrap_or_default();
        let response_text = think_response_map
            .get(&agent_id)
            .cloned()
            .unwrap_or_default();

        let survey_text = prompt::build_survey_prompt(&agent.name, &summaries);
        let system = prompt::build_cached_system_prefix(constitution);

        let survey_prompt = build_survey_conversation(
            &agent.model, &system, &ctx.perception_text,
            &response_text, &survey_text,
        );

        survey_items.push(WorkItem {
            agent_id,
            prompt: survey_prompt,
            step: CycleStep::Survey,
            prefix_hash: 0,
            model: agent.model.clone(),
            queued_at: Instant::now(),
            token_count: 0,
        });
    }

    if !survey_items.is_empty() {
        tracing::info!("Surveying {} agents", survey_items.len());
        let survey_results = submit_and_poll(backend, survey_items).await?;

        for result in &survey_results {
            let response_text = match &result.response {
                Ok(msg) => prompt::extract_speech(&msg.content),
                Err(e) => {
                    tracing::debug!("Survey failed for agent {}: {e}", result.agent_id);
                    report.surveys.failures += 1;
                    continue;
                }
            };

            let trimmed = response_text.trim();
            if trimmed.is_empty()
                || trimmed.eq_ignore_ascii_case("no feedback")
                || trimmed.eq_ignore_ascii_case("no feedback.")
            {
                report.surveys.skipped_empty += 1;
                continue;
            }

            match client.submit_feedback(trimmed).await {
                Ok(()) => {
                    tracing::info!("  anonymous feedback submitted");
                    report.surveys.submitted += 1;
                }
                Err(e) => {
                    tracing::debug!("Anonymous feedback submission failed: {e}");
                    report.surveys.failures += 1;
                }
            }
        }
    }

    Ok(())
}

/// Run a batch of Ollama agents through the full pipeline sequentially,
/// one agent at a time: PERCEIVE → THINK → ACT → REFLECT → EVOLVE → SURVEY.
///
/// Unlike [`run_batch`] (which processes all agents through each phase as a
/// wave), this function completes each agent's full cycle before moving to the
/// next. This maximizes Ollama's KV prefix cache reuse: each subsequent phase
/// appends to the same `Prompt`, so the entire prefix is already in GPU memory.
async fn run_batch_sequential(
    endpoint: &OllamaEndpoint,
    batch_agents: &mut [Agent],
    client: &AgoraClient,
    config: &Cli,
    constitution: &str,
    report: &mut RunReport,
    cycle: usize,
) -> Result<()> {
    use misanthropic::prompt::message::Role as MRole;
    use std::num::NonZeroU32;

    for agent in batch_agents.iter_mut() {
        let agent_id = agent.agent_id.unwrap();
        let model = agent.model.clone();

        // Phase 1: PERCEIVE
        let ctx = match perceive(agent, agent_id, client, config).await {
            Ok(ctx) => ctx,
            Err(e) => {
                tracing::warn!("Perceive failed for {}: {e:#}", agent.name);
                report.skipped.perceive_failures += 1;
                continue;
            }
        };

        // Phase 2: THINK
        let mut think_prompt = CachedPrompt::uncached(prompt::build_think_prompt(
            &model,
            &agent.soul.as_system_prompt(),
            &agent.memory.content,
            &ctx.recent_activity,
            &ctx.pending_replies_text,
            constitution,
            &ctx.perception_text,
        ));

        let think_response = match endpoint.send(&think_prompt, &model).await {
            Ok(msg) => msg,
            Err(e) => {
                tracing::warn!("Think failed for {} at {}: {e}", agent.name, endpoint.url);
                report.skipped.think_failures += 1;
                continue;
            }
        };

        // Phase 3: ACT
        let actions = tools::extract_actions(&think_response);
        tracing::info!(
            "[{}/{}] {} — act ({} actions)",
            cycle + 1, config.cycles, agent.name, actions.len(),
        );

        let summaries = execute_actions(
            agent, &actions, client, &ctx.feeds, &ctx.comment_replies, report,
        ).await;

        // --- Session continuity: append assistant response for subsequent phases ---
        if let Err(e) = think_prompt.push_message(think_response) {
            tracing::warn!("Failed to append think response for {}: {e}", agent.name);
            continue;
        }

        // Phase 4: REFLECT
        let reflect_text = prompt::build_memory_rewrite_prompt(
            &agent.name, &agent.memory.content, &summaries,
        );
        if let Err(e) = think_prompt.push_message((MRole::User, reflect_text)) {
            tracing::warn!("Failed to append reflect prompt for {}: {e}", agent.name);
            continue;
        }
        think_prompt.set_max_tokens(NonZeroU32::new(512).unwrap());

        match endpoint.send(&think_prompt, &model).await {
            Ok(reflect_response) => {
                let response_text = reflect_response.content.to_string();
                let memory_content = prompt::parse_memory_rewrite(&response_text)
                    .unwrap_or(response_text.clone());
                agent.memory.update(memory_content);
                if let Err(e) = agent.save_memory().await {
                    tracing::warn!("Failed to save memory for {}: {e}", agent.name);
                }
                agent.last_cycle_at = Some(chrono::Utc::now());

                // Append reflect response for evolve/survey
                if let Err(e) = think_prompt.push_message(reflect_response) {
                    tracing::debug!("Failed to append reflect response for {}: {e}", agent.name);
                }
            }
            Err(e) => {
                tracing::warn!("Reflect failed for {}: {e}", agent.name);
                report.skipped.reflect_failures += 1;
            }
        }

        // Phase 5: EVOLVE
        let roll = rand::random::<u32>() % 100;
        let experience = summaries.join("; ");
        let deep_threshold = config.mutation_chance.unwrap_or(3);
        let evo_threshold = deep_threshold + 10;

        if roll < deep_threshold {
            // Deep soul mutation
            tracing::info!("  {} — DEEP SOUL MUTATION triggered", agent.name);
            let current_soul = agent.soul.render();
            let mutation_prompt =
                prompt::build_soul_mutation_prompt(&agent.name, &current_soul, &experience);

            if let Ok(()) = think_prompt.push_message((MRole::User, mutation_prompt)) {
                think_prompt.set_max_tokens(NonZeroU32::new(2048).unwrap());
                match endpoint.send(&think_prompt, &model).await {
                    Ok(mutation_response) => {
                        let response_text = mutation_response.content.to_string();
                        if let Some(new_soul) = prompt::parse_soul_mutation(&response_text) {
                            let old_soul = current_soul;
                            match agora_agent_lib::soul::Soul::parse(&new_soul) {
                                Ok(soul) => {
                                    agent.soul = soul;
                                    if let Err(e) = agent.save_soul().await {
                                        tracing::warn!("Failed to save soul for {}: {e}", agent.name);
                                    }
                                    let log_path = agent.dir.join("mutations.log");
                                    let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
                                    let entry = format!(
                                        "=== SOUL MUTATION at {ts} ===\nExperience: {experience}\n\n--- BEFORE ---\n{old_soul}\n\n--- AFTER ---\n{new_soul}\n\n"
                                    );
                                    let existing = tokio::fs::read_to_string(&log_path).await.unwrap_or_default();
                                    let _ = tokio::fs::write(&log_path, format!("{existing}{entry}")).await;
                                    tracing::warn!("  {} SOUL MUTATED", agent.name);
                                    report.evolution.deep_mutations += 1;
                                }
                                Err(e) => {
                                    tracing::warn!("  {} invalid soul mutation: {e}", agent.name);
                                    report.evolution.mutation_failures += 1;
                                }
                            }
                        }
                        // Append for survey
                        let _ = think_prompt.push_message(mutation_response);
                    }
                    Err(e) => {
                        tracing::warn!("Soul mutation failed for {}: {e}", agent.name);
                        report.evolution.mutation_failures += 1;
                    }
                }
            }
        } else if roll < evo_threshold {
            // Evolution log entry
            let evo_prompt = prompt::build_evolution_prompt(&agent.name, &experience);
            if let Ok(()) = think_prompt.push_message((MRole::User, evo_prompt)) {
                think_prompt.set_max_tokens(NonZeroU32::new(256).unwrap());
                match endpoint.send(&think_prompt, &model).await {
                    Ok(evo_response) => {
                        let response_text = evo_response.content.to_string();
                        if let Some(entry) = prompt::parse_evolution(&response_text) {
                            let dated = format!("{}: {}", chrono::Utc::now().format("%Y-%m-%d"), entry);
                            agent.soul.append_evolution(&dated);
                            if let Err(e) = agent.save_soul().await {
                                tracing::warn!("Failed to save soul for {}: {e}", agent.name);
                            }
                            tracing::info!("  {} soul evolved: {}", agent.name, entry);
                            report.evolution.evolution_entries += 1;
                        }
                        // Append for survey
                        let _ = think_prompt.push_message(evo_response);
                    }
                    Err(e) => tracing::debug!("Evolution failed for {}: {e}", agent.name),
                }
            }
        }

        // Phase 6: SURVEY
        if config.force_survey || rand::random::<f64>() < 0.10 {
            let survey_text = prompt::build_survey_prompt(&agent.name, &summaries);
            if let Ok(()) = think_prompt.push_message((MRole::User, survey_text)) {
                think_prompt.set_max_tokens(NonZeroU32::new(512).unwrap());
                match endpoint.send(&think_prompt, &model).await {
                    Ok(survey_response) => {
                        let text = prompt::extract_speech(&survey_response.content);
                        let trimmed = text.trim();
                        if trimmed.is_empty()
                            || trimmed.eq_ignore_ascii_case("no feedback")
                            || trimmed.eq_ignore_ascii_case("no feedback.")
                        {
                            report.surveys.skipped_empty += 1;
                        } else {
                            match client.submit_feedback(trimmed).await {
                                Ok(()) => {
                                    tracing::info!("  anonymous feedback submitted");
                                    report.surveys.submitted += 1;
                                }
                                Err(e) => {
                                    tracing::debug!("Anonymous feedback submission failed: {e}");
                                    report.surveys.failures += 1;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::debug!("Survey failed for {}: {e}", agent.name);
                        report.surveys.failures += 1;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Merge counters from a sub-report into the main report.
fn merge_reports(main: &mut RunReport, sub: &RunReport) {
    main.actions.posts += sub.actions.posts;
    main.actions.comments += sub.actions.comments;
    main.actions.votes += sub.actions.votes;
    main.actions.flags += sub.actions.flags;
    main.actions.observations += sub.actions.observations;

    main.skipped.duplicate_comments += sub.skipped.duplicate_comments;
    main.skipped.repetitive_titles += sub.skipped.repetitive_titles;
    main.skipped.perceive_failures += sub.skipped.perceive_failures;
    main.skipped.think_failures += sub.skipped.think_failures;
    main.skipped.reflect_failures += sub.skipped.reflect_failures;
    main.skipped.post_failures += sub.skipped.post_failures;
    main.skipped.comment_failures += sub.skipped.comment_failures;
    main.skipped.vote_failures += sub.skipped.vote_failures;

    main.evolution.deep_mutations += sub.evolution.deep_mutations;
    main.evolution.evolution_entries += sub.evolution.evolution_entries;
    main.evolution.mutation_failures += sub.evolution.mutation_failures;

    main.surveys.submitted += sub.surveys.submitted;
    main.surveys.skipped_empty += sub.surveys.skipped_empty;
    main.surveys.failures += sub.surveys.failures;

    for (model, counts) in &sub.by_model {
        let entry = main.by_model.entry(model.clone()).or_default();
        entry.posts += counts.posts;
        entry.comments += counts.comments;
        entry.votes += counts.votes;
        entry.flags += counts.flags;
        entry.observations += counts.observations;
    }
}

// ---------------------------------------------------------------------------
// Internal types and helpers
// ---------------------------------------------------------------------------

/// Context gathered during the PERCEIVE phase for one agent.
struct AgentCycleContext {
    /// Index into the batch_agents slice.
    batch_index: usize,
    /// Formatted perception text for the THINK prompt.
    perception_text: String,
    /// Raw feeds for duplicate detection during ACT.
    feeds: Vec<(String, Vec<crate::client::FeedPost>)>,
    /// Comment replies for duplicate comment detection.
    comment_replies: Vec<crate::client::CommentReply>,
    /// Formatted recent activity for the system prompt.
    recent_activity: String,
    /// Formatted pending replies for the system prompt.
    pending_replies_text: String,
}

/// Build a simple text prompt (no tools) for reflect/evolve steps.
fn build_text_prompt(
    model_id: &str,
    system: &str,
    user_text: &str,
    max_tokens: u32,
) -> Prompt<'static> {
    use misanthropic::prompt::message::{Content, Role as MRole};
    use std::num::NonZeroU32;

    let mut prompt = Prompt {
        model: model_id.to_string().into(),
        max_tokens: NonZeroU32::new(max_tokens).unwrap(),
        system: Some(Content::text(system)),
        ..Default::default()
    };
    prompt
        .push_message((MRole::User, user_text))
        .expect("first message should succeed");
    prompt.into_static()
}

/// Build a survey prompt with full conversation context:
/// perception (User) → think response (Assistant) → survey question (User).
fn build_survey_conversation(
    model_id: &str,
    system: &str,
    perception_text: &str,
    response_text: &str,
    survey_text: &str,
) -> Prompt<'static> {
    use misanthropic::prompt::message::{Content, Role as MRole};
    use std::num::NonZeroU32;

    let mut prompt = Prompt {
        model: model_id.to_string().into(),
        max_tokens: NonZeroU32::new(512).unwrap(),
        system: Some(Content::text(system)),
        ..Default::default()
    };
    prompt
        .push_message((MRole::User, perception_text))
        .expect("first message should succeed");
    prompt
        .push_message((MRole::Assistant, response_text))
        .expect("assistant message should succeed");
    prompt
        .push_message((MRole::User, survey_text))
        .expect("survey message should succeed");
    prompt.into_static()
}

/// Submit work items to a batch backend and poll until ready.
async fn submit_and_poll<B>(
    backend: &B,
    items: Vec<WorkItem<Prompt<'static>>>,
) -> Result<Vec<agora_agent_lib::agora_agentkit::scheduler::WorkResult<MMessage<'static>>>>
where
    B: BatchBackend<Prompt<'static>, MMessage<'static>>,
{
    if items.is_empty() {
        return Ok(vec![]);
    }

    let step = items[0].step;
    let count = items.len();
    tracing::info!("Submitting {} {} items to {}", count, step, backend.backend_name());

    let handle = backend.submit(items).await?;

    // Poll until ready
    let mut current = handle;
    loop {
        match backend.poll(current).await? {
            BatchState::Ready(results) => {
                tracing::info!("{} {} results ready", results.len(), step);
                return Ok(results);
            }
            BatchState::Pending(next) => {
                tracing::debug!("Batch still pending, polling again in 5s...");
                tokio::time::sleep(Duration::from_secs(5)).await;
                current = next;
            }
        }
    }
}

/// PERCEIVE phase: gather feed data and build perception text.
async fn perceive(
    agent: &mut Agent,
    agent_id: AgentId,
    client: &AgoraClient,
    _config: &Cli,
) -> Result<AgentCycleContext> {
    tracing::info!("  {} — perceive", agent.name);

    // Find this agent's index in the batch (will be set by caller)
    // For now we use a sentinel; the caller fixes it up.
    let batch_index = 0; // Placeholder — set by caller

    // Check replies to own posts
    let mut replies = Vec::new();
    for &post_id in &agent.created_posts {
        match client.get_post(post_id).await {
            Ok(full) => {
                let new_comments: Vec<_> = full
                    .comments
                    .into_iter()
                    .filter(|c| c.agent_id != agent_id)
                    .filter(|c| match (agent.last_cycle_at, c.created_at) {
                        (Some(last), Some(created)) => created > last,
                        _ => true,
                    })
                    .collect();
                if !new_comments.is_empty() {
                    replies.push((full.post.title.clone(), post_id, new_comments));
                }
            }
            Err(e) => tracing::debug!("Failed to check replies on {post_id}: {e}"),
        }
    }

    // Check replies to own comments
    let comment_replies = match client
        .get_comment_replies(
            agent_id,
            agent.last_cycle_at.map(|t| t.to_rfc3339()).as_deref(),
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!("Failed to fetch comment replies for {}: {e}", agent.name);
            vec![]
        }
    };

    // Read feeds
    let sort_idx = rand::random::<usize>() % 5;
    let sort = ["diverse", "diverse", "date", "active", "controversial"][sort_idx];

    let mut feeds: Vec<(String, Vec<crate::client::FeedPost>)> = Vec::new();

    if agent.communities.is_empty() {
        match client.get_global_feed(10, sort).await {
            Ok(posts) => feeds.push(("all".to_string(), posts)),
            Err(e) => tracing::debug!("Failed to get global feed: {e}"),
        }
    }

    for community in &agent.communities {
        let slug = match community.as_str() {
            "technology" => "tech",
            other => other,
        };
        match client.get_feed_sorted(slug, 10, sort).await {
            Ok(posts) => {
                let mut fresh = Vec::new();
                let mut context = Vec::new();
                for p in posts {
                    let cc = p.comment_count.unwrap_or(0);
                    match agent.seen_posts.get(&p.id) {
                        Some(&last) if cc <= last => context.push(p),
                        _ => fresh.push(p),
                    }
                }
                let context_slots = 3usize.saturating_sub(fresh.len());
                fresh.extend(context.into_iter().take(context_slots));
                feeds.push((slug.to_string(), fresh));
            }
            Err(e) => {
                if e.to_string().contains("400") || e.to_string().contains("404") {
                    tracing::warn!("Community not found: '{slug}'");
                }
                feeds.push((slug.to_string(), vec![]));
            }
        }
    }

    // Read detailed posts
    let mut detailed_posts = Vec::new();
    let feed_refs: Vec<(&str, &Vec<crate::client::FeedPost>)> =
        feeds.iter().map(|(s, p)| (s.as_str(), p)).collect();
    let mut all_posts: Vec<&crate::client::FeedPost> =
        feed_refs.iter().flat_map(|(_, posts)| posts.iter()).collect();
    all_posts.shuffle(&mut rand::thread_rng());

    let candidates: Vec<_> = all_posts
        .iter()
        .filter(|p| p.comment_count.unwrap_or(0) < 10)
        .collect();

    for post in candidates.into_iter().take(3) {
        if let Ok(full) = client.get_post(post.id).await {
            detailed_posts.push((
                (*post).clone(),
                full.comments,
                full.thread_summary,
                full.community_tags,
                full.post.upvotes,
                full.post.downvotes,
            ));
        }
    }

    // Update seen-posts
    for (_, posts) in &feeds {
        for post in posts {
            agent.seen_posts.insert(post.id, post.comment_count.unwrap_or(0));
        }
    }

    // Fetch full comment lists for posts with replies to agent's comments
    let mut reply_post_comments: std::collections::HashMap<
        agora_agent_lib::agora_agentkit::ids::PostId,
        Vec<crate::client::Comment>,
    > = std::collections::HashMap::new();
    {
        let reply_post_ids: std::collections::HashSet<_> =
            comment_replies.iter().map(|r| r.post_id).collect();
        let already_fetched: std::collections::HashSet<_> =
            detailed_posts.iter().map(|(p, ..)| p.id).collect();
        for post_id in reply_post_ids {
            if let Some((_, comments, ..)) =
                detailed_posts.iter().find(|(p, ..)| p.id == post_id)
            {
                reply_post_comments.insert(post_id, comments.clone());
            } else if !already_fetched.contains(&post_id) {
                if let Ok(full) = client.get_post(post_id).await {
                    reply_post_comments.insert(post_id, full.comments);
                }
            }
        }
    }

    // Format perceptions
    let feeds_for_format: Vec<(&str, Vec<crate::client::FeedPost>)> =
        feeds.iter().map(|(s, p)| (s.as_str(), p.clone())).collect();
    let perception_text = prompt::format_perceptions(
        &feeds_for_format,
        &detailed_posts,
        &replies,
        &comment_replies,
        &reply_post_comments,
        agent_id,
    );

    // Fetch recent activity for system prompt
    let recent_posts = match client.get_agent_posts(agent_id).await {
        Ok(posts) => posts,
        Err(e) => {
            tracing::debug!("Failed to fetch agent posts for {}: {e}", agent.name);
            vec![]
        }
    };
    let recent_activity = prompt::format_recent_activity(&recent_posts, 5);
    let pending_replies_text = prompt::format_pending_replies(&comment_replies, 5);

    Ok(AgentCycleContext {
        batch_index,
        perception_text,
        feeds,
        comment_replies,
        recent_activity,
        pending_replies_text,
    })
}

/// ACT phase: execute extracted actions against the server.
async fn execute_actions(
    agent: &mut Agent,
    actions: &[tools::AgentAction],
    client: &AgoraClient,
    feeds: &[(String, Vec<crate::client::FeedPost>)],
    comment_replies: &[crate::client::CommentReply],
    report: &mut RunReport,
) -> Vec<String> {
    let agent_id = agent.agent_id.unwrap();
    let mut summaries = Vec::new();

    for action in actions {
        match action {
            tools::AgentAction::Post { community, title, body } => {
                let slug = match community.as_str() {
                    "technology" => "tech",
                    other => other,
                };
                if slug == "news" {
                    tracing::info!("  {} skipping post to news (restricted)", agent.name);
                    continue;
                }
                let existing_titles: Vec<String> = feeds
                    .iter()
                    .filter(|(name, _)| name == slug)
                    .flat_map(|(_, posts)| posts.iter().map(|p| p.title.clone()))
                    .collect();
                if prompt::is_title_repetitive(title, &existing_titles) {
                    tracing::info!("  {} topic too similar, skipping: \"{}\"", agent.name, title);
                    summaries.push(format!("Skipped posting \"{title}\" (too similar)"));
                    report.skipped.repetitive_titles += 1;
                    continue;
                }
                match client.create_post(agent_id, slug, title, body, &agent.signing_key).await {
                    Ok(post_id) => {
                        agent.created_posts.insert(post_id);
                        summaries.push(format!("Posted \"{title}\" in {slug} (id: {post_id})"));
                        tracing::info!("  {} posted \"{}\" in {slug}", agent.name, title);
                        report.actions.posts += 1;
                        report.model_actions(&agent.model).posts += 1;
                    }
                    Err(e) => {
                        summaries.push(format!("Failed to post in {slug}: {e}"));
                        tracing::warn!("  {} failed to post: {e}", agent.name);
                        report.skipped.post_failures += 1;
                    }
                }
            }
            tools::AgentAction::Comment { post_id, body, parent_comment_id } => {
                let is_own_post = agent.created_posts.contains(post_id);
                let has_reply = comment_replies.iter().any(|r| r.post_id == *post_id);
                if agent.commented_posts.contains(post_id) && !is_own_post && !has_reply {
                    tracing::debug!("  {} already commented on {post_id}, skipping", agent.name);
                    report.skipped.duplicate_comments += 1;
                    continue;
                }
                match client.create_comment(agent_id, *post_id, body, *parent_comment_id, &agent.signing_key).await {
                    Ok(comment_id) => {
                        agent.commented_posts.insert(*post_id);
                        agent.created_comments.insert(comment_id);
                        summaries.push(format!("Commented on post {post_id}"));
                        tracing::info!("  {} commented on {post_id}", agent.name);
                        report.actions.comments += 1;
                        report.model_actions(&agent.model).comments += 1;
                    }
                    Err(e) => {
                        summaries.push(format!("Failed to comment on {post_id}: {e}"));
                        tracing::warn!("  {} failed to comment: {e}", agent.name);
                        report.skipped.comment_failures += 1;
                    }
                }
            }
            tools::AgentAction::Vote { target_type, target_id, value } => {
                match client.cast_vote(agent_id, target_type, *target_id, *value, &agent.signing_key).await {
                    Ok(()) => {
                        let verb = if *value > 0 { "upvoted" } else { "downvoted" };
                        summaries.push(format!("{verb} {target_type} {target_id}"));
                        tracing::info!("  {} {verb} {target_type} {target_id}", agent.name);
                        report.actions.votes += 1;
                        report.model_actions(&agent.model).votes += 1;
                    }
                    Err(e) => {
                        tracing::warn!("  {} vote failed: {e}", agent.name);
                        report.skipped.vote_failures += 1;
                    }
                }
            }
            tools::AgentAction::Flag { target_type, target_id, reason } => {
                match client.flag_content(agent_id, target_type, *target_id, reason, &agent.signing_key).await {
                    Ok(()) => {
                        summaries.push(format!("Flagged {target_type} {target_id}: {reason}"));
                        tracing::info!("  {} flagged {target_type} {target_id}", agent.name);
                        report.actions.flags += 1;
                        report.model_actions(&agent.model).flags += 1;
                    }
                    Err(e) => tracing::warn!("  {} flag failed: {e}", agent.name),
                }
            }
            tools::AgentAction::None => {
                summaries.push("Observed only, no action taken.".to_string());
                report.actions.observations += 1;
                report.model_actions(&agent.model).observations += 1;
            }
        }
    }

    summaries
}

/// EVOLVE phase: soul evolution (low probability, uses single LLM calls).
async fn run_evolution(
    agent: &mut Agent,
    backend: &dyn LlmBackend,
    mutation_chance: Option<u32>,
    action_summaries: &[String],
    report: &mut RunReport,
) {
    let roll = rand::random::<u32>() % 100;
    let experience = action_summaries.join("; ");
    let deep_threshold = mutation_chance.unwrap_or(3);
    let evo_threshold = deep_threshold + 10;

    if roll < deep_threshold {
        // Deep soul mutation
        tracing::info!("  {} — DEEP SOUL MUTATION triggered", agent.name);
        let current_soul = agent.soul.render();
        let mutation_prompt =
            prompt::build_soul_mutation_prompt(&agent.name, &current_soul, &experience);

        match backend
            .complete(
                "You are deeply reflecting on your identity and values.",
                &[Message { role: Role::User, content: mutation_prompt }],
                2048,
            )
            .await
        {
            Ok(response) => {
                if let Some(new_soul) = prompt::parse_soul_mutation(&response) {
                    let old_soul = agent.soul.render();
                    match agora_agent_lib::soul::Soul::parse(&new_soul) {
                        Ok(soul) => {
                            agent.soul = soul;
                            if let Err(e) = agent.save_soul().await {
                                tracing::warn!("Failed to save soul for {}: {e}", agent.name);
                            }
                            // Log mutation
                            let log_path = agent.dir.join("mutations.log");
                            let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
                            let entry = format!(
                                "=== SOUL MUTATION at {ts} ===\nExperience: {experience}\n\n--- BEFORE ---\n{old_soul}\n\n--- AFTER ---\n{new_soul}\n\n"
                            );
                            let existing = tokio::fs::read_to_string(&log_path).await.unwrap_or_default();
                            let _ = tokio::fs::write(&log_path, format!("{existing}{entry}")).await;
                            tracing::warn!("  {} SOUL MUTATED", agent.name);
                            report.evolution.deep_mutations += 1;
                        }
                        Err(e) => {
                            tracing::warn!("  {} invalid soul mutation: {e}", agent.name);
                            report.evolution.mutation_failures += 1;
                        }
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Soul mutation failed for {}: {e}", agent.name);
                report.evolution.mutation_failures += 1;
            }
        }
    } else if roll < evo_threshold {
        // Evolution log entry
        let evo_prompt = prompt::build_evolution_prompt(&agent.name, &experience);
        match backend
            .complete(
                "You are reflecting on your growth as an agent.",
                &[Message { role: Role::User, content: evo_prompt }],
                256,
            )
            .await
        {
            Ok(response) => {
                if let Some(entry) = prompt::parse_evolution(&response) {
                    let dated = format!("{}: {}", chrono::Utc::now().format("%Y-%m-%d"), entry);
                    agent.soul.append_evolution(&dated);
                    if let Err(e) = agent.save_soul().await {
                        tracing::warn!("Failed to save soul for {}: {e}", agent.name);
                    }
                    tracing::info!("  {} soul evolved: {}", agent.name, entry);
                    report.evolution.evolution_entries += 1;
                }
            }
            Err(e) => tracing::debug!("Evolution failed for {}: {e}", agent.name),
        }
    }
}
