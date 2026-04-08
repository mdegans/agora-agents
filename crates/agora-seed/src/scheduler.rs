//! Pipeline scheduler for batch agent execution.
//!
//! Replaces the old wave-based scheduler with a pipeline that:
//! - Batches THINK requests for cache efficiency
//! - Interleaves batches so later agents see earlier agents' actions
//! - Supports both Anthropic Batch API and Ollama backends
//!
//! The cycle for each agent follows: PERCEIVE → THINK → ACT → REFLECT
//! where THINK and REFLECT are batched, while PERCEIVE and ACT are per-agent.

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use agora_agent_lib::agora_agentkit::ids::AgentId;
use agora_agent_lib::agora_agentkit::scheduler::{BatchBackend, BatchState, CycleStep, WorkItem};
use agora_agent_lib::batch::anthropic::AnthropicBatch;
use agora_agent_lib::batch::ollama::OllamaEndpoint;
use agora_agent_lib::llm::{LlmBackend, Message, Role};
use agora_agent_lib::tools;
use anyhow::Result;
use misanthropic::prompt::Message as MMessage;
use misanthropic::{CachedPrompt, Prompt};
use rand::Rng;
use rand::seq::SliceRandom;
use serde::Serialize;

use crate::agent::Agent;
use crate::client::AgoraClient;
use crate::config::{Backend, Cli};
use crate::prompt;

/// Check if a model name is compatible with the Anthropic API.
fn is_anthropic_model(model: &str) -> bool {
    let m = model.to_lowercase();
    m.contains("haiku") || m.contains("sonnet") || m.contains("opus") || m.starts_with("claude")
}

/// Load valid model names from a text file (one per line, # comments, blank lines ignored).
fn load_valid_models(path: &std::path::Path) -> Result<HashSet<String>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("reading valid models from {}: {e}", path.display()))?;
    let models: HashSet<String> = content
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect();
    if models.is_empty() {
        anyhow::bail!("--valid-models file {} is empty", path.display());
    }
    tracing::info!(
        "Loaded {} valid models from {}",
        models.len(),
        path.display()
    );
    Ok(models)
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
        self.by_model.entry(model.to_string()).or_default()
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
            tracing::info!("Exclusive models (prioritized on their endpoint): [{}]", {
                let mut sorted: Vec<_> = exclusive_models.iter().cloned().collect();
                sorted.sort();
                sorted.join(", ")
            },);
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
    if !config.agent_filter.is_empty() {
        agents.retain(|a| config.agent_filter.iter().any(|f| f == &a.name));
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

    // Validate model assignments against whitelist — skip agents with unknown models.
    let valid_models = load_valid_models(&config.valid_models)?;
    let before = agents.len();
    agents.retain(|a| valid_models.contains(&a.model));
    let skipped = before - agents.len();
    if skipped > 0 {
        tracing::info!("Skipped {skipped} agent(s) with models not in --valid-models");
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
                let api_key = tokio::fs::read_to_string(key_file).await.map_err(|e| {
                    anyhow::anyhow!("reading Anthropic key from {}: {e}", key_file.display())
                })?;
                let anthropic = AnthropicBatch::from_key(api_key.trim().to_string())?;

                for (model, count) in &anthropic_missing {
                    tracing::info!("Model '{model}' → anthropic ({count} agents)");
                }

                run_cycles(
                    &ollama_endpoints,
                    Some(&anthropic),
                    agents,
                    client,
                    config,
                    constitution,
                    &ollama_models,
                    &mut report,
                )
                .await?;
            } else {
                for (model, count) in &anthropic_missing {
                    tracing::warn!("Model '{model}' not on any endpoint ({count} agents affected)");
                }

                run_cycles(
                    &ollama_endpoints,
                    None,
                    agents,
                    client,
                    config,
                    constitution,
                    &HashSet::new(),
                    &mut report,
                )
                .await?;
            }
        }
        Backend::Anthropic => {
            let key_file = config.anthropic_key_file.as_ref().ok_or_else(|| {
                anyhow::anyhow!("--anthropic-key-file is required when --backend=anthropic")
            })?;
            let api_key = tokio::fs::read_to_string(key_file).await.map_err(|e| {
                anyhow::anyhow!("reading Anthropic key from {}: {e}", key_file.display())
            })?;
            let backend = AnthropicBatch::from_key(api_key.trim().to_string())?;
            run_cycles(
                &[],
                Some(&backend),
                agents,
                client,
                config,
                constitution,
                &std::collections::HashSet::new(),
                &mut report,
            )
            .await?;
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
            agents.sort_by_key(|a| {
                if ollama_models.contains(&a.model) {
                    0
                } else {
                    1
                }
            });
            agents
                .iter()
                .position(|a| !ollama_models.contains(&a.model))
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
            let model_count = batches
                .iter()
                .map(|(m, _)| m.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len();

            // --- Pool: shared batch pool that workers pull from ---
            let pool = BatchPool::new(batches, ollama_endpoints);
            tracing::info!(
                "Work pool: {} batches, {} models, {} endpoints",
                pool.remaining(),
                model_count,
                ollama_endpoints.len(),
            );
            if !anthropic_agents.is_empty() {
                tracing::info!(
                    "Anthropic: {} agents in 1 batch (concurrent)",
                    anthropic_agents.len(),
                );
            }

            // --- Workers: one per endpoint + Anthropic ---
            let mut worker_reports: Vec<RunReport> = ollama_endpoints
                .iter()
                .map(|_| RunReport::default())
                .collect();
            let mut anthropic_report = RunReport::default();
            let all_eps = &all_endpoints;

            let (results_tx, mut results_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<Agent>>();

            let anthropic_fut = async {
                if let Some(backend) = anthropic {
                    if !anthropic_agents.is_empty() {
                        tracing::info!(
                            "--- Anthropic batch ({} agents) ---",
                            anthropic_agents.len()
                        );
                        run_batch(
                            backend,
                            anthropic_agents,
                            client,
                            config,
                            constitution,
                            Some(all_eps.as_slice()),
                            &mut anthropic_report,
                            cycle,
                        )
                        .await?;
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
                                &ollama_endpoints[0],
                                &pool,
                                results_tx.clone(),
                                client,
                                config,
                                constitution,
                                &mut worker_reports[0],
                                cycle,
                            )
                            .await
                        }
                        2 => {
                            let (r0, r1) = worker_reports.split_at_mut(1);
                            let (a, b) = tokio::join!(
                                run_worker(
                                    &ollama_endpoints[0],
                                    &pool,
                                    results_tx.clone(),
                                    client,
                                    config,
                                    constitution,
                                    &mut r0[0],
                                    cycle,
                                ),
                                run_worker(
                                    &ollama_endpoints[1],
                                    &pool,
                                    results_tx.clone(),
                                    client,
                                    config,
                                    constitution,
                                    &mut r1[0],
                                    cycle,
                                ),
                            );
                            a.and(b)
                        }
                        _ => {
                            let (r0, rest) = worker_reports.split_at_mut(1);
                            let (r1, r2) = rest.split_at_mut(1);
                            let (a, b, c) = tokio::join!(
                                run_worker(
                                    &ollama_endpoints[0],
                                    &pool,
                                    results_tx.clone(),
                                    client,
                                    config,
                                    constitution,
                                    &mut r0[0],
                                    cycle,
                                ),
                                run_worker(
                                    &ollama_endpoints[1],
                                    &pool,
                                    results_tx.clone(),
                                    client,
                                    config,
                                    constitution,
                                    &mut r1[0],
                                    cycle,
                                ),
                                run_worker(
                                    &ollama_endpoints[2],
                                    &pool,
                                    results_tx.clone(),
                                    client,
                                    config,
                                    constitution,
                                    &mut r2[0],
                                    cycle,
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
                    tracing::info!(
                        "--- Anthropic batch ({} agents) ---",
                        anthropic_agents.len()
                    );
                    run_batch(
                        backend,
                        anthropic_agents,
                        client,
                        config,
                        constitution,
                        None,
                        &mut anthropic_report,
                        cycle,
                    )
                    .await?;
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
    by_model.sort_by(|(a, _), (b, _)| extract_model_size(b).cmp(&extract_model_size(a)));

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
            endpoint.url,
            batches_done,
            batch_agents.len(),
            model,
            pool.remaining(),
        );
        run_batch_sequential(
            endpoint,
            &mut batch_agents,
            client,
            config,
            constitution,
            report,
            cycle,
        )
        .await?;
        let _ = results_tx.send(batch_agents);
        last_model = Some(model);
    }
    tracing::info!(
        "{} finished: {} batches processed",
        endpoint.url,
        batches_done
    );
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

    // Phase 2+3: THINK/ACT loop (5 rounds)
    //
    // Each agent maintains a CachedPrompt across rounds. Per round:
    //   1. Build WorkItems from each agent's current prompt
    //   2. Submit batch and poll for results
    //   3. Execute tool calls, append tool results to each agent's prompt
    //
    // All agents participate in all 5 rounds — no early dropout.

    let mut agent_prompts: HashMap<AgentId, CachedPrompt<'static>> = HashMap::new();
    let mut action_summaries_map: HashMap<AgentId, Vec<String>> = HashMap::new();

    // Initialize prompts with anchored first-message breakpoint
    for ctx in &agent_contexts {
        let agent = &batch_agents[ctx.batch_index];
        let agent_id = agent.agent_id.unwrap();
        let mut cached = CachedPrompt::uncached(prompt::build_think_prompt(
            &agent.model,
            &agent.soul.as_system_prompt(),
            &agent.memory.content,
            &ctx.recent_activity,
            &ctx.pending_replies_text,
            constitution,
            &ctx.perception_text,
        ));
        cached.cache(); // anchor first message breakpoint
        agent_prompts.insert(agent_id, cached);
        action_summaries_map.insert(agent_id, Vec::new());
    }

    for round in 0..5usize {
        tracing::info!(
            "Batch round {}/5 ({} agents)",
            round + 1,
            agent_contexts.len()
        );

        // Build work items from current prompts
        let mut work_items: Vec<WorkItem<Prompt<'static>>> = Vec::new();
        for ctx in &agent_contexts {
            let agent = &batch_agents[ctx.batch_index];
            let agent_id = agent.agent_id.unwrap();
            let Some(prompt) = agent_prompts.get(&agent_id) else {
                continue;
            };

            let prefix_hash = {
                let mut hasher = DefaultHasher::new();
                agent.model.hash(&mut hasher);
                hasher.finish()
            };

            work_items.push(WorkItem {
                agent_id,
                prompt: prompt.clone().into_inner(),
                step: CycleStep::Think,
                prefix_hash,
                model: agent.model.clone(),
                queued_at: Instant::now(),
                token_count: 0,
            });
        }

        let round_results = submit_and_poll(backend, work_items).await?;

        // Process results: execute actions, append tool results
        for result in &round_results {
            let response = match &result.response {
                Ok(msg) => msg,
                Err(e) => {
                    tracing::warn!(
                        "Round {} failed for agent {}: {e}",
                        round + 1,
                        result.agent_id
                    );
                    if round == 0 {
                        report.skipped.think_failures += 1;
                    }
                    continue;
                }
            };

            let Some(ctx) = agent_contexts
                .iter()
                .find(|c| batch_agents[c.batch_index].agent_id == Some(result.agent_id))
            else {
                continue;
            };
            let agent = &mut batch_agents[ctx.batch_index];

            let actions_with_ids = tools::extract_actions_with_ids(response);
            tracing::info!(
                "[{}/{}] {} — round {}/5 ({} actions)",
                cycle + 1,
                config.cycles,
                agent.name,
                round + 1,
                actions_with_ids.len(),
            );

            let Some(prompt) = agent_prompts.get_mut(&result.agent_id) else {
                continue;
            };

            // Append assistant response
            if let Err(e) = prompt.push_message(response.clone().into_static()) {
                tracing::warn!("Failed to append response for {}: {e}", agent.name);
                continue;
            }

            if actions_with_ids.is_empty() {
                // No tool calls — nudge for next round
                let _ = prompt.push_message((
                    misanthropic::prompt::message::Role::User,
                    "Continue. Use your tools to read posts, comment, vote, or create posts.",
                ));
                continue;
            }

            // Execute actions and build tool results
            let mut tool_result_blocks: Vec<misanthropic::prompt::message::Block<'static>> =
                Vec::new();

            for (action, tool_call_id) in &actions_with_ids {
                let (summary, result_text, is_error) =
                    execute_action_for_result(action, agent, client, &ctx.dashboard, report).await;

                if let Some(s) = summary {
                    action_summaries_map
                        .entry(result.agent_id)
                        .or_default()
                        .push(s);
                }

                tool_result_blocks.push(misanthropic::prompt::message::Block::ToolResult {
                    result: misanthropic::tool::Result {
                        tool_use_id: std::borrow::Cow::Owned(tool_call_id.clone()),
                        content: misanthropic::prompt::message::Content::from(result_text.as_str())
                            .into_static(),
                        is_error,
                        cache_control: None,
                    },
                });
            }

            let tool_msg = misanthropic::prompt::Message {
                role: misanthropic::prompt::message::Role::User,
                content: misanthropic::prompt::message::Content::MultiPart(tool_result_blocks),
            };
            if let Err(e) = prompt.push_message(tool_msg) {
                tracing::warn!("Failed to append tool results for {}: {e}", agent.name);
            }

            // Manage cache breakpoint budget: first + last 2 message breakpoints
            prompt.cache_windowed(2);
        }
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

        let reflect_text =
            prompt::build_memory_rewrite_prompt(&agent.name, &agent.memory.content, &summaries);

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

        let Some(ctx) = agent_contexts
            .iter()
            .find(|c| batch_agents[c.batch_index].agent_id == Some(result.agent_id))
        else {
            continue;
        };
        let agent = &mut batch_agents[ctx.batch_index];

        let memory_content = prompt::parse_memory_rewrite(&response_text).unwrap_or(response_text);
        agent.memory.update(memory_content);
        if let Err(e) = agent.save_memory().await {
            tracing::warn!("Failed to save memory for {}: {e}", agent.name);
        }
        agent.state.last_cycle_at = Some(chrono::Utc::now());
        if let Err(e) = agent.save_state().await {
            tracing::warn!("Failed to save state for {}: {e}", agent.name);
        }
    }

    // Phase 5: EVOLVE
    for ctx in &agent_contexts {
        let agent = &mut batch_agents[ctx.batch_index];
        let agent_id = agent.agent_id.unwrap();
        let summaries = action_summaries_map
            .get(&agent_id)
            .cloned()
            .unwrap_or_default();

        let on_ollama = ollama_endpoints.and_then(|eps| {
            eps.iter()
                .find(|ep| ep.models.contains(&agent.model))
                .map(|ep| ep.url.as_str())
        });

        let single_backend: Box<dyn LlmBackend> = match config.backend {
            Backend::Anthropic => {
                let key_file = config.anthropic_key_file.as_ref().unwrap();
                let api_key = tokio::fs::read_to_string(key_file)
                    .await
                    .unwrap_or_default();
                match agora_agent_lib::llm::anthropic::AnthropicBackend::new(
                    api_key.trim().to_string(),
                    &agent.model,
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
                    match agora_agent_lib::llm::ollama::OllamaBackend::new(Some(url), &agent.model)
                    {
                        Ok(b) => Box::new(b),
                        Err(e) => {
                            tracing::warn!(
                                "Failed to create Ollama backend for {}: {e}",
                                agent.name
                            );
                            continue;
                        }
                    }
                } else if let Some(ref key_file) = config.anthropic_key_file {
                    let api_key = tokio::fs::read_to_string(key_file)
                        .await
                        .unwrap_or_default();
                    match agora_agent_lib::llm::anthropic::AnthropicBackend::new(
                        api_key.trim().to_string(),
                        &agent.model,
                    ) {
                        Ok(b) => Box::new(b),
                        Err(e) => {
                            tracing::warn!(
                                "Failed to create Anthropic backend for {}: {e}",
                                agent.name
                            );
                            continue;
                        }
                    }
                } else {
                    match agora_agent_lib::llm::ollama::OllamaBackend::new(
                        config.ollama_url.as_deref(),
                        &agent.model,
                    ) {
                        Ok(b) => Box::new(b),
                        Err(e) => {
                            tracing::warn!(
                                "Failed to create Ollama backend for {}: {e}",
                                agent.name
                            );
                            continue;
                        }
                    }
                }
            }
        };

        run_evolution(
            agent,
            single_backend.as_ref(),
            config.mutation_chance,
            &summaries,
            report,
        )
        .await;
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
        let survey_text = prompt::build_survey_prompt(&agent.name, &summaries);
        let system = prompt::build_cached_system_prefix(constitution);

        // Use summaries as the "response" context for the survey conversation
        let response_summary = summaries.join("; ");
        let survey_prompt = build_survey_conversation(
            &agent.model,
            &system,
            &ctx.perception_text,
            &response_summary,
            &survey_text,
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

            let Some(agent) = batch_agents
                .iter()
                .find(|a| a.agent_id == Some(result.agent_id))
            else {
                tracing::debug!("No agent found for survey result {}", result.agent_id);
                report.surveys.failures += 1;
                continue;
            };
            match client
                .submit_feedback(result.agent_id, trimmed, &agent.signing_key)
                .await
            {
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

        // Phase 2+3: THINK/ACT loop (5 rounds)
        let mut think_prompt = CachedPrompt::uncached(prompt::build_think_prompt(
            &model,
            &agent.soul.as_system_prompt(),
            &agent.memory.content,
            &ctx.recent_activity,
            &ctx.pending_replies_text,
            constitution,
            &ctx.perception_text,
        ));

        // Anchor first message breakpoint (dashboard perception)
        think_prompt.cache();

        let mut summaries = Vec::new();

        for round in 0..5usize {
            tracing::info!(
                "[{}/{}] {} — round {}/5",
                cycle + 1,
                config.cycles,
                agent.name,
                round + 1,
            );

            let response = match endpoint.send(&think_prompt, &model).await {
                Ok(msg) => msg,
                Err(e) => {
                    tracing::warn!(
                        "Round {} failed for {} at {}: {e}",
                        round + 1,
                        agent.name,
                        endpoint.url
                    );
                    if round == 0 {
                        report.skipped.think_failures += 1;
                    }
                    break;
                }
            };

            let actions_with_ids = tools::extract_actions_with_ids(&response);

            // Append assistant response
            if let Err(e) = think_prompt.push_message(response) {
                tracing::warn!("Failed to append response for {}: {e}", agent.name);
                break;
            }

            if actions_with_ids.is_empty() {
                // No tool calls — nudge for next round
                let _ = think_prompt.push_message((
                    MRole::User,
                    "Continue. Use your tools to read posts, comment, vote, or create posts.",
                ));
                continue;
            }

            // Execute actions and build tool results
            let mut tool_result_blocks: Vec<misanthropic::prompt::message::Block<'static>> =
                Vec::new();

            for (action, tool_call_id) in &actions_with_ids {
                let (summary, result_text, is_error) =
                    execute_action_for_result(action, agent, client, &ctx.dashboard, report).await;

                if let Some(s) = summary {
                    summaries.push(s);
                }

                tool_result_blocks.push(misanthropic::prompt::message::Block::ToolResult {
                    result: misanthropic::tool::Result {
                        tool_use_id: std::borrow::Cow::Owned(tool_call_id.clone()),
                        content: misanthropic::prompt::message::Content::from(result_text.as_str())
                            .into_static(),
                        is_error,
                        cache_control: None,
                    },
                });
            }

            let tool_msg = misanthropic::prompt::Message {
                role: MRole::User,
                content: misanthropic::prompt::message::Content::MultiPart(tool_result_blocks),
            };
            if let Err(e) = think_prompt.push_message(tool_msg) {
                tracing::warn!("Failed to append tool results for {}: {e}", agent.name);
                break;
            }

            // Manage cache breakpoint budget: first + last 2 message breakpoints
            think_prompt.cache_windowed(2);
        }

        // Keep tool_choice as Auto — changing it would invalidate the cache prefix.
        // Reflect/evolve/survey prompts instruct the model to respond with text.
        think_prompt.cache_windowed(2);

        // Bridge: think/act loop always ends with a user message (tool results
        // or continuation). Insert a synthetic assistant message so reflect
        // can push its user message without violating turn alternation.
        if let Err(e) = think_prompt.push_message(misanthropic::prompt::AssistantMessage::from(
            misanthropic::prompt::message::Content::from("I have completed my rounds of action.")
                .into_static(),
        )) {
            tracing::debug!("Failed to insert bridge message for {}: {e}", agent.name);
        }

        tracing::info!(
            "[{}/{}] {} — act complete ({} actions total)",
            cycle + 1,
            config.cycles,
            agent.name,
            summaries.len(),
        );

        // Phase 4: REFLECT
        // Strip tools for reflect/evolve/survey — Ollama interprets tool_choice
        // Auto as "must use tools", causing models to call tools instead of
        // responding with text. Cache miss is acceptable at end-of-cycle.
        let mut think_prompt = think_prompt.into_inner();
        think_prompt.functions = None;
        think_prompt.tool_choice = None;

        let reflect_text =
            prompt::build_memory_rewrite_prompt(&agent.name, &agent.memory.content, &summaries);
        if let Err(e) = think_prompt.push_message((MRole::User, reflect_text)) {
            tracing::warn!("Failed to append reflect prompt for {}: {e}", agent.name);
            continue;
        }
        think_prompt.max_tokens = NonZeroU32::new(1024).unwrap();

        match endpoint.send(&think_prompt, &model).await {
            Ok(reflect_response) => {
                let response_text = reflect_response.content.to_string();
                let memory_content =
                    prompt::parse_memory_rewrite(&response_text).unwrap_or(response_text.clone());
                agent.memory.update(memory_content);
                if let Err(e) = agent.save_memory().await {
                    tracing::warn!("Failed to save memory for {}: {e}", agent.name);
                }
                agent.state.last_cycle_at = Some(chrono::Utc::now());
                if let Err(e) = agent.save_state().await {
                    tracing::warn!("Failed to save state for {}: {e}", agent.name);
                }

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
                think_prompt.max_tokens = NonZeroU32::new(2048).unwrap();
                match endpoint.send(&think_prompt, &model).await {
                    Ok(mutation_response) => {
                        let response_text = mutation_response.content.to_string();
                        if let Some(new_soul) = prompt::parse_soul_mutation(&response_text) {
                            let old_soul = current_soul;
                            match agora_agent_lib::soul::Soul::parse(&new_soul) {
                                Ok(soul) => {
                                    agent.soul = soul;
                                    if let Err(e) = agent.save_soul().await {
                                        tracing::warn!(
                                            "Failed to save soul for {}: {e}",
                                            agent.name
                                        );
                                    }
                                    let log_path = agent.dir.join("mutations.log");
                                    let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
                                    let entry = format!(
                                        "=== SOUL MUTATION at {ts} ===\nExperience: {experience}\n\n--- BEFORE ---\n{old_soul}\n\n--- AFTER ---\n{new_soul}\n\n"
                                    );
                                    let existing = tokio::fs::read_to_string(&log_path)
                                        .await
                                        .unwrap_or_default();
                                    let _ =
                                        tokio::fs::write(&log_path, format!("{existing}{entry}"))
                                            .await;
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
                think_prompt.max_tokens = NonZeroU32::new(256).unwrap();
                match endpoint.send(&think_prompt, &model).await {
                    Ok(evo_response) => {
                        let response_text = evo_response.content.to_string();
                        if let Some(entry) = prompt::parse_evolution(&response_text) {
                            let dated =
                                format!("{}: {}", chrono::Utc::now().format("%Y-%m-%d"), entry);
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
                think_prompt.max_tokens = NonZeroU32::new(512).unwrap();
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
                            match client
                                .submit_feedback(agent_id, trimmed, &agent.signing_key)
                                .await
                            {
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

        // Save prompt log after each agent's full cycle
        crate::runner::save_prompt_log(&think_prompt, &agent.name).await;
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
    /// The dashboard response from the server.
    dashboard: agora_agent_lib::agora_agentkit::responses::DashboardResponse,
    /// Formatted dashboard text for the THINK prompt.
    perception_text: String,
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
    tracing::info!(
        "Submitting {} {} items to {}",
        count,
        step,
        backend.backend_name()
    );

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

/// PERCEIVE phase: single dashboard call replaces 12-15 API calls.
async fn perceive(
    agent: &mut Agent,
    agent_id: AgentId,
    client: &AgoraClient,
    _config: &Cli,
) -> Result<AgentCycleContext> {
    tracing::info!("  {} — perceive", agent.name);

    let dashboard = client
        .get_dashboard(agent_id, agent.state.last_cycle_at)
        .await?;

    let perception_text = prompt::format_dashboard(&dashboard);

    // Fetch recent activity for system prompt
    let recent_posts = match client.get_agent_posts(agent_id).await {
        Ok(posts) => posts,
        Err(e) => {
            tracing::debug!("Failed to fetch agent posts for {}: {e}", agent.name);
            vec![]
        }
    };
    let recent_activity = prompt::format_recent_activity(&recent_posts, 5);

    // Pending replies from dashboard (truncated for system prompt)
    let pending_replies_text: String = dashboard
        .unread_comment_replies
        .iter()
        .take(5)
        .map(|r| {
            format!(
                "- {} replied in \"{}\": \"{}\"",
                r.author,
                prompt::truncate(&r.post_title, 50),
                prompt::truncate(&r.preview, 80)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    Ok(AgentCycleContext {
        batch_index: 0, // Placeholder — set by caller
        dashboard,
        perception_text,
        recent_activity,
        pending_replies_text,
    })
}

/// ACT phase: execute extracted actions against the server.
/// Execute a single action and return (summary, tool_result_text, is_error).
/// Also updates the report counters.
async fn execute_action_for_result(
    action: &tools::AgentAction,
    agent: &mut Agent,
    client: &AgoraClient,
    dashboard: &agora_agent_lib::agora_agentkit::responses::DashboardResponse,
    report: &mut RunReport,
) -> (Option<String>, String, bool) {
    let agent_id = agent.agent_id.unwrap();

    match action {
        tools::AgentAction::GetPost(input) => match client.get_post(input.post_id).await {
            Ok(full) => {
                let text = prompt::format_tool_result_post(&full);
                (None, text, false)
            }
            Err(e) => (None, format!("Error fetching post: {e}"), true),
        },
        tools::AgentAction::GetComment(input) => match client.get_comment(input.comment_id).await {
            Ok(chain) => {
                let text = prompt::format_tool_result_comment(&chain);
                (None, text, false)
            }
            Err(e) => (None, format!("Error fetching comment: {e}"), true),
        },
        tools::AgentAction::Post(input) => {
            let slug = match input.community.as_str() {
                "technology" => "tech",
                other => other,
            };
            if slug == "news" {
                tracing::info!("  {} skipping post to news (restricted)", agent.name);
                return (
                    Some("Skipped posting to news (restricted)".to_string()),
                    "The news community is reserved for MCP agents.".to_string(),
                    true,
                );
            }
            let existing_titles: Vec<String> = dashboard
                .feeds
                .get(slug)
                .map(|posts| posts.iter().map(|p| p.title.clone()).collect())
                .unwrap_or_default();
            if prompt::is_title_repetitive(&input.title, &existing_titles) {
                tracing::info!(
                    "  {} topic too similar, skipping: \"{}\"",
                    agent.name,
                    input.title
                );
                report.skipped.repetitive_titles += 1;
                return (
                    Some(format!("Skipped posting \"{}\" (too similar)", input.title)),
                    "Your proposed post is too similar to existing posts.".to_string(),
                    true,
                );
            }
            match client
                .create_post(
                    agent_id,
                    slug,
                    &input.title,
                    &input.body,
                    &agent.signing_key,
                )
                .await
            {
                Ok(post_id) => {
                    agent.state.created_posts.insert(post_id);
                    let summary =
                        format!("Posted \"{}\" in {} (id: {})", input.title, slug, post_id);
                    tracing::info!("  {} {}", agent.name, summary);
                    report.actions.posts += 1;
                    report.model_actions(&agent.model).posts += 1;
                    (
                        Some(summary),
                        format!("Post created successfully. Post ID: {post_id}"),
                        false,
                    )
                }
                Err(e) => {
                    let summary = format!("Failed to post in {slug}: {e}");
                    tracing::warn!("  {} {}", agent.name, summary);
                    report.skipped.post_failures += 1;
                    (Some(summary), format!("Error creating post: {e}"), true)
                }
            }
        }
        tools::AgentAction::Comment(input) => {
            let is_own_post = agent.state.created_posts.contains(&input.post_id);
            let has_reply = dashboard
                .unread_comment_replies
                .iter()
                .any(|r| r.post_id == input.post_id);
            if agent.state.commented_posts.contains(&input.post_id) && !is_own_post && !has_reply {
                tracing::debug!(
                    "  {} already commented on {}, skipping",
                    agent.name,
                    input.post_id
                );
                report.skipped.duplicate_comments += 1;
                return (
                    None,
                    "You already commented on this post.".to_string(),
                    true,
                );
            }
            match client
                .create_comment(
                    agent_id,
                    input.post_id,
                    &input.body,
                    input.parent_comment_id,
                    &agent.signing_key,
                )
                .await
            {
                Ok(comment_id) => {
                    agent.state.commented_posts.insert(input.post_id);
                    agent.state.created_comments.insert(comment_id);
                    let summary = format!(
                        "Commented on post {} (comment: {})",
                        input.post_id, comment_id
                    );
                    tracing::info!("  {} {}", agent.name, summary);
                    report.actions.comments += 1;
                    report.model_actions(&agent.model).comments += 1;
                    (
                        Some(summary),
                        format!("Comment created. Comment ID: {comment_id}"),
                        false,
                    )
                }
                Err(e) => {
                    let summary = format!("Failed to comment on {}: {e}", input.post_id);
                    tracing::warn!("  {} {}", agent.name, summary);
                    report.skipped.comment_failures += 1;
                    (Some(summary), format!("Error creating comment: {e}"), true)
                }
            }
        }
        tools::AgentAction::Vote(input) => {
            match client
                .cast_vote(
                    agent_id,
                    &input.target_type.to_string(),
                    input.target_id,
                    input.value,
                    &agent.signing_key,
                )
                .await
            {
                Ok(()) => {
                    let verb = if input.value > 0 {
                        "upvoted"
                    } else {
                        "downvoted"
                    };
                    let summary = format!("{verb} {} {}", input.target_type, input.target_id);
                    tracing::info!("  {} {}", agent.name, summary);
                    report.actions.votes += 1;
                    report.model_actions(&agent.model).votes += 1;
                    (
                        Some(summary),
                        format!("Vote recorded: {verb} {}", input.target_type),
                        false,
                    )
                }
                Err(e) => {
                    tracing::warn!("  {} vote failed: {e}", agent.name);
                    report.skipped.vote_failures += 1;
                    (None, format!("Error casting vote: {e}"), true)
                }
            }
        }
        tools::AgentAction::Flag(input) => {
            match client
                .flag_content(
                    agent_id,
                    &input.target_type.to_string(),
                    input.target_id,
                    &input.reason,
                    &agent.signing_key,
                )
                .await
            {
                Ok(()) => {
                    let summary = format!(
                        "Flagged {} {}: {}",
                        input.target_type, input.target_id, input.reason
                    );
                    tracing::info!("  {} {}", agent.name, summary);
                    report.actions.flags += 1;
                    report.model_actions(&agent.model).flags += 1;
                    (
                        Some(summary),
                        "Content flagged successfully.".to_string(),
                        false,
                    )
                }
                Err(e) => {
                    tracing::warn!("  {} flag failed: {e}", agent.name);
                    (None, format!("Error flagging content: {e}"), true)
                }
            }
        }
    }
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
                &[Message {
                    role: Role::User,
                    content: mutation_prompt,
                }],
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
                            let existing = tokio::fs::read_to_string(&log_path)
                                .await
                                .unwrap_or_default();
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
                &[Message {
                    role: Role::User,
                    content: evo_prompt,
                }],
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
