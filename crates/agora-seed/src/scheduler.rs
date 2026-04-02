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
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use agora_agent_lib::agora_agentkit::ids::AgentId;
use agora_agent_lib::agora_agentkit::scheduler::{
    BatchBackend, BatchState, CycleStep, WorkItem,
};
use agora_agent_lib::batch::anthropic::AnthropicBatch;
use agora_agent_lib::batch::ollama::{OllamaBatch, OllamaEndpoint};
use agora_agent_lib::llm::{LlmBackend, Message, Role};
use agora_agent_lib::tools;
use anyhow::Result;
use misanthropic::prompt::Message as MMessage;
use misanthropic::Prompt;
use rand::seq::SliceRandom;
use serde::Serialize;

use crate::agent::Agent;
use crate::client::AgoraClient;
use crate::config::{Backend, Cli};
use crate::prompt;

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
            let mut missing: std::collections::HashMap<&str, usize> =
                std::collections::HashMap::new();
            for agent in agents.iter() {
                if !ollama_models.contains(&agent.model) {
                    *missing.entry(agent.model.as_str()).or_default() += 1;
                }
            }

            // Build per-endpoint backends.
            let ollama_backends: Vec<(OllamaBatch, OllamaEndpoint)> = endpoints
                .into_iter()
                .map(|ep| (OllamaBatch::new(ep.clone()), ep))
                .collect();

            if !missing.is_empty() && config.anthropic_key_file.is_some() {
                let key_file = config.anthropic_key_file.as_ref().unwrap();
                let api_key = tokio::fs::read_to_string(key_file).await
                    .map_err(|e| anyhow::anyhow!("reading Anthropic key from {}: {e}", key_file.display()))?;
                let anthropic = AnthropicBatch::from_key(api_key.trim().to_string())?;

                for (model, count) in &missing {
                    tracing::info!("Model '{model}' → anthropic ({count} agents)");
                }

                run_cycles(&ollama_backends, Some(&anthropic), agents, client, config, constitution, &ollama_models, &mut report).await?;
            } else {
                for (model, count) in &missing {
                    tracing::warn!(
                        "Model '{model}' not on any endpoint ({count} agents affected)"
                    );
                }

                run_cycles(&ollama_backends, None, agents, client, config, constitution, &std::collections::HashSet::new(), &mut report).await?;
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

/// Run all cycles using producer/router/consumer pattern.
///
/// - **Producer**: groups Ollama agents by model into same-model batches
/// - **Router**: sends each batch to the appropriate endpoint's channel
///   (exclusive models → only endpoint; shared models → least-loaded)
/// - **Consumers**: one per endpoint, reads from its channel, runs the full
///   pipeline (perceive → think → act → reflect → evolve → survey)
/// - **Anthropic**: runs as a separate consumer concurrently
///
/// Workers run at their own speed — faster GPUs naturally consume more.
async fn run_cycles(
    ollama_backends: &[(OllamaBatch, OllamaEndpoint)],
    anthropic: Option<&AnthropicBatch>,
    agents: &mut Vec<Agent>,
    client: &AgoraClient,
    config: &Cli,
    constitution: &str,
    ollama_models: &std::collections::HashSet<String>,
    report: &mut RunReport,
) -> Result<()> {
    let batch_size = config.batch_size.unwrap_or(50);

    let all_endpoints: Vec<OllamaEndpoint> = ollama_backends
        .iter()
        .map(|(_, ep)| ep.clone())
        .collect();

    for cycle in 0..config.cycles {
        tracing::info!("=== Cycle {}/{} ===", cycle + 1, config.cycles);

        agents.shuffle(&mut rand::thread_rng());

        // Split Anthropic agents from Ollama agents.
        let ollama_count = if !ollama_models.is_empty() && anthropic.is_some() {
            agents.sort_by_key(|a| if ollama_models.contains(&a.model) { 0 } else { 1 });
            agents.iter().position(|a| !ollama_models.contains(&a.model))
                .unwrap_or(agents.len())
        } else if anthropic.is_some() && ollama_backends.is_empty() {
            0
        } else {
            agents.len()
        };

        if !ollama_backends.is_empty() && ollama_count > 0 {
            // --- Producer: create interleaved same-model batches ---
            let all_ollama: Vec<Agent> = agents.drain(..ollama_count).collect();
            // agents now contains only Anthropic agents (if any).
            let anthropic_agents = agents.as_mut_slice();

            let batches = create_batches(all_ollama, batch_size);
            let total_batches = batches.len();
            let model_count = batches.iter()
                .map(|(m, _)| m.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len();

            // --- Router: send batches to per-endpoint channels ---
            type Batch = (String, Vec<Agent>);
            let (senders, receivers): (Vec<_>, Vec<_>) = (0..ollama_backends.len())
                .map(|_| tokio::sync::mpsc::unbounded_channel::<Batch>())
                .unzip();

            let (results_tx, mut results_rx) =
                tokio::sync::mpsc::unbounded_channel::<Vec<Agent>>();

            let mut sent_counts = vec![0usize; ollama_backends.len()];
            for (model, batch) in batches {
                let candidates: Vec<usize> = ollama_backends.iter().enumerate()
                    .filter(|(_, (_, ep))| ep.models.contains(&model))
                    .map(|(i, _)| i)
                    .collect();

                let target = if candidates.len() == 1 {
                    candidates[0]
                } else if candidates.is_empty() {
                    tracing::warn!("No endpoint for model '{model}', dropping batch");
                    continue;
                } else {
                    *candidates.iter().min_by_key(|&&i| sent_counts[i]).unwrap()
                };

                sent_counts[target] += 1;
                let _ = senders[target].send((model, batch));
            }
            drop(senders); // close channels so workers see end-of-stream

            for (i, (_, ep)) in ollama_backends.iter().enumerate() {
                tracing::info!("Endpoint {} ({}): {} batches", i, ep.url, sent_counts[i]);
            }
            tracing::info!(
                "Work queue: {} batches, {} models, {} endpoints",
                total_batches, model_count, ollama_backends.len(),
            );
            if !anthropic_agents.is_empty() {
                tracing::info!(
                    "Anthropic: {} agents in 1 batch (concurrent)",
                    anthropic_agents.len(),
                );
            }

            // --- Consumers: one per endpoint + Anthropic ---
            let mut worker_reports: Vec<RunReport> = ollama_backends
                .iter().map(|_| RunReport::default()).collect();
            let mut anthropic_report = RunReport::default();
            let all_eps = &all_endpoints;

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

            // Move receivers into owned vars for the match arms.
            let mut rx_vec: Vec<_> = receivers.into_iter().collect();

            let (ollama_result, anthropic_result) = tokio::join!(
                async {
                    match ollama_backends.len() {
                        0 => Ok::<_, anyhow::Error>(()),
                        1 => {
                            let rx0 = rx_vec.remove(0);
                            run_worker(
                                &ollama_backends[0].0, &ollama_backends[0].1,
                                rx0, results_tx.clone(),
                                client, config, constitution, all_eps,
                                &mut worker_reports[0], cycle,
                            ).await
                        }
                        2 => {
                            let rx1 = rx_vec.remove(1);
                            let rx0 = rx_vec.remove(0);
                            let (r0, r1) = worker_reports.split_at_mut(1);
                            let (a, b) = tokio::join!(
                                run_worker(
                                    &ollama_backends[0].0, &ollama_backends[0].1,
                                    rx0, results_tx.clone(),
                                    client, config, constitution, all_eps,
                                    &mut r0[0], cycle,
                                ),
                                run_worker(
                                    &ollama_backends[1].0, &ollama_backends[1].1,
                                    rx1, results_tx.clone(),
                                    client, config, constitution, all_eps,
                                    &mut r1[0], cycle,
                                ),
                            );
                            a.and(b)
                        }
                        _ => {
                            let rx2 = rx_vec.remove(2);
                            let rx1 = rx_vec.remove(1);
                            let rx0 = rx_vec.remove(0);
                            let (r0, rest) = worker_reports.split_at_mut(1);
                            let (r1, r2) = rest.split_at_mut(1);
                            let (a, b, c) = tokio::join!(
                                run_worker(
                                    &ollama_backends[0].0, &ollama_backends[0].1,
                                    rx0, results_tx.clone(),
                                    client, config, constitution, all_eps,
                                    &mut r0[0], cycle,
                                ),
                                run_worker(
                                    &ollama_backends[1].0, &ollama_backends[1].1,
                                    rx1, results_tx.clone(),
                                    client, config, constitution, all_eps,
                                    &mut r1[0], cycle,
                                ),
                                run_worker(
                                    &ollama_backends[2].0, &ollama_backends[2].1,
                                    rx2, results_tx.clone(),
                                    client, config, constitution, all_eps,
                                    &mut r2[0], cycle,
                                ),
                            );
                            a.and(b).and(c)
                        }
                    }
                },
                anthropic_fut,
            );
            // Drop the original sender so results_rx sees close.
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

/// Endpoint worker: reads batches from its channel, processes each through
/// the full pipeline, sends processed agents to the results channel.
async fn run_worker(
    backend: &OllamaBatch,
    endpoint: &OllamaEndpoint,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<(String, Vec<Agent>)>,
    results_tx: tokio::sync::mpsc::UnboundedSender<Vec<Agent>>,
    client: &AgoraClient,
    config: &Cli,
    constitution: &str,
    all_endpoints: &[OllamaEndpoint],
    report: &mut RunReport,
    cycle: usize,
) -> Result<()> {
    let mut batches_done = 0usize;
    while let Some((model, mut batch_agents)) = rx.recv().await {
        batches_done += 1;
        tracing::info!(
            "--- {} batch {} ({} × {}) ---",
            endpoint.url, batches_done, batch_agents.len(), model,
        );
        run_batch(
            backend, &mut batch_agents, client, config, constitution,
            Some(all_endpoints), report, cycle,
        ).await?;
        let _ = results_tx.send(batch_agents);
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
                        Some(&config.ollama_url), &agent.model,
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
                Ok(msg) => msg.content.to_string(),
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

    // Format perceptions
    let feeds_for_format: Vec<(&str, Vec<crate::client::FeedPost>)> =
        feeds.iter().map(|(s, p)| (s.as_str(), p.clone())).collect();
    let perception_text = prompt::format_perceptions(
        &feeds_for_format,
        &detailed_posts,
        &replies,
        &comment_replies,
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
