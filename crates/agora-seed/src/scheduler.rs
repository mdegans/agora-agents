use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::Semaphore;

use agora_agent_lib::llm::ollama::OllamaBackend;

use crate::agent::Agent;
use crate::client::AgoraClient;
use crate::config::Cli;
use crate::runner;

/// Run all agents, grouped by model.
///
/// Local Ollama models run sequentially (one model at a time, GPU VRAM constraint).
/// Remote Ollama models run in parallel with local waves on a separate machine.
pub async fn run_all(
    agents: &mut Vec<Agent>,
    client: &AgoraClient,
    config: &Cli,
) -> Result<()> {
    // Split agents into local and remote groups
    let mut local_agents: Vec<Agent> = Vec::new();
    let mut remote_agents: Vec<Agent> = Vec::new();
    let mut remaining: Vec<Agent> = Vec::new();

    // Drain agents into local/remote groups
    for agent in agents.drain(..) {
        if config.is_remote_model(&agent.model) {
            remote_agents.push(agent);
        } else {
            local_agents.push(agent);
        }
    }

    tracing::info!(
        "Running {} agents: {} local, {} remote, {} cycles each",
        local_agents.len() + remote_agents.len(),
        local_agents.len(),
        remote_agents.len(),
        config.cycles,
    );

    // Spawn remote agents in a background task (runs on different machine)
    let remote_handle = if !remote_agents.is_empty() {
        let remote_url = config
            .remote_ollama_url
            .as_ref()
            .expect("--remote-ollama-url required for remote models")
            .clone();
        let remote_concurrency = config.remote_ollama_concurrency;
        let cycles = config.cycles;
        let client_url = config.server_url.clone();

        Some(tokio::spawn(async move {
            let client = match AgoraClient::new(&client_url) {
                Ok(c) => c,
                Err(e) => return (remote_agents, Err(e)),
            };
            let result = run_waves(
                &mut remote_agents,
                &client,
                &remote_url,
                remote_concurrency,
                cycles,
                "Remote",
            )
            .await;
            (remote_agents, result)
        }))
    } else {
        None
    };

    // Run local waves sequentially
    run_waves(
        &mut local_agents,
        client,
        &config.ollama_url,
        config.ollama_concurrency,
        config.cycles,
        "Local",
    )
    .await?;

    // Collect local agents back
    remaining.extend(local_agents);

    // Wait for remote to finish and collect agents back
    if let Some(handle) = remote_handle {
        match handle.await {
            Ok((remote_back, Ok(()))) => {
                tracing::info!("Remote waves completed successfully");
                remaining.extend(remote_back);
            }
            Ok((remote_back, Err(e))) => {
                tracing::error!("Remote waves failed: {e:#}");
                remaining.extend(remote_back);
            }
            Err(e) => {
                tracing::error!("Remote task panicked: {e}");
            }
        }
    }

    // Put agents back
    *agents = remaining;

    tracing::info!("All waves complete!");
    Ok(())
}

/// Run model waves against a single Ollama server.
/// Models are loaded one at a time; agents for each model run with the given concurrency.
async fn run_waves(
    agents: &mut [Agent],
    client: &AgoraClient,
    ollama_url: &str,
    concurrency: usize,
    cycles: usize,
    label: &str,
) -> Result<()> {
    // Group by model
    let mut model_groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, agent) in agents.iter().enumerate() {
        model_groups
            .entry(agent.model.clone())
            .or_default()
            .push(i);
    }

    let total_waves = model_groups.len();
    for (wave_num, (model_name, agent_indices)) in model_groups.iter().enumerate() {
        tracing::info!(
            "=== {label} wave {}/{total_waves}: {model_name} ({} agents) ===",
            wave_num + 1,
            agent_indices.len()
        );

        preload_model(ollama_url, model_name).await;
        let backend = Arc::new(OllamaBackend::new(Some(ollama_url), model_name));
        let semaphore = Arc::new(Semaphore::new(concurrency));

        // Interleave cycles: run cycle N for ALL agents before cycle N+1.
        // Shuffle agent order each cycle so different agents go first,
        // preventing the same agents from always setting the conversation tone.
        let mut shuffled_indices = agent_indices.clone();
        for cycle in 0..cycles {
            use rand::seq::SliceRandom;
            shuffled_indices.shuffle(&mut rand::thread_rng());
            tracing::info!(
                "--- {label} {model_name} cycle {}/{cycles} ({} agents, shuffled) ---",
                cycle + 1,
                shuffled_indices.len()
            );
            for &agent_idx in &shuffled_indices {
                let _permit = semaphore.acquire().await?;
                let agent = &mut agents[agent_idx];

                if agent.agent_id.is_none() {
                    continue;
                }

                match runner::run_cycle(agent, backend.as_ref(), client, cycle, cycles).await {
                    Ok(()) => {}
                    Err(e) => {
                        tracing::warn!(
                            "Agent {} cycle {}/{cycles} failed: {e:#}",
                            agent.name,
                            cycle + 1,
                        );
                    }
                }
            }
        }

        tracing::info!(
            "=== {label} wave {}/{total_waves} complete ===",
            wave_num + 1
        );
    }
    Ok(())
}

/// Send a trivial request to Ollama to preload a model into VRAM.
async fn preload_model(ollama_url: &str, model: &str) {
    tracing::info!("Preloading Ollama model: {model} at {ollama_url}");

    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": false,
        "options": {"num_predict": 1}
    });

    match client
        .post(format!("{ollama_url}/api/chat"))
        .json(&body)
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            tracing::info!("Model {model} loaded successfully");
        }
        Ok(resp) => {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            tracing::warn!("Preload {model} returned {status}: {text}");
        }
        Err(e) => {
            tracing::warn!("Failed to preload {model}: {e}");
        }
    }
}
