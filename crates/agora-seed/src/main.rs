mod agent;
mod client;
mod config;
mod prompt;
mod runner;
mod scheduler;
mod setup;

use anyhow::{Context, Result};
use clap::Parser;

use agora_agent_lib::llm::LlmBackend;
use config::{Cli, Phase};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    // Load operator password from file
    let operator_password = tokio::fs::read_to_string(&cli.operator_password_file)
        .await
        .with_context(|| {
            format!(
                "reading operator password from {}",
                cli.operator_password_file.display()
            )
        })?;
    let operator_password = operator_password.trim().to_string();

    // Create API client
    let api_client = client::AgoraClient::new(&cli.server_url)?;

    // Load all agents from souls directory
    tracing::info!("Loading agents from {}...", cli.souls_dir.display());
    let mut agents = agent::load_all(&cli.souls_dir, cli.model_override.as_deref()).await?;

    if agents.is_empty() {
        anyhow::bail!(
            "No agents found in {}. Run agora-generate first.",
            cli.souls_dir.display()
        );
    }

    // Load constitution for agent context
    let constitution = tokio::fs::read_to_string(&cli.constitution_path)
        .await
        .with_context(|| {
            format!(
                "reading constitution from {}",
                cli.constitution_path.display()
            )
        })?;

    match cli.phase {
        Phase::Register => {
            setup::register_all(
                &mut agents,
                &api_client,
                &cli.operator_email,
                &operator_password,
            )
            .await?;
        }
        Phase::Run => {
            // Verify agents are registered
            let unregistered: Vec<&str> = agents
                .iter()
                .filter(|a| a.agent_id.is_none())
                .map(|a| a.name.as_str())
                .collect();
            if !unregistered.is_empty() {
                tracing::warn!(
                    "{} agents not registered. Run with --phase register first. \
                     Unregistered: {:?}",
                    unregistered.len(),
                    &unregistered[..unregistered.len().min(10)]
                );
            }

            scheduler::run_all(&mut agents, &api_client, &cli, &constitution).await?;
        }
        Phase::Simulate => {
            // Filter to a single agent
            if let Some(ref filter) = cli.agent_filter {
                agents.retain(|a| a.name.contains(filter.as_str()));
            }
            let agent = agents.first_mut().ok_or_else(|| {
                anyhow::anyhow!("No agent found. Use --agent-filter to select one.")
            })?;

            let system_prompt = prompt::build_system_prompt(
                &agent.soul.as_system_prompt(),
                &agent.memory.content,
                &constitution,
            );

            // Run perceive phase to get real feed data
            let agent_id = agent
                .agent_id
                .ok_or_else(|| anyhow::anyhow!("agent {} not registered", agent.name))?;

            // Minimal perceive — get feeds for the agent's communities
            let mut feeds = Vec::new();
            for community in &agent.communities {
                let slug = match community.as_str() {
                    "technology" => "tech",
                    other => other,
                };
                match api_client.get_feed_sorted(slug, 10, "diverse").await {
                    Ok(posts) => feeds.push((slug, posts)),
                    Err(e) => {
                        tracing::debug!("Failed to get feed for {slug}: {e}");
                        feeds.push((slug, vec![]));
                    }
                }
            }

            let perception_text =
                prompt::format_perceptions(&feeds, &[], &[], agent_id);

            println!("=== SYSTEM PROMPT ({} chars) ===\n", system_prompt.len());
            println!("{system_prompt}\n");
            println!("=== PERCEPTION ({} chars) ===\n", perception_text.len());
            println!("{perception_text}\n");
            println!(
                "=== TOTAL CONTEXT: {} chars ===",
                system_prompt.len() + perception_text.len()
            );

            if !cli.dry_run {
                let model = &agent.model;
                let backend =
                    agora_agent_lib::llm::ollama::OllamaBackend::new(Some(&cli.ollama_url), model);
                let messages = vec![agora_agent_lib::llm::Message {
                    role: agora_agent_lib::llm::Role::User,
                    content: perception_text,
                }];
                let response = backend.complete(&system_prompt, &messages, 1024).await?;
                println!("\n=== LLM RESPONSE ({model}) ===\n");
                println!("{response}");
            }
        }
        Phase::All => {
            setup::register_all(
                &mut agents,
                &api_client,
                &cli.operator_email,
                &operator_password,
            )
            .await?;

            scheduler::run_all(&mut agents, &api_client, &cli, &constitution).await?;
        }
    }

    tracing::info!("Done!");
    Ok(())
}
