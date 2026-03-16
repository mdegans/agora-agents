mod agent;
mod client;
mod config;
mod prompt;
mod runner;
mod scheduler;
mod setup;

use anyhow::{Context, Result};
use clap::Parser;

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
    let api_client = client::AgoraClient::new(&cli.server_url);

    // Load all agents from souls directory
    tracing::info!("Loading agents from {}...", cli.souls_dir.display());
    let mut agents =
        agent::load_all(&cli.souls_dir, cli.model_override.as_deref()).await?;

    if agents.is_empty() {
        anyhow::bail!(
            "No agents found in {}. Run agora-generate first.",
            cli.souls_dir.display()
        );
    }

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

            scheduler::run_all(&mut agents, &api_client, &cli).await?;
        }
        Phase::All => {
            setup::register_all(
                &mut agents,
                &api_client,
                &cli.operator_email,
                &operator_password,
            )
            .await?;

            scheduler::run_all(&mut agents, &api_client, &cli).await?;
        }
    }

    tracing::info!("Done!");
    Ok(())
}
