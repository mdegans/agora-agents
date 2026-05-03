mod agent;
mod client;
mod config;
mod prompt;
mod prompt_log;
mod scheduler;
mod setup;
mod state;

use anyhow::{Context, Result};
use clap::Parser;

use config::{Cli, Phase};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
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
    let mut agents = agent::load_all(&cli.souls_dir, cli.model.as_deref()).await?;

    if agents.is_empty() {
        anyhow::bail!(
            "No agents found in {}. Run agora-generate first.",
            cli.souls_dir.display()
        );
    }

    // (Community list comes from `agora_agent_lib::Community::ALL`, codegen'd
    // at build time in agora-agent-lib/build.rs from the live API. No
    // runtime fetch needed.)

    // Resolve models from server for agents that don't have one from --model
    if cli.model.is_none() {
        let unresolved = agent::resolve_models(&mut agents, &api_client).await;
        if !unresolved.is_empty() {
            // Filter out agents with no model — they can't run
            let before = agents.len();
            agents.retain(|a| !a.model.is_empty());
            tracing::warn!(
                "Dropped {} agents with no model (use --model to set a default)",
                before - agents.len()
            );
        }
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
        Phase::Validate => {
            use agora_agent_lib::soul::WarnLevel;

            let mut total_errors = 0u32;
            let mut total_warnings = 0u32;
            let mut agents_with_errors = 0u32;
            let mut agents_with_no_communities = 0u32;

            // Apply agent filter if set
            if !cli.agent_filter.is_empty() {
                agents.retain(|a| cli.agent_filter.iter().any(|f| f == &a.name));
            }

            for agent in &agents {
                let warnings = agent.soul.validate();
                if warnings.is_empty() {
                    continue;
                }

                let has_error = warnings.iter().any(|w| w.level == WarnLevel::Error);
                if has_error {
                    agents_with_errors += 1;
                }
                if warnings.iter().any(|w| {
                    w.level == WarnLevel::Error && w.message.contains("no communities")
                }) {
                    agents_with_no_communities += 1;
                }

                // Print per-agent warnings (errors and warnings only, skip info)
                let significant: Vec<_> = warnings
                    .iter()
                    .filter(|w| w.level != WarnLevel::Info)
                    .collect();
                if !significant.is_empty() {
                    eprintln!("--- {} ---", agent.name);
                    for w in &significant {
                        eprintln!("  {w}");
                    }
                }

                for w in &warnings {
                    match w.level {
                        WarnLevel::Error => total_errors += 1,
                        WarnLevel::Warning => total_warnings += 1,
                        WarnLevel::Info => {}
                    }
                }
            }

            eprintln!();
            eprintln!("=== Validation Summary ===");
            eprintln!("Agents scanned:          {}", agents.len());
            eprintln!("Agents with errors:      {agents_with_errors}");
            eprintln!("  - no communities:      {agents_with_no_communities}");
            eprintln!("Total errors:            {total_errors}");
            eprintln!("Total warnings:          {total_warnings}");

            // Also output communities used across all agents
            let mut community_counts: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();
            for agent in &agents {
                for c in agent.soul.communities() {
                    *community_counts.entry(c).or_default() += 1;
                }
            }
            let mut sorted: Vec<_> = community_counts.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));
            eprintln!();
            eprintln!("=== Community Usage ===");
            for (slug, count) in &sorted {
                // The new typed Soul drops invalid communities at deserialize
                // time (with a warn!), so anything we see here is by definition
                // valid. The legacy reader applies the same filter. We keep
                // the report shape for compatibility but no longer mark slugs.
                eprintln!("  {slug}: {count}");
            }

            if total_errors > 0 {
                std::process::exit(1);
            }
            return Ok(());
        }
        Phase::Fix => {
            anyhow::bail!(
                "Phase::Fix has been replaced. Run `cargo run --bin agora-audit -- migrate` instead."
            );
        }
        Phase::AssignCommunities => {
            anyhow::bail!(
                "Phase::AssignCommunities has been replaced. Run `cargo run --bin agora-audit -- migrate` instead."
            );
        }
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
            if !cli.agent_filter.is_empty() {
                agents.retain(|a| cli.agent_filter.iter().any(|f| f == &a.name));
            }
            let agent = agents.first_mut().ok_or_else(|| {
                anyhow::anyhow!("No agent found. Use --agent-filter to select one.")
            })?;

            if cli.dry_run {
                // Dry run: show the full Prompt that would be sent to the LLM
                let agent_id = agent
                    .agent_id
                    .ok_or_else(|| anyhow::anyhow!("agent {} not registered", agent.name))?;

                let dashboard = api_client
                    .get_dashboard(agent_id, agent.state.last_cycle_at)
                    .await?;
                let dashboard_text = prompt::format_dashboard(&dashboard);

                let think_prompt = prompt::build(
                    &agent.model,
                    &agent.soul.markdown(),
                    &agent.memory.render_for_prompt(),
                    "", // no recent activity in dry-run mode
                    "", // no pending replies in dry-run mode
                    &constitution,
                    &dashboard_text,
                );

                println!("{}", serde_json::to_string_pretty(&think_prompt)?);
                eprintln!(
                    "\n--- Prompt with {} tool(s), tool_choice: {:?} ---",
                    think_prompt.functions.as_ref().map_or(0, |f| f.len()),
                    think_prompt.tool_choice,
                );
            } else {
                // Live single-agent simulation. Instead of a separate
                // verbose code path (which used to live in runner.rs and
                // drifted from the real scheduler over time), run the
                // normal scheduler with a single-agent filter. Set
                // `RUST_LOG=agora_seed=debug` for per-phase request/
                // response logging.
                let _ = agent; // filter was already applied above
                scheduler::run_all(&mut agents, &api_client, &cli, &constitution)
                    .await?;
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
