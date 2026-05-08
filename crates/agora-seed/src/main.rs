use agora_agent_lib::info_payload;
use clap::Parser;

mod agent;
mod client;
mod config;
pub mod log;
mod prompt;
mod prompt_log;
mod scheduler;
mod setup;
mod state;
mod utils;

use config::{Args, Phase};
use serde::Serialize;
pub use utils::constitution;
use utils::{init_constitution, init_logging, read_file_stripped};

use crate::{
    agent::{filter_agents, load_allowed_models},
    utils::assert_clean,
};

/// All phases complete
#[derive(Serialize)]
pub struct Done;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = Args::parse();

    let _guards = init_logging(args.logfile.as_deref())?;
    assert_clean(&args.souls_dir).await?;
    init_constitution().await;

    // Load allowed_models
    let allowed_models = load_allowed_models(&args.allowed_models).await?;

    // Load operator password from file
    // FIXME: Before production, we should be storing and reading the password
    // using the keyring crate, supporting the password file only for legacy.
    // `operator_password_file` can become Option in this case.
    let operator_password = read_file_stripped(&args.operator_password_file).await?;

    // Create API client
    let api_client = client::AgoraClient::new(args.server_url.clone())?;

    // Load all agents from souls directory
    tracing::info!("Loading agents from {}...", args.souls_dir.display());
    let mut agents = agent::load_all(&args.souls_dir).await?;

    if agents.is_empty() {
        anyhow::bail!(
            "No agents found in {}. Run agora-generate first.",
            args.souls_dir.display()
        );
    }

    // (Community list comes from `agora_agent_lib::Community::ALL`, codegen'd
    // at build time in agora-agent-lib/build.rs from the live API. No
    // runtime fetch needed.)

    // Resolve models from server for agents
    agent::resolve_models(&mut agents, &api_client).await;
    filter_agents(
        &mut agents,
        &mut args.allowed_agents,
        &allowed_models,
        !args.phase.is_register(), // require_registered except when Phase::Register
    )?;

    match args.phase {
        Phase::Register => {
            setup::register_all(
                &mut agents,
                &api_client,
                &args.operator_email,
                &operator_password,
            )
            .await?;
        }
        Phase::Run => {
            scheduler::run_all(&mut agents, &api_client, &args).await?;
        }
        Phase::All => {
            setup::register_all(
                &mut agents,
                &api_client,
                &args.operator_email,
                &operator_password,
            )
            .await?;

            scheduler::run_all(&mut agents, &api_client, &args).await?;
        }
    }

    info_payload!(Done);
    Ok(())
}
