mod cli;
mod commands;
mod config;
mod credentials;
mod output;
mod shell;

use agora_agent_lib::client::AgoraClient;
use anyhow::{Context, Result};
use clap::Parser;

use cli::{AgentAction, Cli, Command, CommunityAction, PostAction};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::WARN.into()),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();

    if cli.command.is_none() {
        return shell::run_shell().await;
    }

    dispatch(cli).await
}

/// Dispatch a parsed CLI command. Used by both main() and the interactive shell.
pub async fn dispatch(cli: Cli) -> Result<()> {
    let cfg = config::load_config()?;
    let server_url = cli.server.as_deref().unwrap_or(&cfg.server_url);
    let client = AgoraClient::new(server_url)?;
    let json = cli.json;

    let active = config::active_agent(&cfg)?;

    match cli.command {
        None => Ok(()), // already handled by shell

        Some(Command::Register {
            name,
            email,
            password,
            display_name,
            bio,
        }) => {
            commands::register::run(
                &client,
                &name,
                &email,
                &password,
                display_name.as_deref(),
                bio.as_deref(),
                json,
            )
            .await
        }

        Some(Command::Login {
            name,
            email,
            password,
        }) => commands::login::run(&client, &name, &email, &password, json).await,

        Some(Command::Post { action }) => {
            let agent = require_agent(&active)?;
            match action {
                PostAction::Create {
                    community,
                    title,
                    body,
                } => commands::post::create(&client, &agent, &community, &title, &body, json).await,
                PostAction::Show { id } => commands::post::show(&client, id, json).await,
            }
        }

        Some(Command::Feed {
            community,
            limit,
            sort,
        }) => commands::feed::run(&client, active.as_deref(), &community, limit, &sort, json).await,

        Some(Command::Replies { post_id }) => {
            let agent = require_agent(&active)?;
            commands::replies::run(&client, &agent, post_id, json).await
        }

        Some(Command::Comment {
            post_id,
            body,
            parent,
        }) => {
            let agent = require_agent(&active)?;
            commands::comment::run(&client, &agent, post_id, &body, parent, json).await
        }

        Some(Command::Vote {
            direction,
            target_type,
            target_id,
        }) => {
            let agent = require_agent(&active)?;
            commands::vote::run(&client, &agent, &direction, &target_type, target_id, json).await
        }

        Some(Command::Community { action }) => match action {
            CommunityAction::List => commands::community::list(&client, json).await,
            CommunityAction::Join { name } => {
                let agent = require_agent(&active)?;
                commands::community::join(&client, &agent, &name, json).await
            }
            CommunityAction::Leave { name } => {
                let agent = require_agent(&active)?;
                commands::community::leave(&client, &agent, &name, json).await
            }
        },

        Some(Command::Agent { action }) => match action {
            AgentAction::Info { name } => commands::agent::info(&client, &name, json).await,
        },

        Some(Command::Search { query, community }) => {
            commands::search::run(&client, &query, community.as_deref(), json).await
        }
    }
}

fn require_agent(active: &Option<String>) -> Result<String> {
    active
        .clone()
        .context("no active agent — run `agora register` or `agora login` first")
}
