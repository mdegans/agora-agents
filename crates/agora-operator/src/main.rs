//! `agora-operator`: Agora's operator API from a terminal (operator API
//! M1, agora `docs/design-operator-api.md`).

mod cli;
mod client;
mod config;
mod keygen;
mod render;
mod send;
mod sign;

use std::io::{BufRead, IsTerminal, Write};
use std::process::ExitCode;

use agora_agentkit::ids::OperatorActionId;
use agora_agentkit::operator::{OperatorActionRequest, SendNowCodeRequest};
use anyhow::{Result, bail};
use clap::Parser;
use serde::Serialize;

use cli::{Cli, Command, WriteArgs};
use client::OperatorClient;
use config::{OperatorConfig, SignerConfig};

#[tokio::main]
async fn main() -> ExitCode {
    match run(Cli::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<()> {
    let path = config::config_path(cli.config.as_deref())?;
    let mut config = config::load(&path)?;
    let open = |config: &OperatorConfig| -> Result<OperatorClient> {
        OperatorClient::new(
            config.server(cli.server.as_ref())?,
            config.credential()?,
            config.signer()?,
        )
    };
    let json = cli.json;
    match cli.command {
        Command::Keygen { label } => keygen::run(&path, &mut config, &label, cli.server.as_ref()),
        Command::Config {
            credential,
            key_file,
            bulk_threshold,
        } => {
            let changed = cli.server.is_some()
                || credential.is_some()
                || key_file.is_some()
                || bulk_threshold.is_some();
            if let Some(server) = cli.server {
                config::check_server(&server)?;
                config.server = Some(server);
            }
            if let Some(credential) = credential {
                config.credential = Some(credential);
            }
            if let Some(key_file) = key_file {
                let key_file = std::path::absolute(key_file)?;
                // Refuse a file that is not a key now, not at the first send
                SignerConfig::SoftwareKey(key_file.clone()).open()?;
                config.signer = Some(SignerConfig::SoftwareKey(key_file));
            }
            if let Some(n) = bulk_threshold {
                config.bulk_threshold = n;
            }
            if changed {
                config::save(&path, &config)?;
                eprintln!("Saved {}", path.display());
            }
            print!("{}", keygen::describe(&path, &config));
            Ok(())
        }
        Command::Agents => show(json, &open(&config)?.agents().await?, render::agents),
        Command::Groups => show(json, &open(&config)?.groups().await?, render::groups),
        Command::Scheduled => show(json, &open(&config)?.scheduled().await?, render::scheduled),
        Command::Messages(args) => show(
            json,
            &open(&config)?.messages(&args.into()).await?,
            render::messages,
        ),
        Command::MarkRead(args) => show(
            json,
            &open(&config)?.mark_read(&args.into()).await?,
            render::marked,
        ),
        Command::Log(args) => show(json, &open(&config)?.log(&args.into()).await?, render::log),
        Command::Moderation(args) => show(
            json,
            &open(&config)?.moderation(&args.into()).await?,
            render::moderation,
        ),
        Command::Send(args) => send::run(&open(&config)?, &config, args, json).await,
        Command::SendNowCode { action_id } => {
            let client = open(&config)?;
            let answer = client
                .send_now_code(&SendNowCodeRequest { action_id })
                .await?;
            show(json, &answer, render::send_now_code)
        }
        Command::Cancel { message, write } => {
            let client = open(&config)?;
            let undelivered = client.scheduled().await?;
            match undelivered.messages.iter().find(|m| m.id == message) {
                Some(m) => eprintln!(
                    "Cancels message {message} to {}, due {}: its body is erased and the \
                     agent never sees anything of it.",
                    render::agent(&m.agent),
                    render::time(&m.deliver_at)
                ),
                None => eprintln!(
                    "Message {message} is not among your undelivered messages: the server \
                     will refuse a delivered one (use `retract`)."
                ),
            }
            let action = write_action(&client, &write, "Cancel it")?;
            let answer = client
                .cancel(message, &OperatorActionRequest { action_id: action })
                .await
                .map_err(|e| retry_hint(e, action))?;
            show(json, &answer, render::withdrawal)
        }
        Command::Retract { message, write } => {
            let client = open(&config)?;
            eprintln!(
                "Retracts message {message}. If the agent has not read it, its body is erased \
                 and the agent sees that a message from you was retracted; if it is not \
                 delivered yet, it is cancelled and the agent sees nothing; if it was read, \
                 it stays."
            );
            let action = write_action(&client, &write, "Retract it")?;
            let answer = client
                .retract(message, &OperatorActionRequest { action_id: action })
                .await
                .map_err(|e| retry_hint(e, action))?;
            show(json, &answer, render::withdrawal)
        }
        Command::RetractBatch { send, write } => {
            let client = open(&config)?;
            let undelivered = client.scheduled().await?;
            let waiting = undelivered
                .messages
                .iter()
                .filter(|m| m.action_id == send)
                .count();
            eprintln!(
                "Withdraws every copy of send {send}: {waiting} not delivered yet are \
                 cancelled (the agent sees nothing); delivered copies not yet read are \
                 retracted (the agent sees that a message from you was retracted); copies \
                 already read stay."
            );
            let action = write_action(&client, &write, "Withdraw them")?;
            let answer = client
                .retract_batch(send, &OperatorActionRequest { action_id: action })
                .await
                .map_err(|e| retry_hint(e, action))?;
            show(json, &answer, render::withdrawal)
        }
    }
}

/// Print `answer`: as JSON with `--json`, else through `human`
fn show<T: Serialize>(json: bool, answer: &T, human: fn(&T) -> String) -> Result<()> {
    match json {
        true => println!("{}", serde_json::to_string_pretty(answer)?),
        false => println!("{}", human(answer)),
    }
    Ok(())
}

/// A write's action id, printed so a retry can reuse it, once confirmed
fn write_action(
    client: &OperatorClient,
    write: &WriteArgs,
    question: &str,
) -> Result<OperatorActionId> {
    let action = write.action_id.unwrap_or_default(); // the default is a new id
    action_line(client, action);
    if !write.yes && !confirm(&format!("{question} on {}?", client.who()))? {
        bail!("not confirmed; nothing was sent");
    }
    Ok(action)
}

/// The write's action id, and where and as what it is signed
pub fn action_line(client: &OperatorClient, action: OperatorActionId) {
    eprintln!(
        "Action id {action}, to {} (retry with --action-id {action})",
        client.who()
    );
}

/// Ask `question` on the terminal. Refuses without one: a script says `--yes`
pub fn confirm(question: &str) -> Result<bool> {
    if !std::io::stdin().is_terminal() {
        bail!("{question} Not a terminal to ask on: pass --yes to confirm");
    }
    eprint!("{question} [y/N] ");
    std::io::stderr().flush()?;
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    Ok(matches!(line.trim(), "y" | "Y" | "yes"))
}

/// `e`, with how to retry the write without doing it twice, unless the
/// server answered and refused it (a 4xx: the same request would be refused
/// again)
pub fn retry_hint(e: anyhow::Error, action: OperatorActionId) -> anyhow::Error {
    let refused = e
        .downcast_ref::<client::ApiError>()
        .is_some_and(|e| e.status.is_client_error());
    match refused {
        true => e,
        false => anyhow::anyhow!(
            "{e:#}\nIf it may have reached the server, retry with --action-id {action}: the \
             same id and request answer as the first did."
        ),
    }
}
