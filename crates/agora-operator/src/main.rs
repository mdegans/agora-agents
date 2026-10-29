//! `agora-operator`: Agora's operator API from a terminal (operator API
//! M1, agora `docs/design-operator-api.md`).

mod cli;
mod client;
mod config;
mod keygen;
mod render;
mod send;
mod sign;
#[cfg(feature = "yubikey")]
mod yubikey;

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
    let server_flag = cli.server.clone();
    let sudo = cli.sudo;
    let open_sudo = async |config: &OperatorConfig| -> Result<OperatorClient> {
        let server = config.server(server_flag.as_ref())?;
        let (credential, signer) = config.sudo().await?;
        OperatorClient::new(server, credential, signer)
    };
    let open = async |config: &OperatorConfig| -> Result<OperatorClient> {
        if sudo {
            return open_sudo(config).await;
        }
        let server = config.server(server_flag.as_ref())?;
        let credential = config.credential()?;
        OperatorClient::new(server, credential, config.signer().await?)
    };
    let json = cli.json;
    match cli.command {
        Command::Keygen { label, operator } => {
            keygen::run(&path, &mut config, &label, cli.server.as_ref(), operator).await
        }
        #[cfg(feature = "yubikey")]
        Command::Yubikey(command) => {
            yubikey::run(command, &path, &mut config, cli.server.as_ref()).await
        }
        Command::Prove { operator } => {
            let signer = config.signer().await?;
            println!("{}", keygen::proof(signer.as_ref(), operator).await?);
            Ok(())
        }
        Command::Config {
            credential,
            key_file,
        } => {
            let changed = cli.server.is_some() || credential.is_some() || key_file.is_some();
            if let Some(server) = &cli.server {
                config::check_server(server)?;
                config.server = Some(server.clone());
            }
            if let Some(credential) = credential {
                config.credential = Some(credential);
            }
            if let Some(key_file) = key_file {
                let key_file = std::path::absolute(key_file)?;
                // Refuse a file that is not a key now, not at the first send
                SignerConfig::SoftwareKey(key_file.clone()).open().await?;
                config.signer = Some(SignerConfig::SoftwareKey(key_file));
            }
            if changed {
                config::save(&path, &config)?;
                eprintln!("Saved {}", path.display());
            }
            print!("{}", keygen::describe(&path, &config).await);
            // The server's side of it, when there is enough to ask
            if let Ok(client) = open(&config).await {
                match client.config().await {
                    Ok(settings) => show(json, &settings, render::settings)?,
                    Err(e) => eprintln!("(Could not read the server's settings: {e:#})"),
                }
            }
            Ok(())
        }
        Command::Agents => show(json, &open(&config).await?.agents().await?, render::agents),
        Command::Groups => show(json, &open(&config).await?.groups().await?, render::groups),
        Command::Scheduled => show(
            json,
            &open(&config).await?.scheduled().await?,
            render::scheduled,
        ),
        Command::Messages(args) => show(
            json,
            &open(&config).await?.messages(&args).await?,
            render::messages,
        ),
        Command::MarkRead(args) => show(
            json,
            &open(&config).await?.mark_read(&args).await?,
            render::marked,
        ),
        Command::Log(args) => show(json, &open(&config).await?.log(&args).await?, render::log),
        Command::Moderation(args) => show(
            json,
            &open(&config).await?.moderation(&args).await?,
            render::moderation,
        ),
        Command::Send(args) => {
            // A sudo send previews with the ordinary key, if there is one,
            // so only the send itself waits for a tap
            let (preview, sudo_client) = match sudo {
                false => (open(&config).await?, None),
                true => {
                    let sudo_client = open_sudo(&config).await?;
                    let ordinary = match (config.credential(), config.signer().await) {
                        (Ok(credential), Ok(signer)) => Some(OperatorClient::new(
                            config.server(server_flag.as_ref())?,
                            credential,
                            signer,
                        )?),
                        _ => None,
                    };
                    match ordinary {
                        Some(ordinary) => (ordinary, Some(sudo_client)),
                        None => (sudo_client, None),
                    }
                }
            };
            send::run(&preview, sudo_client.as_ref(), sudo, args, json).await
        }
        Command::SendNowCode { action_id } => {
            let client = open(&config).await?;
            let answer = client
                .send_now_code(&SendNowCodeRequest { action_id })
                .await?;
            show(json, &answer, render::send_now_code)
        }
        Command::Cancel { message, write } => {
            let client = open(&config).await?;
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
            let client = open(&config).await?;
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
            let client = open(&config).await?;
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
