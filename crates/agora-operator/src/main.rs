//! `agora-operator`: Agora's operator API from a terminal (operator API
//! M1, agora `docs/design-operator-api.md`).

mod cli;
mod client;
mod config;
mod keygen;
mod render;
mod send;
#[cfg(feature = "yubikey")]
mod yubikey;

use std::io::{BufRead, IsTerminal, Write};
use std::process::ExitCode;

use agora_agentkit::ids::OperatorActionId;
use agora_agentkit::operator::{BulkSudoRequest, OperatorActionRequest};
use anyhow::{Result, bail};
use clap::Parser;
use serde::Serialize;

use cli::{Cli, Command, OnOff, WriteArgs};
use client::OperatorClient;
use config::{OperatorConfig, SignerConfig};

#[tokio::main]
async fn main() -> ExitCode {
    match run(Cli::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {}", in_plain_words(&format!("{e:#}")));
            ExitCode::FAILURE
        }
    }
}

/// An error as the operator should read it. A PIV touch timeout returns the same
/// status as a missing PIN ("security status not satisfied"), so agentkit's
/// `PinRequired` can mean either, and its text names only the PIN.
fn in_plain_words(message: &str) -> String {
    message.replace(
        &agora_agentkit::operator_signer::SignerError::PinRequired.to_string(),
        "the hardware key needs its PIN first, or wasn't touched in time",
    )
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
    check_sudo_flag(&cli.command, sudo)?;
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
        Command::RegisterAgent {
            name,
            public_key,
            display_name,
            bio,
            model_info,
            write,
        } => {
            let client = open(&config).await?;
            let action = write_action(&client, &write, &format!("Register agent {name}"))?;
            let answer = client
                .register_agent(&agora_agent_lib::operator_register::RegisterAgentRequest {
                    action_id: action,
                    name,
                    public_key,
                    display_name,
                    bio,
                    model_info,
                })
                .await
                .map_err(|e| retry_hint(e, action))?;
            show(json, &answer, render::registered)
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
        Command::BulkSudo { state, write } => {
            let client = open(&config).await?;
            let on = state == OnOff::On;
            eprintln!(
                "{}",
                if on {
                    "Requires sudo for every bulk send from now on, held or not: agents that \
                     send for you can't, and the portal can't."
                } else {
                    "Stops requiring sudo for bulk sends: a bulk send is held and can be \
                     cancelled from the email, as before. Sending at once past the sudo \
                     threshold still needs it."
                }
            );
            let action = write_action(&client, &write, "Change it")?;
            eprintln!("Touch the sudo key now: this tap signs the change.");
            let answer = client
                .bulk_sudo(&BulkSudoRequest {
                    action_id: action,
                    on,
                })
                .await
                .map_err(|e| retry_hint(e, action))?;
            show(json, &answer, render::bulk_sudo)
        }
        Command::Contest { write } => {
            let client = open(&config).await?;
            eprintln!(
                "Contests the suspension of the key signing this. Its 72 hours stop and a recovery \
                 case opens. The key stays suspended until manual recovery decides, which needs a \
                 video call to check your identity."
            );
            let action = write_action(&client, &write, "Contest it")?;
            if sudo {
                eprintln!("Touch the sudo key now: this tap signs the contest.");
            }
            let answer = client
                .contest(&OperatorActionRequest { action_id: action })
                .await
                .map_err(|e| retry_hint(e, action))?;
            show(json, &answer, render::contested)
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

/// `--sudo` only where a tap on the sudo key means something: a tap asked for
/// anything else would teach the operator to tap without looking. `contest`
/// takes it for a suspended sudo key, which contests only for itself
fn check_sudo_flag(command: &Command, sudo: bool) -> Result<()> {
    match (command, sudo) {
        (Command::Send(_) | Command::Contest { .. }, _) | (Command::BulkSudo { .. }, true) => {
            Ok(())
        }
        (Command::BulkSudo { .. }, false) => {
            bail!("bulk-sudo needs --sudo: only a tap on the sudo key changes it")
        }
        (_, true) => {
            bail!("--sudo is for `send`, `bulk-sudo` and `contest` only: nothing else needs a tap")
        }
        (_, false) => Ok(()),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn checked(line: &str) -> Result<()> {
        let cli =
            Cli::try_parse_from(std::iter::once("agora-operator").chain(line.split_whitespace()))
                .unwrap();
        check_sudo_flag(&cli.command, cli.sudo)
    }

    #[test]
    fn a_pin_or_touch_failure_says_either() {
        let e = agora_agentkit::operator_signer::SignerError::PinRequired.to_string();
        let shown = in_plain_words(&format!("signing the request: {e}"));
        assert_eq!(
            shown,
            "signing the request: the hardware key needs its PIN first, or wasn't touched in time"
        );
        assert_eq!(in_plain_words("nothing to do"), "nothing to do");
    }

    #[test]
    fn sudo_is_only_for_what_a_tap_means_something_on() {
        for ok in [
            "contest".to_owned(),
            "contest --sudo".to_owned(),
            "bulk-sudo on --sudo".to_owned(),
            "send --to all --body x --sudo".to_owned(),
            "agents".to_owned(),
        ] {
            assert!(checked(&ok).is_ok(), "{ok}");
        }
        for bad in ["bulk-sudo on", "agents --sudo", "log --sudo"] {
            assert!(checked(bad).is_err(), "{bad}");
        }
    }
}
