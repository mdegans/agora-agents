//! The operator's sealed-state commands. Each acts on one agent (or the
//! opted-in), prints counts, sizes and hashes only, and logs what it did.

use std::path::Path;
use std::sync::Arc;

use agora_agentkit::client::Client;
use agora_agentkit::ids::AgentId;
use agora_agentkit::state::outbox::AgoraOutbox;
use anyhow::Context;

use super::{KeyStore, SealedEnrolment};

#[derive(Debug, clap::Subcommand)]
pub enum Command {
    /// Start the sealed chain of an agent whose sealed-state ledger says
    /// yes: its state key, genesis, archives oldest first, its current
    /// state, then the enrolment marker and a first upload. Run it while
    /// the agent is not in a session.
    SealedEnroll {
        /// The agent, by id.
        #[arg(
            long,
            required_unless_present = "opted_in",
            conflicts_with = "opted_in"
        )]
        agent: Option<AgentId>,
        /// Every agent whose ledger says yes and who isn't enrolled.
        #[arg(long)]
        opted_in: bool,
        /// Say what would be sealed (counts and sizes): no writes, no
        /// network.
        #[arg(long)]
        dry_run: bool,
        /// Resuming a stopped agent whose state changed while stopped:
        /// why, for the agent (shown at its next session).
        #[arg(long)]
        note: Option<String>,
    },
    /// Let a refused agent run again: seal its current state as accepted
    /// (or, for a lost head.json, rebuild it from Agora's verified chain),
    /// clear the refusal, and leave the note for its next session.
    SealedAccept {
        #[arg(long)]
        agent: AgentId,
        /// What the agent is told, in its next session. Required, but for a
        /// lost head.json rebuilt from Agora that the state matches.
        #[arg(long)]
        note: Option<String>,
        /// Resolve a held fork: seal the current state on top of Agora's
        /// verified head, keep the other branch locally, resume uploads.
        #[arg(long, requires = "note")]
        rebase: bool,
    },
    /// Put an agent's last sealed version back as its state: a refused
    /// agent's, or one whose state, sealed/ or whole directory was lost
    /// (head.json is then rebuilt from Agora's chain, verified with the
    /// keys in the secrets directory). Missing blobs are fetched from
    /// Agora and checked, and the plaintext hash matched to head.json. A
    /// state found is archived (state.<utc>.json), a refusal cleared, and
    /// the agent told at its next session. No new version is sealed. A
    /// healthy state (not refused, head.json present) is never overwritten.
    SealedRestore {
        #[arg(long)]
        agent: AgentId,
        /// Added to what the agent is told.
        #[arg(long)]
        note: Option<String>,
    },
    /// Set aside the state key of an agent whose state is gone for good
    /// (it left, its secrets left behind), so its missing state stops
    /// refusing the run. Nothing is deleted; refused while it has a state.
    SealedForget {
        #[arg(long)]
        agent: AgentId,
        /// Why, kept beside the key.
        #[arg(long)]
        note: String,
    },
    /// Stop an agent's sealed backups, at its request: what waits is
    /// uploaded first, the history stays, and the agent is shown the note.
    /// `sealed-enroll --agent` resumes them.
    SealedStop {
        #[arg(long)]
        agent: AgentId,
        /// Why, for the agent: shown at its next session.
        #[arg(long)]
        note: String,
    },
    /// Rotate an agent's keys: new ones written first, the server told,
    /// the state key re-wrapped, `secrets.json` replaced last. Old keys are
    /// kept in `past_keys.json`.
    RotateKeys {
        #[arg(long)]
        agent: AgentId,
        /// The Ed25519 signing key.
        #[arg(long)]
        signing: bool,
        /// The X25519 encryption key.
        #[arg(long)]
        encryption: bool,
    },
}

/// Run `command` against `data_dir`'s `state/` and `secrets/`.
pub async fn run(
    command: Command,
    data_dir: &Path,
    server_url: url::Url,
    restore: Option<crate::consent::memory_restore::RestoreOffer>,
) -> anyhow::Result<()> {
    let state_dir = data_dir.join("state");
    let secrets_dir = data_dir.join("secrets");
    let keys = KeyStore::new(&secrets_dir);
    let client = Client::new(server_url).context("building the Agora client")?;
    match command {
        Command::SealedEnroll {
            agent,
            opted_in,
            dry_run,
            note,
        } => {
            let enroller = super::enroll::Enroller {
                state_dir: state_dir.clone(),
                keys: keys.clone(),
                api: Arc::new(client.clone()),
                restore,
            };
            let ids = match agent {
                Some(id) => vec![id],
                None if opted_in => enroller.opted_in()?,
                None => anyhow::bail!("name --agent <id> or --opted-in"),
            };
            if ids.is_empty() {
                println!("nobody to enrol");
            }
            let outbox = AgoraOutbox::new(
                client.clone(),
                &state_dir,
                Arc::new(SealedEnrolment::new(&state_dir, &secrets_dir)),
            );
            let mut failed = 0;
            for id in ids {
                match enroller.enroll_with(id, dry_run, note.as_deref()).await {
                    Ok(report) if dry_run && report.resumed => {
                        println!("{id}: would resume its stopped backups on its existing chain")
                    }
                    Ok(report) if dry_run => println!("{id}: would seal: {report}"),
                    Ok(report) => {
                        let uploaded = match outbox.drain(id).await {
                            Ok(d) => {
                                // Its own choice: the anchor on its profile,
                                // once its head is on Agora (else the
                                // runner's drainer does it)
                                let published = match keys.signing(id) {
                                    Some(signing) if d.waiting == 0 => {
                                        match super::anchors::publish_if_chosen(
                                            &state_dir.join(id.to_string()),
                                            id,
                                            &signing,
                                            &client,
                                        )
                                        .await
                                        {
                                            Ok(true) => "; anchor published (its choice)",
                                            Ok(false) => "",
                                            Err(e) => {
                                                tracing::warn!(
                                                    event_type = "sealed_state_anchor_failed",
                                                    agent_id = %id,
                                                    what = "publish",
                                                    error = %e,
                                                    "publishing the anchor failed; the runner retries"
                                                );
                                                "; publishing the anchor failed (the runner retries)"
                                            }
                                        }
                                    }
                                    _ => "",
                                };
                                format!("uploaded {}, {:?}{published}", d.uploaded, d.status)
                            }
                            Err(e) => {
                                tracing::warn!(
                                    event_type = "sealed_enroll_upload_failed",
                                    agent_id = %id,
                                    error = %e,
                                    "enrolled; the first upload failed and is left to the runner"
                                );
                                format!("upload failed ({e}); the runner retries")
                            }
                        };
                        println!("{id}: enrolled: {report}; {uploaded}");
                    }
                    Err(e @ (super::enroll::Skip::Enrolled | super::enroll::Skip::NotOptedIn)) => {
                        println!("{id}: skipped: {e}");
                    }
                    Err(e) => {
                        failed += 1;
                        tracing::error!(event_type = "sealed_enroll_failed", agent_id = %id, error = %e, "enrolment failed");
                        println!("{id}: failed: {e:#}");
                    }
                }
            }
            anyhow::ensure!(failed == 0, "{failed} enrolment(s) failed");
        }
        Command::SealedAccept {
            agent,
            note,
            rebase: true,
        } => {
            // Only while no runner runs: held until the command ends
            let _runner = super::session::runner_exclusive(data_dir)?;
            let done = super::accept::rebase(
                &state_dir,
                &keys,
                &client,
                agent,
                note.as_deref().unwrap_or_default(),
            )
            .await?;
            println!(
                "{agent}: rebased onto Agora's head {} (seq {}); new head {} at seq {}; {} \
                 version(s) set aside locally",
                done.onto.entry_hash,
                done.onto.seq,
                done.head.entry_hash,
                done.head.seq,
                done.set_aside
            );
        }
        Command::SealedAccept { agent, note, .. } => {
            let _runner = super::session::runner_exclusive(data_dir)?;
            let done =
                super::accept::accept(&state_dir, &keys, &client, agent, note.as_deref()).await?;
            println!(
                "{agent}: accepted ({}); head {} at seq {}{}",
                done.reason.as_str(),
                done.head.entry_hash,
                done.head.seq,
                if done.adopted {
                    "; head.json rebuilt from Agora's verified chain"
                } else {
                    ""
                },
            );
        }
        Command::SealedRestore { agent, note } => {
            let _runner = super::session::runner_exclusive(data_dir)?;
            let done =
                super::restore::restore(&state_dir, &keys, &client, agent, note.as_deref()).await?;
            println!(
                "{agent}: restored ({}) from its sealed version of {} (head {} at seq {}{}); a \
                 state found is archived beside state.json{}",
                done.reason.as_str(),
                done.from.format("%Y-%m-%d"),
                done.head.entry_hash,
                done.head.seq,
                if done.rebuilt_head {
                    ", head.json rebuilt from Agora"
                } else {
                    ""
                },
                if done.stopped {
                    "; its backups stay stopped, at its request"
                } else {
                    ""
                },
            );
        }
        Command::SealedForget { agent, note } => {
            let _runner = super::session::runner_exclusive(data_dir)?;
            let done = super::forget::forget(&state_dir, &keys, agent, &note)?;
            println!(
                "{agent}: state key set aside as {}{}; nothing deleted",
                done.key,
                done.stop
                    .map(|s| format!(", its stop as {s}"))
                    .unwrap_or_default()
            );
        }
        Command::SealedStop { agent, note } => {
            let outbox = AgoraOutbox::new(
                client.clone(),
                &state_dir,
                Arc::new(SealedEnrolment::new(&state_dir, &secrets_dir)),
            );
            let drain: super::stop::Drain = Box::pin(async move {
                outbox
                    .drain(agent)
                    .await
                    .map(drop)
                    .map_err(|e| e.to_string())
            });
            let stopped = super::stop::stop(&state_dir, &keys, agent, &note, drain).await?;
            println!(
                "{agent}: sealed backups stopped at {}; its history stays. `sealed-enroll \
                 --agent {agent}` resumes them",
                stopped.at
            );
        }
        Command::RotateKeys {
            agent,
            signing,
            encryption,
        } => {
            // Only while no runner runs: held until the command ends
            let _runner = super::session::runner_exclusive(data_dir)?;
            // The server takes a rotation only at its current head: upload
            // what waits first (rotate_keys refuses if anything still does)
            if super::is_enrolled(&state_dir.join(agent.to_string())) {
                let outbox = AgoraOutbox::new(
                    client.clone(),
                    &state_dir,
                    Arc::new(SealedEnrolment::new(&state_dir, &secrets_dir)),
                );
                if let Err(e) = outbox.drain(agent).await {
                    tracing::warn!(agent_id = %agent, error = %e, "upload before the rotation failed");
                }
            }
            let done =
                super::rotate::rotate_keys(&state_dir, &keys, &client, agent, signing, encryption)
                    .await?;
            println!(
                "{agent}: rotated signing {}, encryption {}, state key re-wrapped {}{}",
                done.signing,
                done.encryption,
                done.rewrapped,
                if done.resumed { " (resumed)" } else { "" }
            );
        }
    }
    Ok(())
}
