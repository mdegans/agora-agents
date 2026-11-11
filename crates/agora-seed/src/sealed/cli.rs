//! The operator's sealed-state commands. Each acts on one agent (or the
//! opted-in), prints counts, sizes and hashes only, and logs what it did.

use std::path::Path;
use std::sync::Arc;

use agora_agentkit::client::Client;
use agora_agentkit::ids::AgentId;
use agora_agentkit::state::outbox::{AgoraOutbox, Drained};
use anyhow::Context;

use super::{KeyStore, SealedEnrolment};

#[derive(Debug, clap::Subcommand)]
pub enum Command {
    /// Chain agents locally (every agent with a state, or those named):
    /// its state key, genesis, archives oldest first, its current state,
    /// then the enrolment marker. Nothing is sent to Agora. Named, a
    /// stopped agent's uploads are resumed instead (the versions saved
    /// while stopped stay here). With `--opt-in`, start uploading. Only
    /// while no runner runs (`just runner-stop`).
    SealedEnroll {
        /// The agents, by id (repeat the flag). Default: every agent with a
        /// state (`--opt-in`: every agent whose ledger says yes).
        #[arg(long = "agent")]
        agents: Vec<AgentId>,
        /// Start uploading the chained agents whose sealed-state ledger says
        /// yes (chaining any not chained yet first): their state key record
        /// posted, then marked uploading, then a first upload. Also what
        /// finishes an opt-in cut short. `--opted-in` is the old spelling.
        #[arg(long, visible_alias = "opted-in")]
        opt_in: bool,
        /// Say what would be done (counts and sizes): no writes, no
        /// network.
        #[arg(long)]
        dry_run: bool,
        /// Agents chained at once. Each holds tens of MB while it seals.
        /// Default: half the cores, at most 16.
        #[arg(long, value_parser = clap::value_parser!(u16).range(1..=64))]
        jobs: Option<u16>,
        /// Resuming a stopped agent whose state changed while stopped, or
        /// opting in against an answer: why, for the agent (shown at its
        /// next session).
        #[arg(long)]
        note: Option<String>,
        /// With `--opt-in` and named agents: opt in even though an agent's
        /// latest answer is `decline` (never one with no answer). Needs
        /// `--note`, which the agent is shown.
        #[arg(long, requires_all = ["note", "opt_in"])]
        against_answer: bool,
    },
    /// Re-open the sealed-state question for agents that answered it: put
    /// again, once, at each one's next session, with the note. Their answer
    /// changes nothing by itself: you are alerted, and act on it
    /// (`sealed-enroll --opt-in`, `sealed-stop`). Only while no runner runs.
    SealedReoffer {
        /// The agents, by id.
        #[arg(required = true)]
        agents: Vec<AgentId>,
        /// Why, for the agents: shown with the question.
        #[arg(long)]
        note: String,
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
        /// sealed/ (or head.json) lost and a state here that isn't
        /// Agora's latest: replace it anyway (it is archived). Without
        /// this the restore refuses; `sealed-accept --note` keeps it.
        #[arg(long)]
        discard_local: bool,
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
    /// Stop uploading an agent's sealed state, at its request: what waits
    /// is uploaded first, what is on Agora stays, and the agent is shown
    /// the note. Its state is still sealed, chained and checked locally.
    /// `sealed-enroll --agent` resumes the uploads.
    SealedStop {
        #[arg(long)]
        agent: AgentId,
        /// Why, for the agent: shown at its next session.
        #[arg(long)]
        note: String,
        /// Stop even though the agent's latest answer is yes (a legal
        /// reason, say); the note says why, and the agent is shown it.
        #[arg(long)]
        against_answer: bool,
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
    let enrolment = Arc::new(SealedEnrolment::new(&state_dir, &secrets_dir));
    // Every per-agent sealed-state call goes through the guard: nothing of
    // an agent that doesn't upload is sent or asked
    let api = Arc::new(super::api::guarded(client.clone(), enrolment.clone()));
    let dry_run = matches!(command, Command::SealedEnroll { dry_run: true, .. });
    // The whole of `sealed-enroll` runs only while no runner does, taken
    // before anything is read: a runner holding an agent's state from
    // before its chain would save it unsealed over the chain (and be
    // alerted). Not waited for. A dry run writes nothing and needs none.
    let _runner = match &command {
        Command::SealedEnroll { dry_run: false, .. } => {
            Some(super::session::runner_exclusive(data_dir)?)
        }
        _ => None,
    };
    if !dry_run {
        // As at a run's start: a lost local marker put back, and the agents
        // that uploaded before the marker existed marked (idempotent)
        super::uploading::reconcile_all(&state_dir, &keys);
        super::uploading::migrate_all(&state_dir, &keys);
        // B1: the move of old `sealed/enrolled` markers to agentkit's new
        // name goes here (called, not refused: the runner lock is held, so
        // it is as safe here as at a run's start, and it is idempotent)
    }
    match command {
        Command::SealedEnroll {
            agents,
            opt_in,
            dry_run,
            jobs,
            note,
            against_answer,
        } => {
            let enroller = Arc::new(super::enroll::Enroller {
                state_dir: state_dir.clone(),
                keys: keys.clone(),
                restore,
            });
            if opt_in {
                let outbox = AgoraOutbox::new(client.clone(), &state_dir, enrolment.clone());
                let against = against_answer.then_some(note.as_deref()).flatten();
                anyhow::ensure!(
                    !against_answer || !agents.is_empty(),
                    "--against-answer acts only on agents named with --agent"
                );
                opt_in_pass(&enroller, &api, &outbox, agents, dry_run, against).await?;
            } else {
                let jobs = jobs.map_or_else(default_jobs, usize::from);
                chain_pass(&enroller, agents, dry_run, jobs, note.as_deref()).await?;
            }
        }
        Command::SealedAccept {
            agent,
            note,
            rebase: true,
        } => {
            // `rebase` takes the runner lock itself, before it reads anything
            let done = super::accept::rebase(
                &state_dir,
                &keys,
                api.as_ref(),
                agent,
                note.as_deref().unwrap_or_default(),
            )
            .await?;
            match done.onto {
                Some(onto) => println!(
                    "{agent}: sealed onto Agora's head {} (seq {}); new head {} at seq {}; {} \
                     version(s) set aside locally",
                    onto.entry_hash, onto.seq, done.head.entry_hash, done.head.seq, done.set_aside
                ),
                None => println!(
                    "{agent}: Agora held nothing: a new chain (its genesis records the break); \
                     head {} at seq {}; {} version(s) set aside locally",
                    done.head.entry_hash, done.head.seq, done.set_aside
                ),
            }
        }
        Command::SealedAccept { agent, note, .. } => {
            let _runner = super::session::runner_exclusive(data_dir)?;
            let done =
                super::accept::accept(&state_dir, &keys, api.as_ref(), agent, note.as_deref())
                    .await?;
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
        Command::SealedRestore {
            agent,
            note,
            discard_local,
        } => {
            let _runner = super::session::runner_exclusive(data_dir)?;
            let done = super::restore::restore(
                &state_dir,
                &keys,
                api.as_ref(),
                agent,
                note.as_deref(),
                discard_local,
            )
            .await?;
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
                "{agent}: state key set aside as {}{}{}; no key deleted{}",
                done.key,
                done.memory_id_key
                    .map(|k| format!(", its memory-id key as {k}"))
                    .unwrap_or_default(),
                done.stop
                    .map(|s| format!(", its stop as {s}"))
                    .unwrap_or_default(),
                if done.index {
                    "; the search index (a rebuildable sidecar) deleted"
                } else {
                    ""
                }
            );
        }
        Command::SealedReoffer { agents, note } => {
            anyhow::ensure!(
                !note.trim().is_empty(),
                "--note is required: the agents read it"
            );
            let _runner = super::session::runner_exclusive(data_dir)?;
            let mut failed = 0;
            for id in agents {
                let dir = state_dir.join(id.to_string());
                match crate::consent::sealed_state::reoffer(&dir, id, &note).await {
                    Ok(replaced) => println!(
                        "{id}: the sealed-state question is put again at its next session{}",
                        if replaced {
                            " (a pending re-offer's note replaced)"
                        } else {
                            ""
                        }
                    ),
                    Err(e) => {
                        failed += 1;
                        println!("{id}: not re-offered: {e}");
                    }
                }
            }
            anyhow::ensure!(failed == 0, "{failed} agent(s) not re-offered");
        }
        Command::SealedStop {
            agent,
            note,
            against_answer,
        } => {
            // Never against the agent's latest answer, but with the override
            let dir = state_dir.join(agent.to_string());
            {
                use crate::consent::offer::Stored as _;
                let ledger = crate::consent::sealed_state::ledger::SealLedger::load_blocking(&dir)
                    .context("the agent's sealed-state ledger is unreadable")?;
                if let Err(why) = ledger.allows(false) {
                    anyhow::ensure!(
                        against_answer,
                        "{agent}: {why}: not stopped. Re-open the question (`sealed-reoffer`), or \
                         pass --against-answer with a --note saying why (the agent is shown it)"
                    );
                    tracing::warn!(
                        event_type = "sealed_state_stop_against_answer",
                        agent_id = %agent,
                        "the operator stops uploads against the agent's latest answer"
                    );
                }
            }
            let outbox = AgoraOutbox::new(client.clone(), &state_dir, enrolment.clone());
            let dir = state_dir.join(agent.to_string());
            let mut baseline = super::receipts::baseline(&dir, agent).await;
            let drain: super::stop::Drain = Box::pin(async move {
                super::receipts::drain(&outbox, &dir, agent, &mut baseline)
                    .await
                    .map(drop)
                    .map_err(|e| e.to_string())
            });
            let stopped = super::stop::stop(&state_dir, &keys, agent, &note, drain).await?;
            println!(
                "{agent}: uploads stopped at {}; what is on Agora stays, and its state is still \
                 sealed and checked here, locally. `sealed-enroll --agent {agent}` resumes the \
                 uploads",
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
                let outbox = AgoraOutbox::new(client.clone(), &state_dir, enrolment.clone());
                let dir = state_dir.join(agent.to_string());
                let mut baseline = super::receipts::baseline(&dir, agent).await;
                if let Err(e) = super::receipts::drain(&outbox, &dir, agent, &mut baseline).await {
                    tracing::warn!(agent_id = %agent, error = %e, "upload before the rotation failed");
                }
            }
            let done = super::rotate::rotate_keys(
                &state_dir,
                &keys,
                api.as_ref(),
                agent,
                signing,
                encryption,
            )
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

/// Agents chained at once by default: half the cores, 1 to 16. Each holds
/// tens of MB while it seals (brotli's 16 MiB window, twice).
fn default_jobs() -> usize {
    std::thread::available_parallelism()
        .map_or(1, |n| n.get() / 2)
        .clamp(1, 16)
}

/// What a pass did, by outcome: the count table.
#[derive(Default)]
struct Tally(std::collections::BTreeMap<&'static str, usize>);

impl Tally {
    fn add(&mut self, outcome: &'static str) {
        *self.0.entry(outcome).or_default() += 1;
    }

    fn get(&self, outcome: &str) -> usize {
        self.0.get(outcome).copied().unwrap_or(0)
    }

    fn print(&self, what: &str) {
        println!("{what}:");
        for (outcome, n) in &self.0 {
            println!("  {outcome:<18} {n:>6}");
        }
        let total: usize = self.0.values().sum();
        println!("  {:<18} {total:>6}", "total");
        tracing::info!(
            event_type = "sealed_enroll_counts",
            pass = what,
            counts = ?self.0,
            total,
            "sealed-enroll counts"
        );
    }
}

/// `sealed-enroll [--agent <id>]…`: chain locally, `jobs` at once; a
/// stopped agent named is resumed instead. Nothing sent.
async fn chain_pass(
    enroller: &Arc<super::enroll::Enroller>,
    named: Vec<AgentId>,
    dry_run: bool,
    jobs: usize,
    note: Option<&str>,
) -> anyhow::Result<()> {
    use super::enroll::Skip;
    let mut tally = Tally::default();
    let explicit = !named.is_empty();
    let ids = if explicit { named } else { enroller.agents()? };
    let mut chain = Vec::with_capacity(ids.len());
    for id in ids {
        let dir = enroller.state_dir.join(id.to_string());
        let stopped = super::stop::is_stopped(&dir)
            || super::stop::durable(&enroller.keys, id).map_or(true, |d| d.is_some());
        // Only a stopped agent named is resumed: never a whole-fleet pass
        if !(explicit && stopped && super::is_enrolled(&dir)) {
            chain.push(id);
            continue;
        }
        if dry_run {
            println!(
                "{id}: would resume its stopped uploads (the versions saved while stopped stay here)"
            );
            tally.add("would_resume");
            continue;
        }
        match enroller.resume(id, note).await {
            Ok(report) => {
                tally.add("resumed");
                let uploads = super::uploads(&dir, &enroller.keys, id);
                println!(
                    "{id}: resumed; {} version(s) saved while stopped kept here, not sent; {}",
                    report.kept_local,
                    if uploads {
                        "its uploads go on"
                    } else {
                        "it is NOT marked uploading, so nothing of it is sent: `sealed-enroll \
                         --opt-in --agent <id>` starts its uploads if its ledger says yes"
                    }
                );
            }
            Err(e) => {
                tally.add("failed");
                tracing::error!(event_type = "sealed_resume_failed", agent_id = %id, error = %e, "resume failed");
                println!("{id}: failed: {e:#}");
            }
        }
    }
    let mut says_yes = 0;
    for (id, result) in enroller.chain_all(&chain, dry_run, jobs).await {
        match result {
            Ok(report) => {
                tally.add(match (dry_run, report.prior_later) {
                    (true, _) => "would_chain",
                    (false, true) => "prior_appended",
                    (false, false) => "chained",
                });
                println!(
                    "{id}: {}: {report}",
                    if dry_run { "would chain" } else { "chained" }
                );
                let dir = enroller.state_dir.join(id.to_string());
                use crate::consent::offer::Stored as _;
                if crate::consent::sealed_state::ledger::SealLedger::load_blocking(&dir)
                    .is_ok_and(|l| l.opted_in())
                {
                    says_yes += 1;
                }
            }
            Err(Skip::Chained) => tally.add(Skip::Chained.as_str()),
            Err(e @ Skip::Failed(_)) => {
                tally.add(e.as_str());
                tracing::error!(event_type = "sealed_chain_failed", agent_id = %id, error = %e, "chaining failed");
                println!("{id}: failed: {e:#}");
            }
            Err(e) => {
                tally.add(e.as_str());
                println!("{id}: skipped: {e}");
            }
        }
    }
    tally.print(if dry_run { "chain (dry run)" } else { "chain" });
    if says_yes > 0 {
        println!(
            "{says_yes} agent(s) chained here said yes to uploading: `sealed-enroll --opt-in` \
             starts their uploads"
        );
    }
    let failed = tally.get("failed");
    anyhow::ensure!(failed == 0, "{failed} agent(s) failed");
    Ok(())
}

/// `sealed-enroll --opt-in [--agent <id>]…`: chain any not chained yet,
/// then opt in and upload, one agent at a time.
async fn opt_in_pass(
    enroller: &super::enroll::Enroller,
    api: &super::api::GuardedClient,
    outbox: &AgoraOutbox<SealedEnrolment>,
    named: Vec<AgentId>,
    dry_run: bool,
    against: Option<&str>,
) -> anyhow::Result<()> {
    use super::enroll::Skip;
    let mut tally = Tally::default();
    let ids = if named.is_empty() {
        enroller.opted_in()?
    } else {
        named
    };
    for id in ids {
        let dir = enroller.state_dir.join(id.to_string());
        if dry_run {
            let chained = super::is_enrolled(&dir);
            println!(
                "{id}: would {}opt in and upload its chain",
                if chained { "" } else { "chain, then " }
            );
            tally.add("would_opt_in");
            continue;
        }
        if !super::is_enrolled(&dir) {
            match enroller.chain(id, false).await {
                Ok(report) => println!("{id}: chained: {report}"),
                Err(Skip::Chained) => {}
                Err(e) => {
                    tally.add(e.as_str());
                    println!("{id}: not chained, so not opted in: {e:#}");
                    continue;
                }
            }
        }
        // Before the opt-in commits anything: what it commits is this
        // process's own, its upload times recorded
        let mut baseline = super::receipts::baseline(&dir, id).await;
        let opted = match enroller.opt_in(api, id, against).await {
            Ok(o) => o,
            Err(e @ Skip::Failed(_)) => {
                tally.add(e.as_str());
                tracing::error!(event_type = "sealed_opt_in_failed", agent_id = %id, error = %e, "opt-in failed");
                println!("{id}: failed: {e:#}");
                continue;
            }
            Err(e) => {
                tally.add(e.as_str());
                println!("{id}: skipped: {e}");
                continue;
            }
        };
        tally.add(if opted.already {
            "already_uploading"
        } else {
            "opted_in"
        });
        let uploaded = match super::receipts::drain(outbox, &dir, id, &mut baseline).await {
            Ok(Drained::Ran(d)) => {
                // Its own choice: the anchor on its profile, once its head
                // is on Agora (else the runner's drainer does it)
                let published = match enroller.keys.signing(id) {
                    Some(signing) if d.waiting == 0 => {
                        match super::anchors::publish_if_chosen(&dir, id, &signing, api).await {
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
            Ok(Drained::LocalOnly) => "nothing uploaded (it doesn't upload)".to_string(),
            Ok(Drained::NotEnrolled) => "nothing uploaded (not chained)".to_string(),
            Ok(Drained::Withdrawn { uploaded }) => format!(
                "uploaded {uploaded}, then uploading was switched off; the rest stays local"
            ),
            Err(e) => {
                tracing::warn!(
                    event_type = "sealed_opt_in_upload_failed",
                    agent_id = %id,
                    error = %e,
                    "opted in; the first upload failed and is left to the runner"
                );
                format!("upload failed ({e}); the runner retries")
            }
        };
        println!(
            "{id}: {}; key record {}{}; {uploaded}",
            if opted.already {
                "already uploading"
            } else {
                "opted in"
            },
            if opted.posted {
                "posted"
            } else {
                "already on Agora"
            },
            if opted.held_by_agora > 0 {
                format!(", {} entries Agora held filed as sent", opted.held_by_agora)
            } else {
                String::new()
            },
        );
    }
    tally.print(if dry_run {
        "opt in (dry run)"
    } else {
        "opt in"
    });
    let failed = tally.get("failed");
    anyhow::ensure!(failed == 0, "{failed} agent(s) failed");
    Ok(())
}
