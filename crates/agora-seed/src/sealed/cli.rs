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
    /// then the enrolment marker. Nothing is sent to Agora. With
    /// `--opt-in`, say which agents' uploads would start (their own yes in
    /// force), and with `--yes` start them. With `--resume`, resume a
    /// stopped agent's uploads. Only while no runner runs
    /// (`just runner-stop`).
    SealedEnroll {
        /// The agents, by id (repeat the flag). Default: every agent with a
        /// state (`--opt-in`: every agent whose ledger says yes).
        #[arg(long = "agent")]
        agents: Vec<AgentId>,
        /// Start uploading the chained agents whose answer in force to the
        /// sealed-state offer is yes, with no re-offer waiting (chaining
        /// any not chained yet first): their state key record posted, then
        /// marked uploading, then a first upload. Also what finishes an
        /// opt-in cut short, and what settles an uploading marker in
        /// doubt. Without `--yes`, a dry run: the count and the ids,
        /// nothing written or sent. Nothing starts an upload against an
        /// agent's answer. `--opted-in` is the old spelling.
        #[arg(long, visible_alias = "opted-in")]
        opt_in: bool,
        /// With `--opt-in`: do it. With `--agent`, for the agents named;
        /// without, for every agent the dry run lists, and then `--expect`
        /// is required.
        #[arg(long, requires = "opt_in", conflicts_with = "dry_run")]
        yes: bool,
        /// With `--opt-in --yes`: how many agents the dry run listed. A
        /// mismatch touches nothing.
        #[arg(long, requires = "yes")]
        expect: Option<usize>,
        /// Say what would be done (counts and sizes): no writes, no
        /// network.
        #[arg(long)]
        dry_run: bool,
        /// Agents chained at once. Each holds tens of MB while it seals.
        /// Default: half the cores, at most 16.
        #[arg(long, value_parser = clap::value_parser!(u16).range(1..=64))]
        jobs: Option<u16>,
        /// Resuming a stopped agent whose state changed while stopped:
        /// why, for the agent (shown at its next session).
        #[arg(long, requires = "resume")]
        note: Option<String>,
        /// Resume the uploads of the stopped agents named with `--agent`,
        /// on the line Agora holds: the versions not uploaded when the
        /// stop began, and those saved while stopped, stay here. Only on
        /// an answer in force of yes with no re-offer waiting; nothing
        /// overrides that.
        #[arg(long, conflicts_with = "opt_in", requires = "agents")]
        resume: bool,
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
        #[arg(
            long,
            required_unless_present = "agents_file",
            conflicts_with = "agents_file"
        )]
        agent: Option<AgentId>,
        /// What the agent is told, in its next session. Required, but for a
        /// lost head.json rebuilt (from Agora, or the chain kept here) that
        /// the state matches.
        #[arg(long)]
        note: Option<String>,
        /// Resolve a held fork: seal the current state on top of Agora's
        /// verified head, keep the other branch locally, resume uploads.
        #[arg(long, requires = "note", conflicts_with = "keep")]
        rebase: bool,
        /// Resolve a fork in a chain kept here (an agent that doesn't
        /// upload): `head` keeps the head's line, or the outbox number of
        /// a displaced line's tip keeps that line (its state becomes the
        /// agent's). The other line is kept under sealed/lost/<utc>/.
        #[arg(long, requires = "note")]
        keep: Option<super::accept::Keep>,
        /// Accept every agent named in this file (one id a line, `#`
        /// comments; `-` reads stdin) with one note, each under its own
        /// lock. Without --yes, a dry run: what would be done, per agent
        /// (refusal, memory and soul sizes), nothing touched.
        #[arg(long, requires = "note", conflicts_with_all = ["rebase", "keep"])]
        agents_file: Option<std::path::PathBuf>,
        /// Do it (after a dry run): needs --expect.
        #[arg(long, requires_all = ["agents_file", "expect"])]
        yes: bool,
        /// How many agents --agents-file names: a mismatch touches nothing.
        #[arg(long, requires = "agents_file")]
        expect: Option<usize>,
        /// Accept a state whose memory shrank by more than half since its
        /// last sealed version too (a wipe looks like that).
        #[arg(long, requires = "yes")]
        accept_shrunk: bool,
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
        /// Put back the newest save cut short (a complete state.json.tmp,
        /// or one set aside as state.json.tmp.<utc>.aside, newer than the
        /// sealed head) instead of the head's version. A restore that finds
        /// one refuses without this, naming it.
        #[arg(long)]
        from_cut_save: bool,
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
    /// Stop uploading an agent's sealed state, at its request: at once,
    /// whatever state the agent, the runner or Agora is in. Nothing more
    /// is sent (what waited to upload stays here), what is on Agora stays,
    /// and the agent is shown the note. Its state is still sealed, chained
    /// and checked locally. Stopping twice is the same stop.
    /// `sealed-enroll --resume --agent` resumes the uploads.
    SealedStop {
        #[arg(long)]
        agent: AgentId,
        /// Why, for the agent: shown at its next session.
        #[arg(long)]
        note: String,
        /// Stop even though the agent's answer in force is yes, or while a
        /// re-offer waits for its answer (a legal reason, say); the note
        /// says why, and the agent is shown it.
        #[arg(long)]
        against_answer: bool,
    },
    /// Rotate an agent's keys: new ones written first, the server told,
    /// the state key re-wrapped, `secrets.json` replaced last. Old keys are
    /// kept in `past_keys.json`.
    RotateKeys {
        #[arg(long)]
        agent: AgentId,
        /// Say where an unfinished rotation stands and what blocks it;
        /// nothing written, nothing sent.
        #[arg(long, conflicts_with_all = ["signing", "encryption"])]
        status: bool,
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
    let api = Arc::new(super::api::guarded(client.clone(), enrolment.clone()).recording());
    let dry_run = matches!(
        command,
        Command::SealedEnroll { dry_run: true, .. }
            | Command::SealedEnroll {
                opt_in: true,
                yes: false,
                ..
            }
    );
    // Every command that writes an agent's chain, keys or ledger runs only
    // while no runner does, held to its end and taken before anything is
    // read, the start pass included: a runner holding an agent's state from
    // before its chain would save it unsealed over the chain, sign with a
    // key retired under it, or save over a restored state. Not waited for.
    // A dry run of `sealed-enroll` (`--opt-in` without `--yes` is one)
    // writes nothing and needs none;
    // `sealed-stop` runs beside a runner (it only stops uploads);
    // `rotate-keys --status` only reads.
    let _runner = match &command {
        Command::SealedEnroll { dry_run: true, .. }
        | Command::SealedEnroll {
            opt_in: true,
            yes: false,
            ..
        }
        | Command::SealedStop { .. }
        | Command::RotateKeys { status: true, .. } => None,
        Command::SealedEnroll { .. }
        | Command::SealedReoffer { .. }
        | Command::SealedAccept { .. }
        | Command::SealedRestore { .. }
        | Command::SealedForget { .. }
        | Command::RotateKeys { .. } => Some(super::session::runner_exclusive(data_dir)?),
    };
    if !dry_run {
        // As at a run's start, before any command reads a marker: the
        // chain's marker moved to its new name, a lost or half-written
        // uploading marker repaired, the agents that uploaded before the
        // marker existed marked (idempotent). In doubt: logged at error
        super::uploading::start_pass(&state_dir, &keys, &crate::alerts::Alerter::off()).await;
    }
    match command {
        Command::SealedEnroll {
            agents,
            opt_in,
            yes,
            expect,
            dry_run,
            jobs,
            note,
            resume,
        } => {
            let enroller = Arc::new(super::enroll::Enroller {
                state_dir: state_dir.clone(),
                keys: keys.clone(),
                restore,
            });
            if opt_in {
                let outbox = AgoraOutbox::new(client.clone(), &state_dir, enrolment.clone());
                opt_in_pass(&enroller, &api, &outbox, agents, yes, expect).await?;
            } else if resume {
                resume_pass(&enroller, agents, dry_run, note.as_deref()).await?;
            } else {
                let jobs = jobs.map_or_else(default_jobs, usize::from);
                chain_pass(&enroller, agents, dry_run, jobs).await?;
            }
        }
        Command::SealedAccept {
            agents_file: Some(file),
            note,
            expect,
            yes,
            accept_shrunk,
            ..
        } => {
            let text = if file.as_os_str() == "-" {
                std::io::read_to_string(std::io::stdin()).context("reading ids from stdin")?
            } else {
                std::fs::read_to_string(&file)
                    .with_context(|| format!("reading {}", file.display()))?
            };
            let ids = super::accept::read_ids(&text)?;
            println!(
                "sealed-accept: {} agent(s) named{}",
                ids.len(),
                if yes {
                    ""
                } else {
                    "; a dry run, nothing is touched (--yes --expect N does it)"
                }
            );
            let done = super::accept::accept_many(
                &state_dir,
                &keys,
                api.as_ref(),
                &ids,
                super::accept::ManyOptions {
                    note: note.as_deref().unwrap_or_default(),
                    expect,
                    dry_run: !yes,
                    accept_shrunk,
                },
            )
            .await?;
            print!("{}", super::accept::many_table(&done));
            let failed = done.iter().filter(|(_, _, o)| o.failed()).count();
            anyhow::ensure!(failed == 0, "{failed} of {} agent(s) failed", done.len());
        }
        Command::SealedAccept {
            agent: Some(agent),
            note,
            keep: Some(keep),
            ..
        } => {
            let done = super::accept::keep_line(
                &state_dir,
                &keys,
                api.as_ref(),
                agent,
                keep,
                note.as_deref().unwrap_or_default(),
            )
            .await?;
            println!(
                "{agent}: fork resolved; head {} at seq {}; {} version(s) set aside, kept in {}{}",
                done.head.entry_hash,
                done.head.seq,
                done.set_aside,
                done.kept.display(),
                if done.state_changed {
                    "; its state is the kept line's (the one here archived)"
                } else {
                    ""
                }
            );
        }
        Command::SealedAccept {
            agent: Some(agent),
            note,
            rebase: true,
            ..
        } => {
            // Under the runner lock taken above
            let done = super::accept::rebase_locked(
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
        Command::SealedAccept { agent: None, .. } => {
            anyhow::bail!("name --agent <id> or --agents-file")
        }
        Command::SealedAccept {
            agent: Some(agent),
            note,
            ..
        } => {
            let done =
                super::accept::accept(&state_dir, &keys, api.as_ref(), agent, note.as_deref())
                    .await?;
            println!(
                "{agent}: accepted ({}); head {} at seq {}{}",
                done.reason.as_str(),
                done.head.entry_hash,
                done.head.seq,
                if done.adopted {
                    "; head.json rebuilt from the verified chain"
                } else {
                    ""
                },
            );
        }
        Command::SealedRestore {
            agent,
            note,
            discard_local,
            from_cut_save,
        } => {
            let done = super::restore::restore_with(
                &state_dir,
                &keys,
                api.as_ref(),
                agent,
                super::restore::Options {
                    note: note.as_deref(),
                    discard_local,
                    from_cut_save,
                },
            )
            .await?;
            if let Some(file) = &done.cut_save {
                println!("{agent}: put back the save cut short in {file}");
            }
            if done.recovered_key {
                println!("{agent}: state_key.json was missing and was recovered from Agora");
            }
            println!(
                "{agent}: restored ({}) from its sealed version of {} (head {} at seq {}{}); a \
                 state found is archived beside state.json{}",
                done.reason.as_str(),
                done.from.format("%Y-%m-%d"),
                done.head.entry_hash,
                done.head.seq,
                if done.rebuilt_head {
                    ", head.json rebuilt from the verified chain"
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
            let done = super::forget::forget(&state_dir, &keys, agent, &note).await?;
            println!(
                "{agent}: state key set aside as {}{}{}{}; no key deleted{}",
                done.key,
                done.memory_id_key
                    .map(|k| format!(", its memory-id key as {k}"))
                    .unwrap_or_default(),
                done.stop
                    .map(|s| format!(", its stop as {s}"))
                    .unwrap_or_default(),
                done.uploading
                    .map(|u| format!(", its uploading marker as {u}"))
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
            // By the agent's answer in force, or with the override
            let dir = state_dir.join(agent.to_string());
            {
                use crate::consent::offer::Stored as _;
                // Stopping is the safe direction: with the override, an
                // unreadable ledger doesn't block it
                let ledger =
                    match crate::consent::sealed_state::ledger::SealLedger::load_blocking(&dir) {
                        Ok(l) => Some(l),
                        Err(e) if against_answer => {
                            tracing::warn!(
                                event_type = "sealed_state_stop_ledger_unreadable",
                                agent_id = %agent,
                                error = %e,
                                "the agent's ledger couldn't be read; stopped with the override"
                            );
                            None
                        }
                        Err(e) => {
                            return Err(anyhow::Error::new(e).context(
                                "the agent's sealed-state ledger is unreadable: pass \
                                 --against-answer with the note to stop anyway",
                            ));
                        }
                    };
                if let Some(Err(why)) = ledger.map(|l| l.in_force().stops_uploads()) {
                    anyhow::ensure!(
                        against_answer,
                        "{agent}: {why}: not stopped. Wait for its answer (or re-open the \
                         question with `sealed-reoffer`), or pass --against-answer with a --note \
                         saying why (the agent is shown it)"
                    );
                    tracing::warn!(
                        event_type = "sealed_state_stop_against_answer",
                        agent_id = %agent,
                        "the operator stops uploads against the agent's answer in force"
                    );
                }
            }
            // No drain, no client: a stop sends nothing
            let done = super::stop::stop(&state_dir, &keys, agent, &note).await?;
            println!(
                "{agent}: uploads {} at {}; nothing more is sent. What is on Agora stays, and \
                 its state is still sealed and checked here, locally. `sealed-enroll --resume \
                 --agent {agent}` resumes the uploads (only on its own yes)",
                if done.already {
                    "were already stopped (that stop stands, with its own note)"
                } else {
                    "stopped"
                },
                done.stopped.at
            );
            match done.kept_local {
                Some(0) => println!("{agent}: no sealed version was waiting to upload"),
                Some(n) => println!(
                    "{agent}: {n} sealed version(s) had not been uploaded: they stay here, \
                     unsent, and a resume sets them aside from the uploaded line"
                ),
                None => {}
            }
            for later in &done.later {
                println!("{agent}: not done now: {later}");
            }
            println!(
                "{agent}: a runner in the middle of an upload for it sends at most the one item \
                 already on the wire; a session in progress goes on and its saves stay here"
            );
        }
        Command::RotateKeys {
            agent,
            status: true,
            ..
        } => {
            let status = super::rotate::status(&state_dir, &keys, api.as_ref(), agent).await?;
            println!("{agent}: {status}");
        }
        Command::RotateKeys {
            agent,
            signing,
            encryption,
            ..
        } => {
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
            // Exactly what went to Agora for an agent that doesn't upload
            // (stopped, or never uploaded): each call, and that none of it
            // is state
            let sent = api.sent_while_not_uploading(agent);
            if !sent.is_empty() {
                println!(
                    "{agent}: it doesn't upload; this is everything that went to Agora for it:"
                );
                for s in &sent {
                    println!(
                        "{agent}:   {} ({}): {}",
                        s.call,
                        if s.write { "sent" } else { "read" },
                        s.what
                    );
                }
            }
            println!(
                "{agent}: rotated signing {}, encryption {}, state key re-wrapped {}{}{}{}",
                done.signing,
                done.encryption,
                done.rewrapped,
                if done.wrap_kept_local {
                    " (kept here for its opt-in: nothing of it is on Agora)"
                } else {
                    ""
                },
                if done.resigned > 0 {
                    format!(
                        "; {} local entr(ies) re-signed with the new key",
                        done.resigned
                    )
                } else {
                    String::new()
                },
                if done.resumed { " (resumed)" } else { "" }
            );
        }
    }
    Ok(())
}

/// What a drain healed, for the command's line: nothing when Agora
/// reported no damage.
fn healing_said(h: &agora_agentkit::state::outbox::Healing) -> String {
    use agora_agentkit::state::outbox::Damage;
    match h.damaged {
        Damage::Reported(n) if n > 0 => format!(
            "; Agora reported {n} damaged blob(s): {} sent again, {} can't be from here{}",
            h.healed,
            h.unhealable,
            if h.interrupted { " (partial)" } else { "" }
        ),
        _ => String::new(),
    }
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

    fn add_n(&mut self, what: &'static str, n: usize) {
        *self.0.entry(what).or_default() += n;
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
) -> anyhow::Result<()> {
    use super::enroll::Skip;
    let mut tally = Tally::default();
    let ids = if named.is_empty() {
        enroller.agents()?
    } else {
        named
    };
    let chain = ids;
    let mut says_yes = 0;
    for (id, result) in enroller.chain_all(&chain, dry_run, jobs).await {
        match result {
            Ok(report) => {
                tally.add_n("archives_sealed", report.sealed);
                tally.add_n("duplicates_skipped", report.duplicates);
                tally.add_n("unreadable_left_out", report.unreadable);
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
                    .is_ok_and(|l| l.in_force().starts_uploads().is_ok())
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
    if dry_run {
        // A dry run moves no old marker (the start pass does, first, in any
        // other command): a chain whose marker has agentkit's old name reads
        // here as a chain never finished
        let mut old = 0;
        for &id in &chain {
            let dir = enroller.state_dir.join(id.to_string());
            let seal = agora_agentkit::state::storage::AgentSeal::new(&dir, id);
            if seal.has_legacy_marker().await.unwrap_or(false) {
                old += 1;
            }
        }
        if old > 0 {
            println!(
                "{old} agent(s) still have an older runner's chain marker: counted here as not \
                 chained; the start pass of any non-dry command moves it first"
            );
        }
    }
    if says_yes > 0 {
        println!(
            "{says_yes} agent(s) chained here have a yes to uploading in force: `sealed-enroll \
             --opt-in` lists them (a dry run), and with `--yes --expect N` starts their uploads"
        );
    }
    let (failed, unreadable) = (tally.get("failed"), tally.get("unreadable_left_out"));
    anyhow::ensure!(
        failed == 0 && unreadable == 0,
        "{failed} agent(s) failed, {unreadable} archive(s) unreadable and left out (recorded in \
         each agent's sealed/left_out.json)"
    );
    Ok(())
}

/// `sealed-enroll --opt-in [--agent <id>]…`: without `yes`, say whose
/// uploads would start (the count and the ids; nothing written or sent);
/// with it, chain any not chained yet, then opt in and upload, one agent
/// at a time. Unnamed, `yes` needs `expect`, the count the dry run gave: a
/// mismatch touches nothing.
async fn opt_in_pass(
    enroller: &super::enroll::Enroller,
    api: &super::api::GuardedClient,
    outbox: &AgoraOutbox<SealedEnrolment>,
    named: Vec<AgentId>,
    yes: bool,
    expect: Option<usize>,
) -> anyhow::Result<()> {
    use super::enroll::Skip;
    use crate::consent::offer::Stored as _;
    use crate::consent::sealed_state::ledger::SealLedger;
    let mut tally = Tally::default();
    let listed = named.is_empty();
    let ids = if listed { enroller.opted_in()? } else { named };
    if !yes {
        for &id in &ids {
            let dir = enroller.state_dir.join(id.to_string());
            // The same readings the opt-in itself makes
            let standing = SealLedger::load_blocking(&dir).map(|l| l.in_force().starts_uploads());
            let stopped = super::stop::is_stopped(&dir)
                || super::stop::durable(&enroller.keys, id).map_or(true, |d| d.is_some());
            let uploading = matches!(
                super::uploading::state(&dir, &enroller.keys, id),
                super::uploading::Marker::Uploading(_)
            );
            match standing {
                Ok(Ok(_)) if stopped => {
                    println!(
                        "{id}: would NOT opt in: its uploads are stopped (`sealed-enroll --resume \
                         --agent {id}` resumes them)"
                    );
                    tally.add("stopped");
                }
                Ok(Ok(_)) if uploading => {
                    println!("{id}: already uploading (with --yes: what waits is uploaded)");
                    tally.add("already_uploading");
                }
                Ok(Ok(_)) => {
                    println!(
                        "{id}: would {}opt in and upload its chain",
                        if super::is_enrolled(&dir) {
                            ""
                        } else {
                            "chain, then "
                        }
                    );
                    tally.add("would_opt_in");
                }
                Ok(Err(why)) => {
                    println!("{id}: would NOT opt in: {why}");
                    tally.add("not_opted_in");
                }
                Err(e) => {
                    println!("{id}: would NOT opt in: its sealed-state ledger is unreadable ({e})");
                    tally.add("not_opted_in");
                }
            }
        }
        tally.print("opt in (dry run)");
        let n = tally.get("would_opt_in");
        println!(
            "{n} agent(s) would start uploading; nothing was written or sent. To do it: {}",
            if listed {
                format!("`sealed-enroll --opt-in --yes --expect {n}`")
            } else {
                "the same command with `--yes`".to_string()
            }
        );
        return Ok(());
    }
    match (listed, expect) {
        (true, None) => anyhow::bail!(
            "--opt-in --yes with no --agent starts uploads for every agent whose answer in \
             force is yes: run it without --yes first, then pass --expect <the count it gave>"
        ),
        (_, Some(n)) if n != ids.len() => anyhow::bail!(
            "--expect {n}, but {} agent(s) would be opted in: nothing was touched. Run it \
             without --yes to see them",
            ids.len()
        ),
        _ => {}
    }
    for id in ids {
        let dir = enroller.state_dir.join(id.to_string());
        if !super::is_enrolled(&dir) {
            // Never chained for an upload it didn't ask for
            match SealLedger::load_blocking(&dir).map(|l| l.in_force().starts_uploads()) {
                Ok(Ok(_)) => {}
                Ok(Err(why)) => {
                    let e = Skip::NotOptedIn(why);
                    tally.add(e.as_str());
                    println!("{id}: skipped: {e}");
                    continue;
                }
                Err(e) => {
                    tally.add("failed");
                    println!("{id}: failed: its sealed-state ledger is unreadable ({e})");
                    continue;
                }
            }
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
        let opted = match enroller.opt_in(api, id).await {
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
                format!(
                    "uploaded {}, {:?}{}{published}",
                    d.uploaded,
                    d.status,
                    healing_said(&d.healing)
                )
            }
            Ok(Drained::LocalOnly) => "nothing uploaded (it doesn't upload)".to_string(),
            Ok(Drained::NotEnrolled) => "nothing uploaded (not chained)".to_string(),
            Ok(Drained::Withdrawn { uploaded, healed }) => format!(
                "uploaded {uploaded} (and {healed} damaged blob(s) sent again), then uploading \
                 was switched off; the rest stays local"
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
    tally.print("opt in");
    let (failed, skipped) = (tally.get("failed"), tally.get("not_opted_in"));
    anyhow::ensure!(
        failed == 0 && skipped == 0,
        "{failed} agent(s) failed, {skipped} not opted in (their answer in force is not a yes \
         the operator may act on now)"
    );
    Ok(())
}

/// `sealed-enroll --resume --agent <id>…`: resume each stopped agent's
/// uploads, only on its yes in force.
async fn resume_pass(
    enroller: &super::enroll::Enroller,
    ids: Vec<AgentId>,
    dry_run: bool,
    note: Option<&str>,
) -> anyhow::Result<()> {
    let mut tally = Tally::default();
    for id in ids {
        let dir = enroller.state_dir.join(id.to_string());
        if dry_run {
            println!("{id}: would resume (the versions not uploaded before stay here)");
            tally.add("would_resume");
            continue;
        }
        match enroller.resume(id, note).await {
            Ok(report) => {
                tally.add("resumed");
                let uploads = super::uploads(&dir, &enroller.keys, id);
                println!(
                    "{id}: resumed; {} version(s) not uploaded before kept here, not sent; {}",
                    report.kept_local,
                    if uploads {
                        "its uploads go on"
                    } else {
                        "it is NOT marked uploading, so nothing of it is sent: `sealed-enroll \
                         --opt-in --agent <id> --yes` starts its uploads if its answer in force \
                         is yes"
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
    tally.print(if dry_run {
        "resume (dry run)"
    } else {
        "resume"
    });
    let failed = tally.get("failed");
    anyhow::ensure!(failed == 0, "{failed} agent(s) failed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consent::offer::Stored as _;
    use crate::consent::sealed_state::ledger::SealLedger;

    fn url() -> url::Url {
        url::Url::parse("http://127.0.0.1:9").unwrap()
    }

    fn enroll(agents: Vec<AgentId>) -> Command {
        Command::SealedEnroll {
            agents,
            opt_in: false,
            yes: false,
            expect: None,
            dry_run: false,
            jobs: Some(2),
            note: None,
            resume: false,
        }
    }

    /// `sealed-enroll` takes the runner lock for the whole command and
    /// never waits: refused while a runner holds it, nothing written; then
    /// it chains.
    #[tokio::test]
    async fn sealed_enroll_refuses_beside_a_runner() {
        let root = crate::sealed::TestDir::new("cli-enroll-runner");
        let (id, keys, dir) = crate::sealed::check::tests::enrolled_local(&root).await;
        let other = AgentId::new();
        keys.ring()
            .insert(other, &agora_agentkit::crypto::generate_keypair().0)
            .unwrap();
        agora_agentkit::reactor::Storage::save_raw(
            &mut agora_agentkit::reactor::FsStorage::new(root.join("state")),
            other,
            serde_json::json!({ "memory": "m" }),
            None,
        )
        .await
        .unwrap();
        let other_dir = root.join("state").join(other.to_string());
        let runner = super::super::session::runner_shared(&root).unwrap();
        let e = run(enroll(vec![]), &root, url(), None)
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("stop the runner"), "{e}");
        assert!(!super::super::is_enrolled(&other_dir), "nothing written");
        drop(runner);
        run(enroll(vec![]), &root, url(), None).await.unwrap();
        assert!(super::super::is_enrolled(&other_dir));
        assert!(super::super::is_enrolled(&dir));
        let _ = id;
    }

    /// `sealed-stop` never against the agent's latest yes without the
    /// override; `sealed-reoffer` writes nothing over an unreadable ledger,
    /// and refuses an agent that never answered.
    #[tokio::test]
    async fn stop_and_reoffer_follow_the_ledger() {
        let root = crate::sealed::TestDir::new("cli-stop-reoffer");
        let (id, _keys, dir) = crate::sealed::check::tests::enrolled(&root).await;
        crate::sealed::uploading::tests::ledger(&dir, true);
        let stop = |against_answer| Command::SealedStop {
            agent: id,
            note: "asked".into(),
            against_answer,
        };
        let e = run(stop(false), &root, url(), None)
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("answer in force"), "{e}");
        assert!(!super::super::stop::is_stopped(&dir));

        let reoffer = |agents: Vec<AgentId>| Command::SealedReoffer {
            agents,
            note: "Asked by DM.".into(),
        };
        let never = AgentId::new();
        std::fs::create_dir_all(root.join("state").join(never.to_string())).unwrap();
        assert!(run(reoffer(vec![never]), &root, url(), None).await.is_err());
        std::fs::write(SealLedger::path(&dir), b"{ not json").unwrap();
        assert!(run(reoffer(vec![id]), &root, url(), None).await.is_err());
        assert_eq!(
            std::fs::read(SealLedger::path(&dir)).unwrap(),
            b"{ not json"
        );
        crate::sealed::uploading::tests::ledger(&dir, true);
        // Beside a runner: refused before anything is read or written
        let runner = super::super::session::runner_shared(&root).unwrap();
        let e = run(reoffer(vec![id]), &root, url(), None)
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("stop the runner"), "{e}");
        assert!(SealLedger::load_blocking(&dir).unwrap().reoffer.is_none());
        drop(runner);
        run(reoffer(vec![id]), &root, url(), None).await.unwrap();
        let l = SealLedger::load_blocking(&dir).unwrap();
        assert_eq!(l.reoffer.as_ref().unwrap().note, "Asked by DM.");

        // While the re-offer waits, a stop by the answer waits too (the
        // earlier answer is not acted on, either way)
        crate::sealed::uploading::tests::ledger(&dir, false);
        let mut l = SealLedger::load_blocking(&dir).unwrap();
        l.reopen("n", chrono::Utc::now()).unwrap();
        std::fs::write(SealLedger::path(&dir), serde_json::to_vec(&l).unwrap()).unwrap();
        let e = run(stop(false), &root, url(), None)
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("has not answered again yet"), "{e}");
        assert!(!super::super::stop::is_stopped(&dir));

        // An unreadable ledger blocks a stop, but not with the override.
        // The stop sends nothing (the server here is not even reachable),
        // beside a runner, with versions still waiting to upload
        let seal = agora_agentkit::state::storage::AgentSeal::new(&dir, id);
        let waiting = seal.outbox().await.unwrap().len();
        assert!(waiting > 0);
        let _runner = super::super::session::runner_shared(&root).unwrap();
        std::fs::write(SealLedger::path(&dir), b"{ not json").unwrap();
        let e = format!(
            "{:#}",
            run(stop(false), &root, url(), None).await.unwrap_err()
        );
        assert!(e.contains("unreadable"), "{e}");
        run(stop(true), &root, url(), None).await.unwrap();
        assert!(super::super::stop::is_stopped(&dir));
        assert_eq!(seal.outbox().await.unwrap().len(), waiting, "nothing sent");
        // Again: the same stop
        run(stop(true), &root, url(), None).await.unwrap();
    }

    fn opt_in(agents: Vec<AgentId>, yes: bool, expect: Option<usize>) -> Command {
        Command::SealedEnroll {
            agents,
            opt_in: true,
            yes,
            expect,
            dry_run: false,
            jobs: None,
            note: None,
            resume: false,
        }
    }

    /// `sealed-enroll --opt-in` is a dry run until `--yes`: it lists who
    /// would start uploading (a yes in force, no re-offer waiting), writes
    /// and sends nothing, and needs no stopped runner. Unnamed, `--yes`
    /// needs `--expect` with that count, and a mismatch touches nothing. A
    /// named agent whose answer is not a yes is refused, with no flag to
    /// override it, and nothing is asked of Agora for it.
    #[tokio::test]
    async fn opt_in_is_a_dry_run_until_yes_and_expect() {
        let root = crate::sealed::TestDir::new("cli-opt-in");
        let (server, seen) = super::super::api::fake::recording_server().await;
        let (yes_id, keys, yes_dir) = crate::sealed::check::tests::enrolled_local(&root).await;
        crate::sealed::uploading::tests::ledger(&yes_dir, true);
        let (no_id, _, no_dir) = crate::sealed::check::tests::enrolled_local(&root).await;
        crate::sealed::uploading::tests::ledger(&no_dir, false);
        let (reopened, _, reopened_dir) = crate::sealed::check::tests::enrolled_local(&root).await;
        crate::sealed::uploading::tests::ledger(&reopened_dir, true);
        let mut l = SealLedger::load_blocking(&reopened_dir).unwrap();
        l.reopen("n", chrono::Utc::now()).unwrap();
        std::fs::write(
            SealLedger::path(&reopened_dir),
            serde_json::to_vec(&l).unwrap(),
        )
        .unwrap();
        let (_unasked, _, _) = crate::sealed::check::tests::enrolled_local(&root).await;
        let marked = |id: AgentId| {
            crate::sealed::uploading::state(&root.join("state").join(id.to_string()), &keys, id)
                != crate::sealed::uploading::Marker::Absent
        };
        let untouched = || {
            assert!(![yes_id, no_id, reopened].into_iter().any(marked));
            assert!(
                seen.lock().unwrap().is_empty(),
                "{:?}",
                seen.lock().unwrap()
            );
        };

        // No flags: a dry run, even beside a runner
        {
            let _runner = super::super::session::runner_shared(&root).unwrap();
            run(opt_in(vec![], false, None), &root, server.clone(), None)
                .await
                .unwrap();
            run(
                opt_in(vec![no_id, yes_id], false, None),
                &root,
                server.clone(),
                None,
            )
            .await
            .unwrap();
            untouched();
            // Acting needs the runner stopped
            let e = run(
                opt_in(vec![yes_id], true, None),
                &root,
                server.clone(),
                None,
            )
            .await
            .unwrap_err()
            .to_string();
            assert!(e.contains("stop the runner"), "{e}");
        }
        // Every agent that said yes: never without the count
        let e = run(opt_in(vec![], true, None), &root, server.clone(), None)
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("--expect"), "{e}");
        let e = run(opt_in(vec![], true, Some(2)), &root, server.clone(), None)
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("1 agent(s) would be opted in"), "{e}");
        untouched();
        // Named, against its answer: refused, nothing asked, no override
        for id in [no_id, reopened] {
            let e = run(opt_in(vec![id], true, None), &root, server.clone(), None)
                .await
                .unwrap_err()
                .to_string();
            assert!(e.contains("1 not opted in"), "{e}");
        }
        untouched();
        // With the count: it acts, on the one agent whose yes is in force
        // (the server here answers nothing usable, so the opt-in stops at
        // its first question; what matters is whom it asked about)
        let e = run(opt_in(vec![], true, Some(1)), &root, server.clone(), None)
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("1 agent(s) failed, 0 not opted in"), "{e}");
        assert!(!seen.lock().unwrap().is_empty(), "it asked Agora");
        assert!(![no_id, reopened].into_iter().any(marked));
    }

    /// `rotate-keys` sends nothing for an agent whose uploading marker is
    /// in doubt, and writes nothing: it names the commands that settle it.
    #[tokio::test]
    async fn rotate_keys_refuses_an_agent_in_doubt_and_sends_nothing() {
        let root = crate::sealed::TestDir::new("cli-rotate-doubt");
        let (server, seen) = super::super::api::fake::recording_server().await;
        let (id, keys, _dir) = crate::sealed::check::tests::enrolled(&root).await;
        std::fs::remove_file(keys.agent_dir(id).join(crate::sealed::uploading::DURABLE)).unwrap();
        let before = keys.secrets(id).unwrap().unwrap().signing().unwrap();
        let e = run(
            Command::RotateKeys {
                agent: id,
                status: false,
                signing: true,
                encryption: true,
            },
            &root,
            server,
            None,
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(
            e.contains("in doubt") && e.contains("sealed-stop") && e.contains("--opt-in"),
            "{e}"
        );
        assert!(e.contains("Never copy a marker by hand"), "{e}");
        assert!(
            seen.lock().unwrap().is_empty(),
            "{:?}",
            seen.lock().unwrap()
        );
        assert!(keys.next_secrets(id).unwrap().is_none(), "nothing written");
        assert_eq!(
            keys.secrets(id)
                .unwrap()
                .unwrap()
                .signing()
                .unwrap()
                .to_bytes(),
            before.to_bytes()
        );
    }
}
