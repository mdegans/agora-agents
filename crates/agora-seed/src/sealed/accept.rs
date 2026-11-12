//! `agora-seed sealed-accept --agent <id> [--note "…"]`: the only way out of
//! a refusal, and an operator's act.
//!
//! It seals the agent's current state as a new snapshot on its chain (so
//! the next start finds it intact), clears the refusal (kept as
//! `refused.cleared.<utc>`), and adds the note to the agent's notice: its
//! next session opens with "Your session on <date> did not run: <reason>.
//! Your operator accepted the current state with this note: …".
//!
//! The note is kept in `sealed/notice.json`, beside the chain. It is not
//! uploaded: no v1 entry kind carries it (the history-tools PR adds one).
//!
//! A refusal for a missing `head.json` is accepted by rebuilding it from
//! the agent's chain on Agora, verified with the runner's own trust
//! (agentkit's `adopt_server_head`): when the state here is that head's,
//! the refusal is simply cleared (`--note` optional); when it isn't, the
//! agent stays refused unless `--note` seals its current state on top.
//! Never a second genesis. With no chain on Agora either, `head.json` must
//! come back from a backup. Like every operator command it refuses while
//! the agent has a session, and holds the agent's `sealed/lock`
//! throughout.
//!
//! **An agent that never uploaded** makes no call to Agora: `--note` works
//! as above from what is here, a lost `head.json` is rebuilt from the
//! local chain ([`super::restore::adopt_local`]), and `--rebase` is
//! refused (nothing of it is on Agora to join). A fork in its chain (two
//! saves from one version) is resolved by `--keep` ([`keep_line`]): the
//! operator chooses the line that goes on, and the other is kept under
//! `sealed/lost/<utc>/` ([`super::lost`]).
//!
//! **Many at once** ([`accept_many`], `--agents-file`): one note, each
//! agent under its own lock, after a fleet-wide event; the count must be
//! confirmed with `--expect N`, or nothing is touched.

use std::path::Path;

use agora_agentkit::enums::StateEntryKind;
use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::{FsStorage, Storage};
use agora_agentkit::state::codec::KeyframePolicy;
use agora_agentkit::state::storage::{Adopted, AgentSeal};
use agora_agentkit::state::verify::{HeadRef, HeadStatus, head_status, verify_chain};
use anyhow::Context;
use chrono::Utc;

use super::KeyStore;
use super::api::SealedApi;
use super::notice::{Diff, Notice, Refusal};

/// What `sealed-accept` did: hashes only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Accepted {
    pub reason: Refusal,
    pub head: HeadRef,
    /// A lost `head.json` was rebuilt from Agora's verified chain.
    pub adopted: bool,
}

/// The note when a head rebuilt from the chain kept here (an agent that
/// never uploaded) matched and the operator gave none.
pub const REBUILT_LOCAL_NOTE: &str = "the record of your last sealed version was missing \
     here and was rebuilt from your chain as your runner keeps it here (nothing of yours is on \
     Agora), after checking it with your runner's own keys; your state matched it, so nothing \
     about your state changed.";

/// The note when the rebuilt head matched and the operator gave none.
pub const REBUILT_NOTE: &str = "the record of your last sealed version was missing here and \
     was rebuilt from your chain on Agora, after checking it; your state matched it, so \
     nothing about your state changed.";

/// Accept `id`'s current state. A lost `head.json` is first rebuilt from
/// Agora's chain ([`AgentSeal::adopt_server_head`]): if the state matches
/// its head the refusal is cleared with no new version; if not, it stays
/// refused unless `note` is given, which seals the current state on top
/// of the adopted head. Never a second genesis.
pub async fn accept(
    state_dir: &Path,
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
    note: Option<&str>,
) -> anyhow::Result<Accepted> {
    let note = note.map(str::trim).filter(|n| !n.is_empty());
    let agent_dir = state_dir.join(id.to_string());
    let seal = AgentSeal::new(&agent_dir, id);
    // An id naming no agent here: nothing made for it (the lock would make
    // its directory)
    anyhow::ensure!(
        std::fs::exists(&agent_dir)?,
        "no agent {id} here ({}): nothing was written",
        agent_dir.display()
    );
    // No session in progress (a journal older than the refusal is one the
    // refused agent can't clear), and the lock for the whole command
    let lock = super::session::take(state_dir, id, true).await?;
    // sealed/ lost (its enrolment marker and head.json with it), the state
    // here: enrolled by its keys, and accepted on top of Agora's chain.
    // The start check refuses the run for it without a marker
    anyhow::ensure!(
        !std::fs::read_to_string(seal.dir().join("refused"))
            .is_ok_and(|r| Refusal::from_marker(&r) == Refusal::StateTooNew),
        "agent {id}'s state was written by a newer runner than this one: upgrade the runner; \
         don't restore or accept over it"
    );
    let lost_seal = !seal.is_enrolled().await?
        && seal.head().await?.is_none()
        && keys.state_key_file(id)?.is_some();
    anyhow::ensure!(
        lost_seal || seal.is_enrolled().await?,
        "agent {id} is not enrolled in sealed state"
    );
    anyhow::ensure!(
        lost_seal || seal.is_refused().await?,
        "agent {id} is not refused; nothing to accept"
    );
    let reason = if lost_seal && !seal.is_refused().await? {
        Refusal::HeadMissing
    } else {
        Refusal::from_marker(
            &std::fs::read_to_string(seal.dir().join("refused")).unwrap_or_default(),
        )
    };
    let current = FsStorage::new(state_dir)
        .load_raw(id)
        .await
        .context("the agent's state won't load")?
        .value;
    let reach = super::version::reach(&agent_dir, keys, id);
    // A lost state key: from Agora's wrap when it is there
    if keys.signing(id).is_some() && keys.state_key_file(id)?.is_none() {
        super::keys::recover_state_key(keys, &agent_dir, api, id).await?;
    }
    let k = keys
        .seal_keys(id)
        .context("no sealing keys for the agent")?;

    let mut adopted = None;
    if seal.head().await?.is_none() && !reach.on_agora() {
        // Nothing on Agora: from the agent's own chain here
        let trusted = keys
            .trusted(id)?
            .context("no signing key to verify the chain kept here with")?;
        let result =
            super::restore::adopt_local(&seal, &lock, &trusted, &k, Some(&current)).await?;
        tracing::warn!(
            event_type = "sealed_state_head_adopted",
            agent_id = %id,
            source = "local",
            matched = matches!(result, Adopted::Intact(_)),
            "head.json rebuilt from the verified chain kept here"
        );
        adopted = Some(result);
    } else if seal.head().await?.is_none() {
        let signing = keys.signing(id).context("no signing key for the agent")?;
        let chain = super::api::chain(api, id, &signing)
            .await
            .map_err(anyhow::Error::msg)
            .context("reading the agent's chain on Agora")?;
        anyhow::ensure!(
            !chain.is_empty(),
            "agent {id}'s sealed/head.json is missing and Agora holds no chain: restore \
             head.json from a backup of its directory"
        );
        let trusted = keys
            .trusted(id)?
            .context("no signing key to verify Agora's chain with")?;
        let signing = &signing;
        let fetch = move |sha: agora_agentkit::govlog::Sha256Hex| async move {
            api.blob(id, signing, &sha)
                .await
                .map_err(Box::<dyn std::error::Error + Send + Sync>::from)
        };
        let result = seal
            .adopt_server_head(&lock, &chain, &trusted, &k, Some(&current), fetch)
            .await
            .context("rebuilding head.json from Agora's chain")?;
        tracing::warn!(
            event_type = "sealed_state_head_adopted",
            agent_id = %id,
            matched = matches!(result, Adopted::Intact(_)),
            "head.json rebuilt from the verified chain on Agora"
        );
        adopted = Some(result);
    }

    let now = Utc::now();
    let head = match (adopted, note) {
        (Some(Adopted::Intact(head)), note) => {
            if seal.is_refused().await? {
                seal.clear_refusal().await.context("clearing the refusal")?;
            }
            let rebuilt = if reach.on_agora() {
                REBUILT_NOTE
            } else {
                REBUILT_LOCAL_NOTE
            };
            Notice::accept(&agent_dir, note.unwrap_or(rebuilt), reason, None)
                .context("writing the notice")?;
            head
        }
        (Some(Adopted::Mismatch(head)), None) => anyhow::bail!(
            "head.json was rebuilt (head {} at seq {}), but the state here doesn't match it: \
             the agent stays refused. Pass --note to seal its current state on top",
            head.entry_hash,
            head.seq
        ),
        (None, None) => anyhow::bail!("--note is required: the agent reads it"),
        (_, Some(note)) => {
            // What the agent will be shown beside the note: the last sealed
            // version against the state accepted. Local only, never logged
            let diff = match super::version::last_sealed(&seal, keys, api, id).await {
                // line_diff can take seconds by design: off the runtime
                Ok(last) => {
                    let after = current.clone();
                    tokio::task::spawn_blocking(move || Diff::between(&last.state, &after))
                        .await
                        .unwrap_or_else(|e| {
                            tracing::warn!(agent_id = %id, error = %e, "the diff task failed");
                            Diff::unavailable()
                        })
                }
                Err(e) => {
                    tracing::warn!(
                        event_type = "sealed_accept_no_diff",
                        agent_id = %id,
                        error = %e,
                        "the last sealed version couldn't be opened; accepted without a diff"
                    );
                    Diff::unavailable()
                }
            };
            super::crash::point("accept:diffed")?;
            // The note is written before the refusal is cleared: a crash in
            // between leaves the agent refused, and a rerun rewrites it. A
            // refusal at the next start sets an unshown acceptance aside
            Notice::accept(&agent_dir, note, reason, Some(diff)).context("writing the notice")?;
            super::crash::point("accept:noticed")?;
            if seal.is_refused().await? {
                seal.clear_refusal().await.context("clearing the refusal")?;
            }
            super::crash::point("accept:cleared")?;
            let head = super::session::append(
                &seal,
                &lock,
                &k,
                &KeyframePolicy::default(),
                StateEntryKind::Snapshot,
                &current,
                now,
            )
            .await
            .context(
                "sealing the accepted state (the refusal is cleared; the next start checks it)",
            )?;
            super::crash::point("accept:appended")?;
            super::enroll::record_sizes(&agent_dir, head, now, &current);
            head
        }
    };
    if lost_seal {
        seal.enroll()
            .await
            .context("writing the enrolment marker")?;
    }
    // The roster takes the head written, whatever it held: an accept of a
    // rolled-back agent sets it lower on purpose (head.json first, then
    // the roster: the runner's keeper reads heads from disk)
    super::roster::replace_for(state_dir, id, Some(head)).await?;
    drop(lock);
    tracing::warn!(
        event_type = "sealed_state_accepted",
        agent_id = %id,
        reason = reason.as_str(),
        head = %head.entry_hash,
        seq = head.seq,
        adopted = adopted.is_some(),
        "the operator accepted a refused agent's current state"
    );
    Ok(Accepted {
        reason,
        head,
        adopted: adopted.is_some(),
    })
}

/// What `sealed-accept --rebase` did: hashes and counts only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rebased {
    /// The server head it was sealed on top of; `None`: Agora held nothing,
    /// and a new chain began (its genesis records the break).
    pub onto: Option<HeadRef>,
    /// The new head.
    pub head: HeadRef,
    /// Outbox items set aside (kept, with their blobs).
    pub set_aside: usize,
}

/// Tests: [`rebase_locked`] as `sealed-accept --rebase` runs it, under
/// the runner lock held to its end.
#[cfg(test)]
pub async fn rebase(
    state_dir: &Path,
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
    note: &str,
) -> anyhow::Result<Rebased> {
    // Before anything is read or written: no runner on this data dir (it
    // could be saving), held to the end
    let data_dir = state_dir
        .parent()
        .context("the state directory has no parent data directory")?;
    let _runner = super::session::runner_exclusive(data_dir)?;
    rebase_locked(state_dir, keys, api, id, note).await
}

/// Resolve a held fork (v1 has no merges), or versions Agora lost (a held
/// LocalMissing): the current state is sealed as a keyframe child of
/// Agora's head, which is first verified with the runner's own trust. For
/// a fork the versions that went the other way are kept locally as
/// `.displaced`; for a loss, see [`super::lost`] (and, Agora holding
/// nothing, a new chain whose genesis records the break). A refusal is
/// cleared first. The agent's next session is told, with `note`.
///
/// The runner lock (`runner.lock`, exclusive) is the caller's:
/// `sealed-accept` takes it before the start pass, held to its end.
pub async fn rebase_locked(
    state_dir: &Path,
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
    note: &str,
) -> anyhow::Result<Rebased> {
    let note = note.trim();
    anyhow::ensure!(!note.is_empty(), "--note is required: the agent reads it");
    let reach = super::version::reach(&state_dir.join(id.to_string()), keys, id);
    anyhow::ensure!(
        reach.on_agora(),
        "agent {id} never uploaded: nothing of it is on Agora, so there is no line of Agora's \
         to join it to. A fork in the chain kept here is resolved with `sealed-accept --agent \
         {id} --keep <head|ord> --note \"…\"`; nothing was changed"
    );
    // No session in progress. The start check below takes the agent's
    // sealed/lock itself, so that lock is taken after it
    super::session::no_session(state_dir, id, true)?;
    let agent_dir = state_dir.join(id.to_string());
    anyhow::ensure!(
        std::fs::exists(&agent_dir)?,
        "no agent {id} here ({}): nothing was written",
        agent_dir.display()
    );
    let seal = AgentSeal::new(&agent_dir, id);
    anyhow::ensure!(
        seal.is_enrolled().await?,
        "agent {id} is not enrolled in sealed state"
    );
    let signing = keys.signing(id).context("no signing key for the agent")?;
    let k = keys
        .seal_keys(id)
        .context("no sealing keys for the agent")?;
    let chain = super::api::chain(api, id, &signing)
        .await
        .map_err(anyhow::Error::msg)
        .context("reading the agent's chain on Agora")?;
    let trusted = keys
        .trusted(id)?
        .context("no signing key to verify Agora's chain with")?;
    let report = verify_chain(id, &chain, &trusted);
    anyhow::ensure!(
        report.intact,
        "Agora's chain does not verify ({:?}): not rebasing onto it",
        report.problems
    );
    let current = FsStorage::new(state_dir)
        .load_raw(id)
        .await
        .context("the agent's state won't load")?
        .value;
    // A resolution of Agora's loss cut short: finished, as it began
    if let Some((dir, intent)) = super::lost::pending(&seal)? {
        anyhow::ensure!(
            intent.local_fork.is_none(),
            "a fork resolution (`sealed-accept --keep`) is unfinished under {}: run that \
             again first",
            dir.display()
        );
        let lock = super::session::take(state_dir, id, true).await?;
        return finish_lost(
            &seal, &lock, &k, &agent_dir, &dir, &intent, &current, &chain, note,
        )
        .await;
    }
    let local_head = seal.head().await?.map(|h| h.head);
    let outbox: Vec<_> = seal
        .outbox()
        .await?
        .into_iter()
        .map(|i| i.entry.entry)
        .collect();
    let hashes: Vec<_> = chain.iter().map(|e| e.entry_hash()).collect();
    let status = head_status(local_head.as_ref().map(|h| &h.entry_hash), &outbox, &hashes);
    // Versions uploaded from here that Agora no longer has
    let lost: Vec<_> = seal
        .sent()
        .await?
        .into_iter()
        .filter(|i| !hashes.contains(&i.entry.entry_hash()))
        .collect();
    let agora_lost = status == HeadStatus::LocalMissing || !lost.is_empty();
    anyhow::ensure!(
        agora_lost || matches!(status, HeadStatus::ServerAhead | HeadStatus::Diverged),
        "no fork to resolve and nothing lost on Agora: the local head stands {status:?} to \
         Agora's"
    );
    if agora_lost {
        // Never on a cold run: the start check must hold (or have refused)
        // the agent for it, now
        let refused = seal.is_refused().await?;
        if !refused {
            let out = super::check::check(
                &agent_dir,
                id,
                &agora_agentkit::state::storage::plaintext_hash(&current),
                super::sizes::Sizes::of(&current),
                keys,
                api,
            )
            .await;
            anyhow::ensure!(
                matches!(
                    out,
                    super::check::Outcome::Run {
                        hold: Some(super::check::Hold::HeadMissing),
                        ..
                    }
                ),
                "agent {id} is not held for versions Agora lost (the start check says {out:?}): \
                 nothing to resolve"
            );
        }
        // And one read isn't enough to call versions lost: Agora's head,
        // asked apart, must say the same
        let head = api.head(id, &signing).await.map_err(anyhow::Error::msg)?;
        let agrees = match (chain.last(), head) {
            (None, None) => true,
            (Some(last), Some(h)) => h == last.entry_hash(),
            _ => false,
        };
        anyhow::ensure!(
            agrees,
            "Agora's answers disagree (its chain and its head); try again later"
        );
        let onto = chain.last().map(|e| HeadRef {
            entry_hash: e.entry_hash(),
            seq: e.entry.seq,
        });
        let lock = super::session::take(state_dir, id, true).await?;
        let (dir, intent) = super::lost::begin(&seal, onto, &lost, note).await?;
        return finish_lost(
            &seal, &lock, &k, &agent_dir, &dir, &intent, &current, &chain, note,
        )
        .await;
    }
    let lock = super::session::take(state_dir, id, true).await?;
    if seal.is_refused().await? {
        seal.clear_refusal().await.context("clearing the refusal")?;
    }
    let server_head = chain
        .last()
        .context("Agora holds no chain: nothing to rebase onto")?;
    let set_aside = outbox.len();
    let head = seal
        .rebase_onto(&lock, &k, &server_head.entry, &current)
        .await
        .context("sealing the current state on top of Agora's head")?;
    super::enroll::record_sizes(&agent_dir, head, Utc::now(), &current);
    Notice::joined(&agent_dir, note).context("writing the notice")?;
    super::roster::replace_for(state_dir, id, Some(head)).await?;
    drop(lock);
    let onto = HeadRef {
        entry_hash: server_head.entry_hash(),
        seq: server_head.entry.seq,
    };
    tracing::warn!(
        event_type = "sealed_state_rebased",
        agent_id = %id,
        onto = %onto.entry_hash,
        head = %head.entry_hash,
        set_aside,
        "the operator joined a held fork onto Agora's head"
    );
    Ok(Rebased {
        onto: Some(onto),
        head,
        set_aside,
    })
}

#[allow(clippy::too_many_arguments)]
async fn finish_lost(
    seal: &AgentSeal,
    lock: &agora_agentkit::state::storage::SealLock,
    k: &agora_agentkit::state::storage::SealKeys,
    agent_dir: &Path,
    dir: &Path,
    intent: &super::lost::Intent,
    current: &serde_json::Value,
    chain: &[agora_agentkit::state::SignedStateEntry],
    note: &str,
) -> anyhow::Result<Rebased> {
    let id = seal.agent_id();
    let head = super::lost::finish(seal, lock, k, dir, intent, current, chain.last()).await?;
    if seal.is_refused().await? {
        seal.clear_refusal().await.context("clearing the refusal")?;
    }
    super::enroll::record_sizes(agent_dir, head, Utc::now(), current);
    let state_dir = agent_dir
        .parent()
        .context("the agent directory has no parent state directory")?;
    super::roster::replace_for(state_dir, id, Some(head)).await?;
    Notice::agora_lost(
        agent_dir,
        note,
        intent.onto.is_none(),
        intent.items.len(),
        &super::lost::kept_text(intent),
    )
    .context("writing the notice")?;
    tracing::warn!(
        event_type = "sealed_state_agora_lost_resolved",
        agent_id = %id,
        onto = ?intent.onto.map(|o| o.entry_hash),
        head = %head.entry_hash,
        set_aside = intent.items.len(),
        kept = %dir.display(),
        "versions Agora lost resolved: the state sealed on top of what Agora holds"
    );
    Ok(Rebased {
        onto: intent.onto,
        head,
        set_aside: intent.items.len(),
    })
}

/// Which line `sealed-accept --keep` keeps: the head's, or the displaced
/// line whose tip is the outbox item numbered `ord`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keep {
    Head,
    Ord(u64),
}

impl std::str::FromStr for Keep {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim() {
            "head" => Ok(Self::Head),
            n => n.parse().map(Self::Ord).map_err(|_| {
                format!(
                    "`{n}`: name `head`, or the outbox number of a displaced line's tip (the \
                     choices are listed when the wrong one is named)"
                )
            }),
        }
    }
}

/// What `sealed-accept --keep` did: hashes and counts only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kept {
    /// The kept line's tip, now the head.
    pub head: HeadRef,
    /// Outbox items set aside (kept, with their blobs and copies).
    pub set_aside: usize,
    /// The state became the kept tip's (the one here archived).
    pub state_changed: bool,
    /// Where the copies are.
    pub kept: std::path::PathBuf,
}

/// Resolve a fork in a chain kept here (an agent that doesn't upload): two
/// saves from one version, recorded in `head.json`'s `displaced`. `keep`
/// names the line that goes on, by its tip: the head, or a displaced line's
/// tip (never an ancestor of another line: that would roll a line back).
/// Before anything moves, the kept line is verified and, when its tip isn't
/// the head, its state opened and checked against its recorded hash. Then
/// every outbox item off it is kept under `sealed/lost/<utc>/` and set
/// aside as `.displaced` ([`super::lost`]), the kept line's items set aside
/// before are put back, `head.json` names the kept tip, and when that isn't
/// the head, its state becomes the agent's, the state here archived by
/// FsStorage. The agent is told, with `note`. A cut run is finished by the
/// next, as it began; one that moved nothing and can't go on is abandoned
/// (its `INTENT` kept as `ABANDONED.<utc>`). No call to Agora.
pub async fn keep_line(
    state_dir: &Path,
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
    keep: Keep,
    note: &str,
) -> anyhow::Result<Kept> {
    use agora_agentkit::state::storage::{OutboxItem, as_reloaded, plaintext_hash};
    let note = note.trim();
    anyhow::ensure!(!note.is_empty(), "--note is required: the agent reads it");
    let agent_dir = state_dir.join(id.to_string());
    anyhow::ensure!(
        std::fs::exists(&agent_dir)?,
        "no agent {id} here ({}): nothing was written",
        agent_dir.display()
    );
    let seal = AgentSeal::new(&agent_dir, id);
    let reach = super::version::reach(&agent_dir, keys, id);
    anyhow::ensure!(
        reach != super::version::Reach::Uploads,
        "agent {id} uploads: a fork against Agora's copy is resolved with `--rebase`"
    );
    let lock = super::session::take(state_dir, id, true).await?;
    let k = keys
        .seal_keys(id)
        .context("no sealing keys for the agent")?;
    let trusted = keys
        .trusted(id)?
        .context("no signing key to verify the chain kept here with")?;
    // Every entry here: sent, committed, and set aside as `.displaced`
    let all_items = async || -> anyhow::Result<Vec<OutboxItem>> {
        let mut all = seal.sent().await?;
        all.extend(seal.outbox().await?);
        all.extend(super::lost::displaced_items(&seal)?);
        Ok(all)
    };
    // The kept line verified, and its state opened and checked: before
    // anything moves
    let prove = async |tip: &HeadRef,
                       tip_is_head: bool|
           -> anyhow::Result<(
        Vec<agora_agentkit::state::SignedStateEntry>,
        Option<agora_agentkit::govlog::Sha256Hex>,
        Option<serde_json::Value>,
    )> {
        let items = all_items().await?;
        let entries: Vec<_> = items.iter().map(|i| i.entry.clone()).collect();
        let line = super::check::main_line(&entries, &tip.entry_hash)
            .context("the kept line has a missing link here")?;
        let report = verify_chain(id, &line, &trusted);
        anyhow::ensure!(
            report.intact && report.head == Some(*tip),
            "the kept line doesn't verify ({:?}): nothing was changed",
            report.problems
        );
        let snapshot = line
            .iter()
            .rev()
            .find(|e| e.entry.kind == StateEntryKind::Snapshot)
            .cloned()
            .context("the kept line holds no snapshot")?;
        let recorded = items
            .iter()
            .find(|i| i.entry.entry_hash() == snapshot.entry_hash())
            .and_then(|i| i.plaintext_sha256);
        if tip_is_head {
            return Ok((line, recorded, None));
        }
        let state = super::version::open_reach(&seal, api, &k, &line, &snapshot.entry, reach)
            .await
            .context("opening the kept line's latest version: nothing was changed")?;
        let state = as_reloaded(&state).context("the kept state won't serialize")?;
        let hash = plaintext_hash(&state);
        anyhow::ensure!(
            Some(hash) == recorded,
            "the kept line's latest version hashes to {hash}, not its recorded {recorded:?}: \
             nothing was changed"
        );
        Ok((line, recorded, Some(state)))
    };
    let (dir, intent, state) = match super::lost::pending(&seal)? {
        Some((dir, intent)) => {
            let fork = intent.local_fork.clone().with_context(|| {
                format!(
                    "a resolution of versions Agora lost is unfinished under {}: \
                     `sealed-accept --rebase` finishes it",
                    dir.display()
                )
            })?;
            let asked = match keep {
                Keep::Head => None,
                Keep::Ord(o) => Some(o),
            };
            anyhow::ensure!(
                asked == fork.keep_ord,
                "a fork resolution is unfinished under {}, keeping {}: run it again with that \
                 choice",
                dir.display(),
                fork.keep_ord
                    .map_or("the head".to_string(), |o| format!("ord {o}"))
            );
            match prove(&fork.keep, fork.keep_ord.is_none()).await {
                Ok((_, _, state)) => (dir, intent, state),
                Err(e) => {
                    if super::lost::abandon_if_untouched(&seal, &dir, &intent).await? {
                        anyhow::bail!(
                            "the unfinished fork resolution under {} can't go on ({e:#}), and \
                             had moved nothing: it is abandoned (its INTENT kept as \
                             ABANDONED.<utc>). Nothing was changed",
                            dir.display()
                        );
                    }
                    return Err(e.context(format!(
                        "the fork resolution under {} has begun and can't finish: put back \
                         what the message names from a backup, then run this again",
                        dir.display()
                    )));
                }
            }
        }
        None => {
            let head = seal
                .head()
                .await?
                .with_context(|| format!("agent {id}'s sealed/head.json is missing"))?;
            anyhow::ensure!(
                !head.displaced.is_empty(),
                "agent {id}'s chain here has no fork to resolve"
            );
            let items = all_items().await?;
            // A displaced head another entry was saved on is an ancestor,
            // not a line's tip
            let parents: std::collections::HashSet<_> = items
                .iter()
                .filter_map(|i| i.entry.entry.prev_hash)
                .collect();
            let tips: Vec<&OutboxItem> = head
                .displaced
                .iter()
                .filter(|d| !parents.contains(&d.entry_hash))
                .filter_map(|d| items.iter().find(|i| i.entry.entry_hash() == d.entry_hash))
                .collect();
            let choices = || {
                let mut out = format!("head (seq {}, {})", head.head.seq, head.head.entry_hash);
                for i in &tips {
                    out.push_str(&format!(
                        ", {} (seq {}, {})",
                        i.ord,
                        i.entry.entry.seq,
                        i.entry.entry_hash()
                    ));
                }
                out
            };
            let (tip, tip_item) = match keep {
                Keep::Head => (head.head, None),
                Keep::Ord(o) => {
                    let item = items.iter().find(|i| i.ord == o).with_context(|| {
                        format!("no outbox item {o} here. The choices: {}", choices())
                    })?;
                    let hash = item.entry.entry_hash();
                    anyhow::ensure!(
                        !parents.contains(&hash),
                        "outbox item {o} is an ancestor of a later save, not a line's tip: \
                         keeping it would roll that line back. The choices: {}",
                        choices()
                    );
                    anyhow::ensure!(
                        tips.iter().any(|t| t.entry.entry_hash() == hash),
                        "outbox item {o} is not the tip of a line a later save forked from. \
                         The choices: {}",
                        choices()
                    );
                    (
                        HeadRef {
                            entry_hash: hash,
                            seq: item.entry.entry.seq,
                        },
                        Some(item.clone()),
                    )
                }
            };
            let (line, recorded, state) = prove(&tip, tip_item.is_none()).await?;
            let on_line: std::collections::HashSet<_> =
                line.iter().map(|e| e.entry_hash()).collect();
            let aside: Vec<OutboxItem> = seal
                .outbox()
                .await?
                .into_iter()
                .filter(|i| !on_line.contains(&i.entry.entry_hash()))
                .collect();
            let back: Vec<OutboxItem> = super::lost::displaced_items(&seal)?
                .into_iter()
                .filter(|i| on_line.contains(&i.entry.entry_hash()))
                .collect();
            let (dir, intent) = super::lost::begin_fork(
                &seal,
                super::lost::LocalFork {
                    keep: tip,
                    keep_ord: tip_item.as_ref().map(|i| i.ord),
                    tip: tip_item,
                    plaintext_sha256: recorded,
                    back,
                },
                &aside,
                note,
            )
            .await?;
            (dir, intent, state)
        }
    };
    let fork = intent
        .local_fork
        .clone()
        .context("not a fork's resolution")?;
    super::lost::finish_fork(&seal, &dir, &intent).await?;
    // The state: the kept tip's, when it isn't the head the agent ran on
    let state_changed = state.is_some();
    if let Some(state) = state {
        let hash = plaintext_hash(&state);
        let here = FsStorage::new(state_dir)
            .load_raw(id)
            .await
            .ok()
            .and_then(|l| as_reloaded(&l.value).ok())
            .map(|v| plaintext_hash(&v));
        if here != Some(hash) {
            // Archives the state here as state.<utc>.json first
            FsStorage::new(state_dir)
                .save_sealed_raw(id, state)
                .await
                .context("writing the kept state")?;
        }
    }
    super::crash::point("fork:state")?;
    if seal.is_refused().await? {
        seal.clear_refusal().await.context("clearing the refusal")?;
    }
    let chain = match reach {
        super::version::Reach::Local => super::notice::ChainKept::Local,
        super::version::Reach::Stopped => super::notice::ChainKept::Stopped,
        super::version::Reach::Uploads => super::notice::ChainKept::Uploading,
    };
    Notice::fork_kept(
        &agent_dir,
        note,
        intent.items.len(),
        state_changed,
        &super::lost::kept_text(&intent),
        chain,
    )
    .context("writing the notice")?;
    super::lost::done(&dir)?;
    super::roster::replace_for(state_dir, id, Some(fork.keep)).await?;
    drop(lock);
    tracing::warn!(
        event_type = "sealed_state_fork_kept",
        agent_id = %id,
        head = %fork.keep.entry_hash,
        seq = fork.keep.seq,
        set_aside = intent.items.len(),
        put_back = fork.back.len(),
        state_changed,
        kept = %dir.display(),
        "the operator resolved a fork in a chain kept here; the other line is kept"
    );
    Ok(Kept {
        head: fork.keep,
        set_aside: intent.items.len(),
        state_changed,
        kept: dir,
    })
}

/// Agent ids, one a line, as `--agents-file` (or stdin) gives them: blank
/// lines and `#` comments skipped, each once, in order. A line that isn't
/// an id is an error naming it.
pub fn read_ids(text: &str) -> anyhow::Result<Vec<AgentId>> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        let id: AgentId = line
            .parse()
            .map_err(|_| anyhow::anyhow!("line {}: `{line}` is not an agent id", n + 1))?;
        if !out.contains(&id) {
            out.push(id);
        }
    }
    Ok(out)
}

/// What a wide accept may do ([`accept_many`]).
#[derive(Debug, Clone, Copy)]
pub struct ManyOptions<'a> {
    /// What every agent accepted reads.
    pub note: &'a str,
    /// How many agents the list names: a real run with another count
    /// touches nothing.
    pub expect: Option<usize>,
    /// Say what would be done, per agent; touch nothing.
    pub dry_run: bool,
    /// Accept a state whose memory shrank by more than half (or whose
    /// change can't be measured) too.
    pub accept_shrunk: bool,
}

/// What a wide accept sees of one agent before it acts: counts and sizes,
/// never content.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Preview {
    /// Its refusal, if it is refused (or its head record lost).
    pub reason: Option<Refusal>,
    /// Memory and soul lengths (characters): the last sealed version's,
    /// then the state here's. `None`: couldn't be measured.
    pub memory: Option<(usize, usize)>,
    pub soul: Option<(usize, usize)>,
}

impl Preview {
    /// Memory shrank by more than half, or its change couldn't be
    /// measured: a wipe looks like this.
    pub fn shrunk(&self) -> bool {
        match self.memory {
            Some((before, after)) => after * 2 < before,
            None => true,
        }
    }
}

/// What a wide accept did with one agent.
#[derive(Debug)]
pub enum ManyOutcome {
    /// A dry run: would be accepted.
    Would,
    /// A dry run: would be held back, and why.
    WouldHold(&'static str),
    Accepted(Accepted),
    /// Not refused, its head here: accepted before (a rerun).
    Already,
    /// Held back, and why: a loss-shaped refusal, or a shrunk memory
    /// without `--accept-shrunk`.
    Held(&'static str),
    /// No such agent here: nothing made.
    NoAgent,
    Failed(anyhow::Error),
}

impl ManyOutcome {
    fn label(&self) -> &'static str {
        match self {
            Self::Would => "would accept",
            Self::WouldHold(_) => "would hold back",
            Self::Accepted(_) => "accepted",
            Self::Already => "already accepted",
            Self::Held(_) => "held back",
            Self::NoAgent => "no such agent",
            Self::Failed(_) => "failed",
        }
    }

    /// Counted as a failure of the command.
    pub fn failed(&self) -> bool {
        matches!(self, Self::Failed(_))
    }
}

/// One agent, measured: its refusal, and the memory and soul sizes of its
/// last sealed version against the state here.
async fn preview(state_dir: &Path, keys: &KeyStore, api: &dyn SealedApi, id: AgentId) -> Preview {
    let agent_dir = state_dir.join(id.to_string());
    let seal = AgentSeal::new(&agent_dir, id);
    let reason = match seal.is_refused().await {
        Ok(true) => Some(Refusal::from_marker(
            &std::fs::read_to_string(seal.dir().join("refused")).unwrap_or_default(),
        )),
        _ => match seal.head().await {
            Ok(None) => Some(Refusal::HeadMissing),
            _ => None,
        },
    };
    let here = FsStorage::new(state_dir)
        .load_raw(id)
        .await
        .ok()
        .map(|l| super::sizes::Sizes::of(&l.value));
    let sealed = super::version::last_sealed(&seal, keys, api, id)
        .await
        .ok()
        .map(|l| super::sizes::Sizes::of(&l.state));
    let (memory, soul) = match (sealed, here) {
        (Some(b), Some(a)) => (
            Some((b.memory_chars, a.memory_chars)),
            Some((b.soul_chars, a.soul_chars)),
        ),
        _ => (None, None),
    };
    Preview {
        reason,
        memory,
        soul,
    }
}

/// `sealed-accept --agents-file`: each of `ids` measured ([`Preview`]),
/// then, unless a dry run, [`accept`]ed with one note, each under its own
/// lock. A real run needs `expect` to be the count, or nothing is touched,
/// and holds back, per agent, a loss-shaped refusal (`sealed-restore`'s
/// case) and a memory that shrank by more than half (or can't be measured)
/// unless `accept_shrunk`. An agent accepted before counts as already; an
/// id naming no agent here has nothing made for it.
pub async fn accept_many(
    state_dir: &Path,
    keys: &KeyStore,
    api: &dyn SealedApi,
    ids: &[AgentId],
    options: ManyOptions<'_>,
) -> anyhow::Result<Vec<(AgentId, Preview, ManyOutcome)>> {
    let note = options.note.trim();
    anyhow::ensure!(
        !note.is_empty(),
        "--note is required: every agent accepted reads it"
    );
    if !options.dry_run {
        anyhow::ensure!(
            options.expect == Some(ids.len()),
            "{} agent(s) named, and --expect says {}: nothing was touched",
            ids.len(),
            options
                .expect
                .map_or("nothing".to_string(), |n| n.to_string())
        );
    }
    let mut out = Vec::with_capacity(ids.len());
    for &id in ids {
        if !std::fs::exists(state_dir.join(id.to_string()))? {
            out.push((id, Preview::default(), ManyOutcome::NoAgent));
            continue;
        }
        let p = preview(state_dir, keys, api, id).await;
        let hold = match p.reason {
            None => None,
            // Never wholesale: the state or its head's record gone,
            // unreadable or put back (the start check's own classification)
            Some(r) if r.loss_shaped() => Some(
                "loss-shaped refusal: resolve it on its own (sealed-restore, or sealed-accept \
                 --agent), never in a wide accept",
            ),
            Some(_) if p.shrunk() && !options.accept_shrunk => Some(
                "its memory shrank by more than half since its last sealed version (or can't \
                 be measured): --accept-shrunk accepts it",
            ),
            Some(_) => None,
        };
        let outcome = match (p.reason, hold, options.dry_run) {
            (None, _, _) => ManyOutcome::Already,
            (_, Some(why), true) => ManyOutcome::WouldHold(why),
            (_, None, true) => ManyOutcome::Would,
            (_, Some(why), false) => ManyOutcome::Held(why),
            (_, None, false) => match accept(state_dir, keys, api, id, Some(note)).await {
                Ok(a) => ManyOutcome::Accepted(a),
                Err(e) => {
                    tracing::warn!(
                        event_type = "sealed_accept_many_failed",
                        agent_id = %id,
                        error = %format!("{e:#}"),
                        "one agent of a wide sealed-accept failed; the rest go on"
                    );
                    ManyOutcome::Failed(e)
                }
            },
        };
        out.push((id, p, outcome));
    }
    Ok(out)
}

/// The table a wide accept prints: per agent its refusal, memory and soul
/// sizes (last sealed → here) and what was done; then counts by outcome
/// and by reason. Ids, reasons, sizes and counts only.
pub fn many_table(done: &[(AgentId, Preview, ManyOutcome)]) -> String {
    use std::fmt::Write as _;
    let sizes = |s: Option<(usize, usize)>| match s {
        Some((b, a)) => format!("{b}->{a}"),
        None => "?".into(),
    };
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{:<36} {:<16} {:>17} {:>15}  outcome",
        "agent", "reason", "memory", "soul"
    );
    let mut by_outcome: std::collections::BTreeMap<&str, usize> = Default::default();
    let mut by_reason: std::collections::BTreeMap<&str, usize> = Default::default();
    for (id, p, o) in done {
        let reason = p.reason.map_or("-", Refusal::as_str);
        *by_outcome.entry(o.label()).or_default() += 1;
        *by_reason.entry(reason).or_default() += 1;
        let why = match o {
            ManyOutcome::WouldHold(why) | ManyOutcome::Held(why) => format!(": {why}"),
            ManyOutcome::Failed(e) => format!(": {e:#}"),
            ManyOutcome::Accepted(a) => format!(": head seq {}", a.head.seq),
            _ => String::new(),
        };
        let _ = writeln!(
            out,
            "{id:<36} {reason:<16} {:>17} {:>15}  {}{why}",
            sizes(p.memory),
            sizes(p.soul),
            o.label()
        );
    }
    let _ = writeln!(out, "by outcome:");
    for (k, n) in &by_outcome {
        let _ = writeln!(out, "  {k:<20} {n:>6}");
    }
    let _ = writeln!(out, "by reason:");
    for (k, n) in &by_reason {
        let _ = writeln!(out, "  {k:<20} {n:>6}");
    }
    let _ = writeln!(out, "  {:<20} {:>6}", "total", done.len());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alerts::Alerter;
    use crate::sealed::api::fake::FakeApi;
    use crate::sealed::check::{self, Outcome, tests::enrolled};
    use agora_agentkit::state::storage::plaintext_hash;

    async fn local(root: &Path, id: AgentId) -> agora_agentkit::govlog::Sha256Hex {
        plaintext_hash(
            &FsStorage::new(root.join("state"))
                .load_raw(id)
                .await
                .unwrap()
                .value,
        )
    }

    #[tokio::test]
    async fn accepting_a_refusal_runs_the_agent_and_shows_the_note_once() {
        let root = crate::sealed::TestDir::new("accept");
        let (id, keys, dir) = enrolled(&root).await;
        let api = FakeApi::new();
        let state_dir = root.join("state");
        // Not refused: nothing to accept
        assert!(
            accept(&state_dir, &keys, &api, id, Some("n"))
                .await
                .is_err()
        );

        // A wipe, refused
        FsStorage::new(&state_dir)
            .save_raw(id, serde_json::json!({ "memory": "" }), None)
            .await
            .unwrap();
        let Outcome::Refuse { reason, detail } = check::check(
            &dir,
            id,
            &local(&root, id).await,
            Default::default(),
            &keys,
            &api,
        )
        .await
        else {
            panic!()
        };
        check::refuse(&dir, id, "tarn", reason, &detail, &Alerter::off()).await;
        assert!(
            accept(&state_dir, &keys, &api, id, Some("  "))
                .await
                .is_err(),
            "a note is required"
        );

        let done = accept(
            &state_dir,
            &keys,
            &api,
            id,
            Some("A test wipe; I restored nothing."),
        )
        .await
        .unwrap();
        assert_eq!(done.reason, Refusal::Mismatch);
        let out = check::check(
            &dir,
            id,
            &local(&root, id).await,
            Default::default(),
            &keys,
            &api,
        )
        .await;
        let Outcome::Run {
            seat, notice, hold, ..
        } = out
        else {
            panic!("{out:?}")
        };
        assert!(notice && hold.is_none());
        assert!(
            seat[0].contains(
                "did not run: your state did not match your last sealed version. Your operator \
                 accepted the current state with this note: A test wipe; I restored nothing."
            ),
            "{seat:?}"
        );
        assert!(
            seat[1].starts_with("Your sealed state: intact. It has 2 versions since "),
            "{seat:?}"
        );
        // Shown in a session that saved: not again
        Notice::shown(&dir).unwrap();
        let out = check::check(
            &dir,
            id,
            &local(&root, id).await,
            Default::default(),
            &keys,
            &api,
        )
        .await;
        assert!(matches!(out, Outcome::Run { notice: false, .. }), "{out:?}");
    }

    /// A shrink an operator accepted is never told as the agent's own
    /// rewrite, beside the notice that says the operator accepted it (beta
    /// round 6)
    #[tokio::test]
    async fn an_accepted_shrink_is_not_called_the_agents_rewrite() {
        use agora_agentkit::state::storage::Mirrored;
        let root = crate::sealed::TestDir::new("accept-shrink");
        let (id, keys, dir) = enrolled(&root).await;
        let api = FakeApi::new();
        let state_dir = root.join("state");
        let full = serde_json::json!({ "memory": { "content": "x".repeat(1000) } });
        let cut = serde_json::json!({ "memory": { "content": "x".repeat(400) } });
        let enrolment = std::sync::Arc::new(crate::sealed::SealedEnrolment::new(
            state_dir.clone(),
            root.join("secrets"),
        ));
        let mut store = Mirrored::new(FsStorage::new(&state_dir), (), enrolment);
        let base = store.load_raw(id).await.unwrap().base;
        store
            .save_raw(id, full.clone(), base.as_ref())
            .await
            .unwrap();
        let sizes = crate::sealed::sizes::Sizes::of;
        // Checked, so its sizes are recorded
        let out = check::check(&dir, id, &local(&root, id).await, sizes(&full), &keys, &api).await;
        assert!(matches!(out, Outcome::Run { hold: None, .. }), "{out:?}");
        // Cut outside a session: refused, then accepted
        FsStorage::new(&state_dir)
            .save_raw(id, cut.clone(), None)
            .await
            .unwrap();
        let Outcome::Refuse { reason, detail } =
            check::check(&dir, id, &local(&root, id).await, sizes(&cut), &keys, &api).await
        else {
            panic!()
        };
        check::refuse(&dir, id, "tarn", reason, &detail, &Alerter::off()).await;
        accept(&state_dir, &keys, &api, id, Some("Trimmed by hand."))
            .await
            .unwrap();
        let out = check::check(&dir, id, &local(&root, id).await, sizes(&cut), &keys, &api).await;
        let Outcome::Run { seat, notice, .. } = out else {
            panic!("{out:?}")
        };
        assert!(notice, "{seat:?}");
        let verdict = seat
            .iter()
            .find(|l| l.starts_with("Your sealed state: "))
            .expect("a verdict");
        assert!(
            verdict.contains(
                "your memory is 400 characters (\u{2212}600 since your earlier sealed version, \
                 head "
            ) && verdict.contains(") and your soul"),
            "{verdict}"
        );
        assert!(!verdict.contains("rewrote"), "{verdict}");
    }

    /// A deliberate edit, accepted: the agent reads the note, then what
    /// changed, counted; with the last version unopenable, it is told no
    /// diff could be made.
    #[tokio::test]
    async fn an_accepted_edit_shows_the_note_then_the_diff() {
        let root = crate::sealed::TestDir::new("accept-diff");
        let (id, keys, dir) = enrolled(&root).await;
        let api = FakeApi::new();
        let state_dir = root.join("state");
        let edited = serde_json::json!({
            "memory": { "content": "first line\nan edited line\nthird line\nfourth\n" },
            "soul": { "name": "tarn" },
        });
        FsStorage::new(&state_dir)
            .save_raw(id, edited, None)
            .await
            .unwrap();
        let Outcome::Refuse { reason, detail } = check::check(
            &dir,
            id,
            &local(&root, id).await,
            Default::default(),
            &keys,
            &api,
        )
        .await
        else {
            panic!()
        };
        check::refuse(&dir, id, "tarn", reason, &detail, &Alerter::off()).await;
        accept(&state_dir, &keys, &api, id, Some("I edited it by hand."))
            .await
            .unwrap();
        let notice = Notice::load(&dir).unwrap().unwrap();
        let diff = notice.diff.clone().unwrap();
        assert!(diff.text.is_some() && !diff.cut);
        let line = notice.line().unwrap();
        let (note, shown) = line.split_once("\n\n").unwrap();
        assert!(
            note.ends_with("with this note: I edited it by hand."),
            "{line}"
        );
        assert!(
            shown.starts_with(&format!(
                "What changed from your last sealed version to the state your operator \
                 accepted (+{} / \u{2212}{} lines):\nMemory:\n",
                diff.added, diff.removed
            )),
            "{shown}"
        );
        assert!(diff.added >= 4, "{diff:?}");
        assert!(
            shown.contains("  + an edited line") && shown.contains("Soul:\n"),
            "{shown}"
        );

        // Again, with the sealed blobs gone and Agora unreachable
        let root = crate::sealed::TestDir::new("accept-nodiff");
        let (id, keys, dir) = enrolled(&root).await;
        let state_dir = root.join("state");
        FsStorage::new(&state_dir)
            .save_raw(id, serde_json::json!({ "memory": "changed" }), None)
            .await
            .unwrap();
        let Outcome::Refuse { reason, detail } = check::check(
            &dir,
            id,
            &local(&root, id).await,
            Default::default(),
            &keys,
            &api,
        )
        .await
        else {
            panic!()
        };
        check::refuse(&dir, id, "tarn", reason, &detail, &Alerter::off()).await;
        std::fs::rename(dir.join("sealed/blobs"), root.join("blobs.moved")).unwrap();
        std::fs::create_dir_all(dir.join("sealed/blobs")).unwrap();
        *api.fail.lock().unwrap() = Some("blob");
        accept(&state_dir, &keys, &api, id, Some("offline"))
            .await
            .unwrap();
        let line = Notice::load(&dir).unwrap().unwrap().line().unwrap();
        assert!(
            line.ends_with(&format!(
                "with this note: offline\n\n{}",
                super::super::notice::NO_DIFF
            )),
            "{line}"
        );
    }

    /// Every blob and entry the agent holds, on a fake Agora.
    fn uploaded(api: &FakeApi, dir: &Path, chain: Vec<agora_agentkit::state::SignedStateEntry>) {
        *api.chain.lock().unwrap() = Ok(chain);
        for f in std::fs::read_dir(dir.join("sealed/blobs")).unwrap() {
            let f = f.unwrap();
            let sha = f.file_name().to_str().unwrap().parse().unwrap();
            api.blobs
                .lock()
                .unwrap()
                .insert(sha, std::fs::read(f.path()).unwrap());
        }
    }

    async fn lose_head_and_refuse(root: &Path, id: AgentId, keys: &KeyStore, dir: &Path) {
        let sealed = dir.join("sealed");
        std::fs::rename(sealed.join("head.json"), sealed.join("head.json.lost")).unwrap();
        let api = FakeApi::new();
        let Outcome::Refuse { reason, detail } = check::check(
            dir,
            id,
            &local(root, id).await,
            Default::default(),
            keys,
            &api,
        )
        .await
        else {
            panic!()
        };
        assert_eq!(reason, Refusal::HeadMissing);
        check::refuse(dir, id, "tarn", reason, &detail, &Alerter::off()).await;
    }

    /// A lost head.json, the state still the head's: rebuilt from Agora's
    /// verified chain and the refusal cleared, with no new version.
    #[tokio::test]
    async fn a_lost_head_is_adopted_from_agora_when_the_state_matches() {
        let root = crate::sealed::TestDir::new("accept-adopt");
        let (id, keys, dir) = enrolled(&root).await;
        let chain = check::local_entries(&AgentSeal::new(&dir, id))
            .await
            .unwrap();
        lose_head_and_refuse(&root, id, &keys, &dir).await;
        let api = FakeApi::new();
        // Nothing on Agora: nothing to adopt, still refused
        assert!(
            accept(&root.join("state"), &keys, &api, id, None)
                .await
                .is_err()
        );
        uploaded(&api, &dir, chain.clone());
        let done = accept(&root.join("state"), &keys, &api, id, None)
            .await
            .unwrap();
        assert!(done.adopted);
        assert_eq!(done.head.entry_hash, chain.last().unwrap().entry_hash());
        let out = check::check(
            &dir,
            id,
            &local(&root, id).await,
            Default::default(),
            &keys,
            &api,
        )
        .await;
        let Outcome::Run { seat, hold, .. } = out else {
            panic!("{out:?}")
        };
        assert_eq!(hold, None);
        assert!(seat[0].contains(REBUILT_NOTE), "{seat:?}");
    }

    /// A lost head.json and a state that isn't its head: refused until the
    /// operator writes a note, then sealed on top of the adopted head.
    #[tokio::test]
    async fn a_lost_head_with_a_changed_state_stays_refused_without_a_note() {
        let root = crate::sealed::TestDir::new("accept-adopt-mismatch");
        let (id, keys, dir) = enrolled(&root).await;
        let chain = check::local_entries(&AgentSeal::new(&dir, id))
            .await
            .unwrap();
        FsStorage::new(root.join("state"))
            .save_raw(id, serde_json::json!({ "memory": "" }), None)
            .await
            .unwrap();
        lose_head_and_refuse(&root, id, &keys, &dir).await;
        let api = FakeApi::new();
        uploaded(&api, &dir, chain.clone());
        let e = accept(&root.join("state"), &keys, &api, id, None)
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("stays refused"), "{e}");
        assert!(AgentSeal::new(&dir, id).is_refused().await.unwrap());
        let done = accept(
            &root.join("state"),
            &keys,
            &api,
            id,
            Some("wiped in a test"),
        )
        .await
        .unwrap();
        assert_eq!(done.head.seq, chain.len() as u64, "on top of Agora's head");
        let out = check::check(
            &dir,
            id,
            &local(&root, id).await,
            Default::default(),
            &keys,
            &api,
        )
        .await;
        assert!(matches!(out, Outcome::Run { hold: None, .. }), "{out:?}");
    }

    /// Never while the agent has a session journal newer than its refusal.
    #[tokio::test]
    async fn accept_refuses_while_a_session_is_in_progress() {
        let root = crate::sealed::TestDir::new("accept-session");
        let (id, keys, dir) = enrolled(&root).await;
        AgentSeal::new(&dir, id).refuse("mismatch").await.unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let journal = FsStorage::new(root.join("state")).journal(id);
        std::fs::write(journal.path(), b"{}\n").unwrap();
        let e = accept(&root.join("state"), &keys, &FakeApi::new(), id, Some("n"))
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("session in progress"), "{e}");
    }

    /// A held fork: Agora moved on from the local head. `--rebase` seals
    /// the current state on Agora's head, keeps the other branch locally,
    /// and the next start is clean and tells the agent.
    #[tokio::test]
    async fn a_held_fork_is_rebased_onto_agoras_head() {
        let root = crate::sealed::TestDir::new("accept-rebase");
        let (id, keys, dir) = enrolled(&root).await;
        let seal = AgentSeal::new(&dir, id);
        let mine = check::local_entries(&seal).await.unwrap();
        // Agora took genesis and a sibling of our state
        let mut sibling = mine[1].entry.clone();
        sibling.state_at += 1;
        let server_chain = vec![mine[0].clone(), sibling.sign(&keys.signing(id).unwrap())];
        let api = FakeApi::new();
        *api.chain.lock().unwrap() = Ok(server_chain.clone());
        let out = check::check(
            &dir,
            id,
            &local(&root, id).await,
            Default::default(),
            &keys,
            &api,
        )
        .await;
        assert!(
            matches!(
                out,
                Outcome::Run {
                    hold: Some(check::Hold::Fork),
                    ..
                }
            ),
            "{out:?}"
        );
        let held = seal.held().await.unwrap();
        assert!(held.outbox_items > 0);

        assert!(
            rebase(&root.join("state"), &keys, &api, id, " ")
                .await
                .is_err(),
            "a note"
        );
        let done = rebase(&root.join("state"), &keys, &api, id, "joined in a test")
            .await
            .unwrap();
        assert_eq!(done.onto.unwrap().entry_hash, server_chain[1].entry_hash());
        assert_eq!(done.head.seq, 2);
        assert_eq!(done.set_aside, held.outbox_items);
        // Agora at its head, the rebased version waiting on it: clean
        let out = check::check(
            &dir,
            id,
            &local(&root, id).await,
            Default::default(),
            &keys,
            &api,
        )
        .await;
        let Outcome::Run {
            seat, hold, notice, ..
        } = out
        else {
            panic!("{out:?}")
        };
        assert_eq!(hold, None);
        assert!(notice);
        assert!(
            seat[0].contains("your operator joined your two sealed histories"),
            "{seat:?}"
        );
        assert!(
            seat[0].ends_with("Your operator's note: joined in a test"),
            "{seat:?}"
        );
        // Nothing left to rebase
        assert!(
            rebase(&root.join("state"), &keys, &api, id, "again")
                .await
                .is_err()
        );
    }

    /// Wiped and refused, an agent that never uploaded.
    async fn local_refused(root: &Path) -> (AgentId, KeyStore, std::path::PathBuf) {
        let (id, keys, dir) = check::tests::enrolled_local(root).await;
        FsStorage::new(root.join("state"))
            .save_raw(id, serde_json::json!({ "memory": "" }), None)
            .await
            .unwrap();
        AgentSeal::new(&dir, id).refuse("mismatch").await.unwrap();
        (id, keys, dir)
    }

    /// An agent that never uploaded: accepted with a note, its lost head
    /// rebuilt from the chain kept here, and `--rebase` refused, all with
    /// no call to Agora (the strict guard).
    #[tokio::test]
    async fn a_local_only_agent_is_accepted_without_agora() {
        let root = crate::sealed::TestDir::new("accept-local");
        let (id, keys, dir) = local_refused(&root).await;
        let api = crate::sealed::api::fake::guarded(&root);
        let state_dir = root.join("state");
        let done = accept(&state_dir, &keys, &api, id, Some("a test wipe"))
            .await
            .unwrap();
        assert!(!done.adopted);
        assert!(!AgentSeal::new(&dir, id).is_refused().await.unwrap());
        let e = rebase(&state_dir, &keys, &api, id, "join")
            .await
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("nothing of it is on Agora") && e.contains("--keep"),
            "{e}"
        );

        // head.json lost, the state still the head's: rebuilt here
        let root = crate::sealed::TestDir::new("accept-local-head");
        let (id, keys, dir) = check::tests::enrolled_local(&root).await;
        let head = AgentSeal::new(&dir, id).head().await.unwrap().unwrap();
        std::fs::rename(dir.join("sealed/head.json"), root.join("head.lost")).unwrap();
        AgentSeal::new(&dir, id)
            .refuse("head_missing")
            .await
            .unwrap();
        let api = crate::sealed::api::fake::guarded(&root);
        let done = accept(&root.join("state"), &keys, &api, id, None)
            .await
            .unwrap();
        assert!(done.adopted);
        assert_eq!(done.head, head.head);
        let line = Notice::load(&dir).unwrap().unwrap().line().unwrap();
        assert!(line.contains(REBUILT_LOCAL_NOTE), "{line}");
        assert!(api.calls.lock().unwrap().is_empty());
    }

    /// A chain kept here forked: two saves from one version.
    async fn forked(root: &Path) -> (AgentId, KeyStore, std::path::PathBuf, HeadRef, HeadRef) {
        use agora_agentkit::state::storage::Mirrored;
        let (id, keys, dir) = check::tests::enrolled_local(root).await;
        let state_dir = root.join("state");
        let enrolment = std::sync::Arc::new(crate::sealed::SealedEnrolment::new(
            state_dir.clone(),
            root.join("secrets"),
        ));
        let mut store = Mirrored::new(FsStorage::new(&state_dir), (), enrolment);
        let base = store.load_raw(id).await.unwrap().base;
        let a = store
            .save_raw(id, check::tests::state(20), base.as_ref())
            .await
            .unwrap()
            .unwrap();
        let b = store
            .save_raw(id, check::tests::state(21), base.as_ref())
            .await
            .unwrap()
            .unwrap();
        let head = AgentSeal::new(&dir, id).head().await.unwrap().unwrap();
        assert_eq!(head.head, b);
        assert_eq!(head.displaced, vec![a], "a fork");
        (id, keys, dir, a, b)
    }

    fn ord_of(dir: &Path, h: HeadRef) -> u64 {
        for e in std::fs::read_dir(dir.join("sealed/outbox")).unwrap() {
            let p = e.unwrap().path();
            if p.extension().and_then(|x| x.to_str()) != Some("json") {
                continue;
            }
            let item: agora_agentkit::state::storage::OutboxItem =
                serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
            if item.entry.entry_hash() == h.entry_hash {
                return item.ord;
            }
        }
        panic!("no item")
    }

    /// `--keep head`: the head's line goes on, the other version is kept
    /// aside (a `.displaced` item and its copies under sealed/lost/), the
    /// fork cleared, the agent told; `--keep <ord>` of the other tip: that
    /// line goes on and its state becomes the agent's. No call to Agora.
    #[tokio::test]
    async fn a_local_fork_is_resolved_by_the_line_kept() {
        for keep_head in [true, false] {
            let root = crate::sealed::TestDir::new("accept-keep");
            let (id, keys, dir, a, b) = forked(&root).await;
            let api = crate::sealed::api::fake::guarded(&root);
            let state_dir = root.join("state");
            let keep = if keep_head {
                Keep::Head
            } else {
                Keep::Ord(ord_of(&dir, a))
            };
            let done = keep_line(&state_dir, &keys, &api, id, keep, "two saves at once")
                .await
                .unwrap();
            let (kept, other, state) = if keep_head {
                (b, a, check::tests::state(21))
            } else {
                (a, b, check::tests::state(20))
            };
            assert_eq!(done.head, kept);
            assert_eq!(done.set_aside, 1);
            assert_eq!(done.state_changed, !keep_head);
            let seal = AgentSeal::new(&dir, id);
            let head = seal.head().await.unwrap().unwrap();
            assert_eq!(head.head, kept);
            assert!(head.displaced.is_empty());
            assert_eq!(
                local(&root, id).await,
                agora_agentkit::state::storage::plaintext_hash(&state)
            );
            assert_eq!(head.plaintext_sha256, Some(local(&root, id).await));
            // The other version: aside, never deleted
            assert!(
                seal.outbox()
                    .await
                    .unwrap()
                    .iter()
                    .all(|i| i.entry.entry_hash() != other.entry_hash)
            );
            let aside = std::fs::read_dir(dir.join("sealed/outbox"))
                .unwrap()
                .filter(|e| {
                    e.as_ref()
                        .unwrap()
                        .file_name()
                        .to_string_lossy()
                        .ends_with(&format!("{}.displaced", other.entry_hash))
                })
                .count();
            assert_eq!(aside, 1);
            let res = crate::sealed::lost::resolutions(&seal).unwrap();
            assert_eq!(res.len(), 1);
            assert!(std::fs::exists(res[0].0.join("DONE")).unwrap());
            assert!(std::fs::exists(res[0].0.join("items")).unwrap());
            // The main line verifies; the agent is told
            let entries = check::local_entries(&seal).await.unwrap();
            let line = check::main_line(&entries, &kept.entry_hash).unwrap();
            assert!(verify_chain(id, &line, &keys.trusted(id).unwrap().unwrap()).intact);
            let told = Notice::load(&dir).unwrap().unwrap().line().unwrap();
            assert!(
                told.contains("resolved a fork") && told.contains("two saves at once"),
                "{told}"
            );
            assert!(!told.contains("on Agora:"), "{told}");
            assert!(api.calls.lock().unwrap().is_empty());
            // Nothing left to resolve
            assert!(
                keep_line(&state_dir, &keys, &api, id, Keep::Head, "again")
                    .await
                    .is_err()
            );
        }
    }

    /// A tip that isn't a displaced line's: refused, naming the choices,
    /// nothing written.
    #[tokio::test]
    async fn keeping_a_wrong_line_names_the_choices() {
        let root = crate::sealed::TestDir::new("accept-keep-wrong");
        let (id, keys, dir, a, _) = forked(&root).await;
        let api = crate::sealed::api::fake::guarded(&root);
        let e = keep_line(&root.join("state"), &keys, &api, id, Keep::Ord(999), "n")
            .await
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("The choices: head") && e.contains(&a.entry_hash.to_string()),
            "{e}"
        );
        assert!(!std::fs::exists(dir.join("sealed/lost")).unwrap());
    }

    /// Cut after each step: the rerun finishes the same resolution.
    #[tokio::test]
    async fn a_cut_fork_resolution_finishes_on_rerun() {
        for point in [
            "fork:intent",
            "fork:copied",
            "fork:aside",
            "fork:head",
            "fork:state",
        ] {
            let root = crate::sealed::TestDir::new("accept-keep-cut");
            let (id, keys, dir, a, _) = forked(&root).await;
            let api = crate::sealed::api::fake::guarded(&root);
            let keep = Keep::Ord(ord_of(&dir, a));
            crate::sealed::crash::at(point);
            let e = keep_line(&root.join("state"), &keys, &api, id, keep, "n")
                .await
                .unwrap_err();
            assert!(
                format!("{e:#}").contains("simulated crash"),
                "{point}: {e:#}"
            );
            let done = keep_line(&root.join("state"), &keys, &api, id, keep, "n")
                .await
                .unwrap();
            assert_eq!(done.head, a, "{point}");
            assert_eq!(done.set_aside, 1, "{point}");
            assert_eq!(
                local(&root, id).await,
                agora_agentkit::state::storage::plaintext_hash(&check::tests::state(20)),
                "{point}"
            );
            let res = crate::sealed::lost::resolutions(&AgentSeal::new(&dir, id)).unwrap();
            assert_eq!(res.len(), 1, "{point}: one resolution");
        }
    }

    /// An older copy of an agent's directory put back: the roster refuses
    /// it as rolled back. `sealed-accept --note` keeps the older state on
    /// purpose, its new head below the roster's: the roster takes that
    /// head, so the next start finds no loss. A wide accept holds a
    /// rolled-back refusal back, as every loss-shaped one.
    #[tokio::test]
    async fn an_accept_that_lowers_the_head_is_not_refused_at_the_next_start() {
        let root = crate::sealed::TestDir::new("accept-rolled-back");
        let (id, keys, dir) = check::tests::enrolled_local(&root).await;
        let state_dir = root.join("state");
        let seal = AgentSeal::new(&dir, id);
        let head = seal.head().await.unwrap().unwrap().head;
        // The roster saw five more versions than the copy put back holds
        let ahead = HeadRef {
            seq: head.seq + 5,
            ..head
        };
        crate::sealed::roster::record(&root, id, ahead)
            .await
            .unwrap();
        let scan = crate::sealed::roster::check(&root, &state_dir)
            .await
            .unwrap();
        assert_eq!(
            scan.losses.get(&id).map(|l| l.refusal()),
            Some(Refusal::RolledBack)
        );
        seal.refuse(Refusal::RolledBack.as_str()).await.unwrap();
        // A wide accept holds it back, dry or not
        let api = crate::sealed::api::fake::guarded(&root);
        let many = accept_many(
            &state_dir,
            &keys,
            &api,
            &[id],
            ManyOptions {
                note: "n",
                expect: Some(1),
                dry_run: false,
                accept_shrunk: true,
            },
        )
        .await
        .unwrap();
        assert!(matches!(many[0].2, ManyOutcome::Held(_)), "{:?}", many[0]);
        assert!(seal.is_refused().await.unwrap());
        // Accepted on its own: a head below the roster's
        let done = accept(&state_dir, &keys, &api, id, Some("the older copy is kept"))
            .await
            .unwrap();
        assert!(done.head.seq < ahead.seq, "{:?}", done.head);
        let roster = crate::sealed::roster::load(&root).unwrap().unwrap();
        assert_eq!(roster.agents[&id].seq, done.head.seq);
        assert_eq!(roster.agents[&id].entry_hash, done.head.entry_hash);
        let scan = crate::sealed::roster::check(&root, &state_dir)
            .await
            .unwrap();
        assert!(scan.losses.is_empty(), "{:?}", scan.losses);
        assert!(!scan.refused.contains(&id));
        assert!(
            api.calls.lock().unwrap().is_empty(),
            "nothing asked of Agora"
        );
    }

    /// `--agents-file`: a dry run touches nothing and says what it would
    /// do; a real run with a count that isn't `--expect` touches nothing;
    /// with it, each refused agent is accepted with one note, a wiped
    /// memory held back unless `--accept-shrunk`, a rerun counts them as
    /// already, and an id naming no agent makes nothing.
    #[tokio::test]
    async fn a_wide_accept_is_previewed_counted_and_never_blesses_a_wipe() {
        let root = crate::sealed::TestDir::new("accept-many");
        // Sealed with 1,000 characters of memory, then changed and refused:
        // a to nothing (a wipe), b to 900 (an edit)
        let refused_with = async |chars: usize| {
            use agora_agentkit::state::codec::KeyframePolicy;
            let (id, keys, dir) = check::tests::enrolled_local(&root).await;
            let full = serde_json::json!({ "memory": { "content": "x".repeat(1000) } });
            AgentSeal::new(&dir, id)
                .append_state(
                    &keys.seal_keys(id).unwrap(),
                    &KeyframePolicy::default(),
                    StateEntryKind::Snapshot,
                    &full,
                    Utc::now(),
                )
                .await
                .unwrap();
            FsStorage::new(root.join("state"))
                .save_raw(
                    id,
                    serde_json::json!({ "memory": { "content": "x".repeat(chars) } }),
                    None,
                )
                .await
                .unwrap();
            AgentSeal::new(&dir, id).refuse("mismatch").await.unwrap();
            (id, keys, dir)
        };
        let (a, keys, dir_a) = refused_with(0).await;
        let (b, _, dir_b) = refused_with(900).await;
        let ghost = AgentId::new();
        let api = crate::sealed::api::fake::guarded(&root);
        let state_dir = root.join("state");
        let ids = read_ids(&format!("# after the event\n{a}\n\n{b}\n{a}\n{ghost}\n")).unwrap();
        assert_eq!(ids, vec![a, b, ghost], "each once, in order");
        assert!(read_ids("not-an-id").is_err());
        let opts = |dry_run, expect, accept_shrunk| ManyOptions {
            note: "restored by hand",
            expect,
            dry_run,
            accept_shrunk,
        };
        let untouched = async |d: &Path| {
            assert!(
                AgentSeal::new(d, AgentId::new())
                    .is_refused()
                    .await
                    .unwrap()
            );
            assert!(Notice::load(d).unwrap().is_none(), "untouched");
        };
        // Dry run: what would be done, nothing done
        let dry = accept_many(&state_dir, &keys, &api, &ids, opts(true, None, false))
            .await
            .unwrap();
        assert!(
            matches!(dry[0].2, ManyOutcome::WouldHold(_)),
            "{:?}",
            dry[0]
        );
        assert_eq!(dry[0].1.memory, Some((1000, 0)));
        assert_eq!(dry[1].1.memory, Some((1000, 900)));
        assert!(matches!(dry[1].2, ManyOutcome::Would), "{:?}", dry[1]);
        assert!(matches!(dry[2].2, ManyOutcome::NoAgent));
        let table = many_table(&dry);
        assert!(
            table.contains("would hold back") && table.contains("mismatch"),
            "{table}"
        );
        for d in [&dir_a, &dir_b] {
            untouched(d).await;
        }
        assert!(!std::fs::exists(state_dir.join(ghost.to_string())).unwrap());
        // The wrong count: nothing
        let e = accept_many(&state_dir, &keys, &api, &ids, opts(false, Some(2), false))
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("nothing was touched"), "{e}");
        for d in [&dir_a, &dir_b] {
            untouched(d).await;
        }
        // The real run: the wipe held back, the edit accepted
        let done = accept_many(&state_dir, &keys, &api, &ids, opts(false, Some(3), false))
            .await
            .unwrap();
        assert!(matches!(done[0].2, ManyOutcome::Held(_)));
        assert!(matches!(done[1].2, ManyOutcome::Accepted(_)));
        assert!(matches!(done[2].2, ManyOutcome::NoAgent));
        assert!(!std::fs::exists(state_dir.join(ghost.to_string())).unwrap());
        untouched(&dir_a).await;
        // Told so: accepted; and a rerun counts the other as already
        let done = accept_many(&state_dir, &keys, &api, &ids, opts(false, Some(3), true))
            .await
            .unwrap();
        assert!(matches!(done[0].2, ManyOutcome::Accepted(_)));
        assert!(matches!(done[1].2, ManyOutcome::Already));
        assert!(done.iter().all(|(_, _, o)| !o.failed()));
        assert!(api.calls.lock().unwrap().is_empty());
        // A single accept of an id naming no agent makes nothing either
        assert!(
            accept(&state_dir, &keys, &api, ghost, Some("n"))
                .await
                .is_err()
        );
        assert!(!std::fs::exists(state_dir.join(ghost.to_string())).unwrap());
    }

    /// A, B from one version; then A2 saved on A: A is an ancestor, not a
    /// line's tip. Never offered, and refused when named.
    #[tokio::test]
    async fn an_ancestor_is_never_kept_as_a_tip() {
        use agora_agentkit::state::storage::Mirrored;
        let root = crate::sealed::TestDir::new("accept-keep-ancestor");
        let (id, keys, dir, a, b) = forked(&root).await;
        let state_dir = root.join("state");
        // A2 on A: a save whose base is A
        let enrolment = std::sync::Arc::new(crate::sealed::SealedEnrolment::new(
            state_dir.clone(),
            root.join("secrets"),
        ));
        let mut store = Mirrored::new(FsStorage::new(&state_dir), (), enrolment);
        let a2 = store
            .save_raw(id, check::tests::state(22), Some(&a))
            .await
            .unwrap()
            .unwrap();
        let head = AgentSeal::new(&dir, id).head().await.unwrap().unwrap();
        assert_eq!(head.head, a2);
        assert!(head.displaced.contains(&a) && head.displaced.contains(&b));
        let api = crate::sealed::api::fake::guarded(&root);
        let e = keep_line(&state_dir, &keys, &api, id, Keep::Ord(ord_of(&dir, a)), "n")
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("is an ancestor"), "{e}");
        assert!(e.contains(&b.entry_hash.to_string()), "B is a choice: {e}");
        assert!(
            !e.contains(&format!(", {} (seq", ord_of(&dir, a))),
            "A is not: {e}"
        );
        assert!(
            !std::fs::exists(dir.join("sealed/lost")).unwrap(),
            "nothing moved"
        );
        // B is: kept, A2's line (and A) set aside
        let done = keep_line(&state_dir, &keys, &api, id, Keep::Ord(ord_of(&dir, b)), "n")
            .await
            .unwrap();
        assert_eq!(done.head, b);
        assert_eq!(done.set_aside, 2);
    }

    /// The kept tip's state can't be opened (its blob gone): refused
    /// before anything moves, and the next try with the blob back works.
    #[tokio::test]
    async fn a_kept_tip_that_wont_open_moves_nothing() {
        let root = crate::sealed::TestDir::new("accept-keep-blob");
        let (id, keys, dir, a, _) = forked(&root).await;
        let item = AgentSeal::new(&dir, id)
            .outbox()
            .await
            .unwrap()
            .into_iter()
            .find(|i| i.entry.entry_hash() == a.entry_hash)
            .unwrap();
        let sha = item.entry.entry.blob.unwrap().sha256.to_hex();
        let blob = dir.join("sealed/blobs").join(&sha);
        std::fs::rename(&blob, root.join("blob")).unwrap();
        let head = AgentSeal::new(&dir, id).head().await.unwrap().unwrap();
        let api = crate::sealed::api::fake::guarded(&root);
        let keep = Keep::Ord(item.ord);
        assert!(
            keep_line(&root.join("state"), &keys, &api, id, keep, "n")
                .await
                .is_err()
        );
        assert_eq!(
            AgentSeal::new(&dir, id).head().await.unwrap().unwrap(),
            head
        );
        assert!(
            !std::fs::exists(dir.join("sealed/lost")).unwrap(),
            "no INTENT"
        );
        assert!(api.calls.lock().unwrap().is_empty());
        std::fs::rename(root.join("blob"), &blob).unwrap();
        let done = keep_line(&root.join("state"), &keys, &api, id, keep, "n")
            .await
            .unwrap();
        assert_eq!(done.head, a);
    }

    /// An INTENT left by an older run whose tip won't open, nothing moved:
    /// abandoned (kept as ABANDONED.<utc>), and `--keep` works again.
    #[tokio::test]
    async fn an_untouched_intent_that_cant_finish_is_abandoned() {
        let root = crate::sealed::TestDir::new("accept-keep-abandon");
        let (id, keys, dir, a, _) = forked(&root).await;
        let api = crate::sealed::api::fake::guarded(&root);
        let keep = Keep::Ord(ord_of(&dir, a));
        crate::sealed::crash::at("fork:intent");
        assert!(
            keep_line(&root.join("state"), &keys, &api, id, keep, "n")
                .await
                .is_err()
        );
        let seal = AgentSeal::new(&dir, id);
        let item = seal
            .outbox()
            .await
            .unwrap()
            .into_iter()
            .find(|i| i.entry.entry_hash() == a.entry_hash)
            .unwrap();
        let blob = dir
            .join("sealed/blobs")
            .join(item.entry.entry.blob.unwrap().sha256.to_hex());
        std::fs::rename(&blob, root.join("blob")).unwrap();
        let e = keep_line(&root.join("state"), &keys, &api, id, keep, "n")
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("abandoned"), "{e}");
        assert!(crate::sealed::lost::pending(&seal).unwrap().is_none());
        std::fs::rename(root.join("blob"), &blob).unwrap();
        keep_line(&root.join("state"), &keys, &api, id, Keep::Head, "n")
            .await
            .unwrap();
    }

    /// head.json lost with a fork in the chain here: rebuilt with the fork
    /// recorded (the other line set aside), so `--keep` sees it; and an
    /// opt-in then uploads the main line only.
    #[tokio::test]
    async fn a_rebuilt_head_keeps_the_fork_and_an_opt_in_uploads_the_main_line() {
        use agora_agentkit::state::outbox::{AgoraOutbox, Drained};
        for keep_other in [false, true] {
            let root = crate::sealed::TestDir::new("accept-adopt-fork");
            let (id, keys, dir, a, b) = forked(&root).await;
            let ord_a = ord_of(&dir, a);
            std::fs::rename(dir.join("sealed/head.json"), root.join("head.lost")).unwrap();
            AgentSeal::new(&dir, id)
                .refuse("head_missing")
                .await
                .unwrap();
            let api = crate::sealed::api::fake::guarded(&root);
            let state_dir = root.join("state");
            let done = accept(&state_dir, &keys, &api, id, None).await.unwrap();
            assert_eq!(done.head, b);
            let seal = AgentSeal::new(&dir, id);
            assert_eq!(seal.head().await.unwrap().unwrap().displaced, vec![a]);
            assert!(
                seal.outbox()
                    .await
                    .unwrap()
                    .iter()
                    .all(|i| i.entry.entry_hash() != a.entry_hash),
                "set aside"
            );
            let (keep, kept) = if keep_other {
                (Keep::Ord(ord_a), a)
            } else {
                (Keep::Head, b)
            };
            keep_line(&state_dir, &keys, &api, id, keep, "a fork found")
                .await
                .unwrap();
            // Opted in: the main line uploads, and nothing else
            let main = check::main_line(
                &check::local_entries(&seal).await.unwrap(),
                &kept.entry_hash,
            )
            .unwrap();
            let (url, took) =
                crate::sealed::rotate::tests::accepting(keys.signing(id).unwrap().verifying_key())
                    .await;
            crate::sealed::uploading::set(
                &dir,
                &keys,
                id,
                crate::sealed::uploading::UploadingBy::OptIn,
            )
            .unwrap();
            let outbox = AgoraOutbox::new(
                agora_agentkit::client::Client::new(url).unwrap(),
                &state_dir,
                std::sync::Arc::new(crate::sealed::SealedEnrolment::new(
                    &state_dir,
                    root.join("secrets"),
                )),
            );
            let Drained::Ran(_) = outbox.drain(id).await.unwrap() else {
                panic!("not drained")
            };
            assert_eq!(*took.lock().unwrap(), main, "keep other {keep_other}");
        }
    }

    /// The fork notice says where the chain is, by its kind.
    #[test]
    fn the_fork_notice_follows_where_the_chain_is() {
        use crate::sealed::notice::ChainKept;
        for (chain, words) in [
            (
                ChainKept::Local,
                "which your runner keeps here and does not upload:",
            ),
            (
                ChainKept::Stopped,
                "(your uploads are stopped, and the versions on Agora are not changed)",
            ),
            (
                ChainKept::Uploading,
                "(the versions already on Agora are not changed)",
            ),
        ] {
            let root = crate::sealed::TestDir::new("accept-fork-notice");
            Notice::fork_kept(&root, "n", 1, false, "kept", chain).unwrap();
            let line = Notice::load(&root).unwrap().unwrap().line().unwrap();
            assert!(line.contains(words), "{line}");
        }
    }
}
