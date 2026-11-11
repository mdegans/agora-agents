//! `agora-seed sealed-restore --agent <id> [--note "…"]`: put a refused
//! agent's last sealed version back, an operator's act.
//!
//! For damage (a wipe, a bad write): the state here matches no sealed
//! version, so the start check refused the agent and the whole run. The
//! last sealed version is taken from the verified main line to
//! `head.json`'s head ([`super::version::last_sealed`]: blobs no longer
//! here are fetched from Agora and checked against their entries), its
//! plaintext hash must be `head.json`'s, and it is written as the agent's
//! `state.json` by agentkit's `FsStorage::save_sealed_raw`, which first
//! renames the damaged file to `state.<utc>.json`: nothing is deleted. The
//! refusal is cleared (kept as `refused.cleared.<utc>`) and the agent's
//! next session is told it was restored, from which date, and that the
//! state found here is kept by its operator. No new version is sealed: the
//! next start finds the state to be the head's, intact.
//!
//! **When the state or `sealed/` is gone** (a lost disk, a bad `rm`, the
//! May loss): with the agent's keys still in the secrets tree, the chain is
//! read from Agora, verified with those keys, and `head.json` rebuilt from
//! it (agentkit's `adopt_server_head`, as `sealed-accept` does for a lost
//! head; never a genesis), then the head is opened and written as above,
//! and the enrolment marker put back. The directory is recreated if it is
//! gone. Without the keys nothing can be opened, and it says so. The
//! uploaded entries aren't copied into `sent/`: the start check reads the
//! main line through Agora's chain, as after a rebase.
//!
//! **An agent that never uploaded** ([`super::version::Reach::Local`]):
//! the same, from what is here alone, and nothing is asked of Agora. A
//! lost `head.json` is rebuilt from the local outbox entries and blobs
//! ([`adopt_local`]: the newest committed entry is the head, its line
//! verified with the runner's keys, through agentkit's
//! `adopt_server_head` fed the local chain). With `sealed/` itself gone
//! there is nothing to restore from: it says to restore `state/<id>/` and
//! `secrets/<id>/` from a backup, and never falls back to Agora. A missing
//! `state_key.json` is recovered from the wrap on Agora only for an agent
//! whose chain is there ([`super::keys::recover_state_key`]).
//!
//! **A save cut short.** agentkit's FsStorage writes `state.json.tmp`,
//! links the old `state.json` to its archive, then renames the tmp over
//! it; a crash in between leaves the old state in place and the complete
//! new one in the tmp, which the next write sets aside as
//! `state.json.tmp.<utc-ms>.aside`. The seal precedes the local write, so
//! such a save left a prepared outbox item (`.tmp`, or `.abandoned` once a
//! later save passed it) recording its state's hash and following the
//! head: only a file matching one is a cut save ([`cut_saves`]), never one
//! judged by its time. One is never passed over in silence: the restore
//! refuses, naming it; `--from-cut-save --note` puts it in place and seals
//! it ([`restore_cut_save`]). Only the envelope's shape is read, never
//! content into logs.
//!
//! A state here that isn't refused and has its `head.json` is never
//! overwritten (but by `--from-cut-save`): a deliberate change is
//! `sealed-accept --note` instead.
//! Like every operator command it refuses while the agent has a session,
//! and holds the agent's `sealed/lock` throughout.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use agora_agentkit::govlog::Sha256Hex;
use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::FsStorage;
use agora_agentkit::state::SignedStateEntry;
use agora_agentkit::state::storage::{AgentSeal, as_reloaded, plaintext_hash};
use agora_agentkit::state::storage::{OutboxItem, SEALED_DIR};
use agora_agentkit::state::verify::HeadRef;
use anyhow::Context;
use chrono::{DateTime, Utc};

use super::KeyStore;
use super::api::SealedApi;
use super::notice::{Notice, Refusal};

/// What `sealed-restore` did: hashes and dates only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Restored {
    pub reason: Refusal,
    pub head: HeadRef,
    /// The restored version's `state_at`.
    pub from: DateTime<Utc>,
    /// `head.json` was gone and was rebuilt from Agora's verified chain.
    pub rebuilt_head: bool,
    /// The agent's backups stay stopped (its own request, kept in the
    /// secrets tree).
    pub stopped: bool,
    /// The state put back was a save cut short (`--from-cut-save`): the
    /// file it came from.
    pub cut_save: Option<String>,
    /// `state_key.json` was missing and was recovered from Agora.
    pub recovered_key: bool,
}

/// What `sealed-restore` may do beyond the last sealed version.
#[derive(Debug, Clone, Copy, Default)]
pub struct Options<'a> {
    /// Added to what the agent is told.
    pub note: Option<&'a str>,
    /// `head.json` lost and a state here that isn't Agora's latest:
    /// replace it anyway.
    pub discard_local: bool,
    /// Put back the newest save cut short ([`cut_saves`]) instead.
    pub from_cut_save: bool,
}

/// A save that was interrupted, whose complete state is still here. The
/// evidence is the chain's, never a file's time: agentkit seals a save (an
/// outbox `.tmp` item, `.abandoned` once a later save passed it) before
/// the local write, so a genuine cut save is a file whose state hashes to
/// such an item's recorded plaintext hash, the item following the current
/// head (`prev_hash`). Only such a file is one.
pub struct CutSave {
    /// `state.json.tmp`, a `state.json.tmp.<utc>.aside`, or `state.json`
    /// itself (already put in place, not yet sealed: a restore cut after
    /// its write).
    pub path: PathBuf,
    /// What a load of it hashes to: the item's.
    pub hash: Sha256Hex,
    /// The prepared save it is.
    pub item: OutboxItem,
    /// It is `state.json`.
    pub in_place: bool,
    /// Never logged.
    state: serde_json::Value,
}

/// A name `sealed-restore` gives a tmp it sets aside itself, so a rerun
/// doesn't take it for a save of the agent's.
const RESTORE_ASIDE: &str = ".restored-over.aside";

/// The saves cut short here ([`CutSave`]), the one already in place first,
/// then the newest; and the files that look like one (a tmp, an aside) but
/// aren't, by path. A candidate must also load as an agent state, by the
/// runner's own typed load. Reads the envelope's shape and hashes only.
pub async fn cut_saves(seal: &AgentSeal) -> anyhow::Result<(Vec<CutSave>, Vec<PathBuf>)> {
    #[derive(serde::Deserialize)]
    struct Envelope {
        format: u64,
        state: serde_json::Value,
    }
    let agent_dir = seal.dir().parent().unwrap_or(seal.dir()).to_path_buf();
    let Some(head) = seal.head().await? else {
        return Ok((Vec::new(), Vec::new()));
    };
    // Prepared saves that follow the head, with what they sealed
    let mut prepared = Vec::new();
    if let Ok(entries) = std::fs::read_dir(seal.dir().join("outbox")) {
        for e in entries.flatten() {
            let path = e.path();
            if !matches!(
                path.extension().and_then(|x| x.to_str()),
                Some("tmp" | "abandoned")
            ) {
                continue;
            }
            let Ok(item) = std::fs::read(&path)
                .map_err(|e| e.to_string())
                .and_then(|b| serde_json::from_slice::<OutboxItem>(&b).map_err(|e| e.to_string()))
            else {
                continue;
            };
            if item.entry.entry.prev_hash == Some(head.head.entry_hash)
                && item.plaintext_sha256.is_some()
            {
                prepared.push(item);
            }
        }
    }
    let mut found = Vec::new();
    let mut others = Vec::new();
    let Ok(entries) = std::fs::read_dir(&agent_dir) else {
        return Ok((found, others));
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let in_place = name == "state.json";
        let candidate = name == "state.json.tmp"
            || (name.starts_with("state.json.tmp.")
                && name.ends_with(".aside")
                && !name.ends_with(RESTORE_ASIDE));
        if !in_place && !candidate {
            continue;
        }
        let state = std::fs::read(entry.path())
            .ok()
            .and_then(|b| serde_json::from_slice::<Envelope>(&b).ok())
            .filter(|e| e.format <= agora_agentkit::reactor::storage::MAX_FORMAT)
            .map(|e| e.state)
            .filter(|s| {
                serde_json::from_value::<agora_agentkit::reactor::seed::SeedState>(s.clone())
                    .is_ok()
            })
            .and_then(|s| as_reloaded(&s).ok());
        let matched = state.and_then(|state| {
            let hash = plaintext_hash(&state);
            prepared
                .iter()
                .find(|i| i.plaintext_sha256 == Some(hash))
                .map(|item| CutSave {
                    path: entry.path(),
                    hash,
                    item: item.clone(),
                    in_place,
                    state,
                })
        });
        match matched {
            Some(c) => found.push(c),
            None if candidate => others.push(entry.path()),
            None => {}
        }
    }
    found.sort_by_key(|c| (!c.in_place, std::cmp::Reverse(c.item.ord)));
    others.sort();
    Ok((found, others))
}

/// `sealed-restore --from-cut-save --note`: put in place the newest save
/// that was interrupted ([`cut_saves`]), and seal it: the prepared item it
/// was, committed (or, passed by a later save, sealed on the head). The
/// state here is archived by FsStorage. A rerun after a cut finishes it
/// (the save already in place is recognised and sealed). Nothing else is
/// one: a file the evidence doesn't back is named and left as it is.
async fn restore_cut_save(
    state_dir: &Path,
    keys: &KeyStore,
    id: AgentId,
    note: Option<&str>,
) -> anyhow::Result<Restored> {
    let note = note
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .context("--note is required with --from-cut-save: the agent reads it")?;
    let agent_dir = state_dir.join(id.to_string());
    anyhow::ensure!(
        std::fs::exists(&agent_dir)?,
        "no agent {id} here ({}): nothing was written",
        agent_dir.display()
    );
    let seal = AgentSeal::new(&agent_dir, id);
    let lock = super::session::take(state_dir, id, true).await?;
    let k = keys
        .seal_keys(id)
        .context("no sealing keys for the agent")?;
    let (cuts, others) = cut_saves(&seal).await?;
    let Some(cut) = cuts.into_iter().next() else {
        anyhow::bail!(
            "no save that was interrupted is here for agent {id}: a file is one only when a \
             sealed version prepared after the current head records its state.{} Nothing was \
             written",
            if others.is_empty() {
                String::new()
            } else {
                format!(
                    " Not one, left as they are: {}",
                    others
                        .iter()
                        .map(|p| p.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        );
    };
    let refused = seal.is_refused().await?;
    let reason = if refused {
        Refusal::from_marker(
            &std::fs::read_to_string(seal.dir().join("refused")).unwrap_or_default(),
        )
    } else {
        Refusal::Mismatch
    };
    let from = DateTime::from_timestamp_micros(cut.item.entry.entry.state_at)
        .context("the save's date is out of range")?;
    let found = !cut.in_place && std::fs::exists(agent_dir.join("state.json"))?;
    Notice::cut_save_restored(&agent_dir, from, note, reason, found, refused)
        .context("writing the notice")?;
    super::crash::point("restore:noticed")?;
    // A refusal blocks every seal
    if seal.is_refused().await? {
        seal.clear_refusal().await.context("clearing the refusal")?;
    }
    if !cut.in_place {
        // Archives the state here as state.<utc>.json first, and sets a
        // leftover tmp aside whole (agentkit)
        FsStorage::new(state_dir)
            .save_sealed_raw(id, cut.state.clone())
            .await
            .context("writing the interrupted save's state")?;
    }
    super::crash::point("restore:written")?;
    // Sealed, so the next start finds it intact
    let committed = matches!(
        seal.recover_locked(&lock, Some(&cut.hash)).await?,
        agora_agentkit::state::storage::Recovery::Intact(_)
            | agora_agentkit::state::storage::Recovery::Committed(_)
    );
    if !committed {
        let head = super::session::append(
            &seal,
            &lock,
            &k,
            &agora_agentkit::state::codec::KeyframePolicy::default(),
            agora_agentkit::enums::StateEntryKind::Snapshot,
            &cut.state,
            from,
        )
        .await
        .context("sealing the interrupted save on the head")?;
        super::enroll::record_sizes(&agent_dir, head, Utc::now(), &cut.state);
    }
    super::crash::point("restore:cut_sealed")?;
    let head = seal.head().await?.context("head.json vanished")?.head;
    drop(lock);
    tracing::warn!(
        event_type = "sealed_state_cut_save_restored",
        agent_id = %id,
        reason = reason.as_str(),
        refused,
        committed,
        head = %head.entry_hash,
        seq = head.seq,
        from = %from,
        "the operator put an interrupted save in place; a state found here is archived"
    );
    Ok(Restored {
        reason,
        head,
        from,
        rebuilt_head: false,
        stopped: super::stop::is_stopped(&agent_dir),
        cut_save: Some(cut.path.display().to_string()),
        recovered_key: false,
    })
}

/// Before a lost `head.json` of an agent that never uploaded is rebuilt
/// over a state here: that state must be its newest sealed version (the
/// hash recorded when it was sealed), or the restore would roll it back.
/// A state that won't load is damage, and doesn't stop it.
async fn local_is_newest(seal: &AgentSeal, state_dir: &Path, id: AgentId) -> anyhow::Result<()> {
    use agora_agentkit::reactor::Storage as _;
    let Some(here) = FsStorage::new(state_dir)
        .load_raw(id)
        .await
        .ok()
        .and_then(|l| as_reloaded(&l.value).ok())
        .map(|v| plaintext_hash(&v))
    else {
        return Ok(());
    };
    let mut items = committed_items(seal).await?;
    items.extend(seal.sent().await.context("reading sealed/sent")?);
    items.sort_by_key(|i| i.ord);
    let Some(tip) = committed_items(seal)
        .await?
        .into_iter()
        .max_by_key(|i| i.ord)
    else {
        return Ok(());
    };
    // Back from the tip to the newest version that recorded its state
    let mut at = Some(tip.entry.entry_hash());
    let mut newest = None;
    while let Some(hash) = at {
        let Some(item) = items.iter().find(|i| i.entry.entry_hash() == hash) else {
            break;
        };
        if let Some(p) = item.plaintext_sha256 {
            newest = Some((p, item.entry.entry.state_at));
            break;
        }
        at = item.entry.entry.prev_hash;
    }
    let Some((newest, state_at)) = newest else {
        return Ok(());
    };
    if newest == here {
        return Ok(());
    }
    anyhow::bail!(
        "agent {id}'s sealed/head.json is lost, and the state here is not its newest sealed \
         version (of {}): restoring would replace it with that version. Choose: `agora-seed \
         sealed-accept --agent {id} --note \"…\"` keeps the state here, sealed on top of its \
         history; `agora-seed sealed-restore --agent {id} --discard-local` replaces it with the \
         newest sealed version (the one here is archived). Nothing was written",
        DateTime::from_timestamp_micros(state_at)
            .map(|d| d.format("%Y-%m-%d %H:%M UTC").to_string())
            .unwrap_or_default()
    )
}

/// Rebuild a lost `head.json` from the agent's own chain here: the newest
/// committed outbox entry is the head, its line (from `sent/` and the
/// outbox) verified under `trusted`, its state opened from the local
/// blobs, by agentkit's `adopt_server_head` fed the local chain. For an
/// agent that never uploaded; never a genesis. `current`: the state here,
/// to say whether it is the head's.
pub async fn adopt_local(
    seal: &AgentSeal,
    lock: &agora_agentkit::state::storage::SealLock,
    trusted: &agora_agentkit::state::verify::TrustedKeys,
    k: &agora_agentkit::state::storage::SealKeys,
    current: Option<&serde_json::Value>,
) -> anyhow::Result<agora_agentkit::state::storage::Adopted> {
    let id = seal.agent_id();
    let mut items = committed_items(seal).await?;
    items.sort_by_key(|i| i.ord);
    let tip = items.last().with_context(|| {
        format!(
            "agent {id}'s sealed/head.json is missing and its outbox holds no committed version \
             to rebuild it from; nothing of it is on Agora. Restore `state/{id}/` and \
             `secrets/{id}/` from a backup"
        )
    })?;
    let mut entries: Vec<SignedStateEntry> = seal
        .sent()
        .await
        .context("reading sealed/sent")?
        .into_iter()
        .map(|i| i.entry)
        .collect();
    entries.extend(items.iter().map(|i| i.entry.clone()));
    let line = super::check::main_line(&entries, &tip.entry.entry_hash()).with_context(|| {
        format!(
            "the local chain to the newest version (seq {}) has a missing link",
            tip.entry.entry.seq
        )
    })?;
    let fetch = |sha: Sha256Hex| async move {
        seal.read_blob(&sha).await.map_err(|e| {
            Box::<dyn std::error::Error + Send + Sync>::from(format!(
                "blob {sha} isn't here ({e}), and nothing of this agent is on Agora"
            ))
        })
    };
    seal.adopt_server_head(lock, &line, trusted, k, current, fetch)
        .await
        .context("rebuilding head.json from the local chain")
}

/// The outbox's committed items (`*.json`), read directly: agentkit's
/// `outbox()` needs `head.json`.
async fn committed_items(seal: &AgentSeal) -> anyhow::Result<Vec<OutboxItem>> {
    let dir = seal.dir().join("outbox");
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).context("reading sealed/outbox"),
    };
    let mut out = Vec::new();
    for entry in entries {
        let path = entry.context("reading sealed/outbox")?.path();
        if path.extension().and_then(|x| x.to_str()) != Some("json") {
            continue;
        }
        out.push(serde_json::from_slice(&std::fs::read(&path)?).map_err(|e| {
            anyhow::anyhow!(
                "{} isn't an outbox item ({:?})",
                path.display(),
                e.classify()
            )
        })?);
    }
    Ok(out)
}

/// Every entry `sealed/` holds that `head.json` would name: `sent/`, and
/// the outbox read directly (agentkit's `outbox()` needs `head.json`),
/// `.json` and `.tmp`. Unreadable is an error: no evidence, no adoption.
async fn held_entries(seal: &AgentSeal) -> anyhow::Result<Vec<(Sha256Hex, u64)>> {
    let mut out: Vec<(Sha256Hex, u64)> = seal
        .sent()
        .await
        .context("reading sealed/sent")?
        .into_iter()
        .map(|i| (i.entry.entry_hash(), i.entry.entry.seq))
        .collect();
    let dir = seal.dir().join("outbox");
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(e).context("reading sealed/outbox"),
    };
    for entry in entries {
        let path = entry.context("reading sealed/outbox")?.path();
        if !matches!(
            path.extension().and_then(|x| x.to_str()),
            Some("json" | "tmp")
        ) {
            continue;
        }
        let item: OutboxItem = serde_json::from_slice(
            &std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
        )
        .map_err(|e| {
            anyhow::anyhow!(
                "{} isn't an outbox item ({:?})",
                path.display(),
                e.classify()
            )
        })?;
        out.push((item.entry.entry_hash(), item.entry.entry.seq));
    }
    Ok(out)
}

/// Before adopting Agora's chain over a `sealed/` that is still here:
/// nothing held here (sent, waiting, or anchored) may lie past Agora's
/// head or off its line. Otherwise Agora serves an older or other history
/// than this runner sealed, and restoring from it would roll the agent
/// back.
async fn no_rollback(
    seal: &AgentSeal,
    agent_dir: &Path,
    id: AgentId,
    chain: &[SignedStateEntry],
) -> anyhow::Result<()> {
    // Agora's main line: from its head back by prev_hash, not every entry
    // it returned
    let head = chain.last().context("Agora holds no chain")?;
    let line = super::check::main_line(chain, &head.entry_hash())
        .context("Agora's chain has a missing link back from its head: not restoring from it")?;
    let on_line: HashMap<Sha256Hex, u64> =
        line.iter().map(|e| (e.entry_hash(), e.entry.seq)).collect();
    let head_seq = head.entry.seq;
    let mut held = held_entries(seal).await?;
    let anchors =
        super::anchors::AnchorsFile::load(agent_dir).context("reading sealed/anchors.json")?;
    held.extend(anchors.proofs.iter().map(|p| (p.entry_hash, p.seq)));
    let off: Vec<&(Sha256Hex, u64)> = held
        .iter()
        .filter(|(hash, seq)| *seq > head_seq || on_line.get(hash) != Some(seq))
        .collect();
    if let Some((hash, seq)) = off.first() {
        anyhow::bail!(
            "agent {id}'s sealed/ holds {} version(s) past Agora's head (seq {head_seq}) or off \
             its line, e.g. {hash} at seq {seq}: Agora may be serving an older history than \
             this runner sealed, and restoring from it would roll the agent back. Nothing was \
             written. If the state here is right, `sealed-accept --agent {id}` rebuilds \
             head.json with it",
            off.len()
        );
    }
    Ok(())
}

/// How many of Agora's newest versions a local state is compared with.
const COMPARED: usize = 50;

/// Before replacing a state still here with Agora's head (its sealed/ or
/// head.json lost): it must be that head already. Otherwise sessions not
/// yet uploaded would be reverted, so the operator chooses: `sealed-accept`
/// keeps it, `sealed-restore --discard-local` replaces it. Says how far
/// behind it is when it is an older version on Agora's line.
async fn local_is_agoras(
    seal: &AgentSeal,
    api: &dyn SealedApi,
    k: &agora_agentkit::state::storage::SealKeys,
    id: AgentId,
    state_dir: &std::path::PathBuf,
    chain: &[SignedStateEntry],
) -> anyhow::Result<()> {
    use agora_agentkit::enums::StateEntryKind;
    use agora_agentkit::reactor::Storage as _;
    let local = match FsStorage::new(state_dir).load_raw(id).await {
        Ok(l) => Some(plaintext_hash(&as_reloaded(&l.value)?)),
        Err(_) => None,
    };
    let head = chain.last().context("Agora holds no chain")?;
    let line = super::check::main_line(chain, &head.entry_hash())
        .context("Agora's chain has a missing link back from its head")?;
    let mut behind = None;
    if let Some(local) = local {
        for (n, entry) in line
            .iter()
            .rev()
            .filter(|e| e.entry.kind == StateEntryKind::Snapshot)
            .take(COMPARED)
            .enumerate()
        {
            let Ok(state) = super::version::open(seal, api, k, &line, &entry.entry).await else {
                continue;
            };
            if as_reloaded(&state).is_ok_and(|s| plaintext_hash(&s) == local) {
                behind = Some(n);
                break;
            }
        }
    }
    let found = match behind {
        Some(0) => return Ok(()),
        Some(n) => {
            format!("the state here is Agora's version from {n} version(s) before its latest")
        }
        None => format!(
            "the state here matches none of Agora's latest {COMPARED} versions (sessions not yet \
             uploaded when sealed/ was lost?)"
        ),
    };
    anyhow::bail!(
        "agent {id}: {found}. Restoring would replace it with Agora's latest. Choose: \
         `agora-seed sealed-accept --agent {id} --note \"…\"` keeps the state here, sealed on top \
         of Agora's history; `agora-seed sealed-restore --agent {id} --discard-local` replaces it \
         with Agora's latest (the one here is archived)"
    )
}

/// [`restore_with`] without `--from-cut-save`: the tests' form.
#[cfg(test)]
pub async fn restore(
    state_dir: &Path,
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
    note: Option<&str>,
    discard_local: bool,
) -> anyhow::Result<Restored> {
    restore_with(
        state_dir,
        keys,
        api,
        id,
        Options {
            note,
            discard_local,
            from_cut_save: false,
        },
    )
    .await
}

/// Restore `id`'s last sealed version as its state: a refused agent's, or
/// one whose state or `sealed/` (or whole directory) is gone. A state
/// here that isn't refused and has its `head.json` is never overwritten
/// ([`Options`] says what more it may do).
pub async fn restore_with(
    state_dir: &Path,
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
    options: Options<'_>,
) -> anyhow::Result<Restored> {
    let Options {
        note,
        discard_local,
        from_cut_save,
    } = options;
    if from_cut_save {
        return restore_cut_save(state_dir, keys, id, note).await;
    }
    let agent_dir = state_dir.join(id.to_string());
    let seal = AgentSeal::new(&agent_dir, id);
    let reach = super::version::reach(&agent_dir, keys, id);
    let no_keys = || {
        anyhow::anyhow!(
            "no keys for agent {id} in the secrets directory: its backups on Agora can't be \
             opened or checked without them, so there is nothing this runner can restore"
        )
    };
    if keys.signing(id).is_none() {
        return Err(no_keys());
    }
    // Nothing of it on Agora, and nothing sealed here: nothing to restore
    // from, and never a fallback to the server
    anyhow::ensure!(
        reach.on_agora() || std::fs::exists(seal.dir())?,
        "agent {id}'s sealed/ is gone, and nothing of it is on Agora: there is nothing to \
         restore it from here. Restore `state/{id}/` and `secrets/{id}/` from a backup"
    );
    let state_path = agent_dir.join("state.json");
    // A restore cut between FsStorage's archive and its rename leaves the
    // new state as state.json.tmp and the found one archived: found
    let found = std::fs::exists(&state_path)? || std::fs::exists(agent_dir.join("state.json.tmp"))?;
    anyhow::ensure!(
        !std::fs::read_to_string(seal.dir().join("refused"))
            .is_ok_and(|r| Refusal::from_marker(&r) == Refusal::StateTooNew),
        "agent {id}'s state was written by a newer runner than this one: upgrade the runner; \
         don't restore or accept over it"
    );
    let had_head = seal.head().await?.is_some();
    let refused = seal.is_refused().await?;
    let cut_hint = async || -> anyhow::Result<String> {
        Ok(
            match cut_saves(&seal).await?.0.into_iter().find(|c| !c.in_place) {
                Some(c) => format!(
                    " A save that was interrupted is beside it: {} (the sealed version prepared \
                 at seq {} records it). `sealed-restore --agent {id} --from-cut-save --note \
                 \"…\"` puts it in place",
                    c.path.display(),
                    c.item.entry.entry.seq
                ),
                None => String::new(),
            },
        )
    };
    if !(refused || !std::fs::exists(&state_path)? || !had_head || !seal.is_enrolled().await?) {
        anyhow::bail!(
            "agent {id} is not refused and its state and sealed head are here: the start check \
             finds it intact or holds it, and this won't overwrite it (a deliberate change is \
             `sealed-accept --note`).{}",
            cut_hint().await?
        );
    }
    // Made if the directory is gone; no session in progress
    let lock = super::session::take(state_dir, id, true).await?;
    // The state key, recovered from Agora's wrap when it is there: under
    // the lock, after the guards
    let recovered_key = if keys.state_key_file(id)?.is_none() {
        super::keys::recover_state_key(keys, &agent_dir, api, id)
            .await?
            .is_some()
    } else {
        false
    };
    // The keys live in the secrets tree, apart from the state: without
    // them nothing sealed can be opened or verified
    let (Some(k), Some(trusted)) = (keys.seal_keys(id), keys.trusted(id).ok().flatten()) else {
        return Err(no_keys());
    };
    let reason = if refused {
        Refusal::from_marker(
            &std::fs::read_to_string(seal.dir().join("refused")).unwrap_or_default(),
        )
    } else if !found {
        Refusal::StateMissing
    } else {
        Refusal::HeadMissing
    };
    if had_head {
        // A head.json without the marker: a restore cut after it adopted
        // Agora's head. Enrolled, if Agora's chain holds that head (the
        // line to it is verified below)
        if !seal.is_enrolled().await? && reach.on_agora() {
            let head = seal.head().await?.context("head.json vanished")?;
            let chain = super::api::chain(api, id, &k.signing)
                .await
                .map_err(anyhow::Error::msg)
                .context("reading the agent's chain on Agora")?;
            anyhow::ensure!(
                chain.iter().any(|e| e.entry_hash() == head.head.entry_hash),
                "agent {id} is not enrolled in sealed state"
            );
        }
    } else if !reach.on_agora() {
        // Nothing on Agora: from the agent's own chain here, never over a
        // state the start check hasn't judged (not refused) that is newer,
        // unless told
        if std::fs::exists(&state_path)? && !discard_local && !refused {
            local_is_newest(&seal, state_dir, id).await?;
        }
        adopt_local(&seal, &lock, &trusted, &k, None).await?;
        tracing::warn!(
            event_type = "sealed_state_head_adopted",
            agent_id = %id,
            source = "local",
            "head.json rebuilt from the verified local chain, for a restore"
        );
        super::crash::point("restore:adopted")?;
    } else {
        // head.json is gone (with sealed/, or the whole directory): it is
        // rebuilt from Agora's chain, verified with the runner's own keys,
        // the way `sealed-accept` adopts it. Never a genesis
        let chain = super::api::chain(api, id, &k.signing)
            .await
            .map_err(anyhow::Error::msg)
            .context("reading the agent's chain on Agora")?;
        anyhow::ensure!(
            !chain.is_empty(),
            "agent {id} has no sealed head here and Agora holds no chain for it: nothing to \
             restore from"
        );
        no_rollback(&seal, &agent_dir, id, &chain).await?;
        if std::fs::exists(&state_path)? && !discard_local {
            local_is_agoras(&seal, api, &k, id, &state_dir.to_path_buf(), &chain).await?;
        }
        let signing = &k.signing;
        let fetch = move |sha: Sha256Hex| async move {
            api.blob(id, signing, &sha)
                .await
                .map_err(Box::<dyn std::error::Error + Send + Sync>::from)
        };
        seal.adopt_server_head(&lock, &chain, &trusted, &k, None, fetch)
            .await
            .context("rebuilding head.json from Agora's verified chain")?;
        tracing::warn!(
            event_type = "sealed_state_head_adopted",
            agent_id = %id,
            entries = chain.len(),
            "head.json rebuilt from the verified chain on Agora, for a restore"
        );
        super::crash::point("restore:adopted")?;
    }
    let last = super::version::last_sealed(&seal, keys, api, id)
        .await
        .context("opening the last sealed version")?;
    anyhow::ensure!(
        last.line.last().map(|e| e.entry_hash()) == Some(last.snapshot.entry_hash()),
        "the head (seq {}) is not a snapshot: there is no sealed state to restore at it",
        last.head.head.seq
    );
    // What the next start will hash: the value as a load reads it back
    let state = as_reloaded(&last.state).context("the sealed state won't serialize")?;
    let expected = last.head.plaintext_sha256.context(
        "head.json records no plaintext hash, so a restored state couldn't be checked against it",
    )?;
    let got = plaintext_hash(&state);
    anyhow::ensure!(
        got == expected,
        "the opened version hashes to {got}, not head.json's {expected}: not restoring it"
    );
    // A save that was interrupted after the head: offered, never rolled
    // back in silence
    let hint = cut_hint().await?;
    anyhow::ensure!(
        hint.is_empty(),
        "agent {id}: restoring its sealed head (seq {}) would roll back a save made after it.{} \
         instead (the state here is archived). Nothing was written",
        last.head.head.seq,
        hint
    );
    let from = DateTime::from_timestamp_micros(last.snapshot.entry.state_at)
        .context("the version's date is out of range")?;
    // A stop the agent asked for outlives its sealed/: put back before
    // anything marks it enrolled, so it stays stopped (nothing sealed)
    let stopped = super::stop::reconcile(&agent_dir, keys, id)
        .context("putting back the stop kept in the secrets tree")?
        || super::stop::is_stopped(&agent_dir);
    // The notice first: what was found here is known now, and a rerun of
    // a restore cut short keeps it. The marker and the refusal last, so a
    // cut anywhere leaves a rerun to do (refused, or not enrolled)
    Notice::restored(
        &agent_dir,
        from,
        note,
        reason,
        found,
        refused,
        reach == super::version::Reach::Local,
    )
    .context("writing the notice")?;
    super::crash::point("restore:noticed")?;
    // A stray tmp (a save or restore cut short) is kept, never truncated
    // by the write below
    let tmp = agent_dir.join("state.json.tmp");
    if std::fs::exists(&tmp)? {
        let at = Utc::now().format(crate::sealed::STAMP_FORMAT);
        // Not `*.json`: the enrolment's archive listing would take it. Named
        // as restore's own
        std::fs::rename(
            &tmp,
            agent_dir.join(format!("state.json.tmp.{at}{RESTORE_ASIDE}")),
        )
        .context("setting a stray state.json.tmp aside")?;
    }
    super::crash::point("restore:before_write")?;
    // Archives a state.json found here as state.<utc>.json first
    FsStorage::new(state_dir)
        .save_sealed_raw(id, state)
        .await
        .context("writing the restored state")?;
    super::crash::point("restore:written")?;
    // sealed/published, lost with sealed/: always back. Agora holds the
    // agent's choice now, and the agent may have changed it with `config`
    // since the runner's one publish (agentkit 0.151): a profile without an
    // anchor can't tell "turned off" from "never told", so a restore never
    // re-sends `publish: true`, and fails private in the rare second case.
    // The agent's own profile says how to publish with `config`
    let published = agent_dir.join(SEALED_DIR).join(super::anchors::PUBLISHED);
    if !std::fs::exists(&published)? {
        std::fs::write(&published, Utc::now().to_rfc3339().as_bytes())
            .context("writing sealed/published")?;
    }
    if !seal.is_enrolled().await? {
        seal.enroll()
            .await
            .context("writing the enrolment marker")?;
    }
    super::crash::point("restore:enrolled")?;
    if refused && seal.is_refused().await? {
        seal.clear_refusal().await.context("clearing the refusal")?;
    }
    drop(lock);
    let head = last.head.head;
    tracing::warn!(
        event_type = "sealed_state_restored",
        agent_id = %id,
        reason = reason.as_str(),
        refused,
        found,
        stopped,
        recovered_key,
        rebuilt_head = !had_head,
        head = %head.entry_hash,
        seq = head.seq,
        from = %from,
        "the operator restored an agent's last sealed version; a state found here is archived"
    );
    Ok(Restored {
        reason,
        head,
        from,
        rebuilt_head: !had_head,
        stopped,
        cut_save: None,
        recovered_key,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alerts::Alerter;
    use crate::sealed::api::fake::FakeApi;
    use crate::sealed::check::{self, Outcome, tests::enrolled};
    use agora_agentkit::reactor::Storage;

    async fn local(root: &Path, id: AgentId) -> agora_agentkit::govlog::Sha256Hex {
        plaintext_hash(
            &FsStorage::new(root.join("state"))
                .load_raw(id)
                .await
                .unwrap()
                .value,
        )
    }

    async fn wipe_and_refuse(root: &Path, id: AgentId, keys: &KeyStore, dir: &Path) {
        FsStorage::new(root.join("state"))
            .save_raw(id, serde_json::json!({ "memory": "" }), None)
            .await
            .unwrap();
        let Outcome::Refuse { reason, detail } = check::check(
            dir,
            id,
            &local(root, id).await,
            Default::default(),
            keys,
            &FakeApi::new(),
        )
        .await
        else {
            panic!("not refused")
        };
        assert_eq!(reason, Refusal::Mismatch);
        check::refuse(dir, id, "tarn", reason, &detail, &Alerter::off()).await;
    }

    fn archives(dir: &Path) -> usize {
        std::fs::read_dir(dir)
            .unwrap()
            .filter(|e| {
                let n = e
                    .as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .to_string();
                n.starts_with("state.") && n != "state.json"
            })
            .count()
    }

    /// Wiped, refused, restored: the next start is intact, the wiped state
    /// is archived, the agent is told, and no version was added.
    #[tokio::test]
    async fn a_wipe_is_restored_from_the_last_sealed_version() {
        let root = crate::sealed::TestDir::new("restore");
        let (id, keys, dir) = enrolled(&root).await;
        let seal = AgentSeal::new(&dir, id);
        let before = check::local_entries(&seal).await.unwrap().len();
        let head = seal.head().await.unwrap().unwrap().head;
        wipe_and_refuse(&root, id, &keys, &dir).await;
        let archived = archives(&dir);
        let api = FakeApi::new();
        // With a note, which the agent is shown
        let done = restore(
            &root.join("state"),
            &keys,
            &api,
            id,
            Some("a test wipe"),
            false,
        )
        .await
        .unwrap();
        assert_eq!(done.head, head);
        assert_eq!(archives(&dir), archived + 1, "the wiped state is archived");
        assert!(!seal.is_refused().await.unwrap());
        assert!(std::fs::read_dir(dir.join("sealed")).unwrap().any(|e| {
            e.unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("refused.cleared.")
        }));
        assert_eq!(check::local_entries(&seal).await.unwrap().len(), before);
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
            seat[0].contains("did not run: your state did not match your last sealed version. On ")
                && seat[0]
                    .contains(", your operator restored your state from your sealed version of ")
                && seat[0].contains(
                    "The state found here is kept by your operator. Your operator's note: a \
                     test wipe"
                ),
            "{seat:?}"
        );
        assert!(
            seat[1].starts_with("Your sealed state: intact."),
            "{seat:?}"
        );
        assert_eq!(
            check::local_entries(&seal).await.unwrap().len(),
            before,
            "no new version"
        );
        let e = restore(&root.join("state"), &keys, &api, id, None, false)
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("won't overwrite it"), "a healthy state: {e}");
    }

    /// Every entry and blob the agent holds, on a fake Agora.
    async fn on_agora(api: &FakeApi, dir: &Path, id: AgentId) {
        let seal = AgentSeal::new(dir, id);
        *api.chain.lock().unwrap() = Ok(check::local_entries(&seal).await.unwrap());
        for item in seal.outbox().await.unwrap() {
            seal.mark_sent(item.ord).await.unwrap();
        }
        for f in std::fs::read_dir(dir.join("sealed/blobs")).unwrap() {
            let f = f.unwrap();
            let sha = f.file_name().to_str().unwrap().parse().unwrap();
            api.blobs
                .lock()
                .unwrap()
                .insert(sha, std::fs::read(f.path()).unwrap());
        }
    }

    /// The whole directory lost, or only sealed/, the keys kept: restored
    /// from Agora, head.json rebuilt, intact at the next start, and the
    /// next save chains on the head.
    #[tokio::test]
    async fn a_lost_directory_or_sealed_dir_is_restored_from_agora() {
        use agora_agentkit::state::storage::Mirrored;
        for lose in ["agent dir", "sealed/"] {
            let root = crate::sealed::TestDir::new("restore-lost");
            let (id, keys, dir) = enrolled(&root).await;
            let api = FakeApi::new();
            on_agora(&api, &dir, id).await;
            let head = AgentSeal::new(&dir, id).head().await.unwrap().unwrap();
            let gone = root.join("gone");
            match lose {
                "agent dir" => std::fs::rename(&dir, &gone).unwrap(),
                _ => std::fs::rename(dir.join("sealed"), &gone).unwrap(),
            }
            let state_dir = root.join("state");
            let done = restore(&state_dir, &keys, &api, id, None, false)
                .await
                .unwrap();
            assert!(done.rebuilt_head, "{lose}");
            assert_eq!(done.head, head.head, "{lose}");
            let seal = AgentSeal::new(&dir, id);
            assert!(seal.is_enrolled().await.unwrap(), "{lose}");
            assert_eq!(
                local(&root, id).await,
                head.plaintext_sha256.unwrap(),
                "{lose}: the head's state"
            );
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
                panic!("{lose}: {out:?}")
            };
            assert_eq!(hold, None, "{lose}");
            assert!(notice);
            let found = lose == "sealed/";
            assert_eq!(
                seat[0].contains("The state found here is kept by your operator."),
                found,
                "{lose}: {seat:?}"
            );
            if !found {
                assert!(
                    seat[0].starts_with("Your state was lost here. On "),
                    "{seat:?}"
                );
            }
            assert!(
                seat[1].starts_with("Your sealed state: intact, and in step with Agora."),
                "{lose}: {seat:?}"
            );

            // The next save chains on the restored head
            let enrolment = std::sync::Arc::new(crate::sealed::SealedEnrolment::new(
                &state_dir,
                root.join("secrets"),
            ));
            let mut store = Mirrored::new(FsStorage::new(&state_dir), (), enrolment);
            let base = store.load_raw(id).await.unwrap().base;
            store
                .save_raw(id, serde_json::json!({ "memory": "next" }), base.as_ref())
                .await
                .unwrap();
            let next = seal.head().await.unwrap().unwrap().head;
            assert_eq!(next.seq, head.head.seq + 1, "{lose}");
            let entry = seal.outbox().await.unwrap().pop().unwrap().entry.entry;
            assert_eq!(entry.prev_hash, Some(head.head.entry_hash), "{lose}");
        }
    }

    /// head.json lost while sealed/ holds versions Agora doesn't (Agora
    /// serving an older prefix): adopting it would roll the agent back,
    /// so it is refused and nothing is written (review of #337).
    #[tokio::test]
    async fn restore_refuses_to_roll_back_to_an_older_agora_prefix() {
        use agora_agentkit::enums::StateEntryKind;
        use agora_agentkit::state::codec::KeyframePolicy;
        let root = crate::sealed::TestDir::new("restore-rollback");
        let (id, keys, dir) = enrolled(&root).await;
        let api = FakeApi::new();
        on_agora(&api, &dir, id).await;
        let seal = AgentSeal::new(&dir, id);
        let prefix = check::local_entries(&seal).await.unwrap();
        let k = keys.seal_keys(id).unwrap();
        seal.append_state(
            &k,
            &KeyframePolicy::default(),
            StateEntryKind::Snapshot,
            &check::tests::state(2),
            chrono::Utc::now(),
        )
        .await
        .unwrap();
        FsStorage::new(root.join("state"))
            .save_raw(id, check::tests::state(2), None)
            .await
            .unwrap();
        for item in seal.outbox().await.unwrap() {
            seal.mark_sent(item.ord).await.unwrap();
        }
        // Agora's copy is the old prefix only
        *api.chain.lock().unwrap() = Ok(prefix);
        std::fs::rename(dir.join("sealed/head.json"), root.join("head.lost")).unwrap();
        let archived = archives(&dir);
        let e = format!(
            "{:#}",
            restore(&root.join("state"), &keys, &api, id, None, false)
                .await
                .unwrap_err()
        );
        assert!(
            e.contains("Agora may be serving an older history") && e.contains("sealed-accept"),
            "{e}"
        );
        assert!(
            seal.head().await.unwrap().is_none(),
            "head.json not adopted"
        );
        assert_eq!(archives(&dir), archived, "nothing written");
        assert_eq!(
            local(&root, id).await,
            plaintext_hash(&check::tests::state(2))
        );

        // A stored anchor proof past Agora's head is the same evidence
        let root = crate::sealed::TestDir::new("restore-rollback-anchor");
        let (id, keys, dir) = enrolled(&root).await;
        let api = FakeApi::new();
        on_agora(&api, &dir, id).await;
        let head = AgentSeal::new(&dir, id).head().await.unwrap().unwrap().head;
        let file = crate::sealed::anchors::AnchorsFile {
            proofs: vec![agora_agentkit::state::AnchorProof {
                anc_id: "ANC-2026-0009".parse().unwrap(),
                date: chrono::NaiveDate::from_ymd_opt(2026, 11, 9).unwrap(),
                root: Sha256Hex::from([1; 32]),
                entry_hash: Sha256Hex::from([2; 32]),
                seq: head.seq + 3,
                leaf_index: 0,
                leaf_count: 1,
                proof: Vec::new(),
            }],
        };
        std::fs::write(
            dir.join("sealed").join(crate::sealed::anchors::ANCHORS),
            serde_json::to_vec(&file).unwrap(),
        )
        .unwrap();
        std::fs::rename(dir.join("sealed/head.json"), root.join("head.lost")).unwrap();
        let e = format!(
            "{:#}",
            restore(&root.join("state"), &keys, &api, id, None, false)
                .await
                .unwrap_err()
        );
        assert!(e.contains("older history"), "{e}");
    }

    /// A stopped agent that lost sealed/: restored, and still stopped (the
    /// stop kept in the secrets tree is put back), with nothing sealed.
    #[tokio::test]
    async fn a_stopped_agent_stays_stopped_through_a_restore() {
        let root = crate::sealed::TestDir::new("restore-stopped");
        let (id, keys, dir) = enrolled(&root).await;
        let api = FakeApi::new();
        on_agora(&api, &dir, id).await;
        let state_dir = root.join("state");
        crate::sealed::stop::stop(
            &state_dir,
            &keys,
            id,
            "It asked",
            Box::pin(async { Ok(()) }),
        )
        .await
        .unwrap();
        assert!(
            keys.agent_dir(id)
                .join(crate::sealed::stop::DURABLE)
                .exists()
        );
        std::fs::rename(dir.join("sealed"), root.join("sealed.lost")).unwrap();
        let done = restore(&state_dir, &keys, &api, id, None, false)
            .await
            .unwrap();
        assert!(done.stopped);
        assert!(crate::sealed::stop::is_stopped(&dir));
        // Still marked uploading (by the secrets copy), but stopped
        assert!(crate::sealed::uploading::is_uploading(&dir, &keys, id));
        assert!(!crate::sealed::uploads(&dir, &keys, id));
        let stopped = crate::sealed::stop::Stopped::load(&dir).unwrap().unwrap();
        assert_eq!(stopped.note, "It asked");
        // A save is chained, locally: it waits in the outbox, and the
        // drain sends nothing of it
        let enrolment = std::sync::Arc::new(crate::sealed::SealedEnrolment::new(
            &state_dir,
            root.join("secrets"),
        ));
        let outbox = agora_agentkit::state::outbox::AgoraOutbox::new(
            agora_agentkit::client::Client::new("http://127.0.0.1:9".parse().unwrap()).unwrap(),
            &state_dir,
            enrolment.clone(),
        );
        let mut storage = crate::sealed::storage(&state_dir, outbox.clone(), enrolment);
        let base = storage.load_raw(id).await.unwrap().base;
        let saved = storage
            .save_raw(id, serde_json::json!({ "memory": "after" }), base.as_ref())
            .await
            .unwrap()
            .unwrap();
        let seal = AgentSeal::new(&dir, id);
        assert_eq!(seal.outbox().await.unwrap().len(), 1, "chained locally");
        assert_eq!(seal.head().await.unwrap().unwrap().head, saved);
        assert_eq!(
            outbox.drain(id).await.unwrap(),
            agora_agentkit::state::outbox::Drained::LocalOnly
        );

        // The start of a run puts a lost sealed/stopped back too
        std::fs::rename(dir.join("sealed/stopped"), root.join("stopped.lost")).unwrap();
        crate::sealed::stop::reconcile_all(&state_dir, &keys);
        assert!(crate::sealed::stop::is_stopped(&dir));
    }

    /// A restore cut after FsStorage archived the state found but before
    /// the new one took its place: the rerun says the state was found and
    /// kept, not lost.
    #[tokio::test]
    async fn a_rerun_after_a_cut_restore_says_the_state_was_kept() {
        let root = crate::sealed::TestDir::new("restore-cut");
        let (id, keys, dir) = enrolled(&root).await;
        wipe_and_refuse(&root, id, &keys, &dir).await;
        // As FsStorage leaves it: state.json archived, the new one as tmp
        std::fs::copy(dir.join("state.json"), dir.join("state.json.tmp")).unwrap();
        let stray = std::fs::read(dir.join("state.json.tmp")).unwrap();
        std::fs::rename(
            dir.join("state.json"),
            dir.join("state.20261103T000000000Z.json"),
        )
        .unwrap();
        restore(&root.join("state"), &keys, &FakeApi::new(), id, None, false)
            .await
            .unwrap();
        let line = Notice::load(&dir).unwrap().unwrap().line().unwrap();
        assert!(
            line.contains("did not run") && line.contains("The state found here is kept"),
            "{line}"
        );
        // The stray tmp was kept whole, set aside
        let kept: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.to_string_lossy().ends_with(".aside"))
            .collect();
        assert_eq!(kept.len(), 1);
        assert_eq!(std::fs::read(&kept[0]).unwrap(), stray);
    }

    /// sealed/ lost while a session's version was still waiting: the state
    /// here is newer than Agora's latest. Restore refuses to revert it
    /// unless told to; `sealed-accept --note` keeps it, sealed on top.
    #[tokio::test]
    async fn a_newer_local_state_is_never_reverted_silently() {
        let root = crate::sealed::TestDir::new("restore-newer");
        let (id, keys, dir) = enrolled(&root).await;
        let api = FakeApi::new();
        on_agora(&api, &dir, id).await;
        let agora_head = AgentSeal::new(&dir, id).head().await.unwrap().unwrap();
        // A session's save, sealed here, never uploaded
        let state_dir = root.join("state");
        let enrolment = std::sync::Arc::new(crate::sealed::SealedEnrolment::new(
            &state_dir,
            root.join("secrets"),
        ));
        let mut store = agora_agentkit::state::storage::Mirrored::new(
            FsStorage::new(&state_dir),
            (),
            enrolment,
        );
        let base = store.load_raw(id).await.unwrap().base;
        store
            .save_raw(id, check::tests::state(7), base.as_ref())
            .await
            .unwrap();
        let newer = local(&root, id).await;
        std::fs::rename(dir.join("sealed"), root.join("sealed.lost")).unwrap();
        let e = format!(
            "{:#}",
            restore(&state_dir, &keys, &api, id, None, false)
                .await
                .unwrap_err()
        );
        assert!(
            e.contains("matches none of Agora's latest")
                && e.contains("sealed-accept")
                && e.contains("--discard-local"),
            "{e}"
        );
        assert_eq!(local(&root, id).await, newer, "untouched");
        // Kept: sealed on top of Agora's history
        let done = crate::sealed::accept::accept(&state_dir, &keys, &api, id, Some("kept"))
            .await
            .unwrap();
        assert_eq!(done.head.seq, agora_head.head.seq + 1);
        assert_eq!(local(&root, id).await, newer);
        let out = check::check(&dir, id, &newer, Default::default(), &keys, &api).await;
        assert!(matches!(out, Outcome::Run { hold: None, .. }), "{out:?}");

        // The other choice, on a second agent: replaced, archived
        let root = crate::sealed::TestDir::new("restore-newer-discard");
        let (id, keys, dir) = enrolled(&root).await;
        let api = FakeApi::new();
        on_agora(&api, &dir, id).await;
        let agora_head = AgentSeal::new(&dir, id).head().await.unwrap().unwrap();
        FsStorage::new(root.join("state"))
            .save_sealed_raw(id, check::tests::state(8))
            .await
            .unwrap();
        std::fs::rename(dir.join("sealed"), root.join("sealed.lost")).unwrap();
        let archived = archives(&dir);
        restore(&root.join("state"), &keys, &api, id, None, true)
            .await
            .unwrap();
        assert_eq!(archives(&dir), archived + 1);
        assert_eq!(local(&root, id).await, agora_head.plaintext_sha256.unwrap());
    }

    /// Without the keys nothing on Agora can be opened: refused, plainly,
    /// and nothing is written.
    #[tokio::test]
    async fn restore_without_the_keys_is_refused() {
        let root = crate::sealed::TestDir::new("restore-nokeys");
        let (id, _, dir) = enrolled(&root).await;
        let api = FakeApi::new();
        on_agora(&api, &dir, id).await;
        std::fs::rename(&dir, root.join("gone")).unwrap();
        let e = restore(
            &root.join("state"),
            &KeyStore::new(root.join("empty-secrets")),
            &api,
            id,
            None,
            false,
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(e.contains("can't be opened or checked without them"), "{e}");
        assert!(!dir.exists(), "nothing written");
    }

    /// A blob pruned here is fetched from Agora, checked against its entry.
    #[tokio::test]
    async fn a_pruned_blob_is_fetched_from_agora() {
        let root = crate::sealed::TestDir::new("restore-fetch");
        let (id, keys, dir) = enrolled(&root).await;
        wipe_and_refuse(&root, id, &keys, &dir).await;
        let blobs = dir.join("sealed/blobs");
        let moved = root.join("moved-blobs");
        std::fs::rename(&blobs, &moved).unwrap();
        std::fs::create_dir_all(&blobs).unwrap();
        let api = FakeApi::new();
        // Agora doesn't have them either: refused, nothing written
        let archived = archives(&dir);
        assert!(
            restore(&root.join("state"), &keys, &api, id, None, false)
                .await
                .is_err()
        );
        assert_eq!(archives(&dir), archived);
        // A forged blob under the right name: refused
        for f in std::fs::read_dir(&moved).unwrap() {
            let f = f.unwrap();
            let sha = f.file_name().to_str().unwrap().parse().unwrap();
            api.blobs.lock().unwrap().insert(sha, b"forged".to_vec());
        }
        assert!(
            restore(&root.join("state"), &keys, &api, id, None, false)
                .await
                .is_err()
        );
        assert_eq!(archives(&dir), archived);
        for f in std::fs::read_dir(&moved).unwrap() {
            let f = f.unwrap();
            let sha = f.file_name().to_str().unwrap().parse().unwrap();
            api.blobs
                .lock()
                .unwrap()
                .insert(sha, std::fs::read(f.path()).unwrap());
        }
        restore(&root.join("state"), &keys, &api, id, None, false)
            .await
            .unwrap();
        assert!(api.calls.lock().unwrap().iter().any(|c| c == "blob"));
        assert_eq!(
            local(&root, id).await,
            AgentSeal::new(&dir, id)
                .head()
                .await
                .unwrap()
                .unwrap()
                .plaintext_sha256
                .unwrap()
        );
    }

    /// A head.json whose hash isn't the version's, or a chain signed by a
    /// key the runner doesn't trust: nothing is written.
    #[tokio::test]
    async fn restore_refuses_a_hash_mismatch_and_a_broken_chain() {
        let root = crate::sealed::TestDir::new("restore-refuse");
        let (id, keys, dir) = enrolled(&root).await;
        wipe_and_refuse(&root, id, &keys, &dir).await;
        let archived = archives(&dir);
        let api = FakeApi::new();

        // Another runner's keys: the chain's signer isn't trusted
        let other = KeyStore::new(root.join("other-secrets"));
        other
            .ring()
            .insert(id, &agora_agentkit::crypto::generate_keypair().0)
            .unwrap();
        other
            .write_state_key(id, &keys.state_key_file(id).unwrap().unwrap())
            .unwrap();
        let e = restore(&root.join("state"), &other, &api, id, None, false)
            .await
            .unwrap_err();
        assert!(format!("{e:#}").contains("did not verify"), "{e:#}");
        assert_eq!(archives(&dir), archived);

        // head.json's plaintext hash altered
        let path = dir.join("sealed/head.json");
        let mut head: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        head["plaintext_sha256"] = serde_json::json!("00".repeat(32));
        std::fs::write(&path, serde_json::to_vec(&head).unwrap()).unwrap();
        let e = restore(&root.join("state"), &keys, &api, id, None, false)
            .await
            .unwrap_err();
        assert!(format!("{e:#}").contains("not restoring it"), "{e:#}");
        assert_eq!(archives(&dir), archived, "nothing written");
        assert!(AgentSeal::new(&dir, id).is_refused().await.unwrap());
    }

    /// The bytes FsStorage writes for `state`, as a save would.
    async fn envelope(root: &Path, id: AgentId, state: serde_json::Value) -> Vec<u8> {
        let scratch = root.join("scratch-envelope");
        FsStorage::new(&scratch)
            .save_sealed_raw(id, state)
            .await
            .unwrap();
        std::fs::read(scratch.join(id.to_string()).join("state.json")).unwrap()
    }

    /// An agent that never uploaded, wiped and refused: restored from its
    /// own chain, with no call to Agora at all (the strict guard).
    #[tokio::test]
    async fn a_local_only_agent_is_restored_without_agora() {
        let root = crate::sealed::TestDir::new("restore-local");
        let (id, keys, dir) = check::tests::enrolled_local(&root).await;
        let head = AgentSeal::new(&dir, id).head().await.unwrap().unwrap();
        wipe_and_refuse(&root, id, &keys, &dir).await;
        let api = crate::sealed::api::fake::guarded(&root);
        let done = restore(&root.join("state"), &keys, &api, id, Some("a test"), false)
            .await
            .unwrap();
        assert_eq!(done.head, head.head);
        assert_eq!(local(&root, id).await, head.plaintext_sha256.unwrap());
        assert!(api.calls.lock().unwrap().is_empty());
    }

    /// Its head.json lost: rebuilt from the local outbox and blobs, under
    /// the strict guard; its sealed/ lost: nothing to restore from, and
    /// the message says to restore from a backup, with no call to Agora.
    #[tokio::test]
    async fn a_local_only_lost_head_is_rebuilt_here_and_a_lost_sealed_dir_refuses() {
        let root = crate::sealed::TestDir::new("restore-local-head");
        let (id, keys, dir) = check::tests::enrolled_local(&root).await;
        let head = AgentSeal::new(&dir, id).head().await.unwrap().unwrap();
        wipe_and_refuse(&root, id, &keys, &dir).await;
        std::fs::rename(dir.join("sealed/head.json"), root.join("head.lost")).unwrap();
        let api = crate::sealed::api::fake::guarded(&root);
        let done = restore(&root.join("state"), &keys, &api, id, None, false)
            .await
            .unwrap();
        assert!(done.rebuilt_head);
        assert_eq!(done.head, head.head);
        assert_eq!(local(&root, id).await, head.plaintext_sha256.unwrap());
        let line = Notice::load(&dir).unwrap().unwrap().line().unwrap();
        assert!(!line.contains("the copy on Agora"), "{line}");

        let root = crate::sealed::TestDir::new("restore-local-sealed");
        let (id, keys, dir) = check::tests::enrolled_local(&root).await;
        std::fs::rename(dir.join("sealed"), root.join("sealed.lost")).unwrap();
        let api = crate::sealed::api::fake::guarded(&root);
        let e = restore(&root.join("state"), &keys, &api, id, None, false)
            .await
            .unwrap_err()
            .to_string();
        assert!(
            e.contains(&format!(
                "Restore `state/{id}/` and `secrets/{id}/` from a backup"
            )),
            "{e}"
        );
        assert!(api.calls.lock().unwrap().is_empty());
    }

    /// A save cut short as agentkit leaves one: its item prepared (a
    /// `.tmp` following the head, recording the new state's hash), the
    /// local write cut after the tmp, then set aside by the next write.
    async fn cut_save(root: &Path, id: AgentId, keys: &KeyStore, dir: &Path) -> serde_json::Value {
        use agora_agentkit::enums::StateEntryKind;
        use agora_agentkit::state::codec::KeyframePolicy;
        use agora_agentkit::state::storage::Append;
        // A whole agent state: a cut save must load as one
        let newer = as_reloaded(
            &serde_json::to_value(crate::agent_selection_tests::pooled("tarn").1).unwrap(),
        )
        .unwrap();
        let seal = AgentSeal::new(dir, id);
        let lock = seal.lock().await.unwrap();
        let head = seal.head().await.unwrap().unwrap().head;
        let plaintext = agora_agentkit::state::codec::plaintext(&newer);
        seal.prepare(
            &lock,
            &keys.seal_keys(id).unwrap(),
            &KeyframePolicy::default(),
            Append {
                kind: StateEntryKind::Snapshot,
                plaintext: &plaintext,
                plaintext_sha256: Some(plaintext_hash(&newer)),
                state_at: Utc::now(),
                parent: Some(head),
            },
        )
        .await
        .unwrap();
        let aside = dir.join("state.json.tmp.20261111T120000000Z.aside");
        std::fs::write(&aside, envelope(root, id, newer.clone()).await).unwrap();
        newer
    }

    /// A save that was interrupted is never passed over: the restore
    /// refuses, naming it; `--from-cut-save --note` puts it in place,
    /// sealed (its prepared item committed), and says so in true words.
    #[tokio::test]
    async fn a_cut_save_is_offered_not_silently_rolled_back() {
        for refused in [true, false] {
            let root = crate::sealed::TestDir::new("restore-cut-save");
            let (id, keys, dir) = check::tests::enrolled_local(&root).await;
            let newer = cut_save(&root, id, &keys, &dir).await;
            if refused {
                wipe_and_refuse(&root, id, &keys, &dir).await;
            }
            let aside = dir.join("state.json.tmp.20261111T120000000Z.aside");
            let before = std::fs::read(dir.join("state.json")).unwrap();
            let api = crate::sealed::api::fake::guarded(&root);
            let e = format!(
                "{:#}",
                restore(&root.join("state"), &keys, &api, id, None, false)
                    .await
                    .unwrap_err()
            );
            assert!(
                e.contains(&aside.display().to_string()) && e.contains("--from-cut-save"),
                "refused {refused}: {e}"
            );
            assert_eq!(
                std::fs::read(dir.join("state.json")).unwrap(),
                before,
                "untouched"
            );
            let opts = |note| Options {
                note,
                from_cut_save: true,
                ..Options::default()
            };
            assert!(
                restore_with(&root.join("state"), &keys, &api, id, opts(None))
                    .await
                    .is_err(),
                "a note"
            );
            let done = restore_with(&root.join("state"), &keys, &api, id, opts(Some("a crash")))
                .await
                .unwrap();
            assert_eq!(done.cut_save, Some(aside.display().to_string()));
            assert_eq!(local(&root, id).await, plaintext_hash(&newer));
            let seal = AgentSeal::new(&dir, id);
            assert!(!seal.is_refused().await.unwrap());
            assert_eq!(
                seal.head().await.unwrap().unwrap().plaintext_sha256,
                Some(plaintext_hash(&newer)),
                "sealed: the prepared item committed"
            );
            assert!(std::fs::exists(&aside).unwrap(), "kept");
            let line = Notice::load(&dir).unwrap().unwrap().line().unwrap();
            assert!(
                line.contains("was interrupted before it finished")
                    && line.contains("put that saved state in place")
                    && line.contains("Your operator's note: a crash")
                    && !line.contains("sealed version of"),
                "{line}"
            );
            assert!(api.calls.lock().unwrap().is_empty());
        }
    }

    /// A file named like a cut save with no prepared item behind it (here
    /// `{}`, on an intact uploader) is not one: never written, never
    /// sealed, named and left as it is.
    #[tokio::test]
    async fn a_file_without_evidence_is_not_a_cut_save() {
        let root = crate::sealed::TestDir::new("restore-cut-fake");
        let (id, keys, dir) = enrolled(&root).await;
        let fake = dir.join("state.json.tmp.20200101T000000000Z.aside");
        std::fs::write(&fake, b"{\"format\": 3, \"state\": {}}").unwrap();
        let before = std::fs::read(dir.join("state.json")).unwrap();
        let head = AgentSeal::new(&dir, id).head().await.unwrap().unwrap();
        let e = format!(
            "{:#}",
            restore_with(
                &root.join("state"),
                &keys,
                &FakeApi::new(),
                id,
                Options {
                    note: Some("n"),
                    from_cut_save: true,
                    ..Options::default()
                },
            )
            .await
            .unwrap_err()
        );
        assert!(
            e.contains("Not one, left as they are") && e.contains(&fake.display().to_string()),
            "{e}"
        );
        assert_eq!(std::fs::read(dir.join("state.json")).unwrap(), before);
        assert_eq!(
            AgentSeal::new(&dir, id).head().await.unwrap().unwrap(),
            head
        );
        assert!(Notice::load(&dir).unwrap().is_none());
        // And the plain restore doesn't offer it
        let e = restore(&root.join("state"), &keys, &FakeApi::new(), id, None, false)
            .await
            .unwrap_err()
            .to_string();
        assert!(!e.contains("--from-cut-save"), "{e}");
    }

    /// Cut after each step of `--from-cut-save`: the rerun finishes, with
    /// the save sealed once (an in-place save recognised and sealed).
    #[tokio::test]
    async fn a_cut_from_cut_save_finishes_on_rerun() {
        for point in ["restore:noticed", "restore:written"] {
            let root = crate::sealed::TestDir::new("restore-cut-crash");
            let (id, keys, dir) = check::tests::enrolled_local(&root).await;
            let newer = cut_save(&root, id, &keys, &dir).await;
            wipe_and_refuse(&root, id, &keys, &dir).await;
            let api = crate::sealed::api::fake::guarded(&root);
            let opts = Options {
                note: Some("a crash"),
                from_cut_save: true,
                ..Options::default()
            };
            crate::sealed::crash::at(point);
            assert!(
                restore_with(&root.join("state"), &keys, &api, id, opts)
                    .await
                    .is_err()
            );
            restore_with(&root.join("state"), &keys, &api, id, opts)
                .await
                .unwrap_or_else(|e| panic!("{point}: {e:#}"));
            assert_eq!(local(&root, id).await, plaintext_hash(&newer), "{point}");
            let seal = AgentSeal::new(&dir, id);
            assert_eq!(
                seal.head().await.unwrap().unwrap().plaintext_sha256,
                Some(plaintext_hash(&newer)),
                "{point}"
            );
            let line = Notice::load(&dir).unwrap().unwrap().line().unwrap();
            assert!(line.starts_with("Your session on "), "{point}: {line}");
        }
    }

    /// Never uploaded, not refused, its head.json lost, and a state here
    /// newer than its newest sealed version: never replaced without
    /// `--discard-local`.
    #[tokio::test]
    async fn a_lost_local_head_never_rolls_back_a_newer_state() {
        let root = crate::sealed::TestDir::new("restore-local-newer");
        let (id, keys, dir) = check::tests::enrolled_local(&root).await;
        FsStorage::new(root.join("state"))
            .save_sealed_raw(id, check::tests::state(9))
            .await
            .unwrap();
        let newer = local(&root, id).await;
        std::fs::rename(dir.join("sealed/head.json"), root.join("head.lost")).unwrap();
        let api = crate::sealed::api::fake::guarded(&root);
        let e = restore(&root.join("state"), &keys, &api, id, None, false)
            .await
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("sealed-accept") && e.contains("--discard-local"),
            "{e}"
        );
        assert_eq!(local(&root, id).await, newer, "untouched");
        assert!(AgentSeal::new(&dir, id).head().await.unwrap().is_none());
        restore(&root.join("state"), &keys, &api, id, None, true)
            .await
            .unwrap();
        assert_ne!(local(&root, id).await, newer);
    }
}
