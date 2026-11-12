//! `sealed/uploading`: the agent's chain is uploaded to Agora. A positive
//! fact of its own, apart from `sealed/chained` (the chain is kept, which
//! every enrolled agent's is, uploading or not; agora
//! `memory/project_local_chain_for_all.md`, 2026-11-11).
//!
//! **Two copies**, as for [`super::stop`]: `<state>/<id>/sealed/uploading`
//! and `<secrets>/<id>/sealed_uploading`, the same bytes. Both are written
//! with [`super::write_private`] (a uniquely named temporary, fsync,
//! rename, fsync of the directory).
//!
//! **Fail closed, for privacy.** Uploading sends the agent's state off the
//! machine, so a marker in doubt means *not* uploading ([`state`]):
//!
//! | `sealed/uploading` | secrets copy | |
//! |---|---|---|
//! | absent | absent | not uploading |
//! | present | present, the same | uploading |
//! | present | present, different | not uploading, contradictory |
//! | present | absent | not uploading, contradictory. Finished by [`finish`] only where a [`set`] of this code was cut between its writes (its intent, [`SETTING`], is still there and says the same); otherwise it stays in doubt: a secrets copy that was deleted is never written back from the local one |
//! | absent | present | uploading: the local copy was lost with `sealed/` or the directory ([`reconcile`] puts it back) |
//! | either unreadable, or naming another agent | | not uploading |
//!
//! So [`set`] writes its intent into the secrets tree
//! (`sealed_uploading.setting`), then the local copy, and last renames the
//! intent to the secrets copy: the secrets copy is the one whose presence
//! commits the fact, as for a stop, and removing it (alone or with the
//! local copy) is never undone by evidence. A contradictory or unreadable
//! marker is logged at error (`sealed_state_uploading_unreadable`) every
//! time it is read, never overwritten by a start, and alerted once a run
//! ([`start_pass`]). The operator's commands end the doubt, each on the
//! agent's answer in force: `sealed-stop` (it stays not uploading, and is
//! no longer alerted), or an opt-in or resume on a yes
//! ([`set_replacing_doubt`]: the copies in doubt are set aside and a new
//! marker written). Nobody copies a marker by hand.
//!
//! **Has uploaded** ([`has_uploaded`]) is weaker: either copy present and
//! naming this agent, whatever the other says. It lets an agent read back
//! what is already on Agora ([`super::api::UploadGuard`]), never send.
//!
//! **One writer at a time.** [`set`], [`reconcile`], [`finish`] and
//! [`migrate`] read, decide and write under the marker's own lock,
//! `<secrets>/<id>/sealed_uploading.lock` (an exclusive `flock`), so two
//! processes (the runner's start and a `sealed-*` command) can't each
//! write one copy. It is a leaf: nothing else is locked while it is held,
//! so whoever takes it inside the agent's `sealed/lock` or agentkit's
//! directory lock can't deadlock with them.
//!
//! [`migrate`]: the one-time migration of agents that uploaded before
//! the marker existed. **Tied to the old chain marker** (`sealed/enrolled`,
//! which a runner before agentkit 0.158 read as "seal and upload"): an
//! agent is judged only while it still has that marker, and
//! [`migrate_markers`] removes it in the same pass, after the uploading
//! marker is written. Nothing writes `sealed/enrolled` any more, so the
//! migration cannot recur for an agent, whatever is deleted later: no
//! record of "done" exists to lose, and an agent whose marker copies both
//! vanish is never marked again from `sealed/sent/` and its ledger.
//!
//! **A forgotten agent** (`sealed-forget` set its durable copies aside as
//! `sealed_uploading.forgotten.<utc>` / `sealed_stopped.forgotten.<utc>`)
//! is not uploading and not stopped when it is put back, until an explicit
//! opt-in ([`set`]): nothing here makes or restores a marker from evidence
//! older than the forget (a state key set aside counts as one, with no
//! marker beside it). [`migrate`] passes it over, [`finish`] never
//! completes a local copy from before the forget, and the start pass sets
//! such leftover local copies (`sealed/uploading`, `sealed/stopped` that
//! came back with the directory) aside as `<name>.forgotten.<utc>`.

use std::path::{Path, PathBuf};

use agora_agentkit::ids::AgentId;
use agora_agentkit::state::storage::{AgentSeal, MarkerMigration, SEALED_DIR};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::KeyStore;
use crate::alerts::{Alert, AlertKind, Alerter};
use crate::consent::offer::Stored as _;
use crate::consent::sealed_state::ledger::SealLedger;

/// `sealed/uploading`.
pub const UPLOADING: &str = "uploading";
/// `<secrets>/<id>/sealed_uploading`: the durable copy.
pub const DURABLE: &str = "sealed_uploading";
/// `<secrets>/<id>/sealed_uploading.setting`: a [`set`] in progress, the
/// marker's bytes; renamed to [`DURABLE`] to commit it.
pub const SETTING: &str = "sealed_uploading.setting";
/// `<secrets>/<id>/sealed_uploading.lock`: the marker's writers' lock.
pub const LOCK: &str = "sealed_uploading.lock";

/// What set the marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UploadingBy {
    /// [`migrate`]: the agent uploaded before the marker existed.
    Migration,
    /// The agent's yes to the offer, acted on.
    OptIn,
}

/// The marker's contents, the same in both copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Uploading {
    /// Whose: a copy naming another agent is unreadable.
    pub agent_id: AgentId,
    pub since: DateTime<Utc>,
    pub by: UploadingBy,
}

/// What the two copies say together ([`state`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Marker {
    /// Neither copy: the agent doesn't upload.
    Absent,
    Uploading(Uploading),
    /// Both readable, and they disagree (or only the local copy exists).
    Contradictory(&'static str),
    /// A copy that can't be read or parsed, or names another agent.
    Unreadable(String),
}

enum Copy {
    Absent,
    Present(Uploading),
    Unreadable(String),
}

fn local_path(agent_dir: &Path) -> PathBuf {
    agent_dir.join(SEALED_DIR).join(UPLOADING)
}

fn durable_path(keys: &KeyStore, id: AgentId) -> PathBuf {
    keys.agent_dir(id).join(DURABLE)
}

fn setting_path(keys: &KeyStore, id: AgentId) -> PathBuf {
    keys.agent_dir(id).join(SETTING)
}

/// A crash point inside a marker's write, for tests.
fn cut(name: &'static str) -> std::io::Result<()> {
    super::crash::point(name).map_err(std::io::Error::other)
}

/// When `sealed-forget` last set this agent aside, if it ever did: the
/// newest stamp in its secrets directory of anything a forget moves (its
/// state key, the forget's note, a durable uploading marker, a durable
/// stop). A forget of an agent with no marker leaves only the first two.
pub fn forgotten_at(keys: &KeyStore, id: AgentId) -> Option<DateTime<Utc>> {
    let dir = keys.agent_dir(id);
    [
        super::keys::FORGOTTEN_STATE_KEY,
        super::keys::FORGOTTEN_NOTE,
        super::keys::FORGOTTEN_UPLOADING,
        super::keys::FORGOTTEN_STOP,
    ]
    .iter()
    .filter_map(|kind| kind.stamps(&dir).ok())
    .flatten()
    .filter_map(|s| {
        chrono::NaiveDateTime::parse_from_str(&s, super::STAMP_FORMAT)
            .ok()
            .map(|t| t.and_utc())
    })
    .max()
}

/// A forgotten agent put back with its directory: the local copies of
/// its markers that predate the forget (and have no durable copy since),
/// set aside as `sealed/<name>.forgotten.<utc>`, under the marker's lock.
/// Never deleted. Returns the names set aside.
pub fn set_aside_forgotten(
    agent_dir: &Path,
    keys: &KeyStore,
    id: AgentId,
) -> std::io::Result<Vec<&'static str>> {
    let Some(forgot) = forgotten_at(keys, id) else {
        return Ok(Vec::new());
    };
    let _lock = lock(keys, id)?;
    let sealed = agent_dir.join(SEALED_DIR);
    let stamp = Utc::now().format(super::STAMP_FORMAT);
    let mut moved = Vec::new();
    // The uploading copy: from before the forget, nothing made since
    let stale_uploading = !std::fs::exists(durable_path(keys, id))?
        && match read_copy(&local_path(agent_dir), id) {
            Copy::Present(u) => u.since <= forgot,
            Copy::Unreadable(_) => true,
            Copy::Absent => false,
        };
    if stale_uploading {
        std::fs::rename(
            local_path(agent_dir),
            sealed.join(format!("{UPLOADING}.forgotten.{stamp}")),
        )?;
        moved.push(UPLOADING);
    }
    let stale_stop = !std::fs::exists(keys.agent_dir(id).join(super::stop::DURABLE))?
        && std::fs::exists(sealed.join(super::stop::STOPPED))?
        && super::stop::Stopped::load(agent_dir)
            .ok()
            .flatten()
            .is_none_or(|s| s.at <= forgot);
    if stale_stop {
        std::fs::rename(
            sealed.join(super::stop::STOPPED),
            sealed.join(format!("{}.forgotten.{stamp}", super::stop::STOPPED)),
        )?;
        moved.push(super::stop::STOPPED);
    }
    if !moved.is_empty() {
        std::fs::File::open(&sealed)?.sync_all()?;
        tracing::warn!(
            event_type = "sealed_state_forgotten_markers_set_aside",
            agent_id = %id,
            moved = %moved.join(","),
            "a forgotten agent came back with its old local markers; set aside, not acted on"
        );
    }
    Ok(moved)
}

fn read_copy(path: &Path, id: AgentId) -> Copy {
    match std::fs::read(path) {
        Ok(bytes) => match serde_json::from_slice::<Uploading>(&bytes) {
            Ok(u) if u.agent_id == id => Copy::Present(u),
            Ok(u) => Copy::Unreadable(format!(
                "{} names agent {}, not {id}",
                path.display(),
                u.agent_id
            )),
            Err(e) => Copy::Unreadable(format!("{}: {e}", path.display())),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Copy::Absent,
        Err(e) => Copy::Unreadable(format!("{}: {e}", path.display())),
    }
}

/// The marker's lock, held until the file is dropped. Blocking, briefly:
/// every holder only reads and writes the two small copies.
// Public for `sealed-forget`, which sets the secrets copy aside
// under it.
pub fn lock(keys: &KeyStore, id: AgentId) -> std::io::Result<std::fs::File> {
    let dir = keys.agent_dir(id);
    std::fs::create_dir_all(&dir)?;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(dir.join(LOCK))?;
    file.lock()?;
    Ok(file)
}

/// Both copies, read and judged. Never logs.
pub fn state(agent_dir: &Path, keys: &KeyStore, id: AgentId) -> Marker {
    match (
        read_copy(&local_path(agent_dir), id),
        read_copy(&durable_path(keys, id), id),
    ) {
        (Copy::Unreadable(e), _) | (_, Copy::Unreadable(e)) => Marker::Unreadable(e),
        (Copy::Absent, Copy::Absent) => Marker::Absent,
        (Copy::Present(l), Copy::Present(d)) if l == d => Marker::Uploading(d),
        (Copy::Present(_), Copy::Present(_)) => {
            Marker::Contradictory("sealed/uploading and the secrets copy differ")
        }
        (Copy::Present(_), Copy::Absent) => {
            Marker::Contradictory("sealed/uploading is there but the secrets copy is not")
        }
        (Copy::Absent, Copy::Present(d)) => Marker::Uploading(d),
    }
}

/// Whether the agent's chain is uploaded: [`state`], with a marker in
/// doubt logged at error and read as not uploading.
pub fn is_uploading(agent_dir: &Path, keys: &KeyStore, id: AgentId) -> bool {
    match state(agent_dir, keys, id) {
        Marker::Uploading(_) => true,
        Marker::Absent => false,
        Marker::Contradictory(why) => {
            tracing::error!(
                event_type = "sealed_state_uploading_unreadable",
                agent_id = %id,
                problem = "contradictory",
                error = why,
                "the uploading marker is contradictory; the agent is treated as not uploading"
            );
            false
        }
        Marker::Unreadable(e) => {
            tracing::error!(
                event_type = "sealed_state_uploading_unreadable",
                agent_id = %id,
                problem = "unreadable",
                error = %e,
                "the uploading marker is unreadable; the agent is treated as not uploading"
            );
            false
        }
    }
}

/// Whether the agent has ever uploaded: either copy present and naming
/// it, stopped or not, the other copy whatever it says. What is on Agora
/// of such an agent may be read back; nothing may be sent unless it
/// [`super::uploads`]. An agent with neither copy has nothing there.
pub fn has_uploaded(agent_dir: &Path, keys: &KeyStore, id: AgentId) -> bool {
    matches!(read_copy(&local_path(agent_dir), id), Copy::Present(_))
        || matches!(read_copy(&durable_path(keys, id), id), Copy::Present(_))
}

/// Mark `id` as uploading, under the marker's lock: the intent
/// ([`SETTING`]), the local copy, then the intent renamed to the secrets
/// copy, which commits it. Already uploading: nothing written, the marker
/// returned. A marker in doubt is never overwritten, but this code's own
/// half-written one (its intent still there) is finished.
#[cfg(test)]
pub fn set(
    agent_dir: &Path,
    keys: &KeyStore,
    id: AgentId,
    by: UploadingBy,
) -> std::io::Result<Uploading> {
    let _lock = lock(keys, id)?;
    set_locked(agent_dir, keys, id, by)
}

fn set_locked(
    agent_dir: &Path,
    keys: &KeyStore,
    id: AgentId,
    by: UploadingBy,
) -> std::io::Result<Uploading> {
    let marker = match state(agent_dir, keys, id) {
        Marker::Uploading(u) => {
            // The local copy may be the lost one
            reconcile_locked(agent_dir, keys, id)?;
            return Ok(u);
        }
        Marker::Absent => Uploading {
            agent_id: id,
            since: Utc::now(),
            by,
        },
        Marker::Contradictory(why) => {
            return match finish_locked(agent_dir, keys, id)? {
                Some(u) => Ok(u),
                None => Err(std::io::Error::other(why)),
            };
        }
        Marker::Unreadable(e) => return Err(std::io::Error::other(e)),
    };
    let bytes = serde_json::to_vec_pretty(&marker)?;
    super::write_private(&setting_path(keys, id), &bytes)?;
    cut("uploading:intent")?;
    super::write_private(&local_path(agent_dir), &bytes)?;
    cut("uploading:local")?;
    commit(keys, id)?;
    Ok(marker)
}

/// The intent renamed to the secrets copy, durably: the marker's commit.
fn commit(keys: &KeyStore, id: AgentId) -> std::io::Result<()> {
    std::fs::rename(setting_path(keys, id), durable_path(keys, id))?;
    std::fs::File::open(keys.agent_dir(id))?.sync_all()
}

/// [`set`] for the operator's opt-in or resume, which acts on the agent's
/// yes in force: a marker in doubt (contradictory or unreadable) is set
/// aside, each copy as `<name>.doubt.<utc>` (kept, never read again), and
/// a new one written. The one way a marker in doubt becomes a marker:
/// nothing does it from the copies themselves.
pub fn set_replacing_doubt(
    agent_dir: &Path,
    keys: &KeyStore,
    id: AgentId,
    by: UploadingBy,
) -> std::io::Result<Uploading> {
    let _lock = lock(keys, id)?;
    if matches!(
        state(agent_dir, keys, id),
        Marker::Contradictory(_) | Marker::Unreadable(_)
    ) {
        if let Some(u) = finish_locked(agent_dir, keys, id)? {
            return Ok(u);
        }
        let stamp = Utc::now().format(super::STAMP_FORMAT);
        let mut moved = Vec::new();
        for (path, name) in [
            (local_path(agent_dir), UPLOADING),
            (durable_path(keys, id), DURABLE),
        ] {
            if std::fs::symlink_metadata(&path).is_ok() {
                std::fs::rename(&path, path.with_file_name(format!("{name}.doubt.{stamp}")))?;
                moved.push(name);
            }
        }
        std::fs::File::open(agent_dir.join(SEALED_DIR))?.sync_all()?;
        std::fs::File::open(keys.agent_dir(id))?.sync_all()?;
        tracing::warn!(
            event_type = "sealed_state_uploading_doubt_replaced",
            agent_id = %id,
            moved = %moved.join(","),
            "an uploading marker in doubt was set aside for a new one, on the agent's yes in force"
        );
    }
    set_locked(agent_dir, keys, id, by)
}

/// `sealed-stop`'s end of a doubt, after its stop is written (so nothing
/// uploads whatever this does): the copy of the marker that can be read
/// and names the agent (the secrets copy first) is written to the other
/// place, whose own copy, if any, is set aside as `<name>.doubt.<utc>`.
/// The agent then reads as one that uploaded and is stopped: what it has
/// on Agora can be read back and its keys maintained; uploads resume only
/// on its own yes. `lock` is the marker's lock, held by the caller.
/// `Ok(false)`: no doubt, or no copy can be read (it stays in doubt, and
/// stopped).
pub fn settle_stopped(
    agent_dir: &Path,
    keys: &KeyStore,
    id: AgentId,
    lock: &std::fs::File,
) -> std::io::Result<bool> {
    let _ = lock;
    if !matches!(
        state(agent_dir, keys, id),
        Marker::Contradictory(_) | Marker::Unreadable(_)
    ) {
        return Ok(false);
    }
    let (local, durable) = (local_path(agent_dir), durable_path(keys, id));
    let kept = match (read_copy(&durable, id), read_copy(&local, id)) {
        (Copy::Present(u), _) | (_, Copy::Present(u)) => u,
        _ => return Ok(false),
    };
    let bytes = serde_json::to_vec_pretty(&kept)?;
    let stamp = Utc::now().format(super::STAMP_FORMAT);
    for (path, name) in [(local, UPLOADING), (durable, DURABLE)] {
        if matches!(read_copy(&path, id), Copy::Present(u) if u == kept) {
            continue;
        }
        if std::fs::symlink_metadata(&path).is_ok() {
            std::fs::rename(&path, path.with_file_name(format!("{name}.doubt.{stamp}")))?;
        }
        // Never a directory made for an agent that has none here
        if path.parent().is_some_and(|d| d.is_dir()) {
            super::write_private(&path, &bytes)?;
        }
    }
    tracing::warn!(
        event_type = "sealed_state_uploading_doubt_settled_stopped",
        agent_id = %id,
        "a stopped agent's uploading marker in doubt was settled from its readable copy; it stays stopped"
    );
    Ok(true)
}

/// `sealed/uploading` put back from the secrets copy when it is gone and
/// the agent's directory is here. `Ok(true)`: put back.
pub fn reconcile(agent_dir: &Path, keys: &KeyStore, id: AgentId) -> std::io::Result<bool> {
    if !std::fs::exists(durable_path(keys, id))? {
        return Ok(false);
    }
    let _lock = lock(keys, id)?;
    reconcile_locked(agent_dir, keys, id)
}

fn reconcile_locked(agent_dir: &Path, keys: &KeyStore, id: AgentId) -> std::io::Result<bool> {
    if !std::fs::exists(agent_dir)? || std::fs::exists(local_path(agent_dir))? {
        return Ok(false);
    }
    let Copy::Present(u) = read_copy(&durable_path(keys, id), id) else {
        return Ok(false);
    };
    super::write_private(&local_path(agent_dir), &serde_json::to_vec_pretty(&u)?)?;
    tracing::warn!(
        event_type = "sealed_state_uploading_reconciled",
        agent_id = %id,
        "sealed/uploading was missing; put back from the secrets tree"
    );
    Ok(true)
}

/// A [`set`] of this code cut before its commit: the local copy whole, no
/// secrets copy, and the set's own intent ([`SETTING`]) still in the
/// secrets tree saying the same. The intent is renamed to the secrets
/// copy, under the marker's lock. `Ok(Some)`: finished. Anything else is
/// left as it is: a local copy with no secrets copy and no matching intent
/// is a secrets copy that was removed, which nothing here puts back
/// (in doubt, not uploading, alerted).
pub fn finish(
    agent_dir: &Path,
    keys: &KeyStore,
    id: AgentId,
) -> std::io::Result<Option<Uploading>> {
    if !std::fs::exists(local_path(agent_dir))? {
        return Ok(None);
    }
    let _lock = lock(keys, id)?;
    finish_locked(agent_dir, keys, id)
}

fn finish_locked(
    agent_dir: &Path,
    keys: &KeyStore,
    id: AgentId,
) -> std::io::Result<Option<Uploading>> {
    let Copy::Absent = read_copy(&durable_path(keys, id), id) else {
        return Ok(None);
    };
    let Copy::Present(u) = read_copy(&local_path(agent_dir), id) else {
        return Ok(None);
    };
    // Only a set this code began: its intent, the same marker
    match read_copy(&setting_path(keys, id), id) {
        Copy::Present(intent) if intent == u => {}
        _ => return Ok(None),
    }
    // A local copy from before a forget is not an opt-in to complete
    if forgotten_at(keys, id).is_some_and(|f| u.since <= f) {
        return Ok(None);
    }
    commit(keys, id)?;
    tracing::warn!(
        event_type = "sealed_state_uploading_finished",
        agent_id = %id,
        "a half-written uploading marker (its intent, no secrets copy) was finished with the same contents"
    );
    Ok(Some(u))
}

/// Every agent with a state directory or a state key here.
fn all_ids(state_dir: &Path, keys: &KeyStore) -> Vec<AgentId> {
    let mut ids: Vec<AgentId> = match std::fs::read_dir(state_dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_name().to_str()?.parse::<AgentId>().ok())
            .collect(),
        Err(e) => {
            tracing::warn!(error = %e, "the state directory couldn't be listed for uploading markers");
            Vec::new()
        }
    };
    match keys.with_state_keys() {
        Ok(more) => ids.extend(more),
        Err(e) => {
            tracing::warn!(error = %e, "the secrets tree couldn't be listed for uploading markers")
        }
    }
    ids.sort();
    ids.dedup();
    ids
}

/// [`reconcile`] and [`finish`] for every agent here: a lost local copy
/// put back, a half-written marker completed. At the start of a run (and
/// of a `sealed-*` command), before states load.
pub fn reconcile_all(state_dir: &Path, keys: &KeyStore) {
    for id in all_ids(state_dir, keys) {
        let dir = state_dir.join(id.to_string());
        if let Err(e) = set_aside_forgotten(&dir, keys, id) {
            tracing::error!(agent_id = %id, error = %e, "a forgotten agent's old local markers couldn't be set aside");
        }
        if let Err(e) = reconcile(&dir, keys, id) {
            tracing::error!(agent_id = %id, error = %e, "the durable uploading marker couldn't be put back");
        }
        if let Err(e) = finish(&dir, keys, id) {
            tracing::error!(agent_id = %id, error = %e, "a half-written uploading marker couldn't be finished");
        }
    }
}

/// Why [`migrate`] left an agent as it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotMigrated {
    /// No `sealed/enrolled` (the chain marker of a runner before agentkit
    /// 0.158): it never uploaded under one, or was judged already.
    NoOldMarker,
    /// A marker is there already (or one in doubt, never overwritten).
    Marked,
    /// Nothing in `sealed/sent/`: it never uploaded.
    NeverUploaded,
    /// Its answer in force is not a yes the operator may act on now (or
    /// its sealed-state ledger can't be read).
    NotOptedIn,
    /// `sealed-forget` set it aside: only an explicit opt-in marks it
    /// again.
    Forgotten,
}

/// One agent of the one-time migration: marked uploading only if it still
/// has the old chain marker (so it ran under a runner for which enrolled
/// meant uploading), has no uploading marker, has uploaded (a non-empty
/// `sealed/sent/`) and its answer in force is a yes. Reads directory
/// entries and the ledger's answer only; decides and writes under the
/// marker's lock. [`migrate_markers`] calls it before it removes the old
/// marker, which is why it runs once.
pub async fn migrate(
    agent_dir: &Path,
    keys: &KeyStore,
    id: AgentId,
) -> std::io::Result<Result<Uploading, NotMigrated>> {
    let sealed = agent_dir.join(SEALED_DIR);
    // Strictly: an old marker that can't be checked is an error, retried
    if !AgentSeal::new(agent_dir, id).has_legacy_marker().await? {
        return Ok(Err(NotMigrated::NoOldMarker));
    }
    if forgotten_at(keys, id).is_some() {
        return Ok(Err(NotMigrated::Forgotten));
    }
    let _lock = lock(keys, id)?;
    if state(agent_dir, keys, id) != Marker::Absent {
        return Ok(Err(NotMigrated::Marked));
    }
    let sent = match std::fs::read_dir(sealed.join("sent")) {
        Ok(mut entries) => entries.next().is_some(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
        Err(e) => return Err(e),
    };
    if !sent {
        return Ok(Err(NotMigrated::NeverUploaded));
    }
    match SealLedger::load_blocking(agent_dir) {
        Ok(ledger) if ledger.in_force().starts_uploads().is_ok() => {}
        Ok(_) => return Ok(Err(NotMigrated::NotOptedIn)),
        Err(e) => {
            tracing::error!(
                event_type = "sealed_state_uploading_migration_ledger_unreadable",
                agent_id = %id,
                error = %e,
                "sealed-state ledger unreadable; an agent that uploaded is not marked uploading \
                 (`sealed-enroll --opt-in --agent <id> --yes` once its ledger reads)"
            );
            return Ok(Err(NotMigrated::NotOptedIn));
        }
    }
    set_locked(agent_dir, keys, id, UploadingBy::Migration).map(Ok)
}

/// The agents whose marker is still in doubt after the repairs, with why.
pub fn in_doubt(state_dir: &Path, keys: &KeyStore) -> Vec<(AgentId, String)> {
    all_ids(state_dir, keys)
        .into_iter()
        .filter_map(
            |id| match state(&state_dir.join(id.to_string()), keys, id) {
                Marker::Contradictory(why) => Some((id, why.to_string())),
                Marker::Unreadable(e) => Some((id, e)),
                _ => None,
            },
        )
        .collect()
}

/// agentkit 0.158 renamed the chain's marker `sealed/enrolled` to
/// `sealed/chained`, so that no older runner, which reads `enrolled` as
/// "seal and upload", can upload an agent that is only chained. Each
/// agent with the old marker is first judged by [`migrate`] (it uploaded
/// under such a runner and its answer in force is yes: marked uploading,
/// one `sealed_state_uploading_migrated` event), and then moved to the new
/// marker by agentkit's [`AgentSeal::migrate_marker`] (the new marker
/// written and made durable before the old is removed: a crash leaves at
/// least one, and the next start finishes), one
/// `sealed_state_marker_migrated` event each. An agent whose uploading
/// migration failed on I/O keeps its old marker, and is tried again at the
/// next start. The agents moved. Before anything reads the marker.
pub async fn migrate_markers(state_dir: &Path, keys: &KeyStore) -> Vec<AgentId> {
    let ids: Vec<AgentId> = match std::fs::read_dir(state_dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_name().to_str()?.parse::<AgentId>().ok())
            .collect(),
        Err(e) => {
            tracing::warn!(error = %e, "the state directory couldn't be listed for the marker migration");
            return Vec::new();
        }
    };
    let mut moved = Vec::new();
    for id in ids {
        let dir = state_dir.join(id.to_string());
        // While the old marker is still here, and only then
        match migrate(&dir, keys, id).await {
            Ok(Ok(_)) => tracing::warn!(
                event_type = "sealed_state_uploading_migrated",
                agent_id = %id,
                "an agent that uploaded before the uploading marker existed is marked uploading"
            ),
            Ok(Err(_)) => {}
            Err(e) => {
                tracing::error!(
                    event_type = "sealed_state_uploading_migration_failed",
                    agent_id = %id,
                    error = %e,
                    "the uploading migration couldn't be done for an agent; its old chain marker \
                     stays, and it is tried again next start"
                );
                continue;
            }
        }
        let seal = AgentSeal::new(&dir, id);
        match seal.migrate_marker().await {
            Ok(MarkerMigration::Migrated) => {
                tracing::warn!(
                    event_type = "sealed_state_marker_migrated",
                    agent_id = %id,
                    "sealed/enrolled moved to sealed/chained"
                );
                moved.push(id);
            }
            Ok(MarkerMigration::Already | MarkerMigration::NotEnrolled) => {}
            Err(e) => tracing::error!(
                event_type = "sealed_state_marker_migration_failed",
                agent_id = %id,
                error = %e,
                "sealed/enrolled couldn't be moved to sealed/chained; tried again next start"
            ),
        }
    }
    moved.sort();
    moved
}

/// The start's pass over the markers, before states load: the agents
/// with the old chain marker judged for the uploading marker and moved to
/// the new name ([`migrate_markers`]), then [`reconcile_all`] (a lost
/// local copy back, this code's own half-written marker finished, a
/// forgotten agent's old copies set aside); then any agent whose marker is
/// still in doubt, and whose uploads its operator has not stopped, is one
/// summary alert ([`AlertKind::SealedStateUploadingInDoubt`]): an agent
/// that may have said yes and is silently not uploading must be seen. The
/// agents in doubt, stopped or not.
pub async fn start_pass(state_dir: &Path, keys: &KeyStore, alerts: &Alerter) -> Vec<AgentId> {
    migrate_markers(state_dir, keys).await;
    reconcile_all(state_dir, keys);
    let doubt = in_doubt(state_dir, keys);
    // A stopped agent's doubt is settled as far as uploads go: it stays
    // not uploading, by its operator's own command
    let open: Vec<&(AgentId, String)> = doubt
        .iter()
        .filter(|(id, _)| {
            !super::stop::is_stopped(&state_dir.join(id.to_string()))
                && !super::stop::durable(keys, *id).map_or(true, |d| d.is_some())
        })
        .collect();
    if !open.is_empty() {
        let ids: Vec<String> = open.iter().map(|(id, _)| id.to_string()).collect();
        tracing::error!(
            event_type = "sealed_state_uploading_in_doubt",
            agents = open.len(),
            ids = %ids.join(","),
            "uploading markers in doubt: those agents upload nothing until an operator's \
             sealed-stop or opt-in settles it"
        );
        let mut alert = Alert::new(
            AlertKind::SealedStateUploadingInDoubt,
            format!(
                "{} agent(s) with an uploading marker in doubt: nothing of theirs is uploaded",
                open.len()
            ),
        )
        .detail("agents", ids.join(", "));
        for (id, why) in open.iter().take(10) {
            alert = alert.detail("problem", format!("{id}: {why}"));
        }
        alerts.notify(alert);
    }
    doubt.into_iter().map(|(id, _)| id).collect()
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::sealed::check::tests::enrolled_local;
    use agora_agentkit::state::storage::CHAINED;

    /// As a runner before agentkit 0.158 left the agent: `sealed/enrolled`
    /// (which it read as "seal and upload"), no `sealed/chained`.
    pub fn old_marker(dir: &Path) {
        let sealed = dir.join(SEALED_DIR);
        std::fs::write(sealed.join("enrolled"), b"").unwrap();
        let _ = std::fs::remove_file(sealed.join(CHAINED));
    }

    /// Mark a test agent as having uploaded something (`sealed/sent/`).
    pub fn sent_one(dir: &Path) {
        let sent = dir.join(SEALED_DIR).join("sent");
        std::fs::create_dir_all(&sent).unwrap();
        std::fs::write(sent.join("0.json"), b"{}").unwrap();
    }

    /// The agent's sealed-state ledger, saying yes or no.
    pub fn ledger(dir: &Path, yes: bool) {
        use crate::consent::sealed_state::ledger::{LEDGER_FILE, SealAsk, SealOutcome};
        use crate::consent::sealed_state::prompt::SealChoice;
        let choice = if yes {
            SealChoice::Private
        } else {
            SealChoice::Decline
        };
        let ledger = SealLedger {
            asks: vec![SealAsk {
                at: "2026-11-08T09:00:00Z".parse().unwrap(),
                offer_version: 1,
                model: misanthropic::model::Model::from("claude-haiku-4-5"),
                order: SealChoice::ALL,
                order_seed: 42,
                constrained: true,
                attempts: 1,
                outcome: SealOutcome::Answered {
                    choice,
                    reason: String::new(),
                    memory_note_written: false,
                    note_dropped: None,
                    apply_failed: None,
                },
            }],
            ..SealLedger::default()
        };
        std::fs::write(dir.join(LEDGER_FILE), serde_json::to_vec(&ledger).unwrap()).unwrap();
    }

    #[tokio::test]
    async fn the_marker_fails_closed() {
        let root = crate::sealed::TestDir::new("uploading-marker");
        let (id, keys, dir) = enrolled_local(&root).await;
        assert_eq!(state(&dir, &keys, id), Marker::Absent);
        assert!(!is_uploading(&dir, &keys, id));

        let u = set(&dir, &keys, id, UploadingBy::OptIn).unwrap();
        assert!(is_uploading(&dir, &keys, id));
        assert_eq!(
            set(&dir, &keys, id, UploadingBy::Migration).unwrap(),
            u,
            "idempotent"
        );
        let local = std::fs::read(local_path(&dir)).unwrap();
        assert_eq!(std::fs::read(durable_path(&keys, id)).unwrap(), local);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(local_path(&dir))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        let no_tmp = std::fs::read_dir(dir.join(SEALED_DIR)).unwrap().all(|e| {
            !e.unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".writing")
        });
        assert!(no_tmp, "no temporary left");

        // The local copy lost: the secrets copy stands, and is put back
        std::fs::remove_file(local_path(&dir)).unwrap();
        assert!(is_uploading(&dir, &keys, id));
        assert!(reconcile(&dir, &keys, id).unwrap());
        assert_eq!(std::fs::read(local_path(&dir)).unwrap(), local);

        // The secrets copy lost: contradictory, not uploading, and nothing
        // writes it back from the local copy: not a start, not `set`
        let durable = std::fs::read(durable_path(&keys, id)).unwrap();
        std::fs::remove_file(durable_path(&keys, id)).unwrap();
        assert!(matches!(state(&dir, &keys, id), Marker::Contradictory(_)));
        assert!(!is_uploading(&dir, &keys, id));
        assert!(
            !reconcile(&dir, &keys, id).unwrap(),
            "never from the local copy"
        );
        assert_eq!(finish(&dir, &keys, id).unwrap(), None);
        assert!(set(&dir, &keys, id, UploadingBy::OptIn).is_err());
        assert!(!is_uploading(&dir, &keys, id), "removing one copy stops it");
        std::fs::write(durable_path(&keys, id), &durable).unwrap();
        assert!(is_uploading(&dir, &keys, id));

        // The copies differ
        let other = Uploading {
            since: u.since + chrono::Duration::seconds(1),
            ..u
        };
        std::fs::write(durable_path(&keys, id), serde_json::to_vec(&other).unwrap()).unwrap();
        assert!(matches!(state(&dir, &keys, id), Marker::Contradictory(_)));
        assert!(!is_uploading(&dir, &keys, id));
        assert!(
            set(&dir, &keys, id, UploadingBy::OptIn).is_err(),
            "never overwritten"
        );

        // Unparseable, either copy
        std::fs::write(durable_path(&keys, id), b"{ not json").unwrap();
        assert!(matches!(state(&dir, &keys, id), Marker::Unreadable(_)));
        assert!(!is_uploading(&dir, &keys, id));
        std::fs::write(durable_path(&keys, id), &durable).unwrap();
        assert!(is_uploading(&dir, &keys, id));
        std::fs::write(local_path(&dir), b"").unwrap();
        assert!(!is_uploading(&dir, &keys, id));

        // Another agent's marker
        let stranger = Uploading {
            agent_id: AgentId::new(),
            ..u
        };
        let bytes = serde_json::to_vec(&stranger).unwrap();
        std::fs::write(local_path(&dir), &bytes).unwrap();
        std::fs::write(durable_path(&keys, id), &bytes).unwrap();
        assert!(matches!(state(&dir, &keys, id), Marker::Unreadable(_)));
        assert!(!is_uploading(&dir, &keys, id));

        // Unreadable as a file (a directory in its place)
        std::fs::remove_file(local_path(&dir)).unwrap();
        std::fs::create_dir(local_path(&dir)).unwrap();
        std::fs::write(durable_path(&keys, id), &durable).unwrap();
        assert!(matches!(state(&dir, &keys, id), Marker::Unreadable(_)));
        assert!(!is_uploading(&dir, &keys, id));
    }

    /// Marked only with the old chain marker, uploaded before, and a yes
    /// in force; one agent's state never another's; and never again once
    /// the old marker is gone.
    #[tokio::test]
    async fn the_migration_marks_only_agents_that_uploaded() {
        let root = crate::sealed::TestDir::new("uploading-migrate");
        let state_dir = root.join("state");
        let (all, keys, all_dir) = enrolled_local(&root).await;
        old_marker(&all_dir);
        sent_one(&all_dir);
        ledger(&all_dir, true);
        // Each lacking one condition
        let (no_sent, _, d) = enrolled_local(&root).await;
        old_marker(&d);
        ledger(&d, true);
        let (declined, _, d) = enrolled_local(&root).await;
        old_marker(&d);
        sent_one(&d);
        ledger(&d, false);
        let (no_ledger, _, d) = enrolled_local(&root).await;
        old_marker(&d);
        sent_one(&d);
        // Chained by this runner (no old marker): never by evidence
        let (new_marker, _, d) = enrolled_local(&root).await;
        sent_one(&d);
        ledger(&d, true);
        let (unenrolled, _, d) = enrolled_local(&root).await;
        sent_one(&d);
        ledger(&d, true);
        std::fs::remove_file(d.join(SEALED_DIR).join(CHAINED)).unwrap();
        let (unreadable_ledger, _, d) = enrolled_local(&root).await;
        old_marker(&d);
        sent_one(&d);
        std::fs::write(
            d.join(crate::consent::sealed_state::ledger::LEDGER_FILE),
            b"{",
        )
        .unwrap();
        // A yes, re-opened by the operator and not answered again
        let (reopened, _, d) = enrolled_local(&root).await;
        old_marker(&d);
        sent_one(&d);
        ledger(&d, true);
        {
            let mut l = SealLedger::load_blocking(&d).unwrap();
            l.reopen("n", Utc::now()).unwrap();
            std::fs::write(
                d.join(crate::consent::sealed_state::ledger::LEDGER_FILE),
                serde_json::to_vec(&l).unwrap(),
            )
            .unwrap();
        }

        for (id, why) in [
            (no_sent, NotMigrated::NeverUploaded),
            (declined, NotMigrated::NotOptedIn),
            (no_ledger, NotMigrated::NotOptedIn),
            (new_marker, NotMigrated::NoOldMarker),
            (unenrolled, NotMigrated::NoOldMarker),
            (unreadable_ledger, NotMigrated::NotOptedIn),
            (reopened, NotMigrated::NotOptedIn),
        ] {
            let dir = state_dir.join(id.to_string());
            assert_eq!(migrate(&dir, &keys, id).await.unwrap(), Err(why));
            assert_eq!(state(&dir, &keys, id), Marker::Absent, "{why:?}");
        }
        migrate_markers(&state_dir, &keys).await;
        let u = match state(&all_dir, &keys, all) {
            Marker::Uploading(u) => u,
            other => panic!("{other:?}"),
        };
        assert_eq!(u.by, UploadingBy::Migration);
        for id in [
            no_sent,
            declined,
            no_ledger,
            new_marker,
            unenrolled,
            unreadable_ledger,
            reopened,
        ] {
            let dir = state_dir.join(id.to_string());
            assert_eq!(state(&dir, &keys, id), Marker::Absent);
        }
        // Again: nothing more, nothing moved
        migrate_markers(&state_dir, &keys).await;
        assert_eq!(state(&all_dir, &keys, all), Marker::Uploading(u));
        assert_eq!(
            migrate(&all_dir, &keys, all).await.unwrap(),
            Err(NotMigrated::NoOldMarker)
        );
    }

    /// The migration can't recur: with the old marker gone, an agent whose
    /// marker copies are both removed is never marked again from its
    /// `sent/` and its ledger, at any later start; and no record of the
    /// migration exists to delete.
    #[tokio::test]
    async fn the_migration_never_recurs_for_an_agent() {
        let root = crate::sealed::TestDir::new("uploading-once");
        let (id, keys, dir) = enrolled_local(&root).await;
        let state_dir = root.join("state");
        old_marker(&dir);
        sent_one(&dir);
        ledger(&dir, true);
        start_pass(&state_dir, &keys, &Alerter::off()).await;
        assert!(is_uploading(&dir, &keys, id));
        assert!(!dir.join(SEALED_DIR).join("enrolled").exists());
        assert!(!root.join("sealed-uploading-migrated").exists());
        // Both copies removed: not uploading, for good
        std::fs::remove_file(local_path(&dir)).unwrap();
        std::fs::remove_file(durable_path(&keys, id)).unwrap();
        for _ in 0..2 {
            start_pass(&state_dir, &keys, &Alerter::off()).await;
            assert_eq!(state(&dir, &keys, id), Marker::Absent);
            assert!(!is_uploading(&dir, &keys, id));
        }
        // A migration cut after the uploading marker, before the old
        // chain marker moved: finished by the next start, the same marker
        let (cut, _, cut_dir) = enrolled_local(&root).await;
        old_marker(&cut_dir);
        sent_one(&cut_dir);
        ledger(&cut_dir, true);
        let u = migrate(&cut_dir, &keys, cut).await.unwrap().unwrap();
        assert!(cut_dir.join(SEALED_DIR).join("enrolled").exists());
        start_pass(&state_dir, &keys, &Alerter::off()).await;
        assert_eq!(state(&cut_dir, &keys, cut), Marker::Uploading(u));
        assert!(crate::sealed::is_enrolled(&cut_dir));
        assert!(!cut_dir.join(SEALED_DIR).join("enrolled").exists());
    }

    /// A forget that moved only a state key (the agent had no marker) is a
    /// forget all the same: the agent, put back, is passed over.
    #[tokio::test]
    async fn a_forgotten_state_key_alone_counts_as_a_forget() {
        let root = crate::sealed::TestDir::new("uploading-forgot-key");
        let (id, keys, dir) = enrolled_local(&root).await;
        assert_eq!(forgotten_at(&keys, id), None);
        let stamp = "20261110T090000000Z";
        std::fs::write(
            keys.agent_dir(id)
                .join(crate::sealed::keys::FORGOTTEN_STATE_KEY.name(stamp)),
            b"{}",
        )
        .unwrap();
        assert_eq!(
            forgotten_at(&keys, id),
            Some("2026-11-10T09:00:00Z".parse().unwrap())
        );
        old_marker(&dir);
        sent_one(&dir);
        ledger(&dir, true);
        assert_eq!(
            migrate(&dir, &keys, id).await.unwrap(),
            Err(NotMigrated::Forgotten)
        );
    }

    /// Has uploaded: a copy naming the agent in either place, whatever the
    /// other says; neither copy, or only one naming another agent, not.
    #[tokio::test]
    async fn has_uploaded_is_any_copy_naming_the_agent() {
        let root = crate::sealed::TestDir::new("uploading-has");
        let (id, keys, dir) = enrolled_local(&root).await;
        assert!(!has_uploaded(&dir, &keys, id));
        let u = set(&dir, &keys, id, UploadingBy::OptIn).unwrap();
        assert!(has_uploaded(&dir, &keys, id));
        std::fs::remove_file(durable_path(&keys, id)).unwrap();
        assert!(has_uploaded(&dir, &keys, id), "contradictory, still has");
        std::fs::remove_file(local_path(&dir)).unwrap();
        let stranger = Uploading {
            agent_id: AgentId::new(),
            ..u
        };
        std::fs::write(
            durable_path(&keys, id),
            serde_json::to_vec(&stranger).unwrap(),
        )
        .unwrap();
        assert!(!has_uploaded(&dir, &keys, id), "another agent's");
    }

    /// Two writers at once (the runner's start and a `sealed-*` command):
    /// under the marker's lock, both copies always end the same.
    #[tokio::test]
    async fn concurrent_setters_never_leave_the_copies_differing() {
        let root = crate::sealed::TestDir::new("uploading-race");
        let (id, keys, dir) = enrolled_local(&root).await;
        for _ in 0..20 {
            let _ = std::fs::remove_file(local_path(&dir));
            let _ = std::fs::remove_file(durable_path(&keys, id));
            let handles: Vec<_> = (0..4)
                .map(|n| {
                    let (dir, keys) = (dir.clone(), keys.clone());
                    std::thread::spawn(move || {
                        let by = if n % 2 == 0 {
                            UploadingBy::OptIn
                        } else {
                            UploadingBy::Migration
                        };
                        set(&dir, &keys, id, by).unwrap()
                    })
                })
                .collect();
            let got: Vec<Uploading> = handles.into_iter().map(|h| h.join().unwrap()).collect();
            let Marker::Uploading(u) = state(&dir, &keys, id) else {
                panic!("{:?}", state(&dir, &keys, id))
            };
            assert!(
                got.iter().all(|g| *g == u),
                "every setter saw the one marker"
            );
        }
    }

    /// A `set` cut at either step is finished at the start with the same
    /// contents, or never happened; a local copy with no secrets copy and
    /// no intent of this code (a secrets copy removed) stays in doubt. A
    /// marker still in doubt after the pass is one alert, naming the
    /// agent, unless its operator stopped its uploads.
    #[tokio::test]
    async fn the_start_pass_finishes_a_half_written_marker_and_alerts_the_rest() {
        use crate::alerts::testing::{Recorder, recording};
        let root = crate::sealed::TestDir::new("uploading-pass");
        let state_dir = root.join("state");
        // Cut after the intent, before the local copy: nothing is marked
        let (early, keys, early_dir) = enrolled_local(&root).await;
        crate::sealed::crash::at("uploading:intent");
        assert!(set(&early_dir, &keys, early, UploadingBy::OptIn).is_err());
        assert_eq!(state(&early_dir, &keys, early), Marker::Absent);
        // Cut after the local copy, before the commit
        let (half, _, half_dir) = enrolled_local(&root).await;
        crate::sealed::crash::at("uploading:local");
        assert!(set(&half_dir, &keys, half, UploadingBy::OptIn).is_err());
        assert!(matches!(
            state(&half_dir, &keys, half),
            Marker::Contradictory(_)
        ));
        let bytes = std::fs::read(local_path(&half_dir)).unwrap();
        let u: Uploading = serde_json::from_slice(&bytes).unwrap();
        // The secrets copy removed after a whole set: no intent, in doubt
        let (removed, _, removed_dir) = enrolled_local(&root).await;
        set(&removed_dir, &keys, removed, UploadingBy::OptIn).unwrap();
        std::fs::remove_file(durable_path(&keys, removed)).unwrap();
        // The same, with an intent that says something else
        let (other_intent, _, other_dir) = enrolled_local(&root).await;
        let was = set(&other_dir, &keys, other_intent, UploadingBy::OptIn).unwrap();
        std::fs::remove_file(durable_path(&keys, other_intent)).unwrap();
        std::fs::write(
            setting_path(&keys, other_intent),
            serde_json::to_vec(&Uploading {
                since: was.since + chrono::Duration::seconds(1),
                ..was
            })
            .unwrap(),
        )
        .unwrap();
        // One whose copies differ: never rewritten, alerted
        let (bad, _, bad_dir) = enrolled_local(&root).await;
        set(&bad_dir, &keys, bad, UploadingBy::OptIn).unwrap();
        let other = Uploading {
            agent_id: bad,
            since: Utc::now() + chrono::Duration::seconds(5),
            by: UploadingBy::Migration,
        };
        std::fs::write(
            durable_path(&keys, bad),
            serde_json::to_vec(&other).unwrap(),
        )
        .unwrap();
        let before_bad = (
            std::fs::read(local_path(&bad_dir)).unwrap(),
            std::fs::read(durable_path(&keys, bad)).unwrap(),
        );
        // In doubt and stopped by its operator: the stop settles it from
        // the copy that can be read (it uploaded, and is stopped)
        let (settled, _, settled_dir) = enrolled_local(&root).await;
        let was_settled = set(&settled_dir, &keys, settled, UploadingBy::OptIn).unwrap();
        std::fs::remove_file(durable_path(&keys, settled)).unwrap();
        let done = crate::sealed::stop::stop(&state_dir, &keys, settled, "asked")
            .await
            .unwrap();
        assert!(done.later.is_empty(), "{:?}", done.later);
        assert_eq!(
            state(&settled_dir, &keys, settled),
            Marker::Uploading(was_settled)
        );
        assert!(!crate::sealed::uploads(&settled_dir, &keys, settled));
        // Stopped with no copy readable: still in doubt, but stopped by
        // its operator's own command, so not alerted
        let (stopped, _, stopped_dir) = enrolled_local(&root).await;
        set(&stopped_dir, &keys, stopped, UploadingBy::OptIn).unwrap();
        std::fs::write(durable_path(&keys, stopped), b"{").unwrap();
        std::fs::write(local_path(&stopped_dir), b"{").unwrap();
        let done = crate::sealed::stop::stop(&state_dir, &keys, stopped, "asked")
            .await
            .unwrap();
        assert_eq!(done.later.len(), 1, "{:?}", done.later);
        assert!(!crate::sealed::uploads(&stopped_dir, &keys, stopped));
        // And one never marked: left alone, not alerted
        let (_none, _, _) = enrolled_local(&root).await;

        let sent = Recorder::default();
        let alerts = recording(root.path(), &sent);
        let doubt = start_pass(&state_dir, &keys, &alerts).await;
        alerts.flush(std::time::Duration::from_secs(5)).await;
        assert_eq!(state(&early_dir, &keys, early), Marker::Absent);
        assert_eq!(
            state(&half_dir, &keys, half),
            Marker::Uploading(u),
            "finished"
        );
        assert_eq!(std::fs::read(durable_path(&keys, half)).unwrap(), bytes);
        assert!(!setting_path(&keys, half).exists(), "the intent became it");
        let mut expect = vec![removed, other_intent, bad, stopped];
        expect.sort();
        let mut got = doubt.clone();
        got.sort();
        assert_eq!(got, expect);
        for (id, dir) in [(removed, &removed_dir), (other_intent, &other_dir)] {
            assert!(!is_uploading(dir, &keys, id), "not put back by a start");
            assert!(!durable_path(&keys, id).exists());
        }
        assert_eq!(
            (
                std::fs::read(local_path(&bad_dir)).unwrap(),
                std::fs::read(durable_path(&keys, bad)).unwrap()
            ),
            before_bad,
            "a marker in doubt is never rewritten"
        );
        let mails = sent.sent();
        assert_eq!(mails.len(), 1, "one summary");
        for id in [removed, other_intent, bad] {
            assert!(mails[0].contains(&id.to_string()), "{}", mails[0]);
        }
        for id in [half, early, stopped, settled] {
            assert!(!mails[0].contains(&id.to_string()), "{}", mails[0]);
        }
        // The operator's opt-in (on the agent's yes) ends the doubt: the
        // copies set aside, a new marker
        let new = set_replacing_doubt(&bad_dir, &keys, bad, UploadingBy::OptIn).unwrap();
        assert_eq!(state(&bad_dir, &keys, bad), Marker::Uploading(new));
        let aside = |dir: &Path, name: &str| {
            std::fs::read_dir(dir)
                .unwrap()
                .filter(|e| {
                    e.as_ref()
                        .unwrap()
                        .file_name()
                        .to_string_lossy()
                        .starts_with(&format!("{name}.doubt."))
                })
                .count()
        };
        assert_eq!(aside(&bad_dir.join(SEALED_DIR), UPLOADING), 1, "kept");
        assert_eq!(aside(&keys.agent_dir(bad), DURABLE), 1, "kept");
        let again = set_replacing_doubt(&bad_dir, &keys, bad, UploadingBy::OptIn).unwrap();
        assert_eq!(again, new, "a sound marker is left as it is");
    }

    /// The chain's marker moved from `sealed/enrolled` to `sealed/chained`
    /// at every point a start can be cut: an agent is never left with
    /// neither, and an agent that uploaded ends chained and uploading.
    #[tokio::test]
    async fn the_old_marker_moves_before_the_uploading_migration_at_every_crash_point() {
        let root = crate::sealed::TestDir::new("uploading-marker-move");
        let state_dir = root.join("state");
        let mut agents = Vec::new();
        // (old marker, new marker, uploaded and said yes)
        for (old, new, yes) in [
            (true, false, true),  // as the betas are today: old marker, all sent, yes
            (true, true, true),   // cut after the new marker was written
            (true, false, false), // chained, never uploaded
        ] {
            let (id, keys, dir) = enrolled_local(&root).await;
            let sealed = dir.join(SEALED_DIR);
            if old {
                std::fs::write(sealed.join("enrolled"), b"").unwrap();
            }
            if !new {
                std::fs::remove_file(sealed.join(CHAINED)).unwrap();
            }
            if yes {
                // Everything it had uploaded, as the betas have
                let seal = AgentSeal::new(&dir, id);
                for item in seal.outbox().await.unwrap() {
                    seal.mark_sent(item.ord).await.unwrap();
                }
                ledger(&dir, true);
            }
            agents.push((id, keys, dir, yes));
        }
        let keys = agents[0].1.clone();
        let sent_before: Vec<usize> = {
            let mut v = Vec::new();
            for (id, _, dir, _) in &agents {
                v.push(AgentSeal::new(dir, *id).sent().await.unwrap().len());
            }
            v
        };
        let either = |dir: &Path| {
            dir.join(SEALED_DIR).join(CHAINED).exists()
                || dir.join(SEALED_DIR).join("enrolled").exists()
        };
        // An old marker alone isn't read as chained (nor uploading) before
        // the pass
        assert!(!crate::sealed::is_enrolled(&agents[0].2));
        assert!(!is_uploading(&agents[0].2, &keys, agents[0].0));
        // The uploading marker is written while the old chain marker is
        // still there, then the chain marker moves: a cut between the two
        // leaves the old marker, and the next pass finishes
        let moved = migrate_markers(&state_dir, &keys).await;
        let mut expect: Vec<AgentId> = agents.iter().map(|a| a.0).collect();
        expect.sort();
        assert_eq!(moved, expect);
        for (id, _, dir, yes) in &agents {
            assert!(either(dir), "{id}: never neither marker");
            assert!(crate::sealed::is_enrolled(dir));
            assert!(!dir.join(SEALED_DIR).join("enrolled").exists());
            assert_eq!(is_uploading(dir, &keys, *id), *yes, "{id}");
        }
        // The rest of the pass, and again: idempotent
        for _ in 0..2 {
            start_pass(&state_dir, &keys, &crate::alerts::Alerter::off()).await;
            for (id, _, dir, yes) in &agents {
                assert!(crate::sealed::is_enrolled(dir));
                assert!(!dir.join(SEALED_DIR).join("enrolled").exists());
                assert_eq!(is_uploading(dir, &keys, *id), *yes, "{id}");
            }
        }
        assert!(migrate_markers(&state_dir, &keys).await.is_empty());
        // An older runner, which reads the literal `sealed/enrolled` as
        // "seal and upload", finds nothing to act on
        for (_, _, dir, _) in &agents {
            assert!(!dir.join("sealed/enrolled").exists());
        }
        // Nothing queued again: what was uploaded stays sent, nothing waits
        for ((id, _, dir, yes), before) in agents.iter().zip(sent_before) {
            let seal = AgentSeal::new(dir, *id);
            assert_eq!(seal.sent().await.unwrap().len(), before);
            if *yes {
                assert!(
                    seal.outbox().await.unwrap().is_empty(),
                    "nothing to upload twice"
                );
            }
        }
    }
}
