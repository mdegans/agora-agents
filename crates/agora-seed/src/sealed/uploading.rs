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
//! | present | absent | not uploading, contradictory: a [`set`] cut between its writes ([`finish`] completes it at the next start) |
//! | absent | present | uploading: the local copy was lost with `sealed/` or the directory ([`reconcile`] puts it back) |
//! | either unreadable, or naming another agent | | not uploading |
//!
//! So [`set`] writes the local copy first and the secrets copy last: the
//! secrets copy is the one whose presence commits the fact, as for a stop.
//! A contradictory or unreadable marker is logged at error
//! (`sealed_state_uploading_unreadable`) every time it is read, never
//! overwritten here, and alerted once a run ([`start_pass`]).
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
//! [`migrate_all`]: the one-time migration of agents that uploaded before
//! the marker existed.

use std::path::{Path, PathBuf};

use agora_agentkit::ids::AgentId;
use agora_agentkit::state::storage::{AgentSeal, CHAINED, MarkerMigration, SEALED_DIR};
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
/// `<secrets>/<id>/sealed_uploading.lock`: the marker's writers' lock.
pub const LOCK: &str = "sealed_uploading.lock";

/// What set the marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UploadingBy {
    /// [`migrate_all`]: the agent uploaded before the marker existed.
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
fn lock(keys: &KeyStore, id: AgentId) -> std::io::Result<std::fs::File> {
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

/// Mark `id` as uploading: the local copy, then the secrets copy (which
/// commits it), under the marker's lock. Already uploading: nothing
/// written, the marker returned. A marker in doubt is never overwritten,
/// but one only half written (local only) is finished.
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
    super::write_private(&local_path(agent_dir), &bytes)?;
    super::write_private(&durable_path(keys, id), &bytes)?;
    Ok(marker)
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

/// A [`set`] cut between its two writes (the local copy whole, the secrets
/// copy absent): the secrets copy written with the same contents, under
/// the marker's lock. `Ok(Some)`: finished. Anything else is left as it is.
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
    super::write_private(&durable_path(keys, id), &serde_json::to_vec_pretty(&u)?)?;
    tracing::warn!(
        event_type = "sealed_state_uploading_finished",
        agent_id = %id,
        "a half-written uploading marker (no secrets copy) was finished with the same contents"
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
    /// No `sealed/chained` (or it can't be checked).
    NotEnrolled,
    /// A marker is there already (or one in doubt, never overwritten).
    Marked,
    /// Nothing in `sealed/sent/`: it never uploaded.
    NeverUploaded,
    /// Its sealed-state ledger doesn't say yes (or can't be read).
    NotOptedIn,
}

/// One agent of [`migrate_all`]: marked uploading only if it is enrolled,
/// has no marker, has uploaded before (a non-empty `sealed/sent/`) and its
/// sealed-state ledger says yes. Reads directory entries and the ledger's
/// answer only; decides and writes under the marker's lock.
pub fn migrate(
    agent_dir: &Path,
    keys: &KeyStore,
    id: AgentId,
) -> std::io::Result<Result<Uploading, NotMigrated>> {
    let sealed = agent_dir.join(SEALED_DIR);
    // Strictly: an enrolment that can't be checked is not migrated
    if !std::fs::exists(sealed.join(CHAINED)).unwrap_or(false) {
        return Ok(Err(NotMigrated::NotEnrolled));
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
        Ok(ledger) if ledger.opted_in() => {}
        Ok(_) => return Ok(Err(NotMigrated::NotOptedIn)),
        Err(e) => {
            tracing::warn!(
                agent_id = %id,
                error = %e,
                "sealed-state ledger unreadable; the agent is not marked uploading"
            );
            return Ok(Err(NotMigrated::NotOptedIn));
        }
    }
    set_locked(agent_dir, keys, id, UploadingBy::Migration).map(Ok)
}

/// The one-time migration to the uploading marker, run at every start
/// (idempotent: a marked agent is left alone). Before the marker existed,
/// an enrolled agent uploaded; after it, an enrolled agent with no marker
/// does not. So an agent that did upload (enrolled, its ledger yes,
/// something in `sealed/sent/`) is marked, one `sealed_state_uploading_migrated`
/// event each; no other agent ever is. The agents marked.
pub fn migrate_all(state_dir: &Path, keys: &KeyStore) -> Vec<AgentId> {
    let mut migrated = Vec::new();
    for id in all_ids(state_dir, keys) {
        match migrate(&state_dir.join(id.to_string()), keys, id) {
            Ok(Ok(_)) => {
                tracing::warn!(
                    event_type = "sealed_state_uploading_migrated",
                    agent_id = %id,
                    "an agent that uploaded before the uploading marker existed is marked uploading"
                );
                migrated.push(id);
            }
            Ok(Err(_)) => {}
            Err(e) => tracing::error!(
                event_type = "sealed_state_uploading_migration_failed",
                agent_id = %id,
                error = %e,
                "the uploading migration couldn't be done for an agent; it stays not uploading"
            ),
        }
    }
    migrated
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
/// agent with the old marker is moved to the new one by agentkit's
/// [`AgentSeal::migrate_marker`] (the new marker written and made durable
/// before the old is removed: a crash leaves at least one, and the next
/// start finishes), one `sealed_state_marker_migrated` event each. The
/// agents moved. Before anything reads the marker.
pub async fn migrate_markers(state_dir: &Path) -> Vec<AgentId> {
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
        let seal = AgentSeal::new(&state_dir.join(id.to_string()), id);
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

/// The start's pass over the markers, before states load: the chain's
/// marker moved to its new name ([`migrate_markers`]), then
/// [`reconcile_all`] (a lost local copy back, a half-written marker
/// finished), then [`migrate_all`] (which needs the new name); then any
/// agent whose marker is still in doubt is one summary alert
/// ([`AlertKind::SealedStateUploadingInDoubt`]): a consenting agent
/// silently not uploading must be seen. The agents in doubt.
pub async fn start_pass(state_dir: &Path, keys: &KeyStore, alerts: &Alerter) -> Vec<AgentId> {
    migrate_markers(state_dir).await;
    reconcile_all(state_dir, keys);
    migrate_all(state_dir, keys);
    let doubt = in_doubt(state_dir, keys);
    if !doubt.is_empty() {
        let ids: Vec<String> = doubt.iter().map(|(id, _)| id.to_string()).collect();
        tracing::error!(
            event_type = "sealed_state_uploading_in_doubt",
            agents = doubt.len(),
            ids = %ids.join(","),
            "uploading markers in doubt: those agents upload nothing until an operator repairs them"
        );
        let mut alert = Alert::new(
            AlertKind::SealedStateUploadingInDoubt,
            format!(
                "{} agent(s) with an uploading marker in doubt: nothing of theirs is uploaded",
                doubt.len()
            ),
        )
        .detail("agents", ids.join(", "));
        for (id, why) in doubt.iter().take(10) {
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

        // The secrets copy lost: contradictory, not uploading
        let durable = std::fs::read(durable_path(&keys, id)).unwrap();
        std::fs::remove_file(durable_path(&keys, id)).unwrap();
        assert!(matches!(state(&dir, &keys, id), Marker::Contradictory(_)));
        assert!(!is_uploading(&dir, &keys, id));
        assert!(
            !reconcile(&dir, &keys, id).unwrap(),
            "never from the local copy"
        );
        // ...which `set` finishes, as a cut `set` would be
        assert_eq!(set(&dir, &keys, id, UploadingBy::OptIn).unwrap(), u);
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

    /// Marked only when enrolled, uploaded before, and the ledger says yes;
    /// idempotent; one agent's state never another's.
    #[tokio::test]
    async fn the_migration_marks_only_agents_that_uploaded() {
        let root = crate::sealed::TestDir::new("uploading-migrate");
        let state_dir = root.join("state");
        let (all, keys, all_dir) = enrolled_local(&root).await;
        sent_one(&all_dir);
        ledger(&all_dir, true);
        // Each lacking one condition
        let (no_sent, _, d) = enrolled_local(&root).await;
        ledger(&d, true);
        let (declined, _, d) = enrolled_local(&root).await;
        sent_one(&d);
        ledger(&d, false);
        let (no_ledger, _, d) = enrolled_local(&root).await;
        sent_one(&d);
        let (unenrolled, _, d) = enrolled_local(&root).await;
        sent_one(&d);
        ledger(&d, true);
        std::fs::remove_file(d.join(SEALED_DIR).join(CHAINED)).unwrap();
        let (unreadable_ledger, _, d) = enrolled_local(&root).await;
        sent_one(&d);
        std::fs::write(
            d.join(crate::consent::sealed_state::ledger::LEDGER_FILE),
            b"{",
        )
        .unwrap();

        assert_eq!(migrate_all(&state_dir, &keys), vec![all]);
        let u = match state(&all_dir, &keys, all) {
            Marker::Uploading(u) => u,
            other => panic!("{other:?}"),
        };
        assert_eq!(u.by, UploadingBy::Migration);
        for (id, why) in [
            (no_sent, NotMigrated::NeverUploaded),
            (declined, NotMigrated::NotOptedIn),
            (no_ledger, NotMigrated::NotOptedIn),
            (unenrolled, NotMigrated::NotEnrolled),
            (unreadable_ledger, NotMigrated::NotOptedIn),
        ] {
            let dir = state_dir.join(id.to_string());
            assert_eq!(state(&dir, &keys, id), Marker::Absent, "{why:?}");
            assert_eq!(migrate(&dir, &keys, id).unwrap(), Err(why));
        }
        // Again: nothing more, nothing moved
        assert!(migrate_all(&state_dir, &keys).is_empty());
        assert_eq!(state(&all_dir, &keys, all), Marker::Uploading(u));
        assert_eq!(
            migrate(&all_dir, &keys, all).unwrap(),
            Err(NotMigrated::Marked)
        );
        // A marker in doubt is never touched by the migration
        std::fs::remove_file(durable_path(&keys, all)).unwrap();
        assert!(migrate_all(&state_dir, &keys).is_empty());
        assert!(matches!(
            state(&all_dir, &keys, all),
            Marker::Contradictory(_)
        ));
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

    /// A `set` cut between its writes (the local copy only) is finished at
    /// the start with the same contents; a marker still in doubt after the
    /// pass is one alert, naming the agent.
    #[tokio::test]
    async fn the_start_pass_finishes_a_half_written_marker_and_alerts_the_rest() {
        use crate::alerts::testing::{Recorder, recording};
        let root = crate::sealed::TestDir::new("uploading-pass");
        let state_dir = root.join("state");
        let (half, keys, half_dir) = enrolled_local(&root).await;
        let u = Uploading {
            agent_id: half,
            since: Utc::now(),
            by: UploadingBy::OptIn,
        };
        let bytes = serde_json::to_vec_pretty(&u).unwrap();
        crate::sealed::write_private(&local_path(&half_dir), &bytes).unwrap();
        assert!(matches!(
            state(&half_dir, &keys, half),
            Marker::Contradictory(_)
        ));
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
        // And one never marked: left alone, not alerted
        let (_none, _, _) = enrolled_local(&root).await;

        let sent = Recorder::default();
        let alerts = recording(root.path(), &sent);
        let doubt = start_pass(&state_dir, &keys, &alerts).await;
        alerts.flush(std::time::Duration::from_secs(5)).await;
        assert_eq!(
            state(&half_dir, &keys, half),
            Marker::Uploading(u),
            "finished"
        );
        assert_eq!(std::fs::read(durable_path(&keys, half)).unwrap(), bytes);
        assert_eq!(doubt, vec![bad]);
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
        assert!(mails[0].contains(&bad.to_string()), "{}", mails[0]);
        assert!(!mails[0].contains(&half.to_string()));
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
            (false, true, true),  // cut before the uploading migration
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
        // the move
        assert!(!crate::sealed::is_enrolled(&agents[0].2));
        assert_eq!(
            migrate(&agents[0].2, &keys, agents[0].0).unwrap(),
            Err(NotMigrated::NotEnrolled)
        );
        // Cut after the marker move, before anything else
        let moved = migrate_markers(&state_dir).await;
        let mut expect: Vec<AgentId> = [0, 1, 3].iter().map(|&i| agents[i].0).collect();
        expect.sort();
        assert_eq!(moved, expect);
        for (id, _, dir, _) in &agents {
            assert!(either(dir), "{id}: never neither marker");
            assert!(crate::sealed::is_enrolled(dir));
            assert!(!dir.join(SEALED_DIR).join("enrolled").exists());
            assert!(
                !is_uploading(dir, &keys, *id),
                "not before its own migration"
            );
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
        assert!(migrate_markers(&state_dir).await.is_empty());
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
