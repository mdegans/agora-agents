//! `sealed/uploading`: the agent's chain is uploaded to Agora. A positive
//! fact of its own, apart from `sealed/enrolled` (the chain is kept, which
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
//! | present | absent | not uploading, contradictory |
//! | absent | present | uploading: the local copy was lost with `sealed/` or the directory ([`reconcile`] puts it back) |
//! | either unreadable, or naming another agent | | not uploading |
//!
//! So [`set`] writes the local copy first and the secrets copy last: a
//! crash between them leaves "present, absent", which reads as not
//! uploading until `set` is run again. The secrets copy is the one whose
//! presence commits the fact, as for a stop. A contradictory or unreadable
//! marker is logged at error (`sealed_state_uploading_unreadable`) every
//! time it is read, and never overwritten here.
//!
//! [`migrate_all`]: the one-time migration of agents that uploaded before
//! the marker existed.

use std::path::{Path, PathBuf};

use agora_agentkit::ids::AgentId;
use agora_agentkit::state::storage::SEALED_DIR;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::KeyStore;
use crate::consent::offer::Stored as _;
use crate::consent::sealed_state::ledger::SealLedger;

/// `sealed/uploading`.
pub const UPLOADING: &str = "uploading";
/// `<secrets>/<id>/sealed_uploading`: the durable copy.
pub const DURABLE: &str = "sealed_uploading";

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

/// Mark `id` as uploading: the local copy, then the secrets copy (which
/// commits it). Already uploading: nothing written, the marker returned.
/// A marker in doubt is never overwritten: an error.
pub fn set(
    agent_dir: &Path,
    keys: &KeyStore,
    id: AgentId,
    by: UploadingBy,
) -> std::io::Result<Uploading> {
    let marker = match state(agent_dir, keys, id) {
        Marker::Uploading(u) => {
            // The local copy may be the lost one
            reconcile(agent_dir, keys, id)?;
            return Ok(u);
        }
        Marker::Absent => Uploading {
            agent_id: id,
            since: Utc::now(),
            by,
        },
        Marker::Contradictory(why) => {
            // Only a local copy: an earlier `set` cut before its secrets
            // copy. Finished with the same contents
            let durable_absent = matches!(read_copy(&durable_path(keys, id), id), Copy::Absent);
            match read_copy(&local_path(agent_dir), id) {
                Copy::Present(u) if durable_absent => u,
                _ => return Err(std::io::Error::other(why)),
            }
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

/// [`reconcile`] for every agent with a state key in the secrets tree and
/// a state directory here: at the start of a run, before states load.
pub fn reconcile_all(state_dir: &Path, keys: &KeyStore) {
    let ids = match keys.with_state_keys() {
        Ok(ids) => ids,
        Err(e) => {
            tracing::warn!(error = %e, "the secrets tree couldn't be listed for uploading markers");
            return;
        }
    };
    for id in ids {
        if let Err(e) = reconcile(&state_dir.join(id.to_string()), keys, id) {
            tracing::error!(agent_id = %id, error = %e, "the durable uploading marker couldn't be put back");
        }
    }
}

/// Why [`migrate`] left an agent as it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotMigrated {
    /// No `sealed/enrolled` (or it can't be checked).
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
/// answer only.
pub fn migrate(
    agent_dir: &Path,
    keys: &KeyStore,
    id: AgentId,
) -> std::io::Result<Result<Uploading, NotMigrated>> {
    let sealed = agent_dir.join(SEALED_DIR);
    // Strictly: an enrolment that can't be checked is not migrated
    if !std::fs::exists(sealed.join("enrolled")).unwrap_or(false) {
        return Ok(Err(NotMigrated::NotEnrolled));
    }
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
    set(agent_dir, keys, id, UploadingBy::Migration).map(Ok)
}

/// The one-time migration to the uploading marker, run at every start
/// (idempotent: a marked agent is left alone). Before the marker existed,
/// an enrolled agent uploaded; after it, an enrolled agent with no marker
/// does not. So an agent that did upload (enrolled, its ledger yes,
/// something in `sealed/sent/`) is marked, one `sealed_state_uploading_migrated`
/// event each; no other agent ever is. The agents marked.
pub fn migrate_all(state_dir: &Path, keys: &KeyStore) -> Vec<AgentId> {
    let entries = match std::fs::read_dir(state_dir) {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!(error = %e, "the state directory couldn't be listed for the uploading migration");
            return Vec::new();
        }
    };
    let mut ids: Vec<AgentId> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str()?.parse::<AgentId>().ok())
        .collect();
    ids.sort();
    let mut migrated = Vec::new();
    for id in ids {
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
        std::fs::remove_file(d.join(SEALED_DIR).join("enrolled")).unwrap();
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
}
