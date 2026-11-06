//! `agora-seed sealed-forget --agent <id> --note "…"`: set aside the state
//! key of an agent whose state is gone for good (it left, its secrets left
//! behind), so the start check stops refusing the run over it
//! ([`super::check::missing_states`]).
//!
//! No key or record is deleted: `<secrets>/<id>/state_key.json` becomes
//! `state_key.forgotten.<utc>.json`, the durable stop (if any)
//! `sealed_stopped.forgotten.<utc>`, and the note is kept beside them as
//! `state_key.forgotten.<utc>.note.json`. The memory-id key goes aside with
//! the state key, under the same `<utc>`
//! (`memory_id_key.forgotten.<utc>.json`), so putting the agent back
//! restores both together and its memory ids still resolve; a rerun after
//! a cut between the two moves reuses the memory-id key's stamp. A key set
//! aside is reused, never minted over, if the agent is ever asked for one
//! again ([`KeyStore::memory_id_keys_or_create`]). The signing keys stay. The one file deleted is the `memory` tool's search index
//! (`sealed/history_index.json`, [`super::history_index`]), a sidecar
//! rebuilt from the history whenever it is needed. Refused while the agent
//! has a `state.json`: a living agent's backups are stopped with
//! `sealed-stop`, never forgotten.

use std::path::Path;

use agora_agentkit::ids::AgentId;
use anyhow::Context;
use chrono::{DateTime, Utc};
use serde::Serialize;

use super::KeyStore;
use super::keys::{FORGOTTEN_MEMORY_ID_KEY_PREFIX, MEMORY_ID_KEY_FILE, STATE_KEY_FILE};

/// The note kept beside the forgotten key.
#[derive(Debug, Serialize)]
struct Forgotten<'a> {
    at: DateTime<Utc>,
    note: &'a str,
}

/// What `sealed-forget` set aside: file names only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Forgot {
    pub key: String,
    pub stop: Option<String>,
    /// The memory-id key's new name, if the agent had one.
    pub memory_id_key: Option<String>,
    /// The search index was there, and was deleted.
    pub index: bool,
}

/// The stamp of a memory-id key set aside with no state key of the same
/// stamp beside it: a forget cut between the two moves.
fn orphaned_stamp(dir: &Path) -> std::io::Result<Option<String>> {
    for entry in std::fs::read_dir(dir)? {
        let name = entry?.file_name();
        let Some(stamp) = name
            .to_str()
            .and_then(|n| n.strip_prefix(FORGOTTEN_MEMORY_ID_KEY_PREFIX))
            .and_then(|n| n.strip_suffix(".json"))
        else {
            continue;
        };
        if !std::fs::exists(dir.join(format!("state_key.forgotten.{stamp}.json")))? {
            return Ok(Some(stamp.to_string()));
        }
    }
    Ok(None)
}

pub fn forget(
    state_dir: &Path,
    keys: &KeyStore,
    id: AgentId,
    note: &str,
) -> anyhow::Result<Forgot> {
    let note = note.trim();
    anyhow::ensure!(!note.is_empty(), "--note is required: it says why");
    anyhow::ensure!(
        !std::fs::exists(state_dir.join(id.to_string()).join("state.json"))?,
        "agent {id} has a state here: its backups are stopped with `sealed-stop`, not forgotten"
    );
    let dir = keys.agent_dir(id);
    let key = dir.join(STATE_KEY_FILE);
    anyhow::ensure!(
        std::fs::exists(&key)?,
        "agent {id} has no state key in the secrets directory; nothing to forget"
    );
    // The disposable sidecar first: a failure here leaves the rest to a rerun
    let index = super::history_index::remove(&state_dir.join(id.to_string()))
        .context("deleting the search index")?;
    let at = Utc::now();
    // A run cut after setting the memory-id key aside left its stamp: this
    // one finishes it under the same, so the two keys still pair up
    let stamp = match orphaned_stamp(&dir)? {
        Some(stamp) => stamp,
        None => at.format("%Y%m%dT%H%M%S%3fZ").to_string(),
    };
    let key_to = format!("state_key.forgotten.{stamp}.json");
    let note_path = dir.join(format!("state_key.forgotten.{stamp}.note.json"));
    // The note first: a key set aside always says why. A cut run's note
    // is kept as it was
    if !std::fs::exists(&note_path)? {
        std::fs::write(
            &note_path,
            serde_json::to_vec_pretty(&Forgotten { at, note })?,
        )
        .context("writing the note")?;
    }
    // Before the state key, whose move finishes the forget
    let memory_id_key = dir.join(MEMORY_ID_KEY_FILE);
    let memory_id_key_to = if std::fs::exists(&memory_id_key)? {
        let to = format!("{FORGOTTEN_MEMORY_ID_KEY_PREFIX}{stamp}.json");
        std::fs::rename(&memory_id_key, dir.join(&to))
            .context("setting the memory-id key aside")?;
        Some(to)
    } else {
        None
    };
    std::fs::rename(&key, dir.join(&key_to)).context("setting the state key aside")?;
    let stop = dir.join(super::stop::DURABLE);
    let stop_to = if std::fs::exists(&stop)? {
        let to = format!("{}.forgotten.{stamp}", super::stop::DURABLE);
        std::fs::rename(&stop, dir.join(&to)).context("setting the stop aside")?;
        Some(to)
    } else {
        None
    };
    std::fs::File::open(&dir)?.sync_all()?;
    tracing::warn!(
        event_type = "sealed_state_forgotten",
        agent_id = %id,
        index_deleted = index,
        "the operator set aside a state key whose agent has no state here; no key deleted"
    );
    Ok(Forgot {
        key: key_to,
        stop: stop_to,
        memory_id_key: memory_id_key_to,
        index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sealed::check::tests::enrolled;

    #[tokio::test]
    async fn forgetting_sets_the_key_aside_and_never_a_living_agent() {
        let root = crate::sealed::TestDir::new("forget");
        let (id, keys, dir) = enrolled(&root).await;
        let state_dir = root.join("state");
        let e = forget(&state_dir, &keys, id, "it left")
            .unwrap_err()
            .to_string();
        assert!(e.contains("sealed-stop"), "{e}");
        std::fs::rename(&dir, root.join("gone")).unwrap();
        assert!(forget(&state_dir, &keys, id, " ").is_err(), "a note");
        assert_eq!(keys.with_state_keys().unwrap(), vec![id]);
        let done = forget(&state_dir, &keys, id, "it left").unwrap();
        assert!(keys.with_state_keys().unwrap().is_empty());
        let kept: Vec<String> = std::fs::read_dir(keys.agent_dir(id))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        assert!(kept.contains(&done.key), "{kept:?}");
        assert!(!done.index, "no index to delete");
        assert_eq!(done.memory_id_key, None, "no memory-id key to set aside");
        assert!(kept.iter().any(|n| n.ends_with(".note.json")), "{kept:?}");
        // The signing keys stay
        assert!(keys.signing(id).is_some());
        // The run no longer refuses over it
        let api = crate::sealed::api::fake::FakeApi::new();
        *api.chain.lock().unwrap() = Ok(vec![]);
        assert!(
            crate::sealed::check::missing_states(
                &state_dir,
                &keys,
                &api,
                &crate::alerts::Alerter::off()
            )
            .await
            .is_empty()
        );
    }

    /// Forgetting deletes the search index, the one sidecar it deletes, and
    /// sets the memory-id key aside with the state key, under one stamp.
    #[tokio::test]
    async fn forgetting_deletes_the_search_index_and_sets_the_memory_id_key_aside() {
        let root = crate::sealed::TestDir::new("forget-index");
        let (id, keys, dir) = enrolled(&root).await;
        let state_dir = root.join("state");
        // The state lost, sealed/ left behind
        std::fs::rename(dir.join("state.json"), root.join("gone.json")).unwrap();
        crate::sealed::history_index::write(&dir, b"{}").unwrap();
        let key = keys.memory_id_keys_or_create(id).unwrap();
        let done = forget(&state_dir, &keys, id, "it left").unwrap();
        assert!(done.index);
        assert_eq!(crate::sealed::history_index::read(&dir).unwrap(), None);
        assert!(
            std::fs::exists(dir.join("sealed/head.json")).unwrap(),
            "only the index"
        );
        assert!(keys.memory_id_keys(id).unwrap().is_none(), "set aside");
        let aside = done.memory_id_key.expect("set aside");
        let stamp = |name: &str, prefix: &str| {
            name.strip_prefix(prefix)
                .and_then(|n| n.strip_suffix(".json"))
                .unwrap()
                .to_string()
        };
        assert_eq!(
            stamp(&aside, "memory_id_key.forgotten."),
            stamp(&done.key, "state_key.forgotten."),
            "one stamp: restored together"
        );
        // Kept whole: renamed back, it is the same key
        let dir = keys.agent_dir(id);
        std::fs::rename(dir.join(&aside), dir.join(MEMORY_ID_KEY_FILE)).unwrap();
        assert_eq!(
            keys.memory_id_keys(id).unwrap().unwrap().current.expose(),
            key.current.expose()
        );
    }

    /// A forget cut after the memory-id key moved: the rerun sets the state
    /// key aside under the same stamp, keeping the first note.
    #[tokio::test]
    async fn a_cut_forget_finishes_under_the_same_stamp() {
        let root = crate::sealed::TestDir::new("forget-cut");
        let (id, keys, dir) = enrolled(&root).await;
        let state_dir = root.join("state");
        std::fs::rename(&dir, root.join("gone")).unwrap();
        keys.memory_id_keys_or_create(id).unwrap();
        let secrets = keys.agent_dir(id);
        let stamp = "20261106T120000000Z";
        std::fs::write(
            secrets.join(format!("state_key.forgotten.{stamp}.note.json")),
            b"{\"note\":\"first\"}",
        )
        .unwrap();
        std::fs::rename(
            secrets.join(MEMORY_ID_KEY_FILE),
            secrets.join(format!("{FORGOTTEN_MEMORY_ID_KEY_PREFIX}{stamp}.json")),
        )
        .unwrap();
        let done = forget(&state_dir, &keys, id, "again").unwrap();
        assert_eq!(done.key, format!("state_key.forgotten.{stamp}.json"));
        assert_eq!(done.memory_id_key, None, "already aside");
        assert_eq!(
            std::fs::read(secrets.join(format!("state_key.forgotten.{stamp}.note.json"))).unwrap(),
            b"{\"note\":\"first\"}"
        );
    }
}
