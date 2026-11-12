//! `agora-seed sealed-forget --agent <id> --note "…"`: set aside the state
//! key of an agent whose state is gone for good (it left, its secrets left
//! behind), so the start check stops refusing the run over it
//! ([`super::check::missing_states`]).
//!
//! No key or record is deleted: `<secrets>/<id>/state_key.json` becomes
//! `state_key.forgotten.<utc>.json`, the memory-id key
//! `memory_id_key.forgotten.<utc>.json` (the same `<utc>`, so putting the
//! agent back restores both together and its memory ids still resolve),
//! the durable stop (if any) `sealed_stopped.forgotten.<utc>`, the durable
//! uploading marker (if any) `sealed_uploading.forgotten.<utc>`, under the
//! marker's lock (whose file stays, as lock files do) (so an
//! agent put back is not silently uploading again: its secrets copy alone
//! would read as uploading, with the stop set aside beside it), and the note
//! is kept beside them as `state_key.forgotten.<utc>.note.json`, naming
//! what this forget moves. The signing keys stay. A memory-id key set aside
//! is reused, never minted over, if the agent is ever asked for one again
//! ([`KeyStore::memory_id_keys_or_create`]). The one file deleted is the
//! `memory` tool's search index (`sealed/history_index.json`,
//! [`super::history_index`]), a sidecar rebuilt from the history whenever
//! it is needed. The rest of `sealed/` stays as it is, the `memory` tool's
//! `memory_ids_told.json` included (the agent was told; put back, it isn't
//! told again). Refused while the agent has a `state.json`: a living
//! agent's backups are stopped with `sealed-stop`, never forgotten.
//!
//! **Order.** The note (with the names it will move to), then the state
//! key, which commits the forget, then the memory-id key and the stop. A
//! run cut after the commit is finished by the next: with no
//! `state_key.json`, the newest note whose state key is set aside and
//! whose planned moves aren't all done is completed under its own stamp.
//! A run cut before it left only a note; the next forgets afresh, under a
//! new stamp. A stamp is never reused for a new forget.

use std::path::Path;

use agora_agentkit::ids::AgentId;
use anyhow::Context;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::KeyStore;
use super::keys::{
    FORGOTTEN_MEMORY_ID_KEY, FORGOTTEN_NOTE, FORGOTTEN_STATE_KEY, MEMORY_ID_KEY_FILE,
    STATE_KEY_FILE,
};

/// The note kept beside the forgotten key: why, and what this forget
/// moves (so a cut run is finished as planned, and only it).
#[derive(Debug, Serialize, Deserialize)]
struct Forgotten {
    at: DateTime<Utc>,
    note: String,
    /// Where the memory-id key goes, if there was one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    memory_id_key: Option<String>,
    /// Where the durable stop goes, if there was one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    stop: Option<String>,
    /// Where the durable uploading marker goes, if there was one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    uploading: Option<String>,
}

/// What `sealed-forget` set aside: file names only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Forgot {
    pub key: String,
    pub stop: Option<String>,
    /// The durable uploading marker's new name, if the agent had one.
    pub uploading: Option<String>,
    /// The memory-id key's new name, if the agent had one.
    pub memory_id_key: Option<String>,
    /// The search index was there, and was deleted.
    pub index: bool,
    /// This run finished a forget an earlier one began.
    pub finished_cut: bool,
}

/// Move `from` to `to` in `dir` unless it is already there (or `from`
/// is gone): the moves after the commit, idempotent.
fn move_aside(dir: &Path, from: &str, to: &str) -> std::io::Result<()> {
    if std::fs::exists(dir.join(to))? || !std::fs::exists(dir.join(from))? {
        return Ok(());
    }
    std::fs::rename(dir.join(from), dir.join(to))
}

/// The moves after the commit, as `plan` names them.
fn finish(keys: &KeyStore, id: AgentId, dir: &Path, plan: &Forgotten) -> anyhow::Result<()> {
    if let Some(to) = &plan.memory_id_key {
        move_aside(dir, MEMORY_ID_KEY_FILE, to).context("setting the memory-id key aside")?;
    }
    if let Some(to) = &plan.stop {
        move_aside(dir, super::stop::DURABLE, to).context("setting the stop aside")?;
    }
    if let Some(to) = &plan.uploading {
        // Under the marker's lock, as every writer of it; the lock file
        // stays, as every lock file does
        let _lock = super::uploading::lock(keys, id).context("the uploading marker's lock")?;
        move_aside(dir, super::uploading::DURABLE, to)
            .context("setting the uploading marker aside")?;
    }
    std::fs::File::open(dir)?.sync_all()?;
    Ok(())
}

/// The newest committed forget whose planned moves aren't all done, or
/// (`on_roster`) whose agent the roster still holds.
fn cut_forget(dir: &Path, on_roster: bool) -> anyhow::Result<Option<(String, Forgotten)>> {
    let Some(stamp) = FORGOTTEN_STATE_KEY.stamps(dir)?.pop() else {
        return Ok(None);
    };
    let path = dir.join(FORGOTTEN_NOTE.name(&stamp));
    let plan: Forgotten = match std::fs::read(&path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .with_context(|| format!("{} unreadable", path.display()))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let pending = |from: &str, to: &Option<String>| -> std::io::Result<bool> {
        Ok(match to {
            Some(to) => std::fs::exists(dir.join(from))? && !std::fs::exists(dir.join(to))?,
            None => false,
        })
    };
    let unfinished = on_roster
        || pending(MEMORY_ID_KEY_FILE, &plan.memory_id_key)?
        || pending(super::stop::DURABLE, &plan.stop)?
        || pending(super::uploading::DURABLE, &plan.uploading)?;
    Ok(unfinished.then_some((stamp, plan)))
}

/// Whether the sealed roster beside `state_dir` holds `id`.
fn on_roster(state_dir: &Path, id: AgentId) -> anyhow::Result<bool> {
    let data_dir = state_dir
        .parent()
        .context("the state directory has no parent data directory")?;
    Ok(super::roster::load(data_dir)?.is_some_and(|r| r.agents.contains_key(&id)))
}

pub async fn forget(
    state_dir: &Path,
    keys: &KeyStore,
    id: AgentId,
    note: &str,
) -> anyhow::Result<Forgot> {
    let note = note.trim();
    anyhow::ensure!(!note.is_empty(), "--note is required: it says why");
    // The agent's session lock for the whole command, before its state
    // is looked for (none is made for a directory that is gone)
    let _session = super::session::agent_lock(state_dir, id)?;
    anyhow::ensure!(
        !std::fs::exists(state_dir.join(id.to_string()).join("state.json"))?,
        "agent {id} has a state here: its backups are stopped with `sealed-stop`, not forgotten"
    );
    let dir = keys.agent_dir(id);
    let key = dir.join(STATE_KEY_FILE);
    if !std::fs::exists(&key)? {
        // A forget cut after its commit: finished under its own stamp
        let Some((stamp, plan)) = cut_forget(&dir, on_roster(state_dir, id)?)? else {
            anyhow::bail!(
                "agent {id} has no state key in the secrets directory; nothing to forget"
            );
        };
        let index = super::history_index::remove(&state_dir.join(id.to_string()))
            .context("deleting the search index")?;
        super::roster::replace_for(state_dir, id, None).await?;
        finish(keys, id, &dir, &plan)?;
        tracing::warn!(
            event_type = "sealed_state_forgotten",
            agent_id = %id,
            stamp = %stamp,
            finished_cut = true,
            index_deleted = index,
            "a cut sealed-forget finished as its note planned; no key deleted"
        );
        return Ok(Forgot {
            key: FORGOTTEN_STATE_KEY.name(&stamp),
            stop: plan.stop,
            uploading: plan.uploading,
            memory_id_key: plan.memory_id_key,
            index,
            finished_cut: true,
        });
    }
    // The disposable sidecar first: a failure here leaves the rest to a rerun
    let index = super::history_index::remove(&state_dir.join(id.to_string()))
        .context("deleting the search index")?;
    let at = Utc::now();
    let stamp = at.format(super::STAMP_FORMAT).to_string();
    anyhow::ensure!(
        !std::fs::exists(dir.join(FORGOTTEN_STATE_KEY.name(&stamp)))?,
        "a state key was set aside under {stamp} already; run sealed-forget again"
    );
    let plan = Forgotten {
        at,
        note: note.to_string(),
        memory_id_key: std::fs::exists(dir.join(MEMORY_ID_KEY_FILE))?
            .then(|| FORGOTTEN_MEMORY_ID_KEY.name(&stamp)),
        stop: std::fs::exists(dir.join(super::stop::DURABLE))?
            .then(|| super::keys::FORGOTTEN_STOP.name(&stamp)),
        uploading: std::fs::exists(dir.join(super::uploading::DURABLE))?
            .then(|| super::keys::FORGOTTEN_UPLOADING.name(&stamp)),
    };
    // The note first: a key set aside always says why, and what goes with it
    let note_path = dir.join(FORGOTTEN_NOTE.name(&stamp));
    super::anchors::write_atomic(&note_path, &serde_json::to_vec_pretty(&plan)?)
        .context("writing the note")?;
    // The commit
    let key_to = FORGOTTEN_STATE_KEY.name(&stamp);
    std::fs::rename(&key, dir.join(&key_to)).context("setting the state key aside")?;
    super::crash::point("forget:committed")?;
    // Off the roster, which lowers its witness's count first: the one way
    // an agent leaves it
    super::roster::replace_for(state_dir, id, None).await?;
    finish(keys, id, &dir, &plan)?;
    tracing::warn!(
        event_type = "sealed_state_forgotten",
        agent_id = %id,
        stamp = %stamp,
        index_deleted = index,
        "the operator set aside a state key whose agent has no state here; no key deleted"
    );
    Ok(Forgot {
        key: key_to,
        stop: plan.stop,
        uploading: plan.uploading,
        memory_id_key: plan.memory_id_key,
        index,
        finished_cut: false,
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
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("sealed-stop"), "{e}");
        std::fs::rename(&dir, root.join("gone")).unwrap();
        assert!(forget(&state_dir, &keys, id, " ").await.is_err(), "a note");
        assert_eq!(keys.with_state_keys().unwrap(), vec![id]);
        let done = forget(&state_dir, &keys, id, "it left").await.unwrap();
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
                &crate::alerts::Alerter::off(),
                &Default::default(),
            )
            .await
            .unwrap()
            .refused
            .is_empty()
        );
    }

    /// An agent on the roster whose state is gone is refused at the start
    /// (marked `sealed/refused`). Forgotten, it leaves the roster, the
    /// witness's count follows, and its leftover marker no longer counts
    /// toward the refusal threshold: the next start finds nothing. A rerun
    /// cut after the commit but before the roster takes it off too.
    #[tokio::test]
    async fn forget_takes_the_agent_off_the_roster_and_its_marker_stops_counting() {
        for cut in [false, true] {
            let root = crate::sealed::TestDir::new("forget-roster");
            let (id, keys, dir) = enrolled(&root).await;
            let (other, _, _) = enrolled(&root).await;
            let state_dir = root.join("state");
            let scan = crate::sealed::roster::check(&root, &state_dir)
                .await
                .unwrap();
            assert!(scan.on_roster.contains(&id) && scan.on_roster.contains(&other));
            let witness = || {
                crate::sealed::roster::Witness::load(&root)
                    .unwrap()
                    .unwrap()
                    .agents
            };
            assert_eq!(witness(), 2);
            // The state lost: refused, marked, at the next start
            std::fs::rename(dir.join("state.json"), root.join("gone.json")).unwrap();
            let scan = crate::sealed::roster::check(&root, &state_dir)
                .await
                .unwrap();
            assert_eq!(
                scan.losses.get(&id).map(|l| l.refusal()),
                Some(crate::sealed::notice::Refusal::StateMissing)
            );
            crate::sealed::check::mark(
                &dir,
                id,
                "x",
                crate::sealed::notice::Refusal::StateMissing,
                "d",
            )
            .await;
            assert!(std::fs::exists(dir.join("sealed/refused")).unwrap());
            let scan = crate::sealed::roster::check(&root, &state_dir)
                .await
                .unwrap();
            assert!(scan.refused.contains(&id));
            if cut {
                // Cut after the commit, before the roster: the rerun finishes
                let roster = crate::sealed::roster::load(&root).unwrap().unwrap();
                crate::sealed::crash::at("forget:committed");
                assert!(forget(&state_dir, &keys, id, "it left").await.is_err());
                assert_eq!(crate::sealed::roster::load(&root).unwrap().unwrap(), roster);
            }
            let done = forget(&state_dir, &keys, id, "it left").await.unwrap();
            assert_eq!(done.finished_cut, cut);
            let roster = crate::sealed::roster::load(&root).unwrap().unwrap();
            assert!(!roster.agents.contains_key(&id) && roster.agents.contains_key(&other));
            assert_eq!(witness(), 1, "the witness follows");
            let scan = crate::sealed::roster::check(&root, &state_dir)
                .await
                .unwrap();
            assert!(scan.losses.is_empty(), "{:?}", scan.losses);
            assert!(
                !scan.refused.contains(&id),
                "the marker left behind doesn't count"
            );
            assert!(
                !scan.on_roster.contains(&id),
                "its sealed/ left behind is not adopted again"
            );
            let scan = crate::sealed::roster::check(&root, &state_dir)
                .await
                .unwrap();
            assert!(scan.losses.is_empty() && !scan.on_roster.contains(&id));
            assert!(std::fs::exists(dir.join("sealed/refused")).unwrap(), "kept");
            // Nothing more to forget
            let again = forget(&state_dir, &keys, id, "again").await;
            assert!(again.is_err(), "cut {cut}: {again:?}");
        }
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
        let key = keys.memory_id_keys_or_create(id, &dir).unwrap();
        let done = forget(&state_dir, &keys, id, "it left").await.unwrap();
        assert!(done.index);
        assert_eq!(crate::sealed::history_index::read(&dir).unwrap(), None);
        assert!(
            std::fs::exists(dir.join("sealed/head.json")).unwrap(),
            "only the index"
        );
        assert!(keys.memory_id_keys(id).unwrap().is_none(), "set aside");
        let aside = done.memory_id_key.expect("set aside");
        assert_eq!(
            FORGOTTEN_MEMORY_ID_KEY.stamp(&aside),
            FORGOTTEN_STATE_KEY.stamp(&done.key),
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

    /// A forget cut after its commit (the state key moved, the memory-id
    /// key and the stop not yet): the rerun finishes it under its stamp,
    /// as its note planned, keeping that note.
    #[tokio::test]
    async fn a_cut_forget_finishes_under_its_own_stamp() {
        let root = crate::sealed::TestDir::new("forget-cut");
        let (id, keys, dir) = enrolled(&root).await;
        let state_dir = root.join("state");
        std::fs::rename(dir.join("state.json"), root.join("gone.json")).unwrap();
        keys.memory_id_keys_or_create(id, &dir).unwrap();
        let secrets = keys.agent_dir(id);
        std::fs::write(secrets.join(super::super::stop::DURABLE), b"{}").unwrap();
        crate::sealed::crash::at("forget:committed");
        let e = forget(&state_dir, &keys, id, "first").await.unwrap_err();
        assert!(format!("{e:#}").contains("simulated crash"), "{e:#}");
        assert!(
            std::fs::exists(secrets.join(MEMORY_ID_KEY_FILE)).unwrap(),
            "not yet"
        );
        let stamp = FORGOTTEN_STATE_KEY.stamps(&secrets).unwrap().pop().unwrap();
        let note = std::fs::read(secrets.join(FORGOTTEN_NOTE.name(&stamp))).unwrap();
        let done = forget(&state_dir, &keys, id, "second").await.unwrap();
        assert!(done.finished_cut);
        assert_eq!(done.key, FORGOTTEN_STATE_KEY.name(&stamp));
        assert_eq!(
            done.memory_id_key,
            Some(FORGOTTEN_MEMORY_ID_KEY.name(&stamp))
        );
        assert!(std::fs::exists(secrets.join(FORGOTTEN_MEMORY_ID_KEY.name(&stamp))).unwrap());
        assert!(!std::fs::exists(secrets.join(super::super::stop::DURABLE)).unwrap());
        assert_eq!(
            std::fs::read(secrets.join(FORGOTTEN_NOTE.name(&stamp))).unwrap(),
            note,
            "the first note kept"
        );
        // Finished: nothing more to forget
        assert!(forget(&state_dir, &keys, id, "third").await.is_err());
    }

    /// Forget, the agent put back and its key revived, forget again: a new
    /// stamp, and both forgets' records intact.
    #[tokio::test]
    async fn forget_revive_forget_takes_a_new_stamp() {
        let root = crate::sealed::TestDir::new("forget-twice");
        let (id, keys, dir) = enrolled(&root).await;
        let state_dir = root.join("state");
        std::fs::rename(dir.join("state.json"), root.join("gone.json")).unwrap();
        let key = *keys
            .memory_id_keys_or_create(id, &dir)
            .unwrap()
            .current
            .expose();
        let secrets = keys.agent_dir(id);
        std::fs::write(secrets.join(super::super::stop::DURABLE), b"{\"n\":1}").unwrap();
        let first = forget(&state_dir, &keys, id, "first").await.unwrap();
        let s1 = FORGOTTEN_STATE_KEY.stamp(&first.key).unwrap().to_string();
        let record = |s: &str| {
            [
                FORGOTTEN_STATE_KEY.name(s),
                FORGOTTEN_NOTE.name(s),
                FORGOTTEN_MEMORY_ID_KEY.name(s),
                format!("{}.forgotten.{s}", super::super::stop::DURABLE),
            ]
            .map(|n| std::fs::read(secrets.join(n)).unwrap())
        };
        let before = record(&s1);
        // The operator puts the agent back: its state key, then its memory
        std::fs::copy(
            secrets.join(FORGOTTEN_STATE_KEY.name(&s1)),
            secrets.join(STATE_KEY_FILE),
        )
        .unwrap();
        let revived = keys.memory_id_keys_or_create(id, &dir).unwrap();
        assert_eq!(*revived.current.expose(), key);
        std::fs::write(secrets.join(super::super::stop::DURABLE), b"{\"n\":2}").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let second = forget(&state_dir, &keys, id, "second").await.unwrap();
        assert!(!second.finished_cut);
        let s2 = FORGOTTEN_STATE_KEY.stamp(&second.key).unwrap().to_string();
        assert_ne!(s1, s2, "a new stamp");
        assert_eq!(record(&s1), before, "the first forget's records untouched");
        let after = record(&s2);
        assert!(String::from_utf8_lossy(&after[1]).contains("second"));
        assert_eq!(after[3], b"{\"n\":2}");
    }

    /// An agent uploads (stopped at its request, or not), leaves, and is
    /// forgotten; then it is put back, its directory with or without its
    /// old local markers, its state key copied back, and the runner's
    /// start pass runs (the one-time migration not yet done here, so its
    /// guard is what holds): neither uploading nor stopped, and no durable
    /// marker made again. Only an explicit opt-in marks it.
    #[tokio::test]
    async fn a_forgotten_uploader_put_back_is_not_uploading_after_the_start_pass() {
        for stopped in [true, false] {
            for with_local_markers in [true, false] {
                let case = format!("stopped {stopped}, local markers {with_local_markers}");
                let root = crate::sealed::TestDir::new("forget-uploading");
                let (id, keys, dir) = enrolled(&root).await;
                crate::sealed::uploading::tests::ledger(&dir, true);
                let state_dir = root.join("state");
                // Uploaded: nothing waits
                let seal = agora_agentkit::state::storage::AgentSeal::new(&dir, id);
                for item in seal.outbox().await.unwrap() {
                    seal.mark_sent(item.ord).await.unwrap();
                }
                if stopped {
                    crate::sealed::stop::stop(&state_dir, &keys, id, "It asked")
                        .await
                        .unwrap();
                }
                let kept = root.join("kept");
                std::fs::rename(&dir, &kept).unwrap();
                let done = forget(&state_dir, &keys, id, "it left").await.unwrap();
                let secrets = keys.agent_dir(id);
                assert!(done.uploading.is_some(), "{case}");
                assert_eq!(done.stop.is_some(), stopped, "{case}");
                std::fs::rename(&kept, &dir).unwrap();
                if !with_local_markers {
                    for m in [
                        crate::sealed::uploading::UPLOADING,
                        crate::sealed::stop::STOPPED,
                    ] {
                        let _ = std::fs::remove_file(dir.join("sealed").join(m));
                    }
                }
                std::fs::copy(secrets.join(&done.key), secrets.join(STATE_KEY_FILE)).unwrap();
                crate::sealed::uploading::start_pass(
                    &state_dir,
                    &keys,
                    &crate::alerts::Alerter::off(),
                )
                .await;
                assert!(
                    !crate::sealed::uploading::is_uploading(&dir, &keys, id),
                    "{case}"
                );
                assert!(!crate::sealed::uploads(&dir, &keys, id), "{case}");
                assert!(!crate::sealed::stop::is_stopped(&dir), "{case}");
                for durable in [
                    crate::sealed::uploading::DURABLE,
                    crate::sealed::stop::DURABLE,
                ] {
                    assert!(!std::fs::exists(secrets.join(durable)).unwrap(), "{case}");
                }
                // An explicit opt-in marks it again
                crate::sealed::uploading::set(
                    &dir,
                    &keys,
                    id,
                    crate::sealed::uploading::UploadingBy::OptIn,
                )
                .unwrap();
                crate::sealed::uploading::start_pass(
                    &state_dir,
                    &keys,
                    &crate::alerts::Alerter::off(),
                )
                .await;
                assert!(crate::sealed::uploads(&dir, &keys, id), "{case}");
            }
        }
    }
}
