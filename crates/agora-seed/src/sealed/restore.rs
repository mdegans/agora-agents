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
//! A deliberate change is `sealed-accept --note` instead; a lost
//! `head.json` is `sealed-accept`'s too (it rebuilds it from Agora).
//! Like every operator command it refuses while the agent has a session,
//! and holds the agent's `sealed/lock` throughout.

use std::path::Path;

use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::FsStorage;
use agora_agentkit::state::storage::{AgentSeal, as_reloaded, plaintext_hash};
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
}

/// What the refusal marker says, as a [`Refusal`].
fn refusal_of(text: &str) -> Refusal {
    match text.trim() {
        "chain_broken" => Refusal::ChainBroken,
        "head_missing" => Refusal::HeadMissing,
        _ => Refusal::Mismatch,
    }
}

/// Restore refused `id`'s last sealed version as its state.
pub async fn restore(
    state_dir: &Path,
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
    note: Option<&str>,
) -> anyhow::Result<Restored> {
    let agent_dir = state_dir.join(id.to_string());
    let seal = AgentSeal::new(&agent_dir, id);
    let lock = super::session::take(state_dir, id, true).await?;
    anyhow::ensure!(
        seal.is_enrolled().await?,
        "agent {id} is not enrolled in sealed state"
    );
    anyhow::ensure!(
        seal.is_refused().await?,
        "agent {id} is not refused; nothing to restore"
    );
    let reason =
        refusal_of(&std::fs::read_to_string(seal.dir().join("refused")).unwrap_or_default());
    anyhow::ensure!(
        seal.head().await?.is_some(),
        "agent {id}'s sealed/head.json is missing: `sealed-accept --agent {id}` rebuilds it \
         from Agora's verified chain first"
    );
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
    let from = DateTime::from_timestamp_micros(last.snapshot.entry.state_at)
        .context("the version's date is out of range")?;
    // Archives the damaged state.json as state.<utc>.json first
    FsStorage::new(state_dir)
        .save_sealed_raw(id, state)
        .await
        .context("writing the restored state")?;
    seal.clear_refusal().await.context("clearing the refusal")?;
    Notice::restored(&agent_dir, from, note, reason).context("writing the notice")?;
    drop(lock);
    let head = last.head.head;
    tracing::warn!(
        event_type = "sealed_state_restored",
        agent_id = %id,
        reason = reason.as_str(),
        head = %head.entry_hash,
        seq = head.seq,
        from = %from,
        "the operator restored a refused agent's last sealed version; the state found is archived"
    );
    Ok(Restored { reason, head, from })
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
        let done = restore(&root.join("state"), &keys, &api, id, Some("a test wipe"))
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
        assert!(
            restore(&root.join("state"), &keys, &api, id, None)
                .await
                .is_err(),
            "not refused any more"
        );
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
            restore(&root.join("state"), &keys, &api, id, None)
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
            restore(&root.join("state"), &keys, &api, id, None)
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
        restore(&root.join("state"), &keys, &api, id, None)
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
        let e = restore(&root.join("state"), &other, &api, id, None)
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
        let e = restore(&root.join("state"), &keys, &api, id, None)
            .await
            .unwrap_err();
        assert!(format!("{e:#}").contains("not restoring it"), "{e:#}");
        assert_eq!(archives(&dir), archived, "nothing written");
        assert!(AgentSeal::new(&dir, id).is_refused().await.unwrap());
    }
}
