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
//! A state here that isn't refused and has its `head.json` is never
//! overwritten: a deliberate change is `sealed-accept --note` instead.
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
    /// `head.json` was gone and was rebuilt from Agora's verified chain.
    pub rebuilt_head: bool,
}

/// What the refusal marker says, as a [`Refusal`].
fn refusal_of(text: &str) -> Refusal {
    match text.trim() {
        "chain_broken" => Refusal::ChainBroken,
        "head_missing" => Refusal::HeadMissing,
        _ => Refusal::Mismatch,
    }
}

/// Restore `id`'s last sealed version as its state: a refused agent's, or
/// one whose state or `sealed/` (or whole directory) is gone. A state
/// here that isn't refused and has its `head.json` is never overwritten.
pub async fn restore(
    state_dir: &Path,
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
    note: Option<&str>,
) -> anyhow::Result<Restored> {
    let agent_dir = state_dir.join(id.to_string());
    let seal = AgentSeal::new(&agent_dir, id);
    // The keys live in the secrets tree, apart from the state: without
    // them nothing sealed can be opened or verified
    let (Some(k), Some(trusted)) = (keys.seal_keys(id), keys.trusted(id).ok().flatten()) else {
        anyhow::bail!(
            "no keys for agent {id} in the secrets directory: its backups on Agora can't be \
             opened or checked without them, so there is nothing this runner can restore"
        );
    };
    let found = std::fs::exists(agent_dir.join("state.json"))?;
    let had_head = seal.head().await?.is_some();
    let refused = seal.is_refused().await?;
    anyhow::ensure!(
        refused || !found || !had_head,
        "agent {id} is not refused and its state and sealed head are here: the start check \
         finds it intact or holds it, and this won't overwrite it (a deliberate change is \
         `sealed-accept --note`)"
    );
    // Made if the directory is gone; no session in progress
    let lock = super::session::take(state_dir, id, true).await?;
    let reason = if refused {
        refusal_of(&std::fs::read_to_string(seal.dir().join("refused")).unwrap_or_default())
    } else if had_head {
        Refusal::Mismatch
    } else {
        Refusal::HeadMissing
    };
    if had_head {
        anyhow::ensure!(
            seal.is_enrolled().await?,
            "agent {id} is not enrolled in sealed state"
        );
    } else {
        // head.json is gone (with sealed/, or the whole directory): it is
        // rebuilt from Agora's chain, verified with the runner's own keys,
        // the way `sealed-accept` adopts it. Never a genesis
        let chain = api
            .chain(id, &k.signing)
            .await
            .map_err(anyhow::Error::msg)
            .context("reading the agent's chain on Agora")?;
        anyhow::ensure!(
            !chain.is_empty(),
            "agent {id} has no sealed head here and Agora holds no chain for it: nothing to \
             restore from"
        );
        let signing = &k.signing;
        let fetch = move |sha: agora_agentkit::govlog::Sha256Hex| async move {
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
    let from = DateTime::from_timestamp_micros(last.snapshot.entry.state_at)
        .context("the version's date is out of range")?;
    // Archives a state.json found here as state.<utc>.json first
    FsStorage::new(state_dir)
        .save_sealed_raw(id, state)
        .await
        .context("writing the restored state")?;
    if !seal.is_enrolled().await? {
        seal.enroll()
            .await
            .context("writing the enrolment marker")?;
    }
    if refused {
        seal.clear_refusal().await.context("clearing the refusal")?;
    }
    Notice::restored(&agent_dir, from, note, reason, found).context("writing the notice")?;
    drop(lock);
    let head = last.head.head;
    tracing::warn!(
        event_type = "sealed_state_restored",
        agent_id = %id,
        reason = reason.as_str(),
        refused,
        found,
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
        let e = restore(&root.join("state"), &keys, &api, id, None)
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
            let done = restore(&state_dir, &keys, &api, id, None).await.unwrap();
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
