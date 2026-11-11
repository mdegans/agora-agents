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

use std::collections::HashMap;
use std::path::Path;

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

/// Restore `id`'s last sealed version as its state: a refused agent's, or
/// one whose state or `sealed/` (or whole directory) is gone. A state
/// here that isn't refused and has its `head.json` is never overwritten.
pub async fn restore(
    state_dir: &Path,
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
    note: Option<&str>,
    discard_local: bool,
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
    anyhow::ensure!(
        refused || !std::fs::exists(&state_path)? || !had_head || !seal.is_enrolled().await?,
        "agent {id} is not refused and its state and sealed head are here: the start check \
         finds it intact or holds it, and this won't overwrite it (a deliberate change is \
         `sealed-accept --note`)"
    );
    // Made if the directory is gone; no session in progress
    let lock = super::session::take(state_dir, id, true).await?;
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
        if !seal.is_enrolled().await? {
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
    // B5: pass `RestoredFrom::Local` when the version came from the local
    // chain, and set the roster's head (`roster::replace`)
    Notice::restored(
        &agent_dir,
        from,
        note,
        reason,
        found,
        refused,
        super::notice::RestoredFrom::Agora,
    )
    .context("writing the notice")?;
    super::crash::point("restore:noticed")?;
    // A stray tmp (a save or restore cut short) is kept, never truncated
    // by the write below
    let tmp = agent_dir.join("state.json.tmp");
    if std::fs::exists(&tmp)? {
        let at = Utc::now().format(crate::sealed::STAMP_FORMAT);
        // Not `*.json`: the enrolment's archive listing would take it
        std::fs::rename(&tmp, agent_dir.join(format!("state.json.tmp.{at}.aside")))
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
    if refused {
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
}
