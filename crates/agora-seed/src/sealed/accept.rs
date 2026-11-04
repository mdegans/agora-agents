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
    let k = keys
        .seal_keys(id)
        .context("no sealing keys for the agent")?;

    let mut adopted = None;
    if seal.head().await?.is_none() {
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
            Notice::accept(&agent_dir, note.unwrap_or(REBUILT_NOTE), reason, None)
                .context("writing the notice")?;
            head
        }
        (Some(Adopted::Mismatch(head)), None) => anyhow::bail!(
            "head.json was rebuilt from Agora (head {} at seq {}), but the state here doesn't \
             match it: the agent stays refused. Pass --note to seal its current state on top",
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

/// Resolve a held fork (v1 has no merges), or versions Agora lost (a held
/// LocalMissing): the current state is sealed as a keyframe child of
/// Agora's head, which is first verified with the runner's own trust. For
/// a fork the versions that went the other way are kept locally as
/// `.displaced`; for a loss, see [`super::lost`] (and, Agora holding
/// nothing, a new chain whose genesis records the break). A refusal is
/// cleared first. The agent's next session is told, with `note`.
pub async fn rebase(
    state_dir: &Path,
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
    note: &str,
) -> anyhow::Result<Rebased> {
    let note = note.trim();
    anyhow::ensure!(!note.is_empty(), "--note is required: the agent reads it");
    // Before anything is read or written: no runner on this data dir (it
    // could be saving), held to the end, and no session in progress. The
    // start check below takes the agent's sealed/lock itself, so that lock
    // is taken after it
    let data_dir = state_dir
        .parent()
        .context("the state directory has no parent data directory")?;
    let _runner = super::session::runner_exclusive(data_dir)?;
    super::session::no_session(state_dir, id, true)?;
    let agent_dir = state_dir.join(id.to_string());
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
            seat[1].starts_with("Your sealed state: intact. 2 versions since "),
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
                "Your memory is 400 characters (\u{2212}600 since your last checked version) and"
            ),
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
}
