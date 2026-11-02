//! `agora-seed sealed-accept --agent <id> --note "…"`: the only way out of
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
//! A refusal for a missing `head.json` can't be accepted here: there is no
//! head to seal onto, and only agentkit's `AgentSeal` writes `sealed/`.
//! Restore `head.json` from a backup of the agent's directory first.

use std::path::Path;

use agora_agentkit::enums::StateEntryKind;
use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::{FsStorage, Storage};
use agora_agentkit::state::codec::KeyframePolicy;
use agora_agentkit::state::storage::AgentSeal;
use agora_agentkit::state::verify::HeadRef;
use anyhow::Context;
use chrono::Utc;

use super::KeyStore;
use super::notice::{Notice, Refusal};

/// What the refusal marker says, as a [`Refusal`].
fn refusal_of(text: &str) -> Refusal {
    match text.trim() {
        "chain_broken" => Refusal::ChainBroken,
        "head_missing" => Refusal::HeadMissing,
        _ => Refusal::Mismatch,
    }
}

/// What `sealed-accept` did: hashes only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Accepted {
    pub reason: Refusal,
    pub head: HeadRef,
}

/// Accept `id`'s current state with `note`.
pub async fn accept(
    state_dir: &Path,
    keys: &KeyStore,
    id: AgentId,
    note: &str,
) -> anyhow::Result<Accepted> {
    anyhow::ensure!(
        !note.trim().is_empty(),
        "--note is required: the agent reads it"
    );
    let agent_dir = state_dir.join(id.to_string());
    let seal = AgentSeal::new(&agent_dir, id);
    anyhow::ensure!(
        seal.is_enrolled().await?,
        "agent {id} is not enrolled in sealed state"
    );
    anyhow::ensure!(
        seal.is_refused().await?,
        "agent {id} is not refused; nothing to accept"
    );
    let reason =
        refusal_of(&std::fs::read_to_string(seal.dir().join("refused")).unwrap_or_default());
    let current = FsStorage::new(state_dir)
        .load_raw(id)
        .await
        .context("the agent's state won't load")?
        .value;
    let k = keys
        .seal_keys(id)
        .context("no sealing keys for the agent")?;

    anyhow::ensure!(
        seal.head().await?.is_some(),
        "agent {id}'s sealed/head.json is missing: restore it from a backup of its directory \
         first; there is no head to seal onto"
    );
    seal.clear_refusal().await.context("clearing the refusal")?;
    let now = Utc::now();
    let head = seal
        .append_state(
            &k,
            &KeyframePolicy::default(),
            StateEntryKind::Snapshot,
            &current,
            now,
        )
        .await
        .context("sealing the accepted state (the refusal is cleared; the next start checks it)")?;
    super::enroll::record_sizes(&agent_dir, head, now, &current);
    Notice::accept(&agent_dir, note, reason).context("writing the notice")?;
    tracing::warn!(
        event_type = "sealed_state_accepted",
        agent_id = %id,
        reason = reason.as_str(),
        head = %head.entry_hash,
        seq = head.seq,
        "the operator accepted a refused agent's current state"
    );
    Ok(Accepted { reason, head })
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
        assert!(accept(&state_dir, &keys, id, "n").await.is_err());

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
            accept(&state_dir, &keys, id, "  ").await.is_err(),
            "a note is required"
        );

        let done = accept(&state_dir, &keys, id, "A test wipe; I restored nothing.")
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
            seat[1].starts_with("Your sealed state: 2 versions since "),
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

    #[tokio::test]
    async fn a_lost_head_file_is_not_accepted_over() {
        let root = crate::sealed::TestDir::new("accept-head");
        let (id, keys, dir) = enrolled(&root).await;
        let api = FakeApi::new();
        let sealed = dir.join("sealed");
        std::fs::rename(sealed.join("head.json"), sealed.join("head.json.lost")).unwrap();
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
        assert!(accept(&root.join("state"), &keys, id, "n").await.is_err());
        assert!(
            AgentSeal::new(&dir, id).is_refused().await.unwrap(),
            "still refused"
        );
        // The head restored by hand: accepted
        std::fs::rename(sealed.join("head.json.lost"), sealed.join("head.json")).unwrap();
        accept(&root.join("state"), &keys, id, "head file restored")
            .await
            .unwrap();
    }
}
