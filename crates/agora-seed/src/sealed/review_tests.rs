//! Crash and stranding tests for the review round of 2026-11-12 (state
//! loss, stranding, wrongful fleet stops): each fails on 1cfe81eb.
//! Synthetic states, temp dirs, no network.

use std::path::Path;

use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::{FsStorage, Storage};
use agora_agentkit::state::storage::{AgentSeal, plaintext_hash};
use agora_agentkit::state::verify::HeadRef;

use super::api::fake::FakeApi;
use super::check::tests::{enrolled_local, state};
use super::enroll::tests::{agent, enroller};
use super::notice::Refusal;
use super::{accept, check, restore, roster, rotate};

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let target = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &target);
        } else {
            std::fs::copy(e.path(), target).unwrap();
        }
    }
}

async fn head(dir: &Path, id: AgentId) -> HeadRef {
    AgentSeal::new(dir, id).head().await.unwrap().unwrap().head
}

async fn memory(root: &Path, id: AgentId) -> serde_json::Value {
    FsStorage::new(root.join("state"))
        .load_raw(id)
        .await
        .unwrap()
        .value
}

fn rostered(root: &Path, id: AgentId) -> Option<(u64, agora_agentkit::govlog::Sha256Hex, bool)> {
    roster::load(root)
        .unwrap()
        .and_then(|r| r.agents.get(&id).map(|e| (e.seq, e.entry_hash, e.pending)))
}

/// The state emptied outside the store, and the start's refusal of it.
async fn wipe_and_refuse(root: &Path, id: AgentId, dir: &Path) {
    FsStorage::new(root.join("state"))
        .save_raw(id, serde_json::json!({ "memory": "" }), None)
        .await
        .unwrap();
    check::mark(dir, id, "tarn", Refusal::Mismatch, "synthetic").await;
}

/// An enrolment cut after its first archives: a head at an old archive's
/// version, no chain marker, never uploaded, never on the roster.
async fn half_enrolled(root: &Path) -> (AgentId, super::KeyStore, std::path::PathBuf) {
    let (id, keys, dir) = agent(root, &["a", "b", "c", "d"]).await;
    super::crash::at("chain:archive");
    assert!(enroller(root, &keys).chain(id, false).await.is_err());
    assert!(!super::is_enrolled(&dir));
    assert!(AgentSeal::new(&dir, id).head().await.unwrap().is_some());
    (id, keys, dir)
}

/// S1. `sealed-restore` on an agent whose enrolment was cut used to write
/// the old archive its partial chain's head names over the agent's
/// current state, enrol it and set the roster. Refused, naming
/// `sealed-enroll`, with nothing written; `sealed-enroll` then finishes
/// the chain with the state as it was.
#[tokio::test]
async fn a_restore_never_writes_a_half_enrolled_agents_old_archive_as_its_state() {
    let root = super::TestDir::new("review-s1");
    let (id, keys, dir) = half_enrolled(&root).await;
    let before = std::fs::read(dir.join("state.json")).unwrap();
    let files = std::fs::read_dir(&dir).unwrap().count();
    let api = FakeApi::new();
    let e = restore::restore(&root.join("state"), &keys, &api, id, None, false)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("never finished") && e.contains("sealed-enroll --agent"),
        "{e}"
    );
    assert_eq!(std::fs::read(dir.join("state.json")).unwrap(), before);
    assert_eq!(
        std::fs::read_dir(&dir).unwrap().count(),
        files,
        "no archive made"
    );
    assert!(!super::is_enrolled(&dir));
    assert!(rostered(&root, id).is_none());
    assert!(api.calls.lock().unwrap().is_empty());
    // Not with --discard-local either
    assert!(
        restore::restore(&root.join("state"), &keys, &api, id, None, true)
            .await
            .is_err()
    );
    assert_eq!(std::fs::read(dir.join("state.json")).unwrap(), before);
    // The way it names
    let report = enroller(&root, &keys).chain(id, false).await.unwrap();
    assert!(report.resumed);
    assert_eq!(
        std::fs::read(dir.join("state.json")).ok().map(|_| ()),
        Some(())
    );
    assert_eq!(memory(&root, id).await["memory"]["content"], "d");
}

/// S1's other side: an agent that *was* chained (on the roster) and lost
/// its chain marker is still restored from its chain here.
#[tokio::test]
async fn a_chained_agent_that_lost_its_marker_is_still_restored() {
    let root = super::TestDir::new("review-s1-marker");
    let (id, keys, dir) = enrolled_local(&root).await;
    let marker = dir
        .join(agora_agentkit::state::storage::SEALED_DIR)
        .join(agora_agentkit::state::storage::CHAINED);
    std::fs::rename(&marker, root.join("marker.aside")).unwrap();
    restore::restore(&root.join("state"), &keys, &FakeApi::new(), id, None, false)
        .await
        .unwrap();
    assert!(super::is_enrolled(&dir));
    assert_eq!(
        plaintext_hash(&memory(&root, id).await),
        plaintext_hash(&state(1))
    );
}

/// S2. `rotate-keys --signing` on a half-enrolled agent used to rotate
/// with no pre-flight and no re-sign (both are for chained agents),
/// leaving the partial chain signed by a key the finished chain doesn't
/// lead from. Refused, naming `sealed-enroll`; after the chain is
/// finished the rotation goes through and the chain verifies.
#[tokio::test]
async fn a_rotation_waits_for_a_half_enrolled_agents_chain() {
    let root = super::TestDir::new("review-s2");
    let (id, keys, dir) = half_enrolled(&root).await;
    let state_dir = root.join("state");
    let api = FakeApi::new();
    let signing = keys.signing(id).unwrap().verifying_key();
    for (s, e) in [(true, false), (false, true), (true, true)] {
        let err = rotate::rotate_keys(&state_dir, &keys, &api, id, s, e)
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("never finished") && err.contains("sealed-enroll --agent"),
            "{err}"
        );
    }
    assert!(keys.next_secrets(id).unwrap().is_none(), "nothing written");
    assert_eq!(keys.signing(id).unwrap().verifying_key(), signing);
    assert!(api.calls.lock().unwrap().is_empty(), "nothing sent");
    enroller(&root, &keys).chain(id, false).await.unwrap();
    let done = rotate::rotate_keys(&state_dir, &keys, &api, id, true, false)
        .await
        .unwrap();
    assert!(done.signing && done.resigned > 0);
    let entries = check::local_entries(&AgentSeal::new(&dir, id))
        .await
        .unwrap();
    let report = agora_agentkit::state::verify::verify_chain(
        id,
        &entries,
        &keys.trusted(id).unwrap().unwrap(),
    );
    assert!(report.intact, "{:?}", report.problems);
}

/// S6, accept. An accept cut after it cleared the refusal and sealed the
/// state, before the roster took the head: the rerun used to say "not
/// refused; nothing to accept" though the roster's error says to run it
/// again. It finishes the roster step. The agent here was rolled back, so
/// the accepted head is below the roster's: left unfinished, the next
/// start would refuse it as rolled back once more.
#[tokio::test]
async fn an_accept_cut_before_the_roster_is_finished_by_its_rerun() {
    let root = super::TestDir::new("review-s6-accept");
    let state_dir = root.join("state");
    let (id, keys, dir) = enrolled_local(&root).await;
    let backup = root.join("backup");
    copy_dir(&dir, &backup);
    for n in 2..5 {
        super::props::save(&root, id, state(n)).await;
    }
    roster::record_heads(&root, &state_dir, &[id])
        .await
        .unwrap();
    let newer = head(&dir, id).await;
    // The older copy put back; the start refuses it
    std::fs::rename(&dir, root.join("newer")).unwrap();
    std::fs::rename(&backup, &dir).unwrap();
    let scan = roster::check(&root, &state_dir).await.unwrap();
    assert_eq!(scan.losses[&id].refusal(), Refusal::RolledBack);
    check::mark(&dir, id, "tarn", Refusal::RolledBack, "synthetic").await;
    let api = FakeApi::new();
    super::crash::at("accept:appended");
    assert!(
        accept::accept(&state_dir, &keys, &api, id, Some("kept on purpose"))
            .await
            .is_err()
    );
    let accepted = head(&dir, id).await;
    assert!(accepted.seq < newer.seq);
    assert!(!AgentSeal::new(&dir, id).is_refused().await.unwrap());
    assert_eq!(
        rostered(&root, id),
        Some((newer.seq, newer.entry_hash, true)),
        "the roster still holds the newer head, marked pending"
    );
    // The rerun
    let done = accept::accept(&state_dir, &keys, &api, id, Some("kept on purpose"))
        .await
        .unwrap();
    assert_eq!(done.head, accepted, "no second version sealed");
    assert_eq!(
        rostered(&root, id),
        Some((accepted.seq, accepted.entry_hash, false))
    );
    let scan = roster::check(&root, &state_dir).await.unwrap();
    assert!(scan.losses.is_empty(), "{:?}", scan.losses);
    // And only that: an agent that is simply not refused has nothing to
    // accept
    let e = accept::accept(&state_dir, &keys, &api, id, Some("again"))
        .await
        .unwrap_err()
        .to_string();
    assert!(e.contains("nothing to accept"), "{e}");
    assert!(api.calls.lock().unwrap().is_empty());
}

/// S6, accept: a rollback no start has refused is never blessed by the
/// rerun path. Not refused and not pending: nothing to accept, and the
/// roster keeps the newer head.
#[tokio::test]
async fn an_unrefused_rollback_is_not_accepted_through_the_rerun_path() {
    let root = super::TestDir::new("review-s6-guard");
    let state_dir = root.join("state");
    let (id, keys, dir) = enrolled_local(&root).await;
    let backup = root.join("backup");
    copy_dir(&dir, &backup);
    super::props::save(&root, id, state(2)).await;
    roster::record_heads(&root, &state_dir, &[id])
        .await
        .unwrap();
    let newer = head(&dir, id).await;
    std::fs::rename(&dir, root.join("newer")).unwrap();
    std::fs::rename(&backup, &dir).unwrap();
    assert!(
        accept::accept(&state_dir, &keys, &FakeApi::new(), id, Some("x"))
            .await
            .is_err()
    );
    assert_eq!(
        rostered(&root, id),
        Some((newer.seq, newer.entry_hash, false))
    );
}

/// S6, restore. The roster takes the restored head before the chain
/// marker and the refusal go, so a cut at any point leaves a rerun that
/// does the whole restore again; it used to be written last, after the
/// refusal was cleared, where a rerun found nothing to do.
#[tokio::test]
async fn a_restore_cut_anywhere_is_finished_by_its_rerun_roster_included() {
    for point in [
        "restore:noticed",
        "restore:written",
        "restore:rostered",
        "restore:enrolled",
    ] {
        let root = super::TestDir::new("review-s6-restore");
        let state_dir = root.join("state");
        let (id, keys, dir) = enrolled_local(&root).await;
        // The head moved on since the roster last saw it
        super::props::save(&root, id, state(2)).await;
        let sealed = head(&dir, id).await;
        let old = rostered(&root, id).unwrap();
        assert!(old.0 < sealed.seq);
        wipe_and_refuse(&root, id, &dir).await;
        let api = FakeApi::new();
        super::crash::at(point);
        assert!(
            restore::restore(&state_dir, &keys, &api, id, None, false)
                .await
                .is_err(),
            "{point}"
        );
        let seal = AgentSeal::new(&dir, id);
        // Whatever was cut: still refused, so the rerun has its work
        assert!(seal.is_refused().await.unwrap(), "{point}");
        if matches!(point, "restore:rostered" | "restore:enrolled") {
            assert_eq!(
                rostered(&root, id),
                Some((sealed.seq, sealed.entry_hash, false)),
                "{point}: the roster before the refusal goes"
            );
        }
        restore::restore(&state_dir, &keys, &api, id, None, false)
            .await
            .unwrap();
        assert!(!seal.is_refused().await.unwrap(), "{point}");
        assert_eq!(
            rostered(&root, id),
            Some((sealed.seq, sealed.entry_hash, false)),
            "{point}"
        );
        assert_eq!(
            plaintext_hash(&memory(&root, id).await),
            plaintext_hash(&state(2)),
            "{point}"
        );
        let scan = roster::check(&root, &state_dir).await.unwrap();
        assert!(scan.losses.is_empty(), "{point}: {:?}", scan.losses);
        // A never-uploaded agent is given no sealed/published
        assert!(
            !dir.join("sealed").join(super::anchors::PUBLISHED).exists(),
            "{point}"
        );
        assert!(api.calls.lock().unwrap().is_empty(), "{point}");
    }
}
