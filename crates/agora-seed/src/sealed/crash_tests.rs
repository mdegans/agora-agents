//! Crash-point tests: a save, an accept, a restore and the tmp promotion
//! cut after each filesystem step, then a restart. Nothing is lost, and
//! what the next start says is consistent with where it stopped. Plus a
//! property test of anchor proofs. Synthetic states, temp dirs.

use std::path::Path;

use agora_agentkit::govlog::Sha256Hex;
use agora_agentkit::ids::{AgentId, GovernanceLogId};
use agora_agentkit::reactor::{FsStorage, Storage};
use agora_agentkit::state::storage::{AgentSeal, plaintext_hash};
use agora_agentkit::state::{AnchorData, AnchorProof};
use proptest::prelude::*;

use super::api::fake::FakeApi;
use super::check::{self, Outcome};
use super::notice::Refusal;
use super::props::{History, Mode, build, version};
use super::sizes::Sizes;

/// An archive named now, as FsStorage names one.
fn archive_name() -> String {
    format!(
        "state.{}.json",
        chrono::Utc::now().format(crate::sealed::STAMP_FORMAT)
    )
}

/// The restart, once a cut save's tmp has settled.
fn later() -> std::time::SystemTime {
    std::time::SystemTime::now() + crate::promote::SETTLED + std::time::Duration::from_secs(1)
}

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

async fn hash_of(state_dir: &Path, id: AgentId) -> Option<Sha256Hex> {
    FsStorage::new(state_dir)
        .load_raw(id)
        .await
        .ok()
        .map(|l| plaintext_hash(&l.value))
}

async fn check_at(state_dir: &Path, b: &super::props::Built) -> Outcome {
    check::check(
        &b.dir,
        b.id,
        &hash_of(state_dir, b.id).await.unwrap(),
        Sizes::default(),
        &b.keys,
        &b.api,
    )
    .await
}

fn plain() -> History {
    History {
        saves: 1,
        upload: 0,
        rotate: false,
        mode: Mode::Plain,
        upload_rebased: false,
    }
}

/// agentkit's sealed save is prepare (an outbox `.tmp` item and its
/// blob), then FsStorage's write (tmp, fsync, archive, rename), then
/// commit (the item `.json`, head.json). Each stage it can be cut at is
/// rebuilt on disk from the states before and after a real save; the
/// restart (tmp promotion, then the start check) runs on the old or the
/// new state, intact, and never on none.
#[tokio::test]
async fn a_save_cut_at_any_step_loses_nothing() {
    for stage in 1..=5 {
        let b = build(&plain()).await;
        let state_dir = b.state_dir();
        let old = hash_of(&state_dir, b.id).await.unwrap();
        let before = b.root.join("before");
        copy_dir(&b.dir, &before);
        let new_value = version(99);
        {
            let enrolment = std::sync::Arc::new(super::SealedEnrolment::new(
                &state_dir,
                b.root.join("secrets"),
            ));
            let mut store = agora_agentkit::state::storage::Mirrored::new(
                FsStorage::new(&state_dir),
                (),
                enrolment,
            );
            let base = store.load_raw(b.id).await.unwrap().base;
            store
                .save_raw(b.id, new_value.clone(), base.as_ref())
                .await
                .unwrap();
        }
        let new = hash_of(&state_dir, b.id).await.unwrap();
        let after = b.root.join("after");
        copy_dir(&b.dir, &after);
        // The item and blob the save added
        let added = |sub: &str| -> Vec<std::ffi::OsString> {
            let names = |d: &Path| -> std::collections::BTreeSet<std::ffi::OsString> {
                std::fs::read_dir(d.join(sub))
                    .map(|r| r.map(|e| e.unwrap().file_name()).collect())
                    .unwrap_or_default()
            };
            names(&after).difference(&names(&before)).cloned().collect()
        };
        let items = added("sealed/outbox");
        let blobs = added("sealed/blobs");
        std::fs::remove_dir_all(&b.dir).unwrap();
        copy_dir(&before, &b.dir);
        let prepare = |dir: &Path| {
            for name in &items {
                let tmp = Path::new(name).with_extension("tmp");
                std::fs::copy(
                    after.join("sealed/outbox").join(name),
                    dir.join("sealed/outbox").join(tmp),
                )
                .unwrap();
            }
            for name in &blobs {
                std::fs::copy(
                    after.join("sealed/blobs").join(name),
                    dir.join("sealed/blobs").join(name),
                )
                .unwrap();
            }
        };
        let new_state = std::fs::read(after.join("state.json")).unwrap();
        match stage {
            // Prepared only
            1 => prepare(&b.dir),
            // The local write's tmp, fsynced, not yet moved in
            2 => {
                prepare(&b.dir);
                std::fs::write(b.dir.join("state.json.tmp"), &new_state).unwrap();
            }
            // Archived, not yet renamed: no state.json
            3 => {
                prepare(&b.dir);
                std::fs::write(b.dir.join("state.json.tmp"), &new_state).unwrap();
                std::fs::rename(b.dir.join("state.json"), b.dir.join(archive_name())).unwrap();
            }
            // Written, not committed
            4 => {
                prepare(&b.dir);
                std::fs::write(b.dir.join("state.json"), &new_state).unwrap();
            }
            // Done
            _ => {
                std::fs::remove_dir_all(&b.dir).unwrap();
                copy_dir(&after, &b.dir);
            }
        }
        crate::promote::promote_tmp_states(&state_dir, later());
        let now = hash_of(&state_dir, b.id).await;
        let expect = if stage <= 2 { old } else { new };
        assert_eq!(now, Some(expect), "stage {stage}");
        let out = check_at(&state_dir, &b).await;
        assert!(
            matches!(out, Outcome::Run { hold: None, .. }),
            "stage {stage}: {out:?}"
        );
    }
}

/// The save's tmp torn (not a whole envelope) after the archive: never
/// promoted; the state is missing, the run refused, and a restore puts
/// back the last sealed version.
#[tokio::test]
async fn a_torn_tmp_is_never_promoted_and_restore_recovers() {
    let mut h = plain();
    h.upload = 9;
    let b = build(&h).await;
    let state_dir = b.state_dir();
    let head = b.seal().head().await.unwrap().unwrap();
    std::fs::write(b.dir.join("state.json.tmp"), b"{\"format\": 2, \"st").unwrap();
    std::fs::rename(b.dir.join("state.json"), b.dir.join(archive_name())).unwrap();
    assert_eq!(crate::promote::promote_tmp_states(&state_dir, later()), 0);
    let missing =
        check::missing_states(&state_dir, &b.keys, &b.api, &crate::alerts::Alerter::off()).await;
    assert_eq!(missing, vec![(b.id, Refusal::StateMissing)]);
    super::restore::restore(&state_dir, &b.keys, &b.api, b.id, None, false)
        .await
        .unwrap();
    assert_eq!(
        hash_of(&state_dir, b.id).await,
        Some(head.plaintext_sha256.unwrap())
    );
    assert!(matches!(
        check_at(&state_dir, &b).await,
        Outcome::Run { hold: None, .. }
    ));
}

/// A restore cut at each of its steps, then a restart: the start check
/// still refuses (or finds the restored state intact), nothing is lost,
/// and running restore again finishes it.
#[tokio::test]
async fn a_restore_cut_at_any_step_finishes_on_a_rerun() {
    for lost_dir in [false, true] {
        let points: &[&'static str] = if lost_dir {
            &[
                "restore:adopted",
                "restore:noticed",
                "restore:before_write",
                "restore:written",
                "restore:enrolled",
            ]
        } else {
            &[
                "restore:noticed",
                "restore:before_write",
                "restore:written",
                "restore:enrolled",
            ]
        };
        for point in points {
            let mut h = plain();
            h.upload = 9;
            let b = build(&h).await;
            let state_dir = b.state_dir();
            let head = b.seal().head().await.unwrap().unwrap();
            if lost_dir {
                std::fs::rename(&b.dir, b.root.join("lost")).unwrap();
            } else {
                FsStorage::new(&state_dir)
                    .save_sealed_raw(b.id, serde_json::json!({ "memory": "" }))
                    .await
                    .unwrap();
                let Outcome::Refuse { reason, detail } = check_at(&state_dir, &b).await else {
                    panic!("not refused")
                };
                check::refuse(
                    &b.dir,
                    b.id,
                    "tarn",
                    reason,
                    &detail,
                    &crate::alerts::Alerter::off(),
                )
                .await;
            }
            super::crash::at(point);
            let e = super::restore::restore(&state_dir, &b.keys, &b.api, b.id, None, false)
                .await
                .unwrap_err();
            assert!(
                format!("{e:#}").contains("simulated crash"),
                "{point}: {e:#}"
            );
            // The restart
            crate::promote::promote_tmp_states(&state_dir, later());
            if !lost_dir {
                // Still refused until the restore finishes
                assert!(b.seal().is_refused().await.unwrap(), "{point}");
                let wiped = std::fs::read_dir(&b.dir)
                    .unwrap()
                    .filter(|e| {
                        e.as_ref()
                            .unwrap()
                            .file_name()
                            .to_string_lossy()
                            .starts_with("state.")
                    })
                    .count();
                assert!(wiped >= 1, "{point}: the state found is kept");
            }
            let rerun =
                super::restore::restore(&state_dir, &b.keys, &b.api, b.id, None, false).await;
            // Cut after its last step for an agent that wasn't refused: done,
            // and a rerun won't overwrite the state it restored
            if lost_dir && *point == "restore:enrolled" {
                let e = format!("{:#}", rerun.unwrap_err());
                assert!(e.contains("won't overwrite"), "{e}");
            } else {
                rerun.unwrap_or_else(|e| panic!("{lost_dir} {point}: {e:#}"));
            }
            if lost_dir {
                let line = super::notice::Notice::load(&b.dir)
                    .unwrap()
                    .unwrap()
                    .line()
                    .unwrap();
                assert!(
                    line.starts_with("Your state was lost here."),
                    "{point}: {line}"
                );
            }
            assert_eq!(
                hash_of(&state_dir, b.id).await,
                Some(head.plaintext_sha256.unwrap()),
                "{point}"
            );
            let out = check_at(&state_dir, &b).await;
            assert!(
                matches!(out, Outcome::Run { hold: None, .. }),
                "{lost_dir} {point}: {out:?}"
            );
        }
    }
}

/// An accept cut at each of its steps, then a restart: refused again
/// (and accepted on a rerun), or intact with the note shown.
#[tokio::test]
async fn an_accept_cut_at_any_step_is_consistent() {
    for point in [
        "accept:diffed",
        "accept:noticed",
        "accept:cleared",
        "accept:appended",
    ] {
        let b = build(&plain()).await;
        let state_dir = b.state_dir();
        FsStorage::new(&state_dir)
            .save_sealed_raw(b.id, serde_json::json!({ "memory": "edited by hand" }))
            .await
            .unwrap();
        let edited = hash_of(&state_dir, b.id).await.unwrap();
        let api = FakeApi::new();
        let refuse = async || {
            let out = check_at(&state_dir, &b).await;
            if let Outcome::Refuse { reason, detail } = &out {
                check::refuse(
                    &b.dir,
                    b.id,
                    "tarn",
                    *reason,
                    detail,
                    &crate::alerts::Alerter::off(),
                )
                .await;
            }
            out
        };
        assert!(matches!(refuse().await, Outcome::Refuse { .. }));
        super::crash::at(point);
        let e = super::accept::accept(&state_dir, &b.keys, &api, b.id, Some("by hand"))
            .await
            .unwrap_err();
        assert!(
            format!("{e:#}").contains("simulated crash"),
            "{point}: {e:#}"
        );
        // The edited state is never touched
        assert_eq!(hash_of(&state_dir, b.id).await, Some(edited), "{point}");
        let out = refuse().await;
        if point == "accept:appended" {
            let Outcome::Run { seat, hold, .. } = &out else {
                panic!("{point}: {out:?}")
            };
            assert!(hold.is_none() && seat[0].contains("by hand"), "{seat:?}");
        } else {
            assert!(matches!(out, Outcome::Refuse { .. }), "{point}: {out:?}");
            super::accept::accept(&state_dir, &b.keys, &api, b.id, Some("by hand"))
                .await
                .unwrap_or_else(|e| panic!("{point}: {e:#}"));
            let out = refuse().await;
            let Outcome::Run { seat, hold, .. } = &out else {
                panic!("{point}: {out:?}")
            };
            assert!(
                hold.is_none() && seat[0].contains("by hand"),
                "{point}: {seat:?}"
            );
        }
    }
}

/// What a proof Agora serves is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProofKind {
    Genuine,
    /// Agora's root, not the governance log's
    ForgedRoot,
    /// A path that doesn't reach the root
    ForgedPath,
    /// Another agent's leaf, said to be ours
    OtherAgent,
    /// An entry of another chain
    OtherChain,
    /// Our entry at the wrong seq
    WrongSeq,
}

fn proof_kind() -> impl Strategy<Value = ProofKind> {
    prop_oneof![
        3 => Just(ProofKind::Genuine),
        1 => Just(ProofKind::ForgedRoot),
        1 => Just(ProofKind::ForgedPath),
        1 => Just(ProofKind::OtherAgent),
        1 => Just(ProofKind::OtherChain),
        1 => Just(ProofKind::WrongSeq),
    ]
}

/// One ANC entry with `leaves` agents, ours among them at `at`.
fn anc(
    n: u32,
    id: AgentId,
    head: Sha256Hex,
    seq: u64,
    leaves: usize,
    at: usize,
    kind: ProofKind,
) -> (AnchorProof, GovernanceLogId, Sha256Hex) {
    use agora_agentkit::state::anchor::{anchor_leaf, inclusion_proof, merkle_root};
    let other = AgentId::new();
    let mut all: Vec<Sha256Hex> = (0..leaves)
        .map(|i| anchor_leaf(AgentId::new(), &Sha256Hex::from([i as u8; 32])))
        .collect();
    let at = at % (leaves + 1);
    let (leaf_agent, leaf_head) = match kind {
        ProofKind::OtherAgent => (other, head),
        ProofKind::OtherChain => (id, Sha256Hex::from([0xee; 32])),
        _ => (id, head),
    };
    all.insert(at, anchor_leaf(leaf_agent, &leaf_head));
    let root = merkle_root(&all);
    let anc_id: GovernanceLogId = format!("ANC-2026-{n:04}").parse().unwrap();
    // After the agent's versions, so a genuine proof is one it may hold
    let date = chrono::Utc::now().date_naive() + chrono::Days::new(1 + n as u64);
    let attested = super::anchors::anchor_data_hash(&AnchorData {
        date,
        root,
        agents: all.len() as u32,
    })
    .unwrap();
    let mut proof = AnchorProof {
        anc_id: anc_id.clone(),
        date,
        root,
        entry_hash: leaf_head,
        seq,
        leaf_index: at as u32,
        leaf_count: all.len() as u32,
        proof: inclusion_proof(&all, at).unwrap(),
    };
    match kind {
        ProofKind::ForgedRoot => proof.root = Sha256Hex::from([7; 32]),
        ProofKind::ForgedPath if !proof.proof.is_empty() => {
            proof.proof[0].sibling = Sha256Hex::from([8; 32])
        }
        ProofKind::ForgedPath => proof.leaf_count += 1,
        ProofKind::OtherAgent => proof.entry_hash = head,
        ProofKind::WrongSeq => proof.seq += 1,
        _ => {}
    }
    (proof, anc_id, attested)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Random ANCs, random agent counts, genuine and forged proofs in any
    /// order: each anchor is judged once, in order, in one pass; only
    /// genuine own proofs are stored, each judged to hold the version it
    /// should (uploaded with their times recorded); every forged one is a
    /// failure of its own kind, and none stops the ones after it.
    #[test]
    fn each_anchor_is_judged_once_and_only_genuine_proofs_are_stored(
        kinds in proptest::collection::vec((proof_kind(), 0usize..20, 0usize..20), 1..8)
    ) {
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
            let mut h = plain();
            h.upload = 9;
            let b = build(&h).await;
            let head = b.seal().head().await.unwrap().unwrap().head;
            let signing = b.keys.signing(b.id).unwrap();
            let mut served = Vec::new();
            let mut log = Vec::new();
            for (i, (kind, leaves, at)) in kinds.iter().enumerate() {
                let (proof, anc_id, attested) =
                    anc(i as u32 + 2, b.id, head.entry_hash, head.seq, *leaves, *at, *kind);
                served.push((*kind, proof));
                log.push((anc_id, attested, chrono::Utc::now()));
            }
            let verified = super::anchors::VerifiedAnchors::new(log.clone());
            let genuine: Vec<AnchorProof> = served
                .iter()
                .filter(|(k, _)| *k == ProofKind::Genuine)
                .map(|(_, p)| p.clone())
                .collect();
            *b.api.proofs.lock().unwrap() = served.iter().map(|(_, p)| p.clone()).collect();
            let pass = super::anchors::check_anchors(&b.dir, b.id, &b.keys, &signing, &b.api, &verified, true)
                .await
                .unwrap();
            prop_assert_eq!(pass.judged, served.len());
            let judged = super::inclusion::InclusionFile::load(&b.dir).unwrap().checked;
            for ((kind, proof), c) in served.iter().zip(&judged) {
                prop_assert_eq!(&c.anc_id, &proof.anc_id);
                let expect: &[&str] = match kind {
                    ProofKind::Genuine => &["holds"],
                    ProofKind::ForgedRoot => &["not_the_logs"],
                    // An empty path is forged by its count, which the log's
                    // data hash covers
                    ProofKind::ForgedPath => &["did_not_check", "not_the_logs"],
                    ProofKind::OtherAgent => &["did_not_check"],
                    ProofKind::OtherChain | ProofKind::WrongSeq => &["not_on_line"],
                };
                prop_assert!(expect.contains(&c.finding.kind()), "{:?}: {:?}", kind, c.finding);
                prop_assert_eq!(c.finding.failed(), *kind != ProofKind::Genuine);
            }
            prop_assert_eq!(pass.failures.len(), served.len() - genuine.len());
            let stored = super::anchors::AnchorsFile::load(&b.dir).unwrap().proofs;
            prop_assert_eq!(stored, genuine);
            // A second pass judges nothing again
            let again = super::anchors::check_anchors(&b.dir, b.id, &b.keys, &signing, &b.api, &verified, true)
                .await
                .unwrap();
            prop_assert_eq!(again.judged, 0);
            Ok(())
        })?;
    }
}

/// A clock stepped back between saves: the runner's saves and an accept
/// (each through agentkit's `prepare`, which floors `created_at` at the
/// parent's + 1 µs) still make a chain that verifies, with no time going
/// back (the server refuses that with 422 `time_went_back`).
#[tokio::test]
async fn a_clock_stepped_back_still_makes_a_chain_that_verifies() {
    let b = build(&plain()).await;
    let state_dir = b.state_dir();
    // The head an hour in the future: as if the clock then stepped back
    let path = b.dir.join("sealed/head.json");
    let mut head: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let future = chrono::Utc::now().timestamp_micros() + 3_600_000_000;
    head["created_at"] = serde_json::json!(future);
    std::fs::write(&path, serde_json::to_vec(&head).unwrap()).unwrap();
    let enrolment = std::sync::Arc::new(super::SealedEnrolment::new(
        &state_dir,
        b.root.join("secrets"),
    ));
    let mut store =
        agora_agentkit::state::storage::Mirrored::new(FsStorage::new(&state_dir), (), enrolment);
    let base = store.load_raw(b.id).await.unwrap().base;
    store
        .save_raw(b.id, version(50), base.as_ref())
        .await
        .unwrap();
    // And an accept on top
    FsStorage::new(&state_dir)
        .save_sealed_raw(b.id, serde_json::json!({ "memory": "edited" }))
        .await
        .unwrap();
    let Outcome::Refuse { reason, detail } = check_at(&state_dir, &b).await else {
        panic!("not refused")
    };
    check::refuse(
        &b.dir,
        b.id,
        "tarn",
        reason,
        &detail,
        &crate::alerts::Alerter::off(),
    )
    .await;
    super::accept::accept(&state_dir, &b.keys, &FakeApi::new(), b.id, Some("by hand"))
        .await
        .unwrap();
    let entries = check::local_entries(&b.seal()).await.unwrap();
    let mut line = entries.clone();
    line.sort_by_key(|e| e.entry.seq);
    for pair in line.windows(2) {
        // agentkit's one rule (`state::chain::time_went_back`): equal is allowed
        assert!(
            !agora_agentkit::state::chain::time_went_back(
                pair[0].entry.created_at,
                pair[1].entry.created_at
            ),
            "time went back"
        );
    }
    assert!(line.last().unwrap().entry.created_at >= future);
    let report = agora_agentkit::state::verify::verify_chain(
        b.id,
        &line,
        &b.keys.trusted(b.id).unwrap().unwrap(),
    );
    assert!(report.intact, "{:?}", report.problems);
}

/// Agora lost our versions (LocalMissing, held every start): `sealed-accept
/// --rebase --note` seals the state here on top of what Agora holds, or on
/// a new chain whose genesis records the break when Agora holds nothing;
/// the next start is intact. What was set aside is kept in
/// `sealed/lost/<utc>/`, past `prune_blobs`: the old head's keyframe still
/// opens with agentkit's `open_state`. Cut at every step, a rerun finishes.
#[tokio::test]
async fn agoras_lost_versions_are_resolved_and_nothing_sealed_here_is_lost() {
    let points: [Option<&'static str>; 7] = [
        None,
        Some("lost:intent"),
        Some("lost:copied"),
        Some("lost:aside"),
        Some("lost:head_aside"),
        Some("lost:genesis"),
        Some("lost:sealed"),
    ];
    for nothing_left in [false, true] {
        for point in points {
            if !nothing_left && matches!(point, Some("lost:head_aside" | "lost:genesis")) {
                continue;
            }
            let mut h = plain();
            h.saves = 2;
            h.upload = 9;
            let b = build(&h).await;
            let state_dir = b.state_dir();
            let old_head = b.seal().head().await.unwrap().unwrap();
            let server = b.server();
            b.set_server(if nothing_left {
                Vec::new()
            } else {
                server[..server.len() - 1].to_vec()
            });
            let out = check_at(&state_dir, &b).await;
            assert!(
                matches!(
                    out,
                    Outcome::Run {
                        hold: Some(super::check::Hold::HeadMissing),
                        ..
                    }
                ),
                "{nothing_left}: {out:?}"
            );
            if let Some(point) = point {
                super::crash::at(point);
                let e = super::accept::rebase(&state_dir, &b.keys, &b.api, b.id, "Agora lost it")
                    .await
                    .unwrap_err();
                assert!(
                    format!("{e:#}").contains("simulated crash"),
                    "{point}: {e:#}"
                );
            }
            let done = super::accept::rebase(&state_dir, &b.keys, &b.api, b.id, "Agora lost it")
                .await
                .unwrap_or_else(|e| panic!("{nothing_left} {point:?}: {e:#}"));
            assert_eq!(done.onto.is_none(), nothing_left);
            let out = check_at(&state_dir, &b).await;
            let Outcome::Run { seat, hold, .. } = &out else {
                panic!("{nothing_left} {point:?}: {out:?}")
            };
            assert!(hold.is_none(), "{nothing_left} {point:?}: {out:?}");
            assert!(
                seat[0].contains("Agora had lost")
                    && seat[0].contains("Kept by your runner: the full copy of your state from "),
                "{seat:?}"
            );
            // Kept past a prune: the old keyframe opens
            b.seal().prune_blobs().await.unwrap();
            let lost = std::fs::read_dir(b.dir.join("sealed/lost"))
                .unwrap()
                .map(|e| e.unwrap().path())
                .collect::<Vec<_>>();
            assert_eq!(lost.len(), 1);
            assert!(lost[0].join("DONE").exists() && !lost[0].join("INTENT").exists());
            let kept = AgentSeal::new(&lost[0], b.id);
            let kept_head = kept.head().await.unwrap().unwrap();
            assert_eq!(kept_head.head, old_head.head);
            let keyframe = kept_head.keyframe.unwrap().entry;
            let opened = kept
                .open_state(&b.keys.seal_keys(b.id).unwrap(), &keyframe, None)
                .await
                .unwrap();
            assert!(opened.get("memory").is_some());
            let records = std::fs::read_dir(lost[0].join("items")).unwrap().count();
            assert!(records >= 1, "the set-aside versions' records");
            if nothing_left {
                let line = check::local_entries(&b.seal()).await.unwrap();
                assert_eq!(
                    line[0].entry.kind,
                    agora_agentkit::enums::StateEntryKind::Genesis
                );
                assert_eq!(line[0].entry.seq, 0);
            } else {
                let lost_here = std::fs::read_dir(b.dir.join("sealed/sent"))
                    .unwrap()
                    .filter(|e| {
                        e.as_ref()
                            .unwrap()
                            .file_name()
                            .to_string_lossy()
                            .ends_with(".lost")
                    })
                    .count();
                assert!(lost_here >= 1, "numbering kept in sent/");
            }
        }
    }
}

/// Agora's chain and its head disagree, or nothing is held: no rebase.
#[tokio::test]
async fn a_rebase_needs_agora_to_agree_and_the_agent_held() {
    let mut h = plain();
    h.saves = 2;
    h.upload = 9;
    let b = build(&h).await;
    let state_dir = b.state_dir();
    // Nothing lost, nothing forked
    let e = super::accept::rebase(&state_dir, &b.keys, &b.api, b.id, "n")
        .await
        .unwrap_err();
    assert!(format!("{e:#}").contains("nothing lost on Agora"), "{e:#}");
    let server = b.server();
    b.set_server(server[..server.len() - 1].to_vec());
    // The head, asked apart, still names our lost head
    *b.api.head_override.lock().unwrap() = Some(Some(server.last().unwrap().entry_hash()));
    let e = super::accept::rebase(&state_dir, &b.keys, &b.api, b.id, "n")
        .await
        .unwrap_err();
    assert!(format!("{e:#}").contains("answers disagree"), "{e:#}");
    // Nothing on Agora by the chain, a head by /head
    b.set_server(Vec::new());
    let e = super::accept::rebase(&state_dir, &b.keys, &b.api, b.id, "n")
        .await
        .unwrap_err();
    assert!(format!("{e:#}").contains("answers disagree"), "{e:#}");
    assert!(!b.dir.join("sealed/lost").exists(), "nothing begun");
}

/// A state.json a newer runner wrote: its own refusal, and neither restore
/// nor accept will touch it.
#[tokio::test]
async fn a_state_from_a_newer_runner_is_never_restored_over() {
    let mut h = plain();
    h.upload = 9;
    let b = build(&h).await;
    let state_dir = b.state_dir();
    std::fs::write(b.dir.join("state.json"), br#"{"format": 99, "state": {}}"#).unwrap();
    let refused =
        check::missing_states(&state_dir, &b.keys, &b.api, &crate::alerts::Alerter::off()).await;
    assert_eq!(refused, vec![(b.id, Refusal::StateTooNew)]);
    let e = super::restore::restore(&state_dir, &b.keys, &b.api, b.id, None, false)
        .await
        .unwrap_err();
    assert!(format!("{e:#}").contains("newer runner"), "{e:#}");
    let e = super::accept::accept(&state_dir, &b.keys, &b.api, b.id, Some("n"))
        .await
        .unwrap_err();
    assert!(format!("{e:#}").contains("newer runner"), "{e:#}");
    assert_eq!(
        std::fs::read(b.dir.join("state.json")).unwrap(),
        br#"{"format": 99, "state": {}}"#
    );
}

/// An enrolled agent's state.json emptied: refused with the marker (and
/// the run with it), restored with the empty file archived, intact after.
/// A session journal left behind doesn't block the restore, for a missing
/// state either.
#[tokio::test]
async fn an_unreadable_or_missing_state_is_refused_and_restorable() {
    for missing in [false, true] {
        let mut h = plain();
        h.upload = 9;
        let b = build(&h).await;
        let state_dir = b.state_dir();
        let head = b.seal().head().await.unwrap().unwrap();
        // A stalled session's journal
        let journal = FsStorage::new(&state_dir).journal(b.id);
        std::fs::write(journal.path(), b"{}\n").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        if missing {
            std::fs::rename(b.dir.join("state.json"), b.root.join("state.lost")).unwrap();
        } else {
            std::fs::write(b.dir.join("state.json"), b"").unwrap();
        }
        let refused =
            check::missing_states(&state_dir, &b.keys, &b.api, &crate::alerts::Alerter::off())
                .await;
        let reason = if missing {
            Refusal::StateMissing
        } else {
            Refusal::StateUnreadable
        };
        assert_eq!(refused, vec![(b.id, reason)]);
        assert!(b.seal().is_refused().await.unwrap(), "marked");
        super::restore::restore(&state_dir, &b.keys, &b.api, b.id, None, false)
            .await
            .unwrap_or_else(|e| panic!("{missing}: {e:#}"));
        assert_eq!(
            hash_of(&state_dir, b.id).await,
            Some(head.plaintext_sha256.unwrap())
        );
        if !missing {
            let empty = std::fs::read_dir(&b.dir)
                .unwrap()
                .map(|e| e.unwrap().path())
                .filter(|p| {
                    p.file_name()
                        .unwrap()
                        .to_string_lossy()
                        .starts_with("state.2")
                })
                .any(|p| std::fs::metadata(p).unwrap().len() == 0);
            assert!(empty, "the unreadable file is archived, not deleted");
        }
        assert!(
            b.dir
                .join("sealed")
                .join(super::anchors::PUBLISHED)
                .exists(),
            "the marker is back: a restore never re-sends the publish"
        );
        // Even with a ledger saying publish: the agent may have turned it
        // off with `config`, and its profile can't say so
        super::anchors::tests::chose(
            &b.dir,
            crate::consent::sealed_state::prompt::SealChoice::Publish,
        )
        .await;
        let signing = b.keys.signing(b.id).unwrap();
        assert_eq!(
            super::anchors::publish_if_chosen(&b.dir, b.id, &signing, &b.api).await,
            Ok(false)
        );
        assert!(b.api.anchored.lock().unwrap().is_empty(), "nothing sent");
        assert!(
            check::missing_states(&state_dir, &b.keys, &b.api, &crate::alerts::Alerter::off())
                .await
                .is_empty()
        );
        assert!(matches!(
            check_at(&state_dir, &b).await,
            Outcome::Run { hold: None, .. }
        ));
    }
}

#[test]
fn an_enrolled_agent_counts_as_enrolled_by_either_record() {
    let root = super::TestDir::new("enrolled-either");
    let id = AgentId::new();
    let (agent_dir, secrets) = (
        root.join("state").join(id.to_string()),
        root.join("secrets"),
    );
    assert!(!super::enrolled_by_keys_or_marker(&agent_dir, &secrets, id));
    std::fs::create_dir_all(secrets.join(id.to_string())).unwrap();
    std::fs::write(
        secrets
            .join(id.to_string())
            .join(super::keys::STATE_KEY_FILE),
        b"{}",
    )
    .unwrap();
    assert!(
        super::enrolled_by_keys_or_marker(&agent_dir, &secrets, id),
        "its key, its directory gone"
    );
}

/// Beside a live runner (its shared hold on runner.lock), `sealed-accept
/// --rebase` refuses before reading or writing anything: no start check,
/// no size record, no resolution begun.
#[tokio::test]
async fn a_rebase_refuses_beside_a_live_runner_and_writes_nothing() {
    let mut h = plain();
    h.saves = 2;
    h.upload = 9;
    let b = build(&h).await;
    let state_dir = b.state_dir();
    let server = b.server();
    b.set_server(server[..server.len() - 1].to_vec());
    let snapshot = |dir: &Path| {
        let mut all = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap() {
                let e = e.unwrap();
                if e.file_type().unwrap().is_dir() {
                    stack.push(e.path());
                } else if e.file_name() != "lock" {
                    all.push((e.path(), std::fs::read(e.path()).unwrap()));
                }
            }
        }
        all.sort();
        all
    };
    let before = snapshot(&b.dir);
    let runner = super::session::runner_shared(&b.root).unwrap();
    let e = super::accept::rebase(&state_dir, &b.keys, &b.api, b.id, "Agora lost it")
        .await
        .unwrap_err();
    assert!(format!("{e:#}").contains("stop the runner"), "{e:#}");
    assert_eq!(snapshot(&b.dir), before, "nothing written");
    assert!(
        b.api.calls.lock().unwrap().is_empty(),
        "nothing asked of Agora"
    );
    drop(runner);
    super::accept::rebase(&state_dir, &b.keys, &b.api, b.id, "Agora lost it")
        .await
        .unwrap();
}
