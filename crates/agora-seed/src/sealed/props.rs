//! Property tests over random sealed histories (Steward, 2026-11-03: "lots
//! of coverage so these faults are as close to impossible as we can make
//! them").
//!
//! A history is a chain of versions saved through agentkit's `Mirrored`,
//! some uploaded to a fake Agora, with an optional key rotation, and then
//! a plain line, a rebase onto a fork on Agora, or a whole-directory loss
//! restored from Agora. One fault from a catalogue is then applied, and
//! the start check must give exactly the class expected for that fault:
//! never a silent intact for a real fault, never a refusal for none.
//! Synthetic states only, in temp dirs.
//!
//! Run with the rest: 64 cases per property, about 15 s for this module
//! and `crash_tests` together (in `/dev/shm`: the seal fsyncs a lot). A
//! deeper run: `PROPTEST_CASES=256 cargo test -p agora-seed sealed::props`
//! takes about a minute.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use agora_agentkit::govlog::Sha256Hex;
use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::{FsStorage, Storage};
use agora_agentkit::state::SignedStateEntry;
use agora_agentkit::state::storage::{AgentSeal, Mirrored, plaintext_hash};
use proptest::prelude::*;

use super::KeyStore;
use super::check::{self, FaultKind, Hold, Outcome, local_entries};
use super::notice::Refusal;
use super::sizes::Sizes;

/// How the history ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Saves, some uploaded.
    Plain,
    /// A fork on Agora, resolved with `sealed-accept --rebase`: the main
    /// line then runs through an entry only Agora holds.
    Rebase,
    /// Everything uploaded, the whole directory lost and restored from
    /// Agora: every entry on the line is Agora's only.
    Restore,
}

#[derive(Debug, Clone)]
pub struct History {
    /// Saves after enrolment's first version.
    pub saves: usize,
    /// How many of the waiting entries are uploaded (clamped).
    pub upload: usize,
    pub rotate: bool,
    pub mode: Mode,
    /// Upload the rebased entry too.
    pub upload_rebased: bool,
}

pub fn history() -> impl Strategy<Value = History> {
    (
        0usize..4,
        0usize..6,
        any::<bool>(),
        prop_oneof![Just(Mode::Plain), Just(Mode::Rebase), Just(Mode::Restore)],
        any::<bool>(),
    )
        .prop_map(|(saves, upload, rotate, mode, upload_rebased)| History {
            saves,
            upload,
            rotate,
            mode,
            upload_rebased,
        })
}

/// A built history: the agent, its keys, and the fake Agora's chain.
pub struct Built {
    pub root: super::TestDir,
    pub id: AgentId,
    pub keys: KeyStore,
    pub dir: PathBuf,
    /// Strict: a call for an agent that doesn't upload fails the test.
    pub api: super::api::fake::Guarded,
    pub rotated: bool,
    pub mode: Mode,
}

impl Built {
    pub fn state_dir(&self) -> PathBuf {
        self.root.join("state")
    }

    pub fn server(&self) -> Vec<SignedStateEntry> {
        self.api.chain.lock().unwrap().clone().unwrap_or_default()
    }

    pub fn set_server(&self, chain: Vec<SignedStateEntry>) {
        *self.api.chain.lock().unwrap() = Ok(chain);
    }

    pub fn seal(&self) -> AgentSeal {
        AgentSeal::new(&self.dir, self.id)
    }

    pub async fn local_hash(&self) -> Sha256Hex {
        plaintext_hash(
            &FsStorage::new(self.state_dir())
                .load_raw(self.id)
                .await
                .unwrap()
                .value,
        )
    }

    pub async fn check(&self) -> Outcome {
        check::check(
            &self.dir,
            self.id,
            &self.local_hash().await,
            Sizes::default(),
            &self.keys,
            &self.api,
        )
        .await
    }

    /// The entries the line holds that only Agora has.
    pub async fn server_only(&self) -> Vec<Sha256Hex> {
        let ours: std::collections::HashSet<Sha256Hex> = local_entries(&self.seal())
            .await
            .unwrap()
            .iter()
            .map(|e| e.entry_hash())
            .collect();
        self.server()
            .iter()
            .map(|e| e.entry_hash())
            .filter(|h| !ours.contains(h))
            .collect()
    }

    pub async fn sent(&self) -> Vec<Sha256Hex> {
        self.seal()
            .sent()
            .await
            .unwrap()
            .iter()
            .map(|i| i.entry.entry_hash())
            .collect()
    }
}

pub fn version(n: usize) -> serde_json::Value {
    serde_json::json!({
        "memory": { "content": format!("synthetic version {n}\nline two of {n}\n") },
        "soul": { "name": "tarn", "n": n },
    })
}

pub async fn save(root: &Path, id: AgentId, value: serde_json::Value) {
    let enrolment = Arc::new(super::SealedEnrolment::new(
        root.join("state"),
        root.join("secrets"),
    ));
    let mut store = Mirrored::new(FsStorage::new(root.join("state")), (), enrolment);
    let base = store.load_raw(id).await.unwrap().base;
    store.save_raw(id, value, base.as_ref()).await.unwrap();
}

/// Upload up to `n` waiting entries: sent here, on Agora, blobs too, with
/// their upload times recorded as the drainer records them (this process
/// committed them all).
async fn upload(b: &Built, n: usize) {
    let seal = b.seal();
    let before = super::receipts::before(&b.dir, b.id, 0).await.unwrap();
    let mut server = b.server();
    for item in seal.outbox().await.unwrap().into_iter().take(n) {
        seal.mark_sent(item.ord).await.unwrap();
        server.push(item.entry);
    }
    b.set_server(server);
    super::receipts::after(&b.dir, b.id, 0, &before)
        .await
        .unwrap();
    copy_blobs(b);
}

fn copy_blobs(b: &Built) {
    let Ok(dir) = std::fs::read_dir(b.dir.join("sealed/blobs")) else {
        return;
    };
    for f in dir {
        let f = f.unwrap();
        let sha = f.file_name().to_str().unwrap().parse().unwrap();
        b.api
            .blobs
            .lock()
            .unwrap()
            .insert(sha, std::fs::read(f.path()).unwrap());
    }
}

pub async fn build(h: &History) -> Built {
    let root = super::TestDir::in_memory("props");
    let (id, keys, dir) = check::tests::enrolled(&root).await;
    let api = super::api::fake::guarded(&root);
    *api.public_key.lock().unwrap() = Some(keys.signing(id).unwrap().verifying_key());
    let mut b = Built {
        root,
        id,
        keys,
        dir,
        api,
        rotated: false,
        mode: h.mode,
    };
    let mut n = 1;
    for _ in 0..h.saves {
        n += 1;
        save(&b.root, id, version(n)).await;
    }
    if h.rotate {
        upload(&b, usize::MAX).await;
        b.keys.ring().ensure_encryption_key(id).unwrap();
        super::rotate::rotate_keys(&b.state_dir(), &b.keys, &b.api, id, true, false)
            .await
            .unwrap();
        b.rotated = true;
        n += 1;
        save(&b.root, id, version(n)).await;
    }
    match h.mode {
        Mode::Plain => upload(&b, h.upload).await,
        Mode::Rebase => {
            // Everything but one waiting entry uploaded; Agora holds a
            // sibling of that one: a fork, rebased onto Agora's
            let waiting = b.seal().outbox().await.unwrap().len();
            if waiting == 0 {
                n += 1;
                save(&b.root, id, version(n)).await;
            }
            let waiting = b.seal().outbox().await.unwrap().len();
            upload(&b, waiting - 1).await;
            let item = b.seal().outbox().await.unwrap().remove(0);
            let mut sibling = item.entry.entry.clone();
            sibling.state_at += 1;
            let mut server = b.server();
            server.push(sibling.sign(&b.keys.signing(id).unwrap()));
            b.set_server(server);
            super::accept::rebase(&b.state_dir(), &b.keys, &b.api, id, "a property test")
                .await
                .unwrap();
            if h.upload_rebased {
                upload(&b, usize::MAX).await;
            }
        }
        Mode::Restore => {
            upload(&b, usize::MAX).await;
            std::fs::rename(&b.dir, b.root.join("lost")).unwrap();
            super::restore::restore(&b.state_dir(), &b.keys, &b.api, id, None, false)
                .await
                .unwrap();
            super::notice::Notice::shown(&b.dir).unwrap();
        }
    }
    b
}

/// One fault from the catalogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    None,
    LocalEdited,
    HeadDeleted,
    OutboxItemDeleted,
    RotationsDeleted,
    ServerUnreachable,
    ServerReordered,
    ServerDuplicated,
    /// The server's entry at this index (mod its length)
    ServerDropped(usize),
    ServerFlipped(usize),
    ServerResigned(usize),
    /// Agora serves only its first `len - 1 - k` entries
    ServerPrefix(usize),
    /// A signature changed in an entry held here (the newest)
    LocalSignatureCorrupted,
    /// Another byte of an entry held here (the newest)
    LocalBytesCorrupted,
    /// state.json gone
    StateMissing,
    /// sealed/ gone
    SealedLost,
}

pub fn fault() -> impl Strategy<Value = Fault> {
    prop_oneof![
        Just(Fault::None),
        Just(Fault::LocalEdited),
        Just(Fault::HeadDeleted),
        Just(Fault::OutboxItemDeleted),
        Just(Fault::RotationsDeleted),
        Just(Fault::ServerUnreachable),
        Just(Fault::ServerReordered),
        Just(Fault::ServerDuplicated),
        (0usize..8).prop_map(Fault::ServerDropped),
        (0usize..8).prop_map(Fault::ServerFlipped),
        (0usize..8).prop_map(Fault::ServerResigned),
        (0usize..8).prop_map(Fault::ServerPrefix),
        Just(Fault::LocalSignatureCorrupted),
        Just(Fault::LocalBytesCorrupted),
        Just(Fault::StateMissing),
        Just(Fault::SealedLost),
    ]
}

/// What the start check must give.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expect {
    /// Runs, nothing held.
    Intact,
    /// Runs, this hold.
    Held(Hold),
    /// Runs, some hold (the exact one depends on how Agora's copy now
    /// stands against the head).
    AnyHold,
    /// Refused, marked: local evidence.
    Refused(Refusal),
    /// Refused the run, unmarked.
    Fault(FaultKind),
    /// Never checked (no state, or no sealed/): the run's missing-state
    /// pass refuses it with this reason, or skips it (`None`: nothing on
    /// Agora).
    Missing(Option<Refusal>),
}

/// Apply `fault`, and say what the check must then give.
pub async fn apply(b: &Built, fault: Fault) -> Expect {
    let server = b.server();
    let server_only = b.server_only().await;
    let sent = b.sent().await;
    let through_server = b.mode != Mode::Plain;
    let pick = |i: usize| (!server.is_empty()).then(|| i % server.len());
    // What touching Agora's entry `i` means
    let touched = |i: usize, resigned: bool| {
        let hash = server[i].entry_hash();
        if server_only.contains(&hash) {
            Expect::Fault(if resigned {
                FaultKind::UntrustedKey
            } else {
                FaultKind::ServerChainBroken
            })
        } else if sent.contains(&hash) {
            if resigned {
                // Agora's copy of an entry held here differs from ours
                Expect::Fault(FaultKind::ServerChainBroken)
            } else {
                Expect::AnyHold
            }
        } else {
            Expect::Intact
        }
    };
    match fault {
        Fault::None => Expect::Intact,
        Fault::LocalEdited => {
            FsStorage::new(b.state_dir())
                .save_sealed_raw(b.id, serde_json::json!({ "memory": "edited" }))
                .await
                .unwrap();
            Expect::Refused(Refusal::Mismatch)
        }
        Fault::HeadDeleted => {
            std::fs::rename(b.dir.join("sealed/head.json"), b.root.join("head.lost")).unwrap();
            Expect::Refused(Refusal::HeadMissing)
        }
        Fault::OutboxItemDeleted => {
            let items = b.seal().outbox().await.unwrap();
            let Some(last) = items.last() else {
                return Expect::Intact;
            };
            let name = format!("{:020}.json", last.ord);
            std::fs::rename(
                b.dir.join("sealed/outbox").join(name),
                b.root.join("outbox.lost"),
            )
            .unwrap();
            Expect::Refused(Refusal::ChainBroken)
        }
        Fault::RotationsDeleted => {
            if !b.rotated {
                return Expect::Intact;
            }
            std::fs::rename(
                b.keys.agent_dir(b.id).join(super::keys::ROTATIONS_FILE),
                b.root.join("rotations.lost"),
            )
            .unwrap();
            Expect::Fault(FaultKind::UntrustedKey)
        }
        Fault::ServerUnreachable => {
            *b.api.chain.lock().unwrap() = Err("unreachable".into());
            if through_server {
                Expect::Held(Hold::Unchecked)
            } else {
                Expect::Intact
            }
        }
        Fault::ServerReordered => {
            let mut s = server.clone();
            s.reverse();
            b.set_server(s);
            Expect::Intact
        }
        Fault::ServerDuplicated => {
            let mut s = server.clone();
            s.extend(server.iter().cloned());
            b.set_server(s);
            Expect::Intact
        }
        Fault::ServerDropped(i) => {
            let Some(i) = pick(i) else {
                return Expect::Intact;
            };
            let expect = touched(i, false);
            let mut s = server.clone();
            s.remove(i);
            b.set_server(s);
            expect
        }
        Fault::ServerFlipped(i) => {
            let Some(i) = pick(i) else {
                return Expect::Intact;
            };
            let expect = touched(i, false);
            let mut s = server.clone();
            // A byte changed under the signature: another entry, unsigned
            s[i].entry.state_at ^= 1;
            b.set_server(s);
            match expect {
                // An extra entry Agora holds at a seq of ours
                Expect::Intact if sent.is_empty() => Expect::Intact,
                e => e,
            }
        }
        Fault::ServerResigned(i) => {
            let Some(i) = pick(i) else {
                return Expect::Intact;
            };
            let expect = touched(i, true);
            let mut s = server.clone();
            let (forger, _) = agora_agentkit::crypto::generate_keypair();
            s[i] = s[i].entry.clone().sign(&forger);
            b.set_server(s);
            expect
        }
        Fault::LocalSignatureCorrupted | Fault::LocalBytesCorrupted => {
            // The newest item held here: waiting, else uploaded
            let seal = b.seal();
            let newest = match seal.outbox().await.unwrap().pop() {
                Some(i) => Some(("outbox", i.ord)),
                None => seal.sent().await.unwrap().pop().map(|i| ("sent", i.ord)),
            };
            let Some((sub, ord)) = newest else {
                return Expect::Intact;
            };
            let path = b
                .dir
                .join("sealed")
                .join(sub)
                .join(format!("{ord:020}.json"));
            let mut item: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            if fault == Fault::LocalSignatureCorrupted {
                let sig = item["entry"]["signature"].as_str().unwrap().to_string();
                let flipped = if sig.starts_with('0') { "1" } else { "0" };
                item["entry"]["signature"] = serde_json::json!(format!("{flipped}{}", &sig[1..]));
            } else {
                let at = item["entry"]["entry"]["state_at"].as_i64().unwrap();
                item["entry"]["entry"]["state_at"] = serde_json::json!(at ^ 1);
            }
            std::fs::write(&path, serde_json::to_vec(&item).unwrap()).unwrap();
            Expect::Refused(Refusal::ChainBroken)
        }
        Fault::StateMissing | Fault::SealedLost => {
            let (from, reason) = if fault == Fault::StateMissing {
                (b.dir.join("state.json"), Refusal::StateMissing)
            } else {
                (b.dir.join("sealed"), Refusal::HeadMissing)
            };
            std::fs::rename(from, b.root.join("fault.lost")).unwrap();
            Expect::Missing((!server.is_empty()).then_some(reason))
        }
        Fault::ServerPrefix(k) => {
            if server.is_empty() {
                return Expect::Intact;
            }
            let keep = server.len().saturating_sub(1 + k % server.len());
            let removed: Vec<Sha256Hex> = server[keep..].iter().map(|e| e.entry_hash()).collect();
            b.set_server(server[..keep].to_vec());
            if removed.iter().any(|h| server_only.contains(h)) {
                Expect::Fault(FaultKind::ServerChainBroken)
            } else if removed.iter().any(|h| sent.contains(h)) {
                Expect::AnyHold
            } else {
                Expect::Intact
            }
        }
    }
}

/// Whether `out` is what `expect` says.
pub fn meets(out: &Outcome, expect: &Expect) -> bool {
    match (expect, out) {
        (Expect::Intact, Outcome::Run { hold: None, .. }) => true,
        (Expect::Held(h), Outcome::Run { hold: Some(g), .. }) => h == g,
        (Expect::AnyHold, Outcome::Run { hold: Some(_), .. }) => true,
        (Expect::Refused(r), Outcome::Refuse { reason, .. }) => r == reason,
        (Expect::Fault(k), Outcome::ServerFault { kind, .. }) => k == kind,
        _ => false,
    }
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Every random history checks intact, with no fault.
    #[test]
    fn every_history_checks_intact(h in history()) {
        rt().block_on(async {
            let b = build(&h).await;
            let out = b.check().await;
            prop_assert!(meets(&out, &Expect::Intact), "{h:?}: {out:?}");
            Ok(())
        })?;
    }

    /// Anchors over every random history (plain saves, a rebase onto a
    /// fork on Agora, a whole-directory restore, with or without a key
    /// rotation and an accept): an honest anchor is never a failure; a
    /// version older than the one owed is one whenever the runner's
    /// records decide it; after a restore, whose upload times are lost,
    /// nothing is said to pass.
    #[test]
    fn anchors_are_judged_soundly_over_every_history(h in history()) {
        rt().block_on(async {
            use super::anchors::{VerifiedAnchors, anchor_data_hash, check_anchors};
            use super::inclusion::{Finding, InclusionFile};
            use agora_agentkit::state::anchor::{anchor_leaf, inclusion_proof, merkle_root};
            use agora_agentkit::state::{AnchorData, AnchorProof};
            let b = build(&h).await;
            // The anchor work runs only on an empty outbox
            upload(&b, usize::MAX).await;
            let line = local_entries(&b.seal()).await.unwrap();
            let head = b.seal().head().await.unwrap().unwrap().head;
            let line = check::main_line(
                &{
                    let mut all = line;
                    all.extend(b.server());
                    all
                },
                &head.entry_hash,
            )
            .unwrap();
            let anchor = |n: u32, date: chrono::NaiveDate, entry: &SignedStateEntry| {
                let leaves = vec![
                    anchor_leaf(AgentId::new(), &Sha256Hex::from([1; 32])),
                    anchor_leaf(b.id, &entry.entry_hash()),
                ];
                let data = AnchorData { date, root: merkle_root(&leaves), agents: 2 };
                let anc: agora_agentkit::ids::GovernanceLogId =
                    format!("ANC-2026-{n:04}").parse().unwrap();
                (
                    AnchorProof {
                        anc_id: anc.clone(),
                        date,
                        root: data.root,
                        entry_hash: entry.entry_hash(),
                        seq: entry.entry.seq,
                        leaf_index: 1,
                        leaf_count: 2,
                        proof: inclusion_proof(&leaves, 1).unwrap(),
                    },
                    (anc, anchor_data_hash(&data).unwrap(), chrono::Utc::now()),
                )
            };
            let tomorrow = chrono::Utc::now().date_naive() + chrono::Days::new(1);
            let early = chrono::Utc::now().date_naive() - chrono::Days::new(3);
            let (honest, l2) = anchor(2, tomorrow, line.last().unwrap());
            let (stale, l3) = anchor(3, tomorrow, &line[0]);
            // Before the agent existed, no proof served: its data as Agora
            // serves it, hashing to what the log attests
            let (_, l4) = anchor(4, early, &line[0]);
            let leaves = vec![
                anchor_leaf(AgentId::new(), &Sha256Hex::from([1; 32])),
                anchor_leaf(b.id, &line[0].entry_hash()),
            ];
            let early_anchor = AnchorData { date: early, root: merkle_root(&leaves), agents: 2 };
            let l4 = (l4.0, anchor_data_hash(&early_anchor).unwrap(), l4.2);
            b.api.anchor_data.lock().unwrap().insert(
                l4.0.clone(),
                serde_json::to_value(&early_anchor).unwrap(),
            );
            *b.api.proofs.lock().unwrap() = vec![honest, stale];
            let verified = VerifiedAnchors::new([l2, l3, l4]);
            let signing = b.keys.signing(b.id).unwrap();
            check_anchors(&b.dir, b.id, &b.keys, &signing, &b.api, &verified, true)
                .await
                .unwrap();
            let judged = InclusionFile::load(&b.dir).unwrap().checked;
            prop_assert_eq!(judged.len(), 3);
            let timed = h.mode != Mode::Restore;
            // Honest
            prop_assert!(!judged[0].finding.failed(), "{h:?}: {:?}", judged[0]);
            if timed {
                prop_assert!(matches!(judged[0].finding, Finding::Holds { .. }), "{h:?}: {:?}", judged[0]);
            } else {
                prop_assert!(matches!(judged[0].finding, Finding::HoldsUnchecked { .. }), "{h:?}: {:?}", judged[0]);
            }
            // The genesis, though later versions were confirmed: a failure
            // when the records decide it, never a pass
            if timed && line.len() > 1 {
                prop_assert!(matches!(judged[1].finding, Finding::Older { .. }), "{h:?}: {:?}", judged[1]);
            } else if line.len() > 1 {
                prop_assert!(!matches!(judged[1].finding, Finding::Holds { .. }), "{h:?}: {:?}", judged[1]);
            }
            // Before the agent existed: nothing owed
            prop_assert_eq!(&judged[2].finding, &Finding::NotIncluded, "{:?}", h);
            Ok(())
        })?;
    }

    /// One fault: exactly the class expected for it.
    #[test]
    fn each_fault_gets_its_class(h in history(), f in fault()) {
        rt().block_on(async {
            let b = build(&h).await;
            let expect = apply(&b, f).await;
            if let Expect::Missing(reason) = expect {
                let missing = check::missing_states(
                    &b.state_dir(),
                    &b.keys,
                    &b.api,
                    &crate::alerts::Alerter::off(),
                )
                .await;
                let want: Vec<_> = reason.map(|r| (b.id, r)).into_iter().collect();
                prop_assert_eq!(missing, want, "{:?} {:?}", h, f);
                return Ok(());
            }
            let out = b.check().await;
            prop_assert!(meets(&out, &expect), "{h:?} {f:?}: expected {expect:?}, got {out:?}");
            // Never marked against the agent unless the evidence is local
            if !matches!(expect, Expect::Refused(_)) {
                prop_assert!(!b.seal().is_refused().await.unwrap());
            }
            Ok(())
        })?;
    }
}

/// A loss to restore from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loss {
    /// The state wiped (refused first, as the start check does)
    Wiped,
    StateDeleted,
    SealedDeleted,
    HeadDeleted,
    DirDeleted,
}

pub fn loss() -> impl Strategy<Value = Loss> {
    prop_oneof![
        Just(Loss::Wiped),
        Just(Loss::StateDeleted),
        Just(Loss::SealedDeleted),
        Just(Loss::HeadDeleted),
        Just(Loss::DirDeleted),
    ]
}

fn archives(dir: &Path) -> usize {
    std::fs::read_dir(dir).map_or(0, |d| {
        d.filter(|e| {
            let n = e
                .as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .to_string();
            n.starts_with("state.") && n != "state.json"
        })
        .count()
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// For any history and loss, restore either refuses (saying why) or
    /// puts back the newest version there is to restore, never an older
    /// one, the state found archived, and the next check intact.
    #[test]
    fn restore_puts_back_the_newest_version_or_refuses(h in history(), l in loss()) {
        rt().block_on(async {
            let b = build(&h).await;
            let head = b.seal().head().await.unwrap().unwrap();
            let waiting = !b.seal().outbox().await.unwrap().is_empty();
            // The newest version Agora holds, verified, for losses that
            // take sealed/ with them
            let agora_head = b.server().last().map(|e| e.entry_hash());
            let had_state = !matches!(l, Loss::StateDeleted | Loss::DirDeleted);
            match l {
                Loss::Wiped => {
                    FsStorage::new(b.state_dir())
                        .save_sealed_raw(b.id, serde_json::json!({ "memory": "" }))
                        .await
                        .unwrap();
                    let out = b.check().await;
                    let Outcome::Refuse { reason, detail } = out else {
                        return Err(TestCaseError::fail(format!("not refused: {out:?}")));
                    };
                    check::refuse(&b.dir, b.id, "tarn", reason, &detail, &crate::alerts::Alerter::off()).await;
                }
                Loss::StateDeleted => std::fs::rename(b.dir.join("state.json"), b.root.join("state.lost")).unwrap(),
                Loss::SealedDeleted => std::fs::rename(b.dir.join("sealed"), b.root.join("sealed.lost")).unwrap(),
                Loss::HeadDeleted => std::fs::rename(b.dir.join("sealed/head.json"), b.root.join("head.lost")).unwrap(),
                Loss::DirDeleted => std::fs::rename(&b.dir, b.root.join("dir.lost")).unwrap(),
            }
            let before = archives(&b.dir);
            let local_hash = if had_state { Some(b.local_hash().await) } else { None };
            let mut done = super::restore::restore(&b.state_dir(), &b.keys, &b.api, b.id, None, false).await;
            let adopts = matches!(l, Loss::SealedDeleted | Loss::HeadDeleted | Loss::DirDeleted);
            // A state here that isn't Agora's latest, its sealed/ lost: never
            // replaced silently; with --discard-local it is
            if let Err(e) = &done
                && format!("{e:#}").contains("--discard-local")
            {
                prop_assert!(adopts && had_state, "{h:?} {l:?}: {e:#}");
                prop_assert!(local_hash.is_some() && agora_head.is_some());
                done = super::restore::restore(&b.state_dir(), &b.keys, &b.api, b.id, None, true).await;
            }
            match done {
                Err(e) => {
                    let e = format!("{e:#}");
                    // Refused only for a reason: versions here Agora lacks
                    // (a rollback), or nothing on Agora to adopt
                    prop_assert!(
                        (l == Loss::HeadDeleted && waiting && e.contains("older history"))
                            || (adopts && agora_head.is_none() && e.contains("no chain"))
                            || (adopts && e.contains("no snapshot")),
                        "{h:?} {l:?}: {e}"
                    );
                }
                Ok(done) => {
                    let expected = if adopts && matches!(l, Loss::SealedDeleted | Loss::DirDeleted) {
                        agora_head.unwrap()
                    } else {
                        head.head.entry_hash
                    };
                    prop_assert_eq!(done.head.entry_hash, expected, "{:?} {:?}", h, l);
                    if expected == head.head.entry_hash {
                        prop_assert_eq!(b.local_hash().await, head.plaintext_sha256.unwrap());
                    }
                    if had_state {
                        prop_assert_eq!(archives(&b.dir), before + 1, "the state found is archived");
                    }
                    let out = b.check().await;
                    prop_assert!(matches!(out, Outcome::Run { hold: None, .. }), "{h:?} {l:?}: {out:?}");
                }
            }
            Ok(())
        })?;
    }
}

#[test]
fn every_chain_problem_has_exactly_one_class() {
    use agora_agentkit::enums::StateEntryKind;
    use agora_agentkit::state::verify::ChainProblem as P;
    use check::ProblemClass::{Chain, Key};
    let all = [
        (P::MissingGenesis, Chain),
        (P::GenesisNotFirst { seq: 1 }, Chain),
        (P::WrongAgent { seq: 1 }, Chain),
        (P::UnknownVersion { seq: 1, version: 9 }, Chain),
        (
            P::SeqGap {
                expected: 1,
                found: 2,
            },
            Chain,
        ),
        (P::Fork { seq: 1 }, Chain),
        (P::BrokenLink { seq: 1 }, Chain),
        (P::BadSignature { seq: 1 }, Key),
        (P::UntrustedKey { seq: 1 }, Key),
        (P::RetiredKey { seq: 1 }, Chain),
        (
            P::UnknownKind {
                seq: 1,
                kind: StateEntryKind::Merge,
            },
            Chain,
        ),
        (P::DeltaWithoutBase { seq: 1 }, Chain),
        (P::MissingBlob { seq: 1 }, Chain),
        (P::MergeNotAllowed { seq: 1 }, Chain),
        (P::TimeWentBack { seq: 1 }, Chain),
    ];
    for (p, class) in &all {
        assert_eq!(check::class_of(p), *class, "{p:?}");
    }
    // judge_failure places each problem by its entry
    let (key, _) = agora_agentkit::crypto::generate_keypair();
    let (stranger, _) = agora_agentkit::crypto::generate_keypair();
    let (retired, _) = agora_agentkit::crypto::generate_keypair();
    let id = AgentId::new();
    let entry = |seq: u64, by: &agora_agentkit::crypto::SigningKey| {
        agora_agentkit::state::StateEntry {
            agora_state_entry: agora_agentkit::state::chain::STATE_ENTRY_VERSION,
            agent_id: id,
            seq,
            created_at: 1_000 + seq as i64,
            state_at: 1_000,
            kind: if seq == 0 {
                StateEntryKind::Genesis
            } else {
                StateEntryKind::Snapshot
            },
            prev_hash: None,
            merged_from: vec![],
            blob: None,
            target: None,
            key_gen: 0,
        }
        .sign(by)
    };
    let line = vec![entry(0, &key), entry(1, &stranger), entry(2, &retired)];
    let agoras: std::collections::HashSet<Sha256Hex> = [line[1].entry_hash()].into();
    let none = std::collections::HashSet::new();
    let known = vec![key.verifying_key(), retired.verifying_key()];
    let judge = |problems: &[P], agoras: &std::collections::HashSet<Sha256Hex>| {
        check::judge_failure(id, problems, &line, agoras, &known)
    };
    let fault = |o: Outcome| match o {
        Outcome::ServerFault { kind, .. } => Some(kind),
        _ => None,
    };
    let refused = |o: Outcome| {
        matches!(
            o,
            Outcome::Refuse {
                reason: Refusal::ChainBroken,
                ..
            }
        )
    };
    // A key problem on Agora's entry: untrusted key, unmarked
    assert_eq!(
        fault(judge(&[P::BadSignature { seq: 1 }], &agoras)),
        Some(FaultKind::UntrustedKey)
    );
    // On an entry held here, signed by none of our keys: corrupted here
    assert!(refused(judge(&[P::BadSignature { seq: 1 }], &none)));
    // On an entry held here that a retired key of ours signed: the
    // rotation record was lost
    assert_eq!(
        fault(judge(&[P::BadSignature { seq: 2 }], &none)),
        Some(FaultKind::UntrustedKey)
    );
    // A chain problem on Agora's entry: a server fault
    assert_eq!(
        fault(judge(&[P::BrokenLink { seq: 1 }], &agoras)),
        Some(FaultKind::ServerChainBroken)
    );
    // Anything on ours refuses, whatever else
    assert!(refused(judge(
        &[P::BrokenLink { seq: 1 }, P::BrokenLink { seq: 2 }],
        &agoras
    )));
    assert!(refused(judge(&[], &agoras)));
}

// B3: agentkit 0.158 keeps acknowledged blobs (998a07b); the tests of a
// version Agora alone holds simulate an agent pruned before it
/// What agentkit's `prune_blobs` did before 0.158: delete the blobs of
/// `sent/` entries, but never the head keyframe's or one an outbox item
/// names. For tests of versions whose blob is on Agora only.
pub async fn prune_sent_blobs(seal: &AgentSeal) {
    let blobs = seal.dir().join("blobs");
    let mut keep: std::collections::HashSet<Sha256Hex> = std::collections::HashSet::new();
    if let Some(h) = seal.head().await.unwrap() {
        keep.extend(h.keyframe.as_ref().and_then(|k| k.blob_sha256()));
    }
    for item in seal.outbox().await.unwrap() {
        keep.extend(item.entry.entry.blob.as_ref().map(|b| b.sha256));
        keep.extend(item.keyframe_after.as_ref().and_then(|k| k.blob_sha256()));
    }
    for item in seal.sent().await.unwrap() {
        if let Some(b) = item.entry.entry.blob.as_ref()
            && !keep.contains(&b.sha256)
        {
            let _ = std::fs::remove_file(blobs.join(b.sha256.to_hex()));
        }
    }
}
