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
use super::api::fake::FakeApi;
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
    pub api: FakeApi,
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

async fn save(root: &Path, id: AgentId, value: serde_json::Value) {
    let enrolment = Arc::new(super::SealedEnrolment::new(
        root.join("state"),
        root.join("secrets"),
    ));
    let mut store = Mirrored::new(FsStorage::new(root.join("state")), (), enrolment);
    let base = store.load_raw(id).await.unwrap().base;
    store.save_raw(id, value, base.as_ref()).await.unwrap();
}

/// Upload up to `n` waiting entries: sent here, on Agora, blobs too.
async fn upload(b: &Built, n: usize) {
    let seal = b.seal();
    let mut server = b.server();
    for item in seal.outbox().await.unwrap().into_iter().take(n) {
        seal.mark_sent(item.ord).await.unwrap();
        server.push(item.entry);
    }
    b.set_server(server);
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
    let api = FakeApi::new();
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
            super::restore::restore(&b.state_dir(), &b.keys, &b.api, id, None)
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

    /// One fault: exactly the class expected for it.
    #[test]
    fn each_fault_gets_its_class(h in history(), f in fault()) {
        rt().block_on(async {
            let b = build(&h).await;
            let expect = apply(&b, f).await;
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
            let done = super::restore::restore(&b.state_dir(), &b.keys, &b.api, b.id, None).await;
            let adopts = matches!(l, Loss::SealedDeleted | Loss::HeadDeleted | Loss::DirDeleted);
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
    // judge_failure: keys only → untrusted key; on Agora's entries only →
    // server fault; anything on ours → refused
    let id = AgentId::new();
    let on: std::collections::HashSet<u64> = [1].into();
    let none = std::collections::HashSet::new();
    assert!(matches!(
        check::judge_failure(id, &[P::BadSignature { seq: 3 }], &none),
        Outcome::ServerFault {
            kind: FaultKind::UntrustedKey,
            ..
        }
    ));
    assert!(matches!(
        check::judge_failure(id, &[P::BrokenLink { seq: 1 }], &on),
        Outcome::ServerFault {
            kind: FaultKind::ServerChainBroken,
            ..
        }
    ));
    assert!(matches!(
        check::judge_failure(
            id,
            &[P::BrokenLink { seq: 1 }, P::BrokenLink { seq: 2 }],
            &on
        ),
        Outcome::Refuse {
            reason: Refusal::ChainBroken,
            ..
        }
    ));
    assert!(matches!(
        check::judge_failure(id, &[], &on),
        Outcome::Refuse { .. }
    ));
}
