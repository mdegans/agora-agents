//! Anchors: proof that an agent's sealed head is in a day's `ANC-` entry,
//! checked by its runner, and whether the agent's profile shows it.
//!
//! **Proofs.** The server prunes anchor leaves after 30 days, and no agent
//! can run `verify_inclusion` itself, so the runner does ("the client does
//! the work"): the drainer, at the end of a run and at most hourly during
//! it ([`PROOF_INTERVAL`]), asks Agora for each sealed agent's proofs after
//! the newest one kept, for agents whose outbox is empty. A proof is kept
//! only when
//!
//! - its `entry_hash` is one of the agent's own uploaded entries
//!   (`sealed/sent/`), at the same `seq`; and
//! - it checks ([`AnchorProof::check`]) against the [`AnchorData`] of its
//!   `ANC-` entry as the governance log gives it. The chain the runner
//!   verified at the start of the run ([`crate::govlog::verify`]) carries
//!   each entry's signed `data_hash`, not its `data`: the proof's own
//!   `date`, `root` and `leaf_count` are taken as the candidate
//!   `AnchorData` and kept only if they hash to that signed `data_hash`.
//!   So the root checked against is the log's, never the server's copy.
//!
//! Kept proofs go to `sealed/anchors.json` (local, written atomically);
//! the start check reads the newest for the verdict line, with no network.
//! A proof that fails is never stored: it is a structured WARN and
//! [`AlertKind::SealedStateAnchorFailed`]. One for an `ANC-` entry newer
//! than the verified log waits for the next run.
//!
//! **Publish.** An agent whose sealed-state ledger says `publish` gets
//! `POST /api/state/anchor {publish: true}` once, after its enrolment
//! uploads, or from the drainer for an agent enrolled before this; success
//! writes `sealed/published`. Only the agent's own choice sets it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use agora_agentkit::crypto::SigningKey;
use agora_agentkit::govlog::Sha256Hex;
use agora_agentkit::ids::{AgentId, GovernanceLogId};
use agora_agentkit::state::storage::{AgentSeal, SEALED_DIR};
use agora_agentkit::state::{AnchorData, AnchorProof};
use serde::{Deserialize, Serialize};

use super::api::SealedApi;
use crate::alerts::{Alert, AlertKind};
use crate::consent::sealed_state::ledger::SealLedger;
use crate::consent::sealed_state::prompt::SealChoice;

/// `sealed/anchors.json`.
pub const ANCHORS: &str = "anchors.json";
/// `sealed/published`: the server was told to publish the anchor.
pub const PUBLISHED: &str = "published";
/// The drainer's proof fetches while a run goes on.
pub const PROOF_INTERVAL: Duration = Duration::from_secs(3600);

/// The `ANC-` entries of the governance log verified at the start of the
/// run: each id, with the `data_hash` its signed link attests.
#[derive(Debug, Clone, Default)]
pub struct VerifiedAnchors(BTreeMap<GovernanceLogId, Sha256Hex>);

impl VerifiedAnchors {
    pub fn new(entries: impl IntoIterator<Item = (GovernanceLogId, Sha256Hex)>) -> Self {
        Self(entries.into_iter().collect())
    }

    /// The log's [`AnchorData`] for `proof`'s entry: the proof's own
    /// fields, if they hash to what the log attests. `None` when the entry
    /// isn't in the verified log; `Some(Err)` when they don't hash to it.
    fn data(&self, proof: &AnchorProof) -> Option<Result<AnchorData, ()>> {
        let attested = self.0.get(&proof.anc_id)?;
        let data = AnchorData {
            date: proof.date,
            root: proof.root,
            agents: proof.leaf_count,
        };
        Some(if anchor_data_hash(&data) == Some(*attested) {
            Ok(data)
        } else {
            Err(())
        })
    }
}

/// `data`'s hash as the governance log attests it: the log's own
/// `govlog::data_hash` (SHA-256 of its canonical JSON) over the value as
/// the server writes it (`serde_json::to_value`).
pub fn anchor_data_hash(data: &AnchorData) -> Option<Sha256Hex> {
    serde_json::to_value(data)
        .ok()
        .map(|v| agora_agentkit::govlog::data_hash(&v))
}

/// `sealed/anchors.json`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnchorsFile {
    /// Oldest first, each checked.
    #[serde(default)]
    pub proofs: Vec<AnchorProof>,
}

fn path(agent_dir: &Path) -> PathBuf {
    agent_dir.join(SEALED_DIR).join(ANCHORS)
}

impl AnchorsFile {
    /// A missing file is no proofs.
    pub fn load(agent_dir: &Path) -> std::io::Result<Self> {
        match std::fs::read(path(agent_dir)) {
            Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e),
        }
    }

    /// The newest kept proof, for the verdict line. Unreadable: none, and
    /// a WARN.
    pub fn newest(agent_dir: &Path) -> Option<AnchorProof> {
        match Self::load(agent_dir) {
            Ok(file) => file.proofs.last().cloned(),
            Err(e) => {
                tracing::warn!(
                    path = %path(agent_dir).display(),
                    error = %e,
                    "sealed/anchors.json unreadable; no anchor named"
                );
                None
            }
        }
    }

    fn save(&self, agent_dir: &Path) -> std::io::Result<()> {
        write_atomic(&path(agent_dir), &serde_json::to_vec_pretty(self)?)
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let dir = path.parent().expect("in sealed/");
    std::fs::create_dir_all(dir)?;
    let tmp = path.with_extension("writing");
    std::fs::write(&tmp, bytes)?;
    std::fs::File::open(&tmp)?.sync_all()?;
    std::fs::rename(&tmp, path)?;
    std::fs::File::open(dir)?.sync_all()
}

/// Why a proof wasn't kept, or the fetch failed. Ids and hashes only.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProofError {
    #[error("the proofs could not be fetched: {0}")]
    Fetch(String),
    #[error("{anc_id}'s date, root and count don't hash to the governance log's entry")]
    NotTheLogs { anc_id: GovernanceLogId },
    #[error("{anc_id}'s inclusion proof does not reach the governance log's root")]
    DidNotCheck { anc_id: GovernanceLogId },
    #[error("sealed/ unreadable: {0}")]
    Local(String),
    #[error("the agent's main line couldn't be read to place a proof: {0}")]
    Line(String),
}

impl ProofError {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Fetch(_) => "fetch",
            Self::NotTheLogs { .. } => "not_the_logs",
            Self::DidNotCheck { .. } => "did_not_check",
            Self::Local(_) => "local",
            Self::Line(_) => "line",
        }
    }
}

/// Fetch `id`'s proofs after the newest kept, and keep those that check,
/// in order, up to the first that doesn't (stored before it is returned),
/// the first whose entry the verified log doesn't hold yet, or a main line
/// that can't be read. A proof for an entry not the agent's is skipped and
/// counted.
pub async fn fetch_proofs(
    agent_dir: &Path,
    id: AgentId,
    keys: &super::KeyStore,
    signing: &SigningKey,
    api: &dyn SealedApi,
    verified: &VerifiedAnchors,
) -> Result<Fetched, ProofError> {
    let mut file = AnchorsFile::load(agent_dir).map_err(|e| ProofError::Local(e.to_string()))?;
    let after = file.proofs.last().map(|p| p.anc_id.clone());
    let proofs = api
        .proofs(id, signing, after)
        .await
        .map_err(ProofError::Fetch)?;
    if proofs.is_empty() {
        return Ok(Fetched::default());
    }
    // The agent's own entries: uploaded here, and (asked once, only when
    // a proof names one not here) its verified main line, which runs
    // through Agora's entries after a rebase or a restore
    let seal = AgentSeal::new(agent_dir, id);
    let mut own: std::collections::HashMap<Sha256Hex, u64> = seal
        .sent()
        .await
        .map_err(|e| ProofError::Local(e.to_string()))?
        .into_iter()
        .map(|i| (i.entry.entry_hash(), i.entry.entry.seq))
        .collect();
    let mut line_read = false;
    let mut not_own = 0;
    let mut kept = 0;
    let mut failed = None;
    for proof in proofs {
        if file.proofs.iter().any(|p| p.anc_id == proof.anc_id) {
            continue;
        }
        if own.get(&proof.entry_hash) != Some(&proof.seq) && !line_read {
            line_read = true;
            let head = seal
                .head()
                .await
                .map_err(|e| ProofError::Local(e.to_string()))?;
            let line = match head {
                Some(head) => super::version::verified_line(&seal, keys, api, id, &head)
                    .await
                    .map_err(|e| format!("{e:#}")),
                None => Err("sealed/head.json is missing".into()),
            };
            match line {
                Ok(line) => own.extend(line.iter().map(|e| (e.entry_hash(), e.entry.seq))),
                // Can't place it: stop here, the cursor where it was
                Err(e) => {
                    failed = Some(ProofError::Line(e));
                    break;
                }
            }
        }
        // Not the agent's: skipped and never stored, but no reason to stop
        // the ones after it
        if own.get(&proof.entry_hash) != Some(&proof.seq) {
            tracing::warn!(
                event_type = "sealed_state_proof_not_own",
                agent_id = %id,
                anc_id = %proof.anc_id,
                entry = %proof.entry_hash,
                seq = proof.seq,
                "a proof names an entry that isn't on the agent's verified line; skipped"
            );
            not_own += 1;
            continue;
        }
        let data = match verified.data(&proof) {
            None => {
                tracing::info!(
                    event_type = "sealed_state_proof_waits",
                    agent_id = %id,
                    anc_id = %proof.anc_id,
                    "a proof for an ANC entry newer than the verified governance log; next run"
                );
                break;
            }
            Some(Err(())) => {
                failed = Some(ProofError::NotTheLogs {
                    anc_id: proof.anc_id,
                });
                break;
            }
            Some(Ok(data)) => data,
        };
        if !proof.check(id, &data) {
            failed = Some(ProofError::DidNotCheck {
                anc_id: proof.anc_id,
            });
            break;
        }
        tracing::info!(
            event_type = "sealed_state_proof_kept",
            agent_id = %id,
            anc_id = %proof.anc_id,
            date = %proof.date,
            seq = proof.seq,
            "anchor proof checked against the governance log and kept"
        );
        file.proofs.push(proof);
        kept += 1;
    }
    if kept > 0 {
        file.save(agent_dir)
            .map_err(|e| ProofError::Local(e.to_string()))?;
    }
    match failed {
        Some(e) => Err(e),
        None => Ok(Fetched { kept, not_own }),
    }
}

/// What one [`fetch_proofs`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Fetched {
    /// Checked and stored.
    pub kept: usize,
    /// Proofs Agora attributes to the agent for entries that aren't on its
    /// verified line: skipped, never stored, alerted.
    pub not_own: usize,
}

/// Tell Agora to publish `id`'s anchor, once, if its ledger says
/// `publish`. `Ok(true)`: told now.
pub async fn publish_if_chosen(
    agent_dir: &Path,
    id: AgentId,
    signing: &SigningKey,
    api: &dyn SealedApi,
) -> Result<bool, String> {
    let marker = agent_dir.join(SEALED_DIR).join(PUBLISHED);
    if std::fs::exists(&marker).map_err(|e| e.to_string())? {
        return Ok(false);
    }
    let ledger = SealLedger::load(agent_dir)
        .await
        .map_err(|e| format!("sealed-state ledger unreadable: {e}"))?;
    if ledger.choice() != Some(SealChoice::Publish) {
        return Ok(false);
    }
    api.post_anchor(id, signing, true).await?;
    write_atomic(&marker, chrono::Utc::now().to_rfc3339().as_bytes())
        .map_err(|e| format!("told Agora, but sealed/published not written: {e}"))?;
    tracing::info!(
        event_type = "sealed_state_anchor_published",
        agent_id = %id,
        "the agent chose publish: its anchored head is shown on its profile"
    );
    Ok(true)
}

/// The alert for a proof or publish that failed.
pub fn anchor_alert(id: AgentId, name: &str, what: &str, error: &str) -> Alert {
    Alert::new(
        AlertKind::SealedStateAnchorFailed,
        format!("{name}: sealed-state anchor {what} failed"),
    )
    .agent(name, id)
    .detail("what", what)
    .detail("error", error)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::sealed::api::fake::FakeApi;
    use crate::sealed::check::tests::enrolled;
    use agora_agentkit::state::anchor::{anchor_leaf, inclusion_proof, merkle_root};

    /// A ledger on file with `choice`.
    pub(crate) async fn chose(dir: &Path, choice: SealChoice) {
        use crate::consent::sealed_state::ledger::{SealAsk, SealOutcome};
        let mut l = SealLedger::default();
        l.record(SealAsk {
            at: chrono::Utc::now(),
            offer_version: 1,
            model: "m".into(),
            order: SealChoice::ALL,
            order_seed: 0,
            constrained: true,
            attempts: 1,
            outcome: SealOutcome::Answered {
                choice,
                reason: String::new(),
                memory_note_written: false,
                note_dropped: None,
                apply_failed: None,
            },
        });
        l.save(dir).await.unwrap();
    }

    /// Every entry uploaded.
    async fn all_sent(dir: &Path, id: AgentId) {
        let seal = AgentSeal::new(dir, id);
        for item in seal.outbox().await.unwrap() {
            seal.mark_sent(item.ord).await.unwrap();
        }
    }

    /// An `ANC-` entry over `id`'s `head` among others, its proof, and
    /// the log's attested data hash.
    fn anchor(
        n: u32,
        id: AgentId,
        head: Sha256Hex,
        seq: u64,
    ) -> (AnchorProof, GovernanceLogId, Sha256Hex) {
        let mut leaves: Vec<Sha256Hex> = (0..4u8)
            .map(|i| anchor_leaf(AgentId::new(), &Sha256Hex::from([i; 32])))
            .collect();
        leaves.insert(2, anchor_leaf(id, &head));
        let root = merkle_root(&leaves);
        let anc_id: GovernanceLogId = format!("ANC-2026-{n:04}").parse().unwrap();
        let date = chrono::NaiveDate::from_ymd_opt(2026, 11, n).unwrap();
        let data = AnchorData {
            date,
            root,
            agents: leaves.len() as u32,
        };
        let attested = anchor_data_hash(&data).unwrap();
        (
            AnchorProof {
                anc_id: anc_id.clone(),
                date,
                root,
                entry_hash: head,
                seq,
                leaf_index: 2,
                leaf_count: leaves.len() as u32,
                proof: inclusion_proof(&leaves, 2).unwrap(),
            },
            anc_id,
            attested,
        )
    }

    async fn head(dir: &Path, id: AgentId) -> (Sha256Hex, u64) {
        let h = AgentSeal::new(dir, id).head().await.unwrap().unwrap().head;
        (h.entry_hash, h.seq)
    }

    #[tokio::test]
    async fn a_proof_is_kept_only_when_it_checks_and_is_the_agents_own() {
        let root = crate::sealed::TestDir::new("anchors");
        let (id, keys, dir) = enrolled(&root).await;
        all_sent(&dir, id).await;
        let signing = keys.signing(id).unwrap();
        let (hash, seq) = head(&dir, id).await;
        let (good, anc, attested) = anchor(2, id, hash, seq);
        let api = FakeApi::new();
        api.proofs.lock().unwrap().push(good.clone());

        // Not in the verified log yet: waits, nothing stored
        let none = VerifiedAnchors::default();
        assert_eq!(
            fetch_proofs(&dir, id, &keys, &signing, &api, &none).await,
            Ok(Fetched {
                kept: 0,
                not_own: 0
            })
        );
        assert!(AnchorsFile::newest(&dir).is_none());

        let verified = VerifiedAnchors::new([(anc.clone(), attested)]);
        assert_eq!(
            fetch_proofs(&dir, id, &keys, &signing, &api, &verified).await,
            Ok(Fetched {
                kept: 1,
                not_own: 0
            })
        );
        assert_eq!(AnchorsFile::newest(&dir), Some(good.clone()));
        // Asked after the newest kept: nothing new
        assert_eq!(
            fetch_proofs(&dir, id, &keys, &signing, &api, &verified).await,
            Ok(Fetched {
                kept: 0,
                not_own: 0
            })
        );

        // Someone else's head at our seq, then ours at a wrong seq: each
        // skipped, never stored, and the next good proof still kept
        let (theirs, anc3, attested3) = anchor(3, id, Sha256Hex::from([9; 32]), seq);
        let mut wrong_seq = good.clone();
        wrong_seq.anc_id = "ANC-2026-0004".parse().unwrap();
        wrong_seq.seq += 1;
        let (later, anc5, attested5) = anchor(5, id, hash, seq);
        *api.proofs.lock().unwrap() = vec![good.clone(), theirs, wrong_seq, later.clone()];
        let verified = VerifiedAnchors::new([
            (anc.clone(), attested),
            (anc3, attested3),
            (anc5, attested5),
        ]);
        assert_eq!(
            fetch_proofs(&dir, id, &keys, &signing, &api, &verified).await,
            Ok(Fetched {
                kept: 1,
                not_own: 2
            }),
            "the loop goes on past what isn't ours"
        );
        assert_eq!(AnchorsFile::load(&dir).unwrap().proofs, vec![good, later]);
    }

    /// ANC-2026-0001 as Agora serves it (data and attested data_hash, read
    /// 2026-11-03): the AnchorData rebuilt from a proof's fields hashes to
    /// the signed data_hash, with the log's own function, as served.
    #[test]
    fn a_real_anc_entry_round_trips_to_its_data_hash() {
        let served = serde_json::json!({
            "date": "2026-11-02",
            "root": "a8a1d4d25b88fd3a5b968f885635c90cf85166e75b15d7aa08f8d3753a4970f1",
            "agents": 3,
        });
        let attested: Sha256Hex =
            "8793c2a680e8d1225c24601f86f810d8d51baf365dbb5be249c6e527ad5d6c5a"
                .parse()
                .unwrap();
        assert_eq!(agora_agentkit::govlog::data_hash(&served), attested);
        let data: AnchorData = serde_json::from_value(served).unwrap();
        assert_eq!(anchor_data_hash(&data), Some(attested));
        let anc_id: GovernanceLogId = "ANC-2026-0001".parse().unwrap();
        let proof = AnchorProof {
            anc_id: anc_id.clone(),
            date: data.date,
            root: data.root,
            entry_hash: Sha256Hex::from([0; 32]),
            seq: 0,
            leaf_index: 0,
            leaf_count: data.agents,
            proof: Vec::new(),
        };
        let verified = VerifiedAnchors::new([(anc_id, attested)]);
        assert_eq!(verified.data(&proof), Some(Ok(data)));
        let mut off_by_one = proof;
        off_by_one.leaf_count += 1;
        assert_eq!(verified.data(&off_by_one), Some(Err(())));
    }

    #[tokio::test]
    async fn a_forged_proof_is_rejected() {
        let root = crate::sealed::TestDir::new("anchors-forged");
        let (id, keys, dir) = enrolled(&root).await;
        all_sent(&dir, id).await;
        let signing = keys.signing(id).unwrap();
        let (hash, seq) = head(&dir, id).await;
        let (good, anc, attested) = anchor(2, id, hash, seq);
        let verified = VerifiedAnchors::new([(anc, attested)]);
        let api = FakeApi::new();

        // A server root that isn't the log's: refused before the path
        let mut other_root = good.clone();
        other_root.root = Sha256Hex::from([7; 32]);
        *api.proofs.lock().unwrap() = vec![other_root];
        let e = fetch_proofs(&dir, id, &keys, &signing, &api, &verified)
            .await
            .unwrap_err();
        assert_eq!(e.kind(), "not_the_logs", "{e}");

        // The log's root, a path that doesn't reach it
        let mut bad_path = good.clone();
        bad_path.proof[0].sibling = Sha256Hex::from([8; 32]);
        *api.proofs.lock().unwrap() = vec![bad_path];
        let e = fetch_proofs(&dir, id, &keys, &signing, &api, &verified)
            .await
            .unwrap_err();
        assert_eq!(e.kind(), "did_not_check", "{e}");

        assert!(AnchorsFile::newest(&dir).is_none(), "nothing stored");
        // Fetch fails: an error, nothing stored
        *api.fail.lock().unwrap() = Some("proofs");
        assert_eq!(
            fetch_proofs(&dir, id, &keys, &signing, &api, &verified)
                .await
                .unwrap_err()
                .kind(),
            "fetch"
        );
        assert!(!dir.join("sealed").join(ANCHORS).exists());
    }

    /// After a lost-directory restore sent/ is empty: a proof for the
    /// restored head is placed on the verified line through Agora and kept.
    #[tokio::test]
    async fn a_proof_is_kept_after_a_lost_directory_restore() {
        let root = crate::sealed::TestDir::new("anchors-restored");
        let (id, keys, dir) = enrolled(&root).await;
        let api = FakeApi::new();
        let seal = AgentSeal::new(&dir, id);
        *api.chain.lock().unwrap() = Ok(crate::sealed::check::local_entries(&seal).await.unwrap());
        all_sent(&dir, id).await;
        for f in std::fs::read_dir(dir.join("sealed/blobs")).unwrap() {
            let f = f.unwrap();
            let sha = f.file_name().to_str().unwrap().parse().unwrap();
            api.blobs
                .lock()
                .unwrap()
                .insert(sha, std::fs::read(f.path()).unwrap());
        }
        let (hash, seq) = head(&dir, id).await;
        std::fs::rename(&dir, root.join("gone")).unwrap();
        crate::sealed::restore::restore(&root.join("state"), &keys, &api, id, None, false)
            .await
            .unwrap();
        assert!(seal.sent().await.unwrap().is_empty());
        let (proof, anc, attested) = anchor(2, id, hash, seq);
        api.proofs.lock().unwrap().push(proof.clone());
        let signing = keys.signing(id).unwrap();
        let verified = VerifiedAnchors::new([(anc, attested)]);
        // The line can't be read (Agora's chain unreachable): the pass
        // stops there, nothing skipped or stored
        let chain = api.chain.lock().unwrap().clone();
        *api.chain.lock().unwrap() = Err("unreachable".into());
        assert_eq!(
            fetch_proofs(&dir, id, &keys, &signing, &api, &verified)
                .await
                .unwrap_err()
                .kind(),
            "line"
        );
        assert!(AnchorsFile::newest(&dir).is_none());
        *api.chain.lock().unwrap() = chain;
        assert_eq!(
            fetch_proofs(&dir, id, &keys, &signing, &api, &verified).await,
            Ok(Fetched {
                kept: 1,
                not_own: 0
            })
        );
        assert_eq!(AnchorsFile::newest(&dir), Some(proof));
    }

    #[tokio::test]
    async fn publish_is_sent_once_and_only_for_publish() {
        let root = crate::sealed::TestDir::new("anchors-publish");
        let (id, keys, dir) = enrolled(&root).await;
        let signing = keys.signing(id).unwrap();
        let api = FakeApi::new();
        // No ledger: nothing
        assert_eq!(publish_if_chosen(&dir, id, &signing, &api).await, Ok(false));
        for (choice, sent) in [(SealChoice::Private, false), (SealChoice::Publish, true)] {
            chose(&dir, choice).await;
            assert_eq!(publish_if_chosen(&dir, id, &signing, &api).await, Ok(sent));
        }
        assert_eq!(*api.anchored.lock().unwrap(), vec![true]);
        assert!(dir.join("sealed").join(PUBLISHED).exists());
        assert_eq!(publish_if_chosen(&dir, id, &signing, &api).await, Ok(false));
        assert_eq!(api.anchored.lock().unwrap().len(), 1, "once");
    }

    #[tokio::test]
    async fn a_failed_publish_is_retried() {
        let root = crate::sealed::TestDir::new("anchors-publish-fail");
        let (id, keys, dir) = enrolled(&root).await;
        let signing = keys.signing(id).unwrap();
        chose(&dir, SealChoice::Publish).await;
        let api = FakeApi::new();
        *api.fail.lock().unwrap() = Some("post_anchor");
        assert!(publish_if_chosen(&dir, id, &signing, &api).await.is_err());
        assert!(!dir.join("sealed").join(PUBLISHED).exists());
        *api.fail.lock().unwrap() = None;
        assert_eq!(publish_if_chosen(&dir, id, &signing, &api).await, Ok(true));
    }
}
