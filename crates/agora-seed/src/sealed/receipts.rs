//! `sealed/receipts.json`: when this runner first sent each version to
//! Agora, and when Agora acknowledged it, by the runner's own clock. The
//! anchor check ([`super::inclusion`]) bounds when Agora received each
//! version with them, without asking Agora.
//!
//! **Only bounds that hold.** Agora records a version's `received_at` when
//! it takes the upload, after the runner sent it and before it answers, and
//! a version moves to `sent/` only after Agora answered, so:
//!
//! - `sent` is written, and fsynced, **before** the drain that first tries
//!   the upload. It is recorded only for a version committed by this
//!   process (its outbox number at or past the [`baseline`] read when the
//!   process began), because only then can no earlier try, by an earlier
//!   process or an older runner, have reached Agora unrecorded. A version
//!   from before that gets no `sent`: its own signed `created_at` is the
//!   lower bound instead (it can't have been received before it existed).
//!   A version committed while a drain ran is given the drain's start.
//! - `acked` is written after the drain that took it, for a version that
//!   waited in the outbox when that drain began, or was committed during
//!   it. For a version with no `sent` (an earlier process's, whose earlier
//!   try may have reached Agora with its answer lost) it is marked
//!   `observed`: sound, but possibly late. A version found in `sent/` with no `acked` (uploaded by an older
//!   runner, or before a crash between the drain and this record) gets one
//!   when a process first sees it ([`baseline`]), marked `observed`: a
//!   sound upper bound, but a late one, so it never makes an anchor's
//!   window count as decided ([`super::inclusion::Bounds::timed`]).
//!
//! **Never in the way of an upload.** If the file can't be read it is set
//! aside (`receipts.unreadable.<utc>`) and a new one starts, with no `sent`
//! for any version already numbered (their first tries may be in the lost
//! file). If it can't be written, before or after a drain, the drain goes
//! ahead and this process records no further `sent` for the agent (one
//! missed after a drain could let a later drain record a late one).
//!
//! **Two processes** (the runner and `sealed-enroll` or `sealed-stop`,
//! which don't wait for it) may drain one agent at once. One can then
//! record `sent` for a version the other is uploading: late by at most the
//! time between the other's request and its `mark_sent`, or, if that
//! answer was lost, until the next drain (a minute in the runner), both
//! well inside [`super::inclusion::SKEW`].
//!
//! **Format.** A new file; nothing else in `sealed/` changes. Written
//! atomically (a temporary file, fsync, rename, fsync of `sealed/`) under
//! the agent's `sealed/lock`. A runner without this module never reads or
//! writes it. Pruned by the anchor check below the oldest version a later
//! anchor can still need ([`prune_below`]).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use agora_agentkit::govlog::Sha256Hex;
use agora_agentkit::ids::AgentId;
use agora_agentkit::state::outbox::{AgoraOutbox, BeforeUpload, DrainError, Drained};
use agora_agentkit::state::storage::OutboxItem;
use agora_agentkit::state::storage::{AgentSeal, SEALED_DIR};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::SealedEnrolment;

/// `sealed/receipts.json`.
pub const RECEIPTS: &str = "receipts.json";
/// Where an unreadable `sealed/<stem>.json` is set aside:
/// `sealed/<stem>.unreadable.<utc ms>`, never over an earlier one.
pub fn aside_name(agent_dir: &Path, stem: &str) -> PathBuf {
    agent_dir.join(SEALED_DIR).join(format!(
        "{stem}.unreadable.{}",
        Utc::now().timestamp_millis()
    ))
}

fn is_false(b: &bool) -> bool {
    !b
}

/// One version's upload, by the runner's clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Receipt {
    pub entry_hash: Sha256Hex,
    pub seq: u64,
    /// Before the first try to upload it: Agora can't have received it
    /// earlier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sent: Option<DateTime<Utc>>,
    /// Agora had received it by then.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acked: Option<DateTime<Utc>>,
    /// `acked` is when a process first found it uploaded, not the drain
    /// that uploaded it.
    #[serde(default, skip_serializing_if = "is_false")]
    pub observed: bool,
}

/// `sealed/receipts.json`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiptsFile {
    /// By seq.
    #[serde(default)]
    pub versions: Vec<Receipt>,
    /// No `sent` is recorded for a version numbered below this: set when
    /// an unreadable file was set aside, whose `sent` times are lost.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub no_sent_below: u64,
    /// Receipts of versions below this seq were pruned ([`prune_below`]):
    /// none is added back for them.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub pruned_below: u64,
    /// When this runner began recording upload times for the agent
    /// (the first record written to the file): no version
    /// uploaded before it has them. `None` when it isn't known: a file
    /// from 0.152.0 or earlier that was pruned or reset before a later
    /// runner first read it, or a file begun again after it was
    /// unreadable (its earlier times were lost, not never kept).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub began: Option<DateTime<Utc>>,
}

fn is_zero(n: &u64) -> bool {
    *n == 0
}

fn path(agent_dir: &Path) -> PathBuf {
    agent_dir.join(SEALED_DIR).join(RECEIPTS)
}

impl ReceiptsFile {
    /// A missing file is no receipts.
    pub fn load(agent_dir: &Path) -> std::io::Result<Self> {
        match std::fs::read(path(agent_dir)) {
            Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e),
        }
    }

    /// [`Self::load`], under the agent's `sealed/lock`; a file that doesn't
    /// parse is set aside and a new one begun, with no `sent` below
    /// `next_ord` (the next outbox number now).
    fn load_or_reset(agent_dir: &Path, id: AgentId, next_ord: u64) -> std::io::Result<Self> {
        match Self::load(agent_dir) {
            Ok(file) => Ok(file),
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::InvalidData | std::io::ErrorKind::UnexpectedEof
                ) =>
            {
                let aside = aside_name(agent_dir, "receipts");
                std::fs::rename(path(agent_dir), &aside)?;
                tracing::warn!(
                    event_type = "sealed_state_receipts_reset",
                    agent_id = %id,
                    error = %e,
                    aside = %aside.display(),
                    "sealed/receipts.json unreadable: set aside, a new one begun; anchors of the \
                     versions before it read as not checkable"
                );
                let mut file = Self {
                    versions: Vec::new(),
                    no_sent_below: next_ord,
                    pruned_below: 0,
                    began: None,
                };
                file.save(agent_dir)?;
                Ok(file)
            }
            Err(e) => Err(e),
        }
    }

    fn save(&mut self, agent_dir: &Path) -> std::io::Result<()> {
        self.versions
            .sort_by_key(|r| (r.seq, r.entry_hash.to_hex()));
        super::anchors::write_atomic(&path(agent_dir), &serde_json::to_vec_pretty(self)?)
    }

    /// Set [`Self::began`] if it is unset and can be known, as the file is
    /// saved with its first record: the earliest time in it (a new file's
    /// first record; for a file kept by 0.152.0 that was never pruned or
    /// reset, the first process's observed acknowledgements or its first
    /// send). `now` only for a file with nothing in it. Whether it changed.
    fn begin(&mut self, now: DateTime<Utc>) -> bool {
        if self.began.is_some() || self.pruned_below > 0 || self.no_sent_below > 0 {
            return false;
        }
        self.began = if self.versions.is_empty() {
            Some(now)
        } else {
            self.versions
                .iter()
                .flat_map(|r| [r.sent, r.acked])
                .flatten()
                .min()
        };
        self.began.is_some()
    }

    /// By entry hash.
    pub fn by_hash(&self) -> HashMap<Sha256Hex, Receipt> {
        self.versions.iter().map(|r| (r.entry_hash, *r)).collect()
    }

    fn get_mut(&mut self, hash: &Sha256Hex, seq: u64) -> &mut Receipt {
        match self.versions.iter().position(|r| r.entry_hash == *hash) {
            Some(i) => &mut self.versions[i],
            None => {
                self.versions.push(Receipt {
                    entry_hash: *hash,
                    seq,
                    sent: None,
                    acked: None,
                    observed: false,
                });
                self.versions.last_mut().expect("just pushed")
            }
        }
    }
}

async fn next_ord(seal: &AgentSeal) -> Result<u64, String> {
    Ok(seal
        .head()
        .await
        .map_err(|e| e.to_string())?
        .map_or(0, |h| h.next_ord))
}

/// When a process begins, before any of its sessions saves: the agent's
/// next outbox number (a version numbered at or past it is committed by
/// this process), and an `observed` acknowledgement for each version
/// already uploaded that has none. `u64::MAX` when `sealed/` can't be
/// read: nothing counts as this process's own, so no `sent` is recorded
/// (sound, only less precise).
pub async fn baseline(agent_dir: &Path, id: AgentId) -> u64 {
    let seal = AgentSeal::new(agent_dir, id);
    let observe = async {
        let _lock = seal.lock().await.map_err(|e| e.to_string())?;
        let next = next_ord(&seal).await?;
        let now = Utc::now();
        let sent = seal.sent().await.map_err(|e| e.to_string())?;
        let mut file =
            ReceiptsFile::load_or_reset(agent_dir, id, next).map_err(|e| e.to_string())?;
        let mut changed = false;
        let pruned_below = file.pruned_below;
        for item in sent.iter().filter(|i| i.entry.entry.seq >= pruned_below) {
            let r = file.get_mut(&item.entry.entry_hash(), item.entry.entry.seq);
            if r.acked.is_none() {
                r.acked = Some(now);
                r.observed = true;
                changed = true;
            }
        }
        // Only a file with something in it: a dry run, or an agent with
        // nothing uploaded, writes nothing (review of round 13)
        if !file.versions.is_empty() {
            changed |= file.begin(now);
        }
        if changed {
            file.save(agent_dir).map_err(|e| e.to_string())?;
        }
        Ok::<u64, String>(next)
    };
    match observe.await {
        Ok(next) => next,
        Err(e) => {
            tracing::warn!(
                event_type = "sealed_state_receipts_unwritten",
                agent_id = %id,
                error = %e,
                "upload times not begun for this process; none are recorded for its uploads"
            );
            u64::MAX
        }
    }
}

/// [`baseline`] of each agent in `ids`, under `state_dir`.
pub async fn baselines(
    state_dir: &Path,
    ids: impl IntoIterator<Item = AgentId>,
) -> HashMap<AgentId, u64> {
    let mut out = HashMap::new();
    for id in ids {
        out.insert(id, baseline(&state_dir.join(id.to_string()), id).await);
    }
    out
}

/// What [`before`] saw: the drain's start, and what waited.
#[derive(Debug, Clone)]
pub struct Before {
    pub at: DateTime<Utc>,
    /// Hash, seq, outbox number.
    waiting: Vec<(Sha256Hex, u64, u64)>,
    /// The next outbox number then: a version numbered at or past it was
    /// committed after this read, so no drain before this one tried it.
    next_ord: u64,
}

/// Before a drain: `sent` for each waiting version this process committed
/// and that has none, written before the upload is tried.
pub async fn before(agent_dir: &Path, id: AgentId, baseline: u64) -> Result<Before, String> {
    let seal = AgentSeal::new(agent_dir, id);
    let at = Utc::now();
    let _lock = seal.lock().await.map_err(|e| e.to_string())?;
    let next_ord = next_ord(&seal).await?;
    let items = seal.outbox().await.map_err(|e| e.to_string())?;
    let mut file =
        ReceiptsFile::load_or_reset(agent_dir, id, next_ord).map_err(|e| e.to_string())?;
    let floor = baseline.max(file.no_sent_below);
    let mut changed = false;
    for item in &items {
        if item.ord < floor {
            continue;
        }
        let r = file.get_mut(&item.entry.entry_hash(), item.entry.entry.seq);
        if r.sent.is_none() {
            r.sent = Some(at);
            changed = true;
        }
    }
    if changed {
        file.begin(at);
        file.save(agent_dir).map_err(|e| e.to_string())?;
    }
    Ok(Before {
        at,
        waiting: items
            .iter()
            .map(|i| (i.entry.entry_hash(), i.entry.entry.seq, i.ord))
            .collect(),
        next_ord,
    })
}

/// After a drain, whatever it returned: `acked` for each version that
/// waited at [`before`] and has left the outbox (only an upload Agora
/// answered moves it), and for each this process committed during the
/// drain, `sent` the drain's start, `acked` too if it left the outbox.
pub async fn after(
    agent_dir: &Path,
    id: AgentId,
    baseline: u64,
    before: &Before,
) -> Result<usize, String> {
    let seal = AgentSeal::new(agent_dir, id);
    let _lock = seal.lock().await.map_err(|e| e.to_string())?;
    let now = Utc::now();
    let next = next_ord(&seal).await?;
    let mut file = ReceiptsFile::load_or_reset(agent_dir, id, next).map_err(|e| e.to_string())?;
    // Committed after `before` and in this process: numbered past both
    let since = before.next_ord.max(baseline).max(file.no_sent_below);
    if before.waiting.is_empty() && next <= since {
        return Ok(0);
    }
    let waiting_now = seal.outbox().await.map_err(|e| e.to_string())?;
    let left: HashMap<u64, Sha256Hex> = waiting_now
        .iter()
        .map(|i| (i.ord, i.entry.entry_hash()))
        .collect();
    let mut changed = 0;
    for (hash, seq, ord) in &before.waiting {
        if left.get(ord) == Some(hash) {
            continue;
        }
        let r = file.get_mut(hash, *seq);
        if r.acked.is_none() {
            r.acked = Some(now);
            // No `sent`: an earlier process's, which may have tried it
            // before, its answer lost; this acknowledgement may be late
            r.observed = r.sent.is_none();
            changed += 1;
        }
    }
    if next > since {
        // Committed during the drain: those still waiting may have been
        // tried (an answer lost) and get `sent` now; those uploaded are
        // read from `sent/`, the only place they can be
        let gone: Vec<u64> = (since..next).filter(|o| !left.contains_key(o)).collect();
        let still: Vec<(u64, Sha256Hex)> = waiting_now
            .iter()
            .filter(|i| i.ord >= since)
            .map(|i| (i.entry.entry.seq, i.entry.entry_hash()))
            .collect();
        for (seq, hash) in still {
            let r = file.get_mut(&hash, seq);
            if r.sent.is_none() {
                r.sent = Some(before.at);
                changed += 1;
            }
        }
        if !gone.is_empty() {
            for item in seal.sent().await.map_err(|e| e.to_string())? {
                if !gone.contains(&item.ord) {
                    continue;
                }
                let r = file.get_mut(&item.entry.entry_hash(), item.entry.entry.seq);
                if r.sent.is_none() {
                    r.sent = Some(before.at);
                    changed += 1;
                }
                if r.acked.is_none() {
                    r.acked = Some(now);
                    changed += 1;
                }
            }
        }
    }
    if changed > 0 {
        file.begin(now);
        file.save(agent_dir).map_err(|e| e.to_string())?;
    }
    Ok(changed)
}

/// The first-send times of one drain, recorded by agentkit's drain once
/// it has checked that the agent uploads and is about to send
/// ([`BeforeUpload`]): never for a drain that sends nothing, so an agent
/// that doesn't upload gets no receipts at all.
struct FirstSends<'a> {
    agent_dir: &'a Path,
    id: AgentId,
    baseline: &'a mut u64,
    before: Option<Before>,
}

#[async_trait::async_trait]
impl BeforeUpload for FirstSends<'_> {
    async fn before_upload(&mut self, _items: &[OutboxItem]) {
        self.before = match before(self.agent_dir, self.id, *self.baseline).await {
            Ok(b) => Some(b),
            Err(e) => {
                tracing::warn!(
                    event_type = "sealed_state_receipts_unwritten",
                    agent_id = %self.id,
                    error = %e,
                    "upload times could not be recorded; the upload goes ahead, and this \
                     process records no more send times for the agent"
                );
                *self.baseline = u64::MAX;
                None
            }
        };
    }
}

/// Drain `id`'s outbox with its receipts recorded around it. A receipt
/// that can't be written never stops the upload: before the drain, this
/// process then records no `sent` for the agent from here on (`baseline`
/// set past every number), since a version may now reach Agora with no
/// lower bound on disk; after it, the versions simply have no `acked` from
/// this drain, which only makes the check less precise.
///
/// An agent that doesn't upload ([`super::uploads`]) is not drained:
/// [`Drained::LocalOnly`], and nothing read, sent or recorded.
pub async fn drain(
    outbox: &AgoraOutbox<SealedEnrolment>,
    agent_dir: &Path,
    id: AgentId,
    baseline: &mut u64,
) -> Result<Drained, DrainError> {
    let mut sends = FirstSends {
        agent_dir,
        id,
        baseline,
        before: None,
    };
    let result = outbox.drain_with(id, &mut sends).await;
    let FirstSends {
        baseline, before, ..
    } = sends;
    if let Some(before) = before
        && let Err(e) = after(agent_dir, id, *baseline, &before).await
    {
        tracing::warn!(
            event_type = "sealed_state_receipts_unwritten",
            agent_id = %id,
            error = %e,
            "acknowledgement times could not be recorded; those versions' anchors are \
             reported as not checkable, and this process records no more send times for the \
             agent"
        );
        // A version committed during this drain, tried and unanswered,
        // has no `sent` now: a later drain must not give it one
        *baseline = u64::MAX;
    }
    result
}

/// Forget the receipts of versions below `seq`: no anchor the check has
/// yet to judge can need them ([`super::inclusion::Expected`]'s floor only
/// rises with the day).
pub async fn prune_below(agent_dir: &Path, id: AgentId, seq: u64) -> Result<usize, String> {
    let seal = AgentSeal::new(agent_dir, id);
    let _lock = seal.lock().await.map_err(|e| e.to_string())?;
    let mut file = ReceiptsFile::load(agent_dir).map_err(|e| e.to_string())?;
    let n = file.versions.len();
    file.versions.retain(|r| r.seq >= seq);
    let gone = n - file.versions.len();
    let raised = seq > file.pruned_below;
    file.pruned_below = file.pruned_below.max(seq);
    if gone > 0 || raised {
        file.save(agent_dir).map_err(|e| e.to_string())?;
    }
    Ok(gone)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sealed::check::tests::enrolled;

    /// Upload `id`'s outbox as a drain would, without a server.
    async fn upload(dir: &Path, id: AgentId) {
        let seal = AgentSeal::new(dir, id);
        for item in seal.outbox().await.unwrap() {
            seal.mark_sent(item.ord).await.unwrap();
        }
    }

    #[tokio::test]
    async fn versions_this_process_committed_get_both_times() {
        let root = crate::sealed::TestDir::new("receipts-own");
        let (id, _keys, dir) = enrolled(&root).await;
        // Committed by this process: the baseline is from before them
        let b = before(&dir, id, 0).await.unwrap();
        let file = ReceiptsFile::load(&dir).unwrap();
        assert_eq!(file.versions.len(), 2);
        assert!(file.versions.iter().all(|r| r.sent == Some(b.at)));
        assert!(file.versions.iter().all(|r| r.acked.is_none()));
        upload(&dir, id).await;
        assert_eq!(after(&dir, id, 0, &b).await.unwrap(), 2);
        let file = ReceiptsFile::load(&dir).unwrap();
        assert!(
            file.versions
                .iter()
                .all(|r| r.acked.is_some_and(|a| a >= b.at) && !r.observed)
        );
        // A second pass records nothing more, and moves nothing back
        let b2 = before(&dir, id, 0).await.unwrap();
        assert_eq!(after(&dir, id, 0, &b2).await.unwrap(), 0);
        assert_eq!(ReceiptsFile::load(&dir).unwrap(), file);
    }

    /// `began`: set by the first process; a 0.152.0 file never pruned or
    /// reset gets its earliest time; one pruned or reset stays unknown;
    /// never moved once set.
    #[test]
    fn the_record_says_when_it_began_only_when_it_knows() {
        let t = |h: i64| DateTime::<Utc>::from_timestamp(1_762_400_000 + h * 3600, 0).unwrap();
        let r = |seq, sent: Option<DateTime<Utc>>, acked| Receipt {
            entry_hash: Sha256Hex::from([seq as u8; 32]),
            seq,
            sent,
            acked: Some(acked),
            observed: sent.is_none(),
        };
        let mut fresh = ReceiptsFile::default();
        assert!(fresh.begin(t(5)));
        assert_eq!(fresh.began, Some(t(5)));
        assert!(!fresh.begin(t(9)), "never moved");
        let mut kept = ReceiptsFile {
            versions: vec![r(1, None, t(2)), r(2, Some(t(3)), t(4))],
            ..Default::default()
        };
        assert!(kept.begin(t(9)));
        assert_eq!(kept.began, Some(t(2)));
        for (pruned_below, no_sent_below) in [(1, 0), (0, 4)] {
            let mut lost = ReceiptsFile {
                versions: vec![r(2, Some(t(3)), t(4))],
                pruned_below,
                no_sent_below,
                began: None,
            };
            assert!(!lost.begin(t(9)));
            assert_eq!(lost.began, None);
        }
    }

    #[tokio::test]
    async fn versions_from_before_this_process_get_no_send_time() {
        let root = crate::sealed::TestDir::new("receipts-legacy");
        let (id, _keys, dir) = enrolled(&root).await;
        // An earlier process committed them (and may have tried them)
        let base = baseline(&dir, id).await;
        assert!(base > 0);
        let b = before(&dir, id, base).await.unwrap();
        assert!(
            !dir.join("sealed").join(RECEIPTS).exists(),
            "no sent written, and nothing begun: a dry run writes nothing"
        );
        upload(&dir, id).await;
        // Acknowledged during this drain: an upper bound that holds
        assert_eq!(after(&dir, id, base, &b).await.unwrap(), 2);
        let file = ReceiptsFile::load(&dir).unwrap();
        assert!(file.began.is_some(), "begun with its first record");
        // No `sent`: their acknowledgement may be late (an earlier try)
        assert!(
            file.versions
                .iter()
                .all(|r| r.sent.is_none() && r.acked.is_some() && r.observed)
        );
    }

    #[tokio::test]
    async fn versions_found_uploaded_are_acknowledged_as_observed() {
        let root = crate::sealed::TestDir::new("receipts-uploaded");
        let (id, _keys, dir) = enrolled(&root).await;
        // An older runner uploaded them
        upload(&dir, id).await;
        let b = before(&dir, id, 0).await.unwrap();
        assert_eq!(after(&dir, id, 0, &b).await.unwrap(), 0, "not this drain's");
        assert!(ReceiptsFile::load(&dir).unwrap().versions.is_empty());
        // The next process sees them uploaded: a late upper bound, marked
        let started = Utc::now();
        baseline(&dir, id).await;
        let file = ReceiptsFile::load(&dir).unwrap();
        assert_eq!(file.versions.len(), 2);
        assert!(
            file.versions
                .iter()
                .all(|r| r.observed && r.sent.is_none() && r.acked.is_some_and(|a| a >= started))
        );
        // Never moved by a later look
        baseline(&dir, id).await;
        assert_eq!(ReceiptsFile::load(&dir).unwrap(), file);
    }

    #[tokio::test]
    async fn a_crash_between_drain_and_record_leaves_only_an_observed_ack() {
        let root = crate::sealed::TestDir::new("receipts-crash");
        let (id, _keys, dir) = enrolled(&root).await;
        let _b = before(&dir, id, 0).await.unwrap();
        upload(&dir, id).await;
        // ...the process dies here; the next one's drain finds them sent
        let b = before(&dir, id, 0).await.unwrap();
        assert_eq!(after(&dir, id, 0, &b).await.unwrap(), 0);
        baseline(&dir, id).await;
        let file = ReceiptsFile::load(&dir).unwrap();
        assert!(
            file.versions
                .iter()
                .all(|r| r.sent.is_some() && r.acked.is_some() && r.observed)
        );
    }

    #[tokio::test]
    async fn a_version_committed_during_the_drain_is_dated_from_its_start() {
        let root = crate::sealed::TestDir::new("receipts-during");
        let (id, _keys, dir) = enrolled(&root).await;
        // `before` saw nothing waiting (as if the drain's own read came
        // after a commit `before` missed)
        let b = Before {
            at: Utc::now(),
            waiting: Vec::new(),
            next_ord: 0,
        };
        upload(&dir, id).await;
        assert!(after(&dir, id, 0, &b).await.unwrap() > 0);
        let file = ReceiptsFile::load(&dir).unwrap();
        assert_eq!(file.versions.len(), 2);
        assert!(
            file.versions
                .iter()
                .all(|r| r.sent == Some(b.at) && r.acked.is_some())
        );
        // One still waiting (tried, its answer lost, or not tried) gets the
        // drain's start as its send: before any try could reach Agora
        let root = crate::sealed::TestDir::new("receipts-during-waiting");
        let (id, _keys, dir) = enrolled(&root).await;
        after(&dir, id, 0, &b).await.unwrap();
        let file = ReceiptsFile::load(&dir).unwrap();
        assert!(
            file.versions
                .iter()
                .all(|r| r.sent == Some(b.at) && r.acked.is_none())
        );
        // But not one from before the process: that one was maybe tried
        let root = crate::sealed::TestDir::new("receipts-during-old");
        let (id, _keys, dir) = enrolled(&root).await;
        upload(&dir, id).await;
        assert_eq!(after(&dir, id, u64::MAX, &b).await.unwrap(), 0);
        // Nor one already in sent/ when `before` read (its after() lost):
        // its first try came before this drain
        let root = crate::sealed::TestDir::new("receipts-during-sent");
        let (id, _keys, dir) = enrolled(&root).await;
        upload(&dir, id).await;
        let b = before(&dir, id, 0).await.unwrap();
        assert_eq!(after(&dir, id, 0, &b).await.unwrap(), 0);
        assert!(ReceiptsFile::load(&dir).unwrap().versions.is_empty());
    }

    #[tokio::test]
    async fn pruning_keeps_what_a_later_anchor_can_need() {
        let root = crate::sealed::TestDir::new("receipts-prune");
        let (id, _keys, dir) = enrolled(&root).await;
        let b = before(&dir, id, 0).await.unwrap();
        upload(&dir, id).await;
        after(&dir, id, 0, &b).await.unwrap();
        assert_eq!(prune_below(&dir, id, 1).await.unwrap(), 1);
        let file = ReceiptsFile::load(&dir).unwrap();
        assert_eq!(
            file.versions.iter().map(|r| r.seq).collect::<Vec<_>>(),
            vec![1]
        );
        assert_eq!(prune_below(&dir, id, 1).await.unwrap(), 0);
        // A later process doesn't bring the pruned back
        baseline(&dir, id).await;
        assert_eq!(ReceiptsFile::load(&dir).unwrap(), file);
    }

    /// An interrupted write leaves the file as it was: the temporary is
    /// never read as the file.
    #[tokio::test]
    async fn a_torn_write_leaves_the_last_whole_file() {
        let root = crate::sealed::TestDir::new("receipts-torn");
        let (id, _keys, dir) = enrolled(&root).await;
        let b = before(&dir, id, 0).await.unwrap();
        let whole = ReceiptsFile::load(&dir).unwrap();
        std::fs::write(
            dir.join("sealed").join("receipts.writing"),
            b"{\"versions\": [{",
        )
        .unwrap();
        assert_eq!(ReceiptsFile::load(&dir).unwrap(), whole);
        upload(&dir, id).await;
        after(&dir, id, 0, &b).await.unwrap();
        assert!(
            ReceiptsFile::load(&dir)
                .unwrap()
                .versions
                .iter()
                .all(|r| r.acked.is_some())
        );
    }

    /// A file that doesn't parse is set aside, never in an upload's way,
    /// and no `sent` is recorded for what it may have held.
    #[tokio::test]
    async fn an_unreadable_file_is_set_aside_and_records_no_lost_send() {
        let root = crate::sealed::TestDir::new("receipts-unreadable");
        let (id, _keys, dir) = enrolled(&root).await;
        std::fs::write(dir.join("sealed").join(RECEIPTS), b"{ not json").unwrap();
        let b = before(&dir, id, 0).await.unwrap();
        let aside = || {
            std::fs::read_dir(dir.join("sealed"))
                .unwrap()
                .filter(|e| {
                    e.as_ref()
                        .unwrap()
                        .file_name()
                        .to_string_lossy()
                        .starts_with("receipts.unreadable.")
                })
                .count()
        };
        assert_eq!(aside(), 1);
        let file = ReceiptsFile::load(&dir).unwrap();
        assert_eq!(file.no_sent_below, 2);
        assert!(file.versions.is_empty(), "the waiting versions get no send");
        upload(&dir, id).await;
        after(&dir, id, 0, &b).await.unwrap();
        assert!(
            ReceiptsFile::load(&dir)
                .unwrap()
                .versions
                .iter()
                .all(|r| r.sent.is_none() && r.acked.is_some())
        );
        // A second reset keeps the first one set aside
        std::thread::sleep(std::time::Duration::from_millis(2));
        std::fs::write(dir.join("sealed").join(RECEIPTS), b"{ not json").unwrap();
        before(&dir, id, 0).await.unwrap();
        assert_eq!(aside(), 2);
    }

    /// A file that can't be written doesn't stop the upload: the drain
    /// goes on, and no more sends are recorded in this process.
    #[tokio::test]
    async fn an_unwritable_file_never_stops_an_upload() {
        let root = crate::sealed::TestDir::new("receipts-unwritable");
        let (id, _keys, dir) = enrolled(&root).await;
        std::fs::create_dir_all(dir.join("sealed").join(RECEIPTS)).unwrap();
        let outbox = AgoraOutbox::new(
            agora_agentkit::client::Client::new("http://127.0.0.1:9/".parse().unwrap()).unwrap(),
            root.join("state"),
            std::sync::Arc::new(SealedEnrolment::new(
                root.join("state"),
                root.join("secrets"),
            )),
        );
        let mut base = 0;
        let e = drain(&outbox, &dir, id, &mut base).await.unwrap_err();
        assert!(
            matches!(e, DrainError::Client(_)),
            "the upload was tried: {e}"
        );
        assert_eq!(base, u64::MAX);
    }
}
