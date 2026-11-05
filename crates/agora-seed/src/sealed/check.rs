//! The session-start check, for enrolled agents only: run in `main` right
//! after the states load, before any agent is built, as a skip stage
//! (`sealed`). Not in `on_init`: agentkit's reactor would still save a
//! refused agent there.
//!
//! | Found | Done |
//! |---|---|
//! | the local state is the head's (or a pending item's, now committed), the main line verifies, and Agora has it or is behind | run; the verdict line is seated |
//! | the local state matches nothing, the chain fails, or `head.json` is gone | **refuse**: skipped, `sealed/refused`, a notice for the agent, [`SealedStateRefused`] |
//! | the local state is one whose seal failed | run; uploads held; [`SealedStateSealFailed`] |
//! | `sealed/` can't be read | run; saves unsealed, uploads held; [`SealedStateUnreadable`]; never a refusal on its own |
//! | the local head is not on Agora (`LocalMissing`) | run; held; [`SealedStateHeadMissing`] |
//! | Agora moved on (`ServerAhead`, `Diverged`) | run on the local copy; held; [`SealedStateFork`] |
//!
//! Every problem is told to the agent too, in plain words. The line names
//! the file and the commit that checked it, never a model. Nothing here
//! reads any content of the state: the check is over hashes.
//!
//! [`SealedStateRefused`]: crate::alerts::AlertKind::SealedStateRefused
//! [`SealedStateSealFailed`]: crate::alerts::AlertKind::SealedStateSealFailed
//! [`SealedStateHeadMissing`]: crate::alerts::AlertKind::SealedStateHeadMissing
//! [`SealedStateFork`]: crate::alerts::AlertKind::SealedStateFork
//! [`SealedStateUnreadable`]: crate::alerts::AlertKind::SealedStateUnreadable

use std::collections::HashMap;
use std::path::Path;

use agora_agentkit::enums::StateEntryKind;
use agora_agentkit::govlog::Sha256Hex;
use agora_agentkit::ids::{AgentId, GovernanceLogId};
use agora_agentkit::state::storage::{AgentSeal, Recovery};
use agora_agentkit::state::verify::{HeadRef, HeadStatus, head_status, verify_chain};
use agora_agentkit::state::{AnchorProof, SignedStateEntry};

use super::KeyStore;
use super::api::SealedApi;
use super::notice::{Notice, Refusal};
use super::sizes::{Recorded, SizeLog, Sizes, drop_lines};
use crate::alerts::{Alert, AlertKind, Alerter};

/// What checked it, as the verdict line names it. Pinned to the agentkit
/// in `Cargo.lock` by `the_verifier_is_the_locked_agentkit`.
pub const VERIFIER: &str = "git:agora-agentkit:v0.151.2:src/state/verify.rs";

/// Why uploads are held this run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hold {
    SealFailed,
    HeadMissing,
    Fork,
    /// `sealed/` couldn't be read: the agent runs, its saves go unsealed
    /// (no base) and are recorded as such.
    Unreadable,
    /// The chain couldn't be checked (Agora unreachable for a main line
    /// that runs through it): the agent runs, uploads held. No refusal
    /// without evidence.
    Unchecked,
}

/// Agora's entries beside ours, ours winning: one Agora holds that is
/// here too is not taken from Agora. `Err`: Agora's copy of an entry held
/// here differs from it (the hash names it).
pub fn merge_server(
    local: &[SignedStateEntry],
    server: &[SignedStateEntry],
) -> Result<Vec<SignedStateEntry>, Sha256Hex> {
    let ours: HashMap<Sha256Hex, &SignedStateEntry> =
        local.iter().map(|e| (e.entry_hash(), e)).collect();
    let mut all = local.to_vec();
    for e in server {
        let hash = e.entry_hash();
        match ours.get(&hash) {
            Some(mine) if *mine == e => {}
            Some(_) => return Err(hash),
            None => all.push(e.clone()),
        }
    }
    Ok(all)
}

/// The check's answer for one agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// Run. `seat`: lines for the end of the intro; `notice`: one of them
    /// is the refusal notice, renamed aside once the session saves.
    Run {
        seat: Vec<String>,
        notice: bool,
        hold: Option<Hold>,
        /// A memory or soul that shrank past the threshold: what, by how
        /// many percent ([`super::sizes`]).
        drops: Vec<(&'static str, u32)>,
    },
    /// Leave it out, refused. `detail` is for the operator, hashes and
    /// counts only.
    Refuse { reason: Refusal, detail: String },
    /// The local state is the local head's, and the history on Agora (or
    /// the keys it is checked with) fails while nothing here shows a loss.
    /// The whole run is refused (Steward, 2026-11-03), but nothing is
    /// marked against the agent: the next start checks again, and runs once
    /// the cause is fixed. Decided in one place, [`judge_failure`].
    ServerFault { kind: FaultKind, detail: String },
}

/// What a failure that isn't the local state's own is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultKind {
    /// Agora's copy fails: a missing link, a chain problem on entries
    /// only Agora holds, a copy that differs from ours. Likely a
    /// server-side fault.
    ServerChainBroken,
    /// Entries are signed by a key this runner doesn't recognise: most
    /// likely its record of past signing keys was lost after a rotation;
    /// otherwise a forgery. Not reported as a platform fault.
    UntrustedKey,
}

impl FaultKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ServerChainBroken => "server_chain_broken",
            Self::UntrustedKey => "untrusted_key",
        }
    }
}

/// The class of one `verify_chain` problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemClass {
    /// No key this runner knows made the signature, or a known one it
    /// doesn't trust.
    Key,
    /// Anything else: the chain's shape, links, kinds, blobs, times, a
    /// retired key's use.
    Chain,
}

/// Exhaustive, no wildcard: a variant agentkit adds fails to compile here
/// until it is classified.
pub fn class_of(p: &agora_agentkit::state::verify::ChainProblem) -> ProblemClass {
    use agora_agentkit::state::verify::ChainProblem as P;
    match p {
        P::BadSignature { .. } | P::UntrustedKey { .. } => ProblemClass::Key,
        P::MissingGenesis
        | P::GenesisNotFirst { .. }
        | P::WrongAgent { .. }
        | P::UnknownVersion { .. }
        | P::SeqGap { .. }
        | P::Fork { .. }
        | P::BrokenLink { .. }
        | P::RetiredKey { .. }
        | P::UnknownKind { .. }
        | P::DeltaWithoutBase { .. }
        | P::MissingBlob { .. }
        | P::MergeNotAllowed { .. }
        | P::TimeWentBack { .. } => ProblemClass::Chain,
    }
}

/// The one place a chain that didn't verify is judged. Each problem is
/// placed by the entry it lies on:
///
/// - on an entry only Agora holds: Agora's evidence (a key problem there
///   is [`FaultKind::UntrustedKey`], any other [`FaultKind::ServerChainBroken`]);
/// - a key problem on an entry held here whose signature a key this
///   runner once used does make (`past_keys.json`): its rotation record is
///   what was lost, [`FaultKind::UntrustedKey`];
/// - anything else on an entry held here: local evidence, refused and
///   marked ([`Refusal::ChainBroken`]), so `sealed-accept` or
///   `sealed-restore` can resolve it.
///
/// Any local problem refuses; else any chain problem of Agora's is
/// [`FaultKind::ServerChainBroken`]; else [`FaultKind::UntrustedKey`].
pub fn judge_failure(
    id: AgentId,
    problems: &[agora_agentkit::state::verify::ChainProblem],
    line: &[SignedStateEntry],
    server_only: &std::collections::HashSet<Sha256Hex>,
    known_keys: &[agora_agentkit::crypto::VerifyingKey],
) -> Outcome {
    #[derive(PartialEq)]
    enum Place {
        Local,
        AgoraChain,
        Key,
    }
    let place = |p: &agora_agentkit::state::verify::ChainProblem| -> Place {
        let entry = problem_seq(p).and_then(|s| line.iter().find(|e| e.entry.seq == s));
        let agoras = entry.is_some_and(|e| server_only.contains(&e.entry_hash()));
        match (class_of(p), agoras) {
            (ProblemClass::Key, true) => Place::Key,
            (ProblemClass::Chain, true) => Place::AgoraChain,
            (ProblemClass::Key, false)
                if entry.is_some_and(|e| known_keys.iter().any(|k| e.verify(k))) =>
            {
                Place::Key
            }
            _ => Place::Local,
        }
    };
    let places: Vec<Place> = problems.iter().map(place).collect();
    if places.is_empty() || places.contains(&Place::Local) {
        return Outcome::Refuse {
            reason: Refusal::ChainBroken,
            detail: format!("chain did not verify: {problems:?}"),
        };
    }
    if places.contains(&Place::AgoraChain) {
        return on_fault(
            id,
            FaultKind::ServerChainBroken,
            format!("Agora's entries did not verify: {problems:?}"),
        );
    }
    on_fault(
        id,
        FaultKind::UntrustedKey,
        format!("entries signed by a key this runner doesn't recognise: {problems:?}"),
    )
}

/// Agora's copy of an entry held here differs from ours: whichever copy no
/// key this runner has held signed is the corrupted one. Ours: local
/// evidence, refused. Agora's: a server fault.
fn disagreement(
    id: AgentId,
    hash: Sha256Hex,
    local: &[SignedStateEntry],
    keys: &KeyStore,
) -> Outcome {
    let known = known_keys(keys, id);
    let ours_signed = local
        .iter()
        .find(|e| e.entry_hash() == hash)
        .is_some_and(|e| known.iter().any(|k| e.verify(k)));
    if ours_signed {
        on_fault(
            id,
            FaultKind::ServerChainBroken,
            format!("Agora's copy of entry {hash} differs from the one held here"),
        )
    } else {
        Outcome::Refuse {
            reason: Refusal::ChainBroken,
            detail: format!(
                "the copy held here of entry {hash} is signed by no key this runner has held"
            ),
        }
    }
}

/// Every signing key this runner has held for `id`: the current one and
/// those it retired (`past_keys.json`). For telling a lost rotation record
/// from a corrupted signature.
pub fn known_keys(keys: &KeyStore, id: AgentId) -> Vec<agora_agentkit::crypto::VerifyingKey> {
    let mut known: Vec<_> = keys
        .signing(id)
        .map(|k| k.verifying_key())
        .into_iter()
        .collect();
    if let Ok(past) = keys.past_keys(id) {
        known.extend(past.signing_public());
    }
    known
}

fn on_fault(id: AgentId, kind: FaultKind, detail: String) -> Outcome {
    tracing::error!(
        event_type = "sealed_state_server_chain_broken",
        agent_id = %id,
        kind = kind.as_str(),
        detail = %detail,
        "the agent's sealed history failed verification; its local state is intact"
    );
    Outcome::ServerFault { kind, detail }
}

/// Whether `sealed/` shows a local loss: an outbox number below
/// `head.json`'s `next_ord` that no file here carries (`sent/`, or the
/// outbox's waiting, writing, set-aside or abandoned items). Each item
/// keeps its number for good, so a gap is a file deleted here.
pub fn local_gap(seal: &AgentSeal, next_ord: u64) -> std::io::Result<bool> {
    let mut seen = std::collections::BTreeSet::new();
    for sub in ["sent", "outbox"] {
        let dir = seal.dir().join(sub);
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e),
        };
        for entry in entries {
            let name = entry?.file_name();
            let name = name.to_string_lossy();
            let digits: String = name.chars().take_while(char::is_ascii_digit).collect();
            if let Ok(ord) = digits.parse::<u64>() {
                seen.insert(ord);
            }
        }
    }
    Ok((0..next_ord).any(|o| !seen.contains(&o)))
}

/// Where a walk to the head stopped: the hash held nowhere, and the seq it
/// should have had.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MissingLink {
    pub hash: Sha256Hex,
    pub seq: Option<u64>,
}

/// [`main_line`], saying where it broke.
pub fn walk_line(
    entries: &[SignedStateEntry],
    head: &Sha256Hex,
    head_seq: Option<u64>,
) -> Result<Vec<SignedStateEntry>, MissingLink> {
    let by_hash: HashMap<Sha256Hex, &SignedStateEntry> =
        entries.iter().map(|e| (e.entry_hash(), e)).collect();
    let mut line = Vec::new();
    let mut at = Some((*head, head_seq));
    while let Some((hash, seq)) = at {
        let Some(entry) = by_hash.get(&hash) else {
            return Err(MissingLink { hash, seq });
        };
        at = entry
            .entry
            .prev_hash
            .map(|p| (p, entry.entry.seq.checked_sub(1)));
        line.push((*entry).clone());
        if line.len() > entries.len() {
            return Err(MissingLink { hash, seq });
        }
    }
    line.reverse();
    Ok(line)
}

/// The seq a chain problem lies at.
fn problem_seq(p: &agora_agentkit::state::verify::ChainProblem) -> Option<u64> {
    use agora_agentkit::state::verify::ChainProblem as P;
    Some(match p {
        P::MissingGenesis => 0,
        P::SeqGap { found, .. } => *found,
        P::GenesisNotFirst { seq }
        | P::WrongAgent { seq }
        | P::UnknownVersion { seq, .. }
        | P::Fork { seq }
        | P::BrokenLink { seq }
        | P::BadSignature { seq }
        | P::UntrustedKey { seq }
        | P::RetiredKey { seq }
        | P::UnknownKind { seq, .. }
        | P::DeltaWithoutBase { seq }
        | P::MissingBlob { seq }
        | P::MergeNotAllowed { seq }
        | P::TimeWentBack { seq } => *seq,
    })
}

/// The main line ending at `head`, oldest first, from every entry the
/// runner holds; `None` if a link is missing.
pub fn main_line(entries: &[SignedStateEntry], head: &Sha256Hex) -> Option<Vec<SignedStateEntry>> {
    walk_line(entries, head, None).ok()
}

/// Every entry the agent's `sealed/` holds: uploaded, then waiting.
pub async fn local_entries(seal: &AgentSeal) -> Result<Vec<SignedStateEntry>, String> {
    let mut items = seal.sent().await.map_err(|e| e.to_string())?;
    items.extend(seal.outbox().await.map_err(|e| e.to_string())?);
    Ok(items.into_iter().map(|i| i.entry).collect())
}

/// `1234567` as `1,234,567`.
fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, d) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(d);
    }
    out
}

fn day(micros: i64) -> Option<String> {
    chrono::DateTime::from_timestamp_micros(micros).map(|d| d.format("%Y-%m-%d").to_string())
}

/// What Agora's copy said, for the verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Server {
    /// Agora's head is this one.
    InStep,
    /// Agora couldn't be asked.
    Unchecked,
    /// Anything else: uploads waiting, or a problem a line of its own
    /// tells.
    Other,
}

/// The version a verdict compares the state's sizes against
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Previous {
    pub sizes: Sizes,
    /// Its date, which the verdict names as the baseline (beta round 10:
    /// "since your last checked version" and "as at your last sealed
    /// version" read as two different versions, or neither)
    pub dated: Option<chrono::NaiveDate>,
    /// Its entry hash, whose prefix names it beside the date: two versions
    /// can share a date (beta round 11: "since your earlier sealed version
    /// dated 2026-11-05" when the head was dated 2026-11-05 too)
    pub head: Option<Sha256Hex>,
    /// The head is the agent's own end-of-session save and this is the
    /// snapshot just before it on the line, and no notice was seated: a
    /// change between them is then the agent's own rewrite. Anything else
    /// (an operator's accept or resume, sizes not recorded for a session
    /// in between) says nothing about whose it is
    pub own_last_session: bool,
}

/// The verdict line for an intact chain: the verdict, the snapshots since
/// the earliest one's date, a pre-loss copy apart, the newest anchor the
/// runner checked ([`super::anchors`]) and, when the verified governance
/// log already holds a newer one (`newer`), that one with when it is
/// checked, else, when the head is newer than the anchored version and
/// the runner does the anchor work for the agent (`checks_later`), that
/// it isn't anchored yet; the head marker, and the state's sizes (counts
/// only) against the earlier version's, dated.
#[allow(clippy::too_many_arguments)]
pub fn verdict_line(
    line: &[SignedStateEntry],
    head: &HeadRef,
    sizes: Sizes,
    previous: Option<Previous>,
    server: Server,
    anchor: Option<&AnchorProof>,
    newer: Option<&GovernanceLogId>,
    checks_later: bool,
) -> String {
    // "Earlier", not "previous": the baseline is the newest version whose
    // sizes were recorded, which may be older than the one just before
    // (review of 0.151.2).
    // "13,539 characters (−1,204 since your earlier sealed version, head
    // be8c50758835 dated 2026-11-04)": named by its head, as this one is,
    // since two can share a date (beta round 11)
    let dated = previous
        .and_then(|p| p.dated)
        .map(|d| format!(" dated {}", d.format("%Y-%m-%d")))
        .unwrap_or_default();
    let baseline = match previous.and_then(|p| p.head) {
        Some(h) => {
            let mut short = h.to_hex();
            short.truncate(12);
            format!(" since your earlier sealed version, head {short}{dated}")
        }
        None => format!(" since your earlier sealed version{dated}"),
    };
    let change = |now: usize, before: Option<usize>, long: bool| -> String {
        let Some(before) = before else {
            return String::new();
        };
        let since = if long { baseline.as_str() } else { "" };
        match now.cmp(&before) {
            std::cmp::Ordering::Equal => " (unchanged)".into(),
            std::cmp::Ordering::Greater => format!(" (+{}{since})", thousands(now - before)),
            std::cmp::Ordering::Less => format!(" (\u{2212}{}{since})", thousands(before - now)),
        }
    };
    let mut memory_change = change(
        sizes.memory_chars,
        previous.map(|p| p.sizes.memory_chars),
        true,
    );
    // A shrink reads as a loss unless it says whose it is (beta round 6),
    // said only when the runner knows: see [`Previous::own_last_session`]
    if previous.is_some_and(|p| p.own_last_session && sizes.memory_chars < p.sizes.memory_chars) {
        memory_change.insert_str(
            memory_change.len() - 1,
            ", as you rewrote it at the end of your last session",
        );
    }
    let soul_change = change(
        sizes.soul_chars,
        previous.map(|p| p.sizes.soul_chars),
        false,
    );
    let verdict = match server {
        Server::InStep => "intact, and in step with Agora.",
        Server::Unchecked => "intact here; Agora's copy couldn't be checked this session.",
        Server::Other => "intact.",
    };
    let snapshots: Vec<&SignedStateEntry> = line
        .iter()
        .filter(|e| e.entry.kind == StateEntryKind::Snapshot)
        .collect();
    let since = snapshots
        .iter()
        .map(|e| e.entry.state_at)
        .min()
        .and_then(day)
        .unwrap_or_else(|| "an unknown date".into());
    let prior = line
        .iter()
        .find(|e| e.entry.kind == StateEntryKind::ArchivedPrior)
        .and_then(|e| day(e.entry.state_at))
        .map(|d| format!(", plus a copy of your state from before a loss, dated {d}"))
        .unwrap_or_default();
    // In sentences, not one run (beta round 11)
    let anchored = anchor
        .map(|a| {
            format!(
                " Your version of {} is in {} (checked by your runner against the governance \
                 log).",
                a.date.format("%Y-%m-%d"),
                a.anc_id
            )
        })
        .unwrap_or_default();
    // The proof lag read as a missed check (beta round 9): proofs are
    // fetched once uploads drain ([`super::drain`]), so a newer anchor can
    // be in the log before its proof is here
    let newer = anchor
        .zip(newer)
        .filter(|(a, n)| *n > &a.anc_id)
        .map(|(_, n)| {
            format!(
                " {n}, a newer anchor, is in the log too: your runner checks your inclusion in \
                 it after uploading this session's saves, and a later session's verdict names it."
            )
        })
        .unwrap_or_default();
    // The version this session starts from, when it is newer than the one
    // anchored and no newer anchor is in the log: the usual case, the last
    // session's saves not anchored yet (beta round 10: an agent who
    // remembered a session after the anchored day was puzzled). Never
    // that this version will be anchored: this session's save replaces it
    // (review of 0.151.1), nor that a later anchor holds it: what an
    // anchor holds is said as the ANC summary says it (beta round 11: "at
    // the time"); as received, since an upload can lag the version's own
    // date (review of 0.151.2). Said only when the runner will check the next anchor: a
    // held agent's anchor work waits
    let unanchored = anchor
        .filter(|a| a.entry_hash != head.entry_hash && newer.is_empty())
        .filter(|_| checks_later)
        .and_then(|_| line.last())
        .and_then(|h| day(h.entry.state_at))
        .map(|d| {
            format!(
                " The version this session starts from, dated {d}, is newer than that anchor. \
                 Each daily anchor holds the latest version Agora had received from you by the \
                 end of that day (UTC); \
                 your runner checks your inclusion after the anchor is written, and a later \
                 session's verdict names it."
            )
        })
        .unwrap_or_default();
    let mut short = head.entry_hash.to_hex();
    short.truncate(12);
    format!(
        "Your sealed state: {verdict} It has {} version{} since \
         {since}{prior}.{anchored}{newer}{unanchored} This session starts from head {short} (a \
         marker for this version; the next session's will differ). In this version, your \
         memory is {} characters{memory_change} and your soul {}{soul_change}. Checked by \
         {VERIFIER}: links, signatures, and that your state matches the last sealed version.",
        snapshots.len(),
        if snapshots.len() == 1 { "" } else { "s" },
        thousands(sizes.memory_chars),
        thousands(sizes.soul_chars),
    )
}

pub const FORK_LINE: &str = "Your sealed state on Agora has moved on from the version this \
     session starts from: two sessions saved from the same version, or Agora holds versions \
     this runner doesn't. This session runs on your state here, and nothing is lost: new \
     versions are kept here, not uploaded, until your operator reconciles the two. Your \
     operator has been told.";

pub const HEAD_MISSING_LINE: &str = "Your latest sealed version is not on Agora: the copy there \
     is missing versions that were sealed here. This session runs on your state here, which \
     matches your last sealed version; new versions are kept here, not uploaded, and your \
     operator has been told.";

pub const UNREADABLE_LINE: &str = "Your sealed history couldn't be read here at the start of \
     this session, so it couldn't be checked. This session runs on your state as it is; its \
     save is kept here but not sealed, and your operator has been told.";

pub const UNCHECKED_LINE: &str = "Your sealed history couldn't be checked at the start of this \
     session: part of it is on Agora, which couldn't be reached. This session runs on your \
     state as it is, and your operator has been told.";

pub const SEAL_FAILED_LINE: &str = "Your last save was written here but could not be sealed, so \
     your sealed history does not include it yet. This session starts from that save; new \
     versions are kept here, not uploaded, until your operator has looked, and your operator \
     has been told.";

/// The sizes of the newest earlier sealed version: from the sidecar, else
/// by opening it, if its blobs are still here.
async fn earlier_sizes(
    agent_dir: &Path,
    seal: &AgentSeal,
    keys: &KeyStore,
    id: AgentId,
    line: &[SignedStateEntry],
    upto: usize,
) -> Option<Recorded> {
    let snapshots: Vec<&SignedStateEntry> = line[..upto]
        .iter()
        .filter(|e| e.entry.kind == StateEntryKind::Snapshot)
        .collect();
    let hashes: Vec<Sha256Hex> = snapshots.iter().map(|e| e.entry_hash()).collect();
    if let Some(r) = SizeLog::load(agent_dir).latest_among(&hashes) {
        return Some(r.clone());
    }
    let last = snapshots.last()?;
    let k = keys.seal_keys(id)?;
    let keyframe = last.entry.blob.as_ref()?.encoding.base().and_then(|base| {
        line.iter()
            .map(|e| &e.entry)
            .find(|e| e.entry_hash() == base)
    });
    let state = seal.open_state(&k, &last.entry, keyframe).await.ok()?;
    Some(Recorded {
        entry_hash: last.entry_hash(),
        seq: last.entry.seq,
        state_at: chrono::DateTime::from_timestamp_micros(last.entry.state_at)?,
        sizes: Sizes::of(&state),
    })
}

/// One enrolled agent's check. `local`: its state's plaintext hash;
/// `sizes`: its memory's and soul's sizes. Names no newer anchor: see
/// [`check_with`].
pub async fn check(
    agent_dir: &Path,
    id: AgentId,
    local: &Sha256Hex,
    sizes: Sizes,
    keys: &KeyStore,
    server: &dyn SealedApi,
) -> Outcome {
    check_with(agent_dir, id, local, sizes, keys, server, None).await
}

/// [`check`], with the newest `ANC-` entry of the governance log verified
/// this run (`logged`): when it is newer than the newest proof kept, the
/// verdict says so, and that the runner checks it after this session's
/// uploads. Only for an agent the drainer will do the anchor work for:
/// sealing, with its keys here, and not held.
pub async fn check_with(
    agent_dir: &Path,
    id: AgentId,
    local: &Sha256Hex,
    sizes: Sizes,
    keys: &KeyStore,
    server: &dyn SealedApi,
    logged: Option<&GovernanceLogId>,
) -> Outcome {
    let seal = AgentSeal::new(agent_dir, id);
    let refuse = |reason: Refusal, detail: String| Outcome::Refuse { reason, detail };
    let unreadable = || Outcome::Run {
        seat: vec![UNREADABLE_LINE.to_string()],
        notice: false,
        hold: Some(Hold::Unreadable),
        drops: Vec::new(),
    };
    let unchecked = || Outcome::Run {
        seat: vec![UNCHECKED_LINE.to_string()],
        notice: false,
        hold: Some(Hold::Unchecked),
        drops: Vec::new(),
    };

    match seal.is_refused().await {
        Ok(false) => {}
        Ok(true) => {
            let why = std::fs::read_to_string(seal.dir().join("refused")).unwrap_or_default();
            let reason = Notice::load(agent_dir)
                .ok()
                .flatten()
                .map_or(Refusal::Mismatch, |n| n.reason);
            return refuse(
                reason,
                format!("still refused ({why}); clear with sealed-accept"),
            );
        }
        Err(e) => {
            tracing::warn!(agent_id = %id, error = %e, "sealed refusal marker unreadable");
            return unreadable();
        }
    }
    let (head, seal_failed) = match seal.recover(Some(local)).await {
        Ok(Recovery::Intact(h) | Recovery::Committed(h)) => (h, None),
        Ok(Recovery::SealFailed(h, kind)) => (h, Some(kind)),
        Ok(Recovery::NoHead) => {
            return refuse(
                Refusal::HeadMissing,
                "enrolled, but sealed/head.json is missing; `sealed-accept` rebuilds it from the chain on Agora if that verifies".into(),
            );
        }
        Ok(Recovery::Mismatch(h) | Recovery::NoLocalState(h)) => {
            return refuse(
                Refusal::Mismatch,
                format!(
                    "local state matches neither head {} nor a pending version",
                    h.entry_hash
                ),
            );
        }
        Ok(Recovery::NotEnrolled) => {
            return Outcome::Run {
                seat: Vec::new(),
                notice: false,
                hold: None,
                drops: Vec::new(),
            };
        }
        Ok(Recovery::Unreadable(kind)) => {
            tracing::warn!(agent_id = %id, kind = ?kind, "sealed state unreadable");
            return unreadable();
        }
        Err(e) => {
            tracing::warn!(agent_id = %id, error = %e, kind = ?e.kind(), "sealed state unreadable");
            return unreadable();
        }
    };
    let entries = match local_entries(&seal).await {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!(agent_id = %id, error = %e, "local sealed chain unreadable");
            return unreadable();
        }
    };
    // After a rebase the main line runs through Agora's entries, which
    // aren't kept here: they are read from Agora and verified below with
    // the runner's own trust, never taken on its word. Local entries win;
    // what fails only on Agora's data is a hold, never a refusal (no
    // refusal on the server's evidence alone)
    let mut fetched: Option<Result<Vec<SignedStateEntry>, String>> = None;
    // Hashes of the entries only Agora holds, and the seq Agora's chain
    // reaches: what is Agora's evidence, and what isn't
    let mut server_only: HashMap<Sha256Hex, u64> = HashMap::new();
    let mut walked = walk_line(&entries, &head.entry_hash, Some(head.seq));
    // A file deleted here is local evidence, whatever Agora says
    let next_ord = match seal.head().await {
        Ok(Some(h)) => h.next_ord,
        _ => 0,
    };
    let gap = match local_gap(&seal, next_ord) {
        Ok(gap) => gap,
        Err(e) => {
            tracing::warn!(agent_id = %id, error = %e, "sealed/ unreadable for its numbering");
            return unreadable();
        }
    };
    // An entry held here at the missing link's seq, under another hash:
    // that entry changed here
    let corrupted = walked.as_ref().err().is_some_and(|m| {
        m.seq.is_some_and(|s| {
            entries
                .iter()
                .any(|e| e.entry.seq == s && e.entry_hash() != m.hash)
        })
    });
    if let Err(missing) = &walked
        && (gap || corrupted)
    {
        return refuse(
            Refusal::ChainBroken,
            format!(
                "the main line to head {} lacks {} (seq {:?}), and sealed/ is missing an item it \
                 numbered",
                head.entry_hash, missing.hash, missing.seq
            ),
        );
    }
    if walked.is_err()
        && let Some(signing) = keys.signing(id)
    {
        let chain = super::api::chain(server, id, &signing).await;
        if let Ok(chain) = &chain {
            let all = match merge_server(&entries, chain) {
                Ok(all) => all,
                Err(hash) => return disagreement(id, hash, &entries, keys),
            };
            let ours: std::collections::HashSet<Sha256Hex> =
                entries.iter().map(|e| e.entry_hash()).collect();
            server_only = chain
                .iter()
                .filter(|e| !ours.contains(&e.entry_hash()))
                .map(|e| (e.entry_hash(), e.entry.seq))
                .collect();
            walked = walk_line(&all, &head.entry_hash, Some(head.seq));
            // Nothing here was lost (no gap): what is missing is Agora's
            if let Err(missing) = &walked {
                return on_fault(
                    id,
                    FaultKind::ServerChainBroken,
                    format!(
                        "the main line through Agora's chain lacks {} (seq {:?})",
                        missing.hash, missing.seq
                    ),
                );
            }
        }
        fetched = Some(chain);
    }
    let line = match (walked, &fetched) {
        (Ok(line), _) => line,
        // Agora couldn't be asked, or there is no key to ask with: no
        // evidence either way, so the agent runs, unchecked
        (Err(_), Some(Err(e))) => {
            tracing::warn!(agent_id = %id, error = %e, "the main line runs through Agora's chain, which couldn't be read");
            return unchecked();
        }
        (Err(_), _) => return unchecked(),
    };
    // Keys that can't be read are no evidence against the chain
    let trusted = match keys.trusted(id) {
        Ok(Some(t)) => t,
        Ok(None) => {
            tracing::warn!(agent_id = %id, "no signing key to verify the sealed chain with");
            return unreadable();
        }
        Err(e) => {
            tracing::warn!(agent_id = %id, error = %e, "the agent's key records are unreadable");
            return unreadable();
        }
    };
    let report = verify_chain(id, &line, &trusted);
    for notice in &report.notices {
        tracing::warn!(event_type = "sealed_state_key_notice", agent_id = %id, notice = ?notice, "sealed-state key notice");
    }
    if !report.intact || report.head != Some(head) {
        let server_hashes: std::collections::HashSet<Sha256Hex> =
            server_only.keys().copied().collect();
        return judge_failure(
            id,
            &report.problems,
            &line,
            &server_hashes,
            &known_keys(keys, id),
        );
    }

    // Every entry held here is on the main line (a set-aside fork is kept
    // under another name): one that isn't was changed here
    let on_line: std::collections::HashSet<Sha256Hex> =
        line.iter().map(|e| e.entry_hash()).collect();
    if let Some(stray) = entries.iter().find(|e| !on_line.contains(&e.entry_hash())) {
        return refuse(
            Refusal::ChainBroken,
            format!(
                "entry seq {} held here ({}) is not on the main line: changed here",
                stray.entry.seq,
                stray.entry_hash()
            ),
        );
    }

    let mut seat = Vec::new();
    let mut notice = false;
    if let Ok(Some(n)) = Notice::load(agent_dir)
        && let Some(text) = n.line()
    {
        seat.push(text);
        notice = true;
    }
    // Against the head when this state isn't it (a failed seal), else
    // against the version before it
    let upto = if seal_failed.is_some() {
        line.len()
    } else {
        line.len() - 1
    };
    let before = earlier_sizes(agent_dir, &seal, keys, id, &line, upto).await;
    // The snapshot just before the head: compared with that one, and with
    // no notice of an operator's act seated, the head is the agent's own
    // end-of-session save and the change its own rewrite
    let prior_snapshot = line[..line.len() - 1]
        .iter()
        .rev()
        .find(|e| e.entry.kind == StateEntryKind::Snapshot)
        .map(|e| e.entry.seq);
    let own_last_session = !notice
        && seal_failed.is_none()
        && line
            .last()
            .is_some_and(|h| h.entry.kind == StateEntryKind::Snapshot)
        && before
            .as_ref()
            .is_some_and(|b| Some(b.seq) == prior_snapshot);
    let previous = before.as_ref().map(|b| Previous {
        sizes: b.sizes,
        dated: Some(b.state_at.date_naive()),
        head: Some(b.entry_hash),
        own_last_session,
    });
    let drops: Vec<(&'static str, String, u32)> = match &before {
        Some(before) => drop_lines(before, sizes)
            .into_iter()
            .map(|(what, pct, text)| (what, text, pct))
            .collect(),
        None => Vec::new(),
    };
    if seal_failed.is_none()
        && let Some(last) = line.last()
        && let Some(state_at) = chrono::DateTime::from_timestamp_micros(last.entry.state_at)
        && let Err(e) = SizeLog::record(
            agent_dir,
            Recorded {
                entry_hash: head.entry_hash,
                seq: head.seq,
                state_at,
                sizes,
            },
        )
    {
        tracing::warn!(agent_id = %id, error = %e, "sealed sizes not recorded");
    }
    // Read here, never fetched: no network for this at the start
    let anchor = super::anchors::AnchorsFile::newest(agent_dir);
    let drop_lines: Vec<String> = drops.iter().map(|d| d.1.clone()).collect();
    let drops: Vec<(&'static str, u32)> = drops.iter().map(|d| (d.0, d.2)).collect();
    if seal_failed.is_some() {
        seat.push(SEAL_FAILED_LINE.into());
        seat.extend(drop_lines);
        return Outcome::Run {
            seat,
            notice,
            hold: Some(Hold::SealFailed),
            drops,
        };
    }
    let Some(signing) = keys.signing(id) else {
        seat.push(verdict_line(
            &line,
            &head,
            sizes,
            previous,
            Server::Unchecked,
            anchor.as_ref(),
            None,
            false,
        ));
        seat.extend(drop_lines);
        return Outcome::Run {
            seat,
            notice,
            hold: None,
            drops,
        };
    };
    let outbox: Vec<_> = match seal.outbox().await {
        Ok(items) => items.into_iter().map(|i| i.entry.entry).collect(),
        Err(_) => Vec::new(),
    };
    let chain = match fetched {
        Some(chain) => chain,
        None => super::api::chain(server, id, &signing).await,
    };
    let (status, server_said) = match chain {
        Ok(chain) => {
            // Agora's copy of an entry held here must be ours
            if let Err(hash) = merge_server(&entries, &chain) {
                return disagreement(id, hash, &entries, keys);
            }
            let hashes: Vec<Sha256Hex> = chain.iter().map(|e| e.entry_hash()).collect();
            let mut status = head_status(Some(&head.entry_hash), &outbox, &hashes);
            // And every version uploaded from here must still be there, not
            // just the head
            if matches!(status, HeadStatus::Same | HeadStatus::Pending) {
                let on_agora: std::collections::HashSet<&Sha256Hex> = hashes.iter().collect();
                let lost = match seal.sent().await {
                    Ok(sent) => sent
                        .iter()
                        .filter(|i| !on_agora.contains(&i.entry.entry_hash()))
                        .count(),
                    Err(_) => 0,
                };
                if lost > 0 {
                    tracing::warn!(
                        event_type = "sealed_state_uploaded_missing",
                        agent_id = %id,
                        lost,
                        "versions uploaded from here are missing from Agora's chain"
                    );
                    status = HeadStatus::LocalMissing;
                }
            }
            let said = if status == HeadStatus::Same {
                Server::InStep
            } else {
                Server::Other
            };
            (status, said)
        }
        Err(e) => {
            tracing::warn!(
                event_type = "sealed_state_server_unchecked",
                agent_id = %id,
                error = %e,
                "the copy on Agora could not be checked; running on the local check"
            );
            (HeadStatus::Pending, Server::Unchecked)
        }
    };
    // The drainer does the anchor work only for an agent not held, still
    // sealing ([`super::drain::Drainer::anchor_all`])
    let checks_later =
        matches!(status, HeadStatus::Same | HeadStatus::Pending) && super::is_sealing(agent_dir);
    let newer = logged.filter(|_| checks_later);
    seat.push(verdict_line(
        &line,
        &head,
        sizes,
        previous,
        server_said,
        anchor.as_ref(),
        newer,
        checks_later,
    ));
    seat.extend(drop_lines);
    let hold = match status {
        HeadStatus::Same | HeadStatus::Pending => None,
        HeadStatus::LocalMissing => {
            seat.push(HEAD_MISSING_LINE.into());
            Some(Hold::HeadMissing)
        }
        HeadStatus::ServerAhead | HeadStatus::Diverged => {
            seat.push(FORK_LINE.into());
            Some(Hold::Fork)
        }
    };
    Outcome::Run {
        seat,
        notice,
        hold,
        drops,
    }
}

/// Agents enrolled by their keys (a state key in the secrets tree) whose
/// `state.json` is gone, with the whole directory or alone: refused
/// ([`Refusal::StateMissing`]), since a deletion is the May loss's other
/// face. Unless Agora answers that it holds no sealed head for one
/// (`head: null`, never uploaded; or a 404, an unknown or deleted agent
/// whose secrets were left behind), which is warned, alerted and skipped. Agora unreachable, or no key to ask with: refused
/// (fail closed).
pub async fn missing_states(
    state_dir: &Path,
    keys: &KeyStore,
    server: &dyn SealedApi,
    alerts: &Alerter,
) -> Vec<(AgentId, Refusal)> {
    let ids = match keys.with_state_keys() {
        Ok(ids) => ids,
        Err(e) => {
            tracing::warn!(error = %e, "the secrets tree couldn't be listed for missing states");
            return Vec::new();
        }
    };
    let mut missing = Vec::new();
    for id in ids {
        let dir = state_dir.join(id.to_string());
        let state_here = std::fs::exists(dir.join("state.json")).unwrap_or(true);
        // A state whose sealed/ (its enrolment marker) is gone would run
        // unsealed and never be checked again: as missing as a lost state
        let reason = if !state_here {
            Refusal::StateMissing
        } else if !super::is_enrolled(&dir) {
            Refusal::HeadMissing
        } else {
            // There and enrolled: it must load, or it is never checked
            use agora_agentkit::reactor::Storage as _;
            match agora_agentkit::reactor::FsStorage::new(state_dir)
                .load_raw(id)
                .await
            {
                Ok(_) => {}
                Err(e) => {
                    use agora_agentkit::reactor::storage::FsStorageError as E;
                    let reason = match &e {
                        E::Format { .. } => Refusal::StateTooNew,
                        E::Io(_) => Refusal::StateIo,
                        _ => Refusal::StateUnreadable,
                    };
                    refuse(
                        &dir,
                        id,
                        &format!("({})", reason.as_str()),
                        reason,
                        &e.to_string(),
                        alerts,
                    )
                    .await;
                    missing.push((id, reason));
                }
            }
            continue;
        };
        let known = match keys.signing(id) {
            Some(signing) => server.head_known(id, &signing).await,
            None => Err("no signing key to ask Agora with".into()),
        };
        match known {
            Ok(false) => {
                tracing::warn!(
                    event_type = "sealed_state_keys_without_state",
                    agent_id = %id,
                    "a state key is here but no state, and Agora holds no sealed head for it; skipped"
                );
                // A state here: an enrolment cut after its key was cached.
                // None: an agent gone, its keys left behind
                let (summary, what, command) = if state_here {
                    (
                        format!(
                            "{id}: an enrolment that didn't finish (a state key, no enrolment marker, nothing on Agora)"
                        ),
                        "resume",
                        format!("agora-seed sealed-enroll --agent {id}"),
                    )
                } else {
                    (
                        format!(
                            "{id}: a state key but no state here, and nothing on Agora to restore"
                        ),
                        "forget",
                        format!("agora-seed sealed-forget --agent {id} --note \"…\""),
                    )
                };
                alerts.notify(
                    Alert::new(AlertKind::SealedStateKeysWithoutState, summary)
                        .agent(id.to_string(), id)
                        .detail(what, command),
                );
                continue;
            }
            Ok(true) => {}
            Err(e) => tracing::warn!(
                agent_id = %id,
                error = %e,
                "Agora couldn't say whether it holds the missing state's head; refused"
            ),
        }
        // Marked (the directory recreated if it is gone), so a session
        // journal left behind doesn't block the restore
        refuse(
            &dir,
            id,
            &format!("({})", reason.as_str()),
            reason,
            "an enrolled agent's state or sealed/ is missing; `agora-seed sealed-restore` restores it from Agora",
            alerts,
        )
        .await;
        missing.push((id, reason));
    }
    missing
}

/// Rotations the server lists that the runner's own `rotations.json`
/// doesn't: advisory, never trusted, but the operator hears of them (a
/// server that reports a rotation the runner never made). `None` when the
/// server couldn't be asked.
pub async fn unknown_rotations(
    keys: &KeyStore,
    server: &dyn SealedApi,
    id: AgentId,
) -> Option<usize> {
    let signing = keys.signing(id)?;
    let listed = match server.rotations(id, &signing).await {
        Ok(listed) => listed,
        Err(e) => {
            tracing::warn!(agent_id = %id, error = %e, "the server's rotations could not be read");
            return None;
        }
    };
    let ours = keys.rotations(id).unwrap_or_default();
    Some(
        listed
            .iter()
            .filter(|r| {
                !ours
                    .iter()
                    .any(|o| o.seq == r.seq && o.new_public_key == r.new_public_key)
            })
            .count(),
    )
}

/// Act on a refusal: `sealed/refused` (blocks every seal), the notice for
/// the agent, the alert. Never touches the state.
pub async fn refuse(
    agent_dir: &Path,
    id: AgentId,
    name: &str,
    reason: Refusal,
    detail: &str,
    alerts: &Alerter,
) {
    let seal = AgentSeal::new(agent_dir, id);
    if !seal.is_refused().await.unwrap_or(false)
        && let Err(e) = seal.refuse(reason.as_str()).await
    {
        tracing::error!(agent_id = %id, error = %e, "could not write the refusal marker");
    }
    if let Err(e) = Notice::refused(agent_dir, reason) {
        tracing::error!(agent_id = %id, error = %e, "could not write the refusal notice");
    }
    tracing::error!(
        event_type = "sealed_state_refused",
        agent = name,
        agent_id = %id,
        reason = reason.as_str(),
        detail,
        "sealed state did not check out; the agent is left out until `agora-seed sealed-accept`"
    );
    alerts.notify(
        Alert::new(
            AlertKind::SealedStateRefused,
            format!("{name}'s session was refused: {}", reason.plain()),
        )
        .agent(name, id)
        .detail("reason", reason.as_str())
        .detail("detail", detail),
    );
}

/// The alert for a memory or soul that shrank.
pub fn drop_alert(id: AgentId, name: &str, drops: &[(&'static str, u32)]) -> Alert {
    let mut alert = Alert::new(
        AlertKind::SealedStateDrop,
        format!("{name}: memory or soul much smaller than at the previous sealed version"),
    )
    .agent(name, id);
    for (what, pct) in drops {
        alert = alert.detail("smaller", format!("{what} by {pct}%"));
    }
    alert
}

/// The alert for a run that goes on held.
pub fn hold_alert(
    hold: Hold,
    id: AgentId,
    name: &str,
    held: Option<&agora_agentkit::state::storage::Held>,
) -> Alert {
    let (kind, summary) = match hold {
        Hold::SealFailed => (
            AlertKind::SealedStateSealFailed,
            "the last save was written unsealed; uploads held",
        ),
        Hold::HeadMissing => (
            AlertKind::SealedStateHeadMissing,
            "the local sealed head is not on Agora; uploads held",
        ),
        Hold::Fork => (
            AlertKind::SealedStateFork,
            "the sealed chain on Agora moved on from the local head; uploads held",
        ),
        Hold::Unreadable => (
            AlertKind::SealedStateUnreadable,
            "sealed/ or its keys could not be read; the agent runs, uploads held",
        ),
        Hold::Unchecked => (
            AlertKind::SealedStateUnreadable,
            "the sealed chain could not be checked (Agora unreachable); the agent runs, uploads held",
        ),
    };
    let mut alert = Alert::new(kind, format!("{name}: {summary}")).agent(name, id);
    if let Some(held) = held {
        alert = alert
            .detail("held", held.outbox_items)
            .detail("blobs", held.blobs)
            .detail("blob_bytes", held.blob_bytes)
            .detail("forks", held.displaced);
    }
    if hold == Hold::Fork {
        alert = alert.detail(
            "resolve",
            format!("agora-seed sealed-accept --agent {id} --rebase --note \"…\""),
        );
    }
    alert
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use agora_agentkit::reactor::{FsStorage, Storage};
    use agora_agentkit::state::codec::KeyframePolicy;
    use agora_agentkit::state::crypto::StateKey;
    use agora_agentkit::state::storage::plaintext_hash;
    use agora_agentkit::state::{GenesisClass, GenesisOrigin};

    use crate::sealed::api::fake::FakeApi;

    /// A server that holds exactly these entries.
    fn server(chain: Vec<SignedStateEntry>) -> FakeApi {
        let api = FakeApi::new();
        *api.chain.lock().unwrap() = Ok(chain);
        api
    }

    pub fn state(n: usize) -> serde_json::Value {
        serde_json::json!({ "memory": format!("synthetic {n}") })
    }

    /// A synthetic enrolled agent: keys, a state, genesis and that state
    /// sealed, enrolled.
    pub async fn enrolled(root: &Path) -> (AgentId, KeyStore, std::path::PathBuf) {
        let id = AgentId::new();
        let keys = KeyStore::new(root.join("secrets"));
        keys.ring()
            .insert(id, &agora_agentkit::crypto::generate_keypair().0)
            .unwrap();
        keys.write_state_key(
            id,
            &super::super::keys::StateKeyFile::new(&StateKey::generate(), 0, 0),
        )
        .unwrap();
        let mut fs = FsStorage::new(root.join("state"));
        fs.save_raw(id, state(1), None).await.unwrap();
        let dir = fs.agent_dir(id);
        let seal = AgentSeal::new(&dir, id);
        let k = keys.seal_keys(id).unwrap();
        let origin = GenesisOrigin {
            class: GenesisClass::None,
            incident: None,
            prior_copy_held: false,
            verified_at: None,
        };
        seal.genesis(&k, &origin, chrono::Utc::now()).await.unwrap();
        seal.append_state(
            &k,
            &KeyframePolicy::default(),
            StateEntryKind::Snapshot,
            &state(1),
            chrono::Utc::now(),
        )
        .await
        .unwrap();
        seal.enroll().await.unwrap();
        (id, keys, dir)
    }

    async fn local_hash(root: &Path, id: AgentId) -> Sha256Hex {
        let v = FsStorage::new(root.join("state"))
            .load_raw(id)
            .await
            .unwrap()
            .value;
        plaintext_hash(&v)
    }

    async fn sealed_chain(dir: &Path, id: AgentId) -> Vec<SignedStateEntry> {
        local_entries(&AgentSeal::new(dir, id)).await.unwrap()
    }

    #[tokio::test]
    async fn an_intact_state_runs_with_the_verdict_line() {
        let root = crate::sealed::TestDir::new("check-intact");
        let (id, keys, dir) = enrolled(&root).await;
        let server = server(Vec::new());
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &server,
        )
        .await;
        let Outcome::Run {
            seat, notice, hold, ..
        } = out
        else {
            panic!("{out:?}")
        };
        assert_eq!(
            (notice, hold),
            (false, None),
            "pending upload is no problem"
        );
        assert_eq!(seat.len(), 1);
        assert!(
            seat[0].starts_with("Your sealed state: intact. It has 1 version since "),
            "{}",
            seat[0]
        );
        assert!(
            seat[0].contains(VERIFIER)
                && seat[0].contains("(a marker for this version; the next session's will differ)")
                && !seat[0].contains("history tools")
        );

        // Agora has all of it: the same
        *server.chain.lock().unwrap() = Ok(sealed_chain(&dir, id).await);
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &server,
        )
        .await;
        let Outcome::Run {
            seat, hold: None, ..
        } = out
        else {
            panic!("{out:?}")
        };
        assert!(
            seat[0].starts_with("Your sealed state: intact, and in step with Agora."),
            "{seat:?}"
        );
        // Agora unreachable: said so
        *server.chain.lock().unwrap() = Err("unreachable".into());
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &server,
        )
        .await;
        let Outcome::Run { seat, .. } = out else {
            panic!("{out:?}")
        };
        assert!(
            seat[0].starts_with(
                "Your sealed state: intact here; Agora's copy couldn't be checked this session."
            ),
            "{seat:?}"
        );
    }

    /// The newest kept anchor proof is named, read from sealed/anchors.json
    /// at the check; none kept, no clause.
    #[tokio::test]
    async fn the_verdict_names_the_newest_kept_anchor() {
        let root = crate::sealed::TestDir::new("check-anchor");
        let (id, keys, dir) = enrolled(&root).await;
        let server = server(Vec::new());
        let verdict_with = async |logged: Option<&GovernanceLogId>| {
            let out = check_with(
                &dir,
                id,
                &local_hash(&root, id).await,
                Sizes::default(),
                &keys,
                &server,
                logged,
            )
            .await;
            let Outcome::Run { seat, .. } = out else {
                panic!("{out:?}")
            };
            seat[0].clone()
        };
        let verdict = async || verdict_with(None).await;
        let newer: GovernanceLogId = "ANC-2026-0004".parse().unwrap();
        assert!(!verdict().await.contains("governance log"));
        assert!(!verdict_with(Some(&newer)).await.contains("ANC-"));
        let head = AgentSeal::new(&dir, id).head().await.unwrap().unwrap().head;
        let proof = |n: u32| AnchorProof {
            anc_id: format!("ANC-2026-{n:04}").parse().unwrap(),
            date: chrono::NaiveDate::from_ymd_opt(2026, 11, n).unwrap(),
            root: Sha256Hex::from([1; 32]),
            entry_hash: head.entry_hash,
            seq: head.seq,
            leaf_index: 0,
            leaf_count: 1,
            proof: Vec::new(),
        };
        let file = crate::sealed::anchors::AnchorsFile {
            proofs: vec![proof(2), proof(3)],
        };
        std::fs::write(
            dir.join("sealed").join(crate::sealed::anchors::ANCHORS),
            serde_json::to_vec(&file).unwrap(),
        )
        .unwrap();
        let line = verdict().await;
        assert!(
            line.contains(
                ". Your version of 2026-11-03 is in ANC-2026-0003 (checked by your runner \
                 against the governance log). This session starts from head "
            ),
            "{line}"
        );
        // The log's newest is the newest kept: the same line
        let same: GovernanceLogId = "ANC-2026-0003".parse().unwrap();
        assert_eq!(verdict_with(Some(&same)).await, line);
        // A newer one in the verified log: named, with when it is checked
        let lagging = verdict_with(Some(&newer)).await;
        assert!(
            lagging.contains(
                "(checked by your runner against the governance log). ANC-2026-0004, a newer \
                 anchor, is in the log too: your runner checks your inclusion in it after \
                 uploading this session's saves, and a later session's verdict names it. This \
                 session starts from head "
            ),
            "{lagging}"
        );
        assert_eq!(
            server
                .calls
                .lock()
                .unwrap()
                .iter()
                .filter(|c| *c == "proofs")
                .count(),
            0,
            "no network for it"
        );
    }

    #[tokio::test]
    async fn a_wiped_state_is_refused_and_never_sealed() {
        let root = crate::sealed::TestDir::new("check-wipe");
        let (id, keys, dir) = enrolled(&root).await;
        let mut fs = FsStorage::new(root.join("state"));
        fs.save_raw(id, serde_json::json!({ "memory": "" }), None)
            .await
            .unwrap();
        let server = server(Vec::new());
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &server,
        )
        .await;
        let Outcome::Refuse { reason, .. } = out else {
            panic!("{out:?}")
        };
        assert_eq!(reason, Refusal::Mismatch);
        refuse(&dir, id, "tarn", reason, "test", &Alerter::off()).await;
        assert!(AgentSeal::new(&dir, id).is_refused().await.unwrap());
        assert_eq!(
            Notice::load(&dir).unwrap().unwrap().reason,
            Refusal::Mismatch
        );
        // Still refused on the next start, whatever the state
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &server,
        )
        .await;
        assert!(matches!(out, Outcome::Refuse { .. }), "{out:?}");
    }

    #[tokio::test]
    async fn a_missing_head_file_is_refused() {
        let root = crate::sealed::TestDir::new("check-nohead");
        let (id, keys, dir) = enrolled(&root).await;
        let sealed = dir.join("sealed");
        std::fs::rename(sealed.join("head.json"), sealed.join("head.json.moved")).unwrap();
        let server = server(Vec::new());
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &server,
        )
        .await;
        assert!(
            matches!(
                out,
                Outcome::Refuse {
                    reason: Refusal::HeadMissing,
                    ..
                }
            ),
            "{out:?}"
        );
    }

    /// Entries held here signed by a key this runner never held: changed
    /// here (another runner's, or corrupted), so refused and marked, for
    /// `sealed-accept` or `sealed-restore` to resolve. (A key it did hold,
    /// its rotation record lost, is an untrusted key: `props`.)
    #[tokio::test]
    async fn a_chain_signed_by_an_untrusted_key_is_refused() {
        let root = crate::sealed::TestDir::new("check-key");
        let (id, _, dir) = enrolled(&root).await;
        // Another runner's keys: the chain's signer isn't trusted
        let other = KeyStore::new(root.join("other-secrets"));
        other
            .ring()
            .insert(id, &agora_agentkit::crypto::generate_keypair().0)
            .unwrap();
        let server = server(Vec::new());
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &other,
            &server,
        )
        .await;
        assert!(
            matches!(
                out,
                Outcome::Refuse {
                    reason: Refusal::ChainBroken,
                    ..
                }
            ),
            "{out:?}"
        );
    }

    #[tokio::test]
    async fn a_server_that_moved_on_is_a_held_fork_and_a_missing_head_is_held() {
        let root = crate::sealed::TestDir::new("check-fork");
        let (id, keys, dir) = enrolled(&root).await;
        let mine = sealed_chain(&dir, id).await;
        // Agora holds a sibling of our last version
        let signing = keys.signing(id).unwrap();
        let mut sibling = mine[1].entry.clone();
        sibling.state_at += 1;
        let server_chain = vec![mine[0].clone(), sibling.sign(&signing)];
        let server = server(server_chain);
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &server,
        )
        .await;
        let Outcome::Run { seat, hold, .. } = out else {
            panic!("{out:?}")
        };
        assert_eq!(hold, Some(Hold::Fork));
        assert_eq!(seat.last().unwrap(), FORK_LINE);

        // Everything uploaded, nothing waiting, and Agora has lost our
        // last version: the head is missing there
        let seal = AgentSeal::new(&dir, id);
        for item in seal.outbox().await.unwrap() {
            seal.mark_sent(item.ord).await.unwrap();
        }
        *server.chain.lock().unwrap() = Ok(vec![mine[0].clone()]);
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &server,
        )
        .await;
        let Outcome::Run { seat, hold, .. } = out else {
            panic!("{out:?}")
        };
        assert_eq!(hold, Some(Hold::HeadMissing));
        assert_eq!(seat.last().unwrap(), HEAD_MISSING_LINE);
    }

    #[tokio::test]
    async fn a_failed_seal_runs_held() {
        let root = crate::sealed::TestDir::new("check-sealfail");
        let (id, keys, dir) = enrolled(&root).await;
        let mut fs = FsStorage::new(root.join("state"));
        fs.save_raw(id, state(2), None).await.unwrap();
        let local = local_hash(&root, id).await;
        AgentSeal::new(&dir, id)
            .record_seal_failed(local, agora_agentkit::state::storage::SealFailureKind::Io)
            .await
            .unwrap();
        let server = server(Vec::new());
        let out = check(&dir, id, &local, Sizes::default(), &keys, &server).await;
        let Outcome::Run { seat, hold, .. } = out else {
            panic!("{out:?}")
        };
        assert_eq!(hold, Some(Hold::SealFailed));
        assert_eq!(seat, vec![SEAL_FAILED_LINE.to_string()]);
    }

    /// A memory wiped by a session and sealed by its save: the next start
    /// is intact, but tells of the drop. Sizes from the sidecar, or by
    /// opening the earlier version when there is none.
    #[tokio::test]
    async fn a_memory_drop_is_told_but_not_refused() {
        use agora_agentkit::state::storage::Mirrored;
        let root = crate::sealed::TestDir::new("check-drop");
        let (id, keys, dir) = enrolled(&root).await;
        let full =
            serde_json::json!({ "memory": { "content": "x".repeat(1000) }, "soul": {"n": 1} });
        let wiped = serde_json::json!({ "memory": { "content": "" }, "soul": {"n": 1} });
        let enrolment = std::sync::Arc::new(crate::sealed::SealedEnrolment::new(
            root.join("state"),
            root.join("secrets"),
        ));
        let mut store = Mirrored::new(FsStorage::new(root.join("state")), (), enrolment);
        let base = store.load_raw(id).await.unwrap().base;
        let base = store
            .save_raw(id, full.clone(), base.as_ref())
            .await
            .unwrap();
        let server = server(Vec::new());
        let out = check(
            &dir,
            id,
            &plaintext_hash(&full),
            Sizes::of(&full),
            &keys,
            &server,
        )
        .await;
        assert!(
            matches!(&out, Outcome::Run { drops, .. } if drops.is_empty()),
            "{out:?}"
        );
        store
            .save_raw(id, wiped.clone(), base.as_ref())
            .await
            .unwrap();

        for sidecar in [true, false] {
            if !sidecar {
                std::fs::rename(
                    dir.join("sealed/sizes.json"),
                    dir.join("sealed/sizes.moved"),
                )
                .unwrap();
            }
            let out = check(
                &dir,
                id,
                &plaintext_hash(&wiped),
                Sizes::of(&wiped),
                &keys,
                &server,
            )
            .await;
            let Outcome::Run {
                seat, drops, hold, ..
            } = out
            else {
                panic!("{out:?}")
            };
            assert_eq!(hold, None);
            assert_eq!(drops, vec![("memory", 100)], "sidecar {sidecar}");
            assert!(
                seat.last()
                    .unwrap()
                    .starts_with("Your memory is 100% smaller than at your last sealed version ("),
                "{seat:?}"
            );
        }
    }

    /// A shrink is said to be the agent's own rewrite only when the head
    /// is its end-of-session save and the compared version the one just
    /// before it; with a session's sizes unrecorded in between, it isn't
    /// (beta round 6; an operator's accept is in `accept`'s tests)
    #[tokio::test]
    async fn a_shrink_is_the_agents_own_only_against_the_version_just_before() {
        use agora_agentkit::state::storage::Mirrored;
        let root = crate::sealed::TestDir::new("check-own-shrink");
        let (id, keys, dir) = enrolled(&root).await;
        let mem = |n: usize| serde_json::json!({ "memory": { "content": "x".repeat(n) } });
        let enrolment = std::sync::Arc::new(crate::sealed::SealedEnrolment::new(
            root.join("state"),
            root.join("secrets"),
        ));
        let mut store = Mirrored::new(FsStorage::new(root.join("state")), (), enrolment);
        let server = server(Vec::new());
        let verdict = async |state: &serde_json::Value| {
            let out = check(
                &dir,
                id,
                &plaintext_hash(state),
                Sizes::of(state),
                &keys,
                &server,
            )
            .await;
            let Outcome::Run { seat, notice, .. } = out else {
                panic!("{out:?}")
            };
            assert!(!notice);
            seat.into_iter()
                .find(|l| l.starts_with("Your sealed state: "))
                .expect("a verdict")
        };
        let mut base = store.load_raw(id).await.unwrap().base;
        for (n, own) in [(1000, None), (900, Some(true))] {
            base = store.save_raw(id, mem(n), base.as_ref()).await.unwrap();
            let line = verdict(&mem(n)).await;
            match own {
                None => {}
                Some(_) => assert!(
                    line.contains("your memory is 900 characters (\u{2212}100 since your earlier sealed version, head ")
                        && line.contains(", as you rewrote it at the end of your last session)"),
                    "{line}"
                ),
            }
        }
        // A version sealed with no check after it (its session's sizes not
        // recorded): the compared one is two back, and the change may span
        // sessions
        base = store.save_raw(id, mem(800), base.as_ref()).await.unwrap();
        store.save_raw(id, mem(700), base.as_ref()).await.unwrap();
        let line = verdict(&mem(700)).await;
        assert!(
            line.contains(
                "your memory is 700 characters (\u{2212}200 since your earlier sealed version, head "
            ) && !line.contains("rewrote"),
            "{line}"
        );
        assert!(!line.contains("rewrote"), "{line}");
    }

    /// A `sealed/` that can't be read never refuses on its own: the agent
    /// runs, told, held.
    #[tokio::test]
    async fn an_unreadable_seal_runs_held_and_is_never_refused() {
        let root = crate::sealed::TestDir::new("check-unreadable");
        let (id, keys, dir) = enrolled(&root).await;
        std::fs::write(dir.join("sealed/head.json"), b"{ not json").unwrap();
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &server(Vec::new()),
        )
        .await;
        assert_eq!(
            out,
            Outcome::Run {
                seat: vec![UNREADABLE_LINE.to_string()],
                notice: false,
                hold: Some(Hold::Unreadable),
                drops: Vec::new(),
            }
        );
    }

    /// A rotation the server lists and the runner never made is counted,
    /// never trusted.
    #[tokio::test]
    async fn an_unknown_server_rotation_is_counted() {
        let root = crate::sealed::TestDir::new("check-rotations");
        let (id, keys, _) = enrolled(&root).await;
        let api = server(Vec::new());
        assert_eq!(unknown_rotations(&keys, &api, id).await, Some(0));
        let (other, _) = agora_agentkit::crypto::generate_keypair();
        let planted = agora_agentkit::state::KeyRotation::agent_signed(
            id,
            0,
            &keys.signing(id).unwrap(),
            &other,
            None,
            chrono::Utc::now(),
        );
        api.rotations.lock().unwrap().push(planted.clone());
        assert_eq!(unknown_rotations(&keys, &api, id).await, Some(1));
        keys.record_rotation(id, &planted).unwrap();
        assert_eq!(unknown_rotations(&keys, &api, id).await, Some(0));
    }

    /// No refusal without evidence: unreadable key records, or a chain
    /// that runs through an unreachable Agora, run held.
    #[tokio::test]
    async fn transient_failures_run_held_and_are_never_refused() {
        let root = crate::sealed::TestDir::new("check-transient");
        let (id, keys, dir) = enrolled(&root).await;
        std::fs::write(keys.agent_dir(id).join("rotations.json"), b"{ not json").unwrap();
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &server(Vec::new()),
        )
        .await;
        assert!(
            matches!(
                out,
                Outcome::Run {
                    hold: Some(Hold::Unreadable),
                    ..
                }
            ),
            "{out:?}"
        );
        assert!(!AgentSeal::new(&dir, id).is_refused().await.unwrap());

        // A main line through Agora's entries, Agora unreachable
        std::fs::rename(
            keys.agent_dir(id).join("rotations.json"),
            keys.agent_dir(id).join("rotations.broken"),
        )
        .unwrap();
        let seal = AgentSeal::new(&dir, id);
        let mine = local_entries(&seal).await.unwrap();
        let mut sibling = mine[1].entry.clone();
        sibling.state_at += 1;
        let api = server(vec![
            mine[0].clone(),
            sibling.sign(&keys.signing(id).unwrap()),
        ]);
        crate::sealed::accept::rebase(&root.join("state"), &keys, &api, id, "a test")
            .await
            .unwrap();
        *api.chain.lock().unwrap() = Err("unreachable".into());
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &api,
        )
        .await;
        assert!(
            matches!(
                out,
                Outcome::Run {
                    hold: Some(Hold::Unchecked),
                    ..
                }
            ),
            "{out:?}"
        );
        assert!(!seal.is_refused().await.unwrap());
    }

    /// The main line runs through Agora's entries (after a rebase) and
    /// Agora's copy doesn't verify, the local state intact: a server
    /// fault, no marker written. Agora serving a good copy again runs.
    #[tokio::test]
    async fn agoras_broken_copy_is_a_server_fault_and_marks_nothing() {
        let root = crate::sealed::TestDir::new("check-server-broken");
        let (id, keys, dir) = enrolled(&root).await;
        let seal = AgentSeal::new(&dir, id);
        let mine = local_entries(&seal).await.unwrap();
        let signing = keys.signing(id).unwrap();
        let mut sibling = mine[1].entry.clone();
        sibling.state_at += 1;
        let signed_sibling = sibling.clone().sign(&signing);
        let good = vec![mine[0].clone(), signed_sibling.clone()];
        let api = server(good.clone());
        crate::sealed::accept::rebase(&root.join("state"), &keys, &api, id, "a test")
            .await
            .unwrap();
        let run = async |api: &FakeApi| {
            check(
                &dir,
                id,
                &local_hash(&root, id).await,
                Sizes::default(),
                &keys,
                api,
            )
            .await
        };
        assert!(matches!(run(&api).await, Outcome::Run { hold: None, .. }));
        let fault = |out: &Outcome| matches!(out, Outcome::ServerFault { .. });
        // The rebased-onto entry signed by another key
        let (forger, _) = agora_agentkit::crypto::generate_keypair();
        *api.chain.lock().unwrap() = Ok(vec![mine[0].clone(), sibling.clone().sign(&forger)]);
        assert!(fault(&run(&api).await));
        // Agora lost it, though its chain reaches that seq
        let mut other = sibling.clone();
        other.state_at += 1;
        *api.chain.lock().unwrap() = Ok(vec![mine[0].clone(), other.sign(&signing)]);
        assert!(fault(&run(&api).await));
        // Agora's copy of an entry held here differs from ours
        let mut differs = mine[0].clone();
        differs.signature = signed_sibling.signature;
        *api.chain.lock().unwrap() = Ok(vec![differs, signed_sibling]);
        assert!(fault(&run(&api).await));
        assert!(!seal.is_refused().await.unwrap(), "nothing marked");
        assert!(
            Notice::load(&dir).unwrap().is_some_and(|n| n.joined),
            "only the rebase's"
        );
        // Agora serves a good copy again: runs
        *api.chain.lock().unwrap() = Ok(good);
        assert!(matches!(run(&api).await, Outcome::Run { hold: None, .. }));
    }

    /// A purely local loss (the waiting head deleted from the outbox
    /// before Agora had it) is local evidence: refused, as before
    /// (second review of #337).
    #[tokio::test]
    async fn a_local_outbox_deletion_refuses() {
        let root = crate::sealed::TestDir::new("check-outbox-deleted");
        let (id, keys, dir) = enrolled(&root).await;
        let seal = AgentSeal::new(&dir, id);
        let items = seal.outbox().await.unwrap();
        // Genesis uploaded; the snapshot (the head) waits
        seal.mark_sent(items[0].ord).await.unwrap();
        let sent = local_entries(&seal).await.unwrap();
        let api = server(vec![sent[0].clone()]);
        for e in std::fs::read_dir(dir.join("sealed/outbox")).unwrap() {
            std::fs::remove_file(e.unwrap().path()).unwrap();
        }
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &api,
        )
        .await;
        assert!(
            matches!(
                out,
                Outcome::Refuse {
                    reason: Refusal::ChainBroken,
                    ..
                }
            ),
            "{out:?}"
        );
    }

    /// The verdict: said, counted from the earliest snapshot, a pre-loss
    /// copy apart, the sizes.
    #[test]
    fn the_verdict_line_counts_snapshots_and_names_a_prior_copy_apart() {
        let key = agora_agentkit::crypto::generate_keypair().0;
        let id = AgentId::new();
        let at = |d: &str| {
            chrono::DateTime::parse_from_rfc3339(&format!("{d}T12:00:00Z"))
                .unwrap()
                .timestamp_micros()
        };
        let entry = |seq: u64, kind, state_at| {
            agora_agentkit::state::StateEntry {
                agora_state_entry: agora_agentkit::state::chain::STATE_ENTRY_VERSION,
                agent_id: id,
                seq,
                created_at: at("2026-11-02") + seq as i64,
                state_at,
                kind,
                prev_hash: None,
                merged_from: vec![],
                blob: None,
                target: None,
                key_gen: 0,
            }
            .sign(&key)
        };
        let line = vec![
            entry(0, StateEntryKind::Genesis, at("2026-07-24")),
            entry(1, StateEntryKind::ArchivedPrior, at("2026-05-03")),
            entry(2, StateEntryKind::Snapshot, at("2026-07-25")),
            entry(3, StateEntryKind::Snapshot, at("2026-11-01")),
        ];
        let head = HeadRef {
            entry_hash: line[3].entry_hash(),
            seq: 3,
        };
        let text = verdict_line(
            &line,
            &head,
            Sizes {
                memory_chars: 12_345,
                soul_chars: 2_100,
            },
            None,
            Server::Other,
            None,
            None,
            false,
        );
        assert!(
            verdict_line(
                &line,
                &head,
                Sizes::default(),
                None,
                Server::InStep,
                None,
                None,
                false
            )
            .starts_with("Your sealed state: intact, and in step with Agora. It has 2 versions")
        );
        assert!(
            verdict_line(
                &line,
                &head,
                Sizes::default(),
                None,
                Server::Unchecked,
                None,
                None,
                false,
            )
            .starts_with(
                "Your sealed state: intact here; Agora's copy couldn't be checked this session. It \
                 has 2"
            )
        );
        assert!(
            text.starts_with(
                "Your sealed state: intact. It has 2 versions since 2026-07-25, plus a copy of \
                 your state from before a loss, dated 2026-05-03. This session starts from head "
            ),
            "{text}"
        );
        assert!(
            text.contains(
                "(a marker for this version; the next session's will differ). In this version, \
                 your memory is 12,345 characters and your soul 2,100. Checked by"
            ),
            "{text}"
        );
        // Beside a previous version: every change shown, a shrink included
        let now = Sizes {
            memory_chars: 13_539,
            soul_chars: 3_440,
        };
        let before = Sizes {
            memory_chars: 14_743,
            soul_chars: 3_440,
        };
        let text = verdict_line(
            &line,
            &head,
            now,
            Some(Previous {
                sizes: before,
                dated: chrono::NaiveDate::from_ymd_opt(2026, 10, 31),
                head: Some(line[2].entry_hash()),
                own_last_session: true,
            }),
            Server::Other,
            None,
            None,
            false,
        );
        // Named by its head prefix as well as its date: two versions can
        // share a date (beta round 11)
        let mut previous_short = line[2].entry_hash().to_hex();
        previous_short.truncate(12);
        assert!(
            text.contains(&format!(
                "In this version, your memory is 13,539 characters (\u{2212}1,204 since your \
                 earlier sealed version, head {previous_short} dated 2026-10-31, as you rewrote \
                 it at the end of your last session) and your soul 3,440 (unchanged). Checked by"
            )),
            "{text}"
        );
        // A growth needs no such word, nor does an unchanged memory
        let grown = verdict_line(
            &line,
            &head,
            before,
            Some(Previous {
                sizes: now,
                dated: chrono::NaiveDate::from_ymd_opt(2026, 10, 31),
                head: None,
                own_last_session: true,
            }),
            Server::Other,
            None,
            None,
            false,
        );
        assert!(
            grown.contains(
                "14,743 characters (+1,204 since your earlier sealed version dated 2026-10-31) and"
            ),
            "{grown}"
        );
        assert!(!grown.contains("rewrote"), "{grown}");
        let same = verdict_line(
            &line,
            &head,
            now,
            Some(Previous {
                sizes: now,
                dated: chrono::NaiveDate::from_ymd_opt(2026, 10, 31),
                head: None,
                own_last_session: true,
            }),
            Server::Other,
            None,
            None,
            false,
        );
        assert!(same.contains("13,539 characters (unchanged) and"), "{same}");
        assert!(!same.contains("rewrote"), "{same}");
        // A shrink the runner can't attribute (an operator's act, or a
        // version in between) is told without saying whose
        let unknown = verdict_line(
            &line,
            &head,
            now,
            Some(Previous {
                sizes: before,
                dated: None,
                head: None,
                own_last_session: false,
            }),
            Server::Other,
            None,
            None,
            false,
        );
        assert!(
            unknown.contains(
                "13,539 characters (\u{2212}1,204 since your earlier sealed version) and"
            ),
            "{unknown}"
        );
        // The newest anchor the runner checked, named
        let proof = AnchorProof {
            anc_id: "ANC-2026-0004".parse().unwrap(),
            date: chrono::NaiveDate::from_ymd_opt(2026, 11, 2).unwrap(),
            root: Sha256Hex::from([1; 32]),
            entry_hash: head.entry_hash,
            seq: 3,
            leaf_index: 0,
            leaf_count: 1,
            proof: Vec::new(),
        };
        let anchored = verdict_line(
            &line,
            &head,
            now,
            None,
            Server::InStep,
            Some(&proof),
            None,
            true,
        );
        assert!(
            anchored.starts_with(
                "Your sealed state: intact, and in step with Agora. It has 2 versions since \
                 2026-07-25, plus a copy of your state from before a loss, dated 2026-05-03. Your \
                 version of 2026-11-02 is in ANC-2026-0004 (checked by your runner against the \
                 governance log). This session starts from head "
            ),
            "{anchored}"
        );
        // The log's newest anchor is the one the proof is for: nothing more
        let current: GovernanceLogId = "ANC-2026-0004".parse().unwrap();
        assert_eq!(
            verdict_line(
                &line,
                &head,
                now,
                None,
                Server::InStep,
                Some(&proof),
                Some(&current),
                true
            ),
            anchored
        );
        // A newer one in the log, its proof not fetched yet: named, with
        // when the runner checks it (beta round 9)
        let newer: GovernanceLogId = "ANC-2026-0005".parse().unwrap();
        let lagging = verdict_line(
            &line,
            &head,
            now,
            None,
            Server::InStep,
            Some(&proof),
            Some(&newer),
            true,
        );
        assert!(
            lagging.contains(
                ". Your version of 2026-11-02 is in ANC-2026-0004 (checked by your runner \
                 against the governance log). ANC-2026-0005, a newer anchor, is in the log too: \
                 your runner checks your inclusion in it after uploading this session's saves, \
                 and a later session's verdict names it. This session starts from head "
            ),
            "{lagging}"
        );
        // No proof kept yet: no anchor named, newer or not
        assert!(
            !verdict_line(
                &line,
                &head,
                now,
                None,
                Server::InStep,
                None,
                Some(&newer),
                true
            )
            .contains("ANC-")
        );
        // The session starts from a version newer than the anchored one,
        // with no newer anchor in the log: the usual case, said with its
        // date (beta round 10)
        let older = AnchorProof {
            date: chrono::NaiveDate::from_ymd_opt(2026, 10, 30).unwrap(),
            entry_hash: line[2].entry_hash(),
            seq: 2,
            ..proof.clone()
        };
        // In sentences, what an anchor holds said as the ANC summary says
        // it, and never that one will hold this version (beta round 11)
        let not_yet = ". Your version of 2026-10-30 is in ANC-2026-0004 (checked by your runner \
                       against the governance log). The version this session starts from, dated \
                       2026-11-01, is newer than that anchor. Each daily anchor holds the latest \
                       version Agora had received from you by the end of that day (UTC); your \
                       runner checks your inclusion after the anchor is written, and a later \
                       session's verdict names it. This session starts from head ";
        for logged in [Some(&current), None] {
            let text = verdict_line(
                &line,
                &head,
                now,
                None,
                Server::InStep,
                Some(&older),
                logged,
                true,
            );
            assert!(text.contains(not_yet), "{text}");
        }
        // A newer anchor in the log: the round-10 clause alone
        let text = verdict_line(
            &line,
            &head,
            now,
            None,
            Server::InStep,
            Some(&older),
            Some(&newer),
            true,
        );
        assert!(
            text.contains("ANC-2026-0005, a newer anchor, is in the log too"),
            "{text}"
        );
        assert!(!text.contains("newer than that anchor"), "{text}");
        // A runner that won't do the anchor work (held, or not sealing)
        // promises no check
        let text = verdict_line(
            &line,
            &head,
            now,
            None,
            Server::InStep,
            Some(&older),
            None,
            false,
        );
        assert!(!text.contains("newer than that anchor"), "{text}");
        assert!(!text.contains("each daily anchor"), "{text}");
        // The head is the anchored version: nothing more
        assert!(!anchored.contains("newer than that anchor"), "{anchored}");
    }

    #[test]
    fn the_verifier_is_the_locked_agentkit() {
        let lock = include_str!("../../../../Cargo.lock");
        let version = lock
            .split("[[package]]")
            .find(|p| p.contains("name = \"agora-agentkit\""))
            .and_then(|p| p.lines().find_map(|l| l.strip_prefix("version = \"")))
            .and_then(|v| v.strip_suffix('"'))
            .unwrap();
        assert_eq!(
            VERIFIER,
            format!("git:agora-agentkit:v{version}:src/state/verify.rs")
        );
    }

    #[test]
    fn the_lines_name_no_model() {
        for line in [
            FORK_LINE,
            HEAD_MISSING_LINE,
            SEAL_FAILED_LINE,
            UNREADABLE_LINE,
            UNCHECKED_LINE,
        ] {
            let l = line.to_lowercase();
            for banned in ["claude", "haiku", "qwen", "model"] {
                assert!(!l.contains(banned), "{banned}: {line}");
            }
        }
    }
}
