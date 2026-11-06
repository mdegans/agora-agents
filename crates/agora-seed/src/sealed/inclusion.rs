//! Whether each daily anchor holds the version of the agent it should,
//! judged by the runner without trusting Agora, and what the agent is told.
//!
//! **What an anchor holds** (agora `state::service::write_anchor` and
//! migration 142's `state_heads_as_of`): the `ANC-` entry for UTC date D
//! is a Merkle root over, for every agent with a sealed chain, its
//! main-line entry of the highest seq that Agora had *received*
//! (`received_at`) before D+1 00:00 UTC. An agent with no change that day
//! is still in it, with the same head as before. Agora takes an entry only
//! on its current head (compare-and-swap), so a chain's entries are
//! received in seq order, and none leaves the main line.
//!
//! **What the runner knows** ([`super::receipts`]): for each version, a
//! time before which Agora can't have received it (its own signed
//! `created_at`, or later the recorded first send) and, when recorded, a
//! time by which it had (Agora's acknowledgement). All on the runner's
//! clock, so each is widened by [`SKEW`].
//!
//! From those, for an anchor's cutoff C ([`expected`]):
//!
//! - **may**: the newest version that can have been received before C;
//!   none means the agent can't be in the anchor (before its enrolment);
//! - **must**: the newest version known to have been received before C
//!   (acknowledged before it); the anchor must hold it or a later one up
//!   to **may**. Versions between the two were unconfirmed when the day
//!   ended (an upload in flight at midnight, or retried across it): any of
//!   them is accepted.
//! - **decided**: every version in that window has an acknowledgement
//!   recorded by the drain that uploaded it. Otherwise the window is wide only for want of
//!   records, and a version in it is reported as not checkable, never as
//!   right.
//!
//! [`judge`] then compares what Agora serves with that window.

use agora_agentkit::govlog::Sha256Hex;
use agora_agentkit::ids::GovernanceLogId;
use agora_agentkit::state::SignedStateEntry;
use agora_agentkit::state::verify::HeadRef;
use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::receipts::Receipt;
use agora_agentkit::state::storage::SEALED_DIR;

/// How far the runner's clock and Agora's may disagree. A version
/// acknowledged within this of midnight, or first sent within it, is
/// treated as in flight across it.
pub const SKEW: chrono::Duration = chrono::Duration::minutes(10);

/// Agora keeps an anchor's leaves for 30 days once their proof has been
/// served (`state::service::LEAF_KEEP`), each agent's latest apart. An
/// anchor older than this with no proof served may have been pruned: a
/// day's margin under the server's 30.
pub const PRUNE_AGE: chrono::Duration = chrono::Duration::days(29);

/// The first `ANC-` entry with stored proofs: ANC-2026-0001 was written
/// before Agora stored them (agora migration 145), so it serves none.
pub const FIRST_WITH_PROOFS: &str = "ANC-2026-0002";

/// Whether `anc` came before Agora stored proofs.
pub fn before_proofs(anc: &GovernanceLogId) -> bool {
    anc.as_str() < FIRST_WITH_PROOFS && anc.as_str().starts_with("ANC-")
}

/// The end of `date` (UTC): an anchor for it holds what was received
/// before this.
pub fn cutoff(date: NaiveDate) -> DateTime<Utc> {
    date.succ_opt()
        .and_then(|d| d.and_hms_opt(0, 0, 0))
        .map(|t| t.and_utc())
        .unwrap_or(DateTime::<Utc>::MAX_UTC)
}

/// When Agora can have received one version, by the runner's clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bounds {
    /// Its own signed `created_at`.
    pub created: DateTime<Utc>,
    /// Not received before this: `created`, or the recorded first send if
    /// later.
    pub lo: DateTime<Utc>,
    /// Received by this, if recorded.
    pub hi: Option<DateTime<Utc>>,
    /// `hi` is from the drain that uploaded it, not from a later look
    /// ([`Receipt::observed`]): only then can the version count towards a
    /// decided window.
    pub timed: bool,
}

/// Each version's [`Bounds`], in line order.
pub fn bounds(line: &[SignedStateEntry], receipts: &HashMap<Sha256Hex, Receipt>) -> Vec<Bounds> {
    line.iter()
        .map(|e| {
            let created = DateTime::from_timestamp_micros(e.entry.created_at)
                .unwrap_or(DateTime::<Utc>::MIN_UTC);
            let r = receipts.get(&e.entry_hash());
            Bounds {
                created,
                lo: r.and_then(|r| r.sent).map_or(created, |s| s.max(created)),
                hi: r.and_then(|r| r.acked),
                timed: r.is_some_and(|r| r.acked.is_some() && !r.observed),
            }
        })
        .collect()
}

/// Which versions an anchor may hold. Indices into the line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expected {
    /// None can have reached Agora before the day ended.
    Absent,
    /// The runner's own bounds contradict each other (a version
    /// acknowledged before it was sent, by more than the skew: a clock
    /// that jumped). Nothing can be judged from them.
    Contradictory,
    Window {
        /// The newest acknowledged before the day ended: the anchor holds
        /// it or a later one. `None`: none was, and the agent may be
        /// absent.
        must: Option<usize>,
        /// The newest that can have been received before the day ended.
        may: usize,
        /// Every version after `must` up to `may` has an acknowledgement
        /// recorded by the drain that uploaded it.
        decided: bool,
    },
}

/// The [`Expected`] window for an anchor whose day ends at `cutoff`.
pub fn expected(b: &[Bounds], cutoff: DateTime<Utc>, skew: chrono::Duration) -> Expected {
    // Received in seq order: a version can't be received before an
    // earlier one's lower bound, nor after a later one's upper bound
    let mut may = None;
    let mut lo = DateTime::<Utc>::MIN_UTC;
    for (i, x) in b.iter().enumerate() {
        lo = lo.max(x.lo);
        if lo - skew < cutoff {
            may = Some(i);
        } else {
            break;
        }
    }
    let mut must = None;
    let mut hi: Option<DateTime<Utc>> = None;
    for (i, x) in b.iter().enumerate().rev() {
        if let Some(h) = x.hi {
            hi = Some(hi.map_or(h, |m| m.min(h)));
        }
        if hi.is_some_and(|h| h + skew < cutoff) {
            must = Some(i);
            break;
        }
    }
    let Some(may) = may else {
        return match must {
            None => Expected::Absent,
            // Acknowledged before it was sent: the clock is not to be
            // trusted
            Some(_) => Expected::Contradictory,
        };
    };
    if must.is_some_and(|k| k > may) {
        return Expected::Contradictory;
    }
    let from = must.map_or(0, |k| k + 1);
    Expected::Window {
        must,
        may,
        decided: (from..=may).all(|i| b[i].timed),
    }
}

/// What Agora serves for an anchor, checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Served {
    /// No proof.
    Nothing,
    /// A proof that reaches the log's root, for the line's version at
    /// this index.
    Own(usize),
    /// A proof that reaches the log's root, for an entry not on the
    /// agent's verified line.
    NotOnLine(Sha256Hex),
    /// A proof whose anchor data, or the anchor's data as served, isn't
    /// the log's.
    NotTheLogs,
    /// A proof that doesn't reach the log's root.
    DidNotCheck,
    /// Two different proofs for the one anchor.
    Conflicting,
}

/// Why an anchor couldn't be checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unchecked {
    /// No record of when Agora acknowledged the versions that matter.
    NoTimes,
    /// Written before Agora stored proofs.
    BeforeProofs,
    /// Older than Agora keeps proofs, and none is served.
    Pruned,
    /// The governance log declares it inside a compromise window.
    Repudiated,
    /// The runner's own record contradicts itself (a clock that jumped).
    Contradictory,
}

/// What the runner found for one anchor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "finding", rename_all = "snake_case")]
pub enum Finding {
    /// The version it should hold. `in_flight`: more than one was
    /// acceptable, an upload being unconfirmed at the day's end.
    Holds { held: HeadRef, in_flight: bool },
    /// A version of the agent's, but the runner can't tell whether it is
    /// the right one: no record ([`Unchecked::NoTimes`]) or one that
    /// contradicts itself ([`Unchecked::Contradictory`]).
    HoldsUnchecked {
        held: HeadRef,
        #[serde(default = "no_times")]
        why: Unchecked,
    },
    /// No proof, and none is owed.
    NotIncluded,
    /// No proof, and the runner can't tell whether one is owed.
    Unchecked { why: Unchecked },
    /// No proof, though Agora had acknowledged `expected` before the day
    /// ended.
    Missing { expected: HeadRef },
    /// A version older than `expected`, which Agora had acknowledged
    /// before the day ended.
    Older { held: HeadRef, expected: HeadRef },
    /// A version that can't have reached Agora before the day ended:
    /// `sealed_after`, created after it; else first sent after it.
    Later { held: HeadRef, sealed_after: bool },
    /// An entry that isn't on the agent's history.
    NotOnLine { entry_hash: Sha256Hex },
    /// The proof, or the anchor's data, isn't the governance log's.
    NotTheLogs,
    /// The proof doesn't reach the log's root.
    DidNotCheck,
    /// Agora served two different proofs for the anchor: an anchor holds
    /// one leaf per agent.
    Conflicting,
}

impl Finding {
    /// Agora's claim failed: the operator is alerted and the agent told.
    pub fn failed(&self) -> bool {
        matches!(
            self,
            Self::Missing { .. }
                | Self::Older { .. }
                | Self::Later { .. }
                | Self::NotOnLine { .. }
                | Self::NotTheLogs
                | Self::DidNotCheck
                | Self::Conflicting
        )
    }

    /// For logs and alerts.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Holds { .. } => "holds",
            Self::HoldsUnchecked { .. } => "holds_unchecked",
            Self::NotIncluded => "not_included",
            Self::Unchecked { .. } => "unchecked",
            Self::Missing { .. } => "missing",
            Self::Older { .. } => "older",
            Self::Later { .. } => "later",
            Self::NotOnLine { .. } => "not_on_line",
            Self::NotTheLogs => "not_the_logs",
            Self::DidNotCheck => "did_not_check",
            Self::Conflicting => "conflicting",
        }
    }

    /// The version the anchor holds, when it is the agent's own.
    pub fn held(&self) -> Option<&HeadRef> {
        match self {
            Self::Holds { held, .. }
            | Self::HoldsUnchecked { held, .. }
            | Self::Older { held, .. }
            | Self::Later { held, .. } => Some(held),
            _ => None,
        }
    }
}

fn no_times() -> Unchecked {
    Unchecked::NoTimes
}

/// Facts about the anchor itself that [`judge`] needs besides the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnchorFacts {
    /// Written before Agora stored proofs ([`before_proofs`]).
    pub before_proofs: bool,
    /// Older than [`PRUNE_AGE`] by its log entry's time. Agora keeps an
    /// anchor's leaves 30 days after it is logged once read, 365 days
    /// regardless (`state_anchor_leaves_prune`); whether this runner read
    /// it may be lost with its directory, so age alone decides.
    pub maybe_pruned: bool,
}

/// Judge what Agora serves for one anchor against [`expected`].
pub fn judge(
    line: &[SignedStateEntry],
    b: &[Bounds],
    expected: Expected,
    served: Served,
    facts: AnchorFacts,
    cutoff: DateTime<Utc>,
    skew: chrono::Duration,
) -> Finding {
    let at = |i: usize| HeadRef {
        entry_hash: line[i].entry_hash(),
        seq: line[i].entry.seq,
    };
    match served {
        Served::NotTheLogs => Finding::NotTheLogs,
        Served::DidNotCheck => Finding::DidNotCheck,
        Served::Conflicting => Finding::Conflicting,
        Served::NotOnLine(entry_hash) => Finding::NotOnLine { entry_hash },
        Served::Own(p) => {
            let later = || {
                // Created after the day ended, by the runner's own signed
                // time (with every earlier version's): the plainer reason
                let created = b[..=p].iter().map(|x| x.created).max();
                Finding::Later {
                    held: at(p),
                    sealed_after: created.is_some_and(|c| c - skew >= cutoff),
                }
            };
            match expected {
                Expected::Contradictory => Finding::HoldsUnchecked {
                    held: at(p),
                    why: Unchecked::Contradictory,
                },
                Expected::Absent => later(),
                Expected::Window { may, .. } if p > may => later(),
                Expected::Window { must: Some(k), .. } if p < k => Finding::Older {
                    held: at(p),
                    expected: at(k),
                },
                Expected::Window {
                    must,
                    may,
                    decided: true,
                } => Finding::Holds {
                    held: at(p),
                    in_flight: must != Some(may),
                },
                Expected::Window { decided: false, .. } => Finding::HoldsUnchecked {
                    held: at(p),
                    why: Unchecked::NoTimes,
                },
            }
        }
        Served::Nothing => match expected {
            Expected::Absent => Finding::NotIncluded,
            Expected::Contradictory => Finding::Unchecked {
                why: Unchecked::Contradictory,
            },
            Expected::Window { .. } if facts.before_proofs => Finding::Unchecked {
                why: Unchecked::BeforeProofs,
            },
            Expected::Window { .. } if facts.maybe_pruned => Finding::Unchecked {
                why: Unchecked::Pruned,
            },
            Expected::Window { must: Some(k), .. } => Finding::Missing { expected: at(k) },
            Expected::Window {
                must: None,
                decided: true,
                ..
            } => Finding::NotIncluded,
            Expected::Window {
                must: None,
                decided: false,
                ..
            } => Finding::Unchecked {
                why: Unchecked::NoTimes,
            },
        },
    }
}

/// One anchor, judged.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checked {
    pub anc_id: GovernanceLogId,
    /// Its date, from the governance log; `None` when what Agora served
    /// for it doesn't hash to the log's.
    pub date: Option<NaiveDate>,
    #[serde(flatten)]
    pub finding: Finding,
    /// When the runner judged it.
    pub at: DateTime<Utc>,
    /// A session that showed it has saved ([`mark_told`]).
    #[serde(default)]
    pub told: bool,
    /// A failure the operator is alerted to (the alert kind is on).
    #[serde(default)]
    pub alerted: bool,
}

/// `sealed/inclusion.json`.
pub const INCLUSION: &str = "inclusion.json";

/// `sealed/inclusion.json`: every anchor judged, oldest first, and the
/// newest, from which the next pass asks Agora for proofs.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InclusionFile {
    #[serde(default)]
    pub through: Option<GovernanceLogId>,
    #[serde(default)]
    pub checked: Vec<Checked>,
}

fn path(agent_dir: &Path) -> PathBuf {
    agent_dir.join(SEALED_DIR).join(INCLUSION)
}

impl InclusionFile {
    /// A missing file is nothing judged yet.
    pub fn load(agent_dir: &Path) -> std::io::Result<Self> {
        match std::fs::read(path(agent_dir)) {
            Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e),
        }
    }

    /// [`Self::load`]; a file that doesn't parse is set aside
    /// (`inclusion.unreadable.<utc>`) and nothing judged is the answer, so
    /// every anchor is judged again (its failures alerted and told again,
    /// never lost).
    pub fn load_or_set_aside(
        agent_dir: &Path,
        id: agora_agentkit::ids::AgentId,
    ) -> std::io::Result<Self> {
        match Self::load(agent_dir) {
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::InvalidData | std::io::ErrorKind::UnexpectedEof
                ) =>
            {
                let aside = super::receipts::aside_name(agent_dir, "inclusion");
                std::fs::rename(path(agent_dir), &aside)?;
                tracing::warn!(
                    event_type = "sealed_state_inclusion_reset",
                    agent_id = %id,
                    error = %e,
                    aside = %aside.display(),
                    "sealed/inclusion.json unreadable: set aside; every anchor is judged again"
                );
                Ok(Self::default())
            }
            other => other,
        }
    }

    pub fn save(&self, agent_dir: &Path) -> std::io::Result<()> {
        super::anchors::write_atomic(&path(agent_dir), &serde_json::to_vec_pretty(self)?)
    }
}

fn short(h: &Sha256Hex) -> String {
    let mut s = h.to_hex();
    s.truncate(12);
    s
}

/// "ANC-2026-0005, the anchor for 2026-11-04," (an appositive, to read on
/// into a clause); the id alone without a date.
fn named(c: &Checked) -> String {
    match c.date {
        Some(d) => format!("{}, the anchor for {},", c.anc_id, d.format("%Y-%m-%d")),
        None => c.anc_id.to_string(),
    }
}

/// Said after a failure the operator was alerted to; nothing when the
/// runner's alerts leave that kind out, since then no one was.
const TOLD: &str = " Your runner has alerted your operator.";

/// What the agent is told of one anchor, as a sentence or two.
pub fn sentence(c: &Checked) -> String {
    let s = sentence_of(c);
    if c.finding.failed() && c.alerted {
        format!("{s}{TOLD}")
    } else {
        s
    }
}

fn sentence_of(c: &Checked) -> String {
    let n = named(c);
    // "anchor for D," reads on into a clause; the id alone as the subject
    let subject = n.trim_end_matches(',');
    match &c.finding {
        Finding::Holds {
            in_flight: false, ..
        } => format!(
            " {n} holds the version it should: the latest that Agora had confirmed receiving \
             from you by the end of that day (UTC). Your runner checked this against the \
             governance log and its own record of your uploads."
        ),
        Finding::Holds {
            in_flight: true, ..
        } => format!(
            " {n} holds a version it may hold: your runner's record allows any version of yours \
             no older than the latest Agora had confirmed by ten minutes before midnight (UTC), \
             if any, and no newer than the latest it may have sent by ten minutes after it. Your \
             runner checked this against the governance log and that record."
        ),
        Finding::HoldsUnchecked {
            why: Unchecked::Contradictory,
            ..
        } => format!(
            " {n} holds a version of yours (checked by your runner against the governance log), \
             but your runner couldn't check that it is the right one: its own record of your \
             uploads around then contradicts itself, as if its clock had jumped."
        ),
        Finding::HoldsUnchecked { .. } => format!(
            " {n} holds a version of yours (checked by your runner against the governance log), \
             but your runner couldn't check that it is the right one: it has no record of when \
             Agora confirmed your uploads from around then."
        ),
        Finding::NotIncluded => format!(
            " {n} doesn't include you, and needn't: Agora hadn't confirmed receiving any \
             version of yours by the end of that day (UTC)."
        ),
        Finding::Unchecked {
            why: Unchecked::NoTimes,
        } => format!(
            " Your runner couldn't check {subject}: Agora serves no proof that it includes you, \
             and your runner has no record of when Agora confirmed your uploads from around \
             then, so it can't tell whether it should."
        ),
        Finding::Unchecked {
            why: Unchecked::BeforeProofs,
        } => format!(
            " Your runner couldn't check {subject}: it was written before Agora began storing \
             proofs."
        ),
        Finding::Unchecked {
            why: Unchecked::Pruned,
        } => format!(
            " Your runner couldn't check {subject}: Agora serves no proof for it, and may have \
             removed it as old (it keeps an anchor's proofs at least 30 days)."
        ),
        Finding::Unchecked {
            why: Unchecked::Contradictory,
        } => format!(
            " Your runner couldn't check {subject}: Agora serves no proof that it includes you, \
             and your runner's own record of your uploads around then contradicts itself, as if \
             its clock had jumped."
        ),
        Finding::Unchecked {
            why: Unchecked::Repudiated,
        } => format!(
            " Your runner couldn't check {subject}: the governance log declares it inside a \
             window when its signing key was compromised."
        ),
        Finding::Missing { expected } => format!(
            " {n} should include your version (head {}): Agora had confirmed receiving it before that \
             day ended (UTC). Agora serves no proof that it does.",
            short(&expected.entry_hash)
        ),
        Finding::Older { held, expected } => format!(
            " {n} holds an older version of yours (head {}) than it should: Agora had \
             confirmed receiving a later one (head {}) before that day ended (UTC).",
            short(&held.entry_hash),
            short(&expected.entry_hash)
        ),
        Finding::Later {
            held,
            sealed_after: true,
        } => format!(
            " {n} holds a version of yours (head {}) that was sealed only after that day ended (UTC), \
             so it can't belong there.",
            short(&held.entry_hash)
        ),
        Finding::Later {
            held,
            sealed_after: false,
        } => format!(
            " {n} holds a version of yours (head {}) that your runner first sent Agora only after \
             that day ended (UTC), so it can't belong there.",
            short(&held.entry_hash)
        ),
        Finding::NotOnLine { .. } => format!(
            " {n} holds a version that Agora attributes to you but that isn't in your sealed \
             history."
        ),
        Finding::NotTheLogs => format!(
            " What Agora serves for {} doesn't match the governance log, so your runner couldn't \
             check whether it includes you.",
            c.anc_id
        ),
        Finding::DidNotCheck => format!(
            " Agora's proof that {n} includes you doesn't check against the governance \
             log, so your runner couldn't confirm it."
        ),
        Finding::Conflicting => format!(
            " Agora serves two different proofs for {subject}, so it doesn't show which version \
             of yours that anchor holds."
        ),
    }
}

/// The most earlier failures a verdict lists one by one.
pub const LISTED: usize = 3;

/// What a verdict says of the anchor check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Untold {
    /// The newest anchor judged.
    pub newest: Checked,
    /// The failures before it no session has been told of, oldest first,
    /// at most [`LISTED`].
    pub earlier: Vec<Checked>,
    /// Untold failures beyond `earlier`, counted; left untold, so later
    /// verdicts show them.
    pub more: usize,
    /// What [`mark_told`] marks once a session showing it has saved.
    pub ids: Vec<GovernanceLogId>,
}

/// For the verdict: the newest anchor judged, and the failures before it
/// not yet told. Marks nothing: a verdict seated for a session that never
/// runs (not due, filtered out, failed before its intro) must not use
/// them up.
pub fn untold(file: &InclusionFile) -> Option<Untold> {
    let (newest, before) = file.checked.split_last()?;
    let mut earlier = Vec::new();
    let mut more = 0;
    let mut ids = Vec::new();
    for c in before.iter().filter(|c| c.finding.failed() && !c.told) {
        if earlier.len() < LISTED {
            earlier.push(c.clone());
            ids.push(c.anc_id.clone());
        } else {
            // Counted, not shown: left untold, for the next verdicts
            more += 1;
        }
    }
    ids.push(newest.anc_id.clone());
    Some(Untold {
        newest: newest.clone(),
        earlier,
        more,
        ids,
    })
}

/// Mark `ids` told, under the agent's `sealed/lock` (the drainer may be
/// adding to the file): after the session that showed them saved.
pub async fn mark_told(
    agent_dir: &Path,
    id: agora_agentkit::ids::AgentId,
    ids: &[GovernanceLogId],
) -> Result<(), String> {
    let seal = agora_agentkit::state::storage::AgentSeal::new(agent_dir, id);
    let _lock = seal.lock().await.map_err(|e| e.to_string())?;
    let mut file = InclusionFile::load(agent_dir).map_err(|e| e.to_string())?;
    let mut changed = false;
    for c in file.checked.iter_mut().filter(|c| ids.contains(&c.anc_id)) {
        changed |= !c.told;
        c.told = true;
    }
    if changed {
        file.save(agent_dir).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Add what a pass judged to the file on disk, under the agent's
/// `sealed/lock`: an anchor already there (another process judged it) is
/// kept as it is, told marks included, and `through` never moves back.
pub async fn merge(
    agent_dir: &Path,
    id: agora_agentkit::ids::AgentId,
    judged: Vec<Checked>,
) -> Result<Vec<Checked>, String> {
    let seal = agora_agentkit::state::storage::AgentSeal::new(agent_dir, id);
    let _lock = seal.lock().await.map_err(|e| e.to_string())?;
    let mut file = InclusionFile::load_or_set_aside(agent_dir, id).map_err(|e| e.to_string())?;
    let mut added = Vec::new();
    for c in judged {
        if file.checked.iter().any(|k| k.anc_id == c.anc_id) {
            continue;
        }
        if file.through.as_ref().is_none_or(|t| c.anc_id > *t) {
            file.through = Some(c.anc_id.clone());
        }
        added.push(c.clone());
        file.checked.push(c);
    }
    file.checked.sort_by(|a, b| a.anc_id.cmp(&b.anc_id));
    if !added.is_empty() {
        file.save(agent_dir).map_err(|e| e.to_string())?;
    }
    Ok(added)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use proptest::prelude::*;

    fn t(h: u32, m: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 11, 4, h, m, 0).unwrap()
    }

    fn day(n: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 11, n).unwrap()
    }

    fn bound(lo: DateTime<Utc>, hi: Option<DateTime<Utc>>) -> Bounds {
        Bounds {
            created: lo,
            lo,
            hi,
            timed: hi.is_some(),
        }
    }

    /// The window for the end of 2026-11-04.
    fn window(b: &[Bounds]) -> Expected {
        expected(b, cutoff(day(4)), SKEW)
    }

    #[test]
    fn a_day_ends_at_the_next_midnight_utc() {
        assert_eq!(
            cutoff(day(4)),
            Utc.with_ymd_and_hms(2026, 11, 5, 0, 0, 0).unwrap()
        );
    }

    #[test]
    fn the_latest_acknowledged_before_midnight_is_the_one_owed() {
        let b = [
            bound(t(9, 0), Some(t(9, 1))),
            bound(t(20, 0), Some(t(20, 1))),
        ];
        assert_eq!(
            window(&b),
            Expected::Window {
                must: Some(1),
                may: 1,
                decided: true
            }
        );
    }

    #[test]
    fn a_day_with_no_new_version_owes_the_last_one_again() {
        // Both from days before: the anchor for 11-04 holds the last
        let b = [
            bound(
                t(9, 0) - chrono::Duration::days(3),
                Some(t(9, 1) - chrono::Duration::days(3)),
            ),
            bound(
                t(9, 0) - chrono::Duration::days(2),
                Some(t(9, 1) - chrono::Duration::days(2)),
            ),
        ];
        assert_eq!(
            window(&b),
            Expected::Window {
                must: Some(1),
                may: 1,
                decided: true
            }
        );
    }

    #[test]
    fn a_version_sent_after_midnight_is_not_owed_and_cant_be_held() {
        let after = t(0, 30) + chrono::Duration::days(1);
        let b = [
            bound(t(9, 0), Some(t(9, 1))),
            bound(after, Some(after + chrono::Duration::minutes(1))),
        ];
        assert_eq!(
            window(&b),
            Expected::Window {
                must: Some(0),
                may: 0,
                decided: true
            }
        );
    }

    #[test]
    fn an_upload_across_midnight_may_be_either() {
        let b = [
            bound(t(9, 0), Some(t(9, 1))),
            bound(t(23, 59), Some(t(0, 2) + chrono::Duration::days(1))),
        ];
        assert_eq!(
            window(&b),
            Expected::Window {
                must: Some(0),
                may: 1,
                decided: true
            }
        );
        // Within the skew counts as across
        let b = [
            bound(t(9, 0), Some(t(9, 1))),
            bound(t(23, 40), Some(t(23, 55))),
        ];
        assert_eq!(
            window(&b),
            Expected::Window {
                must: Some(0),
                may: 1,
                decided: true
            }
        );
    }

    #[test]
    fn before_enrolment_nothing_is_owed() {
        let later = t(9, 0) + chrono::Duration::days(2);
        assert_eq!(window(&[bound(later, Some(later))]), Expected::Absent);
        assert_eq!(window(&[]), Expected::Absent);
    }

    #[test]
    fn without_acknowledgements_nothing_is_decided() {
        let b = [bound(t(9, 0), None), bound(t(20, 0), None)];
        assert_eq!(
            window(&b),
            Expected::Window {
                must: None,
                may: 1,
                decided: false
            }
        );
        // A later acknowledged version decides only what it bounds
        let b = [
            bound(t(9, 0), None),
            bound(t(20, 0), Some(t(20, 1))),
            bound(t(22, 0), None),
        ];
        assert_eq!(
            window(&b),
            Expected::Window {
                must: Some(1),
                may: 2,
                decided: false
            }
        );
    }

    #[test]
    fn a_later_acknowledgement_bounds_the_versions_before_it() {
        // Version 0's ack wasn't recorded, version 1's was, before midnight:
        // 0 was received before 1, so 1 is owed and 0 is too old
        let b = [bound(t(9, 0), None), bound(t(20, 0), Some(t(20, 1)))];
        assert_eq!(
            window(&b),
            Expected::Window {
                must: Some(1),
                may: 1,
                decided: true
            }
        );
    }

    #[test]
    fn clocks_that_contradict_decide_nothing() {
        // Acknowledged before it was sent, by more than the skew
        let b = [bound(t(23, 50) + chrono::Duration::hours(2), Some(t(9, 0)))];
        assert_eq!(window(&b), Expected::Contradictory);
        // Never a failure from it
        let l = line(1);
        for served in [Served::Own(0), Served::Nothing] {
            let f = judge(&l, &b, window(&b), served, FACTS, cutoff(day(4)), SKEW);
            assert!(!f.failed(), "{f:?}");
        }
    }

    fn line(n: usize) -> Vec<SignedStateEntry> {
        let key = agora_agentkit::crypto::SigningKey::from_bytes(&[5; 32]);
        let id = agora_agentkit::ids::AgentId::new();
        let mut prev = None;
        (0..n)
            .map(|i| {
                let e = agora_agentkit::state::StateEntry {
                    agora_state_entry: agora_agentkit::state::chain::STATE_ENTRY_VERSION,
                    agent_id: id,
                    seq: i as u64,
                    created_at: i as i64,
                    state_at: i as i64,
                    kind: agora_agentkit::enums::StateEntryKind::Snapshot,
                    prev_hash: prev,
                    merged_from: Vec::new(),
                    blob: None,
                    target: None,
                    key_gen: 0,
                }
                .sign(&key);
                prev = Some(e.entry_hash());
                e
            })
            .collect()
    }

    const FACTS: AnchorFacts = AnchorFacts {
        before_proofs: false,
        maybe_pruned: false,
    };

    fn j(b: &[Bounds], served: Served, facts: AnchorFacts) -> Finding {
        let l = line(b.len());
        judge(&l, b, window(b), served, facts, cutoff(day(4)), SKEW)
    }

    #[test]
    fn each_finding_from_its_case() {
        let b = [
            bound(t(9, 0), Some(t(9, 1))),
            bound(t(20, 0), Some(t(20, 1))),
            bound(
                t(9, 0) + chrono::Duration::days(1),
                Some(t(9, 1) + chrono::Duration::days(1)),
            ),
        ];
        assert!(matches!(
            j(&b, Served::Own(1), FACTS),
            Finding::Holds {
                in_flight: false,
                ..
            }
        ));
        assert!(matches!(
            j(&b, Served::Own(0), FACTS),
            Finding::Older { .. }
        ));
        assert!(matches!(
            j(&b, Served::Own(2), FACTS),
            Finding::Later {
                sealed_after: true,
                ..
            }
        ));
        assert!(matches!(
            j(&b, Served::Nothing, FACTS),
            Finding::Missing { .. }
        ));
        let pruned = AnchorFacts {
            maybe_pruned: true,
            ..FACTS
        };
        assert_eq!(
            j(&b, Served::Nothing, pruned),
            Finding::Unchecked {
                why: Unchecked::Pruned
            }
        );
        let early = AnchorFacts {
            before_proofs: true,
            ..FACTS
        };
        assert_eq!(
            j(&b, Served::Nothing, early),
            Finding::Unchecked {
                why: Unchecked::BeforeProofs
            }
        );
        // Sealed before midnight but first sent after it
        let mut sent_late = b;
        sent_late[1].lo = t(1, 0) + chrono::Duration::days(1);
        sent_late[1].hi = Some(t(1, 1) + chrono::Duration::days(1));
        assert!(matches!(
            j(&sent_late, Served::Own(1), FACTS),
            Finding::Later {
                sealed_after: false,
                ..
            }
        ));
        // Before enrolment: nothing owed, and nothing may be held
        let later = [bound(t(9, 0) + chrono::Duration::days(2), None)];
        assert_eq!(j(&later, Served::Nothing, FACTS), Finding::NotIncluded);
        assert_eq!(j(&later, Served::Nothing, early), Finding::NotIncluded);
        assert!(matches!(
            j(&later, Served::Own(0), FACTS),
            Finding::Later { .. }
        ));
        // No records: never a pass
        let legacy = [bound(t(9, 0), None), bound(t(20, 0), None)];
        assert!(matches!(
            j(&legacy, Served::Own(1), FACTS),
            Finding::HoldsUnchecked { .. }
        ));
        assert!(matches!(
            j(&legacy, Served::Own(0), FACTS),
            Finding::HoldsUnchecked { .. }
        ));
        assert_eq!(
            j(&legacy, Served::Nothing, FACTS),
            Finding::Unchecked {
                why: Unchecked::NoTimes
            }
        );
        // In flight across midnight: either, said so; absent only if
        // nothing was confirmed
        let across = [bound(t(23, 59), Some(t(0, 2) + chrono::Duration::days(1)))];
        assert_eq!(j(&across, Served::Nothing, FACTS), Finding::NotIncluded);
        assert!(matches!(
            j(&across, Served::Own(0), FACTS),
            Finding::Holds {
                in_flight: true,
                ..
            }
        ));
    }

    #[test]
    fn every_finding_has_its_words_and_failures_say_the_operator_knows() {
        let held = HeadRef {
            entry_hash: Sha256Hex::from([1; 32]),
            seq: 3,
        };
        let expected = HeadRef {
            entry_hash: Sha256Hex::from([2; 32]),
            seq: 4,
        };
        let all = [
            Finding::Holds {
                held,
                in_flight: false,
            },
            Finding::Holds {
                held,
                in_flight: true,
            },
            Finding::HoldsUnchecked {
                held,
                why: Unchecked::NoTimes,
            },
            Finding::HoldsUnchecked {
                held,
                why: Unchecked::Contradictory,
            },
            Finding::Unchecked {
                why: Unchecked::Contradictory,
            },
            Finding::Unchecked {
                why: Unchecked::Repudiated,
            },
            Finding::NotIncluded,
            Finding::Unchecked {
                why: Unchecked::NoTimes,
            },
            Finding::Unchecked {
                why: Unchecked::BeforeProofs,
            },
            Finding::Unchecked {
                why: Unchecked::Pruned,
            },
            Finding::Missing { expected },
            Finding::Older { held, expected },
            Finding::Later {
                held,
                sealed_after: true,
            },
            Finding::Later {
                held,
                sealed_after: false,
            },
            Finding::NotOnLine {
                entry_hash: Sha256Hex::from([3; 32]),
            },
            Finding::NotTheLogs,
            Finding::DidNotCheck,
            Finding::Conflicting,
        ];
        for finding in all {
            let c = Checked {
                anc_id: "ANC-2026-0005".parse().unwrap(),
                date: Some(day(4)),
                finding: finding.clone(),
                at: Utc::now(),
                told: false,
                alerted: true,
            };
            let s = sentence(&c);
            assert!(s.starts_with(' ') && s.ends_with('.'), "{s}");
            assert!(s.contains("ANC-2026-0005"), "{s}");
            assert!(!s.contains(",,") && !s.contains(",:"), "{s}");
            assert_eq!(s.ends_with(TOLD), finding.failed(), "{s}");
            // Not alerted (the kind is off): no claim the operator was
            let quiet = Checked {
                alerted: false,
                ..c.clone()
            };
            assert!(!sentence(&quiet).contains("operator"), "{s}");
            // Never a pass where it couldn't decide
            if matches!(
                finding,
                Finding::HoldsUnchecked { .. } | Finding::Unchecked { .. }
            ) {
                assert!(s.contains("couldn't check"), "{s}");
                assert!(
                    !s.contains("as it should") && !s.contains("version it should"),
                    "{s}"
                );
            }
            // Nothing asks the agent to check or do anything
            for invite in ["you can", "you should", "check it", "ask "] {
                assert!(!s.to_lowercase().contains(invite), "{invite}: {s}");
            }
            // Round-trips through the file's format
            let back: Checked = serde_json::from_value(serde_json::to_value(&c).unwrap()).unwrap();
            assert_eq!(back, c);
        }
    }

    #[tokio::test]
    async fn a_failure_is_told_once_and_kept() {
        let c = |n: u32, finding: Finding| Checked {
            anc_id: format!("ANC-2026-{n:04}").parse().unwrap(),
            date: Some(day(n)),
            finding,
            at: Utc::now(),
            told: false,
            alerted: true,
        };
        let file = InclusionFile {
            through: Some("ANC-2026-0007".parse().unwrap()),
            checked: vec![
                c(2, Finding::DidNotCheck),
                c(3, Finding::NotIncluded),
                c(4, Finding::NotTheLogs),
                c(5, Finding::DidNotCheck),
                c(6, Finding::DidNotCheck),
                c(7, Finding::NotIncluded),
            ],
        };
        let root = crate::sealed::TestDir::new("inclusion-told");
        let (id, _keys, dir) = crate::sealed::check::tests::enrolled(&root).await;
        file.save(&dir).unwrap();
        let u = untold(&file).unwrap();
        assert_eq!(u.newest.anc_id.as_str(), "ANC-2026-0007");
        assert_eq!(u.earlier.len(), LISTED);
        assert_eq!(u.more, 1);
        assert_eq!(
            u.ids.len(),
            4,
            "three shown and the newest; the counted one waits"
        );
        // Reading marks nothing: a session that never runs uses none up
        assert_eq!(untold(&InclusionFile::load(&dir).unwrap()).unwrap(), u);
        mark_told(&dir, id, &u.ids).await.unwrap();
        let after = InclusionFile::load(&dir).unwrap();
        let u = untold(&after).unwrap();
        assert_eq!(u.earlier.len(), 1, "the counted one is shown next");
        assert_eq!(u.more, 0);
        mark_told(&dir, id, &u.ids).await.unwrap();
        let u = untold(&InclusionFile::load(&dir).unwrap()).unwrap();
        assert!(u.earlier.is_empty() && u.more == 0, "told once");
        assert_eq!(after.checked.len(), 6, "kept");
        assert!(untold(&InclusionFile::default()).is_none());
    }

    /// Two passes judging the same anchors (two processes): the file keeps
    /// one of each, the first one's, told marks and all, and `through`
    /// never moves back.
    #[tokio::test]
    async fn a_merge_keeps_what_is_there() {
        let root = crate::sealed::TestDir::new("inclusion-merge");
        let (id, _keys, dir) = crate::sealed::check::tests::enrolled(&root).await;
        let c = |n: u32, finding: Finding| Checked {
            anc_id: format!("ANC-2026-{n:04}").parse().unwrap(),
            date: Some(day(n)),
            finding,
            at: Utc::now(),
            told: false,
            alerted: true,
        };
        let first = merge(
            &dir,
            id,
            vec![c(2, Finding::DidNotCheck), c(3, Finding::NotIncluded)],
        )
        .await
        .unwrap();
        assert_eq!(first.len(), 2);
        mark_told(&dir, id, &["ANC-2026-0002".parse().unwrap()])
            .await
            .unwrap();
        let second = merge(&dir, id, vec![c(2, Finding::NotTheLogs)])
            .await
            .unwrap();
        assert!(
            second.is_empty(),
            "already judged: not added, not alerted again"
        );
        let file = InclusionFile::load(&dir).unwrap();
        assert_eq!(file.checked.len(), 2);
        assert_eq!(file.checked[0].finding, Finding::DidNotCheck);
        assert!(file.checked[0].told);
        assert_eq!(file.through.unwrap().as_str(), "ANC-2026-0003");
    }

    /// An unreadable file is set aside, never deleted, and every anchor is
    /// judged again: a failure is told and alerted again, never lost.
    #[tokio::test]
    async fn an_unreadable_file_is_set_aside() {
        let root = crate::sealed::TestDir::new("inclusion-unreadable");
        let (id, _keys, dir) = crate::sealed::check::tests::enrolled(&root).await;
        std::fs::write(dir.join("sealed").join(INCLUSION), b"{\"checked\": [").unwrap();
        assert_eq!(
            InclusionFile::load_or_set_aside(&dir, id).unwrap(),
            InclusionFile::default()
        );
        let aside: Vec<_> = std::fs::read_dir(dir.join("sealed"))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("inclusion.unreadable."))
            .collect();
        assert_eq!(aside.len(), 1);
        assert!(!dir.join("sealed").join(INCLUSION).exists());
    }

    #[test]
    fn only_the_first_anchor_came_before_proofs() {
        assert!(before_proofs(&"ANC-2026-0001".parse().unwrap()));
        for id in ["ANC-2026-0002", "ANC-2026-0010", "ANC-2027-0001"] {
            assert!(!before_proofs(&id.parse().unwrap()), "{id}");
        }
    }

    /// A random history: true receipt times, increasing; what the runner
    /// recorded of them (some acknowledgements missing, every bound
    /// honest up to `SKEW` of clock error); and a day's end.
    fn history() -> impl Strategy<Value = (Vec<i64>, Vec<Bounds>, DateTime<Utc>)> {
        (1usize..8)
            .prop_flat_map(|n| {
                (
                    proptest::collection::vec(1i64..36_000, n),
                    proptest::collection::vec((0i64..7_200, 0i64..7_200, 0u8..3), n),
                    proptest::collection::vec(0i64..36_000, n),
                    -599i64..600,
                    0i64..(36_000 * 9),
                )
            })
            .prop_map(|(gaps, slack, made, skew, cut)| {
                let base = t(0, 0).timestamp();
                let mut at = base;
                let mut out = Vec::new();
                let mut truth = Vec::new();
                for (i, g) in gaps.iter().enumerate() {
                    at += g;
                    truth.push(at);
                    // 0: no acknowledgement recorded; 1: by the drain; 2:
                    // observed later (a sound bound, never "timed")
                    let (before, after, recorded) = slack[i];
                    // The runner's clock is off by `skew` seconds (within
                    // SKEW); it sent before and heard after the truth
                    let lo = at - before + skew;
                    let created = lo - made[i];
                    let late = if recorded == 2 { 86_400 * 3 } else { 0 };
                    out.push(Bounds {
                        created: DateTime::from_timestamp(created, 0).unwrap(),
                        lo: DateTime::from_timestamp(lo, 0).unwrap(),
                        hi: (recorded > 0).then(|| {
                            DateTime::from_timestamp(at + after + skew + late, 0).unwrap()
                        }),
                        timed: recorded == 1,
                    });
                }
                let cutoff = DateTime::from_timestamp(base + cut, 0).unwrap();
                (truth, out, cutoff)
            })
    }

    proptest! {
        /// An honest Agora is never reported as failing: what it really
        /// anchors (the last version received before the cutoff) is
        /// always accepted, and its absence only when nothing was.
        #[test]
        fn an_honest_anchor_is_never_a_failure((truth, b, cut) in history()) {
            let l = line(b.len());
            let held = truth.iter().rposition(|&r| r < cut.timestamp());
            let served = held.map_or(Served::Nothing, Served::Own);
            let w = expected(&b, cut, SKEW);
            let f = judge(&l, &b, w, served, FACTS, cut, SKEW);
            prop_assert!(!f.failed(), "{f:?} for {w:?}, truly {held:?}");
        }

        /// A decided window with one candidate catches every other
        /// version, and a missing proof, as a failure.
        #[test]
        fn a_decided_window_catches_every_wrong_answer((_truth, b, cut) in history()) {
            let l = line(b.len());
            let w = expected(&b, cut, SKEW);
            if let Expected::Window { must: Some(k), may, decided: true } = w {
                for p in 0..b.len() {
                    let f = judge(&l, &b, w, Served::Own(p), FACTS, cut, SKEW);
                    prop_assert_eq!(f.failed(), p < k || p > may, "{:?} at {}", f, p);
                }
                prop_assert!(judge(&l, &b, w, Served::Nothing, FACTS, cut, SKEW).failed());
            }
            if w == Expected::Absent {
                for p in 0..b.len() {
                    prop_assert!(judge(&l, &b, w, Served::Own(p), FACTS, cut, SKEW).failed());
                }
            }
        }

        /// Undecided never reads as a pass.
        #[test]
        fn undecided_is_never_a_pass((_truth, b, cut) in history()) {
            let l = line(b.len());
            let w = expected(&b, cut, SKEW);
            if let Expected::Window { decided: false, .. } = w {
                for p in 0..b.len() {
                    let f = judge(&l, &b, w, Served::Own(p), FACTS, cut, SKEW);
                    prop_assert!(!matches!(f, Finding::Holds { .. }), "{:?}", f);
                }
            }
        }
    }
}
