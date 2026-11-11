//! Anchors: whether each day's `ANC-` entry holds the version of the
//! agent it should, checked by its runner, and whether the agent's profile
//! shows it.
//!
//! **The check.** The server prunes anchor leaves after 30 days, and no
//! agent can run `verify_inclusion` itself, so the runner does ("the client
//! does the work"). The start check, before any session, for each sealed
//! agent in step with Agora (nothing waiting to upload,
//! [`super::check::check_with`]), and the drainer, at the end of a run and
//! at most hourly during it ([`PROOF_INTERVAL`]), for each sealed agent
//! whose outbox is empty, when it has anchors in the verified governance
//! log not yet judged ([`check_anchors`]):
//!
//! 1. asks Agora for every proof it serves the agent after the last anchor
//!    judged, page by page until an empty one, so an anchor Agora serves
//!    no proof for is known as such;
//! 2. takes each anchor's data from the governance log verified at the
//!    start of the run ([`crate::govlog::verify`]), which carries each
//!    entry's signed `data_hash`, not its `data`: a proof's own `date`,
//!    `root` and `leaf_count`, or for an anchor with no proof the entry's
//!    data as Agora serves it ([`SealedApi::anchor_data`]), are used only
//!    if they hash to that signed `data_hash`. So the date and root are
//!    the log's, never the server's copy;
//! 3. checks a proof reaches that root ([`AnchorProof::check`]) and names
//!    an entry on the agent's verified line at its seq;
//! 4. judges it against the versions the anchor may hold, from the line
//!    and the runner's own upload times ([`super::inclusion`],
//!    [`super::receipts`]).
//!
//! Each anchor is judged once, in log order, and recorded in
//! `sealed/inclusion.json`; a failure alerts the operator then
//! ([`AlertKind::SealedStateAnchorWrong`]) and the next verdict seated
//! says it (the same session's, when the start check judged it). Proofs that check are also kept in `sealed/anchors.json`
//! (local, written atomically), as before. A pass that can't go on (Agora
//! unreachable, the line unreadable) stops at the anchor it couldn't judge,
//! with what it judged before saved, and the next pass resumes there.
//!
//! **Publish.** An agent whose sealed-state ledger says `publish` gets
//! `POST /api/state/anchor {publish: true}` once, after its enrolment
//! uploads, or from the drainer for an agent enrolled before this; success
//! writes `sealed/published`. Only the agent's own choice sets it. After
//! that the choice is the agent's, changed with `config` (agentkit 0.151):
//! the marker is never removed, a restore puts it back without asking
//! Agora, and the one publish is skipped when Agora already has a choice on
//! record, so the runner never sends `publish: true` behind the agent's back.

use crate::consent::offer::Stored;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agora_agentkit::crypto::SigningKey;
use agora_agentkit::govlog::Sha256Hex;
use agora_agentkit::ids::{AgentId, GovernanceLogId};
use agora_agentkit::state::storage::{AgentSeal, SEALED_DIR};
use agora_agentkit::state::{AnchorData, AnchorProof, SignedStateEntry};
use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};

use super::api::SealedApi;
use super::inclusion::{self, AnchorFacts, Bounds, Checked, Finding, InclusionFile, Served};
use crate::alerts::{Alert, AlertKind};
use crate::consent::sealed_state::ledger::SealLedger;
use crate::consent::sealed_state::prompt::SealChoice;

/// `sealed/anchors.json`.
pub const ANCHORS: &str = "anchors.json";
/// `sealed/published`: the runner's one publish is done (it told Agora, or
/// a restore found the choice already with Agora). Never removed.
pub const PUBLISHED: &str = "published";
/// The drainer's anchor checks while a run goes on.
pub const PROOF_INTERVAL: Duration = Duration::from_secs(3600);
/// The most pages of proofs one pass reads (Agora's page is 100 proofs,
/// one anchor a day): past it, Agora is taken to be paging without end.
pub const MAX_PAGES: usize = 1000;

/// One `ANC-` entry of the verified governance log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoggedAnchor {
    /// The `data_hash` its signed link attests.
    pub data_hash: Sha256Hex,
    /// When it was logged: Agora prunes a leaf by this time.
    pub logged_at: DateTime<Utc>,
    /// Inside a compromise window the log declares: its signature can't be
    /// relied on, so it is judged unchecked.
    pub repudiated: bool,
}

/// The `ANC-` entries of the governance log verified at the start of the
/// run, and the dates learnt of them, shared by every agent's check.
#[derive(Debug, Clone, Default)]
pub struct VerifiedAnchors {
    entries: BTreeMap<GovernanceLogId, LoggedAnchor>,
    dates: Arc<Mutex<HashMap<GovernanceLogId, NaiveDate>>>,
}

impl VerifiedAnchors {
    pub fn new(
        entries: impl IntoIterator<Item = (GovernanceLogId, Sha256Hex, DateTime<Utc>)>,
    ) -> Self {
        Self {
            entries: entries
                .into_iter()
                .map(|(id, data_hash, logged_at)| {
                    (
                        id,
                        LoggedAnchor {
                            data_hash,
                            logged_at,
                            repudiated: false,
                        },
                    )
                })
                .collect(),
            dates: Arc::default(),
        }
    }

    /// Add `ANC-` entries the log declares repudiated: listed, so no
    /// anchor is passed over in silence, and judged unchecked.
    pub fn with_repudiated(
        mut self,
        entries: impl IntoIterator<Item = (GovernanceLogId, Sha256Hex, DateTime<Utc>)>,
    ) -> Self {
        for (id, data_hash, logged_at) in entries {
            self.entries.insert(
                id,
                LoggedAnchor {
                    data_hash,
                    logged_at,
                    repudiated: true,
                },
            );
        }
        self
    }

    /// The newest `ANC-` entry, by id (`ANC-YYYY-NNNN`, as the proofs'
    /// `after` orders them): the verdict names it when its record can't
    /// be read.
    pub fn newest(&self) -> Option<&GovernanceLogId> {
        self.entries.keys().next_back()
    }

    /// The entry `anc`, as the verified log has it.
    pub fn logged(&self, anc: &GovernanceLogId) -> Option<&LoggedAnchor> {
        self.entries.get(anc)
    }

    /// `anc`'s date, if already learnt this run (from a proof, or its data
    /// as Agora serves it, checked against the log). No network.
    pub fn known_date(&self, anc: &GovernanceLogId) -> Option<NaiveDate> {
        self.dates.lock().expect("dates lock").get(anc).copied()
    }

    /// The entries after `after` (all with `None`), oldest first.
    pub fn after<'a>(
        &'a self,
        after: Option<&'a GovernanceLogId>,
    ) -> impl Iterator<Item = (&'a GovernanceLogId, &'a LoggedAnchor)> + 'a {
        self.entries
            .iter()
            .filter(move |(id, _)| after.is_none_or(|a| *id > a))
    }

    /// The log's [`AnchorData`] for `proof`'s entry: the proof's own
    /// fields, if they hash to what the log attests. `None` when the entry
    /// isn't in the verified log; `Some(Err)` when they don't hash to it.
    fn data(&self, proof: &AnchorProof) -> Option<Result<AnchorData, ()>> {
        let logged = self.entries.get(&proof.anc_id)?;
        if logged.repudiated {
            return Some(Err(()));
        }
        let attested = logged.data_hash;
        let data = AnchorData {
            date: proof.date,
            root: proof.root,
            agents: proof.leaf_count,
        };
        Some(if anchor_data_hash(&data) == Some(attested) {
            self.dates
                .lock()
                .expect("dates lock")
                .insert(proof.anc_id.clone(), data.date);
            Ok(data)
        } else {
            Err(())
        })
    }

    /// `anc`'s date, as the log attests it: learnt from a proof, else
    /// from its data as Agora serves it, checked against the log.
    /// `Ok(None)`: what Agora serves doesn't hash to the log's (or `anc`
    /// isn't in the verified log).
    pub async fn date(
        &self,
        anc: &GovernanceLogId,
        api: &dyn SealedApi,
    ) -> Result<Option<NaiveDate>, String> {
        if let Some(d) = self.dates.lock().expect("dates lock").get(anc) {
            return Ok(Some(*d));
        }
        let Some(logged) = self.entries.get(anc).filter(|l| !l.repudiated) else {
            return Ok(None);
        };
        let served = api.anchor_data(anc).await?;
        if agora_agentkit::govlog::data_hash(&served) != logged.data_hash {
            return Ok(None);
        }
        let Ok(data) = serde_json::from_value::<AnchorData>(served) else {
            return Ok(None);
        };
        self.dates
            .lock()
            .expect("dates lock")
            .insert(anc.clone(), data.date);
        Ok(Some(data.date))
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

/// `sealed/anchors.json`: the proofs that checked, oldest first.
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

    fn save(&self, agent_dir: &Path) -> std::io::Result<()> {
        write_atomic(&path(agent_dir), &serde_json::to_vec_pretty(self)?)
    }
}

/// A temporary file beside `path`, fsynced, renamed over it, and the
/// directory fsynced: a crash leaves the old file or the new, whole.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let dir = path.parent().expect("in sealed/");
    std::fs::create_dir_all(dir)?;
    let tmp = path.with_extension("writing");
    std::fs::write(&tmp, bytes)?;
    std::fs::File::open(&tmp)?.sync_all()?;
    std::fs::rename(&tmp, path)?;
    std::fs::File::open(dir)?.sync_all()
}

/// Why a pass stopped before judging every anchor. Ids and hashes only.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProofError {
    #[error("the proofs could not be fetched: {0}")]
    Fetch(String),
    #[error("{anc_id}'s data could not be fetched: {error}")]
    AnchorData {
        anc_id: GovernanceLogId,
        error: String,
    },
    #[error("sealed/ unreadable: {0}")]
    Local(String),
    #[error("the agent's main line couldn't be read to place a proof: {0}")]
    Line(String),
}

impl ProofError {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Fetch(_) => "fetch",
            Self::AnchorData { .. } => "anchor_data",
            Self::Local(_) => "local",
            Self::Line(_) => "line",
        }
    }
}

/// What one [`check_anchors`] pass did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Pass {
    /// Anchors judged.
    pub judged: usize,
    /// Proofs that checked, newly kept in `sealed/anchors.json`.
    pub kept: usize,
    /// The anchors judged as failures: the operator is told of each.
    pub failures: Vec<Checked>,
    /// Why the pass stopped before the newest anchor, with what it judged
    /// before saved (and its failures above, to be alerted all the same).
    pub stopped: Option<ProofError>,
}

/// The agent's verified line, each version's bounds, and where each entry
/// is on it.
struct Line {
    line: Vec<SignedStateEntry>,
    bounds: Vec<Bounds>,
    at: HashMap<Sha256Hex, usize>,
}

async fn read_line(
    agent_dir: &Path,
    id: AgentId,
    keys: &super::KeyStore,
    api: &dyn SealedApi,
) -> Result<Line, ProofError> {
    let seal = AgentSeal::new(agent_dir, id);
    let head = seal
        .head()
        .await
        .map_err(|e| ProofError::Local(e.to_string()))?
        .ok_or_else(|| ProofError::Line("sealed/head.json is missing".into()))?;
    let line = super::version::verified_line(&seal, keys, api, id, &head)
        .await
        .map_err(|e| ProofError::Line(format!("{e:#}")))?;
    // Unreadable: no times, so nothing decided (sound); the next process
    // sets it aside ([`super::receipts::baseline`])
    let receipts = match super::receipts::ReceiptsFile::load(agent_dir) {
        Ok(file) => file.by_hash(),
        Err(e) => {
            tracing::warn!(
                agent_id = %id,
                error = %e,
                "sealed/receipts.json unreadable; anchors judged without upload times"
            );
            HashMap::new()
        }
    };
    Ok(Line {
        bounds: inclusion::bounds(&line, &receipts),
        at: line
            .iter()
            .enumerate()
            .map(|(i, e)| (e.entry_hash(), i))
            .collect(),
        line,
    })
}

/// Every proof Agora serves `id` after `after`, by anchor, read page by
/// page until an empty one. An anchor served two different proofs is
/// listed in the second map.
async fn served_after(
    id: AgentId,
    signing: &SigningKey,
    api: &dyn SealedApi,
    after: Option<GovernanceLogId>,
) -> Result<BTreeMap<GovernanceLogId, Result<AnchorProof, ()>>, ProofError> {
    let mut served: BTreeMap<GovernanceLogId, Result<AnchorProof, ()>> = BTreeMap::new();
    let mut cursor = after;
    for _ in 0..MAX_PAGES {
        let page = api
            .proofs(id, signing, cursor.clone())
            .await
            .map_err(ProofError::Fetch)?;
        let Some(last) = page.iter().map(|p| p.anc_id.clone()).max() else {
            return Ok(served);
        };
        if cursor.as_ref().is_some_and(|c| last <= *c) {
            return Err(ProofError::Fetch(format!(
                "Agora's proof pages don't advance past {last}"
            )));
        }
        for proof in page {
            if cursor.as_ref().is_some_and(|c| proof.anc_id <= *c) {
                continue;
            }
            match served.get(&proof.anc_id) {
                None => {
                    served.insert(proof.anc_id.clone(), Ok(proof));
                }
                Some(Ok(seen)) if *seen == proof => {}
                Some(_) => {
                    served.insert(proof.anc_id.clone(), Err(()));
                }
            }
        }
        cursor = Some(last);
    }
    Err(ProofError::Fetch(format!(
        "Agora served more than {MAX_PAGES} pages of proofs"
    )))
}

/// Judge each anchor of the verified log after the last one judged, in
/// order ([`super::inclusion`]), recording each in `sealed/inclusion.json`
/// and keeping the proofs that check in `sealed/anchors.json`. Only for an
/// agent whose outbox is empty: every version on its line is then on
/// Agora. Stops at an anchor it can't judge yet, with what came before it
/// saved ([`Pass::stopped`]); `Err` only when nothing could be judged.
/// `alerted`: the runner alerts the operator to a failure (its
/// alert kind is on), which the agent is then told.
pub async fn check_anchors(
    agent_dir: &Path,
    id: AgentId,
    keys: &super::KeyStore,
    signing: &SigningKey,
    api: &dyn SealedApi,
    verified: &VerifiedAnchors,
    alerted: bool,
) -> Result<Pass, ProofError> {
    let local = |e: std::io::Error| ProofError::Local(e.to_string());
    let inc = InclusionFile::load_or_set_aside(agent_dir, id).map_err(local)?;
    let pending: Vec<(GovernanceLogId, LoggedAnchor)> = verified
        .after(inc.through.as_ref())
        .map(|(id, l)| (id.clone(), *l))
        .collect();
    if pending.is_empty() {
        return Ok(Pass::default());
    }
    let mut proofs = AnchorsFile::load(agent_dir).map_err(local)?;
    let served = served_after(id, signing, api, inc.through.clone()).await?;
    let now = Utc::now();
    let mut line: Option<Line> = None;
    let mut pass = Pass::default();
    let mut floor: Option<u64> = None;
    let mut stopped = None;
    let mut judged = Vec::new();
    for (anc, logged) in pending {
        let facts = AnchorFacts {
            before_proofs: inclusion::before_proofs(&anc),
            maybe_pruned: now - logged.logged_at > inclusion::PRUNE_AGE,
        };
        // What Agora serves, checked against the log, and the date
        let (date, served_as, proof) = match served.get(&anc) {
            _ if logged.repudiated => (None, Served::Nothing, None),
            Some(Err(())) => (None, Served::Conflicting, None),
            Some(Ok(proof)) => match verified.data(proof) {
                Some(Ok(data)) if proof.check(id, &data) => {
                    (Some(data.date), Served::Nothing, Some(proof))
                }
                Some(Ok(data)) => (Some(data.date), Served::DidNotCheck, None),
                Some(Err(())) | None => (None, Served::NotTheLogs, None),
            },
            None => match verified.date(&anc, api).await {
                Ok(Some(d)) => (Some(d), Served::Nothing, None),
                Ok(None) => (None, Served::NotTheLogs, None),
                Err(error) => {
                    stopped = Some(ProofError::AnchorData {
                        anc_id: anc.clone(),
                        error,
                    });
                    break;
                }
            },
        };
        let finding = match (served_as, date) {
            _ if logged.repudiated => Finding::Unchecked {
                why: inclusion::Unchecked::Repudiated,
            },
            (Served::Nothing, Some(date)) => {
                if line.is_none() {
                    match read_line(agent_dir, id, keys, api).await {
                        Ok(l) => line = Some(l),
                        Err(e) => {
                            stopped = Some(e);
                            break;
                        }
                    }
                }
                let l = line.as_ref().expect("read above");
                let served_as = match proof {
                    None => Served::Nothing,
                    Some(p) => match l.at.get(&p.entry_hash) {
                        Some(&i) if l.line[i].entry.seq == p.seq => Served::Own(i),
                        _ => Served::NotOnLine(p.entry_hash),
                    },
                };
                let cutoff = inclusion::cutoff(date);
                let expected = inclusion::expected(&l.bounds, cutoff, inclusion::SKEW);
                if let inclusion::Expected::Window { must: Some(k), .. } = expected {
                    floor = Some(floor.unwrap_or(0).max(l.line[k].entry.seq));
                }
                inclusion::judge(
                    &l.line,
                    &l.bounds,
                    expected,
                    served_as,
                    facts,
                    cutoff,
                    inclusion::SKEW,
                )
            }
            (Served::Conflicting, _) => Finding::Conflicting,
            (Served::DidNotCheck, _) => Finding::DidNotCheck,
            _ => Finding::NotTheLogs,
        };
        // A proof that reached the log's root, of the agent's own line, is
        // kept whatever the judgement: it proves what the anchor holds
        if let (Some(p), Some(_)) = (proof, finding.held())
            && !proofs.proofs.iter().any(|k| k.anc_id == p.anc_id)
        {
            proofs.proofs.push(p.clone());
            pass.kept += 1;
        }
        let failed = finding.failed();
        judged.push(Checked {
            anc_id: anc.clone(),
            date,
            finding,
            at: now,
            told: false,
            alerted: alerted && failed,
        });
    }
    if pass.kept > 0 {
        proofs.save(agent_dir).map_err(local)?;
    }
    // Merged under the lock: an anchor another process judged meanwhile is
    // kept as it was, and alerted by that one only
    let added = inclusion::merge(agent_dir, id, judged)
        .await
        .map_err(ProofError::Local)?;
    for c in added {
        if c.finding.failed() {
            tracing::warn!(
                event_type = "sealed_state_anchor_judged",
                agent_id = %id,
                anc_id = %c.anc_id,
                finding = c.finding.kind(),
                "an anchor does not hold what it should for the agent"
            );
            pass.failures.push(c);
        } else {
            tracing::info!(
                event_type = "sealed_state_anchor_judged",
                agent_id = %id,
                anc_id = %c.anc_id,
                finding = c.finding.kind(),
                "anchor judged"
            );
        }
        pass.judged += 1;
    }
    if let Some(seq) = floor
        && let Err(e) = super::receipts::prune_below(agent_dir, id, seq).await
    {
        tracing::warn!(agent_id = %id, error = %e, "sealed/receipts.json not pruned");
    }
    pass.stopped = stopped;
    Ok(pass)
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
    // B3: the choice in force is the one acted on when it began uploading
    // (the marker's `since`), never a later re-answer, which changes nothing
    // by itself; the agent's own `config` tool changes it directly
    // (the local copy: absent or unreadable, nothing is published)
    let since = match std::fs::read(agent_dir.join(SEALED_DIR).join(super::uploading::UPLOADING))
        .ok()
        .and_then(|b| serde_json::from_slice::<super::uploading::Uploading>(&b).ok())
    {
        Some(u) if u.agent_id == id => u.since,
        _ => return Ok(false),
    };
    if ledger.choice_as_of(since) != Some(SealChoice::Publish) {
        return Ok(false);
    }
    // A choice already on record is the agent's own (`config`, while this
    // publish waited on a retry or the enrolment's drain) or this runner's
    // whose marker wasn't written: either way, never sent over it
    if api.publish_chosen(id, signing).await? {
        write_atomic(&marker, chrono::Utc::now().to_rfc3339().as_bytes())
            .map_err(|e| format!("sealed/published not written: {e}"))?;
        tracing::info!(
            event_type = "sealed_state_anchor_choice_on_record",
            agent_id = %id,
            "a publish choice is already on record with Agora; the runner sends none"
        );
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

/// The alert for the anchors one pass judged failures
/// ([`Finding::failed`]): once, when they are judged.
pub fn judged_alert(id: AgentId, name: &str, failures: &[Checked]) -> Alert {
    let ids: Vec<&str> = failures.iter().map(|c| c.anc_id.as_str()).collect();
    let mut alert = Alert::new(
        AlertKind::SealedStateAnchorWrong,
        format!(
            "{name}: {} anchor(s) don't hold what they should ({})",
            failures.len(),
            ids.join(", ")
        ),
    )
    .agent(name, id);
    for c in failures {
        alert = alert.detail(
            "anchor",
            format!(
                "{} ({}): {}",
                c.anc_id,
                c.date
                    .map_or_else(|| "date unknown".to_string(), |d| d.to_string()),
                serde_json::to_string(&c.finding).unwrap_or_default()
            ),
        );
    }
    alert
}

/// Alert the operator to what one [`check_anchors`] pass found, as the
/// drainer and the start check both do: the failures it judged (once:
/// they are never judged again, so alerted even when the pass stopped
/// after them), then why it stopped, if it did.
pub fn alert_pass(
    alerts: &crate::alerts::Alerter,
    id: AgentId,
    name: &str,
    result: &Result<Pass, ProofError>,
) {
    let stopped = match result {
        Ok(pass) => {
            if !pass.failures.is_empty() {
                alerts.notify(judged_alert(id, name, &pass.failures));
            }
            pass.stopped.as_ref()
        }
        Err(e) => Some(e),
    };
    if let Some(e) = stopped {
        tracing::warn!(
            event_type = "sealed_state_anchor_failed",
            agent_id = %id,
            what = "check",
            kind = e.kind(),
            error = %e,
            "the anchor check stopped; it resumes at the next pass"
        );
        alerts.notify(anchor_alert(id, name, "check", &e.to_string()));
    }
}

/// The alert for a pass that stopped, or a publish that failed.
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
    // B3: answered before the opt-in (the marker's `since`), as in life
    pub(crate) async fn chose(dir: &Path, choice: SealChoice) {
        use crate::consent::sealed_state::ledger::{SealAsk, SealOutcome};
        let mut l = SealLedger::default();
        l.record(SealAsk {
            at: "2026-11-01T00:00:00Z".parse().unwrap(),
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

    /// Every entry uploaded, by an older runner: no upload times.
    async fn all_sent(dir: &Path, id: AgentId) {
        let seal = AgentSeal::new(dir, id);
        for item in seal.outbox().await.unwrap() {
            seal.mark_sent(item.ord).await.unwrap();
        }
    }

    /// Every entry uploaded by this process, its times recorded as the
    /// drainer records them.
    pub(crate) async fn uploaded(dir: &Path, id: AgentId) {
        let b = crate::sealed::receipts::before(dir, id, 0).await.unwrap();
        all_sent(dir, id).await;
        crate::sealed::receipts::after(dir, id, 0, &b)
            .await
            .unwrap();
    }

    fn today() -> NaiveDate {
        Utc::now().date_naive()
    }

    /// An `ANC-` entry for `date` over `id`'s `head` among others: its
    /// proof, its id, the log's attested data hash, and its data as served.
    pub(crate) fn anchor_on(
        n: u32,
        date: NaiveDate,
        id: AgentId,
        head: Sha256Hex,
        seq: u64,
    ) -> (AnchorProof, GovernanceLogId, Sha256Hex, serde_json::Value) {
        let mut leaves: Vec<Sha256Hex> = (0..4u8)
            .map(|i| anchor_leaf(AgentId::new(), &Sha256Hex::from([i; 32])))
            .collect();
        leaves.insert(2, anchor_leaf(id, &head));
        let root = merkle_root(&leaves);
        let anc_id: GovernanceLogId = format!("ANC-2026-{n:04}").parse().unwrap();
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
            serde_json::to_value(&data).unwrap(),
        )
    }

    /// For an anchor dated tomorrow: its day ends a day from now or more,
    /// so the versions uploaded now were confirmed well before it.
    pub(crate) fn tomorrow() -> NaiveDate {
        today().succ_opt().unwrap()
    }

    fn logged(list: &[(GovernanceLogId, Sha256Hex)]) -> VerifiedAnchors {
        VerifiedAnchors::new(list.iter().map(|(a, h)| (a.clone(), *h, Utc::now())))
    }

    async fn line(dir: &Path, id: AgentId) -> Vec<(Sha256Hex, u64)> {
        let seal = AgentSeal::new(dir, id);
        crate::sealed::check::local_entries(&seal)
            .await
            .unwrap()
            .iter()
            .map(|e| (e.entry_hash(), e.entry.seq))
            .collect()
    }

    fn findings(dir: &Path) -> Vec<(String, &'static str)> {
        InclusionFile::load(dir)
            .unwrap()
            .checked
            .iter()
            .map(|c| (c.anc_id.to_string(), c.finding.kind()))
            .collect()
    }

    struct Agent {
        _root: crate::sealed::TestDir,
        id: AgentId,
        keys: super::super::KeyStore,
        dir: PathBuf,
        signing: SigningKey,
        api: FakeApi,
    }

    impl Agent {
        async fn new(name: &str, timed: bool) -> Self {
            let root = crate::sealed::TestDir::new(name);
            let (id, keys, dir) = enrolled(&root).await;
            if timed {
                uploaded(&dir, id).await;
            } else {
                all_sent(&dir, id).await;
            }
            let signing = keys.signing(id).unwrap();
            Self {
                _root: root,
                id,
                keys,
                dir,
                signing,
                api: FakeApi::new(),
            }
        }

        async fn check(&self, verified: &VerifiedAnchors) -> Result<Pass, ProofError> {
            check_anchors(
                &self.dir,
                self.id,
                &self.keys,
                &self.signing,
                &self.api,
                verified,
                true,
            )
            .await
        }

        fn serve(&self, data: &[(GovernanceLogId, serde_json::Value)]) {
            let mut map = self.api.anchor_data.lock().unwrap();
            for (a, d) in data {
                map.insert(a.clone(), d.clone());
            }
        }
    }

    /// The start check judges what it can before the verdict is written
    /// (beta round 12: on the first session under 0.151.3 nothing had been
    /// judged, and every verdict said nothing of any anchor), for an agent
    /// whose versions are all on Agora: said in that session's verdict,
    /// told only after it saves, never judged twice; a failure goes to the
    /// operator through `judged`; a pass that can't go on is said, with why
    #[tokio::test]
    async fn the_start_check_judges_the_anchors_before_the_verdict() {
        use crate::sealed::check::{Outcome, StartAnchors, check_with, tests::local_hash};
        let a = Agent::new("anc-start", true).await;
        let seal = AgentSeal::new(&a.dir, a.id);
        *a.api.chain.lock().unwrap() =
            Ok(crate::sealed::check::local_entries(&seal).await.unwrap());
        let l = line(&a.dir, a.id).await;
        let (head, seq) = l[1];
        let (proof, anc, attested, _) = anchor_on(2, tomorrow(), a.id, head, seq);
        a.api.proofs.lock().unwrap().push(proof);
        let verified = logged(&[(anc.clone(), attested)]);
        let local = local_hash(a._root.path(), a.id).await;
        let start = StartAnchors {
            verified: &verified,
            alerted: true,
        };
        let run = async || {
            let out = check_with(
                &a.dir,
                a.id,
                &local,
                crate::sealed::sizes::Sizes::default(),
                &a.keys,
                &a.api,
                Some(&start),
            )
            .await;
            let Outcome::Run {
                seat,
                hold: None,
                told,
                judged,
                ..
            } = out
            else {
                panic!("{out:?}")
            };
            (seat[0].clone(), told, judged)
        };
        let (verdict, told, judged) = run().await;
        let pass = judged.expect("judged at the start").unwrap();
        assert_eq!((pass.judged, pass.failures.len()), (1, 0));
        assert!(
            verdict.contains(&format!(
                ". {anc}, the anchor for {}, holds the version it should: the latest that \
                 Agora had confirmed receiving from you by the end of that day (UTC).",
                tomorrow().format("%Y-%m-%d")
            )),
            "{verdict}"
        );
        assert!(!verdict.contains("hasn't been checked"), "{verdict}");
        assert_eq!(told, vec![anc.clone()]);
        // Not marked told by the check: the session that shows it does
        assert!(!InclusionFile::load(&a.dir).unwrap().checked[0].told);
        // Judged once: the next start asks Agora for no proof
        a.api.calls.lock().unwrap().clear();
        let (again, _, judged) = run().await;
        assert_eq!(judged, Some(Ok(Pass::default())));
        assert_eq!(again, verdict);
        assert!(
            a.api.calls.lock().unwrap().iter().all(|c| c != "proofs"),
            "{:?}",
            a.api.calls.lock().unwrap()
        );

        // A newer anchor whose proof can't be fetched: the pass stops, and
        // the verdict says so, with when it is tried again
        let (proof3, anc3, attested3, _) = anchor_on(3, tomorrow(), a.id, head, seq);
        let _ = proof3;
        let verified = logged(&[(anc.clone(), attested), (anc3.clone(), attested3)]);
        let start = StartAnchors {
            verified: &verified,
            alerted: true,
        };
        *a.api.fail.lock().unwrap() = Some("proofs");
        let out = check_with(
            &a.dir,
            a.id,
            &local,
            crate::sealed::sizes::Sizes::default(),
            &a.keys,
            &a.api,
            Some(&start),
        )
        .await;
        let Outcome::Run { seat, judged, .. } = out else {
            panic!("{out:?}")
        };
        assert!(
            matches!(judged, Some(Err(ProofError::Fetch(_)))),
            "{judged:?}"
        );
        assert!(
            seat[0].contains(&format!(
                "holds the version it should: the latest that Agora had confirmed receiving \
                 from you by the end of that day (UTC). Your runner checked this against the \
                 governance log and its own record of your uploads. {anc3}, logged {} (UTC) \
                 and the newest in the governance log, hasn't been checked for you yet: your \
                 runner's check at the start of this session stopped, as Agora's proofs \
                 couldn't be fetched. It tries again after this session's uploads",
                Utc::now().format("%Y-%m-%d")
            )),
            "{}",
            seat[0]
        );
        *a.api.fail.lock().unwrap() = None;

        // A failure judged at the start is alerted through `judged` and
        // said in that session's verdict
        let b = Agent::new("anc-start-fail", true).await;
        let seal = AgentSeal::new(&b.dir, b.id);
        *b.api.chain.lock().unwrap() =
            Ok(crate::sealed::check::local_entries(&seal).await.unwrap());
        let l = line(&b.dir, b.id).await;
        // The genesis, though the later version was confirmed before the
        // day ended: older than it should be
        let (old_head, old_seq) = l[0];
        let (proof, anc, attested, _) = anchor_on(2, tomorrow(), b.id, old_head, old_seq);
        b.api.proofs.lock().unwrap().push(proof);
        let verified = logged(&[(anc.clone(), attested)]);
        let out = check_with(
            &b.dir,
            b.id,
            &local_hash(b._root.path(), b.id).await,
            crate::sealed::sizes::Sizes::default(),
            &b.keys,
            &b.api,
            Some(&StartAnchors {
                verified: &verified,
                alerted: true,
            }),
        )
        .await;
        let Outcome::Run { seat, judged, .. } = out else {
            panic!("{out:?}")
        };
        let pass = judged.unwrap().unwrap();
        assert_eq!(pass.failures.len(), 1);
        assert!(
            seat[0].contains("holds an older version of yours")
                && seat[0].contains("Your runner has alerted your operator."),
            "{}",
            seat[0]
        );
    }

    #[tokio::test]
    async fn an_anchor_holding_the_latest_confirmed_version_passes_and_is_kept() {
        let a = Agent::new("anc-holds", true).await;
        let l = line(&a.dir, a.id).await;
        let (head, seq) = l[1];
        let (proof, anc, attested, _) = anchor_on(2, tomorrow(), a.id, head, seq);
        a.api.proofs.lock().unwrap().push(proof.clone());

        // Not in the verified log yet: nothing asked, nothing judged
        let pass = a.check(&VerifiedAnchors::default()).await.unwrap();
        assert_eq!(pass, Pass::default());
        assert!(a.api.calls.lock().unwrap().is_empty());

        let verified = logged(&[(anc.clone(), attested)]);
        let pass = a.check(&verified).await.unwrap();
        assert_eq!((pass.judged, pass.kept), (1, 1));
        assert!(pass.failures.is_empty());
        assert_eq!(findings(&a.dir), vec![(anc.to_string(), "holds")]);
        assert_eq!(AnchorsFile::load(&a.dir).unwrap().proofs, vec![proof]);
        let inc = InclusionFile::load(&a.dir).unwrap();
        assert_eq!(
            inc.checked[0].finding,
            Finding::Holds {
                held: agora_agentkit::state::verify::HeadRef {
                    entry_hash: head,
                    seq
                },
                in_flight: false
            }
        );
        // Judged once: the next pass asks nothing
        a.api.calls.lock().unwrap().clear();
        assert_eq!(a.check(&verified).await.unwrap(), Pass::default());
        assert!(a.api.calls.lock().unwrap().is_empty());
        // The receipt below the owed version is no longer needed
        let kept: Vec<u64> = crate::sealed::receipts::ReceiptsFile::load(&a.dir)
            .unwrap()
            .versions
            .iter()
            .map(|r| r.seq)
            .collect();
        assert_eq!(kept, vec![1]);
    }

    #[tokio::test]
    async fn an_anchor_with_no_proof_where_one_is_owed_is_missing_once() {
        let a = Agent::new("anc-missing", true).await;
        let (_, anc, attested, data) = anchor_on(2, tomorrow(), a.id, Sha256Hex::from([0; 32]), 0);
        a.serve(&[(anc.clone(), data)]);
        let verified = logged(&[(anc.clone(), attested)]);
        let pass = a.check(&verified).await.unwrap();
        assert_eq!(pass.failures.len(), 1);
        assert!(matches!(
            pass.failures[0].finding,
            Finding::Missing { expected } if expected.seq == 1
        ));
        assert_eq!(pass.failures[0].date, Some(tomorrow()));
        // Never re-alerted
        assert!(a.check(&verified).await.unwrap().failures.is_empty());
        assert_eq!(findings(&a.dir).len(), 1);
    }

    #[tokio::test]
    async fn an_older_or_a_later_version_is_a_failure() {
        let a = Agent::new("anc-older", true).await;
        let l = line(&a.dir, a.id).await;
        // The genesis, though version 1 was confirmed before the day ended
        let (old, anc, attested, _) = anchor_on(2, tomorrow(), a.id, l[0].0, l[0].1);
        a.api.proofs.lock().unwrap().push(old);
        // Before the agent's first version: none can be in it
        let early = today() - chrono::Duration::days(3);
        let (late, anc3, attested3, _) = anchor_on(3, early, a.id, l[1].0, l[1].1);
        a.api.proofs.lock().unwrap().push(late);
        let pass = a
            .check(&logged(&[(anc, attested), (anc3, attested3)]))
            .await
            .unwrap();
        let kinds: Vec<_> = pass.failures.iter().map(|c| c.finding.kind()).collect();
        assert_eq!(kinds, vec!["older", "later"]);
        assert!(matches!(
            pass.failures[1].finding,
            Finding::Later {
                sealed_after: true,
                ..
            }
        ));
        // Both proofs checked against the log: kept as what the anchors hold
        assert_eq!(AnchorsFile::load(&a.dir).unwrap().proofs.len(), 2);
    }

    #[tokio::test]
    async fn anchors_before_enrolment_owe_nothing() {
        let a = Agent::new("anc-before", true).await;
        let early = today() - chrono::Duration::days(3);
        let (_, anc, attested, data) = anchor_on(2, early, a.id, Sha256Hex::from([0; 32]), 0);
        a.serve(&[(anc.clone(), data)]);
        let pass = a.check(&logged(&[(anc.clone(), attested)])).await.unwrap();
        assert!(pass.failures.is_empty());
        assert_eq!(findings(&a.dir), vec![(anc.to_string(), "not_included")]);
    }

    #[tokio::test]
    async fn without_upload_times_nothing_passes_or_fails_by_version() {
        let a = Agent::new("anc-legacy", false).await;
        let l = line(&a.dir, a.id).await;
        let (proof, anc, attested, _) = anchor_on(2, tomorrow(), a.id, l[0].0, l[0].1);
        a.api.proofs.lock().unwrap().push(proof);
        let (_, anc3, attested3, data3) = anchor_on(3, tomorrow(), a.id, l[1].0, l[1].1);
        a.serve(&[(anc3.clone(), data3)]);
        let pass = a
            .check(&logged(&[
                (anc.clone(), attested),
                (anc3.clone(), attested3),
            ]))
            .await
            .unwrap();
        assert!(pass.failures.is_empty());
        assert_eq!(
            findings(&a.dir),
            vec![
                (anc.to_string(), "holds_unchecked"),
                (anc3.to_string(), "unchecked")
            ]
        );
    }

    #[tokio::test]
    async fn the_first_anchor_and_pruned_ones_are_unchecked_not_missing() {
        let a = Agent::new("anc-early-pruned", true).await;
        let (_, anc1, attested1, data1) =
            anchor_on(1, tomorrow(), a.id, Sha256Hex::from([0; 32]), 0);
        let (_, anc2, attested2, data2) =
            anchor_on(2, tomorrow(), a.id, Sha256Hex::from([0; 32]), 0);
        a.serve(&[(anc1.clone(), data1), (anc2.clone(), data2)]);
        let verified = VerifiedAnchors::new([
            (anc1.clone(), attested1, Utc::now()),
            (
                anc2.clone(),
                attested2,
                Utc::now() - chrono::Duration::days(40),
            ),
        ]);
        let pass = a.check(&verified).await.unwrap();
        assert!(pass.failures.is_empty());
        let inc = InclusionFile::load(&a.dir).unwrap();
        assert_eq!(
            inc.checked[0].finding,
            Finding::Unchecked {
                why: inclusion::Unchecked::BeforeProofs
            }
        );
        assert_eq!(
            inc.checked[1].finding,
            Finding::Unchecked {
                why: inclusion::Unchecked::Pruned
            }
        );
    }

    /// An anchor inside a compromise window the log declares is named
    /// unchecked, never passed over, and nothing is fetched for it.
    #[tokio::test]
    async fn a_repudiated_anchor_is_unchecked() {
        let a = Agent::new("anc-repudiated", true).await;
        let (head, seq) = line(&a.dir, a.id).await[1];
        let (p, anc, att, _) = anchor_on(2, tomorrow(), a.id, head, seq);
        a.api.proofs.lock().unwrap().push(p);
        let verified = VerifiedAnchors::default().with_repudiated([(anc.clone(), att, Utc::now())]);
        let pass = a.check(&verified).await.unwrap();
        assert!(pass.failures.is_empty());
        assert_eq!(
            pass.kept, 0,
            "its proof isn't kept: the log can't vouch for it"
        );
        assert_eq!(
            InclusionFile::load(&a.dir).unwrap().checked[0].finding,
            Finding::Unchecked {
                why: inclusion::Unchecked::Repudiated
            }
        );
        assert!(
            !a.api
                .calls
                .lock()
                .unwrap()
                .iter()
                .any(|c| c == "anchor_data")
        );
    }

    /// Versions an older runner uploaded get a late acknowledgement when a
    /// process first sees them: enough to decide anchors after it, so a
    /// quiet agent's later anchors are checked, never before it.
    #[tokio::test]
    async fn versions_found_uploaded_decide_the_anchors_after() {
        let a = Agent::new("anc-observed", false).await;
        crate::sealed::receipts::baseline(&a.dir, a.id).await;
        let (head, seq) = line(&a.dir, a.id).await[1];
        let (_, anc, att, data) = anchor_on(2, tomorrow(), a.id, head, seq);
        a.serve(&[(anc.clone(), data)]);
        let pass = a.check(&logged(&[(anc, att)])).await.unwrap();
        assert!(matches!(
            pass.failures[0].finding,
            Finding::Missing { expected } if expected.seq == seq
        ));
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
        let verified = VerifiedAnchors::new([(anc_id, attested, Utc::now())]);
        assert_eq!(verified.data(&proof), Some(Ok(data)));
        let mut off_by_one = proof;
        off_by_one.leaf_count += 1;
        assert_eq!(verified.data(&off_by_one), Some(Err(())));
    }

    #[tokio::test]
    async fn forged_conflicting_and_foreign_proofs_are_failures() {
        let a = Agent::new("anc-forged", true).await;
        let (head, seq) = line(&a.dir, a.id).await[1];
        let (good2, anc2, att2, _) = anchor_on(2, tomorrow(), a.id, head, seq);
        let mut root_forged = good2;
        root_forged.root = Sha256Hex::from([7; 32]);
        let (good3, anc3, att3, _) = anchor_on(3, tomorrow(), a.id, head, seq);
        let mut path_forged = good3;
        path_forged.proof[0].sibling = Sha256Hex::from([8; 32]);
        // Another entry, really in the tree under the agent's id
        let (foreign, anc4, att4, _) =
            anchor_on(4, tomorrow(), a.id, Sha256Hex::from([9; 32]), seq);
        let (good5, anc5, att5, _) = anchor_on(5, tomorrow(), a.id, head, seq);
        let mut other5 = good5.clone();
        other5.leaf_index = 1;
        let (good6, anc6, att6, _) = anchor_on(6, tomorrow(), a.id, head, seq);
        *a.api.proofs.lock().unwrap() = vec![
            root_forged,
            path_forged,
            foreign,
            good5,
            other5,
            good6.clone(),
        ];
        let pass = a
            .check(&logged(&[
                (anc2, att2),
                (anc3, att3),
                (anc4, att4),
                (anc5, att5),
                (anc6.clone(), att6),
            ]))
            .await
            .unwrap();
        let kinds: Vec<_> = pass.failures.iter().map(|c| c.finding.kind()).collect();
        assert_eq!(
            kinds,
            vec![
                "not_the_logs",
                "did_not_check",
                "not_on_line",
                "conflicting"
            ]
        );
        // The failures don't stop the anchors after them
        assert_eq!(
            findings(&a.dir).last().unwrap(),
            &(anc6.to_string(), "holds")
        );
        assert_eq!(AnchorsFile::load(&a.dir).unwrap().proofs, vec![good6]);
    }

    #[tokio::test]
    async fn served_data_that_isnt_the_logs_is_a_failure() {
        let a = Agent::new("anc-data-forged", true).await;
        let (_, anc, attested, mut data) =
            anchor_on(2, tomorrow(), a.id, Sha256Hex::from([0; 32]), 0);
        data["agents"] = serde_json::json!(99);
        a.serve(&[(anc.clone(), data)]);
        let pass = a.check(&logged(&[(anc, attested)])).await.unwrap();
        assert_eq!(pass.failures.len(), 1);
        assert_eq!(pass.failures[0].finding, Finding::NotTheLogs);
        assert_eq!(pass.failures[0].date, None);
    }

    #[tokio::test]
    async fn a_pass_that_cant_go_on_stops_there_and_resumes() {
        let a = Agent::new("anc-resume", true).await;
        let (head, seq) = line(&a.dir, a.id).await[1];
        let (_, anc2, att2, data2) = anchor_on(2, tomorrow(), a.id, head, seq);
        let (_, anc3, att3, data3) = anchor_on(3, tomorrow(), a.id, head, seq);
        let (p4, anc4, att4, _) = anchor_on(4, tomorrow(), a.id, head, seq);
        *a.api.proofs.lock().unwrap() = vec![p4];
        a.serve(&[(anc2.clone(), data2)]);
        let verified = logged(&[
            (anc2.clone(), att2),
            (anc3.clone(), att3),
            (anc4.clone(), att4),
        ]);
        // ANC 2 judged missing, then ANC 3's data can't be had: the pass
        // stops there, and still returns ANC 2's failure for its alert (it
        // is never judged again)
        let pass = a.check(&verified).await.unwrap();
        assert_eq!(pass.stopped.as_ref().unwrap().kind(), "anchor_data");
        assert_eq!(pass.failures.len(), 1);
        assert_eq!(pass.failures[0].anc_id, anc2);
        assert_eq!(findings(&a.dir), vec![(anc2.to_string(), "missing")]);
        // Fetch fails outright: nothing judged, nothing lost
        *a.api.fail.lock().unwrap() = Some("proofs");
        assert_eq!(a.check(&verified).await.unwrap_err().kind(), "fetch");
        *a.api.fail.lock().unwrap() = None;
        a.serve(&[(anc3.clone(), data3)]);
        let pass = a.check(&verified).await.unwrap();
        assert!(pass.stopped.is_none());
        assert_eq!(pass.judged, 2);
        assert_eq!(
            findings(&a.dir),
            vec![
                (anc2.to_string(), "missing"),
                (anc3.to_string(), "missing"),
                (anc4.to_string(), "holds")
            ]
        );
    }

    #[tokio::test]
    async fn every_page_is_read_and_a_page_that_doesnt_advance_stops_it() {
        let a = Agent::new("anc-pages", true).await;
        let (head, seq) = line(&a.dir, a.id).await[1];
        let mut log = Vec::new();
        for n in 2..7 {
            let (p, anc, att, _) = anchor_on(n, tomorrow(), a.id, head, seq);
            a.api.proofs.lock().unwrap().push(p);
            log.push((anc, att));
        }
        *a.api.proofs_page.lock().unwrap() = Some(2);
        let pass = a.check(&logged(&log)).await.unwrap();
        assert_eq!((pass.judged, pass.kept), (5, 5));
        assert!(pass.failures.is_empty());

        // A server that answers the same page whatever `after` is
        let b = Agent::new("anc-pages-stuck", true).await;
        let (p, anc, att, _) = anchor_on(2, tomorrow(), b.id, head, seq);
        b.api.proofs.lock().unwrap().push(p);
        *b.api.proofs_ignore_after.lock().unwrap() = true;
        let stuck = logged(&[(anc, att)]);
        let e = b.check(&stuck).await.unwrap_err();
        assert_eq!(e.kind(), "fetch", "{e}");
        assert!(InclusionFile::load(&b.dir).unwrap().checked.is_empty());
    }

    /// After a lost-directory restore sent/ and the receipts are gone: a
    /// proof for the restored head is placed on the verified line through
    /// Agora and kept, its version unchecked (no upload times survive).
    #[tokio::test]
    async fn a_proof_is_placed_after_a_lost_directory_restore() {
        let root = crate::sealed::TestDir::new("anchors-restored");
        let (id, keys, dir) = enrolled(&root).await;
        let api = FakeApi::new();
        let seal = AgentSeal::new(&dir, id);
        *api.chain.lock().unwrap() = Ok(crate::sealed::check::local_entries(&seal).await.unwrap());
        uploaded(&dir, id).await;
        for f in std::fs::read_dir(dir.join("sealed/blobs")).unwrap() {
            let f = f.unwrap();
            let sha = f.file_name().to_str().unwrap().parse().unwrap();
            api.blobs
                .lock()
                .unwrap()
                .insert(sha, std::fs::read(f.path()).unwrap());
        }
        let (hash, seq) = line(&dir, id).await[1];
        std::fs::rename(&dir, root.join("gone")).unwrap();
        crate::sealed::restore::restore(&root.join("state"), &keys, &api, id, None, false)
            .await
            .unwrap();
        assert!(seal.sent().await.unwrap().is_empty());
        let (proof, anc, attested, _) = anchor_on(2, tomorrow(), id, hash, seq);
        api.proofs.lock().unwrap().push(proof.clone());
        let signing = keys.signing(id).unwrap();
        let verified = logged(&[(anc.clone(), attested)]);
        // The line can't be read (Agora's chain unreachable): the pass
        // stops there, nothing judged or stored
        let chain = api.chain.lock().unwrap().clone();
        *api.chain.lock().unwrap() = Err("unreachable".into());
        let pass = check_anchors(&dir, id, &keys, &signing, &api, &verified, true)
            .await
            .unwrap();
        assert_eq!(pass.stopped.unwrap().kind(), "line");
        assert_eq!(pass.judged, 0);
        assert!(InclusionFile::load(&dir).unwrap().checked.is_empty());
        *api.chain.lock().unwrap() = chain;
        let pass = check_anchors(&dir, id, &keys, &signing, &api, &verified, true)
            .await
            .unwrap();
        assert_eq!((pass.judged, pass.kept), (1, 1));
        assert_eq!(findings(&dir), vec![(anc.to_string(), "holds_unchecked")]);
        assert_eq!(AnchorsFile::load(&dir).unwrap().proofs, vec![proof]);
    }

    // B3
    /// A re-answer after the opt-in changes nothing by itself: a later
    /// `publish` over the `private` acted on publishes nothing.
    #[tokio::test]
    async fn a_later_publish_answer_publishes_nothing_by_itself() {
        use crate::consent::sealed_state::ledger::{SealAsk, SealOutcome};
        let root = crate::sealed::TestDir::new("anchors-reanswer");
        let (id, keys, dir) = enrolled(&root).await;
        let signing = keys.signing(id).unwrap();
        let api = FakeApi::new();
        chose(&dir, SealChoice::Private).await;
        let mut l = SealLedger::load(&dir).await.unwrap();
        l.record(SealAsk {
            at: chrono::Utc::now() + chrono::Duration::seconds(5),
            offer_version: 9,
            model: "m".into(),
            order: SealChoice::ALL,
            order_seed: 0,
            constrained: true,
            attempts: 1,
            outcome: SealOutcome::Answered {
                choice: SealChoice::Publish,
                reason: String::new(),
                memory_note_written: false,
                note_dropped: None,
                apply_failed: None,
            },
        });
        l.save(&dir).await.unwrap();
        assert_eq!(l.choice(), Some(SealChoice::Publish));
        assert_eq!(publish_if_chosen(&dir, id, &signing, &api).await, Ok(false));
        assert!(api.anchored.lock().unwrap().is_empty());
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

    /// An agent that set its choice with `config` while the runner's one
    /// publish was still waiting (a failed try, or the enrolment's drain)
    /// keeps it: the runner sends nothing over a choice on record
    #[tokio::test]
    async fn a_choice_the_agent_made_first_is_not_overridden() {
        let root = crate::sealed::TestDir::new("anchors-publish-first");
        let (id, keys, dir) = enrolled(&root).await;
        let signing = keys.signing(id).unwrap();
        chose(&dir, SealChoice::Publish).await;
        let api = FakeApi::new();
        *api.fail.lock().unwrap() = Some("post_anchor");
        assert!(publish_if_chosen(&dir, id, &signing, &api).await.is_err());
        *api.fail.lock().unwrap() = None;
        // The agent turns it off with `config` before the retry
        *api.chosen.lock().unwrap() = true;
        assert_eq!(publish_if_chosen(&dir, id, &signing, &api).await, Ok(false));
        assert!(api.anchored.lock().unwrap().is_empty(), "nothing sent");
        assert!(
            dir.join("sealed").join(PUBLISHED).exists(),
            "and never again"
        );
        assert_eq!(publish_if_chosen(&dir, id, &signing, &api).await, Ok(false));
        assert!(api.anchored.lock().unwrap().is_empty());
    }

    /// An agent that answered `private` and later turns publishing on with
    /// `config` (offer v8 says it can) is left alone: the runner sends
    /// nothing for a private answer, marker or not, and never `false`
    #[tokio::test]
    async fn a_private_agent_that_turns_publishing_on_is_left_alone() {
        let root = crate::sealed::TestDir::new("anchors-private-on");
        let (id, keys, dir) = enrolled(&root).await;
        let signing = keys.signing(id).unwrap();
        chose(&dir, SealChoice::Private).await;
        let api = FakeApi::new();
        *api.chosen.lock().unwrap() = true;
        for _ in 0..3 {
            assert_eq!(publish_if_chosen(&dir, id, &signing, &api).await, Ok(false));
        }
        assert!(api.anchored.lock().unwrap().is_empty(), "nothing sent");
    }

    /// Once told, the runner never sends `publish` again, whatever Agora
    /// holds: the agent turning it off with `config` is not undone by a
    /// later drain or enrolment pass (agentkit 0.151)
    #[tokio::test]
    async fn a_publish_the_agent_turned_off_is_not_sent_again() {
        let root = crate::sealed::TestDir::new("anchors-publish-off");
        let (id, keys, dir) = enrolled(&root).await;
        let signing = keys.signing(id).unwrap();
        chose(&dir, SealChoice::Publish).await;
        let api = FakeApi::new();
        assert_eq!(publish_if_chosen(&dir, id, &signing, &api).await, Ok(true));
        // The agent's `config` call goes to Agora, not through the runner;
        // every later pass finds the marker
        for _ in 0..3 {
            assert_eq!(publish_if_chosen(&dir, id, &signing, &api).await, Ok(false));
        }
        assert_eq!(*api.anchored.lock().unwrap(), vec![true], "the one publish");
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
