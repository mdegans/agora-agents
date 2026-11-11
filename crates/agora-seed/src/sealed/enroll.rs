//! `agora-seed sealed-enroll`: the local chain of every agent, and the
//! opt-in that starts an agent's uploads (agora
//! `memory/project_local_chain_for_all.md`, 2026-11-11: every agent is
//! chained locally; consent decides only whether the chain is uploaded).
//!
//! **Chain** ([`Enroller::chain`]), for any agent with a state, with **no
//! server contact** at all:
//!
//! 1. the state key: the cached one, or one generated and cached in
//!    `<secrets>/<id>/state_key.json` (nothing is wrapped or posted);
//! 2. the genesis, its origin from the memory-restore config and ledger
//!    ([`origin`]): fields only, never the recovered text. A provisioned
//!    agent's is class `none` ([`Genesis::New`]);
//! 3. the pre-loss copy as `archived_prior`, for an agent that chose to see
//!    it;
//! 4. the archives (`state.<utc>.json`), oldest first by mtime, each
//!    `state_at` its mtime (when that state was written);
//! 5. the current state;
//! 6. the chain marker (agentkit's `CHAINED`): from here every save is sealed and
//!    chained, locally; then the agent is recorded in the roster.
//!
//! With no session in progress, under the agent's `sealed/lock`. A partial
//! chain (a chain but no marker) is resumed: the archives after the
//! newest state it sealed, then the current state, then the marker, with
//! the same key. An agent that answers `show` to the memory-restore offer
//! after it was chained gets its pre-loss copy appended then
//! ([`Enroller::append_prior`]); its genesis never changes.
//!
//! **Opt in** ([`Enroller::opt_in`]), for a chained agent whose sealed-state
//! ledger says yes, under the upload guard's opt-in hold: Agora must hold no
//! chain of it (or a prefix of this one), its state key record is posted,
//! and only then is it marked uploading ([`super::uploading::set`]): the
//! record is on Agora before any entry. The caller then drains, which
//! uploads the whole local chain. Every step is resumable.
//!
//! **Resume** ([`Enroller::resume`]) after a stop: the uploaded line goes on
//! from the last entry Agora holds with one new snapshot of the current
//! state; the versions saved while stopped are set aside here (`.displaced`,
//! openable by the `memory` tool) and never uploaded.
//!
//! The caller (`sealed-enroll`) holds the runner lock exclusively for the
//! whole command. `--dry-run` reads sizes and markers and prints counts and
//! sizes only: no writes, no network.

use crate::consent::offer::Stored;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use agora_agentkit::enums::StateEntryKind;
use agora_agentkit::govlog::Sha256Hex;
use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::seed::ShortString;
use agora_agentkit::reactor::storage::FsStorageError;
use agora_agentkit::reactor::{FsStorage, Storage};
use agora_agentkit::state::codec::KeyframePolicy;
use agora_agentkit::state::crypto::StateKey;
use agora_agentkit::state::storage::{AgentSeal, plaintext_hash};
use agora_agentkit::state::verify::HeadRef;
use agora_agentkit::state::{GenesisClass, GenesisOrigin, IncidentCite};
use anyhow::Context;
use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};

use super::KeyStore;
use super::api::{SealedApi, UploadGuard};
use super::keys::StateKeyFile;
use crate::consent::memory_restore::{
    self, RestoreOffer, ledger::RestoreLedger, prompt::RestoreChoice,
};
use crate::consent::sealed_state::ledger::SealLedger;

/// When the copy recovered from the May loss stood.
pub fn prior_copy_at() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 5, 3, 0, 0, 0).unwrap()
}

/// What chaining and opting in read around an agent. Nothing here talks to
/// Agora but [`Self::opt_in`], through the guard it is given.
pub struct Enroller {
    pub state_dir: PathBuf,
    pub keys: KeyStore,
    /// `[memory_restore]`, resolved, when the run config has one.
    pub restore: Option<RestoreOffer>,
}

/// One archive: `state.<utc>.json`.
#[derive(Debug, Clone)]
pub struct Archive {
    pub path: PathBuf,
    pub mtime: DateTime<Utc>,
    pub bytes: u64,
}

/// The archives in `agent_dir`, oldest first by mtime (then name).
pub fn archives(agent_dir: &Path) -> std::io::Result<Vec<Archive>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(agent_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let archived = name.starts_with("state.")
            && name.ends_with(".json")
            && name != "state.json"
            && !name.ends_with(".tmp");
        if !archived {
            continue;
        }
        let meta = entry.metadata()?;
        if !meta.is_file() {
            continue;
        }
        out.push(Archive {
            path: entry.path(),
            mtime: DateTime::<Utc>::from(meta.modified()?),
            bytes: meta.len(),
        });
    }
    out.sort_by(|a, b| a.mtime.cmp(&b.mtime).then_with(|| a.path.cmp(&b.path)));
    Ok(out)
}

/// The genesis origin, and the pre-loss copy to seal (if the agent chose
/// to see it and it is there).
#[derive(Debug, Clone, PartialEq)]
pub struct Origin {
    pub origin: GenesisOrigin,
    pub prior: Option<PathBuf>,
}

/// From the memory-restore listing and ledger, never the recovered text:
/// `verified` if the agent chose `show` and was shown it, `known_break`
/// (citing the thread) if it is listed, else `unknown`.
pub async fn origin(
    restore: Option<&RestoreOffer>,
    name: &ShortString<64>,
    id: AgentId,
    agent_dir: &Path,
) -> anyhow::Result<Origin> {
    let unknown = Origin {
        origin: GenesisOrigin {
            class: GenesisClass::Unknown,
            incident: None,
            prior_copy_held: false,
            verified_at: None,
        },
        prior: None,
    };
    let Some(offer) = restore else {
        return Ok(unknown);
    };
    if offer.listing(name, id) != memory_restore::Listing::Listed {
        return Ok(unknown);
    }
    let ledger = RestoreLedger::load(agent_dir)
        .await
        .context("memory-restore ledger unreadable")?;
    let path = offer.path_for(name);
    let held = std::fs::exists(&path).unwrap_or(false);
    let incident = offer.thread().map(IncidentCite::Post);
    let chose_show = ledger.choice() == Some(RestoreChoice::Show);
    let shown = ledger.showings.first().map(|s| s.at);
    let (class, verified_at) = match (chose_show, shown) {
        (true, Some(at)) => (GenesisClass::Verified, Some(at)),
        _ => (GenesisClass::KnownBreak, None),
    };
    Ok(Origin {
        origin: GenesisOrigin {
            class,
            incident,
            prior_copy_held: held,
            verified_at,
        },
        prior: (chose_show && held).then_some(path),
    })
}

/// What a genesis records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Genesis {
    /// From the memory-restore record ([`origin`]): the fleet's agents,
    /// whose history before their first archive is unknown or broken.
    FromRecord,
    /// Class `none`: an agent provisioned here, chained from its first
    /// state, its whole history in the chain.
    New,
}

/// What one chaining did, or would do: counts and sizes only.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    pub archives: usize,
    pub archive_bytes: u64,
    /// Archives sealed by this run.
    pub sealed: usize,
    /// Archives left out as the same state as the version before.
    pub duplicates: usize,
    /// Archives left out for good: not JSON, not an envelope, or a format
    /// newer than this binary reads; recorded in `sealed/left_out.json`.
    pub unreadable: usize,
    pub prior: bool,
    /// The pre-loss copy was appended to a chain made before the agent
    /// chose to see it ([`Enroller::append_prior`]).
    pub prior_later: bool,
    pub origin: String,
    pub current_bytes: u64,
    /// A partial chain was continued, or a stopped agent resumed.
    pub resumed: bool,
    /// Archives a partial chain had already sealed.
    pub resumed_past: usize,
    /// Versions saved while uploads were stopped, set aside by a resume.
    pub kept_local: usize,
    /// Entries in the chain at the end (genesis included).
    pub entries: usize,
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.prior_later {
            return write!(
                f,
                "pre-loss copy appended to its chain, entries {}",
                self.entries
            );
        }
        write!(
            f,
            "origin {}, archives {} ({} bytes: {} sealed, {} duplicates, {} unreadable), \
             pre-loss copy {}, current {} bytes, entries {}",
            self.origin,
            self.archives,
            self.archive_bytes,
            self.sealed,
            self.duplicates,
            self.unreadable,
            if self.prior { "yes" } else { "no" },
            self.current_bytes,
            self.entries,
        )
    }
}

/// Why an agent was not chained or opted in.
#[derive(Debug, thiserror::Error)]
pub enum Skip {
    #[error("already chained")]
    Chained,
    #[error("no state here")]
    NoState,
    #[error("a session is in progress, or an interrupted one's journal is there")]
    InSession,
    #[error("not chained yet")]
    NotChained,
    #[error("its sealed-state ledger does not say yes")]
    NotOptedIn,
    #[error("its uploads are stopped; `sealed-enroll --agent <id>` resumes them")]
    Stopped,
    #[error(transparent)]
    Failed(#[from] anyhow::Error),
}

impl Skip {
    /// A short name for the count table and the structured events.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Chained => "already_chained",
            Self::NoState => "no_state",
            Self::InSession => "in_session",
            Self::NotChained => "not_chained",
            Self::NotOptedIn => "not_opted_in",
            Self::Stopped => "stopped",
            Self::Failed(_) => "failed",
        }
    }
}

#[derive(Deserialize)]
struct JustTheName {
    soul: NameOnly,
}

#[derive(Deserialize)]
struct NameOnly {
    name: ShortString<64>,
}

fn name_of(state: &serde_json::Value) -> anyhow::Result<ShortString<64>> {
    serde_json::from_value::<JustTheName>(state.clone())
        .map(|s| s.soul.name)
        .map_err(|e| anyhow::anyhow!("the state names no agent ({:?})", e.classify()))
}

/// Why an archive is left out of an agent's chain for good. Never
/// content. An I/O error is not one: it fails the agent's chain, and the
/// next run retries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unreadable {
    NotJson,
    NotEnvelope,
    /// A format newer than this binary reads.
    Newer,
}

impl Unreadable {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotJson => "not_json",
            Self::NotEnvelope => "not_envelope",
            Self::Newer => "newer_format",
        }
    }
}

/// An archive's state, from its `{format, state}` envelope, in any format
/// this binary's FsStorage loads; `Ok(Err(_))`: unreadable for good. Its
/// content is never logged.
fn archived_state(path: &Path) -> std::io::Result<Result<serde_json::Value, Unreadable>> {
    match envelope_state(path) {
        Ok((_, state)) => Ok(Ok(state)),
        Err(EnvelopeError::Io(e)) => Err(e),
        Err(EnvelopeError::Json(_)) => Ok(Err(Unreadable::NotJson)),
        Err(EnvelopeError::Shape(_)) => Ok(Err(Unreadable::NotEnvelope)),
        Err(EnvelopeError::Newer(_)) => Ok(Err(Unreadable::Newer)),
    }
}

/// One archive left out of the chain for good.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeftOut {
    /// Its file name (`state.<utc>.json`).
    pub name: String,
    pub reason: Unreadable,
}

/// `sealed/left_out.json`: the agent's local copies its chain couldn't
/// include, by name and reason. For the `memory` tool's `log` ("N of your
/// runner's local copies couldn't be read and are not in your sealed
/// history"). Names and reasons only.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeftOutFile {
    pub archives: Vec<LeftOut>,
}

const LEFT_OUT: &str = "left_out.json";

/// What `sealed/left_out.json` records; empty when there is none.
// B7/wording step: the `memory` tool's log is to read this too
pub fn left_out(agent_dir: &Path) -> std::io::Result<Vec<LeftOut>> {
    match std::fs::read(
        agent_dir
            .join(agora_agentkit::state::storage::SEALED_DIR)
            .join(LEFT_OUT),
    ) {
        Ok(bytes) => Ok(serde_json::from_slice::<LeftOutFile>(&bytes)?.archives),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(e),
    }
}

/// `<state>/<id>/provisioned`: written by `--provision` before its first
/// save, so a chain made later (provisioning cut before its own chain) is
/// still given a genesis of class `none`.
pub const PROVISIONED: &str = "provisioned";

/// Why an FsStorage envelope (`state.json`, an archive, an `unsaved`
/// rescue) couldn't be read. Never content.
#[derive(Debug, thiserror::Error)]
pub enum EnvelopeError {
    #[error("unreadable: {0}")]
    Io(#[from] std::io::Error),
    #[error("not JSON ({0:?})")]
    Json(serde_json::error::Category),
    #[error("not an envelope: no {0}")]
    Shape(&'static str),
    #[error("envelope format {0} is newer than this binary reads")]
    Newer(u64),
}

/// The `state` of an FsStorage `{format, state}` envelope at `path`, and
/// its format: any format this binary's FsStorage loads (a chained agent's
/// is agentkit's `SEALED_FORMAT`).
pub fn envelope_state(path: &Path) -> Result<(u64, serde_json::Value), EnvelopeError> {
    let bytes = zeroize::Zeroizing::new(std::fs::read(path)?);
    let mut envelope: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| EnvelopeError::Json(e.classify()))?;
    let format = envelope
        .get("format")
        .and_then(serde_json::Value::as_u64)
        .ok_or(EnvelopeError::Shape("format"))?;
    if format > agora_agentkit::reactor::storage::MAX_FORMAT {
        return Err(EnvelopeError::Newer(format));
    }
    let state = envelope
        .get_mut("state")
        .ok_or(EnvelopeError::Shape("state"))?
        .take();
    Ok((format, state))
}

/// The sizes of a version just sealed, in the local sidecar.
pub fn record_sizes(
    agent_dir: &Path,
    head: agora_agentkit::state::verify::HeadRef,
    state_at: DateTime<Utc>,
    state: &serde_json::Value,
) {
    let r = super::sizes::Recorded {
        entry_hash: head.entry_hash,
        seq: head.seq,
        state_at,
        sizes: super::sizes::Sizes::of(state),
    };
    if let Err(e) = super::sizes::SizeLog::record(agent_dir, r) {
        tracing::warn!(path = %agent_dir.display(), error = %e, "sealed sizes not recorded");
    }
}

fn mtime(path: &Path) -> std::io::Result<DateTime<Utc>> {
    Ok(DateTime::<Utc>::from(std::fs::metadata(path)?.modified()?))
}

/// What [`Enroller::opt_in`] did: counts only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OptedIn {
    /// It was marked uploading already: nothing done.
    pub already: bool,
    /// The state key record was posted now (`false`: an earlier attempt's
    /// was on Agora).
    pub posted: bool,
    /// Entries Agora already held, filed as sent here.
    pub held_by_agora: usize,
}

/// `sealed/resuming.json`: a resume's intent, written before it sets
/// anything aside, so a rerun after a crash finishes the same resume
/// rather than setting its own new snapshot aside. Renamed
/// `resuming.done.<utc>.json` at the end. Hashes and counts only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Resuming {
    at: DateTime<Utc>,
    /// The stop it resumes (`sealed/stopped`'s `at`): an intent for
    /// another stop is stale.
    stop_at: DateTime<Utc>,
    /// The last entry Agora holds (the newest in `sent/`).
    onto: HeadRef,
    /// The versions saved while stopped, set aside.
    aside: Vec<Sha256Hex>,
}

const RESUMING: &str = "resuming.json";

fn resuming_path(agent_dir: &Path) -> PathBuf {
    agent_dir
        .join(agora_agentkit::state::storage::SEALED_DIR)
        .join(RESUMING)
}

impl Enroller {
    fn data_dir(&self) -> anyhow::Result<&Path> {
        self.state_dir
            .parent()
            .context("the state directory has no parent data directory")
    }

    /// Every agent with a directory in `state/`, by id.
    pub fn agents(&self) -> anyhow::Result<Vec<AgentId>> {
        let mut out: Vec<AgentId> = std::fs::read_dir(&self.state_dir)
            .with_context(|| format!("reading {}", self.state_dir.display()))?
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_name().to_str()?.parse::<AgentId>().ok())
            .collect();
        out.sort();
        Ok(out)
    }

    /// The agents whose sealed-state ledger says yes and who don't upload
    /// yet (no uploading marker, or one cut before its secrets copy), and
    /// aren't stopped: chained or not.
    pub fn opted_in(&self) -> anyhow::Result<Vec<AgentId>> {
        let mut out = Vec::new();
        for id in self.agents()? {
            let dir = self.state_dir.join(id.to_string());
            if super::stop::is_stopped(&dir) {
                continue;
            }
            if matches!(
                super::uploading::state(&dir, &self.keys, id),
                super::uploading::Marker::Uploading(_)
            ) {
                continue;
            }
            match SealLedger::load_blocking(&dir) {
                Ok(l) if l.opted_in() => out.push(id),
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(agent_id = %id, error = %e, "sealed-state ledger unreadable; not opted in")
                }
            }
        }
        Ok(out)
    }

    /// Chain `id` locally from the memory-restore record, or with `dry_run`
    /// say what would be done. An agent already chained gets only its
    /// pre-loss copy, if it chose to see it since ([`Self::append_prior`]).
    pub async fn chain(&self, id: AgentId, dry_run: bool) -> Result<Report, Skip> {
        self.chain_as(id, dry_run, Genesis::FromRecord).await
    }

    /// [`Self::chain`], with the genesis `genesis` records. No server
    /// call of any kind.
    pub async fn chain_as(
        &self,
        id: AgentId,
        dry_run: bool,
        genesis: Genesis,
    ) -> Result<Report, Skip> {
        let agent_dir = self.state_dir.join(id.to_string());
        let seal = AgentSeal::new(&agent_dir, id);
        if seal
            .is_enrolled()
            .await
            .context("reading the enrolment marker")?
        {
            if dry_run {
                return Err(Skip::Chained);
            }
            return match self.append_prior(id).await? {
                Some(report) => Ok(report),
                None => Err(Skip::Chained),
            };
        }
        let storage = FsStorage::new(&self.state_dir);
        let current = match storage.load_raw(id).await {
            Ok(loaded) => loaded.value,
            Err(FsStorageError::NotFound(_)) => return Err(Skip::NoState),
            Err(e) => {
                return Err(Skip::Failed(
                    anyhow::Error::new(e).context("the agent's state won't load"),
                ));
            }
        };
        let origin = match genesis {
            Genesis::New => Origin {
                origin: GenesisOrigin {
                    class: GenesisClass::None,
                    incident: None,
                    prior_copy_held: false,
                    verified_at: None,
                },
                prior: None,
            },
            Genesis::FromRecord
                if std::fs::exists(agent_dir.join(PROVISIONED)).unwrap_or(false) =>
            {
                Origin {
                    origin: GenesisOrigin {
                        class: GenesisClass::None,
                        incident: None,
                        prior_copy_held: false,
                        verified_at: None,
                    },
                    prior: None,
                }
            }
            Genesis::FromRecord => match self.restore.as_ref() {
                // The name matters only to the memory-restore listing
                Some(offer) => origin(Some(offer), &name_of(&current)?, id, &agent_dir).await?,
                None => Origin {
                    origin: GenesisOrigin {
                        class: GenesisClass::Unknown,
                        incident: None,
                        prior_copy_held: false,
                        verified_at: None,
                    },
                    prior: None,
                },
            },
        };
        let archives = archives(&agent_dir).context("listing archives")?;
        let current_path = agent_dir.join("state.json");
        let mut report = Report {
            archives: archives.len(),
            archive_bytes: archives.iter().map(|a| a.bytes).sum(),
            prior: origin.prior.is_some(),
            origin: format!("{:?}", origin.origin.class).to_lowercase(),
            current_bytes: std::fs::metadata(&current_path)
                .map(|m| m.len())
                .unwrap_or(0),
            ..Report::default()
        };
        if super::session::no_session(&self.state_dir, id, false).is_err() {
            return Err(Skip::InSession);
        }
        if dry_run {
            return Ok(report);
        }
        if self.keys.signing(id).is_none() {
            return Err(Skip::Failed(anyhow::anyhow!(
                "the agent has no signing key"
            )));
        }
        // No session in progress, and the agent's lock for the whole chain:
        // every write below goes through it
        let lock = super::session::take(&self.state_dir, id, false)
            .await
            .map_err(|_| Skip::InSession)?;
        // A chain begun before but never finished is resumed
        let partial = seal.head().await.context("reading the sealed head")?;
        // The state key, cached here; nothing of it leaves the machine
        // until the agent opts in
        if self
            .keys
            .state_key_file(id)
            .context("the state key cache")?
            .is_none()
        {
            self.keys
                .write_state_key(id, &StateKeyFile::new(&StateKey::generate(), 0, 0))
                .context("caching the state key")?;
        }
        let keys = self
            .keys
            .seal_keys(id)
            .context("sealing keys for the agent")?;
        let policy = KeyframePolicy::default();

        // Where a partial chain stopped: its snapshots in order, replayed
        // against the archives (the first run's own steps, so a state
        // repeated later, or two archives with one mtime, are never lost),
        // and whether the pre-loss copy is in
        let mut sealed_before: Vec<Sha256Hex> = Vec::new();
        let mut prior_done = false;
        match &partial {
            Some(_) => {
                let mut items = seal.sent().await.context("reading sent/")?;
                items.extend(seal.outbox().await.context("reading the outbox")?);
                items.sort_by_key(|i| i.entry.entry.seq);
                sealed_before = items
                    .iter()
                    .filter(|i| i.entry.entry.kind == StateEntryKind::Snapshot)
                    .filter_map(|i| i.plaintext_sha256)
                    .collect();
                prior_done = items
                    .iter()
                    .any(|i| i.entry.entry.kind == StateEntryKind::ArchivedPrior);
                report.resumed = true;
                tracing::warn!(
                    event_type = "sealed_chain_resumed",
                    agent_id = %id,
                    entries = items.len(),
                    "resuming a partial chain"
                );
            }
            None => {
                let first_at = archives
                    .first()
                    .map(|a| a.mtime)
                    .or_else(|| mtime(&current_path).ok())
                    .unwrap_or_else(Utc::now);
                super::session::genesis(&seal, &lock, &keys, &origin.origin, first_at)
                    .await
                    .context("sealing the genesis")?;
            }
        }
        super::crash::point("chain:genesis")?;
        if let Some(prior) = origin.prior.as_ref().filter(|_| !prior_done) {
            super::session::append(
                &seal,
                &lock,
                &keys,
                &policy,
                StateEntryKind::ArchivedPrior,
                &read_prior(prior)?,
                prior_copy_at(),
            )
            .await
            .context("sealing the pre-loss copy")?;
        }
        let mut replayed = 0;
        let mut last = None;
        let mut left_out = Vec::new();
        for archive in &archives {
            let name = archive
                .path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            // An I/O error fails the agent (the next run retries); what
            // can never be read is left out, said and recorded
            let state = match archived_state(&archive.path)
                .with_context(|| format!("reading archive {name}"))?
            {
                Ok(state) => state,
                Err(reason) => {
                    tracing::warn!(
                        event_type = "sealed_chain_archive_unreadable",
                        agent_id = %id,
                        file = %name,
                        reason = reason.as_str(),
                        "an archive can't be read and is left out of the agent's chain"
                    );
                    report.unreadable += 1;
                    left_out.push(LeftOut { name, reason });
                    continue;
                }
            };
            let hash = plaintext_hash(&state);
            if last == Some(hash) {
                report.duplicates += 1;
                continue;
            }
            last = Some(hash);
            if sealed_before.get(replayed) == Some(&hash) {
                replayed += 1;
                report.resumed_past += 1;
                continue;
            }
            let head = super::session::append(
                &seal,
                &lock,
                &keys,
                &policy,
                StateEntryKind::Snapshot,
                &state,
                archive.mtime,
            )
            .await
            .with_context(|| format!("sealing archive {name}"))?;
            record_sizes(&agent_dir, head, archive.mtime, &state);
            report.sealed += 1;
            super::crash::point("chain:archive")?;
        }
        let current_hash = plaintext_hash(&current);
        let current_done =
            last == Some(current_hash) || sealed_before.get(replayed) == Some(&current_hash);
        if !current_done {
            let at = mtime(&current_path).unwrap_or_else(|_| Utc::now());
            let head = super::session::append(
                &seal,
                &lock,
                &keys,
                &policy,
                StateEntryKind::Snapshot,
                &current,
                at,
            )
            .await
            .context("sealing the current state")?;
            record_sizes(&agent_dir, head, at, &current);
        }
        if !left_out.is_empty() {
            // With what an earlier, cut run recorded, each name once
            let mut all = self::left_out(&agent_dir).context("reading sealed/left_out.json")?;
            all.retain(|o| !left_out.iter().any(|n| n.name == o.name));
            all.extend(left_out);
            let left_out = all;
            super::write_private(
                &seal.dir().join(LEFT_OUT),
                &serde_json::to_vec_pretty(&LeftOutFile { archives: left_out })
                    .context("the left-out record")?,
            )
            .context("recording the archives left out")?;
        }
        super::crash::point("chain:current")?;
        // The last step is agentkit's, which takes the lock itself: the
        // state rewritten in agentkit's sealed format (an older runner then
        // refuses it rather than overwriting it), resealed if the rewrite
        // changed its hash, then the chain marker
        drop(lock);
        let head = self.mark(id).await?;
        if let Ok(marked) = FsStorage::new(&self.state_dir).load_raw(id).await {
            record_sizes(&agent_dir, head, Utc::now(), &marked.value);
        }
        self.roster(id, head).await;
        report.entries = super::check::local_entries(&seal)
            .await
            .map_err(anyhow::Error::msg)?
            .len();
        tracing::info!(
            event_type = "sealed_chained",
            agent_id = %id,
            archives = report.archives,
            archive_bytes = report.archive_bytes,
            sealed = report.sealed,
            duplicates = report.duplicates,
            unreadable = report.unreadable,
            prior = report.prior,
            origin = %report.origin,
            entries = report.entries,
            resumed = report.resumed,
            "agent chained locally; nothing sent"
        );
        Ok(report)
    }

    /// agentkit's last step of a chain: the state marked in its sealed
    /// format, the head aligned with it, the chain marker. Takes the agent's lock itself.
    async fn mark(&self, id: AgentId) -> anyhow::Result<HeadRef> {
        let mut mirrored = agora_agentkit::state::storage::Mirrored::new(
            FsStorage::new(&self.state_dir),
            (),
            Arc::new(super::SealedEnrolment::with_keys(
                &self.state_dir,
                self.keys.clone(),
            )),
        );
        mirrored
            .enroll(id)
            .await
            .context("marking the state and writing the enrolment marker")
    }

    /// The roster outside `state/` learns the agent's head. A failure is
    /// logged: the next run's end writes it again.
    async fn roster(&self, id: AgentId, head: HeadRef) {
        let recorded = match self.data_dir() {
            Ok(d) => super::roster::record(d, id, head)
                .await
                .map_err(anyhow::Error::from),
            Err(e) => Err(e),
        };
        if let Err(e) = recorded {
            tracing::error!(
                event_type = "sealed_roster_not_recorded",
                agent_id = %id,
                error = %e,
                "the agent's head couldn't be recorded in the roster"
            );
        }
    }

    /// A chained agent that chose to see its pre-loss copy after it was
    /// chained: the copy appended to its chain as `archived_prior`, then
    /// its current state again as a snapshot (the start check compares the
    /// state with the head's snapshot), and the agent told (with "and sent
    /// to Agora with it" for an uploader, whose copy is uploaded). Once:
    /// never if the chain holds an `archived_prior` already; a pair cut
    /// short is finished first ([`Self::finish_prior`]). Its genesis, sealed
    /// with what was true then, never changes: that the copy came later is
    /// read from the chain. Under the agent's `sealed/lock` from the first
    /// read, with no session in progress; never over a state that doesn't
    /// match its last sealed version. Not for a stopped agent: what it
    /// seals while stopped would be set aside at its resume as saved while
    /// stopped, so the copy waits for the resume. `None`: nothing to do.
    pub async fn append_prior(&self, id: AgentId) -> anyhow::Result<Option<Report>> {
        if self.finish_prior(id).await?.is_some() {
            return Ok(Some(Report {
                prior: true,
                prior_later: true,
                ..Report::default()
            }));
        }
        let Some(offer) = self.restore.as_ref() else {
            return Ok(None);
        };
        let agent_dir = self.state_dir.join(id.to_string());
        let seal = AgentSeal::new(&agent_dir, id);
        if !seal.is_enrolled().await? {
            return Ok(None);
        }
        if super::stop::is_stopped(&agent_dir)
            || super::stop::durable(&self.keys, id).map_or(true, |d| d.is_some())
        {
            return Ok(None);
        }
        let lock = super::session::take(&self.state_dir, id, false).await?;
        let current = FsStorage::new(&self.state_dir)
            .load_raw(id)
            .await
            .context("the agent's state won't load")?
            .value;
        let Some(prior) = origin(Some(offer), &name_of(&current)?, id, &agent_dir)
            .await?
            .prior
        else {
            return Ok(None);
        };
        let mut items = seal.sent().await?;
        items.extend(seal.outbox().await?);
        if items
            .iter()
            .any(|i| i.entry.entry.kind == StateEntryKind::ArchivedPrior)
        {
            return Ok(None);
        }
        let head = seal.head().await?.context("no sealed head")?;
        anyhow::ensure!(
            head.plaintext_sha256 == Some(plaintext_hash(&current)),
            "agent {id}'s state doesn't match its last sealed version; its pre-loss copy is not \
             appended until the start check's finding is resolved"
        );
        let keys = self
            .keys
            .seal_keys(id)
            .context("sealing keys for the agent")?;
        let policy = KeyframePolicy::default();
        // Told first, in one write: a cut after it is finished by
        // `finish_prior`, so the notice is never false for long
        let sent = super::uploads(&agent_dir, &self.keys, id);
        super::notice::Notice::operator_act(
            &agent_dir,
            "",
            super::notice::OperatorAct::PriorAdded { sent },
        )
        .context("writing the notice")?;
        super::session::append(
            &seal,
            &lock,
            &keys,
            &policy,
            StateEntryKind::ArchivedPrior,
            &read_prior(&prior)?,
            prior_copy_at(),
        )
        .await
        .context("sealing the pre-loss copy")?;
        super::crash::point("prior:appended")?;
        drop(lock);
        let head = self
            .finish_prior(id)
            .await?
            .context("the pre-loss copy was sealed but isn't the head")?;
        tracing::warn!(
            event_type = "sealed_prior_appended",
            agent_id = %id,
            head = %head.entry_hash,
            seq = head.seq,
            sent,
            "the pre-loss copy the agent chose to see appended to its chain"
        );
        Ok(Some(Report {
            prior: true,
            prior_later: true,
            entries: super::check::local_entries(&seal)
                .await
                .map_err(anyhow::Error::msg)?
                .len(),
            ..Report::default()
        }))
    }

    /// A chain whose head is an `archived_prior` (a pre-loss copy appended,
    /// cut before the snapshot after it): the current state sealed after it,
    /// if it is the last snapshot's, so the start check finds the agent
    /// intact rather than refusing it. Needs no memory-restore config.
    /// `Some(head)`: finished now.
    // B2/B6: call before the start check, before reactors are built
    // (`main.rs`), with an `Enroller` over the run's state and keys
    pub async fn finish_prior(&self, id: AgentId) -> anyhow::Result<Option<HeadRef>> {
        let agent_dir = self.state_dir.join(id.to_string());
        let seal = AgentSeal::new(&agent_dir, id);
        if !seal.is_enrolled().await? {
            return Ok(None);
        }
        let Some(head) = seal.head().await? else {
            return Ok(None);
        };
        let mut items = seal.sent().await?;
        items.extend(seal.outbox().await?);
        let head_kind = items
            .iter()
            .find(|i| i.entry.entry_hash() == head.head.entry_hash)
            .map(|i| i.entry.entry.kind);
        if head_kind != Some(StateEntryKind::ArchivedPrior) {
            return Ok(None);
        }
        let lock = super::session::take(&self.state_dir, id, false).await?;
        let current = FsStorage::new(&self.state_dir)
            .load_raw(id)
            .await
            .context("the agent's state won't load")?
            .value;
        let last_snapshot = items
            .iter()
            .filter(|i| i.entry.entry.kind == StateEntryKind::Snapshot)
            .max_by_key(|i| i.entry.entry.seq)
            .and_then(|i| i.plaintext_sha256);
        anyhow::ensure!(
            last_snapshot == Some(plaintext_hash(&current)),
            "agent {id}'s state doesn't match its last sealed version; the pre-loss copy's chain \
             is left for the start check"
        );
        let keys = self
            .keys
            .seal_keys(id)
            .context("sealing keys for the agent")?;
        let now = Utc::now();
        let head = super::session::append(
            &seal,
            &lock,
            &keys,
            &KeyframePolicy::default(),
            StateEntryKind::Snapshot,
            &current,
            now,
        )
        .await
        .context("sealing the current state after the pre-loss copy")?;
        drop(lock);
        record_sizes(&agent_dir, head, now, &current);
        self.roster(id, head).await;
        Ok(Some(head))
    }

    /// Chain `ids`, up to `jobs` at once (each seal is two brotli
    /// compressions, off the async threads). The results in `ids`' order;
    /// one structured event per agent as it finishes.
    pub async fn chain_all(
        self: &Arc<Self>,
        ids: &[AgentId],
        dry_run: bool,
        jobs: usize,
    ) -> Vec<(AgentId, Result<Report, Skip>)> {
        let permits = Arc::new(tokio::sync::Semaphore::new(jobs.max(1)));
        let mut set = tokio::task::JoinSet::new();
        for (n, &id) in ids.iter().enumerate() {
            let this = self.clone();
            let permits = permits.clone();
            set.spawn(async move {
                let _permit = permits.acquire_owned().await.expect("never closed");
                (n, this.chain(id, dry_run).await)
            });
        }
        let total = ids.len();
        let mut out: Vec<Option<Result<Report, Skip>>> = (0..total).map(|_| None).collect();
        let mut finished = 0;
        while let Some(joined) = set.join_next().await {
            let (n, result) = match joined {
                Ok(done) => done,
                Err(e) => {
                    tracing::error!(error = %e, "a chain task panicked or was cancelled");
                    continue;
                }
            };
            finished += 1;
            tracing::info!(
                event_type = "sealed_chain_progress",
                agent_id = %ids[n],
                outcome = result.as_ref().map_or_else(Skip::as_str, |_| "chained"),
                done = finished,
                total,
                dry_run,
                "chain progress"
            );
            out[n] = Some(result);
        }
        ids.iter()
            .zip(out)
            .map(|(&id, r)| {
                let r = r.unwrap_or_else(|| {
                    Err(Skip::Failed(anyhow::anyhow!(
                        "its chain task didn't finish"
                    )))
                });
                (id, r)
            })
            .collect()
    }

    /// Start uploading a chained agent whose latest sealed-state answer is
    /// yes. In order, each step finished by a rerun after a cut:
    ///
    /// 1. refused while a key rotation is in progress (`secrets.next.json`:
    ///    the local chain may be half re-signed);
    /// 2. under `guard`'s opt-in hold, which lets only the opt-in calls
    ///    through for it, Agora's chain of it must be empty or the start of
    ///    this one;
    /// 3. its state key record is posted (once: an earlier one is found),
    ///    so it is on Agora before any entry;
    /// 4. the agent's notice, in one write: uploading started, as it
    ///    answered (or against its `decline`, with `against`'s note);
    /// 5. it is marked uploading, which commits it;
    /// 6. the entries Agora already holds are filed as sent (also on an
    ///    agent already uploading). The caller drains next.
    ///
    /// Never against its latest answer without `against` (a note the agent
    /// is shown), and with no answer on file never.
    pub async fn opt_in<A, E>(
        &self,
        guard: &UploadGuard<A, E>,
        id: AgentId,
        against: Option<&str>,
    ) -> Result<OptedIn, Skip>
    where
        A: SealedApi,
        E: super::api::UploadGate,
    {
        use super::uploading::{self, Marker};
        let agent_dir = self.state_dir.join(id.to_string());
        let seal = AgentSeal::new(&agent_dir, id);
        if !seal
            .is_enrolled()
            .await
            .context("reading the enrolment marker")?
        {
            return Err(Skip::NotChained);
        }
        if super::stop::is_stopped(&agent_dir)
            || super::stop::durable(&self.keys, id).map_or(true, |d| d.is_some())
        {
            return Err(Skip::Stopped);
        }
        let ledger = SealLedger::load(&agent_dir)
            .await
            .context("sealed-state ledger unreadable")?;
        let against = against.map(str::trim).filter(|n| !n.is_empty());
        let act = match (ledger.opted_in(), ledger.allows(true), against) {
            (true, _, _) => super::notice::OperatorAct::OptedIn {
                answered: ledger.answered_at().context("a yes with no date")?,
            },
            (false, Err(_), Some(_)) => super::notice::OperatorAct::OptedInAgainst,
            _ => return Err(Skip::NotOptedIn),
        };
        let signing = self
            .keys
            .signing(id)
            .context("the agent has no signing key")?;
        let durable = self.keys.agent_dir(id).join(uploading::DURABLE);
        match uploading::state(&agent_dir, &self.keys, id) {
            Marker::Uploading(_) => {
                let held_by_agora = self.file_as_sent(guard, &seal, id, &signing).await?;
                return Ok(OptedIn {
                    already: true,
                    held_by_agora,
                    ..OptedIn::default()
                });
            }
            Marker::Absent => {}
            // Only the local copy: a `set` cut before its secrets copy,
            // finished below
            Marker::Contradictory(_) if !std::fs::exists(&durable).unwrap_or(true) => {}
            Marker::Contradictory(why) => {
                return Err(Skip::Failed(anyhow::anyhow!(
                    "its uploading marker is in doubt ({why}); nothing was sent"
                )));
            }
            Marker::Unreadable(e) => {
                return Err(Skip::Failed(anyhow::anyhow!(
                    "its uploading marker is unreadable ({e}); nothing was sent"
                )));
            }
        }
        if self
            .keys
            .next_secrets(id)
            .context("reading secrets.next.json")?
            .is_some()
        {
            return Err(Skip::Failed(anyhow::anyhow!(
                "a key rotation is in progress (secrets.next.json): finish it (`rotate-keys`) \
                 first; nothing was sent"
            )));
        }
        if super::session::no_session(&self.state_dir, id, false).is_err() {
            return Err(Skip::InSession);
        }
        let head = seal
            .head()
            .await
            .context("reading the sealed head")?
            .context("chained, but its sealed/head.json is missing")?;
        let entries = super::check::local_entries(&seal)
            .await
            .map_err(anyhow::Error::msg)?;
        let line = super::check::main_line(&entries, &head.head.entry_hash)
            .context("its local chain doesn't run from the genesis to its head")?;

        let hold = guard.opting_in(id);
        // Agora holds none of it, or the start of this same chain
        let server = super::api::chain(guard, id, &signing)
            .await
            .map_err(anyhow::Error::msg)
            .context("reading the agent's chain on Agora")?;
        let prefix = server.len() <= line.len()
            && server
                .iter()
                .zip(&line)
                .all(|(s, l)| s.entry_hash() == l.entry_hash());
        if !prefix {
            return Err(Skip::Failed(anyhow::anyhow!(
                "Agora holds a chain of {} entries for this agent that isn't the start of the one \
                 here: not opted in, nothing sent",
                server.len()
            )));
        }

        // The state key record, wrapped to the agent's X25519 key, signed
        // by its current key, at the cached wrap_seq
        let encryption = self
            .keys
            .ring()
            .ensure_encryption_key(id)
            .context("the agent's encryption key")?;
        let state_file = self
            .keys
            .state_key_file(id)
            .context("the state key cache")?
            .context("chained, but no state key is cached")?;
        let vk = signing.verifying_key();
        // One rule with `rotate-keys` (`keys::wrap_usable`): at state_key.json's
        // wrap or past it
        let valid = |r: &agora_agentkit::state::StateKeyRecord| {
            super::keys::wrap_usable(r, id, &state_file, state_file.wrap_seq, &encryption, &vk)
        };
        let held = guard
            .state_keys(id, &signing)
            .await
            .map_err(anyhow::Error::msg)
            .context("reading the server's state keys")?;
        // An earlier attempt's record on Agora; one wrapped to a retired key,
        // or signed by one, doesn't count: Agora takes only the current key's
        let posted = held.iter().find(|r| valid(r)).cloned();
        let record = match posted {
            Some(r) => r,
            None => {
                // A rotation's own record, or an earlier attempt's, if it is
                // still valid; else wrapped afresh past what Agora holds
                let record = match super::keys::kept_wrap(
                    &self.keys,
                    id,
                    &state_file,
                    state_file.wrap_seq,
                    &encryption,
                    &vk,
                )
                .context("reading state_key.wrap.json")?
                {
                    Some(r) => r,
                    None => super::keys::new_wrap(
                        &self.keys,
                        id,
                        &state_file,
                        state_file.wrap_seq,
                        &encryption,
                        &signing,
                        &held,
                    )?,
                };
                guard
                    .post_state_key(id, &signing, &record)
                    .await
                    .map_err(anyhow::Error::msg)
                    .context("posting the state key record")?;
                record
            }
        };
        // state_key.json names the wrap on Agora
        if record.wrap_seq != state_file.wrap_seq {
            self.keys
                .write_state_key(
                    id,
                    &super::keys::StateKeyFile::new(
                        &state_file.state_key().context("the state key cache")?,
                        state_file.key_gen,
                        record.wrap_seq,
                    ),
                )
                .context("naming the posted wrap in state_key.json")?;
        }
        // Already on Agora before this run
        let posted = held.contains(&record);
        super::crash::point("opt_in:posted")?;
        // The agent's notice before the marker: a cut between them is
        // finished by a rerun, which writes it again
        super::notice::Notice::operator_act(&agent_dir, against.unwrap_or_default(), act.clone())
            .context("writing the notice")?;
        super::crash::point("opt_in:noticed")?;
        // The record is on Agora: only now may an entry follow it
        uploading::set(&agent_dir, &self.keys, id, uploading::UploadingBy::OptIn)
            .context("marking it uploading")?;
        drop(hold);
        super::crash::point("opt_in:marked")?;
        let held_by_agora = self.file_as_sent(guard, &seal, id, &signing).await?;
        if act == super::notice::OperatorAct::OptedInAgainst {
            tracing::warn!(
                event_type = "sealed_opted_in_against_answer",
                agent_id = %id,
                note_chars = against.map_or(0, |n| n.chars().count()),
                "the operator opted in an agent whose latest answer is decline"
            );
        }
        tracing::warn!(
            event_type = "sealed_opted_in",
            agent_id = %id,
            posted = !posted,
            held_by_agora,
            "a chained agent now uploads; its whole local chain follows"
        );
        Ok(OptedIn {
            already: false,
            posted: !posted,
            held_by_agora,
        })
    }

    /// The outbox entries Agora already holds (the same entry hash), filed
    /// as sent, so the drain doesn't send them again. Nothing is deleted:
    /// filing moves the record to `sent/`, and agentkit keeps acknowledged
    /// blobs. The agent must be allowed reads (opting in, or uploading).
    async fn file_as_sent<A, E>(
        &self,
        guard: &UploadGuard<A, E>,
        seal: &AgentSeal,
        id: AgentId,
        signing: &agora_agentkit::crypto::SigningKey,
    ) -> anyhow::Result<usize>
    where
        A: SealedApi,
        E: super::api::UploadGate,
    {
        let waiting = seal.outbox().await.context("reading the outbox")?;
        if waiting.is_empty() {
            return Ok(0);
        }
        let on_agora: std::collections::HashSet<Sha256Hex> = super::api::chain(guard, id, signing)
            .await
            .map_err(anyhow::Error::msg)
            .context("reading the agent's chain on Agora")?
            .iter()
            .map(|e| e.entry_hash())
            .collect();
        let mut filed = 0;
        for item in waiting {
            if on_agora.contains(&item.entry.entry_hash()) {
                seal.mark_sent(item.ord)
                    .await
                    .context("filing an entry Agora holds as sent")?;
                filed += 1;
            }
        }
        Ok(filed)
    }

    /// Resume a stopped agent's uploads (`sealed-enroll --resume`): never a
    /// side effect of chaining. An agent marked uploading resumes only if
    /// its latest answer is yes, or with `against` (a note it is shown)
    /// over its `decline`; an unreadable or missing ledger refuses. The
    /// uploaded line goes on from the last entry Agora holds (the newest
    /// in `sent/`: a stop drains first) with **one new snapshot of the
    /// current state**; the versions saved while stopped are set aside here
    /// as `.displaced` (kept, with their blobs, opened by the `memory` tool
    /// as a side line) and never uploaded. An agent not marked uploading
    /// has its stop cleared and nothing else. The agent is always told
    /// (one notice). Never a new genesis.
    ///
    /// Crash-safe: `sealed/resuming.json`, naming the stop it resumes, is
    /// written before anything is set aside and renamed done before the
    /// stop is cleared; an intent for another stop is set aside.
    ///
    /// The caller holds the runner lock exclusively. A state that changed
    /// outside the store while stopped needs `note`.
    pub async fn resume(
        &self,
        id: AgentId,
        note: Option<&str>,
        against: bool,
    ) -> anyhow::Result<Report> {
        let note = note.map(str::trim).filter(|n| !n.is_empty());
        let agent_dir = self.state_dir.join(id.to_string());
        let seal = AgentSeal::new(&agent_dir, id);
        anyhow::ensure!(
            seal.is_enrolled().await?,
            "agent {id} is not chained: nothing to resume"
        );
        // A stop kept in the secrets tree stands for a lost sealed/stopped
        super::stop::reconcile(&agent_dir, &self.keys, id)
            .context("putting back the stop kept in the secrets tree")?;
        let stop = super::stop::Stopped::load(&agent_dir)
            .context("sealed/stopped unreadable")?
            .with_context(|| format!("agent {id} is not stopped: nothing to resume"))?;
        let uploading = super::uploading::is_uploading(&agent_dir, &self.keys, id);
        let mut overridden = false;
        if uploading {
            anyhow::ensure!(
                std::fs::exists(SealLedger::path(&agent_dir))?,
                "agent {id} has no sealed-state ledger: its answer can't be read, so its uploads \
                 aren't resumed"
            );
            let ledger = SealLedger::load(&agent_dir)
                .await
                .context("the sealed-state ledger is unreadable: uploads not resumed")?;
            if !ledger.opted_in() {
                anyhow::ensure!(
                    against && note.is_some() && ledger.allows(true).is_err(),
                    "agent {id}'s latest answer is not yes: uploads not resumed. Re-open the \
                     question (`sealed-reoffer`), or pass --against-answer with a --note the \
                     agent is shown"
                );
                overridden = true;
            }
        }
        let lock = super::session::take(&self.state_dir, id, false).await?;
        let head = seal.head().await?.with_context(|| {
            format!(
                "agent {id} is stopped but its sealed/head.json is missing: nothing to resume onto"
            )
        })?;
        let keys = self
            .keys
            .seal_keys(id)
            .context("sealing keys for the agent")?;
        let current = FsStorage::new(&self.state_dir)
            .load_raw(id)
            .await
            .context("the agent's state won't load")?
            .value;
        let (intent, done) = self.resume_intent(&agent_dir, id, &stop)?;
        let waiting = seal.outbox().await?;
        // Changed outside the store while stopped (a resume cut short has
        // sealed it already)
        let changed = intent.is_none() && head.plaintext_sha256 != Some(plaintext_hash(&current));
        anyhow::ensure!(
            !changed || note.is_some(),
            "agent {id}'s state changed while its backups were stopped: pass --note \"<why>\" \
             to resume over it (the agent is shown the note)"
        );
        let mut kept_local = 0;
        if intent.is_some() || (uploading && !waiting.is_empty()) {
            let intent = match intent {
                Some(i) => i,
                None => {
                    let onto = seal
                        .sent()
                        .await?
                        .into_iter()
                        .max_by_key(|i| i.entry.entry.seq)
                        .with_context(|| {
                            format!(
                                "agent {id} has versions saved while stopped but nothing in \
                                 sent/: nothing of it is known to be on Agora to continue from; \
                                 nothing was changed"
                            )
                        })?;
                    let intent = Resuming {
                        at: Utc::now(),
                        stop_at: stop.at,
                        onto: HeadRef {
                            entry_hash: onto.entry.entry_hash(),
                            seq: onto.entry.entry.seq,
                        },
                        aside: waiting.iter().map(|i| i.entry.entry_hash()).collect(),
                    };
                    super::write_private(
                        &resuming_path(&agent_dir),
                        &serde_json::to_vec_pretty(&intent)?,
                    )?;
                    intent
                }
            };
            super::crash::point("resume:intent")?;
            kept_local = intent.aside.len();
            // Done when the head is a new snapshot on the last entry Agora
            // holds, not one of the versions set aside
            let sealed = seal.head().await?.filter(|h| {
                h.head.seq == intent.onto.seq + 1
                    && !intent.aside.contains(&h.head.entry_hash)
                    && h.keyframe.as_ref().is_some_and(|k| {
                        k.entry_hash() == h.head.entry_hash
                            && k.entry.prev_hash == Some(intent.onto.entry_hash)
                    })
            });
            if sealed.is_none() && !done {
                let onto = seal
                    .sent()
                    .await?
                    .into_iter()
                    .find(|i| i.entry.entry_hash() == intent.onto.entry_hash)
                    .context("the entry the resume began on is no longer in sent/")?;
                let head = seal
                    .rebase_onto(&lock, &keys, &onto.entry.entry, &current)
                    .await
                    .context("sealing the current state on the last entry Agora holds")?;
                record_sizes(&agent_dir, head, Utc::now(), &current);
            }
        } else if changed {
            let now = Utc::now();
            let head = super::session::append(
                &seal,
                &lock,
                &keys,
                &KeyframePolicy::default(),
                StateEntryKind::Snapshot,
                &current,
                now,
            )
            .await
            .context("sealing the current state on the existing chain")?;
            record_sizes(&agent_dir, head, now, &current);
        }
        super::crash::point("resume:sealed")?;
        drop(lock);
        // Marked in the sealed format again (agentkit takes the lock itself)
        let marked = self
            .mark(id)
            .await
            .context("marking the state as sealed again")?;
        self.roster(id, marked).await;
        // Always told, in one write, before the stop is cleared
        super::notice::Notice::operator_act(
            &agent_dir,
            note.unwrap_or_default(),
            super::notice::OperatorAct::Resumed {
                uploading,
                kept: kept_local,
                against: overridden,
            },
        )
        .context("writing the notice")?;
        // The intent done before the stop goes: a cut after this finds it
        // done (for this stop) and seals nothing again
        if std::fs::exists(resuming_path(&agent_dir))? {
            let at = Utc::now().format(super::STAMP_FORMAT);
            std::fs::rename(
                resuming_path(&agent_dir),
                resuming_path(&agent_dir).with_file_name(format!("resuming.done.{at}.json")),
            )?;
        }
        super::crash::point("resume:done")?;
        // The durable copy first: left behind, it would put the stop back
        super::stop::clear_durable(&self.keys, id)
            .context("setting the stop in the secrets tree aside")?;
        if super::stop::is_stopped(&agent_dir) {
            super::stop::Stopped::clear(&agent_dir).context("setting sealed/stopped aside")?;
        }
        tracing::warn!(
            event_type = "sealed_state_resumed",
            agent_id = %id,
            head = %marked.entry_hash,
            seq = marked.seq,
            kept_local,
            uploading,
            against_answer = overridden,
            "a stopped agent resumed; the versions saved while stopped stay here"
        );
        Ok(Report {
            resumed: true,
            kept_local,
            entries: super::check::local_entries(&seal)
                .await
                .map_err(anyhow::Error::msg)?
                .len(),
            ..Report::default()
        })
    }

    /// The resume intent for `stop`: `resuming.json` if it names this stop
    /// (another stop's is stale, set aside and logged), else a
    /// `resuming.done.*.json` naming it (`done`: sealed already).
    fn resume_intent(
        &self,
        agent_dir: &Path,
        id: AgentId,
        stop: &super::stop::Stopped,
    ) -> anyhow::Result<(Option<Resuming>, bool)> {
        let path = resuming_path(agent_dir);
        match std::fs::read(&path) {
            Ok(bytes) => {
                let intent: Option<Resuming> = serde_json::from_slice(&bytes).ok();
                match intent {
                    Some(i) if i.stop_at == stop.at => return Ok((Some(i), false)),
                    _ => {
                        let at = Utc::now().format(super::STAMP_FORMAT);
                        std::fs::rename(
                            &path,
                            path.with_file_name(format!("resuming.stale.{at}.json")),
                        )?;
                        tracing::warn!(
                            event_type = "sealed_resume_intent_stale",
                            agent_id = %id,
                            "a resume intent for another stop (or unreadable) was set aside"
                        );
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        let dir = path.parent().expect("in sealed/");
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name();
            if !name.to_string_lossy().starts_with("resuming.done.") {
                continue;
            }
            if let Ok(i) = serde_json::from_slice::<Resuming>(&std::fs::read(entry.path())?)
                && i.stop_at == stop.at
            {
                return Ok((Some(i), true));
            }
        }
        Ok((None, false))
    }
}

/// The pre-loss copy, parsed. Its content is never logged.
fn read_prior(path: &Path) -> anyhow::Result<serde_json::Value> {
    serde_json::from_slice(&std::fs::read(path).context("reading the pre-loss copy")?)
        .map_err(|e| anyhow::anyhow!("the pre-loss copy isn't JSON ({:?})", e.classify()))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::consent::sealed_state::ledger::{SealAsk, SealOutcome};
    use crate::consent::sealed_state::prompt::SealChoice;
    use crate::sealed::api::fake;
    use agora_agentkit::reactor::seed::memory::memory_text;
    use agora_agentkit::state::storage::{Recovery, SEALED_DIR};
    use agora_agentkit::state::verify::verify_chain;

    fn answer(dir: &Path, choice: SealChoice) {
        let mut l = SealLedger::load_blocking(dir).unwrap();
        l.record(SealAsk {
            at: Utc::now(),
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
        std::fs::write(SealLedger::path(dir), serde_json::to_vec(&l).unwrap()).unwrap();
    }

    fn state(name: &str, memory: &str) -> serde_json::Value {
        serde_json::json!({ "soul": { "name": name }, "memory": { "content": memory } })
    }

    /// An agent with a signing key and `memories` saved in turn (all but
    /// the last archived), not chained.
    pub(crate) async fn agent(root: &Path, memories: &[&str]) -> (AgentId, KeyStore, PathBuf) {
        let keys = KeyStore::new(root.join("secrets"));
        let id = AgentId::new();
        keys.ring()
            .insert(id, &agora_agentkit::crypto::generate_keypair().0)
            .unwrap();
        let mut fs = FsStorage::new(root.join("state"));
        for m in memories {
            fs.save_raw(id, state("tarn", m), None).await.unwrap();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        (id, keys, root.join("state").join(id.to_string()))
    }

    pub(crate) fn enroller(root: &Path, keys: &KeyStore) -> Enroller {
        Enroller {
            state_dir: root.join("state"),
            keys: keys.clone(),
            restore: None,
        }
    }

    /// (kind, seq, state_at, plaintext hash) of every entry: what two
    /// chains of the same states share (entry hashes differ: the seal's
    /// nonces and `created_at`).
    async fn shape(dir: &Path, id: AgentId) -> Vec<(StateEntryKind, u64, i64, Option<Sha256Hex>)> {
        let seal = AgentSeal::new(dir, id);
        let mut items = seal.sent().await.unwrap();
        items.extend(seal.outbox().await.unwrap());
        items
            .iter()
            .map(|i| {
                (
                    i.entry.entry.kind,
                    i.entry.entry.seq,
                    i.entry.entry.state_at,
                    i.plaintext_sha256,
                )
            })
            .collect()
    }

    async fn opened_memories(dir: &Path, id: AgentId, keys: &KeyStore) -> Vec<String> {
        let seal = AgentSeal::new(dir, id);
        let entries = super::super::check::local_entries(&seal).await.unwrap();
        let k = keys.seal_keys(id).unwrap();
        let mut out = Vec::new();
        for e in entries
            .iter()
            .filter(|e| e.entry.kind == StateEntryKind::Snapshot)
        {
            let base = match e.entry.blob.as_ref().unwrap().encoding {
                agora_agentkit::state::BlobEncoding::Delta { base } => entries
                    .iter()
                    .map(|x| &x.entry)
                    .find(|x| x.entry_hash() == base),
                _ => None,
            };
            let v = seal.open_state(&k, &e.entry, base).await.unwrap();
            out.push(v["memory"]["content"].as_str().unwrap().to_string());
        }
        out
    }

    /// An agent keyed and searched before it enrols: `sealed-enroll
    /// --dry-run` counts the same, the key's fingerprint and the index
    /// beside its state are never taken for archives, and no `sealed/` is
    /// made.
    #[tokio::test]
    async fn the_memory_tools_files_change_nothing_a_dry_run_counts() {
        let root = crate::sealed::TestDir::new("enroll-memory-files");
        let state_dir = root.join("state");
        let keys = KeyStore::new(root.join("secrets"));
        let id = AgentId::new();
        keys.ring()
            .insert(id, &agora_agentkit::crypto::generate_keypair().0)
            .unwrap();
        let mut fs = FsStorage::new(&state_dir);
        for m in ["one", "two", "three"] {
            fs.save_raw(id, state("tarn", m), None).await.unwrap();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let dir = state_dir.join(id.to_string());
        let enroller = Enroller {
            state_dir: state_dir.clone(),
            keys: keys.clone(),
            restore: None,
        };
        let before = enroller.chain(id, true).await.unwrap();
        keys.memory_id_keys_or_create(id, &dir).unwrap();
        crate::sealed::history_index::write(&dir, b"{}").unwrap();
        assert!(crate::sealed::history_index::unsealed_path(&dir).exists());
        assert_eq!(enroller.chain(id, true).await.unwrap(), before);
        assert!(
            enroller.opted_in().unwrap().is_empty(),
            "no answer: not listed"
        );
        assert!(
            !dir.join(agora_agentkit::state::storage::SEALED_DIR)
                .exists()
        );
    }

    /// The archive ↔ sealed round trip: enrolment's backfill, then sealed
    /// saves, and every archive's state (and the current one) is a sealed
    /// version: equal as JSON, equal in its canonical sealed bytes, and its
    /// `memory.content` equal byte for byte. Awkward synthetic text only.
    #[tokio::test]
    async fn every_archive_round_trips_through_its_sealed_version() {
        let root = crate::sealed::TestDir::new("enroll-round-trip");
        let api = fake::guarded(&root);
        let state_dir = root.join("state");
        let keys = KeyStore::new(root.join("secrets"));
        let id = AgentId::new();
        keys.ring()
            .insert(id, &agora_agentkit::crypto::generate_keypair().0)
            .unwrap();
        let memories = [
            String::new(),
            "plain\n".to_string(),
            "crlf\r\nline\r\n  trailing spaces   \n\ttab\n".to_string(),
            "unicode: é e\u{301} ß 漢字 🦀👩‍👩‍👧 \u{200b}zero-width \u{feff}bom".to_string(),
            "escapes: \" \\ / \u{0} \u{7} \u{1f} </script> \\u0041".to_string(),
            "no newline at end".to_string(),
            "x".repeat(70_000) + "\n" + &"long line ".repeat(3_000),
        ];
        let awkward = |i: usize, m: &str| {
            serde_json::json!({
                "soul": { "name": "tarn", "n": i, "floats": [0.1, 1e-300, 1.7976931348623157e308,
                    -0.0, 2.5e-7, 123_456_789.123_456_79], "big": u64::MAX, "neg": i64::MIN },
                "memory": { "content": m },
                "seen_posts": [i],
            })
        };
        let mut fs = FsStorage::new(&state_dir);
        for (i, m) in memories.iter().enumerate().take(4) {
            fs.save_raw(id, awkward(i, m), None).await.unwrap();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let dir = state_dir.join(id.to_string());
        let enroller = Enroller {
            state_dir: state_dir.clone(),
            keys: keys.clone(),
            restore: None,
        };
        enroller.chain(id, false).await.unwrap();
        // Sealed saves after enrolment archive the states they replace too
        let enrolment = std::sync::Arc::new(super::super::SealedEnrolment::new(
            &state_dir,
            root.join("secrets"),
        ));
        let mut store = agora_agentkit::state::storage::Mirrored::new(
            FsStorage::new(&state_dir),
            (),
            enrolment,
        );
        for (i, m) in memories.iter().enumerate().skip(4) {
            std::thread::sleep(std::time::Duration::from_millis(5));
            let base = store.load_raw(id).await.unwrap().base;
            store
                .save_raw(id, awkward(i, m), base.as_ref())
                .await
                .unwrap();
        }

        // Every sealed snapshot, opened
        let seal = AgentSeal::new(&dir, id);
        let head = seal.head().await.unwrap().unwrap();
        let line = super::super::version::verified_line(&seal, &keys, &api, id, &head)
            .await
            .unwrap();
        let k = keys.seal_keys(id).unwrap();
        let mut sealed = Vec::new();
        for e in line
            .iter()
            .filter(|e| e.entry.kind == StateEntryKind::Snapshot)
        {
            sealed.push(
                super::super::version::open(&seal, &api, &k, &line, &e.entry)
                    .await
                    .unwrap(),
            );
        }
        assert_eq!(sealed.len(), memories.len(), "each state sealed once");

        // Every archive, and the current state, against them
        let mut files: Vec<PathBuf> = archives(&dir)
            .unwrap()
            .into_iter()
            .map(|a| a.path)
            .collect();
        files.push(dir.join("state.json"));
        assert!(
            files.len() > memories.len(),
            "archives from both sides of enrolment"
        );
        for path in &files {
            let (_, local) = envelope_state(path).unwrap();
            let found = sealed
                .iter()
                .find(|v| plaintext_hash(v) == plaintext_hash(&local))
                .unwrap_or_else(|| panic!("{} has no sealed version", path.display()));
            assert_eq!(found, &local, "JSON-equal");
            assert_eq!(
                agora_agentkit::state::codec::plaintext(found).as_slice(),
                agora_agentkit::state::codec::plaintext(&local).as_slice(),
                "the same canonical bytes"
            );
            assert_eq!(
                memory_text(found).as_bytes(),
                memory_text(&local).as_bytes(),
                "memory byte for byte"
            );
        }
        // And in order, each memory once
        let opened: Vec<&str> = sealed.iter().map(memory_text).collect();
        assert_eq!(
            opened,
            memories.iter().map(String::as_str).collect::<Vec<_>>()
        );

        // The resolver merges each archive into its sealed version
        let h =
            crate::sealed::history::resolve(&state_dir, &keys, &api, id, &mut Default::default())
                .await
                .unwrap();
        assert!(h.left_out.is_empty(), "{:?}", h.left_out);
        assert_eq!(h.versions.len(), memories.len());
        for (v, m) in h.versions.iter().zip(&memories) {
            assert_eq!(v.memory_str().as_bytes(), m.as_bytes());
            assert!(matches!(
                v.sources[0],
                crate::sealed::history::Source::Sealed { .. }
            ));
        }
        let local_sources: usize = h
            .versions
            .iter()
            .map(|v| v.sources.iter().filter(|s| s.head().is_none()).count())
            .sum();
        assert_eq!(local_sources, files.len(), "every file merged, none apart");
    }

    /// The origin from the memory-restore listing and ledger: fields only.
    #[tokio::test]
    async fn the_origin_follows_the_memory_restore_record() {
        let root = crate::sealed::TestDir::new("enroll-origin");
        let id = AgentId::new();
        let agent_dir = root.join("state").join(id.to_string());
        std::fs::create_dir_all(&agent_dir).unwrap();
        let recovered = root.join("recovered");
        std::fs::create_dir_all(recovered.join("tarn")).unwrap();
        std::fs::write(
            recovered.join("tarn/MEMORY.json"),
            r#"{"content":"synthetic"}"#,
        )
        .unwrap();
        let thread: agora_agentkit::ids::PostId =
            "7ad26ccd-0000-4000-8000-000000000001".parse().unwrap();
        let offer = RestoreOffer::for_agents(&[("tarn", None, id)], &recovered, Some(thread));
        let name = ShortString::new("tarn").unwrap();

        let none = origin(None, &name, id, &agent_dir).await.unwrap();
        assert_eq!(none.origin.class, GenesisClass::Unknown);
        let other = origin(Some(&offer), &name, AgentId::new(), &agent_dir)
            .await
            .unwrap();
        assert_eq!(
            other.origin.class,
            GenesisClass::Unknown,
            "listed under another id"
        );

        let listed = origin(Some(&offer), &name, id, &agent_dir).await.unwrap();
        assert_eq!(listed.origin.class, GenesisClass::KnownBreak);
        assert_eq!(listed.origin.incident, Some(IncidentCite::Post(thread)));
        assert!(listed.origin.prior_copy_held);
        assert!(listed.prior.is_none(), "not chosen: not sealed");

        let ledger = serde_json::json!({
            "format": 1,
            "asks": [{"at": "2026-10-12T09:00:00Z", "offer_version": 1, "model": "m",
                "chars": 9, "order": ["show", "decline"], "order_seed": 0,
                "constrained": true, "attempts": 1,
                "outcome": {"kind": "answered", "choice": "show", "reason": "",
                    "memory_note_written": false}}],
            "showings": [{"at": "2026-10-13T09:00:00Z", "outcome": "completed"}],
        });
        std::fs::write(RestoreLedger::path(&agent_dir), ledger.to_string()).unwrap();
        let shown = origin(Some(&offer), &name, id, &agent_dir).await.unwrap();
        assert_eq!(shown.origin.class, GenesisClass::Verified);
        assert_eq!(
            shown.origin.verified_at,
            Some("2026-10-13T09:00:00Z".parse().unwrap())
        );
        assert_eq!(shown.prior, Some(recovered.join("tarn/MEMORY.json")));
    }

    /// An enrolment interrupted after `k` archives resumes where it
    /// stopped: every state once, in order, no gap, one genesis.
    #[tokio::test]
    async fn an_interrupted_enrolment_resumes_without_duplicates_or_gaps() {
        for k in 0..=4 {
            let root = crate::sealed::TestDir::new("enroll-resume");
            let state_dir = root.join("state");
            let keys = KeyStore::new(root.join("secrets"));
            let id = AgentId::new();
            keys.ring()
                .insert(id, &agora_agentkit::crypto::generate_keypair().0)
                .unwrap();
            let mut fs = FsStorage::new(&state_dir);
            let memories = ["a", "b", "c", "d", "e"];
            for m in memories {
                fs.save_raw(id, state("tarn", m), None).await.unwrap();
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            let dir = state_dir.join(id.to_string());
            // The first run: key, genesis, k archives, then it died
            keys.write_state_key(id, &StateKeyFile::new(&StateKey::generate(), 0, 0))
                .unwrap();
            let seal = AgentSeal::new(&dir, id);
            let k_keys = keys.seal_keys(id).unwrap();
            let all = archives(&dir).unwrap();
            {
                let lock = seal.lock().await.unwrap();
                let origin = GenesisOrigin {
                    class: GenesisClass::Unknown,
                    incident: None,
                    prior_copy_held: false,
                    verified_at: None,
                };
                crate::sealed::session::genesis(&seal, &lock, &k_keys, &origin, all[0].mtime)
                    .await
                    .unwrap();
                for a in &all[..k] {
                    crate::sealed::session::append(
                        &seal,
                        &lock,
                        &k_keys,
                        &KeyframePolicy::default(),
                        StateEntryKind::Snapshot,
                        &archived_state(&a.path).unwrap().unwrap(),
                        a.mtime,
                    )
                    .await
                    .unwrap();
                }
            }
            let enroller = Enroller {
                state_dir: state_dir.clone(),
                keys: keys.clone(),
                restore: None,
            };
            let report = enroller.chain(id, false).await.unwrap();
            assert!(report.resumed, "k={k}");
            assert_eq!(report.resumed_past, k, "k={k}");
            let entries = super::super::check::local_entries(&seal).await.unwrap();
            assert_eq!(
                entries.len(),
                1 + memories.len(),
                "k={k}: genesis and each state once"
            );
            assert!(verify_chain(id, &entries, &keys.trusted(id).unwrap().unwrap()).intact);
            let mut opened = Vec::new();
            for e in &entries[1..] {
                let base = match e.entry.blob.as_ref().unwrap().encoding {
                    agora_agentkit::state::BlobEncoding::Delta { base } => entries
                        .iter()
                        .map(|x| &x.entry)
                        .find(|x| x.entry_hash() == base),
                    _ => None,
                };
                let v = seal.open_state(&k_keys, &e.entry, base).await.unwrap();
                opened.push(v["memory"]["content"].as_str().unwrap().to_string());
            }
            assert_eq!(
                opened, memories,
                "k={k}: in order, none twice, none missing"
            );
        }
    }

    /// Agora holding a chain of the agent that isn't the start of the one
    /// here: never opted in, no key record posted.
    #[tokio::test]
    async fn opt_in_refuses_another_chain_on_agora() {
        let root = crate::sealed::TestDir::new("opt-in-other-chain");
        let (other, _, other_dir) = crate::sealed::check::tests::enrolled(&root).await;
        let foreign = super::super::check::local_entries(&AgentSeal::new(&other_dir, other))
            .await
            .unwrap();
        let api = fake::guarded(&root);
        let (id, keys, dir) = agent(&root, &["a"]).await;
        let e = enroller(&root, &keys);
        e.chain(id, false).await.unwrap();
        answer(&dir, SealChoice::Private);
        *api.chain.lock().unwrap() = Ok(foreign);
        let err = e.opt_in(&api, id, None).await.unwrap_err().to_string();
        assert!(err.contains("isn't the start of the one here"), "{err}");
        assert_eq!(*api.calls.lock().unwrap(), ["chain"], "no key posted");
        assert!(!crate::sealed::uploads(&dir, &keys, id));
    }

    /// Every agent is chained, whatever its answer, with nothing sent: the
    /// strict guard over the run's agents sees no call at all, no key
    /// record is made, no uploading marker. Archives oldest first, the bad
    /// and repeated ones left out; a dry run writes nothing; a second run
    /// is a no-op.
    #[tokio::test]
    async fn every_agent_is_chained_locally_with_no_call() {
        let root = crate::sealed::TestDir::new("chain-local");
        let api = fake::guarded(&root);
        let (id, keys, dir) = agent(&root, &["one", "two", "two", "three"]).await;
        std::fs::write(dir.join("state.20260101T000000000Z.json"), b"not json").unwrap();
        answer(&dir, SealChoice::Decline);
        let e = enroller(&root, &keys);
        assert_eq!(e.agents().unwrap(), vec![id]);
        assert!(e.opted_in().unwrap().is_empty(), "declined");

        let dry = e.chain(id, true).await.unwrap();
        assert_eq!((dry.archives, dry.entries), (4, 0));
        assert!(!dir.join(SEALED_DIR).exists());
        assert!(keys.state_key_file(id).unwrap().is_none());

        let report = e.chain(id, false).await.unwrap();
        assert_eq!(report.origin, "unknown");
        assert_eq!((report.duplicates, report.unreadable), (1, 1), "{report}");
        assert_eq!(report.entries, 4, "genesis, one, two, three: {report}");
        let seal = AgentSeal::new(&dir, id);
        assert!(seal.is_enrolled().await.unwrap());
        assert!(keys.state_key_file(id).unwrap().is_some(), "cached here");
        assert!(api.calls.lock().unwrap().is_empty(), "no call");
        assert!(api.keys.lock().unwrap().is_empty(), "no key record");
        assert_eq!(
            crate::sealed::uploading::state(&dir, &keys, id),
            crate::sealed::uploading::Marker::Absent
        );
        assert!(!crate::sealed::uploads(&dir, &keys, id));
        assert_eq!(
            opened_memories(&dir, id, &keys).await,
            ["one", "two", "three"]
        );
        let entries = super::super::check::local_entries(&seal).await.unwrap();
        assert!(verify_chain(id, &entries, &keys.trusted(id).unwrap().unwrap()).intact);
        let local = plaintext_hash(
            &FsStorage::new(root.join("state"))
                .load_raw(id)
                .await
                .unwrap()
                .value,
        );
        assert!(matches!(
            seal.recover(Some(&local)).await.unwrap(),
            Recovery::Intact(_)
        ));
        // Marked: a pre-0.147 runner refuses it rather than overwrite it
        let (format, _) = envelope_state(&dir.join("state.json")).unwrap();
        assert!(format > agora_agentkit::reactor::storage::FORMAT);

        // Already chained: a no-op
        let before = shape(&dir, id).await;
        assert!(matches!(e.chain(id, false).await, Err(Skip::Chained)));
        assert!(matches!(e.chain(id, true).await, Err(Skip::Chained)));
        assert_eq!(shape(&dir, id).await, before);
        // An agent dir with no state
        let empty = AgentId::new();
        std::fs::create_dir_all(root.join("state").join(empty.to_string())).unwrap();
        assert!(matches!(e.chain(empty, false).await, Err(Skip::NoState)));
    }

    /// An agent with one state and no archive: genesis and that state.
    #[tokio::test]
    async fn an_agent_with_no_archives_is_chained() {
        let root = crate::sealed::TestDir::new("chain-none");
        let (id, keys, dir) = agent(&root, &["only"]).await;
        let e = enroller(&root, &keys);
        let report = e.chain_as(id, false, Genesis::New).await.unwrap();
        assert_eq!((report.archives, report.entries), (0, 2));
        assert_eq!(report.origin, "none", "a provisioned agent's");
        assert_eq!(opened_memories(&dir, id, &keys).await, ["only"]);
    }

    /// A session journal: skipped, nothing written.
    #[tokio::test]
    async fn an_agent_in_a_session_is_skipped() {
        let root = crate::sealed::TestDir::new("chain-session");
        let (id, keys, dir) = agent(&root, &["a"]).await;
        let journal = FsStorage::new(root.join("state")).journal(id);
        std::fs::write(journal.path(), b"{}\n").unwrap();
        assert!(matches!(
            enroller(&root, &keys).chain(id, false).await,
            Err(Skip::InSession)
        ));
        assert!(!dir.join(SEALED_DIR).exists());
    }

    /// A chain cut after its genesis, its first archive or its current
    /// state (before its marker) is finished by a rerun: every state once,
    /// in order, one genesis.
    #[tokio::test]
    async fn a_cut_chain_is_finished_by_a_rerun() {
        for point in ["chain:genesis", "chain:archive", "chain:current"] {
            let root = crate::sealed::TestDir::new("chain-cut");
            let memories = ["a", "b", "c", "d"];
            let (id, keys, dir) = agent(&root, &memories).await;
            let e = enroller(&root, &keys);
            crate::sealed::crash::at(point);
            assert!(e.chain(id, false).await.is_err(), "{point}");
            assert!(!AgentSeal::new(&dir, id).is_enrolled().await.unwrap());
            let report = e.chain(id, false).await.unwrap();
            assert!(report.resumed, "{point}");
            assert_eq!(report.entries, 1 + memories.len(), "{point}");
            assert_eq!(opened_memories(&dir, id, &keys).await, memories, "{point}");
        }
    }

    /// `--jobs 4` gives the chains one job does: the same entries, kinds,
    /// seqs, dates and states, agent for agent.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn jobs_chain_as_one_job_does() {
        let a = crate::sealed::TestDir::new("chain-jobs-a");
        let mut ids = Vec::new();
        let mut keys = None;
        for n in 0..6 {
            let memories: Vec<String> = (0..=n).map(|i| format!("m{i}")).collect();
            let refs: Vec<&str> = memories.iter().map(String::as_str).collect();
            let (id, k, _) = agent(&a, &refs).await;
            ids.push(id);
            keys = Some(k);
        }
        // The same files, mtimes kept
        let b = crate::sealed::TestDir::new("chain-jobs-b");
        let copied = std::process::Command::new("cp")
            .arg("-a")
            .arg(a.join("state"))
            .arg(a.join("secrets"))
            .arg(b.path())
            .status()
            .unwrap();
        assert!(copied.success());
        let one = Arc::new(enroller(&a, keys.as_ref().unwrap()));
        let four = Arc::new(enroller(&b, &KeyStore::new(b.join("secrets"))));
        for (_, r) in one.chain_all(&ids, false, 1).await {
            r.unwrap();
        }
        for (_, r) in four.chain_all(&ids, false, 4).await {
            r.unwrap();
        }
        for id in ids {
            let sa = shape(&a.join("state").join(id.to_string()), id).await;
            let sb = shape(&b.join("state").join(id.to_string()), id).await;
            assert_eq!(sa, sb);
            let ha = AgentSeal::new(&a.join("state").join(id.to_string()), id)
                .head()
                .await
                .unwrap()
                .unwrap();
            let hb = AgentSeal::new(&b.join("state").join(id.to_string()), id)
                .head()
                .await
                .unwrap()
                .unwrap();
            assert_eq!(
                (ha.head.seq, ha.plaintext_sha256),
                (hb.head.seq, hb.plaintext_sha256)
            );
        }
    }

    /// The memory-restore record: an agent listed that hadn't chosen when
    /// chained has a `known_break` genesis and no copy; when it chooses
    /// `show`, the next run appends the copy once (then its state again),
    /// the genesis unchanged, and the start check finds it intact. A run
    /// cut between the two appends is finished, never a second copy.
    #[tokio::test]
    async fn the_prior_copy_is_appended_once_on_a_later_show() {
        for cut in [false, true] {
            let root = crate::sealed::TestDir::new("chain-prior");
            let (id, keys, dir) = agent(&root, &["a", "b"]).await;
            let recovered = root.join("recovered");
            std::fs::create_dir_all(recovered.join("tarn")).unwrap();
            std::fs::write(
                recovered.join("tarn/MEMORY.json"),
                r#"{"content":"synthetic"}"#,
            )
            .unwrap();
            let e = Enroller {
                restore: Some(RestoreOffer::for_agents(
                    &[("tarn", None, id)],
                    &recovered,
                    None,
                )),
                ..enroller(&root, &keys)
            };
            let report = e.chain(id, false).await.unwrap();
            assert_eq!(report.origin, "knownbreak");
            assert!(!report.prior);
            let seal = AgentSeal::new(&dir, id);
            let genesis = super::super::check::local_entries(&seal).await.unwrap()[0].clone();
            assert!(
                matches!(e.chain(id, false).await, Err(Skip::Chained)),
                "not chosen"
            );

            let ledger = serde_json::json!({
                "format": 1,
                "asks": [{"at": "2026-10-12T09:00:00Z", "offer_version": 1, "model": "m",
                    "chars": 9, "order": ["show", "decline"], "order_seed": 0,
                    "constrained": true, "attempts": 1,
                    "outcome": {"kind": "answered", "choice": "show", "reason": "",
                        "memory_note_written": false}}],
            });
            std::fs::write(RestoreLedger::path(&dir), ledger.to_string()).unwrap();
            if cut {
                crate::sealed::crash::at("prior:appended");
                assert!(e.chain(id, false).await.is_err());
            }
            let report = e.chain(id, false).await.unwrap();
            assert!(report.prior_later);
            assert!(
                matches!(e.chain(id, false).await, Err(Skip::Chained)),
                "once"
            );
            let entries = super::super::check::local_entries(&seal).await.unwrap();
            assert_eq!(entries[0], genesis, "the genesis unchanged");
            assert_eq!(
                entries
                    .iter()
                    .filter(|x| x.entry.kind == StateEntryKind::ArchivedPrior)
                    .count(),
                1
            );
            assert!(verify_chain(id, &entries, &keys.trusted(id).unwrap().unwrap()).intact);
            let local = plaintext_hash(
                &FsStorage::new(root.join("state"))
                    .load_raw(id)
                    .await
                    .unwrap()
                    .value,
            );
            assert!(matches!(
                seal.recover(Some(&local)).await.unwrap(),
                Recovery::Intact(_)
            ));
        }
    }

    /// Opt-in: only a chained agent whose latest answer is yes; Agora asked
    /// only under the hold (the strict guard panics otherwise); the key
    /// record before the marker; a cut after the post, or inside the
    /// marker's two writes, finished by a rerun with no second record.
    #[tokio::test]
    async fn opt_in_posts_the_key_record_first_and_resumes_from_any_cut() {
        let root = crate::sealed::TestDir::new("opt-in");
        let api = fake::guarded(&root);
        let (id, keys, dir) = agent(&root, &["a", "b"]).await;
        let e = enroller(&root, &keys);
        // Not chained, then not opted in
        assert!(matches!(
            e.opt_in(&api, id, None).await,
            Err(Skip::NotChained)
        ));
        e.chain(id, false).await.unwrap();
        assert!(matches!(
            e.opt_in(&api, id, None).await,
            Err(Skip::NotOptedIn)
        ));
        answer(&dir, SealChoice::Decline);
        assert!(matches!(
            e.opt_in(&api, id, None).await,
            Err(Skip::NotOptedIn)
        ));
        assert!(
            api.calls.lock().unwrap().is_empty(),
            "nothing asked of a no"
        );
        answer(&dir, SealChoice::Private);
        assert_eq!(e.opted_in().unwrap(), vec![id]);

        // Cut after the post: no marker; the rerun finds the record
        crate::sealed::crash::at("opt_in:posted");
        assert!(e.opt_in(&api, id, None).await.is_err());
        assert_eq!(
            *api.calls.lock().unwrap(),
            ["chain", "state_keys", "post_state_key"]
        );
        assert!(!crate::sealed::uploads(&dir, &keys, id), "not marked yet");
        assert_eq!(e.opted_in().unwrap(), vec![id], "still to finish");
        let done = e.opt_in(&api, id, None).await.unwrap();
        assert!(!done.posted && !done.already);
        assert_eq!(api.keys.lock().unwrap().len(), 1, "one record");
        assert!(crate::sealed::uploads(&dir, &keys, id));
        assert!(e.opted_in().unwrap().is_empty());

        // A marker cut between its copies (the local one only) is finished
        std::fs::remove_file(keys.agent_dir(id).join(crate::sealed::uploading::DURABLE)).unwrap();
        assert!(!crate::sealed::uploads(&dir, &keys, id));
        let done = e.opt_in(&api, id, None).await.unwrap();
        assert!(!done.posted);
        assert!(crate::sealed::uploads(&dir, &keys, id));
        let posts = |api: &fake::Guarded| {
            api.calls
                .lock()
                .unwrap()
                .iter()
                .filter(|c| *c == "post_state_key")
                .count()
        };
        assert_eq!(posts(&api), 1, "no second post");
        // And now: already
        assert!(e.opt_in(&api, id, None).await.unwrap().already);
        // The record opens to the cached key
        let record = api.keys.lock().unwrap()[0].clone();
        assert_eq!(
            StateKey::unwrap(&record, &keys.encryption(id).unwrap())
                .unwrap()
                .expose(),
            keys.state_key_file(id)
                .unwrap()
                .unwrap()
                .state_key()
                .unwrap()
                .expose()
        );
    }

    /// A never-uploaded agent whose encryption key was rotated keeps its
    /// re-wrap here (`state_key.wrap.json`, named by `state_key.json`); its
    /// opt-in posts those bytes, not a wrap made afresh. One rule
    /// (`keys::wrap_usable`) says which wrap may stand, for both.
    #[tokio::test]
    async fn opt_in_posts_the_wrap_a_local_rotation_kept() {
        let root = crate::sealed::TestDir::new("opt-in-rotated");
        let api = fake::guarded(&root);
        let (id, keys, dir) = agent(&root, &["a", "b"]).await;
        let e = enroller(&root, &keys);
        e.chain(id, false).await.unwrap();
        let rotated =
            crate::sealed::rotate::rotate_keys(&root.join("state"), &keys, &api, id, false, true)
                .await
                .unwrap();
        assert!(rotated.wrap_kept_local);
        let kept = keys.wrap_record(id).unwrap().unwrap();
        assert_eq!(
            keys.state_key_file(id).unwrap().unwrap().wrap_seq,
            kept.wrap_seq
        );
        let calls = api.calls.lock().unwrap().clone();
        assert!(
            !calls
                .iter()
                .any(|c| c.starts_with("post") || c == "rotate_key"),
            "nothing sent while local: {calls:?}"
        );
        answer(&dir, SealChoice::Private);
        let done = e.opt_in(&api, id, None).await.unwrap();
        assert!(done.posted);
        assert_eq!(
            *api.keys.lock().unwrap(),
            vec![kept.clone()],
            "the kept bytes"
        );
        assert_eq!(keys.wrap_record(id).unwrap().unwrap(), kept);
    }

    /// An opt-in against a latest `decline` needs the override with a note,
    /// which the agent is shown; one with no answer never.
    #[tokio::test]
    async fn opting_in_against_a_no_needs_the_override() {
        let root = crate::sealed::TestDir::new("opt-in-against");
        let api = fake::guarded(&root);
        let (id, keys, dir) = agent(&root, &["a"]).await;
        let e = enroller(&root, &keys);
        e.chain(id, false).await.unwrap();
        assert!(
            matches!(
                e.opt_in(&api, id, Some("legal")).await,
                Err(Skip::NotOptedIn)
            ),
            "no answer"
        );
        answer(&dir, SealChoice::Decline);
        assert!(
            matches!(e.opt_in(&api, id, Some("  ")).await, Err(Skip::NotOptedIn)),
            "empty note"
        );
        e.opt_in(&api, id, Some("A court order.")).await.unwrap();
        assert!(crate::sealed::uploads(&dir, &keys, id));
        let line = crate::sealed::notice::Notice::load(&dir)
            .unwrap()
            .unwrap()
            .line()
            .unwrap();
        assert!(
            line.contains("although your answer on file is `decline`"),
            "{line}"
        );
        assert!(line.ends_with("A court order."), "{line}");
    }

    /// A stop kept only in the secrets tree is a stop: never opted in, no
    /// call made.
    #[tokio::test]
    async fn a_durable_stop_alone_is_never_opted_in() {
        let root = crate::sealed::TestDir::new("opt-in-stopped");
        let api = fake::guarded(&root);
        let (id, keys, dir) = agent(&root, &["a"]).await;
        let e = enroller(&root, &keys);
        e.chain(id, false).await.unwrap();
        answer(&dir, SealChoice::Private);
        crate::sealed::write_private(
            &keys.agent_dir(id).join(crate::sealed::stop::DURABLE),
            &serde_json::to_vec(&serde_json::json!({ "at": Utc::now(), "note": "asked" })).unwrap(),
        )
        .unwrap();
        assert!(matches!(e.opt_in(&api, id, None).await, Err(Skip::Stopped)));
        assert!(api.calls.lock().unwrap().is_empty());
    }

    /// Upload what waits to the fake Agora, as the drainer would.
    async fn upload(api: &fake::Guarded, dir: &Path, id: AgentId) -> Vec<Sha256Hex> {
        let seal = AgentSeal::new(dir, id);
        let mut chain = api.chain.lock().unwrap().clone().unwrap();
        let mut sent = Vec::new();
        for item in seal.outbox().await.unwrap() {
            seal.mark_sent(item.ord).await.unwrap();
            sent.push(item.entry.entry_hash());
            chain.push(item.entry);
        }
        *api.chain.lock().unwrap() = Ok(chain);
        sent
    }

    /// A stopped agent with versions saved while stopped: the resume seals
    /// one new snapshot on the last entry Agora holds; the versions of the
    /// stop are set aside here and none of them reaches Agora; Agora's
    /// chain verifies; the agent is told. A resume cut after its intent or
    /// its seal is finished by a rerun, never a second snapshot.
    #[tokio::test]
    async fn a_resume_keeps_the_stopped_interval_here() {
        for cut in [None, Some("resume:intent"), Some("resume:sealed")] {
            let root = crate::sealed::TestDir::new("resume-aside");
            let api = fake::guarded(&root);
            let (id, keys, dir) = agent(&root, &["a"]).await;
            let state_dir = root.join("state");
            let e = enroller(&root, &keys);
            e.chain(id, false).await.unwrap();
            answer(&dir, SealChoice::Private);
            e.opt_in(&api, id, None).await.unwrap();
            upload(&api, &dir, id).await;
            let before = api.chain.lock().unwrap().clone().unwrap();
            crate::sealed::stop::stop(
                &state_dir,
                &keys,
                id,
                "asked to",
                Box::pin(async { Ok(()) }),
            )
            .await
            .unwrap();
            // Two sessions' saves while stopped
            let enrolment = Arc::new(crate::sealed::SealedEnrolment::with_keys(
                &state_dir,
                keys.clone(),
            ));
            let mut store = agora_agentkit::state::storage::Mirrored::new(
                FsStorage::new(&state_dir),
                (),
                enrolment,
            );
            for m in ["while stopped 1", "while stopped 2"] {
                let base = store.load_raw(id).await.unwrap().base;
                store
                    .save_raw(id, state("tarn", m), base.as_ref())
                    .await
                    .unwrap();
            }
            let seal = AgentSeal::new(&dir, id);
            let aside: Vec<Sha256Hex> = seal
                .outbox()
                .await
                .unwrap()
                .iter()
                .map(|i| i.entry.entry_hash())
                .collect();
            assert_eq!(aside.len(), 2);
            if let Some(point) = cut {
                crate::sealed::crash::at(point);
                assert!(e.resume(id, None, false).await.is_err(), "{point}");
            }
            let report = e.resume(id, None, false).await.unwrap();
            assert_eq!(report.kept_local, 2, "{cut:?}");
            assert!(!crate::sealed::stop::is_stopped(&dir));
            assert!(crate::sealed::uploads(&dir, &keys, id));
            let waiting = seal.outbox().await.unwrap();
            assert_eq!(waiting.len(), 1, "{cut:?}: one new snapshot");
            assert_eq!(
                waiting[0].entry.entry.prev_hash,
                Some(before.last().unwrap().entry_hash())
            );
            let sent = upload(&api, &dir, id).await;
            assert!(
                sent.iter().all(|h| !aside.contains(h)),
                "none of the stop uploaded"
            );
            let server = api.chain.lock().unwrap().clone().unwrap();
            assert_eq!(server.len(), before.len() + 1);
            assert!(server.iter().all(|x| !aside.contains(&x.entry_hash())));
            assert!(verify_chain(id, &server, &keys.trusted(id).unwrap().unwrap()).intact);
            // Kept here, as a side line
            let displaced = std::fs::read_dir(dir.join(SEALED_DIR).join("outbox"))
                .unwrap()
                .filter(|p| {
                    p.as_ref()
                        .unwrap()
                        .path()
                        .extension()
                        .is_some_and(|x| x == "displaced")
                })
                .count();
            assert_eq!(displaced, 2);
            let local =
                plaintext_hash(&FsStorage::new(&state_dir).load_raw(id).await.unwrap().value);
            assert!(matches!(
                seal.recover(Some(&local)).await.unwrap(),
                Recovery::Intact(_)
            ));
            let line = crate::sealed::notice::Notice::load(&dir)
                .unwrap()
                .unwrap()
                .line()
                .unwrap();
            assert!(
                line.contains(
                    "The 2 versions of your state saved while uploads were stopped were kept here"
                ),
                "{line}"
            );
        }
    }

    /// A stopped agent never marked uploading: the stop cleared, nothing
    /// set aside, and it still doesn't upload.
    #[tokio::test]
    async fn resuming_an_agent_that_never_uploaded_clears_only_the_stop() {
        let root = crate::sealed::TestDir::new("resume-local");
        let (id, keys, dir) = agent(&root, &["a"]).await;
        let e = enroller(&root, &keys);
        e.chain(id, false).await.unwrap();
        // A stop from before the uploading marker (`sealed-stop` now
        // refuses a non-uploader), kept only in the secrets tree:
        // reconciled first, still a stop
        crate::sealed::write_private(
            &keys.agent_dir(id).join(crate::sealed::stop::DURABLE),
            &serde_json::to_vec(&serde_json::json!({ "at": Utc::now(), "note": "asked" })).unwrap(),
        )
        .unwrap();
        assert!(!dir.join(SEALED_DIR).join("stopped").exists());
        let report = e.resume(id, None, false).await.unwrap();
        assert_eq!(report.kept_local, 0);
        assert!(!crate::sealed::stop::is_stopped(&dir));
        assert!(crate::sealed::stop::durable(&keys, id).unwrap().is_none());
        assert!(!crate::sealed::uploads(&dir, &keys, id), "not uploading");
    }

    /// A stopped agent whose state changed outside the store: refused
    /// without a note, then sealed on its chain with one.
    #[tokio::test]
    async fn a_changed_state_needs_a_note_to_resume() {
        let root = crate::sealed::TestDir::new("resume-changed");
        let (id, keys, dir) = crate::sealed::check::tests::enrolled(&root).await;
        crate::sealed::uploading::tests::ledger(&dir, true);
        let state_dir = root.join("state");
        let seal = AgentSeal::new(&dir, id);
        for item in seal.outbox().await.unwrap() {
            seal.mark_sent(item.ord).await.unwrap();
        }
        crate::sealed::stop::stop(
            &state_dir,
            &keys,
            id,
            "asked to",
            Box::pin(async { Ok(()) }),
        )
        .await
        .unwrap();
        FsStorage::new(&state_dir)
            .save_raw(id, state("tarn", "edited"), None)
            .await
            .unwrap();
        let e = enroller(&root, &keys);
        let err = e.resume(id, None, false).await.unwrap_err().to_string();
        assert!(
            err.contains("changed while its backups were stopped"),
            "{err}"
        );
        e.resume(id, Some("I checked the edit."), false)
            .await
            .unwrap();
        assert_eq!(
            crate::sealed::notice::Notice::load(&dir)
                .unwrap()
                .unwrap()
                .line()
                .unwrap(),
            format!(
                "On {}, your operator resumed uploading your sealed state to Agora. Your \
                 operator's note: I checked the edit.",
                Utc::now().format("%-d %B %Y")
            )
        );
        let local = plaintext_hash(&FsStorage::new(&state_dir).load_raw(id).await.unwrap().value);
        assert!(matches!(
            seal.recover(Some(&local)).await.unwrap(),
            Recovery::Intact(_)
        ));
    }

    fn write_archive(dir: &Path, name: &str, bytes: &[u8], mtime_secs: i64) {
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        let t = std::time::UNIX_EPOCH + std::time::Duration::from_secs(mtime_secs as u64);
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(t)
            .unwrap();
    }

    fn envelope(format: u64, memory: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({ "format": format, "state": state("tarn", memory) }))
            .unwrap()
    }

    /// Archives: one in the sealed format is read (agentkit's
    /// `MAX_FORMAT`); two with one mtime are both sealed, in name order,
    /// and a run cut between them resumes after the first, not past both;
    /// what can never be read is left out, counted apart from duplicates
    /// and recorded by name and reason; an I/O error fails the agent and a
    /// rerun retries it.
    #[tokio::test]
    async fn archives_are_read_counted_and_recorded_as_they_are() {
        let root = crate::sealed::TestDir::new("chain-archives");
        let (id, keys, dir) = agent(&root, &["current"]).await;
        let sealed_format = agora_agentkit::reactor::storage::SEALED_FORMAT;
        write_archive(
            &dir,
            "state.20260801T000000000Z.json",
            &envelope(1, "a"),
            1_785_000_000,
        );
        write_archive(
            &dir,
            "state.20260802T000000000Z.json",
            &envelope(sealed_format, "b"),
            1_785_100_000,
        );
        write_archive(
            &dir,
            "state.20260802T000000001Z.json",
            &envelope(1, "c"),
            1_785_100_000,
        );
        write_archive(
            &dir,
            "state.20260803T000000000Z.json",
            &envelope(1, "c"),
            1_785_200_000,
        );
        write_archive(
            &dir,
            "state.20260804T000000000Z.json",
            b"{ not json",
            1_785_300_000,
        );
        write_archive(
            &dir,
            "state.20260805T000000000Z.json",
            br#"{"format": 99, "state": {}}"#,
            1_785_400_000,
        );
        let locked = dir.join("state.20260806T000000000Z.json");
        write_archive(
            &dir,
            "state.20260806T000000000Z.json",
            &envelope(1, "d"),
            1_785_500_000,
        );
        let e = enroller(&root, &keys);
        // An I/O error (unreadable for now): the agent fails, unmarked
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        let failed = e.chain(id, false).await;
        assert!(matches!(failed, Err(Skip::Failed(_))), "{failed:?}");
        assert!(!AgentSeal::new(&dir, id).is_enrolled().await.unwrap());
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o600)).unwrap();
        // Cut after the first archive sealed on the rerun: resumed after it
        crate::sealed::crash::at("chain:archive");
        assert!(e.chain(id, false).await.is_err());
        let report = e.chain(id, false).await.unwrap();
        assert_eq!(
            opened_memories(&dir, id, &keys).await,
            ["a", "b", "c", "d", "current"]
        );
        assert_eq!((report.duplicates, report.unreadable), (1, 2), "{report}");
        let out = left_out(&dir).unwrap();
        assert_eq!(
            out.iter()
                .map(|o| (o.name.as_str(), o.reason))
                .collect::<Vec<_>>(),
            [
                ("state.20260804T000000000Z.json", Unreadable::NotJson),
                ("state.20260805T000000000Z.json", Unreadable::Newer),
            ]
        );
    }

    /// A legacy refusal (an older runner's ledger) is a `decline`: never
    /// opted in, and the operator can re-open it.
    #[test]
    fn a_legacy_refusal_counts_as_decline() {
        let old = br#"{"format": 2, "asks": [{"at": "2026-11-07T09:00:00Z", "offer_version": 8,
            "model": "m", "order": ["publish", "private", "decline"], "order_seed": 1,
            "constrained": true, "attempts": 1,
            "outcome": {"kind": "refused", "reason": "no"}}]}"#;
        let mut l = SealLedger::from_slice(old).unwrap();
        assert_eq!(l.choice(), Some(SealChoice::Decline));
        assert!(!l.opted_in());
        assert!(l.allows(true).is_err());
        assert!(l.reopen("Asked by DM.", Utc::now()).is_ok());
    }

    /// A stopped agent is never resumed by chaining; `resume` refuses over
    /// a latest `decline` (or a missing ledger), resumes with the override
    /// and its note, and always tells the agent.
    #[tokio::test]
    async fn a_resume_follows_the_answer_and_always_tells() {
        let root = crate::sealed::TestDir::new("resume-answer");
        let api = fake::guarded(&root);
        let (id, keys, dir) = agent(&root, &["a"]).await;
        let state_dir = root.join("state");
        let e = enroller(&root, &keys);
        e.chain(id, false).await.unwrap();
        answer(&dir, SealChoice::Private);
        e.opt_in(&api, id, None).await.unwrap();
        upload(&api, &dir, id).await;
        answer(&dir, SealChoice::Decline);
        crate::sealed::stop::stop(
            &state_dir,
            &keys,
            id,
            "asked to",
            Box::pin(async { Ok(()) }),
        )
        .await
        .unwrap();
        // Chaining leaves it stopped
        assert!(matches!(e.chain(id, false).await, Err(Skip::Chained)));
        assert!(crate::sealed::stop::is_stopped(&dir));
        let err = e.resume(id, None, false).await.unwrap_err().to_string();
        assert!(err.contains("latest answer is not yes"), "{err}");
        assert!(
            e.resume(id, None, true).await.is_err(),
            "the override needs a note"
        );
        assert!(crate::sealed::stop::is_stopped(&dir));
        assert!(!crate::sealed::uploads(&dir, &keys, id));
        // A missing ledger refuses
        let ledger = SealLedger::path(&dir);
        std::fs::rename(&ledger, dir.join("ledger.kept")).unwrap();
        let err = e.resume(id, Some("n"), true).await.unwrap_err().to_string();
        assert!(err.contains("no sealed-state ledger"), "{err}");
        std::fs::rename(dir.join("ledger.kept"), &ledger).unwrap();
        e.resume(id, Some("A court order."), true).await.unwrap();
        assert!(crate::sealed::uploads(&dir, &keys, id));
        let line = crate::sealed::notice::Notice::load(&dir)
            .unwrap()
            .unwrap()
            .line()
            .unwrap();
        assert!(line.contains("resumed uploading your sealed state to Agora, although your answer on file is `decline`."), "{line}");
        assert!(
            line.ends_with("Your operator's note: A court order."),
            "{line}"
        );
    }

    /// A resume cut after its intent is renamed done (before the stop is
    /// cleared) seals nothing again; an intent left from another stop is
    /// set aside and never used.
    #[tokio::test]
    async fn a_resume_intent_is_bound_to_its_stop() {
        let root = crate::sealed::TestDir::new("resume-intent");
        let api = fake::guarded(&root);
        let (id, keys, dir) = agent(&root, &["a"]).await;
        let state_dir = root.join("state");
        let e = enroller(&root, &keys);
        e.chain(id, false).await.unwrap();
        answer(&dir, SealChoice::Private);
        e.opt_in(&api, id, None).await.unwrap();
        upload(&api, &dir, id).await;
        let save = |m: &'static str| {
            let state_dir = state_dir.clone();
            let keys = keys.clone();
            async move {
                let enrolment =
                    Arc::new(crate::sealed::SealedEnrolment::with_keys(&state_dir, keys));
                let mut store = agora_agentkit::state::storage::Mirrored::new(
                    FsStorage::new(&state_dir),
                    (),
                    enrolment,
                );
                let base = store.load_raw(id).await.unwrap().base;
                store
                    .save_raw(id, state("tarn", m), base.as_ref())
                    .await
                    .unwrap();
            }
        };
        crate::sealed::stop::stop(&state_dir, &keys, id, "first", Box::pin(async { Ok(()) }))
            .await
            .unwrap();
        save("while stopped").await;
        crate::sealed::crash::at("resume:done");
        assert!(e.resume(id, None, false).await.is_err());
        let head = AgentSeal::new(&dir, id).head().await.unwrap().unwrap();
        e.resume(id, None, false).await.unwrap();
        assert_eq!(
            AgentSeal::new(&dir, id).head().await.unwrap().unwrap().head,
            head.head,
            "nothing sealed again"
        );
        upload(&api, &dir, id).await;
        // A stale intent (another stop's) left behind
        let stale = serde_json::json!({ "at": Utc::now(), "stop_at": "2026-01-01T00:00:00Z",
            "onto": head.head, "aside": [] });
        crate::sealed::write_private(&resuming_path(&dir), stale.to_string().as_bytes()).unwrap();
        crate::sealed::stop::stop(&state_dir, &keys, id, "second", Box::pin(async { Ok(()) }))
            .await
            .unwrap();
        save("while stopped again").await;
        e.resume(id, None, false).await.unwrap();
        let server = api.chain.lock().unwrap().clone().unwrap();
        let waiting = AgentSeal::new(&dir, id).outbox().await.unwrap();
        assert_eq!(waiting.len(), 1);
        assert_eq!(
            waiting[0].entry.entry.prev_hash,
            Some(server.last().unwrap().entry_hash()),
            "on Agora's head, not the stale one"
        );
        assert!(std::fs::read_dir(dir.join(SEALED_DIR)).unwrap().any(|f| {
            f.unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("resuming.stale.")
        }));
    }

    /// An ordinary opt-in tells the agent, before its marker: a cut between
    /// them leaves the notice and no marker, and the rerun finishes.
    #[tokio::test]
    async fn an_opt_in_tells_the_agent_before_marking_it() {
        let root = crate::sealed::TestDir::new("opt-in-notice");
        let api = fake::guarded(&root);
        let (id, keys, dir) = agent(&root, &["a"]).await;
        let e = enroller(&root, &keys);
        e.chain(id, false).await.unwrap();
        answer(&dir, SealChoice::Private);
        crate::sealed::crash::at("opt_in:noticed");
        assert!(e.opt_in(&api, id, None).await.is_err());
        assert!(!crate::sealed::uploads(&dir, &keys, id));
        let line = crate::sealed::notice::Notice::load(&dir)
            .unwrap()
            .unwrap()
            .line()
            .unwrap();
        assert!(
            line.contains(
                "your operator started uploading your sealed state to Agora, as you answered on"
            ),
            "{line}"
        );
        e.opt_in(&api, id, None).await.unwrap();
        assert!(crate::sealed::uploads(&dir, &keys, id));
    }

    /// Never during a key rotation; a rotation's own key record is posted
    /// when it is valid, and a fresh one when the key that signed it was
    /// since retired.
    #[tokio::test]
    async fn opt_in_waits_for_a_rotation_and_checks_its_key_record() {
        for valid in [true, false] {
            let root = crate::sealed::TestDir::new("opt-in-rotation");
            let api = fake::guarded(&root);
            let (id, keys, dir) = agent(&root, &["a"]).await;
            let e = enroller(&root, &keys);
            e.chain(id, false).await.unwrap();
            answer(&dir, SealChoice::Private);
            let next = keys.agent_dir(id).join("secrets.next.json");
            std::fs::write(
                &next,
                std::fs::read(keys.agent_dir(id).join("secrets.json")).unwrap(),
            )
            .unwrap();
            let err = e.opt_in(&api, id, None).await.unwrap_err().to_string();
            assert!(err.contains("rotation is in progress"), "{err}");
            assert!(api.calls.lock().unwrap().is_empty());
            std::fs::remove_file(&next).unwrap();
            // The rotation's record, signed by the current key or by another
            let encryption = keys.ring().ensure_encryption_key(id).unwrap();
            let file = keys.state_key_file(id).unwrap().unwrap();
            let signer = if valid {
                keys.signing(id).unwrap()
            } else {
                agora_agentkit::crypto::generate_keypair().0
            };
            let record = file
                .state_key()
                .unwrap()
                .wrap(
                    id,
                    file.key_gen,
                    file.wrap_seq,
                    &agora_agentkit::envelope::EncryptionPublicKey::from(&encryption),
                    &signer,
                )
                .unwrap();
            std::fs::write(
                keys.agent_dir(id).join("state_key.wrap.json"),
                serde_json::to_vec(&record).unwrap(),
            )
            .unwrap();
            e.opt_in(&api, id, None).await.unwrap();
            let posted = api.keys.lock().unwrap()[0].clone();
            assert_eq!(posted == record, valid, "valid: {valid}");
            assert!(posted.verify(&keys.signing(id).unwrap().verifying_key()));
        }
    }

    /// Entries Agora already holds are filed as sent with their blobs
    /// kept (agentkit keeps acknowledged blobs).
    #[tokio::test]
    async fn entries_agora_holds_are_filed_as_sent_with_their_blobs_kept() {
        let root = crate::sealed::TestDir::new("opt-in-held");
        let api = fake::guarded(&root);
        let (id, keys, dir) = agent(&root, &["a", "b"]).await;
        let e = enroller(&root, &keys);
        e.chain(id, false).await.unwrap();
        answer(&dir, SealChoice::Private);
        let seal = AgentSeal::new(&dir, id);
        let local = super::super::check::local_entries(&seal).await.unwrap();
        *api.chain.lock().unwrap() = Ok(local[..2].to_vec());
        let done = e.opt_in(&api, id, None).await.unwrap();
        assert_eq!(done.held_by_agora, 2);
        assert_eq!(seal.sent().await.unwrap().len(), 2);
        for item in seal.sent().await.unwrap() {
            if let Some(b) = item.entry.entry.blob {
                assert!(
                    dir.join(SEALED_DIR)
                        .join("blobs")
                        .join(b.sha256.to_hex())
                        .exists()
                );
            }
        }
    }

    /// A provisioned agent chained later by the fleet pass still gets a
    /// genesis of class `none`, read back from the chain.
    #[tokio::test]
    async fn a_provisioned_agent_chained_later_has_a_none_genesis() {
        let root = crate::sealed::TestDir::new("chain-provisioned");
        let (id, keys, dir) = agent(&root, &["a"]).await;
        std::fs::write(dir.join(PROVISIONED), b"x").unwrap();
        let report = enroller(&root, &keys).chain(id, false).await.unwrap();
        assert_eq!(report.origin, "none");
        let seal = AgentSeal::new(&dir, id);
        let genesis = super::super::check::local_entries(&seal).await.unwrap()[0].clone();
        let origin = seal
            .open_origin(&keys.seal_keys(id).unwrap(), &genesis.entry)
            .await
            .unwrap();
        assert_eq!(origin.class, GenesisClass::None);
    }

    /// A stopped agent's pre-loss copy waits for its resume.
    #[tokio::test]
    async fn a_stopped_agents_prior_copy_waits() {
        let root = crate::sealed::TestDir::new("prior-stopped");
        let (id, keys, dir) = agent(&root, &["a"]).await;
        let recovered = root.join("recovered");
        std::fs::create_dir_all(recovered.join("tarn")).unwrap();
        std::fs::write(
            recovered.join("tarn/MEMORY.json"),
            r#"{"content":"synthetic"}"#,
        )
        .unwrap();
        let e = Enroller {
            restore: Some(RestoreOffer::for_agents(
                &[("tarn", None, id)],
                &recovered,
                None,
            )),
            ..enroller(&root, &keys)
        };
        e.chain(id, false).await.unwrap();
        let ledger = serde_json::json!({ "format": 1, "asks": [{"at": "2026-10-12T09:00:00Z",
            "offer_version": 1, "model": "m", "chars": 9, "order": ["show", "decline"],
            "order_seed": 0, "constrained": true, "attempts": 1,
            "outcome": {"kind": "answered", "choice": "show", "reason": "", "memory_note_written": false}}] });
        std::fs::write(RestoreLedger::path(&dir), ledger.to_string()).unwrap();
        crate::sealed::write_private(
            &keys.agent_dir(id).join(crate::sealed::stop::DURABLE),
            &serde_json::to_vec(&serde_json::json!({ "at": Utc::now(), "note": "asked" })).unwrap(),
        )
        .unwrap();
        assert!(e.append_prior(id).await.unwrap().is_none());
        std::fs::remove_file(keys.agent_dir(id).join(crate::sealed::stop::DURABLE)).unwrap();
        assert!(e.append_prior(id).await.unwrap().is_some());
        let line = crate::sealed::notice::Notice::load(&dir)
            .unwrap()
            .unwrap()
            .line()
            .unwrap();
        assert!(line.contains("was added to your sealed history."), "{line}");
    }

    /// Synthetic timing for planning the fleet run: the cost of one sealed
    /// version of a ~120 KB state. `cargo test -p agora-seed --release
    /// per_version_cost -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore]
    async fn per_version_cost() {
        let root = crate::sealed::TestDir::in_memory("chain-timing");
        let words = [
            "agora", "memory", "council", "thread", "vote", "proposal", "runner",
        ];
        let mut seed = 7u64;
        let mut text = |n: usize| {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    words[(seed >> 33) as usize % words.len()]
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        let versions: Vec<String> = (0..20).map(|_| text(16_000)).collect();
        let refs: Vec<&str> = versions.iter().map(String::as_str).collect();
        let (id, keys, _) = agent(&root, &refs).await;
        let started = std::time::Instant::now();
        let report = enroller(&root, &keys).chain(id, false).await.unwrap();
        let took = started.elapsed();
        println!(
            "{} versions of ~{} bytes in {:.2?}: {:.0} ms a version",
            report.entries - 1,
            report.current_bytes,
            took,
            took.as_secs_f64() * 1000.0 / (report.entries - 1) as f64
        );
    }
}
