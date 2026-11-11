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
use super::keys::{StateKeyFile, x25519_public_hex};
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
    /// Archives left out: unreadable, a newer envelope, or the same state
    /// as the version before.
    pub skipped: usize,
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
            "origin {}, archives {} ({} bytes, {} skipped), pre-loss copy {}, current {} bytes, \
             entries {}",
            self.origin,
            self.archives,
            self.archive_bytes,
            self.skipped,
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

/// An archive's state, from its `{format, state}` envelope. `None`: not
/// one this binary reads. Its content is never logged.
fn archived_state(path: &Path) -> Option<serde_json::Value> {
    envelope_state(path)
        .ok()
        .filter(|(format, _)| *format <= agora_agentkit::reactor::storage::FORMAT)
        .map(|(_, state)| state)
}

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

        // Where a partial chain stopped: the newest state it sealed, and
        // whether the pre-loss copy is in
        let mut done_until: Option<i64> = None;
        let mut prior_done = false;
        let mut last = None;
        match &partial {
            Some(head) => {
                let entries = super::check::local_entries(&seal)
                    .await
                    .map_err(anyhow::Error::msg)?;
                done_until = entries
                    .iter()
                    .filter(|e| e.entry.kind == StateEntryKind::Snapshot)
                    .map(|e| e.entry.state_at)
                    .max();
                prior_done = entries
                    .iter()
                    .any(|e| e.entry.kind == StateEntryKind::ArchivedPrior);
                last = head.plaintext_sha256;
                report.resumed = true;
                tracing::warn!(
                    event_type = "sealed_chain_resumed",
                    agent_id = %id,
                    entries = entries.len(),
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
        for (n, archive) in archives.iter().enumerate() {
            if done_until.is_some_and(|t| archive.mtime.timestamp_micros() <= t) {
                report.resumed_past += 1;
                continue;
            }
            let Some(state) = archived_state(&archive.path) else {
                report.skipped += 1;
                continue;
            };
            let hash = plaintext_hash(&state);
            if last == Some(hash) {
                report.skipped += 1;
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
            .with_context(|| format!("sealing archive {}", archive.path.display()))?;
            record_sizes(&agent_dir, head, archive.mtime, &state);
            last = Some(hash);
            if n == 0 {
                super::crash::point("chain:archive")?;
            }
        }
        if last != Some(plaintext_hash(&current)) {
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
        self.roster(id, head);
        report.entries = super::check::local_entries(&seal)
            .await
            .map_err(anyhow::Error::msg)?
            .len();
        tracing::info!(
            event_type = "sealed_chained",
            agent_id = %id,
            archives = report.archives,
            archive_bytes = report.archive_bytes,
            skipped = report.skipped,
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
    fn roster(&self, id: AgentId, head: HeadRef) {
        let recorded = self
            .data_dir()
            .map_err(|e| std::io::Error::other(e.to_string()))
            .and_then(|d| super::roster::record(d, id, head));
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
    /// state with the head's snapshot). Once: never if the chain holds an
    /// `archived_prior` already, and a run cut between the two appends is
    /// finished. Its genesis, sealed with what was true then
    /// (`known_break`, `prior_copy_held` as it was), never changes: that
    /// the copy came later is read from the chain. Under the agent's
    /// `sealed/lock`, with no session in progress; never over a state that
    /// doesn't match its last sealed version (the start check decides
    /// about that one). `None`: nothing to do.
    pub async fn append_prior(&self, id: AgentId) -> anyhow::Result<Option<Report>> {
        let Some(offer) = self.restore.as_ref() else {
            return Ok(None);
        };
        let agent_dir = self.state_dir.join(id.to_string());
        let seal = AgentSeal::new(&agent_dir, id);
        if !seal.is_enrolled().await? {
            return Ok(None);
        }
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
        let head = seal.head().await?.context("no sealed head")?;
        let head_kind = items
            .iter()
            .find(|i| i.entry.entry_hash() == head.head.entry_hash)
            .map(|i| i.entry.entry.kind);
        let has_prior = items
            .iter()
            .any(|i| i.entry.entry.kind == StateEntryKind::ArchivedPrior);
        // A run cut between the copy and the snapshot after it
        let cut_short = head_kind == Some(StateEntryKind::ArchivedPrior);
        if has_prior && !cut_short {
            return Ok(None);
        }
        // The state must be the chain's last sealed version
        let local = plaintext_hash(&current);
        let sealed = if cut_short {
            items
                .iter()
                .filter(|i| i.entry.entry.kind == StateEntryKind::Snapshot)
                .max_by_key(|i| i.entry.entry.seq)
                .and_then(|i| i.plaintext_sha256)
        } else {
            head.plaintext_sha256
        };
        anyhow::ensure!(
            sealed == Some(local),
            "agent {id}'s state doesn't match its last sealed version; its pre-loss copy is not \
             appended until the start check's finding is resolved"
        );
        let keys = self
            .keys
            .seal_keys(id)
            .context("sealing keys for the agent")?;
        let policy = KeyframePolicy::default();
        let lock = super::session::take(&self.state_dir, id, false).await?;
        if !cut_short {
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
        }
        let now = Utc::now();
        let head = super::session::append(
            &seal,
            &lock,
            &keys,
            &policy,
            StateEntryKind::Snapshot,
            &current,
            now,
        )
        .await
        .context("sealing the current state after the pre-loss copy")?;
        drop(lock);
        record_sizes(&agent_dir, head, now, &current);
        self.roster(id, head);
        tracing::warn!(
            event_type = "sealed_prior_appended",
            agent_id = %id,
            head = %head.entry_hash,
            seq = head.seq,
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

    /// Start uploading a chained agent whose sealed-state ledger says yes.
    /// Under `guard`'s opt-in hold, which lets only the opt-in calls
    /// through for it: Agora must hold no chain of it, or a prefix of this
    /// one (whose entries are then filed as sent); its state key record is
    /// posted (once: an earlier attempt's is found); then it is marked
    /// uploading, which commits it. The caller drains next. A crash at any
    /// step is finished by a rerun.
    ///
    /// Never against the agent's latest answer: only a yes is acted on,
    /// unless `against` carries the operator's note for opting in an agent
    /// whose latest answer is `decline` (the agent is shown it). With no
    /// answer on file, never.
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
        let overridden = match (ledger.allows(true), ledger.opted_in(), against) {
            (_, true, _) => None,
            (Err(_), _, Some(note)) => Some(note),
            _ => return Err(Skip::NotOptedIn),
        };
        let durable = self.keys.agent_dir(id).join(uploading::DURABLE);
        match uploading::state(&agent_dir, &self.keys, id) {
            Marker::Uploading(_) => {
                return Ok(OptedIn {
                    already: true,
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
        if super::session::no_session(&self.state_dir, id, false).is_err() {
            return Err(Skip::InSession);
        }
        let signing = self
            .keys
            .signing(id)
            .context("the agent has no signing key")?;
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

        let _hold = guard.opting_in(id);
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
        let on_agora: std::collections::HashSet<Sha256Hex> =
            server.iter().map(|e| e.entry_hash()).collect();
        let mut held_by_agora = 0;
        for item in seal.outbox().await.context("reading the outbox")? {
            if on_agora.contains(&item.entry.entry_hash()) {
                seal.mark_sent(item.ord)
                    .await
                    .context("filing an entry Agora holds as sent")?;
                held_by_agora += 1;
            }
        }

        // The state key record, wrapped to the agent's X25519 key
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
        let state_key = state_file.state_key().context("the state key cache")?;
        let held = guard
            .state_keys(id, &signing)
            .await
            .map_err(anyhow::Error::msg)
            .context("reading the server's state keys")?;
        let vk = signing.verifying_key();
        // A record of this key wrapped to the current X25519 key, signed
        // by the agent: an earlier attempt's. One wrapped to a retired key
        // doesn't count: the agent must be able to open it now.
        let posted = held.iter().any(|r| {
            r.key_gen == state_file.key_gen
                && r.wrapped_to_pub == x25519_public_hex(&encryption)
                && r.verify(&vk)
                && StateKey::unwrap(r, &encryption).is_ok_and(|k| k.expose() == state_key.expose())
        });
        if !posted {
            let record = state_key
                .wrap(
                    id,
                    state_file.key_gen,
                    state_file.wrap_seq,
                    &agora_agentkit::envelope::EncryptionPublicKey::from(&encryption),
                    &signing,
                )
                .context("wrapping the state key")?;
            guard
                .post_state_key(id, &signing, &record)
                .await
                .map_err(anyhow::Error::msg)
                .context("posting the state key record")?;
        }
        super::crash::point("opt_in:posted")?;
        // The record is on Agora: only now may an entry follow it
        uploading::set(&agent_dir, &self.keys, id, uploading::UploadingBy::OptIn)
            .context("marking it uploading")?;
        if let Some(note) = overridden {
            super::notice::Notice::opted_in_against(&agent_dir, note)
                .context("writing the notice")?;
            tracing::warn!(
                event_type = "sealed_opted_in_against_answer",
                agent_id = %id,
                note_chars = note.chars().count(),
                "the operator opted in an agent whose latest answer is decline"
            );
        }
        tracing::warn!(
            event_type = "sealed_opted_in",
            agent_id = %id,
            posted = !posted,
            held_by_agora,
            "a chained agent that said yes now uploads; its whole local chain follows"
        );
        Ok(OptedIn {
            already: false,
            posted: !posted,
            held_by_agora,
        })
    }

    /// Resume a stopped agent's uploads. The uploaded line goes on from
    /// the last entry Agora holds (the newest in `sent/`: a stop drains
    /// first) with **one new snapshot of the current state**; the versions
    /// saved while stopped are set aside here as `.displaced` (kept, with
    /// their blobs, opened by the `memory` tool as a side line) and never
    /// uploaded, and the agent is told at its next session. An agent that
    /// isn't marked uploading has nothing to continue: only its stop is
    /// cleared. Never a new genesis.
    ///
    /// The caller holds the runner lock exclusively (`sealed-enroll`), so
    /// no sweep saves this agent mid-resume. A state that changed outside
    /// the store while stopped needs `note`, which the agent is shown, as
    /// after `sealed-accept`.
    pub async fn resume(&self, id: AgentId, note: Option<&str>) -> anyhow::Result<Report> {
        let note = note.map(str::trim).filter(|n| !n.is_empty());
        let agent_dir = self.state_dir.join(id.to_string());
        let seal = AgentSeal::new(&agent_dir, id);
        // A stop kept in the secrets tree stands for a lost sealed/stopped
        super::stop::reconcile(&agent_dir, &self.keys, id)
            .context("putting back the stop kept in the secrets tree")?;
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
        let uploading = super::uploading::is_uploading(&agent_dir, &self.keys, id);
        let intent: Option<Resuming> =
            match std::fs::read(resuming_path(&agent_dir)) {
                Ok(bytes) => Some(serde_json::from_slice(&bytes).with_context(|| {
                    format!("{} unreadable", resuming_path(&agent_dir).display())
                })?),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
                Err(e) => return Err(e.into()),
            };
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
        let new_head = if intent.is_some() || (uploading && !waiting.is_empty()) {
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
            let done = seal.head().await?.filter(|h| {
                h.head.seq == intent.onto.seq + 1
                    && !intent.aside.contains(&h.head.entry_hash)
                    && h.keyframe.as_ref().is_some_and(|k| {
                        k.entry_hash() == h.head.entry_hash
                            && k.entry.prev_hash == Some(intent.onto.entry_hash)
                    })
            });
            match done {
                Some(h) => h.head,
                None => {
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
                    head
                }
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
            head
        } else {
            head.head
        };
        super::crash::point("resume:sealed")?;
        drop(lock);
        // Marked in the sealed format again (agentkit takes the lock itself)
        let marked = self
            .mark(id)
            .await
            .context("marking the state as sealed again")?;
        self.roster(id, marked);
        if kept_local > 0 || changed {
            super::notice::Notice::resumed_kept(&agent_dir, note, kept_local)
                .context("writing the notice")?;
        }
        // The durable copy first: left behind, it would put the stop back
        super::stop::clear_durable(&self.keys, id)
            .context("setting the stop in the secrets tree aside")?;
        if super::stop::is_stopped(&agent_dir) {
            super::stop::Stopped::clear(&agent_dir).context("setting sealed/stopped aside")?;
        }
        if std::fs::exists(resuming_path(&agent_dir))? {
            let at = Utc::now().format(super::STAMP_FORMAT);
            std::fs::rename(
                resuming_path(&agent_dir),
                resuming_path(&agent_dir).with_file_name(format!("resuming.done.{at}.json")),
            )?;
        }
        tracing::warn!(
            event_type = "sealed_state_resumed",
            agent_id = %id,
            head = %new_head.entry_hash,
            seq = new_head.seq,
            kept_local,
            "a stopped agent's uploads resumed; the versions saved while stopped stay here"
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
}

/// The pre-loss copy, parsed. Its content is never logged.
fn read_prior(path: &Path) -> anyhow::Result<serde_json::Value> {
    serde_json::from_slice(&std::fs::read(path).context("reading the pre-loss copy")?)
        .map_err(|e| anyhow::anyhow!("the pre-loss copy isn't JSON ({:?})", e.classify()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consent::sealed_state::ledger::{SealAsk, SealOutcome};
    use crate::consent::sealed_state::prompt::SealChoice;
    use crate::sealed::api::fake;
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
    async fn agent(root: &Path, memories: &[&str]) -> (AgentId, KeyStore, PathBuf) {
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

    fn enroller(root: &Path, keys: &KeyStore) -> Enroller {
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
        assert_eq!(report.skipped, 2, "{report}");
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
        let calls = api.calls.lock().unwrap().len();
        let done = e.opt_in(&api, id, None).await.unwrap();
        assert!(!done.posted);
        assert!(crate::sealed::uploads(&dir, &keys, id));
        assert_eq!(
            api.calls.lock().unwrap().len(),
            calls + 2,
            "chain, state_keys: no post"
        );
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
                assert!(e.resume(id, None).await.is_err(), "{point}");
            }
            let report = e.resume(id, None).await.unwrap();
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
        let report = e.resume(id, None).await.unwrap();
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
        let err = e.resume(id, None).await.unwrap_err().to_string();
        assert!(
            err.contains("changed while its backups were stopped"),
            "{err}"
        );
        e.resume(id, Some("I checked the edit.")).await.unwrap();
        assert_eq!(
            crate::sealed::notice::Notice::load(&dir)
                .unwrap()
                .unwrap()
                .line()
                .unwrap(),
            "While your backups were stopped, your state changed; your operator resumed them \
             with this note: I checked the edit."
        );
        let local = plaintext_hash(&FsStorage::new(&state_dir).load_raw(id).await.unwrap().value);
        assert!(matches!(
            seal.recover(Some(&local)).await.unwrap(),
            Recovery::Intact(_)
        ));
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
