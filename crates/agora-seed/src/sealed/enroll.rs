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

/// A pre-loss copy's pair finished ([`Enroller::finish_prior`]): the new
/// head, and what the agent's files were once it was written, taken under
/// its lock, so a runner that loaded the agent before carries both forward
/// rather than reading its own write as another process's save.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PriorDone {
    pub head: HeadRef,
    pub seen: super::session::Seen,
}

/// What one chaining did, or would do: counts and sizes only.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    /// A pre-loss copy appended after the chain began: the head it left.
    pub carried: Option<PriorDone>,
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
    /// Versions not uploaded before the stop or saved while uploads were
    /// stopped, set aside by a resume.
    pub kept_local: usize,
    /// Entries in the chain at the end (genesis included).
    pub entries: usize,
    /// A session journal is beside the state (an interrupted or failed
    /// session's, kept for the agent's next session): left untouched; the
    /// agent is chained all the same.
    pub journal: bool,
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
        )?;
        if self.journal {
            write!(
                f,
                ", an unfinished session's journal is there (left as it is for its next session)"
            )?;
        }
        Ok(())
    }
}

/// Why an agent was not chained or opted in.
#[derive(Debug, thiserror::Error)]
pub enum Skip {
    #[error("already chained")]
    Chained,
    #[error("no state here")]
    NoState,
    #[error("not chained yet")]
    NotChained,
    /// Uploads start or resume only on the agent's yes in force
    /// ([`SealLedger::in_force`]): no flag overrides this.
    #[error("not opted in: {0}")]
    NotOptedIn(crate::consent::sealed_state::ledger::NotNow),
    #[error("its uploads are stopped; `sealed-enroll --agent <id>` resumes them")]
    Stopped,
    /// Another process holds the agent (an operator's command; a runner's
    /// session in a dry run): not chained this time, a rerun chains it.
    #[error("held by another process; run `sealed-enroll` again when it has finished")]
    Busy,
    /// It can't be chained until the operator repairs something
    /// ([`super::unchainable`]): recorded, and left out of every run until
    /// a later `sealed-enroll` chains it.
    #[error("can't be chained ({}): {error:#}", class.as_str())]
    Cannot {
        class: super::unchainable::Class,
        error: anyhow::Error,
    },
    #[error(transparent)]
    Failed(#[from] anyhow::Error),
}

impl Skip {
    /// A short name for the count table and the structured events.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Chained => "already_chained",
            Self::NoState => "no_state",
            Self::NotChained => "not_chained",
            Self::NotOptedIn(_) => "not_opted_in",
            Self::Stopped => "stopped",
            Self::Busy => "busy",
            Self::Cannot { class, .. } => class.as_str(),
            Self::Failed(_) => "failed",
        }
    }

    fn cannot(class: super::unchainable::Class, error: anyhow::Error) -> Self {
        Self::Cannot { class, error }
    }

    /// A [`Self::Cannot`] whose cause looks like the machine's, not the
    /// agent's: an I/O or permission error on the way (a file or directory
    /// that exists but can't be read, a disk error), or a secrets tree
    /// that is missing altogether ([`IoShaped`]). A missing file or one
    /// that reads but doesn't parse is the agent's.
    pub fn io_shaped(&self) -> bool {
        let Self::Cannot { error, .. } = self else {
            return false;
        };
        error.chain().any(|c| {
            c.is::<IoShaped>()
                || c.downcast_ref::<std::io::Error>().is_some_and(|e| {
                    !matches!(
                        e.kind(),
                        std::io::ErrorKind::NotFound | std::io::ErrorKind::InvalidData
                    )
                })
        })
    }

    /// What the record of agents that can't be chained
    /// ([`super::unchainable`]) takes from this outcome of a chain.
    pub fn found(&self) -> super::unchainable::Found {
        use super::unchainable::Found;
        match self {
            Self::Chained | Self::NoState => Found::Settled,
            Self::Cannot { class, .. } => Found::Cannot(*class),
            _ => Found::Unknown,
        }
    }
}

/// A cause [`Skip::io_shaped`] reads as the machine's, where no I/O error
/// carries it (a whole tree missing).
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct IoShaped(pub String);

/// Why the agent has no signing key: its `secrets.json` there but
/// unreadable, or the whole secrets tree missing, is the machine's; a file
/// missing alone, or one that doesn't parse, is the agent's.
fn no_signing_key(keys: &KeyStore, id: AgentId) -> anyhow::Error {
    let dir = keys.agent_dir(id);
    if let Some(root) = dir.parent()
        && !std::fs::exists(root).unwrap_or(true)
    {
        return anyhow::Error::new(IoShaped(format!(
            "the secrets tree {} is missing",
            root.display()
        )));
    }
    match std::fs::File::open(dir.join("secrets.json")) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            anyhow::Error::new(e).context("its secrets.json can't be read, so no signing key")
        }
        _ => anyhow::anyhow!("the agent has no signing key"),
    }
}

/// From this many agents a chain pass can't chain for I/O-shaped reasons
/// ([`Skip::io_shaped`]), none is recorded and the pass fails: a fault of
/// the disk or the secrets tree is not the agents', and recorded it would
/// leave them out of every run with one alert instead of refusing it. The
/// strays rule's number ([`super::check::REFUSED_AGENTS_STOP_RUN`]).
pub const IO_SHAPED_MAX: usize = super::check::REFUSED_AGENTS_STOP_RUN;

/// The agents with a state that a chain pass left without a chain
/// ([`Enroller::settle_pass`]): ids by why, never anything of a state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Unchained {
    /// By [`Skip::as_str`]: a class of [`super::unchainable::Class`],
    /// `busy` or `failed`.
    pub by_class: std::collections::BTreeMap<&'static str, Vec<AgentId>>,
    /// Those that can't be chained until repaired: on the record
    /// ([`super::unchainable`]), left out of every run with an alert, not
    /// counted by the start's strays rule.
    pub recorded: usize,
    /// The rest (held by another process, a failure a rerun may get
    /// past): three or more refuse the runner's start.
    pub strays: usize,
    remedies: std::collections::BTreeMap<&'static str, &'static str>,
    dry_run: bool,
}

impl Unchained {
    pub fn is_empty(&self) -> bool {
        self.by_class.is_empty()
    }

    pub fn agents(&self) -> usize {
        self.by_class.values().map(Vec::len).sum()
    }
}

impl std::fmt::Display for Unchained {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            return Ok(());
        }
        writeln!(
            f,
            "{} agent(s) with a state {} without a chain:",
            self.agents(),
            if self.dry_run {
                "would be left"
            } else {
                "are left"
            }
        )?;
        for (class, ids) in &self.by_class {
            let ids: Vec<String> = ids.iter().map(ToString::to_string).collect();
            writeln!(f, "  {class} ({}): {}", ids.len(), ids.join(" "))?;
            if let Some(remedy) = self.remedies.get(class) {
                writeln!(f, "    to chain: {remedy}")?;
            }
        }
        if self.recorded > 0 {
            writeln!(
                f,
                "{} of them can't be chained until repaired: {} in {} beside the roster, left out \
                 of every run (never run unchained), with an alert each run, until a later \
                 `sealed-enroll` chains them.",
                self.recorded,
                if self.dry_run {
                    "a run would record them"
                } else {
                    "recorded"
                },
                super::unchainable::RECORD,
            )?;
        }
        if self.strays > 0 {
            writeln!(
                f,
                "{} of them were not reached or failed: run `sealed-enroll` again. They count at \
                 the runner's start, which refuses to run (exit {}) while {} or more agents have \
                 a state and no chain.",
                self.strays,
                super::REFUSED_RUN_EXIT,
                super::check::REFUSED_AGENTS_STOP_RUN,
            )?;
        }
        Ok(())
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

pub const LEFT_OUT: &str = "left_out.json";

/// What `sealed/left_out.json` records; empty when there is none. The
/// `memory` tool's `log` says the count, once.
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

/// `sealed/chain_began.json`: the head the enrolment that began the chain
/// left, written once, after the chain marker. The versions on the main
/// line up to it were sealed by the enrolment (from the archives and the
/// state as found), not by a save since: a change between two of them
/// predates the chain ([`predates_chain`]). Hashes and a date only.
const CHAIN_BEGAN: &str = "chain_began.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainBegan {
    pub head: HeadRef,
    pub at: DateTime<Utc>,
}

fn chain_began_path(agent_dir: &Path) -> PathBuf {
    agent_dir
        .join(agora_agentkit::state::storage::SEALED_DIR)
        .join(CHAIN_BEGAN)
}

/// The record of the enrolment that began the chain, if it can be read. A
/// chain without one (begun before the record existed, or its write
/// failed) has none of its versions taken as the enrolment's.
pub fn chain_began(agent_dir: &Path) -> Option<ChainBegan> {
    serde_json::from_slice(&std::fs::read(chain_began_path(agent_dir)).ok()?).ok()
}

/// Whether every one of `versions` was sealed by the enrolment that began
/// the chain: each on `line` (the main line, genesis first) at or before
/// the head [`chain_began`] names, which must itself be on it. A line that
/// no longer runs through that head (a fork resolved, a new genesis) has
/// none.
pub fn predates_chain(
    agent_dir: &Path,
    line: &[agora_agentkit::state::SignedStateEntry],
    versions: &[Sha256Hex],
) -> bool {
    let Some(began) = chain_began(agent_dir) else {
        return false;
    };
    let Some(pos) = line
        .iter()
        .position(|e| e.entry.seq == began.head.seq && e.entry_hash() == began.head.entry_hash)
    else {
        return false;
    };
    versions
        .iter()
        .all(|v| line[..=pos].iter().any(|e| e.entry_hash() == *v))
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
    /// The versions on its main line at the opt-in (snapshots, as the
    /// `memory` tool counts them): all uploaded once the outbox is empty
    /// after the command's drain, which the agent's notice then says
    /// ([`super::notice::Notice::opted_in_uploaded`]).
    pub versions: usize,
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

// Read-only, for the `memory` tool's labels (`history.rs`): which
// versions a resume set aside, and when.
/// The versions each resume set aside as saved while uploads were
/// stopped, with when it ran: `resuming.json` (a resume under way) and
/// every `resuming.done.*.json`. Hashes only; an unreadable record is
/// skipped (its versions are then labelled as any other side line).
pub fn resumed_aside(agent_dir: &Path) -> Vec<(DateTime<Utc>, Vec<Sha256Hex>)> {
    let dir = agent_dir.join(agora_agentkit::state::storage::SEALED_DIR);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name == RESUMING || (name.starts_with("resuming.done.") && name.ends_with(".json"))
        })
        .filter_map(|e| serde_json::from_slice::<Resuming>(&std::fs::read(e.path()).ok()?).ok())
        .map(|r| (r.at, r.aside))
        .collect()
}

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

    /// The agents the operator may start uploading now: a yes in force
    /// in the sealed-state ledger with no re-offer waiting
    /// ([`SealLedger::in_force`]), who don't upload yet (no uploading
    /// marker, or one in doubt), and aren't stopped: chained or not.
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
                Ok(l) if l.in_force().starts_uploads().is_ok() => out.push(id),
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
        use super::unchainable::Class;
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
            return match self.append_prior(id).await {
                Ok(Some(report)) => Ok(report),
                Ok(None) => Err(Skip::Chained),
                Err(e) if e.is::<super::session::Busy>() => Err(Skip::Busy),
                Err(e) => Err(e.into()),
            };
        }
        // An older runner's chain marker the start pass couldn't move
        // (`uploading::migrate_markers` logs the failure and goes on): the
        // agent is chained, and may upload. Chained again from here its
        // chain would be read as one cut short and added to
        if !dry_run
            && seal
                .has_legacy_marker()
                .await
                .context("reading the older chain marker")?
        {
            return Err(Skip::Failed(anyhow::anyhow!(
                "an older runner's chain marker (sealed/enrolled) is still here: it couldn't be \
                 moved to sealed/chained (a sealed_state_marker_migration_failed event says \
                 why). The agent is chained already and is not chained again; fix what stopped \
                 the move (permissions, the disk) and run this again"
            )));
        }
        let storage = FsStorage::new(&self.state_dir);
        let current = match storage.load_raw(id).await {
            Ok(loaded) => loaded.value,
            Err(FsStorageError::NotFound(_)) => return Err(Skip::NoState),
            Err(e) => {
                return Err(Skip::cannot(
                    Class::StateUnloadable,
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
                Some(offer) => {
                    let name =
                        name_of(&current).map_err(|e| Skip::cannot(Class::StateUnloadable, e))?;
                    origin(Some(offer), &name, id, &agent_dir).await?
                }
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
        // A journal beside the state is an interrupted or failed session's,
        // kept until the agent's next session completes (`interrupted`). It
        // is not state: the state on disk is the last one saved, and that
        // is what is chained. The journal is left untouched, so the agent
        // is still told at its next session. No session can be live: the
        // caller holds `runner.lock` exclusively, and the agent's session
        // lock is taken below (a dry run only reads)
        report.journal =
            std::fs::exists(FsStorage::new(&self.state_dir).journal(id).path()).unwrap_or(false);
        // What no rerun can chain until the operator repairs it, found
        // before anything is written, and by a dry run too
        if self.keys.signing(id).is_none() {
            return Err(Skip::cannot(
                Class::NoSigningKey,
                no_signing_key(&self.keys, id),
            ));
        }
        let has_key = |keys: &KeyStore| -> anyhow::Result<bool> {
            Ok(keys
                .state_key_file(id)
                .context("the state key cache")?
                .is_some())
        };
        // A chain begun before (a head) whose state key is gone: a new key
        // would leave every version already sealed unopenable, with nothing
        // the start check (which compares hashes) could tell
        let key_lost = || {
            Skip::cannot(
                Class::StateKeyLost,
                anyhow::anyhow!(
                    "a chain was begun (sealed/head.json) but its state key                      (state_key.json) is gone; no new key is made over it"
                ),
            )
        };
        if dry_run {
            if seal
                .head()
                .await
                .context("reading the sealed head")?
                .is_some()
                && !has_key(&self.keys)?
            {
                return Err(key_lost());
            }
            if let Some(prior) = &origin.prior {
                read_prior(prior).map_err(|e| Skip::cannot(Class::PriorCopyUnreadable, e))?;
            }
            return Ok(report);
        }
        // The agent's session lock and its `sealed/lock` for the whole
        // chain: every write below goes through it
        let lock = super::session::take_idle(&self.state_dir, id)
            .await
            .map_err(|e| match e.downcast_ref::<super::session::Busy>() {
                Some(_) => Skip::Busy,
                None => Skip::Failed(e.context("taking the agent's locks")),
            })?;
        // A chain begun before but never finished is resumed
        let partial = seal.head().await.context("reading the sealed head")?;
        // The state key, cached here; nothing of it leaves the machine
        // until the agent opts in. Made only for a chain not yet begun
        if !has_key(&self.keys)? {
            if partial.is_some() {
                return Err(key_lost());
            }
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
        if partial.is_some() {
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
        // The pre-loss copy is read before the genesis is sealed: one that
        // can't be read leaves no chain begun
        let prior = match origin.prior.as_ref().filter(|_| !prior_done) {
            Some(path) => {
                Some(read_prior(path).map_err(|e| Skip::cannot(Class::PriorCopyUnreadable, e))?)
            }
            None => None,
        };
        if partial.is_none() {
            let first_at = archives
                .first()
                .map(|a| a.mtime)
                .or_else(|| mtime(&current_path).ok())
                .unwrap_or_else(Utc::now);
            super::session::genesis(&seal, &lock, &keys, &origin.origin, first_at)
                .await
                .context("sealing the genesis")?;
        }
        super::crash::point("chain:genesis")?;
        if let Some(prior) = &prior {
            super::session::append(
                &seal,
                &lock,
                &keys,
                &policy,
                StateEntryKind::ArchivedPrior,
                prior,
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
        // Once: what the enrolment sealed, told from what is saved after
        if chain_began(&agent_dir).is_none() {
            let began = ChainBegan {
                head,
                at: Utc::now(),
            };
            let written = serde_json::to_vec_pretty(&began)
                .map_err(std::io::Error::from)
                .and_then(|b| super::write_private(&chain_began_path(&agent_dir), &b));
            if let Err(e) = written {
                tracing::warn!(
                    event_type = "sealed_chain_began_unrecorded",
                    agent_id = %id,
                    error = %e,
                    "the head the enrolment left couldn't be recorded; a shrink between its \
                     archives is then reported as if it were new"
                );
            }
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
        if let Some(done) = self.finish_prior(id).await? {
            return Ok(Some(Report {
                prior: true,
                prior_later: true,
                carried: Some(done),
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
        // Whatever its journal says (`session::take_idle`): the session
        // lock proves no session is live
        let lock = super::session::take_idle(&self.state_dir, id).await?;
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
        // Read before the agent is told: a copy that can't be read leaves
        // no notice saying it was added
        let prior = read_prior(&prior)?;
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
            &prior,
            prior_copy_at(),
        )
        .await
        .context("sealing the pre-loss copy")?;
        super::crash::point("prior:appended")?;
        drop(lock);
        let done = self
            .finish_prior(id)
            .await?
            .context("the pre-loss copy was sealed but isn't the head")?;
        let head = done.head;
        tracing::info!(
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
            carried: Some(done),
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
    /// At run start `main.rs`'s start check calls it (through
    /// [`Self::append_prior`]) before the agent is judged.
    pub async fn finish_prior(&self, id: AgentId) -> anyhow::Result<Option<PriorDone>> {
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
        let lock = super::session::take_idle(&self.state_dir, id).await?;
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
        // Still under the lock: no one else's save can be in it
        let seen = super::session::Seen::of(&agent_dir);
        drop(lock);
        record_sizes(&agent_dir, head, now, &current);
        self.roster(id, head).await;
        Ok(Some(PriorDone { head, seen }))
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

    /// What a chain pass ([`Self::chain_all`]) left without a chain, and
    /// (not in a dry run) the record of agents that can't be chained
    /// brought up to it ([`super::unchainable::settle`]). `sealed-enroll`
    /// prints it and exits nonzero unless it is empty.
    pub fn settle_pass(
        &self,
        results: &[(AgentId, Result<Report, Skip>)],
        dry_run: bool,
    ) -> anyhow::Result<Unchained> {
        use super::unchainable::Found;
        let mut left = Unchained {
            dry_run,
            ..Unchained::default()
        };
        // Fleet-shaped I/O: nothing recorded, and the pass fails
        let io: Vec<AgentId> = results
            .iter()
            .filter(|(_, r)| r.as_ref().is_err_and(Skip::io_shaped))
            .map(|(id, _)| *id)
            .collect();
        if io.len() >= IO_SHAPED_MAX {
            let ids: Vec<String> = io.iter().map(ToString::to_string).collect();
            tracing::error!(
                event_type = "sealed_enroll_io_shaped",
                agents = io.len(),
                agent_ids = %ids.join(","),
                "agents couldn't be chained for I/O or permission errors; none recorded as \
                 unchainable"
            );
            anyhow::bail!(
                "{} agent(s) couldn't be chained for what looks like an I/O or permission error \
                 (a file or directory there but unreadable, or the secrets tree missing): {}. \
                 None is recorded in {}: a fault of the disk or the secrets tree is not the \
                 agents', and the next start refuses to run them unchained. Fix it (permissions \
                 on secrets/ and state/, the disk, the mount) and run `sealed-enroll` again",
                io.len(),
                ids.join(", "),
                super::unchainable::RECORD
            );
        }
        let mut found = Vec::with_capacity(results.len());
        for (id, result) in results {
            let f = match result {
                Ok(_) => Found::Settled,
                Err(e) => e.found(),
            };
            found.push((*id, f));
            match result {
                Ok(_) | Err(Skip::Chained | Skip::NoState) => {}
                Err(e) => {
                    left.by_class.entry(e.as_str()).or_default().push(*id);
                    match f {
                        Found::Cannot(class) => {
                            left.recorded += 1;
                            left.remedies.insert(class.as_str(), class.remedy());
                        }
                        _ => left.strays += 1,
                    }
                }
            }
        }
        if !dry_run {
            super::unchainable::settle(self.data_dir()?, found)
                .context("recording the agents that can't be chained")?;
        }
        if !left.by_class.is_empty() {
            let classes: std::collections::BTreeMap<&str, usize> = left
                .by_class
                .iter()
                .map(|(c, ids)| (*c, ids.len()))
                .collect();
            let ids: Vec<String> = left
                .by_class
                .values()
                .flatten()
                .map(ToString::to_string)
                .collect();
            tracing::error!(
                event_type = "sealed_enroll_unchained",
                agents = ids.len(),
                recorded = left.recorded,
                strays = left.strays,
                classes = ?classes,
                agent_ids = %ids.join(","),
                dry_run,
                "agents with a state are left without a chain; three or more not on the \
                 unchainable record refuse the runner's start"
            );
        }
        Ok(left)
    }

    /// Start uploading a chained agent whose sealed-state answer in force
    /// is yes ([`SealLedger::in_force`]). In order, each step finished by a
    /// rerun after a cut:
    ///
    /// 1. refused while a key rotation is in progress (`secrets.next.json`:
    ///    the local chain may be half re-signed);
    /// 2. under `guard`'s opt-in hold, which lets only the opt-in calls
    ///    through for it, Agora's chain of it must be empty or the start of
    ///    this one;
    /// 3. its state key record is posted (once: an earlier one is found),
    ///    so it is on Agora before any entry;
    /// 4. the agent's notice, in one write: uploading started, as it
    ///    answered;
    /// 5. it is marked uploading, which commits it (a marker in doubt is
    ///    set aside for the new one: the opt-in is what settles it);
    /// 6. the entries Agora already holds are filed as sent (also on an
    ///    agent already uploading). The caller drains next.
    ///
    /// Only on its yes: never with no answer on file, never over a
    /// `decline`, never while a re-offer waits for its answer, never on an
    /// answer it gave with no question open. Nothing overrides that: the
    /// operator re-opens the question (`sealed-reoffer`) and the agent
    /// answers.
    pub async fn opt_in<A, E>(
        &self,
        guard: &UploadGuard<A, E>,
        id: AgentId,
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
        // The marker's lock from before the stop is read until the marker
        // is written: a `sealed-stop` issued meanwhile waits for this
        // opt-in, then finds the agent marked and stops it. Without it, the
        // stop would find no marker, be refused, and this would mark it
        let marker_lock = uploading::lock(&self.keys, id).map_err(|e| {
            Skip::Failed(anyhow::Error::new(e).context("the uploading marker's lock"))
        })?;
        if super::stop::is_stopped(&agent_dir)
            || super::stop::durable(&self.keys, id).map_or(true, |d| d.is_some())
        {
            return Err(Skip::Stopped);
        }
        let ledger = SealLedger::load(&agent_dir)
            .await
            .context("sealed-state ledger unreadable")?;
        let (_, answered) = ledger
            .in_force()
            .starts_uploads()
            .map_err(Skip::NotOptedIn)?;
        let act = super::notice::OperatorAct::OptedIn {
            answered,
            uploaded: None,
        };
        let signing = self
            .keys
            .signing(id)
            .context("the agent has no signing key")?;
        match uploading::state(&agent_dir, &self.keys, id) {
            Marker::Uploading(_) => {
                drop(marker_lock);
                let held_by_agora = self.file_as_sent(guard, &seal, id, &signing).await?;
                return Ok(OptedIn {
                    already: true,
                    held_by_agora,
                    ..OptedIn::default()
                });
            }
            Marker::Absent => {}
            // In doubt (a `set` cut short, a copy removed, damaged or
            // differing): this opt-in, on the agent's yes in force, is
            // what settles it. The copies are set aside at the commit
            // below; nothing is sent on their say before it
            Marker::Contradictory(_) | Marker::Unreadable(_) => tracing::warn!(
                event_type = "sealed_opt_in_over_doubt",
                agent_id = %id,
                "the agent's uploading marker is in doubt; the opt-in replaces it"
            ),
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
        // The agent's session lock for the rest of the opt-in, whatever its
        // session journal says (`session::take_idle`'s reasoning): this
        // command holds `runner.lock` exclusively, so the lock alone shows
        // no session is live, and a journal an interrupted session left is
        // not state (left untouched; the agent is still told at its next
        // session). Its `sealed/lock` is taken where each write needs it
        let _session =
            super::session::agent_lock(&self.state_dir, id).map_err(|e| match e
                .downcast_ref::<super::session::Busy>()
            {
                Some(_) => Skip::Busy,
                None => Skip::Failed(e.context("taking the agent's session lock")),
            })?;
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
        super::notice::Notice::operator_act(&agent_dir, "", act).context("writing the notice")?;
        super::crash::point("opt_in:noticed")?;
        // The record is on Agora: only now may an entry follow it
        uploading::set_replacing_doubt_locked(
            &agent_dir,
            &self.keys,
            id,
            uploading::UploadingBy::OptIn,
            &marker_lock,
        )
        .context("marking it uploading")?;
        drop(marker_lock);
        drop(hold);
        super::crash::point("opt_in:marked")?;
        let held_by_agora = self.file_as_sent(guard, &seal, id, &signing).await?;
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
            versions: line
                .iter()
                .filter(|e| e.entry.kind == agora_agentkit::enums::StateEntryKind::Snapshot)
                .count(),
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
    /// side effect of chaining. An agent marked uploading (or whose marker
    /// is in doubt) resumes only if its answer in force is yes
    /// ([`SealLedger::in_force`]: never over a `decline`, never while a
    /// re-offer waits, and nothing overrides that); an unreadable or
    /// missing ledger refuses. A marker in doubt is replaced on that yes.
    /// The uploaded line goes on from the last entry known to be on Agora
    /// (the newest in `sent/`) with **one new snapshot of the current
    /// state**; the versions not uploaded when the stop began, and those
    /// saved while stopped, are set aside here as `.displaced` (kept, with
    /// their blobs, opened by the `memory` tool as a side line) and never
    /// uploaded. An agent with no uploading marker has its stop cleared
    /// and nothing else. The agent is always told (one notice). Never a
    /// new genesis.
    ///
    /// Crash-safe: `sealed/resuming.json`, naming the stop it resumes, is
    /// written before anything is set aside and renamed done before the
    /// stop is cleared; an intent for another stop is set aside.
    ///
    /// The caller holds the runner lock exclusively. A state that changed
    /// outside the store while stopped needs `note`.
    pub async fn resume(&self, id: AgentId, note: Option<&str>) -> anyhow::Result<Report> {
        let note = note.map(str::trim).filter(|n| !n.is_empty());
        let agent_dir = self.state_dir.join(id.to_string());
        let seal = AgentSeal::new(&agent_dir, id);
        anyhow::ensure!(
            seal.is_enrolled().await?,
            "agent {id} is not chained: nothing to resume"
        );
        // The marker's lock from before the stop is read until it is
        // cleared: a `sealed-stop` issued meanwhile waits for this resume,
        // then finds the agent unstopped and stops it again. Without it, the
        // stop would find this one's stop on file, write nothing, and this
        // would set that stop aside after it
        let marker_lock =
            super::uploading::lock(&self.keys, id).context("the uploading marker's lock")?;
        // A stop kept in the secrets tree stands for a lost sealed/stopped
        super::stop::reconcile(&agent_dir, &self.keys, id)
            .context("putting back the stop kept in the secrets tree")?;
        let stop = super::stop::Stopped::load(&agent_dir)
            .context("sealed/stopped unreadable")?
            .with_context(|| format!("agent {id} is not stopped: nothing to resume"))?;
        let marker = super::uploading::state(&agent_dir, &self.keys, id);
        let marked = marker != super::uploading::Marker::Absent;
        if marked {
            anyhow::ensure!(
                std::fs::exists(SealLedger::path(&agent_dir))?,
                "agent {id} has no sealed-state ledger: its answer can't be read, so its uploads \
                 aren't resumed"
            );
            let ledger = SealLedger::load(&agent_dir)
                .await
                .context("the sealed-state ledger is unreadable: uploads not resumed")?;
            if let Err(why) = ledger.in_force().starts_uploads() {
                anyhow::bail!(
                    "agent {id}: {why}: uploads not resumed, and it stays stopped. Uploads \
                     resume only on its own yes: re-open the question (`sealed-reoffer`) and \
                     let it answer"
                );
            }
            // In doubt: settled here, on its yes (the copies set aside)
            if !matches!(marker, super::uploading::Marker::Uploading(_)) {
                super::uploading::set_replacing_doubt_locked(
                    &agent_dir,
                    &self.keys,
                    id,
                    super::uploading::UploadingBy::OptIn,
                    &marker_lock,
                )
                .context("replacing the uploading marker in doubt")?;
            }
        }
        let uploading = super::uploading::is_uploading(&agent_dir, &self.keys, id);
        // Whatever its journal says (`session::take_idle`): this command
        // holds `runner.lock` exclusively, and a journal is not state
        let lock = super::session::take_idle(&self.state_dir, id).await?;
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
                                "agent {id} has versions that were never uploaded but nothing \
                                 in sent/: nothing of it is known to be on Agora to continue \
                                 from, so its uploads can't be resumed on the same line; it \
                                 stays stopped and nothing was changed"
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
        // Set, not merged: the uploaded line resumed from the last entry
        // Agora holds, so the head's seq may be below the one the roster
        // saw while stopped; "higher seq wins" would keep the old one and
        // the next start would refuse the agent as rolled back. `head.json`
        // is written already (the order `roster::replace` asks for)
        super::roster::replace_for(&self.state_dir, id, Some(marked)).await?;
        // Always told, in one write, before the stop is cleared
        super::notice::Notice::operator_act(
            &agent_dir,
            note.unwrap_or_default(),
            super::notice::OperatorAct::Resumed {
                uploading,
                kept: kept_local,
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
        // Only the stop this resume began on is cleared: nothing can write
        // another while the marker's lock is held, but a stop on file that
        // isn't the one loaded (a copy restored by hand, say) is never set
        // aside by this
        let on_file = [
            super::stop::durable(&self.keys, id).context("reading the stop in the secrets tree")?,
            super::stop::Stopped::load(&agent_dir).context("reading sealed/stopped")?,
        ];
        anyhow::ensure!(
            on_file.iter().flatten().all(|s| s.at == stop.at),
            "agent {id}: the stop on file is no longer the one this resume began on: it stays \
             stopped. Run the resume again if that stop should end too"
        );
        // The durable copy first: left behind, it would put the stop back
        super::stop::clear_durable(&self.keys, id)
            .context("setting the stop in the secrets tree aside")?;
        if super::stop::is_stopped(&agent_dir) {
            super::stop::Stopped::clear(&agent_dir).context("setting sealed/stopped aside")?;
        }
        drop(marker_lock);
        tracing::warn!(
            event_type = "sealed_state_resumed",
            agent_id = %id,
            head = %marked.entry_hash,
            seq = marked.seq,
            kept_local,
            uploading,
            "a stopped agent resumed; the versions not uploaded before stay here"
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
    use crate::consent::sealed_state::ledger::{NotNow, SealAsk, SealOutcome};
    use crate::consent::sealed_state::prompt::SealChoice;
    use crate::sealed::api::fake;
    use agora_agentkit::reactor::seed::memory::memory_text;
    use agora_agentkit::state::storage::{Recovery, SEALED_DIR};
    use agora_agentkit::state::verify::verify_chain;

    /// The agent's answer on file: its first, or, when it has answered
    /// before, its answer to a re-offer the operator opens here (the only
    /// way a later answer comes into force).
    fn answer(dir: &Path, choice: SealChoice) {
        let mut l = SealLedger::load_blocking(dir).unwrap();
        if l.choice().is_some() && l.reoffer.is_none() {
            l.reopen("asked", Utc::now()).unwrap();
        }
        record_answer(dir, l, choice);
    }

    /// An answer recorded as it comes, whatever is open.
    fn record_answer(dir: &Path, mut l: SealLedger, choice: SealChoice) {
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
        let err = e.opt_in(&api, id).await.unwrap_err().to_string();
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

    /// A real leftover journal: a session that wrote a post and never
    /// finished.
    pub(crate) async fn leave_journal(root: &Path, id: AgentId) -> Vec<u8> {
        use agora_agentkit::reactor::journal::{Journal, PostWrite, SessionHeader, Written};
        let journal = FsStorage::new(root.join("state")).journal(id);
        journal
            .begin(
                SessionHeader {
                    started_at: Utc::now(),
                    model: "claude-test".into(),
                },
                &Journal::default(),
            )
            .await
            .unwrap();
        journal
            .append(
                &Written::Post(PostWrite {
                    at: Utc::now(),
                    id: agora_agentkit::ids::PostId::new(),
                })
                .into(),
            )
            .await
            .unwrap();
        std::fs::read(journal.path()).unwrap()
    }

    /// B1 (review of 2026-11-12). A journal an interrupted or failed
    /// session left is not state: the agent is chained from its last saved
    /// state, the journal is untouched (so its next session still shows
    /// the notice), and the start check lets it run. Before, such an agent
    /// was skipped (`Skip::InSession`) by every enrolment, and three of
    /// them refused the run that would have cleared their journals.
    #[tokio::test]
    async fn an_agent_with_a_leftover_journal_is_chained_and_still_told() {
        let root = crate::sealed::TestDir::new("chain-journal");
        let (id, keys, dir) = agent(&root, &["a", "b"]).await;
        let before = leave_journal(&root, id).await;
        let e = enroller(&root, &keys);
        let dry = e.chain(id, true).await.unwrap();
        assert!(dry.journal, "a dry run says the journal is there");
        assert!(!dir.join(SEALED_DIR).exists(), "and writes nothing");
        let report = e.chain(id, false).await.unwrap();
        assert!(report.journal);
        assert!(report.to_string().contains("journal"), "{report}");
        assert!(AgentSeal::new(&dir, id).is_enrolled().await.unwrap());
        assert_eq!(opened_memories(&dir, id, &keys).await, ["a", "b"]);
        // The journal as it was: the next session's notice is still due
        let journal = FsStorage::new(root.join("state")).journal(id);
        assert_eq!(std::fs::read(journal.path()).unwrap(), before);
        let notice = crate::interrupted::notice(&journal.read().await.unwrap()).unwrap();
        assert!(
            notice.starts_with(crate::interrupted::NOTICE_PREFIX),
            "{notice}"
        );
        // And the now-chained agent's start check lets that session start
        let hash = crate::sealed::check::tests::local_hash(&root, id).await;
        let out = crate::sealed::check::check(
            &dir,
            id,
            &hash,
            Default::default(),
            &keys,
            &fake::FakeApi::new(),
        )
        .await;
        assert!(
            matches!(out, crate::sealed::check::Outcome::Run { hold: None, .. }),
            "{out:?}"
        );
        assert_eq!(std::fs::read(journal.path()).unwrap(), before);
    }

    /// A leftover journal strands an agent in none of the operator's
    /// commands: `rotate-keys`, `--opt-in --yes` and `--resume` each take
    /// its session lock and act, leaving the journal as it was (its next
    /// session is still told). Each refused it before (`no_session`); a
    /// live session (its lock held) is still refused.
    #[tokio::test]
    async fn a_leftover_journal_strands_no_operator_command() {
        let root = crate::sealed::TestDir::new("journal-commands");
        let api = fake::guarded(&root);
        let (id, keys, dir) = agent(&root, &["a", "b"]).await;
        let e = enroller(&root, &keys);
        e.chain(id, false).await.unwrap();
        let before = leave_journal(&root, id).await;
        let journal = FsStorage::new(root.join("state")).journal(id);
        // A live session holds its lock: every command refuses
        {
            let _live = crate::sealed::session::agent_lock(&root.join("state"), id).unwrap();
            answer(&dir, SealChoice::Private);
            assert!(matches!(e.opt_in(&api, id).await, Err(Skip::Busy)));
            assert!(
                crate::sealed::rotate::rotate_keys(
                    &root.join("state"),
                    &keys,
                    &api,
                    id,
                    true,
                    false
                )
                .await
                .is_err()
            );
        }
        crate::sealed::rotate::rotate_keys(&root.join("state"), &keys, &api, id, true, false)
            .await
            .unwrap();
        assert_eq!(std::fs::read(journal.path()).unwrap(), before);
        e.opt_in(&api, id).await.unwrap();
        assert!(crate::sealed::uploads(&dir, &keys, id));
        assert_eq!(std::fs::read(journal.path()).unwrap(), before);
        // Uploaded, then stopped
        let seal = AgentSeal::new(&dir, id);
        for item in seal.outbox().await.unwrap() {
            seal.mark_sent(item.ord).await.unwrap();
        }
        crate::sealed::stop::stop(&root.join("state"), &keys, id, "It asked.")
            .await
            .unwrap();
        e.resume(id, None).await.unwrap();
        assert!(crate::sealed::uploads(&dir, &keys, id));
        assert_eq!(std::fs::read(journal.path()).unwrap(), before);
    }

    /// An agent another process holds (its session lock) is not chained
    /// this time, and says so apart from a failure: a rerun chains it.
    #[tokio::test]
    async fn an_agent_held_by_another_process_is_busy_not_failed() {
        let root = crate::sealed::TestDir::new("chain-busy");
        let (id, keys, dir) = agent(&root, &["a"]).await;
        let e = enroller(&root, &keys);
        let held = crate::sealed::session::try_session(&root.join("state"), id)
            .unwrap()
            .unwrap();
        let skip = e.chain(id, false).await.unwrap_err();
        assert!(matches!(skip, Skip::Busy), "{skip:?}");
        assert_eq!(skip.found(), crate::sealed::unchainable::Found::Unknown);
        assert!(!dir.join(SEALED_DIR).exists());
        drop(held);
        e.chain(id, false).await.unwrap();
    }

    /// What no rerun can chain is named by class, by a dry run too, with
    /// nothing written: no signing key, a state that won't load, an
    /// unreadable pre-loss copy.
    #[tokio::test]
    async fn what_cant_be_chained_is_classed_and_a_dry_run_says_the_same() {
        use crate::sealed::unchainable::{Class, Found};
        let root = crate::sealed::TestDir::new("chain-cannot");
        let (keyless, keys, keyless_dir) = agent(&root, &["a"]).await;
        std::fs::rename(
            root.join("secrets").join(keyless.to_string()),
            root.join("secrets").join(format!("{keyless}.aside")),
        )
        .unwrap();
        let (broken, _, broken_dir) = agent(&root, &["a"]).await;
        std::fs::write(broken_dir.join("state.json"), b"{ not json").unwrap();
        let e = enroller(&root, &keys);
        for dry in [true, false] {
            for (id, dir, class) in [
                (keyless, &keyless_dir, Class::NoSigningKey),
                (broken, &broken_dir, Class::StateUnloadable),
            ] {
                let skip = e.chain(id, dry).await.unwrap_err();
                assert_eq!(skip.found(), Found::Cannot(class), "dry {dry}: {skip:?}");
                assert_eq!(skip.as_str(), class.as_str());
                assert!(!dir.join(SEALED_DIR).exists(), "nothing written");
                assert!(
                    keys.state_key_file(id).unwrap().is_none(),
                    "no state key made"
                );
            }
        }
    }

    /// Secrets that can't be read (a permission error) for three or more
    /// agents at once are the machine's fault, not theirs: none is recorded
    /// as unchainable, and the pass fails saying why. One or two are
    /// recorded as before.
    #[tokio::test]
    async fn io_errors_across_agents_record_none_and_fail_the_pass() {
        use std::os::unix::fs::PermissionsExt as _;
        let root = crate::sealed::TestDir::new("chain-io-shaped");
        let mut ids = Vec::new();
        let mut keys = None;
        for _ in 0..IO_SHAPED_MAX {
            let (id, k, _) = agent(&root, &["a"]).await;
            ids.push(id);
            keys = Some(k);
        }
        let keys = keys.unwrap();
        let lock = |id: AgentId, mode| {
            std::fs::set_permissions(keys.agent_dir(id), std::fs::Permissions::from_mode(mode))
                .unwrap()
        };
        for id in &ids {
            lock(*id, 0o000);
        }
        let e = std::sync::Arc::new(enroller(&root, &keys));
        let results = e.chain_all(&ids, false, 2).await;
        for id in &ids {
            lock(*id, 0o700);
        }
        assert!(
            results
                .iter()
                .all(|(_, r)| r.as_ref().is_err_and(Skip::io_shaped)),
            "{results:?}"
        );
        let err = e.settle_pass(&results, false).unwrap_err().to_string();
        assert!(err.contains("I/O or permission error"), "{err}");
        assert!(
            crate::sealed::unchainable::load(&root)
                .unwrap()
                .agents
                .is_empty(),
            "none recorded"
        );
        // Fewer than that: recorded, as an agent's own
        e.settle_pass(&results[..IO_SHAPED_MAX - 1], false).unwrap();
        assert_eq!(
            crate::sealed::unchainable::load(&root)
                .unwrap()
                .agents
                .len(),
            IO_SHAPED_MAX - 1
        );
    }

    /// S3. A chain cut short whose state key then goes missing is never
    /// given a new key: the versions already sealed would be unopenable,
    /// and the start check (hashes only) couldn't tell. Refused, classed,
    /// nothing written; with the key back, the rerun finishes the chain.
    #[tokio::test]
    async fn a_partial_chain_whose_state_key_is_gone_gets_no_new_key() {
        use crate::sealed::unchainable::{Class, Found};
        let root = crate::sealed::TestDir::new("chain-key-lost");
        let memories = ["a", "b", "c"];
        let (id, keys, dir) = agent(&root, &memories).await;
        let e = enroller(&root, &keys);
        crate::sealed::crash::at("chain:archive");
        assert!(e.chain(id, false).await.is_err());
        let key = root
            .join("secrets")
            .join(id.to_string())
            .join("state_key.json");
        let aside = key.with_extension("json.aside");
        std::fs::rename(&key, &aside).unwrap();
        let head = std::fs::read(dir.join(SEALED_DIR).join("head.json")).unwrap();
        for dry in [true, false] {
            let skip = e.chain(id, dry).await.unwrap_err();
            assert_eq!(
                skip.found(),
                Found::Cannot(Class::StateKeyLost),
                "dry {dry}: {skip:?}"
            );
            assert!(!key.exists(), "no new key over a chain begun");
            assert_eq!(
                std::fs::read(dir.join(SEALED_DIR).join("head.json")).unwrap(),
                head
            );
        }
        std::fs::rename(&aside, &key).unwrap();
        let report = e.chain(id, false).await.unwrap();
        assert!(report.resumed);
        assert_eq!(opened_memories(&dir, id, &keys).await, memories);
    }

    /// An older runner's chain marker the start pass couldn't move: the
    /// agent is chained already, and is never chained again as if its
    /// chain had been cut short (which would add to it).
    #[tokio::test]
    async fn an_unmoved_older_marker_is_not_chained_again() {
        let root = crate::sealed::TestDir::new("chain-legacy");
        let (id, keys, dir) = agent(&root, &["a", "b"]).await;
        let e = enroller(&root, &keys);
        e.chain(id, false).await.unwrap();
        let before = shape(&dir, id).await;
        std::fs::rename(
            dir.join(SEALED_DIR)
                .join(agora_agentkit::state::storage::CHAINED),
            dir.join(SEALED_DIR).join("enrolled"),
        )
        .unwrap();
        let seal = AgentSeal::new(&dir, id);
        assert!(seal.has_legacy_marker().await.unwrap());
        let skip = e.chain(id, false).await.unwrap_err();
        assert!(
            matches!(&skip, Skip::Failed(e) if e.to_string().contains("older runner's chain marker")),
            "{skip:?}"
        );
        assert_eq!(shape(&dir, id).await, before, "nothing added");
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
        assert!(matches!(e.opt_in(&api, id).await, Err(Skip::NotChained)));
        e.chain(id, false).await.unwrap();
        assert!(matches!(
            e.opt_in(&api, id).await,
            Err(Skip::NotOptedIn(NotNow::Unanswered))
        ));
        answer(&dir, SealChoice::Decline);
        assert!(matches!(
            e.opt_in(&api, id).await,
            Err(Skip::NotOptedIn(NotNow::SaidNo))
        ));
        assert!(
            api.calls.lock().unwrap().is_empty(),
            "nothing asked of a no"
        );
        answer(&dir, SealChoice::Private);
        assert_eq!(e.opted_in().unwrap(), vec![id]);

        // Cut after the post: no marker; the rerun finds the record
        crate::sealed::crash::at("opt_in:posted");
        assert!(e.opt_in(&api, id).await.is_err());
        assert_eq!(
            *api.calls.lock().unwrap(),
            ["chain", "state_keys", "post_state_key"]
        );
        assert!(!crate::sealed::uploads(&dir, &keys, id), "not marked yet");
        assert_eq!(e.opted_in().unwrap(), vec![id], "still to finish");
        let done = e.opt_in(&api, id).await.unwrap();
        assert!(!done.posted && !done.already);
        assert_eq!(api.keys.lock().unwrap().len(), 1, "one record");
        assert!(crate::sealed::uploads(&dir, &keys, id));
        assert!(e.opted_in().unwrap().is_empty());

        // A marker in doubt (its secrets copy gone) uploads nothing, is
        // listed again, and is settled by the opt-in on the agent's yes:
        // the copy in doubt set aside, a new marker
        std::fs::remove_file(keys.agent_dir(id).join(crate::sealed::uploading::DURABLE)).unwrap();
        assert!(!crate::sealed::uploads(&dir, &keys, id));
        crate::sealed::uploading::reconcile_all(&root.join("state"), &keys);
        assert!(
            !crate::sealed::uploads(&dir, &keys, id),
            "no start writes the secrets copy back"
        );
        assert_eq!(e.opted_in().unwrap(), vec![id]);
        let done = e.opt_in(&api, id).await.unwrap();
        assert!(!done.posted);
        assert!(crate::sealed::uploads(&dir, &keys, id));
        assert!(
            std::fs::read_dir(dir.join(SEALED_DIR)).unwrap().any(|f| f
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("uploading.doubt.")),
            "the copy in doubt is kept aside"
        );
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
        assert!(e.opt_in(&api, id).await.unwrap().already);
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
        let done = e.opt_in(&api, id).await.unwrap();
        assert!(done.posted);
        assert_eq!(
            *api.keys.lock().unwrap(),
            vec![kept.clone()],
            "the kept bytes"
        );
        assert_eq!(keys.wrap_record(id).unwrap().unwrap(), kept);
    }

    /// Nothing opts an agent in against its answer: not with no answer,
    /// not over a `decline`, not while the operator's re-offer waits for
    /// its answer (whatever the earlier answer was), not on an answer it
    /// gave with no question open. No override exists, and no call is
    /// made. Its own yes to a re-offer is what does it.
    #[tokio::test]
    async fn nothing_opts_an_agent_in_against_its_answer() {
        let root = crate::sealed::TestDir::new("opt-in-against");
        let api = fake::guarded(&root);
        let (id, keys, dir) = agent(&root, &["a"]).await;
        let e = enroller(&root, &keys);
        e.chain(id, false).await.unwrap();
        let refused = |r: Result<OptedIn, Skip>, why: NotNow| match r {
            Err(Skip::NotOptedIn(w)) => assert_eq!(w, why),
            other => panic!(
                "{why:?}: {:?}",
                other.map(|_| ()).map_err(|e| e.to_string())
            ),
        };
        refused(e.opt_in(&api, id).await, NotNow::Unanswered);
        answer(&dir, SealChoice::Decline);
        refused(e.opt_in(&api, id).await, NotNow::SaidNo);
        // (B) A second answer nobody asked for, a yes: on file, not in force
        record_answer(
            &dir,
            SealLedger::load_blocking(&dir).unwrap(),
            SealChoice::Private,
        );
        refused(e.opt_in(&api, id).await, NotNow::SaidNo);
        assert!(e.opted_in().unwrap().is_empty(), "not listed either");
        // The operator re-opens the question: nothing until it answers
        let mut l = SealLedger::load_blocking(&dir).unwrap();
        l.reopen("Asked by DM.", Utc::now()).unwrap();
        std::fs::write(SealLedger::path(&dir), serde_json::to_vec(&l).unwrap()).unwrap();
        refused(e.opt_in(&api, id).await, NotNow::Reopened("decline"));
        // Its yes to the re-offer
        answer(&dir, SealChoice::Private);
        assert_eq!(e.opted_in().unwrap(), vec![id]);
        // (A) Re-opened again, unanswered: the yes on file is not acted on
        let mut l = SealLedger::load_blocking(&dir).unwrap();
        l.reopen("Asked again.", Utc::now()).unwrap();
        std::fs::write(SealLedger::path(&dir), serde_json::to_vec(&l).unwrap()).unwrap();
        refused(e.opt_in(&api, id).await, NotNow::Reopened("private"));
        assert!(
            e.opted_in().unwrap().is_empty(),
            "not listed while it waits"
        );
        assert!(
            api.calls.lock().unwrap().is_empty(),
            "nothing was asked of Agora for any of them"
        );
        assert!(!crate::sealed::uploads(&dir, &keys, id));
        assert!(crate::sealed::notice::Notice::load(&dir).unwrap().is_none());
        // Answered yes again: opted in, told it is as it answered
        answer(&dir, SealChoice::Publish);
        e.opt_in(&api, id).await.unwrap();
        assert!(crate::sealed::uploads(&dir, &keys, id));
        let line = crate::sealed::notice::Notice::load(&dir)
            .unwrap()
            .unwrap()
            .line()
            .unwrap();
        assert!(line.contains("as you answered on"), "{line}");
        assert!(!line.contains("although"), "{line}");
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
        assert!(matches!(e.opt_in(&api, id).await, Err(Skip::Stopped)));
        assert!(api.calls.lock().unwrap().is_empty());
    }

    /// Run `step` on its own thread (its own runtime), paused at `point`:
    /// returns once it is there, with its handle and its release.
    fn paused<T: Send + 'static>(
        point: &'static str,
        step: impl FnOnce() -> T + Send + 'static,
    ) -> (std::thread::JoinHandle<T>, std::sync::mpsc::Sender<()>) {
        let (reached_tx, reached) = std::sync::mpsc::channel();
        let (release, release_rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            crate::sealed::crash::pause_at(point, reached_tx, release_rx);
            step()
        });
        reached
            .recv_timeout(std::time::Duration::from_secs(60))
            .expect("the step reached its pause");
        (handle, release)
    }

    /// `sealed-stop` on its own thread, as a second process would run it;
    /// it must still be waiting for the step that holds the marker's lock.
    fn stop_meanwhile(
        state_dir: &Path,
        keys: &KeyStore,
        id: AgentId,
    ) -> std::thread::JoinHandle<anyhow::Result<crate::sealed::stop::Stop>> {
        let (state_dir, keys) = (state_dir.to_path_buf(), keys.clone());
        let handle = std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(crate::sealed::stop::stop(&state_dir, &keys, id, "stop now"))
        });
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(
            !handle.is_finished(),
            "the stop waits for the step in progress"
        );
        handle
    }

    /// A stop issued while a resume is in progress (paused after its seal,
    /// before it clears the stop) waits for it, then stops the agent the
    /// resume left: the resume can't set the new stop aside, and nothing
    /// uploads after it.
    #[tokio::test]
    async fn a_stop_during_a_resume_wins() {
        let root = crate::sealed::TestDir::new("resume-stop-race");
        let api = fake::guarded(&root);
        let (id, keys, dir) = agent(&root, &["a"]).await;
        let state_dir = root.join("state");
        let e = enroller(&root, &keys);
        e.chain(id, false).await.unwrap();
        answer(&dir, SealChoice::Private);
        e.opt_in(&api, id).await.unwrap();
        upload(&api, &dir, id).await;
        let first = crate::sealed::stop::stop(&state_dir, &keys, id, "asked to")
            .await
            .unwrap()
            .stopped;
        let (resume, release) = {
            let (root, keys) = (root.to_path_buf(), keys.clone());
            paused("resume:sealed", move || {
                tokio::runtime::Builder::new_current_thread()
                    .build()
                    .unwrap()
                    .block_on(enroller(&root, &keys).resume(id, None))
            })
        };
        let stop = stop_meanwhile(&state_dir, &keys, id);
        release.send(()).unwrap();
        resume.join().unwrap().expect("the resume finishes");
        let stop = stop.join().unwrap().unwrap();
        assert!(!stop.already, "a new stop, not the one the resume ended");
        assert_ne!(stop.stopped.at, first.at);
        assert!(crate::sealed::stop::is_stopped(&dir));
        assert_eq!(
            crate::sealed::stop::durable(&keys, id).unwrap(),
            Some(stop.stopped.clone())
        );
        assert!(
            !crate::sealed::uploads(&dir, &keys, id),
            "stopped after the resume"
        );
    }

    /// A stop issued while an opt-in is in progress (paused after the key
    /// record's post, before the marker) is not refused for want of a
    /// marker: it waits, then stops the agent the opt-in marked.
    #[tokio::test]
    async fn a_stop_during_an_opt_in_wins() {
        let root = crate::sealed::TestDir::new("opt-in-stop-race");
        let (id, keys, dir) = agent(&root, &["a"]).await;
        let state_dir = root.join("state");
        enroller(&root, &keys).chain(id, false).await.unwrap();
        answer(&dir, SealChoice::Private);
        let (opt_in, release) = {
            let (root, keys) = (root.to_path_buf(), keys.clone());
            paused("opt_in:posted", move || {
                tokio::runtime::Builder::new_current_thread()
                    .build()
                    .unwrap()
                    .block_on(async {
                        let api = fake::guarded(&root);
                        enroller(&root, &keys)
                            .opt_in(&api, id)
                            .await
                            .map(|_| ())
                            .map_err(|e| e.to_string())
                    })
            })
        };
        let stop = stop_meanwhile(&state_dir, &keys, id);
        release.send(()).unwrap();
        opt_in.join().unwrap().expect("the opt-in finishes");
        let stop = stop
            .join()
            .unwrap()
            .expect("not refused for want of a marker");
        assert!(!stop.already);
        assert!(crate::sealed::stop::is_stopped(&dir));
        assert!(
            !crate::sealed::uploads(&dir, &keys, id),
            "stopped after the opt-in"
        );
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
            e.opt_in(&api, id).await.unwrap();
            upload(&api, &dir, id).await;
            let before = api.chain.lock().unwrap().clone().unwrap();
            crate::sealed::stop::stop(&state_dir, &keys, id, "asked to")
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
            // As the runner's keeper did while it ran stopped: the roster
            // holds the head of the stopped interval
            crate::sealed::roster::record_heads(&root, &state_dir, &[id])
                .await
                .unwrap();
            let stopped_seq = seal.head().await.unwrap().unwrap().head.seq;
            if let Some(point) = cut {
                crate::sealed::crash::at(point);
                assert!(e.resume(id, None).await.is_err(), "{point}");
            }
            let report = e.resume(id, None).await.unwrap();
            assert_eq!(report.kept_local, 2, "{cut:?}");
            // S4: the uploaded line resumed below the stopped interval's
            // head, and the roster follows it (set, not "higher seq wins"):
            // the next start doesn't refuse the agent as rolled back
            let head = seal.head().await.unwrap().unwrap().head;
            assert!(head.seq < stopped_seq, "{cut:?}");
            let roster = crate::sealed::roster::load(&root).unwrap().unwrap();
            assert_eq!(
                (roster.agents[&id].seq, roster.agents[&id].entry_hash),
                (head.seq, head.entry_hash),
                "{cut:?}"
            );
            let scan = crate::sealed::roster::check(&root, &state_dir)
                .await
                .unwrap();
            assert!(scan.losses.is_empty(), "{cut:?}: {:?}", scan.losses);
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
            // The notice's wording (`notice.rs`)
            assert!(
                line.contains(
                    "The 2 versions not backed up before the stop or saved while backing up was \
                     stopped stay on your operator's machine, off the main line of"
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
        crate::sealed::uploading::tests::ledger(&dir, true);
        let state_dir = root.join("state");
        let seal = AgentSeal::new(&dir, id);
        for item in seal.outbox().await.unwrap() {
            seal.mark_sent(item.ord).await.unwrap();
        }
        crate::sealed::stop::stop(&state_dir, &keys, id, "asked to")
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
            format!(
                "On {}, your operator resumed backing up your sealed history on Agora. Your \
                 operator's note: I checked the edit.",
                Utc::now().format(crate::sealed::text::DATE)
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
        assert_eq!(l.in_force().starts_uploads(), Err(NotNow::SaidNo));
        assert_eq!(l.in_force().stops_uploads(), Ok(()));
        assert!(l.reopen("Asked by DM.", Utc::now()).is_ok());
    }

    /// A stopped agent is never resumed by chaining; `resume` refuses over
    /// a `decline` in force, while a re-offer waits, and with a missing
    /// ledger, with no override; it resumes on the agent's own yes, and
    /// always tells the agent.
    #[tokio::test]
    async fn a_resume_follows_the_answer_and_always_tells() {
        let root = crate::sealed::TestDir::new("resume-answer");
        let api = fake::guarded(&root);
        let (id, keys, dir) = agent(&root, &["a"]).await;
        let state_dir = root.join("state");
        let e = enroller(&root, &keys);
        e.chain(id, false).await.unwrap();
        answer(&dir, SealChoice::Private);
        e.opt_in(&api, id).await.unwrap();
        upload(&api, &dir, id).await;
        answer(&dir, SealChoice::Decline);
        crate::sealed::stop::stop(&state_dir, &keys, id, "asked to")
            .await
            .unwrap();
        // Chaining leaves it stopped
        assert!(matches!(e.chain(id, false).await, Err(Skip::Chained)));
        assert!(crate::sealed::stop::is_stopped(&dir));
        let err = e.resume(id, None).await.unwrap_err().to_string();
        assert!(
            err.contains("answer in force") && err.contains("decline"),
            "{err}"
        );
        let err = e
            .resume(id, Some("A court order."))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("uploads not resumed"),
            "a note overrides nothing: {err}"
        );
        assert!(crate::sealed::stop::is_stopped(&dir));
        assert!(!crate::sealed::uploads(&dir, &keys, id));
        // A missing ledger refuses
        let ledger = SealLedger::path(&dir);
        std::fs::rename(&ledger, dir.join("ledger.kept")).unwrap();
        let err = e.resume(id, Some("n")).await.unwrap_err().to_string();
        assert!(err.contains("no sealed-state ledger"), "{err}");
        std::fs::rename(dir.join("ledger.kept"), &ledger).unwrap();
        // Re-opened: not until it answers
        let mut l = SealLedger::load_blocking(&dir).unwrap();
        l.reopen("Asked by DM.", Utc::now()).unwrap();
        std::fs::write(&ledger, serde_json::to_vec(&l).unwrap()).unwrap();
        let err = e.resume(id, None).await.unwrap_err().to_string();
        assert!(err.contains("has not answered again yet"), "{err}");
        assert!(crate::sealed::stop::is_stopped(&dir));
        // Its yes
        answer(&dir, SealChoice::Private);
        e.resume(id, Some("As you asked.")).await.unwrap();
        assert!(crate::sealed::uploads(&dir, &keys, id));
        let line = crate::sealed::notice::Notice::load(&dir)
            .unwrap()
            .unwrap()
            .line()
            .unwrap();
        assert!(
            line.contains("resumed backing up your sealed history on Agora."),
            "{line}"
        );
        assert!(!line.contains("although"), "{line}");
        assert!(
            line.ends_with("Your operator's note: As you asked."),
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
        e.opt_in(&api, id).await.unwrap();
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
        crate::sealed::stop::stop(&state_dir, &keys, id, "first")
            .await
            .unwrap();
        save("while stopped").await;
        crate::sealed::crash::at("resume:done");
        assert!(e.resume(id, None).await.is_err());
        let head = AgentSeal::new(&dir, id).head().await.unwrap().unwrap();
        e.resume(id, None).await.unwrap();
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
        crate::sealed::stop::stop(&state_dir, &keys, id, "second")
            .await
            .unwrap();
        save("while stopped again").await;
        e.resume(id, None).await.unwrap();
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
        assert!(e.opt_in(&api, id).await.is_err());
        assert!(!crate::sealed::uploads(&dir, &keys, id));
        let line = crate::sealed::notice::Notice::load(&dir)
            .unwrap()
            .unwrap()
            .line()
            .unwrap();
        assert!(
            line.contains(
                "your operator started backing up your sealed history on Agora, as you answered on"
            ),
            "{line}"
        );
        let waiting = crate::sealed::backlog(&dir).unwrap().versions;
        let opted = e.opt_in(&api, id).await.unwrap();
        assert!(crate::sealed::uploads(&dir, &keys, id));
        // The main line's versions, as the `memory` tool counts them: what
        // waited to go up, for the notice to say once they are all up
        // (`cli.rs`, `Notice::opted_in_uploaded`)
        assert_eq!(opted.versions, waiting);
        assert!(waiting > 0);
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
            let err = e.opt_in(&api, id).await.unwrap_err().to_string();
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
            e.opt_in(&api, id).await.unwrap();
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
        let done = e.opt_in(&api, id).await.unwrap();
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
