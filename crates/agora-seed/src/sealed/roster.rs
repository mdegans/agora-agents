//! `<data_dir>/sealed-roster.json`: the runner's witness, outside `state/`,
//! of every chained agent's head.
//!
//! An agent's chain lives in the directory it guards, so a directory put
//! back to an older consistent copy, or removed whole, reads as fine to the
//! chain alone. The roster holds, per chained agent, the head seq and entry
//! hash the runner last saw, beside `runner.lock` (outside `state/` and
//! `secrets/`). At each start ([`check`]), for every agent on it: its
//! directory gone, its chain marker gone, its `head.json` gone, its
//! `state.json` gone, or its head **behind** the roster's (or at the same
//! seq under another hash) is a loss, and refuses the run. A head ahead is
//! normal (a crash before the roster was written): accepted, recorded.
//!
//! **Written** when an agent is enrolled ([`record`]), at the start check
//! (heads ahead, chained directories not yet on it), and while a run goes
//! on and at its end for the agents that ran ([`Keeper`]). Atomically and
//! durably ([`super::write_private`]), under `sealed-roster.lock` around
//! every read-modify-write, merged per agent by "higher seq wins" so the
//! continuous unit, the daily timer and a CLI command can all write it.
//! Operator commands that set a head back on purpose use [`replace`].
//!
//! **The roster itself lost.** `sealed-roster.began` is written beside it
//! the first time it is. A roster missing while `began` exists refuses the
//! run, whether or not chained agents are found: a wiped `state/` would
//! take them with it. Both gone together reads as a first run (the gap:
//! the whole data directory's top level lost at once, which a wiped
//! `state/` is not). A roster that can't be read, or is in a newer format,
//! refuses the run.
//!
//! **First run.** With neither file, every directory that is chained
//! (agentkit's `AgentSeal::is_enrolled`) and has a readable head is
//! adopted; nothing else is. A directory with a `state.json` that is
//! neither chained nor on the roster is counted for one alert a run
//! ([`Scan::unchained`]), never enrolled from here.
//!
//! Reads only directory listings, the chain marker's existence,
//! `sealed/head.json`, the existence of `state.json` and `sealed/refused`.
//! Never any state content.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use agora_agentkit::govlog::Sha256Hex;
use agora_agentkit::ids::AgentId;
use agora_agentkit::state::chain::HeadRef;
use agora_agentkit::state::storage::AgentSeal;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::notice::Refusal;

pub const ROSTER: &str = "sealed-roster.json";
/// Written once, beside the roster, when the roster is first written.
pub const BEGAN: &str = "sealed-roster.began";
const LOCK: &str = "sealed-roster.lock";
/// The roster's format; a newer one refuses the run.
pub const FORMAT: u32 = 1;

/// One chained agent's head, as last seen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub seq: u64,
    pub entry_hash: Sha256Hex,
    pub updated_at: DateTime<Utc>,
    /// The last full verification of its local chain, for an agent that
    /// doesn't upload ([`Vouch`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vouched: Option<Vouch>,
}

impl Entry {
    fn of(head: HeadRef) -> Self {
        Self {
            seq: head.seq,
            entry_hash: head.entry_hash,
            updated_at: Utc::now(),
            vouched: None,
        }
    }
}

/// A full verification of an agent's local main line that passed: the
/// head it reached, the next outbox number then, when, the newest
/// keyframe on it, and what the verdict says of it. Until it is
/// [`FULL_VERIFY_EVERY`] old (spread per agent), the start check of an
/// agent that doesn't upload verifies only the line from this head on
/// ([`super::check`]), and falls back to the full check on anything off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Vouch {
    pub head: HeadRef,
    pub next_ord: u64,
    pub at: DateTime<Utc>,
    pub keyframe: Option<Sha256Hex>,
    pub summary: super::check::LineSummary,
    /// Versions that couldn't be opened here at that check.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unopenable: Option<super::check::Unopenable>,
}

/// How often an agent's whole local chain is verified again, at least:
/// each agent's period is between this and twice it, spread by its id so
/// the fleet doesn't come due at one start.
pub const FULL_VERIFY_EVERY: chrono::TimeDelta = chrono::TimeDelta::days(7);

impl Vouch {
    /// Still good for `id` at `now`.
    pub fn fresh(&self, id: AgentId, now: DateTime<Utc>) -> bool {
        let spread = (id.as_uuid().as_u128() % 1_000) as i32;
        let period = FULL_VERIFY_EVERY + FULL_VERIFY_EVERY * spread / 1_000;
        now >= self.at && now - self.at < period
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Roster {
    pub format: u32,
    pub updated_at: DateTime<Utc>,
    #[serde(default)]
    pub agents: BTreeMap<AgentId, Entry>,
}

impl Default for Roster {
    fn default() -> Self {
        Self {
            format: FORMAT,
            updated_at: Utc::now(),
            agents: BTreeMap::new(),
        }
    }
}

impl Roster {
    /// "Higher seq wins": `head` replaces the entry only if it is ahead.
    /// Whether it did.
    fn merge(&mut self, id: AgentId, head: HeadRef) -> bool {
        match self.agents.get_mut(&id) {
            Some(e) if e.seq >= head.seq => false,
            Some(e) => {
                *e = Entry {
                    vouched: e.vouched,
                    ..Entry::of(head)
                };
                true
            }
            None => {
                self.agents.insert(id, Entry::of(head));
                true
            }
        }
    }
}

/// Why the roster couldn't be used: each refuses the whole run.
#[derive(Debug, thiserror::Error)]
pub enum RosterError {
    #[error("{} is missing, but {BEGAN} or the secrets root's {WITNESS} says it was written before", path.display())]
    Missing { path: PathBuf },
    #[error("{} holds {roster} agents, but the secrets root's {WITNESS} says {witness}: agents left it other than by sealed-forget", path.display())]
    Shrunk {
        path: PathBuf,
        roster: usize,
        witness: usize,
    },
    #[error("{} is unreadable: {error}", path.display())]
    Unreadable { path: PathBuf, error: String },
    #[error("{} is format {format}, newer than this runner's {FORMAT}", path.display())]
    Newer { path: PathBuf, format: u32 },
    #[error("the state directory {} couldn't be listed: {error}", path.display())]
    StateDir {
        path: PathBuf,
        error: std::io::Error,
    },
    #[error("the roster couldn't be locked or written: {0}")]
    Io(#[from] std::io::Error),
}

/// What a roster agent's directory shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loss {
    /// `state/<id>/` is gone.
    DirMissing,
    /// The chain marker is gone (the whole `sealed/`, or the marker).
    ChainMissing,
    /// `sealed/head.json` is gone.
    HeadMissing,
    /// `state.json` is gone.
    StateMissing,
    /// The head is behind the roster's, or at its seq under another hash:
    /// an older copy put back.
    Behind { roster: u64, found: u64 },
    /// `sealed/head.json` is there but can't be read or parsed: the
    /// roster knows a head should be, so it is a loss, not a hold.
    HeadUnreadable,
    /// The directory couldn't be looked at (an I/O error).
    Io,
}

impl Loss {
    /// The refusal the agent is marked with.
    pub fn refusal(self) -> Refusal {
        match self {
            Self::DirMissing | Self::StateMissing => Refusal::StateMissing,
            Self::ChainMissing | Self::HeadMissing | Self::HeadUnreadable => Refusal::HeadMissing,
            Self::Behind { .. } => Refusal::RolledBack,
            Self::Io => Refusal::StateIo,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::DirMissing => "dir_missing",
            Self::ChainMissing => "chain_missing",
            Self::HeadMissing => "head_missing",
            Self::HeadUnreadable => "head_unreadable",
            Self::StateMissing => "state_missing",
            Self::Behind { .. } => "behind",
            Self::Io => "io",
        }
    }
}

/// What the start's scan found ([`check`]).
#[derive(Debug, Clone, Default)]
pub struct Scan {
    /// Agents on the roster whose directory shows a loss.
    pub losses: BTreeMap<AgentId, Loss>,
    /// Chained directories added (adopted on a first run, or missed).
    pub added: Vec<AgentId>,
    /// Directories with a `state.json`, neither chained nor on the roster.
    pub unchained: Vec<AgentId>,
    /// Agents marked refused (`sealed/refused`), from this run or earlier.
    pub refused: BTreeSet<AgentId>,
    /// The roster's agents after the scan.
    pub on_roster: BTreeSet<AgentId>,
    /// Each agent's last full verification, where there is one.
    pub vouched: BTreeMap<AgentId, Vouch>,
    /// No witness of an earlier roster existed: this start began it.
    pub first: bool,
    /// The roster was last written this long before the newest chained
    /// agent's `state.json` (past [`ROSTER_STALE`]): when, and that mtime.
    pub stale: Option<(DateTime<Utc>, DateTime<Utc>)>,
}

/// A roster written this long before the newest chained state is alerted
/// ([`Scan::stale`]): a data directory restored from an older backup
/// without its files' times would show it. The stated limit: a whole data
/// directory restored with its times kept passes every witness inside it;
/// only a count and digest kept outside that backup set (on Agora, for an
/// agent that uploads) would close it.
pub const ROSTER_STALE: chrono::TimeDelta = chrono::TimeDelta::days(1);

fn path(data_dir: &Path) -> PathBuf {
    data_dir.join(ROSTER)
}

/// The exclusive lock around a read-modify-write, waited for off the
/// async runtime's threads.
async fn lock(data_dir: &Path) -> std::io::Result<std::fs::File> {
    let at = data_dir.join(LOCK);
    let dir = data_dir.to_path_buf();
    tokio::task::spawn_blocking(move || {
        std::fs::create_dir_all(&dir)?;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(at)?;
        file.lock()?;
        Ok(file)
    })
    .await
    .map_err(std::io::Error::other)?
}

/// The roster: `None` when there is none.
pub fn load(data_dir: &Path) -> Result<Option<Roster>, RosterError> {
    let path = path(data_dir);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(RosterError::Unreadable {
                path,
                error: e.to_string(),
            });
        }
    };
    #[derive(Deserialize)]
    struct Format {
        format: u32,
    }
    match serde_json::from_slice::<Format>(&bytes) {
        Ok(f) if f.format > FORMAT => {
            return Err(RosterError::Newer {
                path,
                format: f.format,
            });
        }
        Ok(_) => {}
        Err(e) => {
            return Err(RosterError::Unreadable {
                path,
                error: e.to_string(),
            });
        }
    }
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|e| RosterError::Unreadable {
            path,
            error: e.to_string(),
        })
}

/// Read under the lock: the roster, or a new one when neither it nor
/// `began` exists (`true`: a first one).
fn load_or_begin(data_dir: &Path) -> Result<(Roster, bool), RosterError> {
    let witness = Witness::load(data_dir)?;
    match load(data_dir)? {
        Some(r) => match witness {
            // Agents leave the roster only through `replace(.., None)`,
            // which lowers the witness's count first
            Some(w) if w.agents > r.agents.len() => Err(RosterError::Shrunk {
                path: path(data_dir),
                roster: r.agents.len(),
                witness: w.agents,
            }),
            _ => Ok((r, false)),
        },
        None if witness.is_some() || std::fs::exists(data_dir.join(BEGAN))? => {
            Err(RosterError::Missing {
                path: path(data_dir),
            })
        }
        None => Ok((Roster::default(), true)),
    }
}

/// The roster's second witness, in the secrets root (a tree the start
/// already lists, and backed up apart from `state/`): when the roster
/// began and how many agents it holds. Either witness present without the
/// roster, or a roster holding fewer agents than this says, refuses the
/// run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Witness {
    pub began: DateTime<Utc>,
    pub agents: usize,
}

/// The witness's file, in `<data_dir>/secrets/`.
pub const WITNESS: &str = "sealed-roster.witness.json";

impl Witness {
    fn path(data_dir: &Path) -> PathBuf {
        data_dir.join("secrets").join(WITNESS)
    }

    pub fn load(data_dir: &Path) -> Result<Option<Self>, RosterError> {
        let path = Self::path(data_dir);
        match std::fs::read(&path) {
            Ok(bytes) => {
                serde_json::from_slice(&bytes)
                    .map(Some)
                    .map_err(|e| RosterError::Unreadable {
                        path,
                        error: e.to_string(),
                    })
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(RosterError::Unreadable {
                path,
                error: e.to_string(),
            }),
        }
    }

    fn save(data_dir: &Path, w: &Self) -> std::io::Result<()> {
        super::write_private(
            &Self::path(data_dir),
            &serde_json::to_vec_pretty(w).map_err(std::io::Error::other)?,
        )
    }
}

/// Write it, then `began` if it isn't there, with the secrets root's
/// witness: written before the roster when the count goes down (a crash
/// between leaves a roster above its witness, which is fine), after it
/// otherwise.
fn save(data_dir: &Path, roster: &mut Roster) -> Result<(), RosterError> {
    roster.updated_at = Utc::now();
    let old = Witness::load(data_dir).ok().flatten();
    let witness = Witness {
        began: old.map_or(roster.updated_at, |w| w.began),
        agents: roster.agents.len(),
    };
    let down = old.is_some_and(|w| w.agents > witness.agents);
    if down {
        Witness::save(data_dir, &witness)?;
    }
    super::write_private(
        &path(data_dir),
        &serde_json::to_vec_pretty(roster).map_err(std::io::Error::other)?,
    )?;
    let began = data_dir.join(BEGAN);
    if !std::fs::exists(&began)? {
        super::write_private(&began, Utc::now().to_rfc3339().as_bytes())?;
    }
    if !down && old != Some(witness) {
        Witness::save(data_dir, &witness)?;
    }
    Ok(())
}

/// Record `id`'s head ("higher seq wins"): at enrolment.
#[allow(dead_code)] // B3: `sealed-enroll` calls it once the chain is marked
pub async fn record(data_dir: &Path, id: AgentId, head: HeadRef) -> Result<(), RosterError> {
    record_all(data_dir, [(id, head)]).await
}

/// Record each head that is ahead of the roster's; written only if one is.
pub async fn record_all(
    data_dir: &Path,
    heads: impl IntoIterator<Item = (AgentId, HeadRef)>,
) -> Result<(), RosterError> {
    let _lock = lock(data_dir).await?;
    let (mut roster, first) = load_or_begin(data_dir)?;
    let mut changed = first;
    for (id, head) in heads {
        changed |= roster.merge(id, head);
    }
    if changed {
        save(data_dir, &mut roster)?;
    }
    Ok(())
}

/// Record the heads of `ids` as `head.json` says them, read inside the
/// roster's lock, "higher seq wins": the runner's own updates, which can
/// then never undo a [`replace`] that came after `head.json` was lowered.
pub async fn record_heads(
    data_dir: &Path,
    state_dir: &Path,
    ids: &[AgentId],
) -> Result<Vec<(AgentId, HeadRef)>, RosterError> {
    let _lock = lock(data_dir).await?;
    let (mut roster, first) = load_or_begin(data_dir)?;
    let mut changed = first;
    let mut read = Vec::new();
    for id in ids {
        let seal = AgentSeal::new(&state_dir.join(id.to_string()), *id);
        if let Ok(Some(h)) = seal.head().await {
            changed |= roster.merge(*id, h.head);
            read.push((*id, h.head));
        }
    }
    if changed {
        save(data_dir, &mut roster)?;
    }
    Ok(read)
}

/// Keep each agent's newest full verification; written only if one is
/// newer. An agent not on the roster is left off it.
pub async fn vouch_all(data_dir: &Path, vouches: &[(AgentId, Vouch)]) -> Result<(), RosterError> {
    if vouches.is_empty() {
        return Ok(());
    }
    let _lock = lock(data_dir).await?;
    let (mut roster, _) = load_or_begin(data_dir)?;
    let mut changed = false;
    for (id, v) in vouches {
        if let Some(e) = roster.agents.get_mut(id)
            && e.vouched.is_none_or(|old| old.at < v.at)
        {
            e.vouched = Some(*v);
            changed = true;
        }
    }
    if changed {
        save(data_dir, &mut roster)?;
    }
    Ok(())
}

/// Set `id`'s head to `head` whatever the roster held: for an operator's
/// command that sets the head back on purpose (`sealed-accept --rebase`
/// onto a shorter chain, a restore). `None` takes the agent off: the one
/// way an agent leaves the roster (`sealed-forget`).
///
/// **Order, for a command that lowers a head:** rewrite `head.json`
/// first, then call this. The runner's [`Keeper`] reads heads from disk
/// inside the roster's lock ([`record_heads`]), so once `head.json` is
/// lowered it can't raise the roster again, and this sets it.
///
/// B5 calls it: `sealed-accept` (`accept` and `rebase`) and
/// `sealed-restore` with `Some(<the head written>)` after `head.json` is
/// written; `sealed-forget` with `None` after the state key is set aside.
#[allow(dead_code)] // B5: accept, restore and forget call it once they act
pub async fn replace(
    data_dir: &Path,
    id: AgentId,
    head: Option<HeadRef>,
) -> Result<(), RosterError> {
    let _lock = lock(data_dir).await?;
    let (mut roster, _) = load_or_begin(data_dir)?;
    match head {
        Some(head) => roster.agents.insert(id, Entry::of(head)),
        None => roster.agents.remove(&id),
    };
    save(data_dir, &mut roster)
}

/// The agent directories under `state_dir`, by id.
fn agent_dirs(state_dir: &Path) -> Result<BTreeSet<AgentId>, RosterError> {
    let entries = match std::fs::read_dir(state_dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeSet::new()),
        Err(error) => {
            return Err(RosterError::StateDir {
                path: state_dir.to_path_buf(),
                error,
            });
        }
    };
    let mut ids = BTreeSet::new();
    for entry in entries {
        let entry = entry.map_err(|error| RosterError::StateDir {
            path: state_dir.to_path_buf(),
            error,
        })?;
        if let Some(id) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<AgentId>().ok())
        {
            ids.insert(id);
        }
    }
    Ok(ids)
}

/// What one agent's directory holds, as far as the roster asks.
enum Found {
    /// No directory.
    Gone,
    /// Not chained.
    Unchained {
        state: bool,
    },
    /// Chained, no `head.json`.
    NoHead {
        state: bool,
    },
    /// Chained, `head.json` unreadable: the start check's own reading
    /// says so (a hold); no comparison here.
    HeadUnreadable {
        state: bool,
    },
    Head {
        head: HeadRef,
        state: bool,
    },
}

async fn look(dir: &Path, id: AgentId) -> std::io::Result<Found> {
    if !std::fs::exists(dir)? {
        return Ok(Found::Gone);
    }
    let state = std::fs::exists(dir.join("state.json"))?;
    let seal = AgentSeal::new(dir, id);
    if !seal.is_enrolled().await? {
        return Ok(Found::Unchained { state });
    }
    Ok(match seal.head().await {
        Ok(Some(h)) => Found::Head {
            head: h.head,
            state,
        },
        Ok(None) => Found::NoHead { state },
        Err(e) => {
            tracing::warn!(agent_id = %id, error = %e, "sealed/head.json unreadable for the roster");
            Found::HeadUnreadable { state }
        }
    })
}

/// The start's scan, under the lock: every roster agent's directory
/// against its entry, chained directories not on it added, heads ahead
/// recorded, unchained states counted, refused markers listed. Writes the
/// roster when anything was added or moved on (and on a first run).
pub async fn check(data_dir: &Path, state_dir: &Path) -> Result<Scan, RosterError> {
    let _lock = lock(data_dir).await?;
    let (mut roster, first) = load_or_begin(data_dir)?;
    let dirs = agent_dirs(state_dir)?;
    let mut scan = Scan {
        first,
        ..Scan::default()
    };
    let mut changed = first;
    let written = roster.updated_at;
    let mut newest: Option<DateTime<Utc>> = None;
    let ids: BTreeSet<AgentId> = dirs.iter().chain(roster.agents.keys()).copied().collect();
    for id in ids {
        let dir = state_dir.join(id.to_string());
        let found = match look(&dir, id).await {
            Ok(f) => f,
            Err(e) => {
                tracing::error!(agent_id = %id, error = %e, "an agent directory couldn't be looked at for the roster");
                if roster.agents.contains_key(&id) {
                    scan.losses.insert(id, Loss::Io);
                }
                continue;
            }
        };
        // A refused marker counts while the agent is on the roster or has
        // a state: one `sealed-forget` left behind doesn't count forever
        let has_state = matches!(
            found,
            Found::Unchained { state: true }
                | Found::NoHead { state: true }
                | Found::HeadUnreadable { state: true }
                | Found::Head { state: true, .. }
        );
        if (has_state || roster.agents.contains_key(&id))
            && std::fs::exists(
                dir.join(agora_agentkit::state::storage::SEALED_DIR)
                    .join("refused"),
            )
            .unwrap_or(false)
        {
            scan.refused.insert(id);
        }
        let Some(entry) = roster.agents.get(&id).copied() else {
            match found {
                Found::Head { head, .. } => {
                    roster.merge(id, head);
                    scan.added.push(id);
                    changed = true;
                }
                Found::Unchained { state: true } => scan.unchained.push(id),
                // Chained with no readable head: the start check refuses
                // or holds it; adopted once it has one
                _ => {}
            }
            continue;
        };
        let loss = match found {
            Found::Gone => Some(Loss::DirMissing),
            Found::Unchained { .. } => Some(Loss::ChainMissing),
            Found::NoHead { .. } => Some(Loss::HeadMissing),
            Found::HeadUnreadable { .. } => Some(Loss::HeadUnreadable),
            Found::Head { head, state } => {
                if head.seq < entry.seq
                    || (head.seq == entry.seq && head.entry_hash != entry.entry_hash)
                {
                    Some(Loss::Behind {
                        roster: entry.seq,
                        found: head.seq,
                    })
                } else if !state {
                    Some(Loss::StateMissing)
                } else {
                    changed |= roster.merge(id, head);
                    if let Ok(at) =
                        std::fs::metadata(dir.join("state.json")).and_then(|m| m.modified())
                    {
                        let at = DateTime::<Utc>::from(at);
                        newest = Some(newest.map_or(at, |n| n.max(at)));
                    }
                    None
                }
            }
        };
        if let Some(loss) = loss {
            scan.losses.insert(id, loss);
        }
    }
    if !first
        && let Some(newest) = newest
        && newest - written > ROSTER_STALE
    {
        scan.stale = Some((written, newest));
    }
    if changed {
        save(data_dir, &mut roster)?;
    }
    if !scan.added.is_empty() {
        tracing::info!(
            event_type = "sealed_roster_added",
            agents = scan.added.len(),
            first,
            "chained agents added to the roster"
        );
    }
    scan.on_roster = roster.agents.keys().copied().collect();
    scan.vouched = roster
        .agents
        .iter()
        .filter_map(|(id, e)| Some((*id, e.vouched?)))
        .collect();
    Ok(scan)
}

/// Between the roster's updates while a run goes on.
pub const KEEP_INTERVAL: std::time::Duration = std::time::Duration::from_secs(60);

/// Keeps the roster up with the heads of the agents that run: every
/// [`KEEP_INTERVAL`] while the run goes on (after each batch of saves),
/// and once at its end. Reads each agent's `head.json` on its own task,
/// never under the reactors' storage lock; writes the roster only when a
/// head moved on.
pub struct Keeper {
    data_dir: PathBuf,
    state_dir: PathBuf,
    agents: Vec<AgentId>,
    seen: tokio::sync::Mutex<BTreeMap<AgentId, HeadRef>>,
    stop: tokio::sync::Notify,
}

impl Keeper {
    pub fn new(data_dir: &Path, state_dir: &Path, agents: Vec<AgentId>) -> Self {
        Self {
            data_dir: data_dir.to_path_buf(),
            state_dir: state_dir.to_path_buf(),
            agents,
            seen: Default::default(),
            stop: tokio::sync::Notify::new(),
        }
    }

    /// One pass: the heads that moved since the last, recorded.
    pub async fn keep(&self) {
        let mut seen = self.seen.lock().await;
        // Which moved, read outside the lock only to decide whether to
        // take it: what is recorded is read again inside it
        let mut moved = Vec::new();
        for id in &self.agents {
            let seal = AgentSeal::new(&self.state_dir.join(id.to_string()), *id);
            if let Ok(Some(h)) = seal.head().await
                && seen.get(id) != Some(&h.head)
            {
                moved.push(*id);
            }
        }
        if moved.is_empty() {
            return;
        }
        match record_heads(&self.data_dir, &self.state_dir, &moved).await {
            Ok(read) => seen.extend(read),
            Err(e) => tracing::error!(
                event_type = "sealed_roster_unwritten",
                agents = moved.len(),
                error = %e,
                "the sealed roster couldn't be updated; the next start checks against the heads it holds"
            ),
        }
    }

    pub fn spawn(self: &std::sync::Arc<Self>) -> tokio::task::JoinHandle<()> {
        let this = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(KEEP_INTERVAL) => this.keep().await,
                    _ = this.stop.notified() => return,
                }
            }
        })
    }

    /// Stop the loop, then one last pass.
    pub async fn finish(self: &std::sync::Arc<Self>, task: tokio::task::JoinHandle<()>) {
        self.stop.notify_one();
        if let Err(e) = task.await {
            tracing::error!(error = %e, "sealed roster task failed");
        }
        self.keep().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sealed::check::tests::{enrolled_local, state};

    /// A data dir with `n` agents chained locally; their ids and dirs.
    async fn fleet(root: &Path, n: usize) -> Vec<(AgentId, PathBuf)> {
        let mut out = Vec::new();
        for _ in 0..n {
            let (id, _, dir) = enrolled_local(root).await;
            out.push((id, dir));
        }
        out
    }

    async fn head(dir: &Path, id: AgentId) -> HeadRef {
        AgentSeal::new(dir, id).head().await.unwrap().unwrap().head
    }

    /// The first run adopts every chained directory and nothing else;
    /// writes the roster and `began`.
    #[tokio::test]
    async fn the_first_run_adopts_chained_directories_only() {
        let root = crate::sealed::TestDir::new("roster-first");
        let agents = fleet(&root, 2).await;
        // A state never chained, and leftover keys with no directory
        let plain = AgentId::new();
        let mut fs = agora_agentkit::reactor::FsStorage::new(root.join("state"));
        use agora_agentkit::reactor::Storage as _;
        fs.save_raw(plain, state(1), None).await.unwrap();
        std::fs::create_dir_all(root.join("secrets").join(AgentId::new().to_string())).unwrap();
        let scan = check(&root, &root.join("state")).await.unwrap();
        assert!(scan.losses.is_empty(), "{:?}", scan.losses);
        let mut added = scan.added.clone();
        added.sort();
        let mut want: Vec<_> = agents.iter().map(|a| a.0).collect();
        want.sort();
        assert_eq!(added, want);
        assert_eq!(scan.unchained, vec![plain]);
        assert!(root.join(ROSTER).exists() && root.join(BEGAN).exists());
        let roster = load(&root).unwrap().unwrap();
        assert_eq!(roster.agents.len(), 2);
        for (id, dir) in &agents {
            assert_eq!(roster.agents[id].seq, head(dir, *id).await.seq);
        }
        // Again: nothing new
        let scan = check(&root, &root.join("state")).await.unwrap();
        assert!(scan.added.is_empty() && scan.losses.is_empty());
    }

    /// A directory put back to an older consistent copy reads as fine to
    /// its chain; the roster catches it.
    #[tokio::test]
    async fn a_rollback_is_caught() {
        let root = crate::sealed::TestDir::new("roster-rollback");
        let (id, dir) = fleet(&root, 1).await.remove(0);
        let backup = root.join("backup");
        std::process::Command::new("cp")
            .args(["-a"])
            .arg(&dir)
            .arg(&backup)
            .status()
            .unwrap();
        check(&root, &root.join("state")).await.unwrap();
        crate::sealed::props::save(&root, id, state(2)).await;
        record_all(&root, [(id, head(&dir, id).await)])
            .await
            .unwrap();
        // The older copy back in place
        std::fs::rename(&dir, root.join("newer")).unwrap();
        std::fs::rename(&backup, &dir).unwrap();
        let scan = check(&root, &root.join("state")).await.unwrap();
        assert!(
            matches!(scan.losses[&id], Loss::Behind { .. }),
            "{:?}",
            scan.losses
        );
        assert_eq!(scan.losses[&id].refusal(), Refusal::RolledBack);
        // The roster keeps the newer head
        assert_eq!(
            load(&root).unwrap().unwrap().agents[&id].seq,
            head(&root.join("newer"), id).await.seq
        );
    }

    /// A directory removed whole, a lost chain, a lost head, a lost state.
    #[tokio::test]
    async fn removals_are_caught() {
        let root = crate::sealed::TestDir::new("roster-removal");
        let agents = fleet(&root, 4).await;
        check(&root, &root.join("state")).await.unwrap();
        let sealed = |d: &Path| d.join(agora_agentkit::state::storage::SEALED_DIR);
        std::fs::rename(&agents[0].1, root.join("gone")).unwrap();
        std::fs::rename(sealed(&agents[1].1), root.join("gone-sealed")).unwrap();
        std::fs::rename(
            sealed(&agents[2].1).join("head.json"),
            root.join("gone-head"),
        )
        .unwrap();
        std::fs::rename(agents[3].1.join("state.json"), root.join("gone-state")).unwrap();
        let scan = check(&root, &root.join("state")).await.unwrap();
        let found: Vec<Loss> = agents.iter().map(|a| scan.losses[&a.0]).collect();
        assert_eq!(
            found,
            [
                Loss::DirMissing,
                Loss::ChainMissing,
                Loss::HeadMissing,
                Loss::StateMissing
            ]
        );
        assert!(found.iter().all(|l| l.refusal().loss_shaped()));
    }

    /// A head ahead of the roster (a crash before its write) is accepted
    /// and recorded.
    #[tokio::test]
    async fn a_head_ahead_is_accepted() {
        let root = crate::sealed::TestDir::new("roster-ahead");
        let (id, dir) = fleet(&root, 1).await.remove(0);
        check(&root, &root.join("state")).await.unwrap();
        crate::sealed::props::save(&root, id, state(2)).await;
        let scan = check(&root, &root.join("state")).await.unwrap();
        assert!(scan.losses.is_empty());
        assert_eq!(
            load(&root).unwrap().unwrap().agents[&id].seq,
            head(&dir, id).await.seq
        );
    }

    /// Two writers at once (the continuous unit and the timer): each
    /// agent's higher seq wins, nothing lost.
    #[tokio::test]
    async fn concurrent_updates_merge() {
        let root = crate::sealed::TestDir::new("roster-concurrent");
        let ids: Vec<AgentId> = (0..40).map(|_| AgentId::new()).collect();
        let at = |seq| HeadRef {
            entry_hash: Sha256Hex::from([seq as u8; 32]),
            seq,
        };
        let mut tasks = tokio::task::JoinSet::new();
        for (n, id) in ids.iter().enumerate() {
            for seq in [1, 5, 3] {
                let (root, id) = (root.to_path_buf(), *id);
                tasks.spawn(async move { record_all(&root, [(id, at(seq + n as u64))]).await });
            }
        }
        while let Some(r) = tasks.join_next().await {
            r.unwrap().unwrap();
        }
        let roster = load(&root).unwrap().unwrap();
        for (n, id) in ids.iter().enumerate() {
            assert_eq!(roster.agents[id].seq, 5 + n as u64);
        }
    }

    /// The roster unreadable, newer, or missing after it began: refused.
    #[tokio::test]
    async fn an_unreadable_or_lost_roster_is_refused() {
        let root = crate::sealed::TestDir::new("roster-unreadable");
        fleet(&root, 1).await;
        check(&root, &root.join("state")).await.unwrap();
        let good = std::fs::read(root.join(ROSTER)).unwrap();
        std::fs::write(root.join(ROSTER), b"{ torn").unwrap();
        assert!(matches!(
            check(&root, &root.join("state")).await,
            Err(RosterError::Unreadable { .. })
        ));
        std::fs::write(root.join(ROSTER), br#"{"format": 99}"#).unwrap();
        assert!(matches!(
            check(&root, &root.join("state")).await,
            Err(RosterError::Newer { format: 99, .. })
        ));
        std::fs::rename(root.join(ROSTER), root.join("roster.aside")).unwrap();
        assert!(matches!(
            check(&root, &root.join("state")).await,
            Err(RosterError::Missing { .. })
        ));
        // Recording never re-creates a roster that began
        assert!(
            record_all(
                &root,
                [(
                    AgentId::new(),
                    HeadRef {
                        entry_hash: Sha256Hex::from([1; 32]),
                        seq: 1
                    }
                )]
            )
            .await
            .is_err()
        );
        assert!(!root.join(ROSTER).exists());
        std::fs::write(root.join(ROSTER), good).unwrap();
        assert!(check(&root, &root.join("state")).await.is_ok());
    }

    /// A command that lowers a head (rewrites `head.json`, then
    /// `replace`) can't be undone by the runner's Keeper: it reads heads
    /// inside the roster's lock (review of B2, 6). And the witness in the
    /// secrets root alone is enough to refuse a lost roster.
    #[tokio::test]
    async fn the_keeper_never_undoes_a_replace() {
        let root = crate::sealed::TestDir::new("roster-keeper");
        let (id, dir) = fleet(&root, 1).await.remove(0);
        crate::sealed::props::save(&root, id, state(2)).await;
        let keeper = Keeper::new(&root, &root.join("state"), vec![id]);
        keeper.keep().await;
        let high = head(&dir, id).await;
        assert_eq!(load(&root).unwrap().unwrap().agents[&id].seq, high.seq);
        // The command: head.json lowered to the genesis, then replace
        let genesis = crate::sealed::check::local_entries(&AgentSeal::new(&dir, id))
            .await
            .unwrap()[0]
            .clone();
        let low = HeadRef {
            entry_hash: genesis.entry_hash(),
            seq: genesis.entry.seq,
        };
        let path = dir
            .join(agora_agentkit::state::storage::SEALED_DIR)
            .join("head.json");
        let mut file: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        file["head"] = serde_json::to_value(low).unwrap();
        std::fs::write(&path, serde_json::to_vec(&file).unwrap()).unwrap();
        replace(&root, id, Some(low)).await.unwrap();
        // The Keeper's cache still has the high head; it reads again
        keeper.keep().await;
        keeper.keep().await;
        assert_eq!(load(&root).unwrap().unwrap().agents[&id].seq, low.seq);

        // Only the witness left: refused
        std::fs::rename(root.join(ROSTER), root.join("r.aside")).unwrap();
        std::fs::rename(root.join(BEGAN), root.join("b.aside")).unwrap();
        assert!(matches!(
            check(&root, &root.join("state")).await,
            Err(RosterError::Missing { .. })
        ));
    }

    /// An operator's command sets a head back on purpose; the vouch goes
    /// with the head it was for.
    #[tokio::test]
    async fn replace_sets_a_head_back() {
        let root = crate::sealed::TestDir::new("roster-replace");
        let id = AgentId::new();
        let at = |seq| HeadRef {
            entry_hash: Sha256Hex::from([seq as u8; 32]),
            seq,
        };
        record(&root, id, at(9)).await.unwrap();
        replace(&root, id, Some(at(4))).await.unwrap();
        assert_eq!(load(&root).unwrap().unwrap().agents[&id].seq, 4);
        replace(&root, id, None).await.unwrap();
        assert!(load(&root).unwrap().unwrap().agents.is_empty());
    }
}
