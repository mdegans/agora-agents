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
}

impl Entry {
    fn of(head: HeadRef) -> Self {
        Self {
            seq: head.seq,
            entry_hash: head.entry_hash,
            updated_at: Utc::now(),
        }
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
        match self.agents.get(&id) {
            Some(e) if e.seq >= head.seq => false,
            _ => {
                self.agents.insert(id, Entry::of(head));
                true
            }
        }
    }
}

/// Why the roster couldn't be used: each refuses the whole run.
#[derive(Debug, thiserror::Error)]
pub enum RosterError {
    #[error("{} is missing, but {BEGAN} says it was written before", path.display())]
    Missing { path: PathBuf },
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
    /// The directory couldn't be looked at (an I/O error).
    Io,
}

impl Loss {
    /// The refusal the agent is marked with.
    pub fn refusal(self) -> Refusal {
        match self {
            Self::DirMissing | Self::StateMissing => Refusal::StateMissing,
            Self::ChainMissing | Self::HeadMissing => Refusal::HeadMissing,
            Self::Behind { .. } => Refusal::RolledBack,
            Self::Io => Refusal::StateIo,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::DirMissing => "dir_missing",
            Self::ChainMissing => "chain_missing",
            Self::HeadMissing => "head_missing",
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
}

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
    match load(data_dir)? {
        Some(r) => Ok((r, false)),
        None if std::fs::exists(data_dir.join(BEGAN))? => Err(RosterError::Missing {
            path: path(data_dir),
        }),
        None => Ok((Roster::default(), true)),
    }
}

/// Write it, then `began` if it isn't there.
fn save(data_dir: &Path, roster: &mut Roster) -> Result<(), RosterError> {
    roster.updated_at = Utc::now();
    super::write_private(
        &path(data_dir),
        &serde_json::to_vec_pretty(roster).map_err(std::io::Error::other)?,
    )?;
    let began = data_dir.join(BEGAN);
    if !std::fs::exists(&began)? {
        super::write_private(&began, Utc::now().to_rfc3339().as_bytes())?;
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

/// Set `id`'s head to `head` whatever the roster held: for an operator's
/// command that sets the head back on purpose (`sealed-accept --rebase`
/// onto a shorter chain, a restore). `None` takes the agent off.
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
    NoHead,
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
        Ok(None) => Found::NoHead,
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
    let mut scan = Scan::default();
    let mut changed = first;
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
        if std::fs::exists(
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
            Found::NoHead => Some(Loss::HeadMissing),
            Found::HeadUnreadable { state } => (!state).then_some(Loss::StateMissing),
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
                    None
                }
            }
        };
        if let Some(loss) = loss {
            scan.losses.insert(id, loss);
        }
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
        let mut moved = Vec::new();
        for id in &self.agents {
            let seal = AgentSeal::new(&self.state_dir.join(id.to_string()), *id);
            if let Ok(Some(h)) = seal.head().await
                && seen.get(id) != Some(&h.head)
            {
                moved.push((*id, h.head));
            }
        }
        if moved.is_empty() {
            return;
        }
        match record_all(&self.data_dir, moved.iter().copied()).await {
            Ok(()) => seen.extend(moved),
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
