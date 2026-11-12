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
//! the first time it is, and a witness in the secrets root ([`Witness`]).
//! A roster missing while either exists refuses the run, whether or not
//! chained agents are found: a wiped `state/` would take them with it. A
//! roster that can't be read, is in a newer format, or holds fewer agents
//! than its witness says refuses the run.
//!
//! **No roster and no trace of one** (the roster, `began` and the witness
//! all gone, or never there). Enrolment begins the roster: the first
//! agent `sealed-enroll` chains is its first entry ([`record`]). So with
//! no chained directory under `state/` this is a true first run, and a
//! roster begins empty. With **any** chained directory it is never one (a
//! chained agent was recorded when it was chained): the runner's start
//! refuses ([`RosterError::NotBegun`]) rather than begin anew, adopt every
//! directory as it finds it and switch the strays rule off; and no other
//! writer begins one over chained agents either. The operator begins it
//! deliberately, with a note that is logged and kept: `agora-seed
//! sealed-enroll --begin-roster --note "<why>"` ([`begin`]), which adopts
//! every directory that is chained and has a readable head and a
//! `state.json`, and nothing else (a chained directory without a state is
//! a forgotten agent's, or a loss `missing_states` finds by its keys).
//! The same command is the deliberate way out of a roster that is
//! missing, unreadable or shrunk ([`RosterError::way_out`]); it refuses a
//! roster that reads and matches its witness.
//!
//! A directory with a `state.json` that is neither chained nor on the
//! roster is counted for one alert a run ([`Scan::unchained`]), never
//! enrolled from here.
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
    /// An operator's command that will set this head ([`replace`]) has
    /// begun and not yet done so ([`expect`]): if it is cut after it
    /// cleared the agent's refusal, its rerun finds this and finishes the
    /// roster step rather than finding "nothing to accept".
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub pending: bool,
}

impl Entry {
    fn of(head: HeadRef) -> Self {
        Self {
            seq: head.seq,
            entry_hash: head.entry_hash,
            updated_at: Utc::now(),
            vouched: None,
            pending: false,
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
                    pending: e.pending,
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
    #[error(
        "{} is missing and nothing says one was ever begun, but {chained} chained agent \
         director(ies) are under {}: a roster is begun by enrolment, never over agents \
         already chained",
        path.display(),
        state_dir.display()
    )]
    NotBegun {
        path: PathBuf,
        state_dir: PathBuf,
        chained: usize,
    },
    #[error("{} reads and matches its witness: nothing to begin again (one agent is resolved with sealed-accept, sealed-restore or sealed-forget)", path.display())]
    Healthy { path: PathBuf },
    #[error("the roster couldn't be locked or written: {0}")]
    Io(#[from] std::io::Error),
}

/// The one command that begins a roster deliberately ([`begin`]).
pub const BEGIN_COMMAND: &str = "agora-seed sealed-enroll --begin-roster --note \"<why>\"";

impl RosterError {
    /// What the operator does about it, for the refusal's text and
    /// `LOCAL-CHAIN-IMPACT.md`. Every way that isn't "put the file back"
    /// goes through [`begin`], which needs a note and is logged.
    pub fn way_out(&self) -> String {
        match self {
            Self::Missing { .. } | Self::NotBegun { .. } => format!(
                "put sealed-roster.json back from a backup if there was one (the next start \
                 then compares every agent with it); otherwise, the runner stopped, begin one \
                 deliberately from the chained directories as they are: {BEGIN_COMMAND}"
            ),
            Self::Shrunk { .. } => format!(
                "an older or edited roster is in place: put the newer sealed-roster.json back \
                 from a backup; or, the runner stopped, begin it again from the chained \
                 directories as they are (the roster and the witness are set aside, not \
                 deleted): {BEGIN_COMMAND}. Deleting the witness by hand is not the way: it \
                 is the only record of how many agents the roster held"
            ),
            Self::Unreadable { .. } => format!(
                "check permissions and the disk first (the next start reads it again); if the \
                 file is damaged, put it back from a backup, or, the runner stopped, begin it \
                 again from the chained directories as they are (the damaged file is set \
                 aside): {BEGIN_COMMAND}"
            ),
            Self::Newer { .. } => "written by a newer runner than this one: upgrade the runner; \
                 don't begin it again (this runner can't read what the newer one recorded)"
                .to_string(),
            Self::StateDir { .. } | Self::Io(_) => {
                "check permissions and the disk; the next start tries again".to_string()
            }
            Self::Healthy { .. } => "nothing to do".to_string(),
        }
    }
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

/// The chained agent directories under `state_dir`, but `own`.
fn chained_dirs(state_dir: &Path, own: &[AgentId]) -> Result<usize, RosterError> {
    Ok(agent_dirs(state_dir)?
        .into_iter()
        .filter(|id| !own.contains(id))
        .filter(|id| super::is_enrolled(&state_dir.join(id.to_string())))
        .count())
}

/// Read under the lock: the roster, or a new one when there is neither
/// it nor any trace of one (`true`: a first one). A first one is begun
/// only while no agent directory but `own` (the agents the caller is
/// recording as it chains them) is chained: over chained agents it is
/// [`RosterError::NotBegun`], for every caller.
fn load_or_begin(
    data_dir: &Path,
    state_dir: &Path,
    own: &[AgentId],
) -> Result<(Roster, bool), RosterError> {
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
        None => match chained_dirs(state_dir, own)? {
            0 => Ok((Roster::default(), true)),
            chained => Err(RosterError::NotBegun {
                path: path(data_dir),
                state_dir: state_dir.to_path_buf(),
                chained,
            }),
        },
    }
}

/// What [`begin`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Begun {
    /// What it found: `missing`, `unreadable` or `shrunk`.
    pub was: &'static str,
    /// Chained directories with a head and a state, now on the roster.
    pub adopted: usize,
    /// The roster and witness it found, renamed aside (never deleted).
    pub set_aside: Vec<PathBuf>,
}

/// Begin the roster deliberately (`sealed-enroll --begin-roster --note`,
/// under the exclusive runner lock): when it is missing (with or without
/// a trace of an earlier one), unreadable, or holds fewer agents than its
/// witness says. Every directory that is chained and has a readable head
/// and a `state.json` is adopted as it is found; what it finds of an
/// earlier roster or witness is renamed aside; `note` is logged and kept
/// in `sealed-roster.began`. Refused for a roster that reads and matches
/// its witness ([`RosterError::Healthy`]) and for one in a newer format.
///
/// What it gives up, and why it takes a note: every head is taken as
/// found, so a directory rolled back or emptied *before* this is no
/// longer told apart by the roster (agents whose `sealed/` is gone are
/// still found by their state keys, `check::missing_states`).
pub async fn begin(data_dir: &Path, state_dir: &Path, note: &str) -> Result<Begun, RosterError> {
    let _lock = lock(data_dir).await?;
    let roster_path = path(data_dir);
    let witness = Witness::load(data_dir);
    let was = match load(data_dir) {
        Ok(None) => "missing",
        Ok(Some(r)) => match &witness {
            Ok(Some(w)) if w.agents > r.agents.len() => "shrunk",
            Err(_) => "unreadable",
            Ok(_) => return Err(RosterError::Healthy { path: roster_path }),
        },
        Err(e @ RosterError::Newer { .. }) => return Err(e),
        Err(_) => "unreadable",
    };
    let at = Utc::now().format(super::STAMP_FORMAT).to_string();
    let mut set_aside = Vec::new();
    for (file, name) in [
        (
            roster_path.clone(),
            format!("sealed-roster.{was}.{at}.json"),
        ),
        (
            Witness::path(data_dir),
            format!("sealed-roster.witness.{was}.{at}.json"),
        ),
    ] {
        if std::fs::exists(&file)? {
            let aside = file.with_file_name(name);
            std::fs::rename(&file, &aside)?;
            set_aside.push(aside);
        }
    }
    let mut roster = Roster::default();
    for id in agent_dirs(state_dir)? {
        if let Found::Head { head, state: true } = look(&state_dir.join(id.to_string()), id).await?
        {
            roster.merge(id, head);
        }
    }
    save(data_dir, &mut roster)?;
    // The note kept beside the roster, after what `began` already says
    let began = data_dir.join(BEGAN);
    let mut text = std::fs::read_to_string(&began).unwrap_or_default();
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(&format!(
        "{} begun again by the operator (was {was}; {} agents adopted): {}\n",
        Utc::now().to_rfc3339(),
        roster.agents.len(),
        note.trim().replace('\n', " ")
    ));
    super::write_private(&began, text.as_bytes())?;
    tracing::warn!(
        event_type = "sealed_roster_begun",
        was,
        adopted = roster.agents.len(),
        set_aside = set_aside.len(),
        note = %note.trim(),
        "the operator began the sealed roster again from the chained directories as they are"
    );
    Ok(Begun {
        was,
        adopted: roster.agents.len(),
        set_aside,
    })
}

/// Before `sealed-enroll` chains anything: `Err`
/// ([`RosterError::NotBegun`] and the rest) when the start would refuse
/// the roster as it is, so the command says so first. With no roster and
/// no chained directory (a true first enrolment) the roster is begun here,
/// empty: agents chained side by side (`--jobs`) then each find it there,
/// rather than each finding the others chained beside no roster.
pub async fn ready(data_dir: &Path, state_dir: &Path) -> Result<(), RosterError> {
    let _lock = lock(data_dir).await?;
    let (mut roster, first) = load_or_begin(data_dir, state_dir, &[])?;
    if first {
        save(data_dir, &mut roster)?;
    }
    Ok(())
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
pub async fn record(data_dir: &Path, id: AgentId, head: HeadRef) -> Result<(), RosterError> {
    record_all(data_dir, [(id, head)]).await
}

/// Record each head that is ahead of the roster's; written only if one is.
pub async fn record_all(
    data_dir: &Path,
    heads: impl IntoIterator<Item = (AgentId, HeadRef)>,
) -> Result<(), RosterError> {
    let _lock = lock(data_dir).await?;
    let heads: Vec<(AgentId, HeadRef)> = heads.into_iter().collect();
    let own: Vec<AgentId> = heads.iter().map(|(id, _)| *id).collect();
    let (mut roster, first) = load_or_begin(data_dir, &data_dir.join("state"), &own)?;
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
    let (mut roster, first) = load_or_begin(data_dir, state_dir, &[])?;
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
    let (mut roster, _) = load_or_begin(data_dir, &data_dir.join("state"), &[])?;
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
/// Called by `sealed-accept` (`accept`, `rebase`, a loss resolved,
/// `--keep`) and `sealed-restore` (a version, or a cut save) with the head
/// written, after `head.json` is; by `sealed-forget` with `None`.
pub async fn replace(
    data_dir: &Path,
    id: AgentId,
    head: Option<HeadRef>,
) -> Result<(), RosterError> {
    let _lock = lock(data_dir).await?;
    let (mut roster, first) = load_or_begin(data_dir, &data_dir.join("state"), &[id])?;
    match head {
        Some(head) => roster.agents.insert(id, Entry::of(head)),
        // No roster yet: nothing to take it off, and none begun for it (a
        // first start still adopts every chained agent)
        None if first => return Ok(()),
        None => roster.agents.remove(&id),
    };
    save(data_dir, &mut roster)
}

/// Before an operator's command clears an agent's refusal on its way to
/// [`replace`]: the agent's entry, if it has one, is marked
/// [`Entry::pending`], which only `replace` clears. Nothing is added for an
/// agent not on the roster (the next start adopts it once chained).
pub async fn expect(data_dir: &Path, id: AgentId) -> Result<(), RosterError> {
    let _lock = lock(data_dir).await?;
    let (mut roster, _) = load_or_begin(data_dir, &data_dir.join("state"), &[id])?;
    match roster.agents.get_mut(&id) {
        Some(e) if !e.pending => {
            e.pending = true;
            save(data_dir, &mut roster)
        }
        _ => Ok(()),
    }
}

/// [`expect`] for an operator's command, which knows the state directory.
pub async fn expect_for(state_dir: &Path, id: AgentId) -> anyhow::Result<()> {
    use anyhow::Context as _;
    let data_dir = state_dir
        .parent()
        .context("the state directory has no parent data directory")?;
    expect(data_dir, id).await.with_context(|| {
        format!("marking agent {id}'s entry in the sealed roster (nothing was changed)")
    })
}

/// Whether an operator's command left `id`'s roster step unfinished
/// ([`expect`] without its [`replace`]). `false` when the roster can't be
/// read.
pub fn is_pending(state_dir: &Path, id: AgentId) -> bool {
    state_dir
        .parent()
        .and_then(|data_dir| load(data_dir).ok().flatten())
        .and_then(|r| r.agents.get(&id).map(|e| e.pending))
        .unwrap_or(false)
}

/// [`replace`] for an operator's command, which knows the state directory
/// (`<data_dir>/state`): the roster is beside it. A failure is the
/// command's: the head is written, and the next start would refuse it.
pub async fn replace_for(
    state_dir: &Path,
    id: AgentId,
    head: Option<HeadRef>,
) -> anyhow::Result<()> {
    use anyhow::Context as _;
    let data_dir = state_dir
        .parent()
        .context("the state directory has no parent data directory")?;
    replace(data_dir, id, head).await.with_context(|| {
        format!(
            "recording agent {id}'s head in the sealed roster (its head is written; run the \
             command again, or the next start refuses it as rolled back)"
        )
    })
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
/// roster when anything was added or moved on (and on a first run, which
/// is one only with no chained directory: [`RosterError::NotBegun`]).
pub async fn check(data_dir: &Path, state_dir: &Path) -> Result<Scan, RosterError> {
    let _lock = lock(data_dir).await?;
    let (mut roster, first) = load_or_begin(data_dir, state_dir, &[])?;
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
                // Adopted only with its state: a chained directory left
                // without one is a forgotten agent's (`sealed-forget` took
                // it off) or a loss the start check finds by its keys
                // (`missing_states`); adopted, the next start would refuse
                // it as missing
                Found::Head { head, state: true } => {
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

    /// S8 (review of 2026-11-12). With the roster, `began` and the witness
    /// all gone, chained directories are never a first run: the start's
    /// scan refuses rather than begin anew (it used to adopt every
    /// directory as found and switch the strays rule off), and so does
    /// every other writer. The operator's `begin` adopts chained
    /// directories with a head and a state, and nothing else, keeps the
    /// note, and refuses a roster that is healthy.
    #[tokio::test]
    async fn a_roster_is_never_begun_over_chained_directories_but_by_the_operator() {
        let root = crate::sealed::TestDir::new("roster-first");
        let state_dir = root.join("state");
        let agents = fleet(&root, 2).await;
        // A state never chained, and leftover keys with no directory
        let plain = AgentId::new();
        let mut fs = agora_agentkit::reactor::FsStorage::new(&state_dir);
        use agora_agentkit::reactor::Storage as _;
        fs.save_raw(plain, state(1), None).await.unwrap();
        std::fs::create_dir_all(root.join("secrets").join(AgentId::new().to_string())).unwrap();
        // Healthy: nothing to begin again
        check(&root, &state_dir).await.unwrap();
        assert!(matches!(
            begin(&root, &state_dir, "why").await,
            Err(RosterError::Healthy { .. })
        ));
        // Every trace of the roster lost
        for (n, file) in [root.join(ROSTER), root.join(BEGAN), Witness::path(&root)]
            .into_iter()
            .enumerate()
        {
            std::fs::rename(&file, root.join(format!("lost-{n}"))).unwrap();
        }
        let e = check(&root, &state_dir).await.unwrap_err();
        assert!(matches!(e, RosterError::NotBegun { chained: 2, .. }), "{e}");
        assert!(
            e.way_out().contains("--begin-roster --note"),
            "{}",
            e.way_out()
        );
        assert!(matches!(
            ready(&root, &state_dir).await,
            Err(RosterError::NotBegun { .. })
        ));
        // No writer begins one either: not the keeper's, a vouch, or a
        // record or replace of one agent beside another chained one
        let (id, dir) = &agents[0];
        let h = head(dir, *id).await;
        assert!(record_heads(&root, &state_dir, &[*id]).await.is_err());
        assert!(record(&root, *id, h).await.is_err());
        assert!(replace(&root, *id, Some(h)).await.is_err());
        assert!(!root.join(ROSTER).exists() && !root.join(BEGAN).exists());
        // The operator's deliberate begin
        let begun = begin(&root, &state_dir, "the top level was lost\nand found")
            .await
            .unwrap();
        assert_eq!((begun.was, begun.adopted), ("missing", 2));
        let roster = load(&root).unwrap().unwrap();
        assert_eq!(roster.agents.len(), 2);
        for (id, dir) in &agents {
            assert_eq!(roster.agents[id].seq, head(dir, *id).await.seq);
        }
        let kept = std::fs::read_to_string(root.join(BEGAN)).unwrap();
        assert!(
            kept.contains("the top level was lost and found") && kept.contains("was missing"),
            "{kept}"
        );
        assert_eq!(Witness::load(&root).unwrap().unwrap().agents, 2);
        let scan = check(&root, &state_dir).await.unwrap();
        assert!(!scan.first && scan.added.is_empty() && scan.losses.is_empty());
        assert_eq!(scan.unchained, vec![plain]);
    }

    /// A true first run: no roster and no chained directory. The scan
    /// begins an empty one, and enrolment's record of the first agent it
    /// chains begins one too.
    #[tokio::test]
    async fn a_first_run_with_nothing_chained_begins_the_roster() {
        let root = crate::sealed::TestDir::new("roster-true-first");
        let scan = check(&root, &root.join("state")).await.unwrap();
        assert!(scan.first && scan.on_roster.is_empty());
        assert!(root.join(ROSTER).exists() && root.join(BEGAN).exists());
        let other = crate::sealed::TestDir::new("roster-true-first-enrol");
        let (id, _, dir) = crate::sealed::check::tests::chained_local(&other).await;
        record(&other, id, head(&dir, id).await).await.unwrap();
        assert_eq!(load(&other).unwrap().unwrap().agents.len(), 1);
    }

    /// The ways out of a roster the start refuses, each through `begin`
    /// (what it finds set aside, never deleted) or not at all: unreadable
    /// and shrunk are begun again; a newer format is not.
    #[tokio::test]
    async fn begin_is_the_way_out_of_unreadable_and_shrunk_never_of_newer() {
        let root = crate::sealed::TestDir::new("roster-ways-out");
        let state_dir = root.join("state");
        let agents = fleet(&root, 3).await;
        check(&root, &state_dir).await.unwrap();
        let good = std::fs::read(root.join(ROSTER)).unwrap();
        // Shrunk: an agent edited out
        let mut roster = load(&root).unwrap().unwrap();
        roster.agents.remove(&agents[0].0);
        std::fs::write(root.join(ROSTER), serde_json::to_vec(&roster).unwrap()).unwrap();
        let e = check(&root, &state_dir).await.unwrap_err();
        assert!(matches!(e, RosterError::Shrunk { .. }), "{e}");
        assert!(e.way_out().contains("--begin-roster"));
        let begun = begin(&root, &state_dir, "edited by mistake").await.unwrap();
        assert_eq!(
            (begun.was, begun.adopted, begun.set_aside.len()),
            ("shrunk", 3, 2)
        );
        assert!(
            begun.set_aside.iter().all(|p| p.exists()),
            "set aside, kept"
        );
        check(&root, &state_dir).await.unwrap();
        // Unreadable
        std::fs::write(root.join(ROSTER), b"{ torn").unwrap();
        let e = check(&root, &state_dir).await.unwrap_err();
        assert!(matches!(e, RosterError::Unreadable { .. }), "{e}");
        let begun = begin(&root, &state_dir, "torn write").await.unwrap();
        assert_eq!((begun.was, begun.adopted), ("unreadable", 3));
        check(&root, &state_dir).await.unwrap();
        // Newer: only an upgrade
        let mut newer: serde_json::Value = serde_json::from_slice(&good).unwrap();
        newer["format"] = serde_json::json!(FORMAT + 1);
        std::fs::write(root.join(ROSTER), serde_json::to_vec(&newer).unwrap()).unwrap();
        let e = begin(&root, &state_dir, "x").await.unwrap_err();
        assert!(matches!(e, RosterError::Newer { .. }), "{e}");
        assert!(e.way_out().contains("upgrade"));
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
