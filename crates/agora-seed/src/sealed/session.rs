//! What every operator command holds while it works on an agent: proof
//! that no session is in progress, the agent's session lock
//! ([`SESSION_LOCK`]) and its `sealed/lock`, for the whole command. Writes
//! to `sealed/` then go through agentkit's `AgentSeal::prepare` and
//! `commit` under that one lock ([`append`], [`genesis`]).
//!
//! # The session lock
//!
//! `<state>/<id>/session.lock`, an exclusive `flock` on a file of its own
//! (not agentkit's `sealed/lock`, not the directory lock `FsStorage` takes
//! for each save). Whoever holds it is the only one that may run a session
//! of the agent, or change it from outside a session:
//!
//! - a runner takes it for each agent it is about to give a session, after
//!   routing and before the agent's reactor is built, and holds it until
//!   the run has saved (or rescued) that agent's state and told it what it
//!   must be told ([`claim`]);
//! - every operator command that changes an agent takes it for that agent
//!   before its `sealed/lock` ([`take`], [`agent_lock`]), as does the
//!   start's promotion of a cut save (`crate::promote`).
//!
//! It is never waited for: a holder elsewhere means the agent is busy, and
//! the runner leaves it for this sweep ([`Claim::Busy`]); a command is
//! refused. It is released when dropped, and by the kernel when the process
//! ends however it ends (a panic, a signal's immediate exit, a kill): a
//! `flock` lives with its open file description, which Rust opens
//! close-on-exec, so no child process inherits it.
//!
//! **Lock order**, the same in every process: `runner.lock` →
//! `session.lock` → `sealed/lock` → agentkit's directory lock (each save's
//! `FsStorage` write) → the leaf locks (the uploading marker's). The first
//! two are only ever *tried* (`try_lock`, never a wait), so no process ever
//! waits for one of them while holding anything; a wait-for cycle needs a
//! wait on every lock in it, so none can pass through them. The waits that
//! remain (`sealed/lock`, then the directory lock inside a save) are taken
//! in agentkit's one order, by every holder. Across processes the same
//! argument holds: `flock` doesn't distinguish processes from descriptors,
//! and every holder follows the order above.
//!
//! **Descriptors.** A runner holds one descriptor per agent it has claimed:
//! every agent routed to it this run, from routing until the run ends (an
//! agent queued for its session stays claimed, so nothing changes it while
//! it waits). That is the cohort, not the loaded fleet; but a daily batch
//! cohort runs all at once, and a continuous sweep may route most of the
//! fleet (about 1,670 agents in November 2026), against systemd's default
//! soft limit of 1,024 for a user unit. The units raise it
//! (`LimitNOFILE=65536`), and the runner never claims past what the limit
//! leaves ([`fd_room`]): the rest stay due, with one alert.

use std::path::Path;

use agora_agentkit::enums::StateEntryKind;
use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::FsStorage;
use agora_agentkit::state::GenesisOrigin;
use agora_agentkit::state::codec::{self, KeyframePolicy};
use agora_agentkit::state::storage::{
    AgentSeal, Append, SealError, SealKeys, SealLock, plaintext_hash,
};
use agora_agentkit::state::verify::HeadRef;
use chrono::{DateTime, Utc};

/// Refuse while `id` has a session journal: a session is in progress, or
/// one was interrupted (its journal stays until the agent's next session
/// completes). With `refused_ok`, a journal older than the agent's
/// refusal is let pass: a refused agent runs no session, so it can't clear
/// one.
pub fn no_session(state_dir: &Path, id: AgentId, refused_ok: bool) -> anyhow::Result<()> {
    let journal = FsStorage::new(state_dir).journal(id);
    let modified = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    let Some(at) = modified(journal.path()) else {
        return Ok(());
    };
    if refused_ok {
        let refused = state_dir
            .join(id.to_string())
            .join(agora_agentkit::state::storage::SEALED_DIR)
            .join("refused");
        if modified(&refused).is_some_and(|r| r >= at) {
            return Ok(());
        }
    }
    anyhow::bail!(
        "agent {id} has a session in progress, or an interrupted one whose journal ({}) stays \
         until its next session completes; run this when it has no session",
        journal.path().display()
    )
}

/// `<data_dir>/runner.lock`: the runner holds it shared for its whole
/// life; `rotate-keys` needs it exclusive, so it runs only while no runner
/// does.
pub const RUNNER_LOCK: &str = "runner.lock";

fn runner_lock_file(data_dir: &Path) -> anyhow::Result<std::fs::File> {
    std::fs::create_dir_all(data_dir)?;
    Ok(std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(data_dir.join(RUNNER_LOCK))?)
}

/// The runner's shared hold, for as long as the returned file lives.
pub fn runner_shared(data_dir: &Path) -> anyhow::Result<std::fs::File> {
    let file = runner_lock_file(data_dir)?;
    match file.try_lock_shared() {
        Ok(()) => Ok(file),
        Err(std::fs::TryLockError::WouldBlock) => anyhow::bail!(
            "{} is held exclusively: `agora-seed rotate-keys` is running; start the runner \
             when it has finished",
            data_dir.join(RUNNER_LOCK).display()
        ),
        Err(std::fs::TryLockError::Error(e)) => Err(e.into()),
    }
}

/// The exclusive hold of the operator's commands that write (see
/// `sealed::cli::run`): refused while any runner holds it.
pub fn runner_exclusive(data_dir: &Path) -> anyhow::Result<std::fs::File> {
    let file = runner_lock_file(data_dir)?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(std::fs::TryLockError::WouldBlock) => anyhow::bail!(
            "a runner holds {}: stop the runner (`just runner-stop`) first. `sealed-enroll`, \
             `sealed-reoffer`, `sealed-accept`, `sealed-restore`, `sealed-forget` and \
             `rotate-keys` run only while no runner does (a session saving unsealed over a new \
             chain, signing with a key retired under it, or saving over a restored state)",
            data_dir.join(RUNNER_LOCK).display()
        ),
        Err(std::fs::TryLockError::Error(e)) => Err(e.into()),
    }
}

/// `<state>/<id>/session.lock` (see the [module docs](self)).
pub const SESSION_LOCK: &str = "session.lock";

/// An agent's session lock, held until dropped. `None` inside: the agent
/// has no directory, so no session of it can run (a command on an agent
/// whose directory is gone).
#[derive(Debug)]
pub struct SessionLock {
    _file: Option<std::fs::File>,
}

/// Try `id`'s session lock, never waiting: `Ok(None)` when another holds
/// it. The agent's directory is never created (`NotFound` when it is
/// gone).
pub fn try_session(state_dir: &Path, id: AgentId) -> std::io::Result<Option<SessionLock>> {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(state_dir.join(id.to_string()).join(SESSION_LOCK))?;
    match file.try_lock() {
        Ok(()) => Ok(Some(SessionLock { _file: Some(file) })),
        Err(std::fs::TryLockError::WouldBlock) => Ok(None),
        Err(std::fs::TryLockError::Error(e)) => Err(e),
    }
}

/// What a runner may do with an agent it is about to give a session.
#[derive(Debug)]
pub enum Claim {
    /// It runs, holding this until its state is saved and settled.
    Run(SessionLock),
    /// Another process holds the agent: a session in progress, one queued
    /// in another runner, or an operator's command. Left for this sweep;
    /// it stays due.
    Busy,
}

/// The runner's policy for an agent another process holds, in one place:
/// **skip it** this sweep ([`Claim::Busy`]), so two sessions never start
/// from one version (Steward, 2026-11-11: a fork is held, never
/// overwritten, but it is still a wasted session and an operator's chore).
///
/// A deliberate fork, the planned v1.1 merge (the same agent run in
/// parallel and the experiences reconciled, the agent writing the
/// synthesis), would replace `Busy` here with a claim on a *branch*: the
/// second session told it is one, its save sealed on the same parent as a
/// held fork by design rather than by accident (agentkit already holds a
/// save that parents onto an older head, `displaced`), a record of which
/// sessions branched where, and a merge step that ends the fork with the
/// agent's own synthesis before either line moves on. None of that exists
/// yet; until it does, this is the only answer.
pub fn claim(state_dir: &Path, id: AgentId) -> std::io::Result<Claim> {
    Ok(match try_session(state_dir, id)? {
        Some(lock) => Claim::Run(lock),
        None => Claim::Busy,
    })
}

/// An operator command's hold on `id`'s session lock: refused while a
/// runner (or another command) holds it. An agent with no directory has
/// no session to exclude, and its directory isn't made for the lock.
pub fn agent_lock(state_dir: &Path, id: AgentId) -> anyhow::Result<SessionLock> {
    match try_session(state_dir, id) {
        Ok(Some(lock)) => Ok(lock),
        Ok(None) => anyhow::bail!(
            "agent {id} is held by another process ({}): a runner has it in a session or queued \
             for one, or another command is working on it; run this when it has finished",
            state_dir.join(id.to_string()).join(SESSION_LOCK).display()
        ),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(SessionLock { _file: None }),
        Err(e) => Err(e.into()),
    }
}

/// Descriptors kept free for everything but session locks: sockets, log
/// files, and each save's own (its directory lock, `sealed/lock`, the tmp
/// and the fsyncs).
pub const RESERVED_FDS: usize = 256;

/// How many more session locks this process can hold: its soft limit on
/// open files, less those open now and [`RESERVED_FDS`]. `None` where it
/// can't be read (no `/proc`): no cap.
pub fn fd_room() -> Option<usize> {
    let limits = std::fs::read_to_string("/proc/self/limits").ok()?;
    let soft = soft_open_files(&limits)?;
    let open = std::fs::read_dir("/proc/self/fd").ok()?.count();
    Some(soft.saturating_sub(open).saturating_sub(RESERVED_FDS))
}

/// The soft "Max open files" of a `/proc/<pid>/limits`; `unlimited` is
/// `usize::MAX`.
fn soft_open_files(limits: &str) -> Option<usize> {
    let line = limits.lines().find(|l| l.starts_with("Max open files"))?;
    let soft = line
        .trim_start_matches("Max open files")
        .split_whitespace()
        .next()?;
    if soft == "unlimited" {
        return Some(usize::MAX);
    }
    soft.parse().ok()
}

/// What an operator command holds on an agent ([`take`]): its session
/// lock and its `sealed/lock`, released together when dropped. Derefs to
/// the [`SealLock`] agentkit's writes ask for.
#[derive(Debug)]
pub struct Taken {
    seal: SealLock,
    _session: SessionLock,
}

impl std::ops::Deref for Taken {
    type Target = SealLock;
    fn deref(&self) -> &SealLock {
        &self.seal
    }
}

/// [`no_session`], the agent's session lock ([`agent_lock`]), then its
/// `sealed/lock`, held until dropped (lock order: see the module docs).
pub async fn take(state_dir: &Path, id: AgentId, refused_ok: bool) -> anyhow::Result<Taken> {
    no_session(state_dir, id, refused_ok)?;
    let session = agent_lock(state_dir, id)?;
    let seal = AgentSeal::new(&state_dir.join(id.to_string()), id)
        .lock()
        .await?;
    // A session that began while we waited
    no_session(state_dir, id, refused_ok)?;
    Ok(Taken {
        seal,
        _session: session,
    })
}

/// What a process saw of an agent's files before it read its state:
/// `state.json` and `sealed/head.json`, by identity and size. Every save
/// replaces `state.json` by a rename and every commit `head.json`, so a
/// save by anyone else in between changes it. A runner compares it again
/// under the agent's session lock before giving it a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Seen {
    state: Option<Stamp>,
    head: Option<Stamp>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Stamp {
    dev: u64,
    ino: u64,
    len: u64,
    modified: Option<std::time::SystemTime>,
}

impl Stamp {
    fn of(path: &Path) -> Option<Self> {
        use std::os::unix::fs::MetadataExt;
        let m = std::fs::metadata(path).ok()?;
        Some(Self {
            dev: m.dev(),
            ino: m.ino(),
            len: m.len(),
            modified: m.modified().ok(),
        })
    }
}

impl Seen {
    pub fn of(agent_dir: &Path) -> Self {
        Self {
            state: Stamp::of(&agent_dir.join("state.json")),
            head: Stamp::of(
                &agent_dir
                    .join(agora_agentkit::state::storage::SEALED_DIR)
                    .join("head.json"),
            ),
        }
    }
}

/// `AgentSeal::append_state`, under a lock the caller holds.
pub async fn append(
    seal: &AgentSeal,
    lock: &SealLock,
    keys: &SealKeys,
    policy: &KeyframePolicy,
    kind: StateEntryKind,
    state: &serde_json::Value,
    state_at: DateTime<Utc>,
) -> Result<HeadRef, SealError> {
    let head = seal.head().await?.ok_or(SealError::NoHead)?;
    let plaintext = codec::plaintext(state);
    let prepared = seal
        .prepare(
            lock,
            keys,
            policy,
            Append {
                kind,
                plaintext: &plaintext,
                plaintext_sha256: (kind == StateEntryKind::Snapshot).then(|| plaintext_hash(state)),
                state_at,
                parent: Some(head.head),
            },
        )
        .await?;
    seal.commit(lock, &prepared).await
}

/// `AgentSeal::genesis`, under a lock the caller holds.
pub async fn genesis(
    seal: &AgentSeal,
    lock: &SealLock,
    keys: &SealKeys,
    origin: &GenesisOrigin,
    state_at: DateTime<Utc>,
) -> Result<HeadRef, SealError> {
    let plaintext = agora_agentkit::govlog::canonical_json(&serde_json::to_value(origin)?);
    let prepared = seal
        .prepare(
            lock,
            keys,
            &KeyframePolicy::default(),
            Append {
                kind: StateEntryKind::Genesis,
                plaintext: &plaintext,
                plaintext_sha256: None,
                state_at,
                parent: None,
            },
        )
        .await?;
    seal.commit(lock, &prepared).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_journal_blocks_the_commands_unless_older_than_a_refusal() {
        let root = crate::sealed::TestDir::new("session-guard");
        let id = AgentId::new();
        let dir = root.join(id.to_string());
        std::fs::create_dir_all(&dir).unwrap();
        assert!(no_session(&root, id, false).is_ok());
        let journal = FsStorage::new(&*root).journal(id);
        std::fs::write(journal.path(), b"{}\n").unwrap();
        let e = no_session(&root, id, false).unwrap_err().to_string();
        assert!(e.contains("session in progress"), "{e}");
        assert!(no_session(&root, id, true).is_err(), "not refused");
        std::thread::sleep(std::time::Duration::from_millis(20));
        AgentSeal::new(&dir, id).refuse("mismatch").await.unwrap();
        assert!(
            no_session(&root, id, true).is_ok(),
            "older than the refusal"
        );
        assert!(no_session(&root, id, false).is_err());
        assert!(take(&root, id, false).await.is_err());
        let _lock = take(&root, id, true).await.unwrap();
    }

    #[test]
    fn rotate_keys_needs_the_runner_stopped() {
        let root = crate::sealed::TestDir::new("runner-lock");
        let runner = runner_shared(&root).unwrap();
        let second = runner_shared(&root).unwrap();
        let e = runner_exclusive(&root).unwrap_err().to_string();
        assert!(e.contains("stop the runner"), "{e}");
        drop((runner, second));
        let rotating = runner_exclusive(&root).unwrap();
        assert!(
            runner_shared(&root)
                .unwrap_err()
                .to_string()
                .contains("rotate-keys` is running")
        );
        drop(rotating);
    }
}
