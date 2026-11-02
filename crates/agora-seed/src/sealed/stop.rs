//! `agora-seed sealed-stop --agent <id> --note "<why>"`: an operator stops
//! an agent's sealed backups, at the agent's request (the offer tells it to
//! ask its operator).
//!
//! Under the session guard and the agent's `sealed/lock`, after a drain so
//! nothing waiting is lost (refused if anything still waits): writes
//! `sealed/stopped` (`{at, note, shown}`, local; the note is shown to the
//! agent, not uploaded in v1). `enrolled`, `head.json`, the outbox and the
//! whole history stay. From then on nothing is sealed for the agent and no
//! failed seal is recorded ([`super::SealedStorage`] saves it plainly, as
//! FsStorage format 1), and the start check leaves it alone but for a line
//! under "## Your Sealed State".
//!
//! `sealed-enroll --agent` on a stopped agent resumes the same chain
//! ([`super::enroll`]): the current state sealed as the next snapshot,
//! `stopped` set aside, the state marked format 2 again. Never a new
//! genesis.

use std::path::{Path, PathBuf};

use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::Storage as _;
use agora_agentkit::state::storage::{AgentSeal, SEALED_DIR};
use anyhow::Context;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

const STOPPED: &str = "stopped";

/// `sealed/stopped`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stopped {
    pub at: DateTime<Utc>,
    /// The operator's note, shown to the agent.
    pub note: String,
    /// The first session after the stop has shown the note.
    #[serde(default)]
    pub shown: bool,
}

fn path(agent_dir: &Path) -> PathBuf {
    agent_dir.join(SEALED_DIR).join(STOPPED)
}

/// Whether the agent's backups are stopped. A marker that can't be checked
/// counts as stopped: sealing an agent that asked to stop is the worse
/// mistake.
pub fn is_stopped(agent_dir: &Path) -> bool {
    std::fs::exists(path(agent_dir)).unwrap_or(true)
}

fn write(agent_dir: &Path, stopped: &Stopped) -> std::io::Result<()> {
    let path = path(agent_dir);
    let dir = path.parent().expect("in sealed/");
    std::fs::create_dir_all(dir)?;
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(stopped)?)?;
    std::fs::File::open(&tmp)?.sync_all()?;
    std::fs::rename(&tmp, &path)?;
    std::fs::File::open(dir)?.sync_all()
}

impl Stopped {
    pub fn load(agent_dir: &Path) -> std::io::Result<Option<Self>> {
        match std::fs::read(path(agent_dir)) {
            Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// The line for the agent: the note at the first session after the
    /// stop, a short reminder after that.
    pub fn line(&self) -> String {
        let date = self.at.format("%-d %B %Y");
        if self.shown {
            format!(
                "Your sealed backups are stopped (since {date}); your earlier versions stay on \
                 Agora; your state isn't checked while stopped."
            )
        } else {
            format!(
                "Your sealed backups were stopped by your operator on {date}: {}. Your earlier \
                 versions stay on Agora. While stopped, your state isn't checked at the start of \
                 sessions.",
                self.note.trim_end_matches('.')
            )
        }
    }

    /// The note was seated in a session that saved.
    pub fn mark_shown(agent_dir: &Path) -> std::io::Result<()> {
        if let Some(mut s) = Self::load(agent_dir)?
            && !s.shown
        {
            s.shown = true;
            write(agent_dir, &s)?;
        }
        Ok(())
    }

    /// Resumed: kept as `stopped.cleared.<utc>`, never deleted.
    pub fn clear(agent_dir: &Path) -> std::io::Result<()> {
        let from = path(agent_dir);
        let at = Utc::now().format("%Y%m%dT%H%M%S%3fZ");
        std::fs::rename(
            &from,
            from.with_file_name(format!("{STOPPED}.cleared.{at}")),
        )
    }
}

/// Upload what waits, under no lock (the drain takes the agent's lock
/// itself). The command then checks, under the lock, that nothing waits.
pub type Drain = std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send>>;

/// Stop `id`'s sealed backups. `drain` uploads what waits first.
pub async fn stop(
    state_dir: &Path,
    id: AgentId,
    note: &str,
    drain: Drain,
) -> anyhow::Result<Stopped> {
    let note = note.trim();
    anyhow::ensure!(!note.is_empty(), "--note is required: the agent reads it");
    let agent_dir = state_dir.join(id.to_string());
    let seal = AgentSeal::new(&agent_dir, id);
    anyhow::ensure!(
        seal.is_enrolled().await?,
        "agent {id} is not enrolled in sealed state; nothing to stop"
    );
    anyhow::ensure!(
        !is_stopped(&agent_dir),
        "agent {id}'s backups are already stopped"
    );
    super::session::no_session(state_dir, id, false)?;
    // Never a way around sealed-accept: a refused agent, a save that went
    // unsealed, or a state that isn't its head's is resolved first
    anyhow::ensure!(
        !seal.is_refused().await?,
        "agent {id} is refused; resolve that with sealed-accept first"
    );
    anyhow::ensure!(
        seal.seal_failed().await?.is_none(),
        "agent {id}'s last save went unsealed; resolve that with sealed-accept first"
    );
    let current = agora_agentkit::reactor::FsStorage::new(state_dir)
        .load_raw(id)
        .await
        .context("the agent's state won't load")?
        .value;
    let local = agora_agentkit::state::storage::plaintext_hash(&current);
    match seal.recover(Some(&local)).await? {
        agora_agentkit::state::storage::Recovery::Intact(_)
        | agora_agentkit::state::storage::Recovery::Committed(_) => {}
        other => anyhow::bail!(
            "agent {id}'s state doesn't match its sealed head ({other:?}); resolve that with \
             sealed-accept first"
        ),
    }
    if let Err(e) = drain.await {
        anyhow::bail!(
            "the versions waiting to upload could not be uploaded ({e}); nothing was stopped. \
             Fix that first, so nothing waiting is lost"
        );
    }
    let lock = super::session::take(state_dir, id, false).await?;
    let waiting = seal.outbox().await?.len();
    anyhow::ensure!(
        waiting == 0,
        "{waiting} sealed version(s) still wait to upload (a held fork? resolve it with \
         sealed-accept --rebase); nothing was stopped"
    );
    let stopped = Stopped {
        at: Utc::now(),
        note: note.to_string(),
        shown: false,
    };
    write(&agent_dir, &stopped).context("writing sealed/stopped")?;
    drop(lock);
    tracing::warn!(
        event_type = "sealed_state_stopped",
        agent_id = %id,
        "the operator stopped an agent's sealed backups; its history stays"
    );
    Ok(stopped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sealed::check::tests::enrolled;

    fn drained() -> Drain {
        Box::pin(async { Ok(()) })
    }

    #[tokio::test]
    async fn stopping_drains_first_and_keeps_everything() {
        let root = crate::sealed::TestDir::new("stop");
        let (id, _keys, dir) = enrolled(&root).await;
        let state_dir = root.join("state");
        let seal = AgentSeal::new(&dir, id);
        assert!(
            stop(&state_dir, id, " ", drained()).await.is_err(),
            "a note"
        );
        // The drain fails: nothing stopped
        let failing: Drain = Box::pin(async { Err("unreachable".to_string()) });
        let e = stop(&state_dir, id, "asked to", failing)
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("could not be uploaded"), "{e}");
        assert!(!is_stopped(&dir));
        // Drained, but something still waits: refused
        let e = stop(&state_dir, id, "asked to", drained())
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("still wait to upload"), "{e}");
        // All uploaded: stopped, and everything kept
        for item in seal.outbox().await.unwrap() {
            seal.mark_sent(item.ord).await.unwrap();
        }
        let s = stop(&state_dir, id, "It asked me to.", drained())
            .await
            .unwrap();
        assert!(is_stopped(&dir));
        assert!(seal.is_enrolled().await.unwrap() && seal.head().await.unwrap().is_some());
        assert!(!seal.sent().await.unwrap().is_empty());
        assert!(
            s.line()
                .contains(": It asked me to. Your earlier versions stay on Agora. While stopped, your state isn't checked at the start of sessions."),
            "{}",
            s.line()
        );
        assert!(
            stop(&state_dir, id, "again", drained()).await.is_err(),
            "already"
        );
        Stopped::mark_shown(&dir).unwrap();
        let later = Stopped::load(&dir).unwrap().unwrap().line();
        assert!(
            later.starts_with("Your sealed backups are stopped (since ")
                && later.ends_with(
                    "); your earlier versions stay on Agora; your state isn't checked while \
                     stopped."
                ),
            "{later}"
        );
    }

    /// Stop is no way around sealed-accept.
    #[tokio::test]
    async fn stopping_refuses_a_refused_unsealed_or_changed_agent() {
        let root = crate::sealed::TestDir::new("stop-refused");
        let (id, _keys, dir) = enrolled(&root).await;
        let state_dir = root.join("state");
        let seal = AgentSeal::new(&dir, id);
        for item in seal.outbox().await.unwrap() {
            seal.mark_sent(item.ord).await.unwrap();
        }
        // Changed outside the seal
        let mut fs = agora_agentkit::reactor::FsStorage::new(&state_dir);
        agora_agentkit::reactor::Storage::save_raw(
            &mut fs,
            id,
            serde_json::json!({ "memory": "edited" }),
            None,
        )
        .await
        .unwrap();
        let e = stop(&state_dir, id, "n", drained())
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("doesn't match its sealed head"), "{e}");
        // Unsealed save on record
        let local = agora_agentkit::state::storage::plaintext_hash(
            &agora_agentkit::reactor::Storage::load_raw(&fs, id)
                .await
                .unwrap()
                .value,
        );
        seal.record_seal_failed(local, agora_agentkit::state::storage::SealFailureKind::Io)
            .await
            .unwrap();
        let e = stop(&state_dir, id, "n", drained())
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("went unsealed"), "{e}");
        // Refused
        seal.refuse("mismatch").await.unwrap();
        let e = stop(&state_dir, id, "n", drained())
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("is refused"), "{e}");
        assert!(!is_stopped(&dir));
    }
}
