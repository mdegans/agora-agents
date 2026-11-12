//! `agora-seed sealed-stop --agent <id> --note "<why>"`: an operator stops
//! uploading an agent's sealed state, at the agent's request (the offer
//! tells it to ask its operator).
//!
//! **Stop means stop uploading** (agora
//! `memory/project_local_chain_for_all.md`, 2026-11-11). The chain and the
//! start check go on: every save is still sealed and chained locally
//! (agentkit's `Mirrored`, as for any enrolled agent), and checked before
//! each session; only the upload stops ([`super::uploads`] is false while
//! stopped, so nothing of it is drained or asked of Agora).
//!
//! Under the session guard and the agent's `sealed/lock`, after a drain so
//! nothing waiting is lost (refused if anything still waits): writes
//! `sealed/stopped` (`{at, note, shown}`, local; the note is shown
//! to the agent, not uploaded in v1), and its durable copy in the secrets
//! tree first. `chained`, `uploading`, `head.json`, the outbox and the
//! whole history stay.
//!
//! `sealed-enroll --resume --agent <id>` resumes the uploads on the same
//! chain ([`super::enroll`]). The versions saved while stopped are the
//! agent's and stay local, set aside from the uploaded line at the resume
//! (kept, never uploaded; sending them at the agent's request is not
//! built).
//!
//! **A stop from before the chain went on while stopped** (none exists in
//! the fleet): its saves went plainly, unsealed, so its start check finds
//! a mismatch and refuses it. The operator resolves that with
//! `sealed-accept --note`, so the agent is shown why; nothing seals the
//! unchecked interval silently.

use std::path::{Path, PathBuf};

use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::Storage as _;
use agora_agentkit::state::storage::{AgentSeal, SEALED_DIR};
use anyhow::Context;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// Public for the uploading start pass, which sets a forgotten
// agent's leftover local stop aside.
/// `sealed/stopped`.
pub const STOPPED: &str = "stopped";
/// `<secrets>/<id>/sealed_stopped`: the stop, kept beside the keys too,
/// so it survives a lost `sealed/` (or state directory) and a restore
/// leaves the agent stopped. Authoritative when `sealed/stopped` is gone.
pub const DURABLE: &str = "sealed_stopped";

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

/// Whether the agent's uploads are stopped. A marker that can't be checked
/// counts as stopped: uploading for an agent that asked to stop is the
/// worse mistake.
pub fn is_stopped(agent_dir: &Path) -> bool {
    std::fs::exists(path(agent_dir)).unwrap_or(true)
}

fn write(agent_dir: &Path, stopped: &Stopped) -> std::io::Result<()> {
    write_at(&path(agent_dir), stopped)
}

fn write_at(path: &Path, stopped: &Stopped) -> std::io::Result<()> {
    super::write_private(path, &serde_json::to_vec_pretty(stopped)?)
}

/// The stop as kept in the secrets tree, if any.
pub fn durable(keys: &super::KeyStore, id: AgentId) -> std::io::Result<Option<Stopped>> {
    match std::fs::read(keys.agent_dir(id).join(DURABLE)) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// `sealed/stopped` put back from the secrets tree when it is gone: the
/// stop is the agent's request and outlives its `sealed/`. `Ok(true)`: put
/// back.
pub fn reconcile(agent_dir: &Path, keys: &super::KeyStore, id: AgentId) -> std::io::Result<bool> {
    if std::fs::exists(path(agent_dir))? {
        return Ok(false);
    }
    let Some(stopped) = durable(keys, id)? else {
        return Ok(false);
    };
    write(agent_dir, &stopped)?;
    tracing::warn!(
        event_type = "sealed_state_stop_reconciled",
        agent_id = %id,
        "sealed/stopped was missing; put back from the secrets tree (the agent stays stopped)"
    );
    Ok(true)
}

/// [`reconcile`] for every agent with a stop in the secrets tree and a
/// state directory here: at the start of a run, before states load.
pub fn reconcile_all(state_dir: &Path, keys: &super::KeyStore) {
    let ids = match keys.with_state_keys() {
        Ok(ids) => ids,
        Err(e) => {
            tracing::warn!(error = %e, "the secrets tree couldn't be listed for stops");
            return;
        }
    };
    for id in ids {
        let dir = state_dir.join(id.to_string());
        if !std::fs::exists(&dir).unwrap_or(false) {
            continue;
        }
        if let Err(e) = reconcile(&dir, keys, id) {
            tracing::error!(agent_id = %id, error = %e, "the durable stop couldn't be put back");
        }
    }
}

/// Resumed: the durable copy set aside too, as `sealed_stopped.cleared.<utc>`.
pub fn clear_durable(keys: &super::KeyStore, id: AgentId) -> std::io::Result<()> {
    let from = keys.agent_dir(id).join(DURABLE);
    if !std::fs::exists(&from)? {
        return Ok(());
    }
    let at = Utc::now().format(crate::sealed::STAMP_FORMAT);
    std::fs::rename(
        &from,
        from.with_file_name(format!("{DURABLE}.cleared.{at}")),
    )
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
        let date = self.at.format(super::text::DATE);
        if self.shown {
            format!(
                "Uploads of your sealed history are stopped (since {date}); the versions \
                 already uploaded stay on Agora; your state is still sealed and checked here."
            )
        } else {
            format!(
                "Your operator stopped uploading your sealed history on {date}: {}. The \
                 versions already uploaded stay on Agora. Your state is still sealed and checked \
                 at the start of each session; the versions saved from now on stay here.",
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
        let at = Utc::now().format(crate::sealed::STAMP_FORMAT);
        std::fs::rename(
            &from,
            from.with_file_name(format!("{STOPPED}.cleared.{at}")),
        )
    }
}

/// Upload what waits, under no lock (the drain takes the agent's lock
/// itself). The command then checks, under the lock, that nothing waits.
pub type Drain = std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send>>;

/// Stop uploading `id`'s sealed state. `drain` uploads what waits first.
pub async fn stop(
    state_dir: &Path,
    keys: &super::KeyStore,
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
        "agent {id}'s uploads are already stopped"
    );
    anyhow::ensure!(
        super::uploading::is_uploading(&agent_dir, keys, id),
        "agent {id} doesn't upload its sealed state (no uploading marker, or one in doubt): \
         nothing to stop; its chain is kept locally either way"
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
    // The durable copy first: a stop must never be lost with sealed/
    write_at(&keys.agent_dir(id).join(DURABLE), &stopped)
        .context("writing the stop into the secrets tree")?;
    write(&agent_dir, &stopped).context("writing sealed/stopped")?;
    drop(lock);
    tracing::warn!(
        event_type = "sealed_state_stopped",
        agent_id = %id,
        "the operator stopped an agent's sealed uploads; its history stays, and its chain goes on locally"
    );
    Ok(stopped)
}

/// Every sentence this module can tell an agent, for the one table of
/// agent-facing text ([`crate::sealed::text`]): add a new one here, or
/// its banned-phrase test never sees it.
#[cfg(test)]
pub(crate) fn agent_texts() -> Vec<crate::sealed::text::Text> {
    use crate::sealed::text::{Text, To};
    let stopped = |shown| Stopped {
        at: "2026-11-12T09:00:00Z".parse().expect("a date"),
        note: "It asked me to.".into(),
        shown,
    };
    vec![
        Text::new("Stopped::line", To::Uploaded, stopped(false).line()),
        Text::new("Stopped::line", To::Uploaded, stopped(true).line()),
    ]
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
        let (id, keys, dir) = enrolled(&root).await;
        let state_dir = root.join("state");
        let seal = AgentSeal::new(&dir, id);
        assert!(
            stop(&state_dir, &keys, id, " ", drained()).await.is_err(),
            "a note"
        );
        // The drain fails: nothing stopped
        let failing: Drain = Box::pin(async { Err("unreachable".to_string()) });
        let e = stop(&state_dir, &keys, id, "asked to", failing)
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("could not be uploaded"), "{e}");
        assert!(!is_stopped(&dir));
        // Drained, but something still waits: refused
        let e = stop(&state_dir, &keys, id, "asked to", drained())
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("still wait to upload"), "{e}");
        // All uploaded: stopped, and everything kept
        for item in seal.outbox().await.unwrap() {
            seal.mark_sent(item.ord).await.unwrap();
        }
        let s = stop(&state_dir, &keys, id, "It asked me to.", drained())
            .await
            .unwrap();
        assert!(is_stopped(&dir));
        assert!(seal.is_enrolled().await.unwrap() && seal.head().await.unwrap().is_some());
        assert!(!seal.sent().await.unwrap().is_empty());
        assert!(
            s.line()
                .starts_with("Your operator stopped uploading your sealed history on ")
                && s.line().ends_with(
                    ": It asked me to. The versions already uploaded stay on Agora. Your state \
                     is still sealed and checked at the start of each session; the versions \
                     saved from now on stay here."
                ),
            "{}",
            s.line()
        );
        assert!(!super::super::uploads(&dir, &keys, id), "no longer uploads");
        assert!(
            stop(&state_dir, &keys, id, "again", drained())
                .await
                .is_err(),
            "already"
        );
        Stopped::mark_shown(&dir).unwrap();
        let later = Stopped::load(&dir).unwrap().unwrap().line();
        assert!(
            later.starts_with("Uploads of your sealed history are stopped (since ")
                && later.ends_with(
                    "); the versions already uploaded stay on Agora; your state is still \
                     sealed and checked here."
                ),
            "{later}"
        );
    }

    /// Stop is no way around sealed-accept.
    #[tokio::test]
    async fn stopping_refuses_a_refused_unsealed_or_changed_agent() {
        let root = crate::sealed::TestDir::new("stop-refused");
        let (id, keys, dir) = enrolled(&root).await;
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
        let e = stop(&state_dir, &keys, id, "n", drained())
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
        let e = stop(&state_dir, &keys, id, "n", drained())
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("went unsealed"), "{e}");
        // Refused
        seal.refuse("mismatch").await.unwrap();
        let e = stop(&state_dir, &keys, id, "n", drained())
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("is refused"), "{e}");
        assert!(!is_stopped(&dir));
    }

    /// Stop is about uploads: an agent that doesn't upload has none to stop.
    #[tokio::test]
    async fn stopping_an_agent_that_doesnt_upload_is_refused() {
        let root = crate::sealed::TestDir::new("stop-local");
        let (id, keys, dir) = crate::sealed::check::tests::enrolled_local(&root).await;
        let e = stop(&root.join("state"), &keys, id, "asked", drained())
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("doesn't upload"), "{e}");
        assert!(!is_stopped(&dir));
    }
}
