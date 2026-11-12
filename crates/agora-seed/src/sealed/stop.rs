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
//! **Unconditional, and nothing more is sent.** The command writes the
//! two stop markers first and always: `<secrets>/<id>/sealed_stopped`,
//! then `sealed/stopped` (`{at, note, shown}`; the note is shown to the
//! agent, not uploaded in v1). It needs only the uploading marker's lock
//! (so it can't interleave with an opt-in's write): not the session lock,
//! not a reachable Agora, not a stopped runner, not an agent whose state
//! is in order. A refused agent, a save that went unsealed, a state that
//! isn't its head's, a session in progress or one interrupted are all
//! stopped the same, and are still resolved by their own commands
//! afterwards. Nothing is uploaded first: the versions that were waiting
//! stay here with those saved after, and the command says how many. A
//! runner mid-drain asks [`super::uploads`] before each item (agentkit's
//! drain), so it sends at most the one item already on the wire. Either
//! marker alone stops the uploads, so a cut at any step after the first
//! write leaves the agent stopped; the missing copy is written by a rerun
//! (stopping twice is the same stop) or put back by the next start
//! ([`reconcile`]). `chained`, `head.json`, the outbox and the whole
//! history stay; so does `uploading`, but one in doubt is settled, once
//! the stop is written, from its readable copy
//! ([`super::uploading::settle_stopped`]): the agent then reads as one
//! that uploaded and is stopped.
//!
//! `sealed-enroll --resume --agent <id>` resumes the uploads on the same
//! chain ([`super::enroll`]). The versions not uploaded when the stop
//! began, and those saved while stopped, are the agent's and stay local,
//! set aside from the uploaded line at the resume (kept, never uploaded;
//! sending them at the agent's request is not built).
//!
//! **A stop from before the chain went on while stopped** (none exists in
//! the fleet): its saves went plainly, unsealed, so its start check finds
//! a mismatch and refuses it. The operator resolves that with
//! `sealed-accept --note`, so the agent is shown why; nothing seals the
//! unchecked interval silently.

use std::path::{Path, PathBuf};

use agora_agentkit::ids::AgentId;
use agora_agentkit::state::storage::{AgentSeal, SEALED_DIR};
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
                 at the start of each session; the versions not uploaded by then, and those \
                 saved from now on, stay here.",
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

/// What `sealed-stop` did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stop {
    /// The stop on file: this command's, or the one already there.
    pub stopped: Stopped,
    /// Its uploads were stopped before this command (in either copy); the
    /// stop on file stands, with its own note.
    pub already: bool,
    /// Sealed versions that had not been uploaded: they stay here. `None`:
    /// they couldn't be counted now (nothing was sent either way).
    pub kept_local: Option<usize>,
    /// What couldn't be done now, each with what finishes it.
    pub later: Vec<String>,
}

/// A stop marker's copy as found.
enum Found {
    Absent,
    Stop(Stopped),
    Unreadable,
}

fn found(path: &Path) -> Found {
    match std::fs::read(path) {
        Ok(bytes) => match serde_json::from_slice(&bytes) {
            Ok(s) => Found::Stop(s),
            Err(_) => Found::Unreadable,
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Found::Absent,
        Err(_) => Found::Unreadable,
    }
}

/// `stopped` written at `path` unless a readable stop is there already; a
/// copy that can't be read is set aside first (`<name>.unreadable.<utc>`),
/// never written over.
fn put(path: &Path, stopped: &Stopped) -> std::io::Result<()> {
    match found(path) {
        Found::Stop(_) => return Ok(()),
        Found::Absent => {}
        Found::Unreadable => {
            let at = Utc::now().format(crate::sealed::STAMP_FORMAT);
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            std::fs::rename(path, path.with_file_name(format!("{name}.unreadable.{at}")))?;
        }
    }
    write_at(path, stopped)
}

/// Stop uploading `id`'s sealed state: both markers, at once, whatever
/// state the agent or Agora is in (the module's docs). Nothing is sent.
/// Stopping an agent already stopped completes its markers and changes
/// nothing else. Refused only for an agent with no uploading marker at
/// all (nothing of it is uploaded, so there is nothing to stop), or when
/// neither marker could be written.
pub async fn stop(
    state_dir: &Path,
    keys: &super::KeyStore,
    id: AgentId,
    note: &str,
) -> anyhow::Result<Stop> {
    let note = note.trim();
    anyhow::ensure!(!note.is_empty(), "--note is required: the agent reads it");
    let agent_dir = state_dir.join(id.to_string());
    let durable_path = keys.agent_dir(id).join(DURABLE);
    let mut later = Vec::new();
    let (stopped, already) = {
        // An opt-in's write of the uploading marker waits for this, and
        // this for it: the stop is decided on a marker that isn't moving
        let _marker = super::uploading::lock(keys, id)
            .map_err(|e| anyhow::anyhow!("taking the uploading marker's lock: {e}"))?;
        let on_file = match (found(&durable_path), found(&path(&agent_dir))) {
            (Found::Stop(s), _) | (_, Found::Stop(s)) => Some(s),
            _ => None,
        };
        let already = on_file.is_some();
        anyhow::ensure!(
            already
                || super::uploading::state(&agent_dir, keys, id)
                    != super::uploading::Marker::Absent,
            "agent {id} has no uploading marker: nothing of it is uploaded, so there is nothing \
             to stop; its chain is kept locally either way"
        );
        let stopped = on_file.unwrap_or_else(|| Stopped {
            at: Utc::now(),
            note: note.to_string(),
            shown: false,
        });
        // The durable copy first: a stop must never be lost with sealed/
        let durable = put(&durable_path, &stopped);
        if durable.is_ok() {
            super::crash::point("stop:durable")?;
        }
        // Never a directory made for an agent that has none here
        let local = if std::fs::exists(agent_dir.join(SEALED_DIR)).unwrap_or(false) {
            put(&path(&agent_dir), &stopped)
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "the agent has no sealed/ directory here",
            ))
        };
        match (durable, local) {
            (Ok(()), Ok(())) => {}
            (Err(d), Err(l)) => anyhow::bail!(
                "agent {id} was NOT stopped: neither marker could be written (the secrets \
                 tree: {d}; sealed/stopped: {l})"
            ),
            (Err(d), Ok(())) => later.push(format!(
                "the stop's copy in the secrets tree couldn't be written ({d}): sealed/stopped \
                 stops the uploads meanwhile; run this again to write it"
            )),
            (Ok(()), Err(l)) => later.push(format!(
                "sealed/stopped couldn't be written ({l}): the copy in the secrets tree stops \
                 the uploads meanwhile; the next start puts it back, or run this again"
            )),
        }
        super::crash::point("stop:markers")?;
        // Stopped now, whatever follows. A marker in doubt is settled as
        // "uploaded, and stopped" from the copy that can be read
        match super::uploading::settle_stopped(&agent_dir, keys, id, &_marker) {
            Ok(_) => {}
            Err(e) => later.push(format!(
                "its uploading marker is in doubt and couldn't be settled ({e}): it is stopped \
                 all the same; run this again"
            )),
        }
        if matches!(
            super::uploading::state(&agent_dir, keys, id),
            super::uploading::Marker::Contradictory(_) | super::uploading::Marker::Unreadable(_)
        ) {
            later.push(
                "its uploading marker is still in doubt (no copy of it can be read): it is \
                 stopped all the same, and only an opt-in or resume on its own yes writes a new \
                 one"
                .to_string(),
            );
        }
        (stopped, already)
    };
    // Nothing is sent from here: only counted, for the operator
    let kept_local = match AgentSeal::new(&agent_dir, id).outbox().await {
        Ok(waiting) => Some(waiting.len()),
        Err(e) => {
            later.push(format!(
                "the versions waiting in sealed/outbox couldn't be counted ({e}); none is sent"
            ));
            None
        }
    };
    tracing::warn!(
        event_type = "sealed_state_stopped",
        agent_id = %id,
        already,
        kept_local,
        incomplete = later.len(),
        "the operator stopped an agent's sealed uploads; nothing more is sent, its history stays, and its chain goes on locally"
    );
    Ok(Stop {
        stopped,
        already,
        kept_local,
        later,
    })
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

    /// Stopping sends nothing, needs nothing of the agent or of Agora, and
    /// keeps everything: what waited to upload stays here.
    #[tokio::test]
    async fn stopping_is_unconditional_and_keeps_everything() {
        let root = crate::sealed::TestDir::new("stop");
        let (id, keys, dir) = enrolled(&root).await;
        let state_dir = root.join("state");
        let seal = AgentSeal::new(&dir, id);
        assert!(stop(&state_dir, &keys, id, " ").await.is_err(), "a note");
        let waiting = seal.outbox().await.unwrap().len();
        assert!(waiting > 0, "something waits to upload");
        assert!(super::super::uploads(&dir, &keys, id));
        let s = stop(&state_dir, &keys, id, "It asked me to.")
            .await
            .unwrap();
        assert!(!s.already && s.later.is_empty(), "{:?}", s.later);
        assert_eq!(s.kept_local, Some(waiting), "nothing was uploaded first");
        assert_eq!(seal.outbox().await.unwrap().len(), waiting);
        assert!(is_stopped(&dir));
        assert_eq!(durable(&keys, id).unwrap(), Some(s.stopped.clone()));
        assert_eq!(Stopped::load(&dir).unwrap(), Some(s.stopped.clone()));
        assert!(seal.is_enrolled().await.unwrap() && seal.head().await.unwrap().is_some());
        let s = s.stopped;
        assert!(
            s.line()
                .starts_with("Your operator stopped uploading your sealed history on ")
                && s.line().ends_with(
                    ": It asked me to. The versions already uploaded stay on Agora. Your state \
                     is still sealed and checked at the start of each session; the versions \
                     not uploaded by then, and those saved from now on, stay here."
                ),
            "{}",
            s.line()
        );
        assert!(!super::super::uploads(&dir, &keys, id), "no longer uploads");
        // Again: the same stop, its note and date kept
        let again = stop(&state_dir, &keys, id, "again").await.unwrap();
        assert!(again.already);
        assert_eq!(again.stopped, s);
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

    /// Nothing about the agent stands in a stop's way: a session's lock
    /// held, a session journal left, a refusal, a save that went unsealed,
    /// a state that isn't its head's, an uploading marker in doubt.
    #[tokio::test]
    async fn nothing_about_the_agent_refuses_a_stop() {
        for case in [
            "session", "journal", "changed", "unsealed", "refused", "doubt",
        ] {
            let root = crate::sealed::TestDir::new("stop-any");
            let (id, keys, dir) = enrolled(&root).await;
            let state_dir = root.join("state");
            let seal = AgentSeal::new(&dir, id);
            let mut fs = agora_agentkit::reactor::FsStorage::new(&state_dir);
            let mut held = None;
            match case {
                "session" => {
                    held = Some(
                        crate::sealed::session::take(&state_dir, id, false)
                            .await
                            .unwrap(),
                    )
                }
                "journal" => std::fs::write(fs.journal(id).path(), b"{}\n").unwrap(),
                "changed" => {
                    agora_agentkit::reactor::Storage::save_raw(
                        &mut fs,
                        id,
                        serde_json::json!({ "memory": "edited" }),
                        None,
                    )
                    .await
                    .unwrap();
                }
                "unsealed" => seal
                    .record_seal_failed(
                        agora_agentkit::govlog::Sha256Hex::from([7; 32]),
                        agora_agentkit::state::storage::SealFailureKind::Io,
                    )
                    .await
                    .unwrap(),
                "refused" => seal.refuse("mismatch").await.unwrap(),
                "doubt" => {
                    std::fs::remove_file(keys.agent_dir(id).join(crate::sealed::uploading::DURABLE))
                        .unwrap()
                }
                _ => unreachable!(),
            }
            let s = stop(&state_dir, &keys, id, "asked")
                .await
                .unwrap_or_else(|e| panic!("{case}: {e:#}"));
            assert!(s.later.is_empty(), "{case}: {:?}", s.later);
            assert!(
                matches!(
                    crate::sealed::uploading::state(&dir, &keys, id),
                    crate::sealed::uploading::Marker::Uploading(_)
                ),
                "{case}: no doubt left"
            );
            assert!(is_stopped(&dir), "{case}");
            assert!(durable(&keys, id).unwrap().is_some(), "{case}");
            assert!(!super::super::uploads(&dir, &keys, id), "{case}");
            drop(held);
        }
    }

    /// Cut at every step: once anything is written the agent is stopped,
    /// and a rerun (or the next start) completes the same stop.
    #[tokio::test]
    async fn a_stop_cut_at_any_step_leaves_the_agent_stopped() {
        for point in ["stop:durable", "stop:markers"] {
            for by_rerun in [true, false] {
                let root = crate::sealed::TestDir::new("stop-cut");
                let (id, keys, dir) = enrolled(&root).await;
                let state_dir = root.join("state");
                crate::sealed::crash::at(point);
                let e = stop(&state_dir, &keys, id, "asked").await.unwrap_err();
                assert!(e.to_string().contains("simulated crash"), "{point}: {e:#}");
                assert!(
                    !super::super::uploads(&dir, &keys, id),
                    "{point}: stopped from the first write"
                );
                let first = durable(&keys, id).unwrap().expect("the durable copy");
                if by_rerun {
                    let again = stop(&state_dir, &keys, id, "a second note").await.unwrap();
                    assert!(again.already, "{point}");
                    assert_eq!(again.stopped, first, "{point}: the same stop");
                } else {
                    reconcile_all(&state_dir, &keys);
                }
                assert_eq!(Stopped::load(&dir).unwrap(), Some(first.clone()), "{point}");
                assert_eq!(durable(&keys, id).unwrap(), Some(first), "{point}");
                assert!(!super::super::uploads(&dir, &keys, id));
            }
        }
    }

    /// One copy lost or unreadable: stopped all the same, and a rerun puts
    /// the other back (an unreadable one set aside, not written over).
    #[tokio::test]
    async fn a_rerun_completes_a_stop_with_one_copy() {
        let root = crate::sealed::TestDir::new("stop-one-copy");
        let (id, keys, dir) = enrolled(&root).await;
        let state_dir = root.join("state");
        let first = stop(&state_dir, &keys, id, "asked").await.unwrap().stopped;
        let durable_path = keys.agent_dir(id).join(DURABLE);
        std::fs::remove_file(&durable_path).unwrap();
        assert!(!super::super::uploads(&dir, &keys, id));
        assert!(stop(&state_dir, &keys, id, "n").await.unwrap().already);
        assert_eq!(durable(&keys, id).unwrap(), Some(first.clone()));
        std::fs::write(path(&dir), b"{ not json").unwrap();
        assert!(!super::super::uploads(&dir, &keys, id));
        let again = stop(&state_dir, &keys, id, "n").await.unwrap();
        assert!(again.already && again.later.is_empty());
        assert_eq!(Stopped::load(&dir).unwrap(), Some(first));
        let kept = std::fs::read_dir(dir.join(SEALED_DIR))
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with("stopped.unreadable.")
            })
            .count();
        assert_eq!(kept, 1, "set aside, not deleted");
        // sealed/ gone: the secrets copy alone stops it, and says what's left
        let (lost, _, lost_dir) = enrolled(&root).await;
        std::fs::rename(lost_dir.join(SEALED_DIR), root.join("sealed.lost")).unwrap();
        let s = stop(&state_dir, &keys, lost, "asked").await.unwrap();
        assert!(
            s.later
                .iter()
                .any(|l| l.contains("the next start puts it back")),
            "{:?}",
            s.later
        );
        assert!(!lost_dir.join(SEALED_DIR).exists(), "no directory made");
        assert!(!super::super::uploads(&lost_dir, &keys, lost));
    }

    /// Stop is about uploads: an agent that never uploaded has none to stop.
    #[tokio::test]
    async fn stopping_an_agent_that_doesnt_upload_is_refused() {
        let root = crate::sealed::TestDir::new("stop-local");
        let (id, keys, dir) = crate::sealed::check::tests::enrolled_local(&root).await;
        let e = stop(&root.join("state"), &keys, id, "asked")
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("no uploading marker"), "{e}");
        assert!(!is_stopped(&dir));
        assert!(durable(&keys, id).unwrap().is_none());
    }
}
