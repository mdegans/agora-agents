//! Sealed state, the runner's side (agentkit 0.147 `state`; agora
//! `docs/design-sealed-state.md`; the plan of 2026-11-02, "PR C").
//!
//! Every enrolled (chained: `sealed/chained`) agent (`agora-seed
//! sealed-enroll`, [`enroll`]) has every
//! save sealed and chained, locally, by agentkit's `Mirrored` storage;
//! whether the chain is also **uploaded** is a separate fact, the agent's
//! yes to the sealed-state offer ([`crate::consent::sealed_state`]) acted
//! on: [`uploading`], unless its operator stopped it ([`stop`]), together
//! [`uploads`]. The [`drain`]er uploads an uploading agent's outbox, away
//! from the reactor's storage lock; nothing of an agent that doesn't
//! upload leaves the machine (agentkit's drain gate, and
//! [`api::UploadGuard`] behind it). Before any session,
//! the [`check`] compares each enrolled agent's state with its last sealed
//! version: a state that matches nothing is **refused** (a notice kept for
//! the agent, the operator mailed), and the **whole run** with it (exit
//! [`REFUSED_RUN_EXIT`]): a state that matches nothing is likely
//! fleet-wide. `agora-seed sealed-restore` ([`restore`]) puts the last
//! sealed version back; `agora-seed sealed-accept --note` ([`accept`])
//! keeps the state as it is, and the agent is shown the note and a diff.
//! The drainer records when each version was sent and acknowledged
//! ([`receipts`]) and judges each daily anchor against them and the
//! governance log ([`anchors`], [`inclusion`]).
//! `agora-seed rotate-keys` ([`rotate`]) rotates an agent's keys and
//! re-wraps its state key.
//!
//! **What this reads and writes.** Agent state is read only to hash and
//! seal it; nothing here logs or prints any of its content. Writes are new
//! files under `<state>/<id>/sealed/` (agentkit's layout, plus this
//! module's `notice.json`) and the key files in `<secrets>/<id>/`
//! ([`keys`]), plus `sealed/anchors.json`, `sealed/inclusion.json` and
//! `sealed/published` ([`anchors`]), `sealed/receipts.json`
//! ([`receipts`]), and `sealed/uploading` with its copy
//! `<secrets>/<id>/sealed_uploading` ([`uploading`]). The agent's `state.json` is written only by
//! `sealed-restore`, through agentkit's `FsStorage`, which renames the
//! state it replaces to `state.<utc>.json` first. Nothing here deletes a
//! file: what is superseded is renamed aside.
//!
//! **Who is sealed.** An agent is sealed (chained) exactly when its
//! `sealed/chained` marker exists, stopped or not; it uploads exactly
//! when [`uploads`] says so. The `[sealed_state]` table only decides who is
//! asked. Turning it off must not stop the seal, or an enrolled agent's
//! next session would start from a state the seal never saw, and be
//! refused.

pub mod accept;
pub mod anchors;
pub mod api;
pub mod check;
pub mod cli;
pub mod crash;
#[cfg(test)]
mod crash_tests;
pub mod drain;
pub mod enroll;
pub mod forget;
#[cfg(test)]
mod guard_tests;
pub mod history;
pub mod history_index;
pub mod inclusion;
pub mod keys;
pub mod lost;
pub mod memory_tool;
pub mod notice;
#[cfg(test)]
pub(crate) mod props;
pub mod receipts;
pub mod restore;
pub mod roster;
pub mod rotate;
pub mod session;
pub mod sizes;
pub mod stop;
pub mod uploading;
pub mod version;

use std::path::{Path, PathBuf};

use agora_agentkit::ids::AgentId;
use agora_agentkit::state::outbox::AgoraOutbox;
use agora_agentkit::state::storage::{CHAINED, Enrolment, Mirrored, SEALED_DIR, SealKeys};

pub use keys::KeyStore;

/// The runner's exit code when the start check refused any agent, so the
/// whole run did not start. The systemd unit doesn't restart on it
/// (`RestartPreventExitStatus=`): only an operator's `sealed-restore` or
/// `sealed-accept` can resolve it. EX_TEMPFAIL's number, nothing else.
pub const REFUSED_RUN_EXIT: i32 = 75;

/// How the runner stamps the files it sets aside or rescues (`<utc>` in
/// `state_key.forgotten.<utc>.json`, `unsaved.<utc>.json`, …): UTC to the
/// millisecond, so that by name is by time. agentkit's FsStorage names its
/// archives the same way.
pub const STAMP_FORMAT: &str = "%Y%m%dT%H%M%S%3fZ";

/// Replace `path` with `bytes`, whole and 0600: a uniquely named temporary
/// sibling (so concurrent writers never share one), fsynced, renamed over
/// it, the directory fsynced. The directory is created if missing. On a
/// failure only the temporary is removed; the file stays as it was.
pub fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let dir = path.parent().expect("a file in a directory");
    std::fs::create_dir_all(dir)?;
    let tmp = path.with_extension(format!("{}.writing", uuid::Uuid::new_v4()));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let written = (|| {
        let mut file = options.open(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::rename(&tmp, path)
    })();
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written?;
    std::fs::File::open(dir)?.sync_all()
}

/// agentkit's sealed storage, as the runner configures it.
pub type MirroredStorage = Mirrored<
    agora_agentkit::reactor::FsStorage,
    AgoraOutbox<SealedEnrolment>,
    std::sync::Arc<SealedEnrolment>,
>;

/// The storage every reactor of a run saves through: agentkit's
/// [`Mirrored`], for every agent. An enrolled agent is chained whether or
/// not it uploads, stopped or not; [`SealedEnrolment::uploads`] decides
/// only whether its outbox is drained.
pub fn storage(
    state_dir: impl Into<PathBuf>,
    outbox: AgoraOutbox<SealedEnrolment>,
    enrolment: std::sync::Arc<SealedEnrolment>,
) -> MirroredStorage {
    Mirrored::new(
        agora_agentkit::reactor::FsStorage::new(state_dir.into()),
        outbox,
        enrolment,
    )
}

/// Whether the agent's chain is uploaded to Agora: marked uploading
/// ([`uploading`], which fails closed) and not stopped by its operator
/// ([`stop`], in either copy). Nothing of an agent for which this is false
/// may leave the machine: the drain ([`SealedEnrolment`]) and the API
/// ([`api::UploadGuard`]) both ask it.
pub fn uploads(agent_dir: &Path, keys: &KeyStore, id: AgentId) -> bool {
    !stop::is_stopped(agent_dir)
        && !stop::durable(keys, id).map_or(true, |d| d.is_some())
        && uploading::is_uploading(agent_dir, keys, id)
}

/// The runner's say on who uploads ([`uploads`]), with the keys in its
/// own secrets tree. Who is chained is `sealed/chained`'s alone.
#[derive(Debug, Clone)]
pub struct SealedEnrolment {
    state_dir: PathBuf,
    keys: KeyStore,
}

impl SealedEnrolment {
    pub fn new(state_dir: impl Into<PathBuf>, secrets_dir: impl Into<PathBuf>) -> Self {
        Self {
            state_dir: state_dir.into(),
            keys: KeyStore::new(secrets_dir),
        }
    }

    /// Over a [`KeyStore`] already open.
    pub fn with_keys(state_dir: impl Into<PathBuf>, keys: KeyStore) -> Self {
        Self {
            state_dir: state_dir.into(),
            keys,
        }
    }

    pub fn keys_store(&self) -> &KeyStore {
        &self.keys
    }
}

/// Whether `<agent_dir>/sealed/chained` exists (agentkit's [`CHAINED`];
/// `sealed/enrolled` before agentkit 0.158, moved by the start's
/// [`uploading::start_pass`]). A file that can't be
/// checked counts as enrolled: sealing (or refusing) an agent that isn't is
/// recoverable, writing an enrolled one unsealed is not.
pub fn is_enrolled(agent_dir: &Path) -> bool {
    match std::fs::exists(agent_dir.join(SEALED_DIR).join(CHAINED)) {
        Ok(enrolled) => enrolled,
        Err(e) => {
            tracing::error!(
                path = %agent_dir.display(),
                error = %e,
                "sealed-state enrolment unreadable; treated as enrolled"
            );
            true
        }
    }
}

/// Enrolled by either record: `sealed/chained`, or a state key in
/// `<secrets>/<id>/` (which survives a lost state directory).
pub fn enrolled_by_keys_or_marker(agent_dir: &Path, secrets_dir: &Path, id: AgentId) -> bool {
    is_enrolled(agent_dir)
        || std::fs::exists(secrets_dir.join(id.to_string()).join(keys::STATE_KEY_FILE))
            .unwrap_or(true)
}

impl Enrolment for SealedEnrolment {
    /// [`uploads`]: gates agentkit's outbox wake and drain only.
    fn uploads(&self, id: AgentId) -> bool {
        uploads(&self.state_dir.join(id.to_string()), &self.keys, id)
    }

    /// For every chained agent, uploading or not: the local chain is
    /// signed and sealed with them.
    fn keys(&self, id: AgentId) -> Option<SealKeys> {
        self.keys.seal_keys(id)
    }
}

impl api::UploadGate for SealedEnrolment {
    fn may_send(&self, id: AgentId) -> bool {
        uploads(&self.state_dir.join(id.to_string()), &self.keys, id)
    }

    fn may_read(&self, id: AgentId) -> bool {
        uploading::has_uploaded(&self.state_dir.join(id.to_string()), &self.keys, id)
    }
}

/// A scratch directory for tests, removed on drop. Synthetic state only.
#[cfg(test)]
pub struct TestDir(PathBuf);

#[cfg(test)]
impl TestDir {
    pub fn new(label: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("agora-seed-{label}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    /// In `/dev/shm` where there is one: for tests that fsync a lot (the
    /// property tests), which a disk makes slow.
    pub fn in_memory(label: &str) -> Self {
        let shm = Path::new("/dev/shm");
        if !shm.is_dir() {
            return Self::new(label);
        }
        let dir = shm.join(format!("agora-seed-{label}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

#[cfg(test)]
impl std::ops::Deref for TestDir {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

#[cfg(test)]
impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
