//! Sealed state, the runner's side (agentkit 0.147 `state`; agora
//! `docs/design-sealed-state.md`; the plan of 2026-11-02, "PR C").
//!
//! An agent that said yes to the sealed-state offer
//! ([`crate::consent::sealed_state`]) and was enrolled by its operator
//! (`agora-seed sealed-enroll`, [`enroll`]) has every save sealed, chained
//! and queued for upload by agentkit's `Mirrored` storage; the [`drain`]er
//! uploads it, away from the reactor's storage lock. Before any session,
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
//! `sealed/published` ([`anchors`]) and `sealed/receipts.json`
//! ([`receipts`]). The agent's `state.json` is written only by
//! `sealed-restore`, through agentkit's `FsStorage`, which renames the
//! state it replaces to `state.<utc>.json` first. Nothing here deletes a
//! file: what is superseded is renamed aside.
//!
//! **Who is sealed.** An agent is sealed exactly when its
//! `sealed/enrolled` marker exists ([`SealedEnrolment`]); the
//! `[sealed_state]` table only decides who is asked. Turning it off must
//! not stop the seal, or an enrolled agent's next session would start from
//! a state the seal never saw, and be refused.

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
pub mod inclusion;
pub mod keys;
pub mod lost;
pub mod notice;
#[cfg(test)]
mod props;
pub mod receipts;
pub mod restore;
pub mod rotate;
pub mod session;
pub mod sizes;
pub mod stop;
pub mod version;

use std::path::{Path, PathBuf};

use agora_agentkit::ids::AgentId;
use agora_agentkit::state::outbox::AgoraOutbox;
use agora_agentkit::state::storage::{Enrolment, Mirrored, SEALED_DIR, SealKeys};

pub use keys::KeyStore;

/// The runner's exit code when the start check refused any agent, so the
/// whole run did not start. The systemd unit doesn't restart on it
/// (`RestartPreventExitStatus=`): only an operator's `sealed-restore` or
/// `sealed-accept` can resolve it. EX_TEMPFAIL's number, nothing else.
pub const REFUSED_RUN_EXIT: i32 = 75;

/// agentkit's sealed storage, as the runner configures it.
pub type MirroredStorage = Mirrored<
    agora_agentkit::reactor::FsStorage,
    AgoraOutbox<SealedEnrolment>,
    std::sync::Arc<SealedEnrolment>,
>;

/// The storage every reactor of a run saves through: agentkit's
/// [`Mirrored`] for every agent but one whose backups its operator stopped
/// ([`stop`]), which is loaded and saved plainly (FsStorage, format 1): no
/// seal, no failed seal recorded, no base.
pub struct SealedStorage {
    mirrored: MirroredStorage,
    plain: agora_agentkit::reactor::FsStorage,
}

impl SealedStorage {
    pub fn new(
        state_dir: impl Into<PathBuf>,
        outbox: AgoraOutbox<SealedEnrolment>,
        enrolment: std::sync::Arc<SealedEnrolment>,
    ) -> Self {
        let state_dir = state_dir.into();
        Self {
            mirrored: Mirrored::new(
                agora_agentkit::reactor::FsStorage::new(&state_dir),
                outbox,
                enrolment,
            ),
            plain: agora_agentkit::reactor::FsStorage::new(&state_dir),
        }
    }

    fn stopped(&self, id: AgentId) -> bool {
        stop::is_stopped(&self.plain.agent_dir(id))
    }
}

#[async_trait::async_trait]
impl agora_agentkit::reactor::Storage for SealedStorage {
    type Error = agora_agentkit::state::storage::MirroredError<
        agora_agentkit::reactor::storage::FsStorageError,
    >;
    type Version = agora_agentkit::state::verify::HeadRef;

    async fn save_raw(
        &mut self,
        id: AgentId,
        value: serde_json::Value,
        base: Option<&Self::Version>,
    ) -> Result<Option<Self::Version>, Self::Error> {
        if self.stopped(id) {
            self.plain
                .save_raw(id, value, None)
                .await
                .map_err(agora_agentkit::state::storage::MirroredError::Local)?;
            return Ok(None);
        }
        self.mirrored.save_raw(id, value, base).await
    }

    async fn load_raw(
        &self,
        id: AgentId,
    ) -> Result<agora_agentkit::reactor::Loaded<serde_json::Value, Self::Version>, Self::Error>
    {
        if self.stopped(id) {
            let loaded = self
                .plain
                .load_raw(id)
                .await
                .map_err(agora_agentkit::state::storage::MirroredError::Local)?;
            return Ok(agora_agentkit::reactor::Loaded {
                value: loaded.value,
                base: None,
            });
        }
        self.mirrored.load_raw(id).await
    }
}

/// Whether new versions of the agent in `agent_dir` are sealed: enrolled
/// and not stopped.
pub fn is_sealing(agent_dir: &Path) -> bool {
    is_enrolled(agent_dir) && !stop::is_stopped(agent_dir)
}

/// The runner's say on who is sealed: enrolled agents, with the keys in
/// its own secrets tree.
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

/// Whether `<agent_dir>/sealed/enrolled` exists. A file that can't be
/// checked counts as enrolled: sealing (or refusing) an agent that isn't is
/// recoverable, writing an enrolled one unsealed is not.
pub fn is_enrolled(agent_dir: &Path) -> bool {
    match std::fs::exists(agent_dir.join(SEALED_DIR).join("enrolled")) {
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

/// Enrolled by either record: `sealed/enrolled`, or a state key in
/// `<secrets>/<id>/` (which survives a lost state directory).
pub fn enrolled_by_keys_or_marker(agent_dir: &Path, secrets_dir: &Path, id: AgentId) -> bool {
    is_enrolled(agent_dir)
        || std::fs::exists(secrets_dir.join(id.to_string()).join(keys::STATE_KEY_FILE))
            .unwrap_or(true)
}

impl Enrolment for SealedEnrolment {
    /// Enrolled. `sealed-enroll` enrols only agents whose sealed-state
    /// ledger says yes, so this is the opt-in, as acted on.
    fn opted_in(&self, id: AgentId) -> bool {
        is_sealing(&self.state_dir.join(id.to_string()))
    }

    fn keys(&self, id: AgentId) -> Option<SealKeys> {
        self.keys.seal_keys(id)
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
