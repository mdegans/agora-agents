//! Sealed state, the runner's side (agentkit 0.147 `state`; agora
//! `docs/design-sealed-state.md`; the plan of 2026-11-02, "PR C").
//!
//! An agent that said yes to the sealed-state offer
//! ([`crate::consent::sealed_state`]) and was enrolled by its operator
//! (`agora-seed sealed-enroll`, [`enroll`]) has every save sealed, chained
//! and queued for upload by agentkit's `Mirrored` storage; the [`drain`]er
//! uploads it, away from the reactor's storage lock. Before any session,
//! the [`check`] compares each enrolled agent's state with its last sealed
//! version: a state that matches nothing is **refused** (left out, a
//! notice kept for the agent, the operator mailed), and only
//! `agora-seed sealed-accept` ([`accept`]) lets it run again.
//! `agora-seed rotate-keys` ([`rotate`]) rotates an agent's keys and
//! re-wraps its state key.
//!
//! **What this reads and writes.** Agent state is read only to hash and
//! seal it; nothing here logs or prints any of its content. Writes are new
//! files under `<state>/<id>/sealed/` (agentkit's layout, plus this
//! module's `notice.json`) and the key files in `<secrets>/<id>/`
//! ([`keys`]). Nothing here deletes a file: what is superseded is renamed
//! aside.
//!
//! **Who is sealed.** An agent is sealed exactly when its
//! `sealed/enrolled` marker exists ([`SealedEnrolment`]); the
//! `[sealed_state]` table only decides who is asked. Turning it off must
//! not stop the seal, or an enrolled agent's next session would start from
//! a state the seal never saw, and be refused.

pub mod accept;
pub mod api;
pub mod check;
pub mod cli;
pub mod drain;
pub mod enroll;
pub mod keys;
pub mod notice;
pub mod rotate;
pub mod session;
pub mod sizes;

use std::path::{Path, PathBuf};

use agora_agentkit::ids::AgentId;
use agora_agentkit::state::outbox::AgoraOutbox;
use agora_agentkit::state::storage::{Enrolment, Mirrored, SEALED_DIR, SealKeys};

pub use keys::KeyStore;

/// The storage every reactor of a run saves through.
pub type SealedStorage = Mirrored<
    agora_agentkit::reactor::FsStorage,
    AgoraOutbox<SealedEnrolment>,
    std::sync::Arc<SealedEnrolment>,
>;

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

impl Enrolment for SealedEnrolment {
    /// Enrolled. `sealed-enroll` enrols only agents whose sealed-state
    /// ledger says yes, so this is the opt-in, as acted on.
    fn opted_in(&self, id: AgentId) -> bool {
        is_enrolled(&self.state_dir.join(id.to_string()))
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
