//! The drainer: uploads each enrolled agent's outbox during the run (woken
//! by every committed entry, and every [`INTERVAL`] besides) and once more
//! before the runner exits. Never inside a save: agentkit's `Mirrored`
//! only queues, under the reactor's storage lock.
//!
//! An agent the start check held (a fork, a missing head, a failed seal)
//! is not drained. One the server answers with a head that moved on is
//! held for the rest of the run ([`AlertKind::SealedStateFork`] or
//! [`AlertKind::SealedStateHeadMissing`]). A failed upload is retried at
//! the next wake, and raises [`AlertKind::SealedStateUploadFailing`], which
//! the alerter rate-limits per agent.
//!
//! With [`Anchors`], the drainer also does the anchor work
//! ([`super::anchors`]) for each agent not held whose outbox is empty:
//! once at [`Drainer::finish`] and at most every
//! [`super::anchors::PROOF_INTERVAL`] while the run goes on (never at the
//! start check), it publishes the anchor of an agent that chose `publish`
//! and hasn't been, and fetches and checks its proofs.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use agora_agentkit::ids::AgentId;
use agora_agentkit::state::outbox::AgoraOutbox;
use agora_agentkit::state::storage::AgentSeal;
use agora_agentkit::state::verify::HeadStatus;

use super::anchors::{PROOF_INTERVAL, VerifiedAnchors};
use super::api::SealedApi;
use super::check::{Hold, hold_alert};
use super::{KeyStore, SealedEnrolment};
use crate::alerts::{Alert, AlertKind, Alerter};

/// Between drains when nothing wakes it.
pub const INTERVAL: Duration = Duration::from_secs(60);

/// What the anchor work needs: the API and keys to sign with, where the
/// agents' directories are, and the governance log verified this run.
pub struct Anchors {
    pub api: Arc<dyn SealedApi>,
    pub keys: KeyStore,
    pub state_dir: PathBuf,
    pub verified: VerifiedAnchors,
}

pub struct Drainer {
    outbox: AgoraOutbox<SealedEnrolment>,
    anchors: Option<Anchors>,
    /// When the anchor work last ran: the run's start, at first.
    anchored_at: Mutex<Instant>,
    /// The enrolled agents of this run, with their names.
    agents: Vec<(AgentId, String)>,
    held: Mutex<BTreeSet<AgentId>>,
    alerts: Alerter,
    stop: tokio::sync::Notify,
}

impl Drainer {
    pub fn new(
        outbox: AgoraOutbox<SealedEnrolment>,
        agents: Vec<(AgentId, String)>,
        held: impl IntoIterator<Item = AgentId>,
        alerts: Alerter,
    ) -> Self {
        Self {
            outbox,
            anchors: None,
            anchored_at: Mutex::new(Instant::now()),
            agents,
            held: Mutex::new(held.into_iter().collect()),
            alerts,
            stop: tokio::sync::Notify::new(),
        }
    }

    /// Do the anchor work too ([`Anchors`]).
    pub fn with_anchors(mut self, anchors: Anchors) -> Self {
        self.anchors = Some(anchors);
        self
    }

    /// The anchor work, for every agent not held whose outbox is empty.
    pub async fn anchor_all(&self) {
        let Some(a) = &self.anchors else {
            return;
        };
        *self.anchored_at.lock().expect("anchored lock") = Instant::now();
        for (id, name) in &self.agents {
            if self.held(*id) {
                continue;
            }
            let dir = a.state_dir.join(id.to_string());
            if !super::is_sealing(&dir) {
                continue;
            }
            match AgentSeal::new(&dir, *id).outbox().await {
                Ok(items) if items.is_empty() => {}
                Ok(_) => continue,
                Err(e) => {
                    tracing::warn!(agent_id = %id, error = %e, "sealed outbox unreadable; no anchor work");
                    continue;
                }
            }
            let Some(signing) = a.keys.signing(*id) else {
                continue;
            };
            if let Err(e) =
                super::anchors::publish_if_chosen(&dir, *id, &signing, a.api.as_ref()).await
            {
                tracing::warn!(
                    event_type = "sealed_state_anchor_failed",
                    agent_id = %id,
                    what = "publish",
                    error = %e,
                    "publishing the agent's anchor failed; retried later"
                );
                self.alerts
                    .notify(super::anchors::anchor_alert(*id, name, "publish", &e));
            }
            if let Err(e) = super::anchors::fetch_proofs(
                &dir,
                *id,
                &a.keys,
                &signing,
                a.api.as_ref(),
                &a.verified,
            )
            .await
            {
                tracing::warn!(
                    event_type = "sealed_state_anchor_failed",
                    agent_id = %id,
                    what = "proof",
                    kind = e.kind(),
                    error = %e,
                    "an anchor proof was not kept"
                );
                self.alerts.notify(super::anchors::anchor_alert(
                    *id,
                    name,
                    "proof",
                    &e.to_string(),
                ));
            }
        }
    }

    fn anchors_due(&self) -> bool {
        self.anchors.is_some()
            && self.anchored_at.lock().expect("anchored lock").elapsed() >= PROOF_INTERVAL
    }

    fn held(&self, id: AgentId) -> bool {
        self.held.lock().expect("held lock").contains(&id)
    }

    /// One pass over every agent not held.
    pub async fn drain_all(&self) {
        for (id, name) in &self.agents {
            if !self.held(*id) {
                self.drain_one(*id, name).await;
            }
        }
    }

    async fn drain_one(&self, id: AgentId, name: &str) {
        match self.outbox.drain(id).await {
            Ok(report) => {
                if report.uploaded > 0 {
                    tracing::info!(
                        event_type = "sealed_state_uploaded",
                        agent_id = %id,
                        uploaded = report.uploaded,
                        waiting = report.waiting,
                        "sealed versions uploaded"
                    );
                }
                let hold = match report.status {
                    HeadStatus::Same | HeadStatus::Pending => return,
                    HeadStatus::LocalMissing => Hold::HeadMissing,
                    HeadStatus::ServerAhead | HeadStatus::Diverged => Hold::Fork,
                };
                self.held.lock().expect("held lock").insert(id);
                tracing::warn!(
                    event_type = "sealed_state_held",
                    agent_id = %id,
                    status = ?report.status,
                    waiting = report.waiting,
                    "sealed uploads held for the rest of the run"
                );
                self.alerts
                    .notify(hold_alert(hold, id, name, Some(&report.held)));
            }
            Err(e) => {
                tracing::warn!(
                    event_type = "sealed_state_upload_failing",
                    agent_id = %id,
                    error = %e,
                    "sealed upload failed; retried at the next wake"
                );
                self.alerts.notify(
                    Alert::new(
                        AlertKind::SealedStateUploadFailing,
                        format!("{name}: sealed versions could not be uploaded"),
                    )
                    .agent(name, id)
                    .detail("error", e),
                );
            }
        }
    }

    /// Drain until [`Self::finish`]: at every committed entry, and every
    /// [`INTERVAL`].
    pub fn spawn(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let this = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = this.outbox.notified() => {}
                    _ = tokio::time::sleep(INTERVAL) => {}
                    _ = this.stop.notified() => return,
                }
                this.drain_all().await;
                if this.anchors_due() {
                    this.anchor_all().await;
                }
            }
        })
    }

    /// Stop the background loop, then drain once more, then the anchor
    /// work.
    pub async fn finish(self: &Arc<Self>, task: tokio::task::JoinHandle<()>) {
        self.stop.notify_one();
        if let Err(e) = task.await {
            tracing::error!(error = %e, "sealed-state drainer task failed");
        }
        self.drain_all().await;
        self.anchor_all().await;
    }
}
