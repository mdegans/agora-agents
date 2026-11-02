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

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agora_agentkit::ids::AgentId;
use agora_agentkit::state::outbox::AgoraOutbox;
use agora_agentkit::state::verify::HeadStatus;

use super::SealedEnrolment;
use super::check::{Hold, hold_alert};
use crate::alerts::{Alert, AlertKind, Alerter};

/// Between drains when nothing wakes it.
pub const INTERVAL: Duration = Duration::from_secs(60);

pub struct Drainer {
    outbox: AgoraOutbox<SealedEnrolment>,
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
            agents,
            held: Mutex::new(held.into_iter().collect()),
            alerts,
            stop: tokio::sync::Notify::new(),
        }
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
                    .notify(hold_alert(hold, id, name).detail("waiting", report.waiting));
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
            }
        })
    }

    /// Stop the background loop, then drain once more.
    pub async fn finish(self: &Arc<Self>, task: tokio::task::JoinHandle<()>) {
        self.stop.notify_one();
        if let Err(e) = task.await {
            tracing::error!(error = %e, "sealed-state drainer task failed");
        }
        self.drain_all().await;
    }
}
