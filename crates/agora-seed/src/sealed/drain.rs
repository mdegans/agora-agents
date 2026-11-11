//! The drainer: uploads each enrolled agent's outbox during the run (woken
//! by every committed entry, and every [`INTERVAL`] besides) and once more
//! before the runner exits. Never inside a save: agentkit's `Mirrored`
//! only queues, under the reactor's storage lock.
//!
//! An agent that doesn't upload ([`super::uploads`]: never said yes, or
//! stopped) is chained but never drained: agentkit's drain answers
//! [`Drained::LocalOnly`] without reading or sending anything, and the
//! drainer records [`DrainOutcome::LocalOnly`] for it: no receipts, no
//! anchor work, no hold, no alert, however long its outbox grows.
//!
//! An agent the start check held (a fork, a missing head, a failed seal)
//! is not drained. One the server answers with a head that moved on is
//! held for the rest of the run ([`AlertKind::SealedStateFork`] or
//! [`AlertKind::SealedStateHeadMissing`]). A failed upload is retried at
//! the next wake, and raises [`AlertKind::SealedStateUploadFailing`], which
//! the alerter rate-limits per agent.
//!
//! Every drain records its upload times around it
//! ([`super::receipts::drain`]), against the outbox numbers each agent had
//! when the drainer was made, before any session of this run saved.
//!
//! With [`Anchors`], the drainer also does the anchor work
//! ([`super::anchors`]) for each agent not held whose outbox is empty:
//! once at [`Drainer::finish`] and at most every
//! [`super::anchors::PROOF_INTERVAL`] while the run goes on, it publishes
//! the anchor of an agent that chose `publish` and hasn't been, and judges
//! each anchor not yet judged ([`super::anchors::check_anchors`]): a
//! failure is one [`AlertKind::SealedStateAnchorWrong`] per pass, never
//! repeated. The start check runs the same judging pass first, for an
//! agent in step with Agora, so a session's verdict says what was found
//! for the newest anchor ([`super::check::check_with`]); the drainer's
//! passes then judge what came after.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use agora_agentkit::ids::AgentId;
use agora_agentkit::state::outbox::{AgoraOutbox, Drained};
use agora_agentkit::state::storage::AgentSeal;
use agora_agentkit::state::verify::HeadStatus;

use super::anchors::{PROOF_INTERVAL, VerifiedAnchors};
use super::api::SealedApi;
use super::check::{Hold, hold_alert};
use super::{KeyStore, SealedEnrolment};
use crate::alerts::{Alert, AlertKind, Alerter};

/// A drain Agora refused as unprocessable (422): the entry itself is
/// wrong, so a retry can only fail the same way.
pub fn rejected(e: &agora_agentkit::state::outbox::DrainError) -> bool {
    matches!(
        e,
        agora_agentkit::state::outbox::DrainError::Client(
            agora_agentkit::client::Error::Status { status, .. }
        ) if status.as_u16() == 422
    )
}

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

/// What the drainer's last pass came to for one agent ([`Drainer::last`]).
/// Typed so a verdict can say which: [`Self::LocalOnly`] is "nothing of
/// this agent is on Agora from this run", never "in step with Agora".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainOutcome {
    /// The agent doesn't upload: nothing read, sent or recorded.
    LocalOnly,
    /// No `sealed/chained`: no chain to send.
    NotEnrolled,
    /// Uploaded `uploaded`; Agora's head then `status`, `waiting` left.
    Ran {
        uploaded: usize,
        status: HeadStatus,
        waiting: usize,
    },
    /// Uploading was switched off mid-drain, after `uploaded`.
    Withdrawn { uploaded: usize },
    /// The upload failed (retried at the next wake, or held after a 422).
    Failed,
}

pub struct Drainer {
    outbox: AgoraOutbox<SealedEnrolment>,
    /// Where the agents' directories are.
    state_dir: PathBuf,
    /// Each agent's next outbox number when the drainer was made
    /// ([`super::receipts::baseline`]), raised past every number if its
    /// receipts can't be written.
    baselines: Mutex<HashMap<AgentId, u64>>,
    anchors: Option<Anchors>,
    /// When the anchor work last ran: the run's start, at first.
    anchored_at: Mutex<Instant>,
    /// The enrolled agents of this run, with their names.
    agents: Vec<(AgentId, String)>,
    held: Mutex<BTreeSet<AgentId>>,
    /// Each agent's last drain, as it came out.
    last: Mutex<HashMap<AgentId, DrainOutcome>>,
    alerts: Alerter,
    stop: tokio::sync::Notify,
}

impl Drainer {
    /// `baselines`: [`super::receipts::baselines`] of `agents`, read
    /// before any session of the run saves.
    pub fn new(
        outbox: AgoraOutbox<SealedEnrolment>,
        state_dir: PathBuf,
        baselines: HashMap<AgentId, u64>,
        agents: Vec<(AgentId, String)>,
        held: impl IntoIterator<Item = AgentId>,
        alerts: Alerter,
    ) -> Self {
        Self {
            outbox,
            state_dir,
            baselines: Mutex::new(baselines),
            anchors: None,
            anchored_at: Mutex::new(Instant::now()),
            agents,
            held: Mutex::new(held.into_iter().collect()),
            last: Mutex::new(HashMap::new()),
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
            // Nothing of a local-only agent is on Agora to judge, and no
            // publish choice of it may be read or sent
            if !super::uploads(&dir, &a.keys, *id) {
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
            let result = super::anchors::check_anchors(
                &dir,
                *id,
                &a.keys,
                &signing,
                a.api.as_ref(),
                &a.verified,
                self.alerts.sends(AlertKind::SealedStateAnchorWrong),
            )
            .await;
            super::anchors::alert_pass(&self.alerts, *id, name, &result);
        }
    }

    fn anchors_due(&self) -> bool {
        self.anchors.is_some()
            && self.anchored_at.lock().expect("anchored lock").elapsed() >= PROOF_INTERVAL
    }

    fn held(&self, id: AgentId) -> bool {
        self.held.lock().expect("held lock").contains(&id)
    }

    /// What `id`'s last drain this run came to; `None` before its first,
    /// or while it is held. For the verdict (step B2), which must say
    /// "nothing of this is on Agora" for [`DrainOutcome::LocalOnly`].
    #[allow(dead_code)]
    pub fn last(&self, id: AgentId) -> Option<DrainOutcome> {
        self.last.lock().expect("last lock").get(&id).copied()
    }

    fn record(&self, id: AgentId, outcome: DrainOutcome) {
        self.last.lock().expect("last lock").insert(id, outcome);
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
        // An agent the drainer wasn't told of counts nothing as this
        // process's own (sound: no `sent` time recorded)
        let mut baseline = self
            .baselines
            .lock()
            .expect("baselines lock")
            .get(&id)
            .copied()
            .unwrap_or(u64::MAX);
        let dir = self.state_dir.join(id.to_string());
        let drained = super::receipts::drain(&self.outbox, &dir, id, &mut baseline).await;
        self.baselines
            .lock()
            .expect("baselines lock")
            .insert(id, baseline);
        let report = match drained {
            Ok(Drained::Ran(report)) => report,
            Ok(Drained::LocalOnly) => {
                tracing::debug!(agent_id = %id, "local-only: not drained");
                return self.record(id, DrainOutcome::LocalOnly);
            }
            Ok(Drained::NotEnrolled) => return self.record(id, DrainOutcome::NotEnrolled),
            Ok(Drained::Withdrawn { uploaded }) => {
                tracing::info!(
                    event_type = "sealed_state_upload_withdrawn",
                    agent_id = %id,
                    uploaded,
                    "uploading was switched off mid-drain; the rest stays local"
                );
                return self.record(id, DrainOutcome::Withdrawn { uploaded });
            }
            Err(e) => return self.failed(id, name, e),
        };
        self.record(
            id,
            DrainOutcome::Ran {
                uploaded: report.uploaded,
                status: report.status,
                waiting: report.waiting,
            },
        );
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

    /// A drain that failed: held for the run after a 422, else retried at
    /// the next wake; alerted either way.
    fn failed(&self, id: AgentId, name: &str, e: agora_agentkit::state::outbox::DrainError) {
        self.record(id, DrainOutcome::Failed);
        // Agora refused the entry itself (e.g. `time_went_back`): a bug,
        // never fixed by retrying. Held for the run, told
        if rejected(&e) {
            self.held.lock().expect("held lock").insert(id);
            tracing::error!(
                event_type = "sealed_state_upload_rejected",
                agent_id = %id,
                error = %e,
                "Agora rejected a sealed version (422); uploads held for the rest of the run"
            );
            self.alerts.notify(
                Alert::new(
                    AlertKind::SealedStateUploadFailing,
                    format!(
                        "{name}: Agora rejected a sealed version as malformed (422): a bug; \
                             uploads held for this run"
                    ),
                )
                .agent(name, id)
                .detail("error", e),
            );
        } else {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// An Agora that answers every request 422 `time_went_back`, counting.
    async fn rejecting() -> (url::Url, Arc<AtomicUsize>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(AtomicUsize::new(0));
        let counted = hits.clone();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                counted.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 65536];
                    let _ = sock.read(&mut buf).await;
                    let body = "{\"error\":\"time_went_back\"}";
                    let resp = format!(
                        "HTTP/1.1 422 Unprocessable Entity\r\ncontent-type: application/json\r\n\
                         content-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });
        (format!("http://{addr}/").parse().unwrap(), hits)
    }

    /// A held agent gets no anchor work, so nothing is judged while its
    /// history is unsettled; an agent not held does.
    #[tokio::test]
    async fn a_held_agent_is_not_judged() {
        let root = crate::sealed::TestDir::new("drain-held-anchors");
        let (id, keys, dir) = crate::sealed::check::tests::enrolled(&root).await;
        crate::sealed::anchors::tests::uploaded(&dir, id).await;
        let api = Arc::new(crate::sealed::api::fake::FakeApi::new());
        let verified = crate::sealed::anchors::VerifiedAnchors::new([(
            "ANC-2026-0002".parse().unwrap(),
            agora_agentkit::govlog::Sha256Hex::from([0; 32]),
            chrono::Utc::now(),
        )]);
        let make = |held: Vec<AgentId>| {
            let outbox = AgoraOutbox::new(
                agora_agentkit::client::Client::new("http://127.0.0.1:9/".parse().unwrap())
                    .unwrap(),
                root.join("state"),
                Arc::new(SealedEnrolment::new(
                    root.join("state"),
                    root.join("secrets"),
                )),
            );
            Drainer::new(
                outbox,
                root.join("state"),
                HashMap::new(),
                vec![(id, "tarn".into())],
                held,
                Alerter::off(),
            )
            .with_anchors(Anchors {
                api: api.clone(),
                keys: keys.clone(),
                state_dir: root.join("state"),
                verified: verified.clone(),
            })
        };
        make(vec![id]).anchor_all().await;
        assert!(api.calls.lock().unwrap().is_empty(), "held: nothing asked");
        make(vec![]).anchor_all().await;
        assert!(
            api.calls.lock().unwrap().iter().any(|c| c == "proofs"),
            "not held: judged"
        );
    }

    /// A chained agent that doesn't upload, and one whose uploads were
    /// stopped, over several saves and every drainer pass: chained
    /// locally, and not one request reaches Agora (the drain, the
    /// receipts, the anchor work, the guarded API). Synthetic state only.
    #[tokio::test]
    async fn a_local_only_agent_makes_no_request_end_to_end() {
        use agora_agentkit::reactor::Storage as _;
        let root = crate::sealed::TestDir::new("drain-local-only");
        let state_dir = root.join("state");
        let (local, keys, local_dir) = crate::sealed::check::tests::enrolled_local(&root).await;
        let (stopped, _, stopped_dir) = crate::sealed::check::tests::enrolled(&root).await;
        let seal = AgentSeal::new(&stopped_dir, stopped);
        for item in seal.outbox().await.unwrap() {
            seal.mark_sent(item.ord).await.unwrap();
        }
        crate::sealed::stop::stop(
            &state_dir,
            &keys,
            stopped,
            "asked",
            Box::pin(async { Ok(()) }),
        )
        .await
        .unwrap();
        // Agora: counts every connection (and answers each 422)
        let (url, hits) = rejecting().await;
        let enrolment = Arc::new(SealedEnrolment::new(&state_dir, root.join("secrets")));
        let client = agora_agentkit::client::Client::new(url).unwrap();
        let outbox = AgoraOutbox::new(client.clone(), &state_dir, enrolment.clone());
        let mut storage = crate::sealed::storage(&state_dir, outbox.clone(), enrolment.clone());
        let before: Vec<usize> = vec![
            AgentSeal::new(&local_dir, local)
                .outbox()
                .await
                .unwrap()
                .len(),
            seal.outbox().await.unwrap().len(),
        ];
        for n in 0..5 {
            for id in [local, stopped] {
                let base = storage.load_raw(id).await.unwrap().base;
                assert!(base.is_some(), "on its head");
                let value = serde_json::json!({ "memory": format!("{id} {n}").repeat(40) });
                assert!(
                    storage
                        .save_raw(id, value, base.as_ref())
                        .await
                        .unwrap()
                        .is_some()
                );
            }
        }
        for (dir, id, was) in [
            (&local_dir, local, before[0]),
            (&stopped_dir, stopped, before[1]),
        ] {
            let s = AgentSeal::new(dir, id);
            assert_eq!(s.outbox().await.unwrap().len(), was + 5, "chained locally");
            assert!(s.seal_failed().await.unwrap().is_none());
            assert!(!crate::sealed::uploads(dir, &keys, id));
        }
        let drainer = Arc::new(
            Drainer::new(
                outbox,
                state_dir.clone(),
                // As the run makes them: none for an agent that doesn't upload
                HashMap::new(),
                vec![(local, "wren".into()), (stopped, "tarn".into())],
                [],
                Alerter::off(),
            )
            .with_anchors(Anchors {
                api: Arc::new(crate::sealed::api::guarded(client, enrolment)),
                keys: keys.clone(),
                state_dir: state_dir.clone(),
                verified: crate::sealed::anchors::VerifiedAnchors::new([(
                    "ANC-2026-0002".parse().unwrap(),
                    agora_agentkit::govlog::Sha256Hex::from([0; 32]),
                    chrono::Utc::now(),
                )]),
            }),
        );
        let task = drainer.spawn();
        drainer.drain_all().await;
        drainer.anchor_all().await;
        drainer.finish(task).await;
        assert_eq!(hits.load(Ordering::SeqCst), 0, "no request reached Agora");
        for (dir, id) in [(&local_dir, local), (&stopped_dir, stopped)] {
            assert_eq!(drainer.last(id), Some(DrainOutcome::LocalOnly));
            assert!(!drainer.held(id));
            assert!(
                !dir.join("sealed")
                    .join(super::super::receipts::RECEIPTS)
                    .exists(),
                "no receipts"
            );
            assert_eq!(
                AgentSeal::new(dir, id).outbox().await.unwrap().len(),
                if id == local { before[0] } else { before[1] } + 5,
                "nothing moved"
            );
        }
    }

    /// A 422 holds the agent for the run, alerted: no retry loop.
    #[tokio::test]
    async fn a_rejected_upload_is_held_not_retried() {
        let root = crate::sealed::TestDir::new("drain-422");
        let (id, _keys, _) = crate::sealed::check::tests::enrolled(&root).await;
        let (url, hits) = rejecting().await;
        let outbox = AgoraOutbox::new(
            agora_agentkit::client::Client::new(url).unwrap(),
            root.join("state"),
            Arc::new(SealedEnrolment::new(
                root.join("state"),
                root.join("secrets"),
            )),
        );
        let drainer = Drainer::new(
            outbox,
            root.join("state"),
            HashMap::new(),
            vec![(id, "tarn".into())],
            [],
            Alerter::off(),
        );
        drainer.drain_all().await;
        assert!(drainer.held(id), "held after the 422");
        let after = hits.load(Ordering::SeqCst);
        assert!(after > 0);
        drainer.drain_all().await;
        drainer.drain_all().await;
        assert_eq!(hits.load(Ordering::SeqCst), after, "not retried this run");
    }
}
