//! Consumer for blallama's `/probe` SSE stream.
//!
//! Connect-first / take-everything pattern (per slice-2A's broadcast
//! semantics): the consumer connects before the corresponding
//! `/v1/messages` request is sent, accumulates *all* SSE events
//! arriving on the channel, and post-completion filters the
//! accumulated events by the `Message.id` returned by the API. Single
//! in-flight request is the supported case; concurrent probes against
//! the same blallama would need a smarter correlation strategy.
//!
//! # Why connect-first
//!
//! Blallama's broadcast channel only delivers events to currently-
//! connected consumers. If we send `/v1/messages` first and then
//! connect, we'll miss the early `SessionStart` and the first few
//! `Token` events. Connect-first guarantees we see the whole session.
//!
//! # Why match-after
//!
//! `Message.id` is generated server-side at handler entry. The client
//! has no way to pre-declare it. So we collect the entire SSE stream
//! into a buffer keyed by id, and after `/v1/messages` returns the
//! response with its `id` field, we look up the matching session.

use std::collections::HashMap;

use anyhow::Context as _;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, warn};
use uuid::Uuid;

use super::snapshot::{ProbeEvent, ProbeSnapshot};

/// One completed session's worth of accumulated snapshots.
#[derive(Debug, Clone)]
pub struct CompletedSession {
    pub id: Uuid,
    pub model: Option<String>,
    pub snapshots: Vec<ProbeSnapshot>,
}

/// Background SSE consumer. `start` spawns a tokio task that holds the
/// connection open and accumulates events into a per-session buffer.
/// `take` consumes the buffer for a given `Message.id` after the
/// completion has come back.
pub struct ProbeStreamConsumer {
    /// Receiver for completed sessions. The producer task closes this
    /// when the SSE connection drops or when [`ProbeStreamConsumer::stop`]
    /// is called.
    rx: mpsc::UnboundedReceiver<CompletedSession>,
    /// Shutdown signal — drop this to stop the consumer task.
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    /// The spawned task handle, joined on stop.
    task: Option<JoinHandle<()>>,
    /// Cache of completed sessions awaiting `take` by id.
    cache: HashMap<Uuid, CompletedSession>,
}

impl ProbeStreamConsumer {
    /// Start a background consumer connected to the given probe-stream
    /// URL. Returns immediately; the connection is established on the
    /// background task.
    pub async fn start(probe_url: url::Url) -> anyhow::Result<Self> {
        let (tx, rx) = mpsc::unbounded_channel::<CompletedSession>();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        // SSE connections are long-lived; do NOT set a request timeout
        // (reqwest's default is none, which is what we want — a 0-second
        // timeout is the bug, fired immediately). The connection stays
        // open until the producer drops it or we shutdown.
        let client = reqwest::Client::builder()
            .build()
            .context("building reqwest client for probe stream")?;

        // Connect synchronously here so the caller knows the stream is
        // live (and any future events are guaranteed to arrive on the
        // broadcast channel) before they issue /v1/messages.
        let response = client
            .get(probe_url.clone())
            .header("Accept", "text/event-stream")
            .send()
            .await
            .with_context(|| format!("connecting to probe stream {probe_url}"))?;
        let status = response.status();
        anyhow::ensure!(
            status.is_success(),
            "probe stream {probe_url} returned HTTP {status}",
        );

        let task = tokio::spawn(consume_events(response, tx, shutdown_rx));

        Ok(Self {
            rx,
            shutdown_tx: Some(shutdown_tx),
            task: Some(task),
            cache: HashMap::new(),
        })
    }

    /// Block (cooperatively) until a session for `target_id` is
    /// available, or the consumer task exits / `timeout` elapses.
    /// Removes and returns the session.
    ///
    /// All other completed sessions seen while waiting are kept in the
    /// internal cache and remain available via subsequent `take` calls.
    pub async fn take(
        &mut self,
        target_id: Uuid,
        timeout: std::time::Duration,
    ) -> anyhow::Result<CompletedSession> {
        // Already cached?
        if let Some(s) = self.cache.remove(&target_id) {
            return Ok(s);
        }
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let remaining = deadline
                .checked_duration_since(tokio::time::Instant::now())
                .ok_or_else(|| {
                    anyhow::anyhow!("timed out waiting for probe session {target_id}")
                })?;
            match tokio::time::timeout(remaining, self.rx.recv()).await {
                Ok(Some(session)) => {
                    if session.id == target_id {
                        return Ok(session);
                    }
                    // Different session — cache and keep waiting.
                    self.cache.insert(session.id, session);
                }
                Ok(None) => {
                    anyhow::bail!(
                        "probe stream consumer exited before session \
                         {target_id} arrived"
                    );
                }
                Err(_elapsed) => {
                    anyhow::bail!("timed out waiting for probe session {target_id}");
                }
            }
        }
    }

    /// Shut down the background task. Idempotent.
    pub async fn stop(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.task.take() {
            let _ = handle.await;
        }
    }
}

async fn consume_events(
    response: reqwest::Response,
    tx: mpsc::UnboundedSender<CompletedSession>,
    mut shutdown: tokio::sync::oneshot::Receiver<()>,
) {
    // Per-session accumulators keyed by id. session_start opens an
    // entry; tokens push into it; session_end emits the completed
    // session and removes the entry.
    let mut in_flight: HashMap<Uuid, CompletedSession> = HashMap::new();

    let mut sse = response.bytes_stream().eventsource();
    loop {
        tokio::select! {
            _ = &mut shutdown => {
                debug!("probe stream consumer received shutdown");
                break;
            }
            evt = sse.next() => {
                match evt {
                    Some(Ok(event)) => {
                        match serde_json::from_str::<ProbeEvent>(&event.data) {
                            Ok(parsed) => {
                                handle_event(parsed, &mut in_flight, &tx);
                            }
                            Err(e) => {
                                warn!(
                                    error = %e,
                                    data = %event.data,
                                    "failed to parse probe event"
                                );
                            }
                        }
                    }
                    Some(Err(e)) => {
                        warn!(error = %e, "probe SSE stream error");
                        break;
                    }
                    None => {
                        debug!("probe SSE stream ended");
                        break;
                    }
                }
            }
        }
    }

    // Drain any in-flight sessions on shutdown so callers waiting on
    // `take` for an id we did see (but didn't see SessionEnd for)
    // don't hang forever. They'll receive a partial session.
    for (_, session) in in_flight.drain() {
        let _ = tx.send(session);
    }
}

fn handle_event(
    parsed: ProbeEvent,
    in_flight: &mut HashMap<Uuid, CompletedSession>,
    tx: &mpsc::UnboundedSender<CompletedSession>,
) {
    match parsed {
        ProbeEvent::SessionStart { id, model } => {
            in_flight.insert(
                id,
                CompletedSession {
                    id,
                    model: Some(model),
                    snapshots: Vec::new(),
                },
            );
        }
        ProbeEvent::Token { id, ctx } => {
            in_flight
                .entry(id)
                .or_insert_with(|| CompletedSession {
                    id,
                    model: None,
                    snapshots: Vec::new(),
                })
                .snapshots
                .push(ctx);
        }
        ProbeEvent::SessionEnd { id } => {
            if let Some(session) = in_flight.remove(&id) {
                let _ = tx.send(session);
            } else {
                warn!(
                    %id,
                    "received SessionEnd for unknown session — \
                     SessionStart was missed (connect-first violation?)"
                );
            }
        }
    }
}

/// Default probe-stream URL derived from a `/v1/messages` endpoint:
/// same scheme/host/port, path replaced with `/probe`. Returns an
/// error if the input URL doesn't have a base.
pub fn probe_url_from_endpoint(endpoint: &url::Url) -> anyhow::Result<url::Url> {
    let mut probe = endpoint.clone();
    probe
        .path_segments_mut()
        .map_err(|()| anyhow::anyhow!("endpoint {endpoint} has no base path"))?
        .clear()
        .push("probe");
    Ok(probe)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_url_default_replaces_path() {
        let endpoint = url::Url::parse("http://192.168.0.123:11436").unwrap();
        let probe = probe_url_from_endpoint(&endpoint).unwrap();
        assert_eq!(probe.as_str(), "http://192.168.0.123:11436/probe");
    }

    #[test]
    fn probe_url_replaces_existing_path() {
        let endpoint = url::Url::parse("http://192.168.0.123:11436/v1/messages").unwrap();
        let probe = probe_url_from_endpoint(&endpoint).unwrap();
        assert_eq!(probe.as_str(), "http://192.168.0.123:11436/probe");
    }
}
