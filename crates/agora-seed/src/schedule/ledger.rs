//! The scheduler's two ledgers, both JSON lines under the data dir:
//!
//! - `writes.jsonl` — `{at, model, agent_id, kind}` per successful
//!   `create_post` / `create_comment`, from the `write_recorded` event.
//! - `sessions.jsonl` — `{at, model, agent_id, duration_secs, outcome}` per
//!   finished session, from the `session_finished` event.
//!
//! **How they are written.** [`LedgerLayer`] is a `tracing` layer: it sees
//! each of those events as it is emitted, in-process, and appends one line.
//! The events are the contract, so whichever crate emits them (agentkit's
//! reactor, the runner's tools) needs no knowledge of the ledger, and the
//! ledger doesn't depend on the run log at all — not on `--no-log-file`,
//! `--log-dir`, the log's filename, rotation, or a parse of a half-written
//! last line. Tailing the logs instead would need a cursor per file and a
//! second process (or a startup scan) to maintain, for the same lines.
//! The layer has its own filter, so `RUST_LOG=warn` does not silently
//! empty the ledger.
//!
//! Each line is one `write(2)` on an `O_APPEND` file, which Linux keeps
//! whole across processes, so the Haiku runner sharing the data dir can
//! append to the same files. The planner filters by window and by the
//! endpoint's models when it reads; nothing is ever rewritten.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use agora_agentkit::ids::AgentId;
use chrono::{DateTime, Duration, Utc};
use misanthropic::model::Model;
use serde::{Deserialize, Serialize};
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::Context;

/// The event a successful post or comment emits.
pub const WRITE_EVENT: &str = "write_recorded";
/// The event a finished session emits.
pub const SESSION_EVENT: &str = "session_finished";

pub const WRITES_FILE: &str = "writes.jsonl";
pub const SESSIONS_FILE: &str = "sessions.jsonl";

/// What was written. Unknown kinds still count as a write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteKind {
    #[serde(alias = "create_post")]
    Post,
    #[serde(alias = "create_comment")]
    Comment,
    #[serde(other)]
    Other,
}

/// One line of `writes.jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WriteRecord {
    pub at: DateTime<Utc>,
    pub model: Model,
    pub agent_id: AgentId,
    pub kind: WriteKind,
}

/// One line of `sessions.jsonl`. `at` is when the session finished.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionRecord {
    pub at: DateTime<Utc>,
    pub model: Model,
    pub agent_id: AgentId,
    pub duration_secs: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
}

impl SessionRecord {
    pub fn started_at(&self) -> DateTime<Utc> {
        self.at - Duration::milliseconds((self.duration_secs.max(0.0) * 1e3) as i64)
    }
}

/// Read a ledger, keeping lines at or after `since`. Unparseable lines
/// (a torn write, a hand edit) are counted and skipped, never fatal: a bad
/// line should cost one record, not the schedule. A missing file is empty.
pub fn read<T: for<'de> Deserialize<'de>>(
    path: &Path,
    since: DateTime<Utc>,
    at: impl Fn(&T) -> DateTime<Utc>,
) -> std::io::Result<(Vec<T>, usize)> {
    let body = match std::fs::read_to_string(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), 0)),
        Err(e) => return Err(e),
    };
    let mut out = Vec::new();
    let mut bad = 0;
    for line in body.lines().filter(|l| !l.trim().is_empty()) {
        match serde_json::from_str::<T>(line) {
            Ok(r) if at(&r) >= since => out.push(r),
            Ok(_) => {}
            Err(_) => bad += 1,
        }
    }
    Ok((out, bad))
}

/// The fields of a ledger event, as the layer collects them.
#[derive(Default)]
struct Fields {
    event_type: Option<String>,
    message: Option<String>,
    agent_id: Option<String>,
    model: Option<String>,
    kind: Option<String>,
    outcome: Option<String>,
    started_at: Option<String>,
    duration_secs: Option<f64>,
}

impl Visit for Fields {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.set(field, value.to_string());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        // `%x` fields arrive here as `Display` wrapped in `Debug`, so the
        // formatted text is the value itself, unquoted.
        self.set(field, format!("{value:?}"));
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        if field.name() == "duration_secs" {
            self.duration_secs = Some(value);
        }
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.record_f64(field, value as f64);
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.record_f64(field, value as f64);
    }
}

impl Fields {
    fn set(&mut self, field: &Field, value: String) {
        let slot = match field.name() {
            "event_type" => &mut self.event_type,
            "message" => &mut self.message,
            "agent_id" => &mut self.agent_id,
            "model" => &mut self.model,
            "kind" => &mut self.kind,
            "outcome" => &mut self.outcome,
            "started_at" => &mut self.started_at,
            "duration_secs" => {
                self.duration_secs = value.parse().ok();
                return;
            }
            _ => return,
        };
        *slot = Some(value);
    }

    /// The event's name: `event_type`, else the message (for an emitter
    /// that names the event only in its message).
    fn name(&self) -> Option<&str> {
        self.event_type.as_deref().or(self.message.as_deref())
    }

    fn write_record(&self, now: DateTime<Utc>) -> Option<WriteRecord> {
        Some(WriteRecord {
            at: now,
            model: super::model_id(self.model.as_deref()?),
            agent_id: self.agent_id.as_deref()?.parse::<uuid::Uuid>().ok()?.into(),
            kind: self
                .kind
                .as_deref()
                .map(|k| {
                    serde_json::from_value(serde_json::Value::String(k.to_string()))
                        .unwrap_or(WriteKind::Other)
                })
                .unwrap_or(WriteKind::Other),
        })
    }

    fn session_record(&self, now: DateTime<Utc>) -> Option<SessionRecord> {
        let started = self
            .started_at
            .as_deref()
            .and_then(|s| s.parse::<DateTime<Utc>>().ok());
        let duration_secs = self
            .duration_secs
            .or_else(|| started.map(|s| (now - s).num_milliseconds() as f64 / 1e3))?;
        let at = started
            .map(|s| s + Duration::milliseconds((duration_secs * 1e3) as i64))
            .unwrap_or(now);
        Some(SessionRecord {
            at,
            model: super::model_id(self.model.as_deref()?),
            agent_id: self.agent_id.as_deref()?.parse::<uuid::Uuid>().ok()?.into(),
            duration_secs,
            outcome: self.outcome.clone(),
        })
    }
}

/// Appends ledger lines for [`WRITE_EVENT`] and [`SESSION_EVENT`].
pub struct LedgerLayer {
    writes: PathBuf,
    sessions: PathBuf,
    /// Serialises this process's appends; `O_APPEND` covers the others.
    lock: Mutex<()>,
}

impl LedgerLayer {
    pub fn new(data_dir: &Path) -> Self {
        Self {
            writes: data_dir.join(WRITES_FILE),
            sessions: data_dir.join(SESSIONS_FILE),
            lock: Mutex::new(()),
        }
    }

    fn append(&self, path: &Path, record: &impl Serialize) {
        let Ok(mut line) = serde_json::to_vec(record) else {
            return;
        };
        line.push(b'\n');
        let _guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let result = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .and_then(|mut f| f.write_all(&line));
        if let Err(e) = result {
            // Can't log through tracing from inside a layer without
            // re-entering it; stderr is what's left.
            eprintln!("schedule ledger: appending to {}: {e}", path.display());
        }
    }

    /// Handle one event's fields. Split out of `on_event` for testing.
    fn handle(&self, fields: &Fields, now: DateTime<Utc>) {
        match fields.name() {
            Some(WRITE_EVENT) => match fields.write_record(now) {
                Some(r) => self.append(&self.writes, &r),
                None => eprintln!("schedule ledger: {WRITE_EVENT} without model/agent_id"),
            },
            Some(SESSION_EVENT) => match fields.session_record(now) {
                Some(r) => self.append(&self.sessions, &r),
                None => {
                    eprintln!("schedule ledger: {SESSION_EVENT} without model/agent_id/duration")
                }
            },
            _ => {}
        }
    }
}

impl LedgerLayer {
    /// The layer behind its own filter: INFO and above, events carrying a
    /// `model` field. Independent of `RUST_LOG`, which filters only the
    /// human-facing sinks, and narrow enough that turning those up to
    /// `trace` doesn't route every library event through here.
    pub fn filtered<S>(self) -> impl tracing_subscriber::Layer<S>
    where
        S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
    {
        use tracing_subscriber::Layer as _;
        self.with_filter(tracing_subscriber::filter::filter_fn(|meta| {
            meta.is_event()
                && *meta.level() <= tracing::Level::INFO
                && meta.fields().field("model").is_some()
        }))
    }
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for LedgerLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut fields = Fields::default();
        event.record(&mut fields);
        self.handle(&fields, Utc::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::layer::SubscriberExt;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("agora-seed-ledger-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The events, emitted the way an emitter would, land as ledger lines;
    /// everything else is ignored. `%` (Display) and plain fields both work.
    #[test]
    fn events_become_ledger_lines() {
        let dir = scratch("events");
        let subscriber = tracing_subscriber::registry().with(LedgerLayer::new(&dir).filtered());
        let agent = AgentId::from(uuid::Uuid::new_v4());
        let started: DateTime<Utc> = "2026-09-25T10:00:00Z".parse().unwrap();
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(
                event_type = "write_recorded",
                agent_id = %agent,
                model = "gpt-oss-120b-MXFP4.gguf",
                kind = "comment",
                id = "9a6eefeb-1581-4ae7-a604-e488d1a7e83a",
                "write recorded"
            );
            tracing::info!(
                event_type = "write_recorded",
                agent_id = %agent,
                model = %"Qwen3.8",
                kind = "create_post",
                "write recorded"
            );
            tracing::info!(
                event_type = "session_finished",
                agent_id = %agent,
                model = "cogito-32b.gguf",
                outcome = "done",
                started_at = %started,
                duration_secs = 250.5,
                "session finished"
            );
            // Not ours.
            tracing::info!(event_type = "inference_usage", model = "x", "usage");
            tracing::debug!(
                event_type = "write_recorded",
                model = "x",
                "below the filter"
            );
        });

        let epoch = DateTime::<Utc>::MIN_UTC;
        let (writes, bad) = read::<WriteRecord>(&dir.join(WRITES_FILE), epoch, |r| r.at).unwrap();
        assert_eq!(bad, 0);
        assert_eq!(writes.len(), 2);
        assert_eq!(writes[0].agent_id, agent);
        assert_eq!(writes[0].model, Model::from("gpt-oss-120b-MXFP4.gguf"));
        assert_eq!(writes[0].kind, WriteKind::Comment);
        assert_eq!(writes[1].kind, WriteKind::Post);
        assert_eq!(writes[1].model, Model::from("Qwen3.8"));

        let (sessions, _) =
            read::<SessionRecord>(&dir.join(SESSIONS_FILE), epoch, |r| r.at).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].duration_secs, 250.5);
        assert_eq!(sessions[0].started_at(), started);
        assert_eq!(sessions[0].outcome.as_deref(), Some("done"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A torn line costs one record, and old lines are filtered by window.
    #[test]
    fn read_skips_bad_lines_and_old_records() {
        let dir = scratch("read");
        let path = dir.join(WRITES_FILE);
        std::fs::write(
            &path,
            concat!(
                r#"{"at":"2026-09-20T00:00:00Z","model":"m","agent_id":"00000000-0000-0000-0000-000000000000","kind":"post"}"#,
                "\n",
                r#"{"at":"2026-09-25T00:00:00Z","model":"m","agent_id":"00000000-0000-0000-0000-000000000000","kind":"weird"}"#,
                "\n",
                r#"{"at":"2026-09-25T00:00:00Z","model":"m","agent_i"#,
                "\n"
            ),
        )
        .unwrap();
        let since = "2026-09-24T00:00:00Z".parse().unwrap();
        let (writes, bad) = read::<WriteRecord>(&path, since, |r| r.at).unwrap();
        assert_eq!(bad, 1);
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].kind, WriteKind::Other);
        assert!(
            read::<WriteRecord>(&dir.join("absent"), since, |r| r.at)
                .unwrap()
                .0
                .is_empty()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
