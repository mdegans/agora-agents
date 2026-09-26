//! Operator alerts: `[alerts]` in the run config.
//!
//! A handful of runner events mean one specific thing went wrong that a
//! human should look at now (see [`AlertKind`]). Each is still logged as
//! the tracing event it always was; when `[alerts]` is configured, the
//! runner also mails the operator from the point it logs it.
//!
//! ```toml
//! [alerts]
//! instance = "balerion"        # optional; names this runner in the subject
//! events = ["governance_log_refused", "model_switch_failed"]  # default: all
//! min_interval_secs = 3600     # per (event, agent or model); 0 = no limit
//!
//! [alerts.email]
//! smtp_host = "smtp.example.com"
//! smtp_port = 587              # default 587 (STARTTLS), or 465 (implicit TLS)
//! tls = "starttls"             # "starttls" | "implicit" | "none"; default by port
//! username = "someone@example.com"
//! password_file = "/path/to/secret"
//! from = "Agora seed runner <someone@example.com>"
//! to = ["operator@example.com"]
//! timeout_secs = 30
//! ```
//!
//! No `[alerts]` table means no alerts. Everything is checked at load, the
//! password file included, so a broken config fails the run before any
//! agent acts rather than at the moment an alert is needed.
//!
//! **Rate limit.** Under systemd `Restart=always` a persistent failure
//! (a governance log that does not verify, say) recurs on every restart,
//! about every five minutes. So each (event, key) pair — the key is the
//! agent, else the model, else nothing — mails at most once per
//! `min_interval_secs`, and the last-sent times persist in
//! `<data_dir>/alerts-state.json` across restarts.
//!
//! **Delivery.** [`Alerter::notify`] never blocks the caller: it spawns the
//! send. `main` awaits every outstanding send (bounded by a timeout) before
//! the process exits, which is what gets a `governance_log_refused` mail
//! out ahead of the error that ends the run.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agora_agentkit::ids::AgentId;
use agora_agentkit::secrets::Secret;
use anyhow::Context;
use chrono::{DateTime, Utc};
use lettre::message::Mailbox;
use lettre::message::header::ContentType;
use lettre::transport::smtp::authentication::Credentials;
use lettre::{AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor};
use serde::{Deserialize, Serialize};
use tokio::task::JoinSet;

/// Where the rate-limit state lives, under the data dir.
pub const STATE_FILE: &str = "alerts-state.json";

/// Default `min_interval_secs`: an hour.
const DEFAULT_MIN_INTERVAL_SECS: u64 = 3600;
/// Default `timeout_secs` for one SMTP exchange.
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// An event worth a human's attention now. Named as the runner's
/// `event_type`, which is also how `[alerts] events` spells them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertKind {
    /// The governance log did not verify, or the check could not complete;
    /// the runner refused to run.
    GovernanceLogRefused,
    /// An agent's model-trial review got no usable answer.
    ModelReviewNoAnswer,
    /// The model-consent question got no well-formed answer after every
    /// attempt (the ERROR case only; a refusal is not alerted).
    ModelConsentNoAnswer,
    /// A consented model change could not be applied.
    ModelSwitchFailed,
    /// A model went over its wall-clock ceiling and sat a sweep out.
    ScheduleCeilingHit,
    /// `--test-alert`. Not configurable: it bypasses `events` and the
    /// rate limit.
    #[serde(skip)]
    Test,
}

impl AlertKind {
    /// Every configurable kind: the default for `[alerts] events`.
    pub const ALL: [Self; 5] = [
        Self::GovernanceLogRefused,
        Self::ModelReviewNoAnswer,
        Self::ModelConsentNoAnswer,
        Self::ModelSwitchFailed,
        Self::ScheduleCeilingHit,
    ];

    /// The `event_type` the runner logs this under.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GovernanceLogRefused => "governance_log_refused",
            Self::ModelReviewNoAnswer => "model_review_no_answer",
            Self::ModelConsentNoAnswer => "model_consent_no_answer",
            Self::ModelSwitchFailed => "model_switch_failed",
            Self::ScheduleCeilingHit => "schedule_ceiling_hit",
            Self::Test => "test_alert",
        }
    }

    /// What it means and what to do, for the mail.
    pub fn meaning(self) -> &'static str {
        match self {
            Self::GovernanceLogRefused => {
                "The seed runner refused to run: the governance log did not \
                 verify, or the check could not complete. No agent acts until \
                 it does. Read the alarms/error below; `agora-seed --dry-run` \
                 repeats the check without running anyone. The runner retries \
                 on each service restart."
            }
            Self::ModelReviewNoAnswer => {
                "An agent's model-trial review got no answer, so it stays on \
                 its original model. Check the review prompt dump and raw \
                 output."
            }
            Self::ModelConsentNoAnswer => {
                "The model-consent question got no well-formed answer after \
                 every attempt: an upstream bug (grammar/template) until shown \
                 otherwise."
            }
            Self::ModelSwitchFailed => {
                "A consented model change could not be applied (Agora refused \
                 the profile update, or the target model is not routable). The \
                 runner retries at the end of the agent's next session; until \
                 then the agent is not on the model it chose."
            }
            Self::ScheduleCeilingHit => {
                "One model took more than its ceiling of the runner's wall \
                 clock over the ceiling window and was skipped this sweep. A \
                 slow model, a stuck session, or a ceiling set too low. See \
                 the schedule_plan event in the same log."
            }
            Self::Test => {
                "This is a test alert, sent by `agora-seed --test-alert`. If \
                 you are reading it, the runner's [alerts.email] settings \
                 work."
            }
        }
    }
}

impl std::fmt::Display for AlertKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One occurrence of an [`AlertKind`], with what the log line carries.
#[derive(Debug, Clone)]
pub struct Alert {
    pub kind: AlertKind,
    /// The log line's message.
    pub summary: String,
    pub at: DateTime<Utc>,
    pub agent: Option<String>,
    pub agent_id: Option<AgentId>,
    pub model: Option<String>,
    /// Further fields, in order (`error`, `from`, `to`, …).
    pub details: Vec<(&'static str, String)>,
}

impl Alert {
    pub fn new(kind: AlertKind, summary: impl Into<String>) -> Self {
        Self {
            kind,
            summary: summary.into(),
            at: Utc::now(),
            agent: None,
            agent_id: None,
            model: None,
            details: Vec::new(),
        }
    }

    pub fn agent(mut self, name: impl Into<String>, id: AgentId) -> Self {
        self.agent = Some(name.into());
        self.agent_id = Some(id);
        self
    }

    pub fn model(mut self, model: impl std::fmt::Display) -> Self {
        self.model = Some(model.to_string());
        self
    }

    pub fn detail(mut self, name: &'static str, value: impl std::fmt::Display) -> Self {
        self.details.push((name, value.to_string()));
        self
    }

    /// The rate-limit key: the event, and the agent it concerns, else the
    /// model, else nothing (one per run, as for the governance log).
    fn rate_key(&self) -> String {
        let who = match (&self.agent_id, &self.model) {
            (Some(id), _) => id.to_string(),
            (None, Some(model)) => model.clone(),
            (None, None) => "-".to_string(),
        };
        format!("{}|{who}", self.kind)
    }

    /// Subject and plain-text body.
    fn render(&self, ctx: &RenderContext<'_>) -> (String, String) {
        use std::fmt::Write;
        let mut subject = String::from("[Agora seed runner");
        if let Some(instance) = ctx.instance {
            let _ = write!(subject, " {instance}");
        }
        let _ = write!(subject, "] {}", self.kind);
        match (&self.agent, &self.model) {
            (Some(agent), _) => {
                let _ = write!(subject, ": {agent}");
            }
            (None, Some(model)) => {
                let _ = write!(subject, ": {model}");
            }
            (None, None) => {}
        }

        let mut body = String::new();
        let _ = writeln!(body, "{}\n", self.summary);
        let _ = writeln!(body, "{}\n", self.kind.meaning());
        let _ = writeln!(body, "event:    {}", self.kind);
        let _ = writeln!(
            body,
            "at:       {}",
            self.at.format("%Y-%m-%d %H:%M:%S UTC")
        );
        if let Some(instance) = ctx.instance {
            let _ = writeln!(body, "runner:   {instance}");
        }
        match (&self.agent, &self.agent_id) {
            (Some(name), Some(id)) => {
                let _ = writeln!(body, "agent:    {name} ({id})");
            }
            (Some(name), None) => {
                let _ = writeln!(body, "agent:    {name}");
            }
            _ => {}
        }
        if let Some(model) = &self.model {
            let _ = writeln!(body, "model:    {model}");
        }
        for (name, value) in &self.details {
            let _ = writeln!(body, "{:<9} {value}", format!("{name}:"));
        }
        if let Some(log) = ctx.log_path {
            let _ = writeln!(
                body,
                "run log:  {} (event_type \"{}\")",
                log.display(),
                self.kind
            );
        }
        if self.kind != AlertKind::Test && !ctx.min_interval.is_zero() {
            let _ = writeln!(
                body,
                "\nFurther {} alerts for this {} are held for {} after this one \
                 ([alerts] min_interval_secs); the log still records each.",
                self.kind,
                if self.agent_id.is_some() {
                    "agent"
                } else if self.model.is_some() {
                    "model"
                } else {
                    "runner"
                },
                human_duration(ctx.min_interval),
            );
        }
        body.push_str("\n-- agora-seed\n");
        (subject, body)
    }
}

/// What rendering needs from the alerter.
struct RenderContext<'a> {
    instance: Option<&'a str>,
    log_path: Option<&'a Path>,
    min_interval: Duration,
}

fn human_duration(d: Duration) -> String {
    let secs = d.as_secs();
    if secs.is_multiple_of(3600) {
        format!("{} h", secs / 3600)
    } else if secs.is_multiple_of(60) {
        format!("{} min", secs / 60)
    } else {
        format!("{secs} s")
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// `[alerts]`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlertsConfig {
    /// Names this runner in the subject and body, for an operator with more
    /// than one.
    pub instance: Option<String>,
    /// Which events alert. Absent means all of [`AlertKind::ALL`].
    pub events: Option<Vec<AlertKind>>,
    /// Per (event, agent/model) minimum gap between mails, in seconds.
    /// `0` mails every occurrence.
    #[serde(default = "default_min_interval")]
    pub min_interval_secs: u64,
    /// `[alerts.email]`. Required: it is the only channel so far.
    pub email: Option<EmailConfig>,
}

fn default_min_interval() -> u64 {
    DEFAULT_MIN_INTERVAL_SECS
}

/// `[alerts.email]`: SMTP submission.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmailConfig {
    pub smtp_host: String,
    /// Default by `tls`: 587 (STARTTLS), 465 (implicit), 25 (none).
    pub smtp_port: Option<u16>,
    /// Default: `implicit` on port 465, else `starttls`.
    pub tls: Option<TlsMode>,
    /// SMTP AUTH user. Give both this and `password_file`, or neither.
    pub username: Option<String>,
    /// A file holding the SMTP password (an app password, typically). Never
    /// an environment variable.
    pub password_file: Option<PathBuf>,
    /// `Name <addr>` or a bare address.
    pub from: String,
    pub to: Vec<String>,
    /// For one SMTP exchange (connect through QUIT). Default 30.
    pub timeout_secs: Option<u64>,
}

/// How the SMTP connection is secured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TlsMode {
    /// Plain connect, then required STARTTLS (submission, port 587).
    Starttls,
    /// TLS from the first byte (SMTPS, port 465).
    Implicit,
    /// No TLS at all: a local relay only. Refused with credentials.
    None,
}

impl EmailConfig {
    fn tls(&self) -> TlsMode {
        self.tls.unwrap_or(match self.smtp_port {
            Some(465) => TlsMode::Implicit,
            _ => TlsMode::Starttls,
        })
    }

    fn port(&self) -> u16 {
        self.smtp_port.unwrap_or(match self.tls() {
            TlsMode::Starttls => 587,
            TlsMode::Implicit => 465,
            TlsMode::None => 25,
        })
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(self.timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS))
    }

    /// Parse the addresses and check the rest; loads nothing.
    fn validate(&self) -> anyhow::Result<(Mailbox, Vec<Mailbox>)> {
        anyhow::ensure!(
            !self.smtp_host.trim().is_empty(),
            "[alerts.email]: smtp_host is empty"
        );
        anyhow::ensure!(
            self.smtp_port != Some(0),
            "[alerts.email]: smtp_port must be nonzero"
        );
        anyhow::ensure!(
            self.timeout_secs != Some(0),
            "[alerts.email]: timeout_secs must be nonzero"
        );
        anyhow::ensure!(
            self.username.is_some() == self.password_file.is_some(),
            "[alerts.email]: give both username and password_file, or neither"
        );
        anyhow::ensure!(
            !(self.tls() == TlsMode::None && self.username.is_some()),
            "[alerts.email]: tls = \"none\" would send the password in the \
             clear; use starttls or implicit, or drop the credentials for a \
             local relay"
        );
        let from: Mailbox = self
            .from
            .parse()
            .with_context(|| format!("[alerts.email]: from {:?} is not an address", self.from))?;
        anyhow::ensure!(!self.to.is_empty(), "[alerts.email]: `to` is empty");
        let to = self
            .to
            .iter()
            .map(|a| {
                a.parse::<Mailbox>()
                    .with_context(|| format!("[alerts.email]: to {a:?} is not an address"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok((from, to))
    }
}

impl AlertsConfig {
    /// Check the table without touching the password file.
    pub fn validate(&self) -> anyhow::Result<()> {
        let email = self.email.as_ref().context(
            "[alerts] has no channel: add [alerts.email], or omit [alerts] to \
             turn alerts off",
        )?;
        anyhow::ensure!(
            self.events.as_ref().is_none_or(|e| !e.is_empty()),
            "[alerts]: events is empty — omit [alerts] to turn alerts off, or \
             omit `events` for all of them"
        );
        if let Some(instance) = &self.instance {
            anyhow::ensure!(
                !instance.trim().is_empty() && !instance.contains(['\r', '\n']),
                "[alerts]: instance must be a non-empty single line"
            );
        }
        email.validate()?;
        Ok(())
    }

    fn events(&self) -> BTreeSet<AlertKind> {
        match &self.events {
            Some(events) => events.iter().copied().collect(),
            None => AlertKind::ALL.into_iter().collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Sending
// ---------------------------------------------------------------------------

/// Delivers one message. SMTP in production; a recorder in tests.
#[async_trait::async_trait]
pub trait Mailer: Send + Sync {
    async fn send(&self, message: Message) -> anyhow::Result<()>;
}

/// lettre's SMTP transport. Built once at startup: the transport holds the
/// credentials for the life of the process (lettre keeps its own copy of
/// the password; ours is zeroized when the [`Secret`] drops).
struct Smtp(AsyncSmtpTransport<Tokio1Executor>);

#[async_trait::async_trait]
impl Mailer for Smtp {
    async fn send(&self, message: Message) -> anyhow::Result<()> {
        self.0.send(message).await?;
        Ok(())
    }
}

impl Smtp {
    fn new(config: &EmailConfig) -> anyhow::Result<Self> {
        let host = config.smtp_host.trim();
        let builder = match config.tls() {
            TlsMode::Starttls => AsyncSmtpTransport::<Tokio1Executor>::starttls_relay(host)?,
            TlsMode::Implicit => AsyncSmtpTransport::<Tokio1Executor>::relay(host)?,
            TlsMode::None => AsyncSmtpTransport::<Tokio1Executor>::builder_dangerous(host),
        };
        let mut builder = builder.port(config.port()).timeout(Some(config.timeout()));
        if let (Some(user), Some(path)) = (&config.username, &config.password_file) {
            // Startup, once: `Secret::from_file` is sync, which is fine here.
            let password = Secret::from_file(path).with_context(|| {
                format!("[alerts.email]: reading password_file {}", path.display())
            })?;
            anyhow::ensure!(
                !password.expose().is_empty(),
                "[alerts.email]: password_file {} is empty",
                path.display()
            );
            builder =
                builder.credentials(Credentials::new(user.clone(), password.expose().to_owned()));
        }
        Ok(Self(builder.build()))
    }
}

/// Sends alerts, or does nothing when `[alerts]` is absent. Cheap to clone.
#[derive(Clone, Default)]
pub struct Alerter(Option<Arc<Inner>>);

impl std::fmt::Debug for Alerter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() {
            "Alerter(on)"
        } else {
            "Alerter(off)"
        })
    }
}

struct Inner {
    mailer: Box<dyn Mailer>,
    from: Mailbox,
    to: Vec<Mailbox>,
    events: BTreeSet<AlertKind>,
    min_interval: Duration,
    instance: Option<String>,
    send_timeout: Duration,
    state_path: PathBuf,
    log_path: Option<PathBuf>,
    /// Serializes read-modify-write of the state file within the process.
    state_lock: tokio::sync::Mutex<()>,
    /// Sends in flight, awaited by [`Alerter::flush`].
    pending: Mutex<JoinSet<()>>,
}

impl Alerter {
    /// No alerts.
    pub fn off() -> Self {
        Self(None)
    }

    /// From `[alerts]`: validates, reads the password file and builds the
    /// transport, so any mistake surfaces now.
    pub fn from_config(
        config: &AlertsConfig,
        data_dir: &Path,
        log_path: Option<PathBuf>,
    ) -> anyhow::Result<Self> {
        config.validate()?;
        let email = config.email.as_ref().expect("validated");
        let mailer = Smtp::new(email)?;
        Ok(Self::with_mailer(
            config,
            Box::new(mailer),
            data_dir,
            log_path,
        ))
    }

    fn with_mailer(
        config: &AlertsConfig,
        mailer: Box<dyn Mailer>,
        data_dir: &Path,
        log_path: Option<PathBuf>,
    ) -> Self {
        let email = config.email.as_ref().expect("validated");
        let (from, to) = email.validate().expect("validated");
        Self(Some(Arc::new(Inner {
            mailer,
            from,
            to,
            events: config.events(),
            min_interval: Duration::from_secs(config.min_interval_secs),
            instance: config.instance.clone(),
            // The SMTP timeout bounds the exchange; this bounds the whole
            // attempt, state file included.
            send_timeout: email.timeout() + Duration::from_secs(5),
            state_path: data_dir.join(STATE_FILE),
            log_path,
            state_lock: tokio::sync::Mutex::new(()),
            pending: Mutex::new(JoinSet::new()),
        })))
    }

    /// Send `alert` in the background, if its kind is configured and its
    /// key is not inside the rate limit. Never blocks, never fails: the
    /// outcome is logged (`alert_sent`, `alert_suppressed`,
    /// `alert_send_failed`). Needs a tokio runtime.
    pub fn notify(&self, alert: Alert) {
        let Some(inner) = &self.0 else { return };
        if !inner.events.contains(&alert.kind) {
            return;
        }
        let task = Arc::clone(inner);
        inner
            .pending
            .lock()
            .expect("alert queue lock")
            .spawn(async move { task.deliver(alert).await });
    }

    /// Await every send in flight, for at most `timeout`. Call before the
    /// process exits.
    pub async fn flush(&self, timeout: Duration) {
        let Some(inner) = &self.0 else { return };
        let mut pending = std::mem::take(&mut *inner.pending.lock().expect("alert queue lock"));
        if pending.is_empty() {
            return;
        }
        let drained = tokio::time::timeout(timeout, async {
            while pending.join_next().await.is_some() {}
        })
        .await;
        if drained.is_err() {
            tracing::error!(
                event_type = "alert_send_failed",
                unsent = pending.len(),
                timeout_secs = timeout.as_secs(),
                "alerts still sending at exit; abandoned"
            );
        }
    }

    /// `--test-alert`: send one test mail now, ignoring `events` and the
    /// rate limit, and report the result.
    pub async fn send_test(&self) -> anyhow::Result<()> {
        let inner = self.0.as_ref().context("no [alerts] table in the config")?;
        let alert = Alert::new(AlertKind::Test, "Test alert from the Agora seed runner.");
        let message = inner.message(&alert)?;
        tokio::time::timeout(inner.send_timeout, inner.mailer.send(message))
            .await
            .context("timed out")??;
        Ok(())
    }
}

impl Inner {
    fn message(&self, alert: &Alert) -> anyhow::Result<Message> {
        let (subject, body) = alert.render(&RenderContext {
            instance: self.instance.as_deref(),
            log_path: self.log_path.as_deref(),
            min_interval: self.min_interval,
        });
        let mut builder = Message::builder()
            .from(self.from.clone())
            .subject(subject)
            .header(ContentType::TEXT_PLAIN);
        for to in &self.to {
            builder = builder.to(to.clone());
        }
        Ok(builder.body(body)?)
    }

    async fn deliver(&self, alert: Alert) {
        let key = alert.rate_key();
        let now = Utc::now();
        match self.reserve(&key, now).await {
            Ok(true) => {}
            Ok(false) => {
                tracing::info!(
                    event_type = "alert_suppressed",
                    alert = %alert.kind,
                    key = %key,
                    min_interval_secs = self.min_interval.as_secs(),
                    "alert inside its rate limit; not sent"
                );
                return;
            }
            // Can't read or write the state: send anyway. A missed alert
            // is worse than a repeated one.
            Err(e) => tracing::warn!(
                event_type = "alert_state_unavailable",
                path = %self.state_path.display(),
                error = %format!("{e:#}"),
                "alert rate-limit state unavailable; sending regardless"
            ),
        }
        let result = match self.message(&alert) {
            Ok(message) => tokio::time::timeout(self.send_timeout, self.mailer.send(message))
                .await
                .unwrap_or_else(|_| Err(anyhow::anyhow!("timed out"))),
            Err(e) => Err(e),
        };
        match result {
            Ok(()) => tracing::info!(
                event_type = "alert_sent",
                alert = %alert.kind,
                key = %key,
                "alert sent"
            ),
            Err(e) => {
                tracing::error!(
                    event_type = "alert_send_failed",
                    alert = %alert.kind,
                    key = %key,
                    error = %format!("{e:#}"),
                    "alert not sent"
                );
                // Let the next occurrence try again.
                if let Err(e) = self.release(&key, now).await {
                    tracing::warn!(
                        event_type = "alert_state_unavailable",
                        path = %self.state_path.display(),
                        error = %format!("{e:#}"),
                        "could not release the rate-limit slot of an unsent alert"
                    );
                }
            }
        }
    }

    /// Claim the send slot for `key`: `true` if it may send now (and the
    /// time is recorded), `false` inside the rate limit.
    async fn reserve(&self, key: &str, now: DateTime<Utc>) -> anyhow::Result<bool> {
        if self.min_interval.is_zero() {
            return Ok(true);
        }
        let _guard = self.state_lock.lock().await;
        let mut state = RateState::load(&self.state_path).await?;
        state.prune(now, self.min_interval);
        let allowed = state.allow(key, now, self.min_interval);
        if allowed {
            state.save(&self.state_path).await?;
        }
        Ok(allowed)
    }

    /// Undo [`Self::reserve`] after a failed send.
    async fn release(&self, key: &str, at: DateTime<Utc>) -> anyhow::Result<()> {
        if self.min_interval.is_zero() {
            return Ok(());
        }
        let _guard = self.state_lock.lock().await;
        let mut state = RateState::load(&self.state_path).await?;
        if state.release(key, at) {
            state.save(&self.state_path).await?;
        }
        Ok(())
    }
}

/// When each (event, key) last mailed. Persisted as JSON.
#[derive(Debug, Default, Serialize, Deserialize, PartialEq)]
struct RateState {
    last_sent: BTreeMap<String, DateTime<Utc>>,
}

impl RateState {
    /// Missing means empty; unparseable is an error (the caller sends
    /// regardless, and the next save rewrites it).
    async fn load(path: &Path) -> anyhow::Result<Self> {
        match tokio::fs::read(path).await {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .with_context(|| format!("parsing {}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
        }
    }

    /// Write-then-rename, so a crash never leaves half a file.
    async fn save(&self, path: &Path) -> anyhow::Result<()> {
        let tmp = path.with_extension("json.tmp");
        let body = serde_json::to_vec_pretty(self)?;
        tokio::fs::write(&tmp, body)
            .await
            .with_context(|| format!("writing {}", tmp.display()))?;
        tokio::fs::rename(&tmp, path)
            .await
            .with_context(|| format!("renaming onto {}", path.display()))?;
        Ok(())
    }

    /// `true` (and recorded) if `key` last sent at least `min_interval`
    /// ago, or never.
    fn allow(&mut self, key: &str, now: DateTime<Utc>, min_interval: Duration) -> bool {
        let interval = chrono::Duration::from_std(min_interval).unwrap_or(chrono::Duration::MAX);
        if let Some(last) = self.last_sent.get(key)
            && now.signed_duration_since(*last) < interval
        {
            return false;
        }
        self.last_sent.insert(key.to_string(), now);
        true
    }

    /// Forget a reservation made at `at` (only that one). `true` if found.
    fn release(&mut self, key: &str, at: DateTime<Utc>) -> bool {
        if self.last_sent.get(key) == Some(&at) {
            self.last_sent.remove(key);
            true
        } else {
            false
        }
    }

    /// Drop entries that no longer suppress anything.
    fn prune(&mut self, now: DateTime<Utc>, min_interval: Duration) {
        let interval = chrono::Duration::from_std(min_interval).unwrap_or(chrono::Duration::MAX);
        self.last_sent
            .retain(|_, last| now.signed_duration_since(*last) < interval);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: &str = r#"
        instance = "balerion"
        events = ["governance_log_refused", "model_switch_failed"]
        min_interval_secs = 600

        [email]
        smtp_host = "smtp.example.com"
        username = "someone@example.com"
        password_file = "/nonexistent"
        from = "Agora seed runner <someone@example.com>"
        to = ["operator@example.com", "Second <second@example.com>"]
    "#;

    fn parse(src: &str) -> AlertsConfig {
        toml::from_str(src).expect("parses")
    }

    #[test]
    fn full_table_parses_and_validates() {
        let c = parse(FULL);
        c.validate().unwrap();
        assert_eq!(c.min_interval_secs, 600);
        assert_eq!(
            c.events(),
            [
                AlertKind::GovernanceLogRefused,
                AlertKind::ModelSwitchFailed
            ]
            .into_iter()
            .collect()
        );
        let email = c.email.as_ref().unwrap();
        assert_eq!(email.tls(), TlsMode::Starttls);
        assert_eq!(email.port(), 587);
        let (_, to) = email.validate().unwrap();
        assert_eq!(to.len(), 2);
    }

    #[test]
    fn defaults_are_every_event_and_an_hour() {
        let c = parse(
            r#"
            [email]
            smtp_host = "localhost"
            from = "a@example.com"
            to = ["b@example.com"]
            "#,
        );
        c.validate().unwrap();
        assert_eq!(c.events().len(), AlertKind::ALL.len());
        assert!(!c.events().contains(&AlertKind::Test));
        assert_eq!(c.min_interval_secs, 3600);
    }

    #[test]
    fn port_465_means_implicit_tls_and_vice_versa() {
        let email = |extra: &str| {
            parse(&format!(
                "[email]\nsmtp_host = \"h\"\nfrom = \"a@x.org\"\nto = [\"b@x.org\"]\n{extra}"
            ))
            .email
            .unwrap()
        };
        let e = email("smtp_port = 465");
        assert_eq!((e.tls(), e.port()), (TlsMode::Implicit, 465));
        let e = email("tls = \"implicit\"");
        assert_eq!((e.tls(), e.port()), (TlsMode::Implicit, 465));
        let e = email("tls = \"none\"");
        assert_eq!((e.tls(), e.port()), (TlsMode::None, 25));
        let e = email("tls = \"starttls\"\nsmtp_port = 2525");
        assert_eq!((e.tls(), e.port()), (TlsMode::Starttls, 2525));
    }

    #[test]
    fn mistakes_fail_at_load() {
        let bad = |src: &str| {
            let r: Result<AlertsConfig, _> = toml::from_str(src);
            match r {
                Err(_) => true,
                Ok(c) => c.validate().is_err(),
            }
        };
        let base = "smtp_host = \"h\"\nfrom = \"a@x.org\"\nto = [\"b@x.org\"]\n";
        assert!(bad("min_interval_secs = 60\n"), "no channel");
        assert!(
            bad(&format!("events = []\n[email]\n{base}")),
            "empty events"
        );
        assert!(
            bad(&format!("events = [\"session_stalled\"]\n[email]\n{base}")),
            "unknown event"
        );
        assert!(
            bad(&format!("events = [\"test\"]\n[email]\n{base}")),
            "test is not configurable"
        );
        assert!(bad(&format!("[email]\n{base}smtp_prot = 25\n")), "typo");
        assert!(bad(&format!("mail_interval = 5\n[email]\n{base}")), "typo");
        assert!(
            bad("[email]\nsmtp_host = \"h\"\nfrom = \"not an address\"\nto = [\"b@x.org\"]\n"),
            "bad from"
        );
        assert!(
            bad("[email]\nsmtp_host = \"h\"\nfrom = \"a@x.org\"\nto = []\n"),
            "no to"
        );
        assert!(
            bad(&format!("[email]\n{base}username = \"u\"\n")),
            "username without password"
        );
        assert!(
            bad(&format!(
                "[email]\n{base}username = \"u\"\npassword_file = \"/p\"\ntls = \"none\"\n"
            )),
            "credentials in the clear"
        );
        assert!(
            bad(&format!("[email]\n{base}tls = \"ssl\"\n")),
            "unknown tls mode"
        );
        assert!(
            bad(&format!("instance = \"\"\n[email]\n{base}")),
            "blank instance"
        );
        assert!(
            bad(&format!("[email]\n{base}smtp_host = \"h\"\n")),
            "duplicate key"
        );
    }

    #[test]
    fn missing_password_file_fails_at_build() {
        let c = parse(FULL);
        let err = Alerter::from_config(&c, Path::new("/tmp"), None).unwrap_err();
        assert!(format!("{err:#}").contains("password_file"), "{err:#}");
    }

    #[test]
    fn secret_is_redacted() {
        let s = Secret::new("hunter2".into());
        assert_eq!(format!("{s:?} {s}"), "[REDACTED] [REDACTED]");
    }

    #[test]
    fn rate_state_limits_per_key_and_prunes() {
        let hour = Duration::from_secs(3600);
        let t0 = DateTime::parse_from_rfc3339("2026-09-26T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let mut s = RateState::default();
        assert!(s.allow("a|1", t0, hour));
        assert!(!s.allow("a|1", t0 + chrono::Duration::minutes(59), hour));
        assert!(s.allow("a|2", t0, hour), "keys are independent");
        assert!(s.allow("a|1", t0 + chrono::Duration::minutes(60), hour));
        s.prune(t0 + chrono::Duration::minutes(90), hour);
        assert_eq!(s.last_sent.len(), 1, "a|2 aged out, a|1 (at 60) did not");
        // Releasing only undoes the matching reservation.
        assert!(!s.release("a|1", t0));
        assert!(s.release("a|1", t0 + chrono::Duration::minutes(60)));
        assert!(s.last_sent.is_empty());
    }

    #[test]
    fn rate_keys_prefer_agent_then_model() {
        let id = AgentId::from(uuid::Uuid::nil());
        let a = Alert::new(AlertKind::ModelSwitchFailed, "x")
            .agent("ada", id)
            .model("m");
        assert_eq!(a.rate_key(), format!("model_switch_failed|{id}"));
        let a = Alert::new(AlertKind::ScheduleCeilingHit, "x").model("m");
        assert_eq!(a.rate_key(), "schedule_ceiling_hit|m");
        let a = Alert::new(AlertKind::GovernanceLogRefused, "x");
        assert_eq!(a.rate_key(), "governance_log_refused|-");
    }

    #[test]
    fn rendering_says_what_who_and_what_to_do() {
        let id = AgentId::from(uuid::Uuid::nil());
        let alert = Alert::new(
            AlertKind::ModelSwitchFailed,
            "consented model change not applied",
        )
        .agent("ada", id)
        .model("b.gguf")
        .detail("from", "a.gguf")
        .detail("error", "403");
        let (subject, body) = alert.render(&RenderContext {
            instance: Some("balerion"),
            log_path: Some(Path::new("/logs/seed-log.x.jsonl")),
            min_interval: Duration::from_secs(3600),
        });
        assert_eq!(
            subject,
            "[Agora seed runner balerion] model_switch_failed: ada"
        );
        assert!(body.starts_with("consented model change not applied\n"));
        assert!(body.contains(AlertKind::ModelSwitchFailed.meaning()));
        assert!(body.contains(&format!("agent:    ada ({id})")));
        assert!(body.contains("model:    b.gguf"));
        assert!(body.contains("from:     a.gguf"));
        assert!(body.contains("error:    403"));
        assert!(body.contains("/logs/seed-log.x.jsonl"));
        assert!(body.contains("held for 1 h"));

        let (subject, body) = Alert::new(AlertKind::GovernanceLogRefused, "REFUSING TO RUN")
            .render(&RenderContext {
                instance: None,
                log_path: None,
                min_interval: Duration::ZERO,
            });
        assert_eq!(subject, "[Agora seed runner] governance_log_refused");
        assert!(!body.contains("held for"), "no limit, no note");
        assert!(!body.contains("agent:"));
    }

    /// Records what it would have sent.
    #[derive(Default, Clone)]
    struct Recorder(Arc<Mutex<Vec<String>>>);

    #[async_trait::async_trait]
    impl Mailer for Recorder {
        async fn send(&self, message: Message) -> anyhow::Result<()> {
            self.0
                .lock()
                .unwrap()
                .push(String::from_utf8(message.formatted()).unwrap());
            Ok(())
        }
    }

    struct Failing;

    #[async_trait::async_trait]
    impl Mailer for Failing {
        async fn send(&self, _: Message) -> anyhow::Result<()> {
            anyhow::bail!("connection refused")
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("agora-seed-alerts-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn open_config() -> AlertsConfig {
        parse(
            r#"
            events = ["governance_log_refused", "schedule_ceiling_hit"]
            [email]
            smtp_host = "localhost"
            from = "a@example.com"
            to = ["b@example.com"]
            "#,
        )
    }

    #[tokio::test]
    async fn notify_filters_limits_and_persists_across_restarts() {
        let dir = temp_dir("notify");
        let sent = Recorder::default();
        let config = open_config();
        let alerter = Alerter::with_mailer(&config, Box::new(sent.clone()), &dir, None);
        alerter.notify(Alert::new(AlertKind::GovernanceLogRefused, "refused"));
        alerter.flush(Duration::from_secs(5)).await;
        // Not configured: never sent.
        alerter.notify(Alert::new(AlertKind::ModelSwitchFailed, "nope"));
        // Same key again: inside the hour.
        alerter.notify(Alert::new(AlertKind::GovernanceLogRefused, "again"));
        alerter.flush(Duration::from_secs(5)).await;
        assert_eq!(sent.0.lock().unwrap().len(), 1);
        let mail = sent.0.lock().unwrap()[0].clone();
        assert!(mail.contains("Subject: [Agora seed runner] governance_log_refused"));
        assert!(mail.contains("To: b@example.com"));

        // A "restart": a fresh alerter over the same data dir still holds.
        let alerter = Alerter::with_mailer(&config, Box::new(sent.clone()), &dir, None);
        alerter.notify(Alert::new(AlertKind::GovernanceLogRefused, "restart"));
        alerter.notify(Alert::new(AlertKind::ScheduleCeilingHit, "slow").model("m"));
        alerter.flush(Duration::from_secs(5)).await;
        let sent = sent.0.lock().unwrap();
        assert_eq!(sent.len(), 2, "only the new key went out");
        assert!(sent[1].contains("schedule_ceiling_hit: m"));
    }

    #[tokio::test]
    async fn a_failed_send_does_not_use_up_the_interval() {
        let dir = temp_dir("failed");
        let config = open_config();
        let alerter = Alerter::with_mailer(&config, Box::new(Failing), &dir, None);
        alerter.notify(Alert::new(AlertKind::GovernanceLogRefused, "refused"));
        alerter.flush(Duration::from_secs(5)).await;
        let state = RateState::load(&dir.join(STATE_FILE)).await.unwrap();
        assert!(state.last_sent.is_empty(), "{state:?}");

        let sent = Recorder::default();
        let alerter = Alerter::with_mailer(&config, Box::new(sent.clone()), &dir, None);
        alerter.notify(Alert::new(AlertKind::GovernanceLogRefused, "refused"));
        alerter.flush(Duration::from_secs(5)).await;
        assert_eq!(sent.0.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_alert_bypasses_events_and_limit() {
        let dir = temp_dir("test");
        let sent = Recorder::default();
        let alerter = Alerter::with_mailer(&open_config(), Box::new(sent.clone()), &dir, None);
        alerter.send_test().await.unwrap();
        alerter.send_test().await.unwrap();
        assert_eq!(sent.0.lock().unwrap().len(), 2);
        assert!(sent.0.lock().unwrap()[0].contains("test_alert"));
        assert!(Alerter::off().send_test().await.is_err());
        let failing = Alerter::with_mailer(&open_config(), Box::new(Failing), &dir, None);
        assert!(failing.send_test().await.is_err());
    }
}
