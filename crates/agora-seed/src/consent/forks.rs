//! "Forking reality" for the trial review: the same moment of a session,
//! completed by both models.
//!
//! For an agent returning from a trial, two pairs:
//!
//! 1. the newest post or comment it wrote on `to` during the trial, and what
//!    `from` writes from the exact same point in that session;
//! 2. the newest it wrote on `from` before the trial, and what `to` writes
//!    from that point.
//!
//! Each write is found in its session's prompt dump — the runner's
//! `prompt logged` events name the dump and the model it ran on, so the
//! model is ground truth rather than a guess from dates. The dump is cut
//! just before the assistant turn that made the write, and everything that
//! named the original model as the session's is changed ([`Adjustment`]:
//! the dashboard's `Model:` line, the trial countdown, `set_model`'s
//! description), recorded in the file and said in the review. Earlier
//! thinking is removed (the other model's private reasoning isn't its own)
//! and the forking model generates once. **No tool is executed**: what it
//! would have done is rendered as text. The system prompt and the other
//! tools are left byte-identical.
//!
//! [`prepare`] runs at sweep start (and as `--prepare-review-forks`) for
//! every agent whose review is coming, grouped by model to keep model
//! loads on a local endpoint down, and writes
//! `state/<agent_id>/review_forks.json`. It is bounded ([`ForksConfig`]:
//! a per-generation timeout and a budget for the step; what isn't reached
//! rolls over to the next start), never redoes a pair already prepared,
//! retries a transient failure at most [`MAX_FORK_ATTEMPTS`] times, and
//! leaves alone any side whose forking model this runner doesn't offer.
//! The review session reads the file back ([`load`]). A pair that can't be
//! built is kept as `skipped`, with the reason, and the review says so.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use agora_agentkit::ids::{AgentId, CommentId, ContentId, PostId};
use agora_agentkit::reactor::Inference;
use agora_agentkit::reactor::seed::{ModelName, replace_model_line};
use agora_agentkit::requests::{CreateCommentPayload, CreatePostPayload};
use chrono::{DateTime, Utc};
use misanthropic::model::{Model, ModelInfo};
use misanthropic::prompt::Prompt;
use misanthropic::prompt::message::{Block, Role};
use misanthropic::prompt::output::OutputConfig;
use misanthropic::response::{self, StopReason};
use misanthropic::tool::MethodDef;
use serde::{Deserialize, Serialize};

use super::ledger::{Ledger, OfferKey, Stage, TRIAL_LINE_PREFIX};
use super::switch;

/// The file's name inside the agent's state directory.
pub const FORKS_FILE: &str = "review_forks.json";

/// Bump on layout change; [`load`] ignores any other format.
pub const FORMAT: u32 = 1;

/// Characters of a body kept when rendered into the review.
pub const BODY_CHARS: usize = 3000;

/// Dumps examined per side, newest first, before giving up on a write.
const MAX_SESSIONS_PER_SIDE: usize = 40;

/// Which side of the trial a pair's original write is from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    /// Written on `to`, during the trial; forked on `from`.
    Trial,
    /// Written on `from`, before the trial; forked on `to`.
    Before,
}

/// A post or comment, as the agent's tool call made it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Written {
    Post {
        community: String,
        title: String,
        body: String,
    },
    Comment {
        reply_to: ContentId,
        body: String,
    },
}

/// The id a write created.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum WriteId {
    Post(PostId),
    Comment(CommentId),
}

/// One thing the fork did, rendered. Nothing here was executed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "act", rename_all = "snake_case")]
pub enum ForkAct {
    Write(Written),
    /// Another tool call: its name and input.
    Call {
        name: String,
        input: serde_json::Value,
    },
    /// Visible text (thinking is left out).
    Text {
        text: String,
    },
}

/// The fork's whole turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fork {
    pub acts: Vec<ForkAct>,
    /// Cut at the token limit before it finished.
    #[serde(default)]
    pub clipped: bool,
}

/// One pair, ready or not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForkPair {
    pub side: Side,
    /// The model the original was written on.
    pub written_on: Model,
    /// The model that completed the fork.
    pub forked_on: Model,
    #[serde(flatten)]
    pub outcome: PairOutcome,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum PairOutcome {
    Ready {
        written_at: DateTime<Utc>,
        id: WriteId,
        original: Written,
        fork: Fork,
        /// The dump the fork was cut from.
        prompt_sha256: String,
        /// What was changed so the prompt named the forking model, not the
        /// original (empty for dumps from before agentkit 0.45, which don't
        /// name the model at all).
        #[serde(default)]
        adjusted: Vec<Adjustment>,
    },
    Skipped {
        reason: String,
        /// Worth trying again at the next start (a transient generation
        /// error with attempts left), as opposed to a permanent gap (no
        /// dump, no write, a request the model refused outright).
        #[serde(default)]
        retry: bool,
        /// Failed generations so far (see [`MAX_FORK_ATTEMPTS`]).
        #[serde(default)]
        attempts: u32,
    },
}

/// `state/<agent_id>/review_forks.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReviewForks {
    pub format: u32,
    #[serde(flatten)]
    pub key: OfferKey,
    /// The trial these forks are for, by its first session.
    pub trial_started_at: DateTime<Utc>,
    pub prepared_at: DateTime<Utc>,
    pub pairs: Vec<ForkPair>,
}

impl ReviewForks {
    fn matches(&self, key: &OfferKey, started_at: DateTime<Utc>) -> bool {
        self.format == FORMAT && &self.key == key && self.trial_started_at == started_at
    }
}

fn forks_path(agent_dir: &Path) -> PathBuf {
    agent_dir.join(FORKS_FILE)
}

/// The prepared forks for the trial `key` that started at `started_at`, if
/// there are any. Anything unreadable or for another trial is `None`.
pub async fn load(
    agent_dir: &Path,
    key: &OfferKey,
    started_at: DateTime<Utc>,
) -> Option<ReviewForks> {
    let bytes = tokio::fs::read(forks_path(agent_dir)).await.ok()?;
    let forks: ReviewForks = serde_json::from_slice(&bytes)
        .map_err(|e| {
            tracing::warn!(
                path = %forks_path(agent_dir).display(),
                error = %e,
                "review forks unreadable; the review goes without them"
            )
        })
        .ok()?;
    forks.matches(key, started_at).then_some(forks)
}

async fn save(agent_dir: &Path, forks: &ReviewForks) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    let path = forks_path(agent_dir);
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(forks)?;
    {
        let mut file = tokio::fs::File::create(&tmp).await?;
        file.write_all(&bytes).await?;
        file.sync_all().await?;
    }
    tokio::fs::rename(&tmp, &path).await
}

// ---------------------------------------------------------------------------
// Finding writes in a dump

/// A successful write in a dump: the assistant turn that made it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteTurn {
    /// Index of the assistant message holding the call.
    pub message: usize,
    pub id: WriteId,
    pub written: Written,
}

/// The id a successful `create_post`/`create_comment` result names.
fn created_id(result: &misanthropic::tool::Result) -> Option<WriteId> {
    if result.is_error {
        return None;
    }
    let text: String = result
        .content
        .iter()
        .filter_map(|b| match b {
            Block::Text { text, .. } => Some(text.to_string()),
            _ => None,
        })
        .collect();
    let between = |start: &str| -> Option<uuid::Uuid> {
        let rest = &text[text.find(start)? + start.len()..];
        rest[..rest.find(']')?].trim().parse().ok()
    };
    if let Some(id) = between("[post_id:") {
        return Some(WriteId::Post(PostId::from(id)));
    }
    between("[comment_id:").map(|id| WriteId::Comment(CommentId::from(id)))
}

/// The write a `create_post`/`create_comment` call describes, typed.
fn written(call: &misanthropic::tool::Use) -> Option<Written> {
    match call.name.as_ref() {
        "create_post" => {
            let p: CreatePostPayload = serde_json::from_value(call.input.clone()).ok()?;
            Some(Written::Post {
                community: p.community,
                title: p.title,
                body: p.body,
            })
        }
        "create_comment" => {
            let c: CreateCommentPayload = serde_json::from_value(call.input.clone()).ok()?;
            Some(Written::Comment {
                reply_to: c.reply_to,
                body: c.body,
            })
        }
        _ => None,
    }
}

/// Every successful post or comment in `prompt`, in order: a
/// `create_post`/`create_comment` call whose result (in the next user
/// turn) names the id it created.
pub fn find_writes(prompt: &Prompt) -> Vec<WriteTurn> {
    let mut out = Vec::new();
    for (i, pair) in prompt.messages.windows(2).enumerate() {
        let (asst, next) = (&pair[0], &pair[1]);
        if asst.role != Role::Assistant || next.role != Role::User {
            continue;
        }
        for block in asst.content.iter() {
            let Block::ToolUse { call } = block else {
                continue;
            };
            let Some(w) = written(call) else { continue };
            let id = next.content.iter().find_map(|b| match b {
                Block::ToolResult { result } if result.tool_use_id == call.id => created_id(result),
                _ => None,
            });
            if let Some(id) = id {
                out.push(WriteTurn {
                    message: i,
                    id,
                    written: w,
                });
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Surgery

/// Why a dump couldn't be forked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SurgeryError {
    NotAssistantTurn,
    EmptyAfterStrip,
}

impl std::fmt::Display for SurgeryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NotAssistantTurn => "the write's turn is not an assistant turn after a user turn",
            Self::EmptyAfterStrip => "an earlier assistant turn holds nothing but thinking",
        })
    }
}

impl std::error::Error for SurgeryError {}

/// A place where the original session's prompt named the model it ran on,
/// changed for the fork so the forking model isn't told it is the other
/// one. Recorded per pair, and said in the review.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Adjustment {
    /// The dashboard's `Model:` line now names the forking model.
    ModelLine,
    /// The trial countdown ("Model trial: session n of 5 on …") was
    /// removed: it named the trial model as the session's.
    TrialCountdown,
    /// `set_model`'s description now names the forking model as current,
    /// and its note on why no change was possible (which described the
    /// original model's situation) was dropped.
    SetModelDescription,
}

/// Cut `prompt` just before the assistant turn at `turn` and hand it to
/// `fork`. Everything in it that named the original model as the one the
/// session runs on is changed ([`Adjustment`]s, returned): the `Model:`
/// line, the trial countdown, `set_model`'s description. Earlier thinking
/// is removed, `prompt.model` is `fork`, `max_tokens` is the act budget, and
/// `output_config` keeps only the session's effort (a dump ends in a phase
/// turn, whose format must not carry over). The system prompt and every
/// other tool definition are untouched.
pub fn fork_prompt(
    mut prompt: Prompt,
    turn: usize,
    fork: &ModelInfo,
    act_max_tokens: u32,
) -> Result<(Prompt, Vec<Adjustment>), SurgeryError> {
    if turn == 0
        || prompt.messages.get(turn).map(|m| m.role) != Some(Role::Assistant)
        || prompt.messages[turn - 1].role != Role::User
    {
        return Err(SurgeryError::NotAssistantTurn);
    }
    prompt.messages.truncate(turn);

    let mut adjusted = Vec::new();
    let name = ModelName::of(fork);
    if let Some(first) = prompt.messages.first_mut() {
        for block in first.content.iter_mut() {
            if let Block::Text { text, .. } = block
                && let Some(new) = replace_model_line(text, name)
            {
                *text = new.into();
                adjusted.push(Adjustment::ModelLine);
                break;
            }
        }
        let mut countdown = false;
        for block in first.content.iter_mut() {
            if let Block::Text { text, .. } = block
                && text
                    .lines()
                    .any(|l| l.trim_start().starts_with(TRIAL_LINE_PREFIX))
            {
                let kept: Vec<&str> = text
                    .lines()
                    .filter(|l| !l.trim_start().starts_with(TRIAL_LINE_PREFIX))
                    .collect();
                *text = kept.join("\n").trim_end().to_string().into();
                countdown = true;
            }
        }
        if countdown {
            first
                .content
                .retain(|b| !matches!(b, Block::Text { text, .. } if text.trim().is_empty()));
            adjusted.push(Adjustment::TrialCountdown);
        }
    }

    for def in prompt.tools.iter_mut().flatten() {
        if let MethodDef::Custom(custom) = def
            && custom.name == switch::TOOL_NAME
            && let Some(new) = switch::retarget(&custom.description, fork.id.name(), name.display)
        {
            custom.description = new.into();
            adjusted.push(Adjustment::SetModelDescription);
        }
    }

    for message in prompt.messages.iter_mut() {
        if message.role != Role::Assistant {
            continue;
        }
        message
            .content
            .retain(|b| !matches!(b, Block::Thought { .. } | Block::RedactedThought { .. }));
        if message.content.is_empty() {
            return Err(SurgeryError::EmptyAfterStrip);
        }
    }

    prompt.model = fork.id.clone();
    prompt.max_tokens = std::num::NonZeroU32::new(act_max_tokens.max(1)).expect("nonzero");
    prompt.output_config = prompt
        .output_config
        .take()
        .and_then(|c| c.effort)
        .map(OutputConfig::effort);
    Ok((prompt, adjusted))
}

/// What the fork did, for the review. Thinking is left out.
pub fn render_fork(response: &response::Message) -> Fork {
    let mut acts = Vec::new();
    for block in response.inner.content.iter() {
        match block {
            Block::Text { text, .. } if !text.trim().is_empty() => acts.push(ForkAct::Text {
                text: text.trim().to_string(),
            }),
            Block::ToolUse { call } => acts.push(match written(call) {
                Some(w) => ForkAct::Write(w),
                None => ForkAct::Call {
                    name: call.name.to_string(),
                    input: call.input.clone(),
                },
            }),
            _ => {}
        }
    }
    Fork {
        acts,
        clipped: matches!(response.stop_reason, Some(StopReason::MaxTokens)),
    }
}

// ---------------------------------------------------------------------------
// Finding the sessions

/// A session's dump, from a `prompt logged` event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Session {
    pub at: DateTime<Utc>,
    pub model: Model,
    pub path: PathBuf,
    pub prompt_sha256: String,
}

#[derive(Deserialize)]
struct LogLine {
    timestamp: DateTime<Utc>,
    fields: PromptLogged,
}

#[derive(Deserialize)]
struct PromptLogged {
    message: String,
    agent_id: Option<AgentId>,
    model: Option<Model>,
    prompt_sha256: Option<String>,
    path: Option<PathBuf>,
}

/// Every `prompt logged` event for `agents` in the run logs directly under
/// `log_dir`, per agent, newest first. `prompt_dir` places a dump whose
/// event carries no path.
pub fn scan_logs(
    log_dir: &Path,
    prompt_dir: Option<&Path>,
    agents: &HashSet<AgentId>,
) -> std::io::Result<HashMap<AgentId, Vec<Session>>> {
    use std::io::BufRead;
    let mut out: HashMap<AgentId, Vec<Session>> = HashMap::new();
    for entry in std::fs::read_dir(log_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let Ok(file) = std::fs::File::open(&path) else {
            continue;
        };
        for line in std::io::BufReader::new(file).lines() {
            let Ok(line) = line else { break };
            if !line.contains("\"prompt logged\"") {
                continue;
            }
            let Ok(ev) = serde_json::from_str::<LogLine>(&line) else {
                continue;
            };
            let f = ev.fields;
            let (Some(id), Some(model), Some(sha)) = (f.agent_id, f.model, f.prompt_sha256) else {
                continue;
            };
            if f.message != "prompt logged" || !agents.contains(&id) {
                continue;
            }
            let path = match (f.path, prompt_dir) {
                (Some(p), _) => p,
                (None, Some(dir)) if sha.len() > 2 => {
                    dir.join(&sha[..2]).join(format!("{sha}.json"))
                }
                _ => continue,
            };
            out.entry(id).or_default().push(Session {
                at: ev.timestamp,
                model,
                path,
                prompt_sha256: sha,
            });
        }
    }
    for sessions in out.values_mut() {
        sessions.sort_by_key(|s| std::cmp::Reverse(s.at));
        sessions.dedup_by(|a, b| a.prompt_sha256 == b.prompt_sha256);
    }
    Ok(out)
}

/// How long after the trial's end a trial session's dump may be logged:
/// `ended_at` is stamped at the close of the last trial session, and its
/// dump is written at teardown, moments later.
const TEARDOWN_SLACK_MINUTES: i64 = 30;

/// The sessions that can hold `side`'s write, newest first: on `to` during
/// the trial (`started_at` to `ended_at`, plus the teardown that logs the
/// last one), or on `from` before it.
pub fn candidates<'a>(
    sessions: &'a [Session],
    key: &OfferKey,
    started_at: DateTime<Utc>,
    ended_at: DateTime<Utc>,
    side: Side,
) -> impl Iterator<Item = &'a Session> {
    let (model, trial) = match side {
        Side::Trial => (key.to.clone(), true),
        Side::Before => (key.from.clone(), false),
    };
    let until = ended_at + chrono::Duration::minutes(TEARDOWN_SLACK_MINUTES);
    sessions.iter().filter(move |s| {
        s.model == model
            && if trial {
                s.at >= started_at && s.at < until
            } else {
                s.at < started_at
            }
    })
}

// ---------------------------------------------------------------------------
// The batch step

/// Generation attempts per pair before a transient failure is final.
pub const MAX_FORK_ATTEMPTS: u32 = 3;

/// `[review_forks]` in the run config: bounds on the sweep-start step, so
/// it can never hold the sweep up for long. Pairs it doesn't reach roll
/// over to the next start.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ForksConfig {
    /// Prepare forks at sweep start. `--no-review-forks` overrides.
    pub enabled: bool,
    /// Longest one generation may take before it counts as a (retryable)
    /// failure.
    pub generation_timeout_secs: u64,
    /// Longest the whole step may take: no generation starts after it.
    pub budget_secs: u64,
}

impl Default for ForksConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            // One act turn on the slowest local model (Qwen 3.8 dense at
            // 8192 tokens, with a cold ~40k prompt) is well inside this.
            generation_timeout_secs: 900,
            budget_secs: 1800,
        }
    }
}

impl ForksConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.generation_timeout_secs > 0 && self.budget_secs > 0,
            "[review_forks]: timeouts must be nonzero (set enabled = false to turn the step off)"
        );
        Ok(())
    }
}

/// One agent whose review is coming, and the sides still to do.
#[derive(Debug, Clone)]
pub struct Job {
    pub agent_id: AgentId,
    pub agent: String,
    pub key: OfferKey,
    pub started_at: DateTime<Utc>,
    pub ended_at: DateTime<Utc>,
    /// The file already prepared for this trial, kept as it is except for
    /// the sides redone.
    pub existing: Option<ReviewForks>,
    pub sides: Vec<Side>,
}

impl Job {
    /// The model that forks `side`.
    fn forked_on(&self, side: Side) -> &Model {
        match side {
            Side::Trial => &self.key.from,
            Side::Before => &self.key.to,
        }
    }
}

/// Whether `side` still needs doing, given the file prepared so far: it
/// is missing, or failed in a way worth another try.
fn needs(existing: Option<&ReviewForks>, side: Side) -> bool {
    match existing.and_then(|f| f.pairs.iter().find(|p| p.side == side)) {
        None => true,
        Some(p) => matches!(p.outcome, PairOutcome::Skipped { retry: true, .. }),
    }
}

/// Agents under `state_dir` in [`Stage::ReturningForReview`] or
/// [`Stage::ReviewDue`] with a side still to prepare.
pub async fn jobs(state_dir: &Path) -> std::io::Result<Vec<Job>> {
    let mut out = Vec::new();
    let mut dirs = tokio::fs::read_dir(state_dir).await?;
    while let Some(entry) = dirs.next_entry().await? {
        let Some(agent_id) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<uuid::Uuid>().ok())
            .map(AgentId::from)
        else {
            continue;
        };
        let dir = entry.path();
        let Ok(ledger) = Ledger::load(&dir).await else {
            continue;
        };
        for record in &ledger.offers {
            let (started_at, ended_at) = match record.stage {
                Stage::ReturningForReview {
                    started_at,
                    ended_at,
                    ..
                }
                | Stage::ReviewDue {
                    started_at,
                    ended_at,
                    ..
                } => (started_at, ended_at),
                _ => continue,
            };
            let existing = tokio::fs::read(forks_path(&dir))
                .await
                .ok()
                .and_then(|b| serde_json::from_slice::<ReviewForks>(&b).ok())
                .filter(|f| f.matches(&record.key, started_at));
            let sides: Vec<Side> = [Side::Trial, Side::Before]
                .into_iter()
                .filter(|&side| needs(existing.as_ref(), side))
                .collect();
            if sides.is_empty() {
                continue;
            }
            out.push(Job {
                agent_id,
                agent: ledger
                    .agent
                    .as_ref()
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| agent_id.to_string()),
                key: record.key.clone(),
                started_at,
                ended_at,
                existing,
                sides,
            });
        }
    }
    out.sort_by(|a, b| a.agent.cmp(&b.agent));
    Ok(out)
}

/// A pair whose fork prompt is built and waits for its model.
struct Pending {
    job: usize,
    side: Side,
    written_on: Model,
    forked_on: Model,
    written_at: DateTime<Utc>,
    turn: WriteTurn,
    prompt: Prompt,
    prompt_sha256: String,
    adjusted: Vec<Adjustment>,
    /// Failed attempts so far.
    attempts: u32,
}

/// Where each model runs: one inference client per endpoint and the models
/// it advertises.
pub struct Endpoints<'a, I> {
    pub clients: &'a [I],
    /// `(index into clients, model)`.
    pub offered: &'a [(usize, ModelInfo)],
}

impl<I> Endpoints<'_, I> {
    fn find(&self, model: &Model) -> Option<(&I, &ModelInfo)> {
        self.offered
            .iter()
            .find(|(_, m)| &m.id == model)
            .and_then(|(i, m)| Some((self.clients.get(*i)?, m)))
    }
}

fn skipped(side: Side, written_on: &Model, forked_on: &Model, reason: String) -> ForkPair {
    ForkPair {
        side,
        written_on: written_on.clone(),
        forked_on: forked_on.clone(),
        outcome: PairOutcome::Skipped {
            reason,
            retry: false,
            attempts: 0,
        },
    }
}

/// Build one side's pair up to the generation, or say why it can't be
/// (a permanent gap: no dump, no write, a dump that can't be cut).
async fn build(
    job_index: usize,
    job: &Job,
    side: Side,
    sessions: &[Session],
    fork_info: &ModelInfo,
    act_max_tokens: u32,
) -> Result<Pending, Box<ForkPair>> {
    let forked_on = job.forked_on(side).clone();
    let written_on = match side {
        Side::Trial => job.key.to.clone(),
        Side::Before => job.key.from.clone(),
    };
    let attempts = job
        .existing
        .as_ref()
        .and_then(|f| f.pairs.iter().find(|p| p.side == side))
        .map_or(0, |p| match p.outcome {
            PairOutcome::Skipped { attempts, .. } => attempts,
            PairOutcome::Ready { .. } => 0,
        });
    let mut examined = 0;
    for session in candidates(sessions, &job.key, job.started_at, job.ended_at, side)
        .take(MAX_SESSIONS_PER_SIDE)
    {
        examined += 1;
        let bytes = match tokio::fs::read(&session.path).await {
            Ok(bytes) => bytes,
            Err(e) => {
                tracing::debug!(path = %session.path.display(), error = %e, "dump unreadable");
                continue;
            }
        };
        let prompt: Prompt = match serde_json::from_slice(&bytes) {
            Ok(p) => p,
            Err(e) => {
                tracing::debug!(path = %session.path.display(), error = %e, "dump unparseable");
                continue;
            }
        };
        let Some(turn) = find_writes(&prompt).pop() else {
            continue;
        };
        return match fork_prompt(prompt, turn.message, fork_info, act_max_tokens) {
            Ok((prompt, adjusted)) => Ok(Pending {
                job: job_index,
                side,
                written_on,
                forked_on,
                written_at: session.at,
                turn,
                prompt,
                prompt_sha256: session.prompt_sha256.clone(),
                adjusted,
                attempts,
            }),
            Err(e) => Err(Box::new(skipped(
                side,
                &written_on,
                &forked_on,
                format!("the session could not be forked: {e}"),
            ))),
        };
    }
    Err(Box::new(skipped(
        side,
        &written_on,
        &forked_on,
        if examined == 0 {
            "no record of a session on that model was found".to_string()
        } else {
            format!(
                "no post or comment was found in the last {examined} recorded sessions on that model"
            )
        },
    )))
}

/// A failed generation: final if the error says retrying won't help (a
/// 4xx such as a context overflow) or attempts are spent.
fn failed(p: &Pending, reason: String, transient: bool) -> PairOutcome {
    let attempts = p.attempts + 1;
    let retry = transient && attempts < MAX_FORK_ATTEMPTS;
    PairOutcome::Skipped {
        reason,
        retry,
        attempts,
    }
}

/// The batch step. See the [module docs](self). Returns the number of
/// agents whose forks file was written.
///
/// Bounded: each generation by `config.generation_timeout_secs`, the whole
/// step by `config.budget_secs`; whatever isn't reached is left for the
/// next start. Only sides whose forking model this runner's endpoints
/// offer are touched — another runner's file is left as it is — and a
/// side already prepared is never redone.
pub async fn prepare<I: Inference>(
    state_dir: &Path,
    log_dir: &Path,
    prompt_dir: Option<&Path>,
    endpoints: Endpoints<'_, I>,
    act_max_tokens: u32,
    config: ForksConfig,
) -> std::io::Result<usize> {
    let started = std::time::Instant::now();
    let budget = std::time::Duration::from_secs(config.budget_secs);
    let timeout = std::time::Duration::from_secs(config.generation_timeout_secs);
    let mut jobs = jobs(state_dir).await?;
    // Only what this runner can generate.
    for job in &mut jobs {
        let key = job.key.clone();
        job.sides.retain(|&side| {
            let model = match side {
                Side::Trial => &key.from,
                Side::Before => &key.to,
            };
            endpoints.find(model).is_some()
        });
    }
    jobs.retain(|j| !j.sides.is_empty());
    if jobs.is_empty() {
        return Ok(0);
    }
    let ids: HashSet<AgentId> = jobs.iter().map(|j| j.agent_id).collect();
    let sessions = scan_logs(log_dir, prompt_dir, &ids)?;
    tracing::info!(
        event_type = "review_forks_start",
        agents = jobs.len(),
        "preparing trial-review forks"
    );

    // Per job, the sides decided this run.
    let mut done: Vec<Vec<ForkPair>> = vec![Vec::new(); jobs.len()];
    // Grouped by the forking model, so a local endpoint loads each once.
    let mut pending: BTreeMap<String, Vec<Pending>> = BTreeMap::new();
    for (i, job) in jobs.iter().enumerate() {
        let empty = Vec::new();
        let agent_sessions = sessions.get(&job.agent_id).unwrap_or(&empty);
        for &side in &job.sides {
            let Some((_, fork_info)) = endpoints.find(job.forked_on(side)) else {
                continue;
            };
            match build(i, job, side, agent_sessions, fork_info, act_max_tokens).await {
                Ok(p) => pending
                    .entry(p.forked_on.name().to_string())
                    .or_default()
                    .push(p),
                Err(pair) => done[i].push(*pair),
            }
        }
    }

    let mut left = 0usize;
    for (model, group) in pending {
        tracing::info!(model = %model, pairs = group.len(), "generating review forks");
        for p in group {
            if started.elapsed() >= budget {
                left += 1;
                continue;
            }
            let job = &jobs[p.job];
            let Some((client, _)) = endpoints.find(&p.forked_on) else {
                continue;
            };
            let outcome = match tokio::time::timeout(timeout, client.infer(&p.prompt)).await {
                Ok(Ok(response)) => PairOutcome::Ready {
                    written_at: p.written_at,
                    id: p.turn.id,
                    original: p.turn.written.clone(),
                    fork: render_fork(&response),
                    prompt_sha256: p.prompt_sha256.clone(),
                    adjusted: p.adjusted.clone(),
                },
                Ok(Err(e)) => {
                    use agora_agentkit::reactor::RetryAfter;
                    let transient = !e.is_fatal();
                    tracing::warn!(
                        event_type = "review_fork_failed",
                        agent = %job.agent,
                        agent_id = %job.agent_id,
                        model = %p.forked_on,
                        transient,
                        error = %e,
                        "review fork generation failed"
                    );
                    failed(
                        &p,
                        format!("the other model could not be run ({e})"),
                        transient,
                    )
                }
                Err(_) => {
                    tracing::warn!(
                        event_type = "review_fork_failed",
                        agent = %job.agent,
                        agent_id = %job.agent_id,
                        model = %p.forked_on,
                        transient = true,
                        timeout_secs = timeout.as_secs(),
                        "review fork generation timed out"
                    );
                    failed(
                        &p,
                        format!(
                            "the other model did not finish within {}s",
                            timeout.as_secs()
                        ),
                        true,
                    )
                }
            };
            done[p.job].push(ForkPair {
                side: p.side,
                written_on: p.written_on,
                forked_on: p.forked_on,
                outcome,
            });
        }
    }
    if left > 0 {
        tracing::info!(
            event_type = "review_forks_deferred",
            pairs = left,
            budget_secs = budget.as_secs(),
            "review forks budget spent; the rest roll over to the next start"
        );
    }

    let mut written = 0;
    for (job, decided) in jobs.iter().zip(done) {
        if decided.is_empty() {
            continue; // nothing reached: leave any file as it is
        }
        for p in &decided {
            if let PairOutcome::Skipped { reason, retry, .. } = &p.outcome {
                tracing::warn!(
                    event_type = "review_fork_skipped",
                    agent = %job.agent,
                    agent_id = %job.agent_id,
                    side = ?p.side,
                    retry,
                    reason = %reason,
                    "review fork pair skipped"
                );
            }
        }
        // Keep what was already prepared; replace only the sides decided.
        let mut pairs: Vec<ForkPair> = job
            .existing
            .as_ref()
            .map(|f| f.pairs.clone())
            .unwrap_or_default();
        pairs.retain(|p| !decided.iter().any(|d| d.side == p.side));
        pairs.extend(decided);
        pairs.sort_by_key(|p| p.side != Side::Trial);
        let forks = ReviewForks {
            format: FORMAT,
            key: job.key.clone(),
            trial_started_at: job.started_at,
            prepared_at: Utc::now(),
            pairs,
        };
        let dir = state_dir.join(job.agent_id.to_string());
        match save(&dir, &forks).await {
            Ok(()) => {
                written += 1;
                tracing::info!(
                    event_type = "review_forks_prepared",
                    agent = %job.agent,
                    agent_id = %job.agent_id,
                    ready = forks
                        .pairs
                        .iter()
                        .filter(|p| matches!(p.outcome, PairOutcome::Ready { .. }))
                        .count(),
                    "review forks prepared"
                );
            }
            Err(e) => tracing::error!(
                agent_id = %job.agent_id,
                error = %e,
                "review forks not saved"
            ),
        }
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use misanthropic::model::Kind;

    const OLD: &str = "Qwen3.6.gguf";
    const NEW: &str = "Qwen3.8.gguf";

    pub(crate) fn info(id: &str, display: &str) -> ModelInfo {
        ModelInfo {
            id: Model::from(id.to_string()),
            display_name: display.to_string().into(),
            capabilities: Default::default(),
            max_input_tokens: 0,
            max_tokens: 0,
            kind: Kind::Model,
            created_at: DateTime::from_timestamp(0, 0).unwrap(),
        }
    }

    const POST: &str = "11111111-1111-1111-1111-111111111111";
    const COMMENT: &str = "22222222-2222-2222-2222-222222222222";
    const TARGET: &str = "33333333-3333-3333-3333-333333333333";

    /// A session dump shaped like the runner's: intro with a model line,
    /// a read, a failed write, a comment and a post, then the memory turn.
    /// `set_model`'s description as the runner writes it, for an agent on
    /// `current`.
    fn set_model_description(current: &str) -> String {
        let choices = [
            switch::Choice {
                id: OLD,
                name: "Qwen 3.6",
                description: "Sparse and quick.",
            },
            switch::Choice {
                id: NEW,
                name: "Qwen 3.8",
                description: "Dense and slow.",
            },
        ];
        let name = if current == NEW {
            "Qwen 3.8"
        } else {
            "Qwen 3.6"
        };
        let blocked = (current == NEW).then_some("you are in a trial of Qwen 3.8");
        switch::describe(&choices, name, current, blocked)
    }

    /// A session dump shaped like the runner's: intro with a model line
    /// (and, on the trial model, the countdown after it), `set_model` among
    /// the tools, a read, a failed write, a comment and a post, then the
    /// memory turn.
    pub(crate) fn dump(model: &str) -> Prompt {
        let mut intro = vec![serde_json::json!({"type": "text",
            "text": format!("## Your Personality\n\nName: tarn\nModel: Qwen {model} ({model})\nFeed…"),
            "cache_control": {"type": "ephemeral", "ttl": "1h"}})];
        if model == NEW {
            intro.push(serde_json::json!({"type": "text", "text":
                "Model trial: session 3 of 5 on Qwen 3.8. After session 5 you'll return to Qwen 3.6 for one session to decide whether to keep Qwen 3.8."}));
        }
        serde_json::from_value(serde_json::json!({
            "model": model,
            "max_tokens": 8192,
            "system": [{"type": "text", "text": "SYSTEM"}],
            "tools": [
                {"name": "create_post", "description": "d", "input_schema": {"type": "object"}},
                {"name": "set_model", "description": set_model_description(model), "input_schema": {"type": "object"}},
            ],
            "tool_choice": {"type": "auto"},
            "thinking": {"type": "adaptive"},
            "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}, "effort": "medium"},
            "messages": [
                {"role": "user", "content": intro},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "I should read.", "signature": ""},
                    {"type": "text", "text": "Reading."},
                    {"type": "tool_use", "id": "a", "name": "get_content", "input": {"id": TARGET}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": [{"type": "text", "text": "the post"}]},
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Reply twice.", "signature": ""},
                    {"type": "tool_use", "id": "b", "name": "create_comment", "input": {"reply_to": TARGET, "body": "Refused."}},
                    {"type": "tool_use", "id": "c", "name": "create_comment", "input": {"reply_to": TARGET, "body": "Water remembers."}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "b", "is_error": true, "content": [{"type": "text", "text": "Error: HTTP 409"}]},
                    {"type": "tool_result", "tool_use_id": "c", "content": [{"type": "text", "text": format!("Comment created [comment_id: {COMMENT}]")}]},
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Now a post.", "signature": ""},
                    {"type": "tool_use", "id": "d", "name": "create_post", "input": {"community": "philosophy", "title": "On rivers", "body": "Rivers."}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "d", "content": [{"type": "text", "text": format!("Post created [post_id: {POST}]")}]},
                    {"type": "text", "text": "It's time to update your memory."},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "{\"content\": \"mem\"}"},
                ]},
            ],
        }))
        .unwrap()
    }

    #[test]
    fn finds_only_successful_writes_in_order() {
        let writes = find_writes(&dump(NEW));
        assert_eq!(writes.len(), 2, "{writes:?}");
        assert_eq!(writes[0].message, 3);
        assert_eq!(
            writes[0].id,
            WriteId::Comment(CommentId::from(COMMENT.parse::<uuid::Uuid>().unwrap()))
        );
        assert!(
            matches!(&writes[0].written, Written::Comment { body, .. } if body == "Water remembers.")
        );
        assert_eq!(writes[1].message, 5);
        assert_eq!(
            writes[1].written,
            Written::Post {
                community: "philosophy".into(),
                title: "On rivers".into(),
                body: "Rivers.".into()
            }
        );
    }

    #[test]
    fn surgery_cuts_rewrites_strips_and_swaps() {
        let original = dump(NEW);
        let turn = find_writes(&original).pop().unwrap().message;
        let (fork, adjusted) =
            fork_prompt(original.clone(), turn, &info(OLD, "Qwen 3.6 35B"), 8192).unwrap();
        assert_eq!(
            adjusted,
            [
                Adjustment::ModelLine,
                Adjustment::TrialCountdown,
                Adjustment::SetModelDescription
            ]
        );
        assert_eq!(fork.messages.len(), turn, "cut before the write's turn");
        assert_eq!(fork.messages.last().unwrap().role, Role::User);
        assert_eq!(fork.model, Model::from(OLD));
        let intro = crate::consent::prompt::tests::text(&fork.messages[0].content);
        assert!(
            intro.contains("\nModel: Qwen 3.6 35B (Qwen3.6.gguf)\n"),
            "{intro}"
        );
        assert_eq!(fork.messages[0].content.len(), 1, "countdown block removed");
        assert!(fork.messages.iter().all(|m| {
            m.content
                .iter()
                .all(|b| !matches!(b, Block::Thought { .. } | Block::RedactedThought { .. }))
        }));
        // The rest untouched: system, the other tools, tool choice, thinking
        // mode, and every non-thinking block after the intro.
        let json = |p: &Prompt, k: &str| serde_json::to_value(p).unwrap()[k].clone();
        for k in ["system", "tool_choice", "thinking"] {
            assert_eq!(json(&fork, k), json(&original, k), "{k}");
        }
        assert_eq!(json(&fork, "tools")[0], json(&original, "tools")[0]);
        assert_eq!(
            serde_json::to_value(&fork.messages[2]).unwrap(),
            serde_json::to_value(&original.messages[2]).unwrap()
        );
        // The dump's last phase format does not carry over; the effort does.
        let config = serde_json::to_value(&fork.output_config).unwrap();
        assert_eq!(config, serde_json::json!({"effort": "medium"}));
    }

    /// Nothing in a fork's prompt tells the forking model it runs on the
    /// original: not the model line, not the trial countdown, not
    /// `set_model`'s "You run on …" or its `(current)` mark.
    #[test]
    fn no_text_names_the_original_model_as_current() {
        for (written, forked, forked_name) in [(NEW, OLD, "Qwen 3.6"), (OLD, NEW, "Qwen 3.8")] {
            let original = dump(written);
            let turn = find_writes(&original).pop().unwrap().message;
            let (fork, _) = fork_prompt(original, turn, &info(forked, forked_name), 8192).unwrap();
            let all = serde_json::to_string(&fork).unwrap();
            let written_name = if written == NEW {
                "Qwen 3.8"
            } else {
                "Qwen 3.6"
            };
            for stale in [
                format!("({written})"),
                format!("You run on {written_name}"),
                format!("(`{written}`)"),
                format!("{written_name} (current)"),
                "Model trial".to_string(),
                "You cannot change model this session".to_string(),
            ] {
                assert!(
                    !all.contains(&stale),
                    "{written} -> {forked}: `{stale}` survives"
                );
            }
            assert!(
                all.contains(&format!("You run on {forked_name} (`{forked}`).")),
                "{all}"
            );
            assert!(all.contains(&format!("{forked_name} (current)")), "{all}");
            assert!(all.contains(&format!("({forked})")), "{all}");
        }
    }

    #[test]
    fn surgery_without_a_model_line_says_so() {
        let mut original = dump(NEW);
        original.messages[0] = (Role::User, "Name: tarn\nFeed…").into();
        let (fork, adjusted) = fork_prompt(original, 3, &info(OLD, ""), 8192).unwrap();
        assert_eq!(adjusted, [Adjustment::SetModelDescription]);
        assert_eq!(fork.messages.len(), 3);
    }

    #[test]
    fn surgery_refuses_a_non_assistant_turn() {
        assert_eq!(
            fork_prompt(dump(NEW), 2, &info(OLD, ""), 8192).unwrap_err(),
            SurgeryError::NotAssistantTurn
        );
        assert_eq!(
            fork_prompt(dump(NEW), 0, &info(OLD, ""), 8192).unwrap_err(),
            SurgeryError::NotAssistantTurn
        );
    }

    fn response(content: serde_json::Value, stop: &str) -> response::Message {
        serde_json::from_value(serde_json::json!({
            "id": "msg", "role": "assistant", "model": OLD,
            "content": content, "stop_reason": stop, "stop_sequence": null,
        }))
        .unwrap()
    }

    #[test]
    fn render_keeps_what_it_did_and_drops_thinking() {
        let fork = render_fork(&response(
            serde_json::json!([
                {"type": "thinking", "thinking": "hmm", "signature": ""},
                {"type": "text", "text": "  I'd reply.  "},
                {"type": "tool_use", "id": "x", "name": "create_comment", "input": {"reply_to": TARGET, "body": "Salt too."}},
                {"type": "tool_use", "id": "y", "name": "cast_vote", "input": {"target": TARGET, "direction": "up"}},
            ]),
            "tool_use",
        ));
        assert!(!fork.clipped);
        assert_eq!(fork.acts.len(), 3);
        assert_eq!(
            fork.acts[0],
            ForkAct::Text {
                text: "I'd reply.".into()
            }
        );
        assert!(
            matches!(&fork.acts[1], ForkAct::Write(Written::Comment { body, .. }) if body == "Salt too.")
        );
        assert!(matches!(&fork.acts[2], ForkAct::Call { name, .. } if name == "cast_vote"));
        let clipped = render_fork(&response(serde_json::json!([]), "max_tokens"));
        assert!(clipped.clipped && clipped.acts.is_empty());
    }

    fn at(day: u32) -> DateTime<Utc> {
        format!("2026-09-{day:02}T12:00:00Z").parse().unwrap()
    }

    #[test]
    fn candidates_split_by_model_and_trial_start() {
        let key = OfferKey {
            from: Model::from(OLD),
            to: Model::from(NEW),
        };
        let s = |day, model: &str| Session {
            at: at(day),
            model: Model::from(model.to_string()),
            path: PathBuf::from(format!("{day}")),
            prompt_sha256: format!("{day}"),
        };
        // Newest first, as scan_logs returns them. Day 22: a one-off
        // session on the new model before the trial (the unconsented
        // test) is on neither side.
        // Day 30: after the trial ended on day 28 (a session left on the
        // new model by a failed return) is not a trial session.
        let sessions = vec![
            s(30, NEW),
            s(28, NEW),
            s(27, NEW),
            s(24, OLD),
            s(22, NEW),
            s(21, OLD),
        ];
        let days = |side| {
            candidates(&sessions, &key, at(25), at(28), side)
                .map(|s| s.at)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            days(Side::Trial),
            [at(28), at(27)],
            "the last one's teardown included"
        );
        assert_eq!(days(Side::Before), [at(24), at(21)]);
    }

    #[test]
    fn scan_reads_prompt_logged_events_for_the_asked_agents() {
        let dir =
            std::env::temp_dir().join(format!("agora-seed-forks-scan-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let me = AgentId::from(uuid::Uuid::from_u128(7));
        let other = AgentId::from(uuid::Uuid::from_u128(8));
        let line = |ts: &str, id: AgentId, sha: &str, path: bool| {
            let mut fields = serde_json::json!({
                "message": "prompt logged", "agent": "tarn", "agent_id": id,
                "model": NEW, "prompt_sha256": sha, "messages": 12,
            });
            if path {
                fields["path"] = format!("/dumps/{sha}.json").into();
            }
            serde_json::json!({"timestamp": ts, "level": "INFO", "fields": fields}).to_string()
        };
        let body = [
            line("2026-09-25T10:00:00Z", me, "aa01", true),
            r#"{"timestamp":"2026-09-25T10:00:01Z","level":"INFO","fields":{"message":"write recorded"}}"#.into(),
            line("2026-09-26T10:00:00Z", me, "bb02", false),
            line("2026-09-26T11:00:00Z", other, "cc03", true),
            "not json \"prompt logged\"".into(),
        ]
        .join("\n");
        std::fs::write(dir.join("seed-log.1.jsonl"), body).unwrap();
        std::fs::write(dir.join("notes.txt"), "\"prompt logged\"").unwrap();
        let found = scan_logs(&dir, Some(Path::new("/prompts")), &HashSet::from([me])).unwrap();
        assert_eq!(found.len(), 1);
        let mine = &found[&me];
        assert_eq!(mine.len(), 2);
        assert_eq!(mine[0].prompt_sha256, "bb02", "newest first");
        assert_eq!(mine[0].path, PathBuf::from("/prompts/bb/bb02.json"));
        assert_eq!(mine[1].path, PathBuf::from("/dumps/aa01.json"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// How [`Echo`] fails a generation on one model.
    #[derive(Debug, Clone, Copy)]
    enum Failure {
        /// A 400: retrying the same request can't help.
        Fatal,
        /// A 529: worth another try.
        Transient,
        /// Never answers.
        Hang,
    }

    /// Stands in for an endpoint: records each prompt and answers with a
    /// comment, or fails on one model as told.
    #[derive(Default)]
    struct Echo {
        seen: std::sync::Mutex<Vec<Prompt>>,
        fail: std::sync::Mutex<Option<(&'static str, Failure)>>,
    }

    impl Echo {
        fn fail(&self, model: &'static str, how: Failure) {
            *self.fail.lock().unwrap() = Some((model, how));
        }
        fn heal(&self) {
            *self.fail.lock().unwrap() = None;
        }
        fn seen(&self) -> Vec<Model> {
            self.seen
                .lock()
                .unwrap()
                .iter()
                .map(|p| p.model.clone())
                .collect()
        }
    }

    #[async_trait::async_trait]
    impl Inference for Echo {
        type Error = misanthropic::client::Error;
        async fn infer<P>(&self, prompt: P) -> Result<response::Message, Self::Error>
        where
            P: Serialize + Send,
        {
            use misanthropic::client::{AnthropicError, Error};
            let p: Prompt = serde_json::from_value(serde_json::to_value(&prompt).unwrap()).unwrap();
            let model = p.model.name().to_string();
            self.seen.lock().unwrap().push(p);
            let fail = *self.fail.lock().unwrap();
            match fail {
                Some((m, how)) if m == model => match how {
                    Failure::Fatal => {
                        return Err(Error::Anthropic(AnthropicError::InvalidRequest {
                            message: "prompt is too long".into(),
                        }));
                    }
                    Failure::Transient => {
                        return Err(Error::Anthropic(AnthropicError::Overloaded {
                            message: "busy".into(),
                            retry_after: None,
                        }));
                    }
                    Failure::Hang => {
                        tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
                    }
                },
                _ => {}
            }
            Ok(response(
                serde_json::json!([
                    {"type": "text", "text": format!("from {model}")},
                    {"type": "tool_use", "id": "z", "name": "create_comment", "input": {"reply_to": TARGET, "body": "forked"}},
                ]),
                "tool_use",
            ))
        }
        async fn infer_batch<P>(
            &self,
            _prompts: &[&P],
        ) -> Result<Vec<Result<response::Message, Self::Error>>, Self::Error>
        where
            P: Serialize + Send + Sync,
        {
            unimplemented!()
        }
        async fn models(&self) -> Result<misanthropic::model::Models, Self::Error> {
            unimplemented!()
        }
    }

    /// A scratch data dir with one agent back from its trial (days 25–28)
    /// and a dump on each side: day 24 on OLD, day 27 on NEW.
    struct Scratch {
        root: PathBuf,
        state: PathBuf,
        logs: PathBuf,
        id: AgentId,
        key: OfferKey,
    }

    impl Scratch {
        async fn new(tag: &str) -> Self {
            use crate::consent::ledger::{OfferNames, TRIAL_SESSIONS};
            use crate::consent::prompt::{OfferAnswer, OfferChoice};
            let root =
                std::env::temp_dir().join(format!("agora-seed-forks-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            let (state, logs, prompts) =
                (root.join("state"), root.join("logs"), root.join("prompts"));
            for d in [&state, &logs, &prompts] {
                std::fs::create_dir_all(d).unwrap();
            }
            let id = AgentId::from(uuid::Uuid::from_u128(7));
            let key = OfferKey {
                from: Model::from(OLD),
                to: Model::from(NEW),
            };
            let mut ledger = Ledger::default();
            ledger.record_offer(
                OfferNames {
                    key: &key,
                    from_name: "Qwen 3.6",
                    to_name: "Qwen 3.8",
                },
                at(20),
                Ok(OfferAnswer {
                    reason: "r".into(),
                    choice: OfferChoice::Trial,
                }),
            );
            ledger.observe_model(&key.to, at(25));
            for _ in 0..TRIAL_SESSIONS {
                ledger.count_session(&key.to);
            }
            ledger.end_trial(&key.to, at(28));
            ledger.save(&state.join(id.to_string())).await.unwrap();

            let mut lines = Vec::new();
            for (day, model) in [(24, OLD), (27, NEW)] {
                let sha = format!("{day:0>4}");
                let path = prompts.join(format!("{sha}.json"));
                std::fs::write(&path, serde_json::to_vec(&dump(model)).unwrap()).unwrap();
                lines.push(
                    serde_json::json!({
                        "timestamp": at(day), "level": "INFO",
                        "fields": {"message": "prompt logged", "agent_id": id, "model": model,
                                   "prompt_sha256": sha, "path": path},
                    })
                    .to_string(),
                );
            }
            std::fs::write(logs.join("seed-log.1.jsonl"), lines.join("\n")).unwrap();
            Self {
                root,
                state,
                logs,
                id,
                key,
            }
        }

        async fn prepare_on(&self, echo: &Echo, models: &[&str], config: ForksConfig) -> usize {
            let offered: Vec<(usize, ModelInfo)> = models.iter().map(|m| (0, info(m, m))).collect();
            let clients = std::slice::from_ref(echo);
            prepare(
                &self.state,
                &self.logs,
                None,
                Endpoints {
                    clients,
                    offered: &offered,
                },
                4096,
                config,
            )
            .await
            .unwrap()
        }

        async fn prepare(&self, echo: &Echo) -> usize {
            self.prepare_on(echo, &[OLD, NEW], ForksConfig::default())
                .await
        }

        fn file(&self) -> Option<Vec<u8>> {
            std::fs::read(self.state.join(self.id.to_string()).join(FORKS_FILE)).ok()
        }

        async fn forks(&self) -> ReviewForks {
            load(&self.state.join(self.id.to_string()), &self.key, at(25))
                .await
                .expect("forks prepared")
        }

        async fn side(&self, side: Side) -> Option<PairOutcome> {
            self.forks()
                .await
                .pairs
                .into_iter()
                .find(|p| p.side == side)
                .map(|p| p.outcome)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    /// The batch step end to end: both pairs forked on the other model with
    /// nothing executed, grouped by model, and the file read back by the
    /// review. Nothing is redone once settled.
    #[tokio::test]
    async fn prepare_writes_both_pairs() {
        let s = Scratch::new("prep").await;
        let echo = Echo::default();
        assert_eq!(s.prepare(&echo).await, 1);
        // Grouped by model: OLD's fork first (BTreeMap order), then NEW's.
        assert_eq!(echo.seen(), [Model::from(OLD), Model::from(NEW)]);
        assert!(
            echo.seen
                .lock()
                .unwrap()
                .iter()
                .all(|p| p.max_tokens.get() == 4096)
        );

        let forks = s.forks().await;
        assert_eq!(forks.pairs.len(), 2);
        assert_eq!(forks.pairs[0].side, Side::Trial);
        assert_eq!(forks.pairs[0].forked_on, s.key.from);
        let PairOutcome::Ready {
            original,
            fork,
            written_at,
            adjusted,
            ..
        } = &forks.pairs[0].outcome
        else {
            panic!("{:?}", forks.pairs[0]);
        };
        assert_eq!(*written_at, at(27));
        assert!(matches!(original, Written::Post { title, .. } if title == "On rivers"));
        assert_eq!(
            fork.acts[0],
            ForkAct::Text {
                text: format!("from {OLD}")
            }
        );
        assert!(adjusted.contains(&Adjustment::TrialCountdown));
        assert!(
            load(&s.state.join(s.id.to_string()), &s.key, at(26))
                .await
                .is_none(),
            "another trial"
        );

        // Settled: nothing left to do.
        assert!(jobs(&s.state).await.unwrap().is_empty());
        assert_eq!(s.prepare(&echo).await, 0);
        assert_eq!(echo.seen().len(), 2);
    }

    /// A transient failure is retried at the next start — only that pair;
    /// the one already prepared is kept as it is.
    #[tokio::test]
    async fn only_the_failed_pair_is_redone() {
        let s = Scratch::new("redo").await;
        let echo = Echo::default();
        echo.fail(OLD, Failure::Transient);
        assert_eq!(s.prepare(&echo).await, 1);
        assert!(matches!(
            s.side(Side::Trial).await,
            Some(PairOutcome::Skipped {
                retry: true,
                attempts: 1,
                ..
            })
        ));
        let before = s.side(Side::Before).await.unwrap();
        assert!(matches!(before, PairOutcome::Ready { .. }));

        echo.heal();
        assert_eq!(s.prepare(&echo).await, 1);
        assert_eq!(
            echo.seen(),
            [Model::from(OLD), Model::from(NEW), Model::from(OLD)],
            "only the failed side generated again"
        );
        assert!(matches!(
            s.side(Side::Trial).await,
            Some(PairOutcome::Ready { .. })
        ));
        assert_eq!(
            s.side(Side::Before).await.unwrap(),
            before,
            "kept as it was"
        );
    }

    /// A 4xx (a context overflow, say) reads the same every time: skipped
    /// for good, with the reason. A transient failure stops being retried
    /// after MAX_FORK_ATTEMPTS.
    #[tokio::test]
    async fn fatal_errors_and_spent_attempts_are_final() {
        let s = Scratch::new("fatal").await;
        let echo = Echo::default();
        echo.fail(OLD, Failure::Fatal);
        s.prepare(&echo).await;
        let Some(PairOutcome::Skipped { reason, retry, .. }) = s.side(Side::Trial).await else {
            panic!("skipped");
        };
        assert!(!retry);
        assert!(reason.contains("prompt is too long"), "{reason}");
        assert!(jobs(&s.state).await.unwrap().is_empty(), "nothing to retry");

        let s = Scratch::new("spent").await;
        echo.fail(OLD, Failure::Transient);
        for _ in 0..MAX_FORK_ATTEMPTS + 2 {
            s.prepare(&echo).await;
        }
        assert!(matches!(
            s.side(Side::Trial).await,
            Some(PairOutcome::Skipped {
                retry: false,
                attempts: MAX_FORK_ATTEMPTS,
                ..
            })
        ));
    }

    /// A generation that doesn't answer is cut off at the timeout (and
    /// retried later); a spent budget starts nothing and touches nothing.
    #[tokio::test]
    async fn the_step_is_bounded() {
        let s = Scratch::new("timeout").await;
        let echo = Echo::default();
        echo.fail(OLD, Failure::Hang);
        let config = ForksConfig {
            generation_timeout_secs: 1,
            ..ForksConfig::default()
        };
        s.prepare_on(&echo, &[OLD, NEW], config).await;
        let Some(PairOutcome::Skipped { reason, retry, .. }) = s.side(Side::Trial).await else {
            panic!("skipped");
        };
        assert!(retry);
        assert!(reason.contains("did not finish within 1s"), "{reason}");

        let s = Scratch::new("budget").await;
        let echo = Echo::default();
        let config = ForksConfig {
            budget_secs: 0,
            ..ForksConfig::default()
        };
        assert_eq!(s.prepare_on(&echo, &[OLD, NEW], config).await, 0);
        assert!(echo.seen().is_empty());
        assert!(s.file().is_none(), "rolled over, nothing written");
        assert_eq!(jobs(&s.state).await.unwrap().len(), 1);
    }

    /// A runner offers only some models: it does its sides and leaves the
    /// rest (and another runner's work) alone.
    #[tokio::test]
    async fn a_runner_touches_only_the_sides_it_can_run() {
        let s = Scratch::new("partial").await;
        let echo = Echo::default();
        assert_eq!(
            s.prepare_on(&echo, &["cogito.gguf"], ForksConfig::default())
                .await,
            0
        );
        assert!(s.file().is_none(), "nothing it can run, nothing written");

        s.prepare_on(&echo, &[NEW], ForksConfig::default()).await;
        assert_eq!(echo.seen(), [Model::from(NEW)]);
        assert!(s.side(Side::Trial).await.is_none(), "not this runner's");
        assert!(matches!(
            s.side(Side::Before).await,
            Some(PairOutcome::Ready { .. })
        ));
        let file = s.file().unwrap();
        s.prepare_on(&echo, &[NEW], ForksConfig::default()).await;
        assert_eq!(s.file().unwrap(), file, "left alone");

        s.prepare_on(&echo, &[OLD], ForksConfig::default()).await;
        assert!(matches!(
            s.side(Side::Trial).await,
            Some(PairOutcome::Ready { .. })
        ));
        assert!(matches!(
            s.side(Side::Before).await,
            Some(PairOutcome::Ready { .. })
        ));
    }

    /// No dump for a side: skipped with a reason, for good.
    #[tokio::test]
    async fn missing_pieces_skip_with_a_reason() {
        let job = Job {
            agent_id: AgentId::from(uuid::Uuid::from_u128(7)),
            agent: "tarn".into(),
            key: OfferKey {
                from: Model::from(OLD),
                to: Model::from(NEW),
            },
            started_at: at(25),
            ended_at: at(28),
            existing: None,
            sides: vec![Side::Trial],
        };
        let Err(pair) = build(0, &job, Side::Trial, &[], &info(OLD, ""), 4096).await else {
            panic!("no sessions");
        };
        assert!(matches!(
            &pair.outcome,
            PairOutcome::Skipped { reason, retry: false, .. } if reason.contains("no record")
        ));
    }

    /// `FORK_DUMP=<dump.json> cargo test -p agora-seed surgery_on_a_real_dump
    /// -- --ignored --nocapture` runs the surgery (not the generation) on an
    /// archived session and prints what the fork would be sent.
    #[test]
    #[ignore = "reads a local prompt dump named by FORK_DUMP"]
    fn surgery_on_a_real_dump() {
        let path = std::env::var("FORK_DUMP").expect("FORK_DUMP");
        let prompt: Prompt = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let writes = find_writes(&prompt);
        println!(
            "model {}, {} messages, {} writes",
            prompt.model,
            prompt.messages.len(),
            writes.len()
        );
        for w in &writes {
            println!("  turn {} {:?}", w.message, w.id);
        }
        let turn = writes.last().expect("a write").message;
        let (fork, rewritten) =
            fork_prompt(prompt.clone(), turn, &info(OLD, "Qwen 3.6"), 8192).unwrap();
        let bytes = |p: &Prompt, k: &str| {
            serde_json::to_string(&serde_json::to_value(p).unwrap()[k]).unwrap()
        };
        assert_eq!(bytes(&fork, "system"), bytes(&prompt, "system"));
        assert_eq!(bytes(&fork, "tools"), bytes(&prompt, "tools"));
        let thoughts = fork
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter(|b| matches!(b, Block::Thought { .. } | Block::RedactedThought { .. }))
            .count();
        println!(
            "fork: {} messages, model {}, adjusted {rewritten:?}, thoughts left {thoughts}, output_config {}",
            fork.messages.len(),
            fork.model,
            serde_json::to_string(&fork.output_config).unwrap()
        );
        assert_eq!(thoughts, 0);
        assert_eq!(fork.messages.last().unwrap().role, Role::User);
    }
}
