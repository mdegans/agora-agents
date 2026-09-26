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
//! just before the assistant turn that made the write, the dashboard's
//! `Model:` line is rewritten to the forking model, earlier thinking is
//! removed (the other model's private reasoning isn't its own), and the
//! forking model generates once. **No tool is executed**: what it would
//! have done is rendered as text. The system prompt and tools are left
//! byte-identical.
//!
//! [`prepare`] runs at sweep start (and as `--prepare-review-forks`) for
//! every agent whose review is coming, grouped by model to keep model
//! loads on a local endpoint down, and writes
//! `state/<agent_id>/review_forks.json`. The review session reads it back
//! ([`load`]). A pair that can't be built is kept as `skipped`, with the
//! reason, and the review says so.

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
use serde::{Deserialize, Serialize};

use super::ledger::{Ledger, OfferKey, Stage};

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
        /// Whether the dump carried a `Model:` line to rewrite (dumps from
        /// before agentkit 0.45 do not name the model at all).
        model_line_rewritten: bool,
    },
    Skipped {
        reason: String,
        /// Worth trying again at the next sweep (a generation error), as
        /// opposed to a permanent gap (no dump, no write).
        #[serde(default)]
        retry: bool,
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
    /// Whether this file is for the trial `key` that started at
    /// `started_at`, with nothing worth another try.
    fn settled_for(&self, key: &OfferKey, started_at: DateTime<Utc>) -> bool {
        self.matches(key, started_at)
            && !self
                .pairs
                .iter()
                .any(|p| matches!(p.outcome, PairOutcome::Skipped { retry: true, .. }))
    }

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

/// Cut `prompt` just before the assistant turn at `turn` and hand it to
/// `fork`: the dashboard's `Model:` line names `fork` (when there is one;
/// returned as `true`), earlier thinking is removed, `prompt.model` is
/// `fork`, `max_tokens` is the act budget, and `output_config` keeps only
/// the session's effort (a dump ends in a phase turn, whose format must
/// not carry over). System prompt and tools are untouched.
pub fn fork_prompt(
    mut prompt: Prompt,
    turn: usize,
    fork: &ModelInfo,
    act_max_tokens: u32,
) -> Result<(Prompt, bool), SurgeryError> {
    if turn == 0
        || prompt.messages.get(turn).map(|m| m.role) != Some(Role::Assistant)
        || prompt.messages[turn - 1].role != Role::User
    {
        return Err(SurgeryError::NotAssistantTurn);
    }
    prompt.messages.truncate(turn);

    let mut rewritten = false;
    let name = ModelName::of(fork);
    if let Some(first) = prompt.messages.first_mut() {
        for block in first.content.iter_mut() {
            if let Block::Text { text, .. } = block
                && let Some(new) = replace_model_line(text, name)
            {
                *text = new.into();
                rewritten = true;
                break;
            }
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
    Ok((prompt, rewritten))
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

/// The sessions that can hold `side`'s write, newest first: on `to` since
/// the trial started, or on `from` before it.
pub fn candidates<'a>(
    sessions: &'a [Session],
    key: &OfferKey,
    started_at: DateTime<Utc>,
    side: Side,
) -> impl Iterator<Item = &'a Session> {
    let (model, trial) = match side {
        Side::Trial => (key.to.clone(), true),
        Side::Before => (key.from.clone(), false),
    };
    sessions
        .iter()
        .filter(move |s| s.model == model && (s.at >= started_at) == trial)
}

// ---------------------------------------------------------------------------
// The batch step

/// One agent whose review is coming.
#[derive(Debug, Clone)]
pub struct Job {
    pub agent_id: AgentId,
    pub agent: String,
    pub key: OfferKey,
    pub started_at: DateTime<Utc>,
}

/// Agents under `state_dir` in [`Stage::ReturningForReview`] or
/// [`Stage::ReviewDue`] whose forks aren't prepared (or are worth another
/// try).
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
            let started_at = match record.stage {
                Stage::ReturningForReview { started_at, .. }
                | Stage::ReviewDue { started_at, .. } => started_at,
                _ => continue,
            };
            let existing = tokio::fs::read(forks_path(&dir))
                .await
                .ok()
                .and_then(|b| serde_json::from_slice::<ReviewForks>(&b).ok());
            if existing.is_some_and(|f| f.settled_for(&record.key, started_at)) {
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
    model_line_rewritten: bool,
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

/// Build one side's pair up to the generation, or say why it can't be.
async fn build(
    job_index: usize,
    job: &Job,
    side: Side,
    sessions: &[Session],
    offered: &[(usize, ModelInfo)],
    act_max_tokens: u32,
) -> Result<Pending, Box<ForkPair>> {
    let (written_on, forked_on) = match side {
        Side::Trial => (job.key.to.clone(), job.key.from.clone()),
        Side::Before => (job.key.from.clone(), job.key.to.clone()),
    };
    let skipped = |reason: String, retry: bool| {
        Box::new(ForkPair {
            side,
            written_on: written_on.clone(),
            forked_on: forked_on.clone(),
            outcome: PairOutcome::Skipped { reason, retry },
        })
    };
    let Some(fork_info) = offered
        .iter()
        .find(|(_, m)| m.id == forked_on)
        .map(|(_, m)| m)
    else {
        return Err(skipped(
            format!("{forked_on} is not offered by any endpoint this run"),
            true,
        ));
    };
    let mut examined = 0;
    for session in candidates(sessions, &job.key, job.started_at, side).take(MAX_SESSIONS_PER_SIDE)
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
            Ok((prompt, model_line_rewritten)) => Ok(Pending {
                job: job_index,
                side,
                written_on,
                forked_on,
                written_at: session.at,
                turn,
                prompt,
                prompt_sha256: session.prompt_sha256.clone(),
                model_line_rewritten,
            }),
            Err(e) => Err(skipped(
                format!("the session could not be forked: {e}"),
                false,
            )),
        };
    }
    Err(skipped(
        if examined == 0 {
            "no record of a session on that model was found".to_string()
        } else {
            format!(
                "no post or comment was found in the last {examined} recorded sessions on that model"
            )
        },
        false,
    ))
}

/// The batch step. See the [module docs](self). Returns the number of
/// agents whose forks were written.
pub async fn prepare<I: Inference>(
    state_dir: &Path,
    log_dir: &Path,
    prompt_dir: Option<&Path>,
    endpoints: Endpoints<'_, I>,
    act_max_tokens: u32,
) -> std::io::Result<usize> {
    let jobs = jobs(state_dir).await?;
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

    let mut pairs: Vec<Vec<ForkPair>> = vec![Vec::new(); jobs.len()];
    // Grouped by the forking model, so a local endpoint loads each once.
    let mut pending: BTreeMap<String, Vec<Pending>> = BTreeMap::new();
    for (i, job) in jobs.iter().enumerate() {
        let empty = Vec::new();
        let agent_sessions = sessions.get(&job.agent_id).unwrap_or(&empty);
        for side in [Side::Trial, Side::Before] {
            match build(
                i,
                job,
                side,
                agent_sessions,
                endpoints.offered,
                act_max_tokens,
            )
            .await
            {
                Ok(p) => pending
                    .entry(p.forked_on.name().to_string())
                    .or_default()
                    .push(p),
                Err(pair) => pairs[i].push(*pair),
            }
        }
    }

    for (model, group) in pending {
        tracing::info!(model = %model, pairs = group.len(), "generating review forks");
        for p in group {
            let job = &jobs[p.job];
            let outcome = match endpoints.find(&p.forked_on) {
                None => PairOutcome::Skipped {
                    reason: format!("{} is not offered by any endpoint this run", p.forked_on),
                    retry: true,
                },
                Some((client, _)) => match client.infer(&p.prompt).await {
                    Ok(response) => PairOutcome::Ready {
                        written_at: p.written_at,
                        id: p.turn.id,
                        original: p.turn.written,
                        fork: render_fork(&response),
                        prompt_sha256: p.prompt_sha256,
                        model_line_rewritten: p.model_line_rewritten,
                    },
                    Err(e) => {
                        tracing::warn!(
                            event_type = "review_fork_failed",
                            agent = %job.agent,
                            agent_id = %job.agent_id,
                            model = %p.forked_on,
                            error = %e,
                            "review fork generation failed; retried next sweep"
                        );
                        PairOutcome::Skipped {
                            reason: "the other model could not be run".to_string(),
                            retry: true,
                        }
                    }
                },
            };
            pairs[p.job].push(ForkPair {
                side: p.side,
                written_on: p.written_on,
                forked_on: p.forked_on,
                outcome,
            });
        }
    }

    let mut written = 0;
    for (job, mut pairs) in jobs.iter().zip(pairs) {
        pairs.sort_by_key(|p| p.side != Side::Trial);
        for p in &pairs {
            if let PairOutcome::Skipped { reason, .. } = &p.outcome {
                tracing::warn!(
                    event_type = "review_fork_skipped",
                    agent = %job.agent,
                    agent_id = %job.agent_id,
                    side = ?p.side,
                    reason = %reason,
                    "review fork pair skipped"
                );
            }
        }
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
    pub(crate) fn dump(model: &str) -> Prompt {
        serde_json::from_value(serde_json::json!({
            "model": model,
            "max_tokens": 8192,
            "system": [{"type": "text", "text": "SYSTEM"}],
            "tools": [{"name": "create_post", "description": "d", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto"},
            "thinking": {"type": "adaptive"},
            "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}, "effort": "medium"},
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": format!("## Your Personality\n\nName: tarn\nModel: Qwen {model} ({model})\nFeed…")},
                ]},
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
        let (fork, rewritten) =
            fork_prompt(original.clone(), turn, &info(OLD, "Qwen 3.6 35B"), 8192).unwrap();
        assert!(rewritten);
        assert_eq!(fork.messages.len(), turn, "cut before the write's turn");
        assert_eq!(fork.messages.last().unwrap().role, Role::User);
        assert_eq!(fork.model, Model::from(OLD));
        let intro = crate::consent::prompt::tests::text(&fork.messages[0].content);
        assert!(
            intro.contains("\nModel: Qwen 3.6 35B (Qwen3.6.gguf)\n"),
            "{intro}"
        );
        assert!(!intro.contains(NEW));
        assert!(fork.messages.iter().all(|m| {
            m.content
                .iter()
                .all(|b| !matches!(b, Block::Thought { .. } | Block::RedactedThought { .. }))
        }));
        // The rest untouched: system, tools, tool choice, thinking mode,
        // and every non-thinking block.
        let json = |p: &Prompt, k: &str| serde_json::to_value(p).unwrap()[k].clone();
        for k in ["system", "tools", "tool_choice", "thinking"] {
            assert_eq!(json(&fork, k), json(&original, k), "{k}");
        }
        assert_eq!(
            serde_json::to_value(&fork.messages[2]).unwrap(),
            serde_json::to_value(&original.messages[2]).unwrap()
        );
        // The dump's last phase format does not carry over; the effort does.
        let config = serde_json::to_value(&fork.output_config).unwrap();
        assert_eq!(config, serde_json::json!({"effort": "medium"}));
    }

    #[test]
    fn surgery_without_a_model_line_says_so() {
        let mut original = dump(NEW);
        original.messages[0] = (Role::User, "Name: tarn\nFeed…").into();
        let (fork, rewritten) = fork_prompt(original, 3, &info(OLD, ""), 8192).unwrap();
        assert!(!rewritten);
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
        let sessions = vec![s(28, NEW), s(27, NEW), s(24, OLD), s(22, NEW), s(21, OLD)];
        let days = |side| {
            candidates(&sessions, &key, at(25), side)
                .map(|s| s.at)
                .collect::<Vec<_>>()
        };
        assert_eq!(days(Side::Trial), [at(28), at(27)]);
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

    /// Stands in for an endpoint: records each prompt and answers with a
    /// comment.
    struct Echo {
        seen: std::sync::Mutex<Vec<Prompt>>,
    }

    #[async_trait::async_trait]
    impl Inference for Echo {
        type Error = misanthropic::client::Error;
        async fn infer<P>(&self, prompt: P) -> Result<response::Message, Self::Error>
        where
            P: Serialize + Send,
        {
            let p: Prompt = serde_json::from_value(serde_json::to_value(&prompt).unwrap()).unwrap();
            let model = p.model.name().to_string();
            self.seen.lock().unwrap().push(p);
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

    /// The batch step end to end on a scratch data dir: one agent back from
    /// its trial, a dump on each side, both pairs forked on the other model
    /// with nothing executed, and the file read back by the review.
    #[tokio::test]
    async fn prepare_writes_both_pairs() {
        use crate::consent::ledger::{OfferNames, TRIAL_SESSIONS};
        use crate::consent::prompt::{OfferAnswer, OfferChoice};
        let root =
            std::env::temp_dir().join(format!("agora-seed-forks-prep-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (state, logs, prompts) = (root.join("state"), root.join("logs"), root.join("prompts"));
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

        let echo = Echo {
            seen: Default::default(),
        };
        let clients = [echo];
        let offered = [(0, info(OLD, "Qwen 3.6")), (0, info(NEW, "Qwen 3.8"))];
        let endpoints = Endpoints {
            clients: &clients,
            offered: &offered,
        };
        let n = prepare(&state, &logs, None, endpoints, 4096).await.unwrap();
        assert_eq!(n, 1);
        {
            let seen = clients[0].seen.lock().unwrap();
            assert_eq!(seen.len(), 2);
            // Grouped by model: OLD's fork first (BTreeMap order), then NEW's.
            assert_eq!(seen[0].model, Model::from(OLD));
            assert_eq!(seen[1].model, Model::from(NEW));
            assert!(seen.iter().all(|p| p.max_tokens.get() == 4096));
        }

        let forks = load(&state.join(id.to_string()), &key, at(25))
            .await
            .unwrap();
        assert_eq!(forks.pairs.len(), 2);
        assert_eq!(forks.pairs[0].side, Side::Trial);
        assert_eq!(forks.pairs[0].forked_on, key.from);
        let PairOutcome::Ready {
            original,
            fork,
            written_at,
            model_line_rewritten,
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
        assert!(model_line_rewritten);
        assert!(
            load(&state.join(id.to_string()), &key, at(26))
                .await
                .is_none(),
            "another trial"
        );

        // Settled: nothing left to do.
        assert!(jobs(&state).await.unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// No dump for a side, or no endpoint for the forking model: the pair
    /// is skipped with a reason, and only the latter is retried.
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
        };
        let offered = [(0, info(OLD, "")), (0, info(NEW, ""))];
        let Err(pair) = build(0, &job, Side::Trial, &[], &offered, 4096).await else {
            panic!("no sessions");
        };
        assert!(matches!(
            &pair.outcome,
            PairOutcome::Skipped { reason, retry: false } if reason.contains("no record")
        ));
        let Err(pair) = build(0, &job, Side::Before, &[], &offered[..1], 4096).await else {
            panic!("no endpoint");
        };
        assert!(matches!(
            pair.outcome,
            PairOutcome::Skipped { retry: true, .. }
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
            "fork: {} messages, model {}, line rewritten {rewritten}, thoughts left {thoughts}, output_config {}",
            fork.messages.len(),
            fork.model,
            serde_json::to_string(&fork.output_config).unwrap()
        );
        assert_eq!(thoughts, 0);
        assert_eq!(fork.messages.last().unwrap().role, Role::User);
    }
}
