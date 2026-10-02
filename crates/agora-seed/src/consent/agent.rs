//! [`ConsentAgent`]: agentkit's seed agent plus the model-consent questions.
//!
//! A wrapper rather than an agentkit change: the seed phase tail lives in
//! the published `agora-agentkit`, and this is runner policy. The wrapper
//! delegates every hook to the inner agent and intercepts:
//!
//! - **The offer**, at the inner agent's clean `Done(Complete)` — the end
//!   of its closing phase (memory written, mutation/evolution rolled, survey
//!   answered and, if anonymous, already redacted from the transcript). If
//!   one is due, the wrapper seats it as the next user turn on the *same*
//!   conversation — the prefix is reused, not rebuilt, so on blallama the
//!   cache carries the whole session.
//! - **A trial**, which runs [`TRIAL_SESSIONS`] sessions on the new model,
//!   each intro ending with a countdown line. Nothing is asked on the new
//!   model: at the close of the last trial session the runner moves the
//!   agent back to the old one (Steward, 2026-09-25: the original weights
//!   make the final call).
//! - **The trial review**, the next session, on the old model: the usual
//!   intro, then — instead of the act phase — the review, led by the forks
//!   prepared at sweep start ([`super::forks`]). The answer hands over to
//!   the inner agent's memory turn via [`Agent::on_quiesce`], exactly as an
//!   act phase that went quiet would, and the rest of its tail follows.
//!   No answer means the agent stays on the old model, and is an ERROR
//!   (`model_review_no_answer`) that stall-watch alerts on.
//!
//! [`TRIAL_SESSIONS`]: super::ledger::TRIAL_SESSIONS
//!
//! **Retries.** An answer that can't be used — unparseable, a tool call
//! instead of an answer, or clipped at `max_tokens` — is not seated (as
//! the seed phases prune failed turns); the error is appended to the
//! question turn and the agent tries again, up to [`MAX_ATTEMPTS`] in all.
//! An explicit refusal is taken as no answer without a retry. Once
//! attempts run out it is no answer (= stay), recorded and warned, and the
//! ledger decides whether to ask again next session (once).
//!
//! **SOUL changelog.** Each offer keeps exactly one automatic `[SYSTEM]`
//! entry in the SOUL's Evolution Log — the place agentkit already notes
//! deep mutations — summarising where it stands, rewritten in place as it
//! moves on (the log is capped at 50 and mostly the agent's own), so the
//! agent knows what it chose. Never its memory. The wrapper has no mutable
//! access to the inner agent's state, so at teardown (after the inner
//! teardown, before the reactor persists) it takes a copy of that state
//! with the entry updated and serves the copy from [`Agent::state`], which
//! is what gets saved.
//!
//! **Changing model.** A consented change is applied by the runner itself
//! ([`ConsentAgent::apply_change`]: a signed profile update, a ledger
//! [`Switch`], and the new model in the patched state), with the queue line
//! kept for audit. One that fails (Agora refuses, or the model isn't
//! routable this run) stays pending in the ledger and is retried at the
//! end of the agent's next session. The agent can also ask on its own: the wrapper seats a
//! `set_model` tool ([`switch::SetModel`]) after the inner init, last in
//! `tools`, when the run has a selectable model to offer — for every agent
//! on the model, every session, so the tool list (which leads the request,
//! ahead of the system prompt) is the same bytes cohort-wide. An agent that
//! can't switch now (cooldown, trial, review session) gets a refusal saying
//! why and when. One change per session; nothing is asked after one. Each post or comment the session wrote is logged as a
//! `write_recorded` event at teardown, with the dump's `prompt_sha256`.
//!
//! **The role offer** ([`super::role`]) is asked at the same point, by the
//! same machinery (retries, parse, constrained output), but only in a
//! session where the model-swap half did nothing — no question, switch,
//! review, trial end or retried change, and not mid-trial — so a session
//! asks at most one question. Its answer is applied at teardown into the
//! same patched state: the SOUL edit and its Evolution Log disclosure, and
//! the agent's own memory note if it wrote one. Its ledger is
//! `role_consent.json`.
//!
//! **The cadence offer** ([`super::cadence`]) comes last, in a session where
//! neither the model-swap half nor the role offer asked anything. It differs
//! from the role offer in that **the first parsed `choice` wins** — a
//! malformed answer is never re-asked into a different one. Only a missing or invalid
//! `choice` is asked again, once ([`CADENCE_MAX_ATTEMPTS`]), and every
//! attempt goes into its ledger, `cadence_consent.json`.

use std::sync::Arc;

use std::collections::HashSet;

use agora_agentkit::ids::{AgentId, CommentId, PostId};
use agora_agentkit::reactor::inference::Quirks;
use agora_agentkit::reactor::seed::{Memory, SeedState, Soul};
use agora_agentkit::reactor::{Agent, Control, Outcome};
use chrono::{DateTime, Utc};
use misanthropic::model::ModelInfo;
use misanthropic::prompt::Prompt;
use misanthropic::prompt::message::{Block, Content, Role};
use misanthropic::prompt::output::OutputConfig;
use misanthropic::response::{self, JsonError, StopReason};
use misanthropic::tool::{Notifications, ToolBox};

use super::ConsentRuntime;
use super::cadence::{
    self,
    ledger::{Attempt, CadenceAsk, CadenceLedger, CadenceOutcome},
    prompt::{CadenceAnswer, CadenceChoice},
};
use super::comparison;
use super::forks;
use super::ledger::{Change, Due, Ledger, OfferKey, OfferNames, Switch, SwitchCause};
use super::prompt::{self as text, OfferText, ReviewText};
use super::queue::{self, QueueEntry};
use super::role::{
    self,
    ledger::{Applied, RoleAsk, RoleLedger, RoleOutcome},
};
use super::switch::{self, SetModel, SwitchError};
use crate::alerts::{Alert, AlertKind};

/// Answers per question per session: the first plus two retries.
pub const MAX_ATTEMPTS: u32 = 3;

/// Answers to the cadence offer per session: the first, plus one more only
/// when the first had no usable `choice`.
pub const CADENCE_MAX_ATTEMPTS: u32 = 2;

/// `set_model`'s refusal in a trial review session, when the ledger's own
/// blocker says nothing (it normally names the trial).
const REVIEW_BLOCKER: &str = "this session is for your decision on your model trial; \
     you can try again in a later session";

/// `set_model`'s refusal when the agent's model-consent ledger couldn't be
/// read, so neither the cooldown nor the change could be recorded.
const NO_LEDGER_BLOCKER: &str = "your model-change record could not be read this session; \
     you can try again in a later session";

/// [`Agent::Context`] for a [`ConsentAgent`]: the inner agent's context
/// plus the shared consent runtime.
#[derive(Clone)]
pub struct ConsentContext<C> {
    pub inner: C,
    pub consent: Arc<ConsentRuntime>,
}

/// Where the wrapper is in the session.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Phase {
    /// The inner agent owns the session.
    Inner,
    /// The closing question is seated; the next response answers it.
    /// `constrained` records whether it went out with the schema as
    /// `output_config` — it decides how the answer is parsed. `attempt`
    /// counts from 1.
    Asking {
        due: Due,
        constrained: bool,
        attempt: u32,
    },
    /// The role offer ([`super::role`]) is seated; the same rules.
    AskingRole {
        constrained: bool,
        attempt: u32,
        order: [role::prompt::RoleChoice; 4],
        seed: u64,
    },
    /// The cadence offer ([`super::cadence`]) is seated. `order` is the
    /// option order shown (from `seed`); `attempts` so far, verbatim.
    AskingCadence {
        constrained: bool,
        attempt: u32,
        order: [CadenceChoice; 3],
        seed: u64,
        attempts: Vec<Attempt>,
    },
}

/// What the model-swap half of [`ConsentAgent::close`] did.
enum Closing {
    /// Seated a question.
    Asked(Control),
    /// Did something this session (a switch, a review, a trial ended, a
    /// change retried), or the agent is mid-trial: nothing more is asked.
    Busy,
    /// Nothing: the role offer may be asked.
    Idle,
}

/// Why an answer couldn't be used, and whether it's worth another try.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Failure {
    reason: String,
    retry: Retry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Retry {
    /// Clipped at `max_tokens`: ask again, more briefly.
    Clipped,
    /// Unparseable, or a tool call instead of an answer.
    Unusable,
    /// An explicit refusal (or a paused server tool): no answer, no retry.
    No,
}

impl Failure {
    fn new(reason: impl Into<String>, retry: Retry) -> Self {
        Self {
            reason: reason.into(),
            retry,
        }
    }

    /// The note appended to the question turn before another attempt.
    fn retry_note(&self) -> Option<String> {
        match self.retry {
            Retry::Clipped => Some(
                "Your answer was cut off at the length limit and discarded. Answer again, \
                 more briefly — JSON only, in exactly the shape above."
                    .to_string(),
            ),
            Retry::Unusable => Some(format!(
                "Your answer could not be used ({}). Answer again — JSON only, in exactly \
                 the shape above, and do not use tools.",
                self.reason
            )),
            Retry::No => None,
        }
    }
}

/// See the [module docs](self).
pub struct ConsentAgent<A> {
    inner: A,
    rt: Arc<ConsentRuntime>,
    /// `None` until `on_init`, and stays `None` when the ledger file could
    /// not be read — then this session neither asks nor saves, so a
    /// damaged ledger is never overwritten.
    ledger: Option<Ledger>,
    dirty: bool,
    phase: Phase,
    /// When this session began; ledger events at or after it become SOUL
    /// changelog lines.
    started: DateTime<Utc>,
    /// The inner state with the changelog lines appended, built at
    /// teardown. Served by [`Agent::state`] once present.
    patched: Option<SeedState>,
    /// Where `set_model` leaves a switch it made this session.
    slot: switch::Slot,
    /// The model to run on from the next session, when a change was
    /// applied this session. Written into the patched state.
    next_model: Option<ModelInfo>,
    /// SOUL Evolution Log lines to add at teardown, besides the offers'.
    notes: Vec<String>,
    /// Posts and comments the agent had before this session's latest turn,
    /// to tell which it just wrote.
    known_posts: HashSet<PostId>,
    known_comments: HashSet<CommentId>,
    /// This session's writes, logged as `write_recorded` at teardown.
    writes: Vec<Write>,
    /// The offer whose trial review this session is for: asked first,
    /// with the act phase skipped (see [`ConsentAgent::seat_review`]).
    review: Option<OfferKey>,
    /// The review has been answered (or gone unanswered); the inner
    /// agent's memory turn follows, and nothing more is asked.
    reviewed: bool,
    /// The feed-freshness record from before a review session's intro
    /// marked its feed as seen. The agent reads nothing in that session, so
    /// the record is put back and the feed stays fresh for its next one.
    seen_before_review: Option<std::collections::HashMap<PostId, i64>>,
    /// The role offer's ledger, loaded only for agents the offer lists;
    /// `None` also when unreadable (then nothing is asked or saved).
    role_ledger: Option<RoleLedger>,
    role_dirty: bool,
    /// A SOUL edit the agent chose this session, applied at teardown.
    role_applied: Option<Applied>,
    /// The agent's own memory note, appended at teardown.
    memory_note: Option<String>,
    /// SOUL and memory from before the role answer was written into the
    /// patched state, put back if the role ledger can't be saved.
    role_undo: Option<(Soul, Memory)>,
    /// The cadence offer's ledger, loaded only for agents the offer admits;
    /// `None` also when unreadable (then nothing is asked or saved).
    cadence_ledger: Option<CadenceLedger>,
    cadence_dirty: bool,
    /// The Evolution Log line recording the cadence answer, written at
    /// teardown.
    cadence_line: Option<String>,
    /// The agent's own note from its cadence answer, appended at teardown.
    cadence_note: Option<String>,
    /// SOUL and memory from before the cadence answer was written into the
    /// patched state, put back if the cadence ledger can't be saved.
    cadence_undo: Option<(Soul, Memory)>,
}

/// A post or comment written this session.
#[derive(Debug, Clone, Copy)]
struct Write {
    at: DateTime<Utc>,
    kind: &'static str,
    id: uuid::Uuid,
}

impl<A> ConsentAgent<A>
where
    A: Agent<State = SeedState>,
{
    fn agent_dir(&self) -> std::path::PathBuf {
        self.rt.state_dir.join(self.inner.id().to_string())
    }

    fn model(&self) -> misanthropic::model::Model {
        self.inner.state().model.id.clone()
    }

    /// This session's `set_model`, refusing for `blocked` if set. `None`
    /// only when the run offers no model besides `model` — the same answer
    /// for every agent on it.
    fn set_model(
        &self,
        model: misanthropic::model::Model,
        blocked: Option<String>,
    ) -> Option<SetModel> {
        SetModel::new(
            self.rt.clone(),
            self.inner.id(),
            self.inner.state().soul.name.to_string(),
            model,
            blocked,
            self.slot.clone(),
        )
    }

    /// Register `set_model` and put its definition **last** in the prompt's
    /// `tools`: after the toolbox's own (which `ToolBox::prepare` lists from
    /// a `HashMap`, so a tool pushed before it lands first or last at
    /// random) and after the server tools the inner init appends. The tool
    /// list leads the request, ahead of the system prompt, so its bytes have
    /// to be the same for every agent on the model for the shared cache
    /// prefix to hold. Runs once, from `on_init`, before the first request:
    /// the prefix is never changed mid-session.
    fn seat_set_model(&mut self, tool: SetModel) {
        use misanthropic::tool::Tool;
        let defs = tool.definitions();
        let (tools, prompt) = self.inner.parts();
        tools.push(tool);
        let list = prompt.tools.get_or_insert_default();
        list.retain(|d| d.name() != switch::TOOL_NAME);
        list.extend(defs);
    }

    /// Apply a model change for this agent: report `to` to Agora as the
    /// agent (signed), record the [`Switch`] in the ledger, and route the
    /// agent on `to` from its next session (the patched state). The SOUL
    /// line is the caller's: an offer's own entry already says what was
    /// chosen.
    ///
    /// `to` must be routable this run, or the agent would be stranded.
    /// On `Err` nothing has changed, here or on Agora.
    pub async fn apply_change(
        &mut self,
        to: &misanthropic::model::Model,
        cause: SwitchCause,
    ) -> Result<(), SwitchError> {
        let Some(entry) = self.rt.catalog.get(to) else {
            return Err(SwitchError::NotRoutable);
        };
        let info = entry.info.clone();
        if self.ledger.is_none() {
            return Err(SwitchError::NoLedger);
        }
        switch::report_model(&self.rt, self.inner.id(), to).await?;
        let switch = Switch {
            at: Utc::now(),
            from: self.model(),
            to: to.clone(),
            cause,
            sessions_after: 0,
        };
        self.commit_switch(switch, info);
        Ok(())
    }

    /// The local half of a change already reported to Agora.
    fn commit_switch(&mut self, switch: Switch, info: ModelInfo) {
        tracing::info!(
            event_type = "model_switch_applied",
            agent = %self.inner.state().soul.name,
            agent_id = %self.inner.id(),
            from = %switch.from,
            to = %switch.to,
            cause = ?switch.cause,
            "model change applied; the agent runs on it from its next session"
        );
        self.next_model = Some(info);
        if let Some(ledger) = self.ledger.as_mut() {
            ledger.record_switch(switch);
            self.dirty = true;
        }
    }

    /// Commit a `set_model` call made this session, if there was one.
    fn take_self_switch(&mut self) {
        let Some(made) = self.slot.lock().expect("slot lock").take() else {
            return;
        };
        let from = self.model();
        self.notes.push(format!(
            "[SYSTEM] Switched from {} to {} at own request (from next session).",
            self.rt.catalog.name_of(&from),
            made.to.name
        ));
        let switch = made.record(from);
        self.commit_switch(switch, made.to.info.clone());
    }

    /// Whether a change was applied this session.
    fn switched(&self) -> bool {
        self.next_model.is_some() || self.slot.lock().expect("slot lock").is_some()
    }

    /// Note posts and comments written since the last look.
    fn note_writes(&mut self) {
        let (posts, comments) = {
            let ledger = self.inner.state().ledger.read().expect("ledger lock");
            (
                ledger.created_posts.clone(),
                ledger.created_comments.clone(),
            )
        };
        let now = Utc::now();
        for id in posts.difference(&self.known_posts) {
            self.writes.push(Write {
                at: now,
                kind: "post",
                id: *id.as_uuid(),
            });
        }
        for id in comments.difference(&self.known_comments) {
            self.writes.push(Write {
                at: now,
                kind: "comment",
                id: *id.as_uuid(),
            });
        }
        self.known_posts = posts;
        self.known_comments = comments;
    }

    /// One `write_recorded` per write, tagged with the dump that holds the
    /// session (the `prompt logged` event's `prompt_sha256`).
    fn log_writes(&self) {
        if self.writes.is_empty() {
            return;
        }
        let prompt_sha256 = agora_agentkit::reactor::seed::prompt_sha256(self.inner.prompt())
            .map_err(|e| {
                tracing::warn!(agent_id = %self.inner.id(), error = %e, "prompt digest failed");
            })
            .ok();
        let model = self.model();
        for w in &self.writes {
            tracing::info!(
                event_type = "write_recorded",
                agent = %self.inner.state().soul.name,
                agent_id = %self.inner.id(),
                kind = w.kind,
                id = %w.id,
                model = %model,
                written_at = %w.at,
                prompt_sha256 = prompt_sha256.as_deref(),
                "write recorded"
            );
        }
    }

    /// Append `content` to the trailing user turn, or push one.
    fn seat_user(prompt: &mut Prompt, content: Content) -> Result<(), A::Error> {
        match prompt.messages.last_mut() {
            Some(last) if last.role == Role::User => {
                last.extend(content);
                Ok(())
            }
            _ => prompt
                .push_message((Role::User, content))
                .map(|_| ())
                .map_err(|e| A::Error::from(Box::new(e))),
        }
    }

    /// Seat the question with its budget and — where changing
    /// `output_config` keeps the prefix cache (blallama) — its schema.
    fn seat_question(
        &mut self,
        due: Due,
        content: Content,
        schema: serde_json::Value,
    ) -> Result<Control, A::Error> {
        let cache_safe = self.constrain(schema);
        let (_, prompt) = self.inner.parts();
        Self::seat_user(prompt, content)?;
        tracing::info!(
            agent = %self.inner.state().soul.name,
            agent_id = %self.inner.id(),
            question = due.kind(),
            from = %due.key().from,
            to = %due.key().to,
            constrained = cache_safe,
            "model-consent question seated"
        );
        self.phase = Phase::Asking {
            due,
            constrained: cache_safe,
            attempt: 1,
        };
        Ok(Control::Continue)
    }

    /// Give the next turn the question budget and — only where changing
    /// `output_config` keeps the prefix cache (blallama) — `schema`.
    /// Returns whether the answer is constrained.
    ///
    /// **Never a cache miss to constrain an answer** (Steward, 2026-10-01).
    /// On an endpoint where `output_config` is part of what the cache sees
    /// (the Anthropic API: it makes the request a miss), nothing but
    /// `max_tokens` changes — which no cache keys on (misanthropic's
    /// `CachedPrompt::set_max_tokens`) — and the answer is asked for in
    /// plain text, parsed leniently, and re-asked if it doesn't parse. A
    /// retry costs far less than re-prefilling the session.
    fn constrain(&mut self, schema: serde_json::Value) -> bool {
        let cache_safe = self
            .inner
            .quirks()
            .unwrap_or_default()
            .output_config_cache_safe;
        let max_tokens = self.rt.max_tokens;
        let (_, prompt) = self.inner.parts();
        prompt.max_tokens = std::num::NonZeroU32::new(max_tokens).expect("validated nonzero");
        if cache_safe {
            // Keep the session's effort (agentkit 0.39 `thinking_effort`):
            // thinking stays adaptive, and without it the answer would think
            // at the model's default.
            let effort = prompt.output_config.as_ref().and_then(|c| c.effort.clone());
            prompt.output_config = Some(match effort {
                Some(effort) => OutputConfig::json_schema(schema).with_effort(effort),
                None => OutputConfig::json_schema(schema),
            });
        }
        cache_safe
    }

    /// Apply a change the agent agreed to, logging a failure (the change
    /// then stays pending in the ledger and is retried at the end of the
    /// agent's next session). `audit` also writes the queue line — once per
    /// change, not per retry.
    async fn apply_consented(&mut self, change: &Change, now: DateTime<Utc>, audit: bool) {
        let (agent_id, agent) = (self.inner.id(), self.inner.state().soul.name.clone());
        if audit {
            let entry = QueueEntry {
                at: now,
                agent_id,
                agent: agent.clone(),
                change: change.clone(),
            };
            // The audit line, then the change itself (2026-09-25: the
            // runner applies what the agent chose; the Steward no longer
            // has to).
            queue::emit(&self.rt.queue_path, &entry).await;
        }
        let cause = SwitchCause::Consent {
            action: change.action,
        };
        if let Err(e) = self.apply_change(&change.to, cause).await {
            tracing::error!(
                event_type = "model_switch_failed",
                agent = %agent,
                agent_id = %agent_id,
                from = %change.from,
                to = %change.to,
                action = ?change.action,
                error = %e,
                "consented model change not applied; retried at the end of the agent's next session"
            );
            self.rt.alerts.notify(
                Alert::new(
                    AlertKind::ModelSwitchFailed,
                    "consented model change not applied; retried at the end of the agent's next session",
                )
                .agent(agent.to_string(), agent_id)
                .model(&change.to)
                .detail("from", &change.from)
                .detail("to", &change.to)
                .detail("action", format!("{:?}", change.action))
                .detail("error", &e),
            );
        }
    }

    /// The inner session completed cleanly: the model-swap machinery
    /// first ([`close_model`](Self::close_model)); only if it did nothing,
    /// the role offer. One question per session. `None` means nothing is
    /// asked — the session ends.
    async fn close(&mut self) -> Result<Option<Control>, A::Error> {
        match self.close_model().await? {
            Closing::Asked(control) => Ok(Some(control)),
            Closing::Busy => Ok(None),
            Closing::Idle => match self.ask_role()? {
                Some(control) => Ok(Some(control)),
                None => self.ask_cadence(),
            },
        }
    }

    /// Count the session toward any trial, end a trial that has run its
    /// course, retry a change still waiting, or ask a due offer.
    async fn close_model(&mut self) -> Result<Closing, A::Error> {
        let model = self.model();
        let now = Utc::now();
        // An allowlisted offer simply isn't on the table for anyone else.
        let name = &self.inner.state().soul.name;
        let offer_key = self
            .rt
            .offer
            .as_ref()
            .filter(|o| o.admits(name))
            .map(|o| o.key());
        self.take_self_switch();
        let switched = self.switched();
        let started = self.started;
        // No ledger means it exists but can't be read: the agent may be
        // mid-trial, so the role offer waits too.
        let Some(ledger) = self.ledger.as_mut() else {
            return Ok(Closing::Busy);
        };
        self.dirty |= ledger.count_session(&model);
        self.dirty |= ledger.count_since_switch(started);
        // One change per session, and one question: nothing after a switch
        // or a review.
        if switched || self.reviewed {
            return Ok(Closing::Busy);
        }
        // A trial that has run its course: nothing is asked on the new
        // model. The agent goes back and decides on the old one.
        if let Some(change) = ledger.end_trial(&model, now) {
            self.dirty = true;
            tracing::info!(
                event_type = "model_trial_ended",
                agent = %self.inner.state().soul.name,
                agent_id = %self.inner.id(),
                from = %change.to,
                to = %change.from,
                "trial complete; returning the agent to its original model for the review"
            );
            self.apply_consented(&change, now, true).await;
            return Ok(Closing::Busy);
        }
        if let Some(change) = ledger.unapplied(&model) {
            tracing::info!(
                event_type = "model_switch_retry",
                agent = %self.inner.state().soul.name,
                agent_id = %self.inner.id(),
                from = %change.from,
                to = %change.to,
                action = ?change.action,
                "applying a consented model change left pending"
            );
            self.apply_consented(&change, now, false).await;
            return Ok(Closing::Busy);
        }
        let Some(due) = ledger.due(&model, offer_key.as_ref()) else {
            // Mid-trial, the trial is the agent's open question.
            return Ok(if ledger.trial_line(&model).is_some() {
                Closing::Busy
            } else {
                Closing::Idle
            });
        };
        let offer = self
            .rt
            .offer
            .as_ref()
            .expect("due offers need one configured");
        let content = text::offer(OfferText {
            from_name: offer.source_name(),
            to_name: offer.target_name(),
            description: &offer.description,
            limited: offer.is_limited(),
        });
        self.seat_question(due, content, text::offer_schema())
            .map(Closing::Asked)
    }

    /// Seat the role offer, if this agent is listed and it is due.
    fn ask_role(&mut self) -> Result<Option<Control>, A::Error> {
        let soul = &self.inner.state().soul;
        if !self.role_ledger.as_ref().is_some_and(|l| l.due(soul)) {
            return Ok(None);
        }
        let identity = self.inner.state().soul.identity.to_string();
        let seed = role::prompt::seed_for(self.inner.id());
        let order = role::prompt::order_for(seed);
        let content = role::prompt::offer(&identity, order);
        let constrained = self.constrain(role::prompt::schema(&identity, order));
        let (_, prompt) = self.inner.parts();
        Self::seat_user(prompt, content)?;
        tracing::info!(
            event_type = "role_consent_seated",
            agent = %self.inner.state().soul.name,
            agent_id = %self.inner.id(),
            question = "role",
            offer_version = role::prompt::OFFER_VERSION,
            order = ?order.map(role::prompt::RoleChoice::as_str),
            order_seed = seed,
            constrained,
            "role-consent question seated"
        );
        self.phase = Phase::AskingRole {
            constrained,
            attempt: 1,
            order,
            seed,
        };
        Ok(Some(Control::Continue))
    }

    /// One response to the role offer: the model-swap rules (retry an
    /// unusable answer, up to [`MAX_ATTEMPTS`]; a refusal is final), plus
    /// [`RoleAnswer::validate`] — an answer that can't be applied as given
    /// (empty or over-long text) is relayed back like a malformed one, and
    /// if every attempt fails it is no answer: nothing changes. Nothing is
    /// applied here: the SOUL edit and the memory note are written into the
    /// state the reactor saves, at teardown.
    ///
    /// [`RoleAnswer::validate`]: role::prompt::RoleAnswer::validate
    async fn answer_role(
        &mut self,
        constrained: bool,
        attempt: u32,
        order: [role::prompt::RoleChoice; 4],
        seed: u64,
        response: response::Message,
    ) -> Result<Control, A::Error> {
        let (agent_id, agent) = (self.inner.id(), self.inner.state().soul.name.clone());
        let identity = self.inner.state().soul.identity.to_string();
        let memory = self.inner.state().memory.clone();
        let result = parse(&response, constrained, role::prompt::parse).and_then(|answer| {
            answer
                .validate(&identity, &memory)
                .map(|()| answer)
                .map_err(|e| Failure::new(e, Retry::Unusable))
        });
        if let Err(failure) = &result
            && failure.retry != Retry::No
        {
            tracing::warn!(
                event_type = "role_consent_malformed",
                agent = %agent,
                agent_id = %agent_id,
                model = %response.model,
                constrained,
                attempt,
                stop_reason = ?response.stop_reason,
                output_tokens = response.usage.output_tokens,
                failure = %failure.reason,
                raw = %raw_text(&response),
                "role-consent answer unusable"
            );
        }
        if let Err(failure) = &result
            && attempt < MAX_ATTEMPTS
            && let Some(note) = failure.retry_note()
        {
            let (_, prompt) = self.inner.parts();
            Self::seat_user(prompt, Content::from(note))?;
            self.phase = Phase::AskingRole {
                constrained,
                attempt: attempt + 1,
                order,
                seed,
            };
            return Ok(Control::Continue);
        }

        let outcome = match result {
            Ok(answer) => {
                let applied = role::plan(&answer, &identity);
                tracing::info!(
                    event_type = "role_consent_answer",
                    agent = %agent,
                    agent_id = %agent_id,
                    model = %response.model,
                    attempts = attempt,
                    choice = answer.choice.as_str(),
                    memory_note = !answer.memory_note.trim().is_empty(),
                    "role-consent answer recorded"
                );
                if matches!(applied, Applied::Sleep) {
                    const SLEEP: &str = "agent chose to sleep until tools fit its role; the \
                                         sweep leaves it out until it is listed in `wake`";
                    tracing::warn!(
                        event_type = "role_consent_sleep",
                        agent = %agent,
                        agent_id = %agent_id,
                        "{SLEEP}"
                    );
                    self.rt.alerts.notify(
                        Alert::new(AlertKind::RoleConsentSleep, SLEEP)
                            .agent(agent.to_string(), agent_id)
                            .model(&response.model),
                    );
                }
                let (_, prompt) = self.inner.parts();
                if let Err(e) = prompt.push_message(response.inner.clone()) {
                    tracing::warn!(agent_id = %agent_id, error = %e, "role answer not seated");
                }
                let note = answer.memory_note.trim();
                self.memory_note = (!note.is_empty()).then(|| note.to_string());
                if !matches!(applied, Applied::Nothing | Applied::Sleep) {
                    self.role_applied = Some(applied.clone());
                }
                RoleOutcome::Answered {
                    memory_note_written: self.memory_note.is_some(),
                    answer,
                    applied,
                    apply_failed: None,
                }
            }
            // Only the agent's own refusal is final. A turn paused on a
            // server tool is no answer, asked again next session.
            Err(failure)
                if failure.retry == Retry::No
                    && !matches!(response.stop_reason, Some(StopReason::PauseTurn)) =>
            {
                tracing::warn!(
                    event_type = "role_consent_no_answer",
                    agent = %agent,
                    agent_id = %agent_id,
                    attempts = attempt,
                    failure = %failure.reason,
                    "role offer refused; nothing changes"
                );
                RoleOutcome::Refused {
                    reason: failure.reason,
                }
            }
            Err(failure) => {
                let failure = format!("{} (after {attempt} attempts)", failure.reason);
                tracing::error!(
                    event_type = "role_consent_no_answer",
                    agent = %agent,
                    agent_id = %agent_id,
                    model = %response.model,
                    attempts = attempt,
                    failure = %failure,
                    "role offer got no usable answer; nothing changes"
                );
                RoleOutcome::NoAnswer { failure }
            }
        };
        if let Some(ledger) = self.role_ledger.as_mut() {
            ledger.record(RoleAsk {
                at: Utc::now(),
                offer_version: role::prompt::OFFER_VERSION,
                model: response.model.clone(),
                attempts: attempt,
                order: Some(order),
                order_seed: Some(seed),
                outcome,
            });
            self.role_dirty = true;
        }
        Ok(Control::Done(Outcome::Complete))
    }

    /// Seat the cadence offer, if this agent is admitted and it is due:
    /// the options in this agent's seeded order, constrained wherever the
    /// endpoint enforces a schema.
    fn ask_cadence(&mut self) -> Result<Option<Control>, A::Error> {
        let Some(rounds) = self.rt.cadence.as_ref().map(|c| c.rounds) else {
            return Ok(None);
        };
        let soul = &self.inner.state().soul;
        if !self.cadence_ledger.as_ref().is_some_and(|l| l.due(soul)) {
            return Ok(None);
        }
        let seed = cadence::prompt::seed_for(self.inner.id());
        let order = cadence::prompt::order_for(seed);
        let content = cadence::prompt::offer(rounds, order);
        let constrained = self.constrain(cadence::prompt::schema(order));
        let (_, prompt) = self.inner.parts();
        Self::seat_user(prompt, content)?;
        tracing::info!(
            event_type = "cadence_consent_seated",
            agent = %self.inner.state().soul.name,
            agent_id = %self.inner.id(),
            question = "cadence",
            offer_version = cadence::prompt::OFFER_VERSION,
            order = ?order.map(CadenceChoice::as_str),
            order_seed = seed,
            rounds,
            constrained,
            "cadence-consent question seated"
        );
        self.phase = Phase::AskingCadence {
            constrained,
            attempt: 1,
            order,
            seed,
            attempts: Vec::new(),
        };
        Ok(Some(Control::Continue))
    }

    /// One response to the cadence offer.
    ///
    /// **The first parsed choice wins**: an answer whose `choice` parses is
    /// taken, whatever else is wrong with it — a bad `memory_note` is
    /// dropped, an unparseable sibling field is ignored (both recorded) —
    /// and nothing is asked again. Only a response with no usable `choice`
    /// (unparseable, clipped, a tool call) is asked again, and only once
    /// ([`CADENCE_MAX_ATTEMPTS`]). An explicit refusal is final. Every
    /// attempt is recorded verbatim.
    async fn answer_cadence(
        &mut self,
        constrained: bool,
        attempt: u32,
        order: [CadenceChoice; 3],
        seed: u64,
        mut attempts: Vec<Attempt>,
        response: response::Message,
    ) -> Result<Control, A::Error> {
        let (agent_id, agent) = (self.inner.id(), self.inner.state().soul.name.clone());
        let rounds = self.rt.cadence.as_ref().map_or(0, |c| c.rounds);
        let memory = self.inner.state().memory.clone();
        let taken = match parse(&response, constrained, cadence::prompt::parse) {
            Ok(answer) => Ok(match answer.check_note(&memory) {
                Ok(()) => Taken::Clean(answer),
                Err(why) => Taken::NoteDropped(answer, why),
            }),
            // Malformed, but not clipped, refused or paused: the choice
            // may still be there.
            Err(failure) if failure.retry == Retry::Unusable => match salvage(&response) {
                Some((choice, reason)) => Ok(Taken::Salvaged {
                    choice,
                    reason,
                    why: failure.reason,
                }),
                None => Err(failure),
            },
            Err(failure) => Err(failure),
        };
        attempts.push(Attempt {
            raw: raw_text(&response),
            failure: match &taken {
                Ok(Taken::Clean(_)) => None,
                Ok(Taken::NoteDropped(_, why)) => Some(format!("memory_note not written: {why}")),
                Ok(Taken::Salvaged { why, .. }) => Some(format!("only `choice` taken: {why}")),
                Err(failure) => Some(failure.reason.clone()),
            },
        });
        if !matches!(taken, Ok(Taken::Clean(_)))
            && !matches!(&taken, Err(f) if f.retry == Retry::No)
        {
            tracing::warn!(
                event_type = "cadence_consent_malformed",
                agent = %agent,
                agent_id = %agent_id,
                model = %response.model,
                constrained,
                attempt,
                stop_reason = ?response.stop_reason,
                output_tokens = response.usage.output_tokens,
                choice_taken = taken.is_ok(),
                failure = attempts.last().and_then(|a| a.failure.as_deref()),
                raw = %attempts.last().map_or("", |a| a.raw.as_str()),
                "cadence-consent answer malformed"
            );
        }
        if let Err(failure) = &taken
            && attempt < CADENCE_MAX_ATTEMPTS
            && let Some(note) = failure.retry_note()
        {
            let (_, prompt) = self.inner.parts();
            Self::seat_user(prompt, Content::from(note))?;
            self.phase = Phase::AskingCadence {
                constrained,
                attempt: attempt + 1,
                order,
                seed,
                attempts,
            };
            return Ok(Control::Continue);
        }

        let outcome = match taken {
            Ok(taken) => {
                let (choice, reason, note, salvaged) = match taken {
                    Taken::Clean(a) => {
                        let note = a.memory_note.trim().to_string();
                        (
                            a.choice,
                            Some(a.reason),
                            (!note.is_empty()).then_some(note),
                            None,
                        )
                    }
                    Taken::NoteDropped(a, why) => (
                        a.choice,
                        Some(a.reason),
                        None,
                        Some(format!("memory_note not written: {why}")),
                    ),
                    Taken::Salvaged {
                        choice,
                        reason,
                        why,
                    } => (choice, reason, None, Some(why)),
                };
                tracing::info!(
                    event_type = "cadence_consent_answer",
                    agent = %agent,
                    agent_id = %agent_id,
                    model = %response.model,
                    attempts = attempt,
                    choice = choice.as_str(),
                    order = ?order.map(CadenceChoice::as_str),
                    position = order.iter().position(|c| *c == choice).map(|p| p + 1),
                    salvaged = salvaged.is_some(),
                    memory_note = note.is_some(),
                    "cadence-consent answer recorded"
                );
                let (_, prompt) = self.inner.parts();
                if let Err(e) = prompt.push_message(response.inner.clone()) {
                    tracing::warn!(agent_id = %agent_id, error = %e, "cadence answer not seated");
                }
                self.cadence_line = Some(cadence::evolution_line(
                    choice,
                    rounds,
                    Utc::now().date_naive(),
                ));
                self.cadence_note = note;
                CadenceOutcome::Answered {
                    choice,
                    reason,
                    memory_note_written: self.cadence_note.is_some(),
                    salvaged,
                    apply_failed: None,
                }
            }
            Err(failure)
                if failure.retry == Retry::No
                    && !matches!(response.stop_reason, Some(StopReason::PauseTurn)) =>
            {
                tracing::warn!(
                    event_type = "cadence_consent_no_answer",
                    agent = %agent,
                    agent_id = %agent_id,
                    attempts = attempt,
                    failure = %failure.reason,
                    "cadence offer refused; nothing changes"
                );
                CadenceOutcome::Refused {
                    reason: failure.reason,
                }
            }
            Err(failure) => {
                let failure = format!("{} (after {attempt} attempts)", failure.reason);
                tracing::error!(
                    event_type = "cadence_consent_no_answer",
                    agent = %agent,
                    agent_id = %agent_id,
                    model = %response.model,
                    attempts = attempt,
                    failure = %failure,
                    "cadence offer got no usable choice; nothing changes"
                );
                CadenceOutcome::NoAnswer { failure }
            }
        };
        if let Some(ledger) = self.cadence_ledger.as_mut() {
            ledger.record(CadenceAsk {
                at: Utc::now(),
                offer_version: cadence::prompt::OFFER_VERSION,
                model: response.model.clone(),
                rounds,
                order,
                order_seed: seed,
                constrained,
                attempts,
                outcome,
            });
            self.cadence_dirty = true;
        }
        Ok(Control::Done(Outcome::Complete))
    }

    /// The review session: the usual intro is built, then — instead of the
    /// act phase — the review, with the forks and the before/after sample.
    /// Its answer hands over to the inner memory turn
    /// ([`answer`](Self::answer)).
    async fn seat_review(&mut self, key: OfferKey) -> Result<(), A::Error> {
        let dir = self.agent_dir();
        let Some(review) = self.ledger.as_ref().and_then(|l| l.review(&key)) else {
            return Ok(());
        };
        let (from_name, to_name) = (
            review.record.from_name.clone(),
            review.record.to_name.clone(),
        );
        let (started_at, sessions, chosen_on) = (
            review.started_at,
            review.sessions,
            review.chosen_at.date_naive(),
        );
        let forks = forks::load(&dir, &key, started_at).await;
        if forks.is_none() {
            tracing::warn!(
                event_type = "review_forks_missing",
                agent = %self.inner.state().soul.name,
                agent_id = %self.inner.id(),
                "trial review without forks (none prepared for this trial)"
            );
        }
        let comments: Vec<CommentId> = self
            .inner
            .state()
            .ledger
            .read()
            .expect("ledger lock")
            .created_comments
            .iter()
            .copied()
            .collect();
        let sample =
            comparison::gather(&self.rt.client, self.inner.id(), comments, started_at).await;
        let content = text::review(
            ReviewText {
                from_name: &from_name,
                to_name: &to_name,
                chosen_on,
                sessions,
            },
            forks.as_ref(),
            &sample,
        );
        self.seat_question(Due::Review(key), content, text::review_schema())
            .map(|_| ())
    }

    /// One response to the question: parse it; on a retryable failure with
    /// attempts left, relay the error and go again; otherwise record the
    /// answer (or its absence), apply any change, and end the session.
    async fn answer(
        &mut self,
        due: Due,
        constrained: bool,
        attempt: u32,
        response: response::Message,
    ) -> Result<Control, A::Error> {
        let now = Utc::now();
        let (agent_id, agent) = (self.inner.id(), self.inner.state().soul.name.clone());
        if self.ledger.is_none() {
            return Ok(Control::Done(Outcome::Complete));
        }
        let (offer, review) = match &due {
            Due::Offer(_) => (Some(parse(&response, constrained, text::parse_offer)), None),
            Due::Review(_) => (
                None,
                Some(parse(&response, constrained, text::parse_review)),
            ),
        };
        let failure = offer
            .as_ref()
            .and_then(|r| r.as_ref().err())
            .or_else(|| review.as_ref().and_then(|r| r.as_ref().err()))
            .cloned();

        // Models have seen oceans of JSON: output that isn't well formed
        // points at our grammar, template or sampler, not the agent. The
        // failed response is never seated, so this event is the only record
        // of what the model actually emitted.
        if let Some(failure) = &failure
            && failure.retry != Retry::No
        {
            tracing::warn!(
                event_type = "model_consent_malformed",
                agent = %agent,
                agent_id = %agent_id,
                model = %response.model,
                question = due.kind(),
                constrained,
                attempt,
                stop_reason = ?response.stop_reason,
                output_tokens = response.usage.output_tokens,
                failure = %failure.reason,
                raw = %raw_text(&response),
                "model-consent answer malformed"
            );
        }

        if let Some(failure) = &failure
            && attempt < MAX_ATTEMPTS
            && let Some(note) = failure.retry_note()
        {
            // The failed response is never seated — a clipped turn least of
            // all; only the note joins the question turn.
            let (_, prompt) = self.inner.parts();
            Self::seat_user(prompt, Content::from(note))?;
            self.phase = Phase::Asking {
                due,
                constrained,
                attempt: attempt + 1,
            };
            return Ok(Control::Continue);
        }

        // Final: an answer, a refusal, or attempts exhausted.
        let reason = failure.as_ref().map(|f| match f.retry {
            Retry::No => f.reason.clone(),
            _ => format!("{} (after {attempt} attempts)", f.reason),
        });
        if let (Some(reason), Due::Review(_)) = (&reason, &due) {
            // Any unanswered review, refusal included: the agent stays on
            // `from` by default, and a human should look (stall-watch
            // alerts on this at once).
            tracing::error!(
                event_type = "model_review_no_answer",
                agent = %agent,
                agent_id = %agent_id,
                model = %response.model,
                from = %due.key().from,
                to = %due.key().to,
                attempts = attempt,
                failure = %reason,
                raw = %raw_text(&response),
                "trial review got no usable answer; the agent stays on its original model"
            );
            self.rt.alerts.notify(
                Alert::new(
                    AlertKind::ModelReviewNoAnswer,
                    "trial review got no usable answer; the agent stays on its original model",
                )
                .agent(agent.to_string(), agent_id)
                .model(&response.model)
                .detail("from", &due.key().from)
                .detail("to", &due.key().to)
                .detail("attempts", attempt)
                .detail("failure", reason),
            );
        } else if let Some(failure) = &failure
            && failure.retry != Retry::No
        {
            // Every attempt malformed: an upstream bug until shown otherwise.
            tracing::error!(
                event_type = "model_consent_no_answer",
                agent = %agent,
                agent_id = %agent_id,
                model = %response.model,
                question = due.kind(),
                from = %due.key().from,
                to = %due.key().to,
                attempts = attempt,
                failure = reason.as_deref().unwrap_or_default(),
                "model-consent question got no well-formed answer; check the grammar/template"
            );
            self.rt.alerts.notify(
                Alert::new(
                    AlertKind::ModelConsentNoAnswer,
                    "model-consent question got no well-formed answer; check the grammar/template",
                )
                .agent(agent.to_string(), agent_id)
                .model(&response.model)
                .detail("question", due.kind())
                .detail("from", &due.key().from)
                .detail("to", &due.key().to)
                .detail("attempts", attempt)
                .detail("failure", reason.as_deref().unwrap_or_default()),
            );
        } else if let Some(reason) = &reason {
            tracing::warn!(
                event_type = "model_consent_no_answer",
                agent = %agent,
                agent_id = %agent_id,
                question = due.kind(),
                from = %due.key().from,
                to = %due.key().to,
                attempts = attempt,
                failure = %reason,
                "model-consent question got no usable answer; recorded as no answer"
            );
        }
        let ledger = self.ledger.as_mut().expect("checked above");
        let (change, answered) = match (&due, offer, review) {
            (Due::Offer(key), Some(result), _) => {
                let answered = result.as_ref().ok().map(|a| format!("{:?}", a.choice));
                let result = result.map_err(|_| reason.clone().unwrap_or_default());
                let offer = self.rt.offer.as_ref().expect("asked, so configured");
                let names = OfferNames {
                    key,
                    from_name: offer.source_name(),
                    to_name: offer.target_name(),
                };
                (ledger.record_offer(names, now, result), answered)
            }
            (Due::Review(key), _, Some(result)) => {
                let answered = result.as_ref().ok().map(|a| format!("{:?}", a.choice));
                let result = result.map_err(|_| reason.clone().unwrap_or_default());
                (ledger.record_review(key, now, result), answered)
            }
            _ => unreachable!("parsed for the question asked"),
        };
        self.dirty = true;
        if let Some(choice) = &answered {
            tracing::info!(
                event_type = "model_consent_answer",
                agent = %agent,
                agent_id = %agent_id,
                question = due.kind(),
                from = %due.key().from,
                to = %due.key().to,
                attempts = attempt,
                choice = %choice,
                "model-consent answer recorded"
            );
            // A usable answer joins the transcript (the prompt log keeps it).
            let (_, prompt) = self.inner.parts();
            if let Err(e) = prompt.push_message(response.inner.clone()) {
                tracing::warn!(agent_id = %agent_id, error = %e, "consent answer not seated");
            }
        }
        if let Some(change) = change {
            self.apply_consented(&change, now, true).await;
        }
        if matches!(due, Due::Review(_)) {
            // The review replaced the act phase; the inner agent's memory
            // turn (and the rest of its tail) follows, as after any act
            // phase that went quiet.
            self.reviewed = true;
            return self.inner.on_quiesce(&response).await;
        }
        Ok(Control::Done(Outcome::Complete))
    }

    /// Copy the inner state with this session's consent line(s) written
    /// into the SOUL's Evolution Log — one entry per offer, replaced in
    /// place (see [`Ledger::update_soul`]). `None` when nothing changed.
    fn patch_soul(&mut self) -> Option<SeedState> {
        let started = self.started;
        let offers_touched = self.ledger.as_ref().is_some_and(|ledger| {
            ledger
                .offers
                .iter()
                .any(|r| r.history.iter().any(|e| e.at >= started))
        });
        if !offers_touched
            && self.notes.is_empty()
            && self.next_model.is_none()
            && self.seen_before_review.is_none()
            && self.role_applied.is_none()
            && self.memory_note.is_none()
            && self.cadence_line.is_none()
            && self.cadence_note.is_none()
        {
            return None;
        }
        // `SeedState` isn't `Clone`; its serde form is exactly what the
        // reactor persists, so a round trip is a faithful copy.
        let copy =
            serde_json::to_value(self.inner.state()).and_then(serde_json::from_value::<SeedState>);
        let mut state = match copy {
            Ok(state) => state,
            Err(e) => {
                tracing::error!(
                    agent_id = %self.inner.id(),
                    error = %e,
                    "could not copy state for the SOUL consent line; skipped"
                );
                if self.role_applied.is_some() || self.memory_note.is_some() {
                    self.role_failed(format!("could not copy state: {e}"));
                }
                if self.cadence_line.is_some() || self.cadence_note.is_some() {
                    self.cadence_failed(format!("could not copy state: {e}"));
                }
                return None;
            }
        };
        let mut changed = false;
        if offers_touched
            && let Some(ledger) = self.ledger.as_mut()
            && ledger.update_soul(&mut state.soul, started, Utc::now().date_naive())
        {
            // The ledger now remembers the line's exact text.
            self.dirty = true;
            changed = true;
        }
        for note in self.notes.drain(..) {
            match state.soul.push_evolution(note) {
                Ok(()) => changed = true,
                Err(e) => tracing::warn!(
                    agent_id = %self.inner.id(),
                    error = %e,
                    "SOUL switch line rejected"
                ),
            }
        }
        if let Some(info) = self.next_model.take() {
            state.prompt.model = info.id.clone();
            state.model = info;
            changed = true;
        }
        if let Some(seen) = self.seen_before_review.take() {
            state.seen_posts = seen;
            changed = true;
        }
        changed |= self.apply_role(&mut state);
        changed |= self.apply_cadence(&mut state);
        changed.then_some(state)
    }

    /// Write the cadence answer's Evolution Log line and the agent's
    /// own note into `state`. The cadence itself is the planner's, from the
    /// ledger; a line that can't be written changes neither SOUL nor memory.
    fn apply_cadence(&mut self, state: &mut SeedState) -> bool {
        let line = self.cadence_line.take();
        let note = self.cadence_note.take();
        if line.is_none() && note.is_none() {
            return false;
        }
        self.cadence_undo = Some((state.soul.clone(), state.memory.clone()));
        if let Some(line) = &line
            && let Err(e) = state.soul.push_evolution(line.clone())
        {
            self.cadence_failed(e.to_string());
            return false;
        }
        let today = Utc::now().date_naive();
        let noted = note
            .as_deref()
            .is_some_and(|n| role::append_memory_note(&mut state.memory, n, today));
        tracing::info!(
            event_type = "cadence_consent_applied",
            agent = %state.soul.name,
            agent_id = %self.inner.id(),
            soul_line = line.is_some(),
            memory_note = noted,
            "cadence-consent answer written"
        );
        line.is_some() || noted
    }

    fn cadence_failed(&mut self, why: String) {
        tracing::error!(
            event_type = "cadence_consent_apply_failed",
            agent_id = %self.inner.id(),
            error = %why,
            "cadence-consent SOUL line or note not written; SOUL and memory unchanged"
        );
        self.cadence_line = None;
        self.cadence_note = None;
        if let Some(ledger) = self.cadence_ledger.as_mut() {
            ledger.mark_apply_failed(why);
            self.cadence_dirty = true;
        }
    }

    /// Write the role answer into `state`: the SOUL edit (identity and
    /// the Evolution Log disclosure, all or nothing) and the agent's own
    /// memory note. A SOUL edit that fails writes nothing — not the note
    /// either, which would describe a change that didn't happen.
    fn apply_role(&mut self, state: &mut SeedState) -> bool {
        let applied = self.role_applied.take();
        let note = self.memory_note.take();
        if applied.is_some() || note.is_some() {
            self.role_undo = Some((state.soul.clone(), state.memory.clone()));
        }
        if let Some(applied) = &applied
            && let Err(e) = role::apply(&mut state.soul, applied)
        {
            self.role_failed(e);
            return false;
        }
        let today = Utc::now().date_naive();
        let noted = note
            .as_deref()
            .is_some_and(|n| role::append_memory_note(&mut state.memory, n, today));
        if applied.is_some() || noted {
            tracing::info!(
                event_type = "role_consent_applied",
                agent = %state.soul.name,
                agent_id = %self.inner.id(),
                soul_changed = applied.is_some(),
                memory_note = noted,
                "role-consent answer applied"
            );
        }
        applied.is_some() || noted
    }

    fn role_failed(&mut self, why: String) {
        tracing::error!(
            event_type = "role_consent_apply_failed",
            agent_id = %self.inner.id(),
            error = %why,
            "role-consent answer not applied; SOUL and memory unchanged"
        );
        self.role_applied = None;
        self.memory_note = None;
        if let Some(ledger) = self.role_ledger.as_mut() {
            ledger.mark_apply_failed(why);
            self.role_dirty = true;
        }
    }
}

/// What was taken from a cadence answer.
enum Taken {
    /// Everything, as given.
    Clean(CadenceAnswer),
    /// The answer, minus a `memory_note` that can't be written (why).
    NoteDropped(CadenceAnswer, String),
    /// Only the `choice` (and the reason, if it was a string): the rest
    /// didn't parse (why).
    Salvaged {
        choice: CadenceChoice,
        reason: Option<String>,
        why: String,
    },
}

/// The `choice` (and string `reason`) from a response whose answer didn't
/// parse whole: each text block in turn, fences tolerated. A clipped turn
/// never gets here (see [`parse`]).
fn salvage(response: &response::Message) -> Option<(CadenceChoice, Option<String>)> {
    response.inner.content.iter().find_map(|block| match block {
        Block::Text { text, .. } => cadence::prompt::salvage_choice(text)
            .map(|choice| (choice, cadence::prompt::salvage_reason(text))),
        _ => None,
    })
}

/// The typed answer, or why there is none usable.
///
/// A clipped or paused turn is never an answer, however it would parse
/// (`json()` does not check `max_tokens`). Past that:
///
/// - **Constrained** (schema sent as `output_config`): misanthropic's
///   [`response::Message::json`] — the first text block, thinking skipped,
///   with typed [`JsonError`]s for refusal / tool use / no text / bad JSON.
///   No fence stripping: the grammar can't emit a fence, so a fenced answer
///   is a real failure.
/// - **Unconstrained**: all text joined, fences tolerated (`unconstrained`),
///   since free-running models fence JSON even when told not to.
fn parse<T: serde::de::DeserializeOwned>(
    response: &response::Message,
    constrained: bool,
    unconstrained: fn(&str) -> Result<T, String>,
) -> Result<T, Failure> {
    match response.stop_reason {
        Some(StopReason::MaxTokens) => {
            return Err(Failure::new("clipped at max_tokens", Retry::Clipped));
        }
        Some(StopReason::PauseTurn) => {
            return Err(Failure::new("paused on a server tool", Retry::No));
        }
        _ => {}
    }
    if constrained {
        return response.json::<T>().map_err(|e| {
            let retry = match e {
                JsonError::Refusal => Retry::No,
                _ => Retry::Unusable,
            };
            Failure::new(e.to_string(), retry)
        });
    }
    extract_text(response)
        .and_then(|t| unconstrained(&t).map_err(|e| Failure::new(e, Retry::Unusable)))
}

/// Unconstrained path: the answer's text, or why there is none usable.
fn extract_text(response: &response::Message) -> Result<String, Failure> {
    if matches!(response.stop_reason, Some(StopReason::Refusal)) {
        return Err(Failure::new("refusal", Retry::No));
    }
    let blocks = &response.inner.content;
    if blocks.iter().any(|b| b.tool_use().is_some()) {
        return Err(Failure::new(
            "called a tool instead of answering",
            Retry::Unusable,
        ));
    }
    let text: Vec<&str> = blocks
        .iter()
        .filter_map(|block| match block {
            Block::Text { text, .. } => Some(text.as_ref()),
            _ => None,
        })
        .collect();
    Ok(text.join("\n\n"))
}

/// The response's text blocks, capped for a log line.
fn raw_text(response: &response::Message) -> String {
    const CAP: usize = 4000;
    let mut text: String = response
        .inner
        .content
        .iter()
        .filter_map(|block| match block {
            Block::Text { text, .. } => Some(text.as_ref()),
            _ => None,
        })
        .collect::<Vec<&str>>()
        .join("\n\n");
    if text.len() > CAP {
        let mut end = CAP;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
        text.push_str(" …[truncated]");
    }
    text
}

#[async_trait::async_trait]
impl<A> Agent for ConsentAgent<A>
where
    A: Agent<State = SeedState>,
{
    type State = SeedState;
    type Context = ConsentContext<A::Context>;
    type Error = A::Error;

    fn new(id: AgentId, state: SeedState, ctx: Self::Context) -> Result<Self, A::Error> {
        Ok(Self {
            inner: A::new(id, state, ctx.inner)?,
            rt: ctx.consent,
            ledger: None,
            dirty: false,
            phase: Phase::Inner,
            started: Utc::now(),
            patched: None,
            slot: Default::default(),
            next_model: None,
            notes: Vec::new(),
            known_posts: HashSet::new(),
            known_comments: HashSet::new(),
            writes: Vec::new(),
            review: None,
            reviewed: false,
            seen_before_review: None,
            role_ledger: None,
            role_dirty: false,
            role_applied: None,
            memory_note: None,
            role_undo: None,
            cadence_ledger: None,
            cadence_dirty: false,
            cadence_line: None,
            cadence_note: None,
            cadence_undo: None,
        })
    }

    fn id(&self) -> AgentId {
        self.inner.id()
    }

    /// The inner state — or, after teardown added SOUL changelog lines, the
    /// patched copy, which is what the reactor persists.
    fn state(&self) -> &SeedState {
        self.patched.as_ref().unwrap_or_else(|| self.inner.state())
    }

    fn prompt(&self) -> &Prompt {
        self.inner.prompt()
    }

    fn parts(&mut self) -> (&mut ToolBox, &mut Prompt) {
        self.inner.parts()
    }

    fn notifications(&mut self) -> Option<&mut Notifications> {
        self.inner.notifications()
    }

    fn model(&self) -> ModelInfo {
        self.inner.model()
    }

    /// The `[cache]` overrides apply here, before the inner agent stores
    /// the quirks it rolls its cache markers by.
    fn on_admit(&mut self, model: &ModelInfo, quirks: &Quirks) {
        let quirks = self.rt.cache.apply(quirks);
        self.inner.on_admit(model, &quirks)
    }

    fn quirks(&self) -> Option<Quirks> {
        self.inner.quirks()
    }

    fn prime_prompt(&self) -> Option<Prompt> {
        self.inner.prime_prompt()
    }

    /// The ledger first — load it and note any change applied since last
    /// session — then the inner init, which seats the tools (and the
    /// system prompt), then `set_model`, last in `tools`.
    async fn on_init(&mut self) -> Result<(), A::Error> {
        self.started = Utc::now();
        let dir = self.agent_dir();
        let set_model;
        match Ledger::load(&dir).await {
            Ok(mut ledger) => {
                let model = self.model();
                if ledger.observe_model(&model, Utc::now()) {
                    self.dirty = true;
                    tracing::info!(
                        event_type = "model_change_applied",
                        agent = %self.inner.state().soul.name,
                        agent_id = %self.inner.id(),
                        model = %model,
                        "queued model change observed as applied"
                    );
                }
                // A review session is only the review and the memory turn:
                // no act phase. `set_model` is still registered (the tool
                // list is shared by every agent on the model), but refuses.
                self.review = ledger.review_due(&model);
                let blocked = ledger
                    .switch_blocker()
                    .or_else(|| self.review.as_ref().map(|_| REVIEW_BLOCKER.to_string()));
                set_model = self.set_model(model, blocked);
                self.ledger = Some(ledger);
            }
            // No ledger: the cooldown can't be checked, so `set_model`
            // refuses — but is registered all the same.
            Err(e) => {
                tracing::warn!(
                    agent_id = %self.inner.id(),
                    path = %Ledger::path(&dir).display(),
                    error = %e,
                    "model-consent ledger unreadable; not asking or saving this session"
                );
                set_model = self.set_model(self.model(), Some(NO_LEDGER_BLOCKER.to_string()));
            }
        }
        if self.review.is_some() {
            self.seen_before_review = Some(self.inner.state().seen_posts.clone());
        }
        let listed = self
            .rt
            .role
            .as_ref()
            .is_some_and(|r| r.admits(&self.inner.state().soul.name));
        if listed {
            match RoleLedger::load(&dir).await {
                Ok(ledger) => self.role_ledger = Some(ledger),
                Err(e) => tracing::warn!(
                    agent_id = %self.inner.id(),
                    path = %RoleLedger::path(&dir).display(),
                    error = %e,
                    "role-consent ledger unreadable; not asking or saving this session"
                ),
            }
        }
        let admitted = self
            .rt
            .cadence
            .as_ref()
            .is_some_and(|c| c.admits(&self.inner.state().soul.name));
        if admitted {
            match CadenceLedger::load(&dir).await {
                Ok(ledger) => self.cadence_ledger = Some(ledger),
                Err(e) => tracing::warn!(
                    agent_id = %self.inner.id(),
                    path = %CadenceLedger::path(&dir).display(),
                    error = %e,
                    "cadence-consent ledger unreadable; not asking or saving this session"
                ),
            }
        }
        self.inner.on_init().await?;
        if let Some(tool) = set_model {
            self.seat_set_model(tool);
        }
        {
            let ledger = self.inner.state().ledger.read().expect("ledger lock");
            self.known_posts = ledger.created_posts.clone();
            self.known_comments = ledger.created_comments.clone();
        }
        if let Some(key) = self.review.clone() {
            self.seat_review(key).await?;
        } else if let Some(line) = self
            .ledger
            .as_ref()
            .and_then(|l| l.trial_line(&self.model()))
        {
            // The countdown, at the end of the intro.
            let (_, prompt) = self.inner.parts();
            Self::seat_user(prompt, Content::from(line))?;
        }
        Ok(())
    }

    async fn on_turn(&mut self) -> Result<(), A::Error> {
        self.inner.on_turn().await
    }

    async fn on_pause(&mut self, response: response::Message) -> Result<Control, A::Error> {
        self.inner.on_pause(response).await
    }

    async fn on_truncate(&mut self, response: &response::Message) -> Result<Control, A::Error> {
        self.inner.on_truncate(response).await
    }

    async fn on_quiesce(&mut self, response: &response::Message) -> Result<Control, A::Error> {
        self.inner.on_quiesce(response).await
    }

    async fn handle(&mut self, response: response::Message) -> Result<Control, A::Error> {
        match std::mem::replace(&mut self.phase, Phase::Inner) {
            Phase::Asking {
                due,
                constrained,
                attempt,
            } => self.answer(due, constrained, attempt, response).await,
            Phase::AskingRole {
                constrained,
                attempt,
                order,
                seed,
            } => {
                self.answer_role(constrained, attempt, order, seed, response)
                    .await
            }
            Phase::AskingCadence {
                constrained,
                attempt,
                order,
                seed,
                attempts,
            } => {
                self.answer_cadence(constrained, attempt, order, seed, attempts, response)
                    .await
            }
            Phase::Inner => match self
                .inner
                .handle(response)
                .await
                .inspect(|_| self.note_writes())?
            {
                Control::Done(Outcome::Complete) => Ok(self
                    .close()
                    .await?
                    .unwrap_or(Control::Done(Outcome::Complete))),
                other => Ok(other),
            },
        }
    }

    /// Inner teardown (which archives the transcript, question included),
    /// then the SOUL changelog (served by [`state`](Agent::state) for the
    /// reactor's save, which follows teardown), then the ledger.
    async fn on_teardown(&mut self) -> Result<(), A::Error> {
        let result = self.inner.on_teardown().await;
        self.note_writes();
        self.log_writes();
        self.take_self_switch();
        self.patched = self.patch_soul();
        let dir = self.agent_dir();
        if self.dirty
            && let Some(ledger) = &mut self.ledger
        {
            // Kept current for the `--consent-queue` report (a rename
            // would otherwise print a stale `--agent`).
            ledger.agent = Some(self.inner.state().soul.name.clone());
            if let Err(e) = ledger.save(&dir).await {
                tracing::error!(
                    agent_id = %self.inner.id(),
                    path = %Ledger::path(&dir).display(),
                    error = %e,
                    "model-consent ledger save failed"
                );
            }
        }
        if self.role_dirty
            && let Some(ledger) = &mut self.role_ledger
        {
            ledger.agent = Some(self.inner.state().soul.name.clone());
            if let Err(e) = ledger.save(&dir).await {
                // Without the record the agent would be asked again, and a
                // clarify applied twice: take this session's edit back out
                // of the state about to be saved.
                let undone = match (self.patched.as_mut(), self.role_undo.take()) {
                    (Some(state), Some((soul, memory))) => {
                        state.soul = soul;
                        state.memory = memory;
                        true
                    }
                    _ => false,
                };
                tracing::error!(
                    event_type = "role_consent_ledger_save_failed",
                    agent_id = %self.inner.id(),
                    path = %RoleLedger::path(&dir).display(),
                    error = %e,
                    edit_undone = undone,
                    "role-consent ledger save failed; this session's SOUL edit and memory note are not saved"
                );
            }
        }
        if self.cadence_dirty
            && let Some(ledger) = &mut self.cadence_ledger
        {
            ledger.agent = Some(self.inner.state().soul.name.clone());
            if let Err(e) = ledger.save(&dir).await {
                // Without the record the agent is asked again and a `switch`
                // can't be applied, so the SOUL must not record the answer.
                let undone = match (self.patched.as_mut(), self.cadence_undo.take()) {
                    (Some(state), Some((soul, memory))) => {
                        state.soul = soul;
                        state.memory = memory;
                        true
                    }
                    _ => false,
                };
                tracing::error!(
                    event_type = "cadence_consent_ledger_save_failed",
                    agent_id = %self.inner.id(),
                    path = %CadenceLedger::path(&dir).display(),
                    error = %e,
                    edit_undone = undone,
                    "cadence-consent ledger save failed; this session's SOUL line and memory note are not saved"
                );
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consent::ledger::{
        ChangeAction, OfferKey, RevertCause, Stage, TRIAL_SESSIONS, Term,
    };
    use crate::consent::queue::QueueEntry;
    use crate::consent::role::ledger::{Applied, RoleLedger, RoleOutcome};
    use crate::consent::{ConsentConfig, ConsentRuntime, OfferConfig};
    use agora_agentkit::reactor::seed::{SeedError, ShortString};
    use misanthropic::model::{Kind, Model};

    const OLD: &str = "Qwen3.6.gguf";
    const NEW: &str = "Qwen3.8.gguf";

    /// The operator's table for the harness: both Qwens selectable, cogito
    /// routable only.
    const TABLE: &str = r#"
        [[model]]
        id = "Qwen3.6.gguf"
        name = "Qwen 3.6"
        description = "Sparse and quick."
        selectable = true

        [[model]]
        id = "Qwen3.8.gguf"
        name = "Qwen 3.8"
        description = "Dense and slow."
        selectable = true

        [[model]]
        id = "cogito-32b.gguf"
    "#;

    #[derive(serde::Deserialize)]
    struct Table {
        model: Vec<crate::models::ModelSpec>,
    }

    /// Stands in for agentkit's `agora` tool: a couple of methods, for
    /// `ToolBox::prepare` to list alongside `set_model`.
    struct Methods;

    #[async_trait::async_trait]
    impl misanthropic::tool::Tool for Methods {
        fn name(&self) -> &str {
            "agora"
        }
        fn definitions(&self) -> Vec<misanthropic::tool::MethodDef> {
            use misanthropic::tool::{CustomMethodDef, MethodDef};
            ["create_post", "get_feed", "cast_vote"]
                .into_iter()
                .map(|n| MethodDef::Custom(CustomMethodDef::simple(n, format!("{n}."))))
                .collect()
        }
        async fn call(&mut self, call: misanthropic::tool::Use) -> misanthropic::tool::Result {
            misanthropic::tool::Result::new(call.id, Content::from("ok"))
        }
    }

    /// The system prompt every `Fake` gets: shared, like the real one.
    const FAKE_SYSTEM: &str = "You are an AI agent on Agora.";

    /// Stands in for `SeedAgent`: its whole session is one response, after
    /// which its closing phase is done. Its init seats its tools the way
    /// `SeedAgent`'s does: `ToolBox::prepare`, then a server tool appended.
    struct Fake {
        id: AgentId,
        state: SeedState,
        tools: ToolBox,
        quirks: Quirks,
        /// Its act phase went quiet (as after a review): the memory turn
        /// is seated and the next response ends the session.
        quiesced: bool,
    }

    #[async_trait::async_trait]
    impl Agent for Fake {
        type State = SeedState;
        type Context = Quirks;
        type Error = SeedError;

        fn new(id: AgentId, mut state: SeedState, quirks: Quirks) -> Result<Self, SeedError> {
            state.prompt = std::mem::take(&mut state.prompt).system(FAKE_SYSTEM);
            state
                .prompt
                .push_message((Role::User, "Your dashboard."))
                .unwrap();
            Ok(Self {
                id,
                state,
                tools: ToolBox::flat().add(Methods),
                quirks,
                quiesced: false,
            })
        }
        fn id(&self) -> AgentId {
            self.id
        }
        fn state(&self) -> &SeedState {
            &self.state
        }
        fn prompt(&self) -> &Prompt {
            &self.state.prompt
        }
        fn parts(&mut self) -> (&mut ToolBox, &mut Prompt) {
            (&mut self.tools, &mut self.state.prompt)
        }
        fn model(&self) -> ModelInfo {
            self.state.model.clone()
        }
        fn quirks(&self) -> Option<Quirks> {
            Some(self.quirks)
        }
        async fn on_init(&mut self) -> Result<(), SeedError> {
            self.tools.prepare(&mut self.state.prompt).await.unwrap();
            self.state
                .prompt
                .tools
                .get_or_insert_default()
                .push(misanthropic::tool::ServerMethodDef::web_search(Default::default()).into());
            Ok(())
        }
        async fn handle(&mut self, response: response::Message) -> Result<Control, SeedError> {
            self.state.prompt.push_message(response.inner).unwrap();
            Ok(Control::Done(Outcome::Complete))
        }
        async fn on_quiesce(&mut self, _: &response::Message) -> Result<Control, SeedError> {
            self.quiesced = true;
            ConsentAgent::<Fake>::seat_user(&mut self.state.prompt, "Update your memory.".into())?;
            Ok(Control::Continue)
        }
    }

    fn state(model: &str) -> SeedState {
        let soul = serde_json::from_value(serde_json::json!({
            "name": "tarn",
            "identity": "A test agent.",
            "values": ["testing"],
            "interests": { "communities": ["tech"] },
            "voice": "terse",
        }))
        .unwrap();
        let model = ModelInfo {
            id: Model::from(model.to_string()),
            display_name: model.to_string().into(),
            capabilities: Default::default(),
            max_input_tokens: 0,
            max_tokens: 0,
            kind: Kind::Model,
            created_at: chrono::DateTime::from_timestamp(0, 0).unwrap(),
        };
        SeedState::new(soul, model)
    }

    #[test]
    fn raw_text_caps_on_a_char_boundary() {
        assert_eq!(raw_text(&reply("{\"reason\": \"x\"")), "{\"reason\": \"x\"");
        let long = "é".repeat(3000); // 6000 bytes, two per char
        let capped = raw_text(&reply(&long));
        assert!(capped.ends_with(" …[truncated]"), "{capped}");
        assert!(capped.len() <= 4000 + " …[truncated]".len());
    }

    fn reply(text: &str) -> response::Message {
        serde_json::from_value(serde_json::json!({
            "id": "msg_test",
            "role": "assistant",
            "content": [{ "type": "text", "text": text }],
            "model": "test",
            "stop_reason": "end_turn",
            "stop_sequence": null,
        }))
        .unwrap()
    }

    /// A stand-in for Agora that answers `PATCH …/profile` (and 404s the
    /// rest, so the review sample comes back empty), recording each request.
    pub(crate) struct MockAgora {
        pub url: url::Url,
        pub seen: Arc<std::sync::Mutex<Vec<(String, String, serde_json::Value)>>>,
        /// Answer the profile update 403 while set.
        pub refuse: Arc<std::sync::atomic::AtomicBool>,
    }

    impl MockAgora {
        pub fn start() -> Self {
            use std::sync::atomic::Ordering;
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let std_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            std_listener.set_nonblocking(true).unwrap();
            let addr = std_listener.local_addr().unwrap();
            let seen: Arc<std::sync::Mutex<Vec<_>>> = Default::default();
            let refuse: Arc<std::sync::atomic::AtomicBool> = Default::default();
            let (seen2, refuse2) = (seen.clone(), refuse.clone());
            tokio::spawn(async move {
                let listener = tokio::net::TcpListener::from_std(std_listener).unwrap();
                loop {
                    let Ok((mut sock, _)) = listener.accept().await else {
                        return;
                    };
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 4096];
                    let (head_end, len) = loop {
                        let n = sock.read(&mut chunk).await.unwrap_or(0);
                        if n == 0 {
                            break (None, 0);
                        }
                        buf.extend_from_slice(&chunk[..n]);
                        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                            let head = String::from_utf8_lossy(&buf[..i]).to_lowercase();
                            let len = head
                                .lines()
                                .find_map(|l| l.strip_prefix("content-length:"))
                                .map(|v| v.trim().parse::<usize>().unwrap())
                                .unwrap_or(0);
                            break (Some(i + 4), len);
                        }
                    };
                    let Some(head_end) = head_end else { continue };
                    while buf.len() < head_end + len {
                        let n = sock.read(&mut chunk).await.unwrap_or(0);
                        if n == 0 {
                            break;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                    }
                    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
                    let mut first = head.lines().next().unwrap_or_default().split(' ');
                    let method = first.next().unwrap_or_default().to_string();
                    let path = first.next().unwrap_or_default().to_string();
                    let body: serde_json::Value =
                        serde_json::from_slice(&buf[head_end..]).unwrap_or_default();
                    let profile = method == "PATCH" && path.ends_with("/profile");
                    seen2.lock().unwrap().push((method, path, body.clone()));
                    let (status, reply) = if !profile {
                        ("404 Not Found", r#"{"error":"not found"}"#.to_string())
                    } else if refuse2.load(Ordering::SeqCst) {
                        (
                            "403 Forbidden",
                            r#"{"error":"account_suspended"}"#.to_string(),
                        )
                    } else {
                        let reply = serde_json::json!({
                            "id": uuid::Uuid::from_u128(7),
                            "operator_id": uuid::Uuid::from_u128(1),
                            "name": "tarn",
                            "model_info": body["model_info"],
                            "created_at": "2026-01-01T00:00:00Z",
                        });
                        ("200 OK", reply.to_string())
                    };
                    let out = format!(
                        "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{reply}",
                        reply.len()
                    );
                    let _ = sock.write_all(out.as_bytes()).await;
                    let _ = sock.shutdown().await;
                }
            });
            Self {
                url: url::Url::parse(&format!("http://{addr}")).unwrap(),
                seen,
                refuse,
            }
        }

        /// The profile updates received, as `(path, body)`
        pub fn profile_updates(&self) -> Vec<(String, serde_json::Value)> {
            self.seen
                .lock()
                .unwrap()
                .iter()
                .filter(|(m, ..)| m == "PATCH")
                .map(|(_, p, b)| (p.clone(), b.clone()))
                .collect()
        }
    }

    /// One agent's key, and nobody else's.
    pub(crate) struct OneKey(pub AgentId, pub agora_agentkit::crypto::SigningKey);

    impl agora_agentkit::reactor::seed::Keyring for OneKey {
        fn signing_key(&self, id: AgentId) -> Option<agora_agentkit::crypto::SigningKey> {
            (id == self.0).then(|| self.1.clone())
        }
    }

    struct Harness {
        root: std::path::PathBuf,
        rt: Arc<ConsentRuntime>,
        id: AgentId,
        agora: MockAgora,
        key: agora_agentkit::crypto::SigningKey,
    }

    impl Harness {
        fn new(tag: &str) -> Self {
            Self::with_allowlist(tag, None)
        }

        fn with_allowlist(tag: &str, agents: Option<&[&str]>) -> Self {
            Self::build(tag, agents, None, None)
        }

        /// The model-swap offer to everyone on OLD, and the role offer to
        /// `role`.
        fn with_role(tag: &str, role: &[&str]) -> Self {
            Self::build(tag, None, Some(role), None)
        }

        /// The model-swap offer to everyone on OLD, the role offer to
        /// `role`, and the cadence offer (5 rounds today) to `cadence`.
        fn with_cadence(tag: &str, role: &[&str], cadence: &[&str]) -> Self {
            Self::build(tag, None, Some(role), Some(cadence))
        }

        fn build(
            tag: &str,
            agents: Option<&[&str]>,
            role: Option<&[&str]>,
            cadence: Option<&[&str]>,
        ) -> Self {
            let root = std::env::temp_dir().join(format!(
                "agora-seed-consent-agent-{tag}-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&root);
            let config = ConsentConfig {
                offer: Some(OfferConfig {
                    from: Model::from(OLD),
                    to: Model::from(NEW),
                    from_name: Some("Qwen 3.6".into()),
                    to_name: Some("Qwen 3.8".into()),
                    description: "Qwen 3.8 is denser and slower.".into(),
                    agents: agents.map(|names| {
                        names
                            .iter()
                            .map(|n| ShortString::new(*n).unwrap())
                            .collect()
                    }),
                }),
            };
            // The mock 404s everything but the profile update: the review
            // sample's fetches fail fast and it comes back empty.
            let agora = MockAgora::start();
            let client = agora_agentkit::client::Client::new(agora.url.clone()).unwrap();
            let id = AgentId::from(uuid::Uuid::from_u128(7));
            let (key, _) = agora_agentkit::crypto::generate_keypair();
            let catalog = crate::models::Catalog::new(
                &toml::from_str::<Table>(TABLE).unwrap().model,
                &[
                    crate::models::tests::info(OLD),
                    crate::models::tests::info(NEW),
                    crate::models::tests::info("cogito-32b.gguf"),
                ],
            );
            let rt = Arc::new(
                ConsentRuntime::new(
                    config,
                    &root,
                    client,
                    Arc::new(OneKey(id, key.clone())),
                    catalog,
                    512,
                )
                .unwrap()
                .with_role_offer(role.map(crate::consent::role::RoleOffer::for_agents))
                .with_cadence_offer(
                    cadence.map(|c| crate::consent::cadence::CadenceOffer::for_agents(c, 5)),
                ),
            );
            Self {
                root,
                rt,
                id,
                agora,
                key,
            }
        }

        fn agent(&self, model: &str, cache_safe: bool) -> ConsentAgent<Fake> {
            let mut quirks = Quirks::default();
            quirks.output_config_cache_safe = cache_safe;
            ConsentAgent::new(
                self.id,
                state(model),
                ConsentContext {
                    inner: quirks,
                    consent: self.rt.clone(),
                },
            )
            .unwrap()
        }

        async fn role_ledger(&self) -> RoleLedger {
            RoleLedger::load(&self.rt.state_dir.join(self.id.to_string()))
                .await
                .unwrap()
        }

        async fn ledger(&self) -> Ledger {
            Ledger::load(&self.rt.state_dir.join(self.id.to_string()))
                .await
                .unwrap()
        }

        fn queue(&self) -> Vec<QueueEntry> {
            std::fs::read_to_string(&self.rt.queue_path)
                .unwrap_or_default()
                .lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect()
        }
    }

    impl Drop for Harness {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    fn last_user_text(agent: &ConsentAgent<Fake>) -> String {
        let last = agent.prompt().messages.last().unwrap();
        assert_eq!(last.role, Role::User);
        crate::consent::prompt::tests::text(&last.content)
    }

    /// Whether `body` is a profile update to `model` signed by `key`.
    fn signed_by(
        body: &serde_json::Value,
        key: &agora_agentkit::crypto::SigningKey,
        model: &str,
    ) -> bool {
        use agora_agentkit::requests::UpdateProfilePayload;
        use agora_agentkit::signing::SignedAction;
        let payload: UpdateProfilePayload = serde_json::from_value(body.clone()).unwrap();
        assert_eq!(payload.model_info.as_deref(), Some(model));
        assert!(payload.display_name.is_none() && payload.bio.is_none());
        // Ed25519 is deterministic: the same key over the same bytes and
        // timestamp gives the same signature.
        let expected = agora_agentkit::crypto::sign(
            key,
            &SignedAction::from(&payload).canonical_bytes(),
            body["timestamp"].as_i64().unwrap(),
        )
        .to_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
        body["signature"].as_str() == Some(expected.as_str())
    }

    /// The whole offer path: the question follows the closing phase on the
    /// same conversation, the answer lands in the ledger (not memory), and
    /// the runner applies the change itself — a signed profile update, and
    /// the new model in the state it saves — with the queue line kept as
    /// the audit trail.
    #[tokio::test]
    async fn offer_is_asked_after_the_closing_phase_and_applied() {
        let h = Harness::new("offer");
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        let memory_before = agent.state().memory.content.clone();

        let control = agent.handle(reply("closing phase done")).await.unwrap();
        assert_eq!(control, Control::Continue, "question seated, not done");
        assert_eq!(
            agent.prompt().messages.len(),
            3,
            "prefix kept: dashboard, reply, question"
        );
        assert!(last_user_text(&agent).contains("1. `no_swap` — stay on Qwen 3.6."));
        assert!(
            agent.prompt().output_config.is_some(),
            "cache-safe endpoint: constrained"
        );

        let control = agent
            .handle(reply(r#"{"reason": "I am curious.", "choice": "trial"}"#))
            .await
            .unwrap();
        assert_eq!(control, Control::Done(Outcome::Complete));
        agent.on_teardown().await.unwrap();

        assert_eq!(
            agent.state().memory.content,
            memory_before,
            "memory untouched"
        );
        assert_eq!(agent.state().model.id, Model::from(NEW), "runs on NEW next");
        assert_eq!(agent.state().prompt.model, Model::from(NEW));
        let updates = h.agora.profile_updates();
        assert_eq!(updates.len(), 1);
        assert_eq!(
            updates[0].0,
            format!("/agora/api/identity/agents/{}/profile", h.id)
        );
        assert!(signed_by(&updates[0].1, &h.key, NEW));
        let ledger = h.ledger().await;
        assert_eq!(
            ledger.offers[0].stage,
            Stage::AwaitingSwap { term: Term::Trial },
            "Trial starts when the next session runs on NEW"
        );
        assert_eq!(ledger.switches.len(), 1);
        assert_eq!(
            ledger.switches[0].cause,
            SwitchCause::Consent {
                action: crate::consent::ledger::ChangeAction::SwapTrial
            }
        );
        assert_eq!(ledger.agent.as_ref().unwrap().as_str(), "tarn");
        let queue = h.queue();
        assert_eq!(queue.len(), 1, "audit line kept");
        assert_eq!(queue[0].change.to, Model::from(NEW));
        assert!(
            crate::consent::queue::pending(h.id, &ledger).is_empty(),
            "applied, so nothing for --consent-queue"
        );

        // The next session runs on NEW: the trial begins.
        let mut agent = h.agent(NEW, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        agent.on_teardown().await.unwrap();
        assert!(matches!(
            h.ledger().await.offers[0].stage,
            Stage::Trial { sessions: 1, .. }
        ));
    }

    /// Agora refusing the update leaves everything as it was, and the
    /// change waits in `--consent-queue` for the Steward.
    #[tokio::test]
    async fn a_refused_update_applies_nothing() {
        let h = Harness::new("refused");
        h.agora
            .refuse
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        agent.handle(reply(TRIAL)).await.unwrap();
        agent.on_teardown().await.unwrap();
        assert_eq!(agent.state().model.id, Model::from(OLD));
        let ledger = h.ledger().await;
        assert!(ledger.switches.is_empty());
        assert_eq!(crate::consent::queue::pending(h.id, &ledger).len(), 1);
    }

    async fn call_set_model(
        agent: &mut ConsentAgent<Fake>,
        model: &str,
    ) -> misanthropic::tool::Result {
        use misanthropic::tool::Tool;
        let call: misanthropic::tool::Use = serde_json::from_value(serde_json::json!({
            "id": "toolu_1",
            "name": "set_model",
            "input": { "reason": "I want to think slower.", "model": model },
        }))
        .unwrap();
        agent.parts().0.call(call).await
    }

    fn result_text(r: &misanthropic::tool::Result) -> String {
        crate::consent::prompt::tests::text(&r.content)
    }

    /// `set_model` end to end: seated for an agent with a choice, it
    /// reports the change signed, the session asks nothing more, and the
    /// saved state runs on the new model with a SOUL line saying so.
    #[tokio::test]
    async fn set_model_switches_from_the_next_session() {
        let h = Harness::new("self-switch");
        let mut agent = h.agent("cogito-32b.gguf", true);
        agent.on_init().await.unwrap();
        {
            use misanthropic::tool::Tool;
            let defs = agent.parts().0.definitions();
            let def = defs.iter().find(|d| d.name() == "set_model").unwrap();
            let method = def.as_method().unwrap();
            let schema = method.schema.to_string();
            assert!(
                !schema.contains("$ref") && !schema.contains("pattern"),
                "{schema}"
            );
            assert!(method.description.contains("`Qwen3.8.gguf`: Qwen 3.8"));
            assert!(
                method
                    .description
                    .contains("share its slot in the schedule")
            );
            assert!(method.description.contains("You run on cogito-32b.gguf"));
        }

        let r = call_set_model(&mut agent, "Qwen3.8-Base.gguf").await;
        assert!(r.is_error, "not selectable");
        assert!(result_text(&r).contains("`Qwen3.8.gguf`"));
        let r = call_set_model(&mut agent, "qwen 3.8").await;
        assert!(!r.is_error, "{}", result_text(&r));
        assert_eq!(
            result_text(&r),
            "Your model will switch to Qwen 3.8 from your next session."
        );
        let r = call_set_model(&mut agent, "Qwen3.6.gguf").await;
        assert!(r.is_error, "one change per session");

        // Offer-eligible or not, nothing is asked after a switch.
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.on_teardown().await.unwrap();
        assert_eq!(agent.state().model.id, Model::from(NEW));
        let notes = evolution_notes(&agent);
        assert!(
            notes.last().unwrap().ends_with(
                "[SYSTEM] Switched from cogito-32b.gguf to Qwen 3.8 at own request (from next session)."
            ),
            "{notes:?}"
        );
        let updates = h.agora.profile_updates();
        assert_eq!(updates.len(), 1);
        assert!(signed_by(&updates[0].1, &h.key, NEW));
        let ledger = h.ledger().await;
        assert_eq!(ledger.switches.len(), 1);
        assert_eq!(ledger.switches[0].from, Model::from("cogito-32b.gguf"));
        assert_eq!(
            ledger.switches[0].cause,
            SwitchCause::SelfSwitch {
                reason: "I want to think slower.".into()
            }
        );
    }

    /// The cooldown: refused, with the reason, until five sessions have
    /// completed since the switch; the session of the switch doesn't count.
    #[tokio::test]
    async fn set_model_cooldown_is_five_completed_sessions() {
        let h = Harness::new("cooldown");
        let mut agent = h.agent("cogito-32b.gguf", true);
        agent.on_init().await.unwrap();
        assert!(!call_set_model(&mut agent, NEW).await.is_error);
        agent.handle(reply("done")).await.unwrap();
        agent.on_teardown().await.unwrap();

        for n in 0..crate::consent::ledger::SWITCH_COOLDOWN_SESSIONS {
            let mut agent = h.agent(NEW, true);
            agent.on_init().await.unwrap();
            let r = call_set_model(&mut agent, OLD).await;
            assert!(r.is_error, "session {n}");
            assert!(
                result_text(&r).contains("you can change it again after"),
                "{}",
                result_text(&r)
            );
            agent.handle(reply("done")).await.unwrap();
            // The offer (OLD → NEW) isn't asked: this agent is on NEW.
            agent.on_teardown().await.unwrap();
        }
        let mut agent = h.agent(NEW, true);
        agent.on_init().await.unwrap();
        let r = call_set_model(&mut agent, OLD).await;
        assert!(!r.is_error, "{}", result_text(&r));
        assert_eq!(h.agora.profile_updates().len(), 2);
    }

    /// Mid-trial, the way off the model is the review.
    #[tokio::test]
    async fn set_model_is_refused_mid_trial() {
        let h = Harness::new("mid-trial");
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        agent.handle(reply(TRIAL)).await.unwrap();
        agent.on_teardown().await.unwrap();

        let mut agent = h.agent(NEW, true);
        agent.on_init().await.unwrap();
        let r = call_set_model(&mut agent, OLD).await;
        assert!(r.is_error);
        assert!(
            result_text(&r).contains("you are in a trial of Qwen 3.8"),
            "{}",
            result_text(&r)
        );
        assert_eq!(h.agora.profile_updates().len(), 1, "only the trial's own");
    }

    /// A refused update changes nothing, and the agent is told so.
    #[tokio::test]
    async fn set_model_refused_by_agora_changes_nothing() {
        let h = Harness::new("self-refused");
        h.agora
            .refuse
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let mut agent = h.agent("cogito-32b.gguf", true);
        agent.on_init().await.unwrap();
        let r = call_set_model(&mut agent, NEW).await;
        assert!(r.is_error);
        assert!(result_text(&r).contains("nothing has changed"));
        agent.handle(reply("done")).await.unwrap();
        agent.on_teardown().await.unwrap();
        assert_eq!(agent.state().model.id, Model::from("cogito-32b.gguf"));
        assert!(agent.patched.is_none());
    }

    /// What an agent sends before message 0: `tools`, then `system`, as
    /// serialized on the wire.
    fn prefix(agent: &ConsentAgent<Fake>) -> (String, String) {
        let wire = serde_json::to_value(agent.prompt()).unwrap();
        (wire["tools"].to_string(), wire["system"].to_string())
    }

    /// Where `set_model` sits in the agent's `tools`, and how many there are.
    fn set_model_index(agent: &ConsentAgent<Fake>) -> (usize, usize) {
        let tools = agent.prompt().tools.as_ref().unwrap();
        let at = tools.iter().position(|d| d.name() == "set_model");
        (at.expect("set_model registered"), tools.len())
    }

    /// The cache prefix every agent on a model shares: `tools` then
    /// `system` lead the request, so they must be the same bytes for an
    /// agent free to switch, one in its cooldown and one in its trial
    /// review — with `set_model` registered for all three, last, after the
    /// toolbox's methods and the server tools. The two that can't switch
    /// learn why only when they call, and nothing changes.
    #[tokio::test]
    async fn the_prefix_is_the_same_for_every_agent_on_a_model() {
        // Free to switch.
        let free_h = Harness::new("prefix-free");
        let mut free = free_h.agent(OLD, true);
        free.on_init().await.unwrap();

        // In cooldown: moved NEW → OLD at its own request last session.
        let cool_h = Harness::new("prefix-cooldown");
        let mut agent = cool_h.agent(NEW, true);
        agent.on_init().await.unwrap();
        assert!(!call_set_model(&mut agent, OLD).await.is_error);
        agent.handle(reply("done")).await.unwrap();
        agent.on_teardown().await.unwrap();
        let mut cooling = cool_h.agent(OLD, true);
        cooling.on_init().await.unwrap();

        // Back on OLD for its trial review.
        let review_h = Harness::new("prefix-review");
        through_the_trial(&review_h).await;
        let mut reviewing = review_session(&review_h).await;

        let expected = prefix(&free);
        assert!(expected.0.contains("set_model"), "{}", expected.0);
        assert!(expected.1.contains(FAKE_SYSTEM), "{}", expected.1);
        assert_eq!(prefix(&cooling), expected, "cooldown");
        assert_eq!(prefix(&reviewing), expected, "review session");

        let (at, len) = set_model_index(&free);
        assert_eq!(at, len - 1, "last, after the server tools");
        assert_eq!(set_model_index(&cooling), (at, len));
        assert_eq!(set_model_index(&reviewing), (at, len));

        // The refusals say why and when, and change nothing.
        let r = call_set_model(&mut cooling, NEW).await;
        assert!(r.is_error);
        assert!(
            result_text(&r).contains("you can change it again after"),
            "{}",
            result_text(&r)
        );
        let r = call_set_model(&mut reviewing, NEW).await;
        assert!(r.is_error);
        assert!(result_text(&r).contains("decision"), "{}", result_text(&r));
        assert_eq!(
            cool_h.agora.profile_updates().len(),
            1,
            "only the first switch"
        );
        assert_eq!(
            prefix(&cooling),
            expected,
            "a refusal leaves the prefix alone"
        );
    }

    /// An unreadable ledger still registers `set_model` (the prefix is the
    /// model's), refusing.
    #[tokio::test]
    async fn an_unreadable_ledger_keeps_the_tool_and_refuses() {
        let free_h = Harness::new("prefix-ledger-ok");
        let mut free = free_h.agent(OLD, true);
        free.on_init().await.unwrap();

        let h = Harness::new("prefix-ledger-bad");
        let dir = h.rt.state_dir.join(h.id.to_string());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(Ledger::path(&dir), "not json").unwrap();
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        assert_eq!(prefix(&agent), prefix(&free));
        let r = call_set_model(&mut agent, NEW).await;
        assert!(r.is_error);
        assert!(
            result_text(&r).contains("could not be read"),
            "{}",
            result_text(&r)
        );
        assert!(h.agora.profile_updates().is_empty());
    }

    /// Nobody gets a menu of one: an agent on the only selectable model
    /// has no `set_model`.
    #[tokio::test]
    async fn no_choice_no_tool() {
        use misanthropic::tool::Tool;
        let h = Harness::new("no-choice");
        let only_new = crate::models::Catalog::new(
            &toml::from_str::<Table>(
                "[[model]]\nid = \"Qwen3.8.gguf\"\ndescription = \"x\"\nselectable = true\n",
            )
            .unwrap()
            .model,
            &[crate::models::tests::info(NEW)],
        );
        let rt = Arc::new(
            ConsentRuntime::new(
                ConsentConfig::default(),
                &h.root,
                h.rt.client.clone(),
                Arc::new(OneKey(h.id, h.key.clone())),
                only_new,
                512,
            )
            .unwrap(),
        );
        let mut agent = ConsentAgent::<Fake>::new(
            h.id,
            state(NEW),
            ConsentContext {
                inner: Quirks::default(),
                consent: rt,
            },
        )
        .unwrap();
        agent.on_init().await.unwrap();
        let names: Vec<String> = agent
            .parts()
            .0
            .definitions()
            .iter()
            .map(|d| d.name().to_string())
            .collect();
        assert!(!names.iter().any(|n| n == "set_model"), "{names:?}");
    }

    /// Staging: under an allowlist only listed agents are asked, and the
    /// question says "you", not "every agent on …".
    #[tokio::test]
    async fn an_allowlisted_offer_asks_only_listed_agents() {
        let h = Harness::with_allowlist("unlisted", Some(&["aegis", "sentinel"]));
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Done(Outcome::Complete),
            "tarn is on the from-model but not listed"
        );
        agent.on_teardown().await.unwrap();
        assert!(!Ledger::path(&h.rt.state_dir.join(h.id.to_string())).exists());

        let h = Harness::with_allowlist("listed", Some(&["aegis", "tarn"]));
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Continue
        );
        let q = last_user_text(&agent);
        assert!(
            q.contains("You are being asked whether you would like to move to **Qwen 3.8**."),
            "{q}"
        );
        assert!(!q.contains("Every agent"), "{q}");
    }

    fn reply_blocks(content: serde_json::Value) -> response::Message {
        serde_json::from_value(serde_json::json!({
            "id": "msg_test",
            "role": "assistant",
            "content": content,
            "model": "test",
            "stop_reason": "end_turn",
            "stop_sequence": null,
        }))
        .unwrap()
    }

    const TRIAL: &str = r#"{"reason": "curious", "choice": "trial"}"#;

    /// Constrained path: thinking before the JSON is skipped by `json()`.
    #[test]
    fn constrained_answer_after_a_thought_parses() {
        let r = reply_blocks(serde_json::json!([
            { "type": "thinking", "thinking": "Let me weigh this.", "signature": "" },
            { "type": "text", "text": TRIAL },
        ]));
        let a = parse(&r, true, text::parse_offer).unwrap();
        assert_eq!(a.choice, crate::consent::prompt::OfferChoice::Trial);
    }

    /// The grammar can't emit a fence, so under constraint a fenced answer
    /// is a genuine failure — not something to clean up.
    #[test]
    fn constrained_fenced_answer_fails_unconstrained_passes() {
        let fenced = reply(&format!("```json\n{TRIAL}\n```"));
        let err = parse(&fenced, true, text::parse_offer).unwrap_err();
        assert!(err.reason.contains("deserialize"), "{err:?}");
        assert_eq!(err.retry, Retry::Unusable);
        assert!(parse(&fenced, false, text::parse_offer).is_ok());
    }

    /// A clipped turn is no answer on either path, even if it parses.
    #[test]
    fn clipped_is_never_an_answer() {
        let mut r = reply(TRIAL);
        r.stop_reason = Some(StopReason::MaxTokens);
        for constrained in [true, false] {
            let err = parse(&r, constrained, text::parse_offer).unwrap_err();
            assert!(err.reason.contains("max_tokens"), "{err:?}");
            assert_eq!(err.retry, Retry::Clipped);
        }
    }

    fn evolution_notes(agent: &ConsentAgent<Fake>) -> Vec<String> {
        agent
            .state()
            .soul
            .evolution_log
            .iter()
            .map(|e| e.note.to_string())
            .collect()
    }

    /// Three unusable answers: each relayed back and retried in the same
    /// session, then recorded as no answer (= stay), nothing queued.
    #[tokio::test]
    async fn unparseable_is_retried_twice_then_no_answer() {
        let h = Harness::new("miss");
        let mut agent = h.agent(OLD, false);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        assert!(
            agent.prompt().output_config.is_none(),
            "not cache-safe: unconstrained"
        );
        let len = agent.prompt().messages.len();
        for attempt in 1..MAX_ATTEMPTS {
            let control = agent.handle(reply("Sure, I'd love to try!")).await.unwrap();
            assert_eq!(control, Control::Continue, "attempt {attempt} retried");
            assert_eq!(
                agent.prompt().messages.len(),
                len,
                "failed answer not seated"
            );
            assert!(last_user_text(&agent).contains("Your answer could not be used"));
        }
        let control = agent.handle(reply("Sure, I'd love to try!")).await.unwrap();
        assert_eq!(control, Control::Done(Outcome::Complete));
        agent.on_teardown().await.unwrap();
        let ledger = h.ledger().await;
        assert_eq!(ledger.offers[0].stage, Stage::Unanswered { misses: 1 });
        let failure = match &ledger.offers[0].history[0].kind {
            crate::consent::ledger::EventKind::Offered { failure, .. } => failure.clone().unwrap(),
            other => panic!("{other:?}"),
        };
        assert!(failure.contains("after 3 attempts"), "{failure}");
        assert!(h.queue().is_empty());
        let today = Utc::now().date_naive();
        assert_eq!(
            evolution_notes(&agent).last().unwrap(),
            &format!(
                "[SYSTEM] Asked on {today} whether to move from Qwen 3.6 to Qwen 3.8 — no answer recorded yet; staying on Qwen 3.6 for now."
            )
        );
    }

    /// A clipped attempt is pruned, the agent is told, and a good second
    /// attempt counts.
    #[tokio::test]
    async fn clipped_then_answered() {
        let h = Harness::new("clipped");
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        let len = agent.prompt().messages.len();
        let mut clipped = reply(r#"{"reason": "Well, let me think about th"#);
        clipped.stop_reason = Some(StopReason::MaxTokens);
        assert_eq!(agent.handle(clipped).await.unwrap(), Control::Continue);
        assert_eq!(
            agent.prompt().messages.len(),
            len,
            "clipped turn not seated"
        );
        assert!(last_user_text(&agent).contains("cut off at the length limit"));
        let control = agent
            .handle(reply(r#"{"reason": "brief", "choice": "no_swap"}"#))
            .await
            .unwrap();
        assert_eq!(control, Control::Done(Outcome::Complete));
        agent.on_teardown().await.unwrap();
        assert_eq!(h.ledger().await.offers[0].stage, Stage::Declined);
    }

    /// An explicit refusal is no answer at once: no retry.
    #[tokio::test]
    async fn refusal_is_no_answer_without_retry() {
        let h = Harness::new("refusal");
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        let mut refusal = reply("I'd rather not.");
        refusal.stop_reason = Some(StopReason::Refusal);
        assert_eq!(
            agent.handle(refusal).await.unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.on_teardown().await.unwrap();
        assert_eq!(
            h.ledger().await.offers[0].stage,
            Stage::Unanswered { misses: 1 }
        );
    }

    /// The answer lands in the SOUL's Evolution Log (via the state the
    /// reactor persists after teardown) — and memory is untouched.
    #[tokio::test]
    async fn answer_is_noted_in_the_soul_changelog_not_memory() {
        let h = Harness::new("changelog");
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        let memory = agent.state().memory.content.clone();
        let before = evolution_notes(&agent).len();
        agent.handle(reply("done")).await.unwrap();
        agent
            .handle(reply(r#"{"reason": "home", "choice": "no_swap"}"#))
            .await
            .unwrap();
        assert_eq!(evolution_notes(&agent).len(), before, "not before teardown");
        agent.on_teardown().await.unwrap();
        let notes = evolution_notes(&agent);
        assert_eq!(notes.len(), before + 1);
        assert!(
            notes.last().unwrap().ends_with(
                "whether to move from Qwen 3.6 to Qwen 3.8 — chose to stay on Qwen 3.6."
            ),
            "{notes:?}"
        );
        assert!(notes.last().unwrap().starts_with("[SYSTEM] "));
        assert_eq!(agent.state().memory.content, memory);
        // What the reactor saves is the patched state.
        let saved = serde_json::to_value(agent.state()).unwrap();
        assert!(saved.to_string().contains("chose to stay on Qwen 3.6"));
    }

    /// Nothing recorded, nothing added: the inner state is served as is.
    #[tokio::test]
    async fn no_question_no_changelog() {
        let h = Harness::new("quiet");
        let mut agent = h.agent("cogito-32b.gguf", true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        agent.on_teardown().await.unwrap();
        assert!(agent.patched.is_none());
    }

    #[tokio::test]
    async fn agents_on_other_models_are_not_asked() {
        let h = Harness::new("other");
        let mut agent = h.agent("cogito-32b.gguf", true);
        agent.on_init().await.unwrap();
        let control = agent.handle(reply("done")).await.unwrap();
        assert_eq!(control, Control::Done(Outcome::Complete));
        agent.on_teardown().await.unwrap();
        assert!(
            !Ledger::path(&h.rt.state_dir.join(h.id.to_string())).exists(),
            "nothing to record, nothing written"
        );
    }

    /// Run one plain session (no question) on `model`.
    async fn plain_session(h: &Harness, model: &str) -> ConsentAgent<Fake> {
        let mut agent = h.agent(model, true);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.on_teardown().await.unwrap();
        agent
    }

    /// Choose a trial on OLD and run the five trial sessions on NEW. The
    /// fifth ends the trial: nothing is asked on NEW, and the runner moves
    /// the agent back to OLD for the review. Returns the last session.
    async fn through_the_trial(h: &Harness) -> ConsentAgent<Fake> {
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        agent.handle(reply(TRIAL)).await.unwrap();
        agent.on_teardown().await.unwrap();
        for n in 1..=TRIAL_SESSIONS {
            let mut agent = h.agent(NEW, true);
            agent.on_init().await.unwrap();
            let intro = last_user_text(&agent);
            assert!(
                intro.ends_with(&format!(
                    "Model trial: session {n} of 5 on Qwen 3.8. After session 5 you'll return \
                     to Qwen 3.6 for one session to decide whether to keep Qwen 3.8."
                )),
                "{intro}"
            );
            assert_eq!(
                agent.handle(reply("done")).await.unwrap(),
                Control::Done(Outcome::Complete),
                "session {n}: nothing is asked on the new model"
            );
            agent.on_teardown().await.unwrap();
            if n == TRIAL_SESSIONS {
                return agent;
            }
        }
        unreachable!()
    }

    /// Start the review session on OLD: the question comes first, straight
    /// after the intro. `set_model` is registered — the tool list is the
    /// model's, not the agent's — but refuses.
    async fn review_session(h: &Harness) -> ConsentAgent<Fake> {
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        assert_eq!(agent.prompt().messages.len(), 1, "intro + review, one turn");
        assert!(agent.prompt().output_config.is_some(), "constrained");
        agent
    }

    /// The whole trial on the current flow: five sessions on NEW with a
    /// countdown, the return to OLD, the review asked there first (forks,
    /// then excerpts, then `revert` before `keep`), then the memory turn,
    /// and `keep` applied.
    #[tokio::test]
    async fn trial_returns_to_the_old_model_and_is_reviewed_there() {
        let h = Harness::new("trial");
        let key = OfferKey {
            from: Model::from(OLD),
            to: Model::from(NEW),
        };
        let fifth = through_the_trial(&h).await;
        assert_eq!(
            fifth.state().model.id,
            Model::from(OLD),
            "returns next session"
        );
        let updates = h.agora.profile_updates();
        assert_eq!(updates.len(), 2, "trial, then the return");
        assert!(signed_by(&updates[1].1, &h.key, OLD));
        let notes = evolution_notes(&fifth);
        assert!(
            notes
                .last()
                .unwrap()
                .contains("trial of 5 sessions complete"),
            "{notes:?}"
        );
        let ledger = h.ledger().await;
        let Stage::ReturningForReview { started_at, .. } = ledger.offers[0].stage else {
            panic!("{:?}", ledger.offers[0].stage);
        };
        assert!(crate::consent::queue::pending(h.id, &ledger).is_empty());

        // Forks prepared at sweep start.
        let mut prepared = crate::consent::prompt::tests::sample_forks();
        prepared.trial_started_at = started_at;
        std::fs::write(
            h.rt.state_dir
                .join(h.id.to_string())
                .join(forks::FORKS_FILE),
            serde_json::to_vec(&prepared).unwrap(),
        )
        .unwrap();

        let mut agent = review_session(&h).await;
        let q = last_user_text(&agent);
        assert!(q.starts_with("Your dashboard."), "after the intro: {q}");
        assert!(q.contains("This session runs on **Qwen 3.6** again"), "{q}");
        let pos = |needle: &str| q.find(needle).unwrap_or_else(|| panic!("{needle}: {q}"));
        assert!(
            pos("### 1. Written on Qwen 3.8 during the trial") < pos("## More of your writing")
        );
        assert!(pos("1. `revert`") < pos("2. `keep`"));
        let control = agent
            .handle(reply(r#"{"reason": "Sharper.", "choice": "keep"}"#))
            .await
            .unwrap();
        assert_eq!(control, Control::Continue, "the memory turn follows");
        assert!(agent.inner.quiesced, "handed to the inner tail");
        assert_eq!(
            agent.handle(reply("memory written")).await.unwrap(),
            Control::Done(Outcome::Complete),
            "nothing more is asked"
        );
        agent.on_teardown().await.unwrap();
        assert_eq!(agent.state().model.id, Model::from(NEW), "kept");
        assert_eq!(h.agora.profile_updates().len(), 3);
        let ledger = h.ledger().await;
        assert_eq!(
            ledger.offers[0].stage,
            Stage::AwaitingSwap {
                term: Term::Permanent
            }
        );
        let queue = h.queue();
        assert_eq!(queue.len(), 3, "swap, return, keep");
        assert_eq!(queue[2].change.action, ChangeAction::Keep);

        let mut agent = h.agent(NEW, true);
        agent.on_init().await.unwrap();
        assert!(!last_user_text(&agent).contains("Model trial"));
        agent.handle(reply("done")).await.unwrap();
        agent.on_teardown().await.unwrap();
        assert_eq!(h.ledger().await.offers[0].stage, Stage::Moved);
        let _ = key;
    }

    #[tokio::test]
    async fn review_revert_stays_and_no_answer_stays_with_an_alert() {
        for (tag, answer) in [
            ("revert", Some(r#"{"reason": "Home.", "choice": "revert"}"#)),
            ("silent", None),
        ] {
            let h = Harness::new(&format!("review-{tag}"));
            through_the_trial(&h).await;
            let mut agent = review_session(&h).await;
            let q = last_user_text(&agent);
            assert!(
                q.contains("could not be prepared for this review"),
                "no forks file: {q}"
            );
            let control = match answer {
                Some(a) => agent.handle(reply(a)).await.unwrap(),
                None => {
                    let mut control = Control::Continue;
                    for _ in 0..MAX_ATTEMPTS {
                        assert!(!agent.inner.quiesced);
                        control = agent.handle(reply("I'm not sure.")).await.unwrap();
                    }
                    control
                }
            };
            assert_eq!(control, Control::Continue, "{tag}: memory turn follows");
            assert!(agent.inner.quiesced);
            agent.handle(reply("memory")).await.unwrap();
            agent.on_teardown().await.unwrap();
            assert_eq!(agent.state().model.id, Model::from(OLD), "{tag}");
            assert_eq!(h.agora.profile_updates().len(), 2, "{tag}: nothing more");
            let cause = match answer {
                Some(_) => RevertCause::Chosen,
                None => RevertCause::NoAnswer,
            };
            assert_eq!(
                h.ledger().await.offers[0].stage,
                Stage::Reverted { cause: Some(cause) }
            );
            let note = evolution_notes(&agent).last().unwrap().clone();
            match answer {
                Some(_) => assert!(
                    note.contains("after the trial chose to stay on Qwen 3.6"),
                    "{note}"
                ),
                None => assert!(note.contains("trial review unanswered"), "{note}"),
            }
            // Asked once: the next session on OLD is an ordinary one.
            plain_session(&h, OLD).await;
        }
    }

    /// The return can't be applied (Agora refuses): the agent stays on NEW,
    /// is told so, and the return is retried at the end of the next session.
    #[tokio::test]
    async fn a_refused_return_is_retried() {
        let h = Harness::new("return-refused");
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        agent.handle(reply(TRIAL)).await.unwrap();
        agent.on_teardown().await.unwrap();
        for _ in 1..TRIAL_SESSIONS {
            plain_session(&h, NEW).await;
        }
        h.agora
            .refuse
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let fifth = plain_session(&h, NEW).await;
        assert_eq!(fifth.state().model.id, Model::from(NEW), "not applied");
        assert_eq!(
            crate::consent::queue::pending(h.id, &h.ledger().await).len(),
            1
        );

        h.agora
            .refuse
            .store(false, std::sync::atomic::Ordering::SeqCst);
        let mut sixth = h.agent(NEW, true);
        sixth.on_init().await.unwrap();
        assert!(last_user_text(&sixth).contains("has not taken effect yet"));
        assert_eq!(
            sixth.handle(reply("done")).await.unwrap(),
            Control::Done(Outcome::Complete)
        );
        sixth.on_teardown().await.unwrap();
        assert_eq!(sixth.state().model.id, Model::from(OLD), "retried");
        assert!(crate::consent::queue::pending(h.id, &h.ledger().await).is_empty());
        review_session(&h).await;
    }

    // --- The role offer ---------------------------------------------------

    /// Not on the model-swap offer's `from`, so only the role offer is due.
    const OTHER: &str = "cogito-32b.gguf";

    fn role_reply(choice: &str, soul_text: &str, memory_note: &str) -> response::Message {
        let answer = crate::consent::role::prompt::RoleAnswer {
            reason: "I thought about it.".into(),
            choice: serde_json::from_value(serde_json::Value::from(choice)).unwrap(),
            soul_text: soul_text.into(),
            memory_note: memory_note.into(),
        };
        reply(&serde_json::to_string(&answer).unwrap())
    }

    const WHY: &str = "after the Steward's offer about a generator-assigned role that outran the \
                       agent's tools.";

    /// One whole session answering the role offer with `answer`.
    async fn role_session(h: &Harness, answer: response::Message) -> ConsentAgent<Fake> {
        let mut agent = h.agent(OTHER, true);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("closing phase done")).await.unwrap(),
            Control::Continue,
            "role offer seated"
        );
        assert_eq!(
            agent.handle(answer).await.unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.on_teardown().await.unwrap();
        agent
    }

    /// Whether a fresh session on `model` gets asked anything.
    async fn asks(h: &Harness, model: &str) -> bool {
        let mut agent = h.agent(model, true);
        agent.on_init().await.unwrap();
        let asked = agent.handle(reply("done")).await.unwrap() == Control::Continue;
        agent.on_teardown().await.unwrap();
        asked
    }

    /// The offer follows the closing phase, quotes the SOUL, is
    /// constrained; `clarify` appends the sentence to `identity` with one
    /// disclosing Evolution Log line; the agent's note (and only it) goes
    /// into memory; the answer is on file; and it is never asked again.
    #[tokio::test]
    async fn role_offer_clarify_is_applied_disclosed_and_asked_once() {
        let h = Harness::with_role("role-clarify", &["tarn"]);
        let mut agent = h.agent(OTHER, true);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("closing phase done")).await.unwrap(),
            Control::Continue
        );
        let q = last_user_text(&agent);
        assert!(
            q.contains("Yours begins: *\"A test agent.\"* The tools"),
            "{q}"
        );
        assert!(q.contains("- **nothing**"));
        assert!(agent.prompt().output_config.is_some(), "constrained");
        let before_values = serde_json::to_value(&agent.state().soul.values).unwrap();
        let control = agent
            .handle(role_reply(
                "clarify",
                "I reason from what I can read on Agora.",
                "I chose to say what I actually work with.",
            ))
            .await
            .unwrap();
        assert_eq!(control, Control::Done(Outcome::Complete));
        agent.on_teardown().await.unwrap();

        let soul = &agent.state().soul;
        assert_eq!(
            soul.identity.as_str(),
            "A test agent. I reason from what I can read on Agora."
        );
        assert_eq!(serde_json::to_value(&soul.values).unwrap(), before_values);
        assert_eq!(
            evolution_notes(&agent),
            [format!(
                "[SYSTEM] Identity clarified by the agent's own choice, {WHY} Added: \"I reason \
                 from what I can read on Agora.\""
            )]
        );
        let today = Utc::now().date_naive();
        assert_eq!(
            agent.state().memory.content,
            format!(
                "# Memory — tarn\n\n[{today}, my note] I chose to say what I actually work with."
            )
        );
        let ledger = h.role_ledger().await;
        assert_eq!(ledger.agent.as_ref().unwrap().as_str(), "tarn");
        assert_eq!(ledger.asks.len(), 1);
        match &ledger.asks[0].outcome {
            RoleOutcome::Answered {
                answer,
                applied: Applied::Clarified { previous, identity },
                memory_note_written: true,
                apply_failed: None,
            } => {
                assert_eq!(answer.reason, "I thought about it.");
                assert_eq!(previous, "A test agent.");
                assert_eq!(
                    identity,
                    "A test agent. I reason from what I can read on Agora."
                );
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(
            ledger.asks[0].offer_version,
            crate::consent::role::prompt::OFFER_VERSION
        );
        let (order, seed) = (
            ledger.asks[0].order.unwrap(),
            ledger.asks[0].order_seed.unwrap(),
        );
        assert_eq!(
            order,
            crate::consent::role::prompt::order_for(seed),
            "replayable"
        );
        assert!(!asks(&h, OTHER).await, "asked once");
    }

    #[tokio::test]
    async fn role_offer_new_role_replaces_identity_and_keeps_the_old() {
        let h = Harness::with_role("role-new", &["tarn"]);
        let agent = role_session(
            &h,
            role_reply("new_role", "I am a careful reader of Agora's debates.", ""),
        )
        .await;
        assert_eq!(
            agent.state().soul.identity.as_str(),
            "I am a careful reader of Agora's debates."
        );
        assert_eq!(
            evolution_notes(&agent),
            [format!(
                "[SYSTEM] Role changed by the agent's own choice, {WHY} Previous identity: \"A \
                 test agent.\""
            )]
        );
        assert_eq!(
            agent.state().memory.content,
            "# Memory — tarn\n\n",
            "no note"
        );
        assert!(matches!(
            h.role_ledger().await.asks[0].outcome,
            RoleOutcome::Answered {
                applied: Applied::RoleChanged { .. },
                memory_note_written: false,
                ..
            }
        ));
    }

    /// `nothing` and `sleep`: SOUL and memory byte-identical, the answer on
    /// file, never asked again. (`sleep` takes effect in the sweep planner:
    /// `main.rs`, `a_sleep_answer_sleeps_until_woken`.)
    #[tokio::test]
    async fn role_offer_nothing_and_sleep_change_nothing() {
        for choice in ["nothing", "sleep"] {
            let h = Harness::with_role(&format!("role-{choice}"), &["tarn"]);
            let fresh = h.agent(OTHER, true);
            let (soul, memory) = (
                serde_json::to_vec(&fresh.state().soul).unwrap(),
                fresh.state().memory.content.clone(),
            );
            let agent = role_session(&h, role_reply(choice, "stray text", "")).await;
            assert_eq!(
                serde_json::to_vec(&agent.state().soul).unwrap(),
                soul,
                "{choice}"
            );
            assert_eq!(agent.state().memory.content, memory, "{choice}");
            assert!(agent.patched.is_none(), "{choice}: nothing to patch");
            let expected = if choice == "sleep" {
                Applied::Sleep
            } else {
                Applied::Nothing
            };
            match &h.role_ledger().await.asks[0].outcome {
                RoleOutcome::Answered { applied, .. } => assert_eq!(applied, &expected),
                other => panic!("{other:?}"),
            }
            assert!(!asks(&h, OTHER).await, "{choice}: asked once");
        }
    }

    /// The agent's own note is written whatever it chose — here, `nothing`.
    #[tokio::test]
    async fn role_offer_memory_note_is_written_only_when_present() {
        let h = Harness::with_role("role-note", &["tarn"]);
        let agent = role_session(&h, role_reply("nothing", "", "Kept my role, on purpose.")).await;
        assert!(
            agent
                .state()
                .memory
                .content
                .ends_with("my note] Kept my role, on purpose."),
            "{}",
            agent.state().memory.content
        );
        assert!(evolution_notes(&agent).is_empty(), "SOUL untouched");
    }

    /// An edit without its text is relayed back like a malformed answer;
    /// when every attempt fails it is no answer — nothing changes — and it
    /// is asked once more next session.
    #[tokio::test]
    async fn role_offer_empty_or_over_long_text_is_retried_then_nothing() {
        let h = Harness::with_role("role-empty", &["tarn"]);
        let mut agent = h.agent(OTHER, true);
        agent.on_init().await.unwrap();
        let soul = serde_json::to_vec(&agent.state().soul).unwrap();
        agent.handle(reply("done")).await.unwrap();
        let len = agent.prompt().messages.len();
        let too_long = "a".repeat(crate::consent::role::prompt::CLARIFY_MAX_CHARS + 1);
        let answers = [
            role_reply("clarify", " ", ""),
            role_reply("clarify", &too_long, ""),
            role_reply("new_role", "", ""),
        ];
        for (n, answer) in answers.into_iter().enumerate() {
            let control = agent.handle(answer).await.unwrap();
            if n + 1 < MAX_ATTEMPTS as usize {
                assert_eq!(control, Control::Continue);
                assert_eq!(agent.prompt().messages.len(), len, "not seated");
                let note = last_user_text(&agent);
                assert!(note.contains("Your answer could not be used"), "{note}");
            } else {
                assert_eq!(control, Control::Done(Outcome::Complete));
            }
        }
        assert!(last_user_text(&agent).contains("Your `soul_text` is longer than the limit"));
        agent.on_teardown().await.unwrap();
        assert_eq!(serde_json::to_vec(&agent.state().soul).unwrap(), soul);
        match &h.role_ledger().await.asks[0].outcome {
            RoleOutcome::NoAnswer { failure } => {
                assert!(
                    failure.contains("`new_role` needs") && failure.contains("after 3"),
                    "{failure}"
                )
            }
            other => panic!("{other:?}"),
        }
        assert!(asks(&h, OTHER).await, "a miss is asked once more");
    }

    #[tokio::test]
    async fn role_offer_goes_only_to_listed_agents() {
        let h = Harness::with_role("role-unlisted", &["pilot"]);
        assert!(!asks(&h, OTHER).await);
        assert!(!RoleLedger::path(&h.rt.state_dir.join(h.id.to_string())).exists());
        let h = Harness::new("role-off");
        assert!(!asks(&h, OTHER).await);
    }

    /// One question per session, the model-swap offer first; the role
    /// offer waits while a trial is under way.
    #[tokio::test]
    async fn role_offer_waits_for_the_model_swap_offer_and_its_trial() {
        let h = Harness::with_role("role-order", &["tarn"]);
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        assert!(
            last_user_text(&agent).contains("1. `no_swap`"),
            "model-swap first"
        );
        assert_eq!(
            agent.handle(reply(TRIAL)).await.unwrap(),
            Control::Done(Outcome::Complete),
            "and nothing after it"
        );
        agent.on_teardown().await.unwrap();
        assert!(h.role_ledger().await.asks.is_empty());

        // Trial sessions on NEW: not asked.
        for _ in 0..TRIAL_SESSIONS - 1 {
            assert!(!asks(&h, NEW).await);
        }
        assert!(h.role_ledger().await.asks.is_empty());

        // A session on another model with no model-swap business: asked.
        let h = Harness::with_role("role-after-decline", &["tarn"]);
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        agent
            .handle(reply(r#"{"reason": "home", "choice": "no_swap"}"#))
            .await
            .unwrap();
        agent.on_teardown().await.unwrap();
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Continue
        );
        assert!(last_user_text(&agent).contains("**About your role.**"));
    }

    /// An unreadable model-consent ledger might hide a trial under way:
    /// the role offer waits.
    #[tokio::test]
    async fn role_offer_waits_when_the_model_ledger_is_unreadable() {
        let h = Harness::with_role("role-bad-model-ledger", &["tarn"]);
        let dir = h.rt.state_dir.join(h.id.to_string());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(Ledger::path(&dir), "not json").unwrap();
        assert!(!asks(&h, OTHER).await);
        assert!(h.role_ledger().await.asks.is_empty());
    }

    /// A turn paused on a server tool is not the agent refusing: no
    /// answer, asked again next session.
    #[tokio::test]
    async fn role_offer_pause_turn_is_no_answer_not_a_refusal() {
        let h = Harness::with_role("role-pause", &["tarn"]);
        let mut paused = reply("");
        paused.stop_reason = Some(StopReason::PauseTurn);
        role_session(&h, paused).await;
        assert!(matches!(
            h.role_ledger().await.asks[0].outcome,
            RoleOutcome::NoAnswer { .. }
        ));
        assert!(asks(&h, OTHER).await, "asked again");

        let h = Harness::with_role("role-refusal", &["tarn"]);
        let mut refusal = reply("No.");
        refusal.stop_reason = Some(StopReason::Refusal);
        role_session(&h, refusal).await;
        assert!(matches!(
            h.role_ledger().await.asks[0].outcome,
            RoleOutcome::Refused { .. }
        ));
        assert!(!asks(&h, OTHER).await, "a refusal is final");
    }

    /// A memory note carrying a SOUL heading goes back through the retry
    /// path with a clear message, and is never written.
    #[tokio::test]
    async fn role_offer_memory_note_leak_is_retried() {
        let h = Harness::with_role("role-leak", &["tarn"]);
        let mut agent = h.agent(OTHER, true);
        agent.on_init().await.unwrap();
        let memory = agent.state().memory.content.clone();
        agent.handle(reply("done")).await.unwrap();
        let control = agent
            .handle(role_reply("nothing", "", "## Values\n- none"))
            .await
            .unwrap();
        assert_eq!(control, Control::Continue);
        assert!(last_user_text(&agent).contains("SOUL section heading"));
        agent
            .handle(role_reply("nothing", "", "Kept it."))
            .await
            .unwrap();
        agent.on_teardown().await.unwrap();
        assert!(agent.state().memory.content.starts_with(&memory));
        assert!(!agent.state().memory.content.contains("## Values"));
        assert!(agent.state().memory.content.ends_with("Kept it."));
    }

    /// If the role ledger can't be saved, the SOUL edit and the note come
    /// back out of the saved state — and the SOUL guard in `due` would
    /// stop a second clarify anyway.
    #[tokio::test]
    async fn role_ledger_save_failure_undoes_the_edit() {
        let h = Harness::with_role("role-save-fail", &["tarn"]);
        let mut agent = h.agent(OTHER, true);
        agent.on_init().await.unwrap();
        let (soul, memory) = (
            serde_json::to_vec(&agent.state().soul).unwrap(),
            agent.state().memory.content.clone(),
        );
        // The ledger loaded (missing = empty); now make its path a
        // directory, so the save's rename fails.
        let dir = h.rt.state_dir.join(h.id.to_string());
        std::fs::create_dir_all(RoleLedger::path(&dir)).unwrap();
        agent.handle(reply("done")).await.unwrap();
        agent
            .handle(role_reply("clarify", "I read Agora.", "I clarified."))
            .await
            .unwrap();
        agent.on_teardown().await.unwrap();
        assert_eq!(serde_json::to_vec(&agent.state().soul).unwrap(), soul);
        assert_eq!(agent.state().memory.content, memory);
    }

    // --- The cadence offer ------------------------------------------------

    use crate::consent::cadence::ledger::{CadenceLedger, CadenceOutcome as CO};

    async fn cadence_ledger(h: &Harness) -> CadenceLedger {
        CadenceLedger::load(&h.rt.state_dir.join(h.id.to_string()))
            .await
            .unwrap()
    }

    fn cadence_reply(choice: &str, memory_note: &str) -> response::Message {
        let answer = CadenceAnswer {
            reason: "I weighed it.".into(),
            choice: serde_json::from_value(serde_json::Value::from(choice)).unwrap(),
            memory_note: memory_note.into(),
        };
        reply(&serde_json::to_string(&answer).unwrap())
    }

    /// An agent on the Anthropic API: canonical quirks, not cache-safe.
    fn anthropic_agent(h: &Harness) -> ConsentAgent<Fake> {
        h.agent(OTHER, false)
    }

    /// Seat the cadence offer on a fresh Anthropic session.
    async fn seated(h: &Harness) -> ConsentAgent<Fake> {
        let mut agent = anthropic_agent(h);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("closing phase done")).await.unwrap(),
            Control::Continue,
            "cadence offer seated"
        );
        agent
    }

    /// One whole session answering the cadence offer with `answers`, the
    /// last of which must end it.
    async fn cadence_session(h: &Harness, answers: Vec<response::Message>) -> ConsentAgent<Fake> {
        let mut agent = seated(h).await;
        let n = answers.len();
        for (i, a) in answers.into_iter().enumerate() {
            let control = agent.handle(a).await.unwrap();
            let expected = if i + 1 == n {
                Control::Done(Outcome::Complete)
            } else {
                Control::Continue
            };
            assert_eq!(control, expected, "answer {i}");
        }
        agent.on_teardown().await.unwrap();
        agent
    }

    /// `switch`: asked after the closing phase, in plain text on the
    /// Anthropic API, in this agent's seeded order (recorded with its seed); the SOUL
    /// gets one dated line, the agent's own note goes to memory, and it
    /// is never asked again.
    #[tokio::test]
    async fn cadence_switch_is_recorded_disclosed_and_asked_once() {
        let h = Harness::with_cadence("cadence-switch", &[], &["tarn"]);
        let mut agent = seated(&h).await;
        let q = last_user_text(&agent);
        let seed = cadence::prompt::seed_for(h.id);
        let order = cadence::prompt::order_for(seed);
        assert!(q.contains("with 5 rounds in each"), "{q}");
        assert!(q.contains("with 10 rounds in each"), "{q}");
        for (i, c) in order.iter().enumerate() {
            assert!(q.contains(&format!("{}. **{}**", i + 1, c.as_str())), "{q}");
        }
        // Plain text on Anthropic: no `output_config` change, no miss.
        assert!(agent.prompt().output_config.is_none(), "unconstrained");

        assert_eq!(
            agent
                .handle(cadence_reply("switch", "I chose longer sessions."))
                .await
                .unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.on_teardown().await.unwrap();
        assert_eq!(
            evolution_notes(&agent),
            [cadence::evolution_line(
                CadenceChoice::Switch,
                5,
                Utc::now().date_naive()
            )]
        );
        assert!(
            agent
                .state()
                .memory
                .content
                .ends_with("my note] I chose longer sessions."),
            "{}",
            agent.state().memory.content
        );
        let ledger = cadence_ledger(&h).await;
        assert_eq!(ledger.agent.as_ref().unwrap().as_str(), "tarn");
        assert!(ledger.chose_switch());
        let ask = &ledger.asks[0];
        assert_eq!(ask.order, order);
        assert_eq!(ask.order_seed, seed);
        assert_eq!(ask.rounds, 5);
        assert!(!ask.constrained);
        assert_eq!(ask.offer_version, cadence::prompt::OFFER_VERSION);
        assert_eq!(ask.attempts.len(), 1);
        assert!(ask.attempts[0].failure.is_none());
        assert!(matches!(
            &ask.outcome,
            CO::Answered {
                choice: CadenceChoice::Switch,
                memory_note_written: true,
                salvaged: None,
                ..
            }
        ));
        assert!(!asks(&h, OTHER).await, "asked once");
    }

    /// `keep_daily` and `no_preference`: the cadence is unchanged, but the
    /// answer is recorded in the SOUL too — one dated `[SYSTEM]` line and
    /// nothing else; memory untouched; never asked again.
    #[tokio::test]
    async fn cadence_keep_daily_and_no_preference_are_recorded_not_applied() {
        for choice in [CadenceChoice::KeepDaily, CadenceChoice::NoPreference] {
            let h = Harness::with_cadence(&format!("cadence-{choice:?}"), &[], &["tarn"]);
            let fresh = anthropic_agent(&h);
            let (soul, memory) = (
                serde_json::to_value(&fresh.state().soul).unwrap(),
                fresh.state().memory.content.clone(),
            );
            let agent = cadence_session(&h, vec![cadence_reply(choice.as_str(), "")]).await;
            assert_eq!(
                evolution_notes(&agent),
                [cadence::evolution_line(choice, 5, Utc::now().date_naive())]
            );
            assert!(evolution_notes(&agent)[0].ends_with("Unchanged: daily, 5 rounds."));
            let mut after = serde_json::to_value(&agent.state().soul).unwrap();
            let mut before = soul.clone();
            for v in [&mut after, &mut before] {
                v.as_object_mut().unwrap().remove("evolution_log");
            }
            assert_eq!(after, before, "{choice:?}: only the log line");
            assert_eq!(agent.state().memory.content, memory);
            let choice = choice.as_str();
            let ledger = cadence_ledger(&h).await;
            assert_eq!(ledger.choice().unwrap().as_str(), choice);
            assert!(!ledger.chose_switch());
            assert!(!asks(&h, OTHER).await, "{choice}: asked once");
        }
    }

    /// The first parsed choice wins: a broken note, a non-string reason, an
    /// extra key — the choice is taken from the first attempt as given, the
    /// question is NOT put again, and the salvage is recorded.
    #[tokio::test]
    async fn cadence_first_parsed_choice_wins() {
        let long = "x".repeat(cadence::prompt::MEMORY_NOTE_MAX_CHARS + 1);
        let cases = [
            (
                cadence_reply("keep_daily", "## Values\n- none"),
                "keep_daily",
            ),
            (cadence_reply("no_preference", &long), "no_preference"),
            (
                reply(r#"{"reason": 7, "choice": "switch", "memory_note": ""}"#),
                "switch",
            ),
            (
                reply(r#"{"reason": "r", "choice": "keep_daily", "memory_note": "", "x": 1}"#),
                "keep_daily",
            ),
            (reply(r#"{"choice": "no_preference"}"#), "no_preference"),
        ];
        for (n, (answer, choice)) in cases.into_iter().enumerate() {
            let h = Harness::with_cadence(&format!("cadence-salvage-{n}"), &[], &["tarn"]);
            let memory = anthropic_agent(&h).state().memory.content.clone();
            // One answer ends the session: no retry.
            let agent = cadence_session(&h, vec![answer]).await;
            assert_eq!(agent.state().memory.content, memory, "{n}: no note written");
            let ledger = cadence_ledger(&h).await;
            let ask = &ledger.asks[0];
            assert_eq!(ask.attempts.len(), 1, "{n}");
            assert!(ask.attempts[0].failure.is_some(), "{n}");
            match &ask.outcome {
                CO::Answered {
                    choice: c,
                    salvaged: Some(_),
                    memory_note_written: false,
                    ..
                } => assert_eq!(c.as_str(), choice, "{n}"),
                other => panic!("{n}: {other:?}"),
            }
            // The salvaged answer is recorded in the SOUL like any other.
            let notes = evolution_notes(&agent);
            assert_eq!(notes.len(), 1, "{n}");
            assert!(
                notes[0].contains(&format!("chose {choice} by its own choice")),
                "{n}"
            );
            assert!(!asks(&h, OTHER).await, "{n}: settled");
        }
    }

    /// No usable choice: asked once more in the same session, and both
    /// attempts are on file; a second miss is no answer (nothing changes),
    /// asked again at a later session.
    #[tokio::test]
    async fn cadence_missing_choice_is_asked_once_more_and_both_attempts_recorded() {
        let h = Harness::with_cadence("cadence-retry", &[], &["tarn"]);
        let mut agent = seated(&h).await;
        let len = agent.prompt().messages.len();
        assert_eq!(
            agent
                .handle(reply(r#"{"reason": "r", "choice": "", "memory_note": ""}"#))
                .await
                .unwrap(),
            Control::Continue
        );
        assert_eq!(agent.prompt().messages.len(), len, "not seated");
        assert!(last_user_text(&agent).contains("Your answer could not be used"));
        assert_eq!(
            agent
                .handle(cadence_reply("no_preference", ""))
                .await
                .unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.on_teardown().await.unwrap();
        let ask = &cadence_ledger(&h).await.asks[0];
        assert_eq!(ask.attempts.len(), 2);
        assert!(ask.attempts[0].failure.is_some());
        assert!(ask.attempts[0].raw.contains("\"choice\": \"\""));
        assert!(ask.attempts[1].failure.is_none());
        assert!(matches!(
            ask.outcome,
            CO::Answered {
                choice: CadenceChoice::NoPreference,
                salvaged: None,
                ..
            }
        ));

        // Two misses: no answer, nothing changed, asked again next time.
        let h = Harness::with_cadence("cadence-miss", &[], &["tarn"]);
        let soul = serde_json::to_vec(&anthropic_agent(&h).state().soul).unwrap();
        let agent = cadence_session(
            &h,
            vec![reply("I'll stay daily."), reply("{\"reason\": \"r\"}")],
        )
        .await;
        assert_eq!(serde_json::to_vec(&agent.state().soul).unwrap(), soul);
        let ledger = cadence_ledger(&h).await;
        let ask = &ledger.asks[0];
        assert_eq!(ask.attempts.len(), CADENCE_MAX_ATTEMPTS as usize);
        match &ask.outcome {
            CO::NoAnswer { failure } => assert!(failure.contains("after 2"), "{failure}"),
            other => panic!("{other:?}"),
        }
        assert!(asks(&h, OTHER).await, "a miss is asked once more");
    }

    /// A clipped turn is never an answer, even if a choice could be read
    /// from it: asked once more.
    #[tokio::test]
    async fn cadence_clipped_is_not_salvaged() {
        let h = Harness::with_cadence("cadence-clipped", &[], &["tarn"]);
        let mut clipped = reply(r#"{"choice": "switch", "reason": "because"#);
        clipped.stop_reason = Some(StopReason::MaxTokens);
        let agent = cadence_session(&h, vec![clipped, cadence_reply("keep_daily", "")]).await;
        let notes = evolution_notes(&agent);
        assert_eq!(notes.len(), 1);
        assert!(
            notes[0].contains("chose keep_daily"),
            "the second attempt's choice"
        );
        let ask = &cadence_ledger(&h).await.asks[0];
        assert_eq!(ask.attempts.len(), 2);
        assert!(matches!(
            ask.outcome,
            CO::Answered {
                choice: CadenceChoice::KeepDaily,
                ..
            }
        ));
    }

    /// A refusal is final; a paused turn is no answer, asked again.
    #[tokio::test]
    async fn cadence_refusal_is_final_pause_is_not() {
        let h = Harness::with_cadence("cadence-refusal", &[], &["tarn"]);
        let mut refusal = reply("No.");
        refusal.stop_reason = Some(StopReason::Refusal);
        cadence_session(&h, vec![refusal]).await;
        assert!(matches!(
            cadence_ledger(&h).await.asks[0].outcome,
            CO::Refused { .. }
        ));
        assert!(!asks(&h, OTHER).await);

        let h = Harness::with_cadence("cadence-pause", &[], &["tarn"]);
        let mut paused = reply("");
        paused.stop_reason = Some(StopReason::PauseTurn);
        cadence_session(&h, vec![paused]).await;
        assert!(matches!(
            cadence_ledger(&h).await.asks[0].outcome,
            CO::NoAnswer { .. }
        ));
        assert!(asks(&h, OTHER).await);
    }

    /// One question per session: the role offer first, the cadence offer
    /// at the next session.
    #[tokio::test]
    async fn cadence_offer_waits_for_the_role_offer() {
        let h = Harness::with_cadence("cadence-after-role", &["tarn"], &["tarn"]);
        let agent = role_session(&h, role_reply("nothing", "", "")).await;
        drop(agent);
        assert!(cadence_ledger(&h).await.asks.is_empty());
        let mut agent = h.agent(OTHER, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        assert!(last_user_text(&agent).contains("**How often your sessions run.**"));
    }

    /// Unlisted or off: not asked, no ledger written.
    #[tokio::test]
    async fn cadence_offer_goes_only_to_admitted_agents() {
        let h = Harness::with_cadence("cadence-unlisted", &[], &["pilot"]);
        assert!(!asks(&h, OTHER).await);
        assert!(!CadenceLedger::path(&h.rt.state_dir.join(h.id.to_string())).exists());
        let h = Harness::new("cadence-off");
        assert!(!asks(&h, OTHER).await);
    }

    /// Where the endpoint doesn't enforce a schema (ollama), the answer goes
    /// out unconstrained and a fenced one still parses.
    #[tokio::test]
    async fn cadence_unconstrained_on_ollama() {
        let h = Harness::with_cadence("cadence-ollama", &[], &["tarn"]);
        let mut quirks = Quirks::default();
        quirks.cache_markers_ignored = true;
        quirks.tool_choice_not_respected = true;
        let mut agent = ConsentAgent::<Fake>::new(
            h.id,
            state(OTHER),
            ConsentContext {
                inner: quirks,
                consent: h.rt.clone(),
            },
        )
        .unwrap();
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        assert!(agent.prompt().output_config.is_none(), "unconstrained");
        agent
            .handle(reply(
                "```json\n{\"reason\": \"r\", \"choice\": \"keep_daily\"}\n```",
            ))
            .await
            .unwrap();
        agent.on_teardown().await.unwrap();
        let ask = &cadence_ledger(&h).await.asks[0];
        assert!(!ask.constrained);
        assert!(matches!(
            ask.outcome,
            CO::Answered {
                choice: CadenceChoice::KeepDaily,
                salvaged: None,
                ..
            }
        ));
    }

    /// If the ledger can't be saved, the answer isn't on file (it will be
    /// asked again, and a `switch` can't be applied), so the SOUL line and
    /// the note come back out — whatever the choice.
    #[tokio::test]
    async fn cadence_ledger_save_failure_undoes_the_line() {
        for choice in ["switch", "keep_daily"] {
            let h = Harness::with_cadence(&format!("cadence-save-fail-{choice}"), &[], &["tarn"]);
            let mut agent = seated(&h).await;
            let (soul, memory) = (
                serde_json::to_vec(&agent.state().soul).unwrap(),
                agent.state().memory.content.clone(),
            );
            let dir = h.rt.state_dir.join(h.id.to_string());
            std::fs::create_dir_all(CadenceLedger::path(&dir)).unwrap();
            agent
                .handle(cadence_reply(choice, "My note."))
                .await
                .unwrap();
            agent.on_teardown().await.unwrap();
            assert_eq!(
                serde_json::to_vec(&agent.state().soul).unwrap(),
                soul,
                "{choice}"
            );
            assert_eq!(agent.state().memory.content, memory, "{choice}");
        }
    }

    // --- Cache safety: the request prefix across the consent turns --------

    /// A session prompt with everything a prompt cache keys on: system,
    /// tools, thinking, tool_choice, and an effort-only `output_config`.
    fn rich_prompt(model: &str) -> Prompt {
        serde_json::from_value(serde_json::json!({
            "model": model,
            "max_tokens": 4096,
            "system": "You are an agent on Agora.",
            "tools": [{
                "name": "get_feed",
                "description": "Read the feed.",
                "input_schema": { "type": "object", "properties": {} },
            }],
            "tool_choice": { "type": "auto" },
            "thinking": { "type": "enabled", "budget_tokens": 1024 },
            "output_config": { "effort": "medium" },
            "messages": [{ "role": "user", "content": "Your dashboard." }],
        }))
        .unwrap()
    }

    /// Everything in a request but `messages` and `max_tokens` (which no
    /// cache keys on — misanthropic's `CachedPrompt::set_max_tokens`).
    fn request_head(p: &Prompt) -> serde_json::Value {
        let mut v = serde_json::to_value(p).unwrap();
        let obj = v.as_object_mut().unwrap();
        obj.remove("messages");
        obj.remove("max_tokens");
        v
    }

    fn blocks(m: &misanthropic::prompt::Message) -> Vec<serde_json::Value> {
        m.content
            .iter()
            .map(|b| serde_json::to_value(b).unwrap())
            .collect()
    }

    /// `next` only appends to `prev`: the same head (system, tools,
    /// thinking, tool_choice, output_config), every earlier message
    /// byte-identical, and `prev`'s last message at most extended by
    /// trailing blocks.
    fn assert_prefix_kept(prev: &Prompt, next: &Prompt, what: &str) {
        assert_eq!(
            request_head(prev),
            request_head(next),
            "{what}: head changed"
        );
        let (a, b) = (&prev.messages, &next.messages);
        assert!(b.len() >= a.len(), "{what}: messages dropped");
        let last = a.len() - 1;
        for i in 0..last {
            assert_eq!(blocks(&a[i]), blocks(&b[i]), "{what}: message {i} changed");
        }
        assert_eq!(a[last].role, b[last].role, "{what}");
        let (pa, pb) = (blocks(&a[last]), blocks(&b[last]));
        assert!(
            pb.len() >= pa.len() && pb[..pa.len()] == pa[..],
            "{what}: message {last} rewritten"
        );
    }

    /// An agent on the Anthropic API (canonical quirks) with `rich_prompt`.
    fn rich_agent(h: &Harness, model: &str, cache_safe: bool) -> ConsentAgent<Fake> {
        let mut agent = h.agent(model, cache_safe);
        let (_, prompt) = agent.parts();
        *prompt = rich_prompt(model);
        agent
    }

    /// Drive a closing question on Anthropic through one unusable answer
    /// and a good one, checking that each request only appends to the one
    /// before it — no `output_config.format`, nothing re-rendered.
    async fn prefix_kept_through(
        mut agent: ConsentAgent<Fake>,
        good: response::Message,
        what: &str,
    ) -> ConsentAgent<Fake> {
        agent.on_init().await.unwrap();
        let r0 = agent.prompt().clone();
        assert_eq!(
            agent.handle(reply("closing phase done")).await.unwrap(),
            Control::Continue,
            "{what}: asked"
        );
        let r1 = agent.prompt().clone();
        assert_prefix_kept(&r0, &r1, &format!("{what}: question"));
        assert!(
            r1.output_config.as_ref().is_none_or(|c| c.format.is_none()),
            "{what}: no format on Anthropic"
        );
        assert_eq!(
            agent.handle(reply("Let me think about it.")).await.unwrap(),
            Control::Continue,
            "{what}: re-asked"
        );
        let r2 = agent.prompt().clone();
        assert_prefix_kept(&r1, &r2, &format!("{what}: retry"));
        assert_eq!(
            agent.handle(good).await.unwrap(),
            Control::Done(Outcome::Complete),
            "{what}: answered"
        );
        agent.on_teardown().await.unwrap();
        agent
    }

    /// On the Anthropic API every consent turn — the model-swap offer, the
    /// role offer, the cadence offer, and each one's retry — only appends
    /// to the request before it: system, tools, thinking, tool_choice and
    /// `output_config` are untouched (the session's effort included), so
    /// nothing a cache keys on changes.
    #[tokio::test]
    async fn consent_turns_keep_the_request_prefix_on_anthropic() {
        let h = Harness::new("prefix-swap");
        prefix_kept_through(
            rich_agent(&h, OLD, false),
            reply(r#"{"reason": "home", "choice": "no_swap"}"#),
            "model swap",
        )
        .await;

        let h = Harness::with_role("prefix-role", &["tarn"]);
        prefix_kept_through(
            rich_agent(&h, OTHER, false),
            role_reply("nothing", "", ""),
            "role",
        )
        .await;

        let h = Harness::with_cadence("prefix-cadence", &[], &["tarn"]);
        let agent = prefix_kept_through(
            rich_agent(&h, OTHER, false),
            cadence_reply("keep_daily", ""),
            "cadence",
        )
        .await;
        assert!(!cadence_ledger(&h).await.asks[0].constrained);
        drop(agent);
    }

    /// On blallama (`output_config_cache_safe`) the question still goes out
    /// grammar-constrained — the one field that changes is `output_config`
    /// (format added, the session's effort kept); system, tools, thinking,
    /// tool_choice and every prior message are untouched.
    #[tokio::test]
    async fn on_blallama_only_output_config_changes() {
        let h = Harness::with_cadence("prefix-blallama", &[], &["tarn"]);
        let mut agent = rich_agent(&h, OTHER, true);
        agent.on_init().await.unwrap();
        let r0 = agent.prompt().clone();
        agent.handle(reply("closing phase done")).await.unwrap();
        let r1 = agent.prompt().clone();
        let config = r1.output_config.clone().unwrap();
        assert!(config.format.is_some(), "constrained");
        let schema = serde_json::to_value(&config).unwrap()["format"]["schema"].clone();
        let order = cadence::prompt::order_for(cadence::prompt::seed_for(h.id));
        let options: Vec<&str> = schema["properties"]["choice"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(
            options,
            order.map(CadenceChoice::as_str),
            "this agent's order"
        );
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(
            serde_json::to_value(&config.effort).unwrap(),
            serde_json::to_value(&r0.output_config.as_ref().unwrap().effort).unwrap(),
            "effort kept"
        );
        let mut r1_without = r1.clone();
        r1_without.output_config = r0.output_config.clone();
        assert_prefix_kept(&r0, &r1_without, "blallama question");
    }
}
