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
//! `set_model` tool ([`switch::SetModel`]) before the inner init, when the
//! run has a selectable model to offer. One change per session; nothing is
//! asked after one. Each post or comment the session wrote is logged as a
//! `write_recorded` event at teardown, with the dump's `prompt_sha256`.

use std::sync::Arc;

use std::collections::HashSet;

use agora_agentkit::ids::{AgentId, CommentId, PostId};
use agora_agentkit::reactor::inference::Quirks;
use agora_agentkit::reactor::seed::SeedState;
use agora_agentkit::reactor::{Agent, Control, Outcome};
use chrono::{DateTime, Utc};
use misanthropic::model::ModelInfo;
use misanthropic::prompt::Prompt;
use misanthropic::prompt::message::{Block, Content, Role};
use misanthropic::prompt::output::OutputConfig;
use misanthropic::response::{self, JsonError, StopReason};
use misanthropic::tool::{Notifications, ToolBox};

use super::ConsentRuntime;
use super::comparison;
use super::forks;
use super::ledger::{Change, Due, Ledger, OfferKey, OfferNames, Switch, SwitchCause};
use super::prompt::{self as text, OfferText, ReviewText};
use super::queue::{self, QueueEntry};
use super::switch::{self, SetModel, SwitchError};
use crate::alerts::{Alert, AlertKind};

/// Answers per question per session: the first plus two retries.
pub const MAX_ATTEMPTS: u32 = 3;

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
        let cache_safe = self
            .inner
            .quirks()
            .unwrap_or_default()
            .output_config_cache_safe;
        let max_tokens = self.rt.max_tokens;
        let (_, prompt) = self.inner.parts();
        prompt.max_tokens = std::num::NonZeroU32::new(max_tokens).expect("validated nonzero");
        // Keep the session's effort (agentkit 0.39 `thinking_effort`):
        // thinking stays adaptive, and without it the answer would think at
        // the model's default.
        let effort = prompt.output_config.as_ref().and_then(|c| c.effort.clone());
        prompt.output_config = match (cache_safe, effort) {
            (true, Some(effort)) => Some(OutputConfig::json_schema(schema).with_effort(effort)),
            (true, None) => Some(OutputConfig::json_schema(schema)),
            (false, effort) => effort.map(OutputConfig::effort),
        };
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

    /// The inner session completed cleanly: count it toward any trial, end
    /// a trial that has run its course, retry a change still waiting, or
    /// ask a due offer. `None` means nothing is asked — the session ends.
    async fn close(&mut self) -> Result<Option<Control>, A::Error> {
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
        let Some(ledger) = self.ledger.as_mut() else {
            return Ok(None);
        };
        self.dirty |= ledger.count_session(&model);
        self.dirty |= ledger.count_since_switch(started);
        // One change per session, and one question: nothing after a switch
        // or a review.
        if switched || self.reviewed {
            return Ok(None);
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
            return Ok(None);
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
            return Ok(None);
        }
        let Some(due) = ledger.due(&model, offer_key.as_ref()) else {
            return Ok(None);
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
            .map(Some)
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
        let ledger = self.ledger.as_mut()?;
        let offers_touched = ledger
            .offers
            .iter()
            .any(|r| r.history.iter().any(|e| e.at >= started));
        if !offers_touched
            && self.notes.is_empty()
            && self.next_model.is_none()
            && self.seen_before_review.is_none()
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
                return None;
            }
        };
        let mut changed = false;
        if offers_touched && ledger.update_soul(&mut state.soul, started, Utc::now().date_naive()) {
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
        changed.then_some(state)
    }
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

    fn on_admit(&mut self, model: &ModelInfo, quirks: &Quirks) {
        self.inner.on_admit(model, quirks)
    }

    fn quirks(&self) -> Option<Quirks> {
        self.inner.quirks()
    }

    fn prime_prompt(&self) -> Option<Prompt> {
        self.inner.prime_prompt()
    }

    /// The ledger first — load it and note any change applied since last
    /// session — then `set_model` into the toolbox, then the inner init,
    /// which seats the tools.
    async fn on_init(&mut self) -> Result<(), A::Error> {
        self.started = Utc::now();
        let dir = self.agent_dir();
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
                // no act phase, so no `set_model` either.
                self.review = ledger.review_due(&model);
                let tool = SetModel::new(
                    self.rt.clone(),
                    self.inner.id(),
                    self.inner.state().soul.name.to_string(),
                    model,
                    ledger.switch_blocker(),
                    self.slot.clone(),
                )
                .filter(|_| self.review.is_none());
                if let Some(tool) = tool {
                    self.inner.parts().0.push(tool);
                }
                self.ledger = Some(ledger);
            }
            // No ledger, no `set_model`: the cooldown can't be checked.
            Err(e) => tracing::warn!(
                agent_id = %self.inner.id(),
                path = %Ledger::path(&dir).display(),
                error = %e,
                "model-consent ledger unreadable; not asking or saving this session"
            ),
        }
        if self.review.is_some() {
            self.seen_before_review = Some(self.inner.state().seen_posts.clone());
        }
        self.inner.on_init().await?;
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

    /// Stands in for `SeedAgent`: its whole session is one response, after
    /// which its closing phase is done.
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
            state
                .prompt
                .push_message((Role::User, "Your dashboard."))
                .unwrap();
            Ok(Self {
                id,
                state,
                tools: ToolBox::flat(),
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
                .unwrap(),
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
        assert!(agent.parts().0.definitions().is_empty());
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
    /// after the intro, with no `set_model`.
    async fn review_session(h: &Harness) -> ConsentAgent<Fake> {
        use misanthropic::tool::Tool;
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        assert_eq!(agent.prompt().messages.len(), 1, "intro + review, one turn");
        assert!(agent.parts().0.definitions().is_empty(), "no set_model");
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
}
