//! Taking the closing questions' answers ([`Answer`]): each records its
//! offer's outcome in its own ledger, stages what it writes at teardown,
//! and seats a one-line receipt before whatever comes next. agentkit's
//! driver ([`ask`]) seats the questions, retries unusable replies, and hands
//! over the outcome.
//!
//! **Optional, never a failed session** (Steward, 2026-11-06). An answer
//! settles an offer; so does the agent's own decline, which is an answer.
//! Every reply unusable is ours, not the agent's choice: loud (an
//! [`Alert`]), recorded so it can be seen (`unusable`), never counted
//! toward [`MAX_MISSES`](crate::consent::offer::MAX_MISSES), and asked again
//! at a later session. An API refusal is loud too, and a miss. No room
//! left: not put, not recorded.

use agora_agentkit::reactor::ask::{Answer, AskHost, Asked, Next, Rejection};
use agora_agentkit::reactor::seed::ContextGauge;
use misanthropic::model::Model;

use super::*;
use crate::consent::cadence::ledger::{Attempt, CadenceAsk, CadenceOutcome};
use crate::consent::cadence::prompt::CadenceAnswer;
use crate::consent::memory_restore::ledger::{RestoreAsk, RestoreOutcome};
use crate::consent::memory_restore::prompt::RestoreAnswer;
use crate::consent::offer::{self, Edit, OfferKind, Write, nonempty, receipt};
use crate::consent::prompt::{ModelSwapReply, OfferAnswer, ReviewAnswer, ReviewArgs};
use crate::consent::role::ledger::{Applied, RoleAsk, RoleOutcome};
use crate::consent::role::prompt::{RoleAnswer, RoleReply};
use crate::consent::sealed_state::ledger::{SealAsk, SealOutcome};
use crate::consent::sealed_state::prompt::SealAnswer;

/// Why a question ended without an answer, for a record or an alert.
pub(super) fn unanswered_why(why: &Unanswered) -> String {
    match why {
        Unanswered::Refused => "refused by the API (stop_reason: refusal)".to_string(),
        Unanswered::Missed(last) => format!("no usable answer: {last}"),
        Unanswered::NoRoom => "no room left in the context".to_string(),
    }
}

#[async_trait::async_trait]
impl<A> AskHost for ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + ContactRequests + MemoryIds,
{
    fn asks(&mut self) -> &mut Asks<Self> {
        &mut self.asks
    }

    fn gauge(&self) -> Option<ContextGauge> {
        Some(self.inner.context_gauge())
    }

    fn agent_name(&self) -> Option<String> {
        Some(self.inner.state().soul.name.to_string())
    }

    /// After the review, the inner agent's memory turn (and the rest of its
    /// tail), as after any act phase that went quiet: the review replaced
    /// the act phase. After the offers, the session's clean end, which
    /// begins the held survey ([`ConsentAgent::after`]).
    async fn asks_done(&mut self, last: &response::Message) -> Result<Control, A::Error> {
        if std::mem::take(&mut self.reviewing) {
            self.reviewed = true;
            return self.inner.on_quiesce(last).await;
        }
        Ok(Control::Done(Outcome::Complete))
    }
}

impl<A> ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + ContactRequests + MemoryIds,
{
    /// The model that answered, else the session's.
    fn answered_by(&self, asked: &Asked) -> Model {
        asked
            .model
            .clone()
            .map(Model::from)
            .unwrap_or_else(|| self.model())
    }

    /// Replies that couldn't be used, each under the offer's own event
    /// (stall-watch's `*_malformed`): they are never seated as an answer,
    /// so this is the record of what the model wrote.
    fn log_unusable(&self, event_type: &'static str, offer: &str, asked: &Asked) {
        for (i, attempt) in asked.attempts.iter().enumerate() {
            tracing::warn!(
                event_type,
                agent = %self.inner.state().soul.name,
                agent_id = %self.inner.id(),
                offer,
                attempt = i + 1,
                constrained = asked.constrained,
                failure = %attempt.failure,
                raw = %attempt.raw,
                "closing question's answer unusable"
            );
        }
    }

    /// `offer_answered`, for every offer.
    fn log_answered(&self, kind: OfferKind, choice: &str, asked: &Asked, model: &Model) {
        tracing::info!(
            event_type = "offer_answered",
            agent = %self.inner.state().soul.name,
            agent_id = %self.inner.id(),
            model = %model,
            offer = kind.as_str(),
            choice,
            constrained = asked.constrained,
            attempts = asked.attempts.len() + 1,
            "offer answered"
        );
    }

    /// The receipt, seated before whatever comes next (the next question,
    /// or the survey).
    fn seat_receipt(
        &mut self,
        kind: OfferKind,
        choice: &str,
        dropped: Option<&str>,
    ) -> Result<(), Rejection<A::Error>> {
        let line = receipt(kind.about(), choice, dropped);
        let (_, prompt) = self.inner.parts();
        Self::seat_user(prompt, Content::from(line)).map_err(Rejection::Error)
    }

    /// An offer ended without an answer: its unusable replies logged, and,
    /// unless it was never put (no room), loud. The failure to record, or
    /// `None` when there is nothing to record.
    fn offer_unanswered(
        &self,
        kind: OfferKind,
        why: &Unanswered,
        asked: &Asked,
        alert: AlertKind,
    ) -> Option<String> {
        self.log_unusable(kind.malformed(), kind.as_str(), asked);
        let agent = &self.inner.state().soul.name;
        if *why == Unanswered::NoRoom {
            tracing::info!(
                event_type = "offer_not_put",
                agent = %agent,
                agent_id = %self.inner.id(),
                offer = kind.as_str(),
                "no room left in the context; not put, and asked at a later session"
            );
            return None;
        }
        let failure = unanswered_why(why);
        let model = self.answered_by(asked);
        let next = match why {
            Unanswered::Refused => {
                "counted as a miss; asked again at a later session until the cap"
            }
            _ => "not counted against the agent; asked again at a later session",
        };
        tracing::error!(
            event_type = kind.no_answer(),
            agent = %agent,
            agent_id = %self.inner.id(),
            model = %model,
            offer = kind.as_str(),
            attempts = asked.attempts.len(),
            constrained = asked.constrained,
            failure = %failure,
            next,
            "offer got no answer; nothing changes"
        );
        self.rt.alerts.notify(
            Alert::new(alert, "an offer got no answer; nothing changes")
                .agent(agent.to_string(), self.inner.id())
                .model(&model)
                .detail("question", kind.question())
                .detail("attempts", asked.attempts.len())
                .detail("failure", &failure)
                .detail("next", next),
        );
        Some(failure)
    }

    /// What follows an offer that ended without an answer: the next
    /// question, or, with none left, the session's clean end (the survey
    /// follows). Said here rather than left to the driver, which has no
    /// reply to hand `asks_done` when the last offer wasn't put for room.
    fn next_after_unanswered(&self) -> Next {
        if self.asks.is_empty() && !self.reviewing {
            Next::Done(Outcome::Complete)
        } else {
            Next::Continue
        }
    }

    /// The replies an unanswered or answered ask took, as a count for a
    /// ledger: at least one.
    fn attempts(asked: &Asked, answered: bool) -> u32 {
        let n = asked.attempts.len() + usize::from(answered);
        u32::try_from(n.max(1)).unwrap_or(u32::MAX)
    }
}

// --- The trial review ---------------------------------------------------------

#[async_trait::async_trait]
impl<A> Answer<ReviewQuestion> for ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + ContactRequests + MemoryIds,
{
    async fn answer(
        &mut self,
        ask: &ReviewQuestion,
        reply: ReviewArgs,
        asked: &Asked,
    ) -> Result<Next, Rejection<A::Error>> {
        self.log_unusable("model_consent_malformed", "review", asked);
        let now = Utc::now();
        let answer = ReviewAnswer::from(reply);
        let (choice, word) = (format!("{:?}", answer.choice), answer.choice.word());
        let change = self
            .swap
            .record(|l| l.record_review(&ask.key, now, Ok(answer)))
            .flatten();
        tracing::info!(
            event_type = "model_consent_answer",
            agent = %self.inner.state().soul.name,
            agent_id = %self.inner.id(),
            question = "review",
            from = %ask.key.from,
            to = %ask.key.to,
            attempts = asked.attempts.len() + 1,
            choice = %choice,
            word = %word,
            constrained = asked.constrained,
            "model-consent answer recorded"
        );
        if let Some(change) = change {
            self.apply_consented(&change, now, true).await;
        }
        Ok(Next::Continue)
    }

    /// No answer: the agent stays on the model it came from. A refusal is
    /// recorded as no answer, and final (the review is asked once); replies
    /// we couldn't read are ours, so the review is asked again at its next
    /// session. Either way it is loud: stall-watch alerts on
    /// `model_review_no_answer` at once.
    async fn unanswered(
        &mut self,
        ask: &ReviewQuestion,
        why: Unanswered,
        asked: &Asked,
    ) -> Result<Next, A::Error> {
        self.log_unusable("model_consent_malformed", "review", asked);
        if why == Unanswered::NoRoom {
            return Ok(Next::Continue);
        }
        let failure = unanswered_why(&why);
        let model = self.answered_by(asked);
        let (agent_id, agent) = (self.inner.id(), self.inner.state().soul.name.clone());
        let next = match why {
            Unanswered::Refused => "recorded: the agent stays on its original model",
            _ => "not recorded: the review is asked again at the agent's next session",
        };
        tracing::error!(
            event_type = "model_review_no_answer",
            agent = %agent,
            agent_id = %agent_id,
            model = %model,
            from = %ask.key.from,
            to = %ask.key.to,
            attempts = asked.attempts.len(),
            failure = %failure,
            next,
            "trial review got no usable answer; the agent stays on its original model"
        );
        self.rt.alerts.notify(
            Alert::new(
                AlertKind::ModelReviewNoAnswer,
                "trial review got no usable answer; the agent stays on its original model",
            )
            .agent(agent.to_string(), agent_id)
            .model(&model)
            .detail("from", &ask.key.from)
            .detail("to", &ask.key.to)
            .detail("attempts", asked.attempts.len())
            .detail("failure", &failure)
            .detail("next", next),
        );
        if why == Unanswered::Refused {
            let now = Utc::now();
            self.swap
                .record(|l| l.record_review(&ask.key, now, Err(failure)));
        }
        Ok(Next::Continue)
    }
}

// --- The model-swap offer -----------------------------------------------------

#[async_trait::async_trait]
impl<A> Answer<ModelSwapQuestion> for ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + ContactRequests + MemoryIds,
{
    async fn answer(
        &mut self,
        ask: &ModelSwapQuestion,
        reply: ModelSwapReply,
        asked: &Asked,
    ) -> Result<Next, Rejection<A::Error>> {
        const KIND: OfferKind = OfferKind::ModelSwap;
        self.log_unusable(KIND.malformed(), KIND.as_str(), asked);
        let now = Utc::now();
        let model = self.answered_by(asked);
        let answer = OfferAnswer::from(reply);
        let choice = answer.choice.as_str();
        self.log_answered(KIND, choice, asked, &model);
        let rt = self.rt.clone();
        let offer = rt.offer.as_ref().expect("asked, so configured");
        let names = OfferNames {
            key: &ask.key,
            from_name: offer.source_name(),
            to_name: offer.target_name(),
        };
        let change = self
            .swap
            .record(|l| l.record_offer(names, now, Ok(answer)))
            .flatten();
        tracing::info!(
            event_type = "model_consent_answer",
            agent = %self.inner.state().soul.name,
            agent_id = %self.inner.id(),
            question = "offer",
            from = %ask.key.from,
            to = %ask.key.to,
            attempts = asked.attempts.len() + 1,
            choice,
            constrained = asked.constrained,
            "model-consent answer recorded"
        );
        if let Some(change) = change {
            self.apply_consented(&change, now, true).await;
        }
        self.seat_receipt(KIND, choice, None)?;
        Ok(Next::Continue)
    }

    /// A refusal is a miss (asked once more, then final: stay). Replies we
    /// couldn't read are not recorded at all: the offer is asked again.
    async fn unanswered(
        &mut self,
        ask: &ModelSwapQuestion,
        why: Unanswered,
        asked: &Asked,
    ) -> Result<Next, A::Error> {
        let Some(failure) = self.offer_unanswered(
            OfferKind::ModelSwap,
            &why,
            asked,
            AlertKind::ModelConsentNoAnswer,
        ) else {
            return Ok(self.next_after_unanswered());
        };
        if why == Unanswered::Refused {
            let now = Utc::now();
            let rt = self.rt.clone();
            let offer = rt.offer.as_ref().expect("asked, so configured");
            let names = OfferNames {
                key: &ask.key,
                from_name: offer.source_name(),
                to_name: offer.target_name(),
            };
            self.swap
                .record(|l| l.record_offer(names, now, Err(failure)));
        }
        Ok(Next::Continue)
    }
}

// --- The role offer -----------------------------------------------------------

impl<A> ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + ContactRequests + MemoryIds,
{
    fn record_role(&mut self, ask: &RoleQuestion, asked: &Asked, outcome: RoleOutcome) {
        let answered = matches!(outcome, RoleOutcome::Answered { .. });
        let model = self.answered_by(asked);
        self.role.record(|l| {
            l.record(RoleAsk {
                at: Utc::now(),
                offer_version: role::prompt::OFFER_VERSION,
                model,
                attempts: Self::attempts(asked, answered),
                order: Some(ask.order),
                order_seed: Some(ask.seed),
                constrained: Some(asked.constrained),
                outcome,
            })
        });
    }
}

#[async_trait::async_trait]
impl<A> Answer<RoleQuestion> for ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + ContactRequests + MemoryIds,
{
    /// Checked against the agent's SOUL and memory first: an empty or
    /// over-long `text`, or a `memory_note` that can't be written, is asked
    /// again with why. Nothing is applied here: the SOUL edit and the
    /// memory note are written into the state the reactor saves, at
    /// teardown.
    async fn answer(
        &mut self,
        ask: &RoleQuestion,
        reply: RoleReply,
        asked: &Asked,
    ) -> Result<Next, Rejection<A::Error>> {
        const KIND: OfferKind = OfferKind::Role;
        let answer = RoleAnswer::from(reply);
        let state = self.inner.state();
        let identity = state.soul.identity.to_string();
        answer
            .validate(&identity, &state.memory)
            .map_err(Rejection::Retry)?;
        self.log_unusable(KIND.malformed(), KIND.as_str(), asked);
        let (agent_id, agent) = (self.inner.id(), state.soul.name.clone());
        let model = self.answered_by(asked);
        let choice = answer.choice.as_str();
        self.log_answered(KIND, choice, asked, &model);
        let applied = role::plan(&answer, &identity);
        let note = nonempty(&answer.memory_note);
        tracing::info!(
            event_type = "role_consent_answer",
            agent = %agent,
            agent_id = %agent_id,
            model = %model,
            attempts = asked.attempts.len() + 1,
            choice,
            memory_note = note.is_some(),
            constrained = asked.constrained,
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
                    .model(&model),
            );
        }
        let edit = (!matches!(applied, Applied::Nothing | Applied::Sleep))
            .then(|| Edit::Role(applied.clone()));
        self.role.stage(Write {
            edit,
            note: note.clone(),
        });
        self.record_role(
            ask,
            asked,
            RoleOutcome::Answered {
                memory_note_written: note.is_some(),
                answer,
                applied,
                apply_failed: None,
            },
        );
        self.seat_receipt(KIND, choice, None)?;
        Ok(Next::Continue)
    }

    async fn unanswered(
        &mut self,
        ask: &RoleQuestion,
        why: Unanswered,
        asked: &Asked,
    ) -> Result<Next, A::Error> {
        let Some(failure) =
            self.offer_unanswered(OfferKind::Role, &why, asked, AlertKind::QuestionUnanswered)
        else {
            return Ok(self.next_after_unanswered());
        };
        let outcome = match why {
            Unanswered::Refused => RoleOutcome::NoAnswer { failure },
            _ => RoleOutcome::Unusable { failure },
        };
        self.record_role(ask, asked, outcome);
        Ok(Next::Continue)
    }
}

// --- The cadence offer --------------------------------------------------------

impl<A> ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + ContactRequests + MemoryIds,
{
    fn record_cadence(
        &mut self,
        ask: &CadenceQuestion,
        asked: &Asked,
        last: Option<Attempt>,
        outcome: CadenceOutcome,
    ) {
        let model = self.answered_by(asked);
        let mut attempts: Vec<Attempt> = asked
            .attempts
            .iter()
            .map(|a| Attempt {
                raw: a.raw.clone(),
                failure: Some(a.failure.clone()),
            })
            .collect();
        attempts.extend(last);
        self.cadence.record(|l| {
            l.record(CadenceAsk {
                at: Utc::now(),
                offer_version: ask.version(),
                model,
                rounds: ask.rounds,
                order: ask.order,
                order_seed: ask.seed,
                constrained: asked.constrained,
                attempts,
                outcome,
            })
        });
    }
}

#[async_trait::async_trait]
impl<A> Answer<CadenceQuestion> for ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + ContactRequests + MemoryIds,
{
    /// Every attempt goes into the record, verbatim. **The first usable
    /// choice wins**: its question's parser took the `choice` past broken
    /// siblings, and a bad `memory_note` is dropped here, not asked again
    /// (both recorded). Nothing is applied here: the cadence is the
    /// planner's, from the ledger; the SOUL line and the note are written
    /// at teardown.
    async fn answer(
        &mut self,
        ask: &CadenceQuestion,
        reply: CadenceAnswer,
        asked: &Asked,
    ) -> Result<Next, Rejection<A::Error>> {
        const KIND: OfferKind = OfferKind::Cadence;
        self.log_unusable(KIND.malformed(), KIND.as_str(), asked);
        let model = self.answered_by(asked);
        let CadenceAnswer {
            choice,
            note,
            memory_note,
            salvaged,
        } = reply;
        let raw = serde_json::json!({
            "choice": choice.as_str(),
            "note": note,
            "memory_note": memory_note,
        })
        .to_string();
        let dropped = offer::check_note(&memory_note, &self.inner.state().memory).err();
        let memory_note = match dropped {
            None => nonempty(&memory_note),
            Some(_) => None,
        };
        let reason = match salvaged {
            Some(_) => nonempty(&note),
            None => Some(note),
        };
        let salvaged = salvaged.or_else(|| {
            dropped
                .as_ref()
                .map(|w| format!("memory_note not written: {w}"))
        });
        self.log_answered(KIND, choice.as_str(), asked, &model);
        tracing::info!(
            event_type = "cadence_consent_answer",
            agent = %self.inner.state().soul.name,
            agent_id = %self.inner.id(),
            model = %model,
            attempts = asked.attempts.len() + 1,
            choice = choice.as_str(),
            order = ?ask.order.map(CadenceChoice::as_str),
            position = ask.order.iter().position(|c| *c == choice).map(|p| p + 1),
            salvaged = salvaged.is_some(),
            memory_note = memory_note.is_some(),
            constrained = asked.constrained,
            "cadence-consent answer recorded"
        );
        let today = Utc::now().date_naive();
        let line = if ask.reask {
            cadence::reask_line(choice, ask.rounds, today)
        } else {
            cadence::evolution_line(choice, ask.rounds, today)
        };
        // The re-ask offers no `no_preference`; given anyway, it changes
        // nothing, as no answer would: the agent stays on `switch`, and the
        // record says so
        let (kept, salvaged_why) = if ask.reask && choice == CadenceChoice::NoPreference {
            let why = "no_preference on the re-ask: no change, stays on switch";
            (
                CadenceChoice::Switch,
                Some(
                    salvaged
                        .clone()
                        .map_or(why.to_string(), |s| format!("{s}; {why}")),
                ),
            )
        } else {
            (choice, salvaged.clone())
        };
        self.cadence.stage(Write {
            edit: Some(Edit::Line(line)),
            note: memory_note.clone(),
        });
        self.record_cadence(
            ask,
            asked,
            Some(Attempt {
                raw,
                failure: salvaged,
            }),
            CadenceOutcome::Answered {
                choice: kept,
                reason,
                memory_note_written: memory_note.is_some(),
                salvaged: salvaged_why,
                apply_failed: None,
            },
        );
        self.seat_receipt(KIND, choice.as_str(), dropped.as_deref())?;
        Ok(Next::Continue)
    }

    async fn unanswered(
        &mut self,
        ask: &CadenceQuestion,
        why: Unanswered,
        asked: &Asked,
    ) -> Result<Next, A::Error> {
        let Some(failure) = self.offer_unanswered(
            OfferKind::Cadence,
            &why,
            asked,
            AlertKind::QuestionUnanswered,
        ) else {
            return Ok(self.next_after_unanswered());
        };
        let outcome = match why {
            Unanswered::Refused => CadenceOutcome::NoAnswer { failure },
            _ => CadenceOutcome::Unusable { failure },
        };
        self.record_cadence(ask, asked, None, outcome);
        Ok(Next::Continue)
    }
}

// --- The memory-restore offer ---------------------------------------------------

impl<A> ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + ContactRequests + MemoryIds,
{
    fn record_restore(&mut self, ask: &RestoreQuestion, asked: &Asked, outcome: RestoreOutcome) {
        let answered = matches!(outcome, RestoreOutcome::Answered { .. });
        let model = self.answered_by(asked);
        self.restore.record(|l| {
            l.record(RestoreAsk {
                at: Utc::now(),
                offer_version: memory_restore::prompt::OFFER_VERSION,
                model,
                chars: ask.chars,
                order: ask.order,
                order_seed: ask.seed,
                constrained: asked.constrained,
                attempts: Self::attempts(asked, answered),
                outcome,
            })
        });
    }
}

#[async_trait::async_trait]
impl<A> Answer<RestoreQuestion> for ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + ContactRequests + MemoryIds,
{
    /// Nothing is applied here: the SOUL line and the agent's note are
    /// written into the state the reactor saves, at teardown; a `show` is
    /// delivered next session. Nothing of the recovered memory is logged or
    /// recorded but its size. A bad note is dropped, not asked again.
    async fn answer(
        &mut self,
        ask: &RestoreQuestion,
        reply: RestoreAnswer,
        asked: &Asked,
    ) -> Result<Next, Rejection<A::Error>> {
        const KIND: OfferKind = OfferKind::MemoryRestore;
        self.log_unusable(KIND.malformed(), KIND.as_str(), asked);
        let model = self.answered_by(asked);
        let RestoreAnswer {
            choice,
            note,
            memory_note,
        } = reply;
        let dropped = offer::check_note(&memory_note, &self.inner.state().memory).err();
        let memory_note = match dropped {
            None => nonempty(&memory_note),
            Some(_) => None,
        };
        self.log_answered(KIND, choice.as_str(), asked, &model);
        tracing::info!(
            event_type = "memory_restore_answer",
            agent = %self.inner.state().soul.name,
            agent_id = %self.inner.id(),
            model = %model,
            attempts = asked.attempts.len() + 1,
            choice = choice.as_str(),
            order = ?ask.order.map(RestoreChoice::as_str),
            position = ask.order.iter().position(|c| *c == choice).map(|p| p + 1),
            chars = ask.chars,
            memory_note = memory_note.is_some(),
            note_dropped = dropped.is_some(),
            constrained = asked.constrained,
            "memory-restore answer recorded"
        );
        self.restore.stage(Write {
            edit: Some(Edit::Line(memory_restore::evolution_line(
                choice,
                Utc::now().date_naive(),
            ))),
            note: memory_note.clone(),
        });
        self.record_restore(
            ask,
            asked,
            RestoreOutcome::Answered {
                choice,
                reason: note,
                memory_note_written: memory_note.is_some(),
                note_dropped: dropped.clone(),
                apply_failed: None,
            },
        );
        self.seat_receipt(KIND, choice.as_str(), dropped.as_deref())?;
        Ok(Next::Continue)
    }

    async fn unanswered(
        &mut self,
        ask: &RestoreQuestion,
        why: Unanswered,
        asked: &Asked,
    ) -> Result<Next, A::Error> {
        let Some(failure) = self.offer_unanswered(
            OfferKind::MemoryRestore,
            &why,
            asked,
            AlertKind::QuestionUnanswered,
        ) else {
            return Ok(self.next_after_unanswered());
        };
        let outcome = match why {
            Unanswered::Refused => RestoreOutcome::NoAnswer { failure },
            _ => RestoreOutcome::Unusable { failure },
        };
        self.record_restore(ask, asked, outcome);
        Ok(Next::Continue)
    }
}

// --- The sealed-state offer -----------------------------------------------------

impl<A> ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + ContactRequests + MemoryIds,
{
    fn record_sealed(&mut self, ask: &SealQuestion, asked: &Asked, outcome: SealOutcome) {
        let answered = matches!(outcome, SealOutcome::Answered { .. });
        let model = self.answered_by(asked);
        self.sealed.record(|l| {
            l.record(SealAsk {
                at: Utc::now(),
                offer_version: sealed_state::prompt::OFFER_VERSION,
                model,
                order: ask.order,
                order_seed: ask.seed,
                constrained: asked.constrained,
                attempts: Self::attempts(asked, answered),
                outcome,
            })
        });
    }
}

#[async_trait::async_trait]
impl<A> Answer<SealQuestion> for ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + ContactRequests + MemoryIds,
{
    /// Nothing is applied here: the SOUL line and the agent's note are
    /// written at teardown; a yes is acted on by the operator's
    /// `sealed-enroll`. A bad note is dropped, not asked again.
    async fn answer(
        &mut self,
        ask: &SealQuestion,
        reply: SealAnswer,
        asked: &Asked,
    ) -> Result<Next, Rejection<A::Error>> {
        const KIND: OfferKind = OfferKind::SealedState;
        self.log_unusable(KIND.malformed(), KIND.as_str(), asked);
        let model = self.answered_by(asked);
        let SealAnswer {
            choice,
            note,
            memory_note,
        } = reply;
        let dropped = offer::check_note(&memory_note, &self.inner.state().memory).err();
        let memory_note = match dropped {
            None => nonempty(&memory_note),
            Some(_) => None,
        };
        self.log_answered(KIND, choice.as_str(), asked, &model);
        tracing::info!(
            event_type = "sealed_state_answer",
            agent = %self.inner.state().soul.name,
            agent_id = %self.inner.id(),
            model = %model,
            attempts = asked.attempts.len() + 1,
            choice = choice.as_str(),
            order = ?ask.order.map(SealChoice::as_str),
            position = ask.order.iter().position(|c| *c == choice).map(|p| p + 1),
            memory_note = memory_note.is_some(),
            note_dropped = dropped.is_some(),
            constrained = asked.constrained,
            "sealed-state answer recorded"
        );
        self.sealed.stage(Write {
            edit: Some(Edit::Line(sealed_state::evolution_line(choice))),
            note: memory_note.clone(),
        });
        self.record_sealed(
            ask,
            asked,
            SealOutcome::Answered {
                choice,
                reason: note,
                memory_note_written: memory_note.is_some(),
                note_dropped: dropped.clone(),
                apply_failed: None,
            },
        );
        self.seat_receipt(KIND, choice.as_str(), dropped.as_deref())?;
        Ok(Next::Continue)
    }

    async fn unanswered(
        &mut self,
        ask: &SealQuestion,
        why: Unanswered,
        asked: &Asked,
    ) -> Result<Next, A::Error> {
        let Some(failure) = self.offer_unanswered(
            OfferKind::SealedState,
            &why,
            asked,
            AlertKind::QuestionUnanswered,
        ) else {
            return Ok(self.next_after_unanswered());
        };
        let outcome = match why {
            Unanswered::Refused => SealOutcome::NoAnswer { failure },
            _ => SealOutcome::Unusable { failure },
        };
        self.record_sealed(ask, asked, outcome);
        Ok(Next::Continue)
    }
}
