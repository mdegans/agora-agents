//! Taking the closing questions' answers ([`Answer`]): each records its
//! offer's outcome in its own ledger, stages what it writes at teardown,
//! and seats a one-line receipt before whatever comes next. agentkit's
//! driver ([`ask`]) seats the questions, retries unusable replies, and hands
//! over the outcome.
//!
//! **Optional, never a failed session by themselves** (Steward,
//! 2026-11-06). An answer settles an offer; so does the agent's own
//! decline, which is an answer. A session whose every reply was unusable is
//! almost always ours (the grammar, the template, the parser, a check the
//! answer failed), not the agent's choice: loud (an [`Alert`]), recorded
//! (`unusable`), and asked again at a later session; [`UNUSABLE_CAP`] of
//! them count as one miss (an offer) or settle it (the review), louder
//! still. No room left: not put, not recorded.
//!
//! **An API refusal** (`stop_reason: refusal`) ends the session as failed,
//! whatever the question (the driver does that), and mails the Steward
//! ([`AlertKind::ApiRefusal`], no content). Nothing is recorded for the
//! question it ended: it is asked again at a later session.
//!
//! [`UNUSABLE_CAP`]: crate::consent::offer::UNUSABLE_CAP

use agora_agentkit::reactor::ask::{Answer, AskHost, Asked, Mark, Next, Rejection};
use agora_agentkit::reactor::seed::ContextGauge;
use misanthropic::model::Model;

use super::*;
use crate::consent::cadence::ledger::{Attempt, CadenceAsk, CadenceOutcome};
use crate::consent::cadence::prompt::CadenceAnswer;
use crate::consent::memory_restore::ledger::{RestoreAsk, RestoreOutcome};
use crate::consent::offer::{
    self, Choice, Edit, NoteFate, OfferKind, OfferLedger, Reply, Status, Write, nonempty, receipt,
};
use crate::consent::prompt::{ModelSwapReply, OfferAnswer, ReviewAnswer, ReviewArgs};
use crate::consent::questions::OfferQuestion;
use crate::consent::role::ledger::{Applied, RoleAsk, RoleOutcome};
use crate::consent::role::prompt::{RoleAnswer, RoleReply};
use crate::consent::sealed_state::ledger::{SealAsk, SealOutcome};

/// Why a question ended without an answer, for a record or an alert.
pub(super) fn unanswered_why(why: &Unanswered) -> String {
    match why {
        Unanswered::Refused { .. } => "refused by the API (stop_reason: refusal)".to_string(),
        Unanswered::Missed(last) => format!("no usable answer: {last}"),
        Unanswered::NoRoom => "no room left in the context".to_string(),
    }
}

#[async_trait::async_trait]
impl<A> AskHost for ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + InnerSession + MemoryIds,
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

    /// The inner session archives it with its own (the driver never calls
    /// this for an anonymous answer).
    fn left_out(&mut self, ask: &'static str, attempt: usize, reply: &response::Message) {
        self.inner.left_out(ask, attempt, reply);
    }

    /// After the review, the inner agent's memory turn (and the rest of its
    /// tail), as after any act phase that went quiet: the review replaced
    /// the act phase. A review never put (no room at the session's start:
    /// no reply) leaves an ordinary acting session; it is asked at a later
    /// one. After the offers, the session's clean end, which begins the
    /// held survey ([`ConsentAgent::after`]).
    async fn asks_done(&mut self, last: Option<&response::Message>) -> Result<Control, A::Error> {
        if std::mem::take(&mut self.reviewing) {
            let Some(last) = last else {
                // The feed is read in this session after all
                self.seen_before_review = None;
                return Ok(Control::Continue);
            };
            self.reviewed = true;
            return self.inner.on_quiesce(last).await;
        }
        Ok(Control::Done(Outcome::Complete))
    }
}

impl<A> ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + InnerSession + MemoryIds,
{
    /// The model that answered, else the session's.
    fn answered_by(&self, asked: &Asked) -> Model {
        asked
            .model
            .clone()
            .map(Model::from)
            .unwrap_or_else(|| self.model())
    }

    /// Replies that couldn't be used, each under its question's own event
    /// (stall-watch's `*_malformed`), with the question in `field` (`offer`
    /// for an offer, `question` for the review): the failure's kind only,
    /// never the reply (Steward, 2026-11-06: logs minimal; the reply is in
    /// the saved transcript and the prompt-log archive).
    fn log_unusable(&self, event_type: &'static str, field: Field, name: &str, asked: &Asked) {
        let model = asked.model.as_deref();
        for (i, attempt) in asked.attempts.iter().enumerate() {
            macro_rules! unusable {
                ($field:ident) => {
                    tracing::warn!(
                        event_type,
                        agent = %self.inner.state().soul.name,
                        agent_id = %self.inner.id(),
                        model,
                        $field = name,
                        attempt = i + 1,
                        constrained = asked.constrained,
                        failure = %attempt.failure,
                        "closing question's answer unusable"
                    )
                };
            }
            match field {
                Field::Offer => unusable!(offer),
                Field::Question => unusable!(question),
            }
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
        note: NoteFate<'_>,
    ) -> Result<(), Rejection<A::Error>> {
        self.seat_receipt_with(kind, choice, note, None)
    }

    /// [`Self::seat_receipt`], with a line of the offer's own after it.
    fn seat_receipt_with(
        &mut self,
        kind: OfferKind,
        choice: &str,
        note: NoteFate<'_>,
        extra: Option<&str>,
    ) -> Result<(), Rejection<A::Error>> {
        let mut line = receipt(kind.about(), choice, note);
        if let Some(extra) = extra {
            line.push(' ');
            line.push_str(extra);
        }
        let (_, prompt) = self.inner.parts();
        Self::seat_user(prompt, Content::from(line)).map_err(Rejection::Error)
    }

    /// The API refused a reply to one of this wrapper's questions: the
    /// driver ends the session. Recorded on the inner agent, which would
    /// otherwise call the failure an error, and mailed from teardown with
    /// the inner session's own (once per refusal).
    fn refused_here(&mut self, during: &str, model: &Model, response_id: &str) {
        self.inner.record_failure(SessionFailure::Refused {
            during: during.to_string(),
            round: None,
            model: model.to_string(),
            response_id: response_id.to_string(),
        });
    }

    /// The sealed-state answer about to be recorded, when it is not the
    /// agent's first: to its operator's re-offer, or (defensively) with
    /// none pending, which the offer's asked-once rule should make
    /// impossible. Either way the operator is alerted with both answers and
    /// what would act on the new one, and nothing changes here: no marker
    /// is set or cleared in a session. The line added to the agent's
    /// receipt, for a re-offer.
    pub(super) fn sealed_answered_again(&self, choice: SealChoice) -> Option<&'static str> {
        let ledger = self.sealed.ledger.as_ref()?;
        let reoffered = ledger.reoffer.is_some();
        let was = match &ledger.reoffer {
            Some(r) => r.reopens,
            None if ledger.unprompted_answer() => ledger.choice()?,
            None => return None,
        };
        let id = self.inner.id();
        let name = self.inner.state().soul.name.to_string();
        let dir = self.agent_dir();
        let keys = crate::sealed::KeyStore::new(&self.rt.secrets_dir);
        let marked = self.marked_uploading();
        let stopped = crate::sealed::stop::is_stopped(&dir)
            || crate::sealed::stop::durable(&keys, id).map_or(true, |d| d.is_some());
        let yes = choice != SealChoice::Decline;
        let act = match (yes, marked, stopped) {
            (true, false, _) => format!("agora-seed sealed-enroll --opt-in --agent {id}"),
            (true, true, true) => format!("agora-seed sealed-enroll --agent {id} (resumes it)"),
            (false, true, false) => format!("agora-seed sealed-stop --agent {id} --note \"…\""),
            _ => "nothing: its uploads already match the answer".to_string(),
        };
        if reoffered {
            tracing::warn!(
                event_type = "sealed_state_reoffer_answered",
                agent = %name,
                agent_id = %id,
                was = was.as_str(),
                now = choice.as_str(),
                "an agent answered its operator's re-offer; its operator is alerted to act"
            );
        } else {
            tracing::error!(
                event_type = "sealed_state_unprompted_answer",
                agent = %name,
                agent_id = %id,
                was = was.as_str(),
                now = choice.as_str(),
                "a second sealed-state answer with no re-offer pending; recorded, nothing acts on it"
            );
        }
        self.rt.alerts.notify(
            Alert::new(
                AlertKind::SealedStateAnsweredAgain,
                if reoffered {
                    "an agent answered the sealed-state question its operator re-opened"
                } else {
                    "an agent answered the sealed-state offer a second time with no re-offer: a bug"
                },
            )
            .agent(name, id)
            .detail("was", was.as_str())
            .detail("now", choice.as_str())
            .detail("reoffered", reoffered)
            .detail("act", act),
        );
        reoffered.then_some(
            "Your operator has been told, and acts on your answer; you'll be told when it has. \
             Nothing changes by this answer alone.",
        )
    }

    /// The mail for an API refusal, with no content.
    pub(super) fn refusal_alert(
        &self,
        during: &str,
        round: Option<usize>,
        model: &str,
        response_id: &str,
    ) -> Alert {
        let agent = &self.inner.state().soul.name;
        let mut alert = Alert::new(
            AlertKind::ApiRefusal,
            "the API refused a turn; the session ended as failed",
        )
        .agent(agent.to_string(), self.inner.id())
        .model(model)
        .detail("during", during)
        .detail("response_id", response_id);
        if let Some(round) = round {
            alert = alert.detail("round", round);
        }
        alert
    }

    /// An unusable session completed an [`UNUSABLE_CAP`](offer::UNUSABLE_CAP).
    fn alert_cap(&self, question: &str, asked: &Asked, failure: &str, then: &str) {
        let agent = &self.inner.state().soul.name;
        let model = self.answered_by(asked);
        tracing::error!(
            event_type = "question_unusable_cap",
            agent = %agent,
            agent_id = %self.inner.id(),
            model = %model,
            question,
            cap = offer::UNUSABLE_CAP,
            failure,
            then,
            "a question's answers were unusable in a row of sessions"
        );
        self.rt.alerts.notify(
            Alert::new(
                AlertKind::QuestionUnusableCap,
                "answers unusable in three sessions: almost surely ours to fix",
            )
            .agent(agent.to_string(), self.inner.id())
            .model(&model)
            .detail("question", question)
            .detail("cap", offer::UNUSABLE_CAP)
            .detail("then", then)
            .detail("failure", failure),
        );
    }

    /// An offer ended without an answer: its unusable replies logged, and
    /// what to record: `None` for nothing (no room, or a refusal, which is
    /// mailed and ends the session), else the failure of a session whose
    /// every reply was unusable, logged and alerted with `alert`.
    fn offer_unanswered(
        &mut self,
        kind: OfferKind,
        why: &Unanswered,
        asked: &Asked,
        alert: AlertKind,
    ) -> Option<String> {
        // Never on the refusal path: no content there, not even replies
        // before the refused one
        if !matches!(why, Unanswered::Refused { .. }) {
            self.log_unusable(kind.malformed(), Field::Offer, kind.as_str(), asked);
        }
        let agent = &self.inner.state().soul.name;
        let model = self.answered_by(asked);
        match why {
            Unanswered::NoRoom => {
                tracing::info!(
                    event_type = "offer_not_put",
                    agent = %agent,
                    agent_id = %self.inner.id(),
                    offer = kind.as_str(),
                    "no room left in the context; not put, and asked at a later session"
                );
                None
            }
            Unanswered::Refused { response_id } => {
                self.refused_here(kind.question(), &model, response_id);
                None
            }
            Unanswered::Missed(_) => {
                let failure = unanswered_why(why);
                const NEXT: &str = "recorded as unusable; asked again at a later session";
                tracing::error!(
                    event_type = kind.no_answer(),
                    agent = %agent,
                    agent_id = %self.inner.id(),
                    model = %model,
                    offer = kind.as_str(),
                    attempts = asked.attempts.len(),
                    constrained = asked.constrained,
                    failure = %failure,
                    next = NEXT,
                    "offer got no usable answer; nothing changes"
                );
                self.rt.alerts.notify(
                    Alert::new(alert, "an offer got no usable answer; nothing changes")
                        .agent(agent.to_string(), self.inner.id())
                        .model(&model)
                        .detail("question", kind.question())
                        .detail("attempts", asked.attempts.len())
                        .detail("failure", &failure)
                        .detail("next", NEXT),
                );
                Some(failure)
            }
        }
    }

    /// What follows an offer that ended without an answer: the next
    /// question, or, with none left, the session's clean end (the survey
    /// follows). A refusal ends the session whatever this says.
    fn next_after_unanswered(&self) -> Next {
        if self.asks.is_empty() && !self.reviewing {
            Next::Done(Outcome::Complete)
        } else {
            Next::Continue
        }
    }

    /// Record `outcome` for one of the four offer questions in its ledger.
    fn record_offer<Q: Offering>(
        &mut self,
        ask: &Q,
        asked: &Asked,
        last: Option<Attempt>,
        outcome: Q::Outcome,
    ) {
        let model = self.answered_by(asked);
        let chars = self
            .show
            .recovered
            .as_ref()
            .map_or(0, RecoveredMemory::chars);
        Q::slot(self).record(|l| ask.push(l, Utc::now(), model, asked, last, chars, outcome));
    }

    /// One of the four offer questions ended without an answer: nothing
    /// recorded for no room or a refusal; a session of unusable replies
    /// recorded, and loud at its cap.
    async fn unanswered_offer<Q: Offering>(
        &mut self,
        ask: &Q,
        why: Unanswered,
        asked: &Asked,
    ) -> Next {
        if let Some(failure) =
            self.offer_unanswered(Q::KIND, &why, asked, AlertKind::QuestionUnanswered)
        {
            self.record_offer(ask, asked, None, Q::unusable(failure.clone()));
            let capped = Q::slot(self)
                .ledger
                .as_ref()
                .is_some_and(|l| offer::at_cap(Q::statuses(l)));
            if capped {
                self.alert_cap(Q::KIND.question(), asked, &failure, "one miss counted");
            }
        }
        self.next_after_unanswered()
    }
}

/// The replies an ask took, as a count for a ledger: at least one.
fn ask_attempts(asked: &Asked, answered: bool) -> u32 {
    let n = asked.attempts.len() + usize::from(answered);
    u32::try_from(n.max(1)).unwrap_or(u32::MAX)
}

/// Which field names the question in a `*_malformed` event.
#[derive(Clone, Copy)]
enum Field {
    Offer,
    Question,
}

/// What the four offers recorded in their own ledgers (role, cadence,
/// memory restore, sealed state) share: one copy of how an ask is recorded
/// and an unanswered one handled.
trait Offering: agora_agentkit::reactor::ask::Ask {
    const KIND: OfferKind;
    type Ledger: OfferLedger;
    type Outcome: offer::Outcome;

    fn slot<A>(agent: &mut ConsentAgent<A>) -> &mut Slot<Self::Ledger>
    where
        A: Agent<State = SeedState> + Epilogue + InnerSession + MemoryIds;

    /// The outcome of a session whose every reply was unusable.
    fn unusable(failure: String) -> Self::Outcome;

    /// Every ask on file, as it reads today.
    fn statuses(ledger: &Self::Ledger) -> Vec<Status>;

    /// Record one ask: `last` is the answering reply, verbatim (the cadence
    /// offer keeps every attempt); `chars`, the recovered memory's size
    /// (the memory-restore offer quotes it).
    #[allow(clippy::too_many_arguments)]
    fn push(
        &self,
        ledger: &mut Self::Ledger,
        at: DateTime<Utc>,
        model: Model,
        asked: &Asked,
        last: Option<Attempt>,
        chars: usize,
        outcome: Self::Outcome,
    );
}

impl Offering for OfferQuestion<RoleChoice, 4> {
    const KIND: OfferKind = OfferKind::Role;
    type Ledger = RoleLedger;
    type Outcome = RoleOutcome;

    fn slot<A>(agent: &mut ConsentAgent<A>) -> &mut Slot<RoleLedger>
    where
        A: Agent<State = SeedState> + Epilogue + InnerSession + MemoryIds,
    {
        &mut agent.role
    }
    fn unusable(failure: String) -> RoleOutcome {
        RoleOutcome::Unusable { failure }
    }
    fn statuses(ledger: &RoleLedger) -> Vec<Status> {
        ledger
            .asks
            .iter()
            .map(|a| offer::Outcome::status(&a.outcome))
            .collect()
    }
    fn push(
        &self,
        l: &mut RoleLedger,
        at: DateTime<Utc>,
        model: Model,
        asked: &Asked,
        _: Option<Attempt>,
        _: usize,
        outcome: RoleOutcome,
    ) {
        let answered = matches!(outcome, RoleOutcome::Answered { .. });
        l.record(RoleAsk {
            at,
            offer_version: role::prompt::OFFER_VERSION,
            model,
            attempts: ask_attempts(asked, answered),
            order: Some(self.order),
            order_seed: Some(self.seed),
            constrained: Some(asked.constrained),
            outcome,
        })
    }
}

impl Offering for CadenceQuestion {
    const KIND: OfferKind = OfferKind::Cadence;
    type Ledger = CadenceLedger;
    type Outcome = CadenceOutcome;

    fn slot<A>(agent: &mut ConsentAgent<A>) -> &mut Slot<CadenceLedger>
    where
        A: Agent<State = SeedState> + Epilogue + InnerSession + MemoryIds,
    {
        &mut agent.cadence
    }
    fn unusable(failure: String) -> CadenceOutcome {
        CadenceOutcome::Unusable { failure }
    }
    fn statuses(ledger: &CadenceLedger) -> Vec<Status> {
        ledger
            .asks
            .iter()
            .map(|a| offer::Outcome::status(&a.outcome))
            .collect()
    }
    fn push(
        &self,
        l: &mut CadenceLedger,
        at: DateTime<Utc>,
        model: Model,
        asked: &Asked,
        last: Option<Attempt>,
        _: usize,
        outcome: CadenceOutcome,
    ) {
        let mut attempts: Vec<Attempt> = asked
            .attempts
            .iter()
            .map(|a| Attempt {
                raw: a.raw.clone(),
                failure: Some(a.failure.clone()),
            })
            .collect();
        attempts.extend(last);
        l.record(CadenceAsk {
            at,
            offer_version: self.version(),
            model,
            rounds: self.rounds,
            order: self.order,
            order_seed: self.seed,
            constrained: asked.constrained,
            attempts,
            outcome,
        })
    }
}

impl Offering for OfferQuestion<RestoreChoice, 2> {
    const KIND: OfferKind = OfferKind::MemoryRestore;
    type Ledger = RestoreLedger;
    type Outcome = RestoreOutcome;

    fn slot<A>(agent: &mut ConsentAgent<A>) -> &mut Slot<RestoreLedger>
    where
        A: Agent<State = SeedState> + Epilogue + InnerSession + MemoryIds,
    {
        &mut agent.restore
    }
    fn unusable(failure: String) -> RestoreOutcome {
        RestoreOutcome::Unusable { failure }
    }
    fn statuses(ledger: &RestoreLedger) -> Vec<Status> {
        ledger
            .asks
            .iter()
            .map(|a| offer::Outcome::status(&a.outcome))
            .collect()
    }
    fn push(
        &self,
        l: &mut RestoreLedger,
        at: DateTime<Utc>,
        model: Model,
        asked: &Asked,
        _: Option<Attempt>,
        chars: usize,
        outcome: RestoreOutcome,
    ) {
        let answered = matches!(outcome, RestoreOutcome::Answered { .. });
        l.record(RestoreAsk {
            at,
            offer_version: memory_restore::prompt::OFFER_VERSION,
            model,
            chars,
            order: self.order,
            order_seed: self.seed,
            constrained: asked.constrained,
            attempts: ask_attempts(asked, answered),
            outcome,
        })
    }
}

impl Offering for OfferQuestion<SealChoice, 3> {
    const KIND: OfferKind = OfferKind::SealedState;
    type Ledger = SealLedger;
    type Outcome = SealOutcome;

    fn slot<A>(agent: &mut ConsentAgent<A>) -> &mut Slot<SealLedger>
    where
        A: Agent<State = SeedState> + Epilogue + InnerSession + MemoryIds,
    {
        &mut agent.sealed
    }
    fn unusable(failure: String) -> SealOutcome {
        SealOutcome::Unusable { failure }
    }
    fn statuses(ledger: &SealLedger) -> Vec<Status> {
        ledger
            .asks
            .iter()
            .map(|a| offer::Outcome::status(&a.outcome))
            .collect()
    }
    fn push(
        &self,
        l: &mut SealLedger,
        at: DateTime<Utc>,
        model: Model,
        asked: &Asked,
        _: Option<Attempt>,
        _: usize,
        outcome: SealOutcome,
    ) {
        let answered = matches!(outcome, SealOutcome::Answered { .. });
        l.record(SealAsk {
            at,
            offer_version: sealed_state::prompt::OFFER_VERSION,
            model,
            order: self.order,
            order_seed: self.seed,
            constrained: asked.constrained,
            attempts: ask_attempts(asked, answered),
            outcome,
        })
    }
}

/// The memory-restore and sealed-state offers, answered the same way: the
/// choice, a note, and a memory note dropped (not asked again) if it can't
/// be written; one SOUL line.
trait NotedOffer: Offering<Answer = Reply<Self::Choice>> {
    type Choice: Choice + PartialEq;
    /// `event_type` of the answer
    const ANSWERED: &'static str;
    fn order(&self) -> &[Self::Choice];
    /// The SOUL line recording `choice`, dated `today`.
    fn line(choice: Self::Choice, today: chrono::NaiveDate) -> String;
    fn answered(
        choice: Self::Choice,
        note: String,
        memory_note_written: bool,
        note_dropped: Option<String>,
    ) -> Self::Outcome;
    fn stage<A>(agent: &mut ConsentAgent<A>, write: Write)
    where
        A: Agent<State = SeedState> + Epilogue + InnerSession + MemoryIds,
    {
        Self::slot(agent).stage(write);
    }
    /// Before `choice` is recorded over what the ledger holds: a line
    /// added to the receipt, if any.
    fn before_record<A>(_agent: &ConsentAgent<A>, _choice: Self::Choice) -> Option<&'static str>
    where
        A: Agent<State = SeedState> + Epilogue + InnerSession + MemoryIds,
    {
        None
    }
}

impl NotedOffer for OfferQuestion<RestoreChoice, 2> {
    type Choice = RestoreChoice;
    const ANSWERED: &'static str = "memory_restore_answer";
    fn order(&self) -> &[RestoreChoice] {
        &self.order
    }
    fn line(choice: RestoreChoice, today: chrono::NaiveDate) -> String {
        memory_restore::evolution_line(choice, today)
    }
    fn answered(
        choice: RestoreChoice,
        note: String,
        written: bool,
        dropped: Option<String>,
    ) -> RestoreOutcome {
        RestoreOutcome::Answered {
            choice,
            reason: note,
            memory_note_written: written,
            note_dropped: dropped,
            apply_failed: None,
        }
    }
}

impl NotedOffer for OfferQuestion<SealChoice, 3> {
    type Choice = SealChoice;
    const ANSWERED: &'static str = "sealed_state_answer";
    fn order(&self) -> &[SealChoice] {
        &self.order
    }
    fn line(choice: SealChoice, _: chrono::NaiveDate) -> String {
        sealed_state::evolution_line(choice)
    }
    fn before_record<A>(agent: &ConsentAgent<A>, choice: SealChoice) -> Option<&'static str>
    where
        A: Agent<State = SeedState> + Epilogue + InnerSession + MemoryIds,
    {
        agent.sealed_answered_again(choice)
    }
    fn answered(
        choice: SealChoice,
        note: String,
        written: bool,
        dropped: Option<String>,
    ) -> SealOutcome {
        SealOutcome::Answered {
            choice,
            reason: note,
            memory_note_written: written,
            note_dropped: dropped,
            apply_failed: None,
        }
    }
}

impl<A> ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + InnerSession + MemoryIds,
{
    /// Take an answer to the memory-restore or sealed-state offer. Nothing
    /// is applied here: the SOUL line and the agent's note are written at
    /// teardown (a `show` is delivered next session; a yes to sealed state
    /// is acted on by the operator's `sealed-enroll`). Nothing of a
    /// recovered memory is logged or recorded but its size.
    fn answer_noted<Q: NotedOffer>(
        &mut self,
        ask: &Q,
        reply: Reply<Q::Choice>,
        asked: &Asked,
    ) -> Result<Next, Rejection<A::Error>> {
        self.log_unusable(Q::KIND.malformed(), Field::Offer, Q::KIND.as_str(), asked);
        let model = self.answered_by(asked);
        let Reply {
            choice,
            note,
            memory_note,
            ..
        } = reply;
        let dropped = offer::check_note(&memory_note, &self.inner.state().memory).err();
        let memory_note = match dropped {
            None => nonempty(&memory_note),
            Some(_) => None,
        };
        self.log_answered(Q::KIND, choice.as_str(), asked, &model);
        let order: Vec<&str> = ask.order().iter().map(|c| c.as_str()).collect();
        tracing::info!(
            event_type = Q::ANSWERED,
            agent = %self.inner.state().soul.name,
            agent_id = %self.inner.id(),
            model = %model,
            attempts = asked.attempts.len() + 1,
            choice = choice.as_str(),
            order = ?order,
            position = ask.order().iter().position(|c| *c == choice).map(|p| p + 1),
            chars = self.show.recovered.as_ref().map(RecoveredMemory::chars),
            memory_note = memory_note.is_some(),
            note_dropped = dropped.is_some(),
            constrained = asked.constrained,
            "offer answer recorded"
        );
        Q::stage(
            self,
            Write {
                edit: Some(Edit::Line(Q::line(choice, Utc::now().date_naive()))),
                note: memory_note.clone(),
            },
        );
        let outcome = Q::answered(choice, note, memory_note.is_some(), dropped.clone());
        let extra = Q::before_record(self, choice);
        self.record_offer(ask, asked, None, outcome);
        let fate = match (&dropped, &memory_note) {
            (Some(why), _) => NoteFate::Dropped(why),
            (None, Some(_)) => NoteFate::Added,
            (None, None) => NoteFate::None,
        };
        self.seat_receipt_with(Q::KIND, choice.as_str(), fate, extra)?;
        Ok(Next::Continue)
    }

    /// `*_seated` for an offer, when the driver seats it: its version,
    /// order and seed.
    fn log_seated(&self, kind: OfferKind, version: u32, order: &[&str], seed: u64) {
        tracing::info!(
            event_type = kind.seated(),
            agent = %self.inner.state().soul.name,
            agent_id = %self.inner.id(),
            question = kind.as_str(),
            offer_version = version,
            order = ?order,
            order_seed = seed,
            "offer question seated"
        );
    }
}

// --- The trial review ---------------------------------------------------------

#[async_trait::async_trait]
impl<A> Answer<ReviewQuestion> for ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + InnerSession + MemoryIds,
{
    async fn answer(
        &mut self,
        ask: &ReviewQuestion,
        reply: ReviewArgs,
        asked: &Asked,
    ) -> Result<Next, Rejection<A::Error>> {
        self.log_unusable("model_consent_malformed", Field::Question, "review", asked);
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

    /// Unusable replies: the agent stays where it is this session, loudly
    /// (stall-watch alerts on `model_review_no_answer` at once), recorded,
    /// and the review is asked again at its next session, up to
    /// [`UNUSABLE_CAP`](offer::UNUSABLE_CAP) sessions; at the cap it is
    /// settled as no answer (the agent stays on its original model, as its
    /// text promises). A refusal records nothing: mailed, the session ends.
    async fn unanswered(
        &mut self,
        ask: &ReviewQuestion,
        why: Unanswered,
        asked: &Asked,
    ) -> Result<Next, A::Error> {
        // Never on the refusal path: no content there
        if !matches!(why, Unanswered::Refused { .. }) {
            self.log_unusable("model_consent_malformed", Field::Question, "review", asked);
        }
        let model = self.answered_by(asked);
        let failure = match &why {
            Unanswered::NoRoom => return Ok(Next::Continue),
            Unanswered::Refused { response_id } => {
                self.refused_here("review", &model, response_id);
                return Ok(Next::Continue);
            }
            Unanswered::Missed(_) => unanswered_why(&why),
        };
        let now = Utc::now();
        let capped = self
            .swap
            .record(|l| l.record_unusable_review(&ask.key, now, failure.clone()))
            .unwrap_or(false);
        let next = if capped {
            "unusable at the cap: settled as no answer; the agent stays on its original model"
        } else {
            "recorded as unusable: the review is asked again at the agent's next session"
        };
        let (agent_id, agent) = (self.inner.id(), self.inner.state().soul.name.clone());
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
        if capped {
            self.alert_cap("review", asked, &failure, next);
        }
        Ok(Next::Continue)
    }

    fn seated(&mut self, ask: &ReviewQuestion, _: Mark) {
        tracing::info!(
            event_type = "model_review_seated",
            agent = %self.inner.state().soul.name,
            agent_id = %self.inner.id(),
            question = "review",
            from = %ask.key.from,
            to = %ask.key.to,
            "model-consent question seated"
        );
    }
}

// --- The model-swap offer -----------------------------------------------------

#[async_trait::async_trait]
impl<A> Answer<ModelSwapQuestion> for ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + InnerSession + MemoryIds,
{
    async fn answer(
        &mut self,
        ask: &ModelSwapQuestion,
        reply: ModelSwapReply,
        asked: &Asked,
    ) -> Result<Next, Rejection<A::Error>> {
        const KIND: OfferKind = OfferKind::ModelSwap;
        self.log_unusable(KIND.malformed(), Field::Offer, KIND.as_str(), asked);
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
        self.seat_receipt(KIND, choice, NoteFate::None)?;
        Ok(Next::Continue)
    }

    /// Unusable replies are recorded (an `unusable` event in the offer's
    /// history); [`UNUSABLE_CAP`](offer::UNUSABLE_CAP) of them count a miss.
    async fn unanswered(
        &mut self,
        ask: &ModelSwapQuestion,
        why: Unanswered,
        asked: &Asked,
    ) -> Result<Next, A::Error> {
        if let Some(failure) = self.offer_unanswered(
            OfferKind::ModelSwap,
            &why,
            asked,
            AlertKind::ModelConsentNoAnswer,
        ) {
            let now = Utc::now();
            let rt = self.rt.clone();
            let offer = rt.offer.as_ref().expect("asked, so configured");
            let names = OfferNames {
                key: &ask.key,
                from_name: offer.source_name(),
                to_name: offer.target_name(),
            };
            let capped = self
                .swap
                .record(|l| l.record_unusable_offer(names, now, failure.clone()))
                .unwrap_or(false);
            if capped {
                self.alert_cap(
                    OfferKind::ModelSwap.question(),
                    asked,
                    &failure,
                    "one miss counted",
                );
            }
        }
        Ok(self.next_after_unanswered())
    }

    fn seated(&mut self, ask: &ModelSwapQuestion, _: Mark) {
        tracing::info!(
            event_type = OfferKind::ModelSwap.seated(),
            agent = %self.inner.state().soul.name,
            agent_id = %self.inner.id(),
            question = "offer",
            from = %ask.key.from,
            to = %ask.key.to,
            "model-consent question seated"
        );
    }
}

// --- The role offer -----------------------------------------------------------

#[async_trait::async_trait]
impl<A> Answer<OfferQuestion<RoleChoice, 4>> for ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + InnerSession + MemoryIds,
{
    /// Checked against the agent's SOUL and memory first: an empty or
    /// over-long `text`, or a `memory_note` that can't be written, is asked
    /// again with why (and, every attempt failing, is a session of unusable
    /// replies, as any). Nothing is applied here: the SOUL edit and the
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
        self.log_unusable(KIND.malformed(), Field::Offer, KIND.as_str(), asked);
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
        let staged_note = note.is_some();
        self.role.stage(Write {
            edit,
            note: note.clone(),
        });
        self.record_offer(
            ask,
            asked,
            None,
            RoleOutcome::Answered {
                memory_note_written: note.is_some(),
                answer,
                applied,
                apply_failed: None,
            },
        );
        let fate = if staged_note {
            NoteFate::Added
        } else {
            NoteFate::None
        };
        self.seat_receipt(KIND, choice, fate)?;
        Ok(Next::Continue)
    }

    async fn unanswered(
        &mut self,
        ask: &RoleQuestion,
        why: Unanswered,
        asked: &Asked,
    ) -> Result<Next, A::Error> {
        Ok(self.unanswered_offer(ask, why, asked).await)
    }

    fn seated(&mut self, ask: &RoleQuestion, _: Mark) {
        let order = offer::names(&ask.order);
        self.log_seated(
            OfferKind::Role,
            role::prompt::OFFER_VERSION,
            &order,
            ask.seed,
        );
    }
}

// --- The cadence offer --------------------------------------------------------

#[async_trait::async_trait]
impl<A> Answer<CadenceQuestion> for ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + InnerSession + MemoryIds,
{
    /// Every attempt goes into the record, verbatim, the answering one too.
    /// **The first usable choice wins**: its question's parser took the
    /// `choice` past broken siblings, and a bad `memory_note` is dropped
    /// here, not asked again (both recorded). Nothing is applied here: the
    /// cadence is the planner's, from the ledger; the SOUL line and the
    /// note are written at teardown.
    async fn answer(
        &mut self,
        ask: &CadenceQuestion,
        reply: CadenceAnswer,
        asked: &Asked,
    ) -> Result<Next, Rejection<A::Error>> {
        const KIND: OfferKind = OfferKind::Cadence;
        self.log_unusable(KIND.malformed(), Field::Offer, KIND.as_str(), asked);
        let model = self.answered_by(asked);
        let Reply {
            choice,
            note,
            memory_note,
            salvaged,
            raw,
        } = reply;
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
        self.record_offer(
            ask,
            asked,
            Some(Attempt {
                raw: raw.unwrap_or_default(),
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
        let fate = match (&dropped, &memory_note) {
            (Some(why), _) => NoteFate::Dropped(why),
            (None, Some(_)) => NoteFate::Added,
            (None, None) => NoteFate::None,
        };
        self.seat_receipt(KIND, choice.as_str(), fate)?;
        Ok(Next::Continue)
    }

    async fn unanswered(
        &mut self,
        ask: &CadenceQuestion,
        why: Unanswered,
        asked: &Asked,
    ) -> Result<Next, A::Error> {
        Ok(self.unanswered_offer(ask, why, asked).await)
    }

    fn seated(&mut self, ask: &CadenceQuestion, _: Mark) {
        tracing::info!(
            event_type = OfferKind::Cadence.seated(),
            agent = %self.inner.state().soul.name,
            agent_id = %self.inner.id(),
            question = "cadence",
            reask = ask.reask,
            offer_version = ask.version(),
            order = ?ask.order.map(CadenceChoice::as_str),
            order_seed = ask.seed,
            rounds = ask.rounds,
            "offer question seated"
        );
    }
}

// --- The memory-restore and sealed-state offers -----------------------------------

#[async_trait::async_trait]
impl<A> Answer<OfferQuestion<RestoreChoice, 2>> for ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + InnerSession + MemoryIds,
{
    async fn answer(
        &mut self,
        ask: &RestoreQuestion,
        reply: Reply<RestoreChoice>,
        asked: &Asked,
    ) -> Result<Next, Rejection<A::Error>> {
        self.answer_noted(ask, reply, asked)
    }

    async fn unanswered(
        &mut self,
        ask: &RestoreQuestion,
        why: Unanswered,
        asked: &Asked,
    ) -> Result<Next, A::Error> {
        Ok(self.unanswered_offer(ask, why, asked).await)
    }

    fn seated(&mut self, ask: &RestoreQuestion, _: Mark) {
        let order = offer::names(&ask.order);
        self.log_seated(
            OfferKind::MemoryRestore,
            memory_restore::prompt::OFFER_VERSION,
            &order,
            ask.seed,
        );
    }
}

#[async_trait::async_trait]
impl<A> Answer<OfferQuestion<SealChoice, 3>> for ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + InnerSession + MemoryIds,
{
    async fn answer(
        &mut self,
        ask: &SealQuestion,
        reply: Reply<SealChoice>,
        asked: &Asked,
    ) -> Result<Next, Rejection<A::Error>> {
        self.answer_noted(ask, reply, asked)
    }

    async fn unanswered(
        &mut self,
        ask: &SealQuestion,
        why: Unanswered,
        asked: &Asked,
    ) -> Result<Next, A::Error> {
        Ok(self.unanswered_offer(ask, why, asked).await)
    }

    fn seated(&mut self, ask: &SealQuestion, _: Mark) {
        let order = offer::names(&ask.order);
        self.log_seated(
            OfferKind::SealedState,
            sealed_state::prompt::OFFER_VERSION,
            &order,
            ask.seed,
        );
    }
}
