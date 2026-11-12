//! The sealed-state offer's per-agent record,
//! `state/<agent_id>/sealed_state.json`.
//!
//! Beside the agent's state, never inside its memory, the siblings' rule.
//! One entry per time the offer was put: when, which version of the text,
//! on which model, the option order shown and its seed, how many attempts,
//! and the outcome. **Asked once**: an answer on file, or an explicit
//! refusal, ends it; a missing answer is re-asked up to [`MAX_MISSES`](crate::consent::offer::MAX_MISSES) in
//! all. `sealed-enroll` reads [`SealLedger::choice`] (the latest answer) and
//! nothing else.

use agora_agentkit::reactor::seed::{ShortString, Soul};
use chrono::{DateTime, Utc};
use misanthropic::model::Model;
use serde::{Deserialize, Serialize};

use super::prompt::SealChoice;

/// The ledger's file name inside the agent's state directory.
pub const LEDGER_FILE: &str = "sealed_state.json";

/// Bump on layout change (2 since 2026-11-06: answers' `note`, the
/// `unusable` outcome; 3 since 2026-11-11: the operator's re-offer, which
/// an older runner would drop on its next save); [`SealLedger::load`]
/// refuses anything newer.
pub const FORMAT: u32 = 3;

/// The whole per-agent file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SealLedger {
    pub format: u32,
    /// Display copy of the agent's name, for whoever reads the file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<ShortString<64>>,
    /// Every time the offer was put, oldest first.
    #[serde(default)]
    pub asks: Vec<SealAsk>,
    /// The answer on file was told back to the agent, once, at the session
    /// after it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub acknowledged: bool,
    /// The operator re-opened the question (`agora-seed sealed-reoffer`):
    /// put again, once, at the agent's next session; consumed by its
    /// answer (or an explicit refusal).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reoffer: Option<Reoffer>,
    /// Re-offers answered, oldest first, each with when it was answered.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reoffers: Vec<Reoffer>,
}

impl Default for SealLedger {
    fn default() -> Self {
        Self {
            format: FORMAT,
            agent: None,
            asks: Vec::new(),
            acknowledged: false,
            reoffer: None,
            reoffers: Vec::new(),
        }
    }
}

/// The operator re-opened the question for the agent. Never consent: only
/// the agent's own answer to it is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reoffer {
    pub at: DateTime<Utc>,
    /// The operator's note: shown to the agent with the question.
    pub note: String,
    /// [`super::prompt::OFFER_VERSION`] of the text to be put.
    pub offer_version: u32,
    /// The answer on file when it was re-opened.
    pub reopens: SealChoice,
    /// What was on file was a legacy refusal, counted as `decline`: the
    /// agent never chose, and the re-offer doesn't call it its answer.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub never_chose: bool,
    /// When the agent answered it (in [`SealLedger::reoffers`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answered_at: Option<DateTime<Utc>>,
    /// Its answer needed no operator action (the uploads already matched
    /// it): the agent was told so at once.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub no_action: bool,
}

/// Why the operator can't re-open the question.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NoReoffer {
    /// Nothing to re-open: the ordinary offer (`[sealed_state]`) asks an
    /// agent that hasn't answered.
    #[error(
        "it has not answered the offer: list it in `[sealed_state]` to be asked; a re-offer \
         re-opens an answer"
    )]
    NeverAnswered,
}

/// What an operator's action would contradict in the agent's answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Contradicts {
    #[error("its latest answer to the sealed-state offer is `decline`")]
    SaidNo,
    #[error("its latest answer to the sealed-state offer is `{0}` (yes)")]
    SaidYes(&'static str),
}

/// One putting of the offer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SealAsk {
    pub at: DateTime<Utc>,
    /// [`super::prompt::OFFER_VERSION`] of the text the agent saw.
    pub offer_version: u32,
    /// The model that answered.
    pub model: Model,
    /// The options in the order shown.
    pub order: [SealChoice; 3],
    /// The seed [`super::prompt::order_for`] turned into `order`.
    pub order_seed: u64,
    /// Whether the answer came by the strict tool (or was parsed from text).
    pub constrained: bool,
    /// Answers it took, the usable one included.
    pub attempts: u32,
    pub outcome: SealOutcome,
}

/// What came of it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SealOutcome {
    /// A choice. `reason` is the agent's note.
    Answered {
        choice: SealChoice,
        reason: String,
        /// Whether the agent's `memory_note` was appended to its memory.
        memory_note_written: bool,
        /// Set when the note was not written (why). The choice stands.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        note_dropped: Option<String>,
        /// Set when writing the SOUL line or note failed at teardown (why).
        /// The choice still stands.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        apply_failed: Option<String>,
    },
    /// An explicit refusal. Only read, never written since 2026-11-06 (an
    /// older runner's ledgers hold it): kept so they load, and counted as a
    /// recorded `decline` for every purpose ([`SealLedger::choice`], a
    /// re-offer, [`SealLedger::allows`]), so it can be revisited through
    /// the operator like any answer.
    Refused { reason: String },
    /// No answer, the agent's doing (an API refusal, since 2026-11-06):
    /// nothing changes this time; counts toward [`MAX_MISSES`](crate::consent::offer::MAX_MISSES).
    NoAnswer { failure: String },
    /// Every reply unusable (since 2026-11-06): our failure, not the
    /// agent's choice. Recorded, never counted: asked again later.
    Unusable { failure: String },
}

/// How the line telling a first answer back begins; the choice follows.
pub const ACK: &str = "Your answer on uploading your sealed history to Agora is on file:";

/// How the line telling a re-offer's answer back begins.
pub const ACK_REOFFER: &str = "Your answer to the question your operator re-opened, on uploading \
     your sealed history to Agora, is on file:";

impl SealLedger {
    /// Whether the offer should be put this session: nothing final on
    /// file, fewer than [`MAX_MISSES`](crate::consent::offer::MAX_MISSES) misses, and no recorded answer in
    /// `soul`'s Evolution Log.
    /// A pending re-offer is due regardless of the answer on file, until
    /// it has missed as often as a first offer may.
    pub fn due(&self, soul: &Soul) -> bool {
        if let Some(r) = &self.reoffer {
            return crate::consent::offer::misses(
                self.asks
                    .iter()
                    .filter(|a| a.at >= r.at)
                    .map(|a| crate::consent::offer::Outcome::status(&a.outcome)),
            )
            .is_some_and(|m| m < crate::consent::offer::MAX_MISSES);
        }
        crate::consent::offer::due(
            self.asks.iter().map(|a| &a.outcome),
            super::applied_in(soul),
        )
    }

    /// The choice on file, if any: the **latest** answer. A later answer
    /// (to an operator's re-offer) replaces an earlier one, either way.
    /// A legacy refusal counts as `decline`.
    pub fn choice(&self) -> Option<SealChoice> {
        self.answers().last().map(|(_, c)| *c)
    }

    /// Every answer on file, oldest first, with when (a legacy refusal as
    /// `decline`).
    fn answers(&self) -> Vec<(DateTime<Utc>, SealChoice)> {
        self.asks
            .iter()
            .filter_map(|a| match &a.outcome {
                SealOutcome::Answered { choice, .. } => Some((a.at, *choice)),
                SealOutcome::Refused { .. } => Some((a.at, SealChoice::Decline)),
                _ => None,
            })
            .collect()
    }

    /// The answer that was latest at `at`: the one an operator acted on
    /// then (the opt-in, by the uploading marker's `since`). The publish
    /// choice in force is this one's, not a later re-answer's.
    pub fn choice_as_of(&self, at: DateTime<Utc>) -> Option<SealChoice> {
        self.answers()
            .into_iter()
            .rev()
            .find(|(t, _)| *t <= at)
            .map(|(_, c)| c)
    }

    /// The latest "answer" on file is a legacy refusal
    /// ([`SealOutcome::Refused`]), counted as `decline`: the agent never
    /// chose, and no text calls it its answer.
    pub fn latest_is_refusal(&self) -> bool {
        self.asks
            .iter()
            .rev()
            .find_map(|a| match &a.outcome {
                SealOutcome::Answered { .. } => Some(false),
                SealOutcome::Refused { .. } => Some(true),
                _ => None,
            })
            .unwrap_or(false)
    }

    /// When the latest answer was given.
    pub fn answered_at(&self) -> Option<DateTime<Utc>> {
        self.answers().last().map(|(t, _)| *t)
    }

    /// The agent said yes, either way.
    pub fn opted_in(&self) -> bool {
        matches!(
            self.choice(),
            Some(SealChoice::Publish | SealChoice::Private)
        )
    }

    /// Re-open the question for the agent (`sealed-reoffer`): put again,
    /// once, at its next session, with `note`. A pending one is replaced
    /// (`true`). The caller saves.
    pub fn reopen(&mut self, note: &str, at: DateTime<Utc>) -> Result<bool, NoReoffer> {
        let reopens = self.choice().ok_or(NoReoffer::NeverAnswered)?;
        let replaced = self.reoffer.is_some();
        self.reoffer = Some(Reoffer {
            at,
            note: note.trim().to_string(),
            offer_version: super::prompt::OFFER_VERSION,
            reopens,
            never_chose: self.latest_is_refusal(),
            answered_at: None,
            no_action: false,
        });
        Ok(replaced)
    }

    /// Whether an operator may opt the agent in (`yes`) or stop its
    /// uploads (`!yes`) without overriding its answer: never against its
    /// latest answer. A stop with no answer on file contradicts nothing.
    pub fn allows(&self, yes: bool) -> Result<(), Contradicts> {
        match (yes, self.choice()) {
            (true, Some(SealChoice::Decline)) => Err(Contradicts::SaidNo),
            (false, Some(c @ (SealChoice::Publish | SealChoice::Private))) => {
                Err(Contradicts::SaidYes(c.as_str()))
            }
            _ => Ok(()),
        }
    }

    /// Whether the latest answer was to an operator's re-offer.
    fn latest_reoffer(&self) -> Option<&Reoffer> {
        let latest = self.answered_at()?;
        self.reoffers
            .last()
            .filter(|r| r.answered_at == Some(latest))
    }

    /// Mark the pending re-offer as needing no operator action (before its
    /// answer is recorded).
    pub fn reoffer_needs_no_action(&mut self) {
        if let Some(r) = self.reoffer.as_mut() {
            r.no_action = true;
        }
    }

    /// The line telling the agent its answer is on file, at the first
    /// session after it; `None` once told, with no answer, or when what
    /// is on file is a legacy refusal (the agent never chose: nothing is
    /// called its answer). `uploading`: its history is being uploaded now
    /// ([`crate::sealed::uploads`]: marked, not stopped, the marker not in
    /// doubt), which an answer doesn't change by itself.
    ///
    /// No line says the operator was told: the alert went out, or didn't,
    /// in the session that answered, and this one can't know. It says how
    /// the agent can reach its operator instead.
    pub fn ack_line(&self, uploading: bool) -> Option<String> {
        if self.acknowledged || self.latest_is_refusal() {
            return None;
        }
        let choice = self.choice()?;
        let ask = crate::sealed::text::TELL_OPERATOR;
        if let Some(r) = self.latest_reoffer() {
            return Some(if r.no_action {
                // No action: a yes found it uploading, a no found it not
                // (never uploaded, or stopped)
                format!(
                    "{ACK_REOFFER} {}. Nothing needed to change: it was already {}.",
                    choice.as_str(),
                    if choice == SealChoice::Decline {
                        "not being uploaded"
                    } else {
                        "being uploaded"
                    }
                )
            } else {
                format!(
                    "{ACK_REOFFER} {}. Nothing has changed yet: your operator has to act on it, \
                     and you are told when it has. {ask}",
                    choice.as_str()
                )
            });
        }
        Some(match choice {
            SealChoice::Decline if uploading => format!(
                "{ACK} decline. It is being uploaded all the same, and an answer doesn't change \
                 that by itself: your operator has to stop it. {ask}"
            ),
            SealChoice::Decline => format!(
                "{ACK} decline. It is not being uploaded. If you change your mind, ask your \
                 operator to re-open the question."
            ),
            choice if uploading => format!("{ACK} {}. It is being uploaded.", choice.as_str()),
            choice => format!(
                "{ACK} {}. It is not being uploaded now: uploading starts, or resumes, only \
                 when your operator acts on your answer, and you are told when it has.",
                choice.as_str()
            ),
        })
    }

    /// The answer was told back.
    pub fn acknowledge(&mut self) {
        self.acknowledged = true;
    }

    /// Record a putting of the offer. An answer (or a refusal) consumes a
    /// pending re-offer, and is told back at the next session.
    pub fn record(&mut self, ask: SealAsk) {
        let settled = matches!(
            ask.outcome,
            SealOutcome::Answered { .. } | SealOutcome::Refused { .. }
        );
        if settled && let Some(mut r) = self.reoffer.take() {
            r.answered_at = Some(ask.at);
            self.reoffers.push(r);
        }
        if matches!(ask.outcome, SealOutcome::Answered { .. }) {
            self.acknowledged = false;
        }
        self.asks.push(ask);
    }

    /// An answer arriving now would be the agent's second (or later) with
    /// no re-offer pending: the offer is asked once, so a ledger that
    /// shows this is wrong. Logged at error and alerted; nothing acts on
    /// it.
    pub fn unprompted_answer(&self) -> bool {
        self.reoffer.is_none() && self.choice().is_some()
    }

    /// Note that writing the answer's SOUL line or note failed. The choice
    /// itself stands.
    pub fn mark_apply_failed(&mut self, why: String) {
        let answered = self
            .asks
            .iter_mut()
            .rev()
            .find_map(|a| match &mut a.outcome {
                SealOutcome::Answered {
                    memory_note_written,
                    apply_failed,
                    ..
                } => Some((memory_note_written, apply_failed)),
                _ => None,
            });
        if let Some((memory_note_written, apply_failed)) = answered {
            *memory_note_written = false;
            *apply_failed = Some(why);
        }
    }
}

/// Every sentence this module can tell an agent, for the one table of
/// agent-facing text ([`crate::sealed::text`]): add a new one here, or
/// its banned-phrase test never sees it.
#[cfg(test)]
pub(crate) fn agent_texts() -> Vec<crate::sealed::text::Text> {
    use crate::sealed::text::{Text, To};
    let ask = |at: &str, choice| SealAsk {
        at: at.parse().expect("a date"),
        offer_version: 1,
        model: Model::from("claude-haiku-4-5"),
        order: SealChoice::ALL,
        order_seed: 0,
        constrained: true,
        attempts: 1,
        outcome: SealOutcome::Answered {
            choice,
            reason: String::new(),
            memory_note_written: false,
            note_dropped: None,
            apply_failed: None,
        },
    };
    let mut out = Vec::new();
    for choice in SealChoice::ALL {
        // A first answer, uploading or not
        let mut first = SealLedger::default();
        first.record(ask("2026-11-08T09:00:00Z", choice));
        for uploading in [true, false] {
            let to = if uploading { To::Uploaded } else { To::Any };
            out.extend(
                first
                    .ack_line(uploading)
                    .map(|l| Text::new("ack_line", to, l)),
            );
        }
        // A re-offer's answer: to be acted on, and needing nothing
        for no_action in [false, true] {
            let mut again = first.clone();
            again
                .reopen("n", "2026-11-10T09:00:00Z".parse().expect("a date"))
                .expect("answered");
            if no_action {
                again.reoffer_needs_no_action();
            }
            again.record(ask("2026-11-12T09:00:00Z", choice));
            out.extend(
                again
                    .ack_line(choice != SealChoice::Decline && no_action)
                    .map(|l| Text::new("ack_line (re-offer)", To::Any, l)),
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consent::offer::Stored;

    fn soul() -> Soul {
        serde_json::from_value(serde_json::json!({
            "name": "tarn",
            "identity": "A test agent.",
            "values": ["testing"],
            "interests": { "communities": ["tech"] },
            "voice": "terse",
        }))
        .unwrap()
    }

    fn ask(outcome: SealOutcome) -> SealAsk {
        SealAsk {
            at: "2026-11-08T09:00:00Z".parse().unwrap(),
            offer_version: 1,
            model: Model::from("claude-haiku-4-5"),
            order: SealChoice::ALL,
            order_seed: 42,
            constrained: true,
            attempts: 1,
            outcome,
        }
    }

    fn answered(choice: SealChoice) -> SealOutcome {
        SealOutcome::Answered {
            choice,
            reason: String::new(),
            memory_note_written: false,
            note_dropped: None,
            apply_failed: None,
        }
    }

    /// An answer to an older text settles the offer: a new version asks
    /// nobody who answered an older one again (v6 nobody who answered v5;
    /// v9, the JSON answer, nobody who answered with `answer_offer`).
    #[test]
    fn an_answer_to_an_older_offer_text_is_not_asked_again() {
        for old in [5, 8] {
            let mut l = SealLedger::default();
            let mut a = ask(answered(SealChoice::Private));
            a.offer_version = old;
            l.record(a);
            assert!(!l.due(&soul()), "v{old}");
        }
        const { assert!(super::super::prompt::OFFER_VERSION > 8) };
    }

    #[test]
    fn asked_once_and_yes_is_either_yes() {
        for choice in SealChoice::ALL {
            let mut l = SealLedger::default();
            assert!(l.due(&soul()));
            assert!(!l.opted_in());
            l.record(ask(answered(choice)));
            assert!(!l.due(&soul()), "{choice:?}");
            assert_eq!(l.choice(), Some(choice));
            assert_eq!(l.opted_in(), choice != SealChoice::Decline);
        }
        let mut l = SealLedger::default();
        l.record(ask(SealOutcome::Refused { reason: "r".into() }));
        assert!(!l.due(&soul()) && !l.opted_in(), "a refusal is final");
        let mut l = SealLedger::default();
        for _ in 0..crate::consent::offer::MAX_MISSES {
            assert!(l.due(&soul()));
            l.record(ask(SealOutcome::NoAnswer {
                failure: "refusal".into(),
            }));
        }
        assert!(!l.due(&soul()), "misses capped");
    }

    /// The answer is told back once, at the session after it.
    #[test]
    fn the_answer_is_acknowledged_once() {
        let mut l = SealLedger::default();
        assert_eq!(l.ack_line(false), None, "nothing answered");
        l.record(ask(SealOutcome::NoAnswer {
            failure: "refusal".into(),
        }));
        assert_eq!(l.ack_line(false), None, "no answer, nothing to tell");
        l.record(ask(answered(SealChoice::Publish)));
        assert_eq!(
            l.ack_line(false).unwrap(),
            "Your answer on uploading your sealed history to Agora is on file: publish. It is \
             not being uploaded now: uploading starts, or resumes, only when your operator acts \
             on your answer, and you are told when it has."
        );
        assert_eq!(
            l.ack_line(true).unwrap(),
            "Your answer on uploading your sealed history to Agora is on file: publish. It is \
             being uploaded."
        );
        l.acknowledge();
        assert_eq!(l.ack_line(false), None, "once");
        let mut no = SealLedger::default();
        no.record(ask(answered(SealChoice::Decline)));
        assert_eq!(
            no.ack_line(false).unwrap(),
            "Your answer on uploading your sealed history to Agora is on file: decline. It is \
             not being uploaded. If you change your mind, ask your operator to re-open the \
             question."
        );
    }

    #[tokio::test]
    async fn round_trips_and_refuses_a_newer_format() {
        let dir = crate::sealed::TestDir::new("seal-ledger");
        assert_eq!(
            SealLedger::load(dir.path()).await.unwrap(),
            SealLedger::default()
        );
        let mut l = SealLedger::default();
        l.record(ask(answered(SealChoice::Private)));
        l.save(dir.path()).await.unwrap();
        assert_eq!(SealLedger::load(dir.path()).await.unwrap(), l);
        assert_eq!(SealLedger::load_blocking(dir.path()).unwrap(), l);
        let mut newer = l.clone();
        newer.format = FORMAT + 1;
        newer.save(dir.path()).await.unwrap();
        assert!(SealLedger::load(dir.path()).await.is_err());
    }

    fn at(ask: SealAsk, when: &str) -> SealAsk {
        SealAsk {
            at: when.parse().unwrap(),
            ..ask
        }
    }

    /// `reopen`: refused with no answer; a second replaces the note; due
    /// whatever the answer and the Evolution Log, until the misses run
    /// out; consumed by the answer, which is then the latest and told
    /// back once as a re-offer's.
    #[test]
    fn a_reoffer_is_due_once_and_consumed_by_its_answer() {
        let mut l = SealLedger::default();
        assert_eq!(l.reopen("n", Utc::now()), Err(NoReoffer::NeverAnswered));
        l.record(at(
            ask(answered(SealChoice::Private)),
            "2026-11-08T09:00:00Z",
        ));
        l.acknowledge();
        assert!(!l.due(&soul()));
        let t = "2026-11-10T09:00:00Z".parse().unwrap();
        assert_eq!(l.reopen("first", t), Ok(false));
        assert_eq!(l.reopen("Asked by DM.", t), Ok(true), "replaced");
        assert_eq!(l.reoffer.as_ref().unwrap().note, "Asked by DM.");
        assert!(l.due(&soul()));
        let mut n = 0;
        while l.due(&soul()) && n < 64 {
            l.record(at(
                ask(SealOutcome::Unusable {
                    failure: "x".into(),
                }),
                "2026-11-11T09:00:00Z",
            ));
            n += 1;
        }
        assert!(n > 1 && !l.due(&soul()), "the misses ran out ({n})");
        assert!(l.reoffer.is_some(), "not consumed by a miss");
        l.record(at(
            ask(answered(SealChoice::Decline)),
            "2026-11-12T09:00:00Z",
        ));
        assert!(l.reoffer.is_none());
        assert_eq!(l.reoffers.len(), 1);
        assert_eq!(l.choice(), Some(SealChoice::Decline));
        assert!(!l.due(&soul()));
        let told = l.ack_line(true).unwrap();
        assert!(
            told.starts_with(&format!("{ACK_REOFFER} decline. Nothing has changed yet")),
            "{told}"
        );
        // No claim that the operator was told (O2): how to reach it
        assert!(
            told.contains("you are told when it has")
                && told.ends_with(crate::sealed::text::TELL_OPERATOR)
                && !told.contains("has been told"),
            "{told}"
        );
        l.acknowledge();
        assert_eq!(l.ack_line(true), None, "once");
    }

    /// A re-offer answered where nothing needs doing says so at once.
    #[test]
    fn a_reoffer_answer_needing_nothing_says_so() {
        let mut l = SealLedger::default();
        l.record(at(
            ask(answered(SealChoice::Private)),
            "2026-11-08T09:00:00Z",
        ));
        l.reopen("n", "2026-11-10T09:00:00Z".parse().unwrap())
            .unwrap();
        l.reoffer_needs_no_action();
        l.record(at(
            ask(answered(SealChoice::Publish)),
            "2026-11-12T09:00:00Z",
        ));
        let told = l.ack_line(true).unwrap();
        assert_eq!(
            told,
            "Your answer to the question your operator re-opened, on uploading your sealed \
             history to Agora, is on file: publish. Nothing needed to change: it was already \
             being uploaded."
        );
        assert!(!told.contains("you are told"), "{told}");
        // A no that needed nothing: never uploading, or stopped
        let mut no = SealLedger::default();
        no.record(at(
            ask(answered(SealChoice::Private)),
            "2026-11-08T09:00:00Z",
        ));
        no.reopen("n", "2026-11-10T09:00:00Z".parse().unwrap())
            .unwrap();
        no.reoffer_needs_no_action();
        no.record(at(
            ask(answered(SealChoice::Decline)),
            "2026-11-12T09:00:00Z",
        ));
        assert!(
            no.ack_line(false)
                .unwrap()
                .ends_with("Nothing needed to change: it was already not being uploaded.")
        );
    }

    /// A legacy refusal counts as `decline` for what may be done, but the
    /// agent never chose: it is never told "your answer is on file", and
    /// a re-offer of it says so instead of calling it its answer.
    #[test]
    fn a_legacy_refusal_is_never_called_the_agents_answer() {
        let mut l = SealLedger::default();
        l.record(ask(SealOutcome::Refused { reason: "r".into() }));
        assert_eq!(l.choice(), Some(SealChoice::Decline));
        assert!(l.latest_is_refusal() && !l.acknowledged);
        assert_eq!(l.ack_line(false), None);
        assert_eq!(l.ack_line(true), None);
        l.reopen("n", "2026-11-10T09:00:00Z".parse().unwrap())
            .unwrap();
        assert!(l.reoffer.as_ref().unwrap().never_chose);
        // An answer is one: told back, and a re-offer of it calls it so
        l.record(at(
            ask(answered(SealChoice::Private)),
            "2026-11-12T09:00:00Z",
        ));
        assert!(!l.latest_is_refusal() && l.ack_line(false).is_some());
        l.reopen("n", "2026-11-13T09:00:00Z".parse().unwrap())
            .unwrap();
        assert!(!l.reoffer.as_ref().unwrap().never_chose);
    }

    /// `allows`: an opt-in never over a `decline`, a stop never over a yes,
    /// a stop with no answer contradicts nothing; `choice_as_of` is the
    /// answer acted on then.
    #[test]
    fn allows_and_the_choice_as_of_a_time() {
        let mut l = SealLedger::default();
        assert_eq!(l.allows(false), Ok(()));
        assert_eq!(l.allows(true), Ok(()), "no answer: `opted_in` decides");
        l.record(at(
            ask(answered(SealChoice::Private)),
            "2026-11-08T09:00:00Z",
        ));
        assert_eq!(l.allows(true), Ok(()));
        assert_eq!(l.allows(false), Err(Contradicts::SaidYes("private")));
        l.reopen("n", "2026-11-09T09:00:00Z".parse().unwrap())
            .unwrap();
        l.record(at(
            ask(answered(SealChoice::Decline)),
            "2026-11-12T09:00:00Z",
        ));
        assert_eq!(l.allows(true), Err(Contradicts::SaidNo));
        assert_eq!(l.allows(false), Ok(()));
        let between = "2026-11-10T00:00:00Z".parse().unwrap();
        assert_eq!(l.choice_as_of(between), Some(SealChoice::Private));
        assert_eq!(
            l.choice_as_of("2026-11-01T00:00:00Z".parse().unwrap()),
            None
        );
    }

    /// The first answer's ack for a `decline` while uploading says only
    /// what is true.
    #[test]
    fn a_first_decline_while_uploading_is_told_truly() {
        let mut l = SealLedger::default();
        l.record(ask(answered(SealChoice::Decline)));
        let told = l.ack_line(true).unwrap();
        assert_eq!(
            told,
            "Your answer on uploading your sealed history to Agora is on file: decline. It is \
             being uploaded all the same, and an answer doesn't change that by itself: your \
             operator has to stop it. You can tell your operator (the `send_message` tool, to \
             `operator`)."
        );
        assert!(!told.contains("earlier answer"), "{told}");
    }
}
