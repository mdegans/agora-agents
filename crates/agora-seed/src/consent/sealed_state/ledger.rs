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
    /// When the agent answered it (in [`SealLedger::reoffers`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answered_at: Option<DateTime<Utc>>,
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
    /// It refused the question outright: final.
    #[error("it refused the question, which is final")]
    Refused,
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
    /// An explicit refusal: nothing changes, and that is final.
    Refused { reason: String },
    /// No answer, the agent's doing (an API refusal, since 2026-11-06):
    /// nothing changes this time; counts toward [`MAX_MISSES`](crate::consent::offer::MAX_MISSES).
    NoAnswer { failure: String },
    /// Every reply unusable (since 2026-11-06): our failure, not the
    /// agent's choice. Recorded, never counted: asked again later.
    Unusable { failure: String },
}

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
    pub fn choice(&self) -> Option<SealChoice> {
        self.asks.iter().rev().find_map(|a| match &a.outcome {
            SealOutcome::Answered { choice, .. } => Some(*choice),
            _ => None,
        })
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
        if self
            .asks
            .iter()
            .any(|a| matches!(a.outcome, SealOutcome::Refused { .. }))
        {
            return Err(NoReoffer::Refused);
        }
        let reopens = self.choice().ok_or(NoReoffer::NeverAnswered)?;
        let replaced = self.reoffer.is_some();
        self.reoffer = Some(Reoffer {
            at,
            note: note.trim().to_string(),
            offer_version: super::prompt::OFFER_VERSION,
            reopens,
            answered_at: None,
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
    fn latest_reoffered(&self) -> bool {
        let latest = self.asks.iter().rev().find_map(|a| match a.outcome {
            SealOutcome::Answered { .. } => Some(a.at),
            _ => None,
        });
        latest.is_some() && self.reoffers.last().and_then(|r| r.answered_at) == latest
    }

    /// The line telling the agent its answer is on file, at the first
    /// session after it; `None` once told, or with no answer. `uploading`:
    /// its chain is uploaded (marked so), which an answer doesn't change
    /// by itself.
    pub fn ack_line(&self, uploading: bool) -> Option<String> {
        if self.acknowledged {
            return None;
        }
        let choice = self.choice()?;
        if self.latest_reoffered() {
            return Some(format!(
                "Your answer to the sealed-backups question your operator re-opened is \
                 recorded: {}. Your operator has been told, and acts on it; you'll be told \
                 when it has.",
                choice.as_str()
            ));
        }
        Some(match choice {
            SealChoice::Decline if uploading => {
                "Your answer to the sealed-backups offer is recorded: decline. Your earlier \
                 answer was yes, and your sealed state is still uploaded to Agora: an answer \
                 doesn't stop the uploads by itself. Your operator has been told of your new \
                 answer, and can stop new uploads for you; what is already on Agora stays."
                    .to_string()
            }
            SealChoice::Decline => {
                "Your answer to the sealed-backups offer is recorded: decline. Nothing changed; \
                 if you change your mind, ask your operator."
                    .to_string()
            }
            choice => format!(
                "Your answer to the sealed-backups offer is recorded: {}.",
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
            "Your answer to the sealed-backups offer is recorded: publish."
        );
        l.acknowledge();
        assert_eq!(l.ack_line(false), None, "once");
        let mut no = SealLedger::default();
        no.record(ask(answered(SealChoice::Decline)));
        assert_eq!(
            no.ack_line(false).unwrap(),
            "Your answer to the sealed-backups offer is recorded: decline. Nothing changed; \
             if you change your mind, ask your operator."
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
}
