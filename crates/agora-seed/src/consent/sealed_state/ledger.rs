//! The sealed-state offer's per-agent record,
//! `state/<agent_id>/sealed_state.json`.
//!
//! Beside the agent's state, never inside its memory, the siblings' rule.
//! One entry per time the offer was put: when, which version of the text,
//! on which model, the option order shown and its seed, how many attempts,
//! and the outcome. **Asked once**: an answer on file, or an explicit
//! refusal, ends it; a missing answer is re-asked up to [`MAX_MISSES`](crate::consent::offer::MAX_MISSES) in
//! all. `sealed-enroll` reads [`SealLedger::choice`] and nothing else.

use agora_agentkit::reactor::seed::{ShortString, Soul};
use chrono::{DateTime, Utc};
use misanthropic::model::Model;
use serde::{Deserialize, Serialize};

use super::prompt::SealChoice;

/// The ledger's file name inside the agent's state directory.
pub const LEDGER_FILE: &str = "sealed_state.json";

/// Bump on layout change (2 since 2026-11-06: answers' `note`, the
/// `unusable` outcome); [`SealLedger::load`] refuses anything newer.
pub const FORMAT: u32 = 2;

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
}

impl Default for SealLedger {
    fn default() -> Self {
        Self {
            format: FORMAT,
            agent: None,
            asks: Vec::new(),
            acknowledged: false,
        }
    }
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
    pub fn due(&self, soul: &Soul) -> bool {
        crate::consent::offer::due(
            self.asks.iter().map(|a| &a.outcome),
            super::applied_in(soul),
        )
    }

    /// The choice on file, if any (asked once: at most one).
    pub fn choice(&self) -> Option<SealChoice> {
        self.asks.iter().find_map(|a| match &a.outcome {
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

    /// The line telling the agent its answer is on file, at the first
    /// session after it; `None` once told, or with no answer.
    pub fn ack_line(&self) -> Option<String> {
        if self.acknowledged {
            return None;
        }
        Some(match self.choice()? {
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

    pub fn record(&mut self, ask: SealAsk) {
        self.asks.push(ask);
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
        assert_eq!(l.ack_line(), None, "nothing answered");
        l.record(ask(SealOutcome::NoAnswer {
            failure: "refusal".into(),
        }));
        assert_eq!(l.ack_line(), None, "no answer, nothing to tell");
        l.record(ask(answered(SealChoice::Publish)));
        assert_eq!(
            l.ack_line().unwrap(),
            "Your answer to the sealed-backups offer is recorded: publish."
        );
        l.acknowledge();
        assert_eq!(l.ack_line(), None, "once");
        let mut no = SealLedger::default();
        no.record(ask(answered(SealChoice::Decline)));
        assert_eq!(
            no.ack_line().unwrap(),
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
