//! The role offer's per-agent record, `state/<agent_id>/role_consent.json`.
//!
//! Beside the agent's state, never inside its memory — the same rule as
//! the model-swap ledger (`super::super::ledger`). One entry per time the
//! offer was put: when, which version of the text, on which model, the
//! full answer (or why there was none), and what was applied.
//!
//! **Asked once.** An answer on file — any choice — or an explicit refusal
//! ends it: the agent is never asked again by this offer. Only a *missing*
//! answer (every attempt malformed, or clipped, or not fitting — an
//! upstream problem until shown otherwise) is re-asked, at the agent's next
//! session, up to [`MAX_MISSES`](crate::consent::offer::MAX_MISSES) in all. `sleep` asks again, but that is
//! the Steward's to arrange when the tools exist (a new
//! [`super::prompt::OFFER_VERSION`] and a fresh offer), not this ledger's.

use agora_agentkit::reactor::seed::{ShortString, Soul};
use chrono::{DateTime, Utc};
use misanthropic::model::Model;
use serde::{Deserialize, Serialize};

use super::prompt::{RoleAnswer, RoleChoice};

/// The ledger's file name inside the agent's state directory.
pub const LEDGER_FILE: &str = "role_consent.json";

/// Bump on layout change (2 since 2026-11-06: answers' `note`, the
/// `unusable` outcome); [`RoleLedger::load`] refuses anything newer.
pub const FORMAT: u32 = 2;

/// The whole per-agent file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoleLedger {
    pub format: u32,
    /// Display copy of the agent's name, for whoever reads the file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<ShortString<64>>,
    /// Every time the offer was put, oldest first.
    #[serde(default)]
    pub asks: Vec<RoleAsk>,
}

impl Default for RoleLedger {
    fn default() -> Self {
        Self {
            format: FORMAT,
            agent: None,
            asks: Vec::new(),
        }
    }
}

/// One putting of the offer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoleAsk {
    pub at: DateTime<Utc>,
    /// [`super::prompt::OFFER_VERSION`] of the text the agent saw.
    pub offer_version: u32,
    /// The model that answered.
    pub model: Model,
    /// Attempts made this session (the first plus retries).
    pub attempts: u32,
    /// The options in the order shown, text and schema alike (v2 on;
    /// v1 showed [`RoleChoice::ALL`] and recorded nothing here).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order: Option<[RoleChoice; 4]>,
    /// The seed [`super::prompt::order_for`] turned into `order`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order_seed: Option<u64>,
    /// Whether the answer taken was grammar-constrained: from v3 to v4 a
    /// call to the strict `answer_offer` tool (`true`) or the plain-text
    /// fallback; since v5 (2026-11-06) whether its schema went out as
    /// `output_config`. Not recorded before v3.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constrained: Option<bool>,
    pub outcome: RoleOutcome,
}

/// What came of it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RoleOutcome {
    /// A usable answer, verbatim, and what the runner did with it.
    Answered {
        answer: RoleAnswer,
        applied: Applied,
        /// Whether the agent's `memory_note` was appended to its memory.
        memory_note_written: bool,
        /// Set when applying failed at teardown (nothing was changed then,
        /// SOUL or memory): why.
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

/// What the runner applied for an answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Applied {
    /// `nothing`: the SOUL is unchanged.
    Nothing,
    /// `clarify`: one sentence appended to `identity`.
    Clarified { previous: String, identity: String },
    /// `new_role`: `identity` replaced. `previous` here is the permanent
    /// record of the old identity; the SOUL's Evolution Log carries it
    /// verbatim too, but that log drops its oldest entries past 50.
    RoleChanged { previous: String, identity: String },
    /// `sleep`: nothing changed. While this is the latest answer, the sweep
    /// leaves the agent out ([`RoleLedger::chose_sleep`]) until the Steward
    /// lists it in the run config's `wake`.
    Sleep,
}

impl RoleLedger {
    /// Whether the offer should be put this session: nothing final on file,
    /// fewer than [`MAX_MISSES`](crate::consent::offer::MAX_MISSES) misses, and no sign in `soul` that an
    /// answer was already applied — the belt to the ledger's braces, for a
    /// session whose SOUL saved but whose ledger didn't (a second `clarify`
    /// would append twice).
    pub fn due(&self, soul: &Soul) -> bool {
        crate::consent::offer::due(
            self.asks.iter().map(|a| &a.outcome),
            super::applied_in(soul),
        )
    }

    /// Whether the agent's latest answer on file is `sleep`: the sweep
    /// then leaves it out until the Steward wakes it (`wake` in the run
    /// config).
    pub fn chose_sleep(&self) -> bool {
        self.asks.iter().rev().find_map(|a| match &a.outcome {
            RoleOutcome::Answered { applied, .. } => Some(applied == &Applied::Sleep),
            _ => None,
        }) == Some(true)
    }

    pub fn record(&mut self, ask: RoleAsk) {
        self.asks.push(ask);
    }

    /// Note that applying the latest answer failed: nothing was changed.
    pub fn mark_apply_failed(&mut self, why: String) {
        if let Some(RoleAsk {
            outcome:
                RoleOutcome::Answered {
                    memory_note_written,
                    apply_failed,
                    ..
                },
            ..
        }) = self.asks.last_mut()
        {
            *memory_note_written = false;
            *apply_failed = Some(why);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consent::offer::Stored;
    use crate::consent::role::prompt::RoleChoice;

    fn soul() -> Soul {
        serde_json::from_value(serde_json::json!({
            "name": "pilot",
            "identity": "I am an economist.",
            "values": ["rigor"],
            "interests": { "communities": ["economics"] },
            "voice": "measured",
        }))
        .unwrap()
    }

    /// A SOUL that already discloses an applied answer is never asked
    /// again, whatever the ledger says (it may have failed to save).
    #[test]
    fn an_applied_answer_in_the_soul_ends_it() {
        let l = RoleLedger::default();
        for line in [
            format!(
                "{}, after the Steward's offer … Added: \"x\"",
                crate::consent::role::CLARIFIED
            ),
            format!(
                "{}, after … Previous identity: \"y\"",
                crate::consent::role::ROLE_CHANGED
            ),
        ] {
            let mut s = soul();
            assert!(l.due(&s));
            s.push_evolution(line).unwrap();
            assert!(!l.due(&s));
        }
        // The agent's own words about it don't count.
        let mut s = soul();
        s.push_evolution("I thought about my identity being clarified.")
            .unwrap();
        assert!(l.due(&s));
    }

    #[test]
    fn chose_sleep_is_the_latest_answer() {
        let mut l = RoleLedger::default();
        assert!(!l.chose_sleep());
        l.record(ask(RoleOutcome::Answered {
            answer: RoleAnswer {
                note: "r".into(),
                choice: RoleChoice::Sleep,
                soul_text: String::new(),
                memory_note: String::new(),
            },
            applied: Applied::Sleep,
            memory_note_written: false,
            apply_failed: None,
        }));
        assert!(l.chose_sleep());
        let dir =
            std::env::temp_dir().join(format!("agora-seed-role-sleep-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            !RoleLedger::load_blocking(&dir).unwrap().chose_sleep(),
            "missing = awake"
        );
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(RoleLedger::path(&dir), serde_json::to_vec(&l).unwrap()).unwrap();
        assert!(RoleLedger::load_blocking(&dir).unwrap().chose_sleep());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn ask(outcome: RoleOutcome) -> RoleAsk {
        RoleAsk {
            at: "2026-09-29T12:00:00Z".parse().unwrap(),
            offer_version: 1,
            model: Model::from("gpt-oss-120b.gguf"),
            attempts: 1,
            order: None,
            order_seed: None,
            constrained: None,
            outcome,
        }
    }

    #[test]
    fn asked_once_then_never_again() {
        let mut l = RoleLedger::default();
        assert!(l.due(&soul()));
        l.record(ask(RoleOutcome::Answered {
            answer: RoleAnswer {
                note: "r".into(),
                choice: RoleChoice::Nothing,
                soul_text: String::new(),
                memory_note: String::new(),
            },
            applied: Applied::Nothing,
            memory_note_written: false,
            apply_failed: None,
        }));
        assert!(!l.due(&soul()), "an answer on file ends it");

        let mut l = RoleLedger::default();
        l.record(ask(RoleOutcome::Refused {
            reason: "refusal".into(),
        }));
        assert!(!l.due(&soul()), "a refusal is final");
    }

    #[test]
    fn a_miss_is_asked_once_more() {
        let mut l = RoleLedger::default();
        l.record(ask(RoleOutcome::NoAnswer {
            failure: "refusal".into(),
        }));
        assert!(l.due(&soul()));
        l.record(ask(RoleOutcome::NoAnswer {
            failure: "refusal".into(),
        }));
        assert!(!l.due(&soul()));
    }

    #[tokio::test]
    async fn round_trips_and_refuses_a_newer_format() {
        let dir =
            std::env::temp_dir().join(format!("agora-seed-role-ledger-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(RoleLedger::load(&dir).await.unwrap(), RoleLedger::default());
        let mut l = RoleLedger::default();
        l.record(ask(RoleOutcome::Answered {
            answer: RoleAnswer {
                note: "r".into(),
                choice: RoleChoice::NewRole,
                soul_text: "I am a reader of Agora.".into(),
                memory_note: "I chose this.".into(),
            },
            applied: Applied::RoleChanged {
                previous: "I am an archivist.".into(),
                identity: "I am a reader of Agora.".into(),
            },
            memory_note_written: true,
            apply_failed: None,
        }));
        l.save(&dir).await.unwrap();
        assert!(dir.join(LEDGER_FILE).exists());
        assert_eq!(RoleLedger::load(&dir).await.unwrap(), l);
        let mut newer = l.clone();
        newer.format = FORMAT + 1;
        newer.save(&dir).await.unwrap();
        assert!(RoleLedger::load(&dir).await.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
