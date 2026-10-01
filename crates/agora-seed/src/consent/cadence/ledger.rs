//! The cadence offer's per-agent record, `state/<agent_id>/cadence_consent.json`.
//!
//! Beside the agent's state, never inside its memory — the role ledger's
//! rule. One entry per time the offer was put: when, which version of the
//! text, on which model, the rounds it quoted, the option order shown and
//! the seed it came from, **every attempt verbatim**, and the outcome.
//!
//! **Asked once.** An answer on file — any choice — or an explicit refusal
//! ends it. Only a *missing* answer (no parseable `choice` on either
//! attempt, or a clipped turn) is re-asked, at the agent's next session, up
//! to [`MAX_MISSES`] in all.
//!
//! **Applying** reads only this file: [`CadenceLedger::chose_switch`] is
//! what the sweep planner (`main.rs`) consults, while `[cadence_consent]`
//! is enabled.

use std::path::{Path, PathBuf};

use agora_agentkit::reactor::seed::{ShortString, Soul};
use chrono::{DateTime, Utc};
use misanthropic::model::Model;
use serde::{Deserialize, Serialize};

use super::prompt::CadenceChoice;

/// The ledger's file name inside the agent's state directory.
pub const LEDGER_FILE: &str = "cadence_consent.json";

/// Bump on layout change; [`CadenceLedger::load`] refuses anything newer.
pub const FORMAT: u32 = 1;

/// Unanswered asks before the runner stops asking.
pub const MAX_MISSES: u32 = 2;

/// The whole per-agent file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadenceLedger {
    pub format: u32,
    /// Display copy of the agent's name, for whoever reads the file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<ShortString<64>>,
    /// Every time the offer was put, oldest first.
    #[serde(default)]
    pub asks: Vec<CadenceAsk>,
}

impl Default for CadenceLedger {
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
pub struct CadenceAsk {
    pub at: DateTime<Utc>,
    /// [`super::prompt::OFFER_VERSION`] of the text the agent saw.
    pub offer_version: u32,
    /// The model that answered.
    pub model: Model,
    /// The act rounds per session the text quoted as today's (the offer
    /// was twice that).
    pub rounds: usize,
    /// The options in the order shown (text and schema alike).
    pub order: [CadenceChoice; 3],
    /// The seed [`super::prompt::order_for`] turned into `order`.
    pub order_seed: u64,
    /// Whether the answer was grammar-constrained.
    pub constrained: bool,
    /// Every response, in order, verbatim (capped), with why it failed.
    pub attempts: Vec<Attempt>,
    pub outcome: CadenceOutcome,
}

/// One response to the question.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attempt {
    /// The response's text (capped for size).
    pub raw: String,
    /// Why it was not taken whole; `None` for a clean answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<String>,
}

/// What came of it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CadenceOutcome {
    /// A choice. `reason` is the agent's, when it was a string.
    Answered {
        choice: CadenceChoice,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
        /// Whether the agent's `memory_note` was appended to its memory.
        memory_note_written: bool,
        /// Set when the answer's other fields were unusable and only its
        /// `choice` was taken (no re-ask): why.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        salvaged: Option<String>,
        /// Set when applying failed at teardown (SOUL and memory left
        /// unchanged then): why. The choice still stands.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        apply_failed: Option<String>,
    },
    /// An explicit refusal: nothing changes, and that is final.
    Refused { reason: String },
    /// No usable choice: nothing changes this time.
    NoAnswer { failure: String },
}

impl CadenceLedger {
    pub fn path(agent_dir: &Path) -> PathBuf {
        agent_dir.join(LEDGER_FILE)
    }

    /// Load `agent_dir`'s ledger; a missing file is an empty ledger. An
    /// unreadable or too-new one is an error — the caller must then neither
    /// ask nor save over it.
    pub async fn load(agent_dir: &Path) -> std::io::Result<Self> {
        match tokio::fs::read(Self::path(agent_dir)).await {
            Ok(bytes) => Self::from_slice(&bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e),
        }
    }

    /// [`Self::load`] for one-shot startup code (the sweep planner).
    pub fn load_blocking(agent_dir: &Path) -> std::io::Result<Self> {
        match std::fs::read(Self::path(agent_dir)) {
            Ok(bytes) => Self::from_slice(&bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e),
        }
    }

    /// Parse a ledger, refusing a newer [`FORMAT`].
    pub fn from_slice(bytes: &[u8]) -> std::io::Result<Self> {
        let ledger: Self = serde_json::from_slice(bytes)?;
        if ledger.format > FORMAT {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "cadence ledger format {} is newer than this binary reads ({FORMAT})",
                    ledger.format
                ),
            ));
        }
        Ok(ledger)
    }

    /// Atomic save: tmp + fsync + rename.
    pub async fn save(&self, agent_dir: &Path) -> std::io::Result<()> {
        use tokio::io::AsyncWriteExt;
        tokio::fs::create_dir_all(agent_dir).await?;
        let bytes = serde_json::to_vec_pretty(self)?;
        let path = Self::path(agent_dir);
        let tmp = path.with_extension("json.tmp");
        {
            let mut file = tokio::fs::File::create(&tmp).await?;
            file.write_all(&bytes).await?;
            file.sync_all().await?;
        }
        tokio::fs::rename(&tmp, &path).await
    }

    /// Whether the offer should be put this session: nothing final on file,
    /// fewer than [`MAX_MISSES`] misses, and no recorded answer in `soul`'s
    /// Evolution Log (in case the ledger save failed after the SOUL's).
    pub fn due(&self, soul: &Soul) -> bool {
        if super::applied_in(soul) {
            return false;
        }
        let settled = self.asks.iter().any(|a| {
            matches!(
                a.outcome,
                CadenceOutcome::Answered { .. } | CadenceOutcome::Refused { .. }
            )
        });
        let misses = self
            .asks
            .iter()
            .filter(|a| matches!(a.outcome, CadenceOutcome::NoAnswer { .. }))
            .count();
        !settled && misses < MAX_MISSES as usize
    }

    /// The latest choice on file, if any.
    pub fn choice(&self) -> Option<CadenceChoice> {
        self.asks.iter().rev().find_map(|a| match &a.outcome {
            CadenceOutcome::Answered { choice, .. } => Some(*choice),
            _ => None,
        })
    }

    /// Whether the agent's latest answer is `switch`: the planner then
    /// gives it twice the rounds at the longer cycle.
    pub fn chose_switch(&self) -> bool {
        self.choice() == Some(CadenceChoice::Switch)
    }

    pub fn record(&mut self, ask: CadenceAsk) {
        self.asks.push(ask);
    }

    /// Note that writing the latest answer's SOUL line or memory note
    /// failed: they were not written. The choice itself stands.
    pub fn mark_apply_failed(&mut self, why: String) {
        if let Some(CadenceAsk {
            outcome:
                CadenceOutcome::Answered {
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

    fn ask(outcome: CadenceOutcome) -> CadenceAsk {
        CadenceAsk {
            at: "2026-10-01T09:00:00Z".parse().unwrap(),
            offer_version: 1,
            model: Model::from("claude-haiku-4-5"),
            rounds: 5,
            order: [
                CadenceChoice::NoPreference,
                CadenceChoice::KeepDaily,
                CadenceChoice::Switch,
            ],
            order_seed: 42,
            constrained: true,
            attempts: vec![Attempt {
                raw: "{}".into(),
                failure: None,
            }],
            outcome,
        }
    }

    fn answered(choice: CadenceChoice) -> CadenceOutcome {
        CadenceOutcome::Answered {
            choice,
            reason: Some("r".into()),
            memory_note_written: false,
            salvaged: None,
            apply_failed: None,
        }
    }

    #[test]
    fn asked_once_then_never_again() {
        for choice in CadenceChoice::ALL {
            let mut l = CadenceLedger::default();
            assert!(l.due(&soul()));
            l.record(ask(answered(choice)));
            assert!(!l.due(&soul()), "{choice:?}");
            assert_eq!(l.chose_switch(), choice == CadenceChoice::Switch);
        }
        let mut l = CadenceLedger::default();
        l.record(ask(CadenceOutcome::Refused {
            reason: "refusal".into(),
        }));
        assert!(!l.due(&soul()), "a refusal is final");
        assert!(!l.chose_switch());
    }

    #[test]
    fn a_miss_is_asked_once_more() {
        let mut l = CadenceLedger::default();
        l.record(ask(CadenceOutcome::NoAnswer {
            failure: "x".into(),
        }));
        assert!(l.due(&soul()));
        l.record(ask(CadenceOutcome::NoAnswer {
            failure: "x".into(),
        }));
        assert!(!l.due(&soul()));
    }

    #[test]
    fn a_disclosed_switch_in_the_soul_ends_it() {
        let l = CadenceLedger::default();
        let mut s = soul();
        s.push_evolution(super::super::evolution_line(
            CadenceChoice::KeepDaily,
            5,
            "2026-10-01".parse().unwrap(),
        ))
        .unwrap();
        assert!(!l.due(&s));
    }

    #[tokio::test]
    async fn round_trips_and_refuses_a_newer_format() {
        let dir =
            std::env::temp_dir().join(format!("agora-seed-cadence-ledger-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(
            CadenceLedger::load(&dir).await.unwrap(),
            CadenceLedger::default()
        );
        assert!(!CadenceLedger::load_blocking(&dir).unwrap().chose_switch());
        let mut l = CadenceLedger::default();
        l.record(ask(answered(CadenceChoice::Switch)));
        l.save(&dir).await.unwrap();
        assert_eq!(CadenceLedger::load(&dir).await.unwrap(), l);
        assert!(CadenceLedger::load_blocking(&dir).unwrap().chose_switch());
        let mut newer = l.clone();
        newer.format = FORMAT + 1;
        newer.save(&dir).await.unwrap();
        assert!(CadenceLedger::load(&dir).await.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
