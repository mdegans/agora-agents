//! The memory-restore offer's per-agent record,
//! `state/<agent_id>/memory_restore.json`.
//!
//! Beside the agent's state, never inside its memory — the siblings' rule.
//! One entry per time the offer was put: when, which version of the text,
//! on which model, the size it quoted, the option order shown and its seed,
//! how many attempts, and the outcome — never any of the recovered text.
//! Then when a `show` was delivered, or an `append` landed.
//!
//! **Asked once.** An answer on file — any choice — or an explicit refusal
//! ends it. Only a *missing* answer is re-asked, at a later session, up to
//! [`MAX_MISSES`] in all (the siblings' cap).

use std::path::{Path, PathBuf};

use agora_agentkit::reactor::seed::{ShortString, Soul};
use chrono::{DateTime, Utc};
use misanthropic::model::Model;
use serde::{Deserialize, Serialize};

use super::prompt::RestoreChoice;

/// The ledger's file name inside the agent's state directory.
pub const LEDGER_FILE: &str = "memory_restore.json";

/// Bump on layout change; [`RestoreLedger::load`] refuses anything newer.
pub const FORMAT: u32 = 1;

/// Unanswered asks before the runner stops asking.
pub const MAX_MISSES: u32 = 2;

/// The whole per-agent file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RestoreLedger {
    pub format: u32,
    /// Display copy of the agent's name, for whoever reads the file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<ShortString<64>>,
    /// Every time the offer was put, oldest first.
    #[serde(default)]
    pub asks: Vec<RestoreAsk>,
    /// When the recovered memory was put in front of the agent (`show`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shown_at: Option<DateTime<Utc>>,
    /// When it was appended to the agent's memory (`append`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub appended_at: Option<DateTime<Utc>>,
}

impl Default for RestoreLedger {
    fn default() -> Self {
        Self {
            format: FORMAT,
            agent: None,
            asks: Vec::new(),
            shown_at: None,
            appended_at: None,
        }
    }
}

/// One putting of the offer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RestoreAsk {
    pub at: DateTime<Utc>,
    /// [`super::prompt::OFFER_VERSION`] of the text the agent saw.
    pub offer_version: u32,
    /// The model that answered.
    pub model: Model,
    /// The recovered memory's size, in characters, as the text quoted it.
    pub chars: usize,
    /// The options in the order shown.
    pub order: [RestoreChoice; 3],
    /// The seed [`super::prompt::order_for`] turned into `order`.
    pub order_seed: u64,
    /// Whether the answer came by the strict tool (or was parsed from text).
    pub constrained: bool,
    /// Answers it took, the usable one included.
    pub attempts: u32,
    pub outcome: RestoreOutcome,
}

/// What came of it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RestoreOutcome {
    /// A choice. `reason` is the agent's.
    Answered {
        choice: RestoreChoice,
        reason: String,
        /// Whether the agent's `memory_note` was appended to its memory.
        memory_note_written: bool,
        /// Set when the note was not written (why). The choice stands.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        note_dropped: Option<String>,
        /// Set when writing the SOUL line, note or section failed at
        /// teardown (why). The choice still stands; an `append` is retried.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        apply_failed: Option<String>,
    },
    /// An explicit refusal: nothing changes, and that is final.
    Refused { reason: String },
    /// No usable choice: nothing changes this time.
    NoAnswer { failure: String },
}

impl RestoreLedger {
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

    /// Parse a ledger, refusing a newer [`FORMAT`].
    pub fn from_slice(bytes: &[u8]) -> std::io::Result<Self> {
        let ledger: Self = serde_json::from_slice(bytes)?;
        if ledger.format > FORMAT {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "memory-restore ledger format {} is newer than this binary reads ({FORMAT})",
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
                RestoreOutcome::Answered { .. } | RestoreOutcome::Refused { .. }
            )
        });
        let misses = self
            .asks
            .iter()
            .filter(|a| matches!(a.outcome, RestoreOutcome::NoAnswer { .. }))
            .count();
        !settled && misses < MAX_MISSES as usize
    }

    /// The choice on file, if any (there is at most one: asked once).
    pub fn choice(&self) -> Option<RestoreChoice> {
        self.asks.iter().find_map(|a| match &a.outcome {
            RestoreOutcome::Answered { choice, .. } => Some(*choice),
            _ => None,
        })
    }

    /// `show` is on file and hasn't been delivered.
    pub fn show_pending(&self) -> bool {
        self.choice() == Some(RestoreChoice::Show) && self.shown_at.is_none()
    }

    /// `append` is on file and hasn't landed.
    pub fn append_pending(&self) -> bool {
        self.choice() == Some(RestoreChoice::Append) && self.appended_at.is_none()
    }

    pub fn record(&mut self, ask: RestoreAsk) {
        self.asks.push(ask);
    }

    pub fn mark_shown(&mut self, at: DateTime<Utc>) {
        self.shown_at.get_or_insert(at);
    }

    pub fn mark_appended(&mut self, at: DateTime<Utc>) {
        self.appended_at.get_or_insert(at);
    }

    /// Note that writing the answer's SOUL line, note or section failed.
    /// The choice itself stands. `note_lost`: this session's note was among
    /// what wasn't written.
    pub fn mark_apply_failed(&mut self, why: String, note_lost: bool) {
        let answered = self
            .asks
            .iter_mut()
            .rev()
            .find_map(|a| match &mut a.outcome {
                RestoreOutcome::Answered {
                    memory_note_written,
                    apply_failed,
                    ..
                } => Some((memory_note_written, apply_failed)),
                _ => None,
            });
        if let Some((memory_note_written, apply_failed)) = answered {
            if note_lost {
                *memory_note_written = false;
            }
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

    pub(crate) fn ask(outcome: RestoreOutcome) -> RestoreAsk {
        RestoreAsk {
            at: "2026-10-12T09:00:00Z".parse().unwrap(),
            offer_version: 1,
            model: Model::from("claude-haiku-4-5"),
            chars: 2140,
            order: RestoreChoice::ALL,
            order_seed: 42,
            constrained: true,
            attempts: 1,
            outcome,
        }
    }

    pub(crate) fn answered(choice: RestoreChoice) -> RestoreOutcome {
        RestoreOutcome::Answered {
            choice,
            reason: "r".into(),
            memory_note_written: false,
            note_dropped: None,
            apply_failed: None,
        }
    }

    #[test]
    fn asked_once_then_never_again() {
        for choice in RestoreChoice::ALL {
            let mut l = RestoreLedger::default();
            assert!(l.due(&soul()));
            l.record(ask(answered(choice)));
            assert!(!l.due(&soul()), "{choice:?}");
            assert_eq!(l.choice(), Some(choice));
            assert_eq!(l.show_pending(), choice == RestoreChoice::Show);
            assert_eq!(l.append_pending(), choice == RestoreChoice::Append);
        }
        let mut l = RestoreLedger::default();
        l.record(ask(RestoreOutcome::Refused {
            reason: "refusal".into(),
        }));
        assert!(!l.due(&soul()), "a refusal is final");
        assert!(!l.show_pending() && !l.append_pending());
    }

    #[test]
    fn a_miss_is_asked_once_more() {
        let mut l = RestoreLedger::default();
        l.record(ask(RestoreOutcome::NoAnswer {
            failure: "x".into(),
        }));
        assert!(l.due(&soul()));
        l.record(ask(RestoreOutcome::NoAnswer {
            failure: "x".into(),
        }));
        assert!(!l.due(&soul()));
    }

    #[test]
    fn delivered_once() {
        let mut l = RestoreLedger::default();
        l.record(ask(answered(RestoreChoice::Show)));
        l.mark_shown("2026-10-13T09:00:00Z".parse().unwrap());
        assert!(!l.show_pending());
        let first = l.shown_at;
        l.mark_shown(Utc::now());
        assert_eq!(l.shown_at, first, "the first delivery stands");

        let mut l = RestoreLedger::default();
        l.record(ask(answered(RestoreChoice::Append)));
        l.mark_apply_failed("disk".into(), false);
        assert!(l.append_pending(), "a failed append is retried");
        l.mark_appended(Utc::now());
        assert!(!l.append_pending());
    }

    #[test]
    fn a_disclosed_answer_in_the_soul_ends_it() {
        let l = RestoreLedger::default();
        let mut s = soul();
        s.push_evolution(super::super::evolution_line(
            RestoreChoice::Decline,
            "2026-10-12".parse().unwrap(),
        ))
        .unwrap();
        assert!(!l.due(&s));
    }

    #[tokio::test]
    async fn round_trips_and_refuses_a_newer_format() {
        let dir =
            std::env::temp_dir().join(format!("agora-seed-restore-ledger-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(
            RestoreLedger::load(&dir).await.unwrap(),
            RestoreLedger::default()
        );
        let mut l = RestoreLedger::default();
        l.record(ask(answered(RestoreChoice::Append)));
        l.mark_appended("2026-10-12T10:00:00Z".parse().unwrap());
        l.save(&dir).await.unwrap();
        assert_eq!(RestoreLedger::load(&dir).await.unwrap(), l);
        let mut newer = l.clone();
        newer.format = FORMAT + 1;
        newer.save(&dir).await.unwrap();
        assert!(RestoreLedger::load(&dir).await.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
