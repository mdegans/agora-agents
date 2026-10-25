//! The memory-restore offer's per-agent record,
//! `state/<agent_id>/memory_restore.json`.
//!
//! Beside the agent's state, never inside its memory — the siblings' rule.
//! One entry per time the offer was put: when, which version of the text,
//! on which model, the size it quoted, the option order shown and its seed,
//! how many attempts, and the outcome — never any of the recovered text.
//! Then one entry per session the recovered memory was shown in, with how
//! that session ended.
//!
//! **Asked once.** An answer on file — either choice — or an explicit
//! refusal ends it. Only a *missing* answer is re-asked, at a later
//! session, up to [`MAX_MISSES`] in all (the siblings' cap).
//!
//! **Shown until one session completes.** A `show` is delivered at the
//! next session, and again at the following one while no showing session
//! has completed with a successful memory update, up to the configured
//! `show_sessions_max` showings in all.

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
    /// Every session the recovered memory was shown in, oldest first.
    #[serde(default)]
    pub showings: Vec<Showing>,
}

impl Default for RestoreLedger {
    fn default() -> Self {
        Self {
            format: FORMAT,
            agent: None,
            asks: Vec::new(),
            showings: Vec::new(),
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
    pub order: [RestoreChoice; 2],
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
        /// Set when writing the SOUL line or note failed at teardown (why).
        /// The choice still stands.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        apply_failed: Option<String>,
    },
    /// An explicit refusal: nothing changes, and that is final.
    Refused { reason: String },
    /// No usable choice: nothing changes this time.
    NoAnswer { failure: String },
}

/// One session the recovered memory was shown in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Showing {
    /// When it was seated (recorded at the model's first response).
    pub at: DateTime<Utc>,
    pub outcome: ShowOutcome,
    /// The session's prompt dump (the `prompt logged` event's
    /// `prompt_sha256`), once torn down. **It holds the recovered text**:
    /// never a fork or replay source, and not to be shared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_sha256: Option<String>,
}

/// How a showing session ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShowOutcome {
    /// The session completed, its memory update accepted: no more showings.
    Completed,
    /// It didn't (or hasn't yet: a showing is recorded as this at the
    /// model's first response in its session, and saved at once, so a
    /// session that dies from then on, without a teardown, still counts
    /// toward the cap; one that died before any response isn't recorded).
    Interrupted,
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

    /// `show` is on file, no showing session has completed, and fewer than
    /// `max` sessions have shown it.
    pub fn show_pending(&self, max: u32) -> bool {
        self.choice() == Some(RestoreChoice::Show)
            && !self
                .showings
                .iter()
                .any(|s| s.outcome == ShowOutcome::Completed)
            && self.showings.len() < max as usize
    }

    pub fn record(&mut self, ask: RestoreAsk) {
        self.asks.push(ask);
    }

    /// A showing seated `at`: interrupted until its session completes.
    pub fn begin_showing(&mut self, at: DateTime<Utc>) {
        self.showings.push(Showing {
            at,
            outcome: ShowOutcome::Interrupted,
            prompt_sha256: None,
        });
    }

    /// The latest showing's session was torn down: its dump, and whether it
    /// completed with its memory update.
    pub fn end_showing(&mut self, completed: bool, prompt_sha256: Option<String>) {
        if let Some(last) = self.showings.last_mut() {
            if completed {
                last.outcome = ShowOutcome::Completed;
            }
            last.prompt_sha256 = prompt_sha256;
        }
    }

    /// The reactor couldn't save the state of a session that completed a
    /// showing at or after `since`: its memory update is lost, so the
    /// showing counts as interrupted and it is shown again. Whether
    /// anything changed.
    pub fn reopen_since(&mut self, since: DateTime<Utc>) -> bool {
        match self.showings.last_mut() {
            Some(last) if last.at >= since && last.outcome == ShowOutcome::Completed => {
                last.outcome = ShowOutcome::Interrupted;
                true
            }
            _ => false,
        }
    }

    /// Note that writing the answer's SOUL line or note failed. The choice
    /// itself stands.
    pub fn mark_apply_failed(&mut self, why: String) {
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

    fn ask(outcome: RestoreOutcome) -> RestoreAsk {
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

    fn answered(choice: RestoreChoice) -> RestoreOutcome {
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
            assert_eq!(l.show_pending(3), choice == RestoreChoice::Show);
        }
        let mut l = RestoreLedger::default();
        l.record(ask(RestoreOutcome::Refused {
            reason: "refusal".into(),
        }));
        assert!(!l.due(&soul()), "a refusal is final");
        assert!(!l.show_pending(3));
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

    /// Shown until a showing session completes, and never more than `max`
    /// times.
    #[test]
    fn shown_until_one_completes_up_to_the_cap() {
        let mut l = RestoreLedger::default();
        l.record(ask(answered(RestoreChoice::Show)));
        l.begin_showing(Utc::now());
        assert!(l.show_pending(3), "interrupted: shown again");
        let at = Utc::now();
        l.begin_showing(at);
        l.end_showing(true, Some("abc".into()));
        assert!(!l.show_pending(3), "completed: done");
        assert_eq!(l.showings[1].prompt_sha256.as_deref(), Some("abc"));
        // Its state not saved: shown again; an older showing is left alone.
        assert!(!l.reopen_since(at + chrono::Duration::seconds(1)));
        assert!(l.reopen_since(at));
        assert!(l.show_pending(3));
        assert_eq!(
            l.showings.iter().map(|s| s.outcome).collect::<Vec<_>>(),
            [ShowOutcome::Interrupted, ShowOutcome::Interrupted]
        );

        let mut l = RestoreLedger::default();
        l.record(ask(answered(RestoreChoice::Show)));
        for _ in 0..3 {
            assert!(l.show_pending(3));
            l.begin_showing(Utc::now());
        }
        assert!(!l.show_pending(3), "capped");
        assert!(l.show_pending(4), "the cap is the config's");
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
        l.record(ask(answered(RestoreChoice::Show)));
        l.begin_showing("2026-10-13T09:00:00Z".parse().unwrap());
        l.save(&dir).await.unwrap();
        assert_eq!(RestoreLedger::load(&dir).await.unwrap(), l);
        let mut newer = l.clone();
        newer.format = FORMAT + 1;
        newer.save(&dir).await.unwrap();
        assert!(RestoreLedger::load(&dir).await.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
