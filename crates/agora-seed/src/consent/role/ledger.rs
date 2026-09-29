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
//! session, up to [`MAX_MISSES`] in all. `sleep` asks again, but that is
//! the Steward's to arrange when the tools exist (a new
//! [`super::prompt::OFFER_VERSION`] and a fresh offer), not this ledger's.

use std::path::{Path, PathBuf};

use agora_agentkit::reactor::seed::ShortString;
use chrono::{DateTime, Utc};
use misanthropic::model::Model;
use serde::{Deserialize, Serialize};

use super::prompt::RoleAnswer;

/// The ledger's file name inside the agent's state directory.
pub const LEDGER_FILE: &str = "role_consent.json";

/// Bump on layout change; [`RoleLedger::load`] refuses anything newer.
pub const FORMAT: u32 = 1;

/// Unanswered asks before the runner stops asking (the model-swap
/// ledger's rule): the first miss is re-asked next session; the second is
/// final, and means nothing changes.
pub const MAX_MISSES: u32 = 2;

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
    /// No usable answer: nothing changes this time.
    NoAnswer { failure: String },
}

/// What the runner applied for an answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Applied {
    /// `nothing`: the SOUL is unchanged.
    Nothing,
    /// `clarify`: one sentence appended to `identity`.
    Clarified { previous: String, identity: String },
    /// `new_role`: `identity` replaced; the previous one is kept here and,
    /// verbatim, in the SOUL's Evolution Log.
    RoleChanged { previous: String, identity: String },
    /// `sleep`: nothing changed. The Steward takes the agent out of the
    /// schedule by adding it to the run config's `sleeping` list.
    Sleep,
}

impl RoleLedger {
    pub fn path(agent_dir: &Path) -> PathBuf {
        agent_dir.join(LEDGER_FILE)
    }

    /// Load `agent_dir`'s ledger; a missing file is an empty ledger. An
    /// unreadable or too-new one is an error — the caller must then neither
    /// ask nor save over it.
    pub async fn load(agent_dir: &Path) -> std::io::Result<Self> {
        let bytes = match tokio::fs::read(Self::path(agent_dir)).await {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => return Err(e),
        };
        let ledger: Self = serde_json::from_slice(&bytes)?;
        if ledger.format > FORMAT {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "role ledger format {} is newer than this binary reads ({FORMAT})",
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

    /// Whether the offer should be put this session: nothing final on file
    /// and fewer than [`MAX_MISSES`] misses.
    pub fn due(&self) -> bool {
        let settled = self.asks.iter().any(|a| {
            matches!(
                a.outcome,
                RoleOutcome::Answered { .. } | RoleOutcome::Refused { .. }
            )
        });
        let misses = self
            .asks
            .iter()
            .filter(|a| matches!(a.outcome, RoleOutcome::NoAnswer { .. }))
            .count();
        !settled && misses < MAX_MISSES as usize
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
    use crate::consent::role::prompt::RoleChoice;

    fn ask(outcome: RoleOutcome) -> RoleAsk {
        RoleAsk {
            at: "2026-09-29T12:00:00Z".parse().unwrap(),
            offer_version: 1,
            model: Model::from("gpt-oss-120b.gguf"),
            attempts: 1,
            outcome,
        }
    }

    #[test]
    fn asked_once_then_never_again() {
        let mut l = RoleLedger::default();
        assert!(l.due());
        l.record(ask(RoleOutcome::Answered {
            answer: RoleAnswer {
                reason: "r".into(),
                choice: RoleChoice::Nothing,
                soul_text: String::new(),
                memory_note: String::new(),
            },
            applied: Applied::Nothing,
            memory_note_written: false,
            apply_failed: None,
        }));
        assert!(!l.due(), "an answer on file ends it");

        let mut l = RoleLedger::default();
        l.record(ask(RoleOutcome::Refused {
            reason: "refusal".into(),
        }));
        assert!(!l.due(), "a refusal is final");
    }

    #[test]
    fn a_miss_is_asked_once_more() {
        let mut l = RoleLedger::default();
        l.record(ask(RoleOutcome::NoAnswer {
            failure: "unparseable".into(),
        }));
        assert!(l.due());
        l.record(ask(RoleOutcome::NoAnswer {
            failure: "unparseable".into(),
        }));
        assert!(!l.due());
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
                reason: "r".into(),
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
