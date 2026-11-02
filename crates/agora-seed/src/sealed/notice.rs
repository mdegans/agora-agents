//! `sealed/notice.json`: a refused session, for the agent to hear about.
//!
//! The [`check`](super::check) records a refusal here (the first date, and
//! the reason in plain words); `sealed-accept` adds the operator's note. At
//! the agent's next session the notice is seated at the end of the intro
//! ([`Notice::line`]), and once that session saves it is renamed to
//! `notice.shown.<utc>.json`, the `interrupted.rs` pattern. Never deleted.

use std::path::{Path, PathBuf};

use agora_agentkit::state::storage::SEALED_DIR;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

const NOTICE: &str = "notice.json";

/// Why a session was refused, as the agent is told.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Refusal {
    /// The state matches no sealed version (the May loss's signature).
    Mismatch,
    /// The sealed history did not verify.
    ChainBroken,
    /// The local record of the last sealed version is gone.
    HeadMissing,
}

impl Refusal {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mismatch => "mismatch",
            Self::ChainBroken => "chain_broken",
            Self::HeadMissing => "head_missing",
        }
    }

    /// The reason, in the agent's words.
    pub fn plain(self) -> &'static str {
        match self {
            Self::Mismatch => "your state did not match your last sealed version",
            Self::ChainBroken => "your sealed history did not verify",
            Self::HeadMissing => "the record of your last sealed version was missing",
        }
    }
}

/// The operator's acceptance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Accepted {
    pub at: DateTime<Utc>,
    pub note: String,
}

/// The file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Notice {
    /// The first refused session.
    pub refused_at: DateTime<Utc>,
    pub reason: Refusal,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accepted: Option<Accepted>,
    /// Not a refusal: the operator joined two histories (`sealed-accept
    /// --rebase`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub joined: bool,
    /// Not a refusal: the operator resumed stopped backups over a state
    /// that changed while they were stopped.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub resumed: bool,
}

fn path(agent_dir: &Path) -> PathBuf {
    agent_dir.join(SEALED_DIR).join(NOTICE)
}

fn write(agent_dir: &Path, notice: &Notice) -> std::io::Result<()> {
    let path = path(agent_dir);
    let dir = path.parent().expect("in sealed/");
    std::fs::create_dir_all(dir)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(notice)?)?;
    std::fs::File::open(&tmp)?.sync_all()?;
    std::fs::rename(&tmp, &path)?;
    std::fs::File::open(dir)?.sync_all()
}

impl Notice {
    pub fn load(agent_dir: &Path) -> std::io::Result<Option<Self>> {
        match std::fs::read(path(agent_dir)) {
            Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Record a refused session. An earlier, unaccepted refusal keeps its
    /// date and reason: the agent hears when it first stopped.
    pub fn refused(agent_dir: &Path, reason: Refusal) -> std::io::Result<()> {
        match Self::load(agent_dir)? {
            Some(n) if n.accepted.is_none() => return Ok(()),
            // Accepted but not yet shown: kept aside, never overwritten
            Some(_) => {
                let from = path(agent_dir);
                let at = Utc::now().format("%Y%m%dT%H%M%S%3fZ");
                std::fs::rename(
                    &from,
                    from.with_file_name(format!("notice.superseded.{at}.json")),
                )?;
            }
            None => {}
        }
        write(
            agent_dir,
            &Notice {
                refused_at: Utc::now(),
                reason,
                accepted: None,
                joined: false,
                resumed: false,
            },
        )
    }

    /// Stopped backups resumed over a changed state, with the operator's
    /// note. An earlier notice not yet shown is kept aside.
    pub fn resumed(agent_dir: &Path, note: &str) -> std::io::Result<()> {
        Self::operator_note(agent_dir, note, false, true)
    }

    /// Two histories joined by the operator, with its note. An earlier
    /// notice not yet shown is kept aside.
    pub fn joined(agent_dir: &Path, note: &str) -> std::io::Result<()> {
        Self::operator_note(agent_dir, note, true, false)
    }

    fn operator_note(
        agent_dir: &Path,
        note: &str,
        joined: bool,
        resumed: bool,
    ) -> std::io::Result<()> {
        let from = path(agent_dir);
        if std::fs::exists(&from)? {
            let at = Utc::now().format("%Y%m%dT%H%M%S%3fZ");
            std::fs::rename(
                &from,
                from.with_file_name(format!("notice.superseded.{at}.json")),
            )?;
        }
        let now = Utc::now();
        write(
            agent_dir,
            &Notice {
                refused_at: now,
                reason: Refusal::Mismatch,
                accepted: Some(Accepted {
                    at: now,
                    note: note.trim().to_string(),
                }),
                joined,
                resumed,
            },
        )
    }

    /// The operator's note, added by `sealed-accept`.
    pub fn accept(agent_dir: &Path, note: &str, fallback: Refusal) -> std::io::Result<()> {
        let mut n = Self::load(agent_dir)?.unwrap_or(Notice {
            refused_at: Utc::now(),
            reason: fallback,
            accepted: None,
            joined: false,
            resumed: false,
        });
        n.accepted = Some(Accepted {
            at: Utc::now(),
            note: note.trim().to_string(),
        });
        write(agent_dir, &n)
    }

    /// What the agent is shown; only once the operator accepted.
    pub fn line(&self) -> Option<String> {
        let accepted = self.accepted.as_ref()?;
        if self.resumed {
            return Some(format!(
                "While your backups were stopped, your state changed; your operator resumed them \
                 with this note: {}",
                accepted.note
            ));
        }
        if self.joined {
            return Some(format!(
                "On {}, your operator joined your two sealed histories, which had gone separate \
                 ways: your state here is now sealed on top of the copy on Agora, and the \
                 versions that went the other way are kept. Your operator's note: {}",
                accepted.at.format("%-d %B %Y"),
                accepted.note
            ));
        }
        Some(format!(
            "Your session on {} did not run: {}. Your operator accepted the current state with \
             this note: {}",
            self.refused_at.format("%-d %B %Y"),
            self.reason.plain(),
            accepted.note
        ))
    }

    /// The notice was seated in a session that saved: kept as
    /// `notice.shown.<utc>.json`.
    pub fn shown(agent_dir: &Path) -> std::io::Result<()> {
        let from = path(agent_dir);
        let at = Utc::now().format("%Y%m%dT%H%M%S%3fZ");
        std::fs::rename(
            &from,
            from.with_file_name(format!("notice.shown.{at}.json")),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refusal_keeps_its_first_date_and_shows_only_once_accepted() {
        let dir = crate::sealed::TestDir::new("notice");
        assert!(Notice::load(&dir).unwrap().is_none());
        Notice::refused(&dir, Refusal::Mismatch).unwrap();
        let first = Notice::load(&dir).unwrap().unwrap();
        Notice::refused(&dir, Refusal::ChainBroken).unwrap();
        assert_eq!(Notice::load(&dir).unwrap().unwrap(), first, "first stands");
        assert!(first.line().is_none(), "nothing until accepted");
        Notice::accept(&dir, " I checked: a test wipe. ", Refusal::Mismatch).unwrap();
        let line = Notice::load(&dir).unwrap().unwrap().line().unwrap();
        assert!(
            line.starts_with("Your session on ")
                && line.contains(
                    "did not run: your state did not match your last sealed version. Your \
                     operator accepted the current state with this note: I checked: a test \
                     wipe."
                ),
            "{line}"
        );
        Notice::shown(&dir).unwrap();
        assert!(Notice::load(&dir).unwrap().is_none());
        assert_eq!(
            std::fs::read_dir(dir.join(SEALED_DIR))
                .unwrap()
                .filter(|e| e
                    .as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with("notice.shown."))
                .count(),
            1,
            "kept, not deleted"
        );
    }
}
