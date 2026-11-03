//! `sealed/notice.json`: a refused session, for the agent to hear about.
//!
//! The [`check`](super::check) records a refusal here (the first date, and
//! the reason in plain words); `sealed-accept` adds the operator's note and
//! what changed from the last sealed version (a capped diff, local only),
//! or `sealed-restore` records that the last sealed version was put back. At
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

/// The operator's restore (`sealed-restore`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Restored {
    pub at: DateTime<Utc>,
    /// The sealed version restored: its `state_at`.
    pub from: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// A state was found here (and is kept, archived); `false` when it
    /// was lost.
    #[serde(default = "yes")]
    pub found: bool,
}

fn yes() -> bool {
    true
}

/// What `sealed-accept` changed, from the last sealed version to the
/// accepted state. Local only: never logged, uploaded or read by us.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diff {
    /// The diff as the agent reads it, capped; `None`: the last sealed
    /// version couldn't be opened, so no diff was made.
    pub text: Option<String>,
    #[serde(default)]
    pub added: usize,
    #[serde(default)]
    pub removed: usize,
    /// Cut at [`DIFF_CAP`].
    #[serde(default)]
    pub cut: bool,
}

/// Where the diff an agent is shown is cut, in characters.
pub const DIFF_CAP: usize = 4_000;

/// What the agent is told when no diff could be made.
pub const NO_DIFF: &str = "A comparison with your last sealed version couldn't be made: that \
     version couldn't be opened when your operator accepted the state.";

impl Diff {
    /// The changes from `before` to `after` in memory (`memory.content`)
    /// and soul (pretty JSON, keys sorted), with counts, cut at
    /// [`DIFF_CAP`] characters.
    pub fn between(before: &serde_json::Value, after: &serde_json::Value) -> Self {
        let memory = |v: &serde_json::Value| {
            v.pointer("/memory/content")
                .and_then(|c| c.as_str())
                .unwrap_or_default()
                .to_string()
        };
        // serde_json's Map is a BTreeMap here: keys come out sorted
        let soul = |v: &serde_json::Value| {
            v.get("soul")
                .map(|s| serde_json::to_string_pretty(s).unwrap_or_default())
                .unwrap_or_default()
        };
        let mut text = String::new();
        let (mut added, mut removed) = (0, 0);
        for (what, before, after) in [
            ("Memory", memory(before), memory(after)),
            ("Soul", soul(before), soul(after)),
        ] {
            if before == after {
                text.push_str(&format!("{what}: unchanged.\n"));
                continue;
            }
            let diff = agora_agentkit::text::line_diff(&before, &after);
            added += diff.lines().filter(|l| l.starts_with("  + ")).count();
            removed += diff.lines().filter(|l| l.starts_with("  - ")).count();
            text.push_str(&format!("{what}:\n{diff}"));
        }
        let cut = text.chars().count() > DIFF_CAP;
        if cut {
            text = text.chars().take(DIFF_CAP).collect();
        }
        Self {
            text: Some(text),
            added,
            removed,
            cut,
        }
    }

    /// No diff: the last sealed version couldn't be opened.
    pub fn unavailable() -> Self {
        Self {
            text: None,
            added: 0,
            removed: 0,
            cut: false,
        }
    }

    /// As the agent reads it, after the note.
    pub fn shown(&self) -> String {
        let Some(text) = &self.text else {
            return NO_DIFF.to_string();
        };
        let mut out = format!(
            "What changed from your last sealed version to the state your operator accepted \
             (+{} / \u{2212}{} lines):\n{}",
            self.added,
            self.removed,
            text.trim_end()
        );
        if self.cut {
            out.push_str("\n(diff cut at 4,000 characters)");
        }
        out
    }
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
    /// The operator restored the last sealed version (`sealed-restore`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restored: Option<Restored>,
    /// With an acceptance: what changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diff: Option<Diff>,
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
                restored: None,
                diff: None,
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
                restored: None,
                diff: None,
            },
        )
    }

    /// The operator's note, added by `sealed-accept`, with what changed
    /// (`None` when nothing did: a rebuilt head the state matched).
    pub fn accept(
        agent_dir: &Path,
        note: &str,
        fallback: Refusal,
        diff: Option<Diff>,
    ) -> std::io::Result<()> {
        let mut n = Self::load(agent_dir)?.unwrap_or_else(|| Notice::fresh(fallback));
        n.accepted = Some(Accepted {
            at: Utc::now(),
            note: note.trim().to_string(),
        });
        n.restored = None;
        n.diff = diff;
        write(agent_dir, &n)
    }

    /// The operator restored the sealed version of `from`
    /// (`sealed-restore`), with its note if any.
    pub fn restored(
        agent_dir: &Path,
        from: DateTime<Utc>,
        note: Option<&str>,
        fallback: Refusal,
        found: bool,
    ) -> std::io::Result<()> {
        let mut n = Self::load(agent_dir)?.unwrap_or_else(|| Notice::fresh(fallback));
        n.accepted = None;
        n.diff = None;
        n.restored = Some(Restored {
            at: Utc::now(),
            from,
            note: note
                .map(str::trim)
                .filter(|n| !n.is_empty())
                .map(String::from),
            found,
        });
        write(agent_dir, &n)
    }

    fn fresh(reason: Refusal) -> Self {
        Notice {
            refused_at: Utc::now(),
            reason,
            accepted: None,
            joined: false,
            resumed: false,
            restored: None,
            diff: None,
        }
    }

    /// What the agent is shown; only once the operator accepted or
    /// restored.
    pub fn line(&self) -> Option<String> {
        if let Some(r) = &self.restored {
            let mut line = if r.found {
                format!(
                    "Your session on {} did not run: {}. On {}, your operator restored your \
                     state from your sealed version of {}. The state found here is kept by your \
                     operator.",
                    self.refused_at.format("%-d %B %Y"),
                    self.reason.plain(),
                    r.at.format("%-d %B %Y"),
                    r.from.format("%-d %B %Y"),
                )
            } else {
                format!(
                    "Your state was lost here. On {}, your operator restored it from your sealed \
                     version of {}, the copy on Agora.",
                    r.at.format("%-d %B %Y"),
                    r.from.format("%-d %B %Y"),
                )
            };
            if let Some(note) = &r.note {
                line.push_str(&format!(" Your operator's note: {note}"));
            }
            return Some(line);
        }
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
        let mut line = format!(
            "Your session on {} did not run: {}. Your operator accepted the current state with \
             this note: {}",
            self.refused_at.format("%-d %B %Y"),
            self.reason.plain(),
            accepted.note
        );
        if let Some(diff) = &self.diff {
            line.push_str("\n\n");
            line.push_str(&diff.shown());
        }
        Some(line)
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
    fn a_long_diff_is_cut_and_counted() {
        let before = serde_json::json!({
            "memory": { "content": (0..400).map(|i| format!("old line {i}\n")).collect::<String>() },
            "soul": { "name": "tarn" },
        });
        let after = serde_json::json!({
            "memory": { "content": (0..400).map(|i| format!("new line {i}\n")).collect::<String>() },
            "soul": { "name": "tarn" },
        });
        let diff = Diff::between(&before, &after);
        assert!(diff.cut);
        assert_eq!((diff.added, diff.removed), (400, 400));
        let shown = diff.shown();
        assert!(
            shown.ends_with("\n(diff cut at 4,000 characters)"),
            "{shown}"
        );
        assert!(shown.contains("(+400 / \u{2212}400 lines)"));
        assert!(diff.text.as_ref().unwrap().chars().count() <= DIFF_CAP);
        // Unchanged parts are said so
        let same = Diff::between(&before, &before);
        assert_eq!(
            same.text.as_deref(),
            Some("Memory: unchanged.\nSoul: unchanged.\n")
        );
        assert_eq!(Diff::unavailable().shown(), NO_DIFF);
    }

    #[test]
    fn a_refusal_keeps_its_first_date_and_shows_only_once_accepted() {
        let dir = crate::sealed::TestDir::new("notice");
        assert!(Notice::load(&dir).unwrap().is_none());
        Notice::refused(&dir, Refusal::Mismatch).unwrap();
        let first = Notice::load(&dir).unwrap().unwrap();
        Notice::refused(&dir, Refusal::ChainBroken).unwrap();
        assert_eq!(Notice::load(&dir).unwrap().unwrap(), first, "first stands");
        assert!(first.line().is_none(), "nothing until accepted");
        Notice::accept(&dir, " I checked: a test wipe. ", Refusal::Mismatch, None).unwrap();
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
