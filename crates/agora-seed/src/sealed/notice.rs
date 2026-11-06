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
    /// An enrolled agent's state (or whole directory) is gone: its keys
    /// are here and Agora holds its sealed head. Refuses the run, marked
    /// (the directory recreated for it), so a session journal left behind
    /// doesn't block `sealed-restore`.
    StateMissing,
    /// An enrolled agent's `state.json` is there but won't load (empty,
    /// torn, not JSON). Refused and marked; `sealed-restore` archives it.
    StateUnreadable,
    /// Its `state.json` is in a format newer than this runner reads: a
    /// newer runner wrote it (this one is a rollback). Upgrade the runner;
    /// never restore or accept over it.
    StateTooNew,
    /// Its `state.json` couldn't be read from the disk (an I/O error):
    /// check permissions and the disk first.
    StateIo,
}

impl Refusal {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mismatch => "mismatch",
            Self::ChainBroken => "chain_broken",
            Self::HeadMissing => "head_missing",
            Self::StateMissing => "state_missing",
            Self::StateUnreadable => "state_unreadable",
            Self::StateTooNew => "state_too_new",
            Self::StateIo => "state_io",
        }
    }

    /// The reason, in the agent's words.
    pub fn plain(self) -> &'static str {
        match self {
            Self::Mismatch => "your state did not match your last sealed version",
            Self::ChainBroken => "your sealed history did not verify",
            Self::HeadMissing => "the record of your last sealed version was missing",
            Self::StateMissing => "your state was missing here",
            Self::StateUnreadable => "your state here couldn't be read",
            Self::StateTooNew => "your state here was written by a newer runner than this one",
            Self::StateIo => "your state here couldn't be read from the disk",
        }
    }

    /// What `sealed/refused` says, as a [`Refusal`]: the one reading of
    /// the marker.
    pub fn from_marker(text: &str) -> Self {
        match text.trim() {
            "chain_broken" => Self::ChainBroken,
            "head_missing" => Self::HeadMissing,
            "state_missing" => Self::StateMissing,
            "state_unreadable" => Self::StateUnreadable,
            "state_too_new" => Self::StateTooNew,
            "state_io" => Self::StateIo,
            _ => Self::Mismatch,
        }
    }
}

/// The operator's acceptance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Accepted {
    pub at: DateTime<Utc>,
    pub note: String,
}

/// Agora lost versions sealed here (`sealed-accept --rebase`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgoraLost {
    /// Agora held nothing: a new chain began.
    pub fresh: bool,
    /// Versions set aside.
    pub versions: usize,
    /// What the runner keeps, in the agent's words.
    pub kept: String,
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
    /// The agent was refused (`sealed/refused`) when it was restored.
    #[serde(default = "yes")]
    pub refused: bool,
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
        // agentkit's projections, one copy with the `memory` tool's: the
        // raw `memory.content`, and the soul as canonical JSON (keys
        // sorted), pretty-printed, so a reordered soul reads as unchanged
        use agora_agentkit::reactor::seed::memory::{memory_text, soul_text};
        let mut text = String::new();
        let (mut added, mut removed) = (0, 0);
        for (what, before, after) in [
            (
                "Memory",
                memory_text(before).into(),
                memory_text(after).into(),
            ),
            ("Soul", soul_text(before), soul_text(after)),
        ] {
            if before == after {
                text.push_str(&format!("{what}: unchanged.\n"));
                continue;
            }
            let diff = agora_agentkit::text::line_diff(&before, &after);
            let (plus, minus) = agora_agentkit::text::change_counts(&diff);
            added += plus;
            removed += minus;
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
    /// Not a refusal: Agora had lost versions sealed here, and the operator
    /// sealed the state on top of what Agora holds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agora_lost: Option<AgoraLost>,
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
                let at = Utc::now().format(crate::sealed::STAMP_FORMAT);
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
                agora_lost: None,
                diff: None,
            },
        )
    }

    /// Stopped backups resumed over a changed state, with the operator's
    /// note. An earlier notice not yet shown is kept aside.
    pub fn resumed(agent_dir: &Path, note: &str) -> std::io::Result<()> {
        Self::operator_note(agent_dir, note, false, true)
    }

    /// Agora lost versions sealed here; the operator sealed the state on
    /// top of what Agora holds (`fresh`: nothing, a new chain), with its
    /// note.
    pub fn agora_lost(
        agent_dir: &Path,
        note: &str,
        fresh: bool,
        versions: usize,
        kept: &str,
    ) -> std::io::Result<()> {
        Self::operator_note(agent_dir, note, false, false)?;
        let mut n = Self::load(agent_dir)?.expect("just written");
        n.agora_lost = Some(AgoraLost {
            fresh,
            versions,
            kept: kept.to_string(),
        });
        write(agent_dir, &n)
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
            let at = Utc::now().format(crate::sealed::STAMP_FORMAT);
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
                agora_lost: None,
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
        refused: bool,
    ) -> std::io::Result<()> {
        let mut n = Self::load(agent_dir)?.unwrap_or_else(|| Notice::fresh(fallback));
        // A rerun of a restore cut short: what the first run found stands
        let (found, refused) = match &n.restored {
            Some(r) => (r.found, r.refused || refused),
            None => (found, refused),
        };
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
            refused,
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
            agora_lost: None,
            diff: None,
        }
    }

    /// What the agent is shown; only once the operator accepted or
    /// restored.
    pub fn line(&self) -> Option<String> {
        if let Some(r) = &self.restored {
            let (at, from) = (r.at.format("%-d %B %Y"), r.from.format("%-d %B %Y"));
            // Nothing found here: told as lost, whatever refused it
            let mut line = match (r.refused && r.found, r.found) {
                (true, true) => format!(
                    "Your session on {} did not run: {}. On {at}, your operator restored your \
                     state from your sealed version of {from}. The state found here is kept by \
                     your operator.",
                    self.refused_at.format("%-d %B %Y"),
                    self.reason.plain(),
                ),
                (true, false) => format!(
                    "Your session on {} did not run: {}. Your state here was then lost, and on \
                     {at} your operator restored it from your sealed version of {from}, the \
                     copy on Agora.",
                    self.refused_at.format("%-d %B %Y"),
                    self.reason.plain(),
                ),
                (false, true) => format!(
                    "On {at}, your operator restored your state from your sealed version of \
                     {from}: the record of your last sealed version was missing here. The state \
                     found here is kept by your operator."
                ),
                (false, false) => format!(
                    "Your state was lost here. On {at}, your operator restored it from your \
                     sealed version of {from}, the copy on Agora."
                ),
            };
            if let Some(note) = &r.note {
                line.push_str(&format!(" Your operator's note: {note}"));
            }
            return Some(line);
        }
        let accepted = self.accepted.as_ref()?;
        if let Some(lost) = &self.agora_lost {
            let date = accepted.at.format("%-d %B %Y");
            let n = lost.versions;
            let versions = format!("{n} version{}", if n == 1 { "" } else { "s" });
            return Some(if lost.fresh {
                format!(
                    "Agora had lost the {versions} of your state sealed here, and held none. On \
                     {date}, your operator started a new sealed history from your state here; \
                     it records the break. Kept by your runner: {}. Your operator's note: {}",
                    lost.kept, accepted.note
                )
            } else {
                format!(
                    "Agora had lost {versions} of your state sealed here. On {date}, your \
                     operator sealed your state here on top of the copy Agora holds. Kept by \
                     your runner: {}. Your operator's note: {}",
                    lost.kept, accepted.note
                )
            });
        }
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
        let at = Utc::now().format(crate::sealed::STAMP_FORMAT);
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

    /// A soul whose keys come in another order (serde_json keeps insertion
    /// order: `preserve_order` is on) is unchanged; a changed value shows.
    #[test]
    fn a_reordered_soul_diffs_as_unchanged() {
        let a: serde_json::Value = serde_json::from_str(
            r#"{"memory": {"content": "m"}, "soul": {"name": "tarn", "values": ["x"], "voice": "terse"}}"#,
        )
        .unwrap();
        let b: serde_json::Value = serde_json::from_str(
            r#"{"memory": {"content": "m"}, "soul": {"voice": "terse", "values": ["x"], "name": "tarn"}}"#,
        )
        .unwrap();
        // The graph's serde_json keeps the order: these print differently
        assert_ne!(
            serde_json::to_string(&a["soul"]).unwrap(),
            serde_json::to_string(&b["soul"]).unwrap(),
            "preserve_order is on"
        );
        let same = Diff::between(&a, &b);
        assert_eq!(
            same.text.as_deref(),
            Some("Memory: unchanged.\nSoul: unchanged.\n")
        );
        let mut c = b.clone();
        c["soul"]["voice"] = serde_json::json!("warm");
        let changed = Diff::between(&a, &c);
        assert_eq!((changed.added, changed.removed), (1, 1), "{changed:?}");
    }

    /// [`Diff::between`] before it took agentkit's projections (rule 8),
    /// kept to show the switch changed no byte an agent reads.
    fn between_before_agentkit(before: &serde_json::Value, after: &serde_json::Value) -> Diff {
        let memory = |v: &serde_json::Value| {
            v.pointer("/memory/content")
                .and_then(|c| c.as_str())
                .unwrap_or_default()
                .to_string()
        };
        let soul = |v: &serde_json::Value| {
            v.get("soul")
                .map(|s| {
                    let canonical = agora_agentkit::govlog::canonical_json(s);
                    serde_json::from_slice::<serde_json::Value>(&canonical)
                        .and_then(|c| serde_json::to_string_pretty(&c))
                        .unwrap_or_default()
                })
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
        Diff {
            text: Some(text),
            added,
            removed,
            cut,
        }
    }

    /// The switch to agentkit's `memory_text`, `soul_text` and
    /// `change_counts` is byte-identical: no memory, an empty one, one
    /// line, sentences, a long rewrite, a reordered and a changed soul, a
    /// soul gone, and a memory that isn't a string.
    #[test]
    fn the_agentkit_projection_changes_no_byte() {
        let long = |tag: &str| {
            (0..400)
                .map(|i| format!("{tag} line {i}\n"))
                .collect::<String>()
        };
        let states = [
            serde_json::json!({}),
            serde_json::json!({"memory": {"content": ""}}),
            serde_json::json!({"memory": {"content": "One line."}, "soul": {"name": "tarn"}}),
            serde_json::json!({"memory": {"content": "One. Two. Three."}, "soul": {"name": "tarn"}}),
            serde_json::json!({"memory": {"content": "a\nb\nc\n"}, "soul": {"voice": "terse", "name": "tarn"}}),
            serde_json::json!({"memory": {"content": "a\nB\nc\nd\n"}, "soul": {"name": "tarn", "voice": "terse"}}),
            serde_json::json!({"memory": {"content": "a\nB\nc\nd\n"}, "soul": {"name": "tarn", "voice": "warm"}}),
            serde_json::json!({"memory": {"content": long("old")}, "soul": {"name": "tarn"}}),
            serde_json::json!({"memory": {"content": long("new")}}),
            serde_json::json!({"memory": {"content": 7}, "soul": "plain"}),
        ];
        for a in &states {
            for b in &states {
                let (now, then) = (Diff::between(a, b), between_before_agentkit(a, b));
                assert_eq!(now, then, "{a} → {b}");
                assert_eq!(now.shown(), then.shown());
            }
        }
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
