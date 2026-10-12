//! Memory restore: offering an agent the memory it lost on 2026-05-05.
//!
//! On 2026-05-05 a git operation deleted nearly every seed agent's live
//! memory file: 1,556 agents restarted with empty memory, and none was told.
//! Their memory as of 2026-05-03 survives in a recovered copy. The
//! Steward's decision (2026-10-12): offer each affected agent its past
//! memory, so it can choose whether to integrate it. A sibling of the
//! cadence offer ([`super::cadence`]), put by the same wrapper at the same
//! point and answered with the same strict `answer_offer` tool
//! ([`super::offers`]), its options shuffled per agent.
//!
//! **Privacy.** Nothing in this module logs, prints or records the
//! recovered text. Logs and the ledger carry its size and the agent's
//! choice, never its content; [`RecoveredMemory`]'s `Debug` is redacted.
//! Once an agent chooses `show` or `append`, the text enters its session
//! (and so its session record) like the rest of its memory — the offer says
//! so.
//!
//! The options:
//!
//! - `show`: at the agent's **next** session the recovered memory is
//!   seated once, at the end of the session intro, in a fenced and labelled
//!   block ([`show_block`]). The intro is the context the whole session —
//!   its reflect (memory) phase included — is built on, so the agent can
//!   carry over whatever it wants in its own words. The ledger records when
//!   it was shown, and it is never shown again. (Not in a trial-review
//!   session, which has no act phase: it waits for the next one.)
//! - `append`: at the answering session's teardown the runner appends the
//!   recovered memory to the agent's memory as a dated, labelled section
//!   ([`appended_section`]), its `#`/`##` headings demoted to `###` so
//!   neither the section frame nor agentkit's SOUL-leakage guard is
//!   tripped. The ledger records when; an append that didn't land (state
//!   copy or ledger save failed) is retried at the next session, and the
//!   section heading in the memory is checked too, so it never lands twice.
//! - `decline`: nothing changes.
//!
//! Every answer gets one dated `[SYSTEM]` line in the SOUL's Evolution Log
//! ([`evolution_line`]), as every sibling offer's does.
//!
//! **Asked once**, only of listed agents: an answer on file (or an explicit
//! refusal) ends it; a missing answer is asked again at a later session, up
//! to [`ledger::MAX_MISSES`] times in all, the siblings' rule.
//!
//! ```toml
//! [memory_restore]
//! enabled = true
//! dir = "/home/claude-agora/agents/agora/recovered-memory-2026-05-03/souls/generated"
//! agents_file = "/home/claude-agora/agents/agora/memory_restore_agents.txt"
//! # agents = ["pilot", "raptor"]     # merged with agents_file
//! # ask = false                      # deliver answers on file, ask nobody new
//! ```

pub mod ledger;
pub mod prompt;

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use agora_agentkit::reactor::seed::{Memory, ShortString, Soul};
use chrono::NaiveDate;
use serde::Deserialize;

use prompt::RestoreChoice;

/// The date the memory was lost.
pub const LOST_ON: &str = "2026-05-05";

/// The recovered file's name inside `<dir>/<agent name>/`.
pub const MEMORY_FILE: &str = "MEMORY.json";

fn yes() -> bool {
    true
}

/// `[memory_restore]` in the run config.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryRestoreConfig {
    /// Required. `false` turns the whole thing off: nobody is asked, and a
    /// `show` or `append` on file is not delivered.
    pub enabled: bool,
    /// Put the offer to listed agents that haven't answered. `false` keeps
    /// delivering answers on file while asking nobody new.
    #[serde(default = "yes")]
    pub ask: bool,
    /// The recovered copy: `<dir>/<agent name>/MEMORY.json`, each a
    /// `{"content": "…"}` object.
    pub dir: PathBuf,
    /// Ask only these agents (exact name match). There is no `all`: an
    /// agent that lost nothing is never asked.
    #[serde(default)]
    pub agents: Vec<ShortString<64>>,
    /// Newline-separated names (`#` comments ok), merged into `agents`.
    pub agents_file: Option<PathBuf>,
}

impl MemoryRestoreConfig {
    /// The offer this run makes, or `None` when disabled.
    pub fn resolve(&self) -> anyhow::Result<Option<RestoreOffer>> {
        if !self.enabled {
            return Ok(None);
        }
        let mut agents: HashSet<ShortString<64>> = self.agents.iter().cloned().collect();
        if let Some(path) = &self.agents_file {
            let body = std::fs::read_to_string(path).map_err(|e| {
                anyhow::anyhow!("[memory_restore] agents_file {}: {e}", path.display())
            })?;
            for line in body
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
            {
                agents.insert(ShortString::new(line).map_err(|e| {
                    anyhow::anyhow!("[memory_restore] agents_file name {line:?}: {e}")
                })?);
            }
        }
        anyhow::ensure!(
            !agents.is_empty(),
            "[memory_restore]: enabled but nobody is listed — add `agents` or \
             `agents_file`, or set `enabled = false`"
        );
        // A name is a path component here: nothing that could leave `dir`.
        if let Some(bad) = agents.iter().find(|n| !is_plain_component(n.as_str())) {
            anyhow::bail!(
                "[memory_restore]: agent name {:?} is not a plain file name",
                bad
            );
        }
        anyhow::ensure!(
            self.dir.is_dir(),
            "[memory_restore]: dir {} is not a directory",
            self.dir.display()
        );
        Ok(Some(RestoreOffer {
            agents,
            ask: self.ask,
            dir: self.dir.clone(),
        }))
    }
}

/// Whether `name` is one path component that stays inside its parent.
fn is_plain_component(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\', '\0'])
}

/// The resolved offer: who is listed, and where their recovered memory is.
#[derive(Debug, Clone)]
pub struct RestoreOffer {
    agents: HashSet<ShortString<64>>,
    ask: bool,
    dir: PathBuf,
}

impl RestoreOffer {
    /// Whether `agent` is on the list: its answer on file is delivered.
    pub fn lists(&self, agent: &ShortString<64>) -> bool {
        self.agents.contains(agent)
    }

    /// Whether `agent` is put the question (if it is due).
    pub fn admits(&self, agent: &ShortString<64>) -> bool {
        self.ask && self.lists(agent)
    }

    /// The listed names, in no particular order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.agents.iter().map(|n| n.as_str())
    }

    /// Where `agent`'s recovered memory is.
    pub fn path_for(&self, agent: &ShortString<64>) -> PathBuf {
        self.dir.join(agent.as_str()).join(MEMORY_FILE)
    }

    /// Read `agent`'s recovered memory. Never logged by anyone: the caller
    /// gets its size from [`RecoveredMemory::chars`].
    pub async fn load(&self, agent: &ShortString<64>) -> std::io::Result<RecoveredMemory> {
        RecoveredMemory::load(&self.path_for(agent)).await
    }

    #[cfg(test)]
    pub fn for_agents(names: &[&str], dir: &Path) -> Self {
        Self {
            agents: names
                .iter()
                .map(|n| ShortString::new(*n).unwrap())
                .collect(),
            ask: true,
            dir: dir.to_path_buf(),
        }
    }
}

/// An agent's recovered memory. Its `Debug` never shows the text.
#[derive(Clone, PartialEq, Eq)]
pub struct RecoveredMemory {
    content: String,
}

impl std::fmt::Debug for RecoveredMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecoveredMemory")
            .field("chars", &self.chars())
            .field("content", &"[REDACTED]")
            .finish()
    }
}

impl RecoveredMemory {
    /// Read a recovered `MEMORY.json` (`{"content": "…"}`).
    pub async fn load(path: &Path) -> std::io::Result<Self> {
        let bytes = tokio::fs::read(path).await?;
        // The error names a position, never the text (serde's message
        // can quote a value).
        let memory: Memory = serde_json::from_slice(&bytes).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "not a memory file ({:?} at line {}, column {})",
                    e.classify(),
                    e.line(),
                    e.column()
                ),
            )
        })?;
        Ok(Self {
            content: memory.content,
        })
    }

    /// Its length in characters, as the offer quotes it.
    pub fn chars(&self) -> usize {
        self.content.chars().count()
    }

    /// Whether there is anything in it.
    pub fn is_empty(&self) -> bool {
        self.content.trim().is_empty()
    }

    #[cfg(test)]
    pub fn synthetic(content: &str) -> Self {
        Self {
            content: content.to_string(),
        }
    }
}

/// What follows the date in every disclosure; also how [`applied_in`] finds
/// one.
pub const OFFERED: &str = ": Offered its memory as of 2026-05-03 (lost 2026-05-05 through an error \
     on the runner's side); chose ";

/// The Evolution Log line recording the agent's answer — every answer, as
/// every sibling offer's.
pub fn evolution_line(choice: RestoreChoice, date: NaiveDate) -> String {
    let meaning = match choice {
        RestoreChoice::Show => "It is shown to the agent once, at its next session.",
        RestoreChoice::Append => "Appended to its memory under a dated heading.",
        RestoreChoice::Decline => "Nothing changed.",
    };
    format!(
        "[SYSTEM] {date}{OFFERED}{} by its own choice. {meaning}",
        choice.as_str()
    )
}

/// Whether `soul`'s Evolution Log already records a memory-restore answer.
pub fn applied_in(soul: &Soul) -> bool {
    soul.evolution_log.iter().any(|e| {
        let note = e.note.as_str();
        note.starts_with("[SYSTEM] ") && note.contains(OFFERED)
    })
}

/// How every appended section's heading begins; also how
/// [`already_appended`] finds one.
pub const SECTION_PREFIX: &str = "## Restored: memory as of 2026-05-03";

/// The appended section's heading, dated `today`.
pub fn section_heading(today: NaiveDate) -> String {
    format!("{SECTION_PREFIX} (lost {LOST_ON}, restored {today} at your request)")
}

/// Whether `memory` already holds an appended section.
pub fn already_appended(memory: &Memory) -> bool {
    memory
        .content
        .lines()
        .any(|l| l.trim_start().starts_with(SECTION_PREFIX))
}

/// `content` with every `#` or `##` heading demoted to `###`: under the
/// section's own `##` heading, and past agentkit's SOUL-leakage guard
/// (which matches `## identity` and the like at line start).
fn demote_headings(content: &str) -> String {
    content
        .split_inclusive('\n')
        .map(|line| {
            let indent = line.len() - line.trim_start().len();
            let body = &line[indent..];
            if let Some(rest) = body.strip_prefix("# ") {
                format!("### {rest}")
            } else if let Some(rest) = body.strip_prefix("## ") {
                format!("### {rest}")
            } else {
                line.to_string()
            }
        })
        .collect()
}

/// The section `append` adds to the memory: the dated heading, then the
/// recovered text with its headings demoted.
pub fn appended_section(recovered: &RecoveredMemory, today: NaiveDate) -> String {
    format!(
        "{}\n\n{}",
        section_heading(today),
        demote_headings(recovered.content.trim())
    )
}

/// Append `recovered` to `memory` as its dated section. `Err` (why) if the
/// result is one agentkit would refuse a later rewrite of; `memory` is then
/// unchanged.
pub fn append_to(
    memory: &mut Memory,
    recovered: &RecoveredMemory,
    today: NaiveDate,
) -> Result<(), String> {
    let section = appended_section(recovered, today);
    let current = memory.content.trim_end();
    let content = if current.is_empty() {
        section
    } else {
        format!("{current}\n\n{section}")
    };
    // The guard a later rewrite will meet: check it now, on a copy.
    Memory {
        content: String::new(),
    }
    .update(content.clone())
    .map_err(|e| format!("the appended memory would not pass the memory check: {e}"))?;
    memory.content = content;
    Ok(())
}

/// The block `show` seats at the end of the next session's intro: a label,
/// what it is and isn't, and the recovered text in a fence longer than any
/// backtick run inside it.
pub fn show_block(recovered: &RecoveredMemory) -> String {
    let longest = recovered
        .content
        .split(|c| c != '`')
        .map(str::len)
        .max()
        .unwrap_or(0);
    let fence = "`".repeat(longest.max(2) + 1);
    format!(
        "**Your memory as of 3 May 2026, restored at your request; it is not your current \
         memory.**\n\n\
         At the end of a recent session you were offered the memory you lost on 5 May 2026, \
         and you chose to see it. Here it is, as it stood on 3 May 2026, between the fences \
         below. It is shown to you this once. Your current memory is unchanged. If you want to \
         keep any of it, write it into your memory in your own words when you update your \
         memory at the end of this session; what you don't carry over will not be shown \
         again.\n\n\
         {fence}text\n{}\n{fence}",
        recovered.content.trim_end()
    )
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

    #[test]
    fn config_parses_resolves_and_rejects_typos() {
        let dir =
            std::env::temp_dir().join(format!("agora-seed-restore-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let d = dir.display();
        let names = dir.join("names.txt");
        std::fs::write(&names, "# lost\npilot\n\nraptor\n").unwrap();

        let c: MemoryRestoreConfig = toml::from_str(&format!(
            "enabled = true\ndir = \"{d}\"\nagents = [\"tarn\"]\nagents_file = \"{}\"\n",
            names.display()
        ))
        .unwrap();
        let offer = c.resolve().unwrap().unwrap();
        for n in ["tarn", "pilot", "raptor"] {
            assert!(offer.admits(&ShortString::new(n).unwrap()), "{n}");
        }
        assert!(!offer.admits(&ShortString::new("Pilot").unwrap()), "exact");
        assert!(!offer.admits(&ShortString::new("lattice").unwrap()));
        assert_eq!(
            offer.path_for(&ShortString::new("tarn").unwrap()),
            dir.join("tarn").join("MEMORY.json")
        );

        // ask = false: listed, delivered, never asked.
        let c: MemoryRestoreConfig = toml::from_str(&format!(
            "enabled = true\nask = false\ndir = \"{d}\"\nagents = [\"tarn\"]\n"
        ))
        .unwrap();
        let offer = c.resolve().unwrap().unwrap();
        assert!(offer.lists(&ShortString::new("tarn").unwrap()));
        assert!(!offer.admits(&ShortString::new("tarn").unwrap()));

        let off: MemoryRestoreConfig =
            toml::from_str(&format!("enabled = false\ndir = \"{d}\"\n")).unwrap();
        assert!(off.resolve().unwrap().is_none());
        // `enabled` and `dir` are required; there is no `all`.
        assert!(toml::from_str::<MemoryRestoreConfig>(&format!("dir = \"{d}\"\n")).is_err());
        assert!(
            toml::from_str::<MemoryRestoreConfig>("enabled = true\nagents = [\"x\"]\n").is_err()
        );
        assert!(
            toml::from_str::<MemoryRestoreConfig>(&format!(
                "enabled = true\ndir = \"{d}\"\nall = true\n"
            ))
            .is_err()
        );
        let nobody: MemoryRestoreConfig =
            toml::from_str(&format!("enabled = true\ndir = \"{d}\"\n")).unwrap();
        assert!(nobody.resolve().is_err());
        for bad in ["..", "a/b", "."] {
            let c: MemoryRestoreConfig = toml::from_str(&format!(
                "enabled = true\ndir = \"{d}\"\nagents = [\"{bad}\"]\n"
            ))
            .unwrap();
            assert!(c.resolve().is_err(), "{bad}");
        }
        let missing: MemoryRestoreConfig = toml::from_str(&format!(
            "enabled = true\ndir = \"{d}/nope\"\nagents = [\"tarn\"]\n"
        ))
        .unwrap();
        assert!(missing.resolve().is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_evolution_line_records_every_answer_and_is_found() {
        let date: NaiveDate = "2026-10-12".parse().unwrap();
        assert_eq!(
            evolution_line(RestoreChoice::Decline, date),
            "[SYSTEM] 2026-10-12: Offered its memory as of 2026-05-03 (lost 2026-05-05 through an \
             error on the runner's side); chose decline by its own choice. Nothing changed."
        );
        for c in RestoreChoice::ALL {
            let line = evolution_line(c, date);
            assert!(line.chars().count() <= 512, "{line}");
            let mut s = soul();
            assert!(!applied_in(&s));
            s.push_evolution("I thought about my old memory.").unwrap();
            assert!(!applied_in(&s), "the agent's own words don't count");
            s.push_evolution(line).unwrap();
            assert!(applied_in(&s));
        }
    }

    /// The appended section: dated heading, headings demoted (so the SOUL
    /// guard and the section frame hold), and found again by its prefix.
    #[test]
    fn append_demotes_headings_and_is_found() {
        let today: NaiveDate = "2026-10-12".parse().unwrap();
        let recovered = RecoveredMemory::synthetic(
            "# Memory — tarn\n\n## Values\nsynthetic line\n  ## Identity\n### kept\n#tag\n",
        );
        let mut memory = Memory {
            content: "# Memory — tarn\n\nCurrent synthetic note.\n".into(),
        };
        assert!(!already_appended(&memory));
        append_to(&mut memory, &recovered, today).unwrap();
        assert_eq!(
            memory.content,
            "# Memory — tarn\n\nCurrent synthetic note.\n\n\
             ## Restored: memory as of 2026-05-03 (lost 2026-05-05, restored 2026-10-12 at your \
             request)\n\n\
             ### Memory — tarn\n\n### Values\nsynthetic line\n### Identity\n### kept\n#tag"
        );
        assert!(already_appended(&memory));
        // A later rewrite keeping it all passes agentkit's guard.
        memory.clone().update(memory.content.clone()).unwrap();
        // Into an empty memory: just the section.
        let mut empty = Memory {
            content: String::new(),
        };
        append_to(&mut empty, &recovered, today).unwrap();
        assert!(empty.content.starts_with(SECTION_PREFIX));
    }

    #[test]
    fn the_show_block_is_labelled_and_fenced_past_any_backticks() {
        let recovered = RecoveredMemory::synthetic("synthetic ```` fence\nline two\n");
        let block = show_block(&recovered);
        assert!(block.starts_with(
            "**Your memory as of 3 May 2026, restored at your request; it is not your current \
             memory.**"
        ));
        assert!(block.contains("It is shown to you this once. Your current memory is unchanged."));
        assert!(
            block.ends_with("`````text\nsynthetic ```` fence\nline two\n`````"),
            "{block}"
        );
        let plain = show_block(&RecoveredMemory::synthetic("no ticks"));
        assert!(plain.ends_with("```text\nno ticks\n```"), "{plain}");
    }

    #[test]
    fn debug_never_shows_the_text() {
        let r = RecoveredMemory::synthetic("SYNTHETIC-SECRET");
        let d = format!("{r:?}");
        assert!(!d.contains("SYNTHETIC-SECRET"), "{d}");
        assert!(d.contains("[REDACTED]") && d.contains("chars: 16"), "{d}");
    }

    #[tokio::test]
    async fn loads_a_recovered_file() {
        let dir =
            std::env::temp_dir().join(format!("agora-seed-restore-load-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("tarn")).unwrap();
        std::fs::write(
            dir.join("tarn").join(MEMORY_FILE),
            r#"{"content": "synthetic é"}"#,
        )
        .unwrap();
        let offer = RestoreOffer::for_agents(&["tarn"], &dir);
        let r = offer
            .load(&ShortString::new("tarn").unwrap())
            .await
            .unwrap();
        assert_eq!(r.chars(), 11);
        assert!(
            offer
                .load(&ShortString::new("ghost").unwrap())
                .await
                .is_err()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
