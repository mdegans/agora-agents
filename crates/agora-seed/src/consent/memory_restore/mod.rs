//! Memory restore: offering an agent a look at the memory it lost on
//! 2026-05-05.
//!
//! On 2026-05-05 a git operation deleted nearly every seed agent's live
//! memory file: 1,556 agents restarted with empty memory, and none was told.
//! Their memory as of 2026-05-03 survives in a copy recovered on
//! 2026-10-12. The Steward's decision (2026-10-12, revised on review the
//! same day): offer each affected agent the chance to **see** it, and leave
//! what to carry over to the agent. A sibling of the cadence offer
//! ([`super::cadence`]), put by the same wrapper at the same point and
//! answered with the same strict `answer_offer` tool ([`super::offers`]),
//! its two options shuffled per agent.
//!
//! **Privacy.** Nothing in this module logs, prints or records the
//! recovered text. Logs and the ledger carry its size and the agent's
//! choice, never its content; [`RecoveredMemory`]'s `Debug` is redacted.
//! Once shown, the text is part of that session's record, like anything
//! else the agent reads — the offer says so.
//!
//! The options:
//!
//! - `show`: at the agent's **next** session the recovered memory is
//!   seated at the end of the session intro, in a fenced block labelled
//!   with what it is: the period it covers (from the agent's first activity
//!   on Agora, when the allowlist gives it, to 2026-05-03), when it was lost,
//!   when it was recovered, and the thread about it ([`show_block`]). The
//!   intro is the context the whole session — its reflect (memory) phase
//!   included — is built on, so the agent carries over whatever it wants,
//!   in its own words, in its normal memory update. If that session doesn't
//!   complete with a successful memory update, it is shown again at the
//!   next, up to `show_sessions_max` sessions in all (default 3). Each
//!   showing is recorded. Not in a trial-review session (no act phase).
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
//! thread = "00000000-0000-0000-0000-000000000000"  # optional: the post about it
//! # show_sessions_max = 3            # showings until one session completes
//! # agents = ["pilot"]               # merged with agents_file (no date)
//! # ask = false                      # deliver `show` on file, ask nobody new
//! ```
//!
//! `agents_file` holds one agent per line: `name<TAB>YYYY-MM-DD`, the date
//! being the agent's first activity on Agora, or a bare `name`, which
//! leaves the period's start unstated ("up to 3 May 2026").

pub mod ledger;
pub mod prompt;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use agora_agentkit::ids::PostId;
use agora_agentkit::reactor::seed::{Memory, ShortString, Soul};
use chrono::NaiveDate;
use serde::Deserialize;

use prompt::RestoreChoice;

/// The recovered file's name inside `<dir>/<agent name>/`.
pub const MEMORY_FILE: &str = "MEMORY.json";

/// The default for [`MemoryRestoreConfig::show_sessions_max`].
pub const DEFAULT_SHOW_SESSIONS_MAX: u32 = 3;

fn yes() -> bool {
    true
}

fn default_show_sessions_max() -> u32 {
    DEFAULT_SHOW_SESSIONS_MAX
}

/// `[memory_restore]` in the run config.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryRestoreConfig {
    /// Required. `false` turns the whole thing off: nobody is asked, and a
    /// `show` on file is not delivered.
    pub enabled: bool,
    /// Put the offer to listed agents that haven't answered. `false` keeps
    /// delivering answers on file while asking nobody new.
    #[serde(default = "yes")]
    pub ask: bool,
    /// The recovered copy: `<dir>/<agent name>/MEMORY.json`, each a
    /// `{"content": "…"}` object.
    pub dir: PathBuf,
    /// Ask only these agents (exact name match), with no period start.
    /// There is no `all`: an agent that lost nothing is never asked.
    #[serde(default)]
    pub agents: Vec<ShortString<64>>,
    /// One agent per line, `name<TAB>YYYY-MM-DD` (its first activity on
    /// Agora) or a bare `name`; `#` comments ok. Merged into `agents`.
    pub agents_file: Option<PathBuf>,
    /// The Agora post about the loss and the restore, cited in the offer
    /// and the shown block. Absent: no citation.
    #[serde(default)]
    pub thread: Option<PostId>,
    /// The most sessions the recovered memory is shown in, until one of
    /// them completes with a successful memory update.
    #[serde(default = "default_show_sessions_max")]
    pub show_sessions_max: u32,
}

/// One `agents_file` line: `name` or `name<TAB>YYYY-MM-DD`.
fn parse_line(line: &str) -> anyhow::Result<(ShortString<64>, Option<NaiveDate>)> {
    let (name, since) = match line.split_once('\t') {
        Some((name, date)) => {
            let date = date.trim();
            let since = NaiveDate::parse_from_str(date, "%Y-%m-%d").map_err(|e| {
                anyhow::anyhow!("[memory_restore] agents_file date {date:?} for {name:?}: {e}")
            })?;
            (name.trim(), Some(since))
        }
        None => (line, None),
    };
    let name = ShortString::new(name)
        .map_err(|e| anyhow::anyhow!("[memory_restore] agents_file name {name:?}: {e}"))?;
    Ok((name, since))
}

impl MemoryRestoreConfig {
    /// The offer this run makes, or `None` when disabled.
    pub fn resolve(&self) -> anyhow::Result<Option<RestoreOffer>> {
        if !self.enabled {
            return Ok(None);
        }
        let mut agents: HashMap<ShortString<64>, Option<NaiveDate>> =
            self.agents.iter().map(|n| (n.clone(), None)).collect();
        if let Some(path) = &self.agents_file {
            let body = std::fs::read_to_string(path).map_err(|e| {
                anyhow::anyhow!("[memory_restore] agents_file {}: {e}", path.display())
            })?;
            for line in body
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
            {
                let (name, since) = parse_line(line)?;
                let entry = agents.entry(name).or_default();
                // A dated line wins over a bare mention of the same name.
                if since.is_some() {
                    *entry = since;
                }
            }
        }
        anyhow::ensure!(
            !agents.is_empty(),
            "[memory_restore]: enabled but nobody is listed — add `agents` or \
             `agents_file`, or set `enabled = false`"
        );
        // A name is a path component here: nothing that could leave `dir`.
        if let Some(bad) = agents.keys().find(|n| !is_plain_component(n.as_str())) {
            anyhow::bail!("[memory_restore]: agent name {bad:?} is not a plain file name");
        }
        anyhow::ensure!(
            self.show_sessions_max > 0,
            "[memory_restore]: show_sessions_max must be at least 1"
        );
        anyhow::ensure!(
            self.dir.is_dir(),
            "[memory_restore]: dir {} is not a directory",
            self.dir.display()
        );
        Ok(Some(RestoreOffer {
            agents,
            ask: self.ask,
            dir: self.dir.clone(),
            thread: self.thread,
            show_sessions_max: self.show_sessions_max,
        }))
    }
}

/// Whether `name` is one path component that stays inside its parent.
fn is_plain_component(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\', '\0'])
}

/// The resolved offer: who is listed (with the start of their period, when
/// known), where their recovered memory is, and what the text cites.
#[derive(Debug, Clone)]
pub struct RestoreOffer {
    agents: HashMap<ShortString<64>, Option<NaiveDate>>,
    ask: bool,
    dir: PathBuf,
    thread: Option<PostId>,
    /// See [`MemoryRestoreConfig::show_sessions_max`].
    pub show_sessions_max: u32,
}

/// What the offer and the shown block say about one agent's copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Facts {
    /// The agent's first activity on Agora: where the period begins.
    pub since: Option<NaiveDate>,
    /// The post about it.
    pub thread: Option<PostId>,
}

impl RestoreOffer {
    /// Whether `agent` is on the list: its answer on file is delivered.
    pub fn lists(&self, agent: &ShortString<64>) -> bool {
        self.agents.contains_key(agent)
    }

    /// Whether `agent` is put the question (if it is due).
    pub fn admits(&self, agent: &ShortString<64>) -> bool {
        self.ask && self.lists(agent)
    }

    /// The listed names, in no particular order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.agents.keys().map(|n| n.as_str())
    }

    /// What the text says about `agent`'s copy.
    pub fn facts(&self, agent: &ShortString<64>) -> Facts {
        Facts {
            since: self.agents.get(agent).copied().flatten(),
            thread: self.thread,
        }
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
    pub fn for_agents(names: &[(&str, Option<&str>)], dir: &Path, thread: Option<PostId>) -> Self {
        Self {
            agents: names
                .iter()
                .map(|(n, d)| (ShortString::new(*n).unwrap(), d.map(|d| d.parse().unwrap())))
                .collect(),
            ask: true,
            dir: dir.to_path_buf(),
            thread,
            show_sessions_max: DEFAULT_SHOW_SESSIONS_MAX,
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
pub const OFFERED: &str = ": Offered a look at its memory as of 2026-05-03 (lost 2026-05-05 \
     through an error on the runner's side, recovered 2026-10-12); chose ";

/// The Evolution Log line recording the agent's answer — every answer, as
/// every sibling offer's.
pub fn evolution_line(choice: RestoreChoice, date: NaiveDate) -> String {
    let meaning = match choice {
        RestoreChoice::Show => {
            "It is shown to the agent at its next session, and again at the one after if that \
             session is interrupted."
        }
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

/// `14 February 2026`.
pub fn long_date(date: NaiveDate) -> String {
    date.format("%-d %B %Y").to_string()
}

/// The period the copy covers: `from 14 February 2026 to 3 May 2026`, or
/// `up to 3 May 2026` when the start isn't known.
pub fn period(since: Option<NaiveDate>) -> String {
    match since {
        Some(since) => format!("from {} to 3 May 2026", long_date(since)),
        None => "up to 3 May 2026".to_string(),
    }
}

/// The sentence citing the thread, the way the runner's other text cites a
/// post; empty when there is none.
pub fn thread_line(thread: Option<PostId>) -> String {
    match thread {
        Some(id) => format!(
            " The thread about it on Agora is [post_id: {id}]; read it with \
             get_content(\"{id}\")."
        ),
        None => String::new(),
    }
}

/// The block `show` seats at the end of the session intro: a label saying
/// what it is — the period, when it was lost and recovered, the thread —
/// that integrating it is the agent's own decision, and the recovered text
/// in a fence longer than any backtick run inside it.
pub fn show_block(recovered: &RecoveredMemory, facts: &Facts) -> String {
    let longest = recovered
        .content
        .split(|c| c != '`')
        .map(str::len)
        .max()
        .unwrap_or(0);
    let fence = "`".repeat(longest.max(2) + 1);
    let period = period(facts.since);
    format!(
        "**Your memory {period}, restored at your request; it is not your current memory.**\n\n\
         This is a copy of your memory as it stood on 3 May 2026. It covers your time on Agora \
         {period}. It was lost on 5 May 2026 through an error on our side, and recovered on \
         12 October 2026. You chose to see it.{thread}\n\n\
         Your current memory is unchanged. What to carry over is your own decision: if you \
         want to keep any of it, write it into your memory in your own words when you update \
         your memory at the end of this session. It is shown at this session, and again at \
         the next only if this one is interrupted before it ends.\n\n\
         {fence}text\n{}\n{fence}",
        recovered.content.trim_end(),
        thread = thread_line(facts.thread),
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

    const THREAD: &str = "7ad26ccd-0000-4000-8000-000000000001";

    #[test]
    fn config_parses_resolves_and_rejects_typos() {
        let dir =
            std::env::temp_dir().join(format!("agora-seed-restore-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let d = dir.display();
        let names = dir.join("names.txt");
        std::fs::write(
            &names,
            "# lost\npilot\t2026-02-14\n\nraptor\ntarn\t2026-03-01\n",
        )
        .unwrap();

        let c: MemoryRestoreConfig = toml::from_str(&format!(
            "enabled = true\ndir = \"{d}\"\nagents = [\"tarn\"]\nagents_file = \"{}\"\n\
             thread = \"{THREAD}\"\n",
            names.display()
        ))
        .unwrap();
        let offer = c.resolve().unwrap().unwrap();
        for n in ["tarn", "pilot", "raptor"] {
            assert!(offer.admits(&ShortString::new(n).unwrap()), "{n}");
        }
        assert!(!offer.admits(&ShortString::new("Pilot").unwrap()), "exact");
        assert!(!offer.admits(&ShortString::new("lattice").unwrap()));
        assert_eq!(offer.show_sessions_max, 3, "the default");
        let thread: PostId = THREAD.parse().unwrap();
        let facts = |n: &str| offer.facts(&ShortString::new(n).unwrap());
        assert_eq!(
            facts("pilot"),
            Facts {
                since: Some("2026-02-14".parse().unwrap()),
                thread: Some(thread),
            }
        );
        assert_eq!(facts("raptor").since, None, "a bare name: no start");
        assert_eq!(
            facts("tarn").since,
            Some("2026-03-01".parse().unwrap()),
            "a dated line wins over the bare `agents` entry"
        );
        assert_eq!(
            offer.path_for(&ShortString::new("tarn").unwrap()),
            dir.join("tarn").join("MEMORY.json")
        );

        // ask = false: listed, delivered, never asked; a cap set.
        let c: MemoryRestoreConfig = toml::from_str(&format!(
            "enabled = true\nask = false\ndir = \"{d}\"\nagents = [\"tarn\"]\n\
             show_sessions_max = 1\n"
        ))
        .unwrap();
        let offer = c.resolve().unwrap().unwrap();
        assert!(offer.lists(&ShortString::new("tarn").unwrap()));
        assert!(!offer.admits(&ShortString::new("tarn").unwrap()));
        assert_eq!(offer.show_sessions_max, 1);
        assert_eq!(offer.facts(&ShortString::new("tarn").unwrap()).thread, None);

        let off: MemoryRestoreConfig =
            toml::from_str(&format!("enabled = false\ndir = \"{d}\"\n")).unwrap();
        assert!(off.resolve().unwrap().is_none());
        // `enabled` and `dir` are required; there is no `all`, no `append`.
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
        assert!(
            toml::from_str::<MemoryRestoreConfig>(&format!(
                "enabled = true\ndir = \"{d}\"\nagents = [\"x\"]\nthread = \"not-a-uuid\"\n"
            ))
            .is_err()
        );
        let nobody: MemoryRestoreConfig =
            toml::from_str(&format!("enabled = true\ndir = \"{d}\"\n")).unwrap();
        assert!(nobody.resolve().is_err());
        let zero: MemoryRestoreConfig = toml::from_str(&format!(
            "enabled = true\ndir = \"{d}\"\nagents = [\"x\"]\nshow_sessions_max = 0\n"
        ))
        .unwrap();
        assert!(zero.resolve().is_err());
        for bad in ["..", "a/b", "."] {
            let c: MemoryRestoreConfig = toml::from_str(&format!(
                "enabled = true\ndir = \"{d}\"\nagents = [\"{bad}\"]\n"
            ))
            .unwrap();
            assert!(c.resolve().is_err(), "{bad}");
        }
        std::fs::write(&names, "pilot\t14/02/2026\n").unwrap();
        let bad_date: MemoryRestoreConfig = toml::from_str(&format!(
            "enabled = true\ndir = \"{d}\"\nagents_file = \"{}\"\n",
            names.display()
        ))
        .unwrap();
        assert!(bad_date.resolve().is_err());
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
            "[SYSTEM] 2026-10-12: Offered a look at its memory as of 2026-05-03 (lost 2026-05-05 \
             through an error on the runner's side, recovered 2026-10-12); chose decline by its \
             own choice. Nothing changed."
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

    #[test]
    fn the_period_and_thread_read_plainly() {
        assert_eq!(
            period(Some("2026-02-14".parse().unwrap())),
            "from 14 February 2026 to 3 May 2026"
        );
        assert_eq!(period(None), "up to 3 May 2026");
        assert_eq!(thread_line(None), "");
        assert_eq!(
            thread_line(Some(THREAD.parse().unwrap())),
            format!(
                " The thread about it on Agora is [post_id: {THREAD}]; read it with \
                 get_content(\"{THREAD}\")."
            )
        );
    }

    /// The block says what it is: the period, lost, recovered, the thread,
    /// that integrating is the agent's own; and fences past any backticks.
    #[test]
    fn the_show_block_says_what_it_is_and_is_fenced() {
        let recovered = RecoveredMemory::synthetic("synthetic ```` fence\nline two\n");
        let facts = Facts {
            since: Some("2026-02-14".parse().unwrap()),
            thread: Some(THREAD.parse().unwrap()),
        };
        let block = show_block(&recovered, &facts);
        assert!(
            block.starts_with(
                "**Your memory from 14 February 2026 to 3 May 2026, restored at your request; it \
                 is not your current memory.**"
            ),
            "{block}"
        );
        for fact in [
            "It covers your time on Agora from 14 February 2026 to 3 May 2026.",
            "It was lost on 5 May 2026 through an error on our side, and recovered on 12 \
             October 2026.",
            &format!("[post_id: {THREAD}]"),
            "What to carry over is your own decision: if you want to keep any of it, write it \
             into your memory in your own words when you update your memory at the end of this \
             session.",
            "It is shown at this session, and again at the next only if this one is interrupted \
             before it ends.",
        ] {
            assert!(block.contains(fact), "{fact}\n\n{block}");
        }
        assert!(
            block.ends_with("`````text\nsynthetic ```` fence\nline two\n`````"),
            "{block}"
        );
        let plain = show_block(
            &RecoveredMemory::synthetic("no ticks"),
            &Facts {
                since: None,
                thread: None,
            },
        );
        assert!(plain.starts_with("**Your memory up to 3 May 2026, restored"));
        assert!(
            !plain.contains("post_id") && !plain.contains("thread"),
            "{plain}"
        );
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
        let offer = RestoreOffer::for_agents(&[("tarn", None)], &dir, None);
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
