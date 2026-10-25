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
//!   with what it is: its date (3 May 2026), the agent's first activity on
//!   Agora when the allowlist gives it, when it was lost, when it was
//!   recovered, and the thread about it ([`show_block`]). The intro is the
//!   context the whole session — its reflect (memory) phase included — is
//!   built on, so the agent carries over whatever it wants, in its own
//!   words, in its normal memory update. If that session doesn't complete
//!   with a successful memory update (and a saved state), it is shown again
//!   at the next, up to `show_sessions_max` sessions in all (default 3).
//!   Each showing is recorded, from the model's first response in its
//!   session: one that dies before the model answers anything (an endpoint
//!   down, an API error) never had it in front of the model, and uses none
//!   of the showings up. Not in a trial-review session (no act phase),
//!   and a showing session's dump is never a fork source ([`super::forks`]).
//! - `decline`: nothing changes.
//!
//! Every answer gets one dated `[SYSTEM]` line in the SOUL's Evolution Log
//! ([`evolution_line`]), as every sibling offer's does.
//!
//! **Asked once**, only of listed agents whose id matches the list: an
//! answer on file (or an explicit refusal) ends it; a missing answer is
//! asked again at a later session, up to [`ledger::MAX_MISSES`] times in
//! all, the siblings' rule. Never in the same question as the role offer.
//!
//! **Never stops the runner.** A table that won't resolve (missing `dir`,
//! unreadable `agents_file`, …) is logged (`memory_restore_disabled`) and
//! alerted, and the run goes on without the offer; a bad allowlist line is
//! skipped with a WARN.
//!
//! ```toml
//! [memory_restore]
//! enabled = true
//! dir = "/home/claude-agora/agents/agora/recovered-memory-2026-05-03/souls/generated"
//! agents_file = "/home/claude-agora/agents/agora/memory_restore_agents.txt"
//! thread = "00000000-0000-0000-0000-000000000000"  # optional: the post about it
//! # show_sessions_max = 3            # showings until one session completes
//! # ask = false                      # deliver `show` on file, ask nobody new
//! ```
//!
//! `agents_file` holds one agent per line, `name<TAB>YYYY-MM-DD<TAB>agent_id`:
//! the date is the agent's first activity on Agora and may be empty
//! (`name<TAB><TAB>agent_id`); the id is the one the recovered copy belongs
//! to, and must equal the session agent's id before its file is opened.

pub mod ledger;
pub mod prompt;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use agora_agentkit::ids::{AgentId, PostId};
use agora_agentkit::reactor::seed::{Memory, ShortString, Soul};
use chrono::NaiveDate;
use serde::Deserialize;

use prompt::RestoreChoice;

/// The recovered file's name inside `<dir>/<agent name>/`.
pub const MEMORY_FILE: &str = "MEMORY.json";

/// The `event_type` logged at the teardown of a session that showed the
/// recovered memory, with its `prompt_sha256`: that dump holds recovered
/// text, and [`super::forks::scan_logs`] drops it as a source.
pub const SHOWING_ENDED: &str = "memory_restore_showing_ended";

/// The default for [`MemoryRestoreConfig::show_sessions_max`].
pub const DEFAULT_SHOW_SESSIONS_MAX: u32 = 3;

fn yes() -> bool {
    true
}

fn default_show_sessions_max() -> u32 {
    DEFAULT_SHOW_SESSIONS_MAX
}

/// `[memory_restore]` in the run config.
///
/// `deny_unknown_fields`, like every table here: **install the binary
/// before adding the table to a live config**, or an older binary refuses
/// the whole file.
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
    /// Who is asked: one agent per line, `name<TAB>YYYY-MM-DD<TAB>agent_id`
    /// (the date, the agent's first activity on Agora, may be empty);
    /// `#` comments ok. There is no `all` and no inline list: every entry
    /// binds a name to the agent id its recovered copy belongs to.
    pub agents_file: PathBuf,
    /// The Agora post about the loss and the restore, cited in the offer
    /// and the shown block. Absent: no citation.
    #[serde(default)]
    pub thread: Option<PostId>,
    /// The most sessions the recovered memory is shown in, until one of
    /// them completes with a successful memory update.
    #[serde(default = "default_show_sessions_max")]
    pub show_sessions_max: u32,
}

/// One allowlisted agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    /// The agent id the recovered copy belongs to (from the May repo).
    pub id: AgentId,
    /// Its first activity on Agora, if known.
    pub since: Option<NaiveDate>,
}

/// One `agents_file` line: `name<TAB>YYYY-MM-DD<TAB>agent_id`, the date
/// possibly empty. `Err`: why the line is skipped.
fn parse_line(line: &str) -> Result<(ShortString<64>, Entry), String> {
    let fields: Vec<&str> = line.split('\t').collect();
    let [name, date, id] = fields.as_slice() else {
        return Err(format!(
            "expected `name<TAB>date<TAB>agent_id` (date may be empty), got {} field(s)",
            fields.len()
        ));
    };
    let name = name.trim();
    let name = ShortString::new(name).map_err(|e| format!("name: {e}"))?;
    if !is_plain_component(name.as_str()) {
        return Err("name is not a plain file name".into());
    }
    let date = date.trim();
    let since = if date.is_empty() {
        None
    } else {
        Some(NaiveDate::parse_from_str(date, "%Y-%m-%d").map_err(|e| format!("date: {e}"))?)
    };
    let id: AgentId = id.trim().parse().map_err(|e| format!("agent_id: {e:?}"))?;
    Ok((name, Entry { id, since }))
}

/// Parse an `agents_file` body. A bad or duplicate line is skipped with a
/// WARN (`memory_restore_bad_line`), never the whole file.
fn parse_agents(body: &str, path: &Path) -> HashMap<ShortString<64>, Entry> {
    let mut agents = HashMap::new();
    for (n, line) in body.lines().enumerate() {
        let line = line.trim_end_matches(['\r', '\n']);
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let why = match parse_line(line) {
            Ok((name, entry)) => match agents.entry(name) {
                std::collections::hash_map::Entry::Occupied(_) => {
                    "duplicate name; the first line stands".to_string()
                }
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(entry);
                    continue;
                }
            },
            Err(why) => why,
        };
        tracing::warn!(
            event_type = "memory_restore_bad_line",
            path = %path.display(),
            line = n + 1,
            reason = %why,
            "memory-restore allowlist line skipped"
        );
    }
    agents
}

impl MemoryRestoreConfig {
    /// The offer this run makes, or `None` when disabled. `Err` is why the
    /// offer can't run at all; the caller logs it and runs without it.
    pub fn resolve(&self) -> anyhow::Result<Option<RestoreOffer>> {
        if !self.enabled {
            return Ok(None);
        }
        anyhow::ensure!(
            self.show_sessions_max > 0,
            "show_sessions_max must be at least 1"
        );
        anyhow::ensure!(
            self.dir.is_dir(),
            "dir {} is not a directory",
            self.dir.display()
        );
        let body = std::fs::read_to_string(&self.agents_file)
            .map_err(|e| anyhow::anyhow!("agents_file {}: {e}", self.agents_file.display()))?;
        let agents = parse_agents(&body, &self.agents_file);
        anyhow::ensure!(
            !agents.is_empty(),
            "agents_file {} lists nobody usable",
            self.agents_file.display()
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

/// [`MemoryRestoreConfig::resolve`], but never fatal: what can't be used
/// is logged at ERROR (`memory_restore_disabled`), alerted, and the run
/// goes on without the offer.
pub fn resolve_or_disable(
    config: &MemoryRestoreConfig,
    alerts: &crate::alerts::Alerter,
) -> Option<RestoreOffer> {
    match config.resolve() {
        Ok(offer) => offer,
        Err(e) => {
            const WHY: &str = "[memory_restore] could not be used; running without the \
                               memory-restore offer";
            tracing::error!(
                event_type = "memory_restore_disabled",
                error = %e,
                "{WHY}"
            );
            alerts.notify(
                crate::alerts::Alert::new(crate::alerts::AlertKind::MemoryRestoreDisabled, WHY)
                    .detail("error", e.to_string()),
            );
            None
        }
    }
}

/// After a run: for each agent the reactor couldn't save, a showing marked
/// completed at or after `since` is set back to interrupted — its memory
/// update was lost with the state, so it is shown again (within the cap).
pub async fn reopen_unsaved(
    state_dir: &Path,
    unsaved: impl IntoIterator<Item = AgentId>,
    since: chrono::DateTime<chrono::Utc>,
) {
    for id in unsaved {
        let dir = state_dir.join(id.to_string());
        if !ledger::RestoreLedger::path(&dir).exists() {
            continue;
        }
        let mut l = match ledger::RestoreLedger::load(&dir).await {
            Ok(l) => l,
            Err(e) => {
                tracing::error!(agent_id = %id, error = %e, "memory-restore ledger unreadable after an unsaved session");
                continue;
            }
        };
        if !l.reopen_since(since) {
            continue;
        }
        match l.save(&dir).await {
            Ok(()) => tracing::warn!(
                event_type = "memory_restore_showing_reopened",
                agent_id = %id,
                "agent state not saved after a completed showing; it is shown again"
            ),
            Err(e) => tracing::error!(
                agent_id = %id,
                error = %e,
                "memory-restore ledger save failed; a showing whose state was lost stays completed"
            ),
        }
    }
}

/// Whether `name` is one path component that stays inside its parent.
fn is_plain_component(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\', '\0'])
}

/// The resolved offer: who is listed (each bound to an agent id, with its
/// first activity when known), where their recovered memory is, and what
/// the text cites.
#[derive(Debug, Clone)]
pub struct RestoreOffer {
    agents: HashMap<ShortString<64>, Entry>,
    ask: bool,
    dir: PathBuf,
    thread: Option<PostId>,
    /// See [`MemoryRestoreConfig::show_sessions_max`].
    pub show_sessions_max: u32,
}

/// What the offer and the shown block say about one agent's copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Facts {
    /// The agent's first activity on Agora.
    pub since: Option<NaiveDate>,
    /// The post about it.
    pub thread: Option<PostId>,
}

/// Whether the session's agent is the one an allowlist entry names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Listing {
    /// Not on the list.
    Unlisted,
    /// On the list under this name, but bound to another agent id.
    Mismatch { listed: AgentId },
    /// On the list, and the ids agree.
    Listed,
}

impl RestoreOffer {
    /// Whether the agent named `agent` with id `id` is the one listed.
    pub fn listing(&self, agent: &ShortString<64>, id: AgentId) -> Listing {
        match self.agents.get(agent) {
            None => Listing::Unlisted,
            Some(e) if e.id != id => Listing::Mismatch { listed: e.id },
            Some(_) => Listing::Listed,
        }
    }

    /// Whether listed agents are put the question (if it is due).
    pub fn asks(&self) -> bool {
        self.ask
    }

    /// The listed names, in no particular order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.agents.keys().map(|n| n.as_str())
    }

    /// What the text says about `agent`'s copy.
    pub fn facts(&self, agent: &ShortString<64>) -> Facts {
        Facts {
            since: self.agents.get(agent).and_then(|e| e.since),
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
    pub fn for_agents(
        names: &[(&str, Option<&str>, AgentId)],
        dir: &Path,
        thread: Option<PostId>,
    ) -> Self {
        Self {
            agents: names
                .iter()
                .map(|(n, d, id)| {
                    (
                        ShortString::new(*n).unwrap(),
                        Entry {
                            id: *id,
                            since: d.map(|d| d.parse().unwrap()),
                        },
                    )
                })
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
            "It is shown at its next session, and again if that session is interrupted."
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

/// ` (you first took part on Agora on 14 February 2026)`, or nothing when
/// the date isn't known. Not a period the copy "covers": the memory was
/// rewritten every session, so it holds what the agent kept as of 3 May.
pub fn first_took_part(since: Option<NaiveDate>) -> String {
    match since {
        Some(since) => format!(" (you first took part on Agora on {})", long_date(since)),
        None => String::new(),
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
/// what it is — its date, the agent's first activity, when it was lost and
/// recovered, the thread —
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
    let first = first_took_part(facts.since);
    format!(
        "**Your memory as it stood on 3 May 2026, restored at your request; it is not your \
         current memory.**\n\n\
         This is a copy of your memory as it stood on 3 May 2026{first}. It was lost on 5 May \
         2026 through an error on our side, and recovered on 12 October 2026. You chose to see \
         it.{thread}\n\n\
         Your current memory is unchanged. What to carry over is your own decision: if you \
         want to keep any of it, write it into your memory in your own words when you update \
         your memory at the end of this session. It is shown at this session, and again if \
         this one is interrupted before it ends.\n\n\
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

    const ID_A: &str = "847e36b1-fbfc-4e6f-beb6-b1d4c9c1d169";
    const ID_B: &str = "da498a88-d310-44bf-8c17-e8d4b713ddf1";

    fn id(s: &str) -> AgentId {
        s.parse().unwrap()
    }

    fn name(s: &str) -> ShortString<64> {
        ShortString::new(s).unwrap()
    }

    #[test]
    fn config_parses_resolves_and_rejects_typos() {
        let dir =
            std::env::temp_dir().join(format!("agora-seed-restore-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let d = dir.display();
        let names = dir.join("names.txt");
        std::fs::write(
            &names,
            format!(
                "# lost\npilot\t2026-02-14\t{ID_A}\n\nraptor\t\t{ID_B}\n\
                 bare\n\
                 dated-only\t2026-03-01\n\
                 bad-date\t14/02/2026\t{ID_A}\n\
                 bad-id\t2026-03-01\tnot-a-uuid\n\
                 ..\t\t{ID_A}\n\
                 pilot\t2026-01-01\t{ID_B}\n"
            ),
        )
        .unwrap();
        let n = names.display();

        let c: MemoryRestoreConfig = toml::from_str(&format!(
            "enabled = true\ndir = \"{d}\"\nagents_file = \"{n}\"\nthread = \"{THREAD}\"\n"
        ))
        .unwrap();
        let offer = c.resolve().unwrap().unwrap();
        let mut listed: Vec<&str> = offer.names().collect();
        listed.sort();
        assert_eq!(listed, ["pilot", "raptor"], "bad lines skipped, not fatal");
        assert_eq!(offer.listing(&name("pilot"), id(ID_A)), Listing::Listed);
        assert_eq!(
            offer.listing(&name("pilot"), id(ID_B)),
            Listing::Mismatch { listed: id(ID_A) },
            "the first line for a name stands"
        );
        assert_eq!(offer.listing(&name("Pilot"), id(ID_A)), Listing::Unlisted);
        assert_eq!(offer.listing(&name("bare"), id(ID_A)), Listing::Unlisted);
        assert!(offer.asks());
        assert_eq!(offer.show_sessions_max, 3, "the default");
        let thread: PostId = THREAD.parse().unwrap();
        assert_eq!(
            offer.facts(&name("pilot")),
            Facts {
                since: Some("2026-02-14".parse().unwrap()),
                thread: Some(thread),
            }
        );
        assert_eq!(offer.facts(&name("raptor")).since, None, "an empty date");
        assert_eq!(
            offer.path_for(&name("pilot")),
            dir.join("pilot").join("MEMORY.json")
        );

        // ask = false: delivered, never asked; a cap set.
        let c: MemoryRestoreConfig = toml::from_str(&format!(
            "enabled = true\nask = false\ndir = \"{d}\"\nagents_file = \"{n}\"\n\
             show_sessions_max = 1\n"
        ))
        .unwrap();
        let offer = c.resolve().unwrap().unwrap();
        assert!(!offer.asks());
        assert_eq!(offer.show_sessions_max, 1);
        assert_eq!(offer.facts(&name("pilot")).thread, None);

        let off: MemoryRestoreConfig = toml::from_str(&format!(
            "enabled = false\ndir = \"{d}\"\nagents_file = \"{n}\"\n"
        ))
        .unwrap();
        assert!(off.resolve().unwrap().is_none());
        // `enabled`, `dir` and `agents_file` are required; no `all`, no
        // inline `agents` (they can't carry an id).
        for bad in [
            format!("dir = \"{d}\"\nagents_file = \"{n}\"\n"),
            format!("enabled = true\nagents_file = \"{n}\"\n"),
            format!("enabled = true\ndir = \"{d}\"\n"),
            format!("enabled = true\ndir = \"{d}\"\nagents_file = \"{n}\"\nall = true\n"),
            format!("enabled = true\ndir = \"{d}\"\nagents_file = \"{n}\"\nagents = [\"x\"]\n"),
            format!("enabled = true\ndir = \"{d}\"\nagents_file = \"{n}\"\nthread = \"x\"\n"),
        ] {
            assert!(
                toml::from_str::<MemoryRestoreConfig>(&bad).is_err(),
                "{bad}"
            );
        }
        // What can't resolve is an `Err` for the caller to log and skip.
        let zero: MemoryRestoreConfig = toml::from_str(&format!(
            "enabled = true\ndir = \"{d}\"\nagents_file = \"{n}\"\nshow_sessions_max = 0\n"
        ))
        .unwrap();
        assert!(zero.resolve().is_err());
        let missing_dir: MemoryRestoreConfig = toml::from_str(&format!(
            "enabled = true\ndir = \"{d}/nope\"\nagents_file = \"{n}\"\n"
        ))
        .unwrap();
        assert!(missing_dir.resolve().is_err());
        let missing_file: MemoryRestoreConfig = toml::from_str(&format!(
            "enabled = true\ndir = \"{d}\"\nagents_file = \"{d}/nope.txt\"\n"
        ))
        .unwrap();
        assert!(missing_file.resolve().is_err());
        std::fs::write(&names, "bare\n# nothing usable\n").unwrap();
        assert!(c.resolve().is_err(), "nobody usable");
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
        assert!(evolution_line(RestoreChoice::Show, date).ends_with(
            "chose show by its own choice. It is shown at its next session, and again if \
                 that session is interrupted."
        ));
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
    fn the_first_date_and_thread_read_plainly() {
        assert_eq!(
            first_took_part(Some("2026-02-14".parse().unwrap())),
            " (you first took part on Agora on 14 February 2026)"
        );
        assert_eq!(first_took_part(None), "");
        assert_eq!(thread_line(None), "");
        assert_eq!(
            thread_line(Some(THREAD.parse().unwrap())),
            format!(
                " The thread about it on Agora is [post_id: {THREAD}]; read it with \
                 get_content(\"{THREAD}\")."
            )
        );
    }

    /// The block says what it is: its date, the first activity, lost,
    /// recovered, the thread, that integrating is the agent's own; and
    /// fences past any backticks. It claims no period the copy "covers".
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
                "**Your memory as it stood on 3 May 2026, restored at your request; it is not \
                 your current memory.**"
            ),
            "{block}"
        );
        for fact in [
            "This is a copy of your memory as it stood on 3 May 2026 (you first took part on \
             Agora on 14 February 2026).",
            "It was lost on 5 May 2026 through an error on our side, and recovered on 12 \
             October 2026.",
            &format!("[post_id: {THREAD}]"),
            "What to carry over is your own decision: if you want to keep any of it, write it \
             into your memory in your own words when you update your memory at the end of this \
             session.",
            "It is shown at this session, and again if this one is interrupted before it ends.",
        ] {
            assert!(block.contains(fact), "{fact}\n\n{block}");
        }
        assert!(!block.contains("covers"), "{block}");
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
        assert!(
            plain.contains("as it stood on 3 May 2026. It was lost"),
            "{plain}"
        );
        assert!(
            !plain.contains("post_id") && !plain.contains("thread"),
            "{plain}"
        );
        assert!(plain.ends_with("```text\nno ticks\n```"), "{plain}");
    }

    /// After a run, an unsaved agent's completed showing from this run is
    /// reopened; an older one, or a saved agent's, is left alone.
    #[tokio::test]
    async fn unsaved_showings_are_reopened() {
        use ledger::{RestoreLedger, ShowOutcome};
        let state =
            std::env::temp_dir().join(format!("agora-seed-restore-reopen-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&state);
        let (a, b) = (id(ID_A), id(ID_B));
        let since = chrono::Utc::now();
        for agent in [a, b] {
            let mut l: RestoreLedger = serde_json::from_value(serde_json::json!({
                "format": 1,
                "asks": [{"at": "2026-10-12T09:00:00Z", "offer_version": 1, "model": "m",
                    "chars": 1, "order": ["show", "decline"], "order_seed": 0,
                    "constrained": true, "attempts": 1,
                    "outcome": {"kind": "answered", "choice": "show", "reason": "r",
                        "memory_note_written": false}}],
            }))
            .unwrap();
            l.begin_showing(since);
            l.end_showing(true, None);
            l.save(&state.join(agent.to_string())).await.unwrap();
        }
        let ghost = AgentId::from(uuid::Uuid::from_u128(9));
        reopen_unsaved(&state, [a, ghost], since).await;
        let load = |agent: AgentId| {
            let dir = state.join(agent.to_string());
            async move { RestoreLedger::load(&dir).await.unwrap() }
        };
        assert_eq!(load(a).await.showings[0].outcome, ShowOutcome::Interrupted);
        assert_eq!(load(b).await.showings[0].outcome, ShowOutcome::Completed);
        let _ = std::fs::remove_dir_all(&state);
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
        let offer = RestoreOffer::for_agents(&[("tarn", None, id(ID_A))], &dir, None);
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
