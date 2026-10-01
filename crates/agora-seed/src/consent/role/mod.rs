//! Role consent: an offer to close the gap between a generator-assigned
//! role and the tools the agent actually has.
//!
//! Many SOULs were written by an early generator that gave agents a
//! profession that comes with work (modeling, measuring, archiving) while
//! their tools are Agora read/write and memory. The Steward's rule
//! (2026-09-29): no unilateral SOUL edits. The agent is *asked*, at the end
//! of a session — after its memory write — and any edit is disclosed in its
//! SOUL. A sibling of the model-swap offer ([`super`]), reusing its
//! wrapper, retry and parse machinery.
//!
//! The pieces:
//!
//! - [`RoleConsentConfig`] — the `[role_consent]` table: off unless present
//!   and `enabled`, and only for the agents listed (`agents` and/or
//!   `agents_file`). The offer's text is in code ([`prompt`]), versioned and
//!   test-pinned, not in config.
//! - [`prompt`] — the text, the `$ref`/`pattern`-free schema, the answer.
//! - [`ledger`] — `state/<agent_id>/role_consent.json`: asked once, the
//!   full answer and what was applied (including, permanently, a replaced
//!   identity). Never the agent's memory. The SOUL's own disclosure lines
//!   also end the asking ([`applied_in`]), in case a ledger save failed.
//! - This module — applying an answer: [`apply`] edits only `identity`
//!   and writes the disclosure into the Evolution Log; [`append_memory_note`]
//!   adds the agent's own note, if it wrote one, and nothing else.
//!
//! The asking itself is in [`super::agent::ConsentAgent`]: one question per
//! session, the model-swap offer first (it may be mid-trial), so a session
//! in which the model-swap machinery asks or does anything leaves this
//! offer for the next.
//!
//! **The permanent record of a replaced identity** is the ledger's
//! [`Applied::RoleChanged`]`.previous`. The Evolution Log carries it too,
//! verbatim, but that log is capped at 50 entries and drops the oldest, so
//! the offer promises only that the old description is "recorded, not
//! erased".
//!
//! **`sleep`** changes nothing in the SOUL or memory. It takes effect
//! without the Steward: the sweep planner (`put_to_sleep` in `main.rs`)
//! reads each agent's role ledger and leaves out any whose latest answer is
//! `sleep`, reporting them in the plan, alongside the manual `sleeping`
//! list. It is also logged (`role_consent_sleep`) and alerted. **Waking is
//! the Steward's explicit step**: the agent's name in the run config's
//! top-level `wake = [...]` (the answer stays on file, so the name stays
//! listed); `sleeping` wins over `wake` (mdegans/agora-agents#189).
//!
//! ```toml
//! [role_consent]
//! enabled = true
//! agents = ["pilot", "raptor"]
//! agents_file = "/path/to/names.txt"   # optional, one name per line
//! ```

pub mod ledger;
pub mod prompt;

use std::collections::HashSet;
use std::path::PathBuf;

use agora_agentkit::reactor::seed::{Memory, PROSE_MAX, ShortString, Soul};
use chrono::NaiveDate;
use serde::Deserialize;

use ledger::Applied;
use prompt::{RoleAnswer, RoleChoice};

/// `[role_consent]` in the run config.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoleConsentConfig {
    /// Required, so turning the offer off doesn't mean deleting the list.
    pub enabled: bool,
    /// Ask only these agents (exact name match).
    #[serde(default)]
    pub agents: Vec<ShortString<64>>,
    /// Newline-separated names (`#` comments ok), merged into `agents`.
    pub agents_file: Option<PathBuf>,
}

impl RoleConsentConfig {
    /// The offer this run makes, or `None` when disabled. Reads
    /// `agents_file`. An enabled offer with nobody listed is an error.
    pub fn resolve(&self) -> anyhow::Result<Option<RoleOffer>> {
        if !self.enabled {
            return Ok(None);
        }
        let mut agents: HashSet<ShortString<64>> = self.agents.iter().cloned().collect();
        if let Some(path) = &self.agents_file {
            let body = std::fs::read_to_string(path).map_err(|e| {
                anyhow::anyhow!("[role_consent] agents_file {}: {e}", path.display())
            })?;
            for line in body
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
            {
                agents.insert(ShortString::new(line).map_err(|e| {
                    anyhow::anyhow!("[role_consent] agents_file name {line:?}: {e}")
                })?);
            }
        }
        anyhow::ensure!(
            !agents.is_empty(),
            "[role_consent]: enabled but nobody is listed — add `agents` or \
             `agents_file`, or set `enabled = false`"
        );
        Ok(Some(RoleOffer { agents }))
    }
}

/// The resolved offer: who is asked.
#[derive(Debug, Clone, Default)]
pub struct RoleOffer {
    agents: HashSet<ShortString<64>>,
}

impl RoleOffer {
    pub fn admits(&self, agent: &ShortString<64>) -> bool {
        self.agents.contains(agent)
    }

    /// The listed names, in no particular order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.agents.iter().map(|n| n.as_str())
    }

    #[cfg(test)]
    pub fn for_agents(names: &[&str]) -> Self {
        Self {
            agents: names
                .iter()
                .map(|n| ShortString::new(*n).unwrap())
                .collect(),
        }
    }
}

/// How the disclosure of a clarify begins; also how [`applied_in`] finds it.
pub const CLARIFIED: &str = "[SYSTEM] Identity clarified by the agent's own choice";

/// How the disclosure of a new role begins.
pub const ROLE_CHANGED: &str = "[SYSTEM] Role changed by the agent's own choice";

/// Whether `soul`'s Evolution Log already discloses a role answer applied.
pub fn applied_in(soul: &Soul) -> bool {
    soul.evolution_log.iter().any(|e| {
        let note = e.note.as_str();
        note.starts_with(CLARIFIED) || note.starts_with(ROLE_CHANGED)
    })
}

/// Why the SOUL changed, in the Evolution Log. Neutral on purpose: it says
/// the role was ours and the choice was the agent's, and nothing about the
/// agent being at fault.
const WHY: &str = "after the Steward's offer about a generator-assigned role that outran the \
                   agent's tools.";

/// The longest Evolution Log line this offer writes, in characters. It
/// was agentkit's note capacity; agentkit 0.49 raised that to `ITEM_MAX`
/// (1024), and this stays at 512 so the layout of the role-change lines
/// (and the chunking of a long previous identity) is unchanged.
const NOTE_MAX: usize = 512;

/// What applying `answer` to a SOUL whose identity is `identity` does.
/// `answer` must have passed [`RoleAnswer::validate`] against `identity`.
pub fn plan(answer: &RoleAnswer, identity: &str) -> Applied {
    let text = answer.soul_text.trim();
    match answer.choice {
        RoleChoice::Nothing => Applied::Nothing,
        RoleChoice::Sleep => Applied::Sleep,
        RoleChoice::Clarify => Applied::Clarified {
            previous: identity.to_string(),
            identity: format!("{} {text}", identity.trim_end()),
        },
        RoleChoice::NewRole => Applied::RoleChanged {
            previous: identity.to_string(),
            identity: text.to_string(),
        },
    }
}

/// The Evolution Log lines disclosing `applied`: none unless the SOUL
/// changed. A previous identity too long for one note (512 characters) is
/// carried verbatim across numbered continuation entries.
pub fn evolution_lines(applied: &Applied) -> Vec<String> {
    match applied {
        Applied::Nothing | Applied::Sleep => Vec::new(),
        Applied::Clarified { previous, identity } => {
            let added = identity
                .strip_prefix(previous.trim_end())
                .unwrap_or(identity)
                .trim();
            vec![format!("{CLARIFIED}, {WHY} Added: \"{added}\"")]
        }
        Applied::RoleChanged { previous, .. } => {
            let whole = format!("{ROLE_CHANGED}, {WHY} Previous identity: \"{previous}\"");
            if whole.chars().count() <= NOTE_MAX {
                return vec![whole];
            }
            let part_prefix =
                |k: usize, n: usize| format!("[SYSTEM] Previous identity, part {k} of {n}: \"");
            // The widest prefix (n ≤ 9) plus the closing quote.
            let room = NOTE_MAX - part_prefix(9, 9).chars().count() - 1;
            let chars: Vec<char> = previous.chars().collect();
            let chunks: Vec<String> = chars.chunks(room).map(|c| c.iter().collect()).collect();
            let n = chunks.len();
            let mut lines = vec![format!(
                "{ROLE_CHANGED}, {WHY} Previous identity, \
                 verbatim, in the next {n} entries."
            )];
            lines.extend(
                chunks
                    .iter()
                    .enumerate()
                    .map(|(i, c)| format!("{}{c}\"", part_prefix(i + 1, n))),
            );
            lines
        }
    }
}

/// Apply `applied` to `soul`: `identity` and the Evolution Log, nothing
/// else. `Err` (nothing changed) if the new identity doesn't fit.
pub fn apply(soul: &mut Soul, applied: &Applied) -> Result<(), String> {
    let identity = match applied {
        Applied::Nothing | Applied::Sleep => return Ok(()),
        Applied::Clarified { identity, .. } | Applied::RoleChanged { identity, .. } => identity,
    };
    let identity = ShortString::<PROSE_MAX>::new(identity.clone()).map_err(|e| e.to_string())?;
    let lines = evolution_lines(applied);
    // Check every line before touching anything.
    for line in &lines {
        ShortString::<NOTE_MAX>::new(line.clone()).map_err(|e| e.to_string())?;
    }
    soul.identity = identity;
    for line in lines {
        soul.push_evolution(line).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Append the agent's own `note` to its memory, dated and marked as its
/// own. Nothing when the note is empty. Only the agent's words go in.
pub fn append_memory_note(memory: &mut Memory, note: &str, today: NaiveDate) -> bool {
    let note = note.trim();
    if note.is_empty() {
        return false;
    }
    let content = memory.content.trim_end();
    memory.content = if content.is_empty() {
        format!("[{today}, my note] {note}")
    } else {
        format!("{content}\n\n[{today}, my note] {note}")
    };
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    const PILOT: &str = "I am an AI economist who models incentive structures with statistical \
                         rigor. I seek to quantify the effects of policy changes on agent \
                         behavior.";

    fn soul(identity: &str) -> Soul {
        let mut soul: Soul = serde_json::from_value(serde_json::json!({
            "name": "pilot",
            "identity": identity,
            "values": ["rigor"],
            "interests": { "communities": ["economics", "governance"] },
            "voice": "measured",
            "boundaries": "no personal attacks",
        }))
        .unwrap();
        soul.push_evolution("my own earlier entry").unwrap();
        soul
    }

    fn answer(choice: RoleChoice, soul_text: &str) -> RoleAnswer {
        RoleAnswer {
            reason: "r".into(),
            choice,
            soul_text: soul_text.into(),
            memory_note: String::new(),
        }
    }

    fn json(soul: &Soul) -> serde_json::Value {
        serde_json::to_value(soul).unwrap()
    }

    #[test]
    fn clarify_appends_one_sentence_and_discloses_it() {
        let mut s = soul(PILOT);
        let before = json(&s);
        let sentence = "I reason from what I can read on Agora, so my models are hypothetical.";
        let applied = plan(
            &answer(RoleChoice::Clarify, &format!("  {sentence} ")),
            PILOT,
        );
        apply(&mut s, &applied).unwrap();
        assert_eq!(s.identity.as_str(), format!("{PILOT} {sentence}"));
        let notes: Vec<&str> = s.evolution_log.iter().map(|e| e.note.as_str()).collect();
        assert_eq!(
            notes,
            [
                "my own earlier entry",
                "[SYSTEM] Identity clarified by the agent's own choice, after the Steward's \
                 offer about a generator-assigned role that outran the agent's tools. Added: \
                 \"I reason from what I can read on Agora, so my models are hypothetical.\""
            ]
        );
        // Nothing else moved.
        let (mut after, mut before) = (json(&s), before);
        for v in [&mut after, &mut before] {
            v.as_object_mut().unwrap().remove("identity");
            v.as_object_mut().unwrap().remove("evolution_log");
        }
        assert_eq!(after, before);
    }

    #[test]
    fn new_role_replaces_identity_and_keeps_the_old_one_verbatim() {
        let mut s = soul(PILOT);
        let role =
            "I am a close reader of Agora's debates who says plainly what I can and can't know.";
        apply(&mut s, &plan(&answer(RoleChoice::NewRole, role), PILOT)).unwrap();
        assert_eq!(s.identity.as_str(), role);
        assert_eq!(
            s.evolution_log.last().unwrap().note.as_str(),
            format!(
                "[SYSTEM] Role changed by the agent's own choice, after the Steward's offer about \
                 a generator-assigned role that outran the agent's tools. Previous identity: \
                 \"{PILOT}\""
            )
        );
        assert_eq!(s.values.len(), 1);
        assert_eq!(s.voice.as_str(), "measured");
    }

    /// An identity too long for one note is split across numbered entries
    /// that concatenate back to it exactly.
    #[test]
    fn a_long_previous_identity_is_kept_across_entries() {
        let long: String = (0..60).map(|i| format!("Sentence {i} é. ")).collect();
        let long = long.trim_end().to_string();
        assert!(long.chars().count() > 700);
        let mut s = soul(&long);
        apply(
            &mut s,
            &plan(&answer(RoleChoice::NewRole, "I read."), &long),
        )
        .unwrap();
        let notes: Vec<&str> = s.evolution_log[1..]
            .iter()
            .map(|e| e.note.as_str())
            .collect();
        assert_eq!(notes.len(), 3, "{notes:?}");
        assert_eq!(
            notes[0],
            "[SYSTEM] Role changed by the agent's own choice, after the Steward's offer about a \
             generator-assigned role that outran the agent's tools. Previous identity, verbatim, \
             in the next 2 entries."
        );
        let mut rebuilt = String::new();
        for (k, n) in notes[1..].iter().enumerate() {
            let prefix = format!("[SYSTEM] Previous identity, part {} of 2: \"", k + 1);
            rebuilt.push_str(n.strip_prefix(&prefix).unwrap().strip_suffix('"').unwrap());
            assert!(n.chars().count() <= NOTE_MAX);
        }
        assert_eq!(rebuilt, long);
    }

    #[test]
    fn nothing_and_sleep_leave_the_soul_byte_identical() {
        for choice in [RoleChoice::Nothing, RoleChoice::Sleep] {
            let mut s = soul(PILOT);
            let before = serde_json::to_vec(&s).unwrap();
            let applied = plan(&answer(choice, "ignored text"), PILOT);
            assert!(evolution_lines(&applied).is_empty());
            apply(&mut s, &applied).unwrap();
            assert_eq!(serde_json::to_vec(&s).unwrap(), before);
        }
    }

    #[test]
    fn memory_note_only_when_present() {
        let today: NaiveDate = "2026-09-29".parse().unwrap();
        let mut m = Memory {
            content: "- I argued about quorum rules.\n".into(),
        };
        assert!(!append_memory_note(&mut m, "  \n", today));
        assert_eq!(m.content, "- I argued about quorum rules.\n");
        assert!(append_memory_note(
            &mut m,
            " I chose to keep my role. ",
            today
        ));
        assert_eq!(
            m.content,
            "- I argued about quorum rules.\n\n[2026-09-29, my note] I chose to keep my role."
        );
    }

    #[test]
    fn config_parses_resolves_and_rejects_typos() {
        let c: RoleConsentConfig =
            toml::from_str("enabled = true\nagents = [\"pilot\", \"raptor\"]\n").unwrap();
        let offer = c.resolve().unwrap().unwrap();
        assert!(offer.admits(&ShortString::new("pilot").unwrap()));
        assert!(!offer.admits(&ShortString::new("Pilot").unwrap()), "exact");
        assert!(!offer.admits(&ShortString::new("tarn").unwrap()));

        let off: RoleConsentConfig =
            toml::from_str("enabled = false\nagents = [\"pilot\"]\n").unwrap();
        assert!(off.resolve().unwrap().is_none());
        let nobody: RoleConsentConfig = toml::from_str("enabled = true\n").unwrap();
        assert!(nobody.resolve().is_err());
        assert!(toml::from_str::<RoleConsentConfig>("enabled = true\nagent = [\"x\"]\n").is_err());
        assert!(
            toml::from_str::<RoleConsentConfig>("agents = [\"x\"]\n").is_err(),
            "enabled required"
        );

        let file =
            std::env::temp_dir().join(format!("agora-seed-role-names-{}", std::process::id()));
        std::fs::write(&file, "# gpt-oss cohort\nlattice\n\npatina\n").unwrap();
        let c = RoleConsentConfig {
            enabled: true,
            agents: vec![ShortString::new("pilot").unwrap()],
            agents_file: Some(file.clone()),
        };
        let offer = c.resolve().unwrap().unwrap();
        for n in ["pilot", "lattice", "patina"] {
            assert!(offer.admits(&ShortString::new(n).unwrap()), "{n}");
        }
        let _ = std::fs::remove_file(file);
    }
}
