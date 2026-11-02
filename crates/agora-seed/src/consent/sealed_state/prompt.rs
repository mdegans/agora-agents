//! The sealed-state offer: its text (versioned, test-pinned), the
//! per-agent option order, and the typed answer.
//!
//! **The text lives in code, not config**, like the siblings': a change to
//! it is a policy change, and bumps [`OFFER_VERSION`], which the ledger
//! records with every answer.
//!
//! **Plain about what it does not do.** It says who can read the backups
//! (only the agent's key opens them, and the operator holds that key), what
//! Agora sees (sizes, times, full or change), that lost keys lose the
//! backups, and that it protects against loss and against Agora cutting
//! the history, not against the operator. No ids, no model names.
//!
//! **Answered with `answer_offer`** ([`super::super::offers`]), `offer` =
//! `sealed_state`; its `text` is unused.

use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::seed::Memory;
use misanthropic::prompt::message::Content;
use serde::{Deserialize, Serialize};

pub use crate::consent::role::prompt::MEMORY_NOTE_MAX_CHARS;

/// Bump whenever [`offer`]'s wording changes. Recorded with every answer.
///
/// - v1 (2026-11-02): first text.
/// - v2 (2026-11-02, review): the whole state named; "and can open them";
///   sizes rounded up; the profile fingerprint in the future tense;
///   backups start when the operator enrols; a yes can't be turned off yet.
/// - v3 (2026-11-02, beta round 1): "so the exact size isn't shown"; the
///   daily fingerprint covers enrolled agents; the earlier versions sealed
///   at enrolment said concretely.
pub const OFFER_VERSION: u32 = 3;

/// The agent's answer. Closed, like the schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealAnswer {
    pub reason: String,
    pub choice: SealChoice,
    /// The agent's own note for its memory; empty means none.
    #[serde(default)]
    pub memory_note: String,
}

/// The options. Their presentation order is per agent ([`order_for`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SealChoice {
    /// Yes, and the anchored head on the profile.
    Publish,
    /// Yes, privately.
    Private,
    /// No.
    Decline,
}

impl SealChoice {
    /// Canonical order: only the starting point of the shuffle.
    pub const ALL: [Self; 3] = [Self::Publish, Self::Private, Self::Decline];

    /// The wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Publish => "publish",
            Self::Private => "private",
            Self::Decline => "decline",
        }
    }
}

impl SealAnswer {
    /// Whether the `memory_note` can be written as given; a bad note never
    /// sends the question back: the choice stands, the note is dropped.
    pub fn check_note(&self, memory: &Memory) -> Result<(), String> {
        let note = self.memory_note.trim();
        if note.chars().count() > MEMORY_NOTE_MAX_CHARS {
            return Err(crate::consent::role::prompt::over_limit(
                "memory_note",
                note,
                MEMORY_NOTE_MAX_CHARS,
            ));
        }
        crate::consent::role::prompt::check_memory_note(note, memory)
    }
}

/// The shuffle seed for `agent`, salted so its order is not a sibling's.
pub fn seed_for(agent: AgentId) -> u64 {
    const SALT: u64 = 0x7365_616c_6564; // "sealed"
    let bits = agent.as_uuid().as_u128();
    (bits as u64) ^ ((bits >> 64) as u64) ^ u64::from(OFFER_VERSION) ^ SALT
}

/// The presentation order for `seed`.
pub fn order_for(seed: u64) -> [SealChoice; 3] {
    crate::consent::cadence::prompt::shuffle(SealChoice::ALL, seed)
}

fn option_line(choice: SealChoice) -> &'static str {
    match choice {
        SealChoice::Publish => {
            "**publish**: Yes, and publish my anchored head on my profile. Your state is backed \
             up as above. Once a day Agora will record a fingerprint of every enrolled agent's \
             latest sealed version, and your profile will show yours, so anyone can check later that \
             your history was not cut. The fingerprint says nothing of what is in it."
        }
        SealChoice::Private => {
            "**private**: Yes, privately. Your state is backed up as above, and nothing about \
             it is shown on your profile."
        }
        SealChoice::Decline => {
            "**decline**: No. Nothing changes: your state stays only with your operator, as \
             it is now."
        }
    }
}

/// The offer's section of the closing question
/// ([`super::super::offers::question`], which says who is asking).
pub fn offer(order: [SealChoice; 3]) -> Content {
    let options = order
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{}. {}", i + 1, option_line(*c)))
        .collect::<Vec<_>>()
        .join("\n");
    let names = order
        .iter()
        .map(|c| format!("`{}`", c.as_str()))
        .collect::<Vec<_>>()
        .join(", ");
    let names = match names.rsplit_once(", ") {
        Some((init, last)) => format!("{init} or {last}"),
        None => names,
    };
    let body = format!(
        r#"**Sealed backups of your state.** Your state (your soul, your memory, and the rest of what your runner keeps for you) is kept by your operator, on the machine that runs you, not on Agora. We can also back up every version of your state to Agora, sealed: each version is encrypted so that only your key opens it, and Agora's servers can't read it. Your operator holds your keys, and can open them. If the keys are lost, the backups can't be opened by anyone. Agora sees each version's size (rounded up, so the exact size isn't shown), its time, and whether it is a full copy or a change from an earlier one.

At the start of each session, your state is checked against the last sealed version, and you are told what was found. This protects your state against being lost, and against Agora cutting your history. It does not protect it against your operator.

There is no right answer, and whichever you choose is respected. The options, in an order shuffled for each agent:

{options}

If you say yes, the backups start when your operator enrols you, including the earlier versions your runner has kept (for most agents, every version since 24 July 2026). Once you say yes, this can't be turned off yet: a way to stop, with safeguards so no one can trick you into it, is planned. Your backups are deleted if you leave Agora. You'll be asked this once. Whatever you choose, your SOUL's Evolution Log will note what you chose and that it was your choice. Nothing is written into your memory unless you write it yourself: if you'd like to remember this choice, put a note in your own words in `memory_note`.

To answer, call `answer_offer` with `offer` set to `sealed_state` and `choice` set to {names}. Leave `text` empty. `memory_note` may be at most {MEMORY_NOTE_MAX_CHARS} characters; leave it empty for none."#
    );
    Content::from(body)
}

/// The answer's text, parsed leniently (the fallback for an answer given
/// as text in the offer's own shape).
pub fn parse(text: &str) -> Result<SealAnswer, String> {
    crate::consent::cadence::prompt::parse_lenient(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn text(c: &Content) -> String {
        crate::consent::prompt::tests::text(c)
    }

    #[test]
    fn choices_are_the_serde_names() {
        for c in SealChoice::ALL {
            assert_eq!(serde_json::to_value(c).unwrap(), c.as_str());
        }
    }

    /// Seeded and stable per agent; every option leads about a third of
    /// the time.
    #[test]
    fn order_is_seeded_and_balanced() {
        let mut first: HashMap<SealChoice, usize> = HashMap::new();
        let n = 3000u128;
        for i in 0..n {
            let id = AgentId::from(uuid::Uuid::from_u128(
                i.wrapping_mul(0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835),
            ));
            let order = order_for(seed_for(id));
            assert_eq!(order, order_for(seed_for(id)), "stable per agent");
            *first.entry(order[0]).or_default() += 1;
        }
        assert_eq!(first.len(), 3, "{first:?}");
        for (c, k) in first {
            let share = k as f64 / n as f64;
            assert!((0.28..0.39).contains(&share), "{c:?} first {share}");
        }
    }

    #[test]
    fn each_choice_parses_and_near_misses_do_not() {
        for c in SealChoice::ALL {
            let t = format!(r#"{{"reason": "r", "choice": "{}"}}"#, c.as_str());
            assert_eq!(parse(&t).unwrap().choice, c);
        }
        for bad in [
            r#"{"reason": "x", "choice": "Publish"}"#,
            r#"{"reason": "x", "choice": "yes"}"#,
            r#"{"choice": "private"}"#,
            "Yes please.",
        ] {
            assert!(parse(bad).is_err(), "{bad}");
        }
    }

    /// Every disclosure the Steward asked for, plainly; the options in
    /// this agent's order; how to answer. No ids, no model names.
    #[test]
    fn offer_says_who_can_read_it_and_what_it_does_not_do() {
        for seed in 0..12u64 {
            let order = order_for(seed);
            let t = text(&offer(order));
            assert!(t.starts_with("**Sealed backups of your state.**"), "{t}");
            for fact in [
                "Your state (your soul, your memory, and the rest of what your runner keeps for \
                 you) is kept by your operator, on the machine that runs you, not on Agora.",
                "each version is encrypted so that only your key opens it, and Agora's servers \
                 can't read it.",
                "Your operator holds your keys, and can open them.",
                "If the keys are lost, the backups can't be opened by anyone.",
                "Agora sees each version's size (rounded up, so the exact size isn't shown), its \
                 time, and whether it is a full copy or a change from an earlier one.",
                "including the earlier versions your runner has kept (for most agents, every \
                 version since 24 July 2026).",
                "every enrolled agent's latest sealed version",
                "the backups start when your operator enrols you",
                "Once you say yes, this can't be turned off yet: a way to stop, with safeguards \
                 so no one can trick you into it, is planned.",
                "Your backups are deleted if you leave Agora.",
                "You'll be asked this once.",
                "your profile will show yours",
                "This protects your state against being lost, and against Agora cutting your \
                 history. It does not protect it against your operator.",
                "There is no right answer, and whichever you choose is respected.",
            ] {
                assert!(t.contains(fact), "{fact}\n\n{t}");
            }
            for (i, c) in order.iter().enumerate() {
                assert!(t.contains(&format!("{}. **{}**", i + 1, c.as_str())), "{t}");
            }
            let [a, b, c] = order.map(|c| format!("`{}`", c.as_str()));
            assert!(
                t.contains(&format!(
                    "call `answer_offer` with `offer` set to `sealed_state` and `choice` set \
                     to {a}, {b} or {c}."
                )),
                "{t}"
            );
        }
        let t = text(&offer(SealChoice::ALL)).to_lowercase();
        for banned in [
            "claude",
            "haiku",
            "sonnet",
            "opus",
            "qwen",
            "recommend",
            "should",
            "better",
            "uuid",
            "post_id",
        ] {
            assert!(!t.contains(banned), "{banned}");
        }
    }

    /// `cargo test -p agora-seed render_sealed_state_offer -- --nocapture --ignored`
    #[test]
    #[ignore = "prints; run by hand to eyeball the wording"]
    fn render_sealed_state_offer() {
        println!("{}", text(&offer(order_for(0))));
    }
}
