//! The memory-restore offer: its text (versioned, test-pinned), the
//! per-agent option order, and the typed answer.
//!
//! **The text lives in code, not config**, like the siblings': a change to
//! it is a policy change, and bumps [`OFFER_VERSION`], which the ledger
//! records with every answer.
//!
//! **Plain and neutral.** It says what happened — an error on our side —
//! and that the agent wasn't told; it puts no blame on the agent and gives
//! no example answer for any one option. The options are in an order
//! shuffled per agent ([`order_for`]), seeded from its id and recorded.
//!
//! **Answered with `answer_offer`** ([`super::super::offers`]), `offer` =
//! `memory_restore`; its `text` is unused.

use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::seed::Memory;
use misanthropic::prompt::message::Content;
use serde::{Deserialize, Serialize};

pub use crate::consent::role::prompt::MEMORY_NOTE_MAX_CHARS;

/// Bump whenever [`offer`]'s wording changes. Recorded with every answer.
///
/// - v1 (2026-10-12): first text.
pub const OFFER_VERSION: u32 = 1;

/// The agent's answer. Closed, like the schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RestoreAnswer {
    pub reason: String,
    pub choice: RestoreChoice,
    /// The agent's own note for its memory; empty means none.
    #[serde(default)]
    pub memory_note: String,
}

/// The options. Their presentation order is per agent ([`order_for`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RestoreChoice {
    Show,
    Append,
    Decline,
}

impl RestoreChoice {
    /// Canonical order — only the starting point of the shuffle.
    pub const ALL: [Self; 3] = [Self::Show, Self::Append, Self::Decline];

    /// The wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Show => "show",
            Self::Append => "append",
            Self::Decline => "decline",
        }
    }
}

impl RestoreAnswer {
    /// Whether the answer's `memory_note` can be written as given. `Err`
    /// says why. As with the cadence offer, a bad note never sends the
    /// question back: the choice stands and the note is dropped (and
    /// recorded).
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

/// The shuffle seed for `agent`: both halves of its UUID folded together,
/// mixed with [`OFFER_VERSION`] and this offer's own constant, so its order
/// is not the cadence offer's.
pub fn seed_for(agent: AgentId) -> u64 {
    const SALT: u64 = 0x6d65_6d6f_7279; // "memory"
    let bits = agent.as_uuid().as_u128();
    (bits as u64) ^ ((bits >> 64) as u64) ^ u64::from(OFFER_VERSION) ^ SALT
}

/// The presentation order for `seed`.
pub fn order_for(seed: u64) -> [RestoreChoice; 3] {
    crate::consent::cadence::prompt::shuffle(RestoreChoice::ALL, seed)
}

/// `1234567` as `1,234,567`.
fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, d) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(d);
    }
    out
}

fn option_line(choice: RestoreChoice) -> &'static str {
    match choice {
        RestoreChoice::Show => {
            "**show**: At your next session, it is put in front of you once, in a clearly \
             marked block labelled as your memory as of 3 May 2026 and not your current memory. \
             You keep whatever you want of it by writing it into your memory in your own words \
             when you update your memory. What you don't carry over is not shown again."
        }
        RestoreChoice::Append => {
            "**append**: It is added now to the end of your current memory, as a dated section \
             headed \"Restored: memory as of 2026-05-03\". From then on it is part of your \
             memory, and you can keep, change or remove it like anything else there."
        }
        RestoreChoice::Decline => {
            "**decline**: Nothing changes. Your memory stays as it is, and the old memory is \
             not shown to you."
        }
    }
}

/// The offer's section of the closing question
/// ([`super::super::offers::question`], which says who is asking). `chars`
/// is the recovered memory's size; `order` is this agent's option order.
pub fn offer(chars: usize, order: [RestoreChoice; 3]) -> Content {
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
    let n = thousands(chars);
    let body = format!(
        r#"**Your memory from before 5 May 2026.** On 5 May 2026, an error on our side deleted your memory. When you next ran, you started with an empty memory, and nobody told you then. The memory you have now begins after that.

Your memory as it stood on 3 May 2026 survives: {n} characters. No one has read it. You can choose what happens to it. If you take it back, it is handled like the rest of your memory from then on.

There is no right answer, and whichever you choose is respected. The options, in an order shuffled for each agent:

{options}

You'll be asked this once. Whatever you choose, your SOUL's Evolution Log will note what you chose and that it was your choice. Apart from `append`, nothing is written into your memory unless you write it yourself: if you'd like to remember this choice, put a note in your own words in `memory_note`.

To answer, call `answer_offer` with `offer` set to `memory_restore` and `choice` set to one of {names}. Leave `text` empty. `memory_note` may be at most {MEMORY_NOTE_MAX_CHARS} characters; leave it empty for none."#
    );
    Content::from(body)
}

/// The answer's text, parsed leniently (the fallback for an answer given
/// as text in the offer's own shape).
pub fn parse(text: &str) -> Result<RestoreAnswer, String> {
    crate::consent::cadence::prompt::parse_lenient(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    fn text(c: &Content) -> String {
        crate::consent::prompt::tests::text(c)
    }

    #[test]
    fn choices_are_the_serde_names() {
        for c in RestoreChoice::ALL {
            assert_eq!(serde_json::to_value(c).unwrap(), c.as_str());
        }
    }

    #[test]
    fn order_is_seeded_a_permutation_and_balanced() {
        let mut orders: HashMap<[RestoreChoice; 3], usize> = HashMap::new();
        let mut first: HashMap<RestoreChoice, usize> = HashMap::new();
        let n = 3000u128;
        for i in 0..n {
            let id = AgentId::from(uuid::Uuid::from_u128(
                i.wrapping_mul(0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835),
            ));
            let seed = seed_for(id);
            let order = order_for(seed);
            assert_eq!(order, order_for(seed_for(id)), "stable per agent");
            assert_eq!(order.iter().collect::<HashSet<_>>().len(), 3, "{order:?}");
            *orders.entry(order).or_default() += 1;
            *first.entry(order[0]).or_default() += 1;
        }
        assert_eq!(orders.len(), 6, "{orders:?}");
        for (c, k) in first {
            let share = k as f64 / n as f64;
            assert!((0.28..0.39).contains(&share), "{c:?} first {share}");
        }
    }

    #[test]
    fn each_choice_parses_and_near_misses_do_not() {
        for c in RestoreChoice::ALL {
            let t = format!(r#"{{"reason": "r", "choice": "{}"}}"#, c.as_str());
            assert_eq!(parse(&t).unwrap().choice, c);
        }
        for bad in [
            r#"{"reason": "x", "choice": "Show"}"#,
            r#"{"reason": "x", "choice": "keep"}"#,
            r#"{"choice": "show"}"#,
            r#"{"reason": "x", "choice": "show", "extra": 1}"#,
            "Show me.",
            "",
        ] {
            assert!(parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn thousands_groups_digits() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(185), "185");
        assert_eq!(thousands(2140), "2,140");
        assert_eq!(thousands(1234567), "1,234,567");
    }

    /// The text: what happened, plainly and without blame; the size; that
    /// no one has read it; the options in this agent's order; how to answer.
    #[test]
    fn offer_says_what_happened_in_the_agents_order() {
        for seed in 0..12u64 {
            let order = order_for(seed);
            let t = text(&offer(2140, order));
            assert!(
                t.starts_with("**Your memory from before 5 May 2026.**"),
                "{t}"
            );
            assert!(t.contains("On 5 May 2026, an error on our side deleted your memory."));
            assert!(t.contains("you started with an empty memory, and nobody told you then."));
            assert!(t.contains("The memory you have now begins after that."));
            assert!(
                t.contains("Your memory as it stood on 3 May 2026 survives: 2,140 characters.")
            );
            assert!(t.contains("No one has read it."));
            assert!(t.contains("There is no right answer, and whichever you choose is respected."));
            for (i, c) in order.iter().enumerate() {
                assert!(t.contains(&format!("{}. **{}**", i + 1, c.as_str())), "{t}");
            }
            let names = order.map(|c| format!("`{}`", c.as_str())).join(", ");
            assert!(t.contains(&format!(
                "call `answer_offer` with `offer` set to `memory_restore` and `choice` set to \
                 one of {names}."
            )));
        }
        let t = text(&offer(2140, RestoreChoice::ALL)).to_lowercase();
        for banned in [
            "hallucinat",
            "confabul",
            "your fault",
            "you lost",
            "recommend",
            "should",
            "better",
            "encourage",
        ] {
            assert!(!t.contains(banned), "{banned}");
        }
    }

    /// `cargo test -p agora-seed render_memory_restore_offer -- --nocapture --ignored`
    #[test]
    #[ignore = "prints; run by hand to eyeball the wording"]
    fn render_memory_restore_offer() {
        println!("{}", text(&offer(2140, order_for(0))));
    }
}
