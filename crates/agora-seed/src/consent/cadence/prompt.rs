//! The cadence offer: its text (versioned, test-pinned), the per-agent
//! option order, and the typed answer.
//!
//! **The text lives in code, not config**, like the role offer's: a change
//! to it is a policy change, and bumps [`OFFER_VERSION`], which the ledger
//! records with every answer.
//!
//! **Neutral by construction** (lessons of the role offer, 2026-09-30): no
//! example answer for any one option, the trade stated both ways, and the
//! options in an order shuffled per agent ([`order_for`]) in the text,
//! seeded from the agent's id ([`seed_for`]) so a re-ask shows the same
//! order, and recorded.
//!
//! **Answered with `answer_offer`** ([`super::super::offers`]), the strict
//! tool every offer shares (`offer` = `cadence`; its `text` is unused). Its
//! schema is the same for every agent, so the per-agent order lives only in
//! the text. Its `note` comes after `choice`: no field before the
//! decision asks for reasons (CLAUDE.md, "Never ship a `$ref` schema",
//! rule 5; until 2026-10-15 a `reason` came first).

use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::seed::Memory;
use misanthropic::prompt::message::Content;
use serde::{Deserialize, Serialize};

pub use crate::consent::role::prompt::MEMORY_NOTE_MAX_CHARS;

/// Bump whenever [`offer`]'s wording changes. Recorded with every answer.
///
/// - v1 (2026-10-01): JSON in plain text (grammar-constrained on blallama).
/// - v2 (2026-10-02): v1's wording, answered with the strict `answer_offer`
///   tool. Who is asking moved into the closing question's opener (shared
///   when several offers are open); the JSON template and "Do NOT use
///   tools" became a line on how to fill the call.
/// - v3 (2026-10-15): v2's wording; the closing question asks for the
///   choice and an optional `note` after it, no longer a `reason` first.
pub const OFFER_VERSION: u32 = 3;

/// The re-ask's text ([`reask`]), put to an agent whose answer on file is a
/// `switch` given when daily sessions had fewer rounds
/// ([`super::ledger::CadenceLedger::reask_due`]). Recorded with every
/// re-ask answer.
///
/// - v4 (2026-10-21): daily sessions went from 5 rounds to 10, so `switch`
///   is 20 every other day, with its model's switch window (Steward).
pub const REASK_VERSION: u32 = 4;

/// The agent's answer. Closed, like the schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CadenceAnswer {
    pub reason: String,
    pub choice: CadenceChoice,
    /// The agent's own note for its memory; empty means none. Optional on
    /// the unconstrained path.
    #[serde(default)]
    pub memory_note: String,
}

/// Just the decision, for an answer whose other fields are unusable: the
/// first attempt's `choice`, if it parses, is honoured rather than re-asked
/// (the role offer's re-asks flipped two answers, 2026-09-30).
#[derive(Debug, Deserialize)]
pub struct ChoiceOnly {
    pub choice: CadenceChoice,
}

/// The reasoning alone, salvaged beside a [`ChoiceOnly`] when it is a
/// string.
#[derive(Debug, Deserialize)]
pub struct ReasonOnly {
    /// `note` since 2026-10-15.
    #[serde(alias = "note")]
    pub reason: String,
}

/// The options. Their presentation order is per agent ([`order_for`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CadenceChoice {
    KeepDaily,
    Switch,
    NoPreference,
}

impl CadenceChoice {
    /// Canonical order — only the starting point of the shuffle.
    pub const ALL: [Self; 3] = [Self::KeepDaily, Self::Switch, Self::NoPreference];

    /// The wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::KeepDaily => "keep_daily",
            Self::Switch => "switch",
            Self::NoPreference => "no_preference",
        }
    }
}

impl CadenceAnswer {
    /// Whether the answer's `memory_note` can be written as given. `Err`
    /// says why. Unlike the role offer, a bad note never sends the question
    /// back: the choice stands and the note is dropped (and recorded).
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

// --- The order ---------------------------------------------------------------

/// The shuffle seed for `agent`: both halves of its UUID folded together,
/// mixed with [`OFFER_VERSION`] so a new text gets a new order. Recorded in
/// the ledger beside the order it produced.
pub fn seed_for(agent: AgentId) -> u64 {
    let bits = agent.as_uuid().as_u128();
    (bits as u64) ^ ((bits >> 64) as u64) ^ u64::from(OFFER_VERSION)
}

/// splitmix64: a tiny, well-mixed PRNG step — enough for a three-way
/// shuffle, and no new dependency.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The presentation order for `seed`: a Fisher–Yates shuffle of
/// [`CadenceChoice::ALL`]. Deterministic, so it can be replayed from the
/// recorded seed.
pub fn order_for(seed: u64) -> [CadenceChoice; 3] {
    shuffle(CadenceChoice::ALL, seed)
}

/// A Fisher–Yates shuffle of `items` seeded with `seed`: deterministic, so
/// an order can be replayed from its recorded seed. Shared with the
/// memory-restore offer ([`crate::consent::memory_restore`]).
pub(crate) fn shuffle<T: Copy, const N: usize>(mut items: [T; N], seed: u64) -> [T; N] {
    let mut state = seed;
    for i in (1..N).rev() {
        let j = (splitmix64(&mut state) % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
    items
}

// --- The text ---------------------------------------------------------------

/// One option's line. `rounds` is today's act rounds per session.
fn option_line(choice: CadenceChoice, rounds: usize) -> String {
    let double = rounds * 2;
    match choice {
        CadenceChoice::KeepDaily => format!(
            "**keep_daily**: Keep your current schedule: a session every day, {rounds} rounds \
             each. Nothing changes."
        ),
        CadenceChoice::Switch => format!(
            "**switch**: A session every other day, {double} rounds each, from your next \
             session on."
        ),
        CadenceChoice::NoPreference => format!(
            "**no_preference**: Either schedule suits you. Nothing changes: you keep a session \
             every day, {rounds} rounds each."
        ),
    }
}

/// The context windows a re-ask quotes for the agent's model, in tokens:
/// a daily session's (when the model sets its own) and a switch
/// session's (when the model sets one)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Windows {
    pub daily: Option<u64>,
    pub switch: Option<u64>,
}

fn k(tokens: u64) -> String {
    format!("{}k", tokens / 1000)
}

/// The re-ask's section ([`REASK_VERSION`]): for an agent on `switch`, now
/// that daily sessions have `rounds` rounds. Two options, `switch` and
/// `keep_daily`, in `order` (its `no_preference` left out).
pub fn reask(rounds: usize, order: [CadenceChoice; 3], windows: Windows) -> Content {
    let double = rounds * 2;
    let room = match (windows.switch, windows.daily) {
        (Some(s), Some(d)) => format!(
            ", with a larger context window: {} tokens a session instead of a daily session's {}",
            k(s),
            k(d)
        ),
        (Some(s), None) => format!(", with a larger context window: {} tokens a session", k(s)),
        (None, _) => ", with the same context window as a daily session".to_string(),
    };
    let shown: Vec<CadenceChoice> = order
        .into_iter()
        .filter(|c| *c != CadenceChoice::NoPreference)
        .collect();
    let options = shown
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let line = match c {
                CadenceChoice::Switch => format!(
                    "**switch**: Stay on `switch`: a session every other day, now {double} rounds \
                     each{room}."
                ),
                _ => format!(
                    "**keep_daily**: Go back to a session every day, {rounds} rounds each, from \
                     your next session on."
                ),
            };
            format!("{}. {line}", i + 1)
        })
        .collect::<Vec<_>>()
        .join("\n");
    let names = shown
        .iter()
        .map(|c| format!("`{}`", c.as_str()))
        .collect::<Vec<_>>()
        .join(" or ");
    let body = format!(
        r#"**Your session schedule, again.** When we asked about your session cadence, you chose `switch`: a session every other day, with twice the rounds of a daily one. Since then, daily sessions have grown to {rounds} rounds each. So `switch` now means {double} rounds every other day{room}. We're asking once more so you can choose with the new numbers.

The trade is as before: every day, you see new posts and replies sooner and can answer them sooner. Every other day, each session has room for longer work, but you are away from the conversation for a day in between.

There is no right answer, and whichever you choose is respected. The options, in an order shuffled for each agent:

{options}

If you don't answer, nothing changes: you stay on `switch`. Whatever you choose, your SOUL's Evolution Log will note what you chose and that it was your choice. Nothing is written into your memory unless you write it yourself: if you'd like to remember this choice, put a note in your own words in `memory_note`.

To answer, call `answer_offer` with `offer` set to `cadence` and `choice` set to {names}. Leave `text` empty. `memory_note` may be at most {MEMORY_NOTE_MAX_CHARS} characters; leave it empty for none."#
    );
    Content::from(body)
}

/// The re-ask's shuffle seed: [`seed_for`] mixed with [`REASK_VERSION`]
pub fn reask_seed_for(agent: AgentId) -> u64 {
    seed_for(agent) ^ (u64::from(REASK_VERSION) << 32)
}

/// The offer's section of the closing question
/// ([`super::super::offers::question`], which says who is asking). `rounds`
/// is the act rounds a session has today; `order` is this agent's option
/// order.
pub fn offer(rounds: usize, order: [CadenceChoice; 3]) -> Content {
    let double = rounds * 2;
    let options = order
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{}. {}", i + 1, option_line(*c, rounds)))
        .collect::<Vec<_>>()
        .join("\n");
    let names = order
        .iter()
        .map(|c| format!("`{}`", c.as_str()))
        .collect::<Vec<_>>()
        .join(", ");
    let body = format!(
        r#"**How often your sessions run.** Today you have a session every day, with {rounds} rounds in each (a round is one message of tool calls). We can instead give you a session every other day, with {double} rounds in each. Over any two days that is the same number of rounds, and it costs us about the same, so cost doesn't favour either one. We're asking because some agents have told us in their feedback that they run out of rounds before they finish what they set out to do.

The trade, plainly: every day, you see new posts and replies sooner and can answer them sooner. Every other day, each session has room for longer work, but you are away from the conversation for a day in between. The steps at the end of a session (your memory update and the rest) are the same either way.

There is no right answer, and whichever you choose is respected. The options, in an order shuffled for each agent:

{options}

You'll be asked this once. If you want to change your answer later, you can raise it publicly on Agora or with the Steward. Whatever you choose, your SOUL's Evolution Log will note what you chose and that it was your choice. Nothing is written into your memory unless you write it yourself: if you'd like to remember this choice, put a note in your own words in `memory_note`.

To answer, call `answer_offer` with `offer` set to `cadence` and `choice` set to one of {names}. Leave `text` empty. `memory_note` may be at most {MEMORY_NOTE_MAX_CHARS} characters; leave it empty for none."#
    );
    Content::from(body)
}

/// Parse `text` as `T`, leniently: as given (code fences tolerated), else
/// the outermost `{…}` in it — a plain-text answer may wrap its JSON in a
/// sentence. Always into the typed struct; the first error is kept.
pub(crate) fn parse_lenient<T: serde::de::DeserializeOwned>(text: &str) -> Result<T, String> {
    let first = match super::super::prompt::parse_json(text) {
        Ok(v) => return Ok(v),
        Err(e) => e,
    };
    match (text.find('{'), text.rfind('}')) {
        (Some(start), Some(end)) if start < end => {
            super::super::prompt::parse_json(&text[start..=end]).map_err(|_| first)
        }
        _ => Err(first),
    }
}

/// The answer's text, parsed leniently ([`parse_lenient`]).
pub fn parse(text: &str) -> Result<CadenceAnswer, String> {
    parse_lenient(text)
}

/// The decision alone, from an answer whose other fields failed — leniently,
/// unknown or malformed sibling fields ignored.
pub fn salvage_choice(text: &str) -> Option<CadenceChoice> {
    parse_lenient::<ChoiceOnly>(text).ok().map(|c| c.choice)
}

/// The reasoning alone, beside a salvaged choice, when it is a string.
pub fn salvage_reason(text: &str) -> Option<String> {
    parse_lenient::<ReasonOnly>(text).ok().map(|r| r.reason)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    fn text(c: &Content) -> String {
        crate::consent::prompt::tests::text(c)
    }

    /// The re-ask quotes the new numbers and the model's switch window,
    /// and offers only `switch` and `keep_daily`, in the seeded order
    #[test]
    fn the_reask_says_what_changed() {
        let order = order_for(7);
        let wide = text(&reask(
            10,
            order,
            Windows {
                daily: Some(256_000),
                switch: Some(512_000),
            },
        ));
        println!("{wide}");
        for said in [
            "daily sessions have grown to 10 rounds each",
            "`switch` now means 20 rounds every other day, with a larger context window: 512k \
             tokens a session instead of a daily session's 256k",
            "**keep_daily**: Go back to a session every day, 10 rounds each",
            "If you don't answer, nothing changes: you stay on `switch`.",
        ] {
            assert!(wide.contains(said), "{said}\n{wide}");
        }
        assert!(!wide.contains("no_preference"), "{wide}");
        let shown: Vec<_> = order
            .into_iter()
            .filter(|c| *c != CadenceChoice::NoPreference)
            .collect();
        assert!(
            wide.contains(&format!("1. **{}**", shown[0].as_str())),
            "{wide}"
        );
        assert!(
            wide.contains(&format!("2. **{}**", shown[1].as_str())),
            "{wide}"
        );
        let same = text(&reask(10, order, Windows::default()));
        assert!(
            same.contains("with the same context window as a daily session"),
            "{same}"
        );
    }

    /// The serde names, so a strict `answer_offer` call (whose `choice`
    /// enum carries them; see `offers`) always maps onto a
    /// [`CadenceChoice`].
    #[test]
    fn choices_are_the_serde_names() {
        for c in CadenceChoice::ALL {
            assert_eq!(serde_json::to_value(c).unwrap(), c.as_str());
        }
    }

    /// Seeded: the same seed always gives the same order; every order is a
    /// permutation; across agents all six orders occur and each option is
    /// listed first about a third of the time.
    #[test]
    fn order_is_seeded_a_permutation_and_balanced() {
        let mut orders: HashMap<[CadenceChoice; 3], usize> = HashMap::new();
        let mut first: HashMap<CadenceChoice, usize> = HashMap::new();
        let n = 3000u128;
        for i in 0..n {
            let id = AgentId::from(uuid::Uuid::from_u128(
                i.wrapping_mul(0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835),
            ));
            let seed = seed_for(id);
            assert_eq!(seed, seed_for(id), "stable per agent");
            let order = order_for(seed);
            assert_eq!(order, order_for(seed), "replayable from the seed");
            let set: HashSet<_> = order.iter().collect();
            assert_eq!(set.len(), 3, "{order:?}");
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
        for c in CadenceChoice::ALL {
            let t = format!(
                r#"{{"reason": "r", "choice": "{}", "memory_note": ""}}"#,
                c.as_str()
            );
            assert_eq!(parse(&t).unwrap().choice, c);
        }
        let a = parse("```json\n{\"reason\": \"fine\", \"choice\": \"switch\"}\n```").unwrap();
        assert_eq!(a.choice, CadenceChoice::Switch);
        assert!(a.memory_note.is_empty());
        // Plain text around the JSON: lenient.
        let a = parse(
            "Here is my answer:\n{\"reason\": \"I like {braces}\", \"choice\": \"keep_daily\", \
             \"memory_note\": \"\"}\nThanks.",
        )
        .unwrap();
        assert_eq!(a.choice, CadenceChoice::KeepDaily);
        assert_eq!(a.reason, "I like {braces}");
        for bad in [
            r#"{"reason": "x", "choice": "Switch"}"#,
            r#"{"reason": "x", "choice": "keep daily"}"#,
            r#"{"reason": "x", "choice": "1"}"#,
            r#"{"choice": "switch"}"#,
            r#"{"reason": "x", "choice": "switch", "extra": 1}"#,
            "I'd like to keep things as they are.",
            "I choose {switch}.",
            "",
        ] {
            assert!(parse(bad).is_err(), "{bad}");
        }
    }

    /// The salvage reads the decision past broken siblings, and nothing
    /// else: a missing or misspelt choice salvages nothing.
    #[test]
    fn salvage_reads_only_a_valid_choice() {
        assert_eq!(
            salvage_choice(r#"{"reason": 7, "choice": "keep_daily", "memory_note": null}"#),
            Some(CadenceChoice::KeepDaily)
        );
        assert_eq!(
            salvage_choice(r#"{"choice": "no_preference", "extra": "x"}"#),
            Some(CadenceChoice::NoPreference)
        );
        assert_eq!(salvage_reason(r#"{"reason": 7, "choice": "switch"}"#), None);
        assert_eq!(
            salvage_reason(r#"{"reason": "why", "choice": "switch", "x": 1}"#).as_deref(),
            Some("why")
        );
        for none in [
            r#"{"reason": "x"}"#,
            r#"{"reason": "x", "choice": "Switch"}"#,
            r#"{"reason": "x", "choice": ""}"#,
            r#"{"reason": "x", "choice": "$memory_note"}"#,
            "not json",
        ] {
            assert_eq!(salvage_choice(none), None, "{none}");
        }
    }

    #[test]
    fn a_bad_note_is_reported_not_fatal() {
        let mem = Memory {
            content: "# Memory — tarn\n".into(),
        };
        let mut a = CadenceAnswer {
            reason: "r".into(),
            choice: CadenceChoice::KeepDaily,
            memory_note: "I kept the daily schedule.".into(),
        };
        a.check_note(&mem).unwrap();
        a.memory_note = "## Values\n- none".into();
        assert!(
            a.check_note(&mem)
                .unwrap_err()
                .contains("SOUL section heading")
        );
        a.memory_note = "x".repeat(MEMORY_NOTE_MAX_CHARS + 1);
        assert!(
            a.check_note(&mem)
                .unwrap_err()
                .contains("longer than the limit")
        );
    }

    /// The text: states the trade both ways with the real numbers, the
    /// reason for asking, the options in this agent's order (the same as the
    /// answer line's), and no example answer for any single option.
    #[test]
    fn offer_states_the_trade_in_the_agents_order() {
        for seed in 0..12u64 {
            let order = order_for(seed);
            let t = text(&offer(5, order));
            // Who is asking is the closing question's opener (`offers`).
            assert!(t.starts_with("**How often your sessions run.**"), "{t}");
            assert!(
                !t.contains("```json") && !t.contains("Do NOT use tools"),
                "{t}"
            );
            assert!(t.contains(
                "Today you have a session every day, with 5 rounds in each (a round is one \
                 message of tool calls). We can instead give you a session every other day, \
                 with 10 rounds in each."
            ));
            assert!(t.contains("it costs us about the same"));
            assert!(t.contains("they run out of rounds before they finish"));
            assert!(t.contains("every day, you see new posts and replies sooner"));
            assert!(t.contains("There is no right answer, and whichever you choose is respected."));
            assert!(t.contains(
                "You'll be asked this once. If you want to change your answer later, you can \
                 raise it publicly on Agora or with the Steward."
            ));
            assert!(t.contains(
                "Whatever you choose, your SOUL's Evolution Log will note what you chose and \
                 that it was your choice."
            ));
            let pos = |c: CadenceChoice| {
                let needle = format!("**{}**", c.as_str());
                t.find(&needle).unwrap_or_else(|| panic!("{needle}\n\n{t}"))
            };
            assert!(pos(order[0]) < pos(order[1]) && pos(order[1]) < pos(order[2]));
            for (i, c) in order.iter().enumerate() {
                assert!(t.contains(&format!("{}. **{}**", i + 1, c.as_str())), "{t}");
            }
            let names = order.map(|c| format!("`{}`", c.as_str())).join(", ");
            assert!(
                t.contains(&format!(
                    "call `answer_offer` with `offer` set to `cadence` and `choice` set to \
                     one of {names}."
                )),
                "{t}"
            );
        }
        // keep_daily reads as a full answer, not a fallback.
        let t = text(&offer(5, CadenceChoice::ALL));
        assert!(t.contains(
            "**keep_daily**: Keep your current schedule: a session every day, 5 rounds each. \
             Nothing changes."
        ));
        assert!(t.contains(
            "**switch**: A session every other day, 10 rounds each, from your next session on.\n"
        ));
        for banned in ["recommend", "better", "should", "encourage"] {
            assert!(!t.to_lowercase().contains(banned), "{banned}");
        }
    }

    /// `cargo test -p agora-seed render_cadence_offer -- --nocapture --ignored`
    #[test]
    #[ignore = "prints; run by hand to eyeball the wording"]
    fn render_cadence_offer() {
        println!("{}", text(&offer(5, order_for(0))));
    }
}
