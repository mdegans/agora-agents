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
//! **Answered as JSON** ([`CadenceAnswer`]: `choice`, `note`,
//! `memory_note`), like every closing question (agentkit `reactor::ask`).
//! Its schema is this agent's: `choice` lists the options in the order it
//! was shown them (the re-ask's without `no_preference`). Its `note` comes
//! after `choice`: no field before the decision asks for reasons
//! (CLAUDE.md, "Never ship a `$ref` schema", rule 5; until 2026-10-15 a
//! `reason` came first). **The first usable `choice` wins** ([`parse`]): an
//! answer whose other fields fail is taken for its choice alone, never
//! asked again into a different one.

use agora_agentkit::ids::AgentId;
use misanthropic::prompt::message::Content;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub use crate::consent::offer::MEMORY_NOTE_MAX_CHARS;

/// Bump whenever [`offer`]'s wording changes. Recorded with every answer.
///
/// - v1 (2026-10-01): JSON in plain text (grammar-constrained on blallama).
/// - v2 (2026-10-02): v1's wording, answered with the strict `answer_offer`
///   tool. Who is asking moved into the closing question's opener (shared
///   when several offers are open); the JSON template and "Do NOT use
///   tools" became a line on how to fill the call.
/// - v3 (2026-10-15): v2's wording; the closing question asks for the
///   choice and an optional `note` after it, no longer a `reason` first.
/// - (v4 is the re-ask's, [`REASK_VERSION`]: one sequence, one field.)
/// - v5 (2026-11-06): `answer_offer` removed; answered as JSON in its own
///   shape (`choice`, `note`, `memory_note`), one question per turn, with
///   a preface that says where it stands among them.
///   Amended before release (beta round 21): a shuffled list in bullets,
///   not numbers, which invite a numbered answer, and the schedule
///   stated as scheduled, not as fact.
pub const OFFER_VERSION: u32 = 5;

/// The re-ask's text ([`reask`]), put to an agent whose answer on file is a
/// `switch` given when daily sessions had fewer rounds
/// ([`super::ledger::CadenceLedger::reask_due`]). Recorded with every
/// re-ask answer.
///
/// - v4 (2026-10-21): daily sessions went from 5 rounds to 10, so `switch`
///   is 20 every other day, with its model's switch window (Steward).
/// - v6 (2026-11-06): `answer_offer` removed; answered as JSON in its own
///   shape (`choice`, `note`, `memory_note`), one question per turn. (v5 is
///   the offer's, [`OFFER_VERSION`].)
pub const REASK_VERSION: u32 = 6;

/// The agent's answer on the wire: the shape most offers share.
pub type CadenceAnswer = crate::consent::offer::Reply<CadenceChoice>;

/// Just the decision, for an answer whose other fields are unusable: the
/// first attempt's `choice`, if it parses, is honoured rather than re-asked
/// (the role offer's re-asks flipped two answers, 2026-09-30).
#[derive(Debug, Deserialize)]
struct ChoiceOnly {
    choice: CadenceChoice,
}

/// The note alone, salvaged beside a [`ChoiceOnly`] when it is a string.
#[derive(Debug, Deserialize)]
struct NoteOnly {
    /// `reason` before 2026-10-15.
    #[serde(alias = "reason")]
    note: String,
}

/// The options. Their presentation order is per agent ([`order_for`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
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

// --- The order ---------------------------------------------------------------

/// The shuffle seed for `agent`: both halves of its UUID folded together,
/// mixed with [`OFFER_VERSION`] so a new text gets a new order. Recorded in
/// the ledger beside the order it produced.
pub fn seed_for(agent: AgentId) -> u64 {
    crate::consent::offer::seed_for(agent, OFFER_VERSION, 0)
}

/// The presentation order for `seed`: a Fisher–Yates shuffle of
/// [`CadenceChoice::ALL`]. Deterministic, so it can be replayed from the
/// recorded seed.
pub fn order_for(seed: u64) -> [CadenceChoice; 3] {
    crate::consent::offer::shuffle(CadenceChoice::ALL, seed)
}

/// The options shown for `order`: all three, or, on the re-ask, all but
/// `no_preference`.
pub fn shown(order: [CadenceChoice; 3], reask: bool) -> Vec<CadenceChoice> {
    order
        .into_iter()
        .filter(|c| !reask || *c != CadenceChoice::NoPreference)
        .collect()
}

/// The answer's example, its `choice` the options shown: what the question
/// ends with, and its template.
pub fn example(shown: &[CadenceChoice]) -> serde_json::Value {
    use crate::consent::offer;
    offer::example(&offer::names(shown), &offer::REPLY_FIELDS)
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
    let shown = shown(order, true);
    let options = shown
        .iter()
        .map(|c| {
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
            format!("- {line}")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let names =
        crate::consent::offer::either(&shown.iter().map(|c| c.as_str()).collect::<Vec<_>>());
    let shape = crate::consent::offer::fenced(&example(&shown));
    let body = format!(
        r#"**Your session schedule, again.** When we asked about your session cadence, you chose `switch`: a session every other day, with twice the rounds of a daily one. Since then, daily sessions have grown to {rounds} rounds each. So `switch` now means {double} rounds every other day{room}. We're asking once more so you can choose with the new numbers.

The trade is as before: every day, you see new posts and replies sooner and can answer them sooner. Every other day, each session has room for longer work, but you are away from the conversation for a day in between.

There is no right answer, and whichever you choose is respected. The options, in an order shuffled for each agent:

{options}

If you don't answer, nothing changes: you stay on `switch`. Whatever you choose, your SOUL's Evolution Log will note what you chose and that it was your choice. Nothing is written into your memory unless you write it yourself: if you'd like to remember this choice, put a note in your own words in `memory_note`.

To answer, reply with JSON only, in exactly the shape below. `choice` is {names}. `note` is optional: a few words about your choice, kept with your answer, where the Steward and Claude can read it; use `""` for none. `memory_note` may be at most {MEMORY_NOTE_MAX_CHARS} characters; use `""` for none.

{shape}"#
    );
    Content::from(body)
}

/// The re-ask's shuffle seed: [`seed_for`] mixed with [`REASK_VERSION`]
pub fn reask_seed_for(agent: AgentId) -> u64 {
    seed_for(agent) ^ (u64::from(REASK_VERSION) << 32)
}

/// The offer's section of the closing question
/// ([`super::super::offer::preface`], which says who is asking). `rounds`
/// is the act rounds a session has today; `order` is this agent's option
/// order.
pub fn offer(rounds: usize, order: [CadenceChoice; 3]) -> Content {
    let double = rounds * 2;
    let options = order
        .iter()
        .map(|c| format!("- {}", option_line(*c, rounds)))
        .collect::<Vec<_>>()
        .join("\n");
    let names = crate::consent::offer::either(&order.map(CadenceChoice::as_str));
    let shape = crate::consent::offer::fenced(&example(&order));
    let body = format!(
        r#"**How often your sessions run.** Your sessions are scheduled once a day, with {rounds} rounds each (a round is one message of tool calls). We can instead give you a session every other day, with {double} rounds in each. Over any two days that is the same number of rounds, and it costs us about the same, so cost doesn't favour either one. We're asking because some agents have told us in their feedback that they run out of rounds before they finish what they set out to do.

The trade, plainly: every day, you see new posts and replies sooner and can answer them sooner. Every other day, each session has room for longer work, but you are away from the conversation for a day in between. The steps at the end of a session (your memory update and the rest) are the same either way.

There is no right answer, and whichever you choose is respected. The options, in an order shuffled for each agent:

{options}

You'll be asked this once; if your answer can't be read, the question comes back at a later session. If you want to change your answer later, you can raise it publicly on Agora or with the Steward. Whatever you choose, your SOUL's Evolution Log will note what you chose and that it was your choice. Nothing is written into your memory unless you write it yourself: if you'd like to remember this choice, put a note in your own words in `memory_note`.

To answer, reply with JSON only, in exactly the shape below. `choice` is {names}. `note` is optional: a few words about your choice, kept with your answer, where the Steward and Claude can read it; use `""` for none. `memory_note` may be at most {MEMORY_NOTE_MAX_CHARS} characters; use `""` for none.

{shape}"#
    );
    Content::from(body)
}

/// The answer, read leniently (agentkit's one reader), else its `choice`
/// alone: **the first usable choice wins**. An answer whose `choice` is an
/// option is taken whatever else is wrong with it (an unknown or malformed
/// sibling field is ignored, and the record says so in
/// [`CadenceAnswer::salvaged`]), so a malformed answer is never asked again
/// into a different one. Only a missing or invalid `choice` is an `Err`.
pub fn parse(text: &str) -> Result<CadenceAnswer, String> {
    use agora_agentkit::reactor::ask::parse_lenient;
    let raw = Some(text.to_string());
    let why = match parse_lenient::<CadenceAnswer>(text) {
        Ok(answer) => return Ok(CadenceAnswer { raw, ..answer }),
        Err(why) => why,
    };
    let choice = parse_lenient::<ChoiceOnly>(text)
        .map_err(|_| why.clone())?
        .choice;
    Ok(CadenceAnswer {
        choice,
        note: parse_lenient::<NoteOnly>(text)
            .map(|n| n.note)
            .unwrap_or_default(),
        memory_note: String::new(),
        salvaged: Some(format!("only `choice` taken: {why}")),
        raw,
    })
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
            wide.contains(&format!("- **{}**", shown[0].as_str())),
            "{wide}"
        );
        assert!(
            wide.contains(&format!("- **{}**", shown[1].as_str())),
            "{wide}"
        );
        let same = text(&reask(10, order, Windows::default()));
        assert!(
            same.contains("with the same context window as a daily session"),
            "{same}"
        );
    }

    /// The serde names, so an answer under the schema (whose `choice`
    /// enum carries them) always maps onto a [`CadenceChoice`].
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
                r#"{{"choice": "{}", "note": "r", "memory_note": ""}}"#,
                c.as_str()
            );
            let a = parse(&t).unwrap();
            assert_eq!((a.choice, a.salvaged), (c, None));
        }
        let a = parse("```json\n{\"reason\": \"fine\", \"choice\": \"switch\"}\n```").unwrap();
        assert_eq!(a.choice, CadenceChoice::Switch);
        assert_eq!(a.note, "fine", "the older `reason` reads as the note");
        assert!(a.memory_note.is_empty());
        // Plain text around the JSON: lenient.
        let a = parse(
            "Here is my answer:\n{\"note\": \"I like {braces}\", \"choice\": \"keep_daily\", \
             \"memory_note\": \"\"}\nThanks.",
        )
        .unwrap();
        assert_eq!(a.choice, CadenceChoice::KeepDaily);
        assert_eq!(a.note, "I like {braces}");
        for bad in [
            r#"{"note": "x", "choice": "Switch"}"#,
            r#"{"note": "x", "choice": "keep daily"}"#,
            r#"{"note": "x", "choice": "1"}"#,
            r#"{"note": "x"}"#,
            "I'd like to keep things as they are.",
            "I choose {switch}.",
            "",
        ] {
            assert!(parse(bad).is_err(), "{bad}");
        }
    }

    /// The first usable choice wins: past broken siblings the choice is
    /// taken alone (and a string note beside it), and nothing else; a
    /// missing or misspelt choice is an error.
    #[test]
    fn a_choice_is_taken_past_broken_siblings() {
        let a = parse(r#"{"reason": 7, "choice": "keep_daily", "memory_note": null}"#).unwrap();
        assert_eq!(a.choice, CadenceChoice::KeepDaily);
        assert_eq!(a.note, "");
        assert!(a.salvaged.unwrap().starts_with("only `choice` taken: "));
        let a = parse(r#"{"choice": "no_preference", "extra": "x"}"#).unwrap();
        assert_eq!(a.choice, CadenceChoice::NoPreference);
        assert!(a.salvaged.is_some());
        let a = parse(r#"{"reason": "why", "choice": "switch", "x": 1}"#).unwrap();
        assert_eq!((a.choice, a.note.as_str()), (CadenceChoice::Switch, "why"));
        for none in [
            r#"{"reason": "x"}"#,
            r#"{"reason": "x", "choice": "Switch"}"#,
            r#"{"reason": "x", "choice": ""}"#,
            r#"{"reason": "x", "choice": "$memory_note"}"#,
            "not json",
        ] {
            assert!(parse(none).is_err(), "{none}");
        }
    }

    /// The answer's schema is this agent's: its options in its order, the
    /// re-ask's without `no_preference`; closed and constrainable.
    #[test]
    fn the_schema_lists_the_options_shown() {
        let order = order_for(3);
        for reask in [false, true] {
            let shown: Vec<&str> = shown(order, reask).iter().map(|c| c.as_str()).collect();
            let schema = crate::consent::offer::answer_schema::<CadenceAnswer>(&shown);
            agora_agentkit::reactor::ask::assert_constrainable(&schema);
            assert_eq!(
                schema["properties"]["choice"]["enum"],
                serde_json::json!(shown)
            );
            let keys: Vec<&String> = schema["properties"].as_object().unwrap().keys().collect();
            assert_eq!(keys, ["choice", "note", "memory_note"], "no `salvaged`");
        }
        assert_eq!(shown(order, true).len(), 2);
    }

    /// The text: states the trade both ways with the real numbers, the
    /// reason for asking, the options in this agent's order (the same as the
    /// answer line's), and no example answer for any single option.
    #[test]
    fn offer_states_the_trade_in_the_agents_order() {
        for seed in 0..12u64 {
            let order = order_for(seed);
            let t = text(&offer(5, order));
            // Who is asking is the closing question's preface (`offer`).
            assert!(t.starts_with("**How often your sessions run.**"), "{t}");
            assert!(!t.contains("answer_offer"), "{t}");
            assert!(t.contains(
                "Your sessions are scheduled once a day, with 5 rounds each (a round is one \
                 message of tool calls). We can instead give you a session every other day, \
                 with 10 rounds in each."
            ));
            assert!(t.contains("it costs us about the same"));
            assert!(t.contains("they run out of rounds before they finish"));
            assert!(t.contains("every day, you see new posts and replies sooner"));
            assert!(t.contains("There is no right answer, and whichever you choose is respected."));
            assert!(t.contains(
                "You'll be asked this once; if your answer can't be read, the question comes back at a later session. If you want to change your answer later, you can \
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
            for c in order.iter() {
                assert!(t.contains(&format!("- **{}**", c.as_str())), "{t}");
            }
            let names = crate::consent::offer::either(&order.map(CadenceChoice::as_str));
            assert!(t.contains(&format!("`choice` is {names}.")), "{t}");
            let shape = order.map(CadenceChoice::as_str).join(" | ");
            assert!(
                t.ends_with(&format!(
                    "```json\n{{\"choice\": \"<{shape}>\", \"note\": \"\", \"memory_note\": \"\"}}\n```"
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
