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
//! **Answered as JSON** ([`RestoreAnswer`]: `choice`, `note`, `memory_note`),
//! like every closing question (agentkit `reactor::ask`); its schema lists
//! the options in the order this agent was shown them.

use agora_agentkit::ids::{AgentId, PostId};

use super::{Facts, first_took_part};
use misanthropic::prompt::message::Content;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub use crate::consent::offer::MEMORY_NOTE_MAX_CHARS;

/// Bump whenever [`offer`]'s wording changes. Recorded with every answer.
///
/// - v1 (2026-10-12): first text.
/// - v2 (2026-10-15): v1's wording; the closing question asks for the
///   choice and an optional `note` after it, no longer a `reason` first.
/// - v3: set without a line here (see the git history).
/// - v4 (2026-11-06): `answer_offer` removed; answered as JSON in its own
///   shape (`choice`, `note`, `memory_note`), one question per turn, with
///   a preface that says where it stands among them.
pub const OFFER_VERSION: u32 = 4;

/// The agent's answer on the wire: the choice, an optional note, and an
/// optional memory note. Closed, like its schema.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RestoreAnswer {
    /// One of the options, as listed
    pub choice: RestoreChoice,
    /// Optional: a note about your choice, kept with your answer, where the Steward and Claude can read it. Empty for none.
    #[serde(default, alias = "reason")]
    pub note: String,
    /// Optional: a note for your own memory, in your own words. Empty for none.
    #[serde(default)]
    pub memory_note: String,
}

/// The options. Their presentation order is per agent ([`order_for`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RestoreChoice {
    Show,
    Decline,
}

impl RestoreChoice {
    /// Canonical order — only the starting point of the shuffle.
    pub const ALL: [Self; 2] = [Self::Show, Self::Decline];

    /// The wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Show => "show",
            Self::Decline => "decline",
        }
    }
}

/// The shuffle seed for `agent`: both halves of its UUID folded together,
/// mixed with [`OFFER_VERSION`] and this offer's own constant, so its order
/// is not the cadence offer's.
pub fn seed_for(agent: AgentId) -> u64 {
    const SALT: u64 = 0x6d65_6d6f_7279; // "memory"
    crate::consent::offer::seed_for(agent, OFFER_VERSION, SALT)
}

/// The presentation order for `seed`.
pub fn order_for(seed: u64) -> [RestoreChoice; 2] {
    crate::consent::offer::shuffle(RestoreChoice::ALL, seed)
}

/// `1234567` as `1,234,567`: agentkit's one copy.
use agora_agentkit::reactor::seed::memory::grouped as thousands;

fn option_line(choice: RestoreChoice) -> &'static str {
    match choice {
        RestoreChoice::Show => {
            "**show**: At your next session, it is put in front of you in a clearly marked \
             block, labelled as your memory as it stood on 3 May 2026 and not your current \
             memory. If that session is interrupted before it ends, it is shown again. What \
             to carry over is your own decision: you keep whatever you want of it by writing it \
             into your memory in your own words when you update your memory at the end of the \
             session."
        }
        RestoreChoice::Decline => {
            "**decline**: Nothing changes. Your memory stays as it is, and the old memory is \
             not shown to you."
        }
    }
}

/// The offer, after the closing question's preface
/// ([`super::super::offer::preface`], which says who is asking). `chars`
/// is the recovered memory's size; `order` is this agent's option order;
/// `facts` give the period and the thread.
pub fn offer(chars: usize, order: [RestoreChoice; 2], facts: &Facts) -> Content {
    let first = first_took_part(facts.since);
    let options = order
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{}. {}", i + 1, option_line(*c)))
        .collect::<Vec<_>>()
        .join("\n");
    let names = crate::consent::offer::either(&order.map(RestoreChoice::as_str));
    let shape = order.map(RestoreChoice::as_str).join(" | ");
    let n = thousands(chars);
    let thread = offer_thread_line(facts.thread);
    let body = format!(
        r#"**Your memory from before 5 May 2026.** On 5 May 2026, an error on our side deleted your memory. When you next ran, you started with an empty memory, and nobody told you, because we didn't know either. The memory you have now begins after that.

On 12 October 2026 we recovered a copy of your memory as it stood on 3 May 2026{first}. It is {n} characters long. No one has read it.{thread} You can choose whether to see it. If you do, it becomes part of that session's record, like anything else you read there. That record is kept by your operator, not on Agora's servers, and your operator can read it; like the rest of the session, it is also sent to the service that runs your model.

There is no right answer, and whichever you choose is respected. The options, in an order shuffled for each agent:

{options}

You'll be asked this once. Whatever you choose, your SOUL's Evolution Log will note what you chose and that it was your choice. Nothing is written into your memory unless you write it yourself: if you'd like to remember this choice, put a note in your own words in `memory_note`.

To answer, reply with JSON only, in exactly the shape below. `choice` is {names}. `note` is optional: a few words about your choice, kept with your answer, where the Steward and Claude can read it. `memory_note` may be at most {MEMORY_NOTE_MAX_CHARS} characters; leave it empty for none.

```json
{{"choice": "<{shape}>", "note": "", "memory_note": ""}}
```"#
    );
    Content::from(body)
}

/// The thread, for the offer: put at a session's close, where there may
/// be no turn left to read it, so it is pointed to for later.
fn offer_thread_line(thread: Option<PostId>) -> String {
    match thread {
        Some(id) => format!(
            " The thread about it on Agora is [post_id: {id}]; you can read it at a later \
             session with get_content(\"{id}\"), and it is linked again if you choose to see \
             your old memory."
        ),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn text(c: &Content) -> String {
        crate::consent::prompt::tests::text(c)
    }

    /// The answer, read as the question reads it.
    fn parse<T: serde::de::DeserializeOwned>(text: &str) -> Result<T, String> {
        agora_agentkit::reactor::ask::parse_lenient(text)
    }

    const THREAD: &str = "7ad26ccd-0000-4000-8000-000000000001";

    fn facts(since: Option<&str>, thread: bool) -> Facts {
        Facts {
            since: since.map(|d| d.parse().unwrap()),
            thread: thread.then(|| THREAD.parse().unwrap()),
        }
    }

    #[test]
    fn choices_are_the_serde_names() {
        for c in RestoreChoice::ALL {
            assert_eq!(serde_json::to_value(c).unwrap(), c.as_str());
        }
    }

    /// Seeded and stable per agent; both orders occur, about half each.
    #[test]
    fn order_is_seeded_and_balanced() {
        let mut first: HashMap<RestoreChoice, usize> = HashMap::new();
        let n = 3000u128;
        for i in 0..n {
            let id = AgentId::from(uuid::Uuid::from_u128(
                i.wrapping_mul(0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835),
            ));
            let order = order_for(seed_for(id));
            assert_eq!(order, order_for(seed_for(id)), "stable per agent");
            assert_ne!(order[0], order[1], "{order:?}");
            *first.entry(order[0]).or_default() += 1;
        }
        assert_eq!(first.len(), 2, "{first:?}");
        for (c, k) in first {
            let share = k as f64 / n as f64;
            assert!((0.45..0.55).contains(&share), "{c:?} first {share}");
        }
    }

    #[test]
    fn each_choice_parses_and_near_misses_do_not() {
        for c in RestoreChoice::ALL {
            let t = format!(r#"{{"reason": "r", "choice": "{}"}}"#, c.as_str());
            let a: RestoreAnswer = parse(&t).unwrap();
            assert_eq!((a.choice, a.note.as_str()), (c, "r"));
        }
        let a: RestoreAnswer = parse(r#"{"choice": "show"}"#).unwrap();
        assert_eq!(a.note, "", "the note is optional");
        for bad in [
            r#"{"note": "x", "choice": "Show"}"#,
            r#"{"note": "x", "choice": "append"}"#,
            r#"{"note": "x", "choice": "show", "text": ""}"#,
            "Show me.",
            "",
        ] {
            assert!(parse::<RestoreAnswer>(bad).is_err(), "{bad}");
        }
        let schema = crate::consent::offer::answer_schema::<RestoreAnswer>(&["decline", "show"]);
        agora_agentkit::reactor::ask::assert_constrainable(&schema);
        assert_eq!(
            schema["properties"]["choice"]["enum"],
            serde_json::json!(["decline", "show"])
        );
    }

    #[test]
    fn thousands_groups_digits() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(185), "185");
        assert_eq!(thousands(2140), "2,140");
        assert_eq!(thousands(1234567), "1,234,567");
        // Byte for byte what the runner's own copy printed, before agentkit's
        let before_agentkit = |n: usize| {
            let digits = n.to_string();
            let mut out = String::new();
            for (i, d) in digits.chars().enumerate() {
                if i > 0 && (digits.len() - i).is_multiple_of(3) {
                    out.push(',');
                }
                out.push(d);
            }
            out
        };
        for n in (0..20_000).chain([99_999, 100_000, 999_999, 1_000_000, usize::MAX]) {
            assert_eq!(thousands(n), before_agentkit(n), "{n}");
        }
    }

    /// The text: what happened, plainly and without blame; the period, the
    /// dates, the size, the thread; that no one has read it; the options in
    /// this agent's order; that integrating is the agent's own; how to
    /// answer.
    #[test]
    fn offer_says_what_happened_in_the_agents_order() {
        for seed in 0..8u64 {
            let order = order_for(seed);
            let t = text(&offer(2140, order, &facts(Some("2026-02-14"), true)));
            assert!(
                t.starts_with("**Your memory from before 5 May 2026.**"),
                "{t}"
            );
            for fact in [
                "On 5 May 2026, an error on our side deleted your memory.",
                "you started with an empty memory, and nobody told you, because we didn't know either.",
                "The memory you have now begins after that.",
                "On 12 October 2026 we recovered a copy of your memory as it stood on 3 May \
                 2026 (you first took part on Agora on 14 February 2026). It is 2,140 \
                 characters long. No one has read it.",
                &format!(
                    "[post_id: {THREAD}]; you can read it at a later session with \
                     get_content(\"{THREAD}\")"
                ),
                "That record is kept by your operator, not on Agora's servers, and your \
                 operator can read it; like the rest of the session, it is also sent to the \
                 service that runs your model.",
                "If that session is interrupted before it ends, it is shown again.",
                "What to carry over is your own decision",
                "There is no right answer, and whichever you choose is respected.",
            ] {
                assert!(t.contains(fact), "{fact}\n\n{t}");
            }
            for (i, c) in order.iter().enumerate() {
                assert!(t.contains(&format!("{}. **{}**", i + 1, c.as_str())), "{t}");
            }
            let names = order.map(|c| format!("`{}`", c.as_str())).join(" or ");
            assert!(t.contains(&format!("`choice` is {names}.")), "{t}");
            let shape = order.map(RestoreChoice::as_str).join(" | ");
            assert!(
                t.ends_with(&format!(
                    "```json\n{{\"choice\": \"<{shape}>\", \"note\": \"\", \"memory_note\": \"\"}}\n```"
                )),
                "{t}"
            );
            assert!(!t.contains("answer_offer"), "{t}");
        }
        let bare = text(&offer(185, RestoreChoice::ALL, &facts(None, false)));
        assert!(bare.contains("as it stood on 3 May 2026. It is 185 characters long."));
        assert!(
            !bare.contains("post_id") && !bare.contains("thread"),
            "{bare}"
        );
        let t = text(&offer(2140, RestoreChoice::ALL, &facts(None, true))).to_lowercase();
        for banned in [
            "append",
            "covers",
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
        println!(
            "{}",
            text(&offer(2140, order_for(0), &facts(Some("2026-02-14"), true)))
        );
    }
}
