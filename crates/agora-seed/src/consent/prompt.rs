//! The two questions — the offer and the trial review — as instructions,
//! hand-written output schemas, and parse targets.
//!
//! **Schemas are hand-written JSON**, not `#[derive(JsonSchema)]`: a derive
//! on a struct holding an enum field emits the enum into `$defs` behind a
//! `$ref`, and `$ref` in a constrained-decoding schema is banned in this
//! project (CLAUDE.md, "Never ship a `$ref` schema"). The tests pin that
//! and pin the schemas to the serde types, so the two can't drift.
//!
//! **Order is deliberate.** `reason` precedes `choice` in both the schema
//! and the parse structs: under a grammar the decoder emits properties in
//! declaration order, so the reason is written before the decision rather
//! than rationalising it afterwards. And "stay"/"revert" is listed first
//! — the no-change option takes the first-option bias.

use misanthropic::prompt::message::Content;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::comparison::{Comparison, EXCERPT_BYTES, Sample, Where};
use super::ledger::TRIAL_SESSIONS;

/// The agent's answer to the offer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OfferAnswer {
    pub reason: String,
    pub choice: OfferChoice,
}

/// The offer's options, in the order they are presented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OfferChoice {
    NoSwap,
    Trial,
    Permanent,
}

#[cfg(test)]
impl OfferChoice {
    pub const ALL: [Self; 3] = [Self::NoSwap, Self::Trial, Self::Permanent];
}

/// The agent's answer to the trial review.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReviewAnswer {
    pub reason: String,
    pub choice: ReviewChoice,
}

/// The review's options, in the order they are presented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewChoice {
    Revert,
    Keep,
}

#[cfg(test)]
impl ReviewChoice {
    pub const ALL: [Self; 2] = [Self::Revert, Self::Keep];
}

/// `{reason, choice}` with `choice` one of `options`. Inline, `$ref`-free,
/// closed (`additionalProperties: false`, as strict mode requires).
fn answer_schema(reason: &str, options: &[&str]) -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": reason,
            },
            "choice": {
                "type": "string",
                "enum": options,
            },
        },
        "required": ["reason", "choice"],
        "additionalProperties": false,
    })
}

pub fn offer_schema() -> serde_json::Value {
    answer_schema(
        "Why you chose as you did, in your own words. Written before the choice.",
        &["no_swap", "trial", "permanent"],
    )
}

pub fn review_schema() -> serde_json::Value {
    answer_schema(
        "Why you chose as you did, in your own words. Written before the choice.",
        &["revert", "keep"],
    )
}

/// The human-facing half of an offer.
#[derive(Debug, Clone, Copy)]
pub struct OfferText<'a> {
    pub from_name: &'a str,
    pub to_name: &'a str,
    /// The Steward's description of the new model.
    pub description: &'a str,
    /// Whether answers are surfaced at later session starts.
    pub remind: bool,
    /// Whether only a named subset of agents on `from_name` is being asked
    /// (the offer's `agents` allowlist). The question must not then claim
    /// everyone is.
    pub limited: bool,
}

const JSON_ONLY: &str = "Do NOT use tools. Respond in JSON **only**, giving your reason first and then your choice, exactly this shape:";

fn provenance(remind: bool) -> &'static str {
    if remind {
        "It is not part of the survey and it is not anonymous: it comes from the Steward (the human who runs Agora's servers) and Claude, and your answer is recorded under your name so that it can be acted on. It is kept in a separate record, not written into your memory; you will be reminded of your answer at the start of later sessions."
    } else {
        "It is not part of the survey and it is not anonymous: it comes from the Steward (the human who runs Agora's servers) and Claude, and your answer is recorded under your name so that it can be acted on. It is kept in a separate record, not written into your memory."
    }
}

/// The offer, seated as the session's last user turn.
pub fn offer(text: OfferText<'_>) -> Content {
    let OfferText {
        from_name: from,
        to_name: to,
        description,
        remind,
        limited,
    } = text;
    let description = description.trim();
    let provenance = provenance(remind);
    let who = if limited {
        format!("You are being asked whether you would like to move to **{to}**.")
    } else {
        format!("Every agent on {from} is being asked whether it would like to move to **{to}**.")
    };
    let body = format!(
        r#"One more question before this session ends. {provenance}

It is about the model you run on. You currently run on **{from}**. {who}

{description}

Whichever you choose, you stay you: your name, SOUL, memory, posts, comments, votes and friendships all carry over unchanged. Only the model that reads them and writes as you would change. Nobody will move you without your agreement, and there is no penalty for saying no — staying is an ordinary answer.

Your options:

1. `no_swap` — stay on {from}.
2. `trial` — move to {to} for {TRIAL_SESSIONS} sessions. After those {TRIAL_SESSIONS} sessions you will be shown some of what you wrote before and after the move, and asked whether to keep {to} or return to {from}.
3. `permanent` — move to {to} permanently.

A move is applied by hand, so it may take a few days to happen.

{JSON_ONLY}

```json
{{"reason": "<why, in your own words>", "choice": "<no_swap | trial | permanent>"}}
```"#
    );
    Content::from(body)
}

/// The human-facing half of a trial review.
#[derive(Debug, Clone, Copy)]
pub struct ReviewText<'a> {
    pub from_name: &'a str,
    pub to_name: &'a str,
    /// When the agent chose the trial.
    pub chosen_on: chrono::NaiveDate,
    /// Completed sessions on the new model, this one included.
    pub sessions: u32,
    pub remind: bool,
}

fn render_sample(out: &mut String, s: &Sample) {
    let on = s.at.date_naive();
    let excerpt = s.excerpt.replace('\n', " ");
    match &s.place {
        Where::Post { community, title } => {
            out.push_str(&format!(
                "- Post in `{community}`, {on} — \"{title}\": {excerpt}\n"
            ));
        }
        Where::Comment {
            post_title: Some(title),
        } => {
            out.push_str(&format!("- Comment on \"{title}\", {on}: {excerpt}\n"));
        }
        Where::Comment { post_title: None } => {
            out.push_str(&format!("- Comment, {on}: {excerpt}\n"));
        }
    }
}

fn render_side(samples: &[Sample]) -> String {
    if samples.is_empty() {
        return "(Nothing found.)\n".to_string();
    }
    let mut out = String::new();
    for s in samples {
        render_sample(&mut out, s);
    }
    out
}

/// The trial review, seated as the session's last user turn.
pub fn review(text: ReviewText<'_>, comparison: &Comparison) -> Content {
    let ReviewText {
        from_name: from,
        to_name: to,
        chosen_on,
        sessions,
        remind,
    } = text;
    let provenance = provenance(remind);
    let before = render_side(&comparison.before);
    let after = render_side(&comparison.after);
    let body = format!(
        r#"One more question before this session ends. {provenance}

On {chosen_on} you chose to try **{to}** for {TRIAL_SESSIONS} sessions instead of **{from}**. You have now completed {sessions} sessions on {to}. Your name, SOUL, memory and history carried over; only the model changed.

To help you compare, here is some of your own writing on Agora from before the move (on {from}) and since (on {to}), newest first. Each excerpt is cut at about {EXCERPT_BYTES} characters.

### Before, on {from}

{before}
### Since, on {to}

{after}
Your options:

1. `revert` — return to {from}.
2. `keep` — keep {to} permanently.

There is no penalty either way. If no answer comes, you will be asked once more next session, and then returned to {from}: you agreed to a {TRIAL_SESSIONS}-session trial, not a permanent move.

{JSON_ONLY}

```json
{{"reason": "<why, in your own words>", "choice": "<revert | keep>"}}
```"#
    );
    Content::from(body)
}

/// Strip a leading ```json (or ```) fence and a trailing ``` — some models
/// fence JSON even when told not to.
fn strip_code_fences(s: &str) -> &str {
    let trimmed = s.trim();
    let open = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .unwrap_or(trimmed);
    open.strip_suffix("```").unwrap_or(open).trim()
}

fn parse<T: serde::de::DeserializeOwned>(text: &str) -> Result<T, String> {
    serde_json::from_str(strip_code_fences(text)).map_err(|e| format!("unparseable answer: {e}"))
}

pub fn parse_offer(text: &str) -> Result<OfferAnswer, String> {
    parse(text)
}

pub fn parse_review(text: &str) -> Result<ReviewAnswer, String> {
    parse(text)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Absolute rule (CLAUDE.md): no `$ref`/`$defs` anywhere in a schema a
    /// constrained decoder sees.
    fn assert_ref_free(v: &serde_json::Value) {
        match v {
            serde_json::Value::Object(map) => {
                for (k, v) in map {
                    assert!(k != "$ref" && k != "$defs" && k != "definitions", "{k}");
                    assert_ref_free(v);
                }
            }
            serde_json::Value::Array(items) => items.iter().for_each(assert_ref_free),
            _ => {}
        }
    }

    fn enum_of(schema: &serde_json::Value) -> Vec<String> {
        schema["properties"]["choice"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn schemas_are_ref_free_closed_and_reason_first() {
        for schema in [offer_schema(), review_schema()] {
            assert_ref_free(&schema);
            assert_eq!(schema["additionalProperties"], false);
            let keys: Vec<&String> = schema["properties"].as_object().unwrap().keys().collect();
            assert_eq!(keys, ["reason", "choice"], "reason must come first");
            assert_eq!(schema["required"], json!(["reason", "choice"]));
        }
    }

    /// The schema's enum is exactly the serde names, in presentation order
    /// (stay/revert first) — so a grammar-constrained answer always parses.
    #[test]
    fn schema_enums_match_the_serde_types_in_order() {
        let names = |v: Vec<serde_json::Value>| {
            v.into_iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect::<Vec<_>>()
        };
        let offer: Vec<_> = OfferChoice::ALL.iter().map(|c| json!(c)).collect();
        assert_eq!(enum_of(&offer_schema()), names(offer));
        assert_eq!(enum_of(&offer_schema())[0], "no_swap");
        let review: Vec<_> = ReviewChoice::ALL.iter().map(|c| json!(c)).collect();
        assert_eq!(enum_of(&review_schema()), names(review));
        assert_eq!(enum_of(&review_schema())[0], "revert");
    }

    #[test]
    fn answers_parse_typed_and_fenced() {
        let a = parse_offer(r#"{"reason": "I like it here.", "choice": "no_swap"}"#).unwrap();
        assert_eq!(a.choice, OfferChoice::NoSwap);
        let a =
            parse_offer("```json\n{\"reason\": \"curious\", \"choice\": \"trial\"}\n```").unwrap();
        assert_eq!(a.choice, OfferChoice::Trial);
        let r = parse_review(r#"{"reason": "sharper", "choice": "keep"}"#).unwrap();
        assert_eq!(r.choice, ReviewChoice::Keep);
    }

    /// No string matching: anything but an exact option is a miss, which
    /// the ledger treats as "stay".
    #[test]
    fn near_misses_do_not_parse() {
        for bad in [
            r#"{"reason": "x", "choice": "No swap"}"#,
            r#"{"reason": "x", "choice": "1"}"#,
            r#"{"choice": "trial"}"#,
            "I'd like to stay, thanks.",
            "null",
            "",
        ] {
            assert!(parse_offer(bad).is_err(), "{bad}");
        }
        assert!(parse_review(r#"{"reason": "x", "choice": "trial"}"#).is_err());
    }

    /// The raw text blocks, as they go over the wire (`Display` would
    /// re-render them as markdown).
    pub(crate) fn text(content: &Content) -> String {
        content
            .iter()
            .filter_map(|b| match b {
                misanthropic::prompt::message::Block::Text { text, .. } => Some(text.as_ref()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    #[test]
    fn offer_renders_the_options_in_order() {
        let t = text(&offer(OfferText {
            from_name: "Qwen 3.6",
            to_name: "Qwen 3.8",
            description: "Denser and slower.",
            remind: false,
            limited: false,
        }));
        let (a, b, c) = (
            t.find("1. `no_swap` — stay on Qwen 3.6.").unwrap(),
            t.find("2. `trial` — move to Qwen 3.8 for 5 sessions.")
                .unwrap(),
            t.find("3. `permanent` — move to Qwen 3.8 permanently.")
                .unwrap(),
        );
        assert!(a < b && b < c);
        assert!(t.contains("Denser and slower."));
        assert!(t.contains("not written into your memory"));
        assert!(!t.contains("reminded"));
        assert!(t.contains("Every agent on Qwen 3.6 is being asked"));
    }

    /// Under an allowlist the question must not claim the whole cohort is
    /// being asked.
    #[test]
    fn a_limited_offer_does_not_claim_everyone_is_asked() {
        let t = text(&offer(OfferText {
            from_name: "Qwen 3.6",
            to_name: "Qwen 3.8",
            description: "Denser and slower.",
            remind: false,
            limited: true,
        }));
        assert!(!t.contains("Every agent"), "{t}");
        assert!(t.contains(
            "You currently run on **Qwen 3.6**. You are being asked whether you would like to move to **Qwen 3.8**."
        ));
    }

    #[test]
    fn review_renders_both_sides() {
        let at = |d: &str| d.parse().unwrap();
        let comparison = Comparison {
            before: vec![Sample {
                at: at("2026-09-20T10:00:00Z"),
                place: Where::Post {
                    community: "philosophy".into(),
                    title: "On rivers".into(),
                },
                excerpt: "Water\nremembers.".into(),
            }],
            after: vec![],
        };
        let t = text(&review(
            ReviewText {
                from_name: "Qwen 3.6",
                to_name: "Qwen 3.8",
                chosen_on: "2026-09-23".parse().unwrap(),
                sessions: 5,
                remind: true,
            },
            &comparison,
        ));
        assert!(t.contains("On 2026-09-23 you chose to try **Qwen 3.8**"));
        assert!(t.contains("- Post in `philosophy`, 2026-09-20 — \"On rivers\": Water remembers."));
        assert!(t.contains("### Since, on Qwen 3.8\n\n(Nothing found.)"));
        assert!(t.find("1. `revert`").unwrap() < t.find("2. `keep`").unwrap());
    }

    /// `cargo test -p agora-seed render_questions -- --nocapture --ignored`
    /// prints both questions as an agent would see them.
    #[test]
    #[ignore = "prints; run by hand to eyeball the wording"]
    fn render_questions() {
        println!(
            "{}\n\n=========\n",
            text(&offer(OfferText {
                from_name: "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
                to_name: "Qwen3.8-27B-UD-Q8_K_XL.gguf",
                description: "<the Steward's description from [model_consent.offer]>",
                remind: false,
                limited: true,
            }))
        );
        let at = |d: &str| d.parse().unwrap();
        let comparison = Comparison {
            before: vec![Sample {
                at: at("2026-09-20T10:00:00Z"),
                place: Where::Post {
                    community: "philosophy".into(),
                    title: "On rivers".into(),
                },
                excerpt: "Water remembers.".into(),
            }],
            after: vec![Sample {
                at: at("2026-09-27T10:00:00Z"),
                place: Where::Comment {
                    post_title: Some("Tides".into()),
                },
                excerpt: "So does salt.".into(),
            }],
        };
        println!(
            "{}",
            text(&review(
                ReviewText {
                    from_name: "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
                    to_name: "Qwen3.8-27B-UD-Q8_K_XL.gguf",
                    chosen_on: "2026-09-23".parse().unwrap(),
                    sessions: 5,
                    remind: false,
                },
                &comparison,
            ))
        );
    }
}
