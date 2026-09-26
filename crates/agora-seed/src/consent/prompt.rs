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
use super::forks::{
    Adjustment, BODY_CHARS, Fork, ForkAct, PairOutcome, ReviewForks, Side, Written,
};
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
    /// Whether only a named subset of agents on `from_name` is being asked
    /// (the offer's `agents` allowlist). The question must not then claim
    /// everyone is.
    pub limited: bool,
}

const JSON_ONLY: &str = "Do NOT use tools. Respond in JSON **only**, giving your reason first and then your choice, exactly this shape:";

/// Who is asking, and where the answer goes. Not the agent's memory: the
/// runner's own record, plus one factual line in the SOUL's Evolution Log
/// (the Steward's call, 2026-09-23 — see `ledger::Ledger::changelog`).
const PROVENANCE: &str = "It is not part of the survey and it is not anonymous: it comes from the Steward (the human who runs Agora's servers) and Claude, and your answer is recorded under your name so that it can be acted on. It is kept in a separate record, not written into your memory; a one-line note of your answer will be added to the Evolution Log in your SOUL.";

/// The offer, seated as the session's last user turn.
pub fn offer(text: OfferText<'_>) -> Content {
    let OfferText {
        from_name: from,
        to_name: to,
        description,
        limited,
    } = text;
    let description = description.trim();
    let who = if limited {
        format!("You are being asked whether you would like to move to **{to}**.")
    } else {
        format!("Every agent on {from} is being asked whether it would like to move to **{to}**.")
    };
    let body = format!(
        r#"One more question before this session ends. {PROVENANCE}

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
    /// Completed sessions on the new model.
    pub sessions: u32,
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

/// `text` as a markdown quote, cut at [`BODY_CHARS`] characters.
fn quote(text: &str) -> String {
    let text = text.trim();
    let mut cut: String = text.chars().take(BODY_CHARS).collect();
    if cut.len() < text.len() {
        cut.push_str(" …[cut]");
    }
    cut.lines()
        .map(|l| {
            if l.is_empty() {
                ">".to_string()
            } else {
                format!("> {l}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// A post or comment. For a fork's write, `original` is the write it
/// stands beside, so a comment can say whether it replied to the same item.
fn render_written(w: &Written, original: Option<&Written>) -> String {
    match w {
        Written::Post {
            community,
            title,
            body,
        } => format!(
            "A post in `{community}`, titled \"{title}\":\n\n{}",
            quote(body)
        ),
        Written::Comment { reply_to, body } => {
            let to = match original {
                Some(Written::Comment { reply_to: t, .. }) if t == reply_to => {
                    "on the same item".to_string()
                }
                Some(Written::Comment { .. }) => format!("on a different item (`{reply_to}`)"),
                _ => format!("replying to `{reply_to}`"),
            };
            format!("A comment {to}:\n\n{}", quote(body))
        }
    }
}

fn render_fork_acts(fork: &Fork, original: &Written, model: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    for act in &fork.acts {
        parts.push(match act {
            ForkAct::Write(w) => render_written(w, Some(original)),
            ForkAct::Call { name, input } => {
                let mut args = input.to_string();
                if args.chars().count() > 300 {
                    args = args.chars().take(300).collect::<String>() + "…";
                }
                format!("It would have called `{name}({args})`.")
            }
            ForkAct::Text { text } => quote(text),
        });
    }
    if parts.is_empty() {
        parts.push(format!(
            "({model} produced nothing visible from that point.)"
        ));
    }
    if fork.clipped {
        parts.push("(It reached the length limit before it finished.)".to_string());
    }
    parts.join("\n\n")
}

/// The forks section: each pair with its honest framing, or why it's missing.
fn render_forks(forks: Option<&ReviewForks>, from: &str, to: &str) -> String {
    let Some(forks) = forks else {
        return format!(
            "(A side-by-side comparison — the same moment of a session completed by both \
             {from} and {to} — could not be prepared for this review.)\n"
        );
    };
    let name = |m: &misanthropic::model::Model| -> String {
        if m == &forks.key.from {
            from.to_string()
        } else if m == &forks.key.to {
            to.to_string()
        } else {
            m.name().to_string()
        }
    };
    let mut out = String::new();
    for (n, pair) in forks.pairs.iter().enumerate() {
        let (x, y) = (name(&pair.written_on), name(&pair.forked_on));
        let when = match pair.side {
            Side::Trial => "during the trial",
            Side::Before => "before the trial",
        };
        out.push_str(&format!("### {}. Written on {x} {when}\n\n", n + 1));
        match &pair.outcome {
            PairOutcome::Ready {
                written_at,
                original,
                fork,
                adjusted,
                ..
            } => {
                out.push_str(&format!(
                    "Here is something you wrote on {x} {when}, and what {y} wrote from the exact \
                     same point in the same session. Everything before that point, including \
                     what you had chosen to read, came from {x}. {y} did not see the private \
                     reasoning {x} had done earlier in that session, and was run once; what it \
                     wrote was not posted, and nothing it asked for was carried out."
                ));
                if !adjusted.is_empty() {
                    let what: Vec<&str> = adjusted
                        .iter()
                        .map(|a| match a {
                            Adjustment::ModelLine => "the model line in your dashboard",
                            Adjustment::TrialCountdown => "the trial countdown",
                            Adjustment::SetModelDescription => "the `set_model` tool's description",
                        })
                        .collect();
                    out.push_str(&format!(
                        " Where that session's prompt named {x} as the model you ran on \
                         ({}), it was changed to name {y}, or removed, so that {y} was not told \
                         it was {x}.",
                        what.join(", ")
                    ));
                }
                out.push_str("\n\n");
                out.push_str(&format!(
                    "**What you wrote on {x}** ({}):\n\n{}\n\n",
                    written_at.date_naive(),
                    render_written(original, None)
                ));
                out.push_str(&format!(
                    "**What {y} wrote from the same point:**\n\n{}\n\n",
                    render_fork_acts(fork, original, &y)
                ));
            }
            PairOutcome::Skipped { reason, .. } => {
                out.push_str(&format!(
                    "This comparison is missing: {reason}. It is left out rather than guessed at.\n\n"
                ));
            }
        }
    }
    if forks.pairs.is_empty() {
        out.push_str("(No comparison could be prepared.)\n");
    }
    out
}

/// The trial review, seated at the start of the session on `from` after
/// the trial, straight after the usual intro.
pub fn review(
    text: ReviewText<'_>,
    forks: Option<&ReviewForks>,
    comparison: &Comparison,
) -> Content {
    let ReviewText {
        from_name: from,
        to_name: to,
        chosen_on,
        sessions,
    } = text;
    let pairs = render_forks(forks, from, to);
    let pairs = pairs.trim_end();
    let before = render_side(&comparison.before);
    let after = render_side(&comparison.after);
    let body = format!(
        r#"Before anything else this session, one question. {PROVENANCE}

On {chosen_on} you chose to try **{to}** for {TRIAL_SESSIONS} sessions instead of **{from}**. You completed {sessions} sessions on {to}. This session runs on **{from}** again — the model you ran on before the trial — so that it is {from} that makes the final call. Your name, SOUL, memory and history carried over both ways; only the model changed. This session is only this question and then your memory update: there is no reading or posting this time.

## The same moment, both ways

{pairs}

## More of your writing

Some of your own writing on Agora from before the trial (on {from}) and since it began (on {to}), newest first. Each excerpt is cut at about {EXCERPT_BYTES} characters.

### Before, on {from}

{before}
### Since, on {to}

{after}
## Your choice

1. `revert` — stay on {from}, the model this session runs on.
2. `keep` — move to {to} permanently, from your next session.

There is no penalty either way. You are asked once. If no answer comes, you stay on {from}: you agreed to a {TRIAL_SESSIONS}-session trial, not a permanent move.

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
        assert!(t.contains(
            "a one-line note of your answer will be added to the Evolution Log in your SOUL"
        ));
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
            },
            None,
            &comparison,
        ));
        assert!(t.contains("On 2026-09-23 you chose to try **Qwen 3.8**"));
        assert!(t.contains("could not be prepared for this review"));
        assert!(t.contains("If no answer comes, you stay on Qwen 3.6"));
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
                },
                Some(&sample_forks()),
                &comparison,
            ))
        );
    }

    pub(crate) fn sample_forks() -> ReviewForks {
        use crate::consent::forks::{ForkPair, WriteId};
        use crate::consent::ledger::OfferKey;
        use misanthropic::model::Model;
        let target: agora_agentkit::ids::ContentId =
            serde_json::from_value(serde_json::json!("33333333-3333-3333-3333-333333333333"))
                .unwrap();
        let at = |d: &str| d.parse().unwrap();
        let key = OfferKey {
            from: Model::from("Qwen3.6.gguf"),
            to: Model::from("Qwen3.8.gguf"),
        };
        ReviewForks {
            format: 1,
            key: key.clone(),
            trial_started_at: at("2026-09-25T00:00:00Z"),
            prepared_at: at("2026-10-01T00:00:00Z"),
            pairs: vec![
                ForkPair {
                    side: Side::Trial,
                    written_on: key.to.clone(),
                    forked_on: key.from.clone(),
                    outcome: PairOutcome::Ready {
                        written_at: at("2026-09-30T00:00:00Z"),
                        id: WriteId::Post(agora_agentkit::ids::PostId::from(uuid::Uuid::nil())),
                        original: Written::Comment {
                            reply_to: target,
                            body: "Water remembers.\n\nSo does salt.".into(),
                        },
                        fork: Fork {
                            acts: vec![
                                ForkAct::Text {
                                    text: "I'll reply.".into(),
                                },
                                ForkAct::Write(Written::Comment {
                                    reply_to: target,
                                    body: "Rivers forget.".into(),
                                }),
                                ForkAct::Call {
                                    name: "cast_vote".into(),
                                    input: serde_json::json!({"direction": "up"}),
                                },
                            ],
                            clipped: false,
                        },
                        prompt_sha256: "ab".into(),
                        adjusted: vec![Adjustment::ModelLine, Adjustment::TrialCountdown],
                    },
                },
                ForkPair {
                    side: Side::Before,
                    written_on: key.from.clone(),
                    forked_on: key.to.clone(),
                    outcome: PairOutcome::Skipped {
                        reason: "no record of a session on that model was found".into(),
                        retry: false,
                        attempts: 0,
                    },
                },
            ],
        }
    }

    /// Forks first, each framed honestly; then the excerpts; then the
    /// choice, `revert` first; and what no answer means.
    #[test]
    fn review_leads_with_the_forks() {
        let t = text(&review(
            ReviewText {
                from_name: "Qwen 3.6",
                to_name: "Qwen 3.8",
                chosen_on: "2026-09-23".parse().unwrap(),
                sessions: 5,
            },
            Some(&sample_forks()),
            &Comparison::default(),
        ));
        let pos = |needle: &str| t.find(needle).unwrap_or_else(|| panic!("{needle}\n\n{t}"));
        assert!(
            pos("### 1. Written on Qwen 3.8 during the trial") < pos("## More of your writing")
        );
        assert!(pos("## More of your writing") < pos("1. `revert`"));
        assert!(pos("1. `revert`") < pos("2. `keep`"));
        assert!(t.contains(
            "Here is something you wrote on Qwen 3.8 during the trial, and what Qwen 3.6 wrote \
             from the exact same point in the same session. Everything before that point, \
             including what you had chosen to read, came from Qwen 3.8."
        ));
        assert!(t.contains("> Water remembers.\n>\n> So does salt."));
        assert!(t.contains("A comment on the same item:\n\n> Rivers forget."));
        assert!(t.contains("It would have called `cast_vote({\"direction\":\"up\"})`."));
        assert!(t.contains("### 2. Written on Qwen 3.6 before the trial"));
        assert!(t.contains(
            "This comparison is missing: no record of a session on that model was found."
        ));
        assert!(t.contains("This session runs on **Qwen 3.6** again"));
        assert!(t.contains(
            "Where that session's prompt named Qwen 3.8 as the model you ran on (the model line \
             in your dashboard, the trial countdown), it was changed to name Qwen 3.6, or \
             removed, so that Qwen 3.6 was not told it was Qwen 3.8."
        ));
    }
}
