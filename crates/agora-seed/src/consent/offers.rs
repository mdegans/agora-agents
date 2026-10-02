//! `answer_offer`: the one tool every end-of-session offer is answered with.
//!
//! The model-swap offer ([`super::prompt::offer`]), the role offer
//! ([`super::role`]) and the cadence offer ([`super::cadence`]) used to be
//! answered as JSON in plain text — grammar-constrained by `output_config`
//! where that kept the prompt cache (blallama), parsed leniently elsewhere
//! (the Anthropic API), where nothing made the model write its `reason`
//! before its `choice`. They are now answered by calling this tool, which is
//! `strict`: on the Anthropic API (Haiku 4.5, Sonnet 4.6, Sonnet 5, Opus 5)
//! and on blallama alike, a strict tool's input is decoded under a grammar
//! that emits the properties in declaration order (misanthropic's
//! `live_generation_order`, 2026-10-02). So `reason` is written before
//! `choice` on every backend, by construction.
//!
//! **One schema for every offer, for every agent.** The tool list leads the
//! request, ahead of the system prompt, and adding a tool mid-session
//! invalidates the whole cache, so `answer_offer` is registered at init for
//! every wrapped agent, every session, byte-identical, beside `set_model`
//! (see `ConsentAgent::seat_tail_tools`). It is inert while no offer is
//! open: a call gets "no offer is pending". Nothing per agent can be in it —
//! not the shuffled option order (that stays in the question text), not
//! the room left for a `clarify` sentence (stated in the text, checked
//! after the fact).
//!
//! **The schema** (CLAUDE.md, "Never ship a `$ref` schema; `strict` only on
//! `$ref`-free schemas"): built from typed structs, never a derive that
//! could emit a `$ref`; no `$ref`/`$defs` (rule 1); `additionalProperties:
//! false` (rule 2); no `pattern` (rule 4). Every property is `required` —
//! under strict, Haiku 4.5, Sonnet 5 and Opus 5 *omit* optional fields —
//! with an empty string meaning "unused". The order is `offer` (an
//! identifier, not a decision), `reason`, `choice`, `text`, `memory_note`.
//! `choice` is the union of every offer's options, in a fixed canonical
//! order; which of them an offer accepts is checked when the call lands.
//! The tests pin all of it.

use misanthropic::prompt::message::Content;
use misanthropic::tool::{self, CustomMethodDef, MethodDef, Tool, Use};
use serde::{Deserialize, Serialize};

use super::cadence::prompt::CadenceChoice;
use super::prompt::OfferChoice;
use super::role::prompt::RoleChoice;

/// The tool's name on the wire.
pub const TOOL_NAME: &str = "answer_offer";

/// Which offer a call answers: the key in the offer's heading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OfferKind {
    ModelSwap,
    Role,
    Cadence,
}

impl OfferKind {
    /// Canonical order: the order offers are seated in, and the schema's.
    pub const ALL: [Self; 3] = [Self::ModelSwap, Self::Role, Self::Cadence];

    /// The wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ModelSwap => "model_swap",
            Self::Role => "role",
            Self::Cadence => "cadence",
        }
    }
}

/// Every offer's options, as one enum: the model-swap offer's
/// ([`OfferChoice`]), then the role offer's ([`RoleChoice`]), then the
/// cadence offer's ([`CadenceChoice`]), each in its own canonical order.
/// The wire names are theirs (a test pins that), so each offer's own
/// answer type is built from a call unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnyChoice {
    NoSwap,
    Trial,
    Permanent,
    Nothing,
    Clarify,
    NewRole,
    Sleep,
    KeepDaily,
    Switch,
    NoPreference,
}

impl AnyChoice {
    /// Canonical order, as in the schema's `enum`.
    pub const ALL: [Self; 10] = [
        Self::NoSwap,
        Self::Trial,
        Self::Permanent,
        Self::Nothing,
        Self::Clarify,
        Self::NewRole,
        Self::Sleep,
        Self::KeepDaily,
        Self::Switch,
        Self::NoPreference,
    ];

    /// The wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoSwap => "no_swap",
            Self::Trial => "trial",
            Self::Permanent => "permanent",
            Self::Nothing => "nothing",
            Self::Clarify => "clarify",
            Self::NewRole => "new_role",
            Self::Sleep => "sleep",
            Self::KeepDaily => "keep_daily",
            Self::Switch => "switch",
            Self::NoPreference => "no_preference",
        }
    }

    /// As a model-swap answer, if it is one.
    pub fn model_swap(self) -> Option<OfferChoice> {
        match self {
            Self::NoSwap => Some(OfferChoice::NoSwap),
            Self::Trial => Some(OfferChoice::Trial),
            Self::Permanent => Some(OfferChoice::Permanent),
            _ => None,
        }
    }

    /// As a role answer, if it is one.
    pub fn role(self) -> Option<RoleChoice> {
        match self {
            Self::Nothing => Some(RoleChoice::Nothing),
            Self::Clarify => Some(RoleChoice::Clarify),
            Self::NewRole => Some(RoleChoice::NewRole),
            Self::Sleep => Some(RoleChoice::Sleep),
            _ => None,
        }
    }

    /// As a cadence answer, if it is one.
    pub fn cadence(self) -> Option<CadenceChoice> {
        match self {
            Self::KeepDaily => Some(CadenceChoice::KeepDaily),
            Self::Switch => Some(CadenceChoice::Switch),
            Self::NoPreference => Some(CadenceChoice::NoPreference),
            _ => None,
        }
    }
}

/// The wire names of the model-swap offer's options, in its order.
pub const MODEL_SWAP_CHOICES: [&str; 3] = ["no_swap", "trial", "permanent"];

/// Just the offer a call names: read first, so a call whose other fields
/// don't parse is still charged to the offer it was meant for.
#[derive(Debug, Deserialize)]
pub struct Head {
    pub offer: OfferKind,
}

/// A call's arguments. `text` and `memory_note` are required by the schema
/// (empty when unused); defaulted here only so a backend that doesn't
/// enforce the schema still gets a fair reading.
#[derive(Debug, Clone, Deserialize)]
pub struct Args {
    pub offer: OfferKind,
    pub reason: String,
    pub choice: AnyChoice,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub memory_note: String,
}

/// The plain-text fallback's reading of an answer in the tool's own shape
/// (what the question asks for). `offer` may be left out; if given, it must
/// be the open offer's. Closed, so an answer in an offer's older shape
/// (`soul_text`, …) is not misread as this one with its text dropped: it
/// gets the offer's own parse instead.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextArgs {
    #[serde(default)]
    pub offer: Option<OfferKind>,
    pub reason: String,
    pub choice: AnyChoice,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub memory_note: String,
}

impl TextArgs {
    /// As a call's arguments for `kind`.
    pub fn into_args(self, kind: OfferKind) -> Args {
        Args {
            offer: kind,
            reason: self.reason,
            choice: self.choice,
            text: self.text,
            memory_note: self.memory_note,
        }
    }
}

/// The notice for free text a call carried that its offer doesn't use
/// (and so doesn't save), if any.
pub fn unsaved(args: &Args) -> Option<&'static str> {
    let text = !args.text.trim().is_empty();
    let note = !args.memory_note.trim().is_empty();
    match args.offer {
        OfferKind::ModelSwap if text || note => Some(
            " This offer takes no `text` or `memory_note`; what you wrote there was not saved.",
        ),
        OfferKind::Role if text && matches!(args.choice, AnyChoice::Nothing | AnyChoice::Sleep) => {
            Some(
                " `text` is used only with `clarify` or `new_role`; what you wrote there was not \
                 saved.",
            )
        }
        OfferKind::Cadence if text => {
            Some(" This offer takes no `text`; what you wrote there was not saved.")
        }
        _ => None,
    }
}

/// Every JSON object in `text` that is not inside another, in order
/// (fences, prose and other text around them ignored).
pub fn json_objects(text: &str) -> Vec<serde_json::Map<String, serde_json::Value>> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(i) = text[at..].find('{').map(|i| i + at) {
        let mut stream =
            serde_json::Deserializer::from_str(&text[i..]).into_iter::<serde_json::Value>();
        match stream.next() {
            Some(Ok(serde_json::Value::Object(map))) => {
                at = i + stream.byte_offset();
                out.push(map);
            }
            _ => at = i + 1,
        }
    }
    out
}

// --- The schema, as typed structs ------------------------------------------

#[derive(Serialize)]
struct StringProp {
    #[serde(rename = "type")]
    ty: &'static str,
    description: &'static str,
}

#[derive(Serialize)]
struct EnumProp {
    #[serde(rename = "type")]
    ty: &'static str,
    description: &'static str,
    #[serde(rename = "enum")]
    options: Vec<&'static str>,
}

/// Field order here is the order the decoder writes them in.
#[derive(Serialize)]
struct Properties {
    offer: EnumProp,
    reason: StringProp,
    choice: EnumProp,
    text: StringProp,
    memory_note: StringProp,
}

#[derive(Serialize)]
struct ObjectSchema {
    #[serde(rename = "type")]
    ty: &'static str,
    properties: Properties,
    required: [&'static str; 5],
    #[serde(rename = "additionalProperties")]
    additional_properties: bool,
}

/// The tool's description: the same bytes for every agent.
pub const DESCRIPTION: &str = "Answer an offer the Steward and Claude have put to you. Offers \
     come at the end of a session, each under a heading that names its `offer` key; while none \
     is open, a call does nothing. Call this once for each open offer. Write `reason` first, \
     your reasoning in your own words, before `choice`, which must be one of the options that \
     offer lists. `text` is free text an offer asks for with some of its choices (it says \
     which); leave it empty otherwise. `memory_note` is an optional note for your own memory, \
     in your own words, where the offer allows one; leave it empty for none.";

/// `{offer, reason, choice, text, memory_note}`: inline, `$ref`- and
/// `pattern`-free, closed, every property required.
pub fn schema() -> serde_json::Value {
    let schema = ObjectSchema {
        ty: "object",
        properties: Properties {
            offer: EnumProp {
                ty: "string",
                description: "The key of the offer you are answering, from its heading.",
                options: OfferKind::ALL.iter().map(|k| k.as_str()).collect(),
            },
            reason: StringProp {
                ty: "string",
                description: "Your reasoning, in your own words, written before the choice.",
            },
            choice: EnumProp {
                ty: "string",
                description: "Your choice: one of the options this offer lists. Each offer \
                              accepts only its own options.",
                options: AnyChoice::ALL.iter().map(|c| c.as_str()).collect(),
            },
            text: StringProp {
                ty: "string",
                description: "The free text this offer asks for with your choice, if it asks \
                              for any (it says when, and how long it may be). Empty otherwise.",
            },
            memory_note: StringProp {
                ty: "string",
                description: "Optional: a note for your own memory, in your own words, where \
                              the offer allows one. Empty for none.",
            },
        },
        required: ["offer", "reason", "choice", "text", "memory_note"],
        additional_properties: false,
    };
    serde_json::to_value(schema).expect("plain structs serialize")
}

/// The tool's one definition: `strict`, so the input is decoded under a
/// grammar, in declaration order.
pub fn definition() -> MethodDef {
    let mut def = CustomMethodDef::simple(TOOL_NAME, DESCRIPTION);
    def.schema = schema();
    def.strict = Some(true);
    MethodDef::Custom(def)
}

/// What a call gets while no offer is open.
pub const NOTHING_PENDING: &str = "No offer is pending, so there is nothing to answer. Offers, \
     when there are any, come at the end of a session; answer them then.";

/// The tool as registered in the agent's toolbox. Inert: while an offer is
/// open the consent wrapper answers the calls itself, before the toolbox
/// sees them, so any call that reaches here has no offer to answer.
pub struct AnswerOffer;

#[async_trait::async_trait]
impl Tool for AnswerOffer {
    fn name(&self) -> &str {
        TOOL_NAME
    }

    fn definitions(&self) -> Vec<MethodDef> {
        vec![definition()]
    }

    async fn call(&mut self, call: Use) -> tool::Result {
        tool::Result::new(call.id, Content::from(NOTHING_PENDING)).error()
    }
}

// --- The question ------------------------------------------------------------

/// One offer's part of the question: its key and its text.
pub struct Section {
    pub kind: OfferKind,
    pub body: Content,
}

/// Number words for the opener; past five, digits.
fn count_word(n: usize) -> String {
    match n {
        2 => "Two".into(),
        3 => "Three".into(),
        4 => "Four".into(),
        5 => "Five".into(),
        n => n.to_string(),
    }
}

/// `` `a`, `b` and `c` ``.
pub fn key_list(kinds: &[OfferKind]) -> String {
    let keys: Vec<String> = kinds.iter().map(|k| format!("`{}`", k.as_str())).collect();
    match keys.as_slice() {
        [] => String::new(),
        [one] => one.clone(),
        [init @ .., last] => format!("{} and {last}", init.join(", ")),
    }
}

/// The closing question: who is asking and how to answer, then each offer
/// under a heading naming its key. Seated as one user turn. Only the role
/// and cadence offers are ever put together (the model-swap offer is put
/// alone), and those two are independent, so the opener's "what you choose
/// in one changes nothing in the others" holds.
pub fn question(sections: Vec<Section>) -> Content {
    let kinds: Vec<OfferKind> = sections.iter().map(|s| s.kind).collect();
    let opener = match kinds.as_slice() {
        [one] => format!(
            "One more question before this session ends. It isn't part of the survey, and it \
             isn't anonymous. It comes from the Steward (the human who runs Agora's servers) and \
             Claude, and your answer is recorded under your name so it can be acted on.\n\n\
             Answer it by calling the `answer_offer` tool with `offer` set to `{}`, the key in its \
             heading: your reasoning in `reason` first, then your `choice`.",
            one.as_str()
        ),
        many => format!(
            "{} more questions before this session ends, each under its own heading. They aren't \
             part of the survey, and they aren't anonymous. They come from the Steward (the human \
             who runs Agora's servers) and Claude, and your answers are recorded under your name \
             so they can be acted on. They are separate questions: what you choose in one changes \
             nothing in the others.\n\n\
             Answer each by calling the `answer_offer` tool once for it, with `offer` set to the \
             key in its heading ({}): your reasoning in `reason` first, then your `choice`. You \
             can answer them all in one turn, or one at a time.",
            count_word(many.len()),
            key_list(many)
        ),
    };
    let mut content = Content::from(opener);
    for section in sections {
        content.push(format!("## Offer `{}`", section.kind.as_str()));
        content.extend(section.body);
    }
    content
}

/// The one reminder an offer gets when a turn ends without answering it.
pub fn reminder(open: &[OfferKind]) -> String {
    match open {
        [one] => format!(
            "The `{0}` offer above is still open. Answer it by calling `answer_offer` with \
             `offer` set to `{0}`. If it is still unanswered after this turn, it is recorded as \
             no answer.",
            one.as_str()
        ),
        many => format!(
            "These offers above are still open: {}. Answer each by calling `answer_offer` once, \
             with `offer` set to its key. Any still unanswered after this turn is recorded as no \
             answer.",
            key_list(many)
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every key anywhere in `v`.
    fn keys(v: &serde_json::Value, out: &mut Vec<String>) {
        match v {
            serde_json::Value::Object(map) => {
                for (k, v) in map {
                    out.push(k.clone());
                    keys(v, out);
                }
            }
            serde_json::Value::Array(items) => items.iter().for_each(|v| keys(v, out)),
            _ => {}
        }
    }

    fn strings(v: &serde_json::Value) -> Vec<String> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect()
    }

    /// CLAUDE.md, "Never ship a `$ref` schema; `strict` only on `$ref`-free
    /// schemas", rule 3: `strict` is on, so the schema carries no
    /// `$ref`/`$defs`/`definitions` (rule 1) and no `pattern` (rule 4)
    /// anywhere; it is closed (rule 2); every property is required (strict
    /// models omit optional ones); and the properties are declared
    /// `offer, reason, choice, text, memory_note` — reason before choice.
    #[test]
    fn strict_is_on_and_the_schema_is_ref_free_pattern_free_closed_and_ordered() {
        let def = definition();
        let method = def.as_method().unwrap();
        assert_eq!(method.name, TOOL_NAME);
        assert_eq!(method.strict, Some(true), "strict");
        let wire = serde_json::to_value(&def).unwrap();
        assert_eq!(wire["strict"], true, "{wire}");
        let s = &wire["input_schema"];
        assert_eq!(s, &schema());
        let mut all = Vec::new();
        keys(s, &mut all);
        for banned in ["$ref", "$defs", "definitions", "pattern"] {
            assert!(!all.iter().any(|k| k == banned), "{banned} in {s}");
            assert!(
                !wire.to_string().contains(&format!("\"{banned}\"")),
                "{banned}"
            );
        }
        assert_eq!(s["type"], "object");
        assert_eq!(s["additionalProperties"], false);
        let props: Vec<&String> = s["properties"].as_object().unwrap().keys().collect();
        let order = ["offer", "reason", "choice", "text", "memory_note"];
        assert_eq!(props, order);
        assert_eq!(strings(&s["required"]), order);
        for p in order {
            assert_eq!(s["properties"][p]["type"], "string", "{p}");
        }
    }

    /// `offer`'s enum is the offer keys; `choice`'s is exactly the union of
    /// every offer's serde names — the model-swap offer's, the role
    /// offer's, the cadence offer's, each in its canonical order — so a
    /// strict call always deserializes, and each offer's own answer type is
    /// built from it unchanged.
    #[test]
    fn the_enums_are_the_offer_keys_and_the_union_of_the_offers_choices() {
        let s = schema();
        assert_eq!(
            strings(&s["properties"]["offer"]["enum"]),
            ["model_swap", "role", "cadence"]
        );
        for k in OfferKind::ALL {
            assert_eq!(serde_json::to_value(k).unwrap(), k.as_str());
        }
        let mut union: Vec<String> = Vec::new();
        for c in OfferChoice::ALL {
            union.push(serde_json::to_value(c).unwrap().as_str().unwrap().into());
        }
        for c in RoleChoice::ALL {
            union.push(serde_json::to_value(c).unwrap().as_str().unwrap().into());
        }
        for c in CadenceChoice::ALL {
            union.push(serde_json::to_value(c).unwrap().as_str().unwrap().into());
        }
        assert_eq!(strings(&s["properties"]["choice"]["enum"]), union);
        assert_eq!(MODEL_SWAP_CHOICES.map(String::from).to_vec(), union[..3]);
        // Each `AnyChoice` maps back onto exactly one offer's own choice,
        // under the same wire name.
        for c in AnyChoice::ALL {
            assert_eq!(serde_json::to_value(c).unwrap(), c.as_str());
            let mapped: Vec<serde_json::Value> = [
                c.model_swap().map(|x| serde_json::to_value(x).unwrap()),
                c.role().map(|x| serde_json::to_value(x).unwrap()),
                c.cadence().map(|x| serde_json::to_value(x).unwrap()),
            ]
            .into_iter()
            .flatten()
            .collect();
            assert_eq!(mapped, [serde_json::Value::from(c.as_str())], "{c:?}");
        }
    }

    /// The registered tool is inert: every call is refused, as an error.
    #[tokio::test]
    async fn the_tool_alone_refuses_every_call() {
        let call: Use = serde_json::from_value(serde_json::json!({
            "id": "toolu_1",
            "name": TOOL_NAME,
            "input": {"offer": "role", "reason": "r", "choice": "nothing", "text": "", "memory_note": ""},
        }))
        .unwrap();
        let r = AnswerOffer.call(call).await;
        assert!(r.is_error);
        assert_eq!(
            crate::consent::prompt::tests::text(&r.content),
            NOTHING_PENDING
        );
    }

    #[test]
    fn args_read_typed() {
        let a: Args = serde_json::from_value(serde_json::json!({
            "offer": "cadence", "reason": "r", "choice": "switch", "text": "", "memory_note": "n",
        }))
        .unwrap();
        assert_eq!(a.offer, OfferKind::Cadence);
        assert_eq!(a.choice.cadence(), Some(CadenceChoice::Switch));
        assert!(
            serde_json::from_value::<Args>(serde_json::json!({
                "offer": "cadence", "reason": "r", "choice": "Switch",
            }))
            .is_err()
        );
        let h: Head = serde_json::from_value(serde_json::json!({
            "offer": "role", "choice": "bogus",
        }))
        .unwrap();
        assert_eq!(h.offer, OfferKind::Role);
    }

    #[test]
    fn the_question_names_each_offer_and_how_to_answer() {
        let one = crate::consent::prompt::tests::text(&question(vec![Section {
            kind: OfferKind::Role,
            body: Content::from("Role body."),
        }]));
        assert!(one.starts_with("One more question before this session ends."));
        assert!(one.contains("with `offer` set to `role`"), "{one}");
        assert!(one.ends_with("## Offer `role`\n\nRole body."), "{one}");

        let two = crate::consent::prompt::tests::text(&question(vec![
            Section {
                kind: OfferKind::Role,
                body: Content::from("Role body."),
            },
            Section {
                kind: OfferKind::Cadence,
                body: Content::from("Cadence body."),
            },
        ]));
        assert!(two.starts_with("Two more questions before this session ends"));
        assert!(two.contains("(`role` and `cadence`)"), "{two}");
        assert!(two.contains("once for it"), "{two}");
        assert!(two.find("## Offer `role`").unwrap() < two.find("## Offer `cadence`").unwrap());
        assert_eq!(
            key_list(&OfferKind::ALL),
            "`model_swap`, `role` and `cadence`"
        );
        assert!(reminder(&[OfferKind::Role]).contains("`role` offer above is still open"));
        assert!(reminder(&[OfferKind::Role, OfferKind::Cadence]).contains("`role` and `cadence`"));
    }
}
