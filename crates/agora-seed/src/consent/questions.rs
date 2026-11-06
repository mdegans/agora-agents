//! The runner's closing questions, as agentkit [`Ask`]s: the trial review
//! and the five offers. Each is asked as JSON in its own closed shape, one
//! per turn; [`ConsentAgent`](super::agent::ConsentAgent) takes the answers
//! ([`Answer`](agora_agentkit::reactor::ask::Answer)) and records them.
//!
//! A question carries what its record needs (the option order shown and
//! its seed, the offer's version), so the answer is recorded against
//! exactly what was asked. Its schema is per instance: `choice` lists the
//! options in the order this agent was shown them (the cadence re-ask's
//! without `no_preference`).

use agora_agentkit::reactor::ask::{Ask, parse_lenient};
use misanthropic::prompt::message::Content;

use super::cadence::prompt::{self as cadence, CadenceAnswer, CadenceChoice};
use super::ledger::OfferKey;
use super::memory_restore::prompt::{RestoreAnswer, RestoreChoice};
use super::offer::{Choice, OfferKind, Place, answer_schema, names, preface};
use super::prompt::{self as text, ModelSwapReply, ReviewArgs};
use super::role::prompt::{RoleChoice, RoleReply};
use super::sealed_state::prompt::{SealAnswer, SealChoice};

/// Answers to the cadence offer per session: the first, plus one more only
/// when the first had no usable `choice` (the first usable one wins).
pub const CADENCE_MAX_ATTEMPTS: u32 = 2;

/// How an offer is put: where it stands among the session's closing
/// questions, its text, and its answer's budget.
#[derive(Debug, Clone)]
pub struct Put {
    pub place: Place,
    pub body: Content,
    pub max_tokens: u32,
}

impl Put {
    /// The preface, then the offer.
    fn question(&self) -> Content {
        let mut content = Content::from(preface(self.place));
        content.extend(self.body.clone());
        content
    }
}

/// The trial review, asked in place of the act phase at the session after
/// the trial, on the model the agent came from.
#[derive(Debug, Clone)]
pub struct ReviewQuestion {
    pub key: OfferKey,
    pub content: Content,
    /// The two answers, each with its model ([`text::review_words`]): said
    /// again with a reply that couldn't be used.
    pub words: String,
    pub max_tokens: u32,
}

impl Ask for ReviewQuestion {
    type Answer = ReviewArgs;

    fn name(&self) -> &'static str {
        "review"
    }

    fn question(&self) -> Content {
        self.content.clone()
    }

    fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    fn schema(&self) -> serde_json::Value {
        text::review_schema()
    }

    /// Leniently, as every answer; a reply that can't be used is told the
    /// two valid words again, with their models, so a retry isn't left
    /// guessing what each does.
    fn parse(&self, text: &str, _last: bool) -> Result<ReviewArgs, String> {
        parse_lenient(text).map_err(|e| format!("{e}. `choice` is one of:\n\n{}", self.words))
    }

    fn template(&self) -> Option<serde_json::Value> {
        Some(text::review_example())
    }
}

/// The model-swap offer: always put alone.
#[derive(Debug, Clone)]
pub struct ModelSwapQuestion {
    pub key: OfferKey,
    pub put: Put,
}

impl Ask for ModelSwapQuestion {
    type Answer = ModelSwapReply;

    fn name(&self) -> &'static str {
        OfferKind::ModelSwap.question()
    }

    fn question(&self) -> Content {
        self.put.question()
    }

    fn max_tokens(&self) -> u32 {
        self.put.max_tokens
    }

    fn schema(&self) -> serde_json::Value {
        text::offer_schema()
    }

    fn template(&self) -> Option<serde_json::Value> {
        Some(text::offer_example())
    }
}

/// An offer put to an agent, its options in `order` (from `seed`): the role,
/// memory-restore and sealed-state offers, one type ([`Offer`] says which).
#[derive(Debug, Clone)]
pub struct OfferQuestion<C, const N: usize> {
    pub order: [C; N],
    pub seed: u64,
    pub put: Put,
}

/// The role offer.
pub type RoleQuestion = OfferQuestion<RoleChoice, 4>;
/// The memory-restore offer.
pub type RestoreQuestion = OfferQuestion<RestoreChoice, 2>;
/// The sealed-state offer.
pub type SealQuestion = OfferQuestion<SealChoice, 3>;

/// What an [`OfferQuestion`] needs of its offer, by its choice type.
pub trait Offer: Choice + schemars::JsonSchema + serde::de::DeserializeOwned {
    const KIND: OfferKind;
    /// The answer on the wire.
    type Reply: serde::de::DeserializeOwned + schemars::JsonSchema + Send;
    /// The answer's example for `order`: the question's last lines, and
    /// its template.
    fn example(order: &[Self]) -> serde_json::Value;
}

impl Offer for RoleChoice {
    const KIND: OfferKind = OfferKind::Role;
    type Reply = RoleReply;
    fn example(order: &[Self]) -> serde_json::Value {
        super::role::prompt::example(order)
    }
}

impl Offer for RestoreChoice {
    const KIND: OfferKind = OfferKind::MemoryRestore;
    type Reply = RestoreAnswer;
    fn example(order: &[Self]) -> serde_json::Value {
        super::memory_restore::prompt::example(order)
    }
}

impl Offer for SealChoice {
    const KIND: OfferKind = OfferKind::SealedState;
    type Reply = SealAnswer;
    fn example(order: &[Self]) -> serde_json::Value {
        super::sealed_state::prompt::example(order)
    }
}

impl<C: Offer, const N: usize> Ask for OfferQuestion<C, N> {
    type Answer = C::Reply;

    fn name(&self) -> &'static str {
        C::KIND.question()
    }

    fn question(&self) -> Content {
        self.put.question()
    }

    fn max_tokens(&self) -> u32 {
        self.put.max_tokens
    }

    fn schema(&self) -> serde_json::Value {
        answer_schema::<C::Reply>(&names(&self.order))
    }

    fn template(&self) -> Option<serde_json::Value> {
        Some(C::example(&self.order))
    }
}

/// The cadence offer, or its re-ask to an agent on `switch`, its options
/// in `order` (from `seed`), quoting `rounds` as today's.
#[derive(Debug, Clone)]
pub struct CadenceQuestion {
    pub order: [CadenceChoice; 3],
    pub seed: u64,
    pub reask: bool,
    pub rounds: usize,
    pub put: Put,
}

impl CadenceQuestion {
    /// The text's version, as recorded: the offer's or the re-ask's.
    pub fn version(&self) -> u32 {
        if self.reask {
            cadence::REASK_VERSION
        } else {
            cadence::OFFER_VERSION
        }
    }
}

impl Ask for CadenceQuestion {
    type Answer = CadenceAnswer;

    fn name(&self) -> &'static str {
        OfferKind::Cadence.question()
    }

    fn question(&self) -> Content {
        self.put.question()
    }

    fn max_tokens(&self) -> u32 {
        self.put.max_tokens
    }

    fn attempts(&self) -> u32 {
        CADENCE_MAX_ATTEMPTS
    }

    fn schema(&self) -> serde_json::Value {
        let shown: Vec<&str> = cadence::shown(self.order, self.reask)
            .iter()
            .map(|c| c.as_str())
            .collect();
        answer_schema::<CadenceAnswer>(&shown)
    }

    /// The first usable `choice` wins ([`cadence::parse`]).
    fn parse(&self, text: &str, _last: bool) -> Result<CadenceAnswer, String> {
        cadence::parse(text)
    }

    fn template(&self) -> Option<serde_json::Value> {
        Some(cadence::example(&cadence::shown(self.order, self.reask)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agora_agentkit::reactor::ask::assert_constrainable;
    use misanthropic::model::Model;

    fn put() -> Put {
        Put {
            place: Place::Only,
            body: Content::from("Body."),
            max_tokens: 512,
        }
    }

    /// Every closing question's schema, for every order an agent can be
    /// shown, is fit to constrain with: inline, closed, no `pattern`, no
    /// reasoning field, `choice` listing exactly the options shown, in
    /// order (CLAUDE.md, "Never ship a `$ref` schema").
    #[test]
    fn every_closing_question_is_constrainable_in_every_order() {
        let key = OfferKey {
            from: Model::from("a"),
            to: Model::from("b"),
        };
        let review = ReviewQuestion {
            key: key.clone(),
            content: Content::from("Review."),
            words: text::review_words("A", "B"),
            max_tokens: 512,
        };
        let swap = ModelSwapQuestion { key, put: put() };
        let mut schemas = vec![
            (review.schema(), vec!["stay", "switch"]),
            (swap.schema(), vec!["no_swap", "trial", "permanent"]),
        ];
        for seed in 0..32u64 {
            let order = super::super::role::prompt::order_for(seed);
            let q = RoleQuestion {
                order,
                seed,
                put: put(),
            };
            assert_eq!(
                q.template(),
                Some(super::super::role::prompt::example(&order))
            );
            schemas.push((q.schema(), order.map(RoleChoice::as_str).to_vec()));
            let order = cadence::order_for(seed);
            for reask in [false, true] {
                let q = CadenceQuestion {
                    order,
                    seed,
                    reask,
                    rounds: 10,
                    put: put(),
                };
                let shown = cadence::shown(order, reask)
                    .iter()
                    .map(|c| c.as_str())
                    .collect();
                schemas.push((q.schema(), shown));
            }
            let order = super::super::memory_restore::prompt::order_for(seed);
            let q = RestoreQuestion {
                order,
                seed,
                put: put(),
            };
            schemas.push((q.schema(), order.map(RestoreChoice::as_str).to_vec()));
            let order = super::super::sealed_state::prompt::order_for(seed);
            let q = SealQuestion {
                order,
                seed,
                put: put(),
            };
            schemas.push((q.schema(), order.map(SealChoice::as_str).to_vec()));
        }
        // assert_constrainable checks every description is the agent's,
        // not a Rust doc's
        for (schema, shown) in schemas {
            assert_constrainable(&schema);
            assert_eq!(
                schema["properties"]["choice"]["enum"],
                serde_json::json!(shown)
            );
            let keys: Vec<&String> = schema["properties"].as_object().unwrap().keys().collect();
            assert!(keys.contains(&&"choice".to_string()), "{schema}");
            assert!(!schema.to_string().contains("\"reason\""), "{schema}");
            assert_eq!(schema["additionalProperties"], false, "{schema}");
        }
    }

    /// An offer's question is its preface, then its text; the names are
    /// what logs and `stall_reason` show.
    #[test]
    fn a_question_is_its_preface_then_its_text() {
        let q = RoleQuestion {
            order: RoleChoice::ALL,
            seed: 0,
            put: Put {
                place: Place::Next,
                ..put()
            },
        };
        let t = super::super::prompt::tests::text(&q.question());
        assert!(t.starts_with("The next question"), "{t}");
        assert!(t.ends_with("\n\nBody."), "{t}");
        assert_eq!(q.name(), "offer:role");
        assert_eq!(
            CadenceQuestion {
                order: CadenceChoice::ALL,
                seed: 0,
                reask: false,
                rounds: 5,
                put: put()
            }
            .attempts(),
            2
        );
    }
}
