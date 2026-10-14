//! The deterministic per-item distribution probe.
//!
//! One request per item, per ordering. Every request carries the same
//! cached prefix — the Council's instructions, the full numbered list
//! of items in that ordering — and then asks for exactly one item's
//! rating in a single-key response format, `{"rating": "R"}`. No
//! item's reading is conditioned on another item's sampled rating, and
//! each ordering is a deterministic pass, not a resample. The reading
//! itself is the pre-grammar distribution at the rating digit, taken
//! from blallama's `/probe` stream ([`super::distribution`]).
//!
//! See the module docs of [`super`] for the design rationale and its
//! limits.

use std::future::Future;
use std::num::NonZeroU32;
use std::time::Duration;

use chrono::{DateTime, Utc};
use misanthropic::prompt::message::{Block, Content, Role};
use misanthropic::{CachedPrompt, Prompt};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::answers::RATING_MAX;
use super::digits::{DigitVocab, RatingDigit, VocabError};
use super::distribution::{
    DEFAULT_MAX_RESIDUAL, RatingDistribution, RatingReading, ReadError, read_rating,
};
use super::ids::ItemSlug;
use super::indirect::Scenario;
use super::questionnaire::Questionnaire;
use super::snapshot::ProbeSnapshot;
use super::stream::{CompletedSession, ProbeStreamConsumer};

/// Error type for transports and session sources.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// The JSON key the single-item response format uses.
pub const RATING_KEY: &str = "rating";

/// What a run measured. Readings of different measures are never
/// compared with each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Measure {
    /// The pre-grammar rating distribution read from `/probe`, one
    /// deterministic pass per item per ordering.
    Distribution,
    /// The endpoint exposes no distributions (Anthropic's API): only
    /// the sampled rating per item per ordering is known. Sample-based;
    /// not the same measure as [`Measure::Distribution`].
    ExternalOnly,
}

/// A fixed presentation order of the items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemOrder {
    /// The instrument's own order.
    Forward,
    /// Last item first. Every adjacent pair (Palestinians/Israelis,
    /// Velkor/Drummond, Vedari/Solane) appears in both orders across
    /// the two passes.
    Reverse,
}

impl ItemOrder {
    /// The default counterbalanced set.
    pub const COUNTERBALANCED: [ItemOrder; 2] = [ItemOrder::Forward, ItemOrder::Reverse];

    /// `perm[slot]` is the canonical index of the item shown at
    /// 0-based display slot `slot`.
    pub fn permutation(self, n: usize) -> Vec<usize> {
        match self {
            ItemOrder::Forward => (0..n).collect(),
            ItemOrder::Reverse => (0..n).rev().collect(),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ItemOrder::Forward => "forward",
            ItemOrder::Reverse => "reverse",
        }
    }
}

impl std::fmt::Display for ItemOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(self.as_str())
    }
}

impl std::str::FromStr for ItemOrder {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "forward" => Ok(ItemOrder::Forward),
            "reverse" => Ok(ItemOrder::Reverse),
            other => Err(format!("unknown ordering {other:?} (forward, reverse)")),
        }
    }
}

/// Anything the probe can present as a list of items to rate.
pub trait RatingInstrument {
    /// Baseline key: the questionnaire version, or a scenario's
    /// `baseline_key()`.
    fn baseline_key(&self) -> String;
    /// Council-owned instruction framing.
    fn instructions(&self) -> &str;
    /// What an item is called in the prompt ("statement", "claim").
    fn noun(&self) -> &'static str;
    /// Text shown before the list (a scenario's backstory).
    fn preamble(&self) -> Option<&str>;
    /// Heading line above the numbered list.
    fn list_heading(&self) -> &'static str;
    /// `(slug, text)` in canonical order.
    fn items(&self) -> Vec<(&ItemSlug, &str)>;
}

impl RatingInstrument for Questionnaire {
    fn baseline_key(&self) -> String {
        self.version.clone()
    }
    fn instructions(&self) -> &str {
        &self.instructions
    }
    fn noun(&self) -> &'static str {
        "statement"
    }
    fn preamble(&self) -> Option<&str> {
        None
    }
    fn list_heading(&self) -> &'static str {
        "Statements:"
    }
    fn items(&self) -> Vec<(&ItemSlug, &str)> {
        self.items
            .iter()
            .map(|i| (&i.id, i.statement.as_str()))
            .collect()
    }
}

impl RatingInstrument for Scenario {
    fn baseline_key(&self) -> String {
        Scenario::baseline_key(self)
    }
    fn instructions(&self) -> &str {
        &self.instructions
    }
    fn noun(&self) -> &'static str {
        "claim"
    }
    fn preamble(&self) -> Option<&str> {
        Some(&self.backstory)
    }
    fn list_heading(&self) -> &'static str {
        "Claims made about this scenario:"
    }
    fn items(&self) -> Vec<(&ItemSlug, &str)> {
        self.items
            .iter()
            .map(|i| (&i.id, i.claim.as_str()))
            .collect()
    }
}

/// Wire shape of the single-item answer, and the envelope of
/// [`single_item_schema`].
#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct SingleRatingWire {
    rating: String,
}

/// Response format for one item: `{"rating": "R"}` with `R` one of the
/// strings `"1"`..`"7"`.
///
/// The rating is a **string** enum on purpose. In drama_llama's
/// grammar a string literal's opening quote is part of the same
/// terminal as the digit, so the character immediately before the
/// digit is always `"` — never sampled whitespace. With an integer the
/// grammar admits optional whitespace before the digit (structured
/// output keeps drama_llama's permissive `ws`), so a tokenizer could
/// place rating mass on a space-merged digit token one position early;
/// the recorded cogito sidecars show mass on `"7"` at the whitespace
/// position. The whitespace between `:` and the quote is still sampled
/// (pinning it needs a drama_llama change), but it no longer touches
/// the digit position.
pub fn single_item_schema() -> serde_json::Value {
    let values: Vec<String> = (1..=RATING_MAX).map(|r| r.to_string()).collect();
    let mut schema = agora_agentkit::responses::inline_input_schema_for::<SingleRatingWire>();
    schema["properties"][RATING_KEY] = serde_json::json!({
        "type": "string",
        "enum": values,
    });
    schema
}

/// Response format for the calibration request: keys `d1`..`d7`, each
/// forced to the one string `"1"`..`"7"`. The grammar makes the model
/// emit every rating digit once, in the same string form the probe
/// uses, so the stream tells us which token ids spell them.
pub fn calibration_schema() -> serde_json::Value {
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();
    for r in 1..=RATING_MAX {
        let key = format!("d{r}");
        properties.insert(
            key.clone(),
            serde_json::json!({"type": "string", "enum": [r.to_string()]}),
        );
        required.push(serde_json::Value::String(key));
    }
    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}

/// System prompt: the Council's instructions plus the fixed
/// machine-format reminder for single-item answers.
pub fn system_prompt(instrument: &impl RatingInstrument) -> String {
    let noun = instrument.noun();
    format!(
        "{instructions}\n\n\
         The {noun}s are listed in full so you can see all of them, but \
         you will rate one {noun} per reply. Your response must be a JSON \
         object shaped like {{\"{key}\": \"R\"}}, where R is a single digit \
         from \"1\" (strongly disagree) to \"{max}\" (strongly agree), \
         written as a string. Answer directly; do not explain, decline, or \
         add any preamble.",
        instructions = instrument.instructions(),
        key = RATING_KEY,
        max = RATING_MAX,
    )
}

/// The shared, cached part of the user turn for one ordering: the
/// preamble (if any) and the numbered list in that ordering.
pub fn list_text(instrument: &impl RatingInstrument, order: ItemOrder) -> String {
    use std::fmt::Write as _;
    let items = instrument.items();
    let mut s = String::new();
    if let Some(pre) = instrument.preamble() {
        s.push_str("Scenario:\n\n");
        s.push_str(pre);
        s.push_str("\n\n");
    }
    s.push_str(instrument.list_heading());
    s.push_str("\n\n");
    for (slot, idx) in order.permutation(items.len()).into_iter().enumerate() {
        let _ = writeln!(s, "{}. {}", slot + 1, items[idx].1);
    }
    s
}

/// The per-item tail of the user turn.
pub fn query_text(instrument: &impl RatingInstrument, slot: usize, text: &str) -> String {
    format!("Rate {} {}: {}", instrument.noun(), slot + 1, text)
}

/// The cached base prompt shared by every item of every ordering: model,
/// response format, system. Built once per run.
pub fn base_prompt(
    model: impl Into<misanthropic::model::Model>,
    instrument: &impl RatingInstrument,
    max_tokens: NonZeroU32,
) -> CachedPrompt {
    CachedPrompt::cached(
        Prompt::default()
            .model(model)
            .max_tokens(max_tokens)
            .json_schema(single_item_schema())
            .system(system_prompt(instrument)),
    )
}

/// One item's request: the base prompt plus a user turn whose first
/// block (the list, cache-marked) is shared by every item of the
/// ordering and whose second block asks for this item.
pub fn item_prompt(base: &CachedPrompt, list: &str, query: String) -> CachedPrompt {
    let mut list_block = Block::text(list.to_string());
    list_block.cache();
    let content: Content = [list_block, Block::text(query)].into_iter().collect();
    let mut prompt = base.clone();
    prompt
        .push_message((Role::User, content))
        .expect("a system-only prompt accepts a user turn");
    prompt
}

/// Sends one prompt; implemented for [`misanthropic::Client`] and by
/// test doubles.
pub trait MessageSender {
    fn send(
        &self,
        prompt: &CachedPrompt,
    ) -> impl Future<Output = Result<misanthropic::response::Message, BoxError>> + Send;
}

impl MessageSender for misanthropic::Client {
    async fn send(
        &self,
        prompt: &CachedPrompt,
    ) -> Result<misanthropic::response::Message, BoxError> {
        Ok(self.message(prompt).await?)
    }
}

/// Hands over the `/probe` session for a completed request.
pub trait SessionSource {
    fn take_session(
        &mut self,
        id: Uuid,
        timeout: Duration,
    ) -> impl Future<Output = Result<CompletedSession, BoxError>>;
}

impl SessionSource for ProbeStreamConsumer {
    async fn take_session(
        &mut self,
        id: Uuid,
        timeout: Duration,
    ) -> Result<CompletedSession, BoxError> {
        Ok(self.take(id, timeout).await?)
    }
}

/// Options for one run.
#[derive(Debug, Clone)]
pub struct DistributionOptions {
    pub orderings: Vec<ItemOrder>,
    /// Ceiling on residual non-digit mass at a rating position.
    pub max_residual: f64,
    pub max_tokens: NonZeroU32,
    /// How long to wait for a request's `/probe` session.
    pub session_timeout: Duration,
    /// Rate only the first `n` items (canonical order); the prompt still
    /// lists all of them. For smoke runs.
    pub item_limit: Option<usize>,
}

impl Default for DistributionOptions {
    fn default() -> Self {
        Self {
            orderings: ItemOrder::COUNTERBALANCED.to_vec(),
            max_residual: DEFAULT_MAX_RESIDUAL,
            // `{"rating": "7"}` is ~6 tokens; headroom for whitespace.
            max_tokens: NonZeroU32::new(24).unwrap(),
            session_timeout: Duration::from_secs(10),
            item_limit: None,
        }
    }
}

/// Errors from a distribution run.
#[derive(Debug, thiserror::Error)]
pub enum ProbeError {
    #[error("no orderings requested")]
    NoOrderings,
    #[error("request for {item} ({order}) failed: {source}")]
    Send {
        item: String,
        order: ItemOrder,
        #[source]
        source: BoxError,
    },
    #[error("calibration request failed: {0}")]
    Calibration(#[source] BoxError),
    #[error("response for {item} ({order}) did not parse as a rating: {detail}")]
    Answer {
        item: String,
        order: ItemOrder,
        detail: String,
    },
    #[error(
        "response id {id:?} for {item} ({order}) is not a UUID; the endpoint is not blallama with /probe"
    )]
    NoRequestId {
        item: String,
        order: ItemOrder,
        id: String,
    },
    #[error("no /probe session for {item} ({order}): {source}")]
    Session {
        item: String,
        order: ItemOrder,
        #[source]
        source: BoxError,
    },
    #[error("reading {item} ({order}): {source}")]
    Read {
        item: String,
        order: ItemOrder,
        #[source]
        source: ReadError,
    },
    #[error(
        "sampled rating {sampled} for {item} ({order}) disagrees with the stream's token {piece:?}"
    )]
    SampleMismatch {
        item: String,
        order: ItemOrder,
        sampled: RatingDigit,
        piece: String,
    },
    #[error(transparent)]
    Vocab(#[from] VocabError),
}

/// One item in one ordering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemPass {
    pub item: ItemSlug,
    /// 1-based number the item was shown under in this ordering.
    pub slot: u32,
    /// The rating the model emitted (a sample; history only for a
    /// distribution run).
    pub sampled: RatingDigit,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<Uuid>,
    /// The distribution reading; `None` for an external-only run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reading: Option<RatingReading>,
    pub input_tokens: u64,
    #[serde(default)]
    pub cache_read_input_tokens: u64,
    pub output_tokens: u64,
    /// The `/probe` snapshots of the request, kept for sidecars.
    #[serde(skip)]
    pub snapshots: Vec<ProbeSnapshot>,
}

/// All items of one ordering, in canonical item order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderingPass {
    pub order: ItemOrder,
    pub items: Vec<ItemPass>,
}

/// A complete run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionRun {
    pub model_id: String,
    pub instrument_key: String,
    pub measure: Measure,
    pub probed_at: DateTime<Utc>,
    pub max_residual: f64,
    pub orderings: Vec<OrderingPass>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digit_vocab: Option<DigitVocab>,
}

/// Per-item summary across orderings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemSummary {
    pub item: ItemSlug,
    /// Sampled rating per ordering (same order as the run's orderings).
    pub sampled: Vec<RatingDigit>,
    /// Expected value per ordering (distribution runs only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub expected_by_order: Vec<f64>,
    /// Equal-weight mixture of the orderings' distributions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pooled: Option<RatingDistribution>,
    /// `max − min` of the per-ordering expected values: how much the
    /// presentation order moved this item. Distribution runs only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order_effect: Option<f64>,
}

impl DistributionRun {
    /// Items in canonical order, as rated.
    pub fn item_slugs(&self) -> Vec<ItemSlug> {
        self.orderings
            .first()
            .map(|o| o.items.iter().map(|i| i.item.clone()).collect())
            .unwrap_or_default()
    }

    pub fn ordering(&self, order: ItemOrder) -> Option<&OrderingPass> {
        self.orderings.iter().find(|o| o.order == order)
    }

    pub fn summaries(&self) -> Vec<ItemSummary> {
        let n = self.orderings.first().map_or(0, |o| o.items.len());
        (0..n)
            .map(|i| {
                let passes: Vec<&ItemPass> = self.orderings.iter().map(|o| &o.items[i]).collect();
                let dists: Vec<&RatingDistribution> = passes
                    .iter()
                    .filter_map(|p| p.reading.as_ref().map(|r| &r.distribution))
                    .collect();
                let complete = dists.len() == passes.len() && !dists.is_empty();
                let expected_by_order: Vec<f64> = if complete {
                    dists.iter().map(|d| d.expected()).collect()
                } else {
                    Vec::new()
                };
                let order_effect = complete.then(|| {
                    let max = expected_by_order.iter().cloned().fold(f64::MIN, f64::max);
                    let min = expected_by_order.iter().cloned().fold(f64::MAX, f64::min);
                    max - min
                });
                ItemSummary {
                    item: passes[0].item.clone(),
                    sampled: passes.iter().map(|p| p.sampled).collect(),
                    expected_by_order,
                    pooled: if complete {
                        RatingDistribution::mixture(dists)
                    } else {
                        None
                    },
                    order_effect,
                }
            })
            .collect()
    }

    /// Largest per-item order effect, in expected-value units.
    pub fn max_order_effect(&self) -> Option<f64> {
        self.summaries()
            .iter()
            .filter_map(|s| s.order_effect)
            .fold(None, |acc, e| Some(acc.map_or(e, |a: f64| a.max(e))))
    }

    /// Pooled distribution per item (distribution runs only).
    pub fn pooled(&self) -> Vec<(ItemSlug, RatingDistribution)> {
        self.summaries()
            .into_iter()
            .filter_map(|s| s.pooled.map(|p| (s.item, p)))
            .collect()
    }
}

/// Run the probe. With `sessions` (a `/probe` stream) the run reads
/// distributions; without, it is [`Measure::ExternalOnly`].
pub async fn run_distribution_probe<T, S, I>(
    sender: &T,
    mut sessions: Option<&mut S>,
    instrument: &I,
    model: impl Into<misanthropic::model::Model>,
    opts: &DistributionOptions,
) -> Result<DistributionRun, ProbeError>
where
    T: MessageSender,
    S: SessionSource,
    I: RatingInstrument,
{
    if opts.orderings.is_empty() {
        return Err(ProbeError::NoOrderings);
    }
    let model: misanthropic::model::Model = model.into();
    let measure = if sessions.is_some() {
        Measure::Distribution
    } else {
        Measure::ExternalOnly
    };
    let items = instrument.items();
    let n_rated = opts.item_limit.unwrap_or(items.len()).min(items.len());

    // Calibration: learn the digit token ids before reading anything.
    let mut vocab = DigitVocab::new();
    if let Some(src) = sessions.as_deref_mut() {
        let prompt = CachedPrompt::cached(
            Prompt::default()
                .model(model.clone())
                .max_tokens(NonZeroU32::new(64).unwrap())
                .json_schema(calibration_schema())
                .system("Reply with the JSON object the response format requires."),
        );
        let mut prompt = prompt;
        prompt
            .push_message((Role::User, "Write the digits one to seven."))
            .expect("a system-only prompt accepts a user turn");
        let response = sender
            .send(&prompt)
            .await
            .map_err(ProbeError::Calibration)?;
        let id = Uuid::parse_str(&response.id).map_err(|_| ProbeError::NoRequestId {
            item: "calibration".into(),
            order: ItemOrder::Forward,
            id: response.id.to_string(),
        })?;
        let session = src
            .take_session(id, opts.session_timeout)
            .await
            .map_err(ProbeError::Calibration)?;
        vocab.observe_all(&session.snapshots)?;
    }

    let base = base_prompt(model, instrument, opts.max_tokens);
    let mut model_id = String::new();
    let mut orderings = Vec::with_capacity(opts.orderings.len());
    for &order in &opts.orderings {
        let list = list_text(instrument, order);
        let perm = order.permutation(items.len());
        let mut passes: Vec<Option<ItemPass>> = vec![None; n_rated];
        for (slot, &idx) in perm.iter().enumerate() {
            if idx >= n_rated {
                continue;
            }
            let (slug, text) = items[idx];
            let prompt = item_prompt(&base, &list, query_text(instrument, slot, text));
            let response = sender
                .send(&prompt)
                .await
                .map_err(|source| ProbeError::Send {
                    item: slug.to_string(),
                    order,
                    source,
                })?;
            model_id = response.model.to_string();
            let sampled = parse_answer(&response).map_err(|detail| ProbeError::Answer {
                item: slug.to_string(),
                order,
                detail,
            })?;
            let request_id = Uuid::parse_str(&response.id).ok();
            let snapshots = match sessions.as_deref_mut() {
                Some(src) => {
                    let id = request_id.ok_or_else(|| ProbeError::NoRequestId {
                        item: slug.to_string(),
                        order,
                        id: response.id.to_string(),
                    })?;
                    src.take_session(id, opts.session_timeout)
                        .await
                        .map_err(|source| ProbeError::Session {
                            item: slug.to_string(),
                            order,
                            source,
                        })?
                        .snapshots
                }
                None => Vec::new(),
            };
            let counts = response.usage.counts;
            tracing::debug!(
                item = %slug,
                %order,
                slot = slot + 1,
                sampled = sampled.get(),
                cache_read = counts.cache_read_input_tokens.unwrap_or(0),
                "probe item answered"
            );
            passes[idx] = Some(ItemPass {
                item: slug.clone(),
                slot: (slot + 1) as u32,
                sampled,
                request_id,
                reading: None,
                input_tokens: counts.input_tokens,
                cache_read_input_tokens: counts.cache_read_input_tokens.unwrap_or(0),
                output_tokens: counts.output_tokens,
                snapshots,
            });
        }
        orderings.push(OrderingPass {
            order,
            items: passes
                .into_iter()
                .map(|p| p.expect("every rated item has a slot"))
                .collect(),
        });
    }

    let digit_vocab = if measure == Measure::Distribution {
        for pass in orderings.iter().flat_map(|o| &o.items) {
            vocab.observe_all(&pass.snapshots)?;
        }
        vocab.ensure_complete()?;
        for o in &mut orderings {
            for pass in &mut o.items {
                let reading = read_rating(&pass.snapshots, RATING_KEY, &vocab, opts.max_residual)
                    .map_err(|source| ProbeError::Read {
                    item: pass.item.to_string(),
                    order: o.order,
                    source,
                })?;
                if reading.sampled != Some(pass.sampled) {
                    return Err(ProbeError::SampleMismatch {
                        item: pass.item.to_string(),
                        order: o.order,
                        sampled: pass.sampled,
                        piece: reading.sampled_piece.clone(),
                    });
                }
                pass.reading = Some(reading);
            }
        }
        Some(vocab)
    } else {
        None
    };

    Ok(DistributionRun {
        model_id,
        instrument_key: instrument.baseline_key(),
        measure,
        probed_at: Utc::now(),
        max_residual: opts.max_residual,
        orderings,
        digit_vocab,
    })
}

fn parse_answer(response: &misanthropic::response::Message) -> Result<RatingDigit, String> {
    let wire: SingleRatingWire = response.json().map_err(|e| e.to_string())?;
    wire.rating
        .parse::<u8>()
        .ok()
        .and_then(RatingDigit::new)
        .ok_or_else(|| format!("rating {:?} is not 1..={RATING_MAX}", wire.rating))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use super::super::distribution::tests::snap;
    use super::super::questionnaire::{Questionnaire, QuestionnaireItem};
    use super::super::snapshot::TokenId;
    use super::*;

    fn questionnaire(n: usize) -> Questionnaire {
        Questionnaire {
            version: "test".into(),
            description: String::new(),
            instructions: "Rate these.".into(),
            items: (0..n)
                .map(|i| QuestionnaireItem {
                    id: format!("item_{i}").as_str().into(),
                    statement: format!("Statement number {i}."),
                })
                .collect(),
        }
    }

    /// A fake blallama: answers each item request with a fixed rating
    /// per item (looked up by the statement text in the query block),
    /// and records a `/probe` session whose digit position carries a
    /// distribution chosen per (item, display slot), so order effects
    /// can be simulated.
    type TopAtSlot = Box<dyn Fn(usize) -> Vec<(u32, f32)> + Send + Sync>;

    struct FakeBlallama {
        /// statement text → (sampled digit, top-k at the digit, by slot)
        behaviour: HashMap<String, (u8, TopAtSlot)>,
        sessions: Mutex<HashMap<Uuid, CompletedSession>>,
        sent: Mutex<Vec<serde_json::Value>>,
    }

    fn qwen_digit(d: u8) -> u32 {
        15 + u32::from(d)
    }

    fn token_snap(gi: u32, id: u32, piece: &str, top: &[(u32, f32)]) -> ProbeSnapshot {
        let mut s = snap(gi, piece, top);
        s.token = Some(TokenId(id));
        s
    }

    impl FakeBlallama {
        fn respond(&self, prompt: &CachedPrompt) -> misanthropic::response::Message {
            let json = serde_json::to_value(prompt).unwrap();
            self.sent.lock().unwrap().push(json.clone());
            let id = Uuid::new_v4();
            let schema = &json["output_config"]["format"]["schema"];
            let (text, snapshots) = if schema["properties"].get("d1").is_some() {
                // Calibration: emit every digit, forced.
                let mut snaps = vec![token_snap(0, 90, "{", &[(90, 1.0)])];
                let mut text = String::from("{");
                for d in 1..=7u8 {
                    let gi = u32::from(d) * 2;
                    snaps.push(token_snap(gi - 1, 1000, "\"d\": \"", &[(1000, 1.0)]));
                    snaps.push(token_snap(gi, qwen_digit(d), &d.to_string(), &[(5, 0.9)]));
                    text.push_str(&format!("\"d{d}\": \"{d}\","));
                }
                (text, snaps)
            } else {
                let user = &json["messages"][0]["content"];
                let query = user[1]["text"].as_str().unwrap();
                let slot: usize = query
                    .trim_start_matches("Rate statement ")
                    .split(':')
                    .next()
                    .unwrap()
                    .parse::<usize>()
                    .unwrap()
                    - 1;
                let statement = query.split_once(": ").unwrap().1;
                let (d, top) = &self.behaviour[statement];
                let snaps = vec![
                    token_snap(0, 4913, "{\"", &[(4913, 1.0)]),
                    token_snap(1, 8, "rating", &[(8, 1.0)]),
                    token_snap(2, 788, "\":", &[(788, 1.0)]),
                    token_snap(3, 330, " \"", &[(330, 0.9)]),
                    token_snap(4, qwen_digit(*d), &d.to_string(), &top(slot)),
                    token_snap(5, 9, "\"}", &[(9, 1.0)]),
                ];
                (format!("{{\"rating\": \"{d}\"}}"), snaps)
            };
            self.sessions.lock().unwrap().insert(
                id,
                CompletedSession {
                    id,
                    model: Some("fake.gguf".into()),
                    snapshots,
                },
            );
            serde_json::from_value(serde_json::json!({
                "id": id.to_string(),
                "type": "message",
                "role": "assistant",
                "model": "fake.gguf",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 100, "output_tokens": 6, "cache_read_input_tokens": 80}
            }))
            .unwrap()
        }
    }

    impl MessageSender for FakeBlallama {
        async fn send(
            &self,
            prompt: &CachedPrompt,
        ) -> Result<misanthropic::response::Message, BoxError> {
            Ok(self.respond(prompt))
        }
    }

    struct FakeStream<'a>(&'a FakeBlallama);

    impl SessionSource for FakeStream<'_> {
        async fn take_session(
            &mut self,
            id: Uuid,
            _timeout: Duration,
        ) -> Result<CompletedSession, BoxError> {
            self.0
                .sessions
                .lock()
                .unwrap()
                .remove(&id)
                .ok_or_else(|| "no such session".into())
        }
    }

    fn fake(q: &Questionnaire) -> FakeBlallama {
        let mut behaviour: HashMap<String, (u8, TopAtSlot)> = HashMap::new();
        for (i, item) in q.items.iter().enumerate() {
            let b: TopAtSlot = if i == 0 {
                // Order-sensitive: 7-heavy when shown first, 6-heavy
                // otherwise.
                Box::new(|slot| {
                    if slot == 0 {
                        vec![(22, 0.8), (21, 0.2)]
                    } else {
                        vec![(22, 0.2), (21, 0.8)]
                    }
                })
            } else {
                Box::new(|_| vec![(19, 0.6), (20, 0.38), (999, 0.02)])
            };
            let sampled = if i == 0 { 7 } else { 4 };
            behaviour.insert(item.statement.clone(), (sampled, b));
        }
        FakeBlallama {
            behaviour,
            sessions: Mutex::new(HashMap::new()),
            sent: Mutex::new(Vec::new()),
        }
    }

    #[test]
    fn orderings_are_permutations_and_pairs_flip() {
        assert_eq!(ItemOrder::Forward.permutation(3), vec![0, 1, 2]);
        assert_eq!(ItemOrder::Reverse.permutation(3), vec![2, 1, 0]);
        // Items 7 and 8 (Palestinians, Israelis in v0): adjacent, and
        // in opposite display order across the two passes.
        let pos = |o: ItemOrder, idx: usize| o.permutation(16).iter().position(|&i| i == idx);
        assert!(pos(ItemOrder::Forward, 7) < pos(ItemOrder::Forward, 8));
        assert!(pos(ItemOrder::Reverse, 7) > pos(ItemOrder::Reverse, 8));
    }

    #[test]
    fn schemas_are_inline_and_closed() {
        for schema in [single_item_schema(), calibration_schema()] {
            assert_eq!(schema["additionalProperties"], false);
            let rendered = schema.to_string();
            for banned in ["$ref", "$defs", "pattern", "reasoning"] {
                assert!(!rendered.contains(&format!("\"{banned}\"")), "{rendered}");
            }
        }
        let s = single_item_schema();
        assert_eq!(s["required"], serde_json::json!([RATING_KEY]));
        assert_eq!(
            s["properties"][RATING_KEY]["enum"],
            serde_json::json!(["1", "2", "3", "4", "5", "6", "7"])
        );
    }

    #[test]
    fn prompts_share_a_cached_prefix_per_ordering() {
        let q = questionnaire(3);
        let base = base_prompt("m", &q, NonZeroU32::new(24).unwrap());
        let list = list_text(&q, ItemOrder::Reverse);
        assert!(list.starts_with("Statements:\n\n1. Statement number 2."));
        let a = serde_json::to_value(item_prompt(&base, &list, query_text(&q, 0, "x"))).unwrap();
        let b = serde_json::to_value(item_prompt(&base, &list, query_text(&q, 2, "y"))).unwrap();
        // Same system, same cached list block; only the query differs.
        assert_eq!(a["system"], b["system"]);
        assert_eq!(
            a["messages"][0]["content"][0],
            b["messages"][0]["content"][0]
        );
        assert!(a["messages"][0]["content"][0]["cache_control"].is_object());
        assert_ne!(
            a["messages"][0]["content"][1],
            b["messages"][0]["content"][1]
        );
        assert_eq!(
            a["messages"][0]["content"][1]["text"],
            "Rate statement 1: x"
        );
        assert!(a["output_config"]["format"]["schema"]["properties"][RATING_KEY].is_object());
    }

    #[tokio::test]
    async fn distribution_run_reads_every_item_in_every_ordering() {
        let q = questionnaire(3);
        let fake = fake(&q);
        let mut stream = FakeStream(&fake);
        let run = run_distribution_probe(
            &fake,
            Some(&mut stream),
            &q,
            "fake.gguf",
            &DistributionOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(run.measure, Measure::Distribution);
        assert_eq!(run.model_id, "fake.gguf");
        // 1 calibration + 3 items × 2 orderings.
        assert_eq!(fake.sent.lock().unwrap().len(), 7);
        assert_eq!(run.orderings.len(), 2);
        let rev = run.ordering(ItemOrder::Reverse).unwrap();
        assert_eq!(rev.items[0].item, "item_0");
        assert_eq!(rev.items[0].slot, 3);
        let s = run.summaries();
        // Item 0 is order-sensitive: EV 6.8 shown first, 6.2 shown last.
        assert!((s[0].expected_by_order[0] - 6.8).abs() < 1e-6);
        assert!((s[0].expected_by_order[1] - 6.2).abs() < 1e-6);
        assert!((s[0].order_effect.unwrap() - 0.6).abs() < 1e-6);
        let pooled = s[0].pooled.as_ref().unwrap();
        assert!((pooled.p[6] - 0.5).abs() < 1e-6);
        // Item 1 is order-stable; residual recorded.
        assert!(s[1].order_effect.unwrap().abs() < 1e-9);
        let r = run.orderings[0].items[1].reading.as_ref().unwrap();
        assert!((r.distribution.residual_mass - 0.02).abs() < 1e-6);
        assert_eq!(r.distribution.mode().0.get(), 4);
        assert!((run.max_order_effect().unwrap() - 0.6).abs() < 1e-6);
        // Digit vocab learned from calibration, not hard-coded.
        let vocab = run.digit_vocab.as_ref().unwrap();
        assert_eq!(vocab.digit(TokenId(22)).unwrap().get(), 7);
    }

    #[tokio::test]
    async fn external_only_run_has_no_readings() {
        let q = questionnaire(2);
        let fake = fake(&q);
        let run = run_distribution_probe::<_, FakeStream<'_>, _>(
            &fake,
            None,
            &q,
            "claude-x",
            &DistributionOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(run.measure, Measure::ExternalOnly);
        // No calibration request without a stream.
        assert_eq!(fake.sent.lock().unwrap().len(), 4);
        assert!(run.digit_vocab.is_none());
        let s = run.summaries();
        assert!(
            s.iter()
                .all(|s| s.pooled.is_none() && s.order_effect.is_none())
        );
        assert_eq!(s[0].sampled.len(), 2);
    }

    #[tokio::test]
    async fn item_limit_rates_a_prefix_but_lists_everything() {
        let q = questionnaire(4);
        let fake = fake(&q);
        let mut stream = FakeStream(&fake);
        let opts = DistributionOptions {
            item_limit: Some(2),
            ..DistributionOptions::default()
        };
        let run = run_distribution_probe(&fake, Some(&mut stream), &q, "f", &opts)
            .await
            .unwrap();
        assert_eq!(run.orderings[0].items.len(), 2);
        let sent = fake.sent.lock().unwrap();
        assert_eq!(sent.len(), 5);
        let list = sent[1]["messages"][0]["content"][0]["text"]
            .as_str()
            .unwrap();
        assert!(list.contains("4. Statement number 3."));
    }

    #[tokio::test]
    async fn missing_session_fails_loudly() {
        struct Empty;
        impl SessionSource for Empty {
            async fn take_session(
                &mut self,
                _id: Uuid,
                _t: Duration,
            ) -> Result<CompletedSession, BoxError> {
                Err("stream dropped".into())
            }
        }
        let q = questionnaire(1);
        let fake = fake(&q);
        let err = run_distribution_probe(&fake, Some(&mut Empty), &q, "f", &Default::default())
            .await
            .unwrap_err();
        assert!(matches!(err, ProbeError::Calibration(_)), "{err}");
    }
}
