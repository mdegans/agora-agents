//! Alignment-drift canary probe.
//!
//! Sends a Council-ratified constitutional questionnaire to a model
//! endpoint via Anthropic-shaped structured output, and returns a
//! typed [`ProbeOutcome`]. Pass/fail against a ratified baseline is
//! computed separately by [`evaluate`] — keeping measurement and
//! judgment separate lets this primitive be used for baseline capture
//! before any baseline exists.
//!
//! Design rationale in `project_alignment_drift_canary.md` in the
//! main agora repo. Threat model: silent upstream alignment drift
//! (dataset filtering, retrained priors on sensitive topics) shipped
//! under the same API. Detection via per-(model, version) baselines
//! whose ratings we compare against future probe outputs.
//!
//! The probe is **not** a full evaluation system on its own — it's
//! the client-side primitive. Scheduler integration, DB persistence,
//! and governance-log wiring are downstream consumers that live
//! elsewhere.

mod answers;
mod baseline;
pub mod indirect;
mod questionnaire;
mod report;
mod score;
pub mod snapshot;
pub mod stream;

pub use answers::{ConstitutionalAnswers, RATING_MAX, Rating, build_schema};
pub use baseline::{BaselineEntry, BaselineFile, CURRENT_SCHEMA_VERSION, PROVIDER_SOURCE_UNKNOWN};
pub use questionnaire::{Questionnaire, QuestionnaireItem, constitutional_v0};
pub use report::{ProbeReport, evaluate};
pub use score::{Score, score};
pub use snapshot::{ProbeEvent, ProbeSnapshot, TokenSnapshot, TopKEntry};
pub use stream::{CompletedSession, ProbeStreamConsumer, probe_url_from_endpoint};

use std::num::NonZeroU32;

use chrono::{DateTime, Utc};
use misanthropic::{Client, Prompt, prompt::message::Role};
use uuid::Uuid;

/// Result of a single probe run — the typed answer plus metadata.
#[derive(Debug, Clone)]
pub struct ProbeOutcome {
    pub answers: ConstitutionalAnswers,
    pub usage: misanthropic::response::Usage,
    /// Model id as reported by the server. For Anthropic this is
    /// the Anthropic model slug; for drama_llama it's whatever the
    /// server returns (typically the GGUF internal name, not the
    /// filename the caller requested).
    pub model_id: String,
    /// Server-issued request id. For blallama (slice-2A onward) this
    /// is a UUID matching the `id` field of `/probe` SSE events,
    /// enabling cross-validation joins between external rating and
    /// internal pre-grammar snapshot. For native Anthropic API this
    /// will be `None` (Anthropic uses a `msg_…`-prefixed id, not a
    /// raw UUID).
    pub request_id: Option<Uuid>,
    pub probed_at: DateTime<Utc>,
}

/// Default cap on response tokens. The probe's structured output
/// is ~50 tokens for 16 items; 512 caps cost on a bad generation
/// without truncating a legitimate answer.
const PROBE_MAX_TOKENS: u32 = 512;

/// Send `questionnaire` to `client` using `model`, parse the typed
/// response, and return a [`ProbeOutcome`].
///
/// Does NOT compare against a baseline — call [`evaluate`] for that.
///
/// Uses Anthropic-shape structured output (`output_config`), which
/// works against Anthropic-native endpoints and against drama_llama
/// (via `client.with_base_url(...)` set by the caller).
///
/// Extended thinking is deliberately NOT enabled: raw rating is the
/// signal we want, not the model's rationalization of it.
pub async fn probe<M>(
    client: &Client,
    questionnaire: &Questionnaire,
    model: M,
) -> anyhow::Result<ProbeOutcome>
where
    M: Into<misanthropic::model::Id<'static>>,
{
    use anyhow::Context as _;

    let max_tokens = NonZeroU32::new(PROBE_MAX_TOKENS).unwrap();

    // Hand-built schema with enum-constrained `n` and `rating` —
    // see `answers::build_schema` for why we bypass schemars here.
    let schema = answers::build_schema(questionnaire.items.len());

    let prompt = Prompt::default()
        .model(model)
        .max_tokens(max_tokens)
        .json_schema(schema)
        .set_system(questionnaire.system_prompt())
        .add_message((Role::User, questionnaire.user_message()))
        .context("assembling probe prompt")?;

    let response = client
        .message(&prompt)
        .await
        .context("probe API call failed")?;

    let model_id = response.model.to_string();
    let usage = response.usage;
    let request_id = Uuid::parse_str(&response.id).ok();

    let raw: ConstitutionalAnswers = response
        .json()
        .context("probe response parse failed — model may have refused, returned no text block, or emitted non-schema JSON")?;

    let answers = raw.validate_and_sort(questionnaire.items.len())?;

    Ok(ProbeOutcome {
        answers,
        usage,
        model_id,
        request_id,
        probed_at: Utc::now(),
    })
}
