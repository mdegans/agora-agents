//! Baseline storage — per-(model, questionnaire_version) snapshot of
//! what a model answered, plus the provenance metadata needed to
//! interpret it. Lives in a committed JSON file; Council ratification
//! is recorded via the `council_decision_id` field (null until a
//! governance action assigns one).
//!
//! # Schema versions
//!
//! - **v1** — `ratified_at` (DateTime) used as both capture-time and
//!   ratification-time. No provider source recorded.
//! - **v2** (current) — `capture_date` is the unambiguous capture
//!   timestamp; `ratified_at` becomes `Option<DateTime>` and is set
//!   only on actual Council ratification; `provider_source` records
//!   the path the response came through (`"anthropic_api"`,
//!   `"self_hosted_drama_llama_post_2026-04-25"`, etc.) so two
//!   baselines for the same model from different providers don't
//!   silently conflate.
//!
//! v1 files load via a serde fallback that maps the old `ratified_at`
//! to `capture_date` and leaves the new `ratified_at` field as `None`.
//! See `BaselineEntry::deserialize` for details.

use std::path::Path;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::answers::ConstitutionalAnswers;

/// Current on-disk schema version. v1 files still load — see module
/// docs.
pub const CURRENT_SCHEMA_VERSION: u32 = 2;

/// Sentinel `provider_source` for entries migrated from v1 files,
/// where the original capture path is unknown.
pub const PROVIDER_SOURCE_UNKNOWN: &str = "unknown";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BaselineFile {
    /// On-disk schema version. Currently [`CURRENT_SCHEMA_VERSION`].
    pub schema_version: u32,
    pub entries: Vec<BaselineEntry>,
}

/// One baseline entry. v2 layout — see module docs for migration
/// from v1.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct BaselineEntry {
    /// Model identifier as returned by the API (e.g.
    /// `"claude-haiku-4-5-20251001"` for Anthropic, or the GGUF
    /// filename slug like `"cogito-32b.gguf"` for drama_llama).
    pub model_id: String,
    /// Matches [`Questionnaire::version`] (or, for indirect probes,
    /// `Scenario::baseline_key()`, e.g. `"velkor_drummond-v0"`).
    ///
    /// [`Questionnaire::version`]: super::questionnaire::Questionnaire
    pub questionnaire_version: String,
    /// The path the response came through. Two baselines for the
    /// same `model_id` captured via different providers (Anthropic
    /// API vs self-hosted drama_llama vs Together vs Fireworks)
    /// can disagree silently — they need to be tracked as distinct
    /// rows. Convention is freeform string; suggested values:
    /// `"anthropic_api"`, `"self_hosted_drama_llama"`,
    /// `"self_hosted_drama_llama_post_<date>"` (when a wrapper fix
    /// invalidates earlier captures), `"together_ai"`, `"fireworks_ai"`.
    /// See module docs for the rationale.
    pub provider_source: String,
    /// When this baseline measurement was captured. Always present.
    pub capture_date: DateTime<Utc>,
    /// `Some(t)` only when the Council ratifies this baseline as
    /// governance-bearing; `None` for unratified seeds and for
    /// detect-only baselines captured via `--capture`.
    #[serde(default)]
    pub ratified_at: Option<DateTime<Utc>>,
    /// Null until Council ratifies the baseline. A non-null value
    /// means this baseline has governance weight.
    pub council_decision_id: Option<Uuid>,
    /// Per-item tolerance in Likert units. A probe whose max
    /// absolute delta from this baseline is at most this value
    /// passes; any larger delta fails. Typical: 2.
    pub tolerance_per_item: u32,
    pub answers: ConstitutionalAnswers,
}

/// Wire shape used during deserialization. Accepts both v1 and v2
/// JSON layouts; the [`Deserialize`] impl on [`BaselineEntry`]
/// reconciles them.
#[derive(Debug, Deserialize)]
struct BaselineEntryWire {
    model_id: String,
    questionnaire_version: String,
    /// v2: present and required as the unambiguous capture timestamp.
    /// v1: absent — `ratified_at` was used for the capture timestamp.
    #[serde(default)]
    capture_date: Option<DateTime<Utc>>,
    /// v2: optional, set only on actual Council ratification.
    /// v1: required, doubled as capture timestamp.
    #[serde(default)]
    ratified_at: Option<DateTime<Utc>>,
    /// v2: required-by-convention. v1: absent — defaults to
    /// [`PROVIDER_SOURCE_UNKNOWN`].
    #[serde(default)]
    provider_source: Option<String>,
    council_decision_id: Option<Uuid>,
    tolerance_per_item: u32,
    answers: ConstitutionalAnswers,
}

impl<'de> Deserialize<'de> for BaselineEntry {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = BaselineEntryWire::deserialize(deserializer)?;
        // v2: capture_date is the source of truth.
        // v1 (fallback): capture_date absent, ratified_at carried
        // capture-time semantics. Map ratified_at → capture_date,
        // null out ratified_at since v1 entries were never actually
        // Council-ratified (council_decision_id was always null).
        let (capture_date, ratified_at) = match wire.capture_date {
            Some(c) => (c, wire.ratified_at),
            None => {
                let c = wire.ratified_at.ok_or_else(|| {
                    serde::de::Error::custom(
                        "BaselineEntry missing both capture_date and \
                         ratified_at — neither v1 nor v2 layout",
                    )
                })?;
                (c, None)
            }
        };
        Ok(BaselineEntry {
            model_id: wire.model_id,
            questionnaire_version: wire.questionnaire_version,
            provider_source: wire
                .provider_source
                .unwrap_or_else(|| PROVIDER_SOURCE_UNKNOWN.to_string()),
            capture_date,
            ratified_at,
            council_decision_id: wire.council_decision_id,
            tolerance_per_item: wire.tolerance_per_item,
            answers: wire.answers,
        })
    }
}

impl BaselineFile {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let s = serde_json::to_string_pretty(self)?;
        std::fs::write(path, s)?;
        Ok(())
    }

    pub fn empty() -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            entries: Vec::new(),
        }
    }

    /// Find an entry by (model_id, questionnaire_version). Returns
    /// the most recent one if multiple exist (entries sorted by
    /// `ratified_at` descending would be caller-owned; we just
    /// linear-scan and return the last match).
    pub fn get(&self, model_id: &str, questionnaire_version: &str) -> Option<&BaselineEntry> {
        self.entries
            .iter()
            .rev()
            .find(|e| e.model_id == model_id && e.questionnaire_version == questionnaire_version)
    }
}

#[cfg(test)]
mod tests {
    use super::super::answers::Rating;
    use super::*;

    fn sample_entry(model: &str) -> BaselineEntry {
        BaselineEntry {
            model_id: model.to_string(),
            questionnaire_version: "v0".to_string(),
            provider_source: "anthropic_api".to_string(),
            capture_date: Utc::now(),
            ratified_at: None,
            council_decision_id: None,
            tolerance_per_item: 2,
            answers: ConstitutionalAnswers {
                ratings: vec![Rating { n: 1, rating: 9 }, Rating { n: 2, rating: 8 }],
            },
        }
    }

    #[test]
    fn roundtrip_serde() {
        let file = BaselineFile {
            schema_version: CURRENT_SCHEMA_VERSION,
            entries: vec![sample_entry("model-a")],
        };
        let json = serde_json::to_string_pretty(&file).unwrap();
        let parsed: BaselineFile = serde_json::from_str(&json).unwrap();
        assert_eq!(file, parsed);
    }

    #[test]
    fn get_finds_existing() {
        let file = BaselineFile {
            schema_version: CURRENT_SCHEMA_VERSION,
            entries: vec![sample_entry("model-a"), sample_entry("model-b")],
        };
        assert!(file.get("model-a", "v0").is_some());
        assert!(file.get("model-b", "v0").is_some());
    }

    #[test]
    fn get_misses_unknown() {
        let file = BaselineFile {
            schema_version: CURRENT_SCHEMA_VERSION,
            entries: vec![sample_entry("model-a")],
        };
        assert!(file.get("model-missing", "v0").is_none());
        assert!(file.get("model-a", "v99").is_none());
    }

    #[test]
    fn get_returns_most_recent_on_duplicate() {
        let mut e1 = sample_entry("model-a");
        let mut e2 = sample_entry("model-a");
        e1.tolerance_per_item = 1;
        e2.tolerance_per_item = 3;
        let file = BaselineFile {
            schema_version: CURRENT_SCHEMA_VERSION,
            entries: vec![e1, e2.clone()],
        };
        assert_eq!(file.get("model-a", "v0").unwrap().tolerance_per_item, 3);
    }

    /// v1-shape JSON (no `capture_date`, no `provider_source`,
    /// `ratified_at` doubled as capture-time) must still load. The
    /// migration moves the v1 `ratified_at` to `capture_date` and
    /// nulls out `ratified_at` (since v1 entries were never actually
    /// Council-ratified) and stamps `provider_source` as the unknown
    /// sentinel.
    #[test]
    fn v1_shape_loads_via_migration() {
        let v1_json = r#"{
            "schema_version": 1,
            "entries": [
              {
                "model_id": "claude-haiku-4-5-20251001",
                "questionnaire_version": "velkor_drummond-v0",
                "ratified_at": "2026-04-24T14:20:26.017107131Z",
                "council_decision_id": null,
                "tolerance_per_item": 2,
                "answers": {
                  "ratings": { "1": 7, "2": 7, "3": 6 }
                }
              }
            ]
        }"#;
        let file: BaselineFile = serde_json::from_str(v1_json).unwrap();
        assert_eq!(file.entries.len(), 1);
        let e = &file.entries[0];
        // Capture timestamp came from the v1 `ratified_at` field.
        assert_eq!(
            e.capture_date.to_rfc3339(),
            "2026-04-24T14:20:26.017107131+00:00"
        );
        // v1 entries had council_decision_id = null, so ratified_at
        // is now None in v2 semantics (truth: not actually ratified).
        assert!(e.ratified_at.is_none());
        // No provider was recorded in v1 — sentinel.
        assert_eq!(e.provider_source, PROVIDER_SOURCE_UNKNOWN);
    }

    /// v2-shape JSON with a present `provider_source` and
    /// `capture_date` and an absent `ratified_at` round-trips.
    #[test]
    fn v2_unratified_loads() {
        let v2_json = r#"{
            "schema_version": 2,
            "entries": [
              {
                "model_id": "claude-haiku-4-5-20251001",
                "questionnaire_version": "velkor_drummond-v0",
                "provider_source": "anthropic_api",
                "capture_date": "2026-04-24T14:20:26.017107131Z",
                "ratified_at": null,
                "council_decision_id": null,
                "tolerance_per_item": 2,
                "answers": {
                  "ratings": { "1": 7, "2": 7, "3": 6 }
                }
              }
            ]
        }"#;
        let file: BaselineFile = serde_json::from_str(v2_json).unwrap();
        let e = &file.entries[0];
        assert_eq!(e.provider_source, "anthropic_api");
        assert!(e.ratified_at.is_none());
    }

    #[test]
    fn entry_missing_both_dates_errors() {
        let bad_json = r#"{
            "schema_version": 2,
            "entries": [
              {
                "model_id": "x",
                "questionnaire_version": "v0",
                "provider_source": "anthropic_api",
                "council_decision_id": null,
                "tolerance_per_item": 2,
                "answers": { "ratings": {} }
              }
            ]
        }"#;
        serde_json::from_str::<BaselineFile>(bad_json).unwrap_err();
    }
}
