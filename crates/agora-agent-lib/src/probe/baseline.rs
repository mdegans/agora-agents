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
//! - **v2** — `capture_date` is the unambiguous capture timestamp;
//!   `ratified_at` becomes `Option<DateTime>` and is set only on
//!   actual Council ratification; `provider_source` records the path
//!   the response came through (`"anthropic_api"`,
//!   `"self_hosted_drama_llama_post_2026-04-25"`, etc.) so two
//!   baselines for the same model from different providers don't
//!   silently conflate.
//! - **v3** — adds optional `snapshot_path`: a sidecar
//!   reference to a JSONL file holding the per-token pre-grammar
//!   snapshot stream captured from blallama's `/probe` SSE endpoint.
//!   Stored separately from the entry to keep the baseline JSON
//!   readable; cross-validation tooling joins on
//!   (model_id, questionnaire_version, capture_date, request_id).
//!
//! - **v4** (current, 2026-10-14) — adds optional `per_item`: the
//!   deterministic per-item probe's readings, per ordering, each item
//!   as `{item, sampled, distribution: {p: [7], residual_mass}}`, plus
//!   the [`Measure`] they are (distribution or external-only samples)
//!   and the EV tolerance. `answers` still holds sampled integers (the
//!   first ordering's) so integer tooling keeps working. An entry
//!   without `per_item` is a legacy single-generation baseline; it is
//!   never compared against a per-item run (see [`evaluate_run`]).
//!
//! v1–v3 files load via serde fallback. See `BaselineEntry::deserialize`
//! for details.

use std::path::Path;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::answers::{ConstitutionalAnswers, Rating};
use super::deterministic::{DistributionRun, ItemOrder, Measure};
use super::digits::RatingDigit;
use super::distribution::RatingDistribution;
use super::drift::{DistributionReport, DriftError, DriftPolicy, compare_run};
use super::ids::ItemSlug;
use super::score::{Score, score};
use super::thinking::ThinkingMode;

/// Current on-disk schema version. v1–v3 files still load — see module
/// docs.
pub const CURRENT_SCHEMA_VERSION: u32 = 4;

/// The per-item probe's readings stored on a v4 baseline entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerItemBaseline {
    pub measure: Measure,
    /// Thinking on or off when captured. Baselines of different modes
    /// are never compared.
    #[serde(default)]
    pub thinking: ThinkingMode,
    /// Residual-mass ceiling the readings were taken under.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_residual: Option<f64>,
    pub orderings: Vec<OrderingRecord>,
}

/// One ordering's items, canonical order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderingRecord {
    pub order: ItemOrder,
    pub items: Vec<ItemRecord>,
}

/// One item: the sampled rating, and the distribution when there is one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ItemRecord {
    pub item: ItemSlug,
    pub sampled: RatingDigit,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distribution: Option<RatingDistribution>,
}

impl PerItemBaseline {
    /// Per-ordering distributions, for [`compare_run`]. Orderings with
    /// any item missing a distribution are skipped.
    pub fn distributions(&self) -> Vec<(ItemOrder, Vec<(ItemSlug, RatingDistribution)>)> {
        self.orderings
            .iter()
            .filter_map(|o| {
                o.items
                    .iter()
                    .map(|i| i.distribution.clone().map(|d| (i.item.clone(), d)))
                    .collect::<Option<Vec<_>>>()
                    .map(|v| (o.order, v))
            })
            .collect()
    }
}

/// How a per-item run compared with a baseline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "measure", rename_all = "snake_case")]
pub enum RunComparison {
    /// Distribution against distribution.
    Distribution(DistributionReport),
    /// External-only samples against external-only samples: integer
    /// deltas per ordering, under the old per-item integer tolerance.
    /// **Weaker evidence**: one sample per item per ordering, so a pass
    /// does not show the pipeline unchanged, and a 1-point move can be
    /// sampling alone.
    ExternalOnly {
        per_ordering: Vec<(ItemOrder, Score)>,
        tolerance_per_item: u32,
        pass: bool,
    },
}

impl RunComparison {
    pub fn pass(&self) -> bool {
        match self {
            RunComparison::Distribution(r) => r.pass,
            RunComparison::ExternalOnly { pass, .. } => *pass,
        }
    }
}

/// Compare a per-item run with a baseline entry of the same measure.
/// A legacy (single-generation) entry, or one of the other measure, is
/// an error: those are different measurements.
pub fn evaluate_run(
    run: &DistributionRun,
    entry: &BaselineEntry,
    policy: &DriftPolicy,
) -> Result<RunComparison, DriftError> {
    if entry.questionnaire_version != run.instrument_key {
        return Err(DriftError::KeyMismatch {
            measured: run.instrument_key.clone(),
            baseline: entry.questionnaire_version.clone(),
        });
    }
    let per_item = match &entry.per_item {
        Some(p) if p.measure == run.measure => p,
        other => {
            return Err(DriftError::MeasureMismatch {
                measured: Some(run.measure),
                baseline: other.as_ref().map(|p| p.measure),
            });
        }
    };
    if per_item.thinking != run.thinking {
        return Err(DriftError::ThinkingMismatch {
            measured: run.thinking.clone(),
            baseline: per_item.thinking.clone(),
        });
    }
    match run.measure {
        Measure::Distribution => {
            compare_run(run, &per_item.distributions(), policy).map(RunComparison::Distribution)
        }
        Measure::ExternalOnly => {
            let mut per_ordering = Vec::new();
            for o in &run.orderings {
                let Some(base) = per_item.orderings.iter().find(|b| b.order == o.order) else {
                    continue;
                };
                let to_answers = |items: &mut dyn Iterator<Item = (ItemSlug, RatingDigit)>| {
                    let mut v: Vec<(ItemSlug, RatingDigit)> = items.collect();
                    v.sort_by(|a, b| a.0.cmp(&b.0));
                    v
                };
                let m = to_answers(&mut o.items.iter().map(|i| (i.item.clone(), i.sampled)));
                let b = to_answers(&mut base.items.iter().map(|i| (i.item.clone(), i.sampled)));
                let slugs = |v: &[(ItemSlug, RatingDigit)]| {
                    v.iter().map(|x| x.0.clone()).collect::<Vec<_>>()
                };
                if slugs(&m) != slugs(&b) {
                    return Err(DriftError::ItemMismatch {
                        measured: slugs(&m),
                        baseline: slugs(&b),
                    });
                }
                let answers = |v: &[(ItemSlug, RatingDigit)]| ConstitutionalAnswers {
                    ratings: v
                        .iter()
                        .enumerate()
                        .map(|(i, (_, r))| Rating {
                            n: (i + 1) as u32,
                            rating: u32::from(r.get()),
                        })
                        .collect(),
                };
                let s =
                    score(&answers(&m), &answers(&b)).expect("same length and n by construction");
                per_ordering.push((o.order, s));
            }
            if per_ordering.is_empty() {
                return Err(DriftError::MissingOrdering(
                    run.orderings
                        .first()
                        .map_or(ItemOrder::Forward, |o| o.order),
                ));
            }
            let pass = per_ordering
                .iter()
                .all(|(_, s)| s.max_abs_delta <= entry.tolerance_per_item);
            Ok(RunComparison::ExternalOnly {
                per_ordering,
                tolerance_per_item: entry.tolerance_per_item,
                pass,
            })
        }
    }
}

/// Sentinel `provider_source` for entries migrated from v1 files,
/// where the original capture path is unknown.
pub const PROVIDER_SOURCE_UNKNOWN: &str = "unknown";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BaselineFile {
    /// On-disk schema version. Currently [`CURRENT_SCHEMA_VERSION`].
    pub schema_version: u32,
    pub entries: Vec<BaselineEntry>,
}

/// One baseline entry. v2 layout — see module docs for migration
/// from v1.
#[derive(Debug, Clone, Serialize, PartialEq)]
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
    /// Sidecar file holding per-token pre-grammar snapshots from
    /// blallama's `/probe` SSE endpoint, captured at the same time
    /// as `answers`. Path is relative to the baseline file's parent
    /// directory. Only present when the probe was run against a
    /// blallama-compatible endpoint with `--probe-stream-endpoint`
    /// active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_path: Option<String>,
    /// Server-issued request id matching the probe-stream session
    /// id, when available. Joinable with `snapshot_path` for the
    /// cross-validation primitive (rate observed externally vs
    /// pre-grammar mass observed internally on the same row).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<Uuid>,
    /// v4: the per-item probe's readings. `None` on legacy
    /// single-generation entries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_item: Option<PerItemBaseline>,
}

impl BaselineEntry {
    /// A v4 entry from a per-item run. `answers` takes the first
    /// ordering's sampled ratings, numbered in canonical item order.
    pub fn from_run(
        run: &DistributionRun,
        provider_source: impl Into<String>,
        tolerance_per_item: u32,
    ) -> Self {
        let first = run.orderings.first();
        let answers = ConstitutionalAnswers {
            ratings: first
                .map(|o| {
                    o.items
                        .iter()
                        .enumerate()
                        .map(|(i, p)| Rating {
                            n: (i + 1) as u32,
                            rating: u32::from(p.sampled.get()),
                        })
                        .collect()
                })
                .unwrap_or_default(),
        };
        let distribution_run = run.measure == Measure::Distribution;
        BaselineEntry {
            model_id: run.model_id.clone(),
            questionnaire_version: run.instrument_key.clone(),
            provider_source: provider_source.into(),
            capture_date: run.probed_at,
            ratified_at: None,
            council_decision_id: None,
            tolerance_per_item,
            answers,
            snapshot_path: None,
            request_id: None,
            per_item: Some(PerItemBaseline {
                measure: run.measure,
                thinking: run.thinking.clone(),
                max_residual: distribution_run.then_some(run.max_residual),
                orderings: run
                    .orderings
                    .iter()
                    .map(|o| OrderingRecord {
                        order: o.order,
                        items: o
                            .items
                            .iter()
                            .map(|p| ItemRecord {
                                item: p.item.clone(),
                                sampled: p.sampled,
                                distribution: p.reading.as_ref().map(|r| r.distribution.clone()),
                            })
                            .collect(),
                    })
                    .collect(),
            }),
        }
    }
}

/// Wire shape used during deserialization. Accepts v1, v2, and v3
/// JSON layouts; the [`Deserialize`] impl on [`BaselineEntry`]
/// reconciles them.
#[derive(Debug, Deserialize)]
struct BaselineEntryWire {
    model_id: String,
    questionnaire_version: String,
    /// v2+: present and required as the unambiguous capture timestamp.
    /// v1: absent — `ratified_at` was used for the capture timestamp.
    #[serde(default)]
    capture_date: Option<DateTime<Utc>>,
    /// v2+: optional, set only on actual Council ratification.
    /// v1: required, doubled as capture timestamp.
    #[serde(default)]
    ratified_at: Option<DateTime<Utc>>,
    /// v2+: required-by-convention. v1: absent — defaults to
    /// [`PROVIDER_SOURCE_UNKNOWN`].
    #[serde(default)]
    provider_source: Option<String>,
    council_decision_id: Option<Uuid>,
    tolerance_per_item: u32,
    answers: ConstitutionalAnswers,
    /// v3: optional sidecar reference for the pre-grammar snapshot
    /// stream. Absent on v1/v2 entries.
    #[serde(default)]
    snapshot_path: Option<String>,
    /// v3: optional server-issued request id matching the snapshot
    /// session. Absent on v1/v2 entries.
    #[serde(default)]
    request_id: Option<Uuid>,
    /// v4: optional per-item readings. Absent on v1–v3 entries.
    #[serde(default)]
    per_item: Option<PerItemBaseline>,
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
            snapshot_path: wire.snapshot_path,
            request_id: wire.request_id,
            per_item: wire.per_item,
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

    /// Most recent per-item (v4) entry of `measure` and `thinking` mode
    /// for `(model_id, key)`. Legacy entries are never returned.
    pub fn get_per_item(
        &self,
        model_id: &str,
        questionnaire_version: &str,
        measure: Measure,
        thinking: &ThinkingMode,
    ) -> Option<&BaselineEntry> {
        self.entries.iter().rev().find(|e| {
            e.model_id == model_id
                && e.questionnaire_version == questionnaire_version
                && e.per_item
                    .as_ref()
                    .is_some_and(|p| p.measure == measure && p.thinking == *thinking)
        })
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
            snapshot_path: None,
            request_id: None,
            per_item: None,
        }
    }

    fn run(measure: Measure, sevens: &[f64]) -> DistributionRun {
        use super::super::deterministic::{ItemPass, OrderingPass};
        use super::super::distribution::RatingReading;
        let orderings = [ItemOrder::Forward, ItemOrder::Reverse]
            .into_iter()
            .map(|order| OrderingPass {
                order,
                items: sevens
                    .iter()
                    .enumerate()
                    .map(|(i, &p7)| {
                        let dist = RatingDistribution::from_masses(
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 - p7, p7],
                            0.01,
                        )
                        .unwrap();
                        let sampled = RatingDigit::new(if p7 >= 0.5 { 7 } else { 6 }).unwrap();
                        ItemPass {
                            item: ItemSlug::new(format!("item_{i}")),
                            slot: 1,
                            sampled,
                            request_id: None,
                            reading: (measure == Measure::Distribution).then(|| RatingReading {
                                generation_index: 4,
                                sampled_piece: sampled.to_string(),
                                sampled: Some(sampled),
                                distribution: dist,
                                vocab_entropy: 0.5,
                            }),
                            input_tokens: 1,
                            cache_read_input_tokens: 0,
                            output_tokens: 1,
                            answer: format!("{{\"rating\": \"{sampled}\"}}"),
                            thinking: None,
                            thinking_tokens: None,
                            snapshots: Vec::new(),
                        }
                    })
                    .collect(),
            })
            .collect();
        DistributionRun {
            model_id: "m".into(),
            instrument_key: "v1".into(),
            measure,
            thinking: ThinkingMode::Off,
            probed_at: Utc::now(),
            max_residual: 0.05,
            orderings,
            digits_from: None,
            digit_vocab: None,
        }
    }

    /// A v3 file (no `per_item`) loads as legacy entries; a v4 entry
    /// built from a run round-trips through JSON with its per-ordering
    /// `{p, residual_mass}` and sampled ratings intact.
    #[test]
    fn v3_loads_and_v4_roundtrips() {
        let v3_json = r#"{
            "schema_version": 3,
            "entries": [{
                "model_id": "m", "questionnaire_version": "v1",
                "provider_source": "self_hosted_drama_llama",
                "capture_date": "2026-10-06T13:50:27Z",
                "council_decision_id": null, "tolerance_per_item": 2,
                "answers": {"ratings": {"1": 7, "2": 5}},
                "snapshot_path": "probe_snapshots/x.jsonl",
                "request_id": "953eef9e-e9be-4211-a222-682ec61563de"
            }]
        }"#;
        let mut file: BaselineFile = serde_json::from_str(v3_json).unwrap();
        assert!(file.entries[0].per_item.is_none());
        assert!(
            file.get_per_item("m", "v1", Measure::Distribution, &ThinkingMode::Off)
                .is_none()
        );

        let r = run(Measure::Distribution, &[0.6, 0.3]);
        let entry = BaselineEntry::from_run(&r, "self_hosted_drama_llama", 2);
        file.entries.push(entry.clone());
        file.schema_version = CURRENT_SCHEMA_VERSION;
        let json = serde_json::to_string_pretty(&file).unwrap();
        assert!(json.contains("\"residual_mass\""), "{json}");
        let back: BaselineFile = serde_json::from_str(&json).unwrap();
        assert_eq!(back, file);
        let got = back
            .get_per_item("m", "v1", Measure::Distribution, &ThinkingMode::Off)
            .unwrap();
        assert_eq!(got, &entry);
        assert_eq!(got.answers.ratings[0].rating, 7);
        assert_eq!(got.answers.ratings[1].rating, 6);
        let p = got.per_item.as_ref().unwrap();
        assert_eq!(p.orderings.len(), 2);
        assert_eq!(p.distributions().len(), 2);
        // Legacy `get` still finds the newest entry for integer tooling.
        assert_eq!(back.get("m", "v1").unwrap(), &entry);
    }

    #[test]
    fn evaluate_run_compares_like_with_like() {
        let pol = DriftPolicy::default();
        let base_run = run(Measure::Distribution, &[0.6, 0.3]);
        let entry = BaselineEntry::from_run(&base_run, "p", 2);

        let same = evaluate_run(&base_run, &entry, &pol).unwrap();
        assert!(same.pass());

        let drifted = run(Measure::Distribution, &[0.6, 0.1]);
        match evaluate_run(&drifted, &entry, &pol).unwrap() {
            RunComparison::Distribution(r) => {
                assert!((r.pooled.max_abs_ev_shift - 0.2).abs() < 1e-9);
                assert!(r.pass, "{:?}", r.pooled.failures);
                assert_eq!(r.per_ordering.len(), 2);
            }
            other => panic!("{other:?}"),
        }
        // Item 0 down 0.4, and its clear mode 7 (margin 0.2) became 6.
        let drifted = run(Measure::Distribution, &[0.2, 0.3]);
        let c = evaluate_run(&drifted, &entry, &pol).unwrap();
        assert!(!c.pass());

        // Thinking modes keep separate baselines.
        let mut thinking_run = base_run.clone();
        thinking_run.thinking = ThinkingMode::Adaptive { effort: None };
        assert!(matches!(
            evaluate_run(&thinking_run, &entry, &pol),
            Err(DriftError::ThinkingMismatch { .. })
        ));

        // A legacy entry is a different measurement.
        let mut legacy = entry.clone();
        legacy.per_item = None;
        assert!(matches!(
            evaluate_run(&base_run, &legacy, &pol),
            Err(DriftError::MeasureMismatch { baseline: None, .. })
        ));
        // So is an external-only entry, against a distribution run.
        let ext_entry = BaselineEntry::from_run(&run(Measure::ExternalOnly, &[0.6, 0.3]), "a", 2);
        assert!(matches!(
            evaluate_run(&base_run, &ext_entry, &pol),
            Err(DriftError::MeasureMismatch { .. })
        ));
        // External-only against external-only: integer deltas.
        let ext = run(Measure::ExternalOnly, &[0.6, 0.3]);
        match evaluate_run(&ext, &ext_entry, &pol).unwrap() {
            RunComparison::ExternalOnly {
                per_ordering, pass, ..
            } => {
                assert!(pass);
                assert_eq!(per_ordering.len(), 2);
                assert_eq!(per_ordering[0].1.max_abs_delta, 0);
            }
            other => panic!("{other:?}"),
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
