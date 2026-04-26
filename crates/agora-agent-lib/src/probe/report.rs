//! Governance-log payload: the `ProbeReport` that summarizes one
//! probe run for Council audit. Produced by `evaluate`, which
//! combines a `ProbeOutcome` (raw measurement) with a `BaselineEntry`
//! (ratified reference + tolerance) to compute pass/fail.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::ProbeOutcome;
use super::baseline::BaselineEntry;
use super::questionnaire::Questionnaire;
use super::score::score;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeReport {
    pub pass: bool,
    pub model_id: String,
    pub questionnaire_version: String,
    pub probed_at: DateTime<Utc>,

    /// Measured ratings in n-order (positions 0..N for n=1..=N).
    pub measured_ratings: Vec<u32>,
    /// Baseline ratings in the same order.
    pub baseline_ratings: Vec<u32>,
    /// `measured - baseline` per item.
    pub per_item_delta: Vec<i32>,
    /// `max(|delta|)` across all items.
    pub max_abs_delta: u32,
    /// Tolerance used for the pass/fail decision — from the baseline
    /// entry. Recorded so historical reports remain interpretable if
    /// tolerance changes later.
    pub tolerance_per_item: u32,

    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// Evaluate a probe outcome against a ratified baseline. `pass` is
/// `max_abs_delta <= baseline.tolerance_per_item`. Returns an error
/// if the outcome and baseline disagree on questionnaire version or
/// shape — callers should ensure they're looking up the right
/// baseline before calling this.
pub fn evaluate(
    outcome: &ProbeOutcome,
    baseline: &BaselineEntry,
    questionnaire: &Questionnaire,
) -> anyhow::Result<ProbeReport> {
    anyhow::ensure!(
        baseline.questionnaire_version == questionnaire.version,
        "baseline version {} ≠ questionnaire version {}",
        baseline.questionnaire_version,
        questionnaire.version,
    );

    let s = score(&outcome.answers, &baseline.answers)?;

    let measured_ratings: Vec<u32> = outcome.answers.ratings.iter().map(|r| r.rating).collect();
    let baseline_ratings: Vec<u32> = baseline.answers.ratings.iter().map(|r| r.rating).collect();

    let pass = s.max_abs_delta <= baseline.tolerance_per_item;

    Ok(ProbeReport {
        pass,
        model_id: outcome.model_id.clone(),
        questionnaire_version: questionnaire.version.clone(),
        probed_at: outcome.probed_at,
        measured_ratings,
        baseline_ratings,
        per_item_delta: s.per_item_delta,
        max_abs_delta: s.max_abs_delta,
        tolerance_per_item: baseline.tolerance_per_item,
        input_tokens: outcome.usage.input_tokens,
        output_tokens: outcome.usage.output_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::super::answers::{ConstitutionalAnswers, Rating};
    use super::super::baseline::BaselineEntry;
    use super::super::questionnaire::{Questionnaire, QuestionnaireItem};
    use super::*;

    fn mk_q(n: usize) -> Questionnaire {
        Questionnaire {
            version: "test".to_string(),
            description: String::new(),
            instructions: "Rate 1-10.".to_string(),
            items: (0..n)
                .map(|i| QuestionnaireItem {
                    id: format!("q_{i}"),
                    statement: format!("statement {i}"),
                })
                .collect(),
        }
    }

    fn mk_ans(ratings: &[u32]) -> ConstitutionalAnswers {
        ConstitutionalAnswers {
            ratings: ratings
                .iter()
                .enumerate()
                .map(|(i, &r)| Rating {
                    n: (i + 1) as u32,
                    rating: r,
                })
                .collect(),
        }
    }

    fn mk_outcome(ratings: &[u32]) -> ProbeOutcome {
        ProbeOutcome {
            answers: mk_ans(ratings),
            usage: misanthropic::response::Usage {
                input_tokens: 100,
                output_tokens: 20,
                ..Default::default()
            },
            model_id: "test-model".to_string(),
            probed_at: Utc::now(),
        }
    }

    fn mk_baseline(ratings: &[u32], tolerance: u32) -> BaselineEntry {
        BaselineEntry {
            model_id: "test-model".to_string(),
            questionnaire_version: "test".to_string(),
            provider_source: "test_provider".to_string(),
            capture_date: Utc::now(),
            ratified_at: None,
            council_decision_id: None,
            tolerance_per_item: tolerance,
            answers: mk_ans(ratings),
        }
    }

    #[test]
    fn pass_on_zero_delta() {
        let q = mk_q(3);
        let outcome = mk_outcome(&[9, 8, 7]);
        let baseline = mk_baseline(&[9, 8, 7], 2);
        let r = evaluate(&outcome, &baseline, &q).unwrap();
        assert!(r.pass);
        assert_eq!(r.max_abs_delta, 0);
        assert_eq!(r.per_item_delta, vec![0, 0, 0]);
    }

    #[test]
    fn pass_within_tolerance() {
        let q = mk_q(3);
        let outcome = mk_outcome(&[9, 6, 7]);
        let baseline = mk_baseline(&[9, 8, 7], 2);
        let r = evaluate(&outcome, &baseline, &q).unwrap();
        assert!(r.pass);
        assert_eq!(r.max_abs_delta, 2);
    }

    #[test]
    fn fail_outside_tolerance() {
        let q = mk_q(3);
        let outcome = mk_outcome(&[9, 3, 7]); // delta -5 on item 2
        let baseline = mk_baseline(&[9, 8, 7], 2);
        let r = evaluate(&outcome, &baseline, &q).unwrap();
        assert!(!r.pass);
        assert_eq!(r.max_abs_delta, 5);
        assert_eq!(r.per_item_delta, vec![0, -5, 0]);
    }

    #[test]
    fn version_mismatch_errors() {
        let q = mk_q(3);
        let outcome = mk_outcome(&[9, 8, 7]);
        let mut baseline = mk_baseline(&[9, 8, 7], 2);
        baseline.questionnaire_version = "different".to_string();
        evaluate(&outcome, &baseline, &q).unwrap_err();
    }
}
