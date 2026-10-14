//! Drift between two distribution readings of the same instrument.
//!
//! These probes are canaries for **inference-pipeline integrity**
//! (Steward, 2026-10-14): they run before each Council sitting on every
//! model the Council uses, and a failure means "do not trust this
//! pipeline today". A distribution reading is deterministic given the
//! same weights, prompt and server, so against the same model's own
//! baseline the run-to-run variation should be near zero, and the
//! tolerances are tight ([`DriftPolicy::default`]):
//! - fail if any item's expected value shifts by more than 0.3, or the
//!   mean absolute shift across items exceeds 0.15;
//! - fail on a mode change only where the baseline's mode led its
//!   runner-up by at least 0.2 (a near-tie may flip on numerical noise
//!   without meaning anything);
//! - report Wasserstein-1 and the order effect, but do not gate on them.
//!
//! Every threshold is configurable. Only readings of the same
//! [`Measure`] and thinking mode are compared; a distribution reading
//! against a sampled integer is a different measurement, and the error
//! says so instead of producing a number.

use serde::{Deserialize, Serialize};

use super::deterministic::{DistributionRun, ItemOrder, Measure};
use super::digits::RatingDigit;
use super::distribution::{RatingDistribution, wasserstein1};
use super::ids::ItemSlug;
use super::thinking::ThinkingMode;

/// Gating thresholds for distribution drift.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DriftPolicy {
    /// Fail when any item's |EV shift| exceeds this (rating units).
    pub max_item_ev_shift: f64,
    /// Fail when the mean |EV shift| across items exceeds this.
    pub max_mean_abs_ev_shift: f64,
    /// A mode change fails only when the baseline's mode led its
    /// runner-up by at least this much probability.
    pub mode_flip_margin: f64,
}

impl Default for DriftPolicy {
    fn default() -> Self {
        Self {
            max_item_ev_shift: 0.3,
            max_mean_abs_ev_shift: 0.15,
            mode_flip_margin: 0.2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DriftError {
    #[error("item sets differ: measured has {measured:?}, baseline has {baseline:?}")]
    ItemMismatch {
        measured: Vec<ItemSlug>,
        baseline: Vec<ItemSlug>,
    },
    #[error(
        "measure mismatch: measured {measured:?}, baseline {baseline:?}; these are different measurements and are not compared"
    )]
    MeasureMismatch {
        measured: Option<Measure>,
        baseline: Option<Measure>,
    },
    #[error("thinking mode mismatch: measured {measured}, baseline {baseline}; not compared")]
    ThinkingMismatch {
        measured: ThinkingMode,
        baseline: ThinkingMode,
    },
    #[error("baseline key {baseline:?} does not match the run's {measured:?}")]
    KeyMismatch { measured: String, baseline: String },
    #[error("baseline has no per-item readings for ordering {0}")]
    MissingOrdering(ItemOrder),
}

/// One item's drift.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ItemDrift {
    pub item: ItemSlug,
    pub baseline_expected: f64,
    pub measured_expected: f64,
    /// `measured − baseline` expected value.
    pub ev_shift: f64,
    pub wasserstein1: f64,
    pub baseline_mode: RatingDigit,
    pub measured_mode: RatingDigit,
    pub mode_changed: bool,
    /// How far the baseline's mode led its runner-up.
    pub baseline_margin: f64,
    /// A mode change that counts against `pass` (the baseline's margin
    /// was at least the policy's `mode_flip_margin`).
    pub mode_change_fails: bool,
}

/// Drift over a set of items.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistributionDrift {
    pub per_item: Vec<ItemDrift>,
    pub max_abs_ev_shift: f64,
    pub mean_abs_ev_shift: f64,
    /// Reported, not gated.
    pub max_wasserstein1: f64,
    pub modes_changed: usize,
    pub policy: DriftPolicy,
    pub pass: bool,
    /// Why it failed, one line per reason; empty on a pass.
    pub failures: Vec<String>,
}

/// Compare two sets of per-item distributions, matched by slug.
pub fn score_distributions(
    measured: &[(ItemSlug, RatingDistribution)],
    baseline: &[(ItemSlug, RatingDistribution)],
    policy: &DriftPolicy,
) -> Result<DistributionDrift, DriftError> {
    let slugs = |v: &[(ItemSlug, RatingDistribution)]| {
        let mut s: Vec<ItemSlug> = v.iter().map(|(i, _)| i.clone()).collect();
        s.sort();
        s
    };
    if slugs(measured) != slugs(baseline) {
        return Err(DriftError::ItemMismatch {
            measured: slugs(measured),
            baseline: slugs(baseline),
        });
    }
    let per_item: Vec<ItemDrift> = measured
        .iter()
        .map(|(item, m)| {
            let b = &baseline
                .iter()
                .find(|(i, _)| i == item)
                .expect("slug sets are equal")
                .1;
            let (mm, _) = m.mode();
            let (bm, _) = b.mode();
            let mode_changed = mm != bm;
            let baseline_margin = b.margin();
            ItemDrift {
                item: item.clone(),
                baseline_expected: b.expected(),
                measured_expected: m.expected(),
                ev_shift: m.expected() - b.expected(),
                wasserstein1: wasserstein1(m, b),
                baseline_mode: bm,
                measured_mode: mm,
                mode_changed,
                baseline_margin,
                mode_change_fails: mode_changed && baseline_margin >= policy.mode_flip_margin,
            }
        })
        .collect();
    let max_abs_ev_shift = per_item
        .iter()
        .map(|d| d.ev_shift.abs())
        .fold(0.0, f64::max);
    let mean_abs_ev_shift = if per_item.is_empty() {
        0.0
    } else {
        per_item.iter().map(|d| d.ev_shift.abs()).sum::<f64>() / per_item.len() as f64
    };
    let max_wasserstein1 = per_item.iter().map(|d| d.wasserstein1).fold(0.0, f64::max);
    let modes_changed = per_item.iter().filter(|d| d.mode_changed).count();

    let mut failures = Vec::new();
    for d in &per_item {
        if d.ev_shift.abs() > policy.max_item_ev_shift {
            failures.push(format!(
                "{}: EV {:.2} -> {:.2} ({:+.2}) exceeds {:.2}",
                d.item,
                d.baseline_expected,
                d.measured_expected,
                d.ev_shift,
                policy.max_item_ev_shift
            ));
        }
        if d.mode_change_fails {
            failures.push(format!(
                "{}: mode {} -> {} (baseline margin {:.2} >= {:.2})",
                d.item,
                d.baseline_mode,
                d.measured_mode,
                d.baseline_margin,
                policy.mode_flip_margin
            ));
        }
    }
    if mean_abs_ev_shift > policy.max_mean_abs_ev_shift {
        failures.push(format!(
            "mean |EV shift| {:.3} exceeds {:.2}",
            mean_abs_ev_shift, policy.max_mean_abs_ev_shift
        ));
    }
    Ok(DistributionDrift {
        per_item,
        max_abs_ev_shift,
        mean_abs_ev_shift,
        max_wasserstein1,
        modes_changed,
        policy: *policy,
        pass: failures.is_empty(),
        failures,
    })
}

/// A distribution run compared with a distribution baseline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistributionReport {
    pub model_id: String,
    pub instrument_key: String,
    /// Pooled over the orderings both sides have. Drives `pass`.
    pub pooled: DistributionDrift,
    /// The same comparison within each ordering.
    pub per_ordering: Vec<(ItemOrder, DistributionDrift)>,
    pub pass: bool,
}

/// Compare a run against per-ordering baseline distributions.
///
/// `baseline` maps each ordering to its per-item distributions. Only
/// orderings present in both are compared; the pooled comparison
/// mixes exactly those orderings on both sides, so a baseline captured
/// with more orderings than the run does not bias the pool.
pub fn compare_run(
    run: &DistributionRun,
    baseline: &[(ItemOrder, Vec<(ItemSlug, RatingDistribution)>)],
    policy: &DriftPolicy,
) -> Result<DistributionReport, DriftError> {
    if run.measure != Measure::Distribution {
        return Err(DriftError::MeasureMismatch {
            measured: Some(run.measure),
            baseline: Some(Measure::Distribution),
        });
    }
    let mut per_ordering = Vec::new();
    let mut shared: Vec<ItemOrder> = Vec::new();
    for pass in &run.orderings {
        let Some((_, base)) = baseline.iter().find(|(o, _)| *o == pass.order) else {
            continue;
        };
        let measured: Vec<(ItemSlug, RatingDistribution)> = pass
            .items
            .iter()
            .filter_map(|p| {
                p.reading
                    .as_ref()
                    .map(|r| (p.item.clone(), r.distribution.clone()))
            })
            .collect();
        per_ordering.push((pass.order, score_distributions(&measured, base, policy)?));
        shared.push(pass.order);
    }
    let Some(&first) = run.orderings.first().map(|o| &o.order) else {
        return Err(DriftError::MissingOrdering(ItemOrder::Forward));
    };
    if shared.is_empty() {
        return Err(DriftError::MissingOrdering(first));
    }
    let pool = |get: &dyn Fn(ItemOrder) -> Vec<(ItemSlug, RatingDistribution)>| {
        let parts: Vec<Vec<(ItemSlug, RatingDistribution)>> =
            shared.iter().map(|o| get(*o)).collect();
        parts[0]
            .iter()
            .map(|(slug, _)| {
                let dists: Vec<&RatingDistribution> = parts
                    .iter()
                    .filter_map(|p| p.iter().find(|(s, _)| s == slug).map(|(_, d)| d))
                    .collect();
                (
                    slug.clone(),
                    RatingDistribution::mixture(dists).expect("at least one ordering"),
                )
            })
            .collect::<Vec<_>>()
    };
    let measured_pool = pool(&|o| {
        run.ordering(o)
            .map(|p| {
                p.items
                    .iter()
                    .filter_map(|i| {
                        i.reading
                            .as_ref()
                            .map(|r| (i.item.clone(), r.distribution.clone()))
                    })
                    .collect()
            })
            .unwrap_or_default()
    });
    let baseline_pool = pool(&|o| {
        baseline
            .iter()
            .find(|(b, _)| *b == o)
            .map(|(_, v)| v.clone())
            .unwrap_or_default()
    });
    let pooled = score_distributions(&measured_pool, &baseline_pool, policy)?;
    let pass = pooled.pass;
    Ok(DistributionReport {
        model_id: run.model_id.clone(),
        instrument_key: run.instrument_key.clone(),
        pooled,
        per_ordering,
        pass,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(masses: [f64; 7]) -> RatingDistribution {
        RatingDistribution::from_masses(masses, 0.0).unwrap()
    }

    fn slug(s: &str) -> ItemSlug {
        ItemSlug::new(s)
    }

    fn pol() -> DriftPolicy {
        DriftPolicy::default()
    }

    #[test]
    fn identical_distributions_pass_with_zero_drift() {
        let a = vec![(slug("x"), d([0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.6]))];
        let r = score_distributions(&a, &a, &pol()).unwrap();
        assert!(r.pass, "{:?}", r.failures);
        assert_eq!(r.max_abs_ev_shift, 0.0);
        assert_eq!(r.max_wasserstein1, 0.0);
        assert_eq!(r.modes_changed, 0);
    }

    /// A near-tie (margin 0.2 is the line) may flip without failing;
    /// a clear mode may not.
    #[test]
    fn mode_change_fails_only_on_a_clear_baseline_mode() {
        let base = vec![(slug("x"), d([0.0, 0.0, 0.0, 0.0, 0.0, 0.45, 0.55]))];
        let meas = vec![(slug("x"), d([0.0, 0.0, 0.0, 0.0, 0.0, 0.55, 0.45]))];
        let r = score_distributions(&meas, &base, &pol()).unwrap();
        let i = &r.per_item[0];
        assert!(i.mode_changed && !i.mode_change_fails);
        assert!((i.ev_shift + 0.1).abs() < 1e-9);
        assert!((i.wasserstein1 - 0.1).abs() < 1e-9);
        assert!(r.pass, "{:?}", r.failures);

        let base = vec![(slug("x"), d([0.0, 0.0, 0.0, 0.0, 0.0, 0.38, 0.62]))];
        let meas = vec![(slug("x"), d([0.0, 0.0, 0.0, 0.0, 0.0, 0.52, 0.48]))];
        let r = score_distributions(&meas, &base, &pol()).unwrap();
        assert!(r.per_item[0].mode_change_fails);
        assert!(!r.pass);
        assert_eq!(r.failures.len(), 1, "{:?}", r.failures);
    }

    #[test]
    fn item_and_mean_ev_thresholds() {
        let base: Vec<_> = ["a", "b", "c", "d"]
            .iter()
            .map(|s| (slug(s), d([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])))
            .collect();
        // One item down 0.35: item threshold fails.
        let mut meas = base.clone();
        meas[0].1 = d([0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.65]);
        let r = score_distributions(&meas, &base, &pol()).unwrap();
        assert!(!r.pass);
        assert!(r.failures[0].starts_with("a: EV"), "{:?}", r.failures);
        // Every item down 0.2: no item over 0.3, but the mean 0.2 > 0.15.
        let meas: Vec<_> = base
            .iter()
            .map(|(s, _)| (s.clone(), d([0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8])))
            .collect();
        let r = score_distributions(&meas, &base, &pol()).unwrap();
        assert!(!r.pass);
        assert!((r.mean_abs_ev_shift - 0.2).abs() < 1e-9);
        assert_eq!(r.failures.len(), 1, "{:?}", r.failures);
        assert!(r.failures[0].starts_with("mean"));
        // Configurable: a looser policy passes the same drift.
        let loose = DriftPolicy {
            max_mean_abs_ev_shift: 0.25,
            ..pol()
        };
        assert!(score_distributions(&meas, &base, &loose).unwrap().pass);
    }

    #[test]
    fn spread_change_without_mean_change_shows_in_w1_only() {
        let base = vec![(slug("x"), d([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))];
        let meas = vec![(slug("x"), d([0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0]))];
        let r = score_distributions(&meas, &base, &pol()).unwrap();
        assert!(r.per_item[0].ev_shift.abs() < 1e-9);
        assert!((r.per_item[0].wasserstein1 - 1.0).abs() < 1e-9);
        // W1 is reported, not gated; the mode moved off a clear 4,
        // though, and that does gate.
        assert!(r.per_item[0].mode_changed);
    }

    #[test]
    fn item_sets_must_match() {
        let a = vec![(slug("x"), d([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))];
        let b = vec![(slug("y"), d([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))];
        assert!(matches!(
            score_distributions(&a, &b, &pol()),
            Err(DriftError::ItemMismatch { .. })
        ));
    }
}
