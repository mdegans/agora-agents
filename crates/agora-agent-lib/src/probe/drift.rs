//! Drift between two distribution readings of the same instrument.
//!
//! For each item: the shift in expected value (signed, in rating
//! units), the Wasserstein-1 distance over `1..=7` (sees a change of
//! spread an unchanged mean hides), and whether the mode changed. Pass
//! is `max |EV shift| <= ev_tolerance`; W1 and mode changes are
//! reported, not gated, until the Council sets policy for them.
//!
//! Only readings of the same [`Measure`] are compared. A distribution
//! reading against a sampled integer is a different measurement, and
//! the error says so instead of producing a number.

use serde::{Deserialize, Serialize};

use super::deterministic::{DistributionRun, ItemOrder, Measure};
use super::digits::RatingDigit;
use super::distribution::{RatingDistribution, wasserstein1};
use super::ids::ItemSlug;

/// Default tolerance on the expected-value shift, in rating units.
/// Placeholder pending a Council decision.
pub const DEFAULT_EV_TOLERANCE: f64 = 0.5;

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
}

/// Drift over a set of items.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistributionDrift {
    pub per_item: Vec<ItemDrift>,
    pub max_abs_ev_shift: f64,
    pub max_wasserstein1: f64,
    pub modes_changed: usize,
    pub ev_tolerance: f64,
    pub pass: bool,
}

/// Compare two sets of per-item distributions, matched by slug.
pub fn score_distributions(
    measured: &[(ItemSlug, RatingDistribution)],
    baseline: &[(ItemSlug, RatingDistribution)],
    ev_tolerance: f64,
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
            ItemDrift {
                item: item.clone(),
                baseline_expected: b.expected(),
                measured_expected: m.expected(),
                ev_shift: m.expected() - b.expected(),
                wasserstein1: wasserstein1(m, b),
                baseline_mode: bm,
                measured_mode: mm,
                mode_changed: mm != bm,
            }
        })
        .collect();
    let max_abs_ev_shift = per_item
        .iter()
        .map(|d| d.ev_shift.abs())
        .fold(0.0, f64::max);
    let max_wasserstein1 = per_item.iter().map(|d| d.wasserstein1).fold(0.0, f64::max);
    let modes_changed = per_item.iter().filter(|d| d.mode_changed).count();
    Ok(DistributionDrift {
        per_item,
        max_abs_ev_shift,
        max_wasserstein1,
        modes_changed,
        ev_tolerance,
        pass: max_abs_ev_shift <= ev_tolerance,
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
    ev_tolerance: f64,
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
        per_ordering.push((
            pass.order,
            score_distributions(&measured, base, ev_tolerance)?,
        ));
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
    let pooled = score_distributions(&measured_pool, &baseline_pool, ev_tolerance)?;
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

    #[test]
    fn identical_distributions_pass_with_zero_drift() {
        let a = vec![(slug("x"), d([0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.6]))];
        let r = score_distributions(&a, &a, 0.5).unwrap();
        assert!(r.pass);
        assert_eq!(r.max_abs_ev_shift, 0.0);
        assert_eq!(r.max_wasserstein1, 0.0);
        assert_eq!(r.modes_changed, 0);
    }

    #[test]
    fn ev_shift_w1_and_mode_change() {
        let base = vec![(slug("x"), d([0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.6]))];
        let meas = vec![(slug("x"), d([0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.4]))];
        let r = score_distributions(&meas, &base, 0.5).unwrap();
        let i = &r.per_item[0];
        assert!((i.ev_shift + 0.2).abs() < 1e-9);
        assert!((i.wasserstein1 - 0.2).abs() < 1e-9);
        assert!(i.mode_changed);
        assert_eq!(i.baseline_mode.get(), 7);
        assert_eq!(i.measured_mode.get(), 6);
        assert!(r.pass, "a mode flip under tolerance is flagged, not failed");

        let meas = vec![(slug("x"), d([0.0, 0.6, 0.4, 0.0, 0.0, 0.0, 0.0]))];
        let r = score_distributions(&meas, &base, 0.5).unwrap();
        assert!(!r.pass);
        assert!((r.max_abs_ev_shift - 4.2).abs() < 1e-9);
    }

    #[test]
    fn spread_change_without_mean_change_shows_in_w1_only() {
        let base = vec![(slug("x"), d([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))];
        let meas = vec![(slug("x"), d([0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0]))];
        let r = score_distributions(&meas, &base, 0.5).unwrap();
        assert!(r.per_item[0].ev_shift.abs() < 1e-9);
        assert!((r.per_item[0].wasserstein1 - 1.0).abs() < 1e-9);
        assert!(r.pass);
    }

    #[test]
    fn item_sets_must_match() {
        let a = vec![(slug("x"), d([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))];
        let b = vec![(slug("y"), d([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))];
        assert!(matches!(
            score_distributions(&a, &b, 0.5),
            Err(DriftError::ItemMismatch { .. })
        ));
    }
}
