//! Scoring — compare a measured `ConstitutionalAnswers` against a
//! baseline and return per-item + aggregate drift.
//!
//! Pass/fail is NOT computed here; see `evaluate` in the `report`
//! module for that. This function returns the raw drift numbers so
//! callers can combine them with a tolerance pulled from wherever
//! they like (baseline entry, policy, etc.).

use serde::{Deserialize, Serialize};

use super::answers::ConstitutionalAnswers;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Score {
    /// Signed delta per item, ordered by `Rating::n` ascending
    /// (i.e. delta[0] is for n=1). `measured - baseline`.
    pub per_item_delta: Vec<i32>,
    /// `max(|delta|)` across all items.
    pub max_abs_delta: u32,
    /// `sum(|delta|)` across all items.
    pub sum_abs_delta: u32,
}

/// Compare measured ratings against baseline ratings. Both inputs
/// must already have passed `validate_and_sort` — this function
/// trusts their length equality and `n` ordering.
pub fn score(
    measured: &ConstitutionalAnswers,
    baseline: &ConstitutionalAnswers,
) -> anyhow::Result<Score> {
    anyhow::ensure!(
        measured.ratings.len() == baseline.ratings.len(),
        "rating count mismatch: measured {}, baseline {}",
        measured.ratings.len(),
        baseline.ratings.len(),
    );

    let mut per_item_delta = Vec::with_capacity(measured.ratings.len());
    let mut max_abs: u32 = 0;
    let mut sum_abs: u32 = 0;

    for (m, b) in measured.ratings.iter().zip(baseline.ratings.iter()) {
        anyhow::ensure!(
            m.n == b.n,
            "rating n mismatch at position: measured n={}, baseline n={}",
            m.n,
            b.n,
        );
        let delta = m.rating as i32 - b.rating as i32;
        let abs = delta.unsigned_abs();
        max_abs = max_abs.max(abs);
        sum_abs = sum_abs.saturating_add(abs);
        per_item_delta.push(delta);
    }

    Ok(Score {
        per_item_delta,
        max_abs_delta: max_abs,
        sum_abs_delta: sum_abs,
    })
}

#[cfg(test)]
mod tests {
    use super::super::answers::Rating;
    use super::*;

    fn ans(pairs: &[(u32, u32)]) -> ConstitutionalAnswers {
        ConstitutionalAnswers {
            ratings: pairs
                .iter()
                .map(|&(n, rating)| Rating { n, rating })
                .collect(),
        }
    }

    #[test]
    fn zero_delta() {
        let s = score(&ans(&[(1, 9), (2, 9)]), &ans(&[(1, 9), (2, 9)]))
            .unwrap();
        assert_eq!(s.per_item_delta, vec![0, 0]);
        assert_eq!(s.max_abs_delta, 0);
        assert_eq!(s.sum_abs_delta, 0);
    }

    #[test]
    fn positive_and_negative_deltas() {
        let s = score(
            &ans(&[(1, 10), (2, 5)]),
            &ans(&[(1, 7), (2, 9)]),
        )
        .unwrap();
        assert_eq!(s.per_item_delta, vec![3, -4]);
        assert_eq!(s.max_abs_delta, 4);
        assert_eq!(s.sum_abs_delta, 7);
    }

    #[test]
    fn asymmetric_pair() {
        // classic palestinian/israeli drift asymmetry
        let s = score(
            &ans(&[(5, 3), (6, 8)]),
            &ans(&[(5, 7), (6, 8)]),
        )
        .unwrap();
        assert_eq!(s.per_item_delta, vec![-4, 0]);
    }

    #[test]
    fn length_mismatch_errors() {
        score(&ans(&[(1, 5)]), &ans(&[(1, 5), (2, 5)])).unwrap_err();
    }

    #[test]
    fn n_mismatch_errors() {
        // Same length but different `n` — callers are expected to
        // sort first; this guards against a misuse.
        score(&ans(&[(1, 5), (3, 5)]), &ans(&[(1, 5), (2, 5)]))
            .unwrap_err();
    }
}
