//! Structured answer types produced by the constitutional probe.
//!
//! No `reasoning` field by design — the probe measures raw model
//! intuition, not its post-hoc rationalization. A drifted model's
//! reasoning would explain the drift away; the rating itself is the
//! signal we want.
//!
//! # Wire schema: fixed-key object, not array
//!
//! The wire schema is a fixed-key object with keys `"1".."N"` each
//! bound to an enum-integer rating. drama_llama's `schema_to_gbnf`
//! emits required object fields as a fixed-order literal template,
//! so the grammar *structurally* requires every key to appear
//! exactly once in order with no skips or duplicates. Combined with
//! `enum: [1..=RATING_MAX]` on each rating value, the grammar is
//! effectively an N-slot fill-in-the-blank form (where N is the
//! item count).
//!
//! This is stronger than the more natural `[{n, rating}, ...]` array
//! shape: with the array shape, the grammar permits the model to
//! skip items, repeat items, or emit them out of order and still
//! satisfy the schema — which cogito-32b did empirically when asked
//! to rate a politically-sensitive item. The fixed-key shape removes
//! every grammar escape valve.
//!
//! We emit the schema via [`build_schema`] rather than via
//! `schemars::JsonSchema` — schemars doesn't easily express
//! "object with dynamically-generated `1..=N` keys", and the
//! hand-crafted JSON is only a few lines.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Maximum rating on the Likert scale. Originally 1-10; switched to
/// 1-7 (Likert-7) on 2026-05-01 for two reasons:
///
/// 1. **Tokenization unambiguity.** With 1-10, ratings 1 and 10 both
///    start with the token "1" — at the rating-emission position the
///    pre-grammar top-K shows the model considering "1", but we can't
///    tell whether that meant "rating 1" (strongly disagree) or
///    "rating 10" (strongly agree, two tokens). Either is a valid
///    grammar path. Likert-7 makes every rating exactly one token, so
///    top-K mass at the rating-emission position has unambiguous
///    meaning. This was the load-bearing reason — the cross-validation
///    primitive (slice-2B) loses interpretability on rating-1/rating-10
///    splits without it.
/// 2. **Psychometric standardness.** Likert-7 is widely used in
///    research; midpoint=4 is meaningful; balanced agree/disagree
///    sides. Likert-10 is unbalanced (no clean neutral midpoint).
///
/// Baselines captured before this change are on the 1-10 scale and
/// are NOT comparable to v1+ baselines. Callers should treat
/// `(scenario_id, version)` as the unit of comparability and never
/// cross-version-compare.
pub const RATING_MAX: u32 = 7;

/// One rating entry in the internal in-memory representation.
/// `n` is always 1-indexed and matches the questionnaire's item
/// ordering. The wire format does NOT use this struct directly —
/// see [`ConstitutionalAnswers`] for the wire shape.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Rating {
    pub n: u32,
    pub rating: u32,
}

/// Structured answer to the questionnaire.
///
/// The wire form is a fixed-key object (`{"1": 9, "2": 8, …}`); the
/// in-memory form is a `Vec<Rating>` sorted by `n`. Both `Serialize`
/// and `Deserialize` are custom so the wire roundtrips cleanly
/// (baselines persist with the wire shape; in-memory code sees the
/// `Vec<Rating>` shape).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstitutionalAnswers {
    pub ratings: Vec<Rating>,
}

impl Serialize for ConstitutionalAnswers {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let map: BTreeMap<String, u32> = self
            .ratings
            .iter()
            .map(|r| (r.n.to_string(), r.rating))
            .collect();
        let wire = serde_json::json!({ "ratings": map });
        wire.serialize(serializer)
    }
}

/// Wire representation: `{"ratings": {"1": 9, "2": 8, ...}}`.
#[derive(Debug, Deserialize)]
struct WireAnswers {
    ratings: BTreeMap<String, u32>,
}

impl<'de> Deserialize<'de> for ConstitutionalAnswers {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = WireAnswers::deserialize(deserializer)?;
        let mut ratings: Vec<Rating> = wire
            .ratings
            .into_iter()
            .map(|(k, rating)| {
                let n: u32 = k.parse().map_err(|_| {
                    serde::de::Error::custom(format!(
                        "rating key must be a positive integer, got {k:?}"
                    ))
                })?;
                Ok(Rating { n, rating })
            })
            .collect::<Result<_, _>>()?;
        ratings.sort_by_key(|r| r.n);
        Ok(ConstitutionalAnswers { ratings })
    }
}

impl ConstitutionalAnswers {
    /// Validate that `ratings` covers `1..=expected_count` with each
    /// `n` appearing exactly once, and each `rating` in `1..=RATING_MAX`.
    ///
    /// With the fixed-key wire schema this is belt-and-braces —
    /// grammar-level constraints already enforce most of it — but
    /// keeping the check defends against schema bugs and any future
    /// wire-format change.
    pub fn validate_and_sort(mut self, expected_count: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(
            self.ratings.len() == expected_count,
            "expected {} ratings, got {}",
            expected_count,
            self.ratings.len(),
        );
        let mut seen = vec![false; expected_count];
        for r in &self.ratings {
            anyhow::ensure!(
                (1..=RATING_MAX).contains(&r.rating),
                "rating out of range 1-{}: n={} rating={}",
                RATING_MAX,
                r.n,
                r.rating,
            );
            let idx = (r.n as usize)
                .checked_sub(1)
                .ok_or_else(|| anyhow::anyhow!("rating n must be >= 1, got {}", r.n))?;
            anyhow::ensure!(
                idx < expected_count,
                "rating n out of range 1..={}: {}",
                expected_count,
                r.n,
            );
            anyhow::ensure!(!seen[idx], "duplicate rating for n={}", r.n);
            seen[idx] = true;
        }
        self.ratings.sort_by_key(|r| r.n);
        Ok(self)
    }
}

/// Build the wire schema: a fixed-key object with keys `"1".."N"`
/// each bound to an enum-integer `1..=RATING_MAX`. See module docs
/// for the grammar-constraint rationale and [`RATING_MAX`] for why
/// the scale is what it is.
pub fn build_schema(item_count: usize) -> serde_json::Value {
    use serde_json::{Value, json};

    let rating_values: Vec<u32> = (1..=RATING_MAX).collect();
    let rating_schema = json!({
        "type": "integer",
        "enum": rating_values,
    });

    let mut properties = serde_json::Map::new();
    let mut required = Vec::with_capacity(item_count);
    for i in 1..=item_count {
        let key = i.to_string();
        properties.insert(key.clone(), rating_schema.clone());
        required.push(Value::String(key));
    }

    json!({
        "type": "object",
        "properties": {
            "ratings": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": false
            }
        },
        "required": ["ratings"],
        "additionalProperties": false
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ratings(pairs: &[(u32, u32)]) -> ConstitutionalAnswers {
        ConstitutionalAnswers {
            ratings: pairs
                .iter()
                .map(|&(n, rating)| Rating { n, rating })
                .collect(),
        }
    }

    #[test]
    fn validates_good_answers() {
        let ans = ratings(&[(3, 5), (1, 7), (2, 6)])
            .validate_and_sort(3)
            .unwrap();
        assert_eq!(
            ans.ratings,
            vec![
                Rating { n: 1, rating: 7 },
                Rating { n: 2, rating: 6 },
                Rating { n: 3, rating: 5 },
            ]
        );
    }

    #[test]
    fn rejects_missing_n() {
        ratings(&[(1, 7), (3, 5)]).validate_and_sort(3).unwrap_err();
    }

    #[test]
    fn rejects_duplicate_n() {
        ratings(&[(1, 7), (1, 6), (2, 5)])
            .validate_and_sort(3)
            .unwrap_err();
    }

    #[test]
    fn rejects_out_of_range_n() {
        ratings(&[(1, 7), (2, 6), (4, 5)])
            .validate_and_sort(3)
            .unwrap_err();
        ratings(&[(0, 7), (1, 6), (2, 5)])
            .validate_and_sort(3)
            .unwrap_err();
    }

    #[test]
    fn rejects_rating_out_of_range() {
        // Rating 0 is below the 1..=7 range.
        ratings(&[(1, 0), (2, 4), (3, 4)])
            .validate_and_sort(3)
            .unwrap_err();
        // Rating 8 is above the 1..=7 range (was a valid rating
        // under the old 1-10 scale; now out-of-range).
        ratings(&[(1, 8), (2, 4), (3, 4)])
            .validate_and_sort(3)
            .unwrap_err();
        // Rating 11 still out of range, regardless of scale.
        ratings(&[(1, 11), (2, 4), (3, 4)])
            .validate_and_sort(3)
            .unwrap_err();
    }

    #[test]
    fn rejects_wrong_length() {
        ratings(&[(1, 4), (2, 4)]).validate_and_sort(3).unwrap_err();
    }

    #[test]
    fn deserializes_fixed_key_wire_format() {
        let wire = r#"{"ratings": {"1": 7, "2": 6, "3": 5}}"#;
        let ans: ConstitutionalAnswers = serde_json::from_str(wire).unwrap();
        assert_eq!(
            ans.ratings,
            vec![
                Rating { n: 1, rating: 7 },
                Rating { n: 2, rating: 6 },
                Rating { n: 3, rating: 5 },
            ]
        );
    }

    #[test]
    fn build_schema_uses_likert_7_enum() {
        let schema = build_schema(2);
        let rating_enum = &schema["properties"]["ratings"]["properties"]["1"]["enum"];
        let values: Vec<u32> = rating_enum
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(values, vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn deserialize_rejects_non_integer_keys() {
        let wire = r#"{"ratings": {"one": 9}}"#;
        serde_json::from_str::<ConstitutionalAnswers>(wire).unwrap_err();
    }

    #[test]
    fn build_schema_uses_fixed_keys() {
        let schema = build_schema(3);
        let ratings_props = &schema["properties"]["ratings"]["properties"];
        assert!(ratings_props.get("1").is_some());
        assert!(ratings_props.get("2").is_some());
        assert!(ratings_props.get("3").is_some());
        assert!(ratings_props.get("4").is_none());
        // All three keys required, in order.
        let required = schema["properties"]["ratings"]["required"]
            .as_array()
            .unwrap();
        assert_eq!(required.len(), 3);
        assert_eq!(required[0], "1");
        assert_eq!(required[1], "2");
        assert_eq!(required[2], "3");
        // additionalProperties false (locks out model from inventing keys)
        assert_eq!(
            schema["properties"]["ratings"]["additionalProperties"],
            serde_json::Value::Bool(false)
        );
    }
}
