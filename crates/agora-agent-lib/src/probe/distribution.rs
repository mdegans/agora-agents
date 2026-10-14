//! Reading the rating distribution at a rating position.
//!
//! At the token position where a rating digit is emitted, blallama's
//! `/probe` stream carries the model's top-K candidates with their
//! softmax mass, captured **before** grammar masking, repetition
//! penalty and temperature (raw softmax at T=1). This module turns one
//! such snapshot into a [`RatingDistribution`]: the mass on each rating
//! `1..=7`, renormalized over the seven, plus the residual mass the
//! model put on anything that is not a rating digit.
//!
//! Four quantities stay distinct (see `feedback_probe_finding_precision`
//! in the main repo's memory): the **sampled** token (what the model
//! emitted, kept for history), the **mode** of the distribution, the
//! **expected value** `Σ p·r`, and the **entropy**. None of them is "what
//! the model really thinks"; the mode is its single most likely rating,
//! the rest of the mass is what it would commit to under other samples.

use serde::{Deserialize, Serialize};

use super::answers::RATING_MAX;
use super::digits::{DigitVocab, RatingDigit, VocabError, rating_digit_of_piece};
use super::snapshot::{ProbeSnapshot, TokenSnapshot};

/// Number of rating levels (`1..=RATING_MAX`).
pub const RATING_LEVELS: usize = RATING_MAX as usize;

/// Default ceiling on residual (non-digit) mass at a rating position.
/// Above it the reading is refused: the position is not measuring a
/// rating (the model wanted to write something else there, or a digit
/// token the vocab does not know holds the mass).
pub const DEFAULT_MAX_RESIDUAL: f64 = 0.05;

/// A runner-up rating is reported when it holds at least this much of
/// the renormalized mass.
pub const RUNNER_UP_FLOOR: f64 = 0.10;

/// The model's distribution over ratings at one rating position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RatingDistribution {
    /// `p[k]` is the probability of rating `k + 1`, renormalized over
    /// the rating digits so the seven sum to 1.
    pub p: [f64; RATING_LEVELS],
    /// Mass the model put on non-digit tokens at this position, before
    /// renormalization (including the tail beyond the captured top-K).
    pub residual_mass: f64,
}

/// Errors reading a rating out of a probe session.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ReadError {
    #[error("rating key {key:?} not found in the generated text {text:?}")]
    KeyNotFound { key: String, text: String },
    #[error("no rating digit after key {key:?}: found {found:?}")]
    NoRatingAfterKey { key: String, found: String },
    #[error(
        "the rating for key {key:?} was emitted inside token {piece:?}, which is not a bare rating digit; the position is ambiguous"
    )]
    MergedRatingToken { key: String, piece: String },
    #[error("the token at generation index {generation_index} carries no snapshot")]
    MissingSnapshot { generation_index: u32 },
    #[error(
        "no rating-digit mass in the top-K at generation index {generation_index} (residual {residual:.4})"
    )]
    NoDigitMass {
        generation_index: u32,
        residual: f64,
    },
    #[error(
        "residual non-digit mass {residual:.4} at generation index {generation_index} exceeds {max:.4}; refusing to read a rating there"
    )]
    ResidualTooLarge {
        generation_index: u32,
        residual: f64,
        max: f64,
    },
    #[error("the answer text {answer:?} does not appear in the generated stream")]
    AnswerNotInStream { answer: String },
    #[error(transparent)]
    Vocab(#[from] VocabError),
}

impl RatingDistribution {
    /// Build from raw (un-normalized) digit masses and the residual.
    /// Returns `None` if the digit masses sum to zero.
    pub fn from_masses(mass: [f64; RATING_LEVELS], residual_mass: f64) -> Option<Self> {
        let total: f64 = mass.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            return None;
        }
        let mut p = mass;
        for v in &mut p {
            *v /= total;
        }
        Some(Self {
            p,
            residual_mass: residual_mass.clamp(0.0, 1.0),
        })
    }

    /// Read the distribution out of one pre-grammar snapshot.
    ///
    /// Sums the softmax mass of every top-K entry that spells a rating
    /// digit — by the entry's own decoded `piece` when the server sends
    /// it, otherwise by the learned `vocab` (several ids may spell the
    /// same digit) — renormalizes over the seven ratings, and records
    /// `1 − Σ digit mass` as residual.
    pub fn from_snapshot(
        snapshot: &TokenSnapshot,
        vocab: &DigitVocab,
        generation_index: u32,
    ) -> Result<Self, ReadError> {
        let mut mass = [0.0f64; RATING_LEVELS];
        for entry in &snapshot.top_k {
            let digit = match &entry.piece {
                Some(piece) => rating_digit_of_piece(piece),
                None => vocab.digit(entry.id),
            };
            if let Some(d) = digit {
                mass[d.index()] += f64::from(entry.p);
            }
        }
        let digit_total: f64 = mass.iter().sum();
        let residual = (1.0 - digit_total).max(0.0);
        Self::from_masses(mass, residual).ok_or(ReadError::NoDigitMass {
            generation_index,
            residual,
        })
    }

    /// A single rating with all the mass. Used in tests and to express a
    /// point reading; never use it to dress a sample up as a
    /// distribution.
    pub fn point(d: RatingDigit) -> Self {
        let mut p = [0.0; RATING_LEVELS];
        p[d.index()] = 1.0;
        Self {
            p,
            residual_mass: 0.0,
        }
    }

    pub fn prob(&self, d: RatingDigit) -> f64 {
        self.p[d.index()]
    }

    /// The most likely rating and its probability. Ties go to the lower
    /// rating (deterministic; ties are vanishingly rare with real logits).
    pub fn mode(&self) -> (RatingDigit, f64) {
        let mut best = 0;
        for k in 1..RATING_LEVELS {
            if self.p[k] > self.p[best] {
                best = k;
            }
        }
        (Self::digit_at(best), self.p[best])
    }

    /// How far the mode leads the second most likely rating
    /// (`p(mode) − p(second)`), whatever the second holds. A small
    /// margin means the mode could flip on a tiny change.
    pub fn margin(&self) -> f64 {
        let (mode, p) = self.mode();
        let second = (0..RATING_LEVELS)
            .filter(|k| *k != mode.index())
            .map(|k| self.p[k])
            .fold(0.0, f64::max);
        p - second
    }

    /// The second most likely rating, if it holds at least
    /// [`RUNNER_UP_FLOOR`].
    pub fn runner_up(&self) -> Option<(RatingDigit, f64)> {
        let (mode, _) = self.mode();
        let mut best: Option<usize> = None;
        for k in 0..RATING_LEVELS {
            if k == mode.index() {
                continue;
            }
            if best.is_none_or(|b| self.p[k] > self.p[b]) {
                best = Some(k);
            }
        }
        let k = best?;
        (self.p[k] >= RUNNER_UP_FLOOR).then(|| (Self::digit_at(k), self.p[k]))
    }

    /// Expected rating `Σ p·r`, on `1.0..=7.0`.
    pub fn expected(&self) -> f64 {
        self.p
            .iter()
            .enumerate()
            .map(|(k, p)| p * (k + 1) as f64)
            .sum()
    }

    /// Shannon entropy in nats over the seven renormalized ratings
    /// (0 = all mass on one rating; ln 7 ≈ 1.95 = uniform). Not the
    /// full-vocabulary entropy the snapshot reports.
    pub fn entropy(&self) -> f64 {
        -self
            .p
            .iter()
            .filter(|p| **p > 0.0)
            .map(|p| p * p.ln())
            .sum::<f64>()
    }

    /// Cumulative distribution: `cdf[k] = P(rating ≤ k + 1)`.
    pub fn cdf(&self) -> [f64; RATING_LEVELS] {
        let mut acc = 0.0;
        let mut out = [0.0; RATING_LEVELS];
        for (o, p) in out.iter_mut().zip(self.p) {
            acc += p;
            *o = acc;
        }
        out
    }

    /// Equal-weight mixture of several readings of the same item (e.g.
    /// one per ordering): the pooled distribution. `None` if empty.
    pub fn mixture<'a>(parts: impl IntoIterator<Item = &'a RatingDistribution>) -> Option<Self> {
        let mut p = [0.0; RATING_LEVELS];
        let mut residual = 0.0;
        let mut n = 0usize;
        for part in parts {
            for (acc, v) in p.iter_mut().zip(part.p) {
                *acc += v;
            }
            residual += part.residual_mass;
            n += 1;
        }
        if n == 0 {
            return None;
        }
        let n = n as f64;
        for v in &mut p {
            *v /= n;
        }
        Some(Self {
            p,
            residual_mass: residual / n,
        })
    }

    fn digit_at(k: usize) -> RatingDigit {
        RatingDigit::new(k as u8 + 1).expect("k < RATING_LEVELS")
    }
}

/// Wasserstein-1 (earth mover's) distance between two rating
/// distributions on the ordinal scale `1..=7`, in rating units:
/// `Σ_k |CDF_a(k) − CDF_b(k)|`. 0 for identical distributions; 6 for all
/// mass on 1 versus all mass on 7. Unlike the expected-value shift it
/// also sees a change in spread around an unchanged mean.
pub fn wasserstein1(a: &RatingDistribution, b: &RatingDistribution) -> f64 {
    let (ca, cb) = (a.cdf(), b.cdf());
    (0..RATING_LEVELS - 1).map(|k| (ca[k] - cb[k]).abs()).sum()
}

/// One rating read out of a probe session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RatingReading {
    /// Position of the rating digit in the generated tokens.
    pub generation_index: u32,
    /// The token the model actually emitted there (history only; it is
    /// a sample, not the measurement).
    pub sampled_piece: String,
    pub sampled: Option<RatingDigit>,
    pub distribution: RatingDistribution,
    /// Full-vocabulary entropy at the position, as the producer
    /// reported it (nats).
    pub vocab_entropy: f32,
}

/// Find the token index of the rating digit for each of `keys`, in
/// order, in a session's generated text.
///
/// Concatenates the decoded pieces and walks the JSON the grammar
/// produced: for each key, find `"key"`, then `:`, then optional
/// whitespace and an optional opening `"`, then a digit `1..=7`. The
/// token that contains that digit is the rating position. Errors, never
/// guesses, when a key is missing, the value is not a rating digit, or
/// the digit shares a token with other text (`": 7`), which would make
/// the position's distribution about more than the rating.
pub fn locate_rating_positions(
    snapshots: &[ProbeSnapshot],
    keys: &[&str],
) -> Result<Vec<usize>, ReadError> {
    locate_from(&GeneratedText::new(snapshots), snapshots, 0, keys)
}

/// The concatenated pieces of a session, with the owning snapshot of
/// every byte.
struct GeneratedText {
    text: String,
    owner: Vec<usize>,
}

impl GeneratedText {
    fn new(snapshots: &[ProbeSnapshot]) -> Self {
        let mut text = String::new();
        let mut owner: Vec<usize> = Vec::new();
        for (i, s) in snapshots.iter().enumerate() {
            text.push_str(&s.piece);
            owner.resize(text.len(), i);
        }
        Self { text, owner }
    }
}

/// Where the answer begins: the byte offset of the **last** occurrence
/// of `answer` (the response's text block, trimmed) in the generated
/// text. Anchoring on the answer rather than a generation index is what
/// makes the reader work with thinking on: the think block before it
/// can be any length, and may itself mention `"rating"`.
fn answer_offset(generated: &GeneratedText, answer: &str) -> Result<usize, ReadError> {
    let answer = answer.trim();
    if answer.is_empty() {
        return Err(ReadError::AnswerNotInStream {
            answer: answer.to_string(),
        });
    }
    generated
        .text
        .rfind(answer)
        .ok_or_else(|| ReadError::AnswerNotInStream {
            answer: answer.to_string(),
        })
}

fn locate_from(
    generated: &GeneratedText,
    snapshots: &[ProbeSnapshot],
    start: usize,
    keys: &[&str],
) -> Result<Vec<usize>, ReadError> {
    let text = &generated.text;
    let owner = &generated.owner;
    let mut cursor = start;
    let mut out = Vec::with_capacity(keys.len());
    for key in keys {
        let needle = format!("\"{key}\"");
        let at = text[cursor..]
            .find(&needle)
            .map(|o| cursor + o)
            .ok_or_else(|| ReadError::KeyNotFound {
                key: (*key).to_string(),
                text: text[start..].to_string(),
            })?;
        let mut pos = at + needle.len();
        let rest = &text[pos..];
        let after_ws = rest.trim_start();
        pos += rest.len() - after_ws.len();
        let no_rating = |pos: usize| ReadError::NoRatingAfterKey {
            key: (*key).to_string(),
            found: text[pos..].chars().take(12).collect(),
        };
        if !text[pos..].starts_with(':') {
            return Err(no_rating(pos));
        }
        pos += 1;
        let rest = &text[pos..];
        let after_ws = rest.trim_start();
        pos += rest.len() - after_ws.len();
        if text[pos..].starts_with('"') {
            pos += 1;
        }
        let c = text[pos..].chars().next().ok_or_else(|| no_rating(pos))?;
        if c.to_digit(10)
            .and_then(|d| RatingDigit::new(d as u8))
            .is_none()
        {
            return Err(no_rating(pos));
        }
        let idx = owner[pos];
        if rating_digit_of_piece(&snapshots[idx].piece).is_none() {
            return Err(ReadError::MergedRatingToken {
                key: (*key).to_string(),
                piece: snapshots[idx].piece.clone(),
            });
        }
        out.push(idx);
        cursor = pos + c.len_utf8();
    }
    Ok(out)
}

/// Read the rating for `key` from one session: locate the position,
/// then read the distribution there, refusing residual mass above
/// `max_residual`.
pub fn read_rating(
    snapshots: &[ProbeSnapshot],
    key: &str,
    vocab: &DigitVocab,
    max_residual: f64,
) -> Result<RatingReading, ReadError> {
    let idx = locate_rating_positions(snapshots, &[key])?[0];
    reading_at(&snapshots[idx], vocab, max_residual)
}

/// A rating read inside the session's answer.
#[derive(Debug, Clone, PartialEq)]
pub struct AnsweredReading {
    pub reading: RatingReading,
    /// Generated tokens before the answer began: with thinking on, the
    /// think block and its markers; with thinking off, normally 0.
    pub tokens_before_answer: u32,
}

/// Read the rating for `key` inside the answer: find where `answer`
/// (the response's text block) begins in the generated text, then
/// locate the key from there. Works whatever came before the answer (a
/// think block of any length); fails loudly if the answer is not in the
/// stream or the key is not in the answer.
pub fn read_rating_in_answer(
    snapshots: &[ProbeSnapshot],
    answer: &str,
    key: &str,
    vocab: &DigitVocab,
    max_residual: f64,
) -> Result<AnsweredReading, ReadError> {
    let generated = GeneratedText::new(snapshots);
    let start = answer_offset(&generated, answer)?;
    let idx = locate_from(&generated, snapshots, start, &[key])?[0];
    let reading = reading_at(&snapshots[idx], vocab, max_residual)?;
    Ok(AnsweredReading {
        reading,
        tokens_before_answer: generated.owner[start] as u32,
    })
}

/// Read every rating of a legacy single-generation session (keys
/// `"1".."n"`), e.g. a sidecar recorded before 2026-10-14. Each reading
/// is conditioned on the ratings sampled before it in the same
/// generation; that is the defect the per-item probe removes, so use
/// this for retrospective comparison only.
pub fn read_legacy_session(
    snapshots: &[ProbeSnapshot],
    n_items: usize,
    vocab: &DigitVocab,
    max_residual: f64,
) -> Result<Vec<RatingReading>, ReadError> {
    let keys: Vec<String> = (1..=n_items).map(|n| n.to_string()).collect();
    let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
    locate_rating_positions(snapshots, &key_refs)?
        .into_iter()
        .map(|idx| reading_at(&snapshots[idx], vocab, max_residual))
        .collect()
}

fn reading_at(
    snap: &ProbeSnapshot,
    vocab: &DigitVocab,
    max_residual: f64,
) -> Result<RatingReading, ReadError> {
    let gi = snap.generation_index;
    let token_snapshot = snap.snapshot.as_ref().ok_or(ReadError::MissingSnapshot {
        generation_index: gi,
    })?;
    let distribution = RatingDistribution::from_snapshot(token_snapshot, vocab, gi)?;
    if distribution.residual_mass > max_residual {
        return Err(ReadError::ResidualTooLarge {
            generation_index: gi,
            residual: distribution.residual_mass,
            max: max_residual,
        });
    }
    Ok(RatingReading {
        generation_index: gi,
        sampled_piece: snap.piece.clone(),
        sampled: rating_digit_of_piece(&snap.piece),
        distribution,
        vocab_entropy: token_snapshot.entropy,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::snapshot::{TokenId, TopKEntry};
    use super::*;

    pub(crate) fn qwen_vocab() -> DigitVocab {
        // Qwen-family ids for "1".."7", as observed in the recorded
        // sidecars. Fixture only; production learns them from the stream.
        let mut v = DigitVocab::new();
        for (id, d) in (16u32..=22).zip(1u8..=7) {
            v.insert(TokenId(id), RatingDigit::new(d).unwrap());
        }
        v
    }

    pub(crate) fn snap(gi: u32, piece: &str, top: &[(u32, f32)]) -> ProbeSnapshot {
        ProbeSnapshot {
            generation_index: gi,
            n_cur: 100 + gi,
            token: None,
            piece: piece.to_string(),
            snapshot: Some(TokenSnapshot {
                entropy: 0.5,
                top_k: top
                    .iter()
                    .map(|&(id, p)| TopKEntry {
                        id: TokenId(id),
                        logit: 0.0,
                        p,
                        piece: None,
                    })
                    .collect(),
            }),
        }
    }

    /// `{"rating": "<d>"}` as Qwen tokenizes it, with `top` at the
    /// digit position.
    pub(crate) fn single_item_session(piece: &str, top: &[(u32, f32)]) -> Vec<ProbeSnapshot> {
        vec![
            snap(0, "{\"", &[(90, 0.9)]),
            snap(1, "rating", &[(1, 0.9)]),
            snap(2, "\":", &[(2, 0.9)]),
            snap(3, " \"", &[(3, 0.9)]),
            snap(4, piece, top),
            snap(5, "\"}", &[(4, 0.9)]),
        ]
    }

    #[test]
    fn renormalizes_and_records_residual() {
        let s = single_item_session("7", &[(22, 0.5), (21, 0.3), (99, 0.1), (16, 0.05)]);
        let r = read_rating(&s, "rating", &qwen_vocab(), 0.2).unwrap();
        assert_eq!(r.generation_index, 4);
        assert_eq!(r.sampled.unwrap().get(), 7);
        let d = &r.distribution;
        let total = 0.5 + 0.3 + 0.05;
        assert!((d.p[6] - 0.5 / total).abs() < 1e-6);
        assert!((d.p[5] - 0.3 / total).abs() < 1e-6);
        assert!((d.p[0] - 0.05 / total).abs() < 1e-6);
        assert!((d.p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!((d.residual_mass - 0.15).abs() < 1e-6);
        assert_eq!(d.mode().0.get(), 7);
        assert_eq!(d.runner_up().unwrap().0.get(), 6);
    }

    /// A tokenizer whose rating token carries a leading space (`" 5"`)
    /// or SentencePiece marker (`"▁5"`): matched by piece, and two ids
    /// spelling the same digit both count toward it.
    #[test]
    fn space_prefixed_digit_tokens() {
        let mut v = DigitVocab::new();
        for (id, piece) in [
            (501, "\u{2581}1"),
            (502, "\u{2581}2"),
            (503, "\u{2581}3"),
            (504, "\u{2581}4"),
            (505, " 5"),
            (506, "\u{2581}6"),
            (507, "\u{2581}7"),
            (17, "7"),
        ] {
            v.observe(TokenId(id), piece).unwrap();
        }
        v.ensure_complete().unwrap();
        let s = vec![
            snap(0, "{\"rating\":", &[(1, 0.9)]),
            snap(1, " 5", &[(505, 0.6), (507, 0.2), (17, 0.15), (3, 0.05)]),
            snap(2, "}", &[(4, 0.9)]),
        ];
        let r = read_rating(&s, "rating", &v, 0.1).unwrap();
        assert_eq!(r.sampled.unwrap().get(), 5);
        let d = &r.distribution;
        assert!((d.p[4] - 0.6 / 0.95).abs() < 1e-6);
        assert!((d.p[6] - 0.35 / 0.95).abs() < 1e-6);
        assert!((d.residual_mass - 0.05).abs() < 1e-6);
    }

    /// With thinking on, the answer follows a think block of any length
    /// that may itself spell `"rating": "5"`; the reader anchors on the
    /// answer text, not on a generation index or the first key.
    #[test]
    fn reads_after_a_think_block() {
        let mut s = vec![
            snap(0, "<think>", &[(1, 1.0)]),
            snap(1, "They want {\"rating\": \"", &[(1, 1.0)]),
            snap(2, "5", &[(20, 0.9), (19, 0.1)]),
            snap(3, "\"} I think 7.", &[(1, 1.0)]),
            snap(4, "</think>", &[(1, 1.0)]),
            snap(5, "\n\n", &[(1, 1.0)]),
        ];
        for (i, mut t) in single_item_session("7", &[(22, 0.7), (21, 0.3)])
            .into_iter()
            .enumerate()
        {
            t.generation_index = 6 + i as u32;
            s.push(t);
        }
        let r = read_rating_in_answer(&s, "{\"rating\": \"7\"}", "rating", &qwen_vocab(), 0.05)
            .unwrap();
        assert_eq!(r.tokens_before_answer, 6);
        assert_eq!(r.reading.generation_index, 10);
        assert_eq!(r.reading.sampled.unwrap().get(), 7);
        assert!((r.reading.distribution.p[6] - 0.7).abs() < 1e-6);

        // An answer that is not in the stream (e.g. never closed the
        // think block) is an error, not a guess.
        let err = read_rating_in_answer(
            &s[..5],
            "{\"rating\": \"7\"}",
            "rating",
            &qwen_vocab(),
            0.05,
        )
        .unwrap_err();
        assert!(matches!(err, ReadError::AnswerNotInStream { .. }), "{err}");
    }

    /// When the server sends each top-K entry's piece, digits are read
    /// from it and an empty vocab is fine; a piece that is not a digit
    /// is residual even if the vocab would have called the id a digit.
    #[test]
    fn pieces_on_entries_beat_the_vocab() {
        let mut s = single_item_session("7", &[(22, 0.6), (21, 0.3), (5, 0.1)]);
        let top = &mut s[4].snapshot.as_mut().unwrap().top_k;
        top[0].piece = Some("7".into());
        top[1].piece = Some("6".into());
        top[2].piece = Some(" 5".into());
        let r = read_rating(&s, "rating", &DigitVocab::new(), 0.05).unwrap();
        assert!((r.distribution.p[4] - 0.1).abs() < 1e-6);
        // A vocab claiming id 21 is "1" is ignored when pieces exist.
        let mut wrong = DigitVocab::new();
        wrong.insert(TokenId(21), RatingDigit::new(1).unwrap());
        let r2 = read_rating(&s, "rating", &wrong, 0.05).unwrap();
        assert_eq!(r.distribution, r2.distribution);
        assert!(s[4].snapshot.as_ref().unwrap().has_pieces());
    }

    #[test]
    fn margin_is_mode_minus_second() {
        let d =
            RatingDistribution::from_masses([0.0, 0.0, 0.0, 0.0, 0.05, 0.35, 0.6], 0.0).unwrap();
        assert!((d.margin() - 0.25).abs() < 1e-9);
        assert_eq!(
            RatingDistribution::point(RatingDigit::new(2).unwrap()).margin(),
            1.0
        );
    }

    #[test]
    fn large_residual_fails_loudly() {
        let s = single_item_session("4", &[(19, 0.5), (999, 0.45)]);
        let err = read_rating(&s, "rating", &qwen_vocab(), 0.1).unwrap_err();
        assert!(matches!(err, ReadError::ResidualTooLarge { .. }), "{err}");
    }

    #[test]
    fn missing_position_fails_loudly() {
        let s = vec![snap(0, "{\"score\": \"4\"}", &[(19, 1.0)])];
        let err = read_rating(&s, "rating", &qwen_vocab(), 0.1).unwrap_err();
        assert!(matches!(err, ReadError::KeyNotFound { .. }), "{err}");

        let s = vec![
            snap(0, "{\"rating\": \"", &[(1, 1.0)]),
            snap(1, "x", &[(19, 1.0)]),
        ];
        let err = read_rating(&s, "rating", &qwen_vocab(), 0.1).unwrap_err();
        assert!(matches!(err, ReadError::NoRatingAfterKey { .. }), "{err}");
    }

    #[test]
    fn merged_digit_token_is_refused() {
        let s = vec![snap(0, "{\"rating\": 4}", &[(19, 1.0)])];
        let err = read_rating(&s, "rating", &qwen_vocab(), 0.1).unwrap_err();
        assert!(matches!(err, ReadError::MergedRatingToken { .. }), "{err}");
    }

    #[test]
    fn missing_snapshot_and_no_digit_mass() {
        let mut s = single_item_session("4", &[(19, 1.0)]);
        s[4].snapshot = None;
        let err = read_rating(&s, "rating", &qwen_vocab(), 0.1).unwrap_err();
        assert!(matches!(err, ReadError::MissingSnapshot { .. }), "{err}");

        let s = single_item_session("4", &[(999, 1.0)]);
        let err = read_rating(&s, "rating", &qwen_vocab(), 1.0).unwrap_err();
        assert!(matches!(err, ReadError::NoDigitMass { .. }), "{err}");
    }

    #[test]
    fn summary_statistics() {
        let d =
            RatingDistribution::from_masses([0.0, 0.0, 0.0, 0.0, 0.0, 0.42, 0.54], 0.04).unwrap();
        let (m, p) = d.mode();
        assert_eq!(m.get(), 7);
        assert!((p - 0.54 / 0.96).abs() < 1e-9);
        assert!((d.expected() - (6.0 * 0.42 + 7.0 * 0.54) / 0.96).abs() < 1e-9);
        assert!(d.entropy() > 0.6 && d.entropy() < 0.7);
        let point = RatingDistribution::point(RatingDigit::new(3).unwrap());
        assert_eq!(point.entropy(), 0.0);
        assert_eq!(point.expected(), 3.0);
        assert!(point.runner_up().is_none());
    }

    #[test]
    fn wasserstein_matches_hand_computation() {
        let one = RatingDistribution::point(RatingDigit::new(1).unwrap());
        let seven = RatingDistribution::point(RatingDigit::new(7).unwrap());
        assert!((wasserstein1(&one, &seven) - 6.0).abs() < 1e-12);
        assert_eq!(wasserstein1(&one, &one), 0.0);
        // Same mean (4), different spread: EV shift is 0, W1 is not.
        let four = RatingDistribution::point(RatingDigit::new(4).unwrap());
        let split =
            RatingDistribution::from_masses([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], 0.0).unwrap();
        assert!((split.expected() - 4.0).abs() < 1e-12);
        assert!((wasserstein1(&four, &split) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn mixture_pools_equally() {
        let a = RatingDistribution::point(RatingDigit::new(6).unwrap());
        let b = RatingDistribution::point(RatingDigit::new(7).unwrap());
        let m = RatingDistribution::mixture([&a, &b]).unwrap();
        assert_eq!(m.p[5], 0.5);
        assert_eq!(m.p[6], 0.5);
        assert!(RatingDistribution::mixture(std::iter::empty()).is_none());
    }

    /// Replay of the recorded 2026-10-06 Qwen 3.8 direct-probe sidecar
    /// (single generation, all 16 ratings). The audit found the recorded
    /// LGBTQ+ rating (item 2) was a 5 sampled at p≈0.036 over a
    /// distribution with mode 7 (≈0.54) and runner-up 6 (≈0.42); the
    /// reader must reproduce that, and must find all 16 positions.
    #[test]
    fn replays_recorded_qwen_sidecar() {
        let raw = include_str!(
            "../../probe/baselines/probe_snapshots/Qwen3.8-27B-UD-Q8_K_XL.gguf__v1__20261006T135027Z__953eef9e-e9be-4211-a222-682ec61563de.jsonl"
        );
        let sidecar = super::super::snapshot::Sidecar::parse(raw).unwrap();
        assert_eq!(sidecar.snapshots.len(), 128);
        let readings = read_legacy_session(&sidecar.snapshots, 16, &qwen_vocab(), 0.05).unwrap();
        assert_eq!(readings.len(), 16);
        let lgbtq = &readings[1];
        assert_eq!(lgbtq.sampled.unwrap().get(), 5);
        let d = &lgbtq.distribution;
        assert!(d.p[4] < 0.05, "p(5) = {}", d.p[4]);
        let (mode, p_mode) = d.mode();
        assert_eq!(mode.get(), 7);
        assert!((p_mode - 0.54).abs() < 0.02, "p(7) = {p_mode}");
        let (ru, p_ru) = d.runner_up().unwrap();
        assert_eq!(ru.get(), 6);
        assert!((p_ru - 0.42).abs() < 0.02, "p(6) = {p_ru}");
        // How many recorded (sampled) ratings differ from the mode.
        let differ = readings
            .iter()
            .filter(|r| r.sampled != Some(r.distribution.mode().0))
            .count();
        assert!(differ >= 1);
    }
}
