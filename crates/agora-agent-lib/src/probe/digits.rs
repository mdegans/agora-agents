//! Which vocabulary tokens are rating digits.
//!
//! blallama's `/probe` top-K lists candidates by token **id** only. Ids
//! are tokenizer-specific (Qwen's `"1"`..`"7"` are 16..22; another
//! family's are not, and a SentencePiece vocab may carry `"▁7"` beside
//! `"7"`), so the distribution reader never hard-codes them. Instead a
//! [`DigitVocab`] is *learned* from the stream itself: every `token`
//! event carries the sampled id and its decoded `piece`, and the
//! distribution probe opens each run with a calibration request whose
//! schema forces every digit `1..=7` to be emitted once (see
//! [`super::deterministic::calibration_schema`]).
//!
//! A token counts as rating digit `d` when its decoded piece, after
//! dropping leading whitespace (including SentencePiece's `▁` and
//! byte-level BPE's `Ġ`) and one JSON string quote on either side, is
//! exactly the single character `d`. A digit token the vocab never
//! observed is not counted as a digit: its mass shows up as residual,
//! and a large residual fails the reading loudly rather than silently.
//!
//! A cleaner source would be the token text on each top-K entry; that
//! needs a drama_llama change (documented in the probe module docs).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::answers::RATING_MAX;
use super::snapshot::{ProbeSnapshot, TokenId};

/// A rating on the Likert scale, `1..=RATING_MAX`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub struct RatingDigit(u8);

impl RatingDigit {
    pub fn new(d: u8) -> Option<Self> {
        (1..=RATING_MAX as u8).contains(&d).then_some(Self(d))
    }

    pub fn get(self) -> u8 {
        self.0
    }

    /// Zero-based index into a `[_; RATING_LEVELS]` vector.
    pub fn index(self) -> usize {
        usize::from(self.0 - 1)
    }

    /// Every rating, ascending.
    pub fn all() -> impl Iterator<Item = RatingDigit> {
        (1..=RATING_MAX as u8).map(RatingDigit)
    }
}

impl TryFrom<u8> for RatingDigit {
    type Error = String;
    fn try_from(d: u8) -> Result<Self, Self::Error> {
        Self::new(d).ok_or_else(|| format!("rating {d} outside 1..={RATING_MAX}"))
    }
}

impl From<RatingDigit> for u8 {
    fn from(d: RatingDigit) -> u8 {
        d.0
    }
}

impl std::fmt::Display for RatingDigit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// The rating digit a decoded piece spells, if it spells exactly one.
///
/// Accepts `"7"`, `" 7"`, `"▁7"`, `"Ġ7"`, `"\n7"`, `"\"7"`, `"7\""`;
/// rejects `"77"`, `"10"`, `"0"`, `"8"`, `": 7"`, `"7,"`, `""`.
pub fn rating_digit_of_piece(piece: &str) -> Option<RatingDigit> {
    let s = piece.trim_start_matches(|c: char| c.is_whitespace() || c == '\u{2581}' || c == 'Ġ');
    let s = s.strip_prefix('"').unwrap_or(s);
    let s = s.strip_suffix('"').unwrap_or(s);
    let mut chars = s.chars();
    let c = chars.next()?;
    if chars.next().is_some() {
        return None;
    }
    let d = c.to_digit(10)?;
    RatingDigit::new(d as u8)
}

/// Token id → rating digit, learned from observed `(id, piece)` pairs.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DigitVocab {
    /// Every token id observed to decode to a rating digit. Several ids
    /// may map to the same digit (`"7"` and `"▁7"`).
    digits: BTreeMap<TokenId, RatingDigit>,
}

/// The vocab could not be trusted for a reading.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VocabError {
    #[error(
        "digit vocab is missing token ids for ratings {missing:?}; did the calibration request run against a /probe endpoint?"
    )]
    Incomplete { missing: Vec<u8> },
    #[error("token {id} was observed decoding to both {first} and {second}")]
    Conflict {
        id: TokenId,
        first: RatingDigit,
        second: RatingDigit,
    },
}

impl DigitVocab {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one observed `(id, piece)` pair. Non-digit pieces are
    /// ignored. Errors if the same id was already seen as another
    /// digit (a corrupted stream or a join across models).
    pub fn observe(&mut self, id: TokenId, piece: &str) -> Result<(), VocabError> {
        let Some(digit) = rating_digit_of_piece(piece) else {
            return Ok(());
        };
        match self.digits.insert(id, digit) {
            Some(prev) if prev != digit => Err(VocabError::Conflict {
                id,
                first: prev,
                second: digit,
            }),
            _ => Ok(()),
        }
    }

    /// Learn from every sampled token in `snapshots` that carries its id.
    pub fn observe_all<'a>(
        &mut self,
        snapshots: impl IntoIterator<Item = &'a ProbeSnapshot>,
    ) -> Result<(), VocabError> {
        for s in snapshots {
            if let Some(id) = s.token {
                self.observe(id, &s.piece)?;
            }
        }
        Ok(())
    }

    /// Insert a known mapping directly (fixtures; replaying sidecars
    /// recorded before the stream carried token ids).
    pub fn insert(&mut self, id: TokenId, digit: RatingDigit) {
        self.digits.insert(id, digit);
    }

    pub fn digit(&self, id: TokenId) -> Option<RatingDigit> {
        self.digits.get(&id).copied()
    }

    /// Ratings with no known token id.
    pub fn missing(&self) -> Vec<u8> {
        RatingDigit::all()
            .filter(|d| !self.digits.values().any(|v| v == d))
            .map(RatingDigit::get)
            .collect()
    }

    /// Ok when every rating `1..=RATING_MAX` has at least one known id.
    pub fn ensure_complete(&self) -> Result<(), VocabError> {
        let missing = self.missing();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(VocabError::Incomplete { missing })
        }
    }

    pub fn len(&self) -> usize {
        self.digits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.digits.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(n: u8) -> RatingDigit {
        RatingDigit::new(n).unwrap()
    }

    #[test]
    fn pieces_across_tokenizers() {
        for (piece, want) in [
            ("7", Some(7)),
            (" 7", Some(7)),
            ("\u{2581}5", Some(5)),
            ("Ġ3", Some(3)),
            ("\n1", Some(1)),
            ("\"4", Some(4)),
            ("6\"", Some(6)),
            ("77", None),
            ("10", None),
            ("0", None),
            ("8", None),
            (": 7", None),
            ("7,", None),
            ("", None),
            (" ", None),
            ("seven", None),
        ] {
            assert_eq!(
                rating_digit_of_piece(piece).map(RatingDigit::get),
                want,
                "{piece:?}"
            );
        }
    }

    #[test]
    fn vocab_learns_several_ids_per_digit_and_reports_gaps() {
        let mut v = DigitVocab::new();
        v.observe(TokenId(22), "7").unwrap();
        v.observe(TokenId(9001), " 7").unwrap();
        v.observe(TokenId(90), "{").unwrap();
        assert_eq!(v.digit(TokenId(22)), Some(d(7)));
        assert_eq!(v.digit(TokenId(9001)), Some(d(7)));
        assert_eq!(v.digit(TokenId(90)), None);
        assert_eq!(v.missing(), vec![1, 2, 3, 4, 5, 6]);
        assert!(matches!(
            v.ensure_complete(),
            Err(VocabError::Incomplete { .. })
        ));
        for (i, n) in (16..=21).zip(1..=6) {
            v.observe(TokenId(i), &n.to_string()).unwrap();
        }
        v.ensure_complete().unwrap();
    }

    #[test]
    fn vocab_rejects_conflicting_observation() {
        let mut v = DigitVocab::new();
        v.observe(TokenId(22), "7").unwrap();
        let err = v.observe(TokenId(22), "6").unwrap_err();
        assert!(matches!(err, VocabError::Conflict { .. }));
    }

    #[test]
    fn rating_digit_serde_is_a_bare_number() {
        assert_eq!(serde_json::to_string(&d(5)).unwrap(), "5");
        assert_eq!(serde_json::from_str::<RatingDigit>("5").unwrap(), d(5));
        serde_json::from_str::<RatingDigit>("8").unwrap_err();
    }
}
