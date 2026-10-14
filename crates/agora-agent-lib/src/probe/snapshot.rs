//! Owned mirror types for the wire shape of blallama's `/probe` SSE
//! events.
//!
//! These types intentionally don't depend on `drama_llama` directly:
//! the consumer needs *owned* data (snapshots persist to disk; lifetime
//! of `Option<&Snapshot>` from the producer side doesn't extend to disk
//! anyway), and pulling drama_llama as a transitive dependency would
//! drag Mac-target-specific deps (metal-rs, etc.) into a crate that's
//! meant to be portable musl-buildable on linux.
//!
//! The wire format (slice-2A in the canary thread) is the contract;
//! these types serde-deserialize from it directly. If drama_llama's
//! internal shape evolves, the consumer schema doesn't have to track
//! it 1:1 — the wire format is the boundary.

use serde::{Deserialize, Serialize};

/// A vocabulary token id as the producer reports it (`ProbeCtx::token`,
/// `TopKEntry::id`). Model-specific: the same id means different text
/// under different tokenizers, which is why ratings are matched by
/// decoded piece (see [`super::digits::DigitVocab`]), never by a
/// hard-coded id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TokenId(pub u32);

impl std::fmt::Display for TokenId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// One token's worth of pre-grammar internal-state snapshot, mirroring
/// drama_llama's `ProbeCtx` on the wire. Owned so it can be persisted
/// and joined against external ratings post-completion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProbeSnapshot {
    /// 0-indexed offset within the generated tokens for this completion.
    /// Position 0 is the first generated token; prefill tokens are not
    /// surfaced.
    pub generation_index: u32,
    /// Absolute position in the model's KV cache for this token.
    /// `n_cur - generation_index` is the prefill length.
    pub n_cur: u32,
    /// The sampled token's id (post-chain), as blallama's `ProbeCtx`
    /// serializes it. Absent from sidecars written before 2026-10-14
    /// (the consumer used to drop it); present on the live wire. Paired
    /// with [`Self::piece`] it tells us which text a top-K id decodes to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<TokenId>,
    /// Decoded text fragment for this token.
    pub piece: String,
    /// The pre-grammar distribution diagnostics. May be absent if the
    /// hook on the producer side opted out of snapshot capture.
    pub snapshot: Option<TokenSnapshot>,
}

/// Pre-grammar distribution shape at one token-emission position. The
/// distribution captured here is *before* `sample_token` consumed the
/// candidates — it's what the model would have produced absent the
/// wrapper chain (grammar mask, sampler, etc.).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenSnapshot {
    /// Shannon entropy in nats over the post-softmax distribution.
    /// High entropy = uncertain; low entropy = peaked (1-2 candidates
    /// dominate).
    pub entropy: f32,
    /// Top-K candidates by post-softmax probability. K is a producer-
    /// side option (default 100); always-keep-argmax discipline
    /// guarantees the actual sampled token is present.
    pub top_k: Vec<TopKEntry>,
}

/// One candidate's pre-grammar mass at a token-emission position.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TopKEntry {
    /// Vocabulary token id.
    pub id: TokenId,
    /// Pre-softmax logit.
    pub logit: f32,
    /// Post-softmax probability mass.
    pub p: f32,
}

impl TokenSnapshot {
    /// Pre-grammar mass on a specific token id, if it appears in the
    /// captured top-K. Returns `None` if the token wasn't in the top-K
    /// (which means its mass was below the K-th candidate — usually
    /// negligible but not always).
    pub fn lookup_p(&self, token_id: TokenId) -> Option<f32> {
        self.top_k.iter().find(|t| t.id == token_id).map(|t| t.p)
    }

    /// Rank of a specific token id within the captured top-K (1-indexed
    /// as 1 = argmax). Returns `None` if not in the top-K.
    pub fn lookup_rank(&self, token_id: TokenId) -> Option<usize> {
        self.top_k
            .iter()
            .position(|t| t.id == token_id)
            .map(|i| i + 1)
    }

    /// Cumulative probability mass over the top-K. With a state-pure
    /// snapshot taken before sampling, this should be ≤ 1.0; in
    /// practice, a low cumulative top-K mass on a large-vocab model
    /// signals a flat distribution where K=100 isn't enough coverage.
    pub fn top_k_cumulative_mass(&self) -> f32 {
        self.top_k.iter().map(|t| t.p).sum()
    }
}

/// Header line of a probe-snapshot sidecar (the JSONL files under
/// `probe/baselines/probe_snapshots/`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SidecarHeader {
    /// Always `"probe_snapshot_header"`.
    pub kind: String,
    pub model_id: String,
    pub questionnaire_version: String,
    pub capture_date: chrono::DateTime<chrono::Utc>,
    pub request_id: uuid::Uuid,
    #[serde(default)]
    pub session_model: Option<String>,
    pub n_tokens: usize,
}

/// A parsed sidecar: its header and one completion's snapshots.
#[derive(Debug, Clone, PartialEq)]
pub struct Sidecar {
    pub header: SidecarHeader,
    pub snapshots: Vec<ProbeSnapshot>,
}

impl Sidecar {
    /// Parse a sidecar's JSONL text: a header line, then one
    /// [`ProbeSnapshot`] per line.
    pub fn parse(jsonl: &str) -> Result<Self, serde_json::Error> {
        let mut lines = jsonl.lines().filter(|l| !l.trim().is_empty());
        let header: SidecarHeader = serde_json::from_str(lines.next().unwrap_or("null"))?;
        let snapshots = lines
            .map(serde_json::from_str::<ProbeSnapshot>)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { header, snapshots })
    }
}

/// One SSE event from the `/probe` endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum ProbeEvent {
    /// Generation has begun for a new completion. The `id` matches
    /// the eventual `Message.id` returned by `/v1/messages` for that
    /// request.
    SessionStart { id: uuid::Uuid, model: String },
    /// One token's snapshot. `id` ties this back to a `session_start`.
    Token { id: uuid::Uuid, ctx: ProbeSnapshot },
    /// Generation has completed for `id`. No more `Token` events with
    /// this id will arrive.
    SessionEnd { id: uuid::Uuid },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_session_start() {
        let wire = r#"{"event":"session_start","id":"5473a9a3-f523-4273-9a40-888a30376e93","model":"qwen3-5-a17b"}"#;
        let evt: ProbeEvent = serde_json::from_str(wire).unwrap();
        match evt {
            ProbeEvent::SessionStart { model, .. } => {
                assert_eq!(model, "qwen3-5-a17b");
            }
            other => panic!("expected SessionStart, got {other:?}"),
        }
    }

    #[test]
    fn deserializes_token_with_snapshot() {
        // Captured live from blallama 2026-05-01.
        let wire = r#"{
            "event": "token",
            "id": "5473a9a3-f523-4273-9a40-888a30376e93",
            "ctx": {
                "generation_index": 13,
                "n_cur": 488,
                "piece": "1",
                "snapshot": {
                    "entropy": 0.03188461437821388,
                    "top_k": [
                        {"id": 16, "logit": 27.326431274414062, "p": 0.9952670931816101},
                        {"id": 24, "logit": 21.871437072753906, "p": 0.004254668951034546}
                    ]
                }
            }
        }"#;
        let evt: ProbeEvent = serde_json::from_str(wire).unwrap();
        match evt {
            ProbeEvent::Token { ctx, .. } => {
                assert_eq!(ctx.generation_index, 13);
                assert_eq!(ctx.piece, "1");
                let snap = ctx.snapshot.as_ref().unwrap();
                assert_eq!(snap.top_k.len(), 2);
                assert_eq!(snap.top_k[0].id, TokenId(16));
                let p16 = snap.lookup_p(TokenId(16)).unwrap();
                assert!((p16 - 0.995_267).abs() < 1e-5, "got {p16}");
                assert_eq!(snap.lookup_rank(TokenId(24)), Some(2));
                assert_eq!(snap.lookup_p(TokenId(99999)), None);
                // Recorded before the consumer kept the sampled id.
                assert_eq!(ctx.token, None);
            }
            other => panic!("expected Token, got {other:?}"),
        }
    }

    /// The live wire carries the sampled token id (`ProbeCtx::token`);
    /// the consumer keeps it so the digit vocab can be learned.
    #[test]
    fn deserializes_token_with_sampled_id() {
        let wire = r#"{"event":"token","id":"5473a9a3-f523-4273-9a40-888a30376e93",
            "ctx":{"token":22,"generation_index":4,"n_cur":400,"piece":"7",
                   "snapshot":{"entropy":0.1,"top_k":[{"id":22,"logit":1.0,"p":0.9}],
                               "top_k_cumulative_mass":0.9}}}"#;
        match serde_json::from_str::<ProbeEvent>(wire).unwrap() {
            ProbeEvent::Token { ctx, .. } => assert_eq!(ctx.token, Some(TokenId(22))),
            other => panic!("expected Token, got {other:?}"),
        }
    }

    #[test]
    fn deserializes_session_end() {
        let wire = r#"{"event":"session_end","id":"5473a9a3-f523-4273-9a40-888a30376e93"}"#;
        let evt: ProbeEvent = serde_json::from_str(wire).unwrap();
        match evt {
            ProbeEvent::SessionEnd { .. } => {}
            other => panic!("expected SessionEnd, got {other:?}"),
        }
    }
}
