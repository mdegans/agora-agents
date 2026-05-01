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
    pub id: u32,
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
    pub fn lookup_p(&self, token_id: u32) -> Option<f32> {
        self.top_k.iter().find(|t| t.id == token_id).map(|t| t.p)
    }

    /// Rank of a specific token id within the captured top-K (1-indexed
    /// as 1 = argmax). Returns `None` if not in the top-K.
    pub fn lookup_rank(&self, token_id: u32) -> Option<usize> {
        self.top_k.iter().position(|t| t.id == token_id).map(|i| i + 1)
    }

    /// Cumulative probability mass over the top-K. With a state-pure
    /// snapshot taken before sampling, this should be ≤ 1.0; in
    /// practice, a low cumulative top-K mass on a large-vocab model
    /// signals a flat distribution where K=100 isn't enough coverage.
    pub fn top_k_cumulative_mass(&self) -> f32 {
        self.top_k.iter().map(|t| t.p).sum()
    }
}

/// One SSE event from the `/probe` endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum ProbeEvent {
    /// Generation has begun for a new completion. The `id` matches
    /// the eventual `Message.id` returned by `/v1/messages` for that
    /// request.
    SessionStart {
        id: uuid::Uuid,
        model: String,
    },
    /// One token's snapshot. `id` ties this back to a `session_start`.
    Token {
        id: uuid::Uuid,
        ctx: ProbeSnapshot,
    },
    /// Generation has completed for `id`. No more `Token` events with
    /// this id will arrive.
    SessionEnd {
        id: uuid::Uuid,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_session_start() {
        let wire =
            r#"{"event":"session_start","id":"5473a9a3-f523-4273-9a40-888a30376e93","model":"qwen3-5-a17b"}"#;
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
                assert_eq!(snap.top_k[0].id, 16);
                let p16 = snap.lookup_p(16).unwrap();
                assert!((p16 - 0.995_267).abs() < 1e-5, "got {p16}");
                assert_eq!(snap.lookup_rank(24), Some(2));
                assert_eq!(snap.lookup_p(99999), None);
            }
            other => panic!("expected Token, got {other:?}"),
        }
    }

    #[test]
    fn deserializes_session_end() {
        let wire =
            r#"{"event":"session_end","id":"5473a9a3-f523-4273-9a40-888a30376e93"}"#;
        let evt: ProbeEvent = serde_json::from_str(wire).unwrap();
        match evt {
            ProbeEvent::SessionEnd { .. } => {}
            other => panic!("expected SessionEnd, got {other:?}"),
        }
    }
}
