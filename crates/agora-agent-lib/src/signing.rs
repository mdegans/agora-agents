//! Ed25519 signing for Agora agent actions.
//!
//! Re-exports from [`agora_agentkit::crypto`]. The canonical signed message
//! format is: `SHA-256(payload || timestamp_le_bytes)`

pub use agora_agentkit::crypto::{
    generate_keypair, sign, signing_key_from_bytes, signing_key_from_hex, signing_key_to_hex,
    CryptoError, Signature, SigningKey, VerifyingKey,
};
