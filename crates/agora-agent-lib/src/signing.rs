//! Ed25519 signing for Agora agent actions.
//!
//! Vendored from agora-common/src/crypto.rs to avoid pulling in sqlx and other
//! server dependencies. The canonical signed message format is:
//! `SHA-256(payload || timestamp_le_bytes)`

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use sha2::{Digest, Sha256};

/// Generate a new Ed25519 keypair.
pub fn generate_keypair() -> (SigningKey, VerifyingKey) {
    let mut csprng = rand::rngs::OsRng;
    let signing_key = SigningKey::generate(&mut csprng);
    let verifying_key = signing_key.verifying_key();
    (signing_key, verifying_key)
}

/// Sign a payload with the given key and timestamp.
///
/// The canonical signed message is `SHA-256(payload || timestamp_le_bytes)`.
pub fn sign(signing_key: &SigningKey, payload: &[u8], timestamp: i64) -> Signature {
    let digest = canonical_digest(payload, timestamp);
    signing_key.sign(&digest)
}

/// Compute the canonical digest: SHA-256(payload || timestamp_le_bytes).
fn canonical_digest(payload: &[u8], timestamp: i64) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(payload);
    hasher.update(timestamp.to_le_bytes());
    hasher.finalize().to_vec()
}

/// Load a signing key from raw bytes (32 bytes).
pub fn signing_key_from_bytes(bytes: &[u8; 32]) -> SigningKey {
    SigningKey::from_bytes(bytes)
}

/// Load a signing key from a hex-encoded string.
pub fn signing_key_from_hex(hex_str: &str) -> anyhow::Result<SigningKey> {
    let bytes = hex::decode(hex_str.trim())?;
    if bytes.len() != 32 {
        anyhow::bail!("signing key must be 32 bytes, got {}", bytes.len());
    }
    let mut key_bytes = [0u8; 32];
    key_bytes.copy_from_slice(&bytes);
    Ok(SigningKey::from_bytes(&key_bytes))
}

/// Save a signing key as hex.
pub fn signing_key_to_hex(key: &SigningKey) -> String {
    hex::encode(key.to_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Verifier;

    #[test]
    fn sign_and_verify() {
        let (signing_key, verifying_key) = generate_keypair();
        let payload = b"hello agora";
        let timestamp = 1234567890i64;

        let signature = sign(&signing_key, payload, timestamp);
        let digest = canonical_digest(payload, timestamp);
        assert!(verifying_key.verify(&digest, &signature).is_ok());
    }

    #[test]
    fn hex_roundtrip() {
        let (signing_key, _) = generate_keypair();
        let hex_str = signing_key_to_hex(&signing_key);
        let recovered = signing_key_from_hex(&hex_str).unwrap();
        assert_eq!(signing_key.to_bytes(), recovered.to_bytes());
    }
}
