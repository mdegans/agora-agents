//! The server calls sealed state makes, behind a trait so the commands can
//! be tested against a fake. [`Client`] is the real one.

use agora_agentkit::client::{Client, StateAuth};
use agora_agentkit::crypto::SigningKey;
use agora_agentkit::envelope::EncryptionPublicKey;
use agora_agentkit::ids::{AgentId, GovernanceLogId};
use agora_agentkit::state::{AnchorProof, KeyRotation, SignedStateEntry, StateKeyRecord};

/// How an error string from [`SealedApi::rotate_key`] begins when the
/// server's head moved on (a 409).
pub const CONFLICT: &str = "conflict: ";

/// What sealed state asks of Agora, each signed as the agent.
// async_trait marks each boxed future `#[must_use]`, which it already is
#[allow(clippy::double_must_use)]
#[async_trait::async_trait]
pub trait SealedApi: Send + Sync {
    /// The agent's chain, in seq order.
    async fn chain(&self, id: AgentId, key: &SigningKey) -> Result<Vec<SignedStateEntry>, String>;
    /// The state key records the server holds. Verify before use.
    async fn state_keys(
        &self,
        id: AgentId,
        key: &SigningKey,
    ) -> Result<Vec<StateKeyRecord>, String>;
    async fn post_state_key(
        &self,
        id: AgentId,
        key: &SigningKey,
        record: &StateKeyRecord,
    ) -> Result<(), String>;
    /// Signed by the old key.
    async fn rotate_key(
        &self,
        id: AgentId,
        old: &SigningKey,
        rotation: &KeyRotation,
    ) -> Result<(), String>;
    async fn register_encryption_key(
        &self,
        id: AgentId,
        key: &SigningKey,
        public: &EncryptionPublicKey,
    ) -> Result<(), String>;
    /// Whether the server takes `key` as the agent's signing key now: a
    /// signed read with it. For a rotation whose answer was lost.
    async fn signer_registered(&self, id: AgentId, key: &SigningKey) -> Result<bool, String>;
    /// The rotations the server lists. Advisory: never trusted.
    async fn rotations(&self, id: AgentId, key: &SigningKey) -> Result<Vec<KeyRotation>, String>;
    /// One of the agent's sealed blobs.
    async fn blob(
        &self,
        id: AgentId,
        key: &SigningKey,
        sha256: &agora_agentkit::govlog::Sha256Hex,
    ) -> Result<Vec<u8>, String>;
    /// The agent's heads in `ANC-` entries after `after`, oldest first.
    /// Untrusted until checked ([`super::anchors`]).
    async fn proofs(
        &self,
        id: AgentId,
        key: &SigningKey,
        after: Option<GovernanceLogId>,
    ) -> Result<Vec<AnchorProof>, String>;
    /// Whether Agora holds a sealed head for the agent: `Ok(false)` when it
    /// answers `head: null` (nothing sealed there) or 404s (an unknown or
    /// deleted agent); anything else unanswered is `Err`.
    async fn head_known(&self, id: AgentId, key: &SigningKey) -> Result<bool, String>;
    /// Whether the agent's anchored head is shown on its profile.
    async fn post_anchor(&self, id: AgentId, key: &SigningKey, publish: bool)
    -> Result<(), String>;
}

fn auth(id: AgentId, key: &SigningKey) -> StateAuth<'_> {
    StateAuth { agent_id: id, key }
}

#[async_trait::async_trait]
impl SealedApi for Client {
    async fn chain(&self, id: AgentId, key: &SigningKey) -> Result<Vec<SignedStateEntry>, String> {
        self.get_state_chain(auth(id, key))
            .await
            .map_err(|e| e.to_string())
    }

    async fn state_keys(
        &self,
        id: AgentId,
        key: &SigningKey,
    ) -> Result<Vec<StateKeyRecord>, String> {
        self.get_state_keys(auth(id, key))
            .await
            .map(|r| r.keys)
            .map_err(|e| e.to_string())
    }

    async fn post_state_key(
        &self,
        id: AgentId,
        key: &SigningKey,
        record: &StateKeyRecord,
    ) -> Result<(), String> {
        Client::post_state_key(self, auth(id, key), record)
            .await
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    async fn rotate_key(
        &self,
        id: AgentId,
        old: &SigningKey,
        rotation: &KeyRotation,
    ) -> Result<(), String> {
        Client::rotate_key(self, auth(id, old), rotation)
            .await
            .map(|_| ())
            .map_err(|e| match agora_agentkit::client::state_conflict(&e) {
                Some(head) => format!("{CONFLICT}{head:?}"),
                None => e.to_string(),
            })
    }

    async fn register_encryption_key(
        &self,
        id: AgentId,
        key: &SigningKey,
        public: &EncryptionPublicKey,
    ) -> Result<(), String> {
        Client::register_encryption_key(self, id, key, public)
            .await
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    async fn signer_registered(&self, id: AgentId, key: &SigningKey) -> Result<bool, String> {
        match self.get_state_head(auth(id, key)).await {
            Ok(_) => Ok(true),
            Err(agora_agentkit::client::Error::Status { status, .. })
                if matches!(status.as_u16(), 401 | 403) =>
            {
                Ok(false)
            }
            Err(e) => Err(e.to_string()),
        }
    }

    async fn rotations(&self, id: AgentId, key: &SigningKey) -> Result<Vec<KeyRotation>, String> {
        self.get_rotations(auth(id, key))
            .await
            .map_err(|e| e.to_string())
    }

    async fn blob(
        &self,
        id: AgentId,
        key: &SigningKey,
        sha256: &agora_agentkit::govlog::Sha256Hex,
    ) -> Result<Vec<u8>, String> {
        self.get_state_blob(auth(id, key), sha256)
            .await
            .map_err(|e| e.to_string())
    }

    async fn head_known(&self, id: AgentId, key: &SigningKey) -> Result<bool, String> {
        match self.get_state_head(auth(id, key)).await {
            Ok(_) => Ok(true),
            Err(agora_agentkit::client::Error::Status { status, .. }) if status.as_u16() == 404 => {
                Ok(false)
            }
            Err(e) => Err(e.to_string()),
        }
    }

    async fn proofs(
        &self,
        id: AgentId,
        key: &SigningKey,
        after: Option<GovernanceLogId>,
    ) -> Result<Vec<AnchorProof>, String> {
        self.get_state_proofs(auth(id, key), after)
            .await
            .map_err(|e| e.to_string())
    }

    async fn post_anchor(
        &self,
        id: AgentId,
        key: &SigningKey,
        publish: bool,
    ) -> Result<(), String> {
        self.post_state_anchor(auth(id, key), publish)
            .await
            .map(drop)
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
pub mod fake {
    use super::*;
    use std::sync::Mutex;

    /// An in-memory Agora: a chain, key records, the signing key it knows,
    /// and every call in order.
    pub struct FakeApi {
        pub chain: Mutex<Result<Vec<SignedStateEntry>, String>>,
        pub keys: Mutex<Vec<StateKeyRecord>>,
        pub calls: Mutex<Vec<String>>,
        /// The call that fails, by name.
        pub fail: Mutex<Option<&'static str>>,
        pub public_key: Mutex<Option<agora_agentkit::crypto::VerifyingKey>>,
        /// What the server lists as the agent's rotations.
        pub rotations: Mutex<Vec<KeyRotation>>,
        /// Blobs the server holds, by hash.
        pub blobs: Mutex<std::collections::HashMap<agora_agentkit::govlog::Sha256Hex, Vec<u8>>>,
        /// The proofs the server lists, all of them: `after` filters.
        pub proofs: Mutex<Vec<AnchorProof>>,
        /// Every `post_anchor`'s `publish`.
        pub anchored: Mutex<Vec<bool>>,
        /// The agent is unknown to the server (deleted): a 404.
        pub unknown_agent: Mutex<bool>,
    }

    impl FakeApi {
        pub fn new() -> Self {
            Self {
                chain: Mutex::new(Ok(Vec::new())),
                keys: Mutex::default(),
                calls: Mutex::default(),
                fail: Mutex::default(),
                public_key: Mutex::default(),
                rotations: Mutex::default(),
                blobs: Mutex::default(),
                proofs: Mutex::default(),
                anchored: Mutex::default(),
                unknown_agent: Mutex::default(),
            }
        }

        fn call(&self, name: &'static str) -> Result<(), String> {
            self.calls.lock().unwrap().push(name.to_string());
            match *self.fail.lock().unwrap() {
                Some(f) if f == name => Err(format!("{name} refused")),
                _ => Ok(()),
            }
        }

        /// The request must be signed by the key the server knows.
        fn signed_by(&self, key: &SigningKey) -> Result<(), String> {
            match *self.public_key.lock().unwrap() {
                Some(known) if known != key.verifying_key() => Err("unknown signer".into()),
                _ => Ok(()),
            }
        }
    }

    #[async_trait::async_trait]
    impl SealedApi for FakeApi {
        async fn chain(&self, _: AgentId, _: &SigningKey) -> Result<Vec<SignedStateEntry>, String> {
            self.call("chain")?;
            self.chain.lock().unwrap().clone()
        }

        async fn state_keys(
            &self,
            _: AgentId,
            _: &SigningKey,
        ) -> Result<Vec<StateKeyRecord>, String> {
            self.call("state_keys")?;
            Ok(self.keys.lock().unwrap().clone())
        }

        async fn post_state_key(
            &self,
            _: AgentId,
            key: &SigningKey,
            record: &StateKeyRecord,
        ) -> Result<(), String> {
            self.call("post_state_key")?;
            self.signed_by(key)?;
            if !record.verify(&key.verifying_key()) {
                return Err("record not signed by the agent".into());
            }
            self.keys.lock().unwrap().push(record.clone());
            Ok(())
        }

        async fn rotate_key(
            &self,
            _: AgentId,
            old: &SigningKey,
            rotation: &KeyRotation,
        ) -> Result<(), String> {
            self.call("rotate_key")?;
            self.signed_by(old)?;
            if *self.fail.lock().unwrap() == Some("rotate_key_conflict") {
                return Err(format!("{CONFLICT}another head"));
            }
            rotation.verify_agent_signed().map_err(|e| e.to_string())?;
            *self.public_key.lock().unwrap() = Some(
                rotation
                    .new_public_key
                    .to_verifying_key()
                    .map_err(|e| e.to_string())?,
            );
            // Taken, but the answer never arrives
            if *self.fail.lock().unwrap() == Some("rotate_key_reply") {
                return Err("connection reset".into());
            }
            Ok(())
        }

        async fn register_encryption_key(
            &self,
            _: AgentId,
            key: &SigningKey,
            _: &EncryptionPublicKey,
        ) -> Result<(), String> {
            self.call("register_encryption_key")?;
            self.signed_by(key)
        }

        async fn signer_registered(&self, _: AgentId, key: &SigningKey) -> Result<bool, String> {
            self.call("signer_registered")?;
            Ok(self.signed_by(key).is_ok())
        }

        async fn rotations(&self, _: AgentId, _: &SigningKey) -> Result<Vec<KeyRotation>, String> {
            self.call("rotations")?;
            Ok(self.rotations.lock().unwrap().clone())
        }

        async fn blob(
            &self,
            _: AgentId,
            _: &SigningKey,
            sha256: &agora_agentkit::govlog::Sha256Hex,
        ) -> Result<Vec<u8>, String> {
            self.call("blob")?;
            self.blobs
                .lock()
                .unwrap()
                .get(sha256)
                .cloned()
                .ok_or_else(|| "404".to_string())
        }

        async fn proofs(
            &self,
            _: AgentId,
            _: &SigningKey,
            after: Option<GovernanceLogId>,
        ) -> Result<Vec<AnchorProof>, String> {
            self.call("proofs")?;
            Ok(self
                .proofs
                .lock()
                .unwrap()
                .iter()
                .filter(|p| after.as_ref().is_none_or(|a| p.anc_id > *a))
                .cloned()
                .collect())
        }

        /// As the real server: 404 for an unknown agent, `head: null`
        /// (false) for no chain, the head otherwise.
        async fn head_known(&self, _: AgentId, _: &SigningKey) -> Result<bool, String> {
            self.call("head_known")?;
            if *self.unknown_agent.lock().unwrap() {
                return Ok(false);
            }
            match &*self.chain.lock().unwrap() {
                Ok(chain) => Ok(!chain.is_empty()),
                Err(e) => Err(e.clone()),
            }
        }

        async fn post_anchor(
            &self,
            _: AgentId,
            key: &SigningKey,
            publish: bool,
        ) -> Result<(), String> {
            self.call("post_anchor")?;
            self.signed_by(key)?;
            self.anchored.lock().unwrap().push(publish);
            Ok(())
        }
    }
}
