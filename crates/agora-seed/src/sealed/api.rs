//! The server calls sealed state makes, behind a trait so the commands can
//! be tested against a fake. [`Client`] is the real one.

use agora_agentkit::client::{Client, StateAuth};
use agora_agentkit::crypto::SigningKey;
use agora_agentkit::envelope::EncryptionPublicKey;
use agora_agentkit::ids::AgentId;
use agora_agentkit::state::{KeyRotation, SignedStateEntry, StateKeyRecord};

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
            .map_err(|e| e.to_string())
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
    }

    impl FakeApi {
        pub fn new() -> Self {
            Self {
                chain: Mutex::new(Ok(Vec::new())),
                keys: Mutex::default(),
                calls: Mutex::default(),
                fail: Mutex::default(),
                public_key: Mutex::default(),
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
            rotation.verify_agent_signed().map_err(|e| e.to_string())?;
            *self.public_key.lock().unwrap() = Some(
                rotation
                    .new_public_key
                    .to_verifying_key()
                    .map_err(|e| e.to_string())?,
            );
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
    }
}
