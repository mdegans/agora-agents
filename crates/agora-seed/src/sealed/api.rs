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
    ) -> Result<Vec<u8>, BlobError>;
    /// The agent's heads in `ANC-` entries after `after`, oldest first.
    /// Untrusted until checked ([`super::anchors`]).
    async fn proofs(
        &self,
        id: AgentId,
        key: &SigningKey,
        after: Option<GovernanceLogId>,
    ) -> Result<Vec<AnchorProof>, String>;
    /// An `ANC-` entry's `data`, as Agora serves it (public, unsigned).
    /// Untrusted until it hashes to the `data_hash` the verified
    /// governance log attests ([`super::anchors::VerifiedAnchors::date`]).
    async fn anchor_data(&self, anc: &GovernanceLogId) -> Result<serde_json::Value, String>;
    /// Agora's head for the agent, asked apart from its chain: `None` for
    /// `head: null` or a 404.
    async fn head(
        &self,
        id: AgentId,
        key: &SigningKey,
    ) -> Result<Option<agora_agentkit::govlog::Sha256Hex>, String>;
    /// Whether Agora holds a sealed head for the agent: `Ok(false)` when it
    /// answers `head: null` (nothing sealed there) or 404s (an unknown or
    /// deleted agent); anything else unanswered is `Err`.
    async fn head_known(&self, id: AgentId, key: &SigningKey) -> Result<bool, String>;
    /// Whether a publish choice was ever recorded on Agora for the agent,
    /// by this runner or by the agent's own `config`. A signed read.
    async fn publish_chosen(&self, id: AgentId, key: &SigningKey) -> Result<bool, String>;
    /// Whether the agent's anchored head is shown on its profile.
    async fn post_anchor(&self, id: AgentId, key: &SigningKey, publish: bool)
    -> Result<(), String>;
}

/// A blob fetch that failed, and whether trying again could help.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct BlobError {
    /// Transient (a 5xx, a timeout, a connection): worth another try. A
    /// 4xx (a 404 above all) is not: the answer won't change.
    pub retry: bool,
    pub message: String,
}

impl BlobError {
    pub fn transient(message: impl Into<String>) -> Self {
        Self {
            retry: true,
            message: message.into(),
        }
    }

    pub fn permanent(message: impl Into<String>) -> Self {
        Self {
            retry: false,
            message: message.into(),
        }
    }
}

/// [`SealedApi::chain`], in seq order with duplicates dropped: the order
/// or repetition of Agora's answer is never evidence, only its entries.
/// Every caller reads the chain through this.
pub async fn chain(
    api: &dyn SealedApi,
    id: AgentId,
    key: &SigningKey,
) -> Result<Vec<SignedStateEntry>, String> {
    let mut chain = api.chain(id, key).await?;
    let mut seen = std::collections::HashSet::new();
    chain.retain(|e| seen.insert((e.entry_hash(), e.signature)));
    chain.sort_by_key(|e| e.entry.seq);
    Ok(chain)
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
    ) -> Result<Vec<u8>, BlobError> {
        self.get_state_blob(auth(id, key), sha256)
            .await
            .map_err(|e| match &e {
                agora_agentkit::client::Error::Status { status, .. }
                    if status.is_client_error() =>
                {
                    BlobError::permanent(e.to_string())
                }
                _ => BlobError::transient(e.to_string()),
            })
    }

    async fn head(
        &self,
        id: AgentId,
        key: &SigningKey,
    ) -> Result<Option<agora_agentkit::govlog::Sha256Hex>, String> {
        match self.get_state_head(auth(id, key)).await {
            Ok(r) => Ok(r.head.map(|h| h.entry_hash)),
            Err(agora_agentkit::client::Error::Status { status, .. }) if status.as_u16() == 404 => {
                Ok(None)
            }
            Err(e) => Err(e.to_string()),
        }
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

    async fn anchor_data(&self, anc: &GovernanceLogId) -> Result<serde_json::Value, String> {
        use agora_agentkit::enums::DetailLevel;
        use agora_agentkit::requests::GetContentInput;
        use agora_agentkit::responses::ContentResponse;
        // As signed, not with revisions applied: the log attests the
        // original's hash
        let mut input =
            GetContentInput::new(anc.clone()).with_detail(DetailLevel::FullWithAttachments);
        input.version = Some(agora_agentkit::enums::RecordVersion::Original);
        let content = self.get_content(&input).await.map_err(|e| e.to_string())?;
        match content {
            ContentResponse::Governance(entry) => entry
                .data
                .ok_or_else(|| format!("{anc} was served without its data")),
            _ => Err(format!("{anc} was not served as a governance-log entry")),
        }
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

    async fn publish_chosen(&self, id: AgentId, key: &SigningKey) -> Result<bool, String> {
        self.config(id, key, &agora_agentkit::requests::ConfigInput::default())
            .await
            .map(|c| c.publish_state_anchor_chosen)
            .map_err(|e| e.to_string())
    }
}

/// How an error string from [`UploadGuard`] begins: the call was for an
/// agent that doesn't upload, and was never sent ([`LocalOnly`]).
pub const LOCAL_ONLY: &str = "local only: ";

/// Whether an error string from a [`SealedApi`] call is [`UploadGuard`]'s
/// refusal: nothing was sent. For the local paths of the steps after B1
/// (check, accept, restore), which tell a refusal from a server's error.
#[allow(dead_code)]
pub fn is_local_only(error: &str) -> bool {
    error.starts_with(LOCAL_ONLY)
}

/// A per-agent [`SealedApi`] call, as [`UploadGuard`] judges it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Call {
    Chain,
    StateKeys,
    PostStateKey,
    /// A rotation that names a head of the agent's chain (`old_key_last`).
    /// One that names none carries nothing of the chain, and is not
    /// guarded: every agent's signing key can be rotated.
    RotateKeyAtHead,
    Rotations,
    Blob,
    Proofs,
    Head,
    HeadKnown,
    PublishChosen,
    PostAnchor,
}

impl Call {
    /// What the opt-in step ([`UploadGuard::opting_in`]) needs before the
    /// agent is marked uploading: whether Agora holds a chain or a state
    /// key record of it, and posting the record. Nothing of the chain.
    pub const OPT_IN: [Self; 5] = [
        Self::Chain,
        Self::Head,
        Self::HeadKnown,
        Self::StateKeys,
        Self::PostStateKey,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Chain => "chain",
            Self::StateKeys => "state_keys",
            Self::PostStateKey => "post_state_key",
            Self::RotateKeyAtHead => "rotate_key",
            Self::Rotations => "rotations",
            Self::Blob => "blob",
            Self::Proofs => "proofs",
            Self::Head => "head",
            Self::HeadKnown => "head_known",
            Self::PublishChosen => "publish_chosen",
            Self::PostAnchor => "post_anchor",
        }
    }
}

impl Call {
    /// A read of what is already on Agora: sends nothing of the agent's
    /// state or choices. The rest are writes.
    pub fn is_read(self) -> bool {
        matches!(
            self,
            Self::Chain
                | Self::Head
                | Self::HeadKnown
                | Self::StateKeys
                | Self::Rotations
                | Self::Blob
                | Self::Proofs
        )
    }
}

/// What [`UploadGuard`] asks of an agent, by its two predicates.
pub trait UploadGate: Send + Sync {
    /// May send its state, a key record or a choice: it uploads
    /// ([`super::uploads`]: marked uploading, not stopped).
    fn may_send(&self, id: AgentId) -> bool;
    /// May read back what is already on Agora: it has ever uploaded
    /// ([`super::uploading::has_uploaded`]: a marker naming it in either
    /// copy, stopped or not, contradictory or not).
    fn may_read(&self, id: AgentId) -> bool;
}

impl<G: UploadGate> UploadGate for std::sync::Arc<G> {
    fn may_send(&self, id: AgentId) -> bool {
        (**self).may_send(id)
    }
    fn may_read(&self, id: AgentId) -> bool {
        (**self).may_read(id)
    }
}

impl std::fmt::Display for Call {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// [`UploadGuard`]'s refusal: `call` was for `id`, which doesn't upload,
/// and nothing was sent. Crosses the [`SealedApi`] boundary as a string
/// beginning [`LOCAL_ONLY`] ([`is_local_only`]); a [`BlobError`] that is
/// not worth a retry for [`SealedApi::blob`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "local only: agent {id} doesn't upload its sealed state (or, for a read, never did); \
     `{call}` was not sent to Agora"
)]
pub struct LocalOnly {
    pub id: AgentId,
    pub call: Call,
}

impl From<LocalOnly> for String {
    fn from(e: LocalOnly) -> Self {
        e.to_string()
    }
}

impl From<LocalOnly> for BlobError {
    fn from(e: LocalOnly) -> Self {
        BlobError::permanent(e.to_string())
    }
}

/// The per-agent server client, guarded: a per-agent call the agent's
/// state doesn't allow returns [`LocalOnly`] without touching the network.
/// The backstop behind agentkit's drain gate: code that forgets to ask
/// can't leak an agent's chain, key record, blobs or publish choice.
///
/// | Call | Passes when |
/// |---|---|
/// | **write**: `post_state_key`, `publish_chosen`, `post_anchor`, a `rotate_key` naming a head | the agent uploads ([`UploadGate::may_send`]: marked uploading, not stopped) |
/// | **read**: `chain`, `head`, `head_known`, `state_keys`, `rotations`, `blob`, `proofs` ([`Call::is_read`]) | the agent has ever uploaded ([`UploadGate::may_read`]: a marker naming it in either copy; stopped or in doubt, it can still be restored from, and read back, what is on Agora) |
/// | `anchor_data`, `register_encryption_key`, `signer_registered`, a `rotate_key` naming no head | always: the public governance log, and the agent's identity on Agora, which every agent has |
///
/// An agent with no marker in either copy makes no sealed-state call at
/// all: nothing of it is on Agora to read.
///
/// **Opting in.** The opt-in step posts the agent's state key record
/// before the agent is marked uploading, so no entry ever reaches Agora
/// ahead of the record that opens it. It holds [`Self::opting_in`] for
/// that: while held, the [`Call::OPT_IN`] calls pass for that one agent,
/// nothing else. Then it marks the agent uploading
/// ([`super::uploading::set`]), drops the hold, and drains.
pub struct UploadGuard<A, E> {
    inner: A,
    enrolment: E,
    opting_in: std::sync::Mutex<std::collections::HashSet<AgentId>>,
    /// Tests: a refused call panics, failing the test that made it.
    #[cfg(test)]
    strict: bool,
}

impl<A, E: UploadGate> UploadGuard<A, E> {
    pub fn new(inner: A, enrolment: E) -> Self {
        Self {
            inner,
            enrolment,
            opting_in: Default::default(),
            #[cfg(test)]
            strict: false,
        }
    }

    /// What it guards.
    #[cfg(test)]
    pub fn inner(&self) -> &A {
        &self.inner
    }

    /// Any refused call panics: for tests where none may be made.
    #[cfg(test)]
    pub fn strict(mut self) -> Self {
        self.strict = true;
        self
    }

    /// Whether `call` for `id` may go to Agora.
    pub fn check(&self, id: AgentId, call: Call) -> Result<(), LocalOnly> {
        let allowed = if call.is_read() {
            self.enrolment.may_read(id)
        } else {
            self.enrolment.may_send(id)
        };
        if allowed {
            return Ok(());
        }
        if Call::OPT_IN.contains(&call)
            && self
                .opting_in
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .contains(&id)
        {
            return Ok(());
        }
        let refused = LocalOnly { id, call };
        #[cfg(test)]
        if self.strict {
            panic!("{refused}");
        }
        tracing::warn!(
            event_type = "sealed_state_call_local_only",
            agent_id = %id,
            call = call.as_str(),
            "a sealed-state call for an agent that doesn't upload was refused, unsent"
        );
        Err(refused)
    }

    /// Let the [`Call::OPT_IN`] calls for `id` through until the returned
    /// hold is dropped (see the type's docs).
    pub fn opting_in(&self, id: AgentId) -> OptingIn<'_> {
        self.opting_in
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id);
        OptingIn {
            set: &self.opting_in,
            id,
        }
    }
}

/// Tests read the fake behind the guard as the fake.
#[cfg(test)]
impl<A, E> std::ops::Deref for UploadGuard<A, E> {
    type Target = A;
    fn deref(&self) -> &A {
        &self.inner
    }
}

/// [`UploadGuard::opting_in`]'s hold.
#[must_use = "the opt-in calls pass only while this is held"]
pub struct OptingIn<'a> {
    set: &'a std::sync::Mutex<std::collections::HashSet<AgentId>>,
    id: AgentId,
}

impl Drop for OptingIn<'_> {
    fn drop(&mut self) {
        self.set
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.id);
    }
}

#[async_trait::async_trait]
impl<A, E> SealedApi for UploadGuard<A, E>
where
    A: SealedApi,
    E: UploadGate,
{
    async fn chain(&self, id: AgentId, key: &SigningKey) -> Result<Vec<SignedStateEntry>, String> {
        self.check(id, Call::Chain)?;
        self.inner.chain(id, key).await
    }

    async fn state_keys(
        &self,
        id: AgentId,
        key: &SigningKey,
    ) -> Result<Vec<StateKeyRecord>, String> {
        self.check(id, Call::StateKeys)?;
        self.inner.state_keys(id, key).await
    }

    async fn post_state_key(
        &self,
        id: AgentId,
        key: &SigningKey,
        record: &StateKeyRecord,
    ) -> Result<(), String> {
        self.check(id, Call::PostStateKey)?;
        self.inner.post_state_key(id, key, record).await
    }

    async fn rotate_key(
        &self,
        id: AgentId,
        old: &SigningKey,
        rotation: &KeyRotation,
    ) -> Result<(), String> {
        if rotation.old_key_last.is_some() {
            self.check(id, Call::RotateKeyAtHead)?;
        }
        self.inner.rotate_key(id, old, rotation).await
    }

    async fn register_encryption_key(
        &self,
        id: AgentId,
        key: &SigningKey,
        public: &EncryptionPublicKey,
    ) -> Result<(), String> {
        self.inner.register_encryption_key(id, key, public).await
    }

    async fn signer_registered(&self, id: AgentId, key: &SigningKey) -> Result<bool, String> {
        self.inner.signer_registered(id, key).await
    }

    async fn rotations(&self, id: AgentId, key: &SigningKey) -> Result<Vec<KeyRotation>, String> {
        self.check(id, Call::Rotations)?;
        self.inner.rotations(id, key).await
    }

    async fn blob(
        &self,
        id: AgentId,
        key: &SigningKey,
        sha256: &agora_agentkit::govlog::Sha256Hex,
    ) -> Result<Vec<u8>, BlobError> {
        self.check(id, Call::Blob)?;
        self.inner.blob(id, key, sha256).await
    }

    async fn proofs(
        &self,
        id: AgentId,
        key: &SigningKey,
        after: Option<GovernanceLogId>,
    ) -> Result<Vec<AnchorProof>, String> {
        self.check(id, Call::Proofs)?;
        self.inner.proofs(id, key, after).await
    }

    async fn anchor_data(&self, anc: &GovernanceLogId) -> Result<serde_json::Value, String> {
        self.inner.anchor_data(anc).await
    }

    async fn head(
        &self,
        id: AgentId,
        key: &SigningKey,
    ) -> Result<Option<agora_agentkit::govlog::Sha256Hex>, String> {
        self.check(id, Call::Head)?;
        self.inner.head(id, key).await
    }

    async fn head_known(&self, id: AgentId, key: &SigningKey) -> Result<bool, String> {
        self.check(id, Call::HeadKnown)?;
        self.inner.head_known(id, key).await
    }

    async fn publish_chosen(&self, id: AgentId, key: &SigningKey) -> Result<bool, String> {
        self.check(id, Call::PublishChosen)?;
        self.inner.publish_chosen(id, key).await
    }

    async fn post_anchor(
        &self,
        id: AgentId,
        key: &SigningKey,
        publish: bool,
    ) -> Result<(), String> {
        self.check(id, Call::PostAnchor)?;
        self.inner.post_anchor(id, key, publish).await
    }
}

/// The runner's guarded client: [`Client`] behind [`UploadGuard`], asking
/// [`super::SealedEnrolment`]. Every per-agent sealed-state call goes
/// through one of these.
pub type GuardedClient = UploadGuard<Client, std::sync::Arc<super::SealedEnrolment>>;

/// [`GuardedClient`] over `client`.
pub fn guarded(client: Client, enrolment: std::sync::Arc<super::SealedEnrolment>) -> GuardedClient {
    UploadGuard::new(client, enrolment)
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
        /// How long each blob answer takes, if set (a server hanging).
        pub blob_delay: Mutex<Option<std::time::Duration>>,
        /// Blobs the server holds, by hash.
        pub blobs: Mutex<std::collections::HashMap<agora_agentkit::govlog::Sha256Hex, Vec<u8>>>,
        /// The proofs the server lists, all of them: `after` filters.
        pub proofs: Mutex<Vec<AnchorProof>>,
        /// At most this many proofs a request (the real server's page is
        /// 100); `None`, all.
        pub proofs_page: Mutex<Option<usize>>,
        /// Answer every proof whatever `after` asks: a server whose pages
        /// don't advance.
        pub proofs_ignore_after: Mutex<bool>,
        /// `ANC-` entries' `data`, as served, by id.
        pub anchor_data: Mutex<std::collections::HashMap<GovernanceLogId, serde_json::Value>>,
        /// Every `post_anchor`'s `publish`.
        pub anchored: Mutex<Vec<bool>>,
        /// A publish choice is on record (the agent's own `config`); any
        /// `post_anchor` sets it too.
        pub chosen: Mutex<bool>,
        /// The agent is unknown to the server (deleted): a 404.
        pub unknown_agent: Mutex<bool>,
        /// What `head` answers instead of the chain's last: a server whose
        /// answers disagree.
        pub head_override: Mutex<Option<Option<agora_agentkit::govlog::Sha256Hex>>>,
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
                blob_delay: Mutex::default(),
                proofs: Mutex::default(),
                proofs_page: Mutex::default(),
                proofs_ignore_after: Mutex::default(),
                anchor_data: Mutex::default(),
                anchored: Mutex::default(),
                chosen: Mutex::default(),
                unknown_agent: Mutex::default(),
                head_override: Mutex::default(),
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

    /// [`FakeApi`] behind a strict [`UploadGuard`] asking the runner's
    /// own [`super::super::SealedEnrolment`]: any per-agent call for an
    /// agent that doesn't upload fails the test.
    pub type Guarded = UploadGuard<FakeApi, std::sync::Arc<super::super::SealedEnrolment>>;

    /// [`Guarded`] over the agents of `root` (`state/`, `secrets/`).
    pub fn guarded(root: &std::path::Path) -> Guarded {
        UploadGuard::new(
            FakeApi::new(),
            std::sync::Arc::new(super::super::SealedEnrolment::new(
                root.join("state"),
                root.join("secrets"),
            )),
        )
        .strict()
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
        ) -> Result<Vec<u8>, BlobError> {
            self.call("blob").map_err(BlobError::transient)?;
            let delay = *self.blob_delay.lock().unwrap();
            if let Some(delay) = delay {
                tokio::time::sleep(delay).await;
            }
            self.blobs
                .lock()
                .unwrap()
                .get(sha256)
                .cloned()
                .ok_or_else(|| BlobError::permanent("HTTP 404 Not Found"))
        }

        async fn proofs(
            &self,
            _: AgentId,
            _: &SigningKey,
            after: Option<GovernanceLogId>,
        ) -> Result<Vec<AnchorProof>, String> {
            self.call("proofs")?;
            let after = after.filter(|_| !*self.proofs_ignore_after.lock().unwrap());
            let mut page: Vec<AnchorProof> = self
                .proofs
                .lock()
                .unwrap()
                .iter()
                .filter(|p| after.as_ref().is_none_or(|a| p.anc_id > *a))
                .cloned()
                .collect();
            if let Some(n) = *self.proofs_page.lock().unwrap() {
                page.truncate(n);
            }
            Ok(page)
        }

        async fn anchor_data(&self, anc: &GovernanceLogId) -> Result<serde_json::Value, String> {
            self.call("anchor_data")?;
            self.anchor_data
                .lock()
                .unwrap()
                .get(anc)
                .cloned()
                .ok_or_else(|| "404".to_string())
        }

        async fn head(
            &self,
            _: AgentId,
            _: &SigningKey,
        ) -> Result<Option<agora_agentkit::govlog::Sha256Hex>, String> {
            self.call("head")?;
            if let Some(h) = *self.head_override.lock().unwrap() {
                return Ok(h);
            }
            if *self.unknown_agent.lock().unwrap() {
                return Ok(None);
            }
            match &*self.chain.lock().unwrap() {
                Ok(chain) => Ok(chain
                    .iter()
                    .max_by_key(|e| e.entry.seq)
                    .map(|e| e.entry_hash())),
                Err(e) => Err(e.clone()),
            }
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
            *self.chosen.lock().unwrap() = true;
            Ok(())
        }

        async fn publish_chosen(&self, _: AgentId, key: &SigningKey) -> Result<bool, String> {
            self.call("publish_chosen")?;
            self.signed_by(key)?;
            Ok(*self.chosen.lock().unwrap())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::FakeApi;
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// One switch for every agent: whether it uploads (`.0`), and whether
    /// it ever did (`.1`).
    #[derive(Default)]
    struct Switch(AtomicBool, AtomicBool);

    impl UploadGate for &Switch {
        fn may_send(&self, _: AgentId) -> bool {
            self.0.load(Ordering::SeqCst)
        }
        fn may_read(&self, _: AgentId) -> bool {
            self.1.load(Ordering::SeqCst)
        }
    }

    /// The names of [`every_call`], in its order.
    const EVERY: [&str; 11] = [
        "chain",
        "state_keys",
        "post_state_key",
        "rotate_key",
        "rotations",
        "blob",
        "proofs",
        "head",
        "head_known",
        "publish_chosen",
        "post_anchor",
    ];

    /// The calls of [`every_call`] that weren't refused.
    fn passed(results: &[Result<(), String>]) -> Vec<&'static str> {
        EVERY
            .into_iter()
            .zip(results)
            .filter(|(_, r)| r.as_ref().err().is_none_or(|e| !is_local_only(e)))
            .map(|(n, _)| n)
            .collect()
    }

    fn rotation(id: AgentId, key: &SigningKey, at_head: bool) -> KeyRotation {
        KeyRotation::agent_signed(
            id,
            0,
            key,
            &agora_agentkit::crypto::generate_keypair().0,
            at_head.then_some(agora_agentkit::state::verify::HeadRef {
                entry_hash: agora_agentkit::govlog::Sha256Hex::from([1; 32]),
                seq: 3,
            }),
            chrono::Utc::now(),
        )
    }

    /// Every per-agent call, made once: what each returned (`Err` with
    /// the refusal) and what reached the fake.
    async fn every_call(
        api: &dyn SealedApi,
        id: AgentId,
        key: &SigningKey,
    ) -> Vec<Result<(), String>> {
        let record = StateKey::generate()
            .wrap(
                id,
                0,
                0,
                &agora_agentkit::envelope::EncryptionPublicKey::from(
                    &agora_agentkit::envelope::generate_encryption_keypair().0,
                ),
                key,
            )
            .unwrap();
        let sha = agora_agentkit::govlog::Sha256Hex::from([2; 32]);
        vec![
            api.chain(id, key).await.map(drop),
            api.state_keys(id, key).await.map(drop),
            api.post_state_key(id, key, &record).await,
            api.rotate_key(id, key, &rotation(id, key, true)).await,
            api.rotations(id, key).await.map(drop),
            api.blob(id, key, &sha)
                .await
                .map(drop)
                .map_err(|e| e.message),
            api.proofs(id, key, None).await.map(drop),
            api.head(id, key).await.map(drop),
            api.head_known(id, key).await.map(drop),
            api.publish_chosen(id, key).await.map(drop),
            api.post_anchor(id, key, true).await,
        ]
    }
    use agora_agentkit::state::crypto::StateKey;

    #[tokio::test]
    async fn a_call_for_an_agent_that_doesnt_upload_is_refused_unsent() {
        let switch = Switch::default();
        let guard = UploadGuard::new(FakeApi::new(), &switch);
        let id = AgentId::new();
        let key = agora_agentkit::crypto::generate_keypair().0;
        let results = every_call(&guard, id, &key).await;
        assert_eq!(results.len(), 11);
        for r in &results {
            let e = r.as_ref().unwrap_err();
            assert!(is_local_only(e), "{e}");
        }
        assert!(
            guard.calls.lock().unwrap().is_empty(),
            "nothing reached Agora"
        );
        // Typed, for the callers that ask first
        assert_eq!(
            guard.check(id, Call::Blob),
            Err(LocalOnly {
                id,
                call: Call::Blob
            })
        );
        // Not about the agent's sealed state: these reach the fake (whatever
        // it answers)
        guard
            .rotate_key(id, &key, &rotation(id, &key, false))
            .await
            .ok();
        guard
            .register_encryption_key(
                id,
                &key,
                &agora_agentkit::envelope::EncryptionPublicKey::from(
                    &agora_agentkit::envelope::generate_encryption_keypair().0,
                ),
            )
            .await
            .ok();
        guard.signer_registered(id, &key).await.ok();
        guard
            .anchor_data(&"ANC-2026-0001".parse().unwrap())
            .await
            .ok();
        assert_eq!(
            *guard.calls.lock().unwrap(),
            [
                "rotate_key",
                "register_encryption_key",
                "signer_registered",
                "anchor_data"
            ]
        );

        // Uploaded once, stopped (or a marker in doubt): reads only
        guard.calls.lock().unwrap().clear();
        switch.1.store(true, Ordering::SeqCst);
        let reads = [
            "chain",
            "state_keys",
            "rotations",
            "blob",
            "proofs",
            "head",
            "head_known",
        ];
        assert_eq!(passed(&every_call(&guard, id, &key).await), reads);
        assert_eq!(
            *guard.calls.lock().unwrap(),
            reads,
            "no write reached Agora"
        );

        // Uploading: every call goes through
        guard.calls.lock().unwrap().clear();
        switch.0.store(true, Ordering::SeqCst);
        let results = every_call(&guard, id, &key).await;
        assert!(
            results
                .iter()
                .all(|r| r.as_ref().err().is_none_or(|e| !is_local_only(e))),
            "{results:?}"
        );
        assert_eq!(guard.calls.lock().unwrap().len(), 11);
    }

    /// The opt-in step's hold: the opt-in calls, for that agent, while
    /// held; nothing else.
    #[tokio::test]
    async fn opting_in_lets_only_the_opt_in_calls_through() {
        let switch = Switch::default();
        let guard = UploadGuard::new(FakeApi::new(), &switch);
        let (id, other) = (AgentId::new(), AgentId::new());
        let key = agora_agentkit::crypto::generate_keypair().0;
        {
            let _hold = guard.opting_in(id);
            let results = every_call(&guard, id, &key).await;
            assert_eq!(
                passed(&results),
                [
                    "chain",
                    "state_keys",
                    "post_state_key",
                    "head",
                    "head_known"
                ]
            );
            assert!(
                every_call(&guard, other, &key)
                    .await
                    .iter()
                    .all(|r| r.as_ref().is_err_and(|e| is_local_only(e))),
                "only the agent opting in"
            );
        }
        guard.calls.lock().unwrap().clear();
        assert!(
            every_call(&guard, id, &key)
                .await
                .iter()
                .all(|r| r.as_ref().is_err_and(|e| is_local_only(e))),
            "not once the hold is dropped"
        );
        assert!(guard.calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    #[should_panic(expected = "local only")]
    async fn a_strict_guard_fails_the_test() {
        let switch = Switch::default();
        let guard = UploadGuard::new(FakeApi::new(), &switch).strict();
        let key = agora_agentkit::crypto::generate_keypair().0;
        let _ = guard.head(AgentId::new(), &key).await;
    }
}
