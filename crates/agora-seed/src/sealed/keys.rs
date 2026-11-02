//! The runner's own key files, in the secrets tree beside `secrets.json`
//! (never in the state dir):
//!
//! - `<secrets>/<id>/state_key.json`: the agent's state key, its
//!   generation and which wrap of it the server holds ([`StateKeyFile`]);
//! - `<secrets>/<id>/past_keys.json`: every key the agent ever retired
//!   ([`PastKeys`]). Nothing here deletes one: a past X25519 secret opens
//!   old wraps and messages;
//! - `<secrets>/<id>/rotations.json`: the agent-signed rotations of its
//!   signing key, in order. Trust in old chain entries follows them
//!   ([`KeyStore::trusted`]): a key is trusted only until its rotation.
//!
//! All `0600`. Every type that holds a secret prints `[REDACTED]`.

use std::path::{Path, PathBuf};

use agora_agentkit::crypto::SigningKey;
use agora_agentkit::envelope::{EncryptionPublicKey, EncryptionSecretKey};
use agora_agentkit::govlog::PublicKeyHex;
use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::seed::{FsKeyring, Keyring};
use agora_agentkit::state::crypto::StateKey;
use agora_agentkit::state::storage::SealKeys;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use zeroize::Zeroizing;

/// The state key cache's file name.
pub const STATE_KEY_FILE: &str = "state_key.json";
/// Retired keys' file name.
pub const PAST_KEYS_FILE: &str = "past_keys.json";
/// agentkit's `FsKeyring` file.
pub const SECRETS_FILE: &str = "secrets.json";
/// Where `rotate-keys` stages the new secrets until the server takes them.
pub const NEXT_SECRETS_FILE: &str = "secrets.next.json";
/// The signing key's rotations.
pub const ROTATIONS_FILE: &str = "rotations.json";
/// A rotation sent, its answer not yet in hand.
pub const PENDING_ROTATION_FILE: &str = "rotation.pending.json";

/// `state_key.json`.
#[derive(Clone, Serialize, Deserialize)]
pub struct StateKeyFile {
    pub key_gen: u32,
    /// Which wrap of this generation was last posted.
    pub wrap_seq: u32,
    /// 64 hex characters.
    pub key: String,
}

impl std::fmt::Debug for StateKeyFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "StateKeyFile {{ key_gen: {}, wrap_seq: {}, key: [REDACTED] }}",
            self.key_gen, self.wrap_seq
        )
    }
}

impl Drop for StateKeyFile {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.key.zeroize();
    }
}

impl StateKeyFile {
    pub fn new(key: &StateKey, key_gen: u32, wrap_seq: u32) -> Self {
        Self {
            key_gen,
            wrap_seq,
            key: hex::encode(key.expose()),
        }
    }

    pub fn state_key(&self) -> std::io::Result<StateKey> {
        let bytes = Zeroizing::new(hex::decode(self.key.trim()).map_err(std::io::Error::other)?);
        let arr: [u8; 32] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| std::io::Error::other("a state key is 32 bytes"))?;
        Ok(StateKey::from_bytes(Zeroizing::new(arr)))
    }
}

/// One retired key.
#[derive(Clone, Serialize, Deserialize)]
pub struct PastKey {
    /// The public half, hex.
    pub public_key: String,
    /// The secret half, hex.
    pub secret: String,
    pub retired_at: DateTime<Utc>,
}

impl std::fmt::Debug for PastKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PastKey {{ public_key: {}, secret: [REDACTED], retired_at: {} }}",
            self.public_key, self.retired_at
        )
    }
}

impl Drop for PastKey {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.secret.zeroize();
    }
}

/// `past_keys.json`: append-only.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PastKeys {
    #[serde(default)]
    pub signing: Vec<PastKey>,
    #[serde(default)]
    pub encryption: Vec<PastKey>,
}

impl PastKeys {
    /// Past signing keys' public halves.
    #[cfg(test)]
    pub fn signing_public(&self) -> Vec<agora_agentkit::crypto::VerifyingKey> {
        self.signing
            .iter()
            .filter_map(|k| {
                let bytes: [u8; 32] = hex::decode(&k.public_key).ok()?.try_into().ok()?;
                agora_agentkit::crypto::VerifyingKey::from_bytes(&bytes).ok()
            })
            .collect()
    }

    /// Past X25519 secrets, newest first, to open an older wrap.
    #[cfg(test)]
    pub fn encryption_secrets(&self) -> Vec<EncryptionSecretKey> {
        self.encryption
            .iter()
            .rev()
            .filter_map(|k| agora_agentkit::envelope::encryption_secret_from_hex(&k.secret).ok())
            .collect()
    }
}

/// `secrets.json`'s wire shape, as agentkit's `FsKeyring` reads it
/// (pinned by `the_secrets_file_is_what_fskeyring_reads`).
#[derive(Serialize, Deserialize)]
pub struct SecretsFile {
    pub signing_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encryption_key: Option<String>,
}

impl std::fmt::Debug for SecretsFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SecretsFile([REDACTED])")
    }
}

impl Drop for SecretsFile {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.signing_key.zeroize();
        self.encryption_key.zeroize();
    }
}

impl SecretsFile {
    pub fn new(signing: &SigningKey, encryption: Option<&EncryptionSecretKey>) -> Self {
        Self {
            signing_key: agora_agentkit::crypto::signing_key_to_hex(signing),
            encryption_key: encryption.map(agora_agentkit::envelope::encryption_secret_to_hex),
        }
    }

    pub fn signing(&self) -> std::io::Result<SigningKey> {
        agora_agentkit::crypto::signing_key_from_hex(&self.signing_key)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }

    pub fn encryption(&self) -> std::io::Result<Option<EncryptionSecretKey>> {
        self.encryption_key
            .as_deref()
            .map(agora_agentkit::envelope::encryption_secret_from_hex)
            .transpose()
            .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

/// The secrets tree, for sealed state.
#[derive(Debug, Clone)]
pub struct KeyStore {
    dir: PathBuf,
    ring: FsKeyring,
}

/// Write `bytes` to `path` with mode 0600: a `.tmp` sibling, fsync, rename.
pub fn write_secret(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let dir = path.parent().expect("a file in a directory");
    std::fs::create_dir_all(dir)?;
    let tmp = path.with_extension("json.tmp");
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    {
        let mut file = options.open(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    std::fs::File::open(dir)?.sync_all()
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> std::io::Result<Option<T>> {
    match std::fs::read(path) {
        Ok(bytes) => {
            let bytes = Zeroizing::new(bytes);
            serde_json::from_slice(&bytes)
                .map(Some)
                .map_err(|e| std::io::Error::other(format!("{}: {e}", path.display())))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

fn to_json<T: Serialize>(value: &T) -> std::io::Result<Zeroizing<Vec<u8>>> {
    serde_json::to_vec_pretty(value)
        .map(Zeroizing::new)
        .map_err(std::io::Error::other)
}

impl KeyStore {
    /// Over `<data_dir>/secrets`.
    pub fn new(secrets_dir: impl Into<PathBuf>) -> Self {
        let dir = secrets_dir.into();
        Self {
            ring: FsKeyring::new(dir.clone()),
            dir,
        }
    }

    pub fn agent_dir(&self, id: AgentId) -> PathBuf {
        self.dir.join(id.to_string())
    }

    pub fn ring(&self) -> &FsKeyring {
        &self.ring
    }

    pub fn signing(&self, id: AgentId) -> Option<SigningKey> {
        self.ring.signing_key(id)
    }

    #[cfg(test)]
    pub fn encryption(&self, id: AgentId) -> Option<EncryptionSecretKey> {
        self.ring.encryption_key(id)
    }

    pub fn state_key_file(&self, id: AgentId) -> std::io::Result<Option<StateKeyFile>> {
        read_json(&self.agent_dir(id).join(STATE_KEY_FILE))
    }

    /// Write the state key cache. Before a key is posted: the server must
    /// never hold a wrap of a key the runner could lose.
    pub fn write_state_key(&self, id: AgentId, file: &StateKeyFile) -> std::io::Result<()> {
        write_secret(&self.agent_dir(id).join(STATE_KEY_FILE), &to_json(file)?)
    }

    pub fn past_keys(&self, id: AgentId) -> std::io::Result<PastKeys> {
        Ok(read_json(&self.agent_dir(id).join(PAST_KEYS_FILE))?.unwrap_or_default())
    }

    /// Append to `past_keys.json`. Never removes an entry.
    pub fn retire(
        &self,
        id: AgentId,
        signing: Option<&SigningKey>,
        encryption: Option<&EncryptionSecretKey>,
    ) -> std::io::Result<()> {
        let mut past = self.past_keys(id)?;
        let now = Utc::now();
        let seen = |list: &[PastKey], public: &str| list.iter().any(|k| k.public_key == public);
        if let Some(key) = signing {
            let public = hex::encode(key.verifying_key().to_bytes());
            if !seen(&past.signing, &public) {
                past.signing.push(PastKey {
                    public_key: public,
                    secret: agora_agentkit::crypto::signing_key_to_hex(key),
                    retired_at: now,
                });
            }
        }
        if let Some(key) = encryption {
            let public = hex::encode(EncryptionPublicKey::from(key).as_bytes());
            if !seen(&past.encryption, &public) {
                past.encryption.push(PastKey {
                    public_key: public,
                    secret: agora_agentkit::envelope::encryption_secret_to_hex(key),
                    retired_at: now,
                });
            }
        }
        write_secret(&self.agent_dir(id).join(PAST_KEYS_FILE), &to_json(&past)?)
    }

    pub fn secrets(&self, id: AgentId) -> std::io::Result<Option<SecretsFile>> {
        read_json(&self.agent_dir(id).join(SECRETS_FILE))
    }

    pub fn next_secrets(&self, id: AgentId) -> std::io::Result<Option<SecretsFile>> {
        read_json(&self.agent_dir(id).join(NEXT_SECRETS_FILE))
    }

    pub fn write_next_secrets(&self, id: AgentId, file: &SecretsFile) -> std::io::Result<()> {
        write_secret(&self.agent_dir(id).join(NEXT_SECRETS_FILE), &to_json(file)?)
    }

    /// `secrets.next.json` becomes `secrets.json`; the replaced file is
    /// kept as `secrets.<utc>.json`. Only after the server took the change.
    pub fn promote_next_secrets(&self, id: AgentId) -> std::io::Result<()> {
        let dir = self.agent_dir(id);
        let at = Utc::now().format("%Y%m%dT%H%M%S%3fZ");
        std::fs::hard_link(
            dir.join(SECRETS_FILE),
            dir.join(format!("secrets.{at}.json")),
        )?;
        std::fs::rename(dir.join(NEXT_SECRETS_FILE), dir.join(SECRETS_FILE))?;
        std::fs::File::open(&dir)?.sync_all()
    }

    /// The signing key's rotations, in order.
    pub fn rotations(
        &self,
        id: AgentId,
    ) -> std::io::Result<Vec<agora_agentkit::state::KeyRotation>> {
        Ok(read_json(&self.agent_dir(id).join(ROTATIONS_FILE))?.unwrap_or_default())
    }

    /// Append a rotation the server took. Never removes one.
    pub fn record_rotation(
        &self,
        id: AgentId,
        rotation: &agora_agentkit::state::KeyRotation,
    ) -> std::io::Result<()> {
        let mut all = self.rotations(id)?;
        if !all.contains(rotation) {
            all.push(rotation.clone());
        }
        write_secret(&self.agent_dir(id).join(ROTATIONS_FILE), &to_json(&all)?)
    }

    /// The rotation sent but not yet recorded, if any.
    pub fn pending_rotation(
        &self,
        id: AgentId,
    ) -> std::io::Result<Option<agora_agentkit::state::KeyRotation>> {
        read_json(&self.agent_dir(id).join(PENDING_ROTATION_FILE))
    }

    /// Keep the signed rotation before it is sent, so a retry sends (or
    /// records) the same one.
    pub fn write_pending_rotation(
        &self,
        id: AgentId,
        rotation: &agora_agentkit::state::KeyRotation,
    ) -> std::io::Result<()> {
        write_secret(
            &self.agent_dir(id).join(PENDING_ROTATION_FILE),
            &to_json(rotation)?,
        )
    }

    /// The pending rotation is recorded: kept as
    /// `rotation.<seq>.taken.json`, never deleted.
    pub fn settle_pending_rotation(&self, id: AgentId, seq: u32) -> std::io::Result<()> {
        let dir = self.agent_dir(id);
        std::fs::rename(
            dir.join(PENDING_ROTATION_FILE),
            dir.join(format!("rotation.{seq}.taken.json")),
        )
    }

    /// What an enrolled agent's entries are signed and sealed with.
    pub fn seal_keys(&self, id: AgentId) -> Option<SealKeys> {
        let signing = self.signing(id)?;
        let file = match self.state_key_file(id) {
            Ok(Some(file)) => file,
            Ok(None) => return None,
            Err(e) => {
                tracing::error!(agent_id = %id, error = %e, "state key cache unreadable");
                return None;
            }
        };
        let state_key = file
            .state_key()
            .map_err(|e| tracing::error!(agent_id = %id, error = %e, "state key cache malformed"))
            .ok()?;
        Some(SealKeys {
            signing,
            state_key,
            key_gen: file.key_gen,
        })
    }

    /// The signing keys this runner trusts for `id`: its first key and
    /// the rotations it recorded itself, each key until its rotation.
    /// Never the server's word. `Err` too when the rotations don't end at
    /// the key in `secrets.json`.
    pub fn trusted(
        &self,
        id: AgentId,
    ) -> std::io::Result<Option<agora_agentkit::state::verify::TrustedKeys>> {
        let Some(current) = self.signing(id) else {
            return Ok(None);
        };
        let rotations = self.rotations(id)?;
        let first = match rotations.iter().min_by_key(|r| r.seq) {
            Some(r) => r
                .old_public_key
                .to_verifying_key()
                .map_err(|e| std::io::Error::other(format!("rotations.json: {e}")))?,
            None => current.verifying_key(),
        };
        let trusted =
            agora_agentkit::state::verify::TrustedKeys::new(first).with_rotations(&rotations);
        if trusted.current() != current.verifying_key() {
            return Err(std::io::Error::other(
                "the recorded rotations don't end at the current signing key",
            ));
        }
        Ok(Some(trusted))
    }
}

/// The X25519 public key, as a record names it.
pub fn x25519_public_hex(secret: &EncryptionSecretKey) -> PublicKeyHex {
    PublicKeyHex::from(*EncryptionPublicKey::from(secret).as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_secrets_file_is_what_fskeyring_reads() {
        let dir = crate::sealed::TestDir::new("keys-secrets");
        let store = KeyStore::new(dir.path());
        let id = AgentId::new();
        let (sk, _) = agora_agentkit::crypto::generate_keypair();
        let (enc, _) = agora_agentkit::envelope::generate_encryption_keypair();
        store
            .write_next_secrets(id, &SecretsFile::new(&sk, Some(&enc)))
            .unwrap();
        std::fs::rename(
            store.agent_dir(id).join(NEXT_SECRETS_FILE),
            store.agent_dir(id).join(SECRETS_FILE),
        )
        .unwrap();
        assert_eq!(store.signing(id).unwrap().to_bytes(), sk.to_bytes());
        assert_eq!(store.encryption(id).unwrap().to_bytes(), enc.to_bytes());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(store.agent_dir(id).join(SECRETS_FILE))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    #[test]
    fn retiring_appends_and_never_drops() {
        let dir = crate::sealed::TestDir::new("keys-past");
        let store = KeyStore::new(dir.path());
        let id = AgentId::new();
        let (a, _) = agora_agentkit::crypto::generate_keypair();
        let (b, _) = agora_agentkit::crypto::generate_keypair();
        let (e, _) = agora_agentkit::envelope::generate_encryption_keypair();
        store.retire(id, Some(&a), Some(&e)).unwrap();
        store.retire(id, Some(&a), None).unwrap();
        store.retire(id, Some(&b), None).unwrap();
        let past = store.past_keys(id).unwrap();
        assert_eq!(past.signing.len(), 2, "a key is retired once");
        assert_eq!(past.encryption.len(), 1);
        assert_eq!(
            past.signing_public(),
            vec![a.verifying_key(), b.verifying_key()]
        );
        assert_eq!(past.encryption_secrets()[0].to_bytes(), e.to_bytes());
        assert!(!format!("{past:?}").contains(&past.signing[0].secret.clone()));
    }

    #[test]
    fn the_state_key_cache_round_trips_and_is_redacted() {
        let dir = crate::sealed::TestDir::new("keys-state");
        let store = KeyStore::new(dir.path());
        let id = AgentId::new();
        assert!(store.state_key_file(id).unwrap().is_none());
        let key = StateKey::generate();
        let file = StateKeyFile::new(&key, 0, 0);
        assert!(!format!("{file:?}").contains(&file.key.clone()));
        store.write_state_key(id, &file).unwrap();
        let back = store.state_key_file(id).unwrap().unwrap();
        assert_eq!(back.state_key().unwrap().expose(), key.expose());
        assert!(store.seal_keys(id).is_none(), "no signing key yet");
        store
            .ring()
            .insert(id, &agora_agentkit::crypto::generate_keypair().0)
            .unwrap();
        assert_eq!(store.seal_keys(id).unwrap().key_gen, 0);
    }
}
