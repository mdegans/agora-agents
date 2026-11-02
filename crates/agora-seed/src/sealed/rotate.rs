//! `agora-seed rotate-keys --agent <id> [--signing] [--encryption]`.
//!
//! In this order, so no step can lose a key:
//!
//! 1. the new secrets are written to `secrets.next.json` (or, on a retry,
//!    the ones already there are reused), and the old keys are appended to
//!    `past_keys.json`;
//! 2. the server takes the change: an agent-signed rotation of the Ed25519
//!    key (signed by the old key, with the new key's proof; its `seq` the
//!    next after `rotations.json`'s), recorded in `rotations.json` once
//!    taken, then the new X25519 key, registered under the current signing
//!    key;
//! 3. an enrolled agent's state key is re-wrapped to the new X25519 key
//!    with the next `wrap_seq`, and posted;
//! 4. only then `secrets.next.json` becomes `secrets.json`; the replaced
//!    file is kept as `secrets.<utc>.json`.
//!
//! A failure at 2 or 3 leaves `secrets.json` as it was, and a rerun picks
//! up `secrets.next.json`. Old keys are never deleted.

use agora_agentkit::ids::AgentId;
use agora_agentkit::state::KeyRotation;
use anyhow::Context;
use chrono::Utc;

use super::KeyStore;
use super::api::SealedApi;
use super::keys::{SecretsFile, StateKeyFile};

/// What changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rotated {
    pub signing: bool,
    pub encryption: bool,
    /// The state key was re-wrapped to the new X25519 key.
    pub rewrapped: bool,
    /// `secrets.next.json` from an earlier attempt was used.
    pub resumed: bool,
}

pub async fn rotate_keys(
    state_dir: &std::path::Path,
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
    signing: bool,
    encryption: bool,
) -> anyhow::Result<Rotated> {
    anyhow::ensure!(
        signing || encryption,
        "name what to rotate: --signing, --encryption, or both"
    );
    // No session in progress: one would sign with a key retired under it.
    // An enrolled agent's lock is held for the whole command.
    super::session::no_session(state_dir, id, false)?;
    let agent_dir = state_dir.join(id.to_string());
    let _lock = if super::is_enrolled(&agent_dir) {
        Some(super::session::take(state_dir, id, false).await?)
    } else {
        None
    };
    let current = keys
        .secrets(id)?
        .with_context(|| format!("agent {id} has no secrets.json"))?;
    let old_signing = current.signing()?;
    let old_encryption = current.encryption()?;

    // 1. The new keys, on disk before anything else
    let (next, resumed) = match keys.next_secrets(id)? {
        Some(next) => (next, true),
        None => {
            let new_signing = if signing {
                agora_agentkit::crypto::generate_keypair().0
            } else {
                old_signing.clone()
            };
            let new_encryption = if encryption {
                Some(agora_agentkit::envelope::generate_encryption_keypair().0)
            } else {
                old_encryption.clone()
            };
            let next = SecretsFile::new(&new_signing, new_encryption.as_ref());
            keys.write_next_secrets(id, &next)
                .context("writing secrets.next.json")?;
            (next, false)
        }
    };
    let new_signing = next.signing()?;
    let new_encryption = next.encryption()?;
    let mut done = Rotated {
        signing: new_signing.to_bytes() != old_signing.to_bytes(),
        encryption: new_encryption.as_ref().map(|k| k.to_bytes())
            != old_encryption.as_ref().map(|k| k.to_bytes()),
        rewrapped: false,
        resumed,
    };
    keys.retire(
        id,
        done.signing.then_some(&old_signing),
        if done.encryption {
            old_encryption.as_ref()
        } else {
            None
        },
    )
    .context("recording the old keys in past_keys.json")?;

    // 2. The server
    if done.signing {
        let seq = keys
            .rotations(id)?
            .iter()
            .map(|r| r.seq + 1)
            .max()
            .unwrap_or(0);
        let new_public = agora_agentkit::govlog::PublicKeyHex::from(&new_signing.verifying_key());
        // The rotation an earlier attempt signed and sent, if it was this one
        let rotation = match keys.pending_rotation(id)? {
            Some(r) if r.new_public_key == new_public && r.seq == seq => r,
            _ => {
                // The old key signs nothing past the current head
                let last = if super::is_enrolled(&agent_dir) {
                    agora_agentkit::state::storage::AgentSeal::new(&agent_dir, id)
                        .head()
                        .await?
                        .map(|h| h.head)
                } else {
                    None
                };
                let r = KeyRotation::agent_signed(
                    id,
                    seq,
                    &old_signing,
                    &new_signing,
                    last,
                    Utc::now(),
                );
                keys.write_pending_rotation(id, &r)
                    .context("keeping the signed rotation before sending it")?;
                r
            }
        };
        // An earlier attempt's answer may have been lost: if the server
        // already takes the new key, the rotation was taken
        let taken = resumed
            && api
                .signer_registered(id, &new_signing)
                .await
                .map_err(anyhow::Error::msg)
                .context("asking the server which signing key it holds")?;
        if taken {
            tracing::warn!(
                event_type = "sealed_rotation_already_taken",
                agent_id = %id,
                seq,
                "the server already holds the new signing key: an earlier attempt's answer was lost"
            );
        } else {
            api.rotate_key(id, &old_signing, &rotation)
                .await
                .map_err(anyhow::Error::msg)
                .context(
                    "the server refused the signing-key rotation; secrets.json is unchanged",
                )?;
        }
        // Ours to verify the chain by, never the server's
        keys.record_rotation(id, &rotation)
            .context("recording the rotation in rotations.json")?;
        keys.settle_pending_rotation(id, seq)?;
    }
    if done.encryption
        && let Some(secret) = &new_encryption
    {
        let public = agora_agentkit::envelope::EncryptionPublicKey::from(secret);
        api.register_encryption_key(id, &new_signing, &public)
            .await
            .map_err(anyhow::Error::msg)
            .context("the server refused the new encryption key; secrets.json is unchanged")?;
    }

    // 3. The state key, re-wrapped
    if done.encryption
        && let Some(secret) = &new_encryption
        && let Some(file) = keys.state_key_file(id)?
    {
        let state_key = file.state_key()?;
        let wrap_seq = file.wrap_seq + 1;
        let record = state_key
            .wrap(
                id,
                file.key_gen,
                wrap_seq,
                &agora_agentkit::envelope::EncryptionPublicKey::from(secret),
                &new_signing,
            )
            .context("re-wrapping the state key")?;
        api.post_state_key(id, &new_signing, &record)
            .await
            .map_err(anyhow::Error::msg)
            .context("the server refused the re-wrapped state key; secrets.json is unchanged")?;
        keys.write_state_key(id, &StateKeyFile::new(&state_key, file.key_gen, wrap_seq))?;
        done.rewrapped = true;
    }

    // 4. Only now
    keys.promote_next_secrets(id)
        .context("replacing secrets.json")?;
    tracing::warn!(
        event_type = "sealed_keys_rotated",
        agent_id = %id,
        signing = done.signing,
        encryption = done.encryption,
        rewrapped = done.rewrapped,
        resumed = done.resumed,
        "agent keys rotated; the old ones are in past_keys.json"
    );
    Ok(done)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sealed::api::fake::FakeApi;
    use crate::sealed::check::{self, Outcome, tests::enrolled};
    use agora_agentkit::reactor::{FsStorage, Storage};
    use agora_agentkit::state::crypto::StateKey;
    use agora_agentkit::state::storage::plaintext_hash;

    #[tokio::test]
    async fn new_keys_are_written_first_and_secrets_replaced_only_after_the_server() {
        let root = crate::sealed::TestDir::new("rotate");
        let (id, keys, dir) = enrolled(&root).await;
        keys.ring().ensure_encryption_key(id).unwrap();
        let old_signing = keys.signing(id).unwrap();
        let old_enc = keys.encryption(id).unwrap();
        let api = FakeApi::new();
        *api.public_key.lock().unwrap() = Some(old_signing.verifying_key());

        let state_dir = root.join("state");
        assert!(
            rotate_keys(&state_dir, &keys, &api, id, false, false)
                .await
                .is_err()
        );

        // The server refuses: nothing replaced, the new keys kept
        *api.fail.lock().unwrap() = Some("rotate_key");
        assert!(
            rotate_keys(&state_dir, &keys, &api, id, true, true)
                .await
                .is_err()
        );
        assert_eq!(keys.signing(id).unwrap().to_bytes(), old_signing.to_bytes());
        let staged = keys.next_secrets(id).unwrap().unwrap().signing().unwrap();
        let past = keys.past_keys(id).unwrap();
        assert_eq!(past.signing_public(), vec![old_signing.verifying_key()]);
        assert_eq!(past.encryption.len(), 1);

        // A retry resumes with the same new keys
        *api.fail.lock().unwrap() = None;
        api.calls.lock().unwrap().clear();
        let done = rotate_keys(&state_dir, &keys, &api, id, true, true)
            .await
            .unwrap();
        assert_eq!(
            done,
            Rotated {
                signing: true,
                encryption: true,
                rewrapped: true,
                resumed: true
            }
        );
        assert_eq!(
            *api.calls.lock().unwrap(),
            [
                "signer_registered",
                "rotate_key",
                "register_encryption_key",
                "post_state_key"
            ],
            "rotation, then the X25519 key under the new signer, then the re-wrap"
        );
        assert_eq!(keys.signing(id).unwrap().to_bytes(), staged.to_bytes());
        assert!(keys.next_secrets(id).unwrap().is_none());
        let new_enc = keys.encryption(id).unwrap();
        assert_ne!(new_enc.to_bytes(), old_enc.to_bytes());
        // The re-wrap opens with the new X25519 key to the same state key
        let record = api.keys.lock().unwrap().last().unwrap().clone();
        assert_eq!(record.wrap_seq, 1);
        let file = keys.state_key_file(id).unwrap().unwrap();
        assert_eq!(file.wrap_seq, 1);
        assert_eq!(
            StateKey::unwrap(&record, &new_enc).unwrap().expose(),
            file.state_key().unwrap().expose()
        );
        // Old keys are still on disk, never deleted; the rotation is ours
        assert_eq!(keys.past_keys(id).unwrap().signing.len(), 1);
        let rotations = keys.rotations(id).unwrap();
        assert_eq!(rotations.len(), 1);
        assert_eq!(rotations[0].seq, 0);
        assert!(rotations[0].verify_agent_signed().is_ok());
        let backups = std::fs::read_dir(keys.agent_dir(id))
            .unwrap()
            .filter(|e| {
                let n = e.as_ref().unwrap().file_name();
                let n = n.to_string_lossy();
                n.starts_with("secrets.2") && n.ends_with(".json")
            })
            .count();
        assert_eq!(backups, 1);

        // The chain signed by the old key still verifies at the next start
        let local = plaintext_hash(
            &FsStorage::new(root.join("state"))
                .load_raw(id)
                .await
                .unwrap()
                .value,
        );
        let out = check::check(&dir, id, &local, Default::default(), &keys, &FakeApi::new()).await;
        assert!(matches!(out, Outcome::Run { hold: None, .. }), "{out:?}");
    }

    /// The server took the rotation but its answer was lost: the retry
    /// finds the new key registered, records the same signed rotation, and
    /// never rotates twice.
    #[tokio::test]
    async fn a_rotation_whose_answer_was_lost_is_recorded_on_retry() {
        let root = crate::sealed::TestDir::new("rotate-lost");
        let (id, keys, _) = enrolled(&root).await;
        let state_dir = root.join("state");
        let api = FakeApi::new();
        *api.public_key.lock().unwrap() = Some(keys.signing(id).unwrap().verifying_key());
        *api.fail.lock().unwrap() = Some("rotate_key_reply");
        assert!(
            rotate_keys(&state_dir, &keys, &api, id, true, false)
                .await
                .is_err()
        );
        let sent = keys.pending_rotation(id).unwrap().unwrap();
        assert!(
            keys.rotations(id).unwrap().is_empty(),
            "not recorded without an answer"
        );

        *api.fail.lock().unwrap() = None;
        api.calls.lock().unwrap().clear();
        let done = rotate_keys(&state_dir, &keys, &api, id, true, false)
            .await
            .unwrap();
        assert!(done.signing && done.resumed);
        assert_eq!(
            *api.calls.lock().unwrap(),
            ["signer_registered"],
            "no second rotation"
        );
        assert_eq!(
            keys.rotations(id).unwrap(),
            vec![sent.clone()],
            "the one sent"
        );
        assert!(keys.pending_rotation(id).unwrap().is_none());
        assert_eq!(
            agora_agentkit::govlog::PublicKeyHex::from(&keys.signing(id).unwrap().verifying_key()),
            sent.new_public_key
        );
        assert!(keys.trusted(id).unwrap().is_some());
    }

    /// Never while the agent has a session journal.
    #[tokio::test]
    async fn rotation_refuses_while_a_session_is_in_progress() {
        let root = crate::sealed::TestDir::new("rotate-session");
        let (id, keys, _) = enrolled(&root).await;
        let state_dir = root.join("state");
        let journal = FsStorage::new(&state_dir).journal(id);
        std::fs::write(journal.path(), b"{}\n").unwrap();
        let e = rotate_keys(&state_dir, &keys, &FakeApi::new(), id, true, false)
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("session in progress"), "{e}");
        assert!(keys.next_secrets(id).unwrap().is_none(), "nothing written");
    }
}
