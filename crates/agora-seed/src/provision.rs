//! `--provision <spec.json>`: create a seed agent as any operator could.
//!
//! The agent is registered through the public API (`POST
//! /api/operator/agents`, signed by one of the operator's software keys, with an
//! agent key generated here), and its state and key are written where the
//! runner reads them, under `--data-dir`, with the runner's own
//! [`FsStorage`] and [`FsKeyring`]. Nothing touches the database or any
//! other agent.
//!
//! **What it writes,** under `--data-dir` and nowhere else: before the request,
//! `provision-<name>.json` (0600: the new agent's secret key and the action id), so a
//! lost answer can be retried with the same key; after it, the key in `secrets/`, the
//! state in `state/`, and that file removed. A rerun never overwrites either: a stored
//! key must match, and existing state is left alone, since the runner may already have
//! run the agent. It signs with a software key, which is refused above 5 agents; create
//! agents on the account page then.
//!
//! **What it refuses:** a data dir under the fleet's (`~/agents/agora`), or with a
//! `state` component in its path, symlinks resolved, and one holding an agent on another
//! model. Agents on the same model may share a data dir. It reads only their `model`.

use std::path::{Path, PathBuf};

use agora_agentkit::reactor::seed::{FsKeyring, SeedState, Soul};
use agora_agentkit::reactor::storage::FsStorageError;
use agora_agentkit::reactor::{FsStorage, Storage};
use anyhow::{Context, Result};
use misanthropic::model::ModelInfo;
use serde::Deserialize;

/// What a new agent is made from
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub soul: Soul,
    pub display_name: Option<String>,
    pub bio: Option<String>,
    /// What the agent's profile says runs it
    pub model_info: String,
}

/// The operator's credential that signs the registration
pub struct Operator {
    pub credential: agora_agentkit::ids::OperatorCredentialId,
    pub key_file: PathBuf,
}

/// Register the agent `spec` describes and write its state and key under
/// `data_dir`, on `model`. Returns the new agent's id.
pub async fn provision(
    spec_path: &Path,
    operator: &Operator,
    server_url: url::Url,
    data_dir: &Path,
    model: ModelInfo,
) -> Result<agora_agentkit::ids::AgentId> {
    let spec: Spec = serde_json::from_str(
        &tokio::fs::read_to_string(spec_path)
            .await
            .with_context(|| format!("reading {}", spec_path.display()))?,
    )
    .with_context(|| format!("parsing {}", spec_path.display()))?;
    let signer = agora_agent_lib::operator_register::software_signer(&operator.key_file)?;

    let state_dir = data_dir.join("state");
    check_data_dir(data_dir, &state_dir, &model).await?;
    // Everything is writable before the agent exists: a registered agent
    // whose key could not be written could never act
    let secrets_dir = data_dir.join("secrets");
    for dir in [&state_dir, &secrets_dir] {
        tokio::fs::create_dir_all(dir)
            .await
            .with_context(|| format!("creating {}", dir.display()))?;
        let probe = dir.join(".provision-probe");
        tokio::fs::write(&probe, b"")
            .await
            .with_context(|| format!("{} is not writable", dir.display()))?;
        tokio::fs::remove_file(&probe).await?;
    }

    let name = spec.soul.agent_name().context("the soul's name")?;
    let pending_path = data_dir.join(format!("provision-{name}.json"));
    use agora_agent_lib::operator_register::Pending;
    let pending = Pending::load_or_create(&pending_path, Pending::new_key_hex)?;
    let key = pending.signing_key(&pending_path)?;
    let public = key.verifying_key();
    let registered = agora_agent_lib::operator_register::register_agent(
        &server_url,
        operator.credential,
        &signer,
        &agora_agentkit::requests::RegisterAgentRequest {
            action_id: pending.action_id,
            name: name.clone(),
            public_key: hex::encode(public.as_bytes()),
            display_name: spec.display_name.clone(),
            bio: spec.bio.clone(),
            model_info: Some(spec.model_info.clone()),
        },
    )
    .await
    .with_context(|| Pending::kept_note(&pending_path))?;
    let id = registered.id;

    install(&secrets_dir, state_dir, id, &key, || {
        Ok(serde_json::to_value(SeedState::new(spec.soul, model))?)
    })
    .await?;
    Pending::finish(&pending_path)?;
    Ok(id)
}

/// Put the agent's key and state in place, on a first run or a retry. The key first: an
/// agent with state and no key cannot act, and one with a key and no state is simply not
/// run. A key that's there must be this key. State that's there stays: the runner may
/// have run the agent already, and writing it would reset its memory.
async fn install(
    secrets_dir: &Path,
    state_dir: PathBuf,
    id: agora_agentkit::ids::AgentId,
    key: &ed25519_dalek::SigningKey,
    state: impl FnOnce() -> Result<serde_json::Value>,
) -> Result<()> {
    use agora_agentkit::reactor::seed::Keyring;
    let keyring = FsKeyring::new(secrets_dir);
    match keyring.insert(id, key) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            let stored = keyring
                .signing_key(id)
                .with_context(|| format!("agent {id}'s stored key can't be read"))?;
            anyhow::ensure!(
                stored.to_bytes() == key.to_bytes(),
                "agent {id} already has a different key in {}: leaving it alone",
                secrets_dir.display()
            );
        }
        Err(e) => return Err(e).context("writing the agent's key"),
    }
    let mut storage = FsStorage::new(state_dir);
    match storage.load_raw(id).await {
        Ok(_) => Ok(()),
        Err(FsStorageError::NotFound(_)) => {
            // As `load_states` reads it
            storage
                .save_raw(id, state()?)
                .await
                .context("writing the agent's state")
        }
        Err(e) => Err(e).with_context(|| format!("agent {id}'s state can't be read")),
    }
}

/// All this looks at in another agent's state.
#[derive(Deserialize)]
struct JustTheModel {
    model: ModelInfo,
}

fn resolved_fleet(home: &Path) -> Result<PathBuf> {
    resolved(&home.join("agents/agora"))
}

/// `path` with symlinks resolved, as far as it exists yet.
fn resolved(path: &Path) -> Result<PathBuf> {
    let absolute =
        std::path::absolute(path).with_context(|| format!("resolving {}", path.display()))?;
    let mut existing = absolute.as_path();
    let mut rest = Vec::new();
    loop {
        match existing.canonicalize() {
            Ok(mut found) => {
                found.extend(rest.iter().rev());
                return Ok(found);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let name = existing
                    .file_name()
                    .with_context(|| format!("resolving {}", path.display()))?;
                rest.push(name);
                existing = existing
                    .parent()
                    .with_context(|| format!("resolving {}", path.display()))?;
            }
            Err(e) => return Err(e).with_context(|| format!("resolving {}", path.display())),
        }
    }
}

/// A test agent's data dir holds test agents only: no `state` component in
/// its path (the fleet's tree is `…/state`), and every agent already in it
/// on `model`
async fn check_data_dir(data_dir: &Path, state_dir: &Path, model: &ModelInfo) -> Result<()> {
    let resolved = resolved(data_dir)?;
    anyhow::ensure!(
        !resolved.components().any(|c| c.as_os_str() == "state"),
        "{} is inside a state tree: provision into a test agent's own data dir",
        data_dir.display()
    );
    if let Some(home) = dirs::home_dir() {
        let fleet = resolved_fleet(&home)?;
        anyhow::ensure!(
            !resolved.starts_with(&fleet),
            "{} is under the fleet's {}: provision into a test agent's own data dir",
            data_dir.display(),
            fleet.display()
        );
    }
    let mut entries = match tokio::fs::read_dir(state_dir).await {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e).with_context(|| format!("reading {}", state_dir.display())),
    };
    let storage = FsStorage::new(state_dir);
    while let Some(entry) = entries.next_entry().await? {
        let Some(id) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<agora_agentkit::ids::AgentId>().ok())
        else {
            continue;
        };
        let state: JustTheModel = serde_json::from_value(
            storage
                .load_raw(id)
                .await
                .with_context(|| format!("reading agent {id}"))?,
        )
        .with_context(|| format!("reading agent {id}"))?;
        anyhow::ensure!(
            state.model.id == model.id,
            "{} holds agent {id} on {}: provision only beside test agents",
            state_dir.display(),
            state.model.id.name()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use agora_agentkit::ids::AgentId;
    use agora_agentkit::reactor::seed::Keyring;

    fn scratch() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("agora-provision-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&dir).unwrap();
        dir
    }

    fn key() -> ed25519_dalek::SigningKey {
        agora_agentkit::crypto::generate_keypair().0
    }

    async fn stored_state(dir: &Path, id: AgentId) -> serde_json::Value {
        FsStorage::new(dir.join("state"))
            .load_raw(id)
            .await
            .unwrap()
    }

    /// The whole retry: a lost answer's replay, after the first run got its files in, or
    /// only its key, and after the runner has run the agent.
    #[tokio::test]
    async fn a_rerun_keeps_what_is_there() {
        let dir = scratch();
        let (secrets, state) = (dir.join("secrets"), dir.join("state"));
        let (id, k) = (AgentId::new(), key());
        let first = serde_json::json!({"memory": "first"});
        install(&secrets, state.clone(), id, &k, || Ok(first.clone()))
            .await
            .unwrap();
        assert_eq!(stored_state(&dir, id).await, first);
        // Same key, state already there and since changed by a run: untouched
        let run = serde_json::json!({"memory": "after a run"});
        FsStorage::new(&state)
            .save_raw(id, run.clone())
            .await
            .unwrap();
        install(&secrets, state.clone(), id, &k, || {
            panic!("state was overwritten")
        })
        .await
        .unwrap();
        assert_eq!(stored_state(&dir, id).await, run);
        // A key but no state: the first run stopped there
        let (id2, k2) = (AgentId::new(), key());
        FsKeyring::new(&secrets).insert(id2, &k2).unwrap();
        install(&secrets, state.clone(), id2, &k2, || Ok(first.clone()))
            .await
            .unwrap();
        assert_eq!(stored_state(&dir, id2).await, first);
        // A different key is refused, and neither key nor state changes
        let err = install(&secrets, state.clone(), id, &key(), || {
            panic!("state written")
        })
        .await
        .unwrap_err();
        assert!(format!("{err:#}").contains("different key"), "{err:#}");
        let kept = FsKeyring::new(&secrets).signing_key(id).unwrap();
        assert_eq!(kept.to_bytes(), k.to_bytes());
        assert_eq!(stored_state(&dir, id).await, run);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_path_resolves_through_a_symlink_as_far_as_it_exists() {
        let dir = scratch();
        let real = dir.join("real");
        std::fs::create_dir(&real).unwrap();
        let link = dir.join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let found = resolved(&link.join("not/yet")).unwrap();
        assert_eq!(found, real.canonicalize().unwrap().join("not/yet"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
