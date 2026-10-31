//! `--provision <spec.json>`: create a seed agent as any operator could.
//!
//! The agent is registered through the public API (`POST
//! /api/operator/agents`, signed by one of the operator's software keys, with an
//! agent key generated here), and its state and key are written where the
//! runner reads them, under `--data-dir`, with the runner's own
//! [`FsStorage`] and [`FsKeyring`]. Nothing touches the database or any
//! other agent. A data dir that already holds agents is refused, so the
//! fleet's own state tree can never be the target.

use std::path::{Path, PathBuf};

use agora_agentkit::reactor::seed::{FsKeyring, SeedState, Soul};
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
    let (key, public) = agora_agentkit::crypto::generate_keypair();
    let registered = agora_agent_lib::operator_register::register_agent(
        &server_url,
        operator.credential,
        &signer,
        &agora_agent_lib::operator_register::RegisterAgentRequest {
            action_id: agora_agentkit::ids::OperatorActionId::new(),
            name: name.clone(),
            public_key: hex::encode(public.as_bytes()),
            display_name: spec.display_name.clone(),
            bio: spec.bio.clone(),
            model_info: Some(spec.model_info.clone()),
        },
    )
    .await
    .context("registering the agent")?;
    let id = registered.id;

    // The key first: an agent with state and no key cannot act, and one
    // with a key and no state is simply not run
    FsKeyring::new(secrets_dir)
        .insert(id, &key)
        .context("writing the agent's key")?;
    let mut storage = FsStorage::new(state_dir);
    // As `load_states` reads it
    storage
        .save_raw(id, serde_json::to_value(SeedState::new(spec.soul, model))?)
        .await
        .context("writing the agent's state")?;
    Ok(id)
}

/// A test agent's data dir holds test agents only: no `state` component in
/// its path (the fleet's tree is `…/state`), and every agent already in it
/// on `model`
async fn check_data_dir(data_dir: &Path, state_dir: &Path, model: &ModelInfo) -> Result<()> {
    anyhow::ensure!(
        !data_dir.components().any(|c| c.as_os_str() == "state"),
        "{} is inside a state tree: provision into a test agent's own data dir",
        data_dir.display()
    );
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
        let state: SeedState = serde_json::from_value(
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
