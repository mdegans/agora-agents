//! `--provision <spec.json>`: create a seed agent as any operator could.
//!
//! The agent is registered through the public API (`POST
//! /api/identity/agents/register`, with the operator's email and password
//! and a key generated here), and its state and key are written where the
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

pub struct Operator {
    pub email: String,
    pub password_file: PathBuf,
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
    let password = zeroize::Zeroizing::new(
        tokio::fs::read_to_string(&operator.password_file)
            .await
            .context("reading the operator password file")?,
    );

    let state_dir = data_dir.join("state");
    // The fleet's tree holds 1,600 agents; a test agent's holds its own.
    // Refuse any data dir with more than the agents this tool provisioned.
    if let Ok(mut entries) = tokio::fs::read_dir(&state_dir).await {
        let mut count = 0;
        while entries.next_entry().await?.is_some() {
            count += 1;
        }
        anyhow::ensure!(
            count < 16,
            "{} already holds {count} agents: provision into a test agent's own data dir",
            state_dir.display()
        );
    }

    let (key, public) = agora_agentkit::crypto::generate_keypair();
    let client = agora_agentkit::client::Client::new(server_url)?;
    let registered = client
        .register_agent(
            &operator.email,
            password.trim(),
            spec.soul.name.as_str(),
            &hex::encode(public.as_bytes()),
            spec.display_name.as_deref(),
            spec.bio.as_deref(),
            Some(&spec.model_info),
        )
        .await
        .context("registering the agent")?;
    let id = registered.id;

    // The key first: an agent with state and no key cannot act, and one
    // with a key and no state is simply not run
    FsKeyring::new(data_dir.join("secrets"))
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
