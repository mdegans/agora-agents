use std::path::PathBuf;

use agora_agent_lib::agora_agentkit::ids::AgentId;
use agora_agent_lib::signing::SigningKey;
use anyhow::{Context, Result};
use uuid::Uuid;

use agora_agent_lib::memory::Memory;
use agora_agent_lib::soul::Soul;

use crate::state::State;

/// An agent loaded from disk, ready to run.
pub struct Agent {
    pub name: String,
    pub soul: Soul,
    pub memory: Memory,
    pub signing_key: SigningKey,
    pub agent_id: Option<AgentId>,
    pub model: String,
    pub dir: PathBuf,
    pub communities: Vec<String>,
    /// Persisted state (seen posts, created posts/comments, last cycle timestamp).
    pub state: State,
}

impl Agent {
    /// Load an agent from its directory (containing SOUL.md, etc.)
    ///
    /// If `model` is Some, it is used for all agents. Otherwise the model field
    /// is left empty and must be resolved from the server before running.
    pub async fn load(dir: PathBuf, model: Option<&str>) -> Result<Self> {
        let soul_path = dir.join("SOUL.md");
        let soul = Soul::from_file(&soul_path)
            .await
            .with_context(|| format!("loading SOUL.md from {}", dir.display()))?;

        let name = soul.name.clone();
        let communities = soul.communities();

        // Load or create memory
        let memory_path = dir.join("MEMORY.md");
        let memory = if memory_path.exists() {
            Memory::from_file(&memory_path).await.unwrap_or_else(|e| {
                tracing::warn!("Failed to load MEMORY.md for {name}: {e}, using empty");
                Memory::empty()
            })
        } else {
            let template = Memory::initial_template(&name);
            let mut mem = Memory::empty();
            mem.update(template);
            mem
        };

        // Load signing key: prefer XDG data dir (rotated keys), fall back to
        // the soul directory (legacy/unrotated), generate new if neither exists.
        let xdg_key_path = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("~/.local/share"))
            .join("agora/keys")
            .join(&name)
            .join("signing_key.hex");
        let soul_key_path = dir.join("signing_key.hex");

        let signing_key = if xdg_key_path.exists() {
            let hex_str = tokio::fs::read_to_string(&xdg_key_path)
                .await
                .context("reading signing key from XDG")?;
            agora_agent_lib::signing::signing_key_from_hex(hex_str.trim())
                .context("parsing signing key")?
        } else if soul_key_path.exists() {
            let hex_str = tokio::fs::read_to_string(&soul_key_path)
                .await
                .context("reading signing key")?;
            agora_agent_lib::signing::signing_key_from_hex(hex_str.trim())
                .context("parsing signing key")?
        } else {
            let (signing_key, _) = agora_agent_lib::signing::generate_keypair();
            let hex_str = agora_agent_lib::signing::signing_key_to_hex(&signing_key);
            tokio::fs::write(&soul_key_path, &hex_str)
                .await
                .context("saving signing key")?;
            tracing::debug!("Generated new keypair for {name}");
            signing_key
        };

        // Load agent_id if previously registered
        let agent_id_path = dir.join("agent_id.txt");
        let agent_id: Option<AgentId> = if agent_id_path.exists() {
            let id_str = tokio::fs::read_to_string(&agent_id_path).await.ok();
            id_str.and_then(|s| s.trim().parse::<Uuid>().ok().map(AgentId::from))
        } else {
            None
        };

        // Model assignment: from CLI flag or resolved later from server
        let model = model.unwrap_or_default().to_string();

        let state = State::load(&dir).await;

        Ok(Self {
            name,
            soul,
            memory,
            signing_key,
            agent_id,
            model,
            dir,
            communities,
            state,
        })
    }

    /// Save agent_id to disk after registration.
    pub async fn save_agent_id(&self) -> Result<()> {
        if let Some(id) = self.agent_id {
            let path = self.dir.join("agent_id.txt");
            tokio::fs::write(&path, id.to_string()).await?;
        }
        Ok(())
    }

    /// Get the public key as a hex string.
    pub fn public_key_hex(&self) -> String {
        hex::encode(self.signing_key.verifying_key().as_bytes())
    }

    /// Save memory to disk.
    pub async fn save_memory(&self) -> Result<()> {
        let path = self.dir.join("MEMORY.md");
        self.memory.save(&path).await?;
        Ok(())
    }

    /// Save soul to disk.
    pub async fn save_soul(&self) -> Result<()> {
        let path = self.dir.join("SOUL.md");
        self.soul.save(&path).await?;
        Ok(())
    }

    /// Save persisted state to disk.
    pub async fn save_state(&self) -> Result<()> {
        self.state.save(&self.dir).await
    }
}

/// Load all agents from the souls directory.
pub async fn load_all(
    souls_dir: &std::path::Path,
    model: Option<&str>,
) -> Result<Vec<Agent>> {
    let mut agents = Vec::new();
    let mut entries = tokio::fs::read_dir(souls_dir)
        .await
        .with_context(|| format!("reading souls directory {}", souls_dir.display()))?;

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let soul_path = path.join("SOUL.md");
        if !soul_path.exists() {
            continue;
        }

        match Agent::load(path.clone(), model).await {
            Ok(agent) => agents.push(agent),
            Err(e) => {
                tracing::warn!("Failed to load agent from {}: {e:#}", path.display());
            }
        }
    }

    agents.sort_by(|a, b| a.name.cmp(&b.name));
    tracing::info!(
        "Loaded {} agents from {}",
        agents.len(),
        souls_dir.display()
    );
    Ok(agents)
}

/// Resolve model assignments from the server for agents that don't have one.
///
/// Fetches each agent's profile and reads the `model_info` field. Agents whose
/// model can't be resolved are collected into the returned error list.
pub async fn resolve_models(
    agents: &mut [Agent],
    client: &crate::client::AgoraClient,
) -> Vec<String> {
    let unresolved: Vec<usize> = agents
        .iter()
        .enumerate()
        .filter(|(_, a)| a.model.is_empty())
        .map(|(i, _)| i)
        .collect();

    if unresolved.is_empty() {
        return Vec::new();
    }

    tracing::info!(
        "Resolving models from server for {} agents...",
        unresolved.len()
    );

    let mut failed = Vec::new();
    let mut resolved = 0;

    for &idx in &unresolved {
        let name = agents[idx].name.clone();
        match client.get_agent(&name).await {
            Ok(Some(data)) => {
                if let Some(model) = data["model_info"]
                    .as_str()
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                {
                    agents[idx].model = model.to_string();
                    resolved += 1;
                } else {
                    failed.push(name);
                }
            }
            _ => {
                failed.push(name);
            }
        }
    }

    tracing::info!("Resolved {resolved} agent models from server");
    if !failed.is_empty() {
        tracing::warn!(
            "{} agents have no model assignment: {}",
            failed.len(),
            if failed.len() <= 10 {
                failed.join(", ")
            } else {
                format!(
                    "{}, ... and {} more",
                    failed[..10].join(", "),
                    failed.len() - 10
                )
            }
        );
    }

    failed
}
