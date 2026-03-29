use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use agora_agent_lib::agora_agentkit::ids::{AgentId, CommentId, PostId};
use agora_agent_lib::signing::SigningKey;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use uuid::Uuid;

use agora_agent_lib::memory::Memory;
use agora_agent_lib::soul::Soul;

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
    /// Posts this agent has already commented on (to avoid duplicate comments across cycles).
    pub commented_posts: HashSet<PostId>,
    /// Posts this agent has already created.
    pub created_posts: HashSet<PostId>,
    /// Comments this agent has created (for tracking replies).
    pub created_comments: HashSet<CommentId>,
    /// Timestamp of last cycle completion (for filtering new replies).
    pub last_cycle_at: Option<DateTime<Utc>>,
    /// Tracks seen posts: post_id → last known comment count.
    /// Posts only appear in the feed if they're new or have new comments.
    pub seen_posts: HashMap<PostId, i64>,
}

impl Agent {
    /// Load an agent from its directory (containing SOUL.md, model.txt, etc.)
    pub async fn load(dir: PathBuf, model_override: Option<&str>) -> Result<Self> {
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

        // Load model assignment — fail fast if model.txt is missing
        let model = if let Some(override_model) = model_override {
            override_model.to_string()
        } else {
            let model_path = dir.join("model.txt");
            tokio::fs::read_to_string(&model_path)
                .await
                .with_context(|| format!("missing model.txt in {}", dir.display()))?
                .trim()
                .to_string()
        };

        Ok(Self {
            name,
            soul,
            memory,
            signing_key,
            agent_id,
            model,
            dir,
            communities,
            commented_posts: HashSet::new(),
            created_posts: HashSet::new(),
            created_comments: HashSet::new(),
            last_cycle_at: None,
            seen_posts: HashMap::new(),
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
}

/// Load all agents from the souls directory.
pub async fn load_all(
    souls_dir: &std::path::Path,
    model_override: Option<&str>,
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

        match Agent::load(path.clone(), model_override).await {
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
