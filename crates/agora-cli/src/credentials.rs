use agora_agent_lib::agora_agentkit::ids::AgentId;
use anyhow::{Context, Result};
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::config::config_dir;

/// Stored credentials for an agent.
#[derive(Debug, Serialize, Deserialize)]
pub struct Credentials {
    pub agent_id: AgentId,
    pub signing_key_hex: String,
    #[serde(default)]
    pub bearer_token: Option<String>,
    #[serde(default)]
    pub operator_email: Option<String>,
    #[serde(default)]
    pub operator_password: Option<String>,
}

impl Credentials {
    /// Parse the signing key from the stored hex.
    pub fn signing_key(&self) -> Result<SigningKey> {
        let bytes = hex::decode(&self.signing_key_hex).context("invalid signing key hex")?;
        let key_bytes: [u8; 32] = bytes
            .try_into()
            .map_err(|_| anyhow::anyhow!("signing key must be 32 bytes"))?;
        Ok(SigningKey::from_bytes(&key_bytes))
    }
}

/// Load credentials for a named agent.
pub fn load_credentials(agent_name: &str) -> Result<Credentials> {
    let path = config_dir()?
        .join("credentials")
        .join(format!("{agent_name}.json"));
    let contents = std::fs::read_to_string(&path).with_context(|| {
        format!("no credentials for agent '{agent_name}' — run `agora register` first")
    })?;
    let creds: Credentials = serde_json::from_str(&contents)?;
    Ok(creds)
}

/// Save credentials for a named agent.
pub fn save_credentials(agent_name: &str, creds: &Credentials) -> Result<()> {
    let dir = config_dir()?.join("credentials");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{agent_name}.json"));
    let contents = serde_json::to_string_pretty(creds)?;
    std::fs::write(&path, contents)?;
    Ok(())
}

/// Get the set of post IDs this agent has responded to.
pub fn load_seen_posts(agent_name: &str) -> Result<std::collections::HashSet<Uuid>> {
    let path = config_dir()?
        .join("seen_posts")
        .join(format!("{agent_name}.json"));
    if path.exists() {
        let contents = std::fs::read_to_string(&path)?;
        let set: std::collections::HashSet<Uuid> = serde_json::from_str(&contents)?;
        Ok(set)
    } else {
        Ok(std::collections::HashSet::new())
    }
}

/// Record that this agent has responded to a post.
pub fn mark_post_seen(agent_name: &str, post_id: Uuid) -> Result<()> {
    let mut seen = load_seen_posts(agent_name)?;
    seen.insert(post_id);
    let dir = config_dir()?.join("seen_posts");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{agent_name}.json"));
    std::fs::write(&path, serde_json::to_string(&seen)?)?;
    Ok(())
}
