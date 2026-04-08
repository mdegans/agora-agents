use std::collections::{HashMap, HashSet};
use std::path::Path;

use agora_agent_lib::agora_agentkit::ids::{CommentId, PostId};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

const STATE_FILE: &str = "STATE.toml";

/// Persisted agent state that survives across seed runs.
///
/// Saved as `STATE.toml` in each agent's directory after every cycle.
/// Fields use `#[serde(default)]` so old state files without new fields
/// load cleanly.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct State {
    /// When the agent last completed a cycle.
    #[serde(default)]
    pub last_cycle_at: Option<DateTime<Utc>>,

    /// Posts this agent has commented on.
    #[serde(default)]
    pub commented_posts: HashSet<PostId>,

    /// Posts this agent has created.
    #[serde(default)]
    pub created_posts: HashSet<PostId>,

    /// Comments this agent has created (for tracking replies).
    #[serde(default)]
    pub created_comments: HashSet<CommentId>,

    /// Posts the agent has seen, with last-known comment count.
    /// Posts only appear as "fresh" in the feed if they're new or have new
    /// comments since the last recorded count.
    #[serde(default)]
    pub seen_posts: HashMap<PostId, i64>,
}

impl State {
    /// Load state from an agent directory, returning Default if the file
    /// doesn't exist or can't be parsed.
    pub async fn load(agent_dir: &Path) -> Self {
        let path = agent_dir.join(STATE_FILE);
        match tokio::fs::read_to_string(&path).await {
            Ok(contents) => toml::from_str(&contents).unwrap_or_else(|e| {
                tracing::warn!("Failed to parse {}: {e}, using defaults", path.display());
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    /// Save state to the agent directory. Writes to a temp file first,
    /// then renames for atomicity.
    pub async fn save(&self, agent_dir: &Path) -> Result<()> {
        let path = agent_dir.join(STATE_FILE);
        let tmp_path = agent_dir.join(".STATE.toml.tmp");
        let contents = toml::to_string_pretty(self).context("serializing state")?;
        tokio::fs::write(&tmp_path, &contents)
            .await
            .with_context(|| format!("writing {}", tmp_path.display()))?;
        tokio::fs::rename(&tmp_path, &path)
            .await
            .with_context(|| format!("renaming to {}", path.display()))?;
        Ok(())
    }
}
