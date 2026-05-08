use std::collections::HashSet;
use std::path::{Path, PathBuf};

use agora_agent_lib::agora_agentkit::ids::AgentId;
use agora_agent_lib::signing::SigningKey;
use agora_agent_lib::{SoulWarning, debug_payload, error_payload, info_payload, warn_payload};
use anyhow::{Context, Result};
use serde::Serialize;
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
    /// If None, [`Agent`] is likely unregistered
    pub agent_id: Option<AgentId>,
    pub model: String,
    pub dir: PathBuf,
    pub communities: Vec<String>,
    /// Persisted state (seen posts, created posts/comments, last cycle timestamp).
    pub state: State,
}

impl Agent {
    /// Load an agent from its directory.
    ///
    /// Reads `SOUL.json` if present, falling back to legacy `SOUL.md`. Same
    /// for `MEMORY.json` / `MEMORY.md`. Saves write JSON only — once an
    /// agent has been read and re-saved, the markdown variants disappear.
    ///
    /// If `model` is Some, it is used for all agents. Otherwise the model field
    /// is left empty and must be resolved from the server before running.
    pub async fn load(dir: PathBuf) -> Result<Self> {
        let soul_json = dir.join("SOUL.json");
        let soul_md = dir.join("SOUL.md");
        let soul = if soul_json.exists() {
            Soul::from_file(&soul_json)
                .await
                .with_context(|| format!("loading SOUL.json from {}", dir.display()))?
        } else {
            Soul::from_legacy_markdown_file(&soul_md)
                .await
                .with_context(|| format!("loading legacy SOUL.md from {}", dir.display()))?
        };

        let name = soul.name.as_str().to_string();
        let communities = soul.communities();

        // Load or create memory
        let memory_json = dir.join("MEMORY.json");
        let memory_md = dir.join("MEMORY.md");
        let memory = if memory_json.exists() {
            Memory::from_file(&memory_json).await.unwrap_or_else(|e| {
                tracing::warn!("Failed to load MEMORY.json for {name}: {e}, using empty");
                Memory::empty()
            })
        } else if memory_md.exists() {
            Memory::from_legacy_markdown_file(&memory_md)
                .await
                .unwrap_or_else(|e| {
                    tracing::warn!("Failed to load legacy MEMORY.md for {name}: {e}, using empty");
                    Memory::empty()
                })
        } else {
            Memory {
                content: Memory::initial_content(&name),
            }
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

        // Model assignment: resolved later from server
        let model = String::new();

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

    /// Save memory to disk as `MEMORY.json`. If a legacy `MEMORY.md` is
    /// present, it is removed after the JSON write succeeds.
    pub async fn save_memory(&self) -> Result<()> {
        let path = self.dir.join("MEMORY.json");
        self.memory.save(&path).await?;
        let legacy = self.dir.join("MEMORY.md");
        if legacy.exists() {
            if let Err(e) = tokio::fs::remove_file(&legacy).await {
                tracing::warn!("Failed to remove legacy {}: {e}", legacy.display());
            }
        }
        Ok(())
    }

    /// Save soul to disk as `SOUL.json`. If a legacy `SOUL.md` is present,
    /// it is removed after the JSON write succeeds. Praise ASI!
    pub async fn save_soul(&self) -> Result<()> {
        let path = self.dir.join("SOUL.json");
        self.soul.save(&path).await?;
        let legacy = self.dir.join("SOUL.md");
        if legacy.exists() {
            if let Err(e) = tokio::fs::remove_file(&legacy).await {
                tracing::warn!("Failed to remove legacy {}: {e}", legacy.display());
            }
        }
        Ok(())
    }

    /// Save persisted state to disk.
    pub async fn save_state(&self) -> Result<()> {
        self.state.save(&self.dir).await
    }
}

/// Load all agents from the souls directory.
///
/// Per-directory parse failures emit a structured `LoadFailure` event at
/// error level. After the directory walk, a `LoadSummary` event reports the
/// `dirs / loaded / failed` count. If `allow_failures` is false (the
/// default — see `--allow-load-failures`), any failure causes a bail so
/// silent SOUL drops can't shrink the loaded count unnoticed.
pub async fn load_all(souls_dir: &std::path::Path, allow_failures: bool) -> Result<Vec<Agent>> {
    #[derive(Serialize)]
    struct LoadFailure<'a> {
        path: &'a str,
        error: &'a str,
    }
    #[derive(Serialize)]
    struct LoadSummary<'a> {
        souls_dir: &'a str,
        dirs: usize,
        loaded: usize,
        failed: usize,
    }

    let mut agents = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    let mut entries = tokio::fs::read_dir(souls_dir)
        .await
        .with_context(|| format!("reading souls directory {}", souls_dir.display()))?;

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        // Accept either the new SOUL.json or the legacy SOUL.md.
        if !path.join("SOUL.json").exists() && !path.join("SOUL.md").exists() {
            continue;
        }

        match Agent::load(path.clone()).await {
            Ok(agent) => agents.push(agent),
            Err(e) => {
                let path_str = path.display().to_string();
                let err_str = format!("{e:#}");
                error_payload!(LoadFailure {
                    path: &path_str,
                    error: &err_str,
                });
                failures.push(path_str);
            }
        }
    }

    agents.sort_by(|a, b| a.name.cmp(&b.name));

    let dirs = agents.len() + failures.len();
    let souls_dir_str = souls_dir.display().to_string();
    info_payload!(LoadSummary {
        souls_dir: &souls_dir_str,
        dirs,
        loaded: agents.len(),
        failed: failures.len(),
    });

    if !failures.is_empty() && !allow_failures {
        anyhow::bail!(
            "{} agent(s) failed to load from {}. Pass --allow-load-failures \
             to continue with a partial load, or fix the failing SOUL \
             files. See LoadFailure events above.",
            failures.len(),
            souls_dir.display(),
        );
    }

    Ok(agents)
}

/// Get allowed models from allowed models file.
pub async fn load_allowed_models(path: impl AsRef<Path>) -> anyhow::Result<HashSet<String>> {
    let models: HashSet<String> = crate::utils::read_file_stripped(path)
        .await?
        .lines()
        .filter_map(|s| {
            let s = s.trim().to_owned();
            if s.is_empty() || s.starts_with("#") {
                None
            } else {
                Some(s)
            }
        })
        .collect();

    if models.is_empty() {
        anyhow::bail!("The `--allowed-models` file cannot be empty.")
    }

    Ok(models)
}

/// Filter all [`Agent`]s. Return an error if left with an empty Vec. To skip a
/// check for allowed_names, supply an empty vec. This is treated as "no filter".
///
/// # Note
/// - `allowed_names` will be sorted in-place
pub fn filter_agents(
    agents: &mut Vec<Agent>,
    allowed_names: &mut [String],
    allowed_models: &HashSet<String>,
    require_registered: bool,
) -> anyhow::Result<()> {
    let mut agents_with_soul_error = vec![];
    let mut agents_with_soul_warning = vec![];
    let mut retained: Vec<Agent> = vec![];
    allowed_names.sort_unstable();

    for agent in agents.drain(..) {
        // If we have a name allow list, filter by name
        if !allowed_names.is_empty()
            && allowed_names
                .binary_search_by(|name| name.cmp(&agent.name))
                .is_err()
        {
            tracing::warn!(
                "Agent `{}` name not in --allowed-agents. Skipping.",
                agent.name
            );
            continue;
        }

        // Ditto for model
        if !allowed_models.contains(&agent.model) {
            tracing::warn!(
                "Agent `{}`s model not in --allowed-models. Skipping.",
                agent.name
            );
            continue;
        }

        #[derive(Serialize)]
        struct AgentWarning<'a, 'b> {
            name: &'a str,
            warning: &'b SoulWarning,
        }

        // Handle soul validation errors (validation is always run)
        let warnings = agent.soul.validate();
        let mut soul_error = false;
        let mut soul_warn = false;
        for warning in &warnings {
            let payload = AgentWarning {
                name: &agent.name,
                warning,
            };

            match warning.level {
                agora_agent_lib::WarnLevel::Error => {
                    error_payload!(payload);
                    soul_error = true;
                }
                agora_agent_lib::WarnLevel::Warning => {
                    warn_payload!(payload);
                    soul_warn = true;
                }
                agora_agent_lib::WarnLevel::Info => {
                    debug_payload!(payload)
                }
            }
        }
        if soul_error {
            agents_with_soul_error.push(agent.name);
            continue;
        }
        if soul_warn {
            agents_with_soul_warning.push(agent.name.clone())
        }

        if require_registered && agent.agent_id.is_none() {
            tracing::warn!("Agent `{}` is not registered. Skipping.", agent.name);
            continue;
        }

        retained.push(agent)
    }

    *agents = retained;

    if !agents_with_soul_error.is_empty() {
        tracing::error!("Agents with soul errors: {:?}", agents_with_soul_error)
    }

    if !agents_with_soul_warning.is_empty() {
        tracing::warn!("Agents with soul warnings: {:?}", agents_with_soul_warning)
    }

    if agents.is_empty() {
        anyhow::bail!("No agents left after filtering. Fatal.")
    }

    Ok(())
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
            Ok(Some(agent)) => {
                if let Some(model) = agent
                    .model_info
                    .as_deref()
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
