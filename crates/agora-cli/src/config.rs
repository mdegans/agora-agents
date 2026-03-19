use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// CLI configuration stored in XDG config dir.
#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub server_url: String,
    #[serde(default)]
    pub default_agent: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server_url: "http://localhost:8080".to_string(),
            default_agent: None,
        }
    }
}

/// Returns the agora config directory (~/.config/agora/).
pub fn config_dir() -> Result<PathBuf> {
    let base = dirs::config_dir().context("could not determine XDG config directory")?;
    Ok(base.join("agora"))
}

/// Load config from disk, or return defaults.
pub fn load_config() -> Result<Config> {
    let path = config_dir()?.join("config.toml");
    if path.exists() {
        let contents = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let config: Config =
            toml::from_str(&contents).with_context(|| format!("failed to parse {}", path.display()))?;
        Ok(config)
    } else {
        Ok(Config::default())
    }
}

/// Save config to disk.
pub fn save_config(config: &Config) -> Result<()> {
    let dir = config_dir()?;
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("config.toml");
    let contents = toml::to_string_pretty(config)?;
    std::fs::write(&path, contents)?;
    Ok(())
}

/// Get the active agent name (from CLI flag, config, or active_agent file).
pub fn active_agent(config: &Config) -> Result<Option<String>> {
    // First check config default
    if let Some(ref name) = config.default_agent {
        return Ok(Some(name.clone()));
    }
    // Then check active_agent file
    let path = config_dir()?.join("active_agent");
    if path.exists() {
        let name = std::fs::read_to_string(&path)?.trim().to_string();
        if !name.is_empty() {
            return Ok(Some(name));
        }
    }
    Ok(None)
}

/// Set the active agent.
pub fn set_active_agent(name: &str) -> Result<()> {
    let dir = config_dir()?;
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("active_agent"), name)?;
    Ok(())
}
