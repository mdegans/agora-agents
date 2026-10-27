use agora_agent_lib::files::{config_dir, write_atomically};
use agora_agentkit::names::AgentName;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// CLI configuration stored in XDG config dir.
#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub server_url: String,
    #[serde(default)]
    pub default_agent: Option<AgentName>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server_url: "http://localhost:8080".to_string(),
            default_agent: None,
        }
    }
}

/// Load config from disk, or return defaults.
pub fn load_config() -> Result<Config> {
    let path = config_dir()?.join("config.toml");
    if path.exists() {
        let contents = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let config: Config = toml::from_str(&contents)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        Ok(config)
    } else {
        Ok(Config::default())
    }
}

/// Save config to disk.
pub fn save_config(config: &Config) -> Result<()> {
    let dir = config_dir()?;
    std::fs::create_dir_all(&dir)?;
    write_atomically(
        &dir.join("config.toml"),
        toml::to_string_pretty(config)?.as_bytes(),
    )
}

/// The active agent: the config's default, else the `active_agent` file.
pub fn active_agent(config: &Config) -> Result<Option<AgentName>> {
    if let Some(name) = &config.default_agent {
        return Ok(Some(name.clone()));
    }
    let path = config_dir()?.join("active_agent");
    if !path.exists() {
        return Ok(None);
    }
    let name = std::fs::read_to_string(&path)?;
    match name.trim() {
        "" => Ok(None),
        name => name
            .parse()
            .map(Some)
            .with_context(|| format!("{}: {name:?}", path.display())),
    }
}

/// Set the active agent.
pub fn set_active_agent(name: &AgentName) -> Result<()> {
    let dir = config_dir()?;
    std::fs::create_dir_all(&dir)?;
    write_atomically(&dir.join("active_agent"), name.as_str().as_bytes())
}
