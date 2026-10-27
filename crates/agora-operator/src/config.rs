//! The operator CLI's config: which server, which credential, which key.
//!
//! `~/.config/agora/operator/config.toml` by default, beside the keys
//! `keygen` writes (`~/.config/agora/operator/<label>.key`). The config
//! names the key's file; it never holds the key.

use std::path::{Path, PathBuf};

use agora_agent_lib::files::{config_dir, write_atomically};
use agora_agentkit::ids::OperatorCredentialId;
use agora_agentkit::operator_signer::{OperatorSigner, SoftwareKeySigner};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use url::Url;

/// The bulk threshold an account starts with (the server's default)
pub const DEFAULT_BULK_THRESHOLD: u32 = 10;

/// How the CLI signs: one variant per kind of signer.
///
/// A hardware key joins as a variant of its own (its device serial and
/// PIV slot, say), opened by [`SignerConfig::open`] into the same
/// `Box<dyn OperatorSigner>`; nothing else in the CLI changes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum SignerConfig {
    /// An Ed25519 key in a file, as hex (`keygen` writes one)
    SoftwareKey(PathBuf),
}

impl SignerConfig {
    /// The signer, ready to sign
    pub fn open(&self) -> Result<Box<dyn OperatorSigner>> {
        match self {
            Self::SoftwareKey(path) => Ok(Box::new(
                SoftwareKeySigner::from_file(path)
                    .with_context(|| format!("opening the key {}", path.display()))?,
            )),
        }
    }
}

/// The config file. Each field is absent until `config` (or `keygen`)
/// sets it, and a command that needs one says which is missing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorConfig {
    /// The Agora server. No default, so nothing reaches a server, least of
    /// all production, unless it was named
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server: Option<Url>,
    /// The credential the portal gave the key when it was added
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential: Option<OperatorCredentialId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signer: Option<SignerConfig>,
    /// The account's bulk threshold as set in the portal, for `send`'s
    /// preview only: the server applies its own
    #[serde(default = "default_bulk_threshold")]
    pub bulk_threshold: u32,
}

fn default_bulk_threshold() -> u32 {
    DEFAULT_BULK_THRESHOLD
}

impl Default for OperatorConfig {
    fn default() -> Self {
        Self {
            server: None,
            credential: None,
            signer: None,
            bulk_threshold: DEFAULT_BULK_THRESHOLD,
        }
    }
}

impl OperatorConfig {
    /// The server: `--server` if given, else the config's
    pub fn server(&self, flag: Option<&Url>) -> Result<Url> {
        match flag.or(self.server.as_ref()) {
            Some(url) => Ok(url.clone()),
            None => bail!(
                "no server: pass --server URL, or save one with \
                 `agora-operator config --server URL`"
            ),
        }
    }

    pub fn credential(&self) -> Result<OperatorCredentialId> {
        self.credential.with_context(|| {
            "no credential: add your key's public key in the portal (Account, Signing keys), \
             then `agora-operator config --credential <id>`"
        })
    }

    pub fn signer(&self) -> Result<Box<dyn OperatorSigner>> {
        match &self.signer {
            Some(signer) => signer.open(),
            None => bail!(
                "no signing key: `agora-operator keygen`, or \
                 `agora-operator config --key-file PATH`"
            ),
        }
    }
}

/// `~/.config/agora/operator`: the config and the keys
pub fn operator_dir() -> Result<PathBuf> {
    Ok(config_dir()?.join("operator"))
}

/// The config file: `--config`, else `~/.config/agora/operator/config.toml`
pub fn config_path(flag: Option<&Path>) -> Result<PathBuf> {
    match flag {
        Some(path) => Ok(path.to_owned()),
        None => Ok(operator_dir()?.join("config.toml")),
    }
}

/// The config at `path`, or the empty one if there is no file
pub fn load(path: &Path) -> Result<OperatorConfig> {
    if !path.exists() {
        return Ok(OperatorConfig::default());
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
}

pub fn save(path: &Path, config: &OperatorConfig) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    write_atomically(path, toml::to_string_pretty(config)?.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_config_round_trips_and_names_only_the_key_file() {
        let config = OperatorConfig {
            server: Some("http://localhost:8099".parse().unwrap()),
            credential: Some("0b7e2a61-5d3c-4c8e-8f1a-2e9d4b6c7a10".parse().unwrap()),
            signer: Some(SignerConfig::SoftwareKey("/k/laptop.key".into())),
            bulk_threshold: 25,
        };
        let text = toml::to_string_pretty(&config).unwrap();
        assert!(text.contains("software_key = \"/k/laptop.key\""), "{text}");
        assert_eq!(toml::from_str::<OperatorConfig>(&text).unwrap(), config);
    }

    #[test]
    fn an_empty_config_has_no_server() {
        let config: OperatorConfig = toml::from_str("").unwrap();
        assert_eq!(config, OperatorConfig::default());
        assert!(config.server(None).is_err());
        let flag: Url = "http://localhost:8099".parse().unwrap();
        assert_eq!(config.server(Some(&flag)).unwrap(), flag);
    }

    #[test]
    fn unknown_fields_are_refused() {
        assert!(toml::from_str::<OperatorConfig>("server_url = \"x\"").is_err());
    }
}
