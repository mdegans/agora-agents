use agora_agentkit::ids::AgentId;
use agora_agentkit::names::AgentName;
use agora_agentkit::reactor::seed::Ledger;
use anyhow::{Context, Result};
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};

use crate::config::{config_dir, write_atomically, write_secret_atomically};

/// Stored credentials for an agent.
///
/// The agent's id and keys, and nothing else. Files written before
/// 2026-09-21 may also carry `bearer_token`, `operator_email` and
/// `operator_password` (the operator's password, in plaintext) from the
/// removed `login` command: loading rewrites such a file without them, and
/// any file others can read as this user's alone ([`load_credentials`]).
#[derive(Serialize, Deserialize)]
pub struct Credentials {
    pub agent_id: AgentId,
    pub signing_key_hex: String,
    /// X25519 encryption secret (hex), generated lazily on first
    /// messaging use and registered with the server.
    #[serde(default)]
    pub encryption_secret_hex: Option<String>,
}

// Redacted (CLAUDE.md rule 6): both keys are secrets.
impl std::fmt::Debug for Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Credentials")
            .field("agent_id", &self.agent_id)
            .field("signing_key_hex", &"[REDACTED]")
            .field(
                "encryption_secret_hex",
                &self.encryption_secret_hex.as_ref().map(|_| "[REDACTED]"),
            )
            .finish()
    }
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
pub fn load_credentials(agent: &AgentName) -> Result<Credentials> {
    use std::os::unix::fs::PermissionsExt;

    let path = config_dir()?
        .join("credentials")
        .join(format!("{agent}.json"));
    let contents = std::fs::read_to_string(&path).with_context(|| {
        format!("no credentials for agent '{agent}' — run `agora register` first")
    })?;
    let creds: Credentials = serde_json::from_str(&contents)?;
    let mode = std::fs::metadata(&path)?.permissions().mode();
    if needs_rewrite(&contents, mode)? {
        save_credentials(agent, &creds)?;
    }
    Ok(creds)
}

/// Whether a credentials file holds fields the CLI no longer keeps, or can
/// be read by anyone but this user (Steward, 2026-10-26)
fn needs_rewrite(contents: &str, mode: u32) -> Result<bool> {
    const KEPT: &[&str] = &["agent_id", "signing_key_hex", "encryption_secret_hex"];
    let fields: serde_json::Map<String, serde_json::Value> = serde_json::from_str(contents)?;
    Ok(fields.keys().any(|k| !KEPT.contains(&k.as_str())) || mode & 0o077 != 0)
}

/// Save credentials for a named agent.
pub fn save_credentials(agent: &AgentName, creds: &Credentials) -> Result<()> {
    let dir = config_dir()?.join("credentials");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{agent}.json"));
    write_secret_atomically(&path, serde_json::to_string_pretty(creds)?.as_bytes())
}

/// Where the CLI keeps an agent's dedup ledger: the runner's shape, in a
/// file of its own (the runner keeps its in `state.json`)
fn ledger_path(agent: &AgentName) -> Result<std::path::PathBuf> {
    Ok(config_dir()?
        .join("state")
        .join(agent.as_str())
        .join("ledger.json"))
}

/// The agent's ledger, empty if there is none yet
pub fn load_ledger(agent: &AgentName) -> Result<Ledger> {
    let path = ledger_path(agent)?;
    if !path.exists() {
        return Ok(Ledger::default());
    }
    let contents = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&contents).with_context(|| format!("failed to parse {}", path.display()))
}

/// Save the agent's ledger
pub fn save_ledger(agent: &AgentName, ledger: &Ledger) -> Result<()> {
    let path = ledger_path(agent)?;
    std::fs::create_dir_all(path.parent().expect("a ledger path has a parent"))?;
    write_atomically(&path, serde_json::to_string(ledger)?.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_fields_or_open_permissions_need_a_rewrite() {
        let kept = r#"{"agent_id":"7ad26ccd-1111-4222-8333-444455556666","signing_key_hex":"00","encryption_secret_hex":"11"}"#;
        assert!(!needs_rewrite(kept, 0o100600).unwrap());
        assert!(needs_rewrite(kept, 0o100664).unwrap());
        let stale = r#"{"agent_id":"7ad26ccd-1111-4222-8333-444455556666","signing_key_hex":"00","operator_password":"x"}"#;
        assert!(needs_rewrite(stale, 0o100600).unwrap());
    }
}
