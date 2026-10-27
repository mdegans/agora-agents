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
            Self::SoftwareKey(path) => Ok(Box::new({
                check_private(path)?;
                SoftwareKeySigner::from_file(path)
                    .with_context(|| format!("opening the key {}", path.display()))?
            })),
        }
    }
}

/// Refuse a key file others can read or write, as ssh does
fn check_private(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mode = std::fs::metadata(path)
        .with_context(|| format!("opening the key {}", path.display()))?
        .permissions()
        .mode();
    if mode & 0o077 != 0 {
        bail!(
            "the key {} is open to others (mode {:04o}), so it is not used: \
             `chmod 600 {}`",
            path.display(),
            mode & 0o7777,
            path.display()
        );
    }
    Ok(())
}

/// A server the CLI will sign for: `https`, or `http` to this machine only,
/// so a signed request never crosses a network in the clear
pub fn check_server(url: &Url) -> Result<()> {
    let loopback = match url.host() {
        Some(url::Host::Domain(d)) => d.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        None => false,
    };
    match url.scheme() {
        "https" => Ok(()),
        "http" if loopback => Ok(()),
        "http" => bail!("{url}: http is for this machine only (localhost); use https"),
        other => bail!("{url}: `{other}` is not a server URL; use https"),
    }
}

/// The config file. Each field is absent until `config` (or `keygen`)
/// sets it, and a command that needs one says which is missing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
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
}

impl OperatorConfig {
    /// The server: `--server` if given, else the config's
    pub fn server(&self, flag: Option<&Url>) -> Result<Url> {
        match flag.or(self.server.as_ref()) {
            Some(url) => {
                check_server(url)?;
                Ok(url.clone())
            }
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

/// `dir`, made this user's alone if it has to be made (an existing one
/// keeps its mode)
pub fn create_private_dir(dir: &Path) -> Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)
        .with_context(|| format!("creating {}", dir.display()))
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
    toml::from_str(&text).map_err(|e| parse_error(path, &text, &e))
}

/// A parse error by line and column only: toml's own text quotes the line,
/// which this file's reader need not see echoed
fn parse_error(path: &Path, text: &str, e: &toml::de::Error) -> anyhow::Error {
    match e.span() {
        Some(span) => {
            let before = &text[..span.start.min(text.len())];
            let line = before.matches('\n').count() + 1;
            let column = before.rsplit('\n').next().unwrap_or("").chars().count() + 1;
            anyhow::anyhow!(
                "failed to parse {} at line {line}, column {column}",
                path.display()
            )
        }
        None => anyhow::anyhow!("failed to parse {}", path.display()),
    }
}

pub fn save(path: &Path, config: &OperatorConfig) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        create_private_dir(dir)?;
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

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "agora-operator-config-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn http_is_for_this_machine_only() {
        for ok in [
            "https://example.org",
            "http://localhost:8099",
            "http://127.0.0.1:8099",
            "http://[::1]:8099",
        ] {
            assert!(check_server(&ok.parse().unwrap()).is_ok(), "{ok}");
        }
        for bad in [
            "http://example.org",
            "http://10.0.0.2:8099",
            "ftp://localhost",
        ] {
            assert!(check_server(&bad.parse().unwrap()).is_err(), "{bad}");
        }
        let config = OperatorConfig {
            server: Some("http://example.org".parse().unwrap()),
            ..OperatorConfig::default()
        };
        assert!(config.server(None).is_err());
    }

    #[test]
    fn a_key_open_to_others_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("mode");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("k.key");
        std::fs::write(&path, "07".repeat(32)).unwrap();
        let signer = SignerConfig::SoftwareKey(path.clone());
        for mode in [0o644, 0o640, 0o604, 0o660] {
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
            let err = signer.open().err().expect("refused");
            assert!(err.to_string().contains("chmod 600"), "{err}");
        }
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(signer.open().is_ok());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn save_makes_its_directory_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("save");
        let path = dir.join("nested").join("config.toml");
        save(&path, &OperatorConfig::default()).unwrap();
        for d in [&dir, &dir.join("nested")] {
            let mode = std::fs::metadata(d).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o700, "{}", d.display());
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_parse_error_names_the_place_not_the_text() {
        let dir = scratch("parse");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            "server = \"https://example.org\"\ncredential = \"s3cr3t-looking\"\n",
        )
        .unwrap();
        let err = format!("{:#}", load(&path).unwrap_err());
        assert!(!err.contains("s3cr3t"), "{err}");
        assert!(err.contains("line 2, column"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unknown_fields_are_refused() {
        assert!(toml::from_str::<OperatorConfig>("server_url = \"x\"").is_err());
        // The server's settings are read from it, not kept here
        assert!(toml::from_str::<OperatorConfig>("bulk_threshold = 10").is_err());
    }
}
