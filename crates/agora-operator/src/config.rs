//! The operator CLI's config: which server, which credential, which key.
//!
//! `~/.config/agora/operator/config.toml` by default, beside the keys
//! `keygen` writes (`~/.config/agora/operator/<label>.key`). The config
//! names the key's file; it never holds the key.

use std::path::{Path, PathBuf};

use agora_agent_lib::files::{config_dir, write_atomically};
use agora_agentkit::ids::OperatorCredentialId;
use agora_agentkit::operator_signer::{OperatorSigner, SoftwareKeySigner};
#[cfg(feature = "yubikey")]
use agora_agentkit::operator_signer::{Pin, PivSerial, PivSlot};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use url::Url;

/// How the CLI signs: one variant per kind of signer, each opened by
/// [`SignerConfig::open`] into the same `Box<dyn OperatorSigner>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum SignerConfig {
    /// An Ed25519 key in a file, as hex (`keygen` writes one)
    SoftwareKey(PathBuf),
    /// A key in a YubiKey's PIV slot (`yubikey provision` sets one up). The
    /// PIN is read from `pin_file` (this user's alone), never from the
    /// command line or the environment
    #[cfg(feature = "yubikey")]
    Yubikey {
        /// The slot: `authentication` (9a) for unattended signing
        slot: PivSlot,
        /// The file holding the PIV PIN
        pin_file: PathBuf,
        /// The device's serial, when more than one may be connected
        #[serde(default, skip_serializing_if = "Option::is_none")]
        serial: Option<PivSerial>,
    },
}

impl SignerConfig {
    /// The signer, ready to sign (a YubiKey's PIN presented)
    pub async fn open(&self) -> Result<Box<dyn OperatorSigner>> {
        match self {
            Self::SoftwareKey(path) => Ok(Box::new({
                check_private(path)?;
                SoftwareKeySigner::from_file(path)
                    .with_context(|| format!("opening the key {}", path.display()))?
            })),
            #[cfg(feature = "yubikey")]
            Self::Yubikey {
                slot,
                pin_file,
                serial,
            } => {
                let pin = read_pin(pin_file)?;
                let signer =
                    agora_agentkit::operator_signer::yubikey::YubikeySigner::open_with_pin(
                        *serial, *slot, pin,
                    )
                    .await
                    .with_context(|| format!("opening the YubiKey's slot {slot}"))?;
                Ok(Box::new(signer))
            }
        }
    }
}

/// A PIV PIN from `path`: a file this user alone can read, as for key
/// files. Never argv or the environment, where `ps` and logs show them
#[cfg(feature = "yubikey")]
pub fn read_pin(path: &Path) -> Result<Pin> {
    check_private(path)?;
    let secret = agora_agentkit::secrets::Secret::from_file(path)
        .with_context(|| format!("reading the PIN file {}", path.display()))?;
    if secret.expose().trim().is_empty() {
        bail!("the PIN file {} is empty", path.display());
    }
    Ok(Pin::from_secret(agora_agentkit::secrets::Secret::new(
        secret.expose().trim().to_owned(),
    )))
}

/// Refuse a key or PIN file others can read or write, as ssh does
pub fn check_private(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mode = std::fs::metadata(path)
        .with_context(|| format!("opening the key {}", path.display()))?
        .permissions()
        .mode();
    if mode & 0o077 != 0 {
        bail!(
            "{} is open to others (mode {:04o}), so it is not used: \
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
    /// The key `--sudo` signs with: a hardware key attested with touch
    /// policy `always` (a YubiKey's 9c slot, `slot = "signature"`), added in
    /// the portal as a credential of its own. Set by hand:
    ///
    /// ```toml
    /// [sudo]
    /// credential = "<its id from the portal>"
    /// [sudo.signer.yubikey]
    /// slot = "signature"
    /// ```
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sudo: Option<SudoConfig>,
}

/// The `[sudo]` table: a credential and its signer, like the top level's
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SudoConfig {
    pub credential: OperatorCredentialId,
    pub signer: SignerConfig,
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

    /// The `[sudo]` credential and its signer, opened
    pub async fn sudo(&self) -> Result<(OperatorCredentialId, Box<dyn OperatorSigner>)> {
        let sudo = self.sudo.as_ref().with_context(|| {
            "no sudo key: add a hardware key with touch policy `always` in the portal (Account, \
             Signing keys), then give it a `[sudo]` table in the config (see `agora-operator \
             config --help`)"
        })?;
        Ok((sudo.credential, sudo.signer.open().await?))
    }

    pub async fn signer(&self) -> Result<Box<dyn OperatorSigner>> {
        match &self.signer {
            Some(signer) => signer.open().await,
            None => bail!(
                "no signing key: `agora-operator keygen`, or \
                 `agora-operator config --key-file PATH`, or a YubiKey: \
             `agora-operator yubikey provision`"
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
            sudo: None,
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

    /// The `[sudo]` table as its doc writes it
    #[cfg(feature = "yubikey")]
    #[test]
    fn a_sudo_table_names_its_credential_and_slot() {
        let config: OperatorConfig = toml::from_str(
            r#"
            credential = "0b7e2a61-5d3c-4c8e-8f1a-2e9d4b6c7a10"
            [signer.yubikey]
            slot = "authentication"
            pin_file = "/k/pin"
            [sudo]
            credential = "1c8f3b72-6e4d-4d9f-9a2b-3f0e5c7d8b21"
            [sudo.signer.yubikey]
            slot = "signature"
            pin_file = "/k/pin"
            "#,
        )
        .unwrap();
        let sudo = config.sudo.as_ref().expect("a sudo table");
        assert_eq!(
            sudo.credential.to_string(),
            "1c8f3b72-6e4d-4d9f-9a2b-3f0e5c7d8b21"
        );
        assert!(matches!(
            sudo.signer,
            SignerConfig::Yubikey {
                slot: PivSlot::Signature,
                ..
            }
        ));
        let text = toml::to_string_pretty(&config).unwrap();
        assert_eq!(toml::from_str::<OperatorConfig>(&text).unwrap(), config);
    }

    #[cfg(feature = "yubikey")]
    #[test]
    fn a_yubikey_config_names_its_slot_and_pin_file_only() {
        let config = OperatorConfig {
            signer: Some(SignerConfig::Yubikey {
                slot: PivSlot::Authentication,
                pin_file: "/k/pin".into(),
                serial: None,
            }),
            ..OperatorConfig::default()
        };
        let text = toml::to_string_pretty(&config).unwrap();
        assert!(text.contains("slot = \"authentication\""), "{text}");
        assert!(text.contains("pin_file = \"/k/pin\""), "{text}");
        assert_eq!(toml::from_str::<OperatorConfig>(&text).unwrap(), config);
    }

    #[cfg(feature = "yubikey")]
    #[test]
    fn a_pin_file_open_to_others_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("pin");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pin");
        std::fs::write(&path, "123456\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(
            read_pin(&path)
                .unwrap_err()
                .to_string()
                .contains("chmod 600")
        );
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(read_pin(&path).unwrap().expose(), "123456");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn a_key_open_to_others_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("mode");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("k.key");
        std::fs::write(&path, "07".repeat(32)).unwrap();
        let signer = SignerConfig::SoftwareKey(path.clone());
        for mode in [0o644, 0o640, 0o604, 0o660] {
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
            let err = signer.open().await.err().expect("refused");
            assert!(err.to_string().contains("chmod 600"), "{err}");
        }
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(signer.open().await.is_ok());
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
