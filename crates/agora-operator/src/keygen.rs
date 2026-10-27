//! `keygen`: a new Ed25519 key in `~/.config/agora/operator/<label>.key`
//! (hex, mode 600, beside the config file), and its public key as the portal's
//! "Add signing key" form takes it (64 hex digits).

use std::fmt::Write;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use agora_agentkit::crypto;
use anyhow::{Context, Result, bail};
use url::Url;
use zeroize::Zeroizing;

use crate::config::{self, OperatorConfig, SignerConfig};
use crate::sign::public_key_hex;

/// Where the portal adds a key, under the server's URL
const CREDENTIALS_PAGE: &str = "agora/account/credentials";

pub fn run(
    config_path: &Path,
    config: &mut OperatorConfig,
    label: &str,
    server: Option<&Url>,
) -> Result<()> {
    // Beside the config: `~/.config/agora/operator/` unless `--config`
    // names another file
    let dir = config_path
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    // Only the default directory is ours to tighten; one the user named is
    // theirs, and is refused rather than changed
    let default_dir = std::path::absolute(config::operator_dir()?)?;
    let may_tighten = std::path::absolute(dir)? == default_dir;
    let key_path = write_key(dir, label, may_tighten)?;
    let signer = SignerConfig::SoftwareKey(key_path.clone());
    let public = public_key_hex(&signer.open()?.public_key());

    println!("Wrote {} (readable by you alone).", key_path.display());
    println!("Public key:\n\n    {public}\n");
    let page = server
        .or(config.server.as_ref())
        .and_then(|s| page_url(s).ok());
    match page {
        Some(page) => println!("Add it at {page} (label: {label})."),
        None => println!("Add it in the portal: Account, Signing keys (label: {label})."),
    }
    if config.signer.is_none() {
        config.signer = Some(signer);
        config::save(config_path, config)?;
        println!("The config now signs with it ({}).", config_path.display());
    } else {
        println!(
            "The config signs with another key; to use this one, \
             `agora-operator config --key-file {}`.",
            key_path.display()
        );
    }
    println!(
        "Then save the credential id the portal shows: \
         `agora-operator config --credential <id>`."
    );
    Ok(())
}

/// A new key at `<dir>/<label>.key` (an absolute path), never over an
/// existing one, since a key already registered would be lost. Only in a
/// directory this user owns at 0700: one made here is made so. An existing
/// one open to others is tightened (and says so) only if `may_tighten`, for
/// the default directory; any other is refused, never chmodded
fn write_key(dir: &Path, label: &str, may_tighten: bool) -> Result<PathBuf> {
    let dir = std::path::absolute(dir)?;
    config::create_private_dir(&dir)?;
    let mode = std::fs::metadata(&dir)?.permissions().mode() & 0o7777;
    if mode & 0o077 != 0 && !may_tighten {
        bail!(
            "{} is open to others (mode {mode:04o}), so no key is written there: \
             `chmod 700 {}` it yourself, or choose another path",
            dir.display(),
            dir.display()
        );
    }
    if mode & 0o077 != 0 {
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).with_context(
            || {
                format!(
                    "{} is open to others (mode {mode:04o}) and can't be made 0700",
                    dir.display()
                )
            },
        )?;
        eprintln!(
            "Tightened {} from {mode:04o} to 0700: keys are kept only where no one else can \
             look.",
            dir.display()
        );
    }
    let path = dir.join(format!("{label}.key"));
    let (key, _) = crypto::generate_keypair();
    let hex = Zeroizing::new(crypto::signing_key_to_hex(&key));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&path)
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::AlreadyExists => anyhow::anyhow!(
                "{} exists: choose another --label (a key is never overwritten)",
                path.display()
            ),
            _ => anyhow::Error::new(e).context(format!("writing {}", path.display())),
        })?;
    // The directory must be this user's: the file just made is, by its uid
    let dir_meta = std::fs::metadata(&dir)?;
    if dir_meta.uid() != file.metadata()?.uid() || dir_meta.permissions().mode() & 0o077 != 0 {
        drop(file);
        std::fs::remove_file(&path)?;
        bail!(
            "{} is not a directory you own at 0700; keys are written only to one",
            dir.display()
        );
    }
    use std::io::Write as _;
    writeln!(file, "{}", hex.as_str()).with_context(|| format!("writing {}", path.display()))?;
    file.sync_all()?;
    Ok(path)
}

fn page_url(server: &Url) -> Result<Url> {
    let mut base = server.clone();
    if !base.path().ends_with('/') {
        let path = format!("{}/", base.path());
        base.set_path(&path);
    }
    Ok(base.join(CREDENTIALS_PAGE)?)
}

/// The config as `config` shows it: never the key, only its file and
/// public key
pub fn describe(path: &Path, config: &OperatorConfig) -> String {
    let mut out = format!("Config: {}\n", path.display());
    let unset = "(not set)";
    let _ = writeln!(
        out,
        "  server:         {}",
        config.server.as_ref().map_or(unset.into(), Url::to_string)
    );
    let _ = writeln!(
        out,
        "  credential:     {}",
        config.credential.map_or(unset.into(), |c| c.to_string())
    );
    match &config.signer {
        None => {
            let _ = writeln!(out, "  key:            {unset}");
        }
        Some(signer @ SignerConfig::SoftwareKey(file)) => {
            let _ = writeln!(out, "  key:            software key {}", file.display());
            let public = match signer.open() {
                Ok(s) => public_key_hex(&s.public_key()),
                Err(e) => format!("(unreadable: {e:#})"),
            };
            let _ = writeln!(out, "  public key:     {public}");
        }
    }
    let _ = writeln!(out, "  bulk threshold: {}", config.bulk_threshold);
    if config.server.is_none() {
        out.push_str("Set the server with `agora-operator config --server URL`.\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("agora-operator-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_key_is_written_once_readable_by_its_owner_alone() {
        let dir = scratch("keygen");
        let path = write_key(&dir, "laptop", false).unwrap();
        assert_eq!(path, std::path::absolute(dir.join("laptop.key")).unwrap());
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
        let dir_mode = std::fs::metadata(&dir).unwrap().permissions().mode();
        assert_eq!(dir_mode & 0o777, 0o700);
        // It opens as the signer the CLI signs with
        let signer = SignerConfig::SoftwareKey(path.clone()).open().unwrap();
        assert_eq!(public_key_hex(&signer.public_key()).len(), 64);
        // and is never overwritten
        let before = std::fs::read(&path).unwrap();
        let err = write_key(&dir, "laptop", false).unwrap_err();
        assert!(err.to_string().contains("never overwritten"), "{err}");
        assert_eq!(std::fs::read(&path).unwrap(), before);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_open_directory_is_tightened_and_the_path_is_absolute() {
        let dir = scratch("open-dir");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        // A directory the user named is refused, and left as it was
        let err = write_key(&dir, "k", false).unwrap_err().to_string();
        assert!(err.contains("chmod 700"), "{err}");
        let mode = std::fs::metadata(&dir).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o755);
        assert!(!dir.join("k.key").exists());
        // The default directory is tightened
        let path = write_key(&dir, "k", true).unwrap();
        let mode = std::fs::metadata(&dir).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o700);
        assert!(path.is_absolute());
        std::fs::remove_dir_all(&dir).unwrap();

        // A relative directory is stored absolute
        let rel = PathBuf::from(format!(".agora-operator-test-rel-{}", std::process::id()));
        let path = write_key(&rel, "k", false).unwrap();
        assert!(path.is_absolute(), "{}", path.display());
        assert_eq!(path, std::path::absolute(rel.join("k.key")).unwrap());
        std::fs::remove_dir_all(&rel).unwrap();
    }

    #[test]
    fn the_portal_page_is_under_the_server() {
        let page = |s: &str| page_url(&s.parse().unwrap()).unwrap().to_string();
        assert_eq!(
            page("http://localhost:8099"),
            "http://localhost:8099/agora/account/credentials"
        );
        assert_eq!(
            page("https://example.org/"),
            "https://example.org/agora/account/credentials"
        );
    }

    #[test]
    fn describe_never_prints_the_key() {
        let dir = scratch("describe");
        let path = write_key(&dir, "k", false).unwrap();
        let secret = std::fs::read_to_string(&path).unwrap();
        let config = OperatorConfig {
            signer: Some(SignerConfig::SoftwareKey(path)),
            ..OperatorConfig::default()
        };
        let text = describe(&dir.join("config.toml"), &config);
        assert!(!text.contains(secret.trim()), "{text}");
        assert!(text.contains("public key:"), "{text}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
