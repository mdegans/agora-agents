//! The config directory and whole-file writes, shared by the agent CLI
//! (`agora`) and the operator CLI (`agora-operator`).

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// The agora config directory: `~/.config/agora` on Linux,
/// `~/Library/Application Support/agora` on macOS.
pub fn config_dir() -> Result<PathBuf> {
    let base = dirs::config_dir().context("could not determine XDG config directory")?;
    Ok(base.join("agora"))
}

/// Write `path` whole or not at all: two CLIs at once never leave half a file
pub fn write_atomically(path: &Path, contents: &[u8]) -> Result<()> {
    write_atomically_with_mode(path, contents, 0o644)
}

/// [`write_atomically`], readable by this user alone: for keys
pub fn write_secret_atomically(path: &Path, contents: &[u8]) -> Result<()> {
    write_atomically_with_mode(path, contents, 0o600)
}

fn write_atomically_with_mode(path: &Path, contents: &[u8], mode: u32) -> Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;

    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(mode)
        .open(&tmp)
        .with_context(|| format!("failed to write {}", tmp.display()))?;
    file.write_all(contents)
        .with_context(|| format!("failed to write {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("failed to write {}", path.display()))
}
