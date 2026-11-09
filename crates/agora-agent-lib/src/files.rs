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

/// Write `contents` to a new file at `path`, readable by this user alone, fully or
/// not at all, and never over anything already there (a symlink included). `false`
/// if something was. Makes the parent if needed; the temp file doesn't outlive this.
pub fn write_secret_exclusive(path: &Path, contents: &[u8]) -> Result<bool> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;

    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    std::fs::create_dir_all(parent)
        .with_context(|| format!("failed to create {}", parent.display()))?;
    let name = path
        .file_name()
        .with_context(|| format!("{} has no file name", path.display()))?;
    let tmp = parent.join(format!(
        ".{}.tmp-{}",
        name.to_string_lossy(),
        uuid::Uuid::new_v4()
    ));
    // `create_new` is O_EXCL, which doesn't follow a symlink
    let written = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&tmp)
        .and_then(|mut file| {
            file.write_all(contents)?;
            file.sync_all()
        });
    // `link` fails if the name exists, in any form, and only one of two racers wins
    let linked = written.and_then(|()| std::fs::hard_link(&tmp, path));
    let _ = std::fs::remove_file(&tmp);
    match linked {
        Ok(()) => {
            std::fs::File::open(parent)
                .and_then(|dir| dir.sync_all())
                .with_context(|| format!("failed to sync {}", parent.display()))?;
            Ok(true)
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
        Err(e) => Err(e).with_context(|| format!("failed to write {}", path.display())),
    }
}
