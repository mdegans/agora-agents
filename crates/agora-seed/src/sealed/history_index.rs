//! The `memory` tool's search index, a disposable sidecar beside the
//! agent's history: `sealed/history_index.json` for an agent with a
//! `sealed/` directory, else `memory_history_index.json` in its state
//! directory ([`location`]).
//!
//! The format is agentkit's (a deduplicated line table and each version's
//! line ids); this module only keeps the file: where it is, a whole-file
//! atomic write, and its removal by `sealed-forget`. It is built lazily at
//! search time, never on the save path, and rebuilt when it is missing or
//! doesn't match, so deleting it loses nothing. Departure doesn't delete it:
//! the runner's files are the operator's.
//!
//! **Never makes `sealed/`.** An agent without one has never been enrolled,
//! and a `sealed/` without its head reads as a sealed history that can't
//! be checked (the `memory` tool's resolver says so, and the start check
//! and enrolment look there), so its index stays beside its state.
//!
//! It holds memory text, so nothing here logs its content.

use std::path::{Path, PathBuf};

use agora_agentkit::state::storage::SEALED_DIR;

/// The index's file name, in the agent's `sealed/`.
pub const HISTORY_INDEX_FILE: &str = "history_index.json";

/// The index's file name in the state directory of an agent with no
/// `sealed/`.
pub const UNSEALED_INDEX_FILE: &str = "memory_history_index.json";

/// `<agent_dir>/sealed/history_index.json`.
pub fn path(agent_dir: &Path) -> PathBuf {
    agent_dir.join(SEALED_DIR).join(HISTORY_INDEX_FILE)
}

/// `<agent_dir>/memory_history_index.json`.
pub fn unsealed_path(agent_dir: &Path) -> PathBuf {
    agent_dir.join(UNSEALED_INDEX_FILE)
}

/// Where the agent's index is kept: in `sealed/` when it has one, else
/// beside its state. Never a reason to make `sealed/`.
pub fn location(agent_dir: &Path) -> PathBuf {
    match std::fs::metadata(agent_dir.join(SEALED_DIR)) {
        Ok(m) if m.is_dir() => path(agent_dir),
        _ => unsealed_path(agent_dir),
    }
}

/// The index's bytes, or `None` if there is none.
pub fn read(agent_dir: &Path) -> std::io::Result<Option<Vec<u8>>> {
    match std::fs::read(location(agent_dir)) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// Replace the index with `bytes`, whole and 0600 (it is a plaintext
/// corpus of every memory line, removed ones included): a uniquely named
/// temporary sibling, fsynced, renamed over it, the directory fsynced. An
/// agent's index from before it had `sealed/` is removed once the one in
/// `sealed/` is written, so its lines are held once.
pub fn write(agent_dir: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let at = location(agent_dir);
    super::write_private(&at, bytes)?;
    if at == path(agent_dir) {
        remove_file(&unsealed_path(agent_dir))?;
    }
    Ok(())
}

/// Delete the index, if there is one, wherever it is: `Ok(true)` if one
/// was there. Only this sidecar: it is rebuilt from the history whenever
/// it is needed.
pub fn remove(agent_dir: &Path) -> std::io::Result<bool> {
    let sealed = remove_file(&path(agent_dir))?;
    let unsealed = remove_file(&unsealed_path(agent_dir))?;
    Ok(sealed || unsealed)
}

fn remove_file(path: &Path) -> std::io::Result<bool> {
    match std::fs::remove_file(path) {
        Ok(()) => {
            if let Some(dir) = path.parent() {
                std::fs::File::open(dir)?.sync_all()?;
            }
            Ok(true)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_index_is_written_whole_read_back_and_removed() {
        let root = crate::sealed::TestDir::new("history-index");
        let agent = root.join("agent");
        std::fs::create_dir_all(agent.join(SEALED_DIR)).unwrap();
        assert_eq!(read(&agent).unwrap(), None);
        assert!(!remove(&agent).unwrap(), "nothing to remove");
        write(&agent, b"{\"v\":1}").unwrap();
        write(&agent, b"{\"v\":2}").unwrap();
        assert_eq!(read(&agent).unwrap().as_deref(), Some(&b"{\"v\":2}"[..]));
        let names: Vec<String> = std::fs::read_dir(agent.join(SEALED_DIR))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, [HISTORY_INDEX_FILE], "no temporary left");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(path(&agent))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        assert!(remove(&agent).unwrap());
        assert_eq!(read(&agent).unwrap(), None);
    }

    /// An agent with no `sealed/` keeps its index beside its state, and
    /// `sealed/` is never made; once it has one, the index moves into it
    /// at the next write, held once.
    #[test]
    fn an_unsealed_agents_index_is_beside_its_state_and_makes_no_sealed_dir() {
        let root = crate::sealed::TestDir::new("history-index-unsealed");
        let agent = root.join("agent");
        std::fs::create_dir_all(&agent).unwrap();
        write(&agent, b"{\"v\":1}").unwrap();
        assert!(
            !std::fs::exists(agent.join(SEALED_DIR)).unwrap(),
            "no sealed/"
        );
        let names: Vec<String> = std::fs::read_dir(&agent)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, [UNSEALED_INDEX_FILE], "no temporary left");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(unsealed_path(&agent))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        assert_eq!(read(&agent).unwrap().as_deref(), Some(&b"{\"v\":1}"[..]));
        assert!(!crate::sealed::is_enrolled(&agent));
        // Enrolled later: the next write goes to sealed/, the old one goes
        std::fs::create_dir_all(agent.join(SEALED_DIR)).unwrap();
        assert_eq!(
            read(&agent).unwrap(),
            None,
            "rebuilt there, not read from beside"
        );
        write(&agent, b"{\"v\":2}").unwrap();
        assert!(!std::fs::exists(unsealed_path(&agent)).unwrap());
        assert_eq!(read(&agent).unwrap().as_deref(), Some(&b"{\"v\":2}"[..]));
        // Forget removes either
        std::fs::write(unsealed_path(&agent), b"x").unwrap();
        assert!(remove(&agent).unwrap());
        assert!(!std::fs::exists(unsealed_path(&agent)).unwrap());
        assert!(!std::fs::exists(path(&agent)).unwrap());
    }
}
