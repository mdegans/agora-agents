//! `sealed/history_index.json`: the `memory` tool's search index, a
//! disposable sidecar beside the agent's sealed history.
//!
//! The format is agentkit's (a deduplicated line table and each version's
//! line ids); this module only keeps the file: where it is, a whole-file
//! atomic write, and its removal by `sealed-forget`. It is built lazily at
//! search time, never on the save path, and rebuilt when it is missing or
//! doesn't match, so deleting it loses nothing. Departure doesn't delete it:
//! the runner's files are the operator's.
//!
//! It holds memory text, so nothing here logs its content.

use std::path::{Path, PathBuf};

use agora_agentkit::state::storage::SEALED_DIR;

/// The index's file name, in the agent's `sealed/`.
pub const HISTORY_INDEX_FILE: &str = "history_index.json";

/// `<agent_dir>/sealed/history_index.json`.
pub fn path(agent_dir: &Path) -> PathBuf {
    agent_dir.join(SEALED_DIR).join(HISTORY_INDEX_FILE)
}

/// The index's bytes, or `None` if there is none.
pub fn read(agent_dir: &Path) -> std::io::Result<Option<Vec<u8>>> {
    match std::fs::read(path(agent_dir)) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// Replace the index with `bytes`, whole: a temporary sibling, fsynced,
/// renamed over it, the directory fsynced.
pub fn write(agent_dir: &Path, bytes: &[u8]) -> std::io::Result<()> {
    super::anchors::write_atomic(&path(agent_dir), bytes)
}

/// Delete the index, if there is one: `Ok(true)` if it was there. Only
/// this sidecar: it is rebuilt from the history whenever it is needed.
pub fn remove(agent_dir: &Path) -> std::io::Result<bool> {
    let path = path(agent_dir);
    match std::fs::remove_file(&path) {
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
        assert!(remove(&agent).unwrap());
        assert_eq!(read(&agent).unwrap(), None);
    }
}
