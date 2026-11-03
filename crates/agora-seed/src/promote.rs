//! A save cut between FsStorage's archive and its rename: agentkit writes
//! `state.json.tmp` (fsynced), renames `state.json` aside to
//! `state.<utc>.json`, then renames the tmp into place. A crash between
//! the two renames leaves no `state.json`, the old state archived, and the
//! new one complete in the tmp. At the start of a run, before states load,
//! such a tmp is renamed into place (`state_tmp_promoted`), if it parses as
//! an FsStorage envelope (`{format, state}`). A tmp beside a `state.json`
//! is left alone; nothing is deleted. Content is never read beyond the
//! envelope's shape, and never logged.

use std::path::Path;

/// Whether `bytes` are an FsStorage envelope: an object with a numeric
/// `format` and a `state`.
fn is_envelope(bytes: &[u8]) -> bool {
    #[derive(serde::Deserialize)]
    struct Envelope {
        #[allow(dead_code)]
        format: u64,
        #[allow(dead_code)]
        state: serde::de::IgnoredAny,
    }
    serde_json::from_slice::<Envelope>(bytes).is_ok()
}

/// Promote each agent's complete `state.json.tmp` where `state.json` is
/// missing. Returns how many.
pub fn promote_tmp_states(state_dir: &Path) -> usize {
    let entries = match std::fs::read_dir(state_dir) {
        Ok(entries) => entries,
        Err(e) => {
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(path = %state_dir.display(), error = %e, "state dir unreadable for tmp promotion");
            }
            return 0;
        }
    };
    let mut promoted = 0;
    for entry in entries.flatten() {
        let dir = entry.path();
        let (state, tmp) = (dir.join("state.json"), dir.join("state.json.tmp"));
        if std::fs::exists(&state).unwrap_or(true) || !std::fs::exists(&tmp).unwrap_or(false) {
            continue;
        }
        let ok = std::fs::read(&tmp).is_ok_and(|b| is_envelope(&b));
        if !ok {
            tracing::warn!(
                event_type = "state_tmp_not_promoted",
                path = %tmp.display(),
                "a state.json.tmp with no state.json isn't a complete envelope; left alone"
            );
            continue;
        }
        let result =
            std::fs::rename(&tmp, &state).and_then(|()| std::fs::File::open(&dir)?.sync_all());
        match result {
            Ok(()) => {
                promoted += 1;
                tracing::warn!(
                    event_type = "state_tmp_promoted",
                    path = %state.display(),
                    "a save cut between its archive and its rename: its complete tmp put in place"
                );
            }
            Err(e) => {
                tracing::error!(path = %tmp.display(), error = %e, "the tmp couldn't be put in place")
            }
        }
    }
    promoted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_complete_tmp_without_a_state_is_promoted_and_nothing_else() {
        let root = crate::sealed::TestDir::new("promote");
        let mk = |name: &str| {
            let d = root.join(name);
            std::fs::create_dir_all(&d).unwrap();
            d
        };
        let env = br#"{"format": 2, "sealed": true, "state": {"memory": "x"}}"#;
        let cut = mk("cut");
        std::fs::write(cut.join("state.json.tmp"), env).unwrap();
        std::fs::write(cut.join("state.20261103T000000000Z.json"), b"{}").unwrap();
        let beside = mk("beside");
        std::fs::write(beside.join("state.json"), b"{\"format\":1,\"state\":{}}").unwrap();
        std::fs::write(beside.join("state.json.tmp"), env).unwrap();
        let torn = mk("torn");
        std::fs::write(torn.join("state.json.tmp"), b"{\"format\": 2, \"sta").unwrap();
        assert_eq!(promote_tmp_states(&root), 1);
        assert_eq!(std::fs::read(cut.join("state.json")).unwrap(), env);
        assert!(!cut.join("state.json.tmp").exists());
        assert!(
            beside.join("state.json.tmp").exists(),
            "left alone beside a state"
        );
        assert_eq!(
            std::fs::read(beside.join("state.json")).unwrap(),
            b"{\"format\":1,\"state\":{}}"
        );
        assert!(torn.join("state.json.tmp").exists() && !torn.join("state.json").exists());
    }
}
