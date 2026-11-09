//! A save cut between FsStorage's archive and its rename: agentkit writes
//! `state.json.tmp` (fsynced), renames `state.json` aside to
//! `state.<utc>.json`, then renames the tmp into place. A crash between
//! the two renames leaves no `state.json`, the old state archived, and the
//! new one complete in the tmp. At the start of a run, before states load,
//! such a tmp is renamed into place (`state_tmp_promoted`) only when all of
//! these hold:
//!
//! - no other runner shares the data dir (the daily and continuous units
//!   both run on it): `runner.lock` taken exclusively, without waiting;
//!   if it is held, nothing is promoted;
//! - the tmp is older than [`SETTLED`], so it isn't a save in flight;
//! - the newest archive (`state.<utc>.json`) is named at or after the
//!   tmp's mtime and within [`SETTLED`] of it: the archive of the same cut
//!   save, so a tmp beside a `state.json` removed by hand is not promoted
//!   (warned instead);
//! - the tmp parses as an FsStorage envelope (`{format, state}`).
//!
//! A tmp beside a `state.json` is left alone; nothing is deleted. Content
//! is never read beyond the envelope's shape, and never logged.

use std::path::Path;
use std::time::{Duration, SystemTime};

/// How old a tmp must be, and how close its archive.
pub const SETTLED: Duration = Duration::from_secs(600);

/// [`promote_tmp_states`] under `<data_dir>/runner.lock`, taken
/// exclusively without waiting: another runner on the data dir may be
/// mid-save, so if it is held nothing is promoted.
pub fn promote_if_alone(data_dir: &Path) -> usize {
    match crate::sealed::session::runner_exclusive(data_dir) {
        Ok(lock) => {
            let n = promote_tmp_states(&data_dir.join("state"), SystemTime::now());
            drop(lock);
            n
        }
        Err(e) => {
            tracing::info!(
                event_type = "state_tmp_promotion_skipped",
                error = %e,
                "another runner shares the data dir; no tmp promoted this start"
            );
            0
        }
    }
}

/// The newest `state.<utc>.json` archive's time, from its name.
fn newest_archive(dir: &Path) -> Option<SystemTime> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| {
            let name = e.ok()?.file_name().to_string_lossy().to_string();
            let stamp = name.strip_prefix("state.")?.strip_suffix(".json")?;
            // `-N` on a same-millisecond collision
            let stamp = stamp.split('-').next()?;
            let at =
                chrono::NaiveDateTime::parse_from_str(stamp, crate::sealed::STAMP_FORMAT).ok()?;
            Some(SystemTime::from(at.and_utc()))
        })
        .max()
}

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

/// Promote each agent's complete, settled `state.json.tmp` where
/// `state.json` is missing and the newest archive is the same save's.
/// Returns how many. `now` for tests.
pub fn promote_tmp_states(state_dir: &Path, now: SystemTime) -> usize {
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
        let not_promoted = |why: &str| {
            tracing::warn!(
                event_type = "state_tmp_not_promoted",
                path = %tmp.display(),
                why,
                "a state.json.tmp with no state.json; left alone"
            );
        };
        let Some(written) = std::fs::metadata(&tmp).and_then(|m| m.modified()).ok() else {
            not_promoted("its time can't be read");
            continue;
        };
        if now.duration_since(written).unwrap_or_default() < SETTLED {
            not_promoted("written too recently: a save may be in flight");
            continue;
        }
        let same_save = newest_archive(&dir).is_some_and(|archived| {
            archived
                .duration_since(written)
                .is_ok_and(|d| d <= SETTLED)
                // Name stamps are milliseconds; mtimes finer
                || written.duration_since(archived).is_ok_and(|d| d < Duration::from_millis(1))
        });
        if !same_save {
            not_promoted("no archive of the same save beside it (a state.json removed by hand?)");
            continue;
        }
        if !std::fs::read(&tmp).is_ok_and(|b| is_envelope(&b)) {
            not_promoted("not a complete envelope");
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
pub(crate) mod tests {
    use super::*;

    /// A cut save as FsStorage leaves it in `dir`: the tmp written at
    /// `written`, the old state archived under a name stamped `archived`.
    pub fn cut(dir: &Path, tmp: &[u8], written: SystemTime, archived: Option<SystemTime>) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("state.json.tmp"), tmp).unwrap();
        std::fs::File::options()
            .write(true)
            .open(dir.join("state.json.tmp"))
            .unwrap()
            .set_modified(written)
            .unwrap();
        if let Some(at) = archived {
            let stamp =
                chrono::DateTime::<chrono::Utc>::from(at).format(crate::sealed::STAMP_FORMAT);
            std::fs::write(dir.join(format!("state.{stamp}.json")), b"{}").unwrap();
        }
    }

    const ENV: &[u8] = br#"{"format": 2, "sealed": true, "state": {"memory": "x"}}"#;

    #[test]
    fn only_a_settled_complete_tmp_of_the_same_save_is_promoted() {
        let root = crate::sealed::TestDir::new("promote");
        let now = SystemTime::now();
        let ago = |s: u64| now - Duration::from_secs(s);
        // The cut save: tmp an hour old, archived a moment after it
        cut(&root.join("cut"), ENV, ago(3600), Some(ago(3599)));
        // A save in flight
        cut(&root.join("fresh"), ENV, ago(5), Some(ago(4)));
        // No archive: a state.json removed by hand
        cut(&root.join("by-hand"), ENV, ago(3600), None);
        // An archive from another save, a day before
        cut(&root.join("other-save"), ENV, ago(3600), Some(ago(90_000)));
        // Torn
        cut(
            &root.join("torn"),
            b"{\"format\": 2, \"sta",
            ago(3600),
            Some(ago(3599)),
        );
        // Beside a state.json
        cut(&root.join("beside"), ENV, ago(3600), Some(ago(3599)));
        std::fs::write(
            root.join("beside/state.json"),
            b"{\"format\":1,\"state\":{}}",
        )
        .unwrap();
        assert_eq!(promote_tmp_states(&root, now), 1);
        assert_eq!(std::fs::read(root.join("cut/state.json")).unwrap(), ENV);
        for d in ["fresh", "by-hand", "other-save", "torn"] {
            assert!(!root.join(d).join("state.json").exists(), "{d}");
            assert!(root.join(d).join("state.json.tmp").exists(), "{d} kept");
        }
        assert_eq!(
            std::fs::read(root.join("beside/state.json")).unwrap(),
            b"{\"format\":1,\"state\":{}}"
        );
    }

    /// Another runner on the data dir: nothing promoted, whatever is there.
    #[test]
    fn nothing_is_promoted_while_another_runner_shares_the_data_dir() {
        let root = crate::sealed::TestDir::new("promote-shared");
        let now = SystemTime::now();
        let ago = |s: u64| now - Duration::from_secs(s);
        cut(&root.join("state/a"), ENV, ago(3600), Some(ago(3599)));
        let other = crate::sealed::session::runner_shared(&root).unwrap();
        assert_eq!(promote_if_alone(&root), 0);
        assert!(!root.join("state/a/state.json").exists());
        drop(other);
        assert_eq!(promote_if_alone(&root), 1);
    }
}
