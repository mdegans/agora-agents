//! A save cut between FsStorage's archive and its rename, put in place at
//! the start of a run, before states load (`state_tmp_promoted`).
//!
//! **The write order since agentkit 0.158** (`FsStorage::write`): lock the
//! agent's directory, set a leftover `state.json.tmp` aside as
//! `state.json.tmp.<utc-ms>.aside`, hard-link `state.json` to
//! `state.<utc-ms>.json`, create the tmp (`create_new`), write, fsync,
//! rename it over `state.json`, fsync the directory. A failure before the
//! rename removes that save's own tmp and link. So a crash leaves
//! `state.json` present (the old state) and sharing its inode with the
//! newest archive, and maybe a `state.json.tmp`, complete only if the crash
//! came after its fsync. Left alone, the next save sets that tmp aside
//! (nothing lost), but the session it held is dropped unless an operator
//! runs `sealed-restore --from-cut-save`. It is renamed over `state.json`
//! when all of these hold:
//!
//! - `state.json` has more than one link and its inode is the newest
//!   archive's (the cut save's own link);
//! - the tmp is older than [`SETTLED`] and parses as an FsStorage envelope
//!   (`{format, state}`);
//! - for a chained agent, the sealed evidence agrees: a prepared outbox
//!   item following the current head records the tmp's state hash
//!   ([`crate::sealed::restore::cut_saves`], the rule `sealed-restore
//!   --from-cut-save` uses; agentkit seals a save before its local write).
//!   An agent not chained has no such evidence; the link and the envelope
//!   are its whole proof (FsStorage leaves that shape only on a crash).
//!
//! Without them the tmp is left alone and warned (`state_tmp_not_promoted`);
//! a tmp with no link beside it (no save of the new order cut it) is left
//! to agentkit's next save, unwarned.
//!
//! **The old order** (before 0.158: the tmp, then `state.json` renamed to
//! its archive, then the tmp renamed into place) left no `state.json` at
//! all. For a tmp left so by a crash before the upgrade, the tmp is renamed
//! into place when it is settled, a complete envelope, and the newest
//! archive is named at or after its time and within [`SETTLED`] of it (the
//! same save's), so a tmp beside a `state.json` removed by hand is not.
//!
//! **Who else may be writing.** Promotion runs only while no runner shares
//! the data dir: `runner.lock` taken exclusively, not waiting, so neither a
//! runner nor an operator command that writes (each holds it) is running.
//! Then, per agent, the locks every writer of `state.json` takes, in the
//! one order ([`crate::sealed::session`]): its session lock (tried; it
//! also stands for `sealed-stop`, which runs beside a runner), its
//! `sealed/lock` for a chained agent (agentkit's, held across each sealed
//! save and the evidence read), and agentkit's lock on the directory itself
//! (tried; the lock `FsStorage` holds across each save). Nothing is
//! deleted, and content is never read beyond the envelope's shape and the
//! hash, and never logged.

use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::time::{Duration, SystemTime};

use agora_agentkit::ids::AgentId;
use agora_agentkit::state::storage::AgentSeal;

/// How old a tmp must be, and (the old order) how close its archive.
pub const SETTLED: Duration = Duration::from_secs(600);

/// [`promote_tmp_states`] under `<data_dir>/runner.lock`, taken
/// exclusively without waiting: another runner on the data dir may be
/// mid-save, so if it is held nothing is promoted.
pub async fn promote_if_alone(data_dir: &Path) -> usize {
    match crate::sealed::session::runner_exclusive(data_dir) {
        Ok(lock) => {
            let n = promote_tmp_states(&data_dir.join("state"), SystemTime::now()).await;
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

/// The newest `state.<utc>.json` archive in `dir`: its time from its name,
/// and its path.
fn newest_archive(dir: &Path) -> Option<(SystemTime, std::path::PathBuf)> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| {
            let e = e.ok()?;
            let name = e.file_name().to_string_lossy().to_string();
            let stamp = name.strip_prefix("state.")?.strip_suffix(".json")?;
            // `-N` on a same-millisecond collision
            let (stamp, n) = match stamp.split_once('-') {
                Some((stamp, n)) => (stamp, n.parse::<u32>().ok()?),
                None => (stamp, 0),
            };
            let at =
                chrono::NaiveDateTime::parse_from_str(stamp, crate::sealed::STAMP_FORMAT).ok()?;
            Some(((SystemTime::from(at.and_utc()), n), e.path()))
        })
        .max_by_key(|(key, _)| *key)
        .map(|((at, _), path)| (at, path))
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

/// What became of one agent's tmp.
enum Promotion {
    Promoted,
    /// Nothing to do, nothing to say: no tmp, or not a cut save's shape.
    Nothing,
    /// Left alone, and why.
    Left(&'static str),
}

/// Promote each agent's complete, settled `state.json.tmp` left by a cut
/// save (see the [module docs](self)). Returns how many. `now` for tests.
pub async fn promote_tmp_states(state_dir: &Path, now: SystemTime) -> usize {
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
        let Some(id) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<uuid::Uuid>().ok())
            .map(AgentId::from)
        else {
            continue;
        };
        let tmp = dir.join("state.json.tmp");
        if !std::fs::exists(&tmp).unwrap_or(false) {
            continue;
        }
        match promote_one(state_dir, &dir, id, now).await {
            Promotion::Promoted => {
                promoted += 1;
                tracing::warn!(
                    event_type = "state_tmp_promoted",
                    agent_id = %id,
                    path = %dir.join("state.json").display(),
                    "a save cut between its archive and its rename: its complete tmp put in place"
                );
            }
            Promotion::Nothing => {}
            Promotion::Left(why) => tracing::warn!(
                event_type = "state_tmp_not_promoted",
                agent_id = %id,
                path = %tmp.display(),
                why,
                "a state.json.tmp left by a cut save; left alone"
            ),
        }
    }
    promoted
}

/// One agent's tmp, under its locks (see the [module docs](self)).
async fn promote_one(state_dir: &Path, dir: &Path, id: AgentId, now: SystemTime) -> Promotion {
    let (state, tmp) = (dir.join("state.json"), dir.join("state.json.tmp"));
    let _session = match crate::sealed::session::try_session(state_dir, id) {
        Ok(Some(lock)) => lock,
        Ok(None) => return Promotion::Left("the agent is held by another process"),
        Err(_) => return Promotion::Left("the agent's session lock couldn't be taken"),
    };
    let seal = AgentSeal::new(dir, id);
    let chained = match (seal.is_enrolled().await, seal.has_legacy_marker().await) {
        (Ok(a), Ok(b)) => a || b,
        _ => return Promotion::Left("the chain's marker couldn't be read"),
    };
    // agentkit's own lock across a sealed save, then the evidence under it
    let _seal_lock = if chained {
        match seal.lock().await {
            Ok(lock) => Some(lock),
            Err(_) => return Promotion::Left("the agent's sealed/lock couldn't be taken"),
        }
    } else {
        None
    };
    let Ok(dir_lock) = std::fs::File::open(dir) else {
        return Promotion::Left("the agent's directory couldn't be opened");
    };
    if dir_lock.try_lock().is_err() {
        return Promotion::Left("a save holds the agent's directory");
    }
    let Some(written) = std::fs::metadata(&tmp).and_then(|m| m.modified()).ok() else {
        return Promotion::Left("its time can't be read");
    };
    let settled = now.duration_since(written).unwrap_or_default() >= SETTLED;
    let archive = newest_archive(dir);
    match std::fs::metadata(&state) {
        // The old order: no state.json at all
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            if !settled {
                return Promotion::Left("written too recently: a save may be in flight");
            }
            let same_save = archive.is_some_and(|(archived, _)| {
                archived
                    .duration_since(written)
                    .is_ok_and(|d| d <= SETTLED)
                    // Name stamps are milliseconds; mtimes finer
                    || written.duration_since(archived).is_ok_and(|d| d < Duration::from_millis(1))
            });
            if !same_save {
                return Promotion::Left(
                    "no state.json, and no archive of the same save beside it (a state.json \
                     removed by hand?)",
                );
            }
            if !std::fs::read(&tmp).is_ok_and(|b| is_envelope(&b)) {
                return Promotion::Left("no state.json, and the tmp is not a complete envelope");
            }
        }
        Err(_) => return Promotion::Left("state.json can't be read"),
        // The new order: state.json still the old state, linked as the
        // newest archive
        Ok(meta) => {
            if meta.nlink() < 2 {
                // Not a cut save of the new order; agentkit's next save
                // sets the tmp aside
                return Promotion::Nothing;
            }
            let linked = archive
                .and_then(|(_, path)| std::fs::metadata(path).ok())
                .is_some_and(|a| (a.dev(), a.ino()) == (meta.dev(), meta.ino()));
            if !linked {
                return Promotion::Left(
                    "state.json has other links, but the newest archive isn't one of them",
                );
            }
            if !settled {
                return Promotion::Left("written too recently: a save may be in flight");
            }
            if !std::fs::read(&tmp).is_ok_and(|b| is_envelope(&b)) {
                return Promotion::Left(
                    "not a complete envelope: the save was cut before its tmp was whole",
                );
            }
            if chained {
                let evidence = match crate::sealed::restore::cut_saves(&seal).await {
                    Ok((cuts, _)) => cuts.iter().any(|c| c.path == tmp),
                    Err(_) => false,
                };
                if !evidence {
                    return Promotion::Left(
                        "no sealed version prepared after the head records the tmp's state",
                    );
                }
            }
        }
    }
    let result = std::fs::rename(&tmp, &state).and_then(|()| std::fs::File::open(dir)?.sync_all());
    match result {
        Ok(()) => Promotion::Promoted,
        Err(e) => {
            tracing::error!(agent_id = %id, path = %tmp.display(), error = %e, "the tmp couldn't be put in place");
            Promotion::Left("the rename failed")
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use agora_agentkit::reactor::{FsStorage, Storage};

    fn stamp(at: SystemTime) -> String {
        chrono::DateTime::<chrono::Utc>::from(at)
            .format(crate::sealed::STAMP_FORMAT)
            .to_string()
    }

    fn set_mtime(path: &Path, at: SystemTime) {
        std::fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(at)
            .unwrap();
    }

    /// A cut save of the old order as FsStorage left it in `dir`: no
    /// state.json, the tmp written at `written`, the old state archived
    /// under a name stamped `archived`.
    pub fn cut(dir: &Path, tmp: &[u8], written: SystemTime, archived: Option<SystemTime>) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("state.json.tmp"), tmp).unwrap();
        set_mtime(&dir.join("state.json.tmp"), written);
        if let Some(at) = archived {
            std::fs::write(dir.join(format!("state.{}.json", stamp(at))), b"{}").unwrap();
        }
    }

    /// A cut save of the new order: `state.json` hard-linked as the newest
    /// archive (stamped `written`), and `tmp` beside it, if any.
    fn cut_linked(dir: &Path, tmp: Option<&[u8]>, written: SystemTime) {
        std::fs::hard_link(
            dir.join("state.json"),
            dir.join(format!("state.{}.json", stamp(written))),
        )
        .unwrap();
        if let Some(tmp) = tmp {
            std::fs::write(dir.join("state.json.tmp"), tmp).unwrap();
            set_mtime(&dir.join("state.json.tmp"), written);
        }
    }

    const ENV: &[u8] = br#"{"format": 2, "sealed": true, "state": {"memory": "x"}}"#;

    /// A new agent's directory under `root`, named by its id.
    fn agent(root: &Path) -> std::path::PathBuf {
        root.join(AgentId::new().to_string())
    }

    #[tokio::test]
    async fn only_a_settled_complete_tmp_of_the_same_save_is_promoted() {
        let root = crate::sealed::TestDir::new("promote");
        let now = SystemTime::now();
        let ago = |s: u64| now - Duration::from_secs(s);
        let [cut_dir, fresh, by_hand, other_save, torn, beside] = [(); 6].map(|()| agent(&root));
        // The cut save: tmp an hour old, archived a moment after it
        cut(&cut_dir, ENV, ago(3600), Some(ago(3599)));
        // A save in flight
        cut(&fresh, ENV, ago(5), Some(ago(4)));
        // No archive: a state.json removed by hand
        cut(&by_hand, ENV, ago(3600), None);
        // An archive from another save, a day before
        cut(&other_save, ENV, ago(3600), Some(ago(90_000)));
        // Torn
        cut(&torn, b"{\"format\": 2, \"sta", ago(3600), Some(ago(3599)));
        // Beside a state.json of its own (not linked): the old order's
        // crash before its first rename, left to agentkit
        cut(&beside, ENV, ago(3600), Some(ago(3599)));
        std::fs::write(beside.join("state.json"), b"{\"format\":1,\"state\":{}}").unwrap();
        assert_eq!(promote_tmp_states(&root, now).await, 1);
        assert_eq!(std::fs::read(cut_dir.join("state.json")).unwrap(), ENV);
        for d in [&fresh, &by_hand, &other_save, &torn] {
            assert!(!d.join("state.json").exists(), "{}", d.display());
            assert!(d.join("state.json.tmp").exists(), "{} kept", d.display());
        }
        assert_eq!(
            std::fs::read(beside.join("state.json")).unwrap(),
            b"{\"format\":1,\"state\":{}}"
        );
        assert!(beside.join("state.json.tmp").exists());
    }

    /// Another runner on the data dir: nothing promoted, whatever is there.
    #[tokio::test]
    async fn nothing_is_promoted_while_another_runner_shares_the_data_dir() {
        let root = crate::sealed::TestDir::new("promote-shared");
        let now = SystemTime::now();
        let ago = |s: u64| now - Duration::from_secs(s);
        let dir = agent(&root.join("state"));
        cut(&dir, ENV, ago(3600), Some(ago(3599)));
        let other = crate::sealed::session::runner_shared(&root).unwrap();
        assert_eq!(promote_if_alone(&root).await, 0);
        assert!(!dir.join("state.json").exists());
        drop(other);
        assert_eq!(promote_if_alone(&root).await, 1);
    }

    /// An agent not chained, cut under the new order: the link made with no
    /// tmp, or with a torn one, changes nothing; a complete settled tmp is
    /// put in place, and the archive keeps the old state.
    #[tokio::test]
    async fn new_order_cuts_of_an_agent_not_chained() {
        let root = crate::sealed::TestDir::new("promote-new");
        let now = SystemTime::now();
        let hour_ago = now - Duration::from_secs(3600);
        let state_dir = root.join("state");
        let mut fs = FsStorage::new(&state_dir);
        let save = async |fs: &mut FsStorage| {
            let id = AgentId::new();
            fs.save_raw(id, serde_json::json!({ "memory": "old" }), None)
                .await
                .unwrap();
            (id, fs.agent_dir(id))
        };
        let (_, no_tmp) = save(&mut fs).await;
        cut_linked(&no_tmp, None, hour_ago);
        let (_, partial) = save(&mut fs).await;
        cut_linked(&partial, Some(b"{\"format\": 2, \"sta"), hour_ago);
        let (_, fresh) = save(&mut fs).await;
        cut_linked(&fresh, Some(ENV), now - Duration::from_secs(5));
        let (complete_id, complete) = save(&mut fs).await;
        cut_linked(&complete, Some(ENV), hour_ago);
        let old = std::fs::read(complete.join("state.json")).unwrap();
        // Someone else holds the agent: left alone this start
        let held = crate::sealed::session::try_session(&state_dir, complete_id)
            .unwrap()
            .unwrap();
        assert_eq!(promote_tmp_states(&state_dir, now).await, 0);
        drop(held);

        assert_eq!(promote_tmp_states(&state_dir, now).await, 1);
        assert_eq!(std::fs::read(complete.join("state.json")).unwrap(), ENV);
        assert_eq!(
            std::fs::read(complete.join(format!("state.{}.json", stamp(hour_ago)))).unwrap(),
            old,
            "the archive keeps the old state"
        );
        assert!(!complete.join("state.json.tmp").exists());
        for d in [&no_tmp, &partial, &fresh] {
            assert!(
                std::fs::read(d.join("state.json"))
                    .unwrap()
                    .windows(3)
                    .any(|w| w == b"old"),
                "{} unchanged",
                d.display()
            );
        }
        assert!(partial.join("state.json.tmp").exists());
        assert!(fresh.join("state.json.tmp").exists());
        // The next save sets the torn tmp aside, and loads as the old state
        assert_eq!(
            fs.load_raw(
                partial
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .parse::<uuid::Uuid>()
                    .map(AgentId::from)
                    .unwrap()
            )
            .await
            .unwrap()
            .value,
            serde_json::json!({ "memory": "old" })
        );
    }

    /// A chained agent cut under the new order: the complete tmp is put in
    /// place only when a sealed version prepared after its head records the
    /// tmp's state; then the next load commits that version.
    #[tokio::test]
    async fn new_order_cut_of_a_chained_agent_needs_the_sealed_evidence() {
        use agora_agentkit::enums::StateEntryKind;
        use agora_agentkit::state::codec::KeyframePolicy;
        use agora_agentkit::state::storage::{Append, Recovery, plaintext_hash};
        let now = SystemTime::now();
        let hour_ago = now - Duration::from_secs(3600);
        for evidence in [false, true] {
            let root = crate::sealed::TestDir::new("promote-chained");
            let (id, keys, dir) = crate::sealed::check::tests::enrolled_local(&root).await;
            let state_dir = root.join("state");
            // A whole agent state, as a cut save must load as one
            let newer = agora_agentkit::state::storage::as_reloaded(
                &serde_json::to_value(crate::agent_selection_tests::pooled("tarn").1).unwrap(),
            )
            .unwrap();
            let envelope = {
                let scratch = root.join("scratch");
                FsStorage::new(&scratch)
                    .save_sealed_raw(id, newer.clone())
                    .await
                    .unwrap();
                std::fs::read(scratch.join(id.to_string()).join("state.json")).unwrap()
            };
            if evidence {
                // Agentkit prepares the item before its local write
                let seal = AgentSeal::new(&dir, id);
                let lock = seal.lock().await.unwrap();
                let head = seal.head().await.unwrap().unwrap().head;
                let plaintext = agora_agentkit::state::codec::plaintext(&newer);
                seal.prepare(
                    &lock,
                    &keys.seal_keys(id).unwrap(),
                    &KeyframePolicy::default(),
                    Append {
                        kind: StateEntryKind::Snapshot,
                        plaintext: &plaintext,
                        plaintext_sha256: Some(plaintext_hash(&newer)),
                        state_at: chrono::Utc::now(),
                        parent: Some(head),
                    },
                )
                .await
                .unwrap();
            }
            cut_linked(&dir, Some(&envelope), hour_ago);
            let before = std::fs::read(dir.join("state.json")).unwrap();
            let promoted = promote_tmp_states(&state_dir, now).await;
            if !evidence {
                assert_eq!(promoted, 0);
                assert_eq!(std::fs::read(dir.join("state.json")).unwrap(), before);
                assert!(dir.join("state.json.tmp").exists(), "kept");
                continue;
            }
            assert_eq!(promoted, 1);
            assert_eq!(std::fs::read(dir.join("state.json")).unwrap(), envelope);
            let loaded = FsStorage::new(&state_dir).load_raw(id).await.unwrap().value;
            let seal = AgentSeal::new(&dir, id);
            assert!(matches!(
                seal.recover(Some(&plaintext_hash(&loaded))).await.unwrap(),
                Recovery::Committed(_)
            ));
        }
    }

    #[test]
    fn the_newest_archive_counts_collisions() {
        let root = crate::sealed::TestDir::new("promote-newest");
        for name in [
            "state.20261111T120000000Z.json",
            "state.20261111T120000000Z-2.json",
            "state.20261111T120000000Z-1.json",
            "state.20261110T120000000Z.json",
            "state.json.tmp.20261112T120000000Z.aside",
        ] {
            std::fs::write(root.join(name), b"{}").unwrap();
        }
        let (_, path) = newest_archive(&root).unwrap();
        assert!(path.ends_with("state.20261111T120000000Z-2.json"));
    }
}
