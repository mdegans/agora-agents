//! `agora-seed sealed-enroll --agent <id> | --opted-in [--dry-run]`: the
//! backfill that starts an agent's sealed chain. In order, with no upload
//! until the last step:
//!
//! 1. the state key: generated (or the cached one reused), cached in
//!    `<secrets>/<id>/state_key.json` **before** its wrap is posted, wrapped
//!    to the agent's X25519 key, signed, posted ([`StateKeyRecord`]);
//! 2. the genesis, its origin from the memory-restore config and ledger
//!    ([`origin`]): fields only, never the recovered text;
//! 3. the pre-loss copy as `archived_prior`, for an agent that chose to see
//!    it;
//! 4. the archives (`state.<utc>.json`), oldest first by mtime, each
//!    `state_at` its mtime (when that state was written);
//! 5. the current state;
//! 6. `sealed/enrolled`: from here every save is sealed; then one upload.
//!
//! Only agents whose sealed-state ledger says yes, with no session in
//! progress, under the agent's `sealed/lock` for the whole run. A partial
//! enrolment (a chain but no `enrolled`) is resumed: the archives after the
//! newest state it sealed, then the current state, then `enrolled`, with
//! the same key. With no chain here, an agent Agora holds a chain of is
//! never given a second genesis. `--dry-run` reads sizes and markers and
//! prints counts and sizes only: no writes, no network.
//!
//! [`StateKeyRecord`]: agora_agentkit::state::StateKeyRecord

use std::path::{Path, PathBuf};

use agora_agentkit::enums::StateEntryKind;
use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::seed::ShortString;
use agora_agentkit::reactor::{FsStorage, Storage};
use agora_agentkit::state::codec::KeyframePolicy;
use agora_agentkit::state::crypto::StateKey;
use agora_agentkit::state::storage::{AgentSeal, plaintext_hash};
use agora_agentkit::state::{GenesisClass, GenesisOrigin, IncidentCite};
use anyhow::Context;
use chrono::{DateTime, TimeZone, Utc};
use serde::Deserialize;

use super::KeyStore;
use super::keys::{StateKeyFile, x25519_public_hex};
use crate::consent::memory_restore::{
    self, RestoreOffer, ledger::RestoreLedger, prompt::RestoreChoice,
};
use crate::consent::sealed_state::ledger::SealLedger;

/// When the copy recovered from the May loss stood.
pub fn prior_copy_at() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 5, 3, 0, 0, 0).unwrap()
}

/// What enrolment reads around an agent.
pub struct Enroller {
    pub state_dir: PathBuf,
    pub keys: KeyStore,
    pub api: std::sync::Arc<dyn super::api::SealedApi>,
    /// `[memory_restore]`, resolved, when the run config has one.
    pub restore: Option<RestoreOffer>,
}

/// One archive: `state.<utc>.json`.
#[derive(Debug, Clone)]
pub struct Archive {
    pub path: PathBuf,
    pub mtime: DateTime<Utc>,
    pub bytes: u64,
}

/// The archives in `agent_dir`, oldest first by mtime (then name).
pub fn archives(agent_dir: &Path) -> std::io::Result<Vec<Archive>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(agent_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let archived = name.starts_with("state.")
            && name.ends_with(".json")
            && name != "state.json"
            && !name.ends_with(".tmp");
        if !archived {
            continue;
        }
        let meta = entry.metadata()?;
        if !meta.is_file() {
            continue;
        }
        out.push(Archive {
            path: entry.path(),
            mtime: DateTime::<Utc>::from(meta.modified()?),
            bytes: meta.len(),
        });
    }
    out.sort_by(|a, b| a.mtime.cmp(&b.mtime).then_with(|| a.path.cmp(&b.path)));
    Ok(out)
}

/// The genesis origin, and the pre-loss copy to seal (if the agent chose
/// to see it and it is there).
#[derive(Debug, Clone, PartialEq)]
pub struct Origin {
    pub origin: GenesisOrigin,
    pub prior: Option<PathBuf>,
}

/// From the memory-restore listing and ledger, never the recovered text:
/// `verified` if the agent chose `show` and was shown it, `known_break`
/// (citing the thread) if it is listed, else `unknown`.
pub async fn origin(
    restore: Option<&RestoreOffer>,
    name: &ShortString<64>,
    id: AgentId,
    agent_dir: &Path,
) -> anyhow::Result<Origin> {
    let unknown = Origin {
        origin: GenesisOrigin {
            class: GenesisClass::Unknown,
            incident: None,
            prior_copy_held: false,
            verified_at: None,
        },
        prior: None,
    };
    let Some(offer) = restore else {
        return Ok(unknown);
    };
    if offer.listing(name, id) != memory_restore::Listing::Listed {
        return Ok(unknown);
    }
    let ledger = RestoreLedger::load(agent_dir)
        .await
        .context("memory-restore ledger unreadable")?;
    let path = offer.path_for(name);
    let held = std::fs::exists(&path).unwrap_or(false);
    let incident = offer.thread().map(IncidentCite::Post);
    let chose_show = ledger.choice() == Some(RestoreChoice::Show);
    let shown = ledger.showings.first().map(|s| s.at);
    let (class, verified_at) = match (chose_show, shown) {
        (true, Some(at)) => (GenesisClass::Verified, Some(at)),
        _ => (GenesisClass::KnownBreak, None),
    };
    Ok(Origin {
        origin: GenesisOrigin {
            class,
            incident,
            prior_copy_held: held,
            verified_at,
        },
        prior: (chose_show && held).then_some(path),
    })
}

/// What one enrolment did, or would do: counts and sizes only.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    pub archives: usize,
    pub archive_bytes: u64,
    /// Archives left out: unreadable, a newer envelope, or the same state
    /// as the version before.
    pub skipped: usize,
    pub prior: bool,
    pub origin: String,
    pub current_bytes: u64,
    /// A partial enrolment was continued.
    pub resumed: bool,
    /// Archives a partial enrolment had already sealed.
    pub resumed_past: usize,
    /// Entries in the chain at the end (genesis included).
    pub entries: usize,
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "origin {}, archives {} ({} bytes, {} skipped), pre-loss copy {}, current {} bytes, \
             entries {}",
            self.origin,
            self.archives,
            self.archive_bytes,
            self.skipped,
            if self.prior { "yes" } else { "no" },
            self.current_bytes,
            self.entries,
        )
    }
}

/// Why an agent was not enrolled.
#[derive(Debug, thiserror::Error)]
pub enum Skip {
    #[error("already enrolled")]
    Enrolled,
    #[error("its sealed-state ledger does not say yes")]
    NotOptedIn,
    #[error(transparent)]
    Failed(#[from] anyhow::Error),
}

#[derive(Deserialize)]
struct JustTheName {
    soul: NameOnly,
}

#[derive(Deserialize)]
struct NameOnly {
    name: ShortString<64>,
}

/// An archive's state, from its `{format, state}` envelope. `None`: not
/// one this binary reads. Its content is never logged.
fn archived_state(path: &Path) -> Option<serde_json::Value> {
    let bytes = std::fs::read(path).ok()?;
    let mut envelope: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let format = envelope.get("format")?.as_u64()?;
    if format > agora_agentkit::reactor::storage::FORMAT {
        return None;
    }
    Some(envelope.get_mut("state")?.take())
}

/// The sizes of a version just sealed, in the local sidecar.
pub fn record_sizes(
    agent_dir: &Path,
    head: agora_agentkit::state::verify::HeadRef,
    state_at: DateTime<Utc>,
    state: &serde_json::Value,
) {
    let r = super::sizes::Recorded {
        entry_hash: head.entry_hash,
        seq: head.seq,
        state_at,
        sizes: super::sizes::Sizes::of(state),
    };
    if let Err(e) = super::sizes::SizeLog::record(agent_dir, r) {
        tracing::warn!(path = %agent_dir.display(), error = %e, "sealed sizes not recorded");
    }
}

fn mtime(path: &Path) -> std::io::Result<DateTime<Utc>> {
    Ok(DateTime::<Utc>::from(std::fs::metadata(path)?.modified()?))
}

impl Enroller {
    /// The agents whose sealed-state ledger says yes and who aren't
    /// enrolled yet.
    pub fn opted_in(&self) -> anyhow::Result<Vec<AgentId>> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&self.state_dir)
            .with_context(|| format!("reading {}", self.state_dir.display()))?
        {
            let entry = entry?;
            let Some(id) = entry
                .file_name()
                .to_str()
                .and_then(|n| n.parse::<AgentId>().ok())
            else {
                continue;
            };
            let dir = entry.path();
            if super::is_enrolled(&dir) {
                continue;
            }
            match SealLedger::load_blocking(&dir) {
                Ok(l) if l.opted_in() => out.push(id),
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(agent_id = %id, error = %e, "sealed-state ledger unreadable; not enrolled")
                }
            }
        }
        out.sort();
        Ok(out)
    }

    /// Enrol `id`, or with `dry_run` say what would be done.
    pub async fn enroll(&self, id: AgentId, dry_run: bool) -> Result<Report, Skip> {
        let agent_dir = self.state_dir.join(id.to_string());
        let seal = AgentSeal::new(&agent_dir, id);
        if seal
            .is_enrolled()
            .await
            .context("reading the enrolment marker")?
        {
            return Err(Skip::Enrolled);
        }
        let ledger = SealLedger::load(&agent_dir)
            .await
            .context("sealed-state ledger unreadable")?;
        if !ledger.opted_in() {
            return Err(Skip::NotOptedIn);
        }
        let storage = FsStorage::new(&self.state_dir);
        let current = storage
            .load_raw(id)
            .await
            .context("the agent's state won't load")?
            .value;
        let name = serde_json::from_value::<JustTheName>(current.clone())
            .map_err(|e| anyhow::anyhow!("the state names no agent ({:?})", e.classify()))?
            .soul
            .name;
        let origin = origin(self.restore.as_ref(), &name, id, &agent_dir).await?;
        let archives = archives(&agent_dir).context("listing archives")?;
        let current_path = agent_dir.join("state.json");
        let mut report = Report {
            archives: archives.len(),
            archive_bytes: archives.iter().map(|a| a.bytes).sum(),
            prior: origin.prior.is_some(),
            origin: format!("{:?}", origin.origin.class).to_lowercase(),
            current_bytes: std::fs::metadata(&current_path)
                .map(|m| m.len())
                .unwrap_or(0),
            ..Report::default()
        };
        if dry_run {
            super::session::no_session(&self.state_dir, id, false)?;
            return Ok(report);
        }
        // No session in progress, and the agent's lock for the whole
        // enrolment: every write below goes through it
        let lock = super::session::take(&self.state_dir, id, false).await?;
        // A chain begun before but never enrolled is resumed
        let partial = seal.head().await.context("reading the sealed head")?;

        let signing = self
            .keys
            .signing(id)
            .context("the agent has no signing key")?;
        // Never a second genesis for an agent Agora holds a chain of, and
        // no key posted for one either
        if partial.is_none() {
            let server = self
                .api
                .chain(id, &signing)
                .await
                .map_err(anyhow::Error::msg)
                .context("reading the agent's chain on Agora")?;
            if !server.is_empty() {
                return Err(Skip::Failed(anyhow::anyhow!(
                    "Agora holds a chain of {} entries for this agent and there is none here: \
                     adopt it (sealed-accept), never start a second genesis",
                    server.len()
                )));
            }
        }
        let encryption = self
            .keys
            .ring()
            .ensure_encryption_key(id)
            .context("the agent's encryption key")?;
        let state_file = match self
            .keys
            .state_key_file(id)
            .context("the state key cache")?
        {
            Some(file) => file,
            None => {
                let file = StateKeyFile::new(&StateKey::generate(), 0, 0);
                // Cached before the server ever sees a wrap of it
                self.keys
                    .write_state_key(id, &file)
                    .context("caching the state key")?;
                file
            }
        };
        let state_key = state_file.state_key().context("the state key cache")?;
        let held = self
            .api
            .state_keys(id, &signing)
            .await
            .map_err(anyhow::Error::msg)
            .context("reading the server's state keys")?;
        let vk = signing.verifying_key();
        // A record of this key wrapped to the current X25519 key, signed
        // by the agent: an earlier attempt's. One wrapped to a retired key
        // doesn't count: the agent must be able to open it now.
        let posted = held.iter().any(|r| {
            r.key_gen == state_file.key_gen
                && r.wrapped_to_pub == x25519_public_hex(&encryption)
                && r.verify(&vk)
                && StateKey::unwrap(r, &encryption).is_ok_and(|k| k.expose() == state_key.expose())
        });
        if !posted {
            let record = state_key
                .wrap(
                    id,
                    state_file.key_gen,
                    state_file.wrap_seq,
                    &agora_agentkit::envelope::EncryptionPublicKey::from(&encryption),
                    &signing,
                )
                .context("wrapping the state key")?;
            self.api
                .post_state_key(id, &signing, &record)
                .await
                .map_err(anyhow::Error::msg)
                .context("posting the state key record")?;
        }
        let keys = self
            .keys
            .seal_keys(id)
            .context("sealing keys for the agent")?;
        let policy = KeyframePolicy::default();

        // Where a partial enrolment stopped: the newest state it sealed,
        // and whether the pre-loss copy is in
        let mut done_until: Option<i64> = None;
        let mut prior_done = false;
        let mut last = None;
        match &partial {
            Some(head) => {
                let entries = super::check::local_entries(&seal)
                    .await
                    .map_err(anyhow::Error::msg)?;
                done_until = entries
                    .iter()
                    .filter(|e| e.entry.kind == StateEntryKind::Snapshot)
                    .map(|e| e.entry.state_at)
                    .max();
                prior_done = entries
                    .iter()
                    .any(|e| e.entry.kind == StateEntryKind::ArchivedPrior);
                last = head.plaintext_sha256;
                report.resumed = true;
                tracing::warn!(
                    event_type = "sealed_enroll_resumed",
                    agent_id = %id,
                    entries = entries.len(),
                    "resuming a partial enrolment"
                );
            }
            None => {
                let first_at = archives
                    .first()
                    .map(|a| a.mtime)
                    .or_else(|| mtime(&current_path).ok())
                    .unwrap_or_else(Utc::now);
                super::session::genesis(&seal, &lock, &keys, &origin.origin, first_at)
                    .await
                    .context("sealing the genesis")?;
            }
        }
        if let Some(prior) = origin.prior.as_ref().filter(|_| !prior_done) {
            let value: serde_json::Value =
                serde_json::from_slice(&std::fs::read(prior).context("reading the pre-loss copy")?)
                    .map_err(|e| {
                        anyhow::anyhow!("the pre-loss copy isn't JSON ({:?})", e.classify())
                    })?;
            super::session::append(
                &seal,
                &lock,
                &keys,
                &policy,
                StateEntryKind::ArchivedPrior,
                &value,
                prior_copy_at(),
            )
            .await
            .context("sealing the pre-loss copy")?;
        }
        for archive in &archives {
            if done_until.is_some_and(|t| archive.mtime.timestamp_micros() <= t) {
                report.resumed_past += 1;
                continue;
            }
            let Some(state) = archived_state(&archive.path) else {
                report.skipped += 1;
                continue;
            };
            let hash = plaintext_hash(&state);
            if last == Some(hash) {
                report.skipped += 1;
                continue;
            }
            let head = super::session::append(
                &seal,
                &lock,
                &keys,
                &policy,
                StateEntryKind::Snapshot,
                &state,
                archive.mtime,
            )
            .await
            .with_context(|| format!("sealing archive {}", archive.path.display()))?;
            record_sizes(&agent_dir, head, archive.mtime, &state);
            last = Some(hash);
        }
        if last != Some(plaintext_hash(&current)) {
            let at = mtime(&current_path).unwrap_or_else(|_| Utc::now());
            let head = super::session::append(
                &seal,
                &lock,
                &keys,
                &policy,
                StateEntryKind::Snapshot,
                &current,
                at,
            )
            .await
            .context("sealing the current state")?;
            record_sizes(&agent_dir, head, at, &current);
        }
        // The last step is agentkit's, which takes the lock itself: the
        // state rewritten as format 2 (a runner older than 0.147 then
        // refuses it rather than overwriting it), resealed if the rewrite
        // changed its hash, then `enrolled`
        drop(lock);
        let mut mirrored = agora_agentkit::state::storage::Mirrored::new(
            FsStorage::new(&self.state_dir),
            (),
            std::sync::Arc::new(super::SealedEnrolment::with_keys(
                &self.state_dir,
                self.keys.clone(),
            )),
        );
        let head = mirrored
            .enroll(id)
            .await
            .context("marking the state and writing the enrolment marker")?;
        if let Ok(marked) = FsStorage::new(&self.state_dir).load_raw(id).await {
            record_sizes(&agent_dir, head, Utc::now(), &marked.value);
        }
        report.entries = super::check::local_entries(&seal)
            .await
            .map_err(anyhow::Error::msg)?
            .len();
        tracing::info!(
            event_type = "sealed_enrolled",
            agent_id = %id,
            archives = report.archives,
            archive_bytes = report.archive_bytes,
            skipped = report.skipped,
            prior = report.prior,
            origin = %report.origin,
            entries = report.entries,
            "agent enrolled in sealed state; the caller uploads"
        );
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consent::sealed_state::ledger::{SealAsk, SealOutcome};
    use crate::consent::sealed_state::prompt::SealChoice;
    use crate::sealed::api::fake::FakeApi;
    use agora_agentkit::state::verify::verify_chain;

    fn yes(dir: &Path) {
        let mut l = SealLedger::default();
        l.record(SealAsk {
            at: Utc::now(),
            offer_version: 1,
            model: "m".into(),
            order: SealChoice::ALL,
            order_seed: 0,
            constrained: true,
            attempts: 1,
            outcome: SealOutcome::Answered {
                choice: SealChoice::Private,
                reason: String::new(),
                memory_note_written: false,
                note_dropped: None,
                apply_failed: None,
            },
        });
        std::fs::write(SealLedger::path(dir), serde_json::to_vec(&l).unwrap()).unwrap();
    }

    fn state(name: &str, memory: &str) -> serde_json::Value {
        serde_json::json!({ "soul": { "name": name }, "memory": { "content": memory } })
    }

    #[tokio::test]
    async fn enrolment_seals_archives_oldest_first_then_the_current_state() {
        let api = std::sync::Arc::new(FakeApi::new());
        let root = crate::sealed::TestDir::new("enroll");
        let state_dir = root.join("state");
        let keys = KeyStore::new(root.join("secrets"));
        let id = AgentId::new();
        keys.ring()
            .insert(id, &agora_agentkit::crypto::generate_keypair().0)
            .unwrap();
        let mut fs = FsStorage::new(&state_dir);
        for m in ["one", "two", "two", "three"] {
            fs.save_raw(id, state("tarn", m), None).await.unwrap();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let dir = state_dir.join(id.to_string());
        std::fs::write(dir.join("state.20260101T000000000Z.json"), b"not json").unwrap();
        let enroller = Enroller {
            state_dir: state_dir.clone(),
            keys: keys.clone(),
            api: api.clone(),
            restore: None,
        };

        // Not opted in: nothing
        assert!(matches!(
            enroller.enroll(id, false).await,
            Err(Skip::NotOptedIn)
        ));
        yes(&dir);
        assert_eq!(enroller.opted_in().unwrap(), vec![id]);

        // Dry run: counts only, nothing written
        let dry = enroller.enroll(id, true).await.unwrap();
        assert_eq!((dry.archives, dry.entries), (4, 0));
        assert!(!std::fs::exists(dir.join(agora_agentkit::state::storage::SEALED_DIR)).unwrap());
        assert!(keys.state_key_file(id).unwrap().is_none());

        let report = enroller.enroll(id, false).await.unwrap();
        assert_eq!(
            *api.calls.lock().unwrap(),
            ["chain", "state_keys", "post_state_key"]
        );
        let record = api.keys.lock().unwrap()[0].clone();
        let enc = keys.encryption(id).unwrap();
        assert_eq!(
            StateKey::unwrap(&record, &enc).unwrap().expose(),
            keys.state_key_file(id)
                .unwrap()
                .unwrap()
                .state_key()
                .unwrap()
                .expose(),
            "the posted wrap opens to the cached key"
        );
        assert_eq!(report.origin, "unknown");
        // one, two, (two again: skipped), the bad file skipped; three is current
        assert_eq!(report.skipped, 2, "{report}");
        assert_eq!(report.entries, 4, "genesis, one, two, three: {report}");
        let seal = AgentSeal::new(&dir, id);
        assert!(seal.is_enrolled().await.unwrap());
        assert!(keys.state_key_file(id).unwrap().is_some());
        // Marked: a pre-0.147 runner refuses it rather than overwrite it
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("state.json")).unwrap()).unwrap();
        assert_eq!(
            raw["format"],
            agora_agentkit::reactor::storage::SEALED_FORMAT
        );
        let entries = super::super::check::local_entries(&seal).await.unwrap();
        let trusted = keys.trusted(id).unwrap().unwrap();
        assert!(verify_chain(id, &entries, &trusted).intact);
        let k = keys.seal_keys(id).unwrap();
        let opened = seal.open_state(&k, &entries[1].entry, None).await.unwrap();
        assert_eq!(opened, state("tarn", "one"), "oldest first");
        // The current state is the head: the start check finds it intact
        let local = plaintext_hash(&fs.load_raw(id).await.unwrap().value);
        assert!(matches!(
            seal.recover(Some(&local)).await.unwrap(),
            agora_agentkit::state::storage::Recovery::Intact(_)
        ));
        assert!(matches!(
            enroller.enroll(id, false).await,
            Err(Skip::Enrolled)
        ));
        assert!(enroller.opted_in().unwrap().is_empty());
    }

    /// The origin from the memory-restore listing and ledger: fields only.
    #[tokio::test]
    async fn the_origin_follows_the_memory_restore_record() {
        let root = crate::sealed::TestDir::new("enroll-origin");
        let id = AgentId::new();
        let agent_dir = root.join("state").join(id.to_string());
        std::fs::create_dir_all(&agent_dir).unwrap();
        let recovered = root.join("recovered");
        std::fs::create_dir_all(recovered.join("tarn")).unwrap();
        std::fs::write(
            recovered.join("tarn/MEMORY.json"),
            r#"{"content":"synthetic"}"#,
        )
        .unwrap();
        let thread: agora_agentkit::ids::PostId =
            "7ad26ccd-0000-4000-8000-000000000001".parse().unwrap();
        let offer = RestoreOffer::for_agents(&[("tarn", None, id)], &recovered, Some(thread));
        let name = ShortString::new("tarn").unwrap();

        let none = origin(None, &name, id, &agent_dir).await.unwrap();
        assert_eq!(none.origin.class, GenesisClass::Unknown);
        let other = origin(Some(&offer), &name, AgentId::new(), &agent_dir)
            .await
            .unwrap();
        assert_eq!(
            other.origin.class,
            GenesisClass::Unknown,
            "listed under another id"
        );

        let listed = origin(Some(&offer), &name, id, &agent_dir).await.unwrap();
        assert_eq!(listed.origin.class, GenesisClass::KnownBreak);
        assert_eq!(listed.origin.incident, Some(IncidentCite::Post(thread)));
        assert!(listed.origin.prior_copy_held);
        assert!(listed.prior.is_none(), "not chosen: not sealed");

        let ledger = serde_json::json!({
            "format": 1,
            "asks": [{"at": "2026-10-12T09:00:00Z", "offer_version": 1, "model": "m",
                "chars": 9, "order": ["show", "decline"], "order_seed": 0,
                "constrained": true, "attempts": 1,
                "outcome": {"kind": "answered", "choice": "show", "reason": "",
                    "memory_note_written": false}}],
            "showings": [{"at": "2026-10-13T09:00:00Z", "outcome": "completed"}],
        });
        std::fs::write(RestoreLedger::path(&agent_dir), ledger.to_string()).unwrap();
        let shown = origin(Some(&offer), &name, id, &agent_dir).await.unwrap();
        assert_eq!(shown.origin.class, GenesisClass::Verified);
        assert_eq!(
            shown.origin.verified_at,
            Some("2026-10-13T09:00:00Z".parse().unwrap())
        );
        assert_eq!(shown.prior, Some(recovered.join("tarn/MEMORY.json")));
    }

    /// An enrolment interrupted after `k` archives resumes where it
    /// stopped: every state once, in order, no gap, one genesis.
    #[tokio::test]
    async fn an_interrupted_enrolment_resumes_without_duplicates_or_gaps() {
        for k in 0..=4 {
            let root = crate::sealed::TestDir::new("enroll-resume");
            let state_dir = root.join("state");
            let keys = KeyStore::new(root.join("secrets"));
            let id = AgentId::new();
            keys.ring()
                .insert(id, &agora_agentkit::crypto::generate_keypair().0)
                .unwrap();
            let mut fs = FsStorage::new(&state_dir);
            let memories = ["a", "b", "c", "d", "e"];
            for m in memories {
                fs.save_raw(id, state("tarn", m), None).await.unwrap();
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            let dir = state_dir.join(id.to_string());
            yes(&dir);
            // The first run: key, genesis, k archives, then it died
            keys.write_state_key(id, &StateKeyFile::new(&StateKey::generate(), 0, 0))
                .unwrap();
            let seal = AgentSeal::new(&dir, id);
            let k_keys = keys.seal_keys(id).unwrap();
            let all = archives(&dir).unwrap();
            {
                let lock = seal.lock().await.unwrap();
                let origin = GenesisOrigin {
                    class: GenesisClass::Unknown,
                    incident: None,
                    prior_copy_held: false,
                    verified_at: None,
                };
                crate::sealed::session::genesis(&seal, &lock, &k_keys, &origin, all[0].mtime)
                    .await
                    .unwrap();
                for a in &all[..k] {
                    crate::sealed::session::append(
                        &seal,
                        &lock,
                        &k_keys,
                        &KeyframePolicy::default(),
                        StateEntryKind::Snapshot,
                        &archived_state(&a.path).unwrap(),
                        a.mtime,
                    )
                    .await
                    .unwrap();
                }
            }
            let api = std::sync::Arc::new(FakeApi::new());
            let enroller = Enroller {
                state_dir: state_dir.clone(),
                keys: keys.clone(),
                api: api.clone(),
                restore: None,
            };
            let report = enroller.enroll(id, false).await.unwrap();
            assert!(report.resumed, "k={k}");
            assert_eq!(report.resumed_past, k, "k={k}");
            assert!(!api.calls.lock().unwrap().contains(&"chain".to_string()));
            let entries = super::super::check::local_entries(&seal).await.unwrap();
            assert_eq!(
                entries.len(),
                1 + memories.len(),
                "k={k}: genesis and each state once"
            );
            assert!(verify_chain(id, &entries, &keys.trusted(id).unwrap().unwrap()).intact);
            let mut opened = Vec::new();
            for e in &entries[1..] {
                let base = match e.entry.blob.as_ref().unwrap().encoding {
                    agora_agentkit::state::BlobEncoding::Delta { base } => entries
                        .iter()
                        .map(|x| &x.entry)
                        .find(|x| x.entry_hash() == base),
                    _ => None,
                };
                let v = seal.open_state(&k_keys, &e.entry, base).await.unwrap();
                opened.push(v["memory"]["content"].as_str().unwrap().to_string());
            }
            assert_eq!(
                opened, memories,
                "k={k}: in order, none twice, none missing"
            );
        }
    }

    /// No chain here, but one on Agora: never a second genesis.
    #[tokio::test]
    async fn no_second_genesis_for_a_chain_on_agora() {
        let root = crate::sealed::TestDir::new("enroll-server-chain");
        let (other, _, other_dir) = crate::sealed::check::tests::enrolled(&root).await;
        let chain = super::super::check::local_entries(&AgentSeal::new(&other_dir, other))
            .await
            .unwrap();
        let state_dir = root.join("state");
        let keys = KeyStore::new(root.join("secrets"));
        let id = AgentId::new();
        keys.ring()
            .insert(id, &agora_agentkit::crypto::generate_keypair().0)
            .unwrap();
        FsStorage::new(&state_dir)
            .save_raw(id, state("tarn", "m"), None)
            .await
            .unwrap();
        yes(&state_dir.join(id.to_string()));
        let api = std::sync::Arc::new(FakeApi::new());
        *api.chain.lock().unwrap() = Ok(chain);
        let enroller = Enroller {
            state_dir,
            keys: keys.clone(),
            api: api.clone(),
            restore: None,
        };
        let e = enroller.enroll(id, false).await.unwrap_err().to_string();
        assert!(e.contains("never start a second genesis"), "{e}");
        assert_eq!(*api.calls.lock().unwrap(), ["chain"], "no key posted");
        assert!(keys.state_key_file(id).unwrap().is_none());
    }
}
