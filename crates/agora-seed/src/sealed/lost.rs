//! Versions Agora lost, resolved by `sealed-accept --rebase` (a held
//! LocalMissing): the state here is sealed on top of what Agora still
//! holds, or, when it holds nothing, on a new chain whose genesis records
//! the break. Nothing sealed here is lost doing it.
//!
//! `sealed/lost/<utc>/` keeps what is set aside, outside what agentkit's
//! `prune_blobs` scans (`sealed/blobs/` only), laid out as an agent
//! directory so agentkit's own `AgentSeal::open_state` opens it:
//!
//! - `INTENT`: written before anything moves; renamed `DONE` at the end.
//!   A rerun finds an `INTENT` and finishes the same resolution: every
//!   step below is idempotent.
//! - `sealed/head.json`: the old head;
//! - `sealed/blobs/`: copies of every blob still here that the old head's
//!   keyframe or a set-aside version names (a keyframe is a full copy);
//! - `items/`: the set-aside versions' records. On a new chain the items
//!   themselves move here, so the new numbering starts clean; on top of
//!   Agora's head they stay in `sent/` as `<ord>-<hash>.lost` too, so the
//!   numbering goes on with no gap.
//!
//! **A fork in a chain kept here** (an agent whose chain isn't uploaded,
//! `sealed-accept --keep`, [`begin_fork`] and [`finish_fork`]): two saves
//! from one version, the later recorded in `head.json`'s `displaced`. The
//! operator chooses the line that goes on; every outbox item off it is
//! kept the same way (`INTENT`, copies of the head, blobs and records,
//! then renamed `outbox/<ord>-<hash>.displaced`, as agentkit's
//! `rebase_onto` sets one aside), `head.json` names the kept line's tip
//! with its forks cleared, and, if the kept tip isn't the head, its state
//! becomes the agent's (`state.json` archived by FsStorage first). Nothing
//! is deleted.

use std::path::{Path, PathBuf};

use agora_agentkit::enums::StateEntryKind;
use agora_agentkit::state::codec::KeyframePolicy;
use agora_agentkit::state::storage::{AgentSeal, OutboxItem, SealKeys, SealLock};
use agora_agentkit::state::verify::HeadRef;
use agora_agentkit::state::{GenesisClass, GenesisOrigin, SignedStateEntry};
use anyhow::Context;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

const LOST: &str = "lost";
const INTENT: &str = "INTENT";
const DONE: &str = "DONE";

/// What a resolution will do, written before it starts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Intent {
    pub at: DateTime<Utc>,
    /// Agora's head to seal on top of; `None`: a new chain.
    pub onto: Option<HeadRef>,
    /// The versions set aside: Agora lost them (from `sent/`), or they
    /// wait (the outbox, on a new chain).
    pub items: Vec<OutboxItem>,
    /// The old head's keyframe: a full copy, if its blob is here.
    pub full_copy: Option<DateTime<Utc>>,
    pub note: String,
    /// A fork in a chain kept here, not a loss on Agora: the line kept.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_fork: Option<LocalFork>,
}

/// The line `sealed-accept --keep` keeps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalFork {
    /// Its tip.
    pub keep: HeadRef,
    /// The tip's outbox item, when it isn't the head `head.json` named.
    pub keep_ord: Option<u64>,
}

/// The pending resolution, if one was cut short.
pub fn pending(seal: &AgentSeal) -> anyhow::Result<Option<(PathBuf, Intent)>> {
    let root = seal.dir().join(LOST);
    let Ok(dirs) = std::fs::read_dir(&root) else {
        return Ok(None);
    };
    let mut found = Vec::new();
    for d in dirs {
        let d = d?.path();
        let intent = d.join(INTENT);
        if std::fs::exists(&intent)? {
            let i: Intent = serde_json::from_slice(&std::fs::read(&intent)?)
                .with_context(|| format!("{} unreadable", intent.display()))?;
            found.push((d, i));
        }
    }
    anyhow::ensure!(
        found.len() <= 1,
        "more than one unfinished resolution under {}",
        root.display()
    );
    Ok(found.pop())
}

/// Every resolution under `sealed/lost/`, finished or not, oldest first:
/// its directory and its intent (when it began, what it set aside).
/// Read-only.
pub fn resolutions(seal: &AgentSeal) -> anyhow::Result<Vec<(PathBuf, Intent)>> {
    let root = seal.dir().join(LOST);
    let dirs = match std::fs::read_dir(&root) {
        Ok(dirs) => dirs,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).with_context(|| format!("reading {}", root.display())),
    };
    let mut found = Vec::new();
    for d in dirs {
        let d = d?.path();
        for marker in [DONE, INTENT] {
            let path = d.join(marker);
            match std::fs::read(&path) {
                Ok(bytes) => {
                    let intent: Intent = serde_json::from_slice(&bytes)
                        .with_context(|| format!("{} unreadable", path.display()))?;
                    found.push((d.clone(), intent));
                    break;
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
            }
        }
    }
    found.sort_by_key(|f| f.1.at);
    Ok(found)
}

fn sync(dir: &Path) -> std::io::Result<()> {
    std::fs::File::open(dir)?.sync_all()
}

fn write_new(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let dir = path.parent().expect("in a directory");
    std::fs::create_dir_all(dir)?;
    let tmp = path.with_extension("writing");
    std::fs::write(&tmp, bytes)?;
    std::fs::File::open(&tmp)?.sync_all()?;
    std::fs::rename(&tmp, path)?;
    sync(dir)
}

fn copy_if_missing(from: &Path, to: &Path) -> std::io::Result<bool> {
    if std::fs::exists(to)? || !std::fs::exists(from)? {
        return Ok(false);
    }
    write_new(to, &std::fs::read(from)?)?;
    Ok(true)
}

/// Step 1 of a resolution: copies of the old head, the blobs still here
/// (the items' and the old head's keyframe's) and the items' records,
/// each written once.
async fn copy_aside(seal: &AgentSeal, dir: &Path, intent: &Intent) -> anyhow::Result<()> {
    let kept = dir.join("sealed");
    copy_if_missing(&seal.dir().join("head.json"), &kept.join("head.json"))?;
    let blobs = seal.dir().join("blobs");
    let mut shas: Vec<_> = intent
        .items
        .iter()
        .filter_map(|i| i.entry.entry.blob.as_ref().map(|b| b.sha256))
        .collect();
    if let Ok(Some(h)) = seal.head().await
        && let Some(sha) = h.keyframe.as_ref().and_then(|k| k.blob_sha256())
    {
        shas.push(sha);
    }
    for sha in shas {
        copy_if_missing(
            &blobs.join(sha.to_hex()),
            &kept.join("blobs").join(sha.to_hex()),
        )?;
    }
    let records = dir.join("items");
    for item in &intent.items {
        let name = format!("{:020}-{}.json", item.ord, item.entry.entry_hash());
        if !std::fs::exists(records.join(&name))? {
            write_new(&records.join(&name), &serde_json::to_vec_pretty(item)?)?;
        }
    }
    Ok(())
}

/// Start a resolution: its directory and `INTENT`, nothing moved yet.
pub async fn begin(
    seal: &AgentSeal,
    onto: Option<HeadRef>,
    lost: &[OutboxItem],
    note: &str,
) -> anyhow::Result<(PathBuf, Intent)> {
    let at = Utc::now();
    let dir = seal
        .dir()
        .join(LOST)
        .join(at.format(crate::sealed::STAMP_FORMAT).to_string());
    let mut items = lost.to_vec();
    // On a new chain the waiting versions go too
    if onto.is_none() {
        items.extend(seal.outbox().await?);
    }
    let head = seal.head().await?;
    let full_copy = head
        .as_ref()
        .and_then(|h| h.keyframe.as_ref())
        .filter(|k| {
            k.blob_sha256().is_some_and(|sha| {
                std::fs::exists(seal.dir().join("blobs").join(sha.to_hex())).unwrap_or(false)
            })
        })
        .and_then(|k| DateTime::from_timestamp_micros(k.entry.state_at));
    let intent = Intent {
        at,
        onto,
        items,
        full_copy,
        note: note.to_string(),
        local_fork: None,
    };
    write_new(&dir.join(INTENT), &serde_json::to_vec_pretty(&intent)?)?;
    super::crash::point("lost:intent")?;
    Ok((dir, intent))
}

/// Finish the resolution `intent` describes: each step is skipped once
/// done, so a rerun after a cut finishes it. Returns the new head.
pub async fn finish(
    seal: &AgentSeal,
    lock: &SealLock,
    keys: &SealKeys,
    dir: &Path,
    intent: &Intent,
    current: &serde_json::Value,
    // Agora's head entry, read and verified by the caller just now
    server_head: Option<&SignedStateEntry>,
) -> anyhow::Result<HeadRef> {
    let kept = dir.join("sealed");
    copy_aside(seal, dir, intent).await?;
    super::crash::point("lost:copied")?;
    let records = dir.join("items");
    // 2. Set aside
    let sent = seal.dir().join("sent");
    let outbox = seal.dir().join("outbox");
    for item in &intent.items {
        let plain = format!("{:020}.json", item.ord);
        for from in [sent.join(&plain), outbox.join(&plain)] {
            // That item, not a new one under the same number (a new chain
            // numbers from 0 again, so a rerun finds its own items here)
            let same = std::fs::read(&from)
                .ok()
                .and_then(|b| serde_json::from_slice::<OutboxItem>(&b).ok())
                .is_some_and(|i| i.entry.entry_hash() == item.entry.entry_hash());
            if !same {
                continue;
            }
            match intent.onto {
                // Numbering goes on: kept in sent/ under its number
                Some(_) => std::fs::rename(
                    &from,
                    sent.join(format!("{:020}-{}.lost", item.ord, item.entry.entry_hash())),
                )?,
                // A new chain numbers from 0: moved out
                None => std::fs::rename(
                    &from,
                    records.join(format!("{:020}-{}.json", item.ord, item.entry.entry_hash())),
                )?,
            }
        }
    }
    for d in [&sent, &outbox, &records] {
        if std::fs::exists(d)? {
            sync(d)?;
        }
    }
    super::crash::point("lost:aside")?;
    // 3. Seal
    let head = match intent.onto {
        Some(onto) => {
            // Done when the head is a new keyframe child of Agora's head:
            // never the old head (which may be one too: Agora lost only it)
            let old: Option<agora_agentkit::state::storage::HeadFile> =
                std::fs::read(kept.join("head.json"))
                    .ok()
                    .and_then(|b| serde_json::from_slice(&b).ok());
            let done = seal.head().await?.filter(|h| {
                old.as_ref().is_none_or(|o| o.head != h.head)
                    && h.keyframe.as_ref().is_some_and(|k| {
                        k.entry_hash() == h.head.entry_hash
                            && k.entry.prev_hash == Some(onto.entry_hash)
                    })
            });
            match done {
                Some(h) => h.head,
                None => {
                    let server_head = server_head
                        .filter(|e| e.entry_hash() == onto.entry_hash)
                        .context("Agora's head is no longer the one this resolution began on")?;
                    seal.rebase_onto(lock, keys, &server_head.entry, current)
                        .await
                        .context("sealing the current state on top of Agora's head")?
                }
            }
        }
        None => {
            // The old head set aside (its copy is already kept)
            let head_file = seal.dir().join("head.json");
            // The old head is the one copied aside in step 1; another is the
            // new chain's, from a run cut after its genesis
            let old: Option<agora_agentkit::state::storage::HeadFile> =
                std::fs::read(kept.join("head.json"))
                    .ok()
                    .and_then(|b| serde_json::from_slice(&b).ok());
            let fresh = match seal.head().await? {
                Some(h) if old.as_ref().is_none_or(|o| o.head != h.head) => Some(h),
                Some(_) => {
                    std::fs::rename(&head_file, kept.join("head.json"))?;
                    sync(seal.dir())?;
                    None
                }
                None => None,
            };
            super::crash::point("lost:head_aside")?;
            let mut head = match fresh {
                Some(h) => h.head,
                None => {
                    let origin = GenesisOrigin {
                        class: GenesisClass::KnownBreak,
                        incident: None,
                        prior_copy_held: true,
                        verified_at: None,
                    };
                    super::session::genesis(seal, lock, keys, &origin, Utc::now())
                        .await
                        .context("sealing a new genesis")?
                }
            };
            super::crash::point("lost:genesis")?;
            if head.seq == 0 {
                head = super::session::append(
                    seal,
                    lock,
                    keys,
                    &KeyframePolicy::default(),
                    StateEntryKind::Snapshot,
                    current,
                    Utc::now(),
                )
                .await
                .context("sealing the current state on the new genesis")?;
            }
            head
        }
    };
    super::crash::point("lost:sealed")?;
    // 4. Done
    std::fs::rename(dir.join(INTENT), dir.join(DONE))?;
    sync(dir)?;
    Ok(head)
}

/// Start a local fork's resolution: `INTENT` naming the line kept and the
/// items set aside, nothing moved yet.
pub async fn begin_fork(
    seal: &AgentSeal,
    keep: LocalFork,
    aside: &[OutboxItem],
    note: &str,
) -> anyhow::Result<(PathBuf, Intent)> {
    let at = Utc::now();
    let dir = seal
        .dir()
        .join(LOST)
        .join(at.format(crate::sealed::STAMP_FORMAT).to_string());
    let head = seal.head().await?;
    let full_copy = head
        .as_ref()
        .and_then(|h| h.keyframe.as_ref())
        .filter(|k| {
            k.blob_sha256().is_some_and(|sha| {
                std::fs::exists(seal.dir().join("blobs").join(sha.to_hex())).unwrap_or(false)
            })
        })
        .and_then(|k| DateTime::from_timestamp_micros(k.entry.state_at));
    let intent = Intent {
        at,
        onto: None,
        items: aside.to_vec(),
        full_copy,
        note: note.to_string(),
        local_fork: Some(keep),
    };
    write_new(&dir.join(INTENT), &serde_json::to_vec_pretty(&intent)?)?;
    super::crash::point("fork:intent")?;
    Ok((dir, intent))
}

/// Steps 1 to 3 of the local fork resolution `intent` describes, each
/// skipped once done, so a rerun after a cut finishes it: copies, the
/// items off the kept line set aside, `head.json` naming the kept tip.
/// `tip`: the kept tip's outbox item, when it isn't the head `head.json`
/// named (`plaintext_sha256`: its line's last snapshot's). The caller
/// then writes the tip's state, the notice, and calls [`done`].
pub async fn finish_fork(
    seal: &AgentSeal,
    dir: &Path,
    intent: &Intent,
    tip: Option<&OutboxItem>,
    plaintext_sha256: Option<agora_agentkit::govlog::Sha256Hex>,
) -> anyhow::Result<()> {
    let keep = intent
        .local_fork
        .as_ref()
        .context("not a local fork's resolution")?;
    // 1. Copies
    copy_aside(seal, dir, intent).await?;
    super::crash::point("fork:copied")?;
    // 2. Set aside, as agentkit's rebase_onto does: `.displaced`
    let outbox = seal.dir().join("outbox");
    for item in &intent.items {
        let from = outbox.join(format!("{:020}.json", item.ord));
        let same = std::fs::read(&from)
            .ok()
            .and_then(|b| serde_json::from_slice::<OutboxItem>(&b).ok())
            .is_some_and(|i| i.entry.entry_hash() == item.entry.entry_hash());
        if same {
            std::fs::rename(
                &from,
                outbox.join(format!(
                    "{:020}-{}.displaced",
                    item.ord,
                    item.entry.entry_hash()
                )),
            )?;
        }
    }
    if std::fs::exists(&outbox)? {
        sync(&outbox)?;
    }
    super::crash::point("fork:aside")?;
    // 3. head.json: the kept tip, its forks resolved
    let old = seal.head().await?.context("head.json is missing")?;
    if old.head != keep.keep || !old.displaced.is_empty() {
        let file = match tip {
            None => agora_agentkit::state::storage::HeadFile {
                displaced: Vec::new(),
                ..old
            },
            Some(tip) => agora_agentkit::state::storage::HeadFile {
                head: keep.keep,
                plaintext_sha256,
                keyframe: tip.keyframe_after.clone(),
                next_ord: old.next_ord,
                displaced: Vec::new(),
                created_at: tip.entry.entry.created_at,
            },
        };
        super::write_private(
            &seal.dir().join("head.json"),
            &serde_json::to_vec_pretty(&file)?,
        )?;
    }
    super::crash::point("fork:head")?;
    Ok(())
}

/// A resolution finished: `INTENT` renamed `DONE`.
pub fn done(dir: &Path) -> anyhow::Result<()> {
    std::fs::rename(dir.join(INTENT), dir.join(DONE))?;
    sync(dir)?;
    Ok(())
}

/// The entries a resolution's record (`INTENT` or `DONE` in `dir`)
/// holds, by file. Read-only.
pub fn record_entries(dir: &Path) -> anyhow::Result<Vec<(PathBuf, Vec<SignedStateEntry>)>> {
    let mut out = Vec::new();
    for marker in [INTENT, DONE] {
        let path = dir.join(marker);
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        let intent: Intent = serde_json::from_slice(&bytes)
            .with_context(|| format!("{} unreadable", path.display()))?;
        out.push((path, intent.items.into_iter().map(|i| i.entry).collect()));
    }
    Ok(out)
}

/// Re-sign the entries a resolution's record (`INTENT` or `DONE` in
/// `dir`) holds, with `resign` (`rotate-keys`, for an agent that doesn't
/// upload): the record rewritten whole if any changed. Returns how many.
pub fn resign_record(
    dir: &Path,
    mut resign: impl FnMut(&Path, &mut SignedStateEntry) -> anyhow::Result<bool>,
) -> anyhow::Result<usize> {
    let mut count = 0;
    for marker in [INTENT, DONE] {
        let path = dir.join(marker);
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        let mut intent: Intent = serde_json::from_slice(&bytes)
            .with_context(|| format!("{} unreadable", path.display()))?;
        let mut changed = 0;
        for item in &mut intent.items {
            if resign(&path, &mut item.entry)? {
                changed += 1;
            }
        }
        if changed > 0 {
            super::write_private(&path, &serde_json::to_vec_pretty(&intent)?)
                .with_context(|| format!("rewriting {}", path.display()))?;
            count += changed;
        }
    }
    Ok(count)
}

/// What the agent is told is kept.
pub fn kept_text(intent: &Intent) -> String {
    let n = intent.items.len();
    let records = format!(
        "the records of {n} version{}",
        if n == 1 { "" } else { "s" }
    );
    match intent.full_copy {
        Some(at) => format!(
            "the full copy of your state from {} and {records} after it",
            at.format("%-d %B %Y")
        ),
        None => records,
    }
}
