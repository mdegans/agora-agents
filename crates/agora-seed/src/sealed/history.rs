//! The `memory` tool's versions, for one agent: every version of its state
//! the runner can show, from wherever it is held, each with its memory
//! text and, if it was sealed, its head.
//!
//! | Source | Held in | Shown when |
//! |---|---|---|
//! | [`Source::Sealed`], [`Source::ArchivedPrior`] | the main line to `sealed/head.json`'s head | the line verifies ([`super::version::verified_line`]: through Agora's entries where it runs through them) |
//! | [`Source::SideLine`] | `sealed/outbox/*.displaced`, and entries held here off the main line | its own line verifies from the genesis (`verify_chain`); else left out, with a structured warning |
//! | [`Source::SetAside`] | `sealed/lost/<utc>/` (and `sealed/sent/*.lost`) | the same |
//! | [`Source::LocalCopy`] | the archives `state.<utc>.json` | always: not sealed, so no head |
//! | [`Source::Current`] | `state.json` | always |
//! | [`Source::NeverSaved`] | `unsaved.<utc>.json` rescues | always: never saved, never signed |
//!
//! **One version, many sources.** A version is one state: sources holding
//! the same state (the same `plaintext_hash`, the hash of its canonical
//! sealed plaintext) are one [`Version`] listing them all, sealed first.
//! Enrolment backfilled every archive into the chain, so for an enrolled
//! agent most archives are the same version as a sealed one; a fork's
//! displaced version and the rebased head over it usually are too. The
//! content shown is the sealed one's, opened from its verified blob.
//! Versions with the same memory but different states stay apart: the
//! renderer marks them ("same memory as above").
//!
//! **Any head the agent was shown** is answered ([`History::find_head`]):
//! with the version that holds it (a side line's or a set-aside one's
//! source says so), or with why it is held but not shown ([`LeftOut`]),
//! never "not found" for a head this runner holds.
//!
//! **Read-only.** Nothing here writes: not the state, the outbox, the
//! receipts or the inclusion record. Blobs no longer held here are fetched
//! from Agora and checked against their entries ([`super::version::Opener`]),
//! and nothing fetched is written anywhere. Content is never logged:
//! warnings carry hashes, seqs, file names and counts.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use agora_agentkit::enums::StateEntryKind;
use agora_agentkit::govlog::Sha256Hex;
use agora_agentkit::ids::AgentId;
use agora_agentkit::state::SignedStateEntry;
use agora_agentkit::state::storage::{AgentSeal, OutboxItem, plaintext_hash};
use agora_agentkit::state::verify::{HeadRef, TrustedKeys, verify_chain};
use chrono::{DateTime, Utc};
use zeroize::Zeroizing;

use super::KeyStore;
use super::api::SealedApi;
use super::check::{local_entries, merge_server, walk_line};
use super::version::{Opener, verified_line};

/// Which line a sealed version is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Line {
    /// The main line to `sealed/head.json`'s head.
    Main,
    /// Off it: displaced by a fork, or set aside by a rebase.
    Side,
}

/// Where one version was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// A snapshot on the main line, verified.
    Sealed { head: HeadRef },
    /// The pre-loss copy, sealed on the main line (`archived_prior`).
    ArchivedPrior { head: HeadRef },
    /// Sealed on a side line the operator rebased away from (a fork's
    /// displaced entry), verified from the genesis.
    SideLine { head: HeadRef },
    /// Set aside on `on`, when the agent's line was rebased
    /// (`sealed/lost/<utc>/`), verified from the genesis.
    SetAside { head: HeadRef, on: DateTime<Utc> },
    /// A local archive, `state.<utc>.json`: the runner's own copy, not
    /// sealed, so no head.
    LocalCopy { file: String },
    /// `state.json`.
    Current,
    /// `unsaved.<utc>.json`: the run ended before saving it. Unsigned.
    NeverSaved { file: String },
}

impl Source {
    /// Its sealed head, if it has one.
    pub fn head(&self) -> Option<HeadRef> {
        match self {
            Self::Sealed { head }
            | Self::ArchivedPrior { head }
            | Self::SideLine { head }
            | Self::SetAside { head, .. } => Some(*head),
            Self::LocalCopy { .. } | Self::Current | Self::NeverSaved { .. } => None,
        }
    }

    /// Its line, if it is sealed.
    pub fn line(&self) -> Option<Line> {
        match self {
            Self::Sealed { .. } | Self::ArchivedPrior { .. } => Some(Line::Main),
            Self::SideLine { .. } | Self::SetAside { .. } => Some(Line::Side),
            Self::LocalCopy { .. } | Self::Current | Self::NeverSaved { .. } => None,
        }
    }

    /// The order sources are listed in: sealed on the main line first.
    fn rank(&self) -> u8 {
        match self {
            Self::Sealed { .. } | Self::ArchivedPrior { .. } => 0,
            Self::SideLine { .. } | Self::SetAside { .. } => 1,
            Self::Current => 2,
            Self::LocalCopy { .. } => 3,
            Self::NeverSaved { .. } => 4,
        }
    }
}

/// One version of the agent's state, as the `memory` tool shows it.
pub struct Version {
    /// When the state existed: the first source's (a sealed entry's
    /// signed `state_at`; else the file's mtime, when it was written).
    pub at: DateTime<Utc>,
    /// Everywhere it is held, sealed on the main line first.
    pub sources: Vec<Source>,
    /// `plaintext_hash` of the state: what makes two sources one version.
    pub state_hash: Sha256Hex,
    /// `memory.content`, raw (the pre-loss copy's `content`); `None` if the
    /// state holds no memory. Never logged.
    pub memory: Option<Zeroizing<String>>,
}

impl std::fmt::Debug for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Version")
            .field("at", &self.at)
            .field("sources", &self.sources)
            .field("state_hash", &self.state_hash)
            .field(
                "memory",
                &self
                    .memory
                    .as_ref()
                    .map(|m| format!("[{} chars]", m.chars().count())),
            )
            .finish()
    }
}

impl Version {
    /// Its sealed heads (a version can be sealed more than once).
    pub fn heads(&self) -> impl Iterator<Item = HeadRef> + '_ {
        self.sources.iter().filter_map(Source::head)
    }

    /// Its memory text, `""` if it has none.
    pub fn memory_str(&self) -> &str {
        self.memory.as_deref().map_or("", String::as_str)
    }
}

/// What is held but not shown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Held {
    /// The main line itself (no head, no keys, a chain that doesn't verify,
    /// Agora unreachable for a line that runs through it).
    MainLine,
    /// One sealed entry.
    Entry { head: HeadRef, line: Line },
    /// A local file, by name.
    File { name: String },
}

/// Something held that the history leaves out, and why: hashes, seqs,
/// file names and reasons only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeftOut {
    pub held: Held,
    pub why: String,
}

/// Every version an agent can be shown, oldest first, and what was held
/// but left out.
#[derive(Debug, Default)]
pub struct History {
    pub versions: Vec<Version>,
    pub left_out: Vec<LeftOut>,
}

/// What a head the agent gave resolves to.
#[derive(Debug)]
pub enum HeadMatch<'a> {
    /// The version holding it, and the source that is that head (a side
    /// line's or a set-aside one's says so).
    Found {
        version: &'a Version,
        source: &'a Source,
    },
    /// Held here but not shown, and why.
    Held(&'a LeftOut),
    /// More than one head starts so.
    Ambiguous(Vec<HeadRef>),
    /// No head held here starts so.
    NotHeld,
    /// Not a head: fewer than [`MIN_HEAD_HEX`] hex digits, or not hex.
    NotAHead,
}

/// The fewest hex digits a head is looked up by.
pub const MIN_HEAD_HEX: usize = 6;

impl History {
    /// The version (or the record of one left out) whose head starts with
    /// `prefix`, hex, case-insensitive.
    pub fn find_head(&self, prefix: &str) -> HeadMatch<'_> {
        let prefix = prefix.trim().to_ascii_lowercase();
        if prefix.len() < MIN_HEAD_HEX || !prefix.chars().all(|c| c.is_ascii_hexdigit()) {
            return HeadMatch::NotAHead;
        }
        let matches = |h: &HeadRef| h.entry_hash.to_hex().starts_with(&prefix);
        let mut found: Vec<(HeadRef, HeadMatch<'_>)> = Vec::new();
        for version in &self.versions {
            for source in &version.sources {
                if let Some(head) = source.head().filter(matches) {
                    found.push((head, HeadMatch::Found { version, source }));
                }
            }
        }
        for left in &self.left_out {
            if let Held::Entry { head, .. } = &left.held
                && matches(head)
                && !found.iter().any(|(h, _)| h == head)
            {
                found.push((*head, HeadMatch::Held(left)));
            }
        }
        let mut heads: Vec<HeadRef> = found.iter().map(|(h, _)| *h).collect();
        heads.sort_by_key(|h| h.entry_hash.to_hex());
        heads.dedup();
        match heads.len() {
            0 => HeadMatch::NotHeld,
            1 => found
                .into_iter()
                .next()
                .map_or(HeadMatch::NotHeld, |(_, m)| m),
            _ => HeadMatch::Ambiguous(heads),
        }
    }
}

/// Builds a [`History`], merging sources of one state into one version.
#[derive(Default)]
struct Builder {
    history: History,
    by_hash: HashMap<Sha256Hex, usize>,
}

impl Builder {
    fn add(
        &mut self,
        source: Source,
        at: DateTime<Utc>,
        state_hash: Sha256Hex,
        memory: Option<Zeroizing<String>>,
    ) {
        match self.by_hash.get(&state_hash) {
            Some(&i) => {
                let v = &mut self.history.versions[i];
                if !v.sources.contains(&source) {
                    v.sources.push(source);
                }
            }
            None => {
                self.by_hash.insert(state_hash, self.history.versions.len());
                self.history.versions.push(Version {
                    at,
                    sources: vec![source],
                    state_hash,
                    memory,
                });
            }
        }
    }

    fn leave_out(&mut self, held: Held, why: impl Into<String>) {
        self.history.left_out.push(LeftOut {
            held,
            why: why.into(),
        });
    }

    fn finish(mut self) -> History {
        for v in &mut self.history.versions {
            v.sources.sort_by_key(Source::rank);
        }
        // Stable: versions of one instant keep their order (the main line's
        // by seq)
        self.history.versions.sort_by_key(|v| v.at);
        self.history
    }
}

/// The memory text a sealed value holds: a state's `memory.content`, or
/// the pre-loss copy's `content` (a `MEMORY.json`).
fn memory_of(kind: StateEntryKind, value: &serde_json::Value) -> Option<Zeroizing<String>> {
    let text = match kind {
        StateEntryKind::ArchivedPrior => value
            .get("content")
            .and_then(serde_json::Value::as_str)
            .or_else(|| super::notice::memory_content(value)),
        _ => super::notice::memory_content(value),
    };
    text.map(|t| Zeroizing::new(t.to_string()))
}

fn micros(t: i64) -> DateTime<Utc> {
    DateTime::from_timestamp_micros(t).unwrap_or_default()
}

fn head_of(entry: &SignedStateEntry) -> HeadRef {
    HeadRef {
        entry_hash: entry.entry_hash(),
        seq: entry.entry.seq,
    }
}

/// Every version agent `id` can be shown. `Err` only when its directory
/// can't be listed; anything else held but unreadable, unverifiable or
/// unopenable is in [`History::left_out`].
pub async fn resolve(
    state_dir: &Path,
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
) -> anyhow::Result<History> {
    let agent_dir = state_dir.join(id.to_string());
    let mut b = Builder::default();
    sealed(&agent_dir, keys, api, id, &mut b).await;
    let dir = agent_dir.clone();
    let local = tokio::task::spawn_blocking(move || local_files(&dir))
        .await
        .map_err(|e| anyhow::anyhow!("the local read failed: {e}"))??;
    for found in local {
        match found {
            Ok(l) => b.add(l.source, l.at, l.state_hash, l.memory),
            Err(left) => b.history.left_out.push(left),
        }
    }
    Ok(b.finish())
}

/// How a side entry was found.
#[derive(Clone, Copy)]
enum Aside {
    Displaced,
    Lost(DateTime<Utc>),
}

/// `OutboxItem`s in `dir` with extension `ext`, and their files' mtimes.
/// Unreadable ones are left out, by name.
fn read_items(dir: &Path, ext: &str, b: &mut Builder) -> Vec<(OutboxItem, DateTime<Utc>)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|x| x.to_str()) != Some(ext) {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let read = std::fs::read(&path)
            .map_err(|e| e.to_string())
            .and_then(|bytes| {
                serde_json::from_slice::<OutboxItem>(&bytes)
                    .map_err(|e| format!("{:?}", e.classify()))
            });
        match read {
            Ok(item) => {
                let mtime = entry
                    .metadata()
                    .and_then(|m| m.modified())
                    .map(DateTime::<Utc>::from)
                    .unwrap_or_default();
                out.push((item, mtime));
            }
            Err(why) => b.leave_out(Held::File { name }, format!("unreadable: {why}")),
        }
    }
    out
}

/// The sealed versions: the main line, then the side lines.
async fn sealed(
    agent_dir: &Path,
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
    b: &mut Builder,
) {
    let seal = AgentSeal::new(agent_dir, id);
    if !std::fs::exists(seal.dir()).unwrap_or(true) {
        return;
    }
    let head = match seal.head().await {
        Ok(Some(head)) => Some(head),
        Ok(None) => {
            b.leave_out(Held::MainLine, "sealed/head.json is missing");
            None
        }
        Err(e) => {
            b.leave_out(Held::MainLine, format!("sealed/head.json unreadable: {e}"));
            None
        }
    };
    let Some(k) = keys.seal_keys(id) else {
        b.leave_out(
            Held::MainLine,
            "the runner holds no sealing keys for the agent",
        );
        return;
    };
    let trusted = match keys.trusted(id) {
        Ok(Some(t)) => t,
        Ok(None) => {
            b.leave_out(Held::MainLine, "no signing key to verify with");
            return;
        }
        Err(e) => {
            b.leave_out(Held::MainLine, format!("the trusted keys: {e}"));
            return;
        }
    };
    let resolutions = super::lost::resolutions(&seal).unwrap_or_else(|e| {
        b.leave_out(
            Held::File {
                name: "sealed/lost".into(),
            },
            format!("{e:#}"),
        );
        Vec::new()
    });
    let lost_seals: Vec<AgentSeal> = resolutions
        .iter()
        .map(|(dir, _)| AgentSeal::new(dir, id))
        .collect();
    let mut seals = vec![&seal];
    seals.extend(lost_seals.iter());
    let mut opener = Opener::new(seals, api, &k);

    // The main line
    let mut main = Vec::new();
    if let Some(head) = &head {
        match verified_line(&seal, keys, api, id, head).await {
            Ok(line) => main = line,
            Err(e) => {
                tracing::warn!(
                    event_type = "memory_history_main_line_unverified",
                    agent_id = %id,
                    head = %head.head.entry_hash,
                    error = %format!("{e:#}"),
                    "the sealed main line couldn't be read for the history"
                );
                b.leave_out(Held::MainLine, format!("{e:#}"));
            }
        }
    }
    for entry in &main {
        let source = match entry.entry.kind {
            StateEntryKind::Snapshot => Source::Sealed {
                head: head_of(entry),
            },
            StateEntryKind::ArchivedPrior => Source::ArchivedPrior {
                head: head_of(entry),
            },
            _ => continue,
        };
        open_into(b, &mut opener, &main, entry, source, Line::Main).await;
    }

    // The side lines: what is held here off the main line
    let on_main: HashSet<Sha256Hex> = main.iter().map(SignedStateEntry::entry_hash).collect();
    let mut pool = main.clone();
    let mut side: Vec<(SignedStateEntry, Aside)> = Vec::new();
    for (_, intent) in &resolutions {
        for item in &intent.items {
            side.push((item.entry.clone(), Aside::Lost(intent.at)));
        }
    }
    for (item, mtime) in read_items(&seal.dir().join("sent"), "lost", b) {
        side.push((item.entry, Aside::Lost(mtime)));
    }
    for (item, _) in read_items(&seal.dir().join("outbox"), "displaced", b) {
        side.push((item.entry, Aside::Displaced));
    }
    // Entries held off the main line: only against a main line known
    let held = if main.is_empty() {
        Ok(Vec::new())
    } else {
        local_entries(&seal).await
    };
    match held {
        Ok(entries) => {
            for e in entries {
                side.push((e, Aside::Displaced));
            }
        }
        Err(e) => b.leave_out(
            Held::File {
                name: "sealed/sent, sealed/outbox".into(),
            },
            e,
        ),
    }
    pool.extend(side.iter().map(|(e, _)| e.clone()));
    let mut seen = on_main;
    let mut server: Option<Result<Vec<SignedStateEntry>, String>> = None;
    for (entry, aside) in side {
        if !seen.insert(entry.entry_hash()) {
            continue;
        }
        if !matches!(
            entry.entry.kind,
            StateEntryKind::Snapshot | StateEntryKind::ArchivedPrior
        ) {
            continue;
        }
        let head = head_of(&entry);
        let Some(line) = side_line(&pool, &entry, id, &trusted, keys, api, &mut server).await
        else {
            tracing::warn!(
                event_type = "memory_history_side_line_unverified",
                agent_id = %id,
                head = %head.entry_hash,
                seq = head.seq,
                "a side-line version's chain doesn't verify from the genesis; left out"
            );
            b.leave_out(
                Held::Entry {
                    head,
                    line: Line::Side,
                },
                "its line doesn't verify from the genesis",
            );
            continue;
        };
        let source = match aside {
            Aside::Displaced => Source::SideLine { head },
            Aside::Lost(on) => Source::SetAside { head, on },
        };
        open_into(b, &mut opener, &line, &entry, source, Line::Side).await;
    }
}

/// `entry`'s line from the genesis, verified with the runner's trust:
/// from the entries held here, else through Agora's (fetched once).
async fn side_line(
    pool: &[SignedStateEntry],
    entry: &SignedStateEntry,
    id: AgentId,
    trusted: &TrustedKeys,
    keys: &KeyStore,
    api: &dyn SealedApi,
    server: &mut Option<Result<Vec<SignedStateEntry>, String>>,
) -> Option<Vec<SignedStateEntry>> {
    let hash = entry.entry_hash();
    let line = match walk_line(pool, &hash, Some(entry.entry.seq)) {
        Ok(line) => line,
        Err(_) => {
            if server.is_none() {
                let fetched = match keys.signing(id) {
                    Some(signing) => super::api::chain(api, id, &signing).await,
                    None => Err("no signing key".into()),
                };
                *server = Some(fetched);
            }
            let chain = server.as_ref()?.as_ref().ok()?;
            let all = merge_server(pool, chain).ok()?;
            walk_line(&all, &hash, Some(entry.entry.seq)).ok()?
        }
    };
    let report = verify_chain(id, &line, trusted);
    (report.intact && report.head == Some(head_of(entry))).then_some(line)
}

async fn open_into(
    b: &mut Builder,
    opener: &mut Opener<'_>,
    line: &[SignedStateEntry],
    entry: &SignedStateEntry,
    source: Source,
    on: Line,
) {
    match opener.open(line, &entry.entry).await {
        Ok(value) => {
            let state_hash = plaintext_hash(&value);
            let memory = memory_of(entry.entry.kind, &value);
            b.add(source, micros(entry.entry.state_at), state_hash, memory);
        }
        Err(e) => b.leave_out(
            Held::Entry {
                head: head_of(entry),
                line: on,
            },
            format!("couldn't be opened: {e:#}"),
        ),
    }
}

/// A version read from a local file.
struct Local {
    source: Source,
    at: DateTime<Utc>,
    state_hash: Sha256Hex,
    memory: Option<Zeroizing<String>>,
}

/// The archives (oldest first), `state.json`, then the `unsaved` rescues:
/// each state projected to its hash and memory as soon as it is read, so
/// only one whole state is in memory at a time.
fn local_files(agent_dir: &Path) -> anyhow::Result<Vec<Result<Local, LeftOut>>> {
    let archives = match super::enroll::archives(agent_dir) {
        Ok(a) => a,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => {
            return Err(anyhow::Error::new(e).context(format!("listing {}", agent_dir.display())));
        }
    };
    let mut files: Vec<(PathBuf, Source)> = Vec::new();
    for a in archives {
        let file = name_of(&a.path);
        files.push((a.path, Source::LocalCopy { file }));
    }
    files.push((agent_dir.join("state.json"), Source::Current));
    let mut unsaved: Vec<PathBuf> = std::fs::read_dir(agent_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let n = name_of(p);
            n.starts_with("unsaved.") && n.ends_with(".json")
        })
        .collect();
    unsaved.sort();
    for p in unsaved {
        let file = name_of(&p);
        files.push((p, Source::NeverSaved { file }));
    }
    let mut out = Vec::new();
    for (path, source) in files {
        if source == Source::Current && !std::fs::exists(&path).unwrap_or(true) {
            continue;
        }
        let at = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .map(DateTime::<Utc>::from)
            .unwrap_or_default();
        out.push(match super::enroll::envelope_state(&path) {
            Ok((_, state)) => Ok(Local {
                at,
                state_hash: plaintext_hash(&state),
                memory: super::notice::memory_content(&state)
                    .map(|m| Zeroizing::new(m.to_string())),
                source,
            }),
            Err(e) => Err(LeftOut {
                held: Held::File {
                    name: name_of(&path),
                },
                why: e.to_string(),
            }),
        });
    }
    Ok(out)
}

fn name_of(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sealed::props::{History as Built, Mode, build, version};
    use agora_agentkit::reactor::{FsStorage, Storage};

    fn memory(n: usize) -> String {
        version(n)["memory"]["content"]
            .as_str()
            .unwrap()
            .to_string()
    }

    /// Every file under `root`: path, size and mtime. What a read-only
    /// pass must leave as it was.
    fn tree(root: &Path) -> Vec<(PathBuf, u64, std::time::SystemTime)> {
        let mut out = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            for e in std::fs::read_dir(&dir).unwrap() {
                let e = e.unwrap();
                let m = e.metadata().unwrap();
                if m.is_dir() {
                    stack.push(e.path());
                } else {
                    out.push((e.path(), m.len(), m.modified().unwrap()));
                }
            }
        }
        out.sort();
        out
    }

    fn sources(h: &History) -> Vec<Vec<&'static str>> {
        h.versions
            .iter()
            .map(|v| {
                v.sources
                    .iter()
                    .map(|s| match s {
                        Source::Sealed { .. } => "sealed",
                        Source::ArchivedPrior { .. } => "prior",
                        Source::SideLine { .. } => "side",
                        Source::SetAside { .. } => "aside",
                        Source::LocalCopy { .. } => "copy",
                        Source::Current => "current",
                        Source::NeverSaved { .. } => "unsaved",
                    })
                    .collect()
            })
            .collect()
    }

    /// An agent never enrolled: its archives, its state and a rescue, the
    /// same state once, an unreadable archive left out by name.
    #[tokio::test]
    async fn an_unsealed_agent_has_its_local_copies() {
        let root = crate::sealed::TestDir::new("history-plain");
        let state_dir = root.join("state");
        let id = AgentId::new();
        let mut fs = FsStorage::new(&state_dir);
        for n in [1, 2, 2, 3] {
            fs.save_raw(id, version(n), None).await.unwrap();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let dir = state_dir.join(id.to_string());
        std::fs::write(dir.join("state.20260101T000000000Z.json"), b"not json").unwrap();
        std::fs::write(
            dir.join("unsaved.20991231T000000000Z.json"),
            serde_json::to_vec(&serde_json::json!({ "format": 1, "state": version(4) })).unwrap(),
        )
        .unwrap();
        let keys = KeyStore::new(root.join("secrets"));
        let api = crate::sealed::api::fake::FakeApi::new();
        let before = tree(root.path());
        let h = resolve(&state_dir, &keys, &api, id).await.unwrap();
        assert_eq!(tree(root.path()), before, "read-only");
        assert!(api.calls.lock().unwrap().is_empty(), "no network");
        let memories: Vec<&str> = h.versions.iter().map(Version::memory_str).collect();
        assert_eq!(memories, [memory(1), memory(2), memory(3), memory(4)]);
        assert_eq!(
            sources(&h),
            [
                vec!["copy"],
                vec!["copy", "copy"],
                vec!["current"],
                vec!["unsaved"]
            ]
        );
        assert_eq!(
            h.left_out,
            [LeftOut {
                held: Held::File {
                    name: "state.20260101T000000000Z.json".into()
                },
                why: "not JSON (Syntax)".into(),
            }]
        );
        assert!(
            !format!("{h:?}").contains("synthetic"),
            "content never in Debug"
        );
        assert!(matches!(h.find_head("abcdef"), HeadMatch::NotHeld));
        assert!(matches!(h.find_head("abc"), HeadMatch::NotAHead));
    }

    /// An enrolled agent: every sealed version, opened from verified
    /// blobs, fetched from Agora once pruned here; the archives that hold
    /// the same states merged into them; nothing written.
    #[tokio::test]
    async fn an_enrolled_agent_has_its_sealed_versions_and_copies_merged() {
        let b = build(&Built {
            saves: 3,
            upload: 9,
            rotate: false,
            mode: Mode::Plain,
            upload_rebased: false,
        })
        .await;
        b.seal().prune_blobs().await.unwrap();
        b.api.calls.lock().unwrap().clear();
        let before = tree(b.root.path());
        let h = resolve(&b.state_dir(), &b.keys, &b.api, b.id)
            .await
            .unwrap();
        assert_eq!(tree(b.root.path()), before, "read-only");
        assert!(h.left_out.is_empty(), "{:?}", h.left_out);
        assert!(
            b.api.calls.lock().unwrap().iter().any(|c| c == "blob"),
            "pruned blobs fetched from Agora"
        );
        // check::tests::enrolled seals a first state with no `memory.content`
        let memories: Vec<&str> = h.versions.iter().map(Version::memory_str).collect();
        assert_eq!(memories, ["", &memory(2), &memory(3), &memory(4)]);
        assert!(h.versions[0].memory.is_none());
        assert_eq!(
            sources(&h),
            [
                vec!["sealed", "copy"],
                vec!["sealed", "copy"],
                vec!["sealed", "copy"],
                vec!["sealed", "current"]
            ]
        );
        let line = check::local_entries(&b.seal()).await.unwrap();
        let heads: Vec<HeadRef> = h.versions.iter().flat_map(Version::heads).collect();
        assert_eq!(heads, line[1..].iter().map(head_of).collect::<Vec<_>>());
        // A head by its prefix
        let newest = heads[3].entry_hash.to_hex();
        let HeadMatch::Found { version, source } = h.find_head(&newest[..8].to_uppercase()) else {
            panic!()
        };
        assert_eq!(version.memory_str(), memory(4));
        assert_eq!(source, &Source::Sealed { head: heads[3] });
    }

    use crate::sealed::check;

    /// Two heads sharing the prefix: both named, neither chosen.
    #[test]
    fn an_ambiguous_prefix_names_the_candidates() {
        let head = |last: u8| {
            let mut bytes = [0xab; 32];
            bytes[31] = last;
            HeadRef {
                entry_hash: Sha256Hex::from(bytes),
                seq: 1,
            }
        };
        let left = |h| LeftOut {
            held: Held::Entry {
                head: h,
                line: Line::Side,
            },
            why: "test".into(),
        };
        let h = History {
            versions: Vec::new(),
            left_out: vec![left(head(1)), left(head(2))],
        };
        let HeadMatch::Ambiguous(both) = h.find_head("abababab") else {
            panic!()
        };
        assert_eq!(both, [head(1), head(2)]);
        assert!(matches!(
            h.find_head(&head(2).entry_hash.to_hex()),
            HeadMatch::Held(_)
        ));
    }

    /// A fork rebased: the displaced version is shown as on a side line,
    /// and its head still resolves, to the version the rebased head holds.
    #[tokio::test]
    async fn a_displaced_head_resolves_to_its_side_line() {
        let b = build(&Built {
            saves: 2,
            upload: 0,
            rotate: false,
            mode: Mode::Rebase,
            upload_rebased: false,
        })
        .await;
        let displaced: Vec<OutboxItem> = std::fs::read_dir(b.dir.join("sealed/outbox"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "displaced"))
            .map(|p| serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap())
            .collect();
        assert_eq!(displaced.len(), 1);
        let head = head_of(&displaced[0].entry);
        let before = tree(b.root.path());
        let h = resolve(&b.state_dir(), &b.keys, &b.api, b.id)
            .await
            .unwrap();
        assert_eq!(tree(b.root.path()), before, "read-only");
        assert!(h.left_out.is_empty(), "{:?}", h.left_out);
        let HeadMatch::Found { version, source } = h.find_head(&head.entry_hash.to_hex()[..12])
        else {
            panic!("{h:?}")
        };
        assert_eq!(source, &Source::SideLine { head });
        assert_eq!(version.memory_str(), memory(3));
        // The rebased head sealed the same state over Agora's: one version
        assert_eq!(version.sources[0].line(), Some(Line::Main));
    }

    /// A displaced entry whose line doesn't verify: left out, its head
    /// still answered, as held but not shown.
    #[tokio::test]
    async fn an_unverified_side_line_is_left_out_but_answered() {
        let b = build(&Built {
            saves: 2,
            upload: 0,
            rotate: false,
            mode: Mode::Rebase,
            upload_rebased: false,
        })
        .await;
        let path = std::fs::read_dir(b.dir.join("sealed/outbox"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| p.extension().is_some_and(|x| x == "displaced"))
            .unwrap();
        let mut item: OutboxItem = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        // Signed by a key the runner doesn't trust
        item.entry = item
            .entry
            .entry
            .clone()
            .sign(&agora_agentkit::crypto::generate_keypair().0);
        std::fs::write(&path, serde_json::to_vec(&item).unwrap()).unwrap();
        let head = head_of(&item.entry);
        let h = resolve(&b.state_dir(), &b.keys, &b.api, b.id)
            .await
            .unwrap();
        let HeadMatch::Held(left) = h.find_head(&head.entry_hash.to_hex()) else {
            panic!("{h:?}")
        };
        assert_eq!(
            left.held,
            Held::Entry {
                head,
                line: Line::Side
            }
        );
        assert!(
            h.versions
                .iter()
                .all(|v| v.sources.iter().all(|s| s.line() != Some(Line::Side)))
        );
    }

    /// Versions Agora lost, set aside by `sealed-accept --rebase`: shown as
    /// set aside on the day, both on top of Agora's head and on a new
    /// chain, their heads answered.
    #[tokio::test]
    async fn set_aside_versions_resolve() {
        for nothing_left in [false, true] {
            let b = build(&Built {
                saves: 2,
                upload: 9,
                rotate: false,
                mode: Mode::Plain,
                upload_rebased: false,
            })
            .await;
            let old = b.seal().head().await.unwrap().unwrap().head;
            let server = b.server();
            b.set_server(if nothing_left {
                Vec::new()
            } else {
                server[..server.len() - 1].to_vec()
            });
            crate::sealed::accept::rebase(&b.state_dir(), &b.keys, &b.api, b.id, "lost")
                .await
                .unwrap();
            let before = tree(b.root.path());
            let h = resolve(&b.state_dir(), &b.keys, &b.api, b.id)
                .await
                .unwrap();
            assert_eq!(tree(b.root.path()), before, "read-only");
            assert!(h.left_out.is_empty(), "{nothing_left}: {:?}", h.left_out);
            let HeadMatch::Found { version, source } = h.find_head(&old.entry_hash.to_hex()) else {
                panic!("{nothing_left}: {h:?}")
            };
            assert!(
                matches!(source, Source::SetAside { head, .. } if *head == old),
                "{source:?}"
            );
            assert_eq!(version.memory_str(), memory(3));
            assert_eq!(
                version.sources[0].line(),
                Some(Line::Main),
                "re-sealed on the new line"
            );
            // Every memory once per state
            let memories: Vec<&str> = h.versions.iter().map(Version::memory_str).collect();
            assert_eq!(memories, ["", &memory(2), &memory(3)]);
        }
    }
}
