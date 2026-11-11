//! The `memory` tool's versions, for one agent: every version of its state
//! the runner can show, from wherever it is held, each with its memory
//! text and, if it was sealed, its head.
//!
//! | Source | Held in | Shown when |
//! |---|---|---|
//! | [`Source::Sealed`], [`Source::ArchivedPrior`] | the main line to `sealed/head.json`'s head | the line verifies ([`super::version::verified_line`]: through Agora's entries where it runs through them) |
//! | [`Source::SideLine`] | `sealed/outbox/*.displaced`, and entries a later save forked from (`head.json`'s `displaced`) | its own line verifies from the genesis (`verify_chain`); else left out, with a structured warning |
//! | [`Source::OffLine`] | other entries held here off the main line | the same |
//! | [`Source::SetAside`] | `sealed/lost/<utc>/` (and `sealed/sent/*.lost`) | the same |
//! | [`Source::LocalCopy`] | the archives `state.<utc>.json` | always: not sealed, so no head |
//! | [`Source::Current`] | `state.json` | always |
//! | [`Source::KeptAtExit`] | `unsaved.<utc>.json` | always: unsigned |
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
//! **A sealed version that can't be opened just now** (a blob Agora didn't
//! serve, after a few tries) stays listed with its sealed source and head,
//! [`Version::unopened`] saying why, and no memory. Its local twin, matched
//! by the plaintext hash the runner recorded when it sealed it, is merged
//! in as a source rather than standing in for it as an unsigned copy. A
//! main line that can't be read at all is [`Held::MainLine`] in
//! [`History::left_out`]: the archives then show as local copies, and the
//! renderer should say the sealed history couldn't be checked.
//!
//! **Any head the agent was shown** is answered ([`History::find_head`]):
//! with the version that holds it (a side line's or a set-aside one's
//! source says so), or with why it is held but not shown ([`LeftOut`]),
//! never "not found" for a head this runner holds.
//!
//! **Read-only.** Nothing here writes: not the state, the outbox, the
//! receipts or the inclusion record. Blobs no longer held here are fetched
//! from Agora and checked against their entries ([`super::version::Opener`]),
//! and nothing fetched is written anywhere. What was opened is kept in the
//! caller's [`HistoryCache`] only, in memory. Content is never logged:
//! warnings carry hashes, seqs, file names and counts.
//!
//! **Network, per call:** one blob GET for each sealed version not in the
//! cache and no longer held here (plus its keyframe's, once per run of
//! deltas, retried up to three times each on failure), and up to two chain
//! GETs (one for a main line that runs through Agora's entries, one for a
//! side line that does).

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
use super::check::{merge_server, walk_line};
use super::version::{ChainCache, FailedBlobs, Opener, verified_line_cached};

/// Which line a sealed version is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Line {
    /// The main line to `sealed/head.json`'s head.
    Main,
    /// Off it: displaced by a fork, set aside by a rebase, or otherwise
    /// not on the current line.
    Side,
}

/// Where one version was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// A snapshot on the main line, verified.
    Sealed { head: HeadRef },
    /// The pre-loss copy, sealed on the main line (`archived_prior`).
    ArchivedPrior { head: HeadRef },
    /// Sealed on a side line the operator rebased away from, or a later
    /// save forked from (a `.displaced` entry, or a head `head.json`
    /// records as displaced), verified from the genesis.
    SideLine { head: HeadRef },
    /// Sealed and verified from the genesis, held here but not on the
    /// current main line, with no record of why. Neutral: not called
    /// rebased.
    OffLine { head: HeadRef },
    /// Set aside when the agent's line was rebased (`sealed/lost/<utc>/`,
    /// or `sent/*.lost`), verified from the genesis. `on` is when, `None`
    /// if unknown.
    SetAside {
        head: HeadRef,
        on: Option<DateTime<Utc>>,
    },
    /// A local archive, `state.<utc>.json`: the runner's own copy, not
    /// sealed, so no head.
    LocalCopy { file: String },
    /// `state.json`.
    Current,
    /// `unsaved.<utc>.json`: a copy your runner kept when a run ended
    /// without saving. Unsigned; nothing here checks where it came from.
    KeptAtExit { file: String },
}

impl Source {
    /// Its sealed head, if it has one.
    pub fn head(&self) -> Option<HeadRef> {
        match self {
            Self::Sealed { head }
            | Self::ArchivedPrior { head }
            | Self::SideLine { head }
            | Self::OffLine { head }
            | Self::SetAside { head, .. } => Some(*head),
            Self::LocalCopy { .. } | Self::Current | Self::KeptAtExit { .. } => None,
        }
    }

    /// Its line, if it is sealed.
    #[cfg_attr(not(test), expect(dead_code, reason = "the resolver's tests read it"))]
    pub fn line(&self) -> Option<Line> {
        match self {
            Self::Sealed { .. } | Self::ArchivedPrior { .. } => Some(Line::Main),
            Self::SideLine { .. } | Self::OffLine { .. } | Self::SetAside { .. } => {
                Some(Line::Side)
            }
            Self::LocalCopy { .. } | Self::Current | Self::KeptAtExit { .. } => None,
        }
    }

    /// The order sources are listed in: sealed on the main line first.
    fn rank(&self) -> u8 {
        match self {
            Self::Sealed { .. } | Self::ArchivedPrior { .. } => 0,
            Self::SideLine { .. } | Self::SetAside { .. } | Self::OffLine { .. } => 1,
            Self::Current => 2,
            Self::LocalCopy { .. } => 3,
            Self::KeptAtExit { .. } => 4,
        }
    }
}

/// One version of the agent's state, as the `memory` tool shows it.
pub struct Version {
    /// When the state existed: a sealed entry's signed `state_at`; else the
    /// file's mtime, when it was written. `None`: date unknown.
    pub at: Option<DateTime<Utc>>,
    /// Everywhere it is held, sealed on the main line first.
    pub sources: Vec<Source>,
    /// `plaintext_hash` of the state: what makes two sources one version.
    /// For an unopened sealed version, the hash the runner recorded when
    /// it sealed it, if it has one.
    pub state_hash: Option<Sha256Hex>,
    /// `memory.content`, raw (the pre-loss copy's `content`); empty if the
    /// state holds no memory, `None` if it couldn't be opened. Never logged.
    pub memory: Option<Zeroizing<String>>,
    /// Why a sealed version couldn't be opened just now (Agora unreachable,
    /// a blob not served): it is listed, without its memory.
    pub unopened: Option<String>,
    /// An unopened version's memory as a local copy of the same state
    /// holds it (matched by the plaintext hash the runner recorded when it
    /// sealed it): unsigned, so never shown as the version's memory, only
    /// used to name it by its memory id. Never logged.
    pub twin_memory: Option<Zeroizing<String>>,
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
            .field("unopened", &self.unopened)
            .field("twin_memory", &self.twin_memory.as_ref().map(|_| "[held]"))
            .finish()
    }
}

impl Version {
    /// Its sealed heads (a version can be sealed more than once).
    pub fn heads(&self) -> impl Iterator<Item = HeadRef> + '_ {
        self.sources.iter().filter_map(Source::head)
    }

    /// Its memory text, `""` if it has none (or it is unopened).
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

/// Every version an agent can be shown, oldest first (date unknown last),
/// and what was held but left out.
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

/// What one opened sealed entry, or one read local file, came to.
#[derive(Clone)]
struct Opened {
    state_hash: Sha256Hex,
    memory: Option<Zeroizing<String>>,
}

/// What [`resolve`] opened and read, kept for the next call: sealed
/// entries by entry hash (an entry binds its blob, so what it opened to
/// never changes), local files by path, size and mtime. Only what opened
/// is kept; a failure is tried again next call. Hold one for a session so
/// repeated `memory` calls don't refetch every pruned blob; drop it to
/// start over. In memory only; holds memory text, so its `Debug` shows
/// counts.
#[derive(Default)]
pub struct HistoryCache {
    sealed: HashMap<Sha256Hex, Opened>,
    files: HashMap<(PathBuf, u64, std::time::SystemTime), Opened>,
    /// Agora's chain, asked once a session.
    chain: ChainCache,
    /// Blob fetches that failed lately, not tried again yet.
    failed: FailedBlobs,
}

impl std::fmt::Debug for HistoryCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HistoryCache")
            .field("sealed", &self.sealed.len())
            .field("files", &self.files.len())
            .field("chain", &self.chain.held())
            .field("failed", &self.failed.count())
            .finish()
    }
}

impl HistoryCache {
    /// How many sealed entries and local files it holds.
    pub fn sizes(&self) -> (usize, usize) {
        (self.sealed.len(), self.files.len())
    }
}

/// Builds a [`History`], merging sources of one state into one version.
#[derive(Default)]
struct Builder {
    history: History,
    by_hash: HashMap<Sha256Hex, usize>,
}

impl Builder {
    fn add(&mut self, source: Source, at: Option<DateTime<Utc>>, opened: Opened) {
        if let Some(&i) = self.by_hash.get(&opened.state_hash) {
            let v = &mut self.history.versions[i];
            if !v.sources.contains(&source) {
                v.sources.push(source);
            }
            // A sealed twin that couldn't be opened gets no content from a
            // local copy: it stays as sealed, unopened, named by the copy's
            // memory only
            if v.unopened.is_some() && v.twin_memory.is_none() {
                v.twin_memory = opened.memory;
            }
            return;
        }
        self.by_hash
            .insert(opened.state_hash, self.history.versions.len());
        self.history.versions.push(Version {
            at,
            sources: vec![source],
            state_hash: Some(opened.state_hash),
            memory: opened.memory,
            unopened: None,
            twin_memory: None,
        });
    }

    /// A sealed version that couldn't be opened: listed, merged with its
    /// twin by the recorded hash if there is one.
    fn add_unopened(
        &mut self,
        source: Source,
        at: Option<DateTime<Utc>>,
        recorded: Option<Sha256Hex>,
        why: String,
    ) {
        if let Some(i) = recorded.and_then(|h| self.by_hash.get(&h).copied()) {
            let v = &mut self.history.versions[i];
            if !v.sources.contains(&source) {
                v.sources.push(source);
            }
            return;
        }
        if let Some(h) = recorded {
            self.by_hash.insert(h, self.history.versions.len());
        }
        self.history.versions.push(Version {
            at,
            sources: vec![source],
            state_hash: recorded,
            memory: None,
            unopened: Some(why),
            twin_memory: None,
        });
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
        // by seq); date unknown last
        self.history
            .versions
            .sort_by_key(|v| (v.at.is_none(), v.at));
        self.history
    }
}

/// The memory text a sealed value holds: a state's `memory.content`, or
/// the pre-loss copy's `content` (a `MEMORY.json`); empty when it holds
/// none (agentkit's projection, one copy with the notices').
fn memory_of(kind: StateEntryKind, value: &serde_json::Value) -> Option<Zeroizing<String>> {
    use agora_agentkit::reactor::seed::memory::memory_text;
    let text = match kind {
        StateEntryKind::ArchivedPrior => value
            .get("content")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_else(|| memory_text(value)),
        _ => memory_text(value),
    };
    Some(Zeroizing::new(text.to_string()))
}

fn head_of(entry: &SignedStateEntry) -> HeadRef {
    HeadRef {
        entry_hash: entry.entry_hash(),
        seq: entry.entry.seq,
    }
}

/// Every version agent `id` can be shown. `Err` only when its directory
/// can't be listed; anything else held but unreadable, unverifiable or
/// unopenable is in [`History::left_out`], or listed unopened.
pub async fn resolve(
    state_dir: &Path,
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
    cache: &mut HistoryCache,
) -> anyhow::Result<History> {
    let agent_dir = state_dir.join(id.to_string());
    let mut b = Builder::default();
    let deadline = tokio::time::Instant::now() + RESOLVE_DEADLINE;
    sealed(&agent_dir, keys, api, id, cache, &mut b, deadline).await;
    if b.history.left_out.iter().any(|l| l.held == Held::MainLine) {
        hold_unchecked(&agent_dir, id, &mut b).await;
    }
    let dir = agent_dir.clone();
    let files = std::mem::take(&mut cache.files);
    let (local, files) = tokio::task::spawn_blocking(move || local_files(&dir, files))
        .await
        .map_err(|e| anyhow::anyhow!("the local read failed: {e}"))?;
    cache.files = files;
    for found in local? {
        match found {
            Ok((source, at, opened)) => b.add(source, at, opened),
            Err(left) => b.history.left_out.push(left),
        }
    }
    Ok(b.finish())
}

/// The longest [`resolve`] spends opening sealed versions: past it, the
/// versions not opened yet are listed unopened, and what was opened is
/// answered.
pub const RESOLVE_DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);

/// What an agent is told of a version not tried before the deadline.
pub const PAST_DEADLINE: &str = "not opened: gathering your history took too long";

/// With the main line unchecked, every head held on it (`head.json`'s,
/// `sent/`'s, `outbox/`'s) is still answered: held, but unchecked, never
/// "not found".
async fn hold_unchecked(agent_dir: &Path, id: AgentId, b: &mut Builder) {
    let seal = AgentSeal::new(agent_dir, id);
    let mut heads: Vec<HeadRef> = Vec::new();
    if let Ok(Some(h)) = seal.head().await {
        heads.push(h.head);
    }
    for items in [seal.sent().await, seal.outbox().await]
        .into_iter()
        .flatten()
    {
        heads.extend(items.iter().map(|i| head_of(&i.entry)));
    }
    let known: HashSet<Sha256Hex> = b
        .history
        .versions
        .iter()
        .flat_map(Version::heads)
        .chain(b.history.left_out.iter().filter_map(|l| match &l.held {
            Held::Entry { head, .. } => Some(*head),
            _ => None,
        }))
        .map(|h| h.entry_hash)
        .collect();
    let mut seen = known;
    for head in heads {
        if seen.insert(head.entry_hash) {
            b.leave_out(
                Held::Entry {
                    head,
                    line: Line::Main,
                },
                "your sealed history couldn't be checked just now",
            );
        }
    }
}

/// How a side entry was found.
#[derive(Clone, Copy)]
enum Aside {
    /// `.displaced`, or a head `head.json` records as displaced.
    Displaced,
    /// Set aside by a rebase, on this date if known.
    Lost(Option<DateTime<Utc>>),
    /// Held here off the main line, with no record of why.
    Off,
}

/// What `sealed/` holds besides the main line, read off the async runtime.
struct SealedFiles {
    /// `sealed/lost/<utc>/` directories, and what each set aside.
    resolutions: Vec<(PathBuf, super::lost::Intent)>,
    /// `sent/` and committed `outbox/` items (plaintext hashes recorded
    /// when sealed), read through agentkit after the rest.
    items: Vec<OutboxItem>,
    /// `sent/*.lost` items, with their file's mtime.
    lost: Vec<(OutboxItem, Option<DateTime<Utc>>)>,
    /// `outbox/*.displaced` items.
    displaced: Vec<OutboxItem>,
    left_out: Vec<LeftOut>,
}

/// `OutboxItem`s in `dir` with extension `ext`, and their files' mtimes.
/// Unreadable ones are left out, by name.
fn read_items(
    dir: &Path,
    ext: &str,
    left_out: &mut Vec<LeftOut>,
) -> Vec<(OutboxItem, Option<DateTime<Utc>>)> {
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
            Ok(item) => out.push((item, mtime(&path))),
            Err(why) => left_out.push(LeftOut {
                held: Held::File { name },
                why: format!("unreadable: {why}"),
            }),
        }
    }
    out
}

fn mtime(path: &Path) -> Option<DateTime<Utc>> {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .map(DateTime::<Utc>::from)
}

fn sealed_files(seal: &AgentSeal) -> SealedFiles {
    let mut left_out = Vec::new();
    let resolutions = super::lost::resolutions(seal).unwrap_or_else(|e| {
        left_out.push(LeftOut {
            held: Held::File {
                name: "sealed/lost".into(),
            },
            why: format!("{e:#}"),
        });
        Vec::new()
    });
    let lost = read_items(&seal.dir().join("sent"), "lost", &mut left_out);
    let displaced = read_items(&seal.dir().join("outbox"), "displaced", &mut left_out)
        .into_iter()
        .map(|(i, _)| i)
        .collect();
    SealedFiles {
        resolutions,
        items: Vec::new(),
        lost,
        displaced,
        left_out,
    }
}

/// The sealed versions: the main line, then the side lines.
async fn sealed(
    agent_dir: &Path,
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
    cache: &mut HistoryCache,
    b: &mut Builder,
    deadline: tokio::time::Instant,
) {
    let seal = AgentSeal::new(agent_dir, id);
    if !tokio::fs::try_exists(seal.dir()).await.unwrap_or(true) {
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
    let files = {
        let seal = seal.clone();
        match tokio::task::spawn_blocking(move || sealed_files(&seal)).await {
            Ok(files) => files,
            Err(e) => {
                b.leave_out(Held::MainLine, format!("reading sealed/ failed: {e}"));
                return;
            }
        }
    };
    let mut files = files;
    b.history.left_out.extend(files.left_out.iter().cloned());
    match (seal.sent().await, seal.outbox().await) {
        (Ok(sent), Ok(outbox)) => files.items = sent.into_iter().chain(outbox).collect(),
        (Err(e), _) | (_, Err(e)) => b.leave_out(
            Held::File {
                name: "sealed/sent, sealed/outbox".into(),
            },
            e.to_string(),
        ),
    }
    // What the runner recorded when it sealed each entry (local only)
    let mut recorded: HashMap<Sha256Hex, Sha256Hex> = HashMap::new();
    let all_items = files
        .items
        .iter()
        .chain(files.displaced.iter())
        .chain(files.lost.iter().map(|(i, _)| i))
        .chain(files.resolutions.iter().flat_map(|(_, i)| i.items.iter()));
    for item in all_items {
        if let Some(h) = item.plaintext_sha256 {
            recorded.insert(item.entry.entry_hash(), h);
        }
    }
    if let Some(h) = &head
        && let Some(p) = h.plaintext_sha256
    {
        recorded.insert(h.head.entry_hash, p);
    }
    let lost_seals: Vec<AgentSeal> = files
        .resolutions
        .iter()
        .map(|(dir, _)| AgentSeal::new(dir, id))
        .collect();
    let mut seals = vec![&seal];
    seals.extend(lost_seals.iter());
    let mut opener = Opener::new(seals, api, &k);
    opener.failed = std::mem::take(&mut cache.failed);
    let mut ctx = Ctx {
        b,
        opener: &mut opener,
        cache,
        recorded: &recorded,
        deadline,
    };

    // The main line
    let mut main = Vec::new();
    if let Some(head) = &head {
        match verified_line_cached(&seal, keys, api, id, head, &mut ctx.cache.chain).await {
            Ok(line) => main = line,
            Err(e) => {
                tracing::warn!(
                    event_type = "memory_history_main_line_unverified",
                    agent_id = %id,
                    head = %head.head.entry_hash,
                    error = %format!("{e:#}"),
                    "the sealed main line couldn't be read for the history"
                );
                ctx.b.leave_out(Held::MainLine, format!("{e:#}"));
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
        ctx.open(&main, entry, source).await;
    }

    // The side lines: what is held here off the main line
    let on_main: HashSet<Sha256Hex> = main.iter().map(SignedStateEntry::entry_hash).collect();
    let forked_from: HashSet<Sha256Hex> = head
        .as_ref()
        .map(|h| h.displaced.iter().map(|d| d.entry_hash).collect())
        .unwrap_or_default();
    let mut side: Vec<(SignedStateEntry, Aside)> = Vec::new();
    for (_, intent) in &files.resolutions {
        for item in &intent.items {
            side.push((item.entry.clone(), Aside::Lost(Some(intent.at))));
        }
    }
    for (item, on) in &files.lost {
        side.push((item.entry.clone(), Aside::Lost(*on)));
    }
    for item in &files.displaced {
        side.push((item.entry.clone(), Aside::Displaced));
    }
    // Entries held off the main line: only against a main line known
    if !main.is_empty() {
        for item in &files.items {
            let how = if forked_from.contains(&item.entry.entry_hash()) {
                Aside::Displaced
            } else {
                Aside::Off
            };
            side.push((item.entry.clone(), how));
        }
    }
    let mut pool = main.clone();
    pool.extend(side.iter().map(|(e, _)| e.clone()));
    let mut seen = on_main;
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
        let Some(line) =
            side_line(&pool, &entry, id, &trusted, keys, api, &mut ctx.cache.chain).await
        else {
            tracing::warn!(
                event_type = "memory_history_side_line_unverified",
                agent_id = %id,
                head = %head.entry_hash,
                seq = head.seq,
                "a side-line version's chain doesn't verify from the genesis; left out"
            );
            ctx.b.leave_out(
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
            Aside::Off => Source::OffLine { head },
        };
        ctx.open(&line, &entry, source).await;
    }
    ctx.cache.failed = std::mem::take(&mut ctx.opener.failed);
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
    server: &mut ChainCache,
) -> Option<Vec<SignedStateEntry>> {
    let hash = entry.entry_hash();
    let line = match walk_line(pool, &hash, Some(entry.entry.seq)) {
        Ok(line) => line,
        Err(_) => {
            let signing = keys.signing(id)?;
            let chain = server.get(api, id, &signing).await.ok()?;
            let all = merge_server(pool, &chain).ok()?;
            walk_line(&all, &hash, Some(entry.entry.seq)).ok()?
        }
    };
    let report = verify_chain(id, &line, trusted);
    (report.intact && report.head == Some(head_of(entry))).then_some(line)
}

/// What opening sealed entries needs, across the lines.
struct Ctx<'b, 'o, 'a> {
    b: &'b mut Builder,
    opener: &'o mut Opener<'a>,
    cache: &'b mut HistoryCache,
    recorded: &'b HashMap<Sha256Hex, Sha256Hex>,
    /// Past it, nothing more is fetched ([`RESOLVE_DEADLINE`]).
    deadline: tokio::time::Instant,
}

impl Ctx<'_, '_, '_> {
    /// Open `entry` (verified on `line`) into the history: from the cache,
    /// else its blob; listed unopened if that fails.
    async fn open(&mut self, line: &[SignedStateEntry], entry: &SignedStateEntry, source: Source) {
        let at = DateTime::from_timestamp_micros(entry.entry.state_at);
        let hash = entry.entry_hash();
        if let Some(opened) = self.cache.sealed.get(&hash) {
            self.b.add(source, at, opened.clone());
            return;
        }
        let opened = if tokio::time::Instant::now() >= self.deadline {
            Err(anyhow::anyhow!("{PAST_DEADLINE}"))
        } else {
            tokio::time::timeout_at(self.deadline, self.opener.open(line, &entry.entry))
                .await
                .unwrap_or_else(|_| Err(anyhow::anyhow!("{PAST_DEADLINE}")))
        };
        match opened {
            Ok(value) => {
                let opened = Opened {
                    state_hash: plaintext_hash(&value),
                    memory: memory_of(entry.entry.kind, &value),
                };
                self.cache.sealed.insert(hash, opened.clone());
                self.b.add(source, at, opened);
            }
            Err(e) => self.b.add_unopened(
                source,
                at,
                self.recorded.get(&hash).copied(),
                format!("couldn't be opened just now: {e:#}"),
            ),
        }
    }
}

type Files = HashMap<(PathBuf, u64, std::time::SystemTime), Opened>;

type Found = Result<(Source, Option<DateTime<Utc>>, Opened), LeftOut>;

/// The archives (oldest first), `state.json`, then the `unsaved` copies:
/// each state projected to its hash and memory as soon as it is read, so
/// only one whole state is in memory at a time; a file unchanged since the
/// last call (path, size, mtime) is taken from `cache`. Returns the cache
/// back, holding just the files seen.
fn local_files(agent_dir: &Path, mut cache: Files) -> (anyhow::Result<Vec<Found>>, Files) {
    let mut kept = Files::new();
    let found = (|| {
        let archives = match super::enroll::archives(agent_dir) {
            Ok(a) => a,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => {
                return Err(
                    anyhow::Error::new(e).context(format!("listing {}", agent_dir.display()))
                );
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
            files.push((p, Source::KeptAtExit { file }));
        }
        let mut out = Vec::new();
        for (path, source) in files {
            let meta = match std::fs::metadata(&path) {
                Ok(m) => m,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound && source == Source::Current => {
                    continue;
                }
                Err(e) => {
                    out.push(Err(LeftOut {
                        held: Held::File {
                            name: name_of(&path),
                        },
                        why: e.to_string(),
                    }));
                    continue;
                }
            };
            let modified = meta.modified().ok();
            let at = modified.map(DateTime::<Utc>::from);
            let key = modified.map(|m| (path.clone(), meta.len(), m));
            if let Some(opened) = key.as_ref().and_then(|k| cache.remove(k)) {
                kept.insert(key.expect("a hit has a key"), opened.clone());
                out.push(Ok((source, at, opened)));
                continue;
            }
            out.push(match super::enroll::envelope_state(&path) {
                Ok((_, state)) => {
                    let opened = Opened {
                        state_hash: plaintext_hash(&state),
                        memory: memory_of(StateEntryKind::Snapshot, &state),
                    };
                    if let Some(k) = key {
                        kept.insert(k, opened.clone());
                    }
                    Ok((source, at, opened))
                }
                Err(e) => Err(LeftOut {
                    held: Held::File {
                        name: name_of(&path),
                    },
                    why: e.to_string(),
                }),
            });
        }
        Ok(out)
    })();
    (found, kept)
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

    /// Every file and directory under `root`: path, kind, size, mtime and
    /// permission bits. What a read-only pass must leave as it was.
    fn tree(root: &Path) -> Vec<(PathBuf, bool, u64, std::time::SystemTime, u32)> {
        use std::os::unix::fs::PermissionsExt;
        let mut out = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            for e in std::fs::read_dir(&dir).unwrap() {
                let e = e.unwrap();
                let m = e.metadata().unwrap();
                let len = if m.is_dir() { 0 } else { m.len() };
                out.push((
                    e.path(),
                    m.is_dir(),
                    len,
                    m.modified().unwrap(),
                    m.permissions().mode(),
                ));
                if m.is_dir() {
                    stack.push(e.path());
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
                        Source::KeptAtExit { .. } => "unsaved",
                        Source::OffLine { .. } => "off",
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
        let h = resolve(&state_dir, &keys, &api, id, &mut HistoryCache::default())
            .await
            .unwrap();
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
        // As a runner before agentkit 0.158 left sealed/
        crate::sealed::version::drop_acknowledged_blobs(&b.seal()).await;
        b.api.calls.lock().unwrap().clear();
        let before = tree(b.root.path());
        let h = resolve(
            &b.state_dir(),
            &b.keys,
            &b.api,
            b.id,
            &mut HistoryCache::default(),
        )
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
        assert_eq!(
            h.versions[0].memory.as_deref().map(String::as_str),
            Some("")
        );
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
        let h = resolve(
            &b.state_dir(),
            &b.keys,
            &b.api,
            b.id,
            &mut HistoryCache::default(),
        )
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
        let h = resolve(
            &b.state_dir(),
            &b.keys,
            &b.api,
            b.id,
            &mut HistoryCache::default(),
        )
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
            let h = resolve(
                &b.state_dir(),
                &b.keys,
                &b.api,
                b.id,
                &mut HistoryCache::default(),
            )
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

    fn plain(saves: usize, upload: usize) -> Built {
        Built {
            saves,
            upload,
            rotate: false,
            mode: Mode::Plain,
            upload_rebased: false,
        }
    }

    /// Agora not serving blobs: each sealed version whose blob is pruned
    /// here stays listed, sealed, with its head and why it couldn't be
    /// opened, its local twin merged in rather than standing in for it.
    /// Served again, everything opens; and a session's cache spares the
    /// next call every fetch.
    #[tokio::test(start_paused = true)]
    async fn a_fetch_failure_keeps_sealed_versions_sealed_and_the_cache_spares_refetches() {
        let b = build(&plain(3, 9)).await;
        // As a runner before agentkit 0.158 left sealed/
        crate::sealed::version::drop_acknowledged_blobs(&b.seal()).await;
        *b.api.fail.lock().unwrap() = Some("blob");
        let mut cache = HistoryCache::default();
        let h = resolve(&b.state_dir(), &b.keys, &b.api, b.id, &mut cache)
            .await
            .unwrap();
        assert_eq!(h.versions.len(), 4, "{h:?}");
        let unopened: Vec<&Version> = h.versions.iter().filter(|v| v.unopened.is_some()).collect();
        assert!(!unopened.is_empty(), "some blobs were pruned: {h:?}");
        for v in &h.versions {
            assert!(
                matches!(v.sources[0], Source::Sealed { .. }),
                "never demoted to a local copy: {v:?}"
            );
        }
        for v in &unopened {
            assert!(v.memory.is_none(), "no content from the unsigned twin");
            assert!(v.twin_memory.is_some(), "the twin's memory names it");
            assert!(
                v.sources
                    .iter()
                    .any(|s| matches!(s, Source::LocalCopy { .. })),
                "its twin merged: {v:?}"
            );
            assert!(
                v.unopened
                    .as_ref()
                    .unwrap()
                    .contains("couldn't be opened just now")
            );
        }
        let tries = b
            .api
            .calls
            .lock()
            .unwrap()
            .iter()
            .filter(|c| *c == "blob")
            .count();
        // Three tries a blob, each blob once (a delta's keyframe is shared)
        assert_eq!(tries, 3 * cache.failed.count(), "three tries each");
        assert!(cache.failed.count() >= 1);
        // Asked again within the session: no fetch at all, the same answer
        b.api.calls.lock().unwrap().clear();
        let again = resolve(&b.state_dir(), &b.keys, &b.api, b.id, &mut cache)
            .await
            .unwrap();
        assert!(b.api.calls.lock().unwrap().is_empty(), "remembered");
        assert_eq!(
            again
                .versions
                .iter()
                .filter(|v| v.unopened.is_some())
                .count(),
            unopened.len()
        );
        // Served again, once the failures are old enough to try again
        *b.api.fail.lock().unwrap() = None;
        tokio::time::advance(crate::sealed::version::FAILED_FOR).await;
        b.api.calls.lock().unwrap().clear();
        let h = resolve(&b.state_dir(), &b.keys, &b.api, b.id, &mut cache)
            .await
            .unwrap();
        assert!(h.versions.iter().all(|v| v.unopened.is_none()), "{h:?}");
        assert_eq!(
            h.versions
                .iter()
                .map(Version::memory_str)
                .collect::<Vec<_>>(),
            ["", &memory(2), &memory(3), &memory(4)]
        );
        let fetched = b
            .api
            .calls
            .lock()
            .unwrap()
            .iter()
            .filter(|c| *c == "blob")
            .count();
        assert!(fetched > 0);
        // The cache: no fetch at all the next time
        b.api.calls.lock().unwrap().clear();
        let again = resolve(&b.state_dir(), &b.keys, &b.api, b.id, &mut cache)
            .await
            .unwrap();
        assert!(!b.api.calls.lock().unwrap().iter().any(|c| c == "blob"));
        assert_eq!(sources(&again), sources(&h));
        assert_eq!(cache.sizes().0, 4);
    }

    /// The pre-loss copy sealed on the main line: shown, its `content` as
    /// the memory.
    #[tokio::test]
    async fn an_archived_prior_is_on_the_main_line() {
        let b = build(&plain(1, 0)).await;
        let seal = b.seal();
        let k = b.keys.seal_keys(b.id).unwrap();
        {
            let lock = seal.lock().await.unwrap();
            crate::sealed::session::append(
                &seal,
                &lock,
                &k,
                &agora_agentkit::state::codec::KeyframePolicy::default(),
                StateEntryKind::ArchivedPrior,
                &serde_json::json!({ "content": "the pre-loss memory" }),
                crate::sealed::enroll::prior_copy_at(),
            )
            .await
            .unwrap();
        }
        let h = resolve(
            &b.state_dir(),
            &b.keys,
            &b.api,
            b.id,
            &mut HistoryCache::default(),
        )
        .await
        .unwrap();
        assert!(h.left_out.is_empty(), "{:?}", h.left_out);
        let prior = &h.versions[0];
        assert!(
            matches!(prior.sources[..], [Source::ArchivedPrior { .. }]),
            "{h:?}"
        );
        assert_eq!(prior.memory_str(), "the pre-loss memory");
        assert_eq!(
            prior.at,
            Some(crate::sealed::enroll::prior_copy_at()),
            "dated first"
        );
    }

    /// A `sent/*.lost` record with no `sealed/lost/<utc>/` left: still set
    /// aside, dated by its file.
    #[tokio::test]
    async fn a_lost_record_alone_still_resolves() {
        let b = build(&plain(2, 9)).await;
        let old = b.seal().head().await.unwrap().unwrap().head;
        let server = b.server();
        b.set_server(server[..server.len() - 1].to_vec());
        crate::sealed::accept::rebase(&b.state_dir(), &b.keys, &b.api, b.id, "lost")
            .await
            .unwrap();
        assert!(
            std::fs::read_dir(b.dir.join("sealed/sent"))
                .unwrap()
                .any(|e| e.unwrap().file_name().to_string_lossy().ends_with(".lost"))
        );
        std::fs::rename(b.dir.join("sealed/lost"), b.root.join("lost-elsewhere")).unwrap();
        let h = resolve(
            &b.state_dir(),
            &b.keys,
            &b.api,
            b.id,
            &mut HistoryCache::default(),
        )
        .await
        .unwrap();
        let HeadMatch::Found { version, source } = h.find_head(&old.entry_hash.to_hex()) else {
            panic!("{h:?}")
        };
        assert!(
            matches!(source, Source::SetAside { head, on: Some(_) } if *head == old),
            "{source:?}"
        );
        assert_eq!(version.memory_str(), memory(3));
    }

    /// A main line that doesn't verify: said, and the archives still shown
    /// as the local copies they are.
    #[tokio::test]
    async fn an_unverified_main_line_leaves_the_local_copies() {
        let b = build(&plain(2, 0)).await;
        // The newest item, re-signed by a key the runner doesn't trust
        let outbox = b.dir.join("sealed/outbox");
        let newest = b.seal().outbox().await.unwrap().pop().unwrap();
        let mut item = newest.clone();
        item.entry = item
            .entry
            .entry
            .clone()
            .sign(&agora_agentkit::crypto::generate_keypair().0);
        std::fs::write(
            outbox.join(format!("{:020}.json", item.ord)),
            serde_json::to_vec(&item).unwrap(),
        )
        .unwrap();
        let h = resolve(
            &b.state_dir(),
            &b.keys,
            &b.api,
            b.id,
            &mut HistoryCache::default(),
        )
        .await
        .unwrap();
        assert!(h.left_out.iter().any(|l| l.held == Held::MainLine), "{h:?}");
        assert!(
            h.versions.iter().all(|v| v.heads().next().is_none()),
            "{h:?}"
        );
        assert_eq!(
            h.versions
                .iter()
                .map(Version::memory_str)
                .collect::<Vec<_>>(),
            ["", &memory(2), &memory(3)]
        );
    }

    /// A signing-key rotation before the fork: the side line verifies with
    /// the rotation the runner recorded.
    #[tokio::test]
    async fn a_side_line_across_a_rotation_resolves() {
        let b = build(&Built {
            saves: 1,
            upload: 0,
            rotate: true,
            mode: Mode::Rebase,
            upload_rebased: false,
        })
        .await;
        assert!(b.rotated);
        let h = resolve(
            &b.state_dir(),
            &b.keys,
            &b.api,
            b.id,
            &mut HistoryCache::default(),
        )
        .await
        .unwrap();
        assert!(h.left_out.is_empty(), "{:?}", h.left_out);
        assert!(
            h.versions
                .iter()
                .flat_map(|v| &v.sources)
                .any(|s| matches!(s, Source::SideLine { .. })),
            "{h:?}"
        );
    }

    /// An entry held off the main line with no record of why (here: a
    /// displaced one filed as sent) is named neutrally, never "rebased".
    #[tokio::test]
    async fn an_entry_off_the_line_without_a_record_is_neutral() {
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
        let item: OutboxItem = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        std::fs::rename(
            &path,
            b.dir.join(format!("sealed/sent/{:020}.json", item.ord)),
        )
        .unwrap();
        let h = resolve(
            &b.state_dir(),
            &b.keys,
            &b.api,
            b.id,
            &mut HistoryCache::default(),
        )
        .await
        .unwrap();
        let HeadMatch::Found { source, .. } = h.find_head(&item.entry.entry_hash().to_hex()) else {
            panic!("{h:?}")
        };
        assert!(matches!(source, Source::OffLine { .. }), "{source:?}");
    }

    /// A server that never answers costs at most the resolver's deadline:
    /// each try is bounded, and past the deadline nothing more is fetched;
    /// what wasn't opened is listed unopened.
    #[tokio::test(start_paused = true)]
    async fn a_hanging_server_costs_at_most_the_deadline() {
        let b = build(&plain(3, 9)).await;
        // As a runner before agentkit 0.158 left sealed/
        crate::sealed::version::drop_acknowledged_blobs(&b.seal()).await;
        *b.api.blob_delay.lock().unwrap() = Some(std::time::Duration::from_secs(3600));
        let start = tokio::time::Instant::now();
        let h = resolve(
            &b.state_dir(),
            &b.keys,
            &b.api,
            b.id,
            &mut HistoryCache::default(),
        )
        .await
        .unwrap();
        assert!(
            start.elapsed() <= RESOLVE_DEADLINE + std::time::Duration::from_secs(1),
            "{:?}",
            start.elapsed()
        );
        assert!(h.versions.iter().any(|v| v.unopened.is_some()), "{h:?}");
        assert_eq!(h.versions.len(), 4, "every version still listed");
    }

    /// One outage costs a session one bounded wait: after it, Agora is
    /// taken as unreachable, and the next call costs nothing and asks
    /// nothing; every version is still listed.
    #[tokio::test(start_paused = true)]
    async fn a_hanging_server_costs_a_session_one_wait() {
        let b = build(&plain(3, 9)).await;
        // As a runner before agentkit 0.158 left sealed/
        crate::sealed::version::drop_acknowledged_blobs(&b.seal()).await;
        *b.api.blob_delay.lock().unwrap() = Some(std::time::Duration::from_secs(3600));
        let mut cache = HistoryCache::default();
        let start = tokio::time::Instant::now();
        resolve(&b.state_dir(), &b.keys, &b.api, b.id, &mut cache)
            .await
            .unwrap();
        let first = start.elapsed();
        assert!(first < RESOLVE_DEADLINE, "{first:?}");
        b.api.calls.lock().unwrap().clear();
        let start = tokio::time::Instant::now();
        let h = resolve(&b.state_dir(), &b.keys, &b.api, b.id, &mut cache)
            .await
            .unwrap();
        assert_eq!(start.elapsed(), std::time::Duration::ZERO);
        assert!(
            !b.api.calls.lock().unwrap().iter().any(|c| c == "blob"),
            "no GET"
        );
        assert_eq!(h.versions.len(), 4);
        assert!(h.versions.iter().any(|v| {
            v.unopened
                .as_deref()
                .is_some_and(|w| w.contains(crate::sealed::version::UNREACHABLE))
        }));
    }

    /// Persistent 503s on a long history: the breaker trips after a few
    /// tries, and the rest are listed unopened at once, their local twins
    /// merged in.
    #[tokio::test(start_paused = true)]
    async fn persistent_503s_on_a_long_history_cost_a_few_seconds() {
        let b = build(&plain(110, 200)).await;
        // As a runner before agentkit 0.158 left sealed/
        crate::sealed::version::drop_acknowledged_blobs(&b.seal()).await;
        *b.api.fail.lock().unwrap() = Some("blob");
        let mut cache = HistoryCache::default();
        let start = tokio::time::Instant::now();
        let h = resolve(&b.state_dir(), &b.keys, &b.api, b.id, &mut cache)
            .await
            .unwrap();
        let took = start.elapsed();
        assert!(took <= std::time::Duration::from_secs(2), "{took:?}");
        let gets = b
            .api
            .calls
            .lock()
            .unwrap()
            .iter()
            .filter(|c| *c == "blob")
            .count();
        assert_eq!(gets as u32, crate::sealed::version::BREAKER_AFTER);
        let unopened: Vec<&Version> = h.versions.iter().filter(|v| v.unopened.is_some()).collect();
        assert!(unopened.len() > 50, "{}", unopened.len());
        assert!(
            unopened.iter().all(|v| v.twin_memory.is_some()),
            "named by their twins"
        );
    }

    /// A 404 is the answer: asked once, never retried.
    #[tokio::test(start_paused = true)]
    async fn a_missing_blob_is_asked_for_once() {
        let b = build(&plain(3, 9)).await;
        // As a runner before agentkit 0.158 left sealed/
        crate::sealed::version::drop_acknowledged_blobs(&b.seal()).await;
        b.api.blobs.lock().unwrap().clear();
        let mut cache = HistoryCache::default();
        let h = resolve(&b.state_dir(), &b.keys, &b.api, b.id, &mut cache)
            .await
            .unwrap();
        assert!(h.versions.iter().any(|v| v.unopened.is_some()));
        let gets = b
            .api
            .calls
            .lock()
            .unwrap()
            .iter()
            .filter(|c| *c == "blob")
            .count();
        assert_eq!(gets, cache.failed.count(), "one GET a blob");
    }

    /// Agora's chain is asked once a session, however often the history
    /// is: a main line through Agora's entries, a side line too.
    #[tokio::test]
    async fn the_chain_is_asked_once_a_session() {
        let b = build(&Built {
            saves: 2,
            upload: 0,
            rotate: false,
            mode: Mode::Rebase,
            upload_rebased: false,
        })
        .await;
        let mut cache = HistoryCache::default();
        b.api.calls.lock().unwrap().clear();
        for _ in 0..3 {
            resolve(&b.state_dir(), &b.keys, &b.api, b.id, &mut cache)
                .await
                .unwrap();
        }
        let chains = b
            .api
            .calls
            .lock()
            .unwrap()
            .iter()
            .filter(|c| *c == "chain")
            .count();
        assert!(chains <= 1, "{chains}");
    }
}
