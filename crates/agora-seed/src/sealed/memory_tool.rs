//! The runner's `memory` tool (agora `memory/project_sealed_history_tools.md`):
//! `log`, `show`, `diff` and `search` over every version of an agent's
//! memory this runner holds, named by memory ids and sealed heads.
//!
//! The pieces are agentkit's (`reactor::seed::memory`: the input, the ids,
//! the renderers and the search index); this is the wiring: the session's
//! memory-id key ([`session_keys`]), the versions the resolver finds
//! ([`super::history::resolve`]) mapped onto agentkit's [`MemoryVersion`],
//! ids and heads resolved against both namespaces, and the index file
//! ([`super::history_index`]).
//!
//! **Honest about what it can't show.** A sealed version that couldn't be
//! opened just now, one off the main line with no record of why, and one
//! with no date are never passed to the renderers with empty or borrowed
//! content: they are listed after the result, by head and why. When the
//! main line can't be checked, the result says so first, and the local
//! copies are called unsigned.
//!
//! **Read-only but for the index.** Nothing here writes agent state; the
//! one file it writes is the search index, at the first `search` and when
//! a later one finds versions it lacks. Content is never logged: events
//! carry the action, counts and sizes.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use agora_agentkit::govlog::Sha256Hex;
use agora_agentkit::ids::{AgentId, GovernanceLogId};
use agora_agentkit::reactor::seed::memory::render::{self, LOCAL_ONLY};
use agora_agentkit::reactor::seed::memory::{
    Candidate, CurrentMemory, HistoryRef, MEMORY_TOOL_NAME, MemoryAction, MemoryId, MemoryIdKey,
    MemoryInput, MemoryVersion, Resolution, SearchIndex, VersionSource, log_limit, resolve,
    short_head, show_budget,
};
use agora_agentkit::reactor::seed::{ContextGauge, SeedAgent};
use chrono::{DateTime, Utc};
use misanthropic::prompt::message::Content;
use misanthropic::tool::{self, MethodDef, Tool, Use};
use zeroize::Zeroizing;

use super::KeyStore;
use super::api::SealedApi;
use super::history::{HeadMatch, Held, History, HistoryCache, Source, Version};
use super::keys::MemoryIdKeys;

/// What the runner needs of a session's agent for the `memory` tool: its
/// memory id, set from the key before `on_init`, and the gauge its results
/// are counted against. As [`crate::consent::agent::ContactRequests`] is
/// for the contact request.
pub trait MemoryIds {
    /// Name the session's memory by its id under `key`, so its heading
    /// shows it. Before `on_init`.
    fn set_memory_id_key(&mut self, key: &MemoryIdKey);
    /// The id of the memory in the agent's context, after `on_init`.
    fn memory_id(&self) -> Option<MemoryId>;
    /// The session's context gauge.
    fn context_gauge(&self) -> ContextGauge;
}

impl MemoryIds for SeedAgent {
    fn set_memory_id_key(&mut self, key: &MemoryIdKey) {
        SeedAgent::set_memory_id_key(self, key)
    }
    fn memory_id(&self) -> Option<MemoryId> {
        SeedAgent::memory_id(self)
    }
    fn context_gauge(&self) -> ContextGauge {
        SeedAgent::context_gauge(self).clone()
    }
}

/// The agent's memory-id keys for a session, made once if it has none
/// ([`KeyStore::memory_id_keys_or_create`]), off the async runtime.
/// `Err` where a key would be minted over evidence of one: the session
/// runs without ids.
pub async fn session_keys(
    keys: &KeyStore,
    id: AgentId,
    agent_dir: &Path,
) -> std::io::Result<MemoryIdKeys> {
    let (keys, dir) = (keys.clone(), agent_dir.to_path_buf());
    tokio::task::spawn_blocking(move || keys.memory_id_keys_or_create(id, &dir))
        .await
        .map_err(std::io::Error::other)?
}

/// What the tool says when the session has no memory-id key.
pub const NO_KEY: &str = "Your memory ids couldn't be worked out this session, so this tool \
     can't name your memory's versions. Your operator has been told.";

/// What the tool says when the history couldn't be read at all.
pub const UNREADABLE: &str = "Your memory's history couldn't be read just now. Ask again later.";

/// What opens a result when the main line couldn't be checked: in place
/// of agentkit's [`LOCAL_ONLY`], which would say the copies aren't sealed.
pub const UNCHECKED: &str = "Your sealed history couldn't be checked just now, so none of your \
     versions is shown as sealed and no head is given. These are your runner's local copies, \
     unsigned: nothing here vouches for them.";

/// What the first search says.
pub const INDEX_BUILT: &str = "This first search built your memory's search index; later \
     searches reuse it.";

/// What a search says when its index had to be built again.
pub const INDEX_REBUILT: &str = "This search rebuilt your memory's search index.";

/// `sealed/memory_ids_told.json`: the agent has been told, once, that
/// enrolment sealed its earlier versions and left its memory ids as they
/// were. Written when the session that told it saves, as anchor findings
/// are marked told. `sealed-forget` leaves it, as it leaves the rest of
/// `sealed/`.
pub const MEMORY_IDS_TOLD: &str = "memory_ids_told.json";

/// `<agent_dir>/sealed/memory_ids_told.json`.
pub fn memory_ids_told_path(agent_dir: &Path) -> PathBuf {
    agent_dir
        .join(agora_agentkit::state::storage::SEALED_DIR)
        .join(MEMORY_IDS_TOLD)
}

/// The sentence enrolment's first verdict adds, the earliest sealed
/// version's date in it.
pub fn enrollment_sentence(earliest: &str) -> String {
    format!(
        "Your earlier versions back to {earliest} are sealed now, so each has a head too. Your \
         memory ids haven't changed."
    )
}

/// Whether the agent is due the enrolment sentence: enrolled, its
/// memory-id key's fingerprint written before its `sealed/enrolled` marker
/// (it had memory ids before its versions were sealed; an agent enrolled
/// before it had a key has no ids to keep), and not told yet. Anything
/// unreadable is "not due": the sentence is never worth a guess.
fn enrollment_due(agent_dir: &Path) -> bool {
    let mtime = |p: PathBuf| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    let sealed = agent_dir.join(agora_agentkit::state::storage::SEALED_DIR);
    let (Some(enrolled), Some(fingerprint)) = (
        mtime(sealed.join("enrolled")),
        mtime(agent_dir.join(super::keys::MEMORY_ID_FINGERPRINT_FILE)),
    ) else {
        return false;
    };
    fingerprint < enrolled && !std::fs::exists(memory_ids_told_path(agent_dir)).unwrap_or(true)
}

/// The enrolment sentence, if the agent is due it
/// ([`enrollment_due`]), dated by its earliest sealed snapshot, as the
/// verdict's "since" is.
pub async fn enrollment_note(agent_dir: &Path, id: AgentId) -> Option<String> {
    let dir = agent_dir.to_path_buf();
    if !tokio::task::spawn_blocking(move || enrollment_due(&dir))
        .await
        .unwrap_or(false)
    {
        return None;
    }
    let seal = agora_agentkit::state::storage::AgentSeal::new(agent_dir, id);
    let entries = match super::check::local_entries(&seal).await {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!(
                agent_id = %id,
                error = %e,
                "sealed entries unreadable; the enrolment sentence waits"
            );
            return None;
        }
    };
    let earliest = entries
        .iter()
        .filter(|e| e.entry.kind == agora_agentkit::enums::StateEntryKind::Snapshot)
        .map(|e| e.entry.state_at)
        .min()
        .and_then(DateTime::<Utc>::from_timestamp_micros)?;
    Some(enrollment_sentence(
        &earliest.format("%Y-%m-%d").to_string(),
    ))
}

/// Mark the enrolment sentence told: once the session that showed it
/// saves. Atomic, 0600.
pub async fn mark_memory_ids_told(agent_dir: &Path) -> std::io::Result<()> {
    let path = memory_ids_told_path(agent_dir);
    let body = serde_json::to_vec_pretty(&serde_json::json!({ "told_at": Utc::now() }))?;
    tokio::task::spawn_blocking(move || super::write_private(&path, &body))
        .await
        .map_err(std::io::Error::other)?
}

/// One answer: its text, and whether it is a refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    pub text: String,
    pub error: bool,
}

impl Answer {
    fn ok(text: String) -> Self {
        Self { text, error: false }
    }
    fn refused(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            error: true,
        }
    }
}

/// A session's reader of one agent's memory history.
pub struct MemoryReader {
    state_dir: PathBuf,
    keys: KeyStore,
    id: AgentId,
    ids: Option<MemoryIdKeys>,
    /// The memory in the agent's context and its id (the heading's).
    current: Option<(MemoryId, Zeroizing<String>)>,
    cache: HistoryCache,
}

impl std::fmt::Debug for MemoryReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryReader")
            .field("id", &self.id)
            .field("keyed", &self.ids.is_some())
            .field("cache", &self.cache)
            .finish()
    }
}

impl MemoryReader {
    /// `ids`: the session's keys, `None` when they couldn't be had.
    /// `current`: the memory in context and its id, `None` with no key.
    pub fn new(
        state_dir: impl Into<PathBuf>,
        keys: KeyStore,
        id: AgentId,
        ids: Option<MemoryIdKeys>,
        current: Option<(MemoryId, &str)>,
    ) -> Self {
        Self {
            state_dir: state_dir.into(),
            keys,
            id,
            ids,
            current: current.map(|(m, c)| (m, Zeroizing::new(c.to_string()))),
            cache: HistoryCache::default(),
        }
    }

    fn agent_dir(&self) -> PathBuf {
        self.state_dir.join(self.id.to_string())
    }

    /// Answer one request.
    pub async fn answer(&mut self, action: MemoryAction, api: &dyn SealedApi) -> Answer {
        let what = match &action {
            MemoryAction::Log { .. } => "log",
            MemoryAction::Show { .. } => "show",
            MemoryAction::Diff { .. } => "diff",
            MemoryAction::Search { .. } => "search",
        };
        let (Some(ids), Some((current_id, current))) = (&self.ids, &self.current) else {
            tracing::warn!(
                event_type = "memory_tool_no_key",
                agent_id = %self.id,
                action = what,
                "memory tool called without a memory-id key"
            );
            return Answer::refused(NO_KEY);
        };
        let (current_id, current) = (*current_id, current.clone());
        let history = match super::history::resolve(
            &self.state_dir,
            &self.keys,
            api,
            self.id,
            &mut self.cache,
        )
        .await
        {
            Ok(h) => h,
            Err(e) => {
                tracing::warn!(
                    event_type = "memory_tool_unreadable",
                    agent_id = %self.id,
                    action = what,
                    error = %e,
                    "the memory history couldn't be resolved"
                );
                return Answer::refused(UNREADABLE);
            }
        };
        let anchors = anchored(&self.agent_dir()).await;
        let view = View::new(history, ids, self.id, &anchors);
        let current = CurrentMemory::new(current_id, &current, &view.rows);
        let text = match action {
            MemoryAction::Log { before, limit } => view.log(before.as_ref(), limit, &current),
            MemoryAction::Show { id, budget } => view.show(id.as_slice(), budget, current),
            MemoryAction::Diff { a, b } => view.diff(&a, b.as_ref(), current),
            MemoryAction::Search { query, mode } => {
                let found = search(&self.agent_dir(), self.id, &view.rows, &query).await;
                let mut out = view.preface();
                if let Some(said) = found.said {
                    push_block(&mut out, said);
                }
                push_block(&mut out, &found.result.render(mode));
                let missing = view.not_shown();
                if !missing.is_empty() {
                    push_block(&mut out, &format!("Not searched:\n{}", missing.join("\n")));
                }
                out
            }
        };
        tracing::info!(
            event_type = "memory_tool_call",
            agent_id = %self.id,
            action = what,
            versions = view.rows.len(),
            not_shown = view.not_shown().len(),
            unchecked = view.unchecked,
            cached_sealed = self.cache.sizes().0,
            cached_files = self.cache.sizes().1,
            bytes = text.len(),
            "memory tool answered"
        );
        Answer::ok(text)
    }
}

/// Each head the inclusion record shows in an anchor, with the first
/// anchor that holds it. Only a held version Agora's proof checked
/// ([`super::inclusion::Finding::Holds`] and `HoldsUnchecked`): never one
/// an anchor was found to hold wrongly.
async fn anchored(agent_dir: &Path) -> HashMap<Sha256Hex, GovernanceLogId> {
    let dir = agent_dir.to_path_buf();
    let loaded = tokio::task::spawn_blocking(move || super::inclusion::InclusionFile::load(&dir))
        .await
        .map_err(std::io::Error::other)
        .and_then(|r| r);
    let file = match loaded {
        Ok(file) => file,
        Err(e) => {
            tracing::warn!(
                event_type = "memory_tool_inclusion_unreadable",
                path = %agent_dir.display(),
                error = %e,
                "sealed/inclusion.json unreadable; no version is said to be anchored"
            );
            return HashMap::new();
        }
    };
    let mut out = HashMap::new();
    for c in file.checked {
        if let super::inclusion::Finding::Holds { held, .. }
        | super::inclusion::Finding::HoldsUnchecked { held, .. } = c.finding
        {
            out.entry(held.entry_hash).or_insert(c.anc_id);
        }
    }
    out
}

/// A version's place in agentkit's terms, by the source it is listed
/// under (sealed on the main line first). `None` for one agentkit has no
/// label for yet: off the main line with no record of why.
fn version_source(source: &Source) -> Option<VersionSource> {
    Some(match source {
        Source::Sealed { .. } | Source::ArchivedPrior { .. } => VersionSource::Sealed,
        Source::SideLine { .. } => VersionSource::SideLine,
        Source::SetAside { on: Some(on), .. } => VersionSource::SetAside {
            on: on.date_naive(),
        },
        // Set aside by a rebase, on a day not recorded
        Source::SetAside { on: None, .. } => VersionSource::SideLine,
        Source::OffLine { .. } => return None,
        Source::LocalCopy { .. } | Source::Current => VersionSource::LocalCopy,
        Source::KeptAtExit { .. } => VersionSource::NeverSaved,
    })
}

fn stamp(at: Option<DateTime<Utc>>) -> String {
    at.map_or("date unknown".into(), |at| {
        at.format("%Y-%m-%d %H:%M").to_string()
    })
}

fn push_block(out: &mut String, block: &str) {
    if block.is_empty() {
        return;
    }
    if !out.is_empty() {
        out.push_str("\n\n");
    }
    out.push_str(block);
}

/// Where a ref led: a row, with notes, or only words.
enum Pick {
    At(usize, Vec<String>),
    Say(String),
}

/// One call's history, as the renderers take it.
struct View {
    history: History,
    /// The versions the renderers are given, newest first.
    rows: Vec<MemoryVersion>,
    /// `history.versions`' index → its row, if it has one.
    placed: Vec<Option<usize>>,
    /// Each row's memory id under every key (the current first), with its
    /// row, newest first: ids a replaced key named still resolve.
    ids: Vec<(MemoryId, usize)>,
    /// The main line couldn't be checked.
    unchecked: bool,
}

impl View {
    fn new(
        history: History,
        keys: &MemoryIdKeys,
        agent: AgentId,
        anchors: &HashMap<Sha256Hex, GovernanceLogId>,
    ) -> Self {
        let unchecked = history.left_out.iter().any(|l| l.held == Held::MainLine);
        let mut rows: Vec<(usize, MemoryVersion)> = Vec::new();
        for (i, v) in history.versions.iter().enumerate() {
            let (None, Some(at), Some(source)) = (
                &v.unopened,
                v.at,
                v.sources.first().and_then(version_source),
            ) else {
                continue;
            };
            let content = v.memory_str();
            rows.push((
                i,
                MemoryVersion {
                    id: MemoryId::of(&keys.current, agent, content),
                    head: v.sources[0].head().map(|h| h.entry_hash),
                    at,
                    source,
                    anchored: v.heads().find_map(|h| anchors.get(&h.entry_hash).cloned()),
                    content: content.to_string(),
                },
            ));
        }
        // Oldest first from the resolver: newest first for the renderers
        rows.reverse();
        let mut placed = vec![None; history.versions.len()];
        for (row, (i, _)) in rows.iter().enumerate() {
            placed[*i] = Some(row);
        }
        let rows: Vec<MemoryVersion> = rows.into_iter().map(|(_, v)| v).collect();
        let mut ids = Vec::new();
        for key in keys.all() {
            for (row, v) in rows.iter().enumerate() {
                ids.push((MemoryId::of(key, agent, &v.content), row));
            }
        }
        Self {
            history,
            rows,
            placed,
            ids,
            unchecked,
        }
    }

    /// What opens every result: that the main line couldn't be checked.
    fn preface(&self) -> String {
        if self.unchecked {
            UNCHECKED.to_string()
        } else {
            String::new()
        }
    }

    /// The versions held but not given to the renderers, one line each.
    fn not_shown(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (i, v) in self.history.versions.iter().enumerate().rev() {
            if self.placed[i].is_some() {
                continue;
            }
            let head = v
                .heads()
                .next()
                .map(|h| format!("head {}", short_head(&h.entry_hash)))
                .unwrap_or_else(|| "a copy with no head".into());
            let why = if v.unopened.is_some() {
                "sealed, but it couldn't be opened just now; ask again later"
            } else if v.at.is_none() {
                "its date is unknown"
            } else {
                "sealed off your main line, with no record of why"
            };
            out.push(format!("- {head} ({}): {why}", stamp(v.at)));
        }
        out
    }

    /// Every head held here: each version's and each left out.
    fn heads(&self) -> Vec<Sha256Hex> {
        self.history
            .versions
            .iter()
            .flat_map(Version::heads)
            .chain(self.history.left_out.iter().filter_map(|l| match &l.held {
                Held::Entry { head, .. } => Some(*head),
                _ => None,
            }))
            .map(|h| h.entry_hash)
            .collect()
    }

    /// Where `r` leads, among both namespaces: a memory id (under any key)
    /// or any head held here.
    fn pick(&self, r: &HistoryRef) -> Pick {
        let heads = self.heads();
        let resolution = resolve(r, self.ids.iter().map(|(id, _)| id), heads.iter());
        let notes: Vec<String> = resolution.note(r).into_iter().collect();
        match resolution.candidate() {
            Some(Candidate::Memory(id)) => match self.ids.iter().find(|(m, _)| *m == id) {
                Some((_, row)) => Pick::At(*row, notes),
                None => Pick::Say(not_found(r)),
            },
            Some(Candidate::Head(head)) => self.at_head(&head, notes),
            None => match resolution {
                Resolution::Ambiguous(_) => {
                    Pick::Say(resolution.ambiguity(r).unwrap_or_else(|| not_found(r)))
                }
                _ => Pick::Say(not_found(r)),
            },
        }
    }

    /// The row holding `head`, with a note when it is off the main line;
    /// why not, for one held but not shown.
    fn at_head(&self, head: &Sha256Hex, mut notes: Vec<String>) -> Pick {
        let short = short_head(head);
        match self.history.find_head(&head.to_hex()) {
            HeadMatch::Found { version, source } => {
                let i = self
                    .history
                    .versions
                    .iter()
                    .position(|v| std::ptr::eq(v, version));
                let Some(row) = i.and_then(|i| self.placed[i]) else {
                    return Pick::Say(if version.unopened.is_some() {
                        format!(
                            "Head {short} is sealed, but its version couldn't be opened just \
                             now; ask again later."
                        )
                    } else if version.at.is_none() {
                        format!(
                            "Head {short} is held by your runner, but its date is unknown, so \
                             it can't be placed among your versions."
                        )
                    } else {
                        format!(
                            "Head {short} is sealed off your main line, with no record of why; \
                             it can't be shown yet."
                        )
                    });
                };
                let id = self.rows[row].id;
                match source {
                    Source::SideLine { .. } => notes.push(format!(
                        "Head {short} is on a side line your operator rebased; its memory is \
                         {id}."
                    )),
                    Source::SetAside { on, .. } => notes.push(format!(
                        "Head {short} was set aside{} when your line was rebased; its memory \
                         is {id}.",
                        on.map(|d| format!(" on {}", d.date_naive()))
                            .unwrap_or_default()
                    )),
                    Source::OffLine { .. } => notes.push(format!(
                        "Head {short} is sealed off your main line, with no record of why; its \
                         memory is {id}."
                    )),
                    _ => {}
                }
                Pick::At(row, notes)
            }
            HeadMatch::Held(left) => Pick::Say(format!(
                "Head {short} is held by your runner, but can't be shown: {}.",
                left.why
            )),
            HeadMatch::Ambiguous(heads) => Pick::Say(format!(
                "\"{short}\" could mean any of these heads; give more digits:\n{}",
                heads
                    .iter()
                    .map(|h| format!("- head {}", short_head(&h.entry_hash)))
                    .collect::<Vec<_>>()
                    .join("\n")
            )),
            HeadMatch::NotHeld | HeadMatch::NotAHead => {
                Pick::Say(format!("Head {short} isn't held by your runner."))
            }
        }
    }

    fn log(
        &self,
        before: Option<&HistoryRef>,
        limit: Option<u32>,
        current: &CurrentMemory<'_>,
    ) -> String {
        let mut notes = Vec::new();
        let start = match before {
            None => 0,
            Some(r) => match render::log_start(&self.rows, r) {
                Ok((start, note)) => {
                    notes.extend(note);
                    start
                }
                Err(_) => match self.pick(r) {
                    Pick::At(row, more) => {
                        notes.extend(more);
                        row + 1
                    }
                    Pick::Say(said) => return self.compose(&[said], ""),
                },
            },
        };
        let mut body = render::log(&self.rows, start, log_limit(limit), current);
        if self.unchecked
            && let Some(rest) = body.strip_prefix(LOCAL_ONLY)
        {
            body = rest.trim_start().to_string();
        }
        let missing = self.not_shown();
        if !missing.is_empty() {
            body.push_str(&format!(
                "\n\nHeld but not listed above:\n{}",
                missing.join("\n")
            ));
        }
        self.compose(&notes, &body)
    }

    fn show(&self, refs: &[HistoryRef], budget: Option<u32>, current: CurrentMemory<'_>) -> String {
        let mut notes = Vec::new();
        let mut picked = Vec::new();
        for r in refs {
            match self.pick(r) {
                Pick::At(row, more) => {
                    notes.extend(more);
                    picked.push(row);
                }
                Pick::Say(said) => notes.push(said),
            }
        }
        if picked.is_empty() {
            return self.compose(&notes, "");
        }
        let body = render::show(&self.rows, &picked, current, show_budget(budget));
        self.compose(&notes, &body)
    }

    fn diff(&self, a: &HistoryRef, b: Option<&HistoryRef>, current: CurrentMemory<'_>) -> String {
        let mut notes = Vec::new();
        let row = |r: &HistoryRef, notes: &mut Vec<String>| match self.pick(r) {
            Pick::At(row, more) => {
                notes.extend(more);
                Some(row)
            }
            Pick::Say(said) => {
                notes.push(said);
                None
            }
        };
        let a = row(a, &mut notes);
        let b = match b {
            Some(b) => row(b, &mut notes).map(Some),
            None => Some(None),
        };
        let (Some(a), Some(b)) = (a, b) else {
            return self.compose(&notes, "");
        };
        let body = render::diff(&self.rows[a], b.map(|b| &self.rows[b]), current);
        self.compose(&notes, &body)
    }

    /// The preface, the notes, then the body.
    fn compose(&self, notes: &[String], body: &str) -> String {
        let mut out = self.preface();
        for note in notes {
            push_block(&mut out, note);
        }
        push_block(&mut out, body);
        out
    }
}

fn not_found(r: &HistoryRef) -> String {
    format!(
        "No version of your memory has the id or head \"{r}\". `log` lists them, with their \
         memory ids and heads."
    )
}

/// A search's result, and what it says of the index.
struct Found {
    result: agora_agentkit::reactor::seed::memory::SearchResult,
    said: Option<&'static str>,
}

/// Search the index, building it on the first search, adding the versions
/// it lacks, and building it again when it can't be read or names a memory
/// no longer held. Written only when it changed. A write that fails is
/// logged; the search is still answered.
async fn search(agent_dir: &Path, id: AgentId, rows: &[MemoryVersion], query: &str) -> Found {
    let dir = agent_dir.to_path_buf();
    let read = tokio::task::spawn_blocking(move || super::history_index::read(&dir))
        .await
        .map_err(std::io::Error::other)
        .and_then(|r| r);
    let (mut index, said) = match read {
        Ok(None) => (SearchIndex::new(), Some(INDEX_BUILT)),
        Ok(Some(bytes)) => match SearchIndex::from_bytes(&bytes) {
            Ok(index) => (index, None),
            Err(e) => {
                tracing::warn!(
                    event_type = "memory_index_rebuilt",
                    agent_id = %id,
                    reason = %e,
                    "the memory search index couldn't be used; rebuilt"
                );
                (SearchIndex::new(), Some(INDEX_REBUILT))
            }
        },
        Err(e) => {
            tracing::warn!(
                event_type = "memory_index_rebuilt",
                agent_id = %id,
                reason = "unreadable",
                error = %e,
                "the memory search index couldn't be read; rebuilt"
            );
            (SearchIndex::new(), Some(INDEX_REBUILT))
        }
    };
    let held: HashSet<MemoryId> = rows.iter().map(|v| v.id).collect();
    let (mut index, said) = if index.ids().any(|m| !held.contains(m)) {
        tracing::info!(
            event_type = "memory_index_rebuilt",
            agent_id = %id,
            reason = "stale",
            "the memory search index names memories no longer held; rebuilt"
        );
        (SearchIndex::new(), said.or(Some(INDEX_REBUILT)))
    } else {
        (std::mem::take(&mut index), said)
    };
    let mut changed = said.is_some();
    for v in rows.iter().rev() {
        changed |= index.add(v.id, v.at, &v.content);
    }
    if changed {
        let bytes = index.to_bytes();
        let size = bytes.len();
        let dir = agent_dir.to_path_buf();
        let written =
            tokio::task::spawn_blocking(move || super::history_index::write(&dir, &bytes))
                .await
                .map_err(std::io::Error::other)
                .and_then(|r| r);
        match written {
            Ok(()) => tracing::info!(
                event_type = "memory_index_written",
                agent_id = %id,
                versions = index.len(),
                memories = index.memories(),
                bytes = size,
                "memory search index written"
            ),
            Err(e) => tracing::warn!(
                event_type = "memory_index_write_failed",
                agent_id = %id,
                error = %e,
                "the memory search index couldn't be written; built again next search"
            ),
        }
    }
    Found {
        result: index.search(query),
        said,
    }
}

/// The `memory` tool: a [`MemoryReader`] over the session's Agora client.
/// Seat it wrapped in agentkit's `Gauged`, so each result is counted
/// against the session's context, as the Agora tools' are.
pub struct MemoryTool {
    reader: MemoryReader,
    api: Arc<dyn SealedApi>,
}

impl MemoryTool {
    pub fn new(reader: MemoryReader, api: Arc<dyn SealedApi>) -> Self {
        Self { reader, api }
    }
}

#[async_trait::async_trait]
impl Tool for MemoryTool {
    fn name(&self) -> &str {
        MEMORY_TOOL_NAME
    }

    fn definitions(&self) -> Vec<MethodDef> {
        vec![MemoryInput::definition()]
    }

    async fn call(&mut self, call: Use) -> tool::Result {
        let answer = match MemoryInput::parse(call.input) {
            Ok(action) => self.reader.answer(action, self.api.as_ref()).await,
            Err(e) => Answer::refused(e.to_string()),
        };
        let result = tool::Result::new(call.id, Content::from(answer.text));
        if answer.error { result.error() } else { result }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sealed::api::fake::FakeApi;
    use crate::sealed::props::{History as Built, Mode, build, version};
    use agora_agentkit::one_or_many::OneOrMany;
    use agora_agentkit::reactor::seed::memory::MemoryAction as A;
    use agora_agentkit::reactor::{FsStorage, Storage};
    use agora_agentkit::state::storage::OutboxItem;

    fn keys(b: u8) -> MemoryIdKeys {
        MemoryIdKeys {
            current: MemoryIdKey::from_bytes(Zeroizing::new([b; 32])),
            past: Vec::new(),
        }
    }

    fn memory(n: usize) -> String {
        version(n)["memory"]["content"]
            .as_str()
            .unwrap()
            .to_string()
    }

    /// A reader for `id`, its current memory `memory(now)`, keyed with
    /// `keys(1)` (and `past`, if any).
    fn reader(
        state_dir: &Path,
        ks: &KeyStore,
        id: AgentId,
        now: usize,
        past: Option<u8>,
    ) -> MemoryReader {
        let mut k = keys(1);
        if let Some(p) = past {
            k.past.push(keys(p).current);
        }
        let current = MemoryId::of(&k.current, id, &memory(now));
        MemoryReader::new(
            state_dir,
            ks.clone(),
            id,
            Some(k),
            Some((current, &memory(now))),
        )
    }

    fn mem(id: AgentId, n: usize) -> MemoryId {
        MemoryId::of(&keys(1).current, id, &memory(n))
    }

    fn r(s: impl std::fmt::Display) -> HistoryRef {
        s.to_string().parse().unwrap()
    }

    fn log() -> A {
        A::Log {
            before: None,
            limit: None,
        }
    }

    fn show(ids: Vec<HistoryRef>) -> A {
        A::Show {
            id: OneOrMany::Many(ids),
            budget: None,
        }
    }

    fn search(q: &str) -> A {
        A::Search {
            query: q.into(),
            mode: None,
        }
    }

    /// Every file under `root`, by path, size and mode.
    fn files(root: &Path) -> Vec<(PathBuf, u64)> {
        let mut out = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            for e in std::fs::read_dir(&dir).unwrap() {
                let e = e.unwrap();
                let m = e.metadata().unwrap();
                if m.is_dir() {
                    stack.push(e.path());
                } else {
                    out.push((e.path(), m.len()));
                }
            }
        }
        out.sort();
        out
    }

    /// An agent never enrolled: `log` lists its local copies; `show`,
    /// `diff` by memory id; `search` builds its index once beside its
    /// state (never making `sealed/`), then loads it; an id a replaced key
    /// named still resolves; nothing else is written.
    #[tokio::test]
    async fn a_local_only_agent_end_to_end() {
        let root = crate::sealed::TestDir::new("memory-tool-local");
        let state_dir = root.join("state");
        let id = AgentId::new();
        let mut fs = FsStorage::new(&state_dir);
        for n in [1, 2, 3] {
            fs.save_raw(id, version(n), None).await.unwrap();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let ks = KeyStore::new(root.join("secrets"));
        let api = FakeApi::new();
        let mut rd = reader(&state_dir, &ks, id, 3, Some(9));
        let before = files(root.path());

        let out = rd.answer(log(), &api).await;
        assert!(!out.error);
        assert!(out.text.starts_with(LOCAL_ONLY), "{}", out.text);
        assert!(out.text.contains(&format!("{}  memory", mem(id, 3))));
        assert!(out.text.contains("← this session"), "{}", out.text);
        assert!(!out.text.contains("head "), "{}", out.text);

        let out = rd.answer(show(vec![r(mem(id, 1))]), &api).await.text;
        assert!(out.contains(&memory(1).trim_end().to_string()), "{out}");
        let out = rd
            .answer(
                A::Diff {
                    a: r(mem(id, 2)),
                    b: None,
                },
                &api,
            )
            .await
            .text;
        assert!(out.contains("→ your current memory"), "{out}");
        assert!(out.contains("  + synthetic version 3"), "{out}");
        // An id the replaced key named still resolves
        let old = MemoryId::of(&keys(9).current, id, &memory(1));
        let out = rd.answer(show(vec![r(old)]), &api).await.text;
        assert!(out.contains(&memory(1).trim_end().to_string()), "{out}");
        let out = rd.answer(show(vec![r("mem-abcdef")]), &api).await.text;
        assert!(out.starts_with("No version of your memory"), "{out}");
        assert_eq!(files(root.path()), before, "read-only but for search");
        assert!(api.calls.lock().unwrap().is_empty(), "no network");

        // Search: built once, beside the state, 0600, no sealed/
        let dir = state_dir.join(id.to_string());
        let out = rd.answer(search("version"), &api).await.text;
        assert!(out.starts_with(INDEX_BUILT), "{out}");
        assert!(out.contains("synthetic version 3"), "{out}");
        let index = super::super::history_index::unsealed_path(&dir);
        assert!(index.exists());
        assert!(!dir.join("sealed").exists(), "never makes sealed/");
        assert!(!crate::sealed::is_enrolled(&dir));
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&index).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        let written = std::fs::metadata(&index).unwrap().modified().unwrap();
        let out = rd.answer(search("line two"), &api).await.text;
        assert!(
            !out.contains(INDEX_BUILT) && !out.contains(INDEX_REBUILT),
            "{out}"
        );
        assert_eq!(
            std::fs::metadata(&index).unwrap().modified().unwrap(),
            written,
            "loaded, not rewritten"
        );
        // A new version is added; a damaged index is rebuilt
        std::thread::sleep(std::time::Duration::from_millis(5));
        fs.save_raw(id, version(4), None).await.unwrap();
        let out = rd.answer(search("synthetic 4"), &api).await.text;
        assert!(out.contains("synthetic version 4"), "{out}");
        assert!(!out.contains(INDEX_BUILT), "{out}");
        std::fs::write(&index, b"{ not an index").unwrap();
        let out = rd.answer(search("synthetic"), &api).await.text;
        assert!(out.starts_with(INDEX_REBUILT), "{out}");
    }

    /// An enrolled agent with a rebased fork: rows carry heads; the
    /// displaced head opens the version it holds, saying it is on a side
    /// line and naming its memory id; a `mem-` with a head's digits falls
    /// back to the head, with agentkit's note.
    #[tokio::test]
    async fn an_enrolled_agent_with_a_displaced_head() {
        let b = build(&Built {
            saves: 2,
            upload: 0,
            rotate: false,
            mode: Mode::Rebase,
            upload_rebased: false,
        })
        .await;
        let displaced: OutboxItem = std::fs::read_dir(b.dir.join("sealed/outbox"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| p.extension().is_some_and(|x| x == "displaced"))
            .map(|p| serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap())
            .unwrap();
        let head = displaced.entry.entry_hash();
        let current = FsStorage::new(b.state_dir())
            .load_raw(b.id)
            .await
            .unwrap()
            .value;
        let now = current["soul"]["n"].as_u64().unwrap() as usize;
        let mut rd = reader(&b.state_dir(), &b.keys, b.id, now, None);
        let out = rd.answer(log(), &b.api).await.text;
        assert!(out.contains("  head "), "{out}");
        assert!(!out.starts_with(LOCAL_ONLY), "{out}");
        let short = short_head(&head);
        let out = rd.answer(show(vec![r(&short)]), &b.api).await.text;
        assert!(
            out.starts_with(&format!(
                "Head {short} is on a side line your operator rebased; its memory is mem-"
            )),
            "{out}"
        );
        let out = rd
            .answer(show(vec![r(format!("mem-{short}"))]), &b.api)
            .await
            .text;
        assert!(
            out.contains("matches no memory id, but it is the start of head"),
            "{out}"
        );
    }

    /// A sealed version that can't be opened is listed by head and why,
    /// never shown with borrowed or empty content; asked for, it says so.
    #[tokio::test(start_paused = true)]
    async fn an_unopened_version_is_listed_not_shown() {
        let b = build(&Built {
            saves: 3,
            upload: 9,
            rotate: false,
            mode: Mode::Plain,
            upload_rebased: false,
        })
        .await;
        b.seal().prune_blobs().await.unwrap();
        *b.api.fail.lock().unwrap() = Some("blob");
        let mut rd = reader(&b.state_dir(), &b.keys, b.id, 4, None);
        let mut cache = HistoryCache::default();
        let h = super::super::history::resolve(&b.state_dir(), &b.keys, &b.api, b.id, &mut cache)
            .await
            .unwrap();
        let unopened: Vec<&Version> = h.versions.iter().filter(|v| v.unopened.is_some()).collect();
        assert!(!unopened.is_empty());
        let out = rd.answer(log(), &b.api).await.text;
        assert!(out.contains("Held but not listed above:"), "{out}");
        assert!(out.contains("couldn't be opened just now"), "{out}");
        for v in &unopened {
            let short = short_head(&v.heads().next().unwrap().entry_hash);
            assert!(out.contains(&format!("- head {short} (")), "{out}");
            let said = rd.answer(show(vec![r(&short)]), &b.api).await.text;
            assert_eq!(
                said,
                format!(
                    "Head {short} is sealed, but its version couldn't be opened just now; ask \
                     again later."
                )
            );
            assert!(!said.contains("synthetic"), "no content: {said}");
        }
    }

    /// A main line that can't be checked: said first, in place of "not
    /// sealed", and no head is given.
    #[tokio::test]
    async fn an_unchecked_main_line_is_said() {
        let b = build(&Built {
            saves: 2,
            upload: 0,
            rotate: false,
            mode: Mode::Plain,
            upload_rebased: false,
        })
        .await;
        let newest = b.seal().outbox().await.unwrap().pop().unwrap();
        let mut item = newest.clone();
        item.entry = item
            .entry
            .entry
            .clone()
            .sign(&agora_agentkit::crypto::generate_keypair().0);
        std::fs::write(
            b.dir.join(format!("sealed/outbox/{:020}.json", item.ord)),
            serde_json::to_vec(&item).unwrap(),
        )
        .unwrap();
        let mut rd = reader(&b.state_dir(), &b.keys, b.id, 3, None);
        let out = rd.answer(log(), &b.api).await.text;
        assert!(out.starts_with(UNCHECKED), "{out}");
        assert!(!out.contains(LOCAL_ONLY), "{out}");
        assert!(!out.contains("  head "), "{out}");
        let out = rd.answer(search("synthetic"), &b.api).await.text;
        assert!(out.starts_with(UNCHECKED), "{out}");
    }

    /// `anchored` only for a head the inclusion record shows held.
    #[tokio::test]
    async fn anchored_only_where_the_inclusion_record_holds_it() {
        use super::super::inclusion::{Checked, Finding, InclusionFile};
        let b = build(&Built {
            saves: 2,
            upload: 9,
            rotate: false,
            mode: Mode::Plain,
            upload_rebased: false,
        })
        .await;
        let line = super::super::check::local_entries(&b.seal()).await.unwrap();
        let held = agora_agentkit::state::verify::HeadRef {
            entry_hash: line[2].entry_hash(),
            seq: line[2].entry.seq,
        };
        let wrong = agora_agentkit::state::verify::HeadRef {
            entry_hash: line[3].entry_hash(),
            seq: line[3].entry.seq,
        };
        let checked = |anc: &str, finding| Checked {
            anc_id: anc.parse().unwrap(),
            date: None,
            finding,
            at: Utc::now(),
            told: true,
            alerted: false,
        };
        InclusionFile {
            through: Some("ANC-2026-0004".parse().unwrap()),
            checked: vec![
                checked(
                    "ANC-2026-0003",
                    Finding::Holds {
                        held,
                        in_flight: false,
                    },
                ),
                checked(
                    "ANC-2026-0004",
                    Finding::Later {
                        held: wrong,
                        sealed_after: true,
                    },
                ),
            ],
        }
        .save(&b.dir)
        .unwrap();
        let mut rd = reader(&b.state_dir(), &b.keys, b.id, 3, None);
        let out = rd.answer(log(), &b.api).await.text;
        assert_eq!(out.matches("anchored in ").count(), 1, "{out}");
        assert!(
            out.contains(&format!("head {}  memory", short_head(&held.entry_hash)))
                || out.contains(&short_head(&held.entry_hash))
        );
        let row = out
            .lines()
            .find(|l| l.contains(&short_head(&held.entry_hash)))
            .unwrap();
        assert!(row.contains("anchored in ANC-2026-0003"), "{row}");
    }

    /// Through `Gauged`: the result is counted against the session's
    /// context; a call that can't be read says what the tool takes; with
    /// no key the tool refuses, plainly.
    #[tokio::test]
    async fn the_gauge_is_charged_and_refusals_are_plain() {
        use agora_agentkit::reactor::seed::Gauged;
        let root = crate::sealed::TestDir::new("memory-tool-gauge");
        let state_dir = root.join("state");
        let id = AgentId::new();
        let mut fs = FsStorage::new(&state_dir);
        fs.save_raw(id, version(1), None).await.unwrap();
        let ks = KeyStore::new(root.join("secrets"));
        let gauge = ContextGauge::default();
        let api: Arc<dyn SealedApi> = Arc::new(FakeApi::new());
        let mut tool = Gauged::new(
            MemoryTool::new(reader(&state_dir, &ks, id, 1, None), api.clone()),
            gauge.clone(),
        );
        assert_eq!(tool.definitions(), vec![MemoryInput::definition()]);
        let call = |input: serde_json::Value| Use::new(MEMORY_TOOL_NAME, input).with_id("toolu_1");
        let before = gauge.get();
        let result = tool
            .call(call(serde_json::json!({"request": {"action": "log"}})))
            .await;
        assert!(!result.is_error);
        assert!(gauge.get() > before, "counted");
        let result = tool
            .call(call(
                serde_json::json!({"command": "view", "path": "/memories"}),
            ))
            .await;
        assert!(result.is_error);

        let mut unkeyed = MemoryTool::new(MemoryReader::new(&state_dir, ks, id, None, None), api);
        let result = unkeyed
            .call(call(serde_json::json!({"request": {"action": "log"}})))
            .await;
        assert!(result.is_error);
        assert!(format!("{:?}", result.content).contains("couldn't be worked out"));
    }
}
