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
//! opened just now is a row of its own, unopened, never given a local
//! twin's content or an empty one; one off the main line or with no date
//! is labelled so (agentkit's labels). When the main line can't be
//! checked, only the runner's own copies are rows, and agentkit says why
//! there are no heads ([`SealCheck::Unchecked`]).
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
use agora_agentkit::reactor::seed::memory::render;
use agora_agentkit::reactor::seed::memory::{
    Candidate, CurrentMemory, HistoryRef, IndexError, IndexState, MEMORY_TOOL_NAME, MemoryAction,
    MemoryId, MemoryIdKey, MemoryInput, MemoryVersion, Resolution, SealCheck, SearchIndex,
    UNCHECKED, VersionSource, VersionText, log_limit, newest_first, resolve, search_limit,
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
/// are counted against. As [`crate::consent::agent::InnerSession`] is
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

/// Told once, in the session that makes the agent's memory-id key (the
/// first after the `memory` tool ships), so a new tool reads as new.
pub const NEW_TOOL_LINE: &str = "New: the `memory` tool shows your memory's earlier versions; \
     `search` finds what you wrote, `diff` shows what changed.";

/// What the tool says when the session has no memory-id key.
pub const NO_KEY: &str = "Your memory ids couldn't be worked out this session, so this tool \
     can't name your memory's versions. Your operator has been told.";

/// What the tool says when the history couldn't be read at all.
pub const UNREADABLE: &str = "Your memory's history couldn't be read just now. Ask again later.";

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
/// memory-id key's fingerprint written before its `sealed/chained` marker
/// (it had memory ids before its versions were sealed; an agent enrolled
/// before it had a key has no ids to keep), and not told yet. Anything
/// unreadable is "not due": the sentence is never worth a guess.
fn enrollment_due(agent_dir: &Path) -> bool {
    let mtime = |p: PathBuf| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    let sealed = agent_dir.join(agora_agentkit::state::storage::SEALED_DIR);
    let (Some(enrolled), Some(fingerprint)) = (
        mtime(sealed.join(agora_agentkit::state::storage::CHAINED)),
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
        let current = view.current(current_id, &current);
        let text = match action {
            MemoryAction::Log { before, limit } => view.log(before.as_ref(), limit, &current),
            MemoryAction::Show { id, budget } => view.show(id.as_slice(), budget, current),
            MemoryAction::Diff { a, b } => view.diff(&a, b.as_ref(), current),
            MemoryAction::Search { query, limit } => {
                let limit = search_limit(limit);
                let found = search(
                    &self.agent_dir(),
                    self.id,
                    &view.rows,
                    &query,
                    limit.value,
                    view.unreachable,
                )
                .await;
                let mut out = String::new();
                // Search takes no `SealCheck`: say it as `log` and `show` do
                if view.seal == SealCheck::Unchecked {
                    out.push_str(UNCHECKED);
                }
                push_block(&mut out, &found.result.render(found.state));
                adjusted(limit.note, out)
            }
        };
        tracing::info!(
            event_type = "memory_tool_call",
            agent_id = %self.id,
            action = what,
            versions = view.rows.len(),
            unopened = view.rows.iter().filter(|v| v.content().is_none()).count(),
            unchecked = view.seal == SealCheck::Unchecked,
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
/// under (sealed on the main line first).
fn version_source(source: &Source) -> VersionSource {
    match source {
        Source::Sealed { .. } => VersionSource::Sealed,
        Source::ArchivedPrior { .. } => VersionSource::PriorCopy,
        Source::SideLine { .. } => VersionSource::SideLine,
        Source::OffLine { .. } => VersionSource::OffLine,
        Source::SetAside { on, .. } => VersionSource::SetAside {
            on: on.map(|on| on.date_naive()),
        },
        Source::LocalCopy { .. } | Source::Current => VersionSource::LocalCopy,
        Source::KeptAtExit { .. } => VersionSource::NeverSaved,
    }
}

/// Whether a source is the runner's own copy, unsealed.
fn is_local(source: &Source) -> bool {
    matches!(
        source,
        Source::LocalCopy { .. } | Source::Current | Source::KeptAtExit { .. }
    )
}

/// Why a sealed version couldn't be opened, in the agent's words (the
/// resolver's reason is the operator's: kept for the log).
const NOT_FETCHED: &str = "its sealed copy couldn't be fetched";

/// Why a sealed version wasn't tried, past the resolver's deadline.
const PAST_DEADLINE: &str = "gathering your history took too long";

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
    /// The versions the renderers are given, newest first
    /// ([`newest_first`]).
    rows: Vec<MemoryVersion>,
    /// `history.versions`' index → its row, if it has one.
    placed: Vec<Option<usize>>,
    /// Each opened row's memory id under every key (the current first),
    /// with its row, newest first: ids a replaced key named still resolve.
    ids: Vec<(MemoryId, usize)>,
    /// Whether the main line was checked. When it wasn't, only the
    /// runner's own copies are rows.
    seal: SealCheck,
    /// Agora was taken as unreachable: some sealed versions weren't asked
    /// for.
    unreachable: bool,
}

impl View {
    fn new(
        history: History,
        keys: &MemoryIdKeys,
        agent: AgentId,
        anchors: &HashMap<Sha256Hex, GovernanceLogId>,
    ) -> Self {
        let unreachable = history.versions.iter().any(|v| {
            v.unopened
                .as_deref()
                .is_some_and(|w| w.contains(super::version::UNREACHABLE))
        });
        let seal = if history.left_out.iter().any(|l| l.held == Held::MainLine) {
            SealCheck::Unchecked
        } else {
            SealCheck::Checked
        };
        // Each row with its place in the history and the text it is named
        // by: its memory, or an unopened version's local twin's
        let mut rows: Vec<(usize, MemoryVersion, Option<&str>)> = Vec::new();
        for (i, v) in history.versions.iter().enumerate() {
            let Some(primary) = v.sources.first() else {
                continue;
            };
            if seal == SealCheck::Unchecked && !is_local(primary) {
                continue;
            }
            let source = version_source(primary);
            let (mut row, named) = match &v.unopened {
                None => {
                    let content = v.memory_str();
                    (
                        MemoryVersion::opened(
                            MemoryId::of(&keys.current, agent, content),
                            content,
                            v.at,
                            source,
                        ),
                        Some(content),
                    )
                }
                Some(why) => {
                    let twin = v.twin_memory.as_deref().map(String::as_str);
                    let reason = if why.contains(super::version::UNREACHABLE) {
                        super::version::UNREACHABLE
                    } else if why.contains(super::history::PAST_DEADLINE) {
                        PAST_DEADLINE
                    } else {
                        NOT_FETCHED
                    };
                    (
                        MemoryVersion {
                            head: None,
                            at: v.at,
                            source,
                            anchored: None,
                            text: VersionText::Unopened {
                                id: twin.map(|t| MemoryId::of(&keys.current, agent, t)),
                                reason: reason.into(),
                            },
                        },
                        twin,
                    )
                }
            };
            row.head = primary.head().map(|h| h.entry_hash);
            row.anchored = v.heads().find_map(|h| anchors.get(&h.entry_hash).cloned());
            rows.push((i, row, named));
        }
        // Newest first, undated last, in agentkit's order: sorted here with
        // the rows' places, then checked unchanged by `newest_first`
        rows.sort_by_key(|(_, v, _)| (v.at.is_none(), std::cmp::Reverse(v.at)));
        let mut placed = vec![None; history.versions.len()];
        let mut ids = Vec::new();
        for key in keys.all() {
            for (row, (_, _, named)) in rows.iter().enumerate() {
                if let Some(text) = named {
                    ids.push((MemoryId::of(key, agent, text), row));
                }
            }
        }
        for (row, (i, _, _)) in rows.iter().enumerate() {
            placed[*i] = Some(row);
        }
        let mut rows: Vec<MemoryVersion> = rows.into_iter().map(|(_, v, _)| v).collect();
        let order: Vec<_> = rows.iter().map(|v| v.at).collect();
        newest_first(&mut rows);
        debug_assert_eq!(order, rows.iter().map(|v| v.at).collect::<Vec<_>>());
        Self {
            history,
            rows,
            placed,
            ids,
            seal,
            unreachable,
        }
    }

    /// The memory in context, placed at its newest row by memory id: an
    /// unopened row named by its local twin places it too, where
    /// agentkit's [`CurrentMemory::new`] matches opened text only.
    fn current<'a>(&self, id: MemoryId, content: &'a str) -> CurrentMemory<'a> {
        CurrentMemory {
            id,
            content,
            position: self.rows.iter().position(|v| v.id() == Some(id)),
        }
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
        let sealed = !heads.is_empty();
        let resolution = resolve(r, self.ids.iter().map(|(id, _)| id), heads.iter());
        let notes: Vec<String> = resolution.note(r).into_iter().collect();
        match resolution.candidate() {
            Some(Candidate::Memory(id)) => match self.ids.iter().find(|(m, _)| *m == id) {
                Some((_, row)) => Pick::At(*row, notes),
                None => Pick::Say(not_found(r, sealed)),
            },
            Some(Candidate::Head(head)) => self.at_head(&head, notes),
            None => match resolution {
                Resolution::Ambiguous(_) => Pick::Say(
                    resolution
                        .ambiguity(r)
                        .unwrap_or_else(|| not_found(r, sealed)),
                ),
                _ => Pick::Say(not_found(r, sealed)),
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
                    // Only when the main line couldn't be checked: its
                    // sealed versions aren't rows
                    return Pick::Say(format!(
                        "Head {short} is held by your runner, but your sealed history couldn't \
                         be checked just now, so it isn't shown; ask again later."
                    ));
                };
                let its = match self.rows[row].id() {
                    Some(id) => format!("; its memory is {id}"),
                    None => String::new(),
                };
                match source {
                    Source::SideLine { .. } => notes.push(format!(
                        "Head {short} is on a side line your operator rebased{its}."
                    )),
                    Source::SetAside { on, .. } => notes.push(format!(
                        "Head {short} was set aside{} when your line was rebased{its}.",
                        on.map(|d| format!(" on {}", d.date_naive()))
                            .unwrap_or_default()
                    )),
                    Source::OffLine { .. } => notes.push(format!(
                        "Head {short} is sealed off your main line, with no record of why{its}."
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
                    Pick::Say(said) => return compose(&[said], ""),
                },
            },
        };
        let limit = log_limit(limit);
        let body = render::log(&self.rows, start, limit.value, current, self.seal);
        adjusted(limit.note, compose(&notes, &body))
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
            return compose(&notes, "");
        }
        let budget = show_budget(budget);
        let body = render::show(&self.rows, &picked, current, budget.value, self.seal);
        adjusted(budget.note, compose(&notes, &body))
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
            return compose(&notes, "");
        };
        let body = render::diff(&self.rows[a], b.map(|b| &self.rows[b]), current);
        compose(&notes, &body)
    }
}

/// `out`, opened with what agentkit says of a limit or budget it
/// adjusted, if it did.
fn adjusted(note: Option<String>, out: String) -> String {
    match note {
        Some(note) => {
            let mut said = note;
            push_block(&mut said, &out);
            said
        }
        None => out,
    }
}

/// The notes, then the body.
fn compose(notes: &[String], body: &str) -> String {
    let mut out = String::new();
    for note in notes {
        push_block(&mut out, note);
    }
    push_block(&mut out, body);
    out
}

/// `sealed`: whether any head is held here, so the reply names heads only
/// to an agent that has them (beta round 17)
fn not_found(r: &HistoryRef, sealed: bool) -> String {
    if sealed {
        format!(
            "No version of your memory has the id or head \"{}\". `log` lists them, with their \
             memory ids and heads.",
            r.shown()
        )
    } else {
        format!(
            "No version of your memory has the id \"{}\". `log` lists them, with their memory \
             ids.",
            r.shown()
        )
    }
}

/// A search's result, and how the index was come by.
struct Found {
    result: agora_agentkit::reactor::seed::memory::SearchResult,
    state: IndexState,
}

/// Search the index, building it on the first search, adding the versions
/// it lacks, and building it again when it can't be read or names a memory
/// no longer held. Only opened, dated versions are indexed. Written only
/// when it changed. A write that fails is logged; the search is still
/// answered.
async fn search(
    agent_dir: &Path,
    id: AgentId,
    rows: &[MemoryVersion],
    query: &str,
    limit: usize,
    unreachable: bool,
) -> Found {
    let dir = agent_dir.to_path_buf();
    let read = tokio::task::spawn_blocking(move || {
        super::history_index::read(&dir).map(|r| (r, super::history_index::moving(&dir)))
    })
    .await
    .map_err(std::io::Error::other)
    .and_then(|r| r);
    let (read, moved) = match read {
        Ok((r, moved)) => (Ok(r), moved),
        Err(e) => (Err(e), false),
    };
    let (index, state) = match read {
        // Moved into `sealed/` since the last search: built again there
        Ok(None) if moved => (SearchIndex::new(), IndexState::Rebuilt),
        Ok(None) => (SearchIndex::new(), IndexState::Built),
        Ok(Some(bytes)) => match SearchIndex::from_bytes(&bytes) {
            Ok(index) => (index, IndexState::Loaded),
            Err(e) => {
                tracing::warn!(
                    event_type = "memory_index_rebuilt",
                    agent_id = %id,
                    reason = index_error_kind(&e),
                    "the memory search index couldn't be used; rebuilt"
                );
                (SearchIndex::new(), IndexState::Rebuilt)
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
            (SearchIndex::new(), IndexState::Rebuilt)
        }
    };
    let indexable: Vec<(MemoryId, DateTime<Utc>, &str)> = rows
        .iter()
        .rev()
        .filter_map(|v| Some((v.id()?, v.at?, v.content()?)))
        .collect();
    // Every row's id is held, unopened ones' too; a row with no id (an
    // unopened version with no local twin) may be any indexed memory, so
    // nothing is dropped for it: an Agora blip never costs a rebuild
    let held: HashSet<MemoryId> = rows.iter().filter_map(MemoryVersion::id).collect();
    let all_named = rows.iter().all(|v| v.id().is_some());
    let (mut index, state) = if all_named && index.ids().any(|m| !held.contains(m)) {
        tracing::info!(
            event_type = "memory_index_rebuilt",
            agent_id = %id,
            reason = "stale",
            "the memory search index names memories no longer held; rebuilt"
        );
        let state = match state {
            IndexState::Loaded => IndexState::Rebuilt,
            other => other,
        };
        (SearchIndex::new(), state)
    } else {
        (index, state)
    };
    let mut changed = state != IndexState::Loaded;
    for (m, at, content) in indexable {
        changed |= index.add(m, at, content);
    }
    // Not while Agora can't be reached: the versions it holds would be
    // missing, and the next search would only add them
    if changed && unreachable {
        tracing::info!(
            event_type = "memory_index_not_written",
            agent_id = %id,
            "Agora unreachable; the memory search index is kept in memory this time"
        );
    } else if changed {
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
        result: index.search(query, limit),
        state,
    }
}

/// What was wrong with an index file, for the log: the variant only (a
/// JSON error's text can quote the file, which holds memory).
fn index_error_kind(e: &IndexError) -> &'static str {
    match e {
        IndexError::Json(_) => "json",
        IndexError::Format(_) => "format",
        IndexError::Mismatch => "mismatch",
        IndexError::LineOutOfRange => "line_out_of_range",
        IndexError::Unsorted => "unsorted",
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
    use agora_agentkit::reactor::seed::memory::{INDEX_BUILT, INDEX_REBUILT, LOCAL_ONLY};
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
            limit: None,
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
        // A limit agentkit adjusted is said first
        let lowered = rd
            .answer(
                A::Log {
                    before: None,
                    limit: Some(500),
                },
                &api,
            )
            .await
            .text;
        assert!(lowered.starts_with("Limit lowered to 50."), "{lowered}");
        assert!(lowered.contains(LOCAL_ONLY), "{lowered}");
        assert!(out.text.contains(&format!("{}  memory", mem(id, 3))));
        assert!(out.text.contains("← in your context"), "{}", out.text);
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
        // Nothing here is sealed, so the reply doesn't speak of heads
        assert!(!out.contains("head"), "{out}");
        // A whole id asked for is echoed short
        let whole = format!("mem-{}", "ab".repeat(32));
        let out = rd.answer(show(vec![r(&whole)]), &api).await.text;
        assert!(out.contains("\"mem-ababababab\""), "{out}");
        assert!(!out.contains(&whole), "{out}");
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
        // Enrolled since: the next search moves the index into sealed/, as
        // a rebuild, never a "first search"
        std::fs::create_dir_all(dir.join("sealed")).unwrap();
        assert!(index.exists(), "not removed until the next search");
        let out = rd.answer(search("synthetic"), &api).await.text;
        assert!(
            out.contains(INDEX_REBUILT) && !out.contains(INDEX_BUILT),
            "{out}"
        );
        assert!(!index.exists(), "moved");
        assert!(super::super::history_index::path(&dir).exists());
        // A search limit agentkit adjusted is said first
        let searched = rd
            .answer(
                A::Search {
                    query: "synthetic".into(),
                    limit: Some(99),
                },
                &api,
            )
            .await
            .text;
        assert!(searched.starts_with("Limit lowered to 30."), "{searched}");
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

    /// A sealed version that can't be opened is a row of its own, by head,
    /// never given borrowed or empty content; asked for, it says so in the
    /// agent's words.
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
        // As a runner before agentkit 0.158 left sealed/
        crate::sealed::version::drop_acknowledged_blobs(&b.seal()).await;
        *b.api.fail.lock().unwrap() = Some("blob");
        let mut rd = reader(&b.state_dir(), &b.keys, b.id, 4, None);
        let mut cache = HistoryCache::default();
        let h = super::super::history::resolve(&b.state_dir(), &b.keys, &b.api, b.id, &mut cache)
            .await
            .unwrap();
        let unopened: Vec<&Version> = h.versions.iter().filter(|v| v.unopened.is_some()).collect();
        assert!(!unopened.is_empty());
        let out = rd.answer(log(), &b.api).await.text;
        assert!(!out.contains("synthetic"), "{out}");
        for v in &unopened {
            let short = short_head(&v.heads().next().unwrap().entry_hash);
            let row = out
                .lines()
                .find(|l| l.contains(&format!("head {short}")))
                .unwrap_or_else(|| panic!("a row of its own: {out}"));
            assert!(row.contains("couldn't be opened just now"), "{row}");
            let said = rd.answer(show(vec![r(&short)]), &b.api).await.text;
            assert!(
                [NOT_FETCHED, super::super::version::UNREACHABLE]
                    .iter()
                    .any(|why| said.contains(&format!("couldn't be opened just now ({why})"))),
                "{said}"
            );
            assert!(!said.contains("synthetic"), "no content: {said}");
            // The resolver's own reason is the operator's, never shown
            assert!(!said.contains("blob"), "{said}");
            // Named by its local twin's memory: its id reaches it too
            if let Some(twin) = &v.twin_memory {
                let id = MemoryId::of(&keys(1).current, b.id, twin);
                let said = rd.answer(show(vec![r(id)]), &b.api).await.text;
                assert!(said.contains("couldn't be opened just now"), "{said}");
                assert!(!said.contains("No version"), "{said}");
            }
        }
    }

    /// No index is written while Agora can't be reached: it would lack
    /// the versions Agora holds.
    #[tokio::test(start_paused = true)]
    async fn no_index_is_written_while_agora_is_unreachable() {
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
        *b.api.blob_delay.lock().unwrap() = Some(std::time::Duration::from_secs(3600));
        let mut rd = reader(&b.state_dir(), &b.keys, b.id, 4, None);
        let out = rd.answer(search("synthetic"), &b.api).await.text;
        assert!(out.contains("synthetic"), "{out}");
        assert!(!super::super::history_index::path(&b.dir).exists());
        let out = rd.answer(log(), &b.api).await.text;
        assert!(out.contains(super::super::version::UNREACHABLE), "{out}");
    }

    /// An index built while Agora served is kept through a blip: the
    /// unopened versions' memories are still held.
    #[tokio::test(start_paused = true)]
    async fn a_blip_does_not_rebuild_the_index() {
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
        let mut rd = reader(&b.state_dir(), &b.keys, b.id, 4, None);
        let out = rd.answer(search("synthetic"), &b.api).await.text;
        assert!(out.starts_with(INDEX_BUILT), "{out}");
        let index = super::super::history_index::path(&b.dir);
        let written = std::fs::read(&index).unwrap();
        *b.api.fail.lock().unwrap() = Some("blob");
        let mut fresh = reader(&b.state_dir(), &b.keys, b.id, 4, None);
        let out = fresh.answer(search("synthetic"), &b.api).await.text;
        assert!(
            !out.contains(INDEX_REBUILT) && !out.contains(INDEX_BUILT),
            "{out}"
        );
        assert_eq!(std::fs::read(&index).unwrap(), written, "not rewritten");
    }

    /// A main line that can't be checked: agentkit says so first, only
    /// the local copies are rows, and no head is given.
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
        assert!(!out.contains("  head "), "{out}");
        assert!(out.contains("in your context"), "{out}");
        let out = rd.answer(search("synthetic"), &b.api).await.text;
        assert!(out.starts_with(UNCHECKED), "{out}");
        // Every head held on the line is answered, never "not found"
        let mut heads = vec![b.seal().head().await.unwrap().unwrap().head.entry_hash];
        heads.extend(
            b.seal()
                .sent()
                .await
                .unwrap()
                .iter()
                .chain(b.seal().outbox().await.unwrap().iter())
                .map(|i| i.entry.entry_hash()),
        );
        for h in heads {
            let short = short_head(&h);
            let said = rd.answer(show(vec![r(&short)]), &b.api).await.text;
            assert!(
                said.contains("your sealed history couldn't be checked just now"),
                "{short}: {said}"
            );
            assert!(!said.contains("No version"), "{said}");
        }
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
        // An empty query is refused at parse
        let result = tool
            .call(call(
                serde_json::json!({"request": {"action": "search", "query": " "}}),
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

    fn head(n: u8) -> agora_agentkit::state::verify::HeadRef {
        agora_agentkit::state::verify::HeadRef {
            entry_hash: Sha256Hex::from([n; 32]),
            seq: u64::from(n),
        }
    }

    fn at(day: u32) -> Option<DateTime<Utc>> {
        use chrono::TimeZone;
        Some(Utc.with_ymd_and_hms(2026, 11, day, 21, 14, 0).unwrap())
    }

    fn held(at: Option<DateTime<Utc>>, sources: Vec<Source>, memory: Option<&str>) -> Version {
        Version {
            at,
            sources,
            state_hash: None,
            memory: memory.map(|m| Zeroizing::new(m.to_string())),
            twin_memory: None,
            unopened: memory
                .is_none()
                .then(|| "couldn't be opened just now: GET /blob 503".into()),
        }
    }

    /// Every source the resolver finds, as agentkit rows: off the line,
    /// set aside with no date (last, "date unknown"), unopened (its own
    /// row, the resolver's reason never shown), and the current memory.
    fn synthetic() -> History {
        History {
            versions: vec![
                held(at(1), vec![Source::Sealed { head: head(1) }], Some("one\n")),
                held(
                    at(2),
                    vec![Source::OffLine { head: head(2) }],
                    Some("one\ntwo\n"),
                ),
                held(at(3), vec![Source::Sealed { head: head(3) }], None),
                held(
                    at(4),
                    vec![Source::Sealed { head: head(4) }, Source::Current],
                    Some("one\ntwo\nfour\n"),
                ),
                held(
                    None,
                    vec![Source::SetAside {
                        head: head(5),
                        on: None,
                    }],
                    Some("five\n"),
                ),
            ],
            left_out: Vec::new(),
        }
    }

    /// The copy from before the May loss is labelled as that, and the
    /// first real version isn't sized against it (beta-memory: a −449
    /// against the prior copy read as a loss).
    #[test]
    fn the_prior_copy_is_labelled_and_not_compared() {
        let agent = AgentId::new();
        let k = keys(1);
        let h = History {
            versions: vec![
                held(
                    at(1),
                    vec![Source::ArchivedPrior { head: head(1) }],
                    Some("a much longer memory from before the loss\n"),
                ),
                held(
                    at(2),
                    vec![Source::Sealed { head: head(2) }],
                    Some("first\n"),
                ),
            ],
            left_out: Vec::new(),
        };
        let view = View::new(h, &k, agent, &HashMap::new());
        let now = "first\n";
        let current = view.current(MemoryId::of(&k.current, agent, now), now);
        let out = view.log(None, None, &current);
        let rows: Vec<&str> = out.lines().collect();
        assert!(
            rows[1].contains("a copy from before a memory loss"),
            "{out}"
        );
        assert!(
            !rows[0].contains('\u{2212}'),
            "no delta across kinds: {out}"
        );
    }

    #[test]
    fn every_source_maps_onto_agentkit_rows() {
        let agent = AgentId::new();
        let k = keys(1);
        let view = View::new(synthetic(), &k, agent, &HashMap::new());
        let now = "one\ntwo\nfour\n";
        let current = view.current(MemoryId::of(&k.current, agent, now), now);
        let out = view.log(None, None, &current);
        let rows: Vec<&str> = out.lines().collect();
        assert_eq!(rows.len(), 5, "{out}");
        assert!(rows[0].contains("head 040404040404") && rows[0].ends_with("← in your context"));
        assert!(
            rows[1].contains("head 030303030303")
                && rows[1].contains("couldn't be opened just now"),
            "{out}"
        );
        assert!(
            rows[2].contains("head 020202020202") && rows[2].contains("sealed, off your main line"),
            "{out}"
        );
        assert!(
            rows[4].contains("head 050505050505") && rows[4].contains("date unknown"),
            "{out}"
        );
        assert!(!out.contains("503"), "the operator's reason: {out}");
        // Asked for by head
        let said = view.show(&[r("030303030303")], None, current);
        assert!(
            said.contains(&format!("({NOT_FETCHED}); ask again later.")),
            "{said}"
        );
        let said = view.show(&[r("050505050505")], None, current);
        assert!(
            said.starts_with(&format!(
                "Head 050505050505 was set aside when your line was rebased; its memory is {}.",
                MemoryId::of(&k.current, agent, "five\n")
            )),
            "{said}"
        );
        let said = view.show(&[r("020202020202")], None, current);
        assert!(said.starts_with("Head 020202020202 is sealed off your main line, with no record of why; its memory is mem-"), "{said}");

        // The memory in context whose sealed version couldn't be opened,
        // named by its local twin: still placed, still reachable by id
        let mut h = synthetic();
        h.versions[3].memory = None;
        h.versions[3].unopened = Some("couldn't be opened just now: 503".into());
        h.versions[3].twin_memory = Some(Zeroizing::new(now.to_string()));
        let view = View::new(h, &k, agent, &HashMap::new());
        let id = MemoryId::of(&k.current, agent, now);
        let current = view.current(id, now);
        assert_eq!(current.position, Some(0));
        let out = view.log(None, None, &current);
        assert!(
            out.lines().next().unwrap().ends_with("← in your context"),
            "{out}"
        );
        let said = view.show(&[r(id)], None, current);
        assert!(said.contains("couldn't be opened just now"), "{said}");

        // Unchecked: only the runner's own copies are rows, no heads
        let mut h = synthetic();
        h.left_out.push(super::super::history::LeftOut {
            held: Held::MainLine,
            why: "no signing key to verify with".into(),
        });
        h.versions[3].sources = vec![Source::Current];
        let view = View::new(h, &k, agent, &HashMap::new());
        assert_eq!(view.seal, SealCheck::Unchecked);
        assert_eq!(view.rows.len(), 1);
        let current = CurrentMemory::new(MemoryId::of(&k.current, agent, now), now, &view.rows);
        let out = view.log(None, None, &current);
        assert!(out.starts_with(UNCHECKED), "{out}");
        assert!(!out.contains("  head "), "{out}");
        let said = view.show(&[r("030303030303")], None, current);
        assert!(
            said.contains("your sealed history couldn't be checked just now"),
            "{said}"
        );
    }
}
