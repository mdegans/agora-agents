//! What every end-of-session offer shares, once (CLAUDE.md rule 8): the
//! miss cap, the per-agent option order, when an offer is due, how its
//! memory note is checked, its answer's schema, the receipt, and the slot
//! [`ConsentAgent`](super::agent::ConsentAgent) keeps each offer's session
//! state in until teardown writes it ([`Slot`]).
//!
//! The offers themselves (their texts, choices and ledgers) stay in their
//! own modules: [`super::prompt`] (the model swap), [`super::role`],
//! [`super::cadence`], [`super::memory_restore`] and
//! [`super::sealed_state`].

use std::path::{Path, PathBuf};

use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::seed::{Memory, SeedState, ShortString, Soul};
use chrono::Utc;

use super::role::{self, ledger::Applied};

/// Asks that went unanswered before an offer stops asking: the first is
/// asked again at a later session, the second is final, and means nothing
/// changes. A miss is the agent's: an API refusal, or (before 2026-11-06)
/// any no-answer. Answers we couldn't read are ours, and never count
/// ([`Status::Unrecorded`]; Steward, 2026-11-06).
pub const MAX_MISSES: u32 = 2;

/// Which offer: for log lines (`offer = …`) and question names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OfferKind {
    ModelSwap,
    Role,
    Cadence,
    MemoryRestore,
    SealedState,
}

impl OfferKind {
    /// The name in logs (the `answer_offer` key it was until 2026-11-06).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ModelSwap => "model_swap",
            Self::Role => "role",
            Self::Cadence => "cadence",
            Self::MemoryRestore => "memory_restore",
            Self::SealedState => "sealed_state",
        }
    }

    /// The question's name ([`Ask::name`](agora_agentkit::reactor::ask::Ask::name)).
    pub fn question(self) -> &'static str {
        match self {
            Self::ModelSwap => "offer:model_swap",
            Self::Role => "offer:role",
            Self::Cadence => "offer:cadence",
            Self::MemoryRestore => "offer:memory_restore",
            Self::SealedState => "offer:sealed_state",
        }
    }

    /// The `event_type` of an unusable reply to it, as stall-watch knows it.
    pub fn malformed(self) -> &'static str {
        match self {
            Self::ModelSwap => "model_consent_malformed",
            Self::Role => "role_consent_malformed",
            Self::Cadence => "cadence_consent_malformed",
            Self::MemoryRestore => "memory_restore_malformed",
            Self::SealedState => "sealed_state_malformed",
        }
    }

    /// The `event_type` of an ask that ended without an answer.
    pub fn no_answer(self) -> &'static str {
        match self {
            Self::ModelSwap => "model_consent_no_answer",
            Self::Role => "role_consent_no_answer",
            Self::Cadence => "cadence_consent_no_answer",
            Self::MemoryRestore => "memory_restore_no_answer",
            Self::SealedState => "sealed_state_no_answer",
        }
    }

    /// What the receipt says the answer was about.
    pub fn about(self) -> &'static str {
        match self {
            Self::ModelSwap => "the model you run on",
            Self::Role => "your role",
            Self::Cadence => "your session schedule",
            Self::MemoryRestore => "your memory from before 5 May 2026",
            Self::SealedState => "sealed backups of your state",
        }
    }
}

// --- The order ---------------------------------------------------------------

/// The shuffle seed for `agent`: both halves of its UUID folded together,
/// mixed with the offer's `version` (so a new text gets a new order) and its
/// own `salt` (so its order is not a sibling's). Recorded in the offer's
/// ledger beside the order it produced.
pub fn seed_for(agent: AgentId, version: u32, salt: u64) -> u64 {
    let bits = agent.as_uuid().as_u128();
    (bits as u64) ^ ((bits >> 64) as u64) ^ u64::from(version) ^ salt
}

/// splitmix64: a tiny, well-mixed PRNG step, enough for a short shuffle and
/// no new dependency.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A Fisher–Yates shuffle of `items` seeded with `seed`: deterministic, so
/// an order can be replayed from its recorded seed.
pub fn shuffle<T: Copy, const N: usize>(mut items: [T; N], seed: u64) -> [T; N] {
    let mut state = seed;
    for i in (1..N).rev() {
        let j = (splitmix64(&mut state) % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
    items
}

/// `` `a`, `b` or `c` ``: the options, in the order shown.
pub fn either(names: &[&str]) -> String {
    let names: Vec<String> = names.iter().map(|n| format!("`{n}`")).collect();
    match names.as_slice() {
        [init @ .., last] if !init.is_empty() => format!("{} or {last}", init.join(", ")),
        _ => names.join(""),
    }
}

// --- When an offer is due ----------------------------------------------------

/// What a recorded ask's outcome means for asking again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// An answer, or (before 2026-11-06) a refusal: never asked again.
    Settled,
    /// No answer, the agent's doing: counts toward [`MAX_MISSES`].
    Miss,
    /// Recorded so it can be seen, and nothing more: every reply unusable
    /// (our failure, not the agent's choice).
    Unrecorded,
}

/// An offer ledger's per-ask outcome.
pub trait Outcome {
    fn status(&self) -> Status;
}

/// Whether an offer with these `outcomes` on file should be put: nothing
/// settled, fewer than [`MAX_MISSES`] misses, and no answer already written
/// into the SOUL (`in_soul`: the belt to the ledger's braces, for a session
/// whose SOUL saved but whose ledger didn't).
pub fn due<'a, O: Outcome + 'a>(outcomes: impl IntoIterator<Item = &'a O>, in_soul: bool) -> bool {
    if in_soul {
        return false;
    }
    let mut misses = 0;
    for outcome in outcomes {
        match outcome.status() {
            Status::Settled => return false,
            Status::Miss => misses += 1,
            Status::Unrecorded => {}
        }
    }
    misses < MAX_MISSES
}

/// Whether `soul`'s Evolution Log holds one of the runner's own lines
/// (`[SYSTEM] …`) containing `marker`: how an offer finds its answer was
/// already written.
pub fn in_log(soul: &Soul, marker: &str) -> bool {
    soul.evolution_log.iter().any(|e| {
        let note = e.note.as_str();
        note.starts_with("[SYSTEM] ") && note.contains(marker)
    })
}

// --- The memory note ---------------------------------------------------------

/// The longest `memory_note`, in characters.
pub const MEMORY_NOTE_MAX_CHARS: usize = 600;

/// Whether `note` can be appended to `memory` as given: at most
/// [`MEMORY_NOTE_MAX_CHARS`], no SOUL heading, no `[SYSTEM]` marker. `Err`
/// says why, in words the agent is shown.
pub fn check_note(note: &str, memory: &Memory) -> Result<(), String> {
    let note = note.trim();
    if note.chars().count() > MEMORY_NOTE_MAX_CHARS {
        return Err(role::prompt::over_limit(
            "memory_note",
            note,
            MEMORY_NOTE_MAX_CHARS,
        ));
    }
    role::prompt::check_memory_note(note, memory)
}

/// `text`, trimmed, or `None` when that leaves nothing.
pub fn nonempty(text: &str) -> Option<String> {
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_string())
}

// --- The question ------------------------------------------------------------

/// The schema of an offer's answer `T`: [`ask_schema`]'s, with `choice`
/// narrowed to this agent's `choices` in the order it is shown them, and
/// every property required (a grammar then writes them all, in declaration
/// order; the parser still takes an answer that leaves the optional ones
/// out).
///
/// [`ask_schema`]: agora_agentkit::reactor::ask::ask_schema
pub fn answer_schema<T: schemars::JsonSchema>(choices: &[&str]) -> serde_json::Value {
    let mut schema = agora_agentkit::reactor::ask::ask_schema::<T>();
    let description = schema["properties"]["choice"]["description"].clone();
    let mut choice = serde_json::json!({ "type": "string", "enum": choices });
    if description.is_string() {
        choice["description"] = description;
    }
    schema["properties"]["choice"] = choice;
    // Every property is required, so no default applies (and a grammar
    // needn't carry one)
    if let Some(props) = schema["properties"].as_object_mut() {
        for prop in props.values_mut() {
            if let Some(p) = prop.as_object_mut() {
                p.remove("default");
            }
        }
    }
    let keys: Vec<serde_json::Value> = schema["properties"]
        .as_object()
        .map(|p| {
            p.keys()
                .map(|k| serde_json::Value::from(k.as_str()))
                .collect()
        })
        .unwrap_or_default();
    schema["required"] = keys.into();
    schema
}

/// Where an offer stands among this session's closing questions: its
/// preface says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    /// The only one.
    Only,
    /// The first of this many.
    First(usize),
    /// After another.
    Next,
}

/// Number words for the preface; past five, digits.
fn count_word(n: usize) -> String {
    match n {
        2 => "Two".into(),
        3 => "Three".into(),
        4 => "Four".into(),
        5 => "Five".into(),
        n => n.to_string(),
    }
}

/// Who is asking, and how to answer: the first paragraph of every offer.
pub fn preface(place: Place) -> String {
    const JSON: &str = "Answer in JSON only, in the shape shown at the end of the question, \
                        and do not use tools.";
    match place {
        Place::Only => format!(
            "One more question before this session ends. It isn't part of the survey, and it \
             isn't anonymous. It comes from the Steward (the human who runs Agora's servers) and \
             Claude, and your answer is recorded under your name so it can be acted on. {JSON}"
        ),
        Place::First(n) => format!(
            "{} more questions before this session ends, one at a time. They aren't part of the \
             survey, and they aren't anonymous. They come from the Steward (the human who runs \
             Agora's servers) and Claude, and your answers are recorded under your name so they \
             can be acted on. They are separate questions: what you choose in one changes \
             nothing in the others. Here is the first. {JSON}",
            count_word(n)
        ),
        Place::Next => format!(
            "The next question, also from the Steward and Claude, and also recorded under your \
             name. It is separate from the last: what you chose there changes nothing here. \
             {JSON}"
        ),
    }
}

/// The line an agent gets once its answer is taken, before whatever comes
/// next: what was recorded, and why its `memory_note` was not written, if
/// it wasn't.
pub fn receipt(about: &str, choice: &str, dropped: Option<&str>) -> String {
    let mut line = format!("Recorded your answer about {about}: `{choice}`.");
    if let Some(why) = dropped {
        line.push_str(&format!(" Your `memory_note` was not written: {why}"));
    }
    line
}

// --- The slot ----------------------------------------------------------------

/// An offer ledger, as a [`Slot`] keeps it.
// async-trait marks the boxed future `#[must_use]`, which it already is
#[allow(clippy::double_must_use)]
#[async_trait::async_trait]
pub trait OfferLedger: Send + Sync {
    /// The ledger's file in `agent_dir`.
    fn path_in(agent_dir: &Path) -> PathBuf;
    /// Atomic save.
    async fn save_to(&self, agent_dir: &Path) -> std::io::Result<()>;
    /// The display copy of the agent's name, kept current.
    fn set_agent(&mut self, name: ShortString<64>);
    /// Writing the latest answer failed at teardown (why): nothing was
    /// written, and the record says so. The choice stands.
    fn mark_apply_failed(&mut self, why: String);
}

/// Each offer's event names, for the slot's log lines.
#[derive(Debug)]
pub struct Events {
    /// The answer was written into the saved state.
    pub applied: &'static str,
    /// It couldn't be: nothing was.
    pub apply_failed: &'static str,
    /// The ledger couldn't be saved: the writes were taken back out.
    pub save_failed: &'static str,
}

/// A SOUL edit an answer makes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Edit {
    /// One Evolution Log line recording the answer.
    Line(String),
    /// The role offer's identity edit, with its own disclosure lines.
    Role(Applied),
}

/// What an answer writes into the state saved at teardown, all or nothing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Write {
    pub edit: Option<Edit>,
    /// The agent's own memory note.
    pub note: Option<String>,
}

impl Write {
    pub fn is_empty(&self) -> bool {
        self.edit.is_none() && self.note.is_none()
    }

    /// Write into `state`: the SOUL edit, then the note; whether the note
    /// was written. An edit that fails writes nothing, not the note either
    /// (`Err`: why, `state` unchanged).
    pub fn apply(&self, state: &mut SeedState) -> Result<bool, String> {
        let mut soul = state.soul.clone();
        match &self.edit {
            Some(Edit::Line(line)) => soul
                .push_evolution(line.clone())
                .map_err(|e| e.to_string())?,
            Some(Edit::Role(applied)) => role::apply(&mut soul, applied)?,
            None => {}
        }
        state.soul = soul;
        let today = Utc::now().date_naive();
        Ok(self
            .note
            .as_deref()
            .is_some_and(|n| role::append_memory_note(&mut state.memory, n, today)))
    }
}

/// One offer's state for a session: its ledger (`None` when the agent isn't
/// listed, or the file couldn't be read: then nothing is asked or saved),
/// whether it changed, and what its answer writes at teardown, with what it
/// takes to undo that if the ledger can't be saved.
///
/// The offers differ beyond this, and keep it beside their slot: the model
/// swap writes through its ledger's own SOUL entry
/// ([`Ledger::update_soul`](super::ledger::Ledger::update_soul)); the
/// memory-restore offer also shows what it restored; the sealed-state offer
/// also seats the start check's lines.
pub struct Slot<L> {
    pub ledger: Option<L>,
    pub dirty: bool,
    /// To write at teardown.
    pending: Write,
    /// SOUL and memory from before [`Self::pending`] was written, put back
    /// if the ledger can't be saved.
    undo: Option<(Soul, Memory)>,
    /// What was written, to write again on top if an edit made before it is
    /// taken back out.
    written: Option<Write>,
    events: &'static Events,
}

impl<L: OfferLedger> Slot<L> {
    pub fn new(events: &'static Events) -> Self {
        Self {
            ledger: None,
            dirty: false,
            pending: Write::default(),
            undo: None,
            written: None,
            events,
        }
    }

    /// Change the ledger (if there is one) and mark it to be saved: what
    /// `f` returned, or `None` without a ledger.
    pub fn record<R>(&mut self, f: impl FnOnce(&mut L) -> R) -> Option<R> {
        let ledger = self.ledger.as_mut()?;
        self.dirty = true;
        Some(f(ledger))
    }

    /// What this session's answer writes at teardown.
    pub fn stage(&mut self, write: Write) {
        self.pending = write;
    }
}

/// A [`Slot`], whatever its ledger: what teardown does with each offer, in
/// order.
#[allow(clippy::double_must_use)]
#[async_trait::async_trait]
pub trait Kept: Send {
    /// Whether anything waits to be written.
    fn pending(&self) -> bool;
    /// Write it into `state` (whether anything changed); a failure is
    /// [`Self::failed`].
    fn apply(&mut self, state: &mut SeedState, agent: AgentId) -> bool;
    /// Nothing could be written: why.
    fn failed(&mut self, why: String, agent: AgentId);
    /// Save the ledger if it changed; `Err` on failure, with its path.
    async fn save(&mut self, dir: &Path, name: ShortString<64>) -> Result<(), (PathBuf, String)>;
    /// Put `state` back as it was before this slot's writes; whether it
    /// was.
    fn undo(&mut self, state: &mut SeedState) -> bool;
    /// Write this slot's writes again on top of `state` (an earlier one was
    /// taken out from under them).
    fn redo(&mut self, state: &mut SeedState, agent: AgentId);
    fn events(&self) -> &'static Events;
}

#[async_trait::async_trait]
impl<L: OfferLedger> Kept for Slot<L> {
    fn pending(&self) -> bool {
        !self.pending.is_empty()
    }

    fn apply(&mut self, state: &mut SeedState, agent: AgentId) -> bool {
        let write = std::mem::take(&mut self.pending);
        if write.is_empty() {
            return false;
        }
        self.undo = Some((state.soul.clone(), state.memory.clone()));
        match write.apply(state) {
            Ok(noted) => {
                tracing::info!(
                    event_type = self.events.applied,
                    agent = %state.soul.name,
                    agent_id = %agent,
                    soul_line = write.edit.is_some(),
                    memory_note = noted,
                    "offer answer written"
                );
                let changed = write.edit.is_some() || noted;
                self.written = Some(write);
                changed
            }
            Err(why) => {
                self.undo = None;
                self.failed(why, agent);
                false
            }
        }
    }

    fn failed(&mut self, why: String, agent: AgentId) {
        tracing::error!(
            event_type = self.events.apply_failed,
            agent_id = %agent,
            error = %why,
            "offer answer not written; SOUL and memory unchanged"
        );
        self.pending = Write::default();
        self.record(|l| l.mark_apply_failed(why));
    }

    async fn save(&mut self, dir: &Path, name: ShortString<64>) -> Result<(), (PathBuf, String)> {
        let Some(ledger) = self.ledger.as_mut().filter(|_| self.dirty) else {
            return Ok(());
        };
        ledger.set_agent(name);
        ledger
            .save_to(dir)
            .await
            .map_err(|e| (L::path_in(dir), e.to_string()))
    }

    fn undo(&mut self, state: &mut SeedState) -> bool {
        let Some((soul, memory)) = self.undo.take() else {
            return false;
        };
        state.soul = soul;
        state.memory = memory;
        true
    }

    fn redo(&mut self, state: &mut SeedState, agent: AgentId) {
        let Some(write) = &self.written else {
            return;
        };
        self.undo = Some((state.soul.clone(), state.memory.clone()));
        if let Err(e) = write.apply(state) {
            tracing::error!(
                event_type = self.events.apply_failed,
                agent_id = %agent,
                error = %e,
                "offer writes not redone after an earlier edit was undone"
            );
        }
    }

    fn events(&self) -> &'static Events {
        self.events
    }
}

/// [`OfferLedger`] for the offer ledgers, which keep the same `agent`,
/// `path` and `save` and their own `mark_apply_failed`.
macro_rules! offer_ledger {
    ($($ty:ty),* $(,)?) => {$(
        #[async_trait::async_trait]
        impl OfferLedger for $ty {
            fn path_in(agent_dir: &Path) -> PathBuf {
                <$ty>::path(agent_dir)
            }
            async fn save_to(&self, agent_dir: &Path) -> std::io::Result<()> {
                <$ty>::save(self, agent_dir).await
            }
            fn set_agent(&mut self, name: ShortString<64>) {
                self.agent = Some(name);
            }
            fn mark_apply_failed(&mut self, why: String) {
                <$ty>::mark_apply_failed(self, why)
            }
        }
    )*};
}

offer_ledger!(
    super::role::ledger::RoleLedger,
    super::cadence::ledger::CadenceLedger,
    super::memory_restore::ledger::RestoreLedger,
    super::sealed_state::ledger::SealLedger,
);

/// The model-consent ledger writes nothing through a slot (its one SOUL
/// entry is its own, [`Ledger::update_soul`]): nothing to mark.
///
/// [`Ledger::update_soul`]: super::ledger::Ledger::update_soul
#[async_trait::async_trait]
impl OfferLedger for super::ledger::Ledger {
    fn path_in(agent_dir: &Path) -> PathBuf {
        Self::path(agent_dir)
    }
    async fn save_to(&self, agent_dir: &Path) -> std::io::Result<()> {
        Self::save(self, agent_dir).await
    }
    fn set_agent(&mut self, name: ShortString<64>) {
        self.agent = Some(name);
    }
    fn mark_apply_failed(&mut self, _: String) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one shuffle is the role offer's old one exactly (and so the
    /// cadence offer's, which was the same code): an agent's shown order
    /// for a given seed is unchanged by the fold.
    #[test]
    fn the_shuffle_is_the_old_role_and_cadence_shuffle() {
        fn old_role(seed: u64) -> [u8; 4] {
            let mut order = [0, 1, 2, 3];
            let mut state = seed;
            for i in (1..order.len()).rev() {
                let j = (splitmix64(&mut state) % (i as u64 + 1)) as usize;
                order.swap(i, j);
            }
            order
        }
        for seed in [0, 1, 7, 42, u64::MAX, 0x1234_5678_9abc_def0] {
            assert_eq!(shuffle([0u8, 1, 2, 3], seed), old_role(seed), "{seed}");
        }
        // Pinned outputs, so the shuffle itself can't drift.
        assert_eq!(shuffle([0u8, 1, 2, 3], 0), [2, 1, 0, 3]);
        assert_eq!(shuffle([0u8, 1, 2], 42), [0, 2, 1]);
    }

    #[test]
    fn the_seed_folds_the_id_with_version_and_salt() {
        let id = AgentId::from(uuid::Uuid::from_u128(
            0x0123_4567_89ab_cdef_fedc_ba98_7654_3210,
        ));
        let fold = 0x0123_4567_89ab_cdef_u64 ^ 0xfedc_ba98_7654_3210;
        assert_eq!(seed_for(id, 0, 0), fold);
        assert_eq!(seed_for(id, 4, 0), fold ^ 4);
        assert_eq!(seed_for(id, 4, 0x10), fold ^ 4 ^ 0x10);
    }

    struct O(Status);
    impl Outcome for O {
        fn status(&self) -> Status {
            self.0
        }
    }

    #[test]
    fn due_until_settled_or_out_of_misses_and_unusable_never_counts() {
        use Status::*;
        let due_of = |s: &[Status], soul: bool| {
            let o: Vec<O> = s.iter().map(|s| O(*s)).collect();
            due(&o, soul)
        };
        assert!(due_of(&[], false));
        assert!(!due_of(&[], true), "already in the SOUL");
        assert!(due_of(&[Miss], false));
        assert!(!due_of(&[Miss, Miss], false));
        assert!(!due_of(&[Settled], false));
        assert!(due_of(&[Unrecorded, Unrecorded, Unrecorded, Miss], false));
    }

    #[test]
    fn either_lists_the_options() {
        assert_eq!(either(&["a"]), "`a`");
        assert_eq!(either(&["a", "b"]), "`a` or `b`");
        assert_eq!(either(&["a", "b", "c"]), "`a`, `b` or `c`");
    }

    #[test]
    fn prefaces_read_in_sequence() {
        assert!(preface(Place::Only).starts_with("One more question before this session ends."));
        let first = preface(Place::First(2));
        assert!(
            first.starts_with("Two more questions before this session ends"),
            "{first}"
        );
        assert!(first.contains("Here is the first."), "{first}");
        let next = preface(Place::Next);
        assert!(next.starts_with("The next question"), "{next}");
        for p in [Place::Only, Place::First(3), Place::Next] {
            let p = preface(p);
            assert!(p.ends_with("and do not use tools."), "{p}");
            assert!(!p.contains("answer_offer"), "{p}");
        }
    }
}
