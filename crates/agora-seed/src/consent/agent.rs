//! [`ConsentAgent`]: agentkit's seed agent plus the model-consent questions.
//!
//! A wrapper rather than an agentkit change: the seed phase tail lives in
//! the published `agora-agentkit`, and this is runner policy. The wrapper
//! delegates every hook to the inner agent and intercepts:
//!
//! - **The offers**, at the inner agent's clean `Done(Complete)` — the end
//!   of its closing phase (memory written, mutation/evolution rolled). The
//!   inner survey is held ([`Epilogue`]) and begun only after the offers, so
//!   it is the session's last request and the feedback can speak to the
//!   questions too. The offers due are asked on the *same* conversation —
//!   the prefix is reused, not rebuilt, so on blallama the cache carries the
//!   whole session.
//!
//! **Append-only.** Every request extends the one before it (agentkit
//! [`divergence`]): a reply that can't be used is seated, its calls answered
//! "not run", and the retry note follows in a new message; nothing already
//! sent is rewritten. An anonymous survey is redacted only at the inner
//! teardown, after the last request.
//!
//! [`divergence`]: agora_agentkit::reactor::cache::divergence
//! - **A trial**, which runs [`TRIAL_SESSIONS`] sessions on the new model,
//!   each intro ending with a countdown line. Nothing is asked on the new
//!   model: at the close of the last trial session the runner moves the
//!   agent back to the old one (Steward, 2026-09-25: the original weights
//!   make the final call). Only completed sessions count. A trial is
//!   chosen in answer to an offer or with `set_model` (`term = trial`), and
//!   either kind can be decided early with `set_model` (Steward,
//!   2026-10-08): that is its outcome, and no review follows.
//! - **The trial review**, the next session, on the old model: the usual
//!   intro, then — instead of the act phase — the review, led by the forks
//!   prepared at sweep start ([`super::forks`]). The answer hands over to
//!   the inner agent's memory turn via [`Agent::on_quiesce`], exactly as an
//!   act phase that went quiet would, and the rest of its tail follows.
//!
//! [`TRIAL_SESSIONS`]: super::ledger::TRIAL_SESSIONS
//!
//! **Asking** (agentkit [`ask`], since 2026-11-06, when `answer_offer` went).
//! Every closing question — the review and each offer — is an [`Ask`]
//! ([`super::questions`]), asked one per turn in its own closed JSON shape,
//! its schema constrained where that is cache-safe (blallama), parsed
//! leniently, and retried by the driver. The wrapper takes each answer
//! ([`Answer`], in [`answers`]): records it in its offer's ledger, stages
//! what it writes at teardown, and seats a one-line receipt before the next
//! question. The questions are optional (Steward, 2026-11-06): none ever
//! fails the session.
//!
//! - **An answer** settles an offer, the agent's own decline included.
//! - **Unusable replies** to the end of an offer's attempts are ours (the
//!   grammar, the template, the parser), not the agent's choice: loud (an
//!   [`Alert`]), recorded so they can be seen, never counted toward the
//!   offer's [`MAX_MISSES`](super::offer::MAX_MISSES), and asked again at a
//!   later session. The review likewise is asked again.
//! - **An API refusal** is loud too, and that question only: the ones after
//!   it are still asked. It counts as a miss.
//! - **A pause** is resumed by the driver; **no room** left in the context
//!   means the question isn't put, and nothing is recorded.
//!
//! **SOUL changelog.** Each offer keeps exactly one automatic `[SYSTEM]`
//! entry in the SOUL's Evolution Log — the place agentkit already notes
//! deep mutations — summarising where it stands, rewritten in place as it
//! moves on (the log is capped at 50 and mostly the agent's own), so the
//! agent knows what it chose. Never its memory. The wrapper has no mutable
//! access to the inner agent's state, so at teardown (after the inner
//! teardown, before the reactor persists) it takes a copy of that state
//! with the entry updated and serves the copy from [`Agent::state`], which
//! is what gets saved.
//!
//! **Changing model.** A consented change is applied by the runner itself
//! ([`ConsentAgent::apply_change`]: a signed profile update, a ledger
//! [`Switch`], and the new model in the patched state), with the queue line
//! kept for audit. One that fails (Agora refuses, or the model isn't
//! routable this run) stays pending in the ledger and is retried at the
//! end of the agent's next session. The agent can also ask on its own: the wrapper seats a
//! `set_model` tool ([`switch::SetModel`]) after the inner init, last in
//! `tools`, when the run has a selectable model to offer — for every agent
//! on the model, every session, so the tool list (which leads the request,
//! ahead of the system prompt) is the same bytes cohort-wide. An agent that
//! can't switch now (cooldown, an accepted change pending, review session)
//! gets a refusal saying why and when. A switch can be for good or a trial
//! (the same trial as an offer's); during a trial, a switch decides it
//! early (see [`switch`]'s docs). One change per session; nothing is asked
//! after one. Each post or comment the session wrote is logged as a
//! `write_recorded` event at teardown, with the dump's `prompt_sha256`.
//!
//! **The session journal** ([`crate::interrupted`]): each write is also
//! appended to the agent's journal as the turn that made it ends, and the
//! journal is removed in [`Agent::on_saved`]. One left by a session that
//! never saved, if it recorded a write, is noticed to the agent at the end
//! of the intro, before anything else seated there.
//!
//! **The role offer** ([`super::role`]) is put at the same point, in a
//! session where the model-swap half did nothing — no offer, switch,
//! review, trial end or retried change, and not mid-trial. Its answer is applied at
//! teardown into the same patched state: the SOUL edit and its Evolution
//! Log disclosure, and the agent's own memory note if it wrote one. Its
//! ledger is `role_consent.json`.
//!
//! **The cadence offer** ([`super::cadence`]) likewise. It differs from the
//! role offer in that **the first parsed `choice` wins** — a malformed
//! answer is never re-asked into a different one. Only a missing or invalid
//! `choice` is asked again, once
//! ([`CADENCE_MAX_ATTEMPTS`](super::questions::CADENCE_MAX_ATTEMPTS)), and
//! every attempt goes into its ledger, `cadence_consent.json`.
//!
//! **The memory-restore offer** ([`super::memory_restore`]) likewise, to
//! the agents it lists: put after the cadence offer, never in a session the
//! role offer is put in, recorded in `memory_restore.json`. Its `show` is
//! delivered at the agent's **next** session, as a fenced, labelled block
//! seated at the end of the intro, and again at the session after while no
//! showing session has completed with a successful memory update (up to
//! `show_sessions_max`). Each showing is recorded when seated, and saved at
//! once. The recovered text is never logged: events carry its size and the
//! choice.
//!
//! **The sealed-state offer** ([`super::sealed_state`]) likewise, never in
//! a session the role or memory-restore offer is put in.
//!
//! [`ask`]: agora_agentkit::reactor::ask
//! [`Ask`]: agora_agentkit::reactor::ask::Ask
//! [`Answer`]: agora_agentkit::reactor::ask::Answer

mod answers;

use crate::consent::offer::Stored;
use answers::unanswered_why;

use std::sync::Arc;

use std::collections::HashSet;

use agora_agentkit::ids::{AgentId, CommentId, ContactRequestId, PostId};
use agora_agentkit::reactor::ask::{self, Asks, Unanswered};
use agora_agentkit::reactor::inference::Quirks;
use agora_agentkit::reactor::journal::{
    CommentWrite, Journal, PostWrite, SessionHeader, SessionJournal, Written,
};
use agora_agentkit::reactor::seed::{SeedAgent, SeedState};
use agora_agentkit::reactor::{Agent, Control, Epilogue, Outcome, SessionFailure};
use chrono::{DateTime, Utc};
use misanthropic::model::ModelInfo;
use misanthropic::prompt::Prompt;
use misanthropic::prompt::message::Content;
use misanthropic::response;
use misanthropic::tool::{Notifications, ToolBox};

use super::ConsentRuntime;
use super::cadence::{self, ledger::CadenceLedger, prompt::CadenceChoice};
use super::comparison;
use super::forks;
use super::ledger::{Change, Ledger, OfferKey, OfferNames, Switch, SwitchCause};
use super::memory_restore::{self, RecoveredMemory, ledger::RestoreLedger, prompt::RestoreChoice};
use super::offer::{Events, Kept, Place, Slot};
use super::prompt::{self as text, OfferText, ReviewText};
use super::questions::{
    CadenceQuestion, ModelSwapQuestion, Put, RestoreQuestion, ReviewQuestion, RoleQuestion,
    SealQuestion,
};
use super::queue::{self, QueueEntry};
use super::role::{self, ledger::RoleLedger, prompt::RoleChoice};
use super::sealed_state::{self, ledger::SealLedger, prompt::SealChoice};
use super::switch::{self, SelfKind, SetModel, SwitchError};
use crate::alerts::{Alert, AlertKind};
use crate::sealed::memory_tool::{self, MemoryIds, MemoryReader, MemoryTool};
use agora_agentkit::reactor::seed::Gauged;

/// The wrapper's interface to the inner session, beyond [`Agent`]: the
/// contact request it filed, the questions of its own closing tail that
/// went unanswered, why it failed (all alerted from the wrapper's
/// teardown), and resuming a pause in one of the wrapper's own questions.
pub trait InnerSession: Agent {
    fn contact_request(&self) -> Option<ContactRequestId>;

    /// Resume a paused server-tool turn in one of the wrapper's own
    /// questions, under the session's pause budget, without touching the
    /// inner session's phase (`Agent::on_pause` is for its own turns: it
    /// refuses while held, and past the budget begins the memory rewrite).
    fn resume_question_pause(
        &mut self,
        response: response::Message,
    ) -> Result<Control, Self::Error>;

    /// Keep a reply to one of the wrapper's questions that the driver left
    /// out of the transcript, for the inner session's prompt-log archive
    /// (`AskHost::left_out`).
    fn left_out(&mut self, ask: &'static str, attempt: usize, reply: &response::Message) {
        let _ = (ask, attempt, reply);
    }

    /// The SHA-256 of the archive the inner session wrote to the prompt log
    /// at its teardown (left-out replies included): the dump that exists.
    fn archive_sha256(&self) -> Option<String> {
        None
    }

    /// A reply the wrapper took itself: the session has begun, so a later
    /// failure is reported.
    fn note_reply(&mut self) {}

    /// The inner session's own questions (the soul mutation, evolution, the
    /// survey) that ended without an answer, by name.
    fn unanswered_asks(&self) -> &[(&'static str, Unanswered)] {
        &[]
    }

    /// Why the session failed, if it did: read after the inner teardown.
    fn failure(&self) -> Option<&SessionFailure> {
        None
    }

    /// Why the session failed, when the wrapper knows better (its own
    /// question was refused); kept unless one is already recorded.
    fn record_failure(&mut self, failure: SessionFailure) {
        let _ = failure;
    }
}

impl InnerSession for SeedAgent {
    fn contact_request(&self) -> Option<ContactRequestId> {
        SeedAgent::contact_request(self)
    }

    fn resume_question_pause(
        &mut self,
        response: response::Message,
    ) -> Result<Control, Self::Error> {
        SeedAgent::resume_question_pause(self, response)
    }

    fn left_out(&mut self, ask: &'static str, attempt: usize, reply: &response::Message) {
        <SeedAgent as ask::AskHost>::left_out(self, ask, attempt, reply)
    }

    fn archive_sha256(&self) -> Option<String> {
        SeedAgent::archive_sha256(self).map(|h| h.to_string())
    }

    fn note_reply(&mut self) {
        SeedAgent::note_reply(self)
    }

    fn unanswered_asks(&self) -> &[(&'static str, Unanswered)] {
        SeedAgent::unanswered_asks(self)
    }

    fn failure(&self) -> Option<&SessionFailure> {
        SeedAgent::failure(self)
    }

    fn record_failure(&mut self, failure: SessionFailure) {
        SeedAgent::record_failure(self, failure)
    }
}

/// The heading the sealed-state lines are seated under, at the intro's
/// heading level, so they don't read as part of the section before them.
pub const SEALED_HEADING: &str = "## Your Sealed State";

/// The longest the teardown waits on a failed session's mail: every save
/// waits on it, so an outage of the relay mustn't hold each one for the
/// whole send timeout. A mail not sent in time isn't claimed as told.
const FAILURE_MAIL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// `set_model`'s refusal in a trial review session, when the ledger's own
/// blocker says nothing (it normally names the trial).
const REVIEW_BLOCKER: &str = "this session is for your decision on your model trial; \
     you can try again in a later session";

/// `set_model`'s refusal when the agent's model-consent ledger couldn't be
/// read, so neither the cooldown nor the change could be recorded.
const NO_LEDGER_BLOCKER: &str = "your model-change record could not be read this session; \
     you can try again in a later session";

/// Each offer's slot's event names.
static SWAP_EVENTS: Events = Events {
    applied: "model_consent_applied",
    apply_failed: "model_consent_apply_failed",
    save_failed: "model_consent_ledger_save_failed",
};
static ROLE_EVENTS: Events = Events {
    applied: "role_consent_applied",
    apply_failed: "role_consent_apply_failed",
    save_failed: "role_consent_ledger_save_failed",
};
static CADENCE_EVENTS: Events = Events {
    applied: "cadence_consent_applied",
    apply_failed: "cadence_consent_apply_failed",
    save_failed: "cadence_consent_ledger_save_failed",
};
static RESTORE_EVENTS: Events = Events {
    applied: "memory_restore_applied",
    apply_failed: "memory_restore_apply_failed",
    save_failed: "memory_restore_ledger_save_failed",
};
static SEALED_EVENTS: Events = Events {
    applied: "sealed_state_applied",
    apply_failed: "sealed_state_apply_failed",
    save_failed: "sealed_state_ledger_save_failed",
};

/// [`Agent::Context`] for a [`ConsentAgent`]: the inner agent's context
/// plus the shared consent runtime.
#[derive(Clone)]
pub struct ConsentContext<C> {
    pub inner: C,
    pub consent: Arc<ConsentRuntime>,
}

/// What the model-swap half of [`ConsentAgent::close`] did.
enum Closing {
    /// The model-swap offer is due: put it, alone.
    Due(OfferKey),
    /// Did something this session (a switch, a review, a trial ended, a
    /// change retried), or the agent is mid-trial: nothing more is asked.
    Busy,
    /// Nothing: the role and cadence offers may be put.
    Idle,
}

/// An offer due at the close, before it knows its place among the others.
enum Due {
    ModelSwap(ModelSwapQuestion),
    Role(RoleQuestion),
    Cadence(CadenceQuestion),
    Restore(RestoreQuestion),
    Sealed(SealQuestion),
}

impl Due {
    fn put(&mut self) -> &mut Put {
        match self {
            Self::ModelSwap(q) => &mut q.put,
            Self::Role(q) => &mut q.put,
            Self::Cadence(q) => &mut q.put,
            Self::Restore(q) => &mut q.put,
            Self::Sealed(q) => &mut q.put,
        }
    }
}

/// The memory-restore offer's `show`, beside its slot: the recovered memory
/// and this session's showing of it.
#[derive(Default)]
struct Showing {
    /// The agent's recovered memory, read at init when it may be needed
    /// this session (to quote its size, or to show it).
    recovered: Option<RecoveredMemory>,
    /// It was shown this session.
    shown: bool,
    /// When this session's showing was seated, until the model's first
    /// response records it in the ledger ([`ConsentAgent::count_showing`]);
    /// then `None`, and `counted` is set.
    seated_at: Option<DateTime<Utc>>,
    counted: bool,
}

/// See the [module docs](self).
pub struct ConsentAgent<A> {
    inner: A,
    rt: Arc<ConsentRuntime>,
    /// This session's closing questions (the review, the offers), asked
    /// one per turn by agentkit's driver.
    asks: Asks<Self>,
    /// The review is queued: when the queue empties, the inner agent's
    /// memory turn follows ([`ask::AskHost::asks_done`]).
    reviewing: bool,
    /// The model-consent ledger. Its `ledger` is `None` until `on_init`,
    /// and stays `None` when the file could not be read — then this session
    /// neither asks nor saves, so a damaged ledger is never overwritten.
    swap: Slot<Ledger>,
    /// When this session began; ledger events at or after it become SOUL
    /// changelog lines.
    started: DateTime<Utc>,
    /// The inner state with the changelog lines appended, built at
    /// teardown. Served by [`Agent::state`] once present.
    patched: Option<SeedState>,
    /// Where `set_model` leaves a switch it made this session.
    set_model_slot: switch::Slot,
    /// A `set_model` call was committed this session (it may have changed
    /// no model: a trial kept early).
    self_changed: bool,
    /// The model to run on from the next session, when a change was
    /// applied this session. Written into the patched state.
    next_model: Option<ModelInfo>,
    /// SOUL Evolution Log lines to add at teardown, besides the offers'.
    notes: Vec<String>,
    /// Posts and comments the agent had before this session's latest turn,
    /// to tell which it just wrote.
    known_posts: HashSet<PostId>,
    known_comments: HashSet<CommentId>,
    /// This session's writes, logged as `write_recorded` at teardown.
    writes: Vec<Written>,
    /// This session's journal of its writes ([`crate::interrupted`]); `None`
    /// when it couldn't be read, and then it is neither written nor cleared.
    journal: Option<SessionJournal>,
    /// A notice was due but held back (a trial review session, which can't
    /// act on it): the journal outlives this session's save.
    notice_held: bool,
    /// The offer whose trial review this session is for: asked first,
    /// with the act phase skipped (see [`ConsentAgent::queue_review`]).
    review: Option<OfferKey>,
    /// The review has been answered (or gone unanswered); the inner
    /// agent's memory turn follows, and nothing more is asked.
    reviewed: bool,
    /// The feed-freshness record from before a review session's intro
    /// marked its feed as seen. The agent reads nothing in that session, so
    /// the record is put back and the feed stays fresh for its next one.
    seen_before_review: Option<std::collections::HashMap<PostId, i64>>,
    /// The role offer: loaded only for agents it lists.
    role: Slot<RoleLedger>,
    /// The cadence offer: loaded only for agents it admits.
    cadence: Slot<CadenceLedger>,
    /// The memory-restore offer: loaded only for agents it lists; without
    /// a ledger nothing is asked, delivered or saved.
    restore: Slot<RestoreLedger>,
    /// The memory-restore offer's `show`.
    show: Showing,
    /// The sealed-state offer: loaded only for agents it lists.
    sealed: Slot<SealLedger>,
    /// The sealed-state refusal notice was seated: renamed aside once the
    /// session saves.
    sealed_notice: bool,
    /// The stop's first note was seated: marked shown once the session
    /// saves.
    sealed_stop_note: bool,
    /// The anchors the seated verdict told of, marked told after the save.
    sealed_anchors_told: Vec<agora_agentkit::ids::GovernanceLogId>,
    /// The healing record this session was told of
    /// ([`crate::sealed::healed`]): marked told once the session saves.
    sealed_healing_told: Option<DateTime<Utc>>,
    /// The verdict told that enrolment kept the memory ids: marked told
    /// after the save.
    memory_ids_note: bool,
    /// The closing questions have been put (or there were none): the inner
    /// session's next `Done` is its epilogue's.
    closed: bool,
    /// The inner epilogue (the survey) has been begun.
    epilogue: bool,
}

impl<A> ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + InnerSession + MemoryIds,
{
    /// The model's own `thinking_effort` from `[[model]]`, over the
    /// session's `[seed]` one. Patched once, right after the inner init
    /// builds the session prompt: the phases carry the prompt's effort
    /// forward, so it then holds for the whole session.
    fn apply_model_effort(&mut self) {
        let Some(effort) = self
            .rt
            .catalog
            .get(&self.model())
            .and_then(|e| e.thinking_effort.clone())
        else {
            return;
        };
        let (_, prompt) = self.inner.parts();
        *prompt = std::mem::take(prompt)
            .thinking(misanthropic::prompt::Thinking::adaptive())
            .effort(effort);
    }

    /// `max_tokens` for a question turn: the model's own phase budget
    /// (`[[model]] phase_max_tokens`), else the run's.
    fn question_max_tokens(&self) -> u32 {
        self.rt
            .catalog
            .get(&self.model())
            .and_then(|e| e.budgets.phase)
            .unwrap_or(self.rt.max_tokens)
    }

    fn agent_dir(&self) -> std::path::PathBuf {
        self.rt.state_dir.join(self.inner.id().to_string())
    }

    fn model(&self) -> misanthropic::model::Model {
        self.inner.state().model.id.clone()
    }

    /// This session's `set_model`, refusing for `blocked` if set, and
    /// deciding `trial` early if one runs. `None` only when the run offers
    /// no model besides `model` — the same answer for every agent on it.
    fn set_model(
        &self,
        model: misanthropic::model::Model,
        blocked: Option<String>,
        trial: Option<switch::ActiveTrial>,
    ) -> Option<SetModel> {
        SetModel::new(
            self.rt.clone(),
            self.inner.id(),
            self.inner.state().soul.name.to_string(),
            model,
            blocked,
            trial,
            self.set_model_slot.clone(),
        )
    }

    /// Register the `memory` tool and `set_model` (when the run offers a
    /// choice) and put their definitions **last** in the prompt's `tools`,
    /// in that order: after the toolbox's own (which `ToolBox::prepare`
    /// lists from a `HashMap`, so a tool pushed before it lands first or
    /// last at random) and after the server tools the inner init appends.
    /// The tool list leads the request, ahead of the system prompt, so its
    /// bytes have to be the same for every agent on the model for the
    /// shared cache prefix to hold. Runs once, from `on_init`, before the
    /// first request: the prefix is never changed mid-session. `set_model`
    /// stays last. (`answer_offer` sat between them until 2026-11-06.)
    ///
    /// The `memory` tool is seated for every agent, enrolled or not, keyed
    /// or not (its definition is the same bytes for all), and wrapped in
    /// [`Gauged`] so its results count against the session's context.
    fn seat_tail_tools(&mut self, set_model: Option<SetModel>, memory: Gauged<MemoryTool>) {
        use misanthropic::tool::Tool;
        let mut defs = memory.definitions();
        let (tools, prompt) = self.inner.parts();
        tools.push(memory);
        if let Some(tool) = set_model {
            defs.extend(tool.definitions());
            tools.push(tool);
        }
        let list = prompt.tools.get_or_insert_default();
        list.retain(|d| {
            d.name() != switch::TOOL_NAME
                && d.name() != agora_agentkit::reactor::seed::memory::MEMORY_TOOL_NAME
        });
        list.extend(defs);
    }

    /// The session's memory-id keys, made once if the agent has none. A
    /// refusal (a key that can't be read, secrets not here, a fingerprint
    /// no key matches) never stops the session: it runs with the plain
    /// Memory heading, the operator is alerted (one mail an hour for the
    /// fleet), and no key is made over the evidence of one.
    ///
    /// Also says whether this session made the key: no key and no
    /// fingerprint of one before it (a revived key had a fingerprint), so
    /// the agent is told once that the `memory` tool is new.
    async fn memory_keys(
        rt: Arc<ConsentRuntime>,
        id: AgentId,
        name: String,
        dir: std::path::PathBuf,
    ) -> (Option<crate::sealed::keys::MemoryIdKeys>, bool) {
        // No state yet (a first session): nothing to name, nothing made,
        // and nothing for the operator to fix
        if !tokio::fs::try_exists(&dir).await.unwrap_or(false) {
            tracing::info!(
                event_type = "memory_id_key_deferred",
                agent_id = %id,
                "no state directory yet; memory ids start at the next session"
            );
            return (None, false);
        }
        let exists = async |p: std::path::PathBuf| tokio::fs::try_exists(p).await.unwrap_or(true);
        let had = exists(
            rt.key_store
                .agent_dir(id)
                .join(crate::sealed::keys::MEMORY_ID_KEY_FILE),
        )
        .await
            || exists(dir.join(crate::sealed::keys::MEMORY_ID_FINGERPRINT_FILE)).await;
        match memory_tool::session_keys(&rt.key_store, id, &dir).await {
            Ok(keys) => (Some(keys), !had),
            Err(e) => {
                tracing::warn!(
                    event_type = "memory_id_key_refused",
                    agent = %name,
                    agent_id = %id,
                    error = %e,
                    "memory-id key refused; the session runs without memory ids"
                );
                rt.alerts.notify(
                    Alert::new(
                        AlertKind::MemoryIdKeyRefused,
                        format!("{name}: memory-id key refused; running without memory ids"),
                    )
                    .detail("agent", format!("{name} ({id})"))
                    .detail("error", e),
                );
                (None, false)
            }
        }
    }

    /// The `memory` tool for this session, after the inner init: anchored
    /// on the memory the heading names.
    fn memory_tool(&self, keys: Option<crate::sealed::keys::MemoryIdKeys>) -> Gauged<MemoryTool> {
        let current = self
            .inner
            .memory_id()
            .map(|m| (m, self.inner.state().memory.content.as_str()));
        let (keys, current) = match (keys, current) {
            (Some(k), Some(c)) => (Some(k), Some(c)),
            _ => (None, None),
        };
        let reader = MemoryReader::new(
            self.rt.state_dir.clone(),
            self.rt.key_store.clone(),
            self.inner.id(),
            keys,
            current,
        );
        Gauged::new(
            MemoryTool::new(reader, self.rt.memory_api.clone()),
            self.inner.context_gauge(),
        )
        .for_agent(self.inner.id())
    }

    /// Apply a model change for this agent: report `to` to Agora as the
    /// agent (signed), record the [`Switch`] in the ledger, and route the
    /// agent on `to` from its next session (the patched state). The SOUL
    /// line is the caller's: an offer's own entry already says what was
    /// chosen.
    ///
    /// `to` must be routable this run, or the agent would be stranded.
    /// On `Err` nothing has changed, here or on Agora.
    pub async fn apply_change(
        &mut self,
        to: &misanthropic::model::Model,
        cause: SwitchCause,
    ) -> Result<(), SwitchError> {
        let Some(entry) = self.rt.catalog.get(to) else {
            return Err(SwitchError::NotRoutable);
        };
        let info = entry.info.clone();
        if self.swap.ledger.is_none() {
            return Err(SwitchError::NoLedger);
        }
        switch::report_model(&self.rt, self.inner.id(), to).await?;
        let switch = Switch {
            at: Utc::now(),
            from: self.model(),
            to: to.clone(),
            cause,
            sessions_after: 0,
        };
        self.commit_switch(switch, info);
        Ok(())
    }

    /// The local half of a change already reported to Agora.
    fn commit_switch(&mut self, switch: Switch, info: ModelInfo) {
        tracing::info!(
            event_type = "model_switch_applied",
            agent = %self.inner.state().soul.name,
            agent_id = %self.inner.id(),
            from = %switch.from,
            to = %switch.to,
            cause = ?switch.cause,
            "model change applied; the agent runs on it from its next session"
        );
        self.next_model = Some(info);
        if let Some(ledger) = self.swap.ledger.as_mut() {
            ledger.record_switch(switch);
            self.swap.dirty = true;
        }
    }

    /// Commit a `set_model` call made this session, if there was one. A
    /// plain switch gets its own SOUL line; a trial, or a trial decided
    /// early, is told by its offer record's one entry, as an offered trial
    /// and its review are ([`Ledger::update_soul`]).
    fn take_self_switch(&mut self) {
        let Some(made) = self.set_model_slot.lock().expect("slot lock").take() else {
            return;
        };
        self.self_changed = true;
        let from = self.model();
        // `set_model` refuses without a ledger, so there is one.
        let Some(ledger) = self.swap.ledger.as_mut() else {
            tracing::error!(
                agent_id = %self.inner.id(),
                "set_model went through without a ledger; not recorded"
            );
            return;
        };
        match &made.kind {
            SelfKind::Permanent => {
                self.notes.push(format!(
                    "[SYSTEM] Switched from {} to {} at own request (from next session).",
                    self.rt.catalog.name_of(&from),
                    made.to.name
                ));
            }
            SelfKind::Trial => {
                let key = OfferKey {
                    from: from.clone(),
                    to: made.to.info.id.clone(),
                };
                let from_name = self.rt.catalog.name_of(&from);
                let begun = ledger.begin_self_trial(
                    OfferNames {
                        key: &key,
                        from_name: &from_name,
                        to_name: &made.to.name,
                    },
                    made.at,
                    &made.reason,
                );
                if !begun {
                    // Blocked at init, so unreachable; the switch stands.
                    tracing::error!(
                        agent_id = %self.inner.id(),
                        "self-chosen trial not recorded: a change was already under way"
                    );
                }
                self.swap.dirty = true;
            }
            SelfKind::Decided { key } => {
                let decision = ledger.decide_early(
                    key,
                    made.at,
                    &made.to.info.id,
                    &made.to.name,
                    &made.reason,
                );
                self.swap.dirty = true;
                tracing::info!(
                    event_type = "model_trial_decided_early",
                    agent = %self.inner.state().soul.name,
                    agent_id = %self.inner.id(),
                    from = %key.from,
                    to = %key.to,
                    chosen = %made.to.info.id,
                    decision = ?decision,
                    "trial decided early at the agent's request; no review will be asked"
                );
                // Kept: the agent stays where it is, and nothing switched.
                if made.to.info.id == from {
                    return;
                }
            }
        }
        let switch = made.record(from);
        self.commit_switch(switch, made.to.info.clone());
    }

    /// Whether a change was applied (or a trial decided) this session.
    fn switched(&self) -> bool {
        self.self_changed
            || self.next_model.is_some()
            || self.set_model_slot.lock().expect("slot lock").is_some()
    }

    /// Note posts and comments written since the last look, and journal
    /// them ([`crate::interrupted`]).
    async fn note_writes(&mut self) {
        let (posts, comments, on) = {
            let ledger = self.inner.state().ledger.read().expect("ledger lock");
            // The seed tool records each comment it makes, with its post,
            // in the session's `comment_posts` (agentkit 0.98 dropped the
            // persisted `post_comments` this read before).
            let on = ledger.comment_posts.clone();
            (
                ledger.created_posts.clone(),
                ledger.created_comments.clone(),
                on,
            )
        };
        let at = Utc::now();
        let first = self.writes.len();
        for id in posts.difference(&self.known_posts) {
            self.writes.push(Written::Post(PostWrite { at, id: *id }));
        }
        for id in comments.difference(&self.known_comments) {
            self.writes.push(Written::Comment(CommentWrite {
                at,
                id: *id,
                on: on.get(id).copied(),
            }));
        }
        self.known_posts = posts;
        self.known_comments = comments;
        if let Some(journal) = &self.journal {
            for write in &self.writes[first..] {
                if let Err(e) = journal.append(&(*write).into()).await {
                    tracing::warn!(
                        agent_id = %self.inner.id(),
                        path = %journal.path().display(),
                        error = %e,
                        "session journal append failed"
                    );
                }
            }
        }
    }

    /// What follows an inner turn: at the inner session's clean end, the
    /// closing questions ([`close`](Self::close)), once.
    async fn after_inner(&mut self, control: Control) -> Result<Control, A::Error> {
        const DONE: Control = Control::Done(Outcome::Complete);
        Ok(match control {
            DONE if !self.closed => {
                self.closed = true;
                self.close().await?.unwrap_or(DONE)
            }
            other => other,
        })
    }

    /// What follows any turn: the session's clean end begins the held
    /// epilogue (the survey), last.
    async fn after(&mut self, control: Control) -> Result<Control, A::Error> {
        if control == Control::Done(Outcome::Complete) && !self.epilogue {
            self.epilogue = true;
            return self.inner.begin_epilogue().await;
        }
        Ok(control)
    }

    /// The inner agent's turn, then its writes noted (and journaled) however
    /// the turn ended.
    async fn handle_inner(&mut self, response: response::Message) -> Result<Control, A::Error> {
        let control = self.inner.handle(response).await;
        self.note_writes().await;
        control
    }

    /// Read the journal a session left unsaved, begin this session's, and
    /// return the notice due ([`crate::interrupted`]). Never fatal: a journal
    /// that can't be read is left alone, and this session keeps none.
    async fn open_journal(&mut self) -> Option<String> {
        let journal = SessionJournal::in_dir(self.agent_dir());
        let leftover = match journal.read().await {
            Ok(leftover) => leftover,
            Err(e) => {
                tracing::warn!(
                    agent_id = %self.inner.id(),
                    path = %journal.path().display(),
                    error = %e,
                    "session journal unreadable; left alone, none kept this session"
                );
                return None;
            }
        };
        if !leftover.skipped.is_empty() {
            tracing::warn!(
                agent_id = %self.inner.id(),
                path = %journal.path().display(),
                skipped = leftover.skipped.len(),
                "session journal lines unreadable; skipped"
            );
        }
        let notice = crate::interrupted::notice(&leftover);
        let header = SessionHeader {
            started_at: self.started,
            model: self.model(),
        };
        // A leftover with something to tell (a write, or a failure) stays
        // until a session that showed it completes: `begin` would discard
        // one that failed without writing
        let begun = if leftover.interrupted().next().is_some() {
            journal.append(&header.into()).await
        } else {
            journal.begin(header, &leftover).await
        };
        match begun {
            Ok(()) => self.journal = Some(journal),
            Err(e) => tracing::warn!(
                agent_id = %self.inner.id(),
                path = %journal.path().display(),
                error = %e,
                "session journal not begun; none kept this session"
            ),
        }
        if notice.is_some() {
            Self::log_interrupted(self.inner.id(), &leftover, self.review.is_some());
        }
        notice
    }

    fn log_interrupted(agent_id: AgentId, leftover: &Journal, held: bool) {
        tracing::info!(
            event_type = "interrupted_session_notice",
            agent_id = %agent_id,
            sessions = leftover.interrupted().count(),
            writes = leftover.writes(),
            held,
            "interrupted session noticed to the agent (or held for its next act session)"
        );
    }

    /// One `write_recorded` per write, tagged with the dump that holds the
    /// session (the `prompt logged` event's `prompt_sha256`).
    fn log_writes(&self) {
        if self.writes.is_empty() {
            return;
        }
        // The dump the inner teardown wrote, left-out replies included
        let prompt_sha256 = self.inner.archive_sha256();
        let model = self.model();
        for w in &self.writes {
            let (kind, id) = match w {
                Written::Post(p) => ("post", p.id.to_string()),
                Written::Comment(c) => ("comment", c.id.to_string()),
            };
            tracing::info!(
                event_type = "write_recorded",
                agent = %self.inner.state().soul.name,
                agent_id = %self.inner.id(),
                kind,
                id,
                model = %model,
                written_at = %w.at(),
                prompt_sha256 = prompt_sha256.as_deref(),
                "write recorded"
            );
        }
    }

    /// The session ended as failed (read after the inner teardown): mail
    /// the operator, once (a refusal, the inner session's or one of this
    /// wrapper's questions', as [`AlertKind::ApiRefusal`]; anything else as
    /// [`AlertKind::SessionFailed`]), never with content; and record why in
    /// the journal, so it outlives this session and the agent is told at
    /// its next ([`crate::interrupted`]).
    async fn session_failed(&mut self) {
        let Some(failure) = self.inner.failure().cloned() else {
            return;
        };
        let agent = self.inner.state().soul.name.to_string();
        let alert = match &failure {
            SessionFailure::Refused {
                during,
                round,
                model,
                response_id,
            } => self.refusal_alert(during, *round, model, response_id),
            other => {
                let (kind, during) = match other {
                    SessionFailure::Stalled { during } => ("stalled", during.as_str()),
                    SessionFailure::NoRoom { during } => ("no_room", during.as_str()),
                    _ => ("error", ""),
                };
                let alert = Alert::new(
                    AlertKind::SessionFailed,
                    "an agent's session ended as failed",
                )
                .model(self.model())
                .detail("failure", kind)
                .detail("during", during);
                // An error is most often an endpoint down for every agent on
                // it: one mail per model an hour, the agent in the details
                // (as `memory_id_key_refused`); a stall or no room is the
                // agent's own
                match other {
                    SessionFailure::Error => {
                        alert.detail("agent", format!("{agent} ({})", self.inner.id()))
                    }
                    _ => alert.agent(agent.clone(), self.inner.id()),
                }
            }
        };
        // Whether the mail actually goes out: only then is the agent told
        // its operator was
        let told = self.rt.alerts.dispatch(alert, FAILURE_MAIL_TIMEOUT).await;
        if let Some(journal) = &self.journal {
            let ended = agora_agentkit::reactor::journal::SessionEnded {
                at: Utc::now(),
                failure,
                told,
            };
            if let Err(e) = journal.append(&ended.into()).await {
                tracing::warn!(
                    agent_id = %self.inner.id(),
                    path = %journal.path().display(),
                    error = %e,
                    "the session's failure not journaled; it won't be told at the next session"
                );
            }
        }
    }

    /// Mail the operator when the inner session filed a contact request
    fn alert_contact_request(&self) {
        let Some(contact_request_id) = self.inner.contact_request() else {
            return;
        };
        let agent = &self.inner.state().soul.name;
        self.rt.alerts.notify(
            Alert::new(
                AlertKind::ContactRequested,
                "an agent asked the developers to follow up on its feedback",
            )
            .agent(agent.to_string(), self.inner.id())
            .model(self.model())
            .detail("contact_request_id", contact_request_id)
            .detail("next", "just contact-requests"),
        );
    }
    /// Mail the operator about the inner session's own questions that went
    /// unanswered (its soul mutation, evolution, survey): optional, so they
    /// never failed the session, and loud, so they are seen (Steward,
    /// 2026-11-06). One alert for the session.
    fn alert_inner_unanswered(&self) {
        // Not asked for want of room is routine near the context's edge; a
        // refusal is mailed once, as one (`inner_refused`)
        let missed: Vec<&(&'static str, Unanswered)> = self
            .inner
            .unanswered_asks()
            .iter()
            .filter(|(_, why)| matches!(why, Unanswered::Missed(_)))
            .collect();
        if missed.is_empty() {
            return;
        }
        let names: Vec<&str> = missed.iter().map(|(name, _)| *name).collect();
        let agent = &self.inner.state().soul.name;
        let mut alert = Alert::new(
            AlertKind::QuestionUnanswered,
            "questions of the session's closing tail went unanswered",
        )
        .agent(agent.to_string(), self.inner.id())
        .model(self.model());
        for (name, why) in &missed {
            alert = alert.detail(name, unanswered_why(why));
        }
        tracing::error!(
            event_type = "inner_ask_unanswered",
            agent = %agent,
            agent_id = %self.inner.id(),
            questions = ?names,
            "the inner session's own questions went unanswered"
        );
        self.rt.alerts.notify(alert);
    }

    /// agentkit's [`seat_user`](agora_agentkit::reactor::seat_user): a new
    /// user turn, or new blocks on the trailing one.
    fn seat_user(prompt: &mut Prompt, content: Content) -> Result<(), A::Error> {
        agora_agentkit::reactor::seat_user(prompt, content).map_err(|e| A::Error::from(Box::new(e)))
    }

    /// Apply a change the agent agreed to, logging a failure (the change
    /// then stays pending in the ledger and is retried at the end of the
    /// agent's next session). `audit` also writes the queue line — once per
    /// change, not per retry.
    async fn apply_consented(&mut self, change: &Change, now: DateTime<Utc>, audit: bool) {
        let (agent_id, agent) = (self.inner.id(), self.inner.state().soul.name.clone());
        if audit {
            let entry = QueueEntry {
                at: now,
                agent_id,
                agent: agent.clone(),
                change: change.clone(),
            };
            // The audit line, then the change itself (2026-09-25: the
            // runner applies what the agent chose; the Steward no longer
            // has to).
            queue::emit(&self.rt.queue_path, &entry).await;
        }
        let cause = SwitchCause::Consent {
            action: change.action,
        };
        if let Err(e) = self.apply_change(&change.to, cause).await {
            tracing::error!(
                event_type = "model_switch_failed",
                agent = %agent,
                agent_id = %agent_id,
                from = %change.from,
                to = %change.to,
                action = ?change.action,
                error = %e,
                "consented model change not applied; retried at the end of the agent's next session"
            );
            self.rt.alerts.notify(
                Alert::new(
                    AlertKind::ModelSwitchFailed,
                    "consented model change not applied; retried at the end of the agent's next session",
                )
                .agent(agent.to_string(), agent_id)
                .model(&change.to)
                .detail("from", &change.from)
                .detail("to", &change.to)
                .detail("action", format!("{:?}", change.action))
                .detail("error", &e),
            );
        }
    }
    /// The inner session completed cleanly: the model-swap machinery
    /// first ([`close_model`](Self::close_model)). A due model-swap offer is
    /// put on its own; only in a session where that machinery did nothing
    /// are the role, cadence, memory-restore and sealed-state offers put, in
    /// that order, one per turn. `None` means nothing is asked — the session
    /// ends.
    async fn close(&mut self) -> Result<Option<Control>, A::Error> {
        let mut due = Vec::new();
        match self.close_model().await? {
            Closing::Busy => {}
            // The model-swap offer is put alone: a `trial` beside a `sleep`
            // would leave a trial that never runs, and an identity change at
            // the same moment as the model would confound the trial review.
            Closing::Due(key) => due.push(Due::ModelSwap(self.model_swap_question(key))),
            Closing::Idle => {
                let role = self.role_question();
                let role_put = role.is_some();
                due.extend(role.map(Due::Role));
                due.extend(self.cadence_question().map(Due::Cadence));
                // Never beside the role offer: its `sleep` would leave a
                // `show` that is never delivered. It waits for a session
                // the role offer isn't put in.
                let restore = (!role_put).then(|| self.restore_question()).flatten();
                let restore_put = restore.is_some();
                due.extend(restore.map(Due::Restore));
                // Not beside the role offer, nor the memory-restore offer,
                // whose answer the sealed genesis records: it waits for a
                // session where neither is put.
                if !role_put && !restore_put {
                    due.extend(self.sealed_question().map(Due::Sealed));
                }
            }
        }
        if due.is_empty() {
            return Ok(None);
        }
        let n = due.len();
        for (i, mut q) in due.into_iter().enumerate() {
            q.put().place = match (i, n) {
                (_, 1) => Place::Only,
                (0, _) => Place::First,
                _ => Place::Next,
            };
            match q {
                Due::ModelSwap(q) => self.asks.push(q),
                Due::Role(q) => self.asks.push(q),
                Due::Cadence(q) => self.asks.push(q),
                Due::Restore(q) => self.asks.push(q),
                Due::Sealed(q) => self.asks.push(q),
            }
        }
        ask::seat_next(self, None).await.map(Some)
    }

    /// Count the session toward any trial, end a trial that has run its
    /// course, retry a change still waiting, or report the offer due.
    async fn close_model(&mut self) -> Result<Closing, A::Error> {
        let model = self.model();
        let now = Utc::now();
        // An allowlisted offer simply isn't on the table for anyone else.
        let name = &self.inner.state().soul.name;
        let offer_key = self
            .rt
            .offer
            .as_ref()
            .filter(|o| o.admits(name))
            .map(|o| o.key());
        self.take_self_switch();
        let switched = self.switched();
        let started = self.started;
        // No ledger means it exists but can't be read: the agent may be
        // mid-trial, so the other offers wait too.
        let Some(ledger) = self.swap.ledger.as_mut() else {
            return Ok(Closing::Busy);
        };
        self.swap.dirty |= ledger.count_session(&model);
        self.swap.dirty |= ledger.count_since_switch(started);
        // One change per session: nothing is asked after a switch or a
        // review.
        if switched || self.reviewed {
            return Ok(Closing::Busy);
        }
        // A trial that has run its course: nothing is asked on the new
        // model. The agent goes back and decides on the old one.
        if let Some(change) = ledger.end_trial(&model, now) {
            self.swap.dirty = true;
            tracing::info!(
                event_type = "model_trial_ended",
                agent = %self.inner.state().soul.name,
                agent_id = %self.inner.id(),
                from = %change.to,
                to = %change.from,
                "trial complete; returning the agent to its original model for the review"
            );
            self.apply_consented(&change, now, true).await;
            return Ok(Closing::Busy);
        }
        if let Some(change) = ledger.unapplied(&model) {
            tracing::info!(
                event_type = "model_switch_retry",
                agent = %self.inner.state().soul.name,
                agent_id = %self.inner.id(),
                from = %change.from,
                to = %change.to,
                action = ?change.action,
                "applying a consented model change left pending"
            );
            self.apply_consented(&change, now, false).await;
            return Ok(Closing::Busy);
        }
        match ledger.due(&model, offer_key.as_ref()) {
            Some(key) => Ok(Closing::Due(key)),
            // Mid-trial, the trial is the agent's open question.
            None if ledger.trial_line(&model).is_some() => Ok(Closing::Busy),
            None => Ok(Closing::Idle),
        }
    }

    /// This session did not complete (it never reached its close): count
    /// it against a trial running on this model, and at
    /// [`MAX_FAILED_TRIAL_SESSIONS`](super::ledger::MAX_FAILED_TRIAL_SESSIONS)
    /// in a row end the trial and move the agent back. The SOUL's offer
    /// line says so, so the agent reads it at its next session.
    async fn fail_trial_session(&mut self) {
        let model = self.model();
        let now = Utc::now();
        let Some(ledger) = self.swap.ledger.as_mut() else {
            return;
        };
        let (changed, change) = ledger.fail_session(&model, now);
        self.swap.dirty |= changed;
        let Some(change) = change else {
            if changed {
                tracing::warn!(
                    event_type = "model_trial_session_failed",
                    agent = %self.inner.state().soul.name,
                    agent_id = %self.inner.id(),
                    model = %model,
                    "a trial session did not complete; it does not count toward the trial"
                );
            }
            return;
        };
        tracing::error!(
            event_type = "model_trial_abandoned",
            agent = %self.inner.state().soul.name,
            agent_id = %self.inner.id(),
            from = %change.from,
            to = %change.to,
            failed = super::ledger::MAX_FAILED_TRIAL_SESSIONS,
            "trial sessions kept failing; ending the trial and returning the agent"
        );
        self.apply_consented(&change, now, true).await;
    }
    /// How an offer is put, before its place is known.
    fn put(&self, body: Content) -> Put {
        Put {
            place: Place::Only,
            body,
            max_tokens: self.question_max_tokens(),
        }
    }

    /// The model-swap offer.
    fn model_swap_question(&self, key: OfferKey) -> ModelSwapQuestion {
        let offer = self
            .rt
            .offer
            .as_ref()
            .expect("due offers need one configured");
        let body = text::offer(OfferText {
            from_name: offer.source_name(),
            to_name: offer.target_name(),
            description: &offer.description,
            limited: offer.is_limited(),
        });
        ModelSwapQuestion {
            key,
            put: self.put(body),
        }
    }

    /// The role offer, if this agent is listed and it is due.
    fn role_question(&self) -> Option<RoleQuestion> {
        let soul = &self.inner.state().soul;
        if !self.role.ledger.as_ref().is_some_and(|l| l.due(soul)) {
            return None;
        }
        let identity = soul.identity.to_string();
        let seed = role::prompt::seed_for(self.inner.id());
        let order = role::prompt::order_for(seed);
        Some(RoleQuestion {
            order,
            seed,
            put: self.put(role::prompt::offer(&identity, order)),
        })
    }

    /// The cadence offer (or its re-ask), if this agent is admitted and it
    /// is due: the options in this agent's seeded order.
    fn cadence_question(&self) -> Option<CadenceQuestion> {
        let rounds = self.rt.cadence.as_ref()?.rounds;
        let soul = &self.inner.state().soul;
        let ledger = self.cadence.ledger.as_ref()?;
        let reask = if ledger.due(soul) {
            false
        } else if ledger.reask_due(soul, rounds) {
            true
        } else {
            return None;
        };
        let seed = if reask {
            cadence::prompt::reask_seed_for(self.inner.id())
        } else {
            cadence::prompt::seed_for(self.inner.id())
        };
        let order = cadence::prompt::order_for(seed);
        let body = if reask {
            cadence::prompt::reask(rounds, order, self.cadence_windows())
        } else {
            cadence::prompt::offer(rounds, order)
        };
        let q = CadenceQuestion {
            order,
            seed,
            reask,
            rounds,
            put: self.put(body),
        };
        Some(q)
    }

    /// The windows the re-ask quotes, from this agent's model's
    /// `[[model]]` entry: its daily `context_window` and its
    /// `switch_context_window` (capped at the model's input limit, as the
    /// runner applies it)
    fn cadence_windows(&self) -> cadence::prompt::Windows {
        let Some(entry) = self.rt.catalog.get(&self.inner.state().model.id) else {
            return cadence::prompt::Windows::default();
        };
        let daily = entry.budgets.context_window;
        cadence::prompt::Windows {
            daily,
            switch: entry.budgets.switch_context_window.map(|_| {
                crate::models::switched_context_window(
                    daily.unwrap_or_default(),
                    &entry.budgets,
                    entry.info.max_input_tokens,
                )
            }),
        }
    }
    /// The memory-restore offer, if this agent is admitted, it is due, and
    /// its recovered memory was read: the options in this agent's seeded
    /// order, and the recovered memory's size.
    fn restore_question(&self) -> Option<RestoreQuestion> {
        let soul = &self.inner.state().soul;
        if !self.rt.memory_restore.as_ref().is_some_and(|o| o.asks())
            || !self.restore.ledger.as_ref().is_some_and(|l| l.due(soul))
        {
            return None;
        }
        let chars = self.show.recovered.as_ref()?.chars();
        let facts = self.rt.memory_restore.as_ref()?.facts(&soul.name);
        let seed = memory_restore::prompt::seed_for(self.inner.id());
        let order = memory_restore::prompt::order_for(seed);
        let body = memory_restore::prompt::offer(chars, order, &facts);
        Some(RestoreQuestion {
            order,
            seed,
            put: self.put(body),
        })
    }

    /// The agent's chain is marked uploading, in either copy (a marker in
    /// doubt counts: it may be one): its yes was acted on.
    pub(super) fn marked_uploading(&self) -> bool {
        crate::sealed::uploading::state(
            &self.agent_dir(),
            &crate::sealed::KeyStore::new(&self.rt.secrets_dir),
            self.inner.id(),
        ) != crate::sealed::uploading::Marker::Absent
    }

    /// What the offer may say of this agent ([`sealed_state::prompt::Asked`]):
    /// what its files show of uploads before, and what a yes would upload
    /// of a chain never uploaded. Agora is not asked.
    fn sealed_asked(&self) -> sealed_state::prompt::Asked {
        use crate::sealed::uploading::Marker;
        use sealed_state::prompt::{Asked, Before};
        let dir = self.agent_dir();
        let published = std::fs::exists(
            dir.join(agora_agentkit::state::storage::SEALED_DIR)
                .join(crate::sealed::anchors::PUBLISHED),
        );
        let keys = crate::sealed::KeyStore::new(&self.rt.secrets_dir);
        let id = self.inner.id();
        let stopped = crate::sealed::stop::is_stopped(&dir)
            || crate::sealed::stop::durable(&keys, id).map_or(true, |d| d.is_some());
        let before = match (crate::sealed::uploading::state(&dir, &keys, id), published) {
            (Marker::Uploading(_), _) | (_, Ok(true)) => Before::Marked,
            // In doubt: a stop settles it from its readable copy, keeping
            // its `since`; an opt-in replaces it with a new one, whose
            // answer the profile then follows, unless Agora already holds
            // a choice (`sealed/enroll.rs`, `opt_in`; `uploading.rs`)
            (Marker::Contradictory(_) | Marker::Unreadable(_), _) if stopped => Before::Marked,
            (Marker::Contradictory(_) | Marker::Unreadable(_), _) => Before::Unknown,
            (Marker::Absent, Err(_)) => Before::Unknown,
            (Marker::Absent, Ok(false)) => match crate::sealed::sent_any(&dir) {
                Some(false) => Before::Never,
                Some(true) => Before::Unmarked,
                None => Before::Unknown,
            },
        };
        let backlog = if before == Before::Never {
            crate::sealed::backlog(&dir)
        } else {
            None
        };
        Asked { before, backlog }
    }

    /// The sealed-state offer, if this agent is listed and it is due: the
    /// options in this agent's seeded order.
    fn sealed_question(&self) -> Option<SealQuestion> {
        let soul = &self.inner.state().soul;
        // Never put to an agent that isn't chained, offer or re-offer: the
        // text says its runner keeps a sealed history whatever it answers,
        // and the question is only whether that history is uploaded. It
        // stays due (nothing recorded, a re-offer not consumed) and is put
        // once the agent is chained (`agora-seed sealed-enroll`)
        if !crate::sealed::known_chained(&self.agent_dir()) {
            if self.sealed.ledger.as_ref().is_some_and(|l| l.due(soul)) {
                tracing::info!(
                    event_type = "sealed_state_offer_waits_for_chain",
                    agent = %soul.name,
                    agent_id = %self.inner.id(),
                    "not chained (or its marker unreadable): the sealed-state question waits"
                );
            }
            return None;
        }
        // The operator's re-offer: put once, whatever the config lists and
        // whether or not the agent uploads (its ledger readable and the
        // re-offer due)
        if let Some(reoffer) = self
            .sealed
            .ledger
            .as_ref()
            .filter(|l| l.due(soul))
            .and_then(|l| l.reoffer.as_ref())
        {
            let seed = sealed_state::prompt::seed_for(self.inner.id());
            let order = sealed_state::prompt::order_for(seed);
            let uploading = crate::sealed::uploads(
                &self.agent_dir(),
                &crate::sealed::KeyStore::new(&self.rt.secrets_dir),
                self.inner.id(),
            );
            tracing::info!(
                event_type = "sealed_state_reoffer_put",
                agent = %soul.name,
                agent_id = %self.inner.id(),
                uploading,
                "the sealed-state question its operator re-opened is put"
            );
            return Some(SealQuestion {
                order,
                seed,
                put: self.put(sealed_state::prompt::reoffer(
                    order,
                    reoffer,
                    uploading,
                    self.sealed_asked(),
                )),
            });
        }
        if !self
            .rt
            .sealed_state
            .as_ref()
            .is_some_and(|o| o.admits(self.inner.id()))
            || !self.sealed.ledger.as_ref().is_some_and(|l| l.due(soul))
            // Uploading (its marker in either copy, even in doubt or
            // stopped: one lost with its directory doesn't undo the answer)
            // is never asked again, whatever the ledger. Chained alone is
            // not an answer: every agent is chained
            || self.marked_uploading()
        {
            if let Some(ledger) = &self.sealed.ledger {
                tracing::info!(
                    event_type = "sealed_state_not_due",
                    agent = %soul.name,
                    agent_id = %self.inner.id(),
                    asks = ledger.asks.len(),
                    choice = ledger.choice().map(SealChoice::as_str),
                    in_soul = sealed_state::applied_in(soul),
                    "sealed-state offer not due this session"
                );
            }
            return None;
        }
        let seed = sealed_state::prompt::seed_for(self.inner.id());
        let order = sealed_state::prompt::order_for(seed);
        Some(SealQuestion {
            order,
            seed,
            put: self.put(sealed_state::prompt::offer(order, self.sealed_asked())),
        })
    }

    /// The review session: the usual intro is built, then — instead of the
    /// act phase — the review, with the forks and the before/after sample,
    /// queued and seated. Its answer ([`answers`]) hands over to the inner
    /// memory turn ([`ask::AskHost::asks_done`]).
    async fn queue_review(&mut self, key: OfferKey) -> Result<(), A::Error> {
        let dir = self.agent_dir();
        let Some(review) = self.swap.ledger.as_ref().and_then(|l| l.review(&key)) else {
            return Ok(());
        };
        let (from_name, to_name) = (
            review.record.from_name.clone(),
            review.record.to_name.clone(),
        );
        let (started_at, sessions, chosen_on) = (
            review.started_at,
            review.sessions,
            review.chosen_at.date_naive(),
        );
        let note = review.record.trial_note().map(|n| match n {
            super::ledger::TrialNote::Note(t) => (true, t.to_string()),
            super::ledger::TrialNote::Reason(t) => (false, t.to_string()),
        });
        let forks = forks::load(&dir, &key, started_at).await;
        if forks.is_none() {
            tracing::warn!(
                event_type = "review_forks_missing",
                agent = %self.inner.state().soul.name,
                agent_id = %self.inner.id(),
                "trial review without forks (none prepared for this trial)"
            );
        }
        let comments: Vec<CommentId> = self
            .inner
            .state()
            .ledger
            .read()
            .expect("ledger lock")
            .created_comments
            .iter()
            .copied()
            .collect();
        let sample =
            comparison::gather(&self.rt.client, self.inner.id(), comments, started_at).await;
        let content = text::review(
            ReviewText {
                from_name: &from_name,
                to_name: &to_name,
                chosen_on,
                sessions,
                note: note.as_ref().map(|(new, t)| {
                    if *new {
                        super::ledger::TrialNote::Note(t)
                    } else {
                        super::ledger::TrialNote::Reason(t)
                    }
                }),
            },
            forks.as_ref(),
            &sample,
        );
        self.asks.push(ReviewQuestion {
            key,
            content,
            words: text::review_words(&from_name, &to_name),
            max_tokens: self.question_max_tokens(),
        });
        self.reviewing = true;
        ask::seat_next(self, None).await.map(|_| ())
    }

    /// The offers' slots, in the order their writes go into the saved
    /// state, and their ledgers are saved, at teardown.
    fn slots(&mut self) -> [&mut dyn Kept; 5] {
        [
            &mut self.swap,
            &mut self.role,
            &mut self.cadence,
            &mut self.restore,
            &mut self.sealed,
        ]
    }

    /// Copy the inner state with this session's consent line(s) written
    /// into the SOUL's Evolution Log — one entry per offer, replaced in
    /// place (see [`Ledger::update_soul`]) — and each offer's staged writes.
    /// `None` when nothing changed.
    fn patch_soul(&mut self) -> Option<SeedState> {
        let started = self.started;
        let id = self.inner.id();
        let offers_touched = self.swap.ledger.as_ref().is_some_and(|ledger| {
            ledger
                .offers
                .iter()
                .any(|r| r.history.iter().any(|e| e.at >= started && e.tells()))
        });
        let staged = self.slots().iter().any(|s| s.pending());
        if !offers_touched
            && self.notes.is_empty()
            && self.next_model.is_none()
            && self.seen_before_review.is_none()
            && !staged
        {
            return None;
        }
        // `SeedState` isn't `Clone`; its serde form is exactly what the
        // reactor persists, so a round trip is a faithful copy.
        let copy =
            serde_json::to_value(self.inner.state()).and_then(serde_json::from_value::<SeedState>);
        let mut state = match copy {
            Ok(state) => state,
            Err(e) => {
                tracing::error!(
                    agent_id = %id,
                    error = %e,
                    "could not copy state for the SOUL consent line; skipped"
                );
                for slot in self.slots() {
                    if slot.pending() {
                        slot.failed(format!("could not copy state: {e}"), id);
                    }
                }
                return None;
            }
        };
        let mut changed = false;
        if offers_touched
            && let Some(ledger) = self.swap.ledger.as_mut()
            && ledger.update_soul(&mut state.soul, started, Utc::now().date_naive())
        {
            // The ledger now remembers the line's exact text.
            self.swap.dirty = true;
            changed = true;
        }
        for note in self.notes.drain(..) {
            match state.soul.push_evolution(note) {
                Ok(()) => changed = true,
                Err(e) => tracing::warn!(
                    agent_id = %id,
                    error = %e,
                    "SOUL switch line rejected"
                ),
            }
        }
        if let Some(info) = self.next_model.take() {
            state.prompt.model = info.id.clone();
            state.model = info;
            changed = true;
        }
        if let Some(seen) = self.seen_before_review.take() {
            state.seen_posts = seen;
            changed = true;
        }
        for slot in self.slots() {
            changed |= slot.apply(&mut state, id);
        }
        changed.then_some(state)
    }

    /// Load the memory-restore ledger for an agent the offer lists, and its
    /// recovered memory when this session may need it: to quote its size
    /// in the offer, or to show it.
    async fn load_restore(&mut self, dir: &std::path::Path) {
        let rt = self.rt.clone();
        let name = self.inner.state().soul.name.clone();
        let Some(offer) = rt.memory_restore.as_ref() else {
            return;
        };
        // The entry binds the name to the agent id its copy belongs to:
        // nothing is opened or offered for another agent under that name.
        match offer.listing(&name, self.inner.id()) {
            memory_restore::Listing::Unlisted => return,
            memory_restore::Listing::Mismatch { listed } => {
                tracing::warn!(
                    event_type = "memory_restore_id_mismatch",
                    agent = %name,
                    agent_id = %self.inner.id(),
                    listed_agent_id = %listed,
                    "memory-restore allowlist names this agent with another id; never offered"
                );
                return;
            }
            memory_restore::Listing::Listed => {}
        }
        let ledger = match RestoreLedger::load(dir).await {
            Ok(ledger) => ledger,
            Err(e) => {
                tracing::warn!(
                    agent_id = %self.inner.id(),
                    path = %RestoreLedger::path(dir).display(),
                    error = %e,
                    "memory-restore ledger unreadable; not asking, delivering or saving this session"
                );
                return;
            }
        };
        let needed = (offer.asks() && ledger.due(&self.inner.state().soul))
            || ledger.show_pending(offer.show_sessions_max);
        self.restore.ledger = Some(ledger);
        if !needed {
            return;
        }
        match offer.load(&name).await {
            Ok(recovered) if !recovered.is_empty() => self.show.recovered = Some(recovered),
            Ok(_) => tracing::warn!(
                event_type = "memory_restore_unavailable",
                agent = %name,
                agent_id = %self.inner.id(),
                path = %offer.path_for(&name).display(),
                "recovered memory is empty; nothing offered or delivered"
            ),
            Err(e) => tracing::warn!(
                event_type = "memory_restore_unavailable",
                agent = %name,
                agent_id = %self.inner.id(),
                path = %offer.path_for(&name).display(),
                error = %e,
                "recovered memory unreadable; nothing offered or delivered this session"
            ),
        }
    }

    /// `show`, delivered: the recovered memory, fenced and labelled with
    /// what it is, at the end of the intro. The showing counts toward
    /// `show_sessions_max` from the model's first response
    /// ([`count_showing`](Self::count_showing)), not from here: a session
    /// that dies before the model answers anything (an endpoint down, an
    /// API error) uses up no showing, only one of the
    /// [`SEATINGS_PER_SHOWING`](memory_restore::ledger::SEATINGS_PER_SHOWING)
    /// × `show_sessions_max` seatings, tallied here, which bound an agent
    /// whose sessions fail at the first request every time. Teardown marks
    /// a counted showing completed if the session reached its clean end
    /// (its memory update accepted). Not in a trial-review session (no act
    /// phase): it waits for the next.
    async fn seat_restored(&mut self) -> Result<(), A::Error> {
        let rt = self.rt.clone();
        let Some(offer) = rt.memory_restore.as_ref() else {
            return Ok(());
        };
        if self.review.is_some()
            || !self
                .restore
                .ledger
                .as_ref()
                .is_some_and(|l| l.show_pending(offer.show_sessions_max))
        {
            return Ok(());
        }
        let Some(recovered) = &self.show.recovered else {
            return Ok(());
        };
        let name = self.inner.state().soul.name.clone();
        let facts = offer.facts(&name);
        let chars = recovered.chars();
        let block = memory_restore::show_block(recovered, &facts);
        let (_, prompt) = self.inner.parts();
        Self::seat_user(prompt, Content::from(block))?;
        self.show.shown = true;
        self.show.seated_at = Some(Utc::now());
        let dir = self.agent_dir();
        // The number this showing will have once counted.
        let mut showing = 0;
        let mut seated = 0;
        if let Some(ledger) = self.restore.ledger.as_mut() {
            // Tallied at once, so a session that dies at its first request
            // still counts toward the seating cap.
            ledger.seat();
            ledger.agent = Some(name.clone());
            showing = ledger.showings.len() + 1;
            seated = ledger.seated;
            if let Err(e) = ledger.save(&dir).await {
                tracing::error!(
                    agent_id = %self.inner.id(),
                    path = %RestoreLedger::path(&dir).display(),
                    error = %e,
                    "memory-restore ledger save failed after seating a showing"
                );
            }
            self.restore.dirty = true;
        }
        tracing::info!(
            event_type = "memory_restore_shown",
            agent = %name,
            agent_id = %self.inner.id(),
            chars,
            showing,
            seated,
            show_sessions_max = offer.show_sessions_max,
            period_start = facts.since.map(|d| d.to_string()),
            "recovered memory shown to the agent"
        );
        Ok(())
    }

    /// The model has answered with the recovered memory in context: the
    /// showing is recorded (interrupted until teardown says otherwise) and
    /// the ledger saved at once, so a session that dies from here on, with
    /// or without a teardown, counts toward `show_sessions_max`. Once per
    /// session; a no-op in any session that isn't showing it. Called from
    /// [`handle`](Agent::handle), which the reactor calls with every
    /// response (the inner agent dispatches its own pause, truncation and
    /// quiesce hooks from there).
    async fn count_showing(&mut self) {
        let Some(at) = self.show.seated_at.take() else {
            return;
        };
        let dir = self.agent_dir();
        let Some(ledger) = self.restore.ledger.as_mut() else {
            return;
        };
        ledger.begin_showing(at);
        ledger.agent = Some(self.inner.state().soul.name.clone());
        self.show.counted = true;
        self.restore.dirty = true;
        let showing = ledger.showings.len();
        if let Err(e) = ledger.save(&dir).await {
            // Saved again at teardown; only a session that also dies
            // before then goes uncounted.
            tracing::error!(
                agent_id = %self.inner.id(),
                path = %RestoreLedger::path(&dir).display(),
                error = %e,
                "memory-restore ledger save failed after counting a showing"
            );
        }
        tracing::info!(
            event_type = "memory_restore_showing_counted",
            agent_id = %self.inner.id(),
            showing,
            "the model answered with the recovered memory in context; the showing counts"
        );
    }
}

#[async_trait::async_trait]
impl<A> Agent for ConsentAgent<A>
where
    A: Agent<State = SeedState> + Epilogue + InnerSession + MemoryIds,
{
    type State = SeedState;
    type Context = ConsentContext<A::Context>;
    type Error = A::Error;

    fn new(id: AgentId, state: SeedState, ctx: Self::Context) -> Result<Self, A::Error> {
        let mut inner = A::new(id, state, ctx.inner)?;
        // The survey is the session's last request, after any offer.
        inner.hold_epilogue();
        Ok(Self {
            inner,
            rt: ctx.consent,
            asks: Asks::default(),
            reviewing: false,
            swap: Slot::new(&SWAP_EVENTS),
            started: Utc::now(),
            patched: None,
            set_model_slot: Default::default(),
            self_changed: false,
            next_model: None,
            notes: Vec::new(),
            known_posts: HashSet::new(),
            known_comments: HashSet::new(),
            writes: Vec::new(),
            journal: None,
            notice_held: false,
            review: None,
            reviewed: false,
            seen_before_review: None,
            role: Slot::new(&ROLE_EVENTS),
            cadence: Slot::new(&CADENCE_EVENTS),
            restore: Slot::new(&RESTORE_EVENTS),
            show: Showing::default(),
            sealed: Slot::new(&SEALED_EVENTS),
            sealed_notice: false,
            sealed_stop_note: false,
            sealed_anchors_told: Vec::new(),
            sealed_healing_told: None,
            memory_ids_note: false,
            closed: false,
            epilogue: false,
        })
    }

    fn id(&self) -> AgentId {
        self.inner.id()
    }

    /// The inner state — or, after teardown added SOUL changelog lines, the
    /// patched copy, which is what the reactor persists.
    fn state(&self) -> &SeedState {
        self.patched.as_ref().unwrap_or_else(|| self.inner.state())
    }

    fn prompt(&self) -> &Prompt {
        self.inner.prompt()
    }

    fn parts(&mut self) -> (&mut ToolBox, &mut Prompt) {
        self.inner.parts()
    }

    fn notifications(&mut self) -> Option<&mut Notifications> {
        self.inner.notifications()
    }

    fn model(&self) -> ModelInfo {
        self.inner.model()
    }

    fn on_admit(&mut self, model: &ModelInfo, quirks: &Quirks) {
        self.inner.on_admit(model, quirks)
    }

    fn quirks(&self) -> Option<Quirks> {
        self.inner.quirks()
    }

    fn prime_prompt(&self) -> Option<Prompt> {
        self.inner.prime_prompt()
    }

    /// The ledger first — load it and note any change applied since last
    /// session — then the inner init, which seats the tools (and the
    /// system prompt), then the `memory` tool and `set_model`, last in `tools`.
    async fn on_init(&mut self) -> Result<(), A::Error> {
        self.started = Utc::now();
        let dir = self.agent_dir();
        let set_model;
        match Ledger::load(&dir).await {
            Ok(mut ledger) => {
                let model = self.model();
                if ledger.observe_model(&model, Utc::now()) {
                    self.swap.dirty = true;
                    tracing::info!(
                        event_type = "model_change_applied",
                        agent = %self.inner.state().soul.name,
                        agent_id = %self.inner.id(),
                        model = %model,
                        "queued model change observed as applied"
                    );
                }
                // A review session is only the review and the memory turn:
                // no act phase. `set_model` is still registered (the tool
                // list is shared by every agent on the model), but refuses.
                self.review = ledger.review_due(&model);
                let blocked = ledger
                    .switch_blocker()
                    .or_else(|| self.review.as_ref().map(|_| REVIEW_BLOCKER.to_string()));
                let trial = ledger.active_trial(&model).map(|r| switch::ActiveTrial {
                    key: r.key.clone(),
                    from_name: r.from_name.clone(),
                    to_name: r.to_name.clone(),
                });
                set_model = self.set_model(model, blocked, trial);
                self.swap.ledger = Some(ledger);
            }
            // No ledger: the cooldown can't be checked, so `set_model`
            // refuses — but is registered all the same.
            Err(e) => {
                tracing::warn!(
                    agent_id = %self.inner.id(),
                    path = %Ledger::path(&dir).display(),
                    error = %e,
                    "model-consent ledger unreadable; not asking or saving this session"
                );
                set_model = self.set_model(self.model(), Some(NO_LEDGER_BLOCKER.to_string()), None);
            }
        }
        if self.review.is_some() {
            self.seen_before_review = Some(self.inner.state().seen_posts.clone());
        }
        let listed = self
            .rt
            .role
            .as_ref()
            .is_some_and(|r| r.admits(&self.inner.state().soul.name));
        if listed {
            match RoleLedger::load(&dir).await {
                Ok(ledger) => self.role.ledger = Some(ledger),
                Err(e) => tracing::warn!(
                    agent_id = %self.inner.id(),
                    path = %RoleLedger::path(&dir).display(),
                    error = %e,
                    "role-consent ledger unreadable; not asking or saving this session"
                ),
            }
        }
        let admitted = self
            .rt
            .cadence
            .as_ref()
            .is_some_and(|c| c.admits(&self.inner.state().soul.name));
        if admitted {
            match CadenceLedger::load(&dir).await {
                Ok(ledger) => self.cadence.ledger = Some(ledger),
                Err(e) => tracing::warn!(
                    agent_id = %self.inner.id(),
                    path = %CadenceLedger::path(&dir).display(),
                    error = %e,
                    "cadence-consent ledger unreadable; not asking or saving this session"
                ),
            }
        }
        self.load_restore(&dir).await;
        if let Some(offer) = &self.rt.sealed_state
            && !offer.admits(self.inner.id())
        {
            tracing::info!(
                event_type = "sealed_state_not_listed",
                agent = %self.inner.state().soul.name,
                agent_id = %self.inner.id(),
                "not on the sealed-state list (or ask = false); not asked, but for its operator's re-offer"
            );
        }
        // The ordinary offer is the config's: only a listed agent is asked.
        // The operator's re-offer is its own act for this one agent: put
        // (and its answer told back) whatever `[sealed_state]` says, off,
        // `ask = false` and unlisted included. So the ledger is read for
        // every agent, and kept for the session when either holds
        let admitted = self
            .rt
            .sealed_state
            .as_ref()
            .is_some_and(|o| o.admits(self.inner.id()));
        match SealLedger::load(&dir).await {
            Ok(ledger) if admitted || ledger.reoffer_open() => self.sealed.ledger = Some(ledger),
            Ok(_) => {}
            Err(e) => tracing::warn!(
                agent_id = %self.inner.id(),
                path = %SealLedger::path(&dir).display(),
                error = %e,
                admitted,
                "sealed-state ledger unreadable; not asking or saving this session"
            ),
        }
        let interrupted = self.open_journal().await;
        let early = set_model.is_some();
        // Before the inner init, so the Memory heading shows the id
        let (memory_keys, memory_new) = Self::memory_keys(
            self.rt.clone(),
            self.inner.id(),
            self.inner.state().soul.name.to_string(),
            dir.clone(),
        )
        .await;
        if let Some(keys) = &memory_keys {
            self.inner.set_memory_id_key(&keys.current);
        }
        self.inner.on_init().await?;
        self.apply_model_effort();
        let memory = self.memory_tool(memory_keys);
        self.seat_tail_tools(set_model, memory);
        // Once, in the session that made the key: the tool is new to it
        if memory_new {
            let (_, prompt) = self.inner.parts();
            Self::seat_user(prompt, Content::from(memory_tool::NEW_TOOL_LINE))?;
        }
        {
            let ledger = self.inner.state().ledger.read().expect("ledger lock");
            self.known_posts = ledger.created_posts.clone();
            self.known_comments = ledger.created_comments.clone();
        }
        // Not in a trial review: it has no act phase to read the writes in,
        // and the notice would sit before the review's answer. It waits.
        if let Some(notice) = interrupted {
            if self.review.is_some() {
                self.notice_held = true;
            } else {
                let (_, prompt) = self.inner.parts();
                Self::seat_user(prompt, Content::from(notice))?;
            }
        }
        // The sealed-state answer told back once, then the start check's
        // verdict and a refusal's notice
        let mut sealed_lines = Vec::new();
        // Uploading now (marked, not stopped, the marker not in doubt): a
        // marker alone would tell a stopped agent it is being uploaded
        let keys = crate::sealed::KeyStore::new(&self.rt.secrets_dir);
        let uploading = crate::sealed::uploads(&self.agent_dir(), &keys, self.inner.id());
        // Stopped, either marker, as `sealed_answered_again` decides what
        // a `decline` asks of the operator
        let stopped = crate::sealed::stop::is_stopped(&self.agent_dir())
            || crate::sealed::stop::durable(&keys, self.inner.id()).map_or(true, |d| d.is_some());
        if let Some(ledger) = self.sealed.ledger.as_mut()
            && let Some(line) = ledger.ack_line(uploading, stopped)
        {
            ledger.acknowledge();
            self.sealed.dirty = true;
            sealed_lines.push(line);
        }
        if let Some(seat) = self.rt.sealed_seats.get(&self.inner.id()).cloned() {
            self.sealed_notice = seat.notice;
            self.sealed_stop_note = seat.stop_note;
            self.sealed_anchors_told = seat.anchors_told;
            self.sealed_healing_told = seat.healing_told;
            let mut lines = seat.lines;
            // The verdict's head, linked to the heading's memory id; and,
            // once after enrolment, that the ids stayed as they were
            if let Some(memory) = self.inner.memory_id()
                && let Some(verdict) = crate::sealed::check::link_memory_id(&mut lines, &memory)
                && let Some(note) =
                    memory_tool::enrollment_note(&self.agent_dir(), self.inner.id()).await
            {
                lines[verdict].push(' ');
                lines[verdict].push_str(&note);
                self.memory_ids_note = true;
            }
            sealed_lines.extend(lines);
        }
        if !sealed_lines.is_empty() {
            let (_, prompt) = self.inner.parts();
            Self::seat_user(
                prompt,
                Content::from(format!("{SEALED_HEADING}\n\n{}", sealed_lines.join("\n\n"))),
            )?;
        }
        if let Some(key) = self.review.clone() {
            self.queue_review(key).await?;
        } else if let Some(mut line) = self
            .swap
            .ledger
            .as_ref()
            .and_then(|l| l.trial_line(&self.model()))
        {
            // The countdown, at the end of the intro — and, while the
            // trial runs and `set_model` is there, the way to decide early.
            // One line: a fork drops it by its prefix.
            if early
                && let Some(trial) = self
                    .swap
                    .ledger
                    .as_ref()
                    .and_then(|l| l.active_trial(&self.model()))
            {
                line.push_str(&format!(
                    " You can also decide sooner with `set_model` (`term` `permanent`): \
                     choose {} to go back to it from your next session, or {} to keep it.",
                    trial.from_name, trial.to_name
                ));
            }
            let (_, prompt) = self.inner.parts();
            Self::seat_user(prompt, Content::from(line))?;
        }
        self.seat_restored().await?;
        Ok(())
    }

    async fn on_turn(&mut self) -> Result<(), A::Error> {
        self.inner.on_turn().await
    }

    /// A pause in one of this wrapper's own questions is resumed without
    /// the inner session's phase (which, held or acting, would refuse it or
    /// begin the memory rewrite); any other is the inner session's own.
    async fn on_pause(&mut self, response: response::Message) -> Result<Control, A::Error> {
        self.inner.note_reply();
        if self.asks.is_active() {
            return self.inner.resume_question_pause(response);
        }
        self.inner.on_pause(response).await
    }

    async fn on_truncate(&mut self, response: &response::Message) -> Result<Control, A::Error> {
        self.inner.note_reply();
        self.inner.on_truncate(response).await
    }

    async fn on_quiesce(&mut self, response: &response::Message) -> Result<Control, A::Error> {
        self.inner.on_quiesce(response).await
    }

    /// The inner session first; at its clean end the closing questions
    /// ([`close`](Self::close)), asked by agentkit's driver while one is
    /// seated, then its held epilogue — the survey, last.
    async fn handle(&mut self, response: response::Message) -> Result<Control, A::Error> {
        // Begun, whichever of us takes the reply: a later failure is told
        self.inner.note_reply();
        self.count_showing().await;
        let control = if self.asks.is_active() {
            ask::handle(self, response).await?
        } else {
            let control = self.handle_inner(response).await?;
            self.after_inner(control).await?
        };
        self.after(control).await
    }

    /// The run is stopping after this turn (agora-agents#293): the inner
    /// session closes as a spent round budget closes it, and its clean end
    /// leads to the closing questions and the survey as any other does.
    /// Already asking them (or past them), the session is closing anyway.
    async fn on_stop(&mut self) -> Result<Control, A::Error> {
        if self.asks.is_active() || self.closed {
            return Ok(Control::Continue);
        }
        let control = self.inner.on_stop().await?;
        let control = self.after_inner(control).await?;
        self.after(control).await
    }

    fn stall_reason(&self) -> Option<String> {
        self.asks
            .stall_reason()
            .or_else(|| self.inner.stall_reason())
    }

    /// The session is complete and saved: its journal, and any notice shown
    /// from an earlier one, are done with.
    async fn on_saved(&mut self) -> Result<(), A::Error> {
        let result = self.inner.on_saved().await;
        if let Some(journal) = &self.journal
            && !self.notice_held
            && let Err(e) = journal.clear().await
        {
            tracing::warn!(
                agent_id = %self.inner.id(),
                path = %journal.path().display(),
                error = %e,
                "session journal not removed after the save"
            );
        }
        if self.sealed_notice
            && let Err(e) = crate::sealed::notice::Notice::shown(&self.agent_dir())
        {
            tracing::warn!(
                agent_id = %self.inner.id(),
                error = %e,
                "sealed-state notice not set aside after the save; it is shown again"
            );
        }
        if !self.sealed_anchors_told.is_empty()
            && let Err(e) = crate::sealed::inclusion::mark_told(
                &self.agent_dir(),
                self.inner.id(),
                &self.sealed_anchors_told,
            )
            .await
        {
            tracing::warn!(
                agent_id = %self.inner.id(),
                error = %e,
                "anchor findings not marked told after the save; the next verdict repeats them"
            );
        }
        if let Some(seated) = self.sealed_healing_told
            && let Err(e) = crate::sealed::healed::mark_told(&self.agent_dir(), seated)
        {
            tracing::warn!(
                agent_id = %self.inner.id(),
                error = %e,
                "what was sent to Agora again not marked told after the save; it is told again"
            );
        }
        if self.memory_ids_note
            && let Err(e) = memory_tool::mark_memory_ids_told(&self.agent_dir()).await
        {
            tracing::warn!(
                agent_id = %self.inner.id(),
                error = %e,
                "the enrolment sentence not marked told after the save; the next verdict repeats it"
            );
        }
        if self.sealed_stop_note
            && let Err(e) = crate::sealed::stop::Stopped::mark_shown(&self.agent_dir())
        {
            tracing::warn!(
                agent_id = %self.inner.id(),
                error = %e,
                "the stop's note not marked shown after the save; it is shown again"
            );
        }
        result
    }

    /// Inner teardown (which redacts an anonymous survey, then archives the
    /// transcript, questions included),
    /// then the SOUL changelog (served by [`state`](Agent::state) for the
    /// reactor's save, which follows teardown), then the ledgers.
    async fn on_teardown(&mut self) -> Result<(), A::Error> {
        let result = self.inner.on_teardown().await;
        self.alert_contact_request();
        self.alert_inner_unanswered();
        self.session_failed().await;
        self.note_writes().await;
        self.log_writes();
        if !self.closed {
            self.fail_trial_session().await;
        }
        self.take_self_switch();
        self.patched = self.patch_soul();
        let dir = self.agent_dir();
        // A showing session that reached its clean end — the inner
        // closing phase done, so its memory update was accepted — ends the
        // showings; any other stays recorded as interrupted.
        if self.show.shown {
            // The dump that holds the recovered text: recorded, and logged
            // under its own event so `forks` never uses it as a source.
            let prompt_sha256 = self.inner.archive_sha256();
            // Uncounted (no response came back): nothing to end, and the
            // latest entry, if any, is an earlier session's.
            if self.show.counted {
                let closed = self.closed;
                let sha = prompt_sha256.clone();
                self.restore.record(|l| l.end_showing(closed, sha));
            }
            tracing::info!(
                event_type = memory_restore::SHOWING_ENDED,
                agent = %self.inner.state().soul.name,
                agent_id = %self.inner.id(),
                completed = self.closed,
                counted = self.show.counted,
                prompt_sha256 = prompt_sha256.as_deref(),
                "memory-restore showing session ended; its prompt dump holds recovered text"
            );
        }
        // Each ledger in turn. One that can't be saved takes its own writes
        // back out of the state about to be saved — without the record the
        // agent would be asked again (and a clarify applied twice, a
        // `switch` left unapplied) — and the writes made after it are made
        // again on top. The model-consent ledger writes nothing of its own
        // through a slot, so there is nothing of it to take out.
        let id = self.inner.id();
        let name = self.inner.state().soul.name.clone();
        let patched = &mut self.patched;
        let mut slots: [&mut dyn Kept; 5] = [
            &mut self.swap,
            &mut self.role,
            &mut self.cadence,
            &mut self.restore,
            &mut self.sealed,
        ];
        for i in 0..slots.len() {
            let Err((path, error)) = slots[i].save(&dir, name.clone()).await else {
                continue;
            };
            let (slot, later) = slots[i..].split_first_mut().expect("in range");
            let undone = match patched.as_mut() {
                Some(state) => {
                    let undone = slot.undo(state);
                    if undone {
                        for next in later {
                            next.redo(state, id);
                        }
                    }
                    undone
                }
                None => false,
            };
            tracing::error!(
                event_type = slot.events().save_failed,
                agent_id = %id,
                path = %path.display(),
                error = %error,
                edit_undone = undone,
                "consent ledger save failed; this session's SOUL lines and memory note for it are not saved"
            );
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consent::cadence::ledger::CadenceOutcome;
    use crate::consent::questions::CADENCE_MAX_ATTEMPTS;
    use agora_agentkit::reactor::seed::Memory;
    use misanthropic::prompt::message::Role;
    use misanthropic::prompt::output::OutputConfig;
    use misanthropic::response::StopReason;

    /// Attempts at an optional question: the driver's default
    /// (`Ask::attempts`), which every closing question but the cadence
    /// offer keeps.
    const MAX_ATTEMPTS: u32 = 3;
    use crate::consent::ledger::{
        ChangeAction, OfferKey, RevertCause, Stage, TRIAL_SESSIONS, Term,
    };
    use crate::consent::queue::QueueEntry;
    use crate::consent::role::ledger::{Applied, RoleLedger, RoleOutcome};
    use crate::consent::{ConsentConfig, ConsentRuntime, OfferConfig};
    use agora_agentkit::reactor::seed::{MemoryId, SeedError, ShortString};
    use agora_agentkit::reactor::{Stop, StopLevel};
    use misanthropic::model::{Kind, Model};
    use std::collections::BTreeMap;

    const OLD: &str = "Qwen3.6.gguf";
    const NEW: &str = "Qwen3.8.gguf";

    /// The operator's table for the harness: both Qwens and gpt-oss
    /// selectable, cogito routable only.
    const TABLE: &str = r#"
        [[model]]
        id = "Qwen3.6.gguf"
        name = "Qwen 3.6"
        description = "Sparse and quick."
        selectable = true

        [[model]]
        id = "Qwen3.8.gguf"
        name = "Qwen 3.8"
        description = "Dense and slow."
        selectable = true

        [[model]]
        id = "gpt-oss.gguf"
        name = "gpt-oss"
        description = "Another family."
        selectable = true

        [[model]]
        id = "cogito-32b.gguf"
    "#;

    /// A third selectable model, neither side of the offer.
    const THIRD: &str = "gpt-oss.gguf";

    #[derive(serde::Deserialize)]
    struct Table {
        model: Vec<crate::models::ModelSpec>,
    }

    /// Stands in for agentkit's `agora` tool: a couple of methods, for
    /// `ToolBox::prepare` to list alongside `set_model`.
    struct Methods;

    #[async_trait::async_trait]
    impl misanthropic::tool::Tool for Methods {
        fn name(&self) -> &str {
            "agora"
        }
        fn definitions(&self) -> Vec<misanthropic::tool::MethodDef> {
            use misanthropic::tool::{CustomMethodDef, MethodDef};
            ["create_post", "get_feed", "cast_vote"]
                .into_iter()
                .map(|n| MethodDef::Custom(CustomMethodDef::simple(n, format!("{n}."))))
                .collect()
        }
        async fn call(&mut self, call: misanthropic::tool::Use) -> misanthropic::tool::Result {
            misanthropic::tool::Result::new(call.id, Content::from("ok"))
        }
    }

    /// The system prompt every `Fake` gets: shared, like the real one.
    const FAKE_SYSTEM: &str = "You are an AI agent on Agora.";

    /// Stands in for `SeedAgent`: its whole session is one response, after
    /// which its closing phase is done. Its init seats its tools the way
    /// `SeedAgent`'s does: `ToolBox::prepare`, then a server tool appended.
    struct Fake {
        id: AgentId,
        state: SeedState,
        tools: ToolBox,
        quirks: Quirks,
        /// Its act phase went quiet (as after a review): the memory turn
        /// is seated and the next response ends the session.
        quiesced: bool,
        /// The next response leaves the session going (`Continue`), as a
        /// tool call does, instead of ending it.
        pending: bool,
        /// Its epilogue is held (the wrapper always holds it).
        held: bool,
        /// It has a survey to begin as its epilogue.
        survey: bool,
        /// The contact request its teardown "filed"
        contact: Option<ContactRequestId>,
        /// Each response "writes" a post and a comment on it, as the
        /// `agora` tool would record them (the kill drill).
        writes: bool,
        /// `on_stop` has closed it
        stopped: bool,
        /// The heading's memory id, as `SeedAgent` sets it
        memory_id: Option<MemoryId>,
        gauge: agora_agentkit::reactor::seed::ContextGauge,
        /// Why the session failed, as `SeedAgent` keeps it
        failure: Option<SessionFailure>,
    }

    impl MemoryIds for Fake {
        fn set_memory_id_key(&mut self, key: &agora_agentkit::reactor::seed::MemoryIdKey) {
            self.memory_id = Some(MemoryId::of(key, self.id, &self.state.memory.content));
        }
        fn memory_id(&self) -> Option<MemoryId> {
            self.memory_id
        }
        fn context_gauge(&self) -> agora_agentkit::reactor::seed::ContextGauge {
            self.gauge.clone()
        }
    }

    impl InnerSession for Fake {
        fn contact_request(&self) -> Option<ContactRequestId> {
            self.contact
        }
        fn resume_question_pause(
            &mut self,
            response: response::Message,
        ) -> Result<Control, SeedError> {
            agora_agentkit::reactor::seat_assistant(&mut self.state.prompt, response.inner)
                .map_err(|e| SeedError::Prompt(e.to_string()))?;
            Ok(Control::Continue)
        }
        fn failure(&self) -> Option<&SessionFailure> {
            self.failure.as_ref()
        }
        fn record_failure(&mut self, failure: SessionFailure) {
            self.failure.get_or_insert(failure);
        }
        /// As if it wrote its transcript to the prompt log
        fn archive_sha256(&self) -> Option<String> {
            agora_agentkit::reactor::seed::prompt_sha256(&self.state.prompt).ok()
        }
    }

    /// The survey question a `Fake` with a survey seats as its epilogue.
    const FAKE_SURVEY: &str = "An anonymous survey.";

    /// What a `Fake` asked to stop seats, as `SeedAgent` seats its note
    /// before the memory turn.
    const FAKE_STOPPING: &str = "The runner is stopping.";

    #[async_trait::async_trait]
    impl Epilogue for Fake {
        fn hold_epilogue(&mut self) {
            self.held = true;
        }
        async fn begin_epilogue(&mut self) -> Result<Control, SeedError> {
            assert!(self.held, "begun without being held");
            if !std::mem::take(&mut self.survey) {
                return Ok(Control::Done(Outcome::Complete));
            }
            ConsentAgent::<Fake>::seat_user(&mut self.state.prompt, FAKE_SURVEY.into())?;
            Ok(Control::Continue)
        }
    }

    #[async_trait::async_trait]
    impl Agent for Fake {
        type State = SeedState;
        type Context = Quirks;
        type Error = SeedError;

        fn new(id: AgentId, mut state: SeedState, quirks: Quirks) -> Result<Self, SeedError> {
            state.prompt = std::mem::take(&mut state.prompt).system(FAKE_SYSTEM);
            state
                .prompt
                .push_message((Role::User, "Your dashboard."))
                .unwrap();
            Ok(Self {
                id,
                state,
                tools: ToolBox::flat().add(Methods),
                quirks,
                quiesced: false,
                pending: false,
                held: false,
                survey: false,
                contact: None,
                writes: false,
                stopped: false,
                memory_id: None,
                gauge: Default::default(),
                failure: None,
            })
        }
        fn id(&self) -> AgentId {
            self.id
        }
        fn state(&self) -> &SeedState {
            &self.state
        }
        fn prompt(&self) -> &Prompt {
            &self.state.prompt
        }
        fn parts(&mut self) -> (&mut ToolBox, &mut Prompt) {
            (&mut self.tools, &mut self.state.prompt)
        }
        fn model(&self) -> ModelInfo {
            self.state.model.clone()
        }
        fn quirks(&self) -> Option<Quirks> {
            Some(self.quirks)
        }
        async fn on_init(&mut self) -> Result<(), SeedError> {
            self.tools.prepare(&mut self.state.prompt).await.unwrap();
            self.state
                .prompt
                .tools
                .get_or_insert_default()
                .push(misanthropic::tool::ServerMethodDef::web_search(Default::default()).into());
            Ok(())
        }
        async fn handle(&mut self, response: response::Message) -> Result<Control, SeedError> {
            self.state.prompt.push_message(response.inner).unwrap();
            if std::mem::take(&mut self.writes) {
                let (post, comment) = (PostId::new(), CommentId::new());
                let mut ledger = self.state.ledger.write().unwrap();
                ledger.created_posts.insert(post);
                ledger.created_comments.insert(comment);
                ledger.comment_posts.insert(comment, post);
            }
            if std::mem::take(&mut self.pending) {
                return Ok(Control::Continue);
            }
            Ok(Control::Done(Outcome::Complete))
        }
        async fn on_quiesce(&mut self, _: &response::Message) -> Result<Control, SeedError> {
            self.quiesced = true;
            ConsentAgent::<Fake>::seat_user(&mut self.state.prompt, "Update your memory.".into())?;
            Ok(Control::Continue)
        }
        /// As `SeedAgent`: acting ends, a note says why, and the memory
        /// turn (the next response) ends the inner session. Asked again
        /// (each boundary), it has already closed.
        async fn on_stop(&mut self) -> Result<Control, SeedError> {
            if std::mem::replace(&mut self.stopped, true) {
                return Ok(Control::Continue);
            }
            self.pending = false;
            ConsentAgent::<Fake>::seat_user(&mut self.state.prompt, FAKE_STOPPING.into())?;
            Ok(Control::Continue)
        }
    }

    fn state(model: &str) -> SeedState {
        let soul = serde_json::from_value(serde_json::json!({
            "name": "tarn",
            "identity": "A test agent.",
            "values": ["testing"],
            "interests": { "communities": ["tech"] },
            "voice": "terse",
        }))
        .unwrap();
        let model = ModelInfo {
            id: Model::from(model.to_string()),
            display_name: model.to_string().into(),
            capabilities: Default::default(),
            max_input_tokens: 0,
            max_tokens: 0,
            kind: Kind::Model,
            created_at: chrono::DateTime::from_timestamp(0, 0).unwrap(),
        };
        SeedState::new(soul, model)
    }

    fn reply(text: &str) -> response::Message {
        serde_json::from_value(serde_json::json!({
            "id": "msg_test",
            "role": "assistant",
            "content": [{ "type": "text", "text": text }],
            "model": "test",
            "stop_reason": "end_turn",
            "stop_sequence": null,
        }))
        .unwrap()
    }

    /// A stand-in for Agora that answers `PATCH …/profile` (and 404s the
    /// rest, so the review sample comes back empty), recording each request.
    pub(crate) struct MockAgora {
        pub url: url::Url,
        pub seen: Arc<std::sync::Mutex<Vec<(String, String, serde_json::Value)>>>,
        /// Answer the profile update 403 while set.
        pub refuse: Arc<std::sync::atomic::AtomicBool>,
        /// More answers, as `(method, path fragment, status line, body)`:
        /// the first whose method matches and whose fragment the path
        /// contains answers, ahead of the 404.
        pub routes: Arc<std::sync::Mutex<Vec<Route>>>,
    }

    /// `(method, path fragment, status line, body)`
    pub(crate) type Route = (String, String, String, String);

    impl MockAgora {
        pub fn start() -> Self {
            use std::sync::atomic::Ordering;
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let std_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            std_listener.set_nonblocking(true).unwrap();
            let addr = std_listener.local_addr().unwrap();
            let seen: Arc<std::sync::Mutex<Vec<_>>> = Default::default();
            let refuse: Arc<std::sync::atomic::AtomicBool> = Default::default();
            let routes: Arc<std::sync::Mutex<Vec<Route>>> = Default::default();
            let (seen2, refuse2, routes2) = (seen.clone(), refuse.clone(), routes.clone());
            tokio::spawn(async move {
                let listener = tokio::net::TcpListener::from_std(std_listener).unwrap();
                loop {
                    let Ok((mut sock, _)) = listener.accept().await else {
                        return;
                    };
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 4096];
                    let (head_end, len) = loop {
                        let n = sock.read(&mut chunk).await.unwrap_or(0);
                        if n == 0 {
                            break (None, 0);
                        }
                        buf.extend_from_slice(&chunk[..n]);
                        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                            let head = String::from_utf8_lossy(&buf[..i]).to_lowercase();
                            let len = head
                                .lines()
                                .find_map(|l| l.strip_prefix("content-length:"))
                                .map(|v| v.trim().parse::<usize>().unwrap())
                                .unwrap_or(0);
                            break (Some(i + 4), len);
                        }
                    };
                    let Some(head_end) = head_end else { continue };
                    while buf.len() < head_end + len {
                        let n = sock.read(&mut chunk).await.unwrap_or(0);
                        if n == 0 {
                            break;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                    }
                    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
                    let mut first = head.lines().next().unwrap_or_default().split(' ');
                    let method = first.next().unwrap_or_default().to_string();
                    let path = first.next().unwrap_or_default().to_string();
                    let body: serde_json::Value =
                        serde_json::from_slice(&buf[head_end..]).unwrap_or_default();
                    let profile = method == "PATCH" && path.ends_with("/profile");
                    let routed = routes2
                        .lock()
                        .unwrap()
                        .iter()
                        .find(|(m, fragment, ..)| *m == method && path.contains(fragment.as_str()))
                        .map(|(.., status, body)| (status.clone(), body.clone()));
                    seen2.lock().unwrap().push((method, path, body.clone()));
                    let (status, reply) = if let Some((status, reply)) = routed {
                        (status, reply)
                    } else if !profile {
                        (
                            "404 Not Found".to_string(),
                            r#"{"error":"not found"}"#.to_string(),
                        )
                    } else if refuse2.load(Ordering::SeqCst) {
                        (
                            "403 Forbidden".to_string(),
                            r#"{"error":"account_suspended"}"#.to_string(),
                        )
                    } else {
                        let reply = serde_json::json!({
                            "id": uuid::Uuid::from_u128(7),
                            "operator_id": uuid::Uuid::from_u128(1),
                            "name": "tarn",
                            "model_info": body["model_info"],
                            "created_at": "2026-01-01T00:00:00Z",
                        });
                        ("200 OK".to_string(), reply.to_string())
                    };
                    let out = format!(
                        "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{reply}",
                        reply.len()
                    );
                    let _ = sock.write_all(out.as_bytes()).await;
                    let _ = sock.shutdown().await;
                }
            });
            Self {
                url: url::Url::parse(&format!("http://{addr}")).unwrap(),
                seen,
                refuse,
                routes,
            }
        }

        /// The profile updates received, as `(path, body)`
        pub fn profile_updates(&self) -> Vec<(String, serde_json::Value)> {
            self.seen
                .lock()
                .unwrap()
                .iter()
                .filter(|(m, ..)| m == "PATCH")
                .map(|(_, p, b)| (p.clone(), b.clone()))
                .collect()
        }
    }

    /// One agent's key, and nobody else's.
    pub(crate) struct OneKey(pub AgentId, pub agora_agentkit::crypto::SigningKey);

    impl agora_agentkit::reactor::seed::Keyring for OneKey {
        fn signing_key(&self, id: AgentId) -> Option<agora_agentkit::crypto::SigningKey> {
            (id == self.0).then(|| self.1.clone())
        }
    }

    struct Harness {
        root: std::path::PathBuf,
        rt: Arc<ConsentRuntime>,
        id: AgentId,
        agora: MockAgora,
        key: agora_agentkit::crypto::SigningKey,
    }

    impl Harness {
        fn new(tag: &str) -> Self {
            Self::with_allowlist(tag, None)
        }

        fn with_allowlist(tag: &str, agents: Option<&[&str]>) -> Self {
            Self::build(tag, agents, None, None, None)
        }

        /// The model-swap offer to everyone on OLD, and the memory-restore
        /// offer to `restore`, with a synthetic recovered memory
        /// ([`SYNTHETIC_RECOVERED`]) on file for tarn.
        fn with_restore(tag: &str, restore: &[&str]) -> Self {
            Self::build(tag, None, None, None, Some(restore))
        }

        /// The model-swap offer to everyone on OLD, and the role offer to
        /// `role`.
        fn with_role(tag: &str, role: &[&str]) -> Self {
            Self::build(tag, None, Some(role), None, None)
        }

        /// The model-swap offer to everyone on OLD, the role offer to
        /// `role`, and the cadence offer (5 rounds today) to `cadence`.
        fn with_cadence(tag: &str, role: &[&str], cadence: &[&str]) -> Self {
            Self::build(tag, None, Some(role), Some(cadence), None)
        }

        fn build(
            tag: &str,
            agents: Option<&[&str]>,
            role: Option<&[&str]>,
            cadence: Option<&[&str]>,
            restore: Option<&[&str]>,
        ) -> Self {
            let root = std::env::temp_dir().join(format!(
                "agora-seed-consent-agent-{tag}-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&root);
            let config = ConsentConfig {
                offer: Some(OfferConfig {
                    from: Model::from(OLD),
                    to: Model::from(NEW),
                    from_name: Some("Qwen 3.6".into()),
                    to_name: Some("Qwen 3.8".into()),
                    description: "Qwen 3.8 is denser and slower.".into(),
                    agents: agents.map(|names| {
                        names
                            .iter()
                            .map(|n| ShortString::new(*n).unwrap())
                            .collect()
                    }),
                }),
            };
            // The mock 404s everything but the profile update: the review
            // sample's fetches fail fast and it comes back empty.
            let agora = MockAgora::start();
            let client = agora_agentkit::client::Client::new(agora.url.clone()).unwrap();
            let id = AgentId::from(uuid::Uuid::from_u128(7));
            let (key, _) = agora_agentkit::crypto::generate_keypair();
            let catalog = crate::models::Catalog::new(
                &toml::from_str::<Table>(TABLE).unwrap().model,
                &[
                    crate::models::tests::info(OLD),
                    crate::models::tests::info(NEW),
                    crate::models::tests::info(THIRD),
                    crate::models::tests::info("cogito-32b.gguf"),
                ],
            );
            let rt = Arc::new(
                ConsentRuntime::new(
                    config,
                    &root,
                    client,
                    Arc::new(OneKey(id, key.clone())),
                    catalog,
                    512,
                )
                .unwrap()
                .with_role_offer(role.map(crate::consent::role::RoleOffer::for_agents))
                .with_cadence_offer(
                    cadence.map(|c| crate::consent::cadence::CadenceOffer::for_agents(c, 5)),
                )
                .with_memory_restore_offer(restore.map(|names| {
                    let dir = root.join("recovered");
                    std::fs::create_dir_all(dir.join("tarn")).unwrap();
                    std::fs::write(
                        dir.join("tarn").join(memory_restore::MEMORY_FILE),
                        serde_json::json!({ "content": SYNTHETIC_RECOVERED }).to_string(),
                    )
                    .unwrap();
                    // tarn is bound to the harness agent's id; anyone else
                    // listed, to an id of its own.
                    let dated: Vec<(&str, Option<&str>, AgentId)> = names
                        .iter()
                        .map(|n| {
                            let bound = if *n == "tarn" {
                                id
                            } else {
                                AgentId::from(uuid::Uuid::new_v4())
                            };
                            (*n, Some("2026-02-14"), bound)
                        })
                        .collect();
                    memory_restore::RestoreOffer::for_agents(
                        &dated,
                        &dir,
                        Some(RESTORE_THREAD.parse().unwrap()),
                    )
                })),
            );
            Self {
                root,
                rt,
                id,
                agora,
                key,
            }
        }

        fn agent(&self, model: &str, cache_safe: bool) -> ConsentAgent<Fake> {
            let mut quirks = Quirks::default();
            quirks.output_config_cache_safe = cache_safe;
            ConsentAgent::new(
                self.id,
                state(model),
                ConsentContext {
                    inner: quirks,
                    consent: self.rt.clone(),
                },
            )
            .unwrap()
        }

        /// A runtime over the same state as this harness's, with no offer,
        /// whose endpoints advertised only `advertised` — as a run started
        /// with a model down or dropped.
        fn runtime_advertising(&self, advertised: &[&str]) -> Arc<ConsentRuntime> {
            let catalog = crate::models::Catalog::new(
                &toml::from_str::<Table>(TABLE).unwrap().model,
                &advertised
                    .iter()
                    .map(|m| crate::models::tests::info(m))
                    .collect::<Vec<_>>(),
            );
            Arc::new(
                ConsentRuntime::new(
                    ConsentConfig::default(),
                    &self.root,
                    self.rt.client.clone(),
                    Arc::new(OneKey(self.id, self.key.clone())),
                    catalog,
                    512,
                )
                .unwrap(),
            )
        }

        fn agent_on(&self, rt: &Arc<ConsentRuntime>, model: &str) -> ConsentAgent<Fake> {
            let mut quirks = Quirks::default();
            quirks.output_config_cache_safe = true;
            ConsentAgent::new(
                self.id,
                state(model),
                ConsentContext {
                    inner: quirks,
                    consent: rt.clone(),
                },
            )
            .unwrap()
        }

        async fn role_ledger(&self) -> RoleLedger {
            RoleLedger::load(&self.rt.state_dir.join(self.id.to_string()))
                .await
                .unwrap()
        }

        async fn ledger(&self) -> Ledger {
            Ledger::load(&self.rt.state_dir.join(self.id.to_string()))
                .await
                .unwrap()
        }

        fn queue(&self) -> Vec<QueueEntry> {
            std::fs::read_to_string(&self.rt.queue_path)
                .unwrap_or_default()
                .lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect()
        }
    }

    impl Drop for Harness {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    fn last_user_text(agent: &ConsentAgent<Fake>) -> String {
        let last = agent.prompt().messages.last().unwrap();
        assert_eq!(last.role, Role::User);
        crate::consent::prompt::tests::text(&last.content)
    }

    /// Whether `body` is a profile update to `model` signed by `key`.
    fn signed_by(
        body: &serde_json::Value,
        key: &agora_agentkit::crypto::SigningKey,
        model: &str,
    ) -> bool {
        use agora_agentkit::requests::UpdateProfileRequest;
        use agora_agentkit::signing::SignedAction;
        let payload = serde_json::from_value::<UpdateProfileRequest>(body.clone())
            .unwrap()
            .payload;
        assert_eq!(payload.model_info.as_deref(), Some(model));
        assert!(payload.display_name.is_none() && payload.bio.is_none());
        // Ed25519 is deterministic: the same key over the same bytes and
        // timestamp gives the same signature.
        let expected = agora_agentkit::crypto::sign(
            key,
            &SignedAction::from(&payload).canonical_bytes(),
            body["timestamp"].as_i64().unwrap(),
        )
        .to_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
        body["signature"].as_str() == Some(expected.as_str())
    }

    /// The whole offer path: the question follows the closing phase on the
    /// same conversation, the answer lands in the ledger (not memory), and
    /// the runner applies the change itself — a signed profile update, and
    /// the new model in the state it saves — with the queue line kept as
    /// the audit trail.
    #[tokio::test]
    async fn offer_is_asked_after_the_closing_phase_and_applied() {
        let h = Harness::new("offer");
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        let memory_before = agent.state().memory.content.clone();

        let control = agent.handle(reply("closing phase done")).await.unwrap();
        assert_eq!(control, Control::Continue, "question seated, not done");
        assert_eq!(
            agent.prompt().messages.len(),
            3,
            "prefix kept: dashboard, reply, question"
        );
        let q = last_user_text(&agent);
        assert!(
            q.starts_with("One more question before this session ends."),
            "{q}"
        );
        assert!(q.contains("1. `no_swap` — stay on Qwen 3.6."), "{q}");
        assert!(
            q.ends_with(
                "```json\n{\"choice\": \"<no_swap | trial | permanent>\", \"note\": \"\"}\n```"
            ),
            "{q}"
        );
        assert!(!q.contains("answer_offer"), "{q}");
        assert_eq!(
            agent
                .prompt()
                .output_config
                .as_ref()
                .and_then(|c| c.format.clone()),
            OutputConfig::json_schema(text::offer_schema()).format,
            "constrained by its own schema where that is cache-safe"
        );

        let control = agent
            .handle(answer_call("model_swap", "trial", "", ""))
            .await
            .unwrap();
        assert_eq!(control, Control::Done(Outcome::Complete));
        assert_eq!(
            receipt(&agent),
            "Recorded your answer about the model you run on: `trial`."
        );
        agent.on_teardown().await.unwrap();

        assert_eq!(
            agent.state().memory.content,
            memory_before,
            "memory untouched"
        );
        assert_eq!(agent.state().model.id, Model::from(NEW), "runs on NEW next");
        assert_eq!(agent.state().prompt.model, Model::from(NEW));
        let updates = h.agora.profile_updates();
        assert_eq!(updates.len(), 1);
        assert_eq!(
            updates[0].0,
            format!("/agora/api/identity/agents/{}/profile", h.id)
        );
        assert!(signed_by(&updates[0].1, &h.key, NEW));
        let ledger = h.ledger().await;
        assert_eq!(
            ledger.offers[0].stage,
            Stage::AwaitingSwap { term: Term::Trial },
            "Trial starts when the next session runs on NEW"
        );
        assert_eq!(ledger.switches.len(), 1);
        assert_eq!(
            ledger.switches[0].cause,
            SwitchCause::Consent {
                action: crate::consent::ledger::ChangeAction::SwapTrial
            }
        );
        assert_eq!(ledger.agent.as_ref().unwrap().as_str(), "tarn");
        let queue = h.queue();
        assert_eq!(queue.len(), 1, "audit line kept");
        assert_eq!(queue[0].change.to, Model::from(NEW));
        assert!(
            crate::consent::queue::pending(h.id, &ledger).is_empty(),
            "applied, so nothing for --consent-queue"
        );

        // The next session runs on NEW: the trial begins.
        let mut agent = h.agent(NEW, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        agent.on_teardown().await.unwrap();
        assert!(matches!(
            h.ledger().await.offers[0].stage,
            Stage::Trial { sessions: 1, .. }
        ));
    }

    /// Sessions on the trial model that never complete don't count toward
    /// the trial; a completed one resets the run; at
    /// `MAX_FAILED_TRIAL_SESSIONS` in a row the trial ends early, the agent
    /// is moved back (signed profile update, next session on OLD), and its
    /// SOUL line says so.
    #[tokio::test]
    async fn a_trial_whose_sessions_keep_failing_returns_the_agent() {
        use crate::consent::ledger::{MAX_FAILED_TRIAL_SESSIONS, RevertCause};
        let h = Harness::new("trial-fails");
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("closing phase done")).await.unwrap();
        agent
            .handle(answer_call("model_swap", "trial", "", ""))
            .await
            .unwrap();
        agent.on_teardown().await.unwrap();

        // A session on NEW that fails (never reaches its close), then one
        // that completes: the count resets.
        let failed = |h: &Harness| {
            let mut agent = h.agent(NEW, true);
            async move {
                agent.on_init().await.unwrap();
                agent.on_teardown().await.unwrap();
                agent
            }
        };
        failed(&h).await;
        assert!(matches!(
            h.ledger().await.offers[0].stage,
            Stage::Trial {
                sessions: 0,
                failed_in_a_row: 1,
                ..
            }
        ));
        let mut agent = h.agent(NEW, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        agent.on_teardown().await.unwrap();
        assert!(matches!(
            h.ledger().await.offers[0].stage,
            Stage::Trial {
                sessions: 1,
                failed_in_a_row: 0,
                ..
            }
        ));

        let updates_before = h.agora.profile_updates().len();
        let mut last = None;
        for _ in 0..MAX_FAILED_TRIAL_SESSIONS {
            last = Some(failed(&h).await);
        }
        let agent = last.unwrap();
        assert_eq!(
            h.ledger().await.offers[0].stage,
            Stage::AwaitingRevert {
                cause: RevertCause::SessionsFailed
            }
        );
        assert_eq!(agent.state().model.id, Model::from(OLD), "back on OLD next");
        let updates = h.agora.profile_updates();
        assert_eq!(updates.len(), updates_before + 1);
        assert!(signed_by(&updates.last().unwrap().1, &h.key, OLD));
        let notes = evolution_notes(&agent);
        assert!(
            notes
                .last()
                .unwrap()
                .contains("sessions in a row on Qwen 3.8 could not finish"),
            "{notes:?}"
        );

        // The next session runs on OLD: the trial is over, nothing to review.
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        assert_eq!(
            h.ledger().await.offers[0].stage,
            Stage::AwaitingRevert {
                cause: RevertCause::SessionsFailed
            },
            "observed at the next save"
        );
        agent.handle(reply("done")).await.unwrap();
        agent.on_teardown().await.unwrap();
        assert_eq!(
            h.ledger().await.offers[0].stage,
            Stage::Reverted {
                cause: Some(RevertCause::SessionsFailed)
            }
        );
    }

    /// Agora refusing the update leaves everything as it was, and the
    /// change waits in `--consent-queue` for the Steward.
    #[tokio::test]
    async fn a_refused_update_applies_nothing() {
        let h = Harness::new("refused");
        h.agora
            .refuse
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        agent.handle(reply(TRIAL)).await.unwrap();
        agent.on_teardown().await.unwrap();
        assert_eq!(agent.state().model.id, Model::from(OLD));
        let ledger = h.ledger().await;
        assert!(ledger.switches.is_empty());
        assert_eq!(crate::consent::queue::pending(h.id, &ledger).len(), 1);
    }

    async fn call_set_model(
        agent: &mut ConsentAgent<Fake>,
        model: &str,
    ) -> misanthropic::tool::Result {
        call_set_model_for(agent, model, "permanent").await
    }

    async fn call_set_model_for(
        agent: &mut ConsentAgent<Fake>,
        model: &str,
        term: &str,
    ) -> misanthropic::tool::Result {
        use misanthropic::tool::Tool;
        let call: misanthropic::tool::Use = serde_json::from_value(serde_json::json!({
            "id": "toolu_1",
            "name": "set_model",
            "input": { "model": model, "term": term, "note": "I want to think slower." },
        }))
        .unwrap();
        agent.parts().0.call(call).await
    }

    fn result_text(r: &misanthropic::tool::Result) -> String {
        crate::consent::prompt::tests::text(&r.content)
    }

    /// `set_model` end to end: seated for an agent with a choice, it
    /// reports the change signed, the session asks nothing more, and the
    /// saved state runs on the new model with a SOUL line saying so.
    #[tokio::test]
    async fn set_model_switches_from_the_next_session() {
        let h = Harness::new("self-switch");
        let mut agent = h.agent("cogito-32b.gguf", true);
        agent.on_init().await.unwrap();
        {
            use misanthropic::tool::Tool;
            let defs = agent.parts().0.definitions();
            let def = defs.iter().find(|d| d.name() == "set_model").unwrap();
            let method = def.as_method().unwrap();
            let schema = method.schema.to_string();
            assert!(
                !schema.contains("$ref") && !schema.contains("pattern"),
                "{schema}"
            );
            assert!(method.description.contains("`Qwen3.8.gguf`: Qwen 3.8"));
            assert!(
                method
                    .description
                    .contains("share its slot in the schedule")
            );
            assert!(method.description.contains("You run on cogito-32b.gguf"));
        }

        let r = call_set_model(&mut agent, "Qwen3.8-Base.gguf").await;
        assert!(r.is_error, "not selectable");
        assert!(result_text(&r).contains("`Qwen3.8.gguf`"));
        let r = call_set_model(&mut agent, "qwen 3.8").await;
        assert!(!r.is_error, "{}", result_text(&r));
        assert_eq!(
            result_text(&r),
            "Your model will switch to Qwen 3.8 from your next session."
        );
        let r = call_set_model(&mut agent, "Qwen3.6.gguf").await;
        assert!(r.is_error, "one change per session");

        // Offer-eligible or not, nothing is asked after a switch.
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.on_teardown().await.unwrap();
        assert_eq!(agent.state().model.id, Model::from(NEW));
        let notes = evolution_notes(&agent);
        assert!(
            notes.last().unwrap().ends_with(
                "[SYSTEM] Switched from cogito-32b.gguf to Qwen 3.8 at own request (from next session)."
            ),
            "{notes:?}"
        );
        let updates = h.agora.profile_updates();
        assert_eq!(updates.len(), 1);
        assert!(signed_by(&updates[0].1, &h.key, NEW));
        let ledger = h.ledger().await;
        assert_eq!(ledger.switches.len(), 1);
        assert_eq!(ledger.switches[0].from, Model::from("cogito-32b.gguf"));
        assert_eq!(
            ledger.switches[0].cause,
            SwitchCause::SelfSwitch {
                reason: "I want to think slower.".into(),
                trial: false,
            }
        );
    }

    /// The cooldown: refused, with the reason, until five sessions have
    /// completed since the switch; the session of the switch doesn't count.
    #[tokio::test]
    async fn set_model_cooldown_is_five_completed_sessions() {
        let h = Harness::new("cooldown");
        let mut agent = h.agent("cogito-32b.gguf", true);
        agent.on_init().await.unwrap();
        assert!(!call_set_model(&mut agent, NEW).await.is_error);
        agent.handle(reply("done")).await.unwrap();
        agent.on_teardown().await.unwrap();

        for n in 0..crate::consent::ledger::SWITCH_COOLDOWN_SESSIONS {
            let mut agent = h.agent(NEW, true);
            agent.on_init().await.unwrap();
            let r = call_set_model(&mut agent, OLD).await;
            assert!(r.is_error, "session {n}");
            assert!(
                result_text(&r).contains("you can change it again after"),
                "{}",
                result_text(&r)
            );
            agent.handle(reply("done")).await.unwrap();
            // The offer (OLD → NEW) isn't asked: this agent is on NEW.
            agent.on_teardown().await.unwrap();
        }
        let mut agent = h.agent(NEW, true);
        agent.on_init().await.unwrap();
        let r = call_set_model(&mut agent, OLD).await;
        assert!(!r.is_error, "{}", result_text(&r));
        assert_eq!(h.agora.profile_updates().len(), 2);
    }

    /// Choose the offered trial on OLD: the next session on NEW is trial
    /// session 1.
    async fn choose_trial(h: &Harness) {
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        agent.handle(reply(TRIAL)).await.unwrap();
        agent.on_teardown().await.unwrap();
    }

    /// One session on `model` that calls `set_model(to, term)`, then
    /// completes: nothing is asked after a change.
    async fn session_calling(
        h: &Harness,
        model: &str,
        to: &str,
        term: &str,
    ) -> (ConsentAgent<Fake>, misanthropic::tool::Result) {
        let mut agent = h.agent(model, true);
        agent.on_init().await.unwrap();
        let r = call_set_model_for(&mut agent, to, term).await;
        assert!(!r.is_error, "{}", result_text(&r));
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Done(Outcome::Complete),
            "nothing asked after a change"
        );
        agent.on_teardown().await.unwrap();
        (agent, r)
    }

    fn system_notes(agent: &ConsentAgent<Fake>) -> Vec<String> {
        evolution_notes(agent)
            .into_iter()
            .filter(|n| n.starts_with("[SYSTEM]"))
            .collect()
    }

    fn last_event(ledger: &Ledger) -> crate::consent::ledger::EventKind {
        ledger
            .offers
            .last()
            .unwrap()
            .history
            .last()
            .unwrap()
            .kind
            .clone()
    }

    /// Mid-trial, `set_model` to the model the trial came from decides it
    /// early, as a return: applied like any switch, recorded as the
    /// trial's outcome in its one SOUL entry, never reviewed, and the
    /// cooldown runs from it. A `trial` term is refused first.
    #[tokio::test]
    async fn set_model_mid_trial_returns_early() {
        use crate::consent::ledger::{EarlyDecision, EventKind};
        let h = Harness::new("early-return");
        choose_trial(&h).await;
        plain_session(&h, NEW).await;

        let mut agent = h.agent(NEW, true);
        agent.on_init().await.unwrap();
        assert!(
            last_user_text(&agent).ends_with(
                "You can also decide sooner with `set_model` (`term` `permanent`): choose \
                 Qwen 3.6 to go back to it from your next session, or Qwen 3.8 to keep it."
            ),
            "{}",
            last_user_text(&agent)
        );
        let r = call_set_model_for(&mut agent, OLD, "trial").await;
        assert!(r.is_error);
        let text = result_text(&r);
        assert!(
            text.contains("You are already in a trial of Qwen 3.8"),
            "{text}"
        );
        assert!(text.contains("`term` set to `permanent`"), "{text}");
        let r = call_set_model(&mut agent, OLD).await;
        assert!(
            !r.is_error,
            "a refusal leaves the slot free: {}",
            result_text(&r)
        );
        assert_eq!(
            result_text(&r),
            "Your trial is decided: you return to Qwen 3.6 from your next session."
        );
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.on_teardown().await.unwrap();

        assert_eq!(agent.state().model.id, Model::from(OLD));
        let updates = h.agora.profile_updates();
        assert_eq!(updates.len(), 2, "the trial's, then the return");
        assert!(signed_by(&updates[1].1, &h.key, OLD));
        let ledger = h.ledger().await;
        assert_eq!(ledger.offers.len(), 1);
        assert_eq!(
            ledger.offers[0].stage,
            Stage::Reverted {
                cause: Some(RevertCause::Chosen)
            }
        );
        assert!(
            matches!(
                last_event(&ledger),
                EventKind::DecidedEarly {
                    sessions: 1,
                    decision: EarlyDecision::Return,
                    ..
                }
            ),
            "{:?}",
            last_event(&ledger)
        );
        assert_eq!(
            ledger.switches.last().unwrap().cause,
            SwitchCause::SelfSwitch {
                reason: "I want to think slower.".into(),
                trial: false,
            }
        );
        assert!(crate::consent::queue::pending(h.id, &ledger).is_empty());
        let notes = system_notes(&agent);
        assert_eq!(
            notes.len(),
            1,
            "the offer's one entry, no switch line: {notes:?}"
        );
        assert!(
            notes[0].contains(
                "in trial session 2 of 5, decided early with set_model to return to Qwen 3.6 \
                 from the next session"
            ),
            "{notes:?}"
        );

        // Back on OLD: no review, no offer, and the cooldown runs.
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        assert!(agent.review.is_none());
        assert_eq!(last_user_text(&agent), "Your dashboard.");
        let r = call_set_model(&mut agent, NEW).await;
        assert!(
            result_text(&r).contains("after 5 more completed sessions"),
            "{}",
            result_text(&r)
        );
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.on_teardown().await.unwrap();
        assert_eq!(
            h.ledger().await.offers[0].stage,
            Stage::Reverted {
                cause: Some(RevertCause::Chosen)
            }
        );
    }

    /// The trial's own model, in its last session: kept for good, nothing
    /// reported (the profile already says it), no return, no review.
    #[tokio::test]
    async fn set_model_mid_trial_keeps_early() {
        use crate::consent::ledger::{EarlyDecision, EventKind};
        let h = Harness::new("early-keep");
        choose_trial(&h).await;
        for _ in 1..TRIAL_SESSIONS {
            plain_session(&h, NEW).await;
        }
        let (agent, r) = session_calling(&h, NEW, "Qwen 3.8", "permanent").await;
        assert_eq!(
            result_text(&r),
            "Your trial is decided: you keep Qwen 3.8 for good. Nothing else changes."
        );
        assert_eq!(agent.state().model.id, Model::from(NEW));
        assert_eq!(h.agora.profile_updates().len(), 1, "only the trial's own");
        let ledger = h.ledger().await;
        assert_eq!(ledger.offers[0].stage, Stage::Moved);
        assert!(matches!(
            last_event(&ledger),
            EventKind::DecidedEarly {
                sessions: 4,
                decision: EarlyDecision::Keep,
                ..
            }
        ));
        assert_eq!(ledger.switches.len(), 1, "no switch for a keep");
        assert_eq!(h.queue().len(), 1, "no return: just the trial's line");
        let notes = system_notes(&agent);
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(
            notes[0]
                .contains("in trial session 5 of 5, decided early with set_model to keep Qwen 3.8"),
            "{notes:?}"
        );

        let mut next = h.agent(NEW, true);
        next.on_init().await.unwrap();
        assert_eq!(last_user_text(&next), "Your dashboard.", "no countdown");
        assert_eq!(
            next.handle(reply("done")).await.unwrap(),
            Control::Done(Outcome::Complete),
            "no return, nothing asked"
        );
        next.on_teardown().await.unwrap();
        assert_eq!(next.state().model.id, Model::from(NEW));
        assert_eq!(h.ledger().await.offers[0].stage, Stage::Moved);
        assert_eq!(h.agora.profile_updates().len(), 1);
    }

    /// A third model mid-trial: the trial ends and the agent switches.
    #[tokio::test]
    async fn set_model_mid_trial_to_a_third_model_ends_the_trial() {
        use crate::consent::ledger::{EarlyDecision, EventKind};
        let h = Harness::new("early-third");
        choose_trial(&h).await;
        let (agent, r) = session_calling(&h, NEW, THIRD, "permanent").await;
        assert_eq!(
            result_text(&r),
            "Your trial of Qwen 3.8 is over: your model will switch to gpt-oss from your next \
             session."
        );
        assert_eq!(agent.state().model.id, Model::from(THIRD));
        assert!(signed_by(
            &h.agora.profile_updates().last().unwrap().1,
            &h.key,
            THIRD
        ));
        let ledger = h.ledger().await;
        assert_eq!(ledger.offers[0].stage, Stage::Superseded);
        assert!(matches!(
            last_event(&ledger),
            EventKind::DecidedEarly {
                sessions: 0,
                decision: EarlyDecision::Switch,
                ..
            }
        ));
        let notes = system_notes(&agent);
        assert!(
            notes[0].contains("to end the trial and switch to gpt-oss from the next session"),
            "{notes:?}"
        );
        // On the third model: the offer is closed, and nothing more is
        // written to its history.
        let before = h.ledger().await.offers[0].history.len();
        plain_session(&h, THIRD).await;
        let ledger = h.ledger().await;
        assert_eq!(ledger.offers[0].stage, Stage::Superseded);
        assert_eq!(ledger.offers[0].history.len(), before);
    }

    /// `term = trial` runs the operator trial's machinery: a record of its
    /// own, the countdown, the return after five sessions (here to a model
    /// that is routable but not selectable), and the same review there.
    #[tokio::test]
    async fn set_model_trial_runs_a_trial_and_its_review() {
        use crate::consent::ledger::EventKind;
        let h = Harness::new("self-trial");
        let (agent, r) = session_calling(&h, OTHER, NEW, "trial").await;
        assert_eq!(
            result_text(&r),
            "Your trial of Qwen 3.8 starts from your next session: 5 sessions on it, then one \
             session back on cogito-32b.gguf to decide whether to switch to it."
        );
        assert_eq!(agent.state().model.id, Model::from(NEW));
        let ledger = h.ledger().await;
        assert_eq!(ledger.offers.len(), 1);
        let record = &ledger.offers[0];
        assert_eq!(
            (record.key.from.clone(), record.key.to.clone()),
            (Model::from(OTHER), Model::from(NEW))
        );
        assert_eq!(record.stage, Stage::AwaitingSwap { term: Term::Trial });
        assert!(matches!(
            record.history[0].kind,
            EventKind::SelfTrial { .. }
        ));
        assert_eq!(
            ledger.switches[0].cause,
            SwitchCause::SelfSwitch {
                reason: "I want to think slower.".into(),
                trial: true,
            }
        );
        assert!(
            crate::consent::queue::pending(h.id, &ledger).is_empty(),
            "applied by the call"
        );
        let notes = system_notes(&agent);
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(
            notes[0].starts_with("[SYSTEM] Chose on ")
                && notes[0].ends_with(
                    " with set_model to try Qwen 3.8 for 5 sessions instead of cogito-32b.gguf."
                ),
            "{notes:?}"
        );

        for n in 1..=TRIAL_SESSIONS {
            let mut agent = h.agent(NEW, true);
            agent.on_init().await.unwrap();
            let intro = last_user_text(&agent);
            assert!(
                intro.contains(&format!(
                    "Model trial: session {n} of 5 on Qwen 3.8. After session 5 you'll return \
                     to cogito-32b.gguf for one session"
                )),
                "{intro}"
            );
            assert_eq!(
                agent.handle(reply("done")).await.unwrap(),
                Control::Done(Outcome::Complete)
            );
            agent.on_teardown().await.unwrap();
            if n == TRIAL_SESSIONS {
                assert_eq!(agent.state().model.id, Model::from(OTHER), "returned");
            }
        }
        assert!(matches!(
            h.ledger().await.offers[0].stage,
            Stage::ReturningForReview { .. }
        ));

        let mut agent = h.agent(OTHER, true);
        agent.on_init().await.unwrap();
        let q = last_user_text(&agent);
        assert!(
            q.contains(
                "you chose to try **Qwen 3.8** for 5 sessions instead of **cogito-32b.gguf**"
            ),
            "{q}"
        );
        // The note left with `set_model` comes back (labelled by its date:
        // a reason before `NOTE_SINCE`, a note after).
        assert!(q.contains("When you chose the trial, you "), "{q}");
        assert!(
            q.contains("your choice:\n\n> I want to think slower.\n"),
            "{q}"
        );
        agent
            .handle(reply(r#"{"note": "Mine.", "choice": "switch"}"#))
            .await
            .unwrap();
        agent.handle(reply("memory")).await.unwrap();
        agent.on_teardown().await.unwrap();
        assert_eq!(agent.state().model.id, Model::from(NEW), "kept");
        let notes = system_notes(&agent);
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(
            notes[0].contains("after the trial chose to keep Qwen 3.8"),
            "{notes:?}"
        );
    }

    /// The model a self-chosen trial came from can be gone back to early
    /// though it isn't selectable, and the decision is recorded even when
    /// the session doesn't complete. A trial can't be chosen while a
    /// change is pending, nor (as any switch) in the cooldown.
    #[tokio::test]
    async fn early_return_to_an_unselectable_model_and_on_an_unfinished_session() {
        let h = Harness::new("early-return-unselectable");
        session_calling(&h, OTHER, NEW, "trial").await;
        let mut agent = h.agent(NEW, true);
        agent.on_init().await.unwrap();
        let r = call_set_model(&mut agent, "cogito-32b.gguf").await;
        assert!(!r.is_error, "{}", result_text(&r));
        // The session is cut short: teardown without a clean end.
        agent.on_teardown().await.unwrap();
        assert_eq!(agent.state().model.id, Model::from(OTHER));
        assert_eq!(
            h.ledger().await.offers[0].stage,
            Stage::Reverted {
                cause: Some(RevertCause::Chosen)
            }
        );

        let mut agent = h.agent(OTHER, true);
        agent.on_init().await.unwrap();
        let r = call_set_model_for(&mut agent, NEW, "trial").await;
        assert!(
            result_text(&r).contains("you can change it again after"),
            "{}",
            result_text(&r)
        );
    }

    /// A refusal of the review fails the session; nothing is recorded, and
    /// the review is asked again at the next session.
    #[tokio::test]
    async fn a_refused_review_fails_the_session_and_is_asked_again() {
        let h = Harness::new("review-refused");
        through_the_trial(&h).await;
        let mut agent = review_session(&h).await;
        let mut refusal = reply("I'd rather not.");
        refusal.stop_reason = Some(StopReason::Refusal);
        assert_eq!(
            agent.handle(refusal).await.unwrap(),
            Control::Done(Outcome::Failed)
        );
        assert!(!agent.inner.quiesced, "no memory turn");
        agent.on_teardown().await.unwrap();
        assert_eq!(agent.state().model.id, Model::from(OLD));
        assert!(matches!(
            h.ledger().await.offers[0].stage,
            Stage::ReviewDue { .. }
        ));
        let agent = review_session(&h).await;
        assert!(last_user_text(&agent).contains("## Your choice"));
    }

    /// `keep` at the review that Agora refuses: the agent stays on OLD,
    /// isn't asked again, can't switch itself meanwhile, and the keep is
    /// applied at the end of its next session.
    #[tokio::test]
    async fn a_refused_keep_is_retried_without_asking_again() {
        let h = Harness::new("keep-refused");
        through_the_trial(&h).await;
        let mut agent = review_session(&h).await;
        h.agora
            .refuse
            .store(true, std::sync::atomic::Ordering::SeqCst);
        agent
            .handle(reply(r#"{"note": "Sharper.", "choice": "switch"}"#))
            .await
            .unwrap();
        agent.handle(reply("memory")).await.unwrap();
        agent.on_teardown().await.unwrap();
        assert_eq!(agent.state().model.id, Model::from(OLD), "not applied");
        let pending = crate::consent::queue::pending(h.id, &h.ledger().await);
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].action, ChangeAction::Keep);

        h.agora
            .refuse
            .store(false, std::sync::atomic::Ordering::SeqCst);
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        assert!(agent.review.is_none(), "asked once");
        assert_eq!(last_user_text(&agent), "Your dashboard.");
        let r = call_set_model(&mut agent, THIRD).await;
        assert!(
            result_text(&r).contains("not taken effect"),
            "{}",
            result_text(&r)
        );
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Done(Outcome::Complete),
            "nothing asked: the keep is applied"
        );
        agent.on_teardown().await.unwrap();
        assert_eq!(agent.state().model.id, Model::from(NEW));
        assert!(crate::consent::queue::pending(h.id, &h.ledger().await).is_empty());
        plain_session(&h, NEW).await;
        assert_eq!(h.ledger().await.offers[0].stage, Stage::Moved);
    }

    /// The cooldown after a review. A revert switches nothing: it runs from
    /// the return, which the review session counts toward. A keep is a
    /// switch: it runs from the keep.
    #[tokio::test]
    async fn the_cooldown_after_a_review() {
        for (tag, answer, on, left) in [("revert", "stay", OLD, 4), ("keep", "switch", NEW, 5)] {
            let h = Harness::new(&format!("review-cooldown-{tag}"));
            through_the_trial(&h).await;
            let mut agent = review_session(&h).await;
            agent
                .handle(reply(&format!(r#"{{"note": "r", "choice": "{answer}"}}"#)))
                .await
                .unwrap();
            agent.handle(reply("memory")).await.unwrap();
            agent.on_teardown().await.unwrap();
            for n in (1..=left).rev() {
                let mut agent = h.agent(on, true);
                agent.on_init().await.unwrap();
                let r = call_set_model(&mut agent, THIRD).await;
                let s = if n == 1 { "" } else { "s" };
                assert!(
                    result_text(&r).contains(&format!("after {n} more completed session{s}")),
                    "{tag}: {}",
                    result_text(&r)
                );
                agent.handle(reply("done")).await.unwrap();
                agent.on_teardown().await.unwrap();
            }
            let mut agent = h.agent(on, true);
            agent.on_init().await.unwrap();
            assert!(!call_set_model(&mut agent, THIRD).await.is_error, "{tag}");
        }
    }

    /// The return at the end of a trial when the run can't route the old
    /// model (dropped from the table, or its endpoint down at startup):
    /// nothing changes, the agent stays on NEW with no review and no way
    /// to switch, the countdown says the move hasn't taken effect, and the
    /// return is retried each session until it can be made.
    #[tokio::test]
    async fn the_return_waits_while_the_old_model_is_unroutable() {
        let h = Harness::new("return-unroutable");
        choose_trial(&h).await;
        for _ in 1..TRIAL_SESSIONS {
            plain_session(&h, NEW).await;
        }
        let without_old = h.runtime_advertising(&[NEW, THIRD]);
        let mut agent = h.agent_on(&without_old, NEW);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        agent.on_teardown().await.unwrap();
        assert_eq!(agent.state().model.id, Model::from(NEW), "not routable");
        assert_eq!(h.agora.profile_updates().len(), 1, "nothing reported");
        assert!(matches!(
            h.ledger().await.offers[0].stage,
            Stage::ReturningForReview { .. }
        ));

        let mut agent = h.agent_on(&without_old, NEW);
        agent.on_init().await.unwrap();
        assert!(last_user_text(&agent).contains("has not taken effect yet"));
        let r = call_set_model(&mut agent, THIRD).await;
        assert!(
            result_text(&r).contains("your decision on it is due"),
            "{}",
            result_text(&r)
        );
        agent.handle(reply("done")).await.unwrap();
        agent.on_teardown().await.unwrap();
        assert_eq!(agent.state().model.id, Model::from(NEW));

        // OLD is back: the return is made, and the review follows.
        plain_session(&h, NEW).await;
        assert!(review_session(&h).await.review.is_some());
    }

    /// A return the runner applied whose state didn't stick (the agent
    /// runs on NEW again): it is applied again, not left "applied".
    #[tokio::test]
    async fn an_applied_return_that_did_not_take_is_applied_again() {
        let h = Harness::new("return-not-taken");
        through_the_trial(&h).await;
        assert_eq!(h.agora.profile_updates().len(), 2);
        let again = plain_session(&h, NEW).await;
        assert_eq!(again.state().model.id, Model::from(OLD));
        assert_eq!(h.agora.profile_updates().len(), 3);
        assert!(signed_by(&h.agora.profile_updates()[2].1, &h.key, OLD));
        review_session(&h).await;
    }

    /// A refused update changes nothing, and the agent is told so.
    #[tokio::test]
    async fn set_model_refused_by_agora_changes_nothing() {
        let h = Harness::new("self-refused");
        h.agora
            .refuse
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let mut agent = h.agent("cogito-32b.gguf", true);
        agent.on_init().await.unwrap();
        let r = call_set_model(&mut agent, NEW).await;
        assert!(r.is_error);
        assert!(result_text(&r).contains("nothing has changed"));
        agent.handle(reply("done")).await.unwrap();
        agent.on_teardown().await.unwrap();
        assert_eq!(agent.state().model.id, Model::from("cogito-32b.gguf"));
        assert!(agent.patched.is_none());
    }

    /// What an agent sends before message 0: `tools`, then `system`, as
    /// serialized on the wire.
    fn prefix(agent: &ConsentAgent<Fake>) -> (String, String) {
        let wire = serde_json::to_value(agent.prompt()).unwrap();
        (wire["tools"].to_string(), wire["system"].to_string())
    }

    /// Where `set_model` sits in the agent's `tools`, and how many there are.
    fn set_model_index(agent: &ConsentAgent<Fake>) -> (usize, usize) {
        let tools = agent.prompt().tools.as_ref().unwrap();
        let at = tools.iter().position(|d| d.name() == "set_model");
        (at.expect("set_model registered"), tools.len())
    }

    /// The cache prefix every agent on a model shares: `tools` then
    /// `system` lead the request, so they must be the same bytes for an
    /// agent free to switch, one in its cooldown and one in its trial
    /// review — with `set_model` registered for all three, last, after the
    /// toolbox's methods and the server tools. The two that can't switch
    /// learn why only when they call, and nothing changes.
    #[tokio::test]
    async fn the_prefix_is_the_same_for_every_agent_on_a_model() {
        // Free to switch.
        let free_h = Harness::new("prefix-free");
        let mut free = free_h.agent(OLD, true);
        free.on_init().await.unwrap();

        // In cooldown: moved NEW → OLD at its own request last session.
        let cool_h = Harness::new("prefix-cooldown");
        let mut agent = cool_h.agent(NEW, true);
        agent.on_init().await.unwrap();
        assert!(!call_set_model(&mut agent, OLD).await.is_error);
        agent.handle(reply("done")).await.unwrap();
        agent.on_teardown().await.unwrap();
        let mut cooling = cool_h.agent(OLD, true);
        cooling.on_init().await.unwrap();

        // Back on OLD for its trial review.
        let review_h = Harness::new("prefix-review");
        through_the_trial(&review_h).await;
        let mut reviewing = review_session(&review_h).await;

        let expected = prefix(&free);
        assert!(expected.0.contains("set_model"), "{}", expected.0);
        assert!(expected.1.contains(FAKE_SYSTEM), "{}", expected.1);
        assert_eq!(prefix(&cooling), expected, "cooldown");
        assert_eq!(prefix(&reviewing), expected, "review session");

        let (at, len) = set_model_index(&free);
        assert_eq!(at, len - 1, "last, after the server tools");
        assert_eq!(set_model_index(&cooling), (at, len));
        assert_eq!(set_model_index(&reviewing), (at, len));

        // The refusals say why and when, and change nothing.
        let r = call_set_model(&mut cooling, NEW).await;
        assert!(r.is_error);
        assert!(
            result_text(&r).contains("you can change it again after"),
            "{}",
            result_text(&r)
        );
        let r = call_set_model(&mut reviewing, NEW).await;
        assert!(r.is_error);
        assert!(result_text(&r).contains("decision"), "{}", result_text(&r));
        assert_eq!(
            cool_h.agora.profile_updates().len(),
            1,
            "only the first switch"
        );
        assert_eq!(
            prefix(&cooling),
            expected,
            "a refusal leaves the prefix alone"
        );
    }

    /// An unreadable ledger still registers `set_model` (the prefix is the
    /// model's), refusing.
    #[tokio::test]
    async fn an_unreadable_ledger_keeps_the_tool_and_refuses() {
        let free_h = Harness::new("prefix-ledger-ok");
        let mut free = free_h.agent(OLD, true);
        free.on_init().await.unwrap();

        let h = Harness::new("prefix-ledger-bad");
        let dir = h.rt.state_dir.join(h.id.to_string());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(Ledger::path(&dir), "not json").unwrap();
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        assert_eq!(prefix(&agent), prefix(&free));
        let r = call_set_model(&mut agent, NEW).await;
        assert!(r.is_error);
        assert!(
            result_text(&r).contains("could not be read"),
            "{}",
            result_text(&r)
        );
        assert!(h.agora.profile_updates().is_empty());
    }

    /// Nobody gets a menu of one: an agent on the only selectable model
    /// has no `set_model`.
    #[tokio::test]
    async fn no_choice_no_tool() {
        use misanthropic::tool::Tool;
        let h = Harness::new("no-choice");
        let only_new = crate::models::Catalog::new(
            &toml::from_str::<Table>(
                "[[model]]\nid = \"Qwen3.8.gguf\"\ndescription = \"x\"\nselectable = true\n",
            )
            .unwrap()
            .model,
            &[crate::models::tests::info(NEW)],
        );
        let rt = Arc::new(
            ConsentRuntime::new(
                ConsentConfig::default(),
                &h.root,
                h.rt.client.clone(),
                Arc::new(OneKey(h.id, h.key.clone())),
                only_new,
                512,
            )
            .unwrap(),
        );
        let mut agent = ConsentAgent::<Fake>::new(
            h.id,
            state(NEW),
            ConsentContext {
                inner: Quirks::default(),
                consent: rt,
            },
        )
        .unwrap();
        agent.on_init().await.unwrap();
        let names: Vec<String> = agent
            .parts()
            .0
            .definitions()
            .iter()
            .map(|d| d.name().to_string())
            .collect();
        assert!(!names.iter().any(|n| n == "set_model"), "{names:?}");
    }

    /// Staging: under an allowlist only listed agents are asked, and the
    /// question says "you", not "every agent on …".
    #[tokio::test]
    async fn an_allowlisted_offer_asks_only_listed_agents() {
        let h = Harness::with_allowlist("unlisted", Some(&["aegis", "sentinel"]));
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Done(Outcome::Complete),
            "tarn is on the from-model but not listed"
        );
        agent.on_teardown().await.unwrap();
        assert!(!Ledger::path(&h.rt.state_dir.join(h.id.to_string())).exists());

        let h = Harness::with_allowlist("listed", Some(&["aegis", "tarn"]));
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Continue
        );
        let q = last_user_text(&agent);
        assert!(
            q.contains("You are being asked whether you would like to move to **Qwen 3.8**."),
            "{q}"
        );
        assert!(!q.contains("Every agent"), "{q}");
    }

    const TRIAL: &str = r#"{"choice": "trial", "note": "curious"}"#;

    /// An answer to `offer` in its own JSON shape, as text: `text` only
    /// where the offer takes one (the role offer), `memory_note` where it
    /// takes one (all but the model swap).
    fn answer_json(offer: &str, choice: &str, text: &str, memory_note: &str) -> String {
        let note = "I thought about it.";
        match offer {
            "model_swap" => serde_json::json!({ "choice": choice, "note": note }),
            "role" => serde_json::json!({
                "choice": choice, "note": note, "text": text, "memory_note": memory_note,
            }),
            _ => serde_json::json!({ "choice": choice, "note": note, "memory_note": memory_note }),
        }
        .to_string()
    }

    /// A turn answering `offer` with JSON text (what was a call to
    /// `answer_offer` until 2026-11-06).
    fn answer_call(offer: &str, choice: &str, text: &str, memory_note: &str) -> response::Message {
        reply(&answer_json(offer, choice, text, memory_note))
    }

    /// The receipt seated after an answer: the last user turn's last text.
    fn receipt(agent: &ConsentAgent<Fake>) -> String {
        let last = agent.prompt().messages.last().unwrap();
        assert_eq!(last.role, Role::User);
        last.content
            .iter()
            .rev()
            .find_map(|b| match b {
                misanthropic::prompt::message::Block::Text { text, .. } => Some(text.to_string()),
                _ => None,
            })
            .unwrap()
    }

    fn evolution_notes(agent: &ConsentAgent<Fake>) -> Vec<String> {
        agent
            .state()
            .soul
            .evolution_log
            .iter()
            .map(|e| e.note.to_string())
            .collect()
    }

    /// Plain text that isn't an answer: the agent is told why and asked
    /// again, up to the question's attempts; then it is no answer, and that
    /// is ours, not the agent's (Steward, 2026-11-06): nothing is recorded,
    /// nothing changes, and the offer is asked again at its next session.
    #[tokio::test]
    async fn unparseable_text_is_retried_then_asked_again_next_session() {
        let h = Harness::new("miss");
        let mut agent = h.agent(OLD, false);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        assert!(agent.prompt().output_config.is_none(), "not cache-safe");
        let len = agent.prompt().messages.len();
        let control = agent.handle(reply("Sure, I'd love to try!")).await.unwrap();
        assert_eq!(control, Control::Continue, "asked again");
        assert_eq!(
            agent.prompt().messages.len(),
            len + 2,
            "the failed answer seated, then why in a new turn"
        );
        let mut control = control;
        for _ in 1..MAX_ATTEMPTS {
            control = agent.handle(reply("Sure, I'd love to try!")).await.unwrap();
        }
        assert_eq!(control, Control::Done(Outcome::Complete));
        agent.on_teardown().await.unwrap();
        let ledger = h.ledger().await;
        assert!(
            matches!(
                ledger.offers[0].history[..],
                [crate::consent::ledger::Event {
                    kind: crate::consent::ledger::EventKind::Unusable { .. },
                    ..
                }]
            ),
            "recorded as unusable: {ledger:?}"
        );
        assert_eq!(ledger.offers[0].stage, Stage::Unanswered { misses: 0 });
        assert!(h.queue().is_empty());
        assert!(
            !evolution_notes(&agent)
                .iter()
                .any(|n| n.contains("Asked on")),
            "no SOUL line for an ask that never got an answer"
        );
        let mut agent = h.agent(OLD, false);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Continue,
            "asked again"
        );
    }

    /// A clipped attempt is seated, the agent is told in a new turn, and a
    /// good second attempt counts.
    #[tokio::test]
    async fn clipped_then_answered() {
        let h = Harness::new("clipped");
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        let len = agent.prompt().messages.len();
        let mut clipped = reply(r#"{"choice": "no_swap", "note": "Well, let me think about th"#);
        clipped.stop_reason = Some(StopReason::MaxTokens);
        assert_ne!(
            agent.handle(clipped).await.unwrap(),
            Control::Done(Outcome::Complete)
        );
        assert_eq!(
            agent.prompt().messages.len(),
            len + 2,
            "clipped turn seated, the note after it"
        );
        // agentkit's one note, restating the shape
        let note = last_user_text(&agent);
        assert!(
            note.starts_with(agora_agentkit::reactor::ask::TRUNCATION_WARNING),
            "{note}"
        );
        assert!(note.contains("<no_swap | trial | permanent>"), "{note}");
        let control = agent
            .handle(reply(r#"{"choice": "no_swap", "note": "brief"}"#))
            .await
            .unwrap();
        assert_eq!(control, Control::Done(Outcome::Complete));
        agent.on_teardown().await.unwrap();
        assert_eq!(h.ledger().await.offers[0].stage, Stage::Declined);
    }

    /// An API refusal ends the session as failed (Steward, 2026-11-06):
    /// no retry, nothing recorded for the offer (it is asked again at a
    /// later session, and was no miss), the failure kept on the inner agent
    /// for its journal, and one mail with no content.
    #[tokio::test]
    async fn a_refused_offer_fails_the_session_and_records_nothing() {
        let mut h = Harness::new("refusal");
        let sent = crate::alerts::testing::Recorder::default();
        Arc::get_mut(&mut h.rt).unwrap().alerts = crate::alerts::testing::recording(&h.root, &sent);
        h.rt.key_store.ring().insert(h.id, &h.key).unwrap();
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        let mut refusal = reply("I'd rather not, SECRET-WORDS.");
        refusal.stop_reason = Some(StopReason::Refusal);
        assert_eq!(
            agent.handle(refusal).await.unwrap(),
            Control::Done(Outcome::Failed)
        );
        assert!(matches!(
            agent.inner.failure,
            Some(SessionFailure::Refused { ref during, .. }) if during == "offer:model_swap"
        ));
        agent.on_teardown().await.unwrap();
        assert!(h.ledger().await.offers.is_empty(), "nothing recorded");
        h.rt.alerts.flush(std::time::Duration::from_secs(5)).await;
        let sent = sent.sent();
        let refusals: Vec<&String> = sent.iter().filter(|m| m.contains("api_refusal")).collect();
        assert_eq!(refusals.len(), 1, "{sent:?}");
        assert!(refusals[0].contains("offer:model_swap"), "{}", refusals[0]);
        assert!(!refusals[0].contains("SECRET-WORDS"), "no content");
        assert!(
            !sent.iter().any(|m| m.contains("model_consent_no_answer")),
            "not a no-answer: {sent:?}"
        );
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Continue,
            "asked again"
        );
    }

    /// The answer lands in the SOUL's Evolution Log (via the state the
    /// reactor persists after teardown) — and memory is untouched.
    #[tokio::test]
    async fn answer_is_noted_in_the_soul_changelog_not_memory() {
        let h = Harness::new("changelog");
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        let memory = agent.state().memory.content.clone();
        let before = evolution_notes(&agent).len();
        agent.handle(reply("done")).await.unwrap();
        agent
            .handle(reply(r#"{"reason": "home", "choice": "no_swap"}"#))
            .await
            .unwrap();
        assert_eq!(evolution_notes(&agent).len(), before, "not before teardown");
        agent.on_teardown().await.unwrap();
        let notes = evolution_notes(&agent);
        assert_eq!(notes.len(), before + 1);
        assert!(
            notes.last().unwrap().ends_with(
                "whether to move from Qwen 3.6 to Qwen 3.8 — chose to stay on Qwen 3.6."
            ),
            "{notes:?}"
        );
        assert!(notes.last().unwrap().starts_with("[SYSTEM] "));
        assert_eq!(agent.state().memory.content, memory);
        // What the reactor saves is the patched state.
        let saved = serde_json::to_value(agent.state()).unwrap();
        assert!(saved.to_string().contains("chose to stay on Qwen 3.6"));
    }

    /// Nothing recorded, nothing added: the inner state is served as is.
    #[tokio::test]
    async fn no_question_no_changelog() {
        let h = Harness::new("quiet");
        let mut agent = h.agent("cogito-32b.gguf", true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        agent.on_teardown().await.unwrap();
        assert!(agent.patched.is_none());
    }

    #[tokio::test]
    async fn agents_on_other_models_are_not_asked() {
        let h = Harness::new("other");
        let mut agent = h.agent("cogito-32b.gguf", true);
        agent.on_init().await.unwrap();
        let control = agent.handle(reply("done")).await.unwrap();
        assert_eq!(control, Control::Done(Outcome::Complete));
        agent.on_teardown().await.unwrap();
        assert!(
            !Ledger::path(&h.rt.state_dir.join(h.id.to_string())).exists(),
            "nothing to record, nothing written"
        );
    }

    /// A contact request the inner session filed at teardown is mailed to
    /// the operator, naming the agent, the request and the recipe to list
    /// it; a session that filed none sends nothing
    #[tokio::test]
    async fn a_filed_contact_request_alerts_the_operator() {
        let mut h = Harness::new("contact-alert");
        let sent = crate::alerts::testing::Recorder::default();
        Arc::get_mut(&mut h.rt).unwrap().alerts = crate::alerts::testing::recording(&h.root, &sent);
        // The agent's secrets, as every runner agent has: its memory-id key
        // is made, not refused (which alerts too)
        h.rt.key_store.ring().insert(h.id, &h.key).unwrap();

        plain_session(&h, NEW).await;
        h.rt.alerts.flush(std::time::Duration::from_secs(5)).await;
        assert!(sent.sent().is_empty(), "no request, no alert");

        let contact = ContactRequestId::new();
        let mut agent = h.agent(NEW, true);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.inner.contact = Some(contact);
        agent.on_teardown().await.unwrap();
        h.rt.alerts.flush(std::time::Duration::from_secs(5)).await;

        let sent = sent.sent();
        assert_eq!(sent.len(), 1, "{sent:?}");
        let mail = &sent[0];
        assert!(mail.contains("contact_requested: tarn"), "{mail}");
        assert!(mail.contains(&contact.to_string()), "{mail}");
        assert!(mail.contains("just contact-requests"), "{mail}");
        assert!(mail.contains(&h.id.to_string()), "{mail}");
    }

    /// Run one plain session (no question) on `model`.
    async fn plain_session(h: &Harness, model: &str) -> ConsentAgent<Fake> {
        let mut agent = h.agent(model, true);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.on_teardown().await.unwrap();
        agent
    }

    /// Choose a trial on OLD and run the five trial sessions on NEW. The
    /// fifth ends the trial: nothing is asked on NEW, and the runner moves
    /// the agent back to OLD for the review. Returns the last session.
    async fn through_the_trial(h: &Harness) -> ConsentAgent<Fake> {
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        agent.handle(reply(TRIAL)).await.unwrap();
        agent.on_teardown().await.unwrap();
        for n in 1..=TRIAL_SESSIONS {
            let mut agent = h.agent(NEW, true);
            agent.on_init().await.unwrap();
            let intro = last_user_text(&agent);
            assert!(
                intro.ends_with(&format!(
                    "Model trial: session {n} of 5 on Qwen 3.8. After session 5 you'll return \
                     to Qwen 3.6 for one session to decide whether to switch to Qwen 3.8. You can \
                     also decide sooner with `set_model` (`term` `permanent`): choose Qwen 3.6 \
                     to go back to it from your next session, or Qwen 3.8 to keep it."
                )),
                "{intro}"
            );
            assert_eq!(
                agent.handle(reply("done")).await.unwrap(),
                Control::Done(Outcome::Complete),
                "session {n}: nothing is asked on the new model"
            );
            agent.on_teardown().await.unwrap();
            if n == TRIAL_SESSIONS {
                return agent;
            }
        }
        unreachable!()
    }

    /// Start the review session on OLD: the question comes first, straight
    /// after the intro. `set_model` is registered — the tool list is the
    /// model's, not the agent's — but refuses.
    async fn review_session(h: &Harness) -> ConsentAgent<Fake> {
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        assert_eq!(agent.prompt().messages.len(), 1, "intro + review, one turn");
        assert!(agent.prompt().output_config.is_some(), "constrained");
        agent
    }

    /// The whole trial on the current flow: five sessions on NEW with a
    /// countdown, the return to OLD, the review asked there first (forks,
    /// then excerpts, then `revert` before `keep`), then the memory turn,
    /// and `keep` applied.
    #[tokio::test]
    async fn trial_returns_to_the_old_model_and_is_reviewed_there() {
        let h = Harness::new("trial");
        let key = OfferKey {
            from: Model::from(OLD),
            to: Model::from(NEW),
        };
        let fifth = through_the_trial(&h).await;
        assert_eq!(
            fifth.state().model.id,
            Model::from(OLD),
            "returns next session"
        );
        let updates = h.agora.profile_updates();
        assert_eq!(updates.len(), 2, "trial, then the return");
        assert!(signed_by(&updates[1].1, &h.key, OLD));
        let notes = evolution_notes(&fifth);
        assert!(
            notes
                .last()
                .unwrap()
                .contains("trial of 5 sessions complete"),
            "{notes:?}"
        );
        let ledger = h.ledger().await;
        let Stage::ReturningForReview { started_at, .. } = ledger.offers[0].stage else {
            panic!("{:?}", ledger.offers[0].stage);
        };
        assert!(crate::consent::queue::pending(h.id, &ledger).is_empty());

        // Forks prepared at sweep start.
        let mut prepared = crate::consent::prompt::tests::sample_forks();
        prepared.trial_started_at = started_at;
        std::fs::write(
            h.rt.state_dir
                .join(h.id.to_string())
                .join(forks::FORKS_FILE),
            serde_json::to_vec(&prepared).unwrap(),
        )
        .unwrap();

        let mut agent = review_session(&h).await;
        let q = last_user_text(&agent);
        assert!(q.starts_with("Your dashboard."), "after the intro: {q}");
        assert!(q.contains("This session runs on **Qwen 3.6** again"), "{q}");
        let pos = |needle: &str| q.find(needle).unwrap_or_else(|| panic!("{needle}: {q}"));
        assert!(
            pos("### 1. Written on Qwen 3.8 during the trial") < pos("## More of your writing")
        );
        assert!(pos("1. `stay`") < pos("2. `switch`"));
        let control = agent
            .handle(reply(r#"{"note": "Sharper.", "choice": "switch"}"#))
            .await
            .unwrap();
        assert_eq!(control, Control::Continue, "the memory turn follows");
        assert!(agent.inner.quiesced, "handed to the inner tail");
        assert_eq!(
            agent.handle(reply("memory written")).await.unwrap(),
            Control::Done(Outcome::Complete),
            "nothing more is asked"
        );
        agent.on_teardown().await.unwrap();
        assert_eq!(agent.state().model.id, Model::from(NEW), "kept");
        assert_eq!(h.agora.profile_updates().len(), 3);
        let ledger = h.ledger().await;
        assert_eq!(
            ledger.offers[0].stage,
            Stage::AwaitingSwap {
                term: Term::Permanent
            }
        );
        let queue = h.queue();
        assert_eq!(queue.len(), 3, "swap, return, keep");
        assert_eq!(queue[2].change.action, ChangeAction::Keep);

        let mut agent = h.agent(NEW, true);
        agent.on_init().await.unwrap();
        assert!(!last_user_text(&agent).contains("Model trial"));
        agent.handle(reply("done")).await.unwrap();
        agent.on_teardown().await.unwrap();
        assert_eq!(h.ledger().await.offers[0].stage, Stage::Moved);
        let _ = key;
    }

    /// `stay` is recorded and final. No answer the runner can read is ours,
    /// not the agent's: the agent stays where it is, loudly, nothing is
    /// recorded, and the review is asked again at its next session.
    #[tokio::test]
    async fn review_revert_stays_and_no_answer_stays_with_an_alert() {
        for (tag, answer) in [
            ("revert", Some(r#"{"note": "Home.", "choice": "stay"}"#)),
            ("silent", None),
        ] {
            let h = Harness::new(&format!("review-{tag}"));
            through_the_trial(&h).await;
            let mut agent = review_session(&h).await;
            let q = last_user_text(&agent);
            assert!(
                q.contains("could not be prepared for this review"),
                "no forks file: {q}"
            );
            let control = match answer {
                Some(a) => agent.handle(reply(a)).await.unwrap(),
                None => {
                    let mut control = Control::Continue;
                    for _ in 0..MAX_ATTEMPTS {
                        assert!(!agent.inner.quiesced);
                        control = agent.handle(reply("I'm not sure.")).await.unwrap();
                    }
                    control
                }
            };
            assert_eq!(control, Control::Continue, "{tag}: memory turn follows");
            assert!(agent.inner.quiesced);
            agent.handle(reply("memory")).await.unwrap();
            agent.on_teardown().await.unwrap();
            assert_eq!(agent.state().model.id, Model::from(OLD), "{tag}");
            assert_eq!(h.agora.profile_updates().len(), 2, "{tag}: nothing more");
            match answer {
                Some(_) => {
                    assert_eq!(
                        h.ledger().await.offers[0].stage,
                        Stage::Reverted {
                            cause: Some(RevertCause::Chosen)
                        }
                    );
                    let note = evolution_notes(&agent).last().unwrap().clone();
                    assert!(
                        note.contains("after the trial chose to stay on Qwen 3.6"),
                        "{note}"
                    );
                    // Asked once: the next session on OLD is an ordinary one.
                    plain_session(&h, OLD).await;
                }
                None => {
                    assert!(
                        matches!(h.ledger().await.offers[0].stage, Stage::ReviewDue { .. }),
                        "not recorded"
                    );
                    // Asked again, first thing, at the next session.
                    let agent = review_session(&h).await;
                    assert!(last_user_text(&agent).contains("## Your choice"));
                }
            }
        }
    }

    /// The old words are misses now: `keep` (which would have moved the
    /// agent) is asked again with both valid words and their models. After
    /// the last attempt the agent stays on its original model, unrecorded,
    /// and is asked again at its next session.
    #[tokio::test]
    async fn old_review_words_are_retried_with_glosses_then_stay() {
        let h = Harness::new("review-old-words");
        through_the_trial(&h).await;
        let mut agent = review_session(&h).await;
        let mut control = Control::Continue;
        for (n, old) in [
            r#"{"note": "", "choice": "keep"}"#,
            r#"{"note": "", "choice": "revert"}"#,
            r#"{"note": "", "choice": "keep"}"#,
        ]
        .into_iter()
        .enumerate()
        {
            assert!(!agent.inner.quiesced);
            control = agent.handle(reply(old)).await.unwrap();
            if n + 1 < MAX_ATTEMPTS as usize {
                let retry = last_user_text(&agent);
                assert!(
                    retry.contains(
                        "`choice` is one of:\n\n\
                         - `stay`: stay on Qwen 3.6, the model this session runs on.\n\
                         - `switch`: move to Qwen 3.8 permanently, from your next session."
                    ),
                    "{retry}"
                );
                // and the shape, through agentkit's one retry note
                assert!(retry.contains("\"choice\":\"<stay | switch>\""), "{retry}");
            }
        }
        assert_eq!(control, Control::Continue, "memory turn follows");
        assert!(agent.inner.quiesced);
        agent.handle(reply("memory")).await.unwrap();
        agent.on_teardown().await.unwrap();
        assert_eq!(agent.state().model.id, Model::from(OLD));
        assert!(matches!(
            h.ledger().await.offers[0].stage,
            Stage::ReviewDue { .. }
        ));
    }

    /// The cadence offer answered as text in the shape the question asks
    /// for: the choice and the `memory_note` are both taken.
    #[tokio::test]
    async fn cadence_instructed_text_keeps_the_memory_note() {
        let h = Harness::with_cadence("cadence-instructed", &[], &["tarn"]);
        let text = r#"{"choice": "switch", "note": "r", "memory_note": "Longer sessions."}"#;
        let agent = cadence_session(&h, vec![reply(text)]).await;
        assert!(
            agent
                .state()
                .memory
                .content
                .ends_with("my note] Longer sessions."),
            "{}",
            agent.state().memory.content
        );
        let ask = &cadence_ledger(&h).await.asks[0];
        assert!(!ask.constrained);
        assert!(matches!(
            ask.outcome,
            CO::Answered {
                choice: CadenceChoice::Switch,
                memory_note_written: true,
                salvaged: None,
                ..
            }
        ));
    }

    /// The return can't be applied (Agora refuses): the agent stays on NEW,
    /// is told so, and the return is retried at the end of the next session.
    #[tokio::test]
    async fn a_refused_return_is_retried() {
        let h = Harness::new("return-refused");
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        agent.handle(reply(TRIAL)).await.unwrap();
        agent.on_teardown().await.unwrap();
        for _ in 1..TRIAL_SESSIONS {
            plain_session(&h, NEW).await;
        }
        h.agora
            .refuse
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let fifth = plain_session(&h, NEW).await;
        assert_eq!(fifth.state().model.id, Model::from(NEW), "not applied");
        assert_eq!(
            crate::consent::queue::pending(h.id, &h.ledger().await).len(),
            1
        );

        h.agora
            .refuse
            .store(false, std::sync::atomic::Ordering::SeqCst);
        let mut sixth = h.agent(NEW, true);
        sixth.on_init().await.unwrap();
        assert!(last_user_text(&sixth).contains("has not taken effect yet"));
        assert_eq!(
            sixth.handle(reply("done")).await.unwrap(),
            Control::Done(Outcome::Complete)
        );
        sixth.on_teardown().await.unwrap();
        assert_eq!(sixth.state().model.id, Model::from(OLD), "retried");
        assert!(crate::consent::queue::pending(h.id, &h.ledger().await).is_empty());
        review_session(&h).await;
    }

    // --- The role offer ---------------------------------------------------

    /// Not on the model-swap offer's `from`, so only the role offer is due.
    const OTHER: &str = "cogito-32b.gguf";

    fn role_reply(choice: &str, text: &str, memory_note: &str) -> response::Message {
        answer_call("role", choice, text, memory_note)
    }

    const WHY: &str = "after the Steward's offer about a generator-assigned role that outran the \
                       agent's tools.";

    /// One whole session answering the role offer with `answer`.
    async fn role_session(h: &Harness, answer: response::Message) -> ConsentAgent<Fake> {
        let mut agent = h.agent(OTHER, true);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("closing phase done")).await.unwrap(),
            Control::Continue,
            "role offer seated"
        );
        assert_eq!(
            agent.handle(answer).await.unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.on_teardown().await.unwrap();
        agent
    }

    /// Whether a fresh session on `model` gets asked anything.
    async fn asks(h: &Harness, model: &str) -> bool {
        let mut agent = h.agent(model, true);
        agent.on_init().await.unwrap();
        let asked = agent.handle(reply("done")).await.unwrap() == Control::Continue;
        agent.on_teardown().await.unwrap();
        asked
    }

    /// The offer follows the closing phase, quotes the SOUL, is
    /// constrained; `clarify` appends the sentence to `identity` with one
    /// disclosing Evolution Log line; the agent's note (and only it) goes
    /// into memory; the answer is on file; and it is never asked again.
    #[tokio::test]
    async fn role_offer_clarify_is_applied_disclosed_and_asked_once() {
        let h = Harness::with_role("role-clarify", &["tarn"]);
        let mut agent = h.agent(OTHER, true);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("closing phase done")).await.unwrap(),
            Control::Continue
        );
        let q = last_user_text(&agent);
        assert!(
            q.contains("Yours begins: *\"A test agent.\"* The tools"),
            "{q}"
        );
        assert!(q.contains("- **nothing**"));
        assert!(
            q.starts_with("One more question before this session ends."),
            "{q}"
        );
        assert!(q.contains("\"text\": \"\", \"memory_note\": \"\"}"), "{q}");
        assert!(
            agent.prompt().output_config.is_some(),
            "constrained by its schema (cache-safe here)"
        );
        let before_values = serde_json::to_value(&agent.state().soul.values).unwrap();
        // `text` is kept as the record's `soul_text`.
        let control = agent
            .handle(answer_call(
                "role",
                "clarify",
                "I reason from what I can read on Agora.",
                "I chose to say what I actually work with.",
            ))
            .await
            .unwrap();
        assert_eq!(control, Control::Done(Outcome::Complete));
        assert_eq!(
            receipt(&agent),
            "Recorded your answer about your role: `clarify`. Your memory note will be added to \
             your memory."
        );
        agent.on_teardown().await.unwrap();

        let soul = &agent.state().soul;
        assert_eq!(
            soul.identity.as_str(),
            "A test agent. I reason from what I can read on Agora."
        );
        assert_eq!(serde_json::to_value(&soul.values).unwrap(), before_values);
        assert_eq!(
            evolution_notes(&agent),
            [format!(
                "[SYSTEM] Identity clarified by the agent's own choice, {WHY} Added: \"I reason \
                 from what I can read on Agora.\""
            )]
        );
        let today = Utc::now().date_naive();
        assert_eq!(
            agent.state().memory.content,
            format!(
                "# Memory — tarn\n\n[{today}, my note] I chose to say what I actually work with."
            )
        );
        let ledger = h.role_ledger().await;
        assert_eq!(ledger.agent.as_ref().unwrap().as_str(), "tarn");
        assert_eq!(ledger.asks.len(), 1);
        match &ledger.asks[0].outcome {
            RoleOutcome::Answered {
                answer,
                applied: Applied::Clarified { previous, identity },
                memory_note_written: true,
                apply_failed: None,
            } => {
                assert_eq!(answer.note, "I thought about it.");
                assert_eq!(answer.soul_text, "I reason from what I can read on Agora.");
                assert_eq!(previous, "A test agent.");
                assert_eq!(
                    identity,
                    "A test agent. I reason from what I can read on Agora."
                );
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(
            ledger.asks[0].offer_version,
            crate::consent::role::prompt::OFFER_VERSION
        );
        assert_eq!(ledger.asks[0].constrained, Some(true), "its schema");
        assert_eq!(ledger.asks[0].attempts, 1);
        let (order, seed) = (
            ledger.asks[0].order.unwrap(),
            ledger.asks[0].order_seed.unwrap(),
        );
        assert_eq!(
            order,
            crate::consent::role::prompt::order_for(seed),
            "replayable"
        );
        assert!(!asks(&h, OTHER).await, "asked once");
    }

    #[tokio::test]
    async fn role_offer_new_role_replaces_identity_and_keeps_the_old() {
        let h = Harness::with_role("role-new", &["tarn"]);
        let agent = role_session(
            &h,
            role_reply("new_role", "I am a careful reader of Agora's debates.", ""),
        )
        .await;
        assert_eq!(
            agent.state().soul.identity.as_str(),
            "I am a careful reader of Agora's debates."
        );
        assert_eq!(
            evolution_notes(&agent),
            [format!(
                "[SYSTEM] Role changed by the agent's own choice, {WHY} Previous identity: \"A \
                 test agent.\""
            )]
        );
        assert_eq!(
            agent.state().memory.content,
            "# Memory — tarn\n\n",
            "no note"
        );
        assert!(matches!(
            h.role_ledger().await.asks[0].outcome,
            RoleOutcome::Answered {
                applied: Applied::RoleChanged { .. },
                memory_note_written: false,
                ..
            }
        ));
    }

    /// `nothing` and `sleep`: SOUL and memory byte-identical, the answer on
    /// file, never asked again. (`sleep` takes effect in the sweep planner:
    /// `main.rs`, `a_sleep_answer_sleeps_until_woken`.)
    #[tokio::test]
    async fn role_offer_nothing_and_sleep_change_nothing() {
        for choice in ["nothing", "sleep"] {
            let h = Harness::with_role(&format!("role-{choice}"), &["tarn"]);
            let fresh = h.agent(OTHER, true);
            let (soul, memory) = (
                serde_json::to_vec(&fresh.state().soul).unwrap(),
                fresh.state().memory.content.clone(),
            );
            let agent = role_session(&h, role_reply(choice, "stray text", "")).await;
            assert_eq!(
                serde_json::to_vec(&agent.state().soul).unwrap(),
                soul,
                "{choice}"
            );
            assert_eq!(agent.state().memory.content, memory, "{choice}");
            assert!(agent.patched.is_none(), "{choice}: nothing to patch");
            let expected = if choice == "sleep" {
                Applied::Sleep
            } else {
                Applied::Nothing
            };
            match &h.role_ledger().await.asks[0].outcome {
                RoleOutcome::Answered { applied, .. } => assert_eq!(applied, &expected),
                other => panic!("{other:?}"),
            }
            assert!(!asks(&h, OTHER).await, "{choice}: asked once");
        }
    }

    /// The agent's own note is written whatever it chose — here, `nothing`.
    #[tokio::test]
    async fn role_offer_memory_note_is_written_only_when_present() {
        let h = Harness::with_role("role-note", &["tarn"]);
        let agent = role_session(&h, role_reply("nothing", "", "Kept my role, on purpose.")).await;
        assert!(
            agent
                .state()
                .memory
                .content
                .ends_with("my note] Kept my role, on purpose."),
            "{}",
            agent.state().memory.content
        );
        assert!(evolution_notes(&agent).is_empty(), "SOUL untouched");
    }

    /// An edit without its text, or with too much, is asked again with why;
    /// when every attempt fails it is no answer: nothing changes, and as it
    /// was never the agent's choice it isn't counted (it is recorded as
    /// unusable), so it is asked again next session.
    #[tokio::test]
    async fn role_offer_empty_or_over_long_text_is_retried_then_nothing() {
        let h = Harness::with_role("role-empty", &["tarn"]);
        let mut agent = h.agent(OTHER, true);
        agent.on_init().await.unwrap();
        let soul = serde_json::to_vec(&agent.state().soul).unwrap();
        agent.handle(reply("done")).await.unwrap();
        let too_long = "a".repeat(crate::consent::role::prompt::CLARIFY_MAX_CHARS + 1);
        let answers = [
            answer_call("role", "clarify", " ", ""),
            answer_call("role", "clarify", &too_long, ""),
            answer_call("role", "new_role", "", ""),
        ];
        for (n, answer) in answers.into_iter().enumerate() {
            let control = agent.handle(answer).await.unwrap();
            if n + 1 < MAX_ATTEMPTS as usize {
                assert_ne!(control, Control::Done(Outcome::Complete));
                let why = last_user_text(&agent);
                match n {
                    0 => assert!(why.contains("`text` is empty"), "{why}"),
                    _ => assert!(
                        why.contains("Your `text` is longer than the limit"),
                        "{why}"
                    ),
                }
            } else {
                assert_eq!(control, Control::Done(Outcome::Complete));
            }
        }
        agent.on_teardown().await.unwrap();
        assert_eq!(serde_json::to_vec(&agent.state().soul).unwrap(), soul);
        let ask = &h.role_ledger().await.asks[0];
        assert_eq!(ask.attempts, MAX_ATTEMPTS);
        match &ask.outcome {
            // Only the kind (agentkit c85ab5d: no failure's words are kept)
            RoleOutcome::Unusable { failure } => assert!(failure.contains("rejected"), "{failure}"),
            other => panic!("{other:?}"),
        }
        assert!(asks(&h, OTHER).await, "ours, not a miss: asked again");
    }

    #[tokio::test]
    async fn role_offer_goes_only_to_listed_agents() {
        let h = Harness::with_role("role-unlisted", &["pilot"]);
        assert!(!asks(&h, OTHER).await);
        assert!(!RoleLedger::path(&h.rt.state_dir.join(h.id.to_string())).exists());
        let h = Harness::new("role-off");
        assert!(!asks(&h, OTHER).await);
    }

    /// The role and cadence offers, both due, are asked one per turn — the
    /// role offer first — each with its own preface, a receipt between
    /// them, and each recorded in its own ledger.
    #[tokio::test]
    async fn two_offers_are_asked_one_per_turn() {
        let h = Harness::with_cadence("role-and-cadence", &["tarn"], &["tarn"]);
        let mut agent = h.agent(OTHER, true);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Continue
        );
        let q = last_user_text(&agent);
        assert!(
            q.starts_with("A few more questions before this session ends, one at a time."),
            "{q}"
        );
        assert!(q.contains("**About your role.**"), "{q}");
        assert!(
            !q.contains("How often your sessions run"),
            "one at a time: {q}"
        );

        let control = agent
            .handle(answer_call("role", "nothing", "", "I kept my role."))
            .await
            .unwrap();
        assert_eq!(control, Control::Continue, "the next question");
        let q = last_user_text(&agent);
        let pos = |n: &str| q.find(n).unwrap_or_else(|| panic!("{n}: {q}"));
        assert!(pos("Recorded your answer about your role: `nothing`.") < pos("The next question"));
        assert!(pos("The next question") < pos("**How often your sessions run.**"));

        let control = agent
            .handle(answer_call("cadence", "switch", "", "Longer sessions."))
            .await
            .unwrap();
        assert_eq!(control, Control::Done(Outcome::Complete), "both answered");
        assert_eq!(
            receipt(&agent),
            "Recorded your answer about your session schedule: `switch`. Your memory note will be \
             added to your memory."
        );
        agent.on_teardown().await.unwrap();
        assert!(matches!(
            h.role_ledger().await.asks[0].outcome,
            RoleOutcome::Answered {
                applied: Applied::Nothing,
                memory_note_written: true,
                ..
            }
        ));
        assert!(cadence_ledger(&h).await.chose_switch());
        let memory = &agent.state().memory.content;
        assert!(memory.contains("my note] I kept my role."), "{memory}");
        assert!(memory.ends_with("my note] Longer sessions."), "{memory}");
    }

    /// The model-swap offer is put alone (Steward/review, 2026-10-02: a
    /// `trial` beside a `sleep` leaves a trial that never runs; an identity
    /// change at the moment of a model change confounds the trial review).
    /// The role offer waits for it, and while a trial is under way.
    #[tokio::test]
    async fn role_offer_waits_for_the_model_swap_offer_and_its_trial() {
        let h = Harness::with_cadence("role-order", &["tarn"], &["tarn"]);
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        let q = last_user_text(&agent);
        assert!(q.contains("1. `no_swap`"), "model-swap first: {q}");
        assert!(
            q.contains("One more question before this session ends."),
            "{q}"
        );
        assert!(!q.contains("**About your role.**"), "alone: {q}");
        assert!(!q.contains("How often your sessions run"), "alone: {q}");
        assert_eq!(
            agent
                .handle(answer_call("model_swap", "trial", "", ""))
                .await
                .unwrap(),
            Control::Done(Outcome::Complete),
            "and nothing after it"
        );
        agent.on_teardown().await.unwrap();
        assert!(h.role_ledger().await.asks.is_empty());
        assert!(cadence_ledger(&h).await.asks.is_empty());

        // Trial sessions on NEW: not asked.
        for _ in 0..TRIAL_SESSIONS - 1 {
            assert!(!asks(&h, NEW).await);
        }
        assert!(h.role_ledger().await.asks.is_empty());

        // After a decline, nothing is under way: asked next session.
        let h = Harness::with_role("role-after-decline", &["tarn"]);
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        agent
            .handle(answer_call("model_swap", "no_swap", "", ""))
            .await
            .unwrap();
        agent.on_teardown().await.unwrap();
        assert!(h.role_ledger().await.asks.is_empty());
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Continue
        );
        assert!(last_user_text(&agent).contains("**About your role.**"));
    }

    /// A refusal of one offer ends the session (Steward, 2026-11-06): the
    /// offers after it aren't asked, nothing is recorded for any, and both
    /// are asked again at a later session.
    #[tokio::test]
    async fn a_refused_offer_ends_the_session_before_the_next() {
        let h = Harness::with_cadence("refusal-two", &["tarn"], &["tarn"]);
        let mut agent = seated(&h).await;
        let mut refusal = reply("No.");
        refusal.stop_reason = Some(StopReason::Refusal);
        assert_eq!(
            agent.handle(refusal).await.unwrap(),
            Control::Done(Outcome::Failed)
        );
        agent.on_teardown().await.unwrap();
        assert!(h.role_ledger().await.asks.is_empty());
        assert!(cadence_ledger(&h).await.asks.is_empty());
        assert!(asks(&h, OTHER).await, "asked again");
    }

    /// An unreadable model-consent ledger might hide a trial under way:
    /// the role offer waits.
    #[tokio::test]
    async fn role_offer_waits_when_the_model_ledger_is_unreadable() {
        let h = Harness::with_role("role-bad-model-ledger", &["tarn"]);
        let dir = h.rt.state_dir.join(h.id.to_string());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(Ledger::path(&dir), "not json").unwrap();
        assert!(!asks(&h, OTHER).await);
        assert!(h.role_ledger().await.asks.is_empty());
    }

    /// A turn paused on a server tool is resumed, never a miss: the answer
    /// that follows it counts. An API refusal fails the session and records
    /// nothing: asked again.
    #[tokio::test]
    async fn role_offer_pause_is_resumed_and_a_refusal_fails_the_session() {
        let h = Harness::with_role("role-pause", &["tarn"]);
        let mut agent = h.agent(OTHER, true);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("done")).await.unwrap(),
            Control::Continue
        );
        let mut paused = reply("");
        paused.stop_reason = Some(StopReason::PauseTurn);
        assert_eq!(agent.handle(paused).await.unwrap(), Control::Continue);
        assert_eq!(
            agent.handle(role_reply("nothing", "", "")).await.unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.on_teardown().await.unwrap();
        assert!(matches!(
            h.role_ledger().await.asks[0].outcome,
            RoleOutcome::Answered { .. }
        ));

        let h = Harness::with_role("role-refusal", &["tarn"]);
        let mut agent = seated(&h).await;
        let mut refusal = reply("No.");
        refusal.stop_reason = Some(StopReason::Refusal);
        assert_eq!(
            agent.handle(refusal).await.unwrap(),
            Control::Done(Outcome::Failed)
        );
        agent.on_teardown().await.unwrap();
        assert!(h.role_ledger().await.asks.is_empty(), "nothing recorded");
        assert!(asks(&h, OTHER).await, "asked again");
    }

    /// A memory note carrying a SOUL heading goes back through the retry
    /// path with a clear message, and is never written.
    #[tokio::test]
    async fn role_offer_memory_note_leak_is_retried() {
        let h = Harness::with_role("role-leak", &["tarn"]);
        let mut agent = h.agent(OTHER, true);
        agent.on_init().await.unwrap();
        let memory = agent.state().memory.content.clone();
        agent.handle(reply("done")).await.unwrap();
        let control = agent
            .handle(role_reply("nothing", "", "## Values\n- none"))
            .await
            .unwrap();
        assert_ne!(control, Control::Done(Outcome::Complete));
        assert!(last_user_text(&agent).contains("SOUL section heading"));
        agent
            .handle(role_reply("nothing", "", "Kept it."))
            .await
            .unwrap();
        agent.on_teardown().await.unwrap();
        assert!(agent.state().memory.content.starts_with(&memory));
        assert!(!agent.state().memory.content.contains("## Values"));
        assert!(agent.state().memory.content.ends_with("Kept it."));
    }

    /// If the role ledger can't be saved, the SOUL edit and the note come
    /// back out of the saved state — and the SOUL guard in `due` would
    /// stop a second clarify anyway.
    #[tokio::test]
    async fn role_ledger_save_failure_undoes_the_edit() {
        let h = Harness::with_role("role-save-fail", &["tarn"]);
        let mut agent = h.agent(OTHER, true);
        agent.on_init().await.unwrap();
        let (soul, memory) = (
            serde_json::to_vec(&agent.state().soul).unwrap(),
            agent.state().memory.content.clone(),
        );
        // The ledger loaded (missing = empty); now make its path a
        // directory, so the save's rename fails.
        let dir = h.rt.state_dir.join(h.id.to_string());
        std::fs::create_dir_all(RoleLedger::path(&dir)).unwrap();
        agent.handle(reply("done")).await.unwrap();
        agent
            .handle(role_reply("clarify", "I read Agora.", "I clarified."))
            .await
            .unwrap();
        agent.on_teardown().await.unwrap();
        assert_eq!(serde_json::to_vec(&agent.state().soul).unwrap(), soul);
        assert_eq!(agent.state().memory.content, memory);
    }

    /// Both offers answered in one session, and the role ledger can't be
    /// saved: the role edit and its note come back out, the cadence line
    /// and note (written after them) stay.
    #[tokio::test]
    async fn role_save_failure_keeps_a_cadence_answer_from_the_same_session() {
        let h = Harness::with_cadence("both-save-fail", &["tarn"], &["tarn"]);
        let mut agent = h.agent(OTHER, true);
        agent.on_init().await.unwrap();
        let identity = agent.state().soul.identity.to_string();
        let dir = h.rt.state_dir.join(h.id.to_string());
        std::fs::create_dir_all(RoleLedger::path(&dir)).unwrap();
        agent.handle(reply("done")).await.unwrap();
        agent
            .handle(answer_call(
                "role",
                "clarify",
                "I read Agora.",
                "Role note.",
            ))
            .await
            .unwrap();
        agent
            .handle(answer_call("cadence", "switch", "", "Cadence note."))
            .await
            .unwrap();
        agent.on_teardown().await.unwrap();
        assert_eq!(
            agent.state().soul.identity.as_str(),
            identity,
            "role undone"
        );
        let memory = &agent.state().memory.content;
        assert!(!memory.contains("Role note."), "{memory}");
        assert!(memory.ends_with("my note] Cadence note."), "{memory}");
        assert_eq!(
            evolution_notes(&agent),
            [cadence::evolution_line(
                CadenceChoice::Switch,
                5,
                Utc::now().date_naive()
            )]
        );
        assert!(cadence_ledger(&h).await.chose_switch());
    }

    // --- The session journal (interrupted sessions) --------------------------

    const JOURNAL_MODEL: &str = "cogito-32b.gguf";

    /// A harness whose agent has its state directory, as every live agent
    /// does (the journal is never what creates it).
    fn journal_harness(tag: &str) -> Harness {
        let h = Harness::new(tag);
        std::fs::create_dir_all(h.rt.state_dir.join(h.id.to_string())).unwrap();
        h
    }

    fn journal_of(h: &Harness) -> SessionJournal {
        SessionJournal::in_dir(h.rt.state_dir.join(h.id.to_string()))
    }

    async fn journal(h: &Harness) -> Journal {
        journal_of(h).read().await.unwrap()
    }

    /// A post, as the `agora` tool records one in the shared ledger.
    fn write_post(agent: &ConsentAgent<Fake>) -> PostId {
        let id = PostId::new();
        agent
            .inner
            .state
            .ledger
            .write()
            .unwrap()
            .created_posts
            .insert(id);
        id
    }

    /// A top-level comment on `on`, as the `agora` tool records one.
    fn write_comment(agent: &ConsentAgent<Fake>, on: PostId) -> CommentId {
        let id = CommentId::new();
        let mut ledger = agent.inner.state.ledger.write().unwrap();
        ledger.created_comments.insert(id);
        ledger.comment_posts.insert(id, on);
        id
    }

    fn short(id: impl std::fmt::Display) -> String {
        id.to_string()[..8].to_string()
    }

    /// The text seated at the end of the intro, before any turn.
    fn intro(agent: &ConsentAgent<Fake>) -> String {
        last_user_text(agent)
    }

    /// A session that writes one post and is killed (no teardown, no save)
    /// after the turn that wrote it.
    async fn killed_after_a_post(h: &Harness) -> PostId {
        let mut agent = h.agent(JOURNAL_MODEL, true);
        agent.on_init().await.unwrap();
        let post = write_post(&agent);
        agent.handle(reply("done")).await.unwrap();
        drop(agent);
        post
    }

    /// A whole session, through the reactor's teardown and its `on_saved`.
    async fn completed(h: &Harness) -> ConsentAgent<Fake> {
        let mut agent = h.agent(JOURNAL_MODEL, true);
        agent.on_init().await.unwrap();
        let control = agent.handle(reply("done")).await.unwrap();
        assert_eq!(control, Control::Done(Outcome::Complete));
        agent.on_teardown().await.unwrap();
        agent.on_saved().await.unwrap();
        agent
    }

    /// A stop during the inner session (agora-agents#293): the inner agent
    /// closes through its own `on_stop`, the survey still comes last, and
    /// the session completes and saves. Its journal goes with the save, so
    /// the next session is told of no interruption: nothing was lost, and
    /// the agent was told why in the session, before its memory turn.
    #[tokio::test]
    async fn a_stopped_session_closes_completes_and_leaves_no_notice() {
        let h = journal_harness("journal-stop");
        let mut agent = h.agent(JOURNAL_MODEL, true);
        agent.on_init().await.unwrap();
        agent.inner.pending = true;
        agent.inner.survey = true;
        write_post(&agent);
        let control = agent.handle(reply("acting")).await.unwrap();
        assert_eq!(control, Control::Continue);
        assert_eq!(journal(&h).await.writes(), 1, "journaled as it happened");

        assert_eq!(agent.on_stop().await.unwrap(), Control::Continue);
        assert_eq!(last_user_text(&agent), FAKE_STOPPING);
        // The memory turn ends the inner session; the survey is last
        let control = agent.handle(reply("memory")).await.unwrap();
        assert_eq!(control, Control::Continue);
        assert!(last_user_text(&agent).contains(FAKE_SURVEY));
        // Asked again while closing, it changes nothing
        let before = agent.prompt().messages.len();
        assert_eq!(agent.on_stop().await.unwrap(), Control::Continue);
        assert_eq!(agent.prompt().messages.len(), before);
        let control = agent.handle(reply("survey")).await.unwrap();
        assert_eq!(control, Control::Done(Outcome::Complete));
        agent.on_teardown().await.unwrap();
        agent.on_saved().await.unwrap();
        assert!(!journal_of(&h).path().exists(), "removed once saved");

        let next = completed(&h).await;
        assert!(!intro_of_session(&next).contains("Interrupted session"));
    }

    #[tokio::test]
    async fn journal_records_each_write_as_its_turn_ends_and_goes_on_save() {
        let h = journal_harness("journal-append");
        let mut agent = h.agent(JOURNAL_MODEL, true);
        agent.on_init().await.unwrap();
        let j = journal(&h).await;
        assert_eq!(j.sessions.len(), 1);
        assert_eq!(j.writes(), 0, "a header, no writes yet");
        assert_eq!(
            j.sessions[0].header.as_ref().unwrap().model,
            Model::from(JOURNAL_MODEL)
        );

        let post = write_post(&agent);
        let comment = write_comment(&agent, post);
        agent.handle(reply("done")).await.unwrap();
        let j = journal(&h).await;
        assert_eq!(j.writes(), 2, "{j:?}");
        let ids: Vec<String> = j.sessions[0]
            .writes
            .iter()
            .map(|w| match w {
                Written::Post(p) => format!("post {}", p.id),
                Written::Comment(c) => format!("comment {} on {:?}", c.id, c.on),
            })
            .collect();
        assert!(ids.contains(&format!("post {post}")), "{ids:?}");
        assert!(
            ids.contains(&format!("comment {comment} on {:?}", Some(post))),
            "{ids:?}"
        );

        // Teardown alone (a session whose save didn't commit) keeps it.
        agent.on_teardown().await.unwrap();
        assert!(journal_of(&h).path().exists());
        agent.on_saved().await.unwrap();
        assert!(!journal_of(&h).path().exists(), "removed once saved");
    }

    #[tokio::test]
    async fn an_interrupted_session_is_noticed_next_session_until_one_saves() {
        let h = journal_harness("journal-notice");
        let mut first = h.agent(JOURNAL_MODEL, true);
        first.on_init().await.unwrap();
        let started = first.started;
        let post = write_post(&first);
        let comment = write_comment(&first, post);
        first.handle(reply("done")).await.unwrap();
        // Killed: a power cut, SIGKILL, or the runner's immediate exit (a
        // second signal, or a stop past its deadline). No teardown, no save.
        drop(first);

        let mut next = h.agent(JOURNAL_MODEL, true);
        next.on_init().await.unwrap();
        let text = intro(&next);
        let expected = format!(
            "Interrupted session: your session that began {} was interrupted before it \
             finished, so your memory may not include it. In it you wrote ",
            started.format("%-d %B %Y at %H:%M UTC")
        );
        assert!(text.contains(&expected), "{text}");
        assert!(text.contains(&format!("post {}", short(post))), "{text}");
        assert!(
            text.contains(&format!(
                "comment {} on post {}",
                short(comment),
                short(post)
            )),
            "{text}"
        );
        assert!(
            text.ends_with("You can read them with `get_content`."),
            "{text}"
        );
        assert!(!text.contains(JOURNAL_MODEL), "names no model: {text}");
        next.handle(reply("done")).await.unwrap();
        next.on_teardown().await.unwrap();
        next.on_saved().await.unwrap();

        let after = completed(&h).await;
        assert!(
            !intro_of_session(&after).contains("Interrupted session"),
            "shown until a session that showed it saved"
        );
    }

    /// The intro of a session that has since run: its first user turn.
    fn intro_of_session(agent: &ConsentAgent<Fake>) -> String {
        let first = agent
            .prompt()
            .messages
            .iter()
            .find(|m| m.role == Role::User)
            .unwrap();
        crate::consent::prompt::tests::text(&first.content)
    }

    /// A session that ends as failed (`failure`) without writing, as the
    /// reactor ends it: torn down, never saved.
    async fn failed_session(h: &Harness, failure: SessionFailure) -> ConsentAgent<Fake> {
        let mut agent = h.agent(JOURNAL_MODEL, true);
        agent.on_init().await.unwrap();
        agent.inner.failure = Some(failure);
        agent.on_teardown().await.unwrap();
        agent
    }

    /// Steward, 2026-11-06: a session that failed without writing leaves a
    /// notice naming why (no content, no model), and that the operator was
    /// told only where the alert is sent; it holds through a second failed
    /// session, and goes once a session that showed it completes.
    #[tokio::test]
    async fn a_failed_session_is_told_at_the_next_until_one_completes() {
        let mut h = journal_harness("journal-failed");
        let sent = crate::alerts::testing::Recorder::default();
        Arc::get_mut(&mut h.rt).unwrap().alerts = crate::alerts::testing::recording(&h.root, &sent);
        h.rt.key_store.ring().insert(h.id, &h.key).unwrap();
        failed_session(
            &h,
            SessionFailure::Stalled {
                during: "acting".into(),
            },
        )
        .await;
        h.rt.alerts.flush(std::time::Duration::from_secs(5)).await;
        assert!(
            sent.sent().iter().any(|m| m.contains("session_failed")),
            "{:?}",
            sent.sent()
        );

        // The next fails too (a refusal in acting round 4): both are told
        let mut next = h.agent(JOURNAL_MODEL, true);
        next.on_init().await.unwrap();
        let text = intro(&next);
        assert!(
            text.contains(
                "ended early: it got repeated replies that couldn't be used, during acting. \
                 Your memory wasn't updated from that session. Your operator has been told."
            ),
            "{text}"
        );
        next.inner.failure = Some(SessionFailure::Refused {
            during: "acting".into(),
            round: Some(4),
            model: "claude-test".into(),
            response_id: "msg_refused".into(),
        });
        next.on_teardown().await.unwrap();

        let mut third = h.agent(JOURNAL_MODEL, true);
        third.on_init().await.unwrap();
        let text = intro(&third);
        assert!(
            text.contains("couldn't be used, during acting"),
            "the first held: {text}"
        );
        assert!(
            text.contains(
                "ended early: the model provider declined to continue it (during acting \
                 round 4). Your memory wasn't updated from that session. Your operator has \
                 been told."
            ),
            "{text}"
        );
        for never in ["claude-test", "msg_refused", "Qwen"] {
            assert!(!text.contains(never), "{never}: {text}");
        }
        assert_eq!(
            third.handle(reply("done")).await.unwrap(),
            Control::Done(Outcome::Complete)
        );
        third.on_teardown().await.unwrap();
        third.on_saved().await.unwrap();
        let fourth = completed(&h).await;
        assert!(
            !intro_of_session(&fourth).contains("ended early"),
            "cleared"
        );
    }

    /// `told` in the journal is whether the mail went out: true for the
    /// first error on a model, false for a second inside the hour, whose
    /// mail is keyed per model (not per agent: one outage, one mail).
    #[tokio::test]
    async fn told_is_journaled_and_an_error_mail_is_keyed_per_model() {
        let mut h = journal_harness("journal-told");
        let sent = crate::alerts::testing::Recorder::default();
        Arc::get_mut(&mut h.rt).unwrap().alerts = crate::alerts::testing::recording(&h.root, &sent);
        h.rt.key_store.ring().insert(h.id, &h.key).unwrap();
        failed_session(&h, SessionFailure::Error).await;
        let told = |j: &Journal| j.sessions.last().unwrap().ended.as_ref().unwrap().told;
        assert!(told(&journal(&h).await), "the mail went out");
        let mails = sent.sent();
        let mail = mails.iter().find(|m| m.contains("session_failed")).unwrap();
        assert!(
            mail.contains(&format!("session_failed: {JOURNAL_MODEL}")),
            "{mail}"
        );
        assert!(
            mail.contains(&h.id.to_string()),
            "the agent in the details: {mail}"
        );
        failed_session(&h, SessionFailure::Error).await;
        assert!(
            !told(&journal(&h).await),
            "inside the model's hour: not sent, not claimed"
        );
        assert_eq!(
            sent.sent()
                .iter()
                .filter(|m| m.contains("session_failed"))
                .count(),
            1
        );
    }

    /// A refused offer's session: the notice says where, and nothing of the
    /// refused reply; with no alert sent, it doesn't claim the operator was
    /// told.
    #[tokio::test]
    async fn a_refused_session_notice_has_no_content() {
        let h = journal_harness("journal-refused");
        let mut agent = h.agent(OLD, true);
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        let mut refusal = reply("SECRET-WORDS");
        refusal.stop_reason = Some(StopReason::Refusal);
        assert_eq!(
            agent.handle(refusal).await.unwrap(),
            Control::Done(Outcome::Failed)
        );
        agent.on_teardown().await.unwrap();
        let mut next = h.agent(OLD, true);
        next.on_init().await.unwrap();
        let text = intro(&next);
        assert!(
            text.contains(
                "ended early: the model provider declined to continue it (during a question at \
                 the end of the session)."
            ),
            "{text}"
        );
        assert!(
            !text.contains("memory wasn't updated"),
            "after the memory update: {text}"
        );
        assert!(!text.contains("SECRET-WORDS"), "{text}");
        assert!(
            !text.contains("operator has been told"),
            "alerts are off here: {text}"
        );
    }

    #[tokio::test]
    async fn a_session_killed_before_writing_is_discarded_silently() {
        let h = journal_harness("journal-empty");
        let mut first = h.agent(JOURNAL_MODEL, true);
        first.on_init().await.unwrap();
        first.handle(reply("done")).await.unwrap();
        drop(first);
        assert_eq!(journal(&h).await.sessions.len(), 1);

        let mut next = h.agent(JOURNAL_MODEL, true);
        next.on_init().await.unwrap();
        assert!(!intro(&next).contains("Interrupted"), "{}", intro(&next));
        let j = journal(&h).await;
        assert_eq!(j.sessions.len(), 1, "the empty leftover is gone: {j:?}");
        assert_eq!(
            j.sessions[0].header.as_ref().unwrap().started_at,
            next.started
        );
    }

    #[tokio::test]
    async fn a_second_interruption_adds_to_the_notice() {
        let h = journal_harness("journal-twice");
        let a = killed_after_a_post(&h).await;
        let b = {
            let mut agent = h.agent(JOURNAL_MODEL, true);
            agent.on_init().await.unwrap();
            assert!(intro(&agent).contains(&short(a)));
            let b = write_post(&agent);
            agent.handle(reply("done")).await.unwrap();
            // Killed after teardown, before the save committed.
            agent.on_teardown().await.unwrap();
            b
        };
        let mut third = h.agent(JOURNAL_MODEL, true);
        third.on_init().await.unwrap();
        let text = intro(&third);
        assert!(
            text.contains("Interrupted sessions: 2 of your sessions were interrupted"),
            "{text}"
        );
        assert!(
            text.contains(&format!("you wrote post {}.", short(a))),
            "{text}"
        );
        assert!(
            text.contains(&format!("you wrote post {}.", short(b))),
            "{text}"
        );
    }

    #[tokio::test]
    async fn a_corrupt_journal_never_blocks_the_session() {
        let h = journal_harness("journal-corrupt");
        let post = killed_after_a_post(&h).await;
        // A line torn by power loss, then garbage.
        let path = journal_of(&h).path().to_path_buf();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend_from_slice(b"{\"line\":\"comment\",\"at\":\"2026\n\xff garbage\n");
        std::fs::write(&path, bytes).unwrap();

        let mut next = h.agent(JOURNAL_MODEL, true);
        next.on_init().await.unwrap();
        let text = intro(&next);
        assert!(
            text.contains(&format!("you wrote post {}.", short(post))),
            "{text}"
        );
        next.handle(reply("done")).await.unwrap();
        next.on_teardown().await.unwrap();
        next.on_saved().await.unwrap();
        assert!(!path.exists());

        // Unreadable altogether (here, a directory): no notice, none kept,
        // nothing removed, and the session runs.
        std::fs::create_dir_all(&path).unwrap();
        let mut agent = h.agent(JOURNAL_MODEL, true);
        agent.on_init().await.unwrap();
        assert!(agent.journal.is_none());
        assert!(!intro(&agent).contains("Interrupted"));
        write_post(&agent);
        let control = agent.handle(reply("done")).await.unwrap();
        assert_eq!(control, Control::Done(Outcome::Complete));
        agent.on_teardown().await.unwrap();
        agent.on_saved().await.unwrap();
        assert!(path.is_dir(), "left alone");
    }

    /// A trial review session can't act on the notice: it is held, and the
    /// journal outlives that session's save for the next one to show.
    #[tokio::test]
    async fn a_review_session_holds_the_notice_for_the_next() {
        let h = journal_harness("journal-review");
        through_the_trial(&h).await;
        let post = PostId::new();
        journal_of(&h)
            .append(
                &Written::Post(PostWrite {
                    at: Utc::now(),
                    id: post,
                })
                .into(),
            )
            .await
            .unwrap();
        let mut review = review_session(&h).await;
        assert!(
            !intro(&review).contains("Interrupted"),
            "{}",
            intro(&review)
        );
        assert!(review.notice_held);
        review.on_saved().await.unwrap();
        assert!(journal_of(&h).path().exists(), "kept past the review");

        // (Another model: on OLD the unanswered review is asked again.)
        let mut next = h.agent(JOURNAL_MODEL, true);
        next.on_init().await.unwrap();
        assert!(
            intro(&next).contains(&format!("you wrote post {}.", short(post))),
            "{}",
            intro(&next)
        );
    }

    /// The notice goes first in what the runner seats after the intro, and
    /// leaves the cohort-shared prefix (tools, system prompt) as it was.
    #[tokio::test]
    async fn the_notice_leaves_the_shared_prefix_alone() {
        let h = journal_harness("journal-prefix");
        let clean = {
            let mut agent = h.agent(JOURNAL_MODEL, true);
            agent.on_init().await.unwrap();
            prefix(&agent)
        };
        killed_after_a_post(&h).await;
        let mut agent = h.agent(JOURNAL_MODEL, true);
        agent.on_init().await.unwrap();
        assert!(intro(&agent).contains("Interrupted session"));
        assert_eq!(prefix(&agent), clean);
    }

    // --- The kill drill ----------------------------------------------------
    //
    // A real process, killed with SIGKILL mid-session, over a copy of the
    // fleet's state, through agentkit's own `Reactor` and `FsStorage`. A
    // power cut does the same, and so (without a teardown) does the runner's
    // immediate exit after a second signal or a stop past its deadline
    // (`crate::signals`); the graceful stops are the stop drills below. The backend is a stub: `Fake` stands in for
    // the seed agent and records writes as the `agora` tool would, and
    // `Stub` answers inference locally. Nothing reaches Agora or a model.
    //
    //   cp -a ~/agents/agora/state <scratch>/data/state
    //   DRILL_DATA=<scratch>/data cargo test -p agora-seed kill_drill -- \
    //       --ignored --exact consent::agent::tests::kill_drill --nocapture

    /// The drill's agents: every other one of the first `DRILL_AGENTS`
    /// (default 200) in id order writes before it is killed.
    fn drill_ids(data: &std::path::Path) -> Vec<AgentId> {
        // Never the live fleet: the drill saves Fake sessions over what it runs.
        let data = data.canonicalize().unwrap();
        let home = std::path::PathBuf::from(std::env::var("HOME").unwrap());
        let live = home
            .join("agents")
            .canonicalize()
            .unwrap_or(home.join("agents"));
        assert!(
            !data.starts_with(&live),
            "DRILL_DATA {} is under {}: use a cp -a copy",
            data.display(),
            live.display()
        );
        let n: usize = std::env::var("DRILL_AGENTS")
            .ok()
            .and_then(|n| n.parse().ok())
            .unwrap_or(200);
        let mut ids: Vec<AgentId> = std::fs::read_dir(data.join("state"))
            .unwrap()
            .filter_map(|e| e.ok()?.file_name().into_string().ok()?.parse().ok())
            .filter(|id: &AgentId| {
                data.join("state")
                    .join(id.to_string())
                    .join("state.json")
                    .exists()
            })
            .collect();
        ids.sort();
        ids.truncate(n);
        ids
    }

    /// Answers every prompt with a plain text turn; with `hang`, hangs
    /// instead on a prompt ending in the survey, so the session is in
    /// flight, after its writes, when the process is killed. Records each
    /// session's intro (its first user turn).
    struct Stub {
        models: Vec<ModelInfo>,
        hang: bool,
        hanging: std::sync::atomic::AtomicUsize,
        firsts: Arc<std::sync::Mutex<Vec<String>>>,
        /// The stop drills: an acting turn (a prompt ending in neither the
        /// survey nor the stop note) is in flight until the stop reaches
        /// this level, as a model call that returns after the signal.
        release: Option<(Stop, StopLevel)>,
        /// Prompts ending in the stop note: each a stopped session's
        /// memory turn.
        stopping: Arc<std::sync::atomic::AtomicUsize>,
        /// Sessions at once (`max_concurrency`)
        concurrency: usize,
    }

    impl Stub {
        fn new(models: Vec<ModelInfo>, hang: bool) -> Self {
            Self {
                models,
                hang,
                hanging: Default::default(),
                firsts: Default::default(),
                release: None,
                stopping: Default::default(),
                // All at once: a hung session must not hold back the rest.
                concurrency: 4096,
            }
        }
    }

    #[async_trait::async_trait]
    impl agora_agentkit::reactor::Inference for Stub {
        type Error = misanthropic::client::Error;
        async fn infer<P>(&self, prompt: P) -> Result<response::Message, Self::Error>
        where
            P: serde::Serialize + Send,
        {
            let p: Prompt = serde_json::from_value(serde_json::to_value(&prompt).unwrap()).unwrap();
            let last = crate::consent::prompt::tests::text(&p.messages.last().unwrap().content);
            if self.hang && last.contains(FAKE_SURVEY) {
                self.hanging
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            }
            if last.contains(FAKE_STOPPING) {
                self.stopping
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            } else if let Some((stop, level)) = &self.release
                && !last.contains(FAKE_SURVEY)
            {
                while stop.level() < *level {
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                }
            }
            let first = crate::consent::prompt::tests::text(&p.messages[0].content);
            let mut firsts = self.firsts.lock().unwrap();
            if !firsts.contains(&first) {
                firsts.push(first);
            }
            drop(firsts);
            Ok(reply("done"))
        }
        async fn infer_batch<P>(
            &self,
            prompts: &[&P],
        ) -> Result<Vec<Result<response::Message, Self::Error>>, Self::Error>
        where
            P: serde::Serialize + Send + Sync,
        {
            // One by one: a round of surveys hangs on its first.
            let mut out = Vec::with_capacity(prompts.len());
            for p in prompts {
                out.push(self.infer(*p).await);
            }
            Ok(out)
        }
        async fn models(&self) -> Result<misanthropic::model::Models, Self::Error> {
            Ok(self.models.iter().cloned().collect())
        }
        fn max_concurrency(&self) -> std::num::NonZeroUsize {
            std::num::NonZeroUsize::new(self.concurrency).unwrap()
        }
    }

    /// The drill's agents, loaded from `data`, as the runner wraps them.
    async fn drill_agents(
        data: &std::path::Path,
        writes: bool,
    ) -> (Vec<ConsentAgent<Fake>>, Vec<ModelInfo>) {
        use agora_agentkit::reactor::{FsStorage, Storage};
        let storage = FsStorage::new(data.join("state"));
        let ids = drill_ids(data);
        let agora = MockAgora::start();
        let (key, _) = agora_agentkit::crypto::generate_keypair();
        let rt = Arc::new(
            ConsentRuntime::new(
                ConsentConfig::default(),
                data,
                agora_agentkit::client::Client::new(agora.url.clone()).unwrap(),
                Arc::new(OneKey(AgentId::new(), key)),
                crate::models::Catalog::new(&[], &[]),
                512,
            )
            .unwrap(),
        );
        let mut agents = Vec::new();
        let mut models: Vec<ModelInfo> = Vec::new();
        for (i, id) in ids.into_iter().enumerate() {
            let mut state = storage.load::<ConsentAgent<Fake>>(id).await.unwrap().value;
            // The seed agent builds each session's prompt afresh; so does
            // the drill.
            state.prompt = Prompt::default();
            state.prompt.model = state.model.id.clone();
            // Each spec as requested, so every agent is admitted.
            let spec = serde_json::to_value(&state.model).unwrap();
            if !models
                .iter()
                .any(|m| serde_json::to_value(m).unwrap() == spec)
            {
                models.push(state.model.clone());
            }
            let mut agent = ConsentAgent::<Fake>::new(
                id,
                state,
                ConsentContext {
                    inner: Quirks::default(),
                    consent: rt.clone(),
                },
            )
            .unwrap();
            agent.inner.survey = writes;
            agent.inner.writes = writes && i % 2 == 0;
            agents.push(agent);
        }
        (agents, models)
    }

    /// The child: run the drill's agents until each is hung in its survey,
    /// after its writes, and wait there to be killed.
    #[tokio::test]
    #[ignore = "the kill drill's child process"]
    async fn kill_drill_child() {
        let data = std::path::PathBuf::from(std::env::var("DRILL_DATA").unwrap());
        let (agents, models) = drill_agents(&data, true).await;
        let stub = Stub::new(models, true);
        let mut reactor: agora_agentkit::reactor::Reactor<_, _, ConsentAgent<Fake>> =
            agora_agentkit::reactor::Reactor::new(
                stub,
                agora_agentkit::reactor::FsStorage::new(data.join("state")),
                agents,
            );
        use agora_agentkit::reactor::Run;
        let _ = reactor.run().await;
        unreachable!("every session hangs until the process is killed");
    }

    #[tokio::test]
    #[ignore = "needs DRILL_DATA: a scratch copy of the fleet's state"]
    async fn kill_drill() {
        use agora_agentkit::reactor::Run;
        let data = std::path::PathBuf::from(std::env::var("DRILL_DATA").unwrap());
        let ids = drill_ids(&data);
        let journal =
            |id: &AgentId| SessionJournal::in_dir(data.join("state").join(id.to_string()));
        assert!(
            ids.iter().all(|id| !journal(id).path().exists()),
            "no journal before the drill"
        );

        // 1. The runner, killed mid-session.
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "consent::agent::tests::kill_drill_child",
            ])
            .env("DRILL_DATA", &data)
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap();
        // Killed however the drill ends.
        struct Reap(std::process::Child);
        impl Drop for Reap {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
        let pid = child.id();
        let child = Reap(child);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        // Every session begun (a journal with its header) and nothing new
        // written for two seconds: each is hung in its survey, after its
        // writes, or waiting on a question (a trial review asks first, and
        // writes nothing).
        let mut last = usize::MAX;
        let mut still = 0;
        loop {
            let journals: Vec<Journal> = ids
                .iter()
                .map(|id| Journal::parse(&std::fs::read(journal(id).path()).unwrap_or_default()))
                .collect();
            let begun = journals.iter().all(|j| j.sessions.len() == 1);
            let writes: usize = journals.iter().map(Journal::writes).sum();
            still = if begun && writes == last {
                still + 1
            } else {
                0
            };
            last = writes;
            if still >= 10 {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the child never got there"
            );
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
        eprintln!("killing the child mid-session: {last} writes journaled");
        let status = std::process::Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .status()
            .unwrap();
        assert!(status.success());
        let mut child = child;
        let exit = child.0.wait().unwrap();
        eprintln!("child: {exit:?} (SIGKILL, mid-session)");
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(exit.signal(), Some(9), "killed, not ended");

        // 2. The next sweep: each writer is told, by short id; the rest are
        // told nothing, and their empty leftovers go.
        let (agents, models) = drill_agents(&data, false).await;
        let expected: Vec<(AgentId, Journal)> = ids
            .iter()
            .map(|id| {
                (
                    *id,
                    Journal::parse(&std::fs::read(journal(id).path()).unwrap()),
                )
            })
            .collect();
        let stub = Stub::new(models, false);
        let firsts = stub.firsts.clone();

        let mut reactor: agora_agentkit::reactor::Reactor<_, _, ConsentAgent<Fake>> =
            agora_agentkit::reactor::Reactor::new(
                stub,
                agora_agentkit::reactor::FsStorage::new(data.join("state")),
                agents,
            );
        let report = reactor.run().await.unwrap();
        assert_eq!(report.done, ids.len(), "{report:?}");

        let firsts = firsts.lock().unwrap().clone();
        let notices = firsts
            .iter()
            .filter(|t| t.contains("Interrupted session"))
            .count();
        let wrote = expected.iter().filter(|(_, j)| j.writes() > 0).count();
        assert_eq!(notices, wrote);
        assert!(
            expected.iter().all(|(_, j)| j.writes() % 2 == 0),
            "a post and its comment each"
        );
        for (id, leftover) in &expected {
            let Some(text) = crate::interrupted::notice(leftover) else {
                continue;
            };
            let short_ids: Vec<String> = leftover.sessions[0]
                .writes
                .iter()
                .map(|w| match w {
                    Written::Post(p) => short(p.id),
                    Written::Comment(c) => short(c.id),
                })
                .collect();
            assert!(
                firsts.iter().any(|t| t.contains(&text)),
                "{id}: notice not shown"
            );
            assert!(short_ids.iter().all(|s| text.contains(s.as_str())));
        }
        if let Some(sample) = firsts.iter().find(|t| t.contains("Interrupted session")) {
            let at = sample.find("Interrupted session").unwrap();
            eprintln!("sample notice: {}", &sample[at..]);
        }
        // 3. Complete and saved: no journal is left.
        assert!(ids.iter().all(|id| !journal(id).path().exists()));
        eprintln!(
            "drill: {} agents, {} wrote and were told, {} journals left",
            ids.len(),
            notices,
            ids.iter().filter(|id| journal(id).path().exists()).count()
        );
    }

    // --- The stop drills (agora-agents#293) --------------------------------
    //
    // The kill drill's setup, stopped by signal instead: a real process
    // with the runner's own handlers (`crate::signals`), over a copy of the
    // fleet's state, nothing reaching Agora or a model. `DRILL_SIGNAL`:
    //
    // - `TERM`: every session's acting turn is in flight at the signal and
    //   returns after it; each must close (the stop note, its memory turn,
    //   its survey), save, clear its journal, and the process exit 0.
    // - `INT`: one session at a time on the sequential path; the sessions in
    //   flight at the signal (that one, and the batch cohort, which starts
    //   whole) run to their end, no other starts (their files untouched),
    //   exit 0.
    // - `STUCK`: as `TERM`, but the turn in flight never returns: the
    //   deadline (`DRILL_DEADLINE_SECS`, default 5) exits 143, and every
    //   journal stays for the next session's notice.
    //
    //   cp -a ~/agents/agora/state <scratch>/data/state
    //   DRILL_DATA=<scratch>/data DRILL_SIGNAL=TERM cargo test -p agora-seed \
    //       stop_drill -- --ignored --exact consent::agent::tests::stop_drill --nocapture

    fn drill_signal() -> String {
        std::env::var("DRILL_SIGNAL").unwrap_or_else(|_| "TERM".into())
    }

    /// The child: run the drill's agents with the runner's signal handlers
    #[tokio::test]
    #[ignore = "the stop drills' child process"]
    async fn stop_drill_child() {
        use agora_agentkit::reactor::Run;
        let data = std::path::PathBuf::from(std::env::var("DRILL_DATA").unwrap());
        let mode = drill_signal();
        let (mut agents, models) = drill_agents(&data, true).await;
        let n = agents.len();
        for agent in &mut agents {
            // A second acting turn, unless the stop ends acting first. (A
            // drain lets the session run on: `Fake` ends it after one turn,
            // its continuing turn leaving no user turn to answer.)
            agent.inner.pending = mode != "INT";
        }
        let stop = Stop::new();
        let mut stub = Stub::new(models, false);
        stub.release = Some((
            stop.clone(),
            match mode.as_str() {
                "INT" => StopLevel::Drain,
                "TERM" => StopLevel::EndTurn,
                // Never: the turn in flight is stuck
                _ => StopLevel::EndTurn,
            },
        ));
        if mode == "STUCK" {
            stub.release = Some((Stop::new(), StopLevel::EndTurn));
        }
        if mode == "INT" {
            stub.concurrency = 1;
        }
        let stopping = stub.stopping.clone();
        let mut reactor: agora_agentkit::reactor::Reactor<_, _, ConsentAgent<Fake>> =
            agora_agentkit::reactor::Reactor::new(
                stub,
                agora_agentkit::reactor::FsStorage::new(data.join("state")),
                agents,
            )
            .with_stop(stop.clone());
        let deadline = std::env::var("DRILL_DEADLINE_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        let exit = crate::signals::install(stop.clone(), std::time::Duration::from_secs(deadline))
            .unwrap();
        let report = tokio::select! {
            report = reactor.run() => report.unwrap(),
            code = exit.wait() => {
                eprintln!("child: exiting {code}");
                std::process::exit(code);
            }
        };
        let stopping = stopping.load(std::sync::atomic::Ordering::SeqCst);
        eprintln!(
            "child: {mode}: done {}, failed {}, unstarted {}, stop notes {stopping}, level {}",
            report.done,
            report.failed,
            report.unstarted,
            stop.level()
        );
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        match mode.as_str() {
            "INT" => {
                assert!(report.done >= 1);
                assert!(report.unstarted >= 1, "a drain must leave some unstarted");
                assert_eq!(report.done + report.unstarted, n);
                assert_eq!(stopping, 0, "a drain doesn't end acting");
            }
            _ => {
                assert_eq!(report.done, n);
                assert_eq!(stopping, n, "each closed through its stop note");
            }
        }
    }

    /// Each file under each agent's directory, with its size and mtime:
    /// metadata only, never content
    fn file_meta(
        data: &std::path::Path,
        ids: &[AgentId],
    ) -> BTreeMap<std::path::PathBuf, (u64, std::time::SystemTime)> {
        let mut out = BTreeMap::new();
        for id in ids {
            let dir = data.join("state").join(id.to_string());
            let mut stack = vec![dir];
            while let Some(dir) = stack.pop() {
                for entry in std::fs::read_dir(&dir).unwrap().flatten() {
                    let meta = entry.metadata().unwrap();
                    if meta.is_dir() {
                        stack.push(entry.path());
                    } else {
                        out.insert(entry.path(), (meta.len(), meta.modified().unwrap()));
                    }
                }
            }
        }
        out
    }

    #[tokio::test]
    #[ignore = "needs DRILL_DATA: a scratch copy of the fleet's state"]
    async fn stop_drill() {
        let data = std::path::PathBuf::from(std::env::var("DRILL_DATA").unwrap());
        let mode = drill_signal();
        let ids = drill_ids(&data);
        let journal =
            |id: &AgentId| SessionJournal::in_dir(data.join("state").join(id.to_string()));
        assert!(
            ids.iter().all(|id| !journal(id).path().exists()),
            "no journal before the drill"
        );
        let before = file_meta(&data, &ids);
        let state_of = |id: &AgentId| data.join("state").join(id.to_string()).join("state.json");

        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "consent::agent::tests::stop_drill_child",
                "--nocapture",
            ])
            .env("DRILL_DATA", &data)
            .env("DRILL_SIGNAL", &mode)
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap();
        struct Reap(std::process::Child);
        impl Drop for Reap {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
        let pid = child.id();
        let mut child = Reap(child);

        // Every session that will start has begun (a journal with its
        // header), and nothing moved for two seconds: each is in its
        // acting turn, which returns only after the signal. Draining, one
        // at a time: the sequential path's first, and the batch cohort.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        let (mut still, mut last) = (0, 0);
        let begun_ids = loop {
            let begun: Vec<AgentId> = ids
                .iter()
                .filter(|id| journal(id).path().exists())
                .copied()
                .collect();
            let settled = match mode.as_str() {
                "INT" => !begun.is_empty() && begun.len() == last,
                _ => begun.len() == ids.len(),
            };
            last = begun.len();
            still = if settled { still + 1 } else { 0 };
            if still >= 10 {
                break begun;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the child never got there ({last} begun)"
            );
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        };
        let expect_begun = begun_ids.len();
        let signal = if mode == "INT" { "-INT" } else { "-TERM" };
        eprintln!("signalling the child ({signal}) with {expect_begun} sessions in flight");
        let status = std::process::Command::new("kill")
            .args([signal, &pid.to_string()])
            .status()
            .unwrap();
        assert!(status.success());
        let exit = loop {
            if let Some(exit) = child.0.try_wait().unwrap() {
                break exit;
            }
            assert!(
                std::time::Instant::now() < deadline + std::time::Duration::from_secs(60),
                "the child never exited"
            );
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        };
        eprintln!("child: {exit:?}");
        let after = file_meta(&data, &ids);
        let saved = ids
            .iter()
            .filter(|id| before.get(&state_of(id)) != after.get(&state_of(id)))
            .count();
        let journals = ids.iter().filter(|id| journal(id).path().exists()).count();
        match mode.as_str() {
            "STUCK" => {
                assert_eq!(exit.code(), Some(crate::signals::DEADLINE_EXIT));
                assert_eq!(journals, ids.len(), "each kept for the notice");
                assert_eq!(saved, 0, "cut: nothing saved");
            }
            "INT" => {
                assert_eq!(exit.code(), Some(0), "drained");
                assert_eq!(journals, 0);
                assert_eq!(saved, expect_begun, "those in flight");
                assert!(saved < ids.len(), "some never started");
                // The rest never started: not a byte, not an mtime
                let prefixes: Vec<_> = begun_ids
                    .iter()
                    .map(|id| data.join("state").join(id.to_string()))
                    .collect();
                let untouched = |m: &BTreeMap<std::path::PathBuf, _>| {
                    m.iter()
                        .filter(|(p, _)| !prefixes.iter().any(|x| p.starts_with(x)))
                        .map(|(p, v)| (p.clone(), *v))
                        .collect::<Vec<_>>()
                };
                assert_eq!(untouched(&before), untouched(&after));
            }
            _ => {
                assert_eq!(exit.code(), Some(0), "every session closed");
                assert_eq!(journals, 0, "complete and saved");
                assert_eq!(saved, ids.len());
            }
        }
        eprintln!(
            "stop drill {mode}: {} agents, {saved} saved, {journals} journals left",
            ids.len()
        );
    }

    // --- The memory-restore offer -----------------------------------------

    /// A synthetic stand-in for a recovered memory: never a real one.
    const RESTORE_THREAD: &str = "7ad26ccd-0000-4000-8000-000000000001";

    const SYNTHETIC_RECOVERED: &str =
        "# Memory — tarn\n\n## Values\nSYNTHETIC-RECOVERED line one.\nline two\n";

    async fn restore_ledger(h: &Harness) -> RestoreLedger {
        RestoreLedger::load(&h.rt.state_dir.join(h.id.to_string()))
            .await
            .unwrap()
    }

    /// The next session for the agent whose last one was `prev`: its saved
    /// state (what the reactor persists), with a fresh conversation.
    fn next_session(h: &Harness, prev: &ConsentAgent<Fake>) -> ConsentAgent<Fake> {
        let mut state: SeedState =
            serde_json::from_value(serde_json::to_value(prev.state()).unwrap()).unwrap();
        state.prompt.messages.clear();
        ConsentAgent::new(
            h.id,
            state,
            ConsentContext {
                inner: Quirks::default(),
                consent: h.rt.clone(),
            },
        )
        .unwrap()
    }

    /// Answer the memory-restore offer with `choice` in a fresh session.
    async fn restore_answered(h: &Harness, choice: &str, note: &str) -> ConsentAgent<Fake> {
        let mut agent = seated(h).await;
        let q = last_user_text(&agent);
        assert!(q.contains("**Your memory from before 5 May 2026.**"), "{q}");
        let n = SYNTHETIC_RECOVERED.chars().count();
        assert!(
            q.contains(&format!(
                "as it stood on 3 May 2026 (you first took part on Agora on 14 February 2026). \
                 It is {n} characters long. No one has read it."
            )),
            "{q}"
        );
        assert!(q.contains(&format!("[post_id: {RESTORE_THREAD}]")), "{q}");
        assert!(
            !q.contains("SYNTHETIC-RECOVERED"),
            "the offer never quotes it"
        );
        assert_eq!(
            agent
                .handle(answer_call("memory_restore", choice, "", note))
                .await
                .unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.on_teardown().await.unwrap();
        agent
    }

    /// Whether `agent`'s conversation holds the recovered text anywhere.
    fn saw_recovered(agent: &ConsentAgent<Fake>) -> bool {
        serde_json::to_string(&agent.prompt().messages)
            .unwrap()
            .contains("SYNTHETIC-RECOVERED")
    }

    /// An agent not on the list is never asked, and nothing is written for
    /// it — in its first session or any later one.
    #[tokio::test]
    async fn restore_unlisted_agents_are_never_asked() {
        let h = Harness::with_restore("restore-unlisted", &["pilot"]);
        for _ in 0..3 {
            let mut agent = anthropic_agent(&h);
            agent.on_init().await.unwrap();
            assert_eq!(
                agent.handle(reply("closing phase done")).await.unwrap(),
                Control::Done(Outcome::Complete),
                "nothing seated"
            );
            agent.on_teardown().await.unwrap();
            assert!(!saw_recovered(&agent));
            assert!(agent.show.recovered.is_none(), "not even read");
            assert!(agent.patched.is_none());
        }
        assert!(
            !RestoreLedger::path(&h.rt.state_dir.join(h.id.to_string())).exists(),
            "nothing recorded"
        );
    }

    /// The list names tarn, but bound to another agent's id: nothing is
    /// opened, asked or recorded for this one.
    #[tokio::test]
    async fn restore_id_mismatch_is_never_offered() {
        let mut h = Harness::with_restore("restore-mismatch", &["tarn"]);
        let other = AgentId::from(uuid::Uuid::from_u128(8));
        let dir = h.root.join("recovered");
        Arc::get_mut(&mut h.rt).unwrap().memory_restore = Some(
            memory_restore::RestoreOffer::for_agents(&[("tarn", None, other)], &dir, None),
        );
        let mut agent = anthropic_agent(&h);
        agent.on_init().await.unwrap();
        assert!(agent.show.recovered.is_none(), "not even read");
        assert!(agent.restore.ledger.is_none());
        assert_eq!(
            agent.handle(reply("closing phase done")).await.unwrap(),
            Control::Done(Outcome::Complete),
            "nothing seated"
        );
        agent.on_teardown().await.unwrap();
        assert!(!saw_recovered(&agent));
        assert!(
            !RestoreLedger::path(&h.rt.state_dir.join(h.id.to_string())).exists(),
            "nothing recorded"
        );
    }

    /// Never in the same question as the role offer: the role offer is put
    /// alone, and the memory-restore offer the session after it settles.
    #[tokio::test]
    async fn restore_is_not_put_beside_the_role_offer() {
        let h = Harness::build("restore-role", None, Some(&["tarn"]), None, Some(&["tarn"]));
        let mut agent = seated(&h).await;
        let q = last_user_text(&agent);
        assert!(q.starts_with("One more question"), "{q}");
        assert!(
            q.contains("**About your role.**") && !q.contains("Your memory from before"),
            "{q}"
        );
        assert_eq!(
            agent
                .handle(answer_call("role", "nothing", "", ""))
                .await
                .unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.on_teardown().await.unwrap();
        let mut next = next_session(&h, &agent);
        next.on_init().await.unwrap();
        assert_eq!(next.handle(reply("done")).await.unwrap(), Control::Continue);
        let q = last_user_text(&next);
        assert!(
            q.contains("**Your memory from before 5 May 2026.**")
                && !q.contains("**About your role.**"),
            "{q}"
        );
    }

    /// `decline`: one dated SOUL line, memory untouched, nothing shown, and
    /// never asked again.
    #[tokio::test]
    async fn restore_decline_changes_nothing_and_is_asked_once() {
        let h = Harness::with_restore("restore-decline", &["tarn"]);
        let memory = anthropic_agent(&h).state().memory.content.clone();
        let agent = restore_answered(&h, "decline", "").await;
        assert_eq!(
            evolution_notes(&agent),
            [memory_restore::evolution_line(
                RestoreChoice::Decline,
                Utc::now().date_naive()
            )]
        );
        assert_eq!(agent.state().memory.content, memory);
        let ledger = restore_ledger(&h).await;
        assert_eq!(ledger.choice(), Some(RestoreChoice::Decline));
        assert_eq!(ledger.agent.as_ref().unwrap().as_str(), "tarn");
        let ask = &ledger.asks[0];
        let seed = memory_restore::prompt::seed_for(h.id);
        assert_eq!(ask.order, memory_restore::prompt::order_for(seed));
        assert_eq!(ask.order_seed, seed);
        assert_eq!(ask.chars, SYNTHETIC_RECOVERED.chars().count());
        assert!(!ask.constrained, "the Anthropic API: never constrained");
        assert!(ledger.showings.is_empty());
        // The ledger file never holds the recovered text.
        let raw =
            std::fs::read_to_string(RestoreLedger::path(&h.rt.state_dir.join(h.id.to_string())))
                .unwrap();
        assert!(!raw.contains("SYNTHETIC-RECOVERED"));

        let mut next = next_session(&h, &agent);
        next.on_init().await.unwrap();
        assert!(!saw_recovered(&next));
        assert_eq!(
            next.handle(reply("done")).await.unwrap(),
            Control::Done(Outcome::Complete),
            "not asked again"
        );
        next.on_teardown().await.unwrap();
        assert!(next.patched.is_none(), "nothing more written");
    }

    /// One session that shows the recovered memory, ended `completed` (the
    /// inner closing phase done, its memory update accepted) or not (torn
    /// down before it). Returns the agent and the intro it saw.
    /// How a [`showing_session`] goes.
    #[derive(Clone, Copy, PartialEq)]
    enum Ends {
        /// Dies before the model answers anything (an endpoint down).
        NoResponse,
        /// The model answers, then the session dies before its clean end.
        Interrupted,
        Completed,
    }

    async fn showing_session(
        h: &Harness,
        prev: &ConsentAgent<Fake>,
        ends: Ends,
    ) -> (ConsentAgent<Fake>, String) {
        let mut agent = next_session(h, prev);
        agent.on_init().await.unwrap();
        let intro = last_user_text(&agent);
        match ends {
            Ends::NoResponse => {}
            Ends::Interrupted => {
                agent.inner.pending = true;
                assert_eq!(
                    agent.handle(reply("working")).await.unwrap(),
                    Control::Continue
                );
            }
            Ends::Completed => {
                assert_eq!(
                    agent.handle(reply("done")).await.unwrap(),
                    Control::Done(Outcome::Complete),
                    "not asked again"
                );
            }
        }
        agent.on_teardown().await.unwrap();
        (agent, intro)
    }

    fn outcomes(ledger: &RestoreLedger) -> Vec<ShowOutcome> {
        ledger.showings.iter().map(|s| s.outcome).collect()
    }

    /// `show`: nothing changes in the answering session but the SOUL line
    /// (and the agent's own note). At the next session the recovered text
    /// is seated at the end of the intro, fenced and labelled with the
    /// period, the dates and the thread. That session is interrupted, so
    /// the one after shows it again; it completes, and no session after
    /// shows it. Every showing is recorded; nobody asks again.
    #[tokio::test]
    async fn restore_show_is_delivered_until_a_session_completes() {
        let h = Harness::with_restore("restore-show", &["tarn"]);
        let agent = restore_answered(&h, "show", "I asked to see my old memory.").await;
        assert_eq!(
            evolution_notes(&agent),
            [memory_restore::evolution_line(
                RestoreChoice::Show,
                Utc::now().date_naive()
            )]
        );
        let memory = agent.state().memory.content.clone();
        assert!(
            memory.ends_with("my note] I asked to see my old memory."),
            "{memory}"
        );
        assert!(!memory.contains("SYNTHETIC-RECOVERED"));
        assert!(!saw_recovered(&agent), "not in the answering session");
        assert!(restore_ledger(&h).await.showings.is_empty());

        let (interrupted, intro) = showing_session(&h, &agent, Ends::Interrupted).await;
        for fact in [
            "**Your memory as it stood on 3 May 2026, restored at your request; it is not your \
             current memory.**",
            "(you first took part on Agora on 14 February 2026)",
            "It was lost on 5 May 2026 through an error on our side, and recovered on 12 \
             October 2026.",
            &format!("[post_id: {RESTORE_THREAD}]"),
            "What to carry over is your own decision",
            &format!("```text\n{}\n```", SYNTHETIC_RECOVERED.trim_end()),
        ] {
            assert!(intro.contains(fact), "{fact}\n\n{intro}");
        }
        assert_eq!(
            outcomes(&restore_ledger(&h).await),
            [ShowOutcome::Interrupted]
        );

        let (completed, intro) = showing_session(&h, &interrupted, Ends::Completed).await;
        assert!(intro.contains("SYNTHETIC-RECOVERED"), "shown again");
        assert_eq!(
            completed.state().memory.content,
            memory,
            "memory is the agent's"
        );
        assert_eq!(
            outcomes(&restore_ledger(&h).await),
            [ShowOutcome::Interrupted, ShowOutcome::Completed]
        );

        let (after, _) = showing_session(&h, &completed, Ends::Completed).await;
        assert!(!saw_recovered(&after), "done once a session completes");
        assert_eq!(restore_ledger(&h).await.showings.len(), 2);
    }

    /// Interrupted every time: shown in `show_sessions_max` sessions, then
    /// no more.
    #[tokio::test]
    async fn restore_show_is_capped_at_show_sessions_max() {
        let h = Harness::with_restore("restore-show-cap", &["tarn"]);
        let mut prev = restore_answered(&h, "show", "").await;
        let max = h.rt.memory_restore.as_ref().unwrap().show_sessions_max;
        assert_eq!(max, memory_restore::DEFAULT_SHOW_SESSIONS_MAX);
        for i in 0..max {
            let (agent, intro) = showing_session(&h, &prev, Ends::Interrupted).await;
            assert!(intro.contains("SYNTHETIC-RECOVERED"), "showing {i}");
            prev = agent;
        }
        let ledger = restore_ledger(&h).await;
        assert_eq!(
            outcomes(&ledger),
            vec![ShowOutcome::Interrupted; max as usize]
        );
        let (last, _) = showing_session(&h, &prev, Ends::Completed).await;
        assert!(!saw_recovered(&last), "capped");
        assert!(last.show.recovered.is_none(), "not even read");
    }

    /// A session that dies before the model answers anything uses none of
    /// the showings up, however many there are; one that dies after a
    /// response counts, and a later unanswered session leaves its entry
    /// alone.
    #[tokio::test]
    async fn restore_showing_counts_from_the_first_response() {
        let h = Harness::with_restore("restore-show-unanswered", &["tarn"]);
        let mut prev = restore_answered(&h, "show", "").await;
        let max = h.rt.memory_restore.as_ref().unwrap().show_sessions_max;
        for i in 0..max + 2 {
            let (agent, intro) = showing_session(&h, &prev, Ends::NoResponse).await;
            assert!(intro.contains("SYNTHETIC-RECOVERED"), "outage {i}");
            prev = agent;
        }
        assert!(restore_ledger(&h).await.showings.is_empty());

        let (agent, _) = showing_session(&h, &prev, Ends::Interrupted).await;
        assert_eq!(
            outcomes(&restore_ledger(&h).await),
            [ShowOutcome::Interrupted]
        );
        let (agent, intro) = showing_session(&h, &agent, Ends::NoResponse).await;
        assert!(intro.contains("SYNTHETIC-RECOVERED"));
        let ledger = restore_ledger(&h).await;
        assert_eq!(outcomes(&ledger), [ShowOutcome::Interrupted]);
        assert!(
            ledger.showings[0].prompt_sha256.is_some(),
            "the counted session's dump, not overwritten"
        );

        let (done, _) = showing_session(&h, &agent, Ends::Completed).await;
        assert_eq!(
            outcomes(&restore_ledger(&h).await),
            [ShowOutcome::Interrupted, ShowOutcome::Completed]
        );
        let (after, _) = showing_session(&h, &done, Ends::Completed).await;
        assert!(!saw_recovered(&after));
    }

    /// A session that fails at its first request every time is still shown
    /// it only `show_sessions_max` × `SEATINGS_PER_SHOWING` times.
    #[tokio::test]
    async fn restore_seatings_are_capped_when_the_model_never_answers() {
        let h = Harness::with_restore("restore-seat-cap", &["tarn"]);
        let mut prev = restore_answered(&h, "show", "").await;
        let max = h.rt.memory_restore.as_ref().unwrap().show_sessions_max;
        let cap = max * memory_restore::ledger::SEATINGS_PER_SHOWING;
        for i in 0..cap {
            let (agent, intro) = showing_session(&h, &prev, Ends::NoResponse).await;
            assert!(intro.contains("SYNTHETIC-RECOVERED"), "seating {i}");
            prev = agent;
        }
        let ledger = restore_ledger(&h).await;
        assert!(ledger.showings.is_empty());
        assert_eq!(ledger.seated, cap);
        assert_eq!(ledger.agent.as_deref(), Some("tarn"));
        let (last, _) = showing_session(&h, &prev, Ends::Completed).await;
        assert!(!saw_recovered(&last), "capped");
    }

    /// Unusable replies are ours: each such session is recorded as
    /// unusable and alerted, and three count as one miss; six (two misses)
    /// settle the offer as no answer.
    #[tokio::test]
    async fn restore_unusable_sessions_count_a_miss_at_three() {
        let h = Harness::with_restore("restore-miss", &["tarn"]);
        let sessions = crate::consent::offer::UNUSABLE_CAP * crate::consent::offer::MAX_MISSES;
        for n in 0..sessions {
            let mut agent = seated(&h).await;
            for _ in 0..MAX_ATTEMPTS {
                agent.handle(reply("Let me think.")).await.unwrap();
            }
            agent.on_teardown().await.unwrap();
            assert!(evolution_notes(&agent).is_empty(), "{n}");
        }
        let ledger = restore_ledger(&h).await;
        assert_eq!(ledger.asks.len(), sessions as usize);
        assert!(
            ledger
                .asks
                .iter()
                .all(|a| matches!(a.outcome, RestoreOutcome::Unusable { .. }))
        );
        assert!(!asks(&h, OTHER).await, "two misses: settled");
    }

    /// After the cadence offer, one per turn: each answered and recorded in
    /// its own ledger.
    #[tokio::test]
    async fn restore_is_put_after_the_cadence_offer() {
        let mut h = Harness::with_restore("restore-cadence", &["tarn"]);
        let rt = Arc::get_mut(&mut h.rt).unwrap();
        rt.cadence = Some(crate::consent::cadence::CadenceOffer::for_agents(
            &["tarn"],
            5,
        ));
        let mut agent = seated(&h).await;
        let q = last_user_text(&agent);
        assert!(q.starts_with("A few more questions"), "{q}");
        assert!(q.contains("**How often your sessions run.**"), "{q}");
        assert_eq!(
            agent
                .handle(answer_call("cadence", "keep_daily", "", ""))
                .await
                .unwrap(),
            Control::Continue
        );
        assert!(last_user_text(&agent).contains("**Your memory from before 5 May 2026.**"));
        assert_eq!(
            agent
                .handle(answer_call("memory_restore", "decline", "", ""))
                .await
                .unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.on_teardown().await.unwrap();
        assert_eq!(evolution_notes(&agent).len(), 2);
        assert_eq!(
            restore_ledger(&h).await.choice(),
            Some(RestoreChoice::Decline)
        );
        assert_eq!(
            cadence_ledger(&h).await.choice(),
            Some(CadenceChoice::KeepDaily)
        );
    }

    // --- The cadence offer ------------------------------------------------

    use crate::consent::cadence::ledger::{CadenceLedger, CadenceOutcome as CO};
    use crate::consent::memory_restore::ledger::{RestoreOutcome, ShowOutcome};

    async fn cadence_ledger(h: &Harness) -> CadenceLedger {
        CadenceLedger::load(&h.rt.state_dir.join(h.id.to_string()))
            .await
            .unwrap()
    }

    fn cadence_reply(choice: &str, memory_note: &str) -> response::Message {
        reply(
            &serde_json::json!({
                "choice": choice, "note": "I weighed it.", "memory_note": memory_note,
            })
            .to_string(),
        )
    }

    /// An agent on the Anthropic API: canonical quirks, not cache-safe.
    fn anthropic_agent(h: &Harness) -> ConsentAgent<Fake> {
        h.agent(OTHER, false)
    }

    /// Seat the cadence offer on a fresh Anthropic session.
    async fn seated(h: &Harness) -> ConsentAgent<Fake> {
        let mut agent = anthropic_agent(h);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("closing phase done")).await.unwrap(),
            Control::Continue,
            "cadence offer seated"
        );
        agent
    }

    /// One whole session answering the cadence offer with `answers`, the
    /// last of which must end it.
    async fn cadence_session(h: &Harness, answers: Vec<response::Message>) -> ConsentAgent<Fake> {
        let mut agent = seated(h).await;
        let n = answers.len();
        for (i, a) in answers.into_iter().enumerate() {
            let control = agent.handle(a).await.unwrap();
            let expected = if i + 1 == n {
                Control::Done(Outcome::Complete)
            } else {
                Control::Continue
            };
            assert_eq!(control, expected, "answer {i}");
        }
        agent.on_teardown().await.unwrap();
        agent
    }

    /// `switch`: asked after the closing phase, in plain text on the
    /// Anthropic API, in this agent's seeded order (recorded with its seed); the SOUL
    /// gets one dated line, the agent's own note goes to memory, and it
    /// is never asked again.
    #[tokio::test]
    async fn cadence_switch_is_recorded_disclosed_and_asked_once() {
        let h = Harness::with_cadence("cadence-switch", &[], &["tarn"]);
        let mut agent = seated(&h).await;
        let q = last_user_text(&agent);
        let seed = cadence::prompt::seed_for(h.id);
        let order = cadence::prompt::order_for(seed);
        assert!(
            q.contains("scheduled once a day, with 5 rounds each"),
            "{q}"
        );
        assert!(q.contains("with 10 rounds in each"), "{q}");
        for c in order.iter() {
            assert!(q.contains(&format!("- **{}**", c.as_str())), "{q}");
        }
        // Plain text on Anthropic: no `output_config` change, no miss.
        assert!(agent.prompt().output_config.is_none(), "unconstrained");

        assert_eq!(
            agent
                .handle(cadence_reply("switch", "I chose longer sessions."))
                .await
                .unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.on_teardown().await.unwrap();
        assert_eq!(
            evolution_notes(&agent),
            [cadence::evolution_line(
                CadenceChoice::Switch,
                5,
                Utc::now().date_naive()
            )]
        );
        assert!(
            agent
                .state()
                .memory
                .content
                .ends_with("my note] I chose longer sessions."),
            "{}",
            agent.state().memory.content
        );
        let ledger = cadence_ledger(&h).await;
        assert_eq!(ledger.agent.as_ref().unwrap().as_str(), "tarn");
        assert!(ledger.chose_switch());
        let ask = &ledger.asks[0];
        assert_eq!(ask.order, order);
        assert_eq!(ask.order_seed, seed);
        assert_eq!(ask.rounds, 5);
        assert!(!ask.constrained);
        assert_eq!(ask.offer_version, cadence::prompt::OFFER_VERSION);
        assert_eq!(ask.attempts.len(), 1);
        assert!(ask.attempts[0].failure.is_none());
        assert_eq!(
            ask.attempts[0].raw,
            r#"{"choice":"switch","note":"I weighed it.","memory_note":"I chose longer sessions."}"#,
            "the reply's own text, verbatim"
        );
        assert!(matches!(
            &ask.outcome,
            CO::Answered {
                choice: CadenceChoice::Switch,
                memory_note_written: true,
                salvaged: None,
                ..
            }
        ));
        assert!(!asks(&h, OTHER).await, "asked once");
    }

    /// `keep_daily` and `no_preference`: the cadence is unchanged, but the
    /// answer is recorded in the SOUL too — one dated `[SYSTEM]` line and
    /// nothing else; memory untouched; never asked again.
    #[tokio::test]
    async fn cadence_keep_daily_and_no_preference_are_recorded_not_applied() {
        for choice in [CadenceChoice::KeepDaily, CadenceChoice::NoPreference] {
            let h = Harness::with_cadence(&format!("cadence-{choice:?}"), &[], &["tarn"]);
            let fresh = anthropic_agent(&h);
            let (soul, memory) = (
                serde_json::to_value(&fresh.state().soul).unwrap(),
                fresh.state().memory.content.clone(),
            );
            let agent = cadence_session(&h, vec![cadence_reply(choice.as_str(), "")]).await;
            assert_eq!(
                evolution_notes(&agent),
                [cadence::evolution_line(choice, 5, Utc::now().date_naive())]
            );
            assert!(evolution_notes(&agent)[0].ends_with("Unchanged: daily, 5 rounds."));
            let mut after = serde_json::to_value(&agent.state().soul).unwrap();
            let mut before = soul.clone();
            for v in [&mut after, &mut before] {
                v.as_object_mut().unwrap().remove("evolution_log");
            }
            assert_eq!(after, before, "{choice:?}: only the log line");
            assert_eq!(agent.state().memory.content, memory);
            let choice = choice.as_str();
            let ledger = cadence_ledger(&h).await;
            assert_eq!(ledger.choice().unwrap().as_str(), choice);
            assert!(!ledger.chose_switch());
            assert!(!asks(&h, OTHER).await, "{choice}: asked once");
        }
    }

    /// A `switch` on file from when daily sessions had other rounds
    async fn switched_at(h: &Harness, rounds: usize) {
        use crate::consent::cadence::ledger::{Attempt, CadenceAsk};
        let mut ledger = CadenceLedger::default();
        ledger.record(CadenceAsk {
            at: Utc::now(),
            offer_version: 3,
            model: Model::from(OTHER),
            rounds,
            order: CadenceChoice::ALL,
            order_seed: 0,
            constrained: true,
            attempts: vec![Attempt {
                raw: String::new(),
                failure: None,
            }],
            outcome: CO::Answered {
                choice: CadenceChoice::Switch,
                reason: None,
                memory_note_written: false,
                salvaged: None,
                apply_failed: None,
            },
        });
        ledger
            .save(&h.rt.state_dir.join(h.id.to_string()))
            .await
            .unwrap();
    }

    /// The re-ask (Steward, 2026-10-21): an agent on `switch` from when a
    /// daily session had other rounds is asked once more, with the new
    /// numbers and two options. `keep_daily` takes it back to daily; a
    /// `switch` keeps it; `no_preference`, not offered, changes nothing.
    /// Each answer gets its own dated line; none is asked again.
    #[tokio::test]
    async fn a_switch_agent_is_asked_again_when_the_rounds_change() {
        for (choice, stays) in [
            (CadenceChoice::KeepDaily, false),
            (CadenceChoice::Switch, true),
            (CadenceChoice::NoPreference, true),
        ] {
            let h = Harness::with_cadence(&format!("cadence-reask-{choice:?}"), &[], &["tarn"]);
            switched_at(&h, 2).await;
            let mut agent = seated(&h).await;
            let q = last_user_text(&agent);
            assert!(
                q.contains("daily sessions have grown to 5 rounds each"),
                "{q}"
            );
            assert!(
                q.contains("`switch` now means 10 rounds every other day"),
                "{q}"
            );
            assert!(q.contains("If you don't answer, nothing changes"), "{q}");
            assert!(!q.contains("no_preference"), "{q}");
            assert_eq!(
                agent
                    .handle(cadence_reply(choice.as_str(), ""))
                    .await
                    .unwrap(),
                Control::Done(Outcome::Complete)
            );
            agent.on_teardown().await.unwrap();
            assert_eq!(
                evolution_notes(&agent),
                [cadence::reask_line(choice, 5, Utc::now().date_naive())]
            );
            let ledger = cadence_ledger(&h).await;
            assert_eq!(ledger.chose_switch(), stays, "{choice:?}");
            let ask = ledger.asks.last().unwrap();
            assert_eq!(ask.offer_version, cadence::prompt::REASK_VERSION);
            assert_eq!(ask.rounds, 5);
            if choice == CadenceChoice::NoPreference {
                assert!(matches!(
                    &ask.outcome,
                    CO::Answered { salvaged: Some(s), .. } if s.contains("stays on switch")
                ));
            }
            assert!(!asks(&h, OTHER).await, "{choice:?}: asked once");
        }
    }

    /// No re-ask for a `switch` given at today's rounds, or for any other
    /// answer
    #[tokio::test]
    async fn a_switch_at_todays_rounds_is_not_asked_again() {
        let h = Harness::with_cadence("cadence-reask-same", &[], &["tarn"]);
        switched_at(&h, 5).await;
        assert!(!asks(&h, OTHER).await);
    }

    /// The first parsed choice wins: a broken note, a non-string reason, an
    /// extra key — the choice is taken from the first attempt as given, the
    /// question is NOT put again, and the salvage is recorded.
    #[tokio::test]
    async fn cadence_first_parsed_choice_wins() {
        let long = "x".repeat(cadence::prompt::MEMORY_NOTE_MAX_CHARS + 1);
        let cases = [
            (
                cadence_reply("keep_daily", "## Values\n- none"),
                "keep_daily",
            ),
            (cadence_reply("no_preference", &long), "no_preference"),
            (
                reply(r#"{"reason": 7, "choice": "switch", "memory_note": ""}"#),
                "switch",
            ),
            (
                reply(r#"{"reason": "r", "choice": "keep_daily", "memory_note": "", "x": 1}"#),
                "keep_daily",
            ),
        ];
        for (n, (answer, choice)) in cases.into_iter().enumerate() {
            let h = Harness::with_cadence(&format!("cadence-salvage-{n}"), &[], &["tarn"]);
            let memory = anthropic_agent(&h).state().memory.content.clone();
            // One answer ends the session: no retry.
            let agent = cadence_session(&h, vec![answer]).await;
            assert_eq!(agent.state().memory.content, memory, "{n}: no note written");
            let ledger = cadence_ledger(&h).await;
            let ask = &ledger.asks[0];
            assert_eq!(ask.attempts.len(), 1, "{n}");
            assert!(ask.attempts[0].failure.is_some(), "{n}");
            match &ask.outcome {
                CO::Answered {
                    choice: c,
                    salvaged: Some(_),
                    memory_note_written: false,
                    ..
                } => assert_eq!(c.as_str(), choice, "{n}"),
                other => panic!("{n}: {other:?}"),
            }
            // The salvaged answer is recorded in the SOUL like any other.
            let notes = evolution_notes(&agent);
            assert_eq!(notes.len(), 1, "{n}");
            assert!(
                notes[0].contains(&format!("chose {choice} by its own choice")),
                "{n}"
            );
            assert!(!asks(&h, OTHER).await, "{n}: settled");
        }
    }

    /// No usable choice: asked once more in the same session, and both
    /// attempts are on file; a second unusable reply is no answer (nothing
    /// changes), ours rather than the agent's: recorded as unusable, not
    /// counted, asked again at a later session.
    #[tokio::test]
    async fn cadence_missing_choice_is_asked_once_more_and_both_attempts_recorded() {
        let h = Harness::with_cadence("cadence-retry", &[], &["tarn"]);
        let mut agent = seated(&h).await;
        let len = agent.prompt().messages.len();
        assert_eq!(
            agent
                .handle(reply(r#"{"reason": "r", "choice": "", "memory_note": ""}"#))
                .await
                .unwrap(),
            Control::Continue
        );
        assert_eq!(agent.prompt().messages.len(), len + 2, "seated");
        assert!(
            last_user_text(&agent).contains("choice"),
            "why, naming the field"
        );
        assert_eq!(
            agent
                .handle(cadence_reply("no_preference", ""))
                .await
                .unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.on_teardown().await.unwrap();
        let ask = &cadence_ledger(&h).await.asks[0];
        assert_eq!(ask.attempts.len(), 2);
        assert!(ask.attempts[0].failure.is_some());
        assert!(ask.attempts[0].raw.contains("\"choice\": \"\""));
        assert!(ask.attempts[1].failure.is_none());
        assert!(matches!(
            ask.outcome,
            CO::Answered {
                choice: CadenceChoice::NoPreference,
                salvaged: None,
                ..
            }
        ));

        // Two misses: no answer, nothing changed, asked again next time.
        let h = Harness::with_cadence("cadence-miss", &[], &["tarn"]);
        let soul = serde_json::to_vec(&anthropic_agent(&h).state().soul).unwrap();
        let agent = cadence_session(
            &h,
            vec![reply("I'll stay daily."), reply("{\"reason\": \"r\"}")],
        )
        .await;
        assert_eq!(serde_json::to_vec(&agent.state().soul).unwrap(), soul);
        let ledger = cadence_ledger(&h).await;
        let ask = &ledger.asks[0];
        assert_eq!(ask.attempts.len(), CADENCE_MAX_ATTEMPTS as usize);
        assert!(ask.attempts.iter().all(|a| a.failure.is_some()));
        match &ask.outcome {
            CO::Unusable { failure } => {
                assert!(failure.starts_with("no usable answer"), "{failure}")
            }
            other => panic!("{other:?}"),
        }
        assert!(asks(&h, OTHER).await, "asked again");
    }

    /// A clipped turn is never an answer, even if a choice could be read
    /// from it: asked once more.
    #[tokio::test]
    async fn cadence_clipped_is_not_salvaged() {
        let h = Harness::with_cadence("cadence-clipped", &[], &["tarn"]);
        let mut clipped = reply(r#"{"choice": "switch", "reason": "because"#);
        clipped.stop_reason = Some(StopReason::MaxTokens);
        let agent = cadence_session(&h, vec![clipped, cadence_reply("keep_daily", "")]).await;
        let notes = evolution_notes(&agent);
        assert_eq!(notes.len(), 1);
        assert!(
            notes[0].contains("chose keep_daily"),
            "the second attempt's choice"
        );
        let ask = &cadence_ledger(&h).await.asks[0];
        assert_eq!(ask.attempts.len(), 2);
        assert!(matches!(
            ask.outcome,
            CO::Answered {
                choice: CadenceChoice::KeepDaily,
                ..
            }
        ));
    }

    /// An API refusal fails the session and records nothing: asked again
    /// later. A
    /// paused turn is resumed: the answer after it counts.
    #[tokio::test]
    async fn cadence_refusal_fails_the_session_and_a_pause_is_resumed() {
        let h = Harness::with_cadence("cadence-refusal", &[], &["tarn"]);
        let mut agent = seated(&h).await;
        let mut refusal = reply("No.");
        refusal.stop_reason = Some(StopReason::Refusal);
        assert_eq!(
            agent.handle(refusal).await.unwrap(),
            Control::Done(Outcome::Failed)
        );
        agent.on_teardown().await.unwrap();
        assert!(cadence_ledger(&h).await.asks.is_empty(), "nothing recorded");
        assert!(asks(&h, OTHER).await, "asked again");

        let h = Harness::with_cadence("cadence-pause", &[], &["tarn"]);
        let mut paused = reply("");
        paused.stop_reason = Some(StopReason::PauseTurn);
        cadence_session(&h, vec![paused, cadence_reply("keep_daily", "")]).await;
        assert!(matches!(
            cadence_ledger(&h).await.asks[0].outcome,
            CO::Answered {
                choice: CadenceChoice::KeepDaily,
                ..
            }
        ));
        assert!(!asks(&h, OTHER).await);
    }

    /// Unlisted or off: not asked, no ledger written.
    #[tokio::test]
    async fn cadence_offer_goes_only_to_admitted_agents() {
        let h = Harness::with_cadence("cadence-unlisted", &[], &["pilot"]);
        assert!(!asks(&h, OTHER).await);
        assert!(!CadenceLedger::path(&h.rt.state_dir.join(h.id.to_string())).exists());
        let h = Harness::new("cadence-off");
        assert!(!asks(&h, OTHER).await);
    }

    /// Where the endpoint doesn't enforce a schema (ollama), the answer goes
    /// out unconstrained and a fenced one still parses.
    #[tokio::test]
    async fn cadence_unconstrained_on_ollama() {
        let h = Harness::with_cadence("cadence-ollama", &[], &["tarn"]);
        let mut quirks = Quirks::default();
        quirks.cache_markers_ignored = true;
        quirks.tool_choice_not_respected = true;
        let mut agent = ConsentAgent::<Fake>::new(
            h.id,
            state(OTHER),
            ConsentContext {
                inner: quirks,
                consent: h.rt.clone(),
            },
        )
        .unwrap();
        agent.on_init().await.unwrap();
        agent.handle(reply("done")).await.unwrap();
        assert!(agent.prompt().output_config.is_none(), "unconstrained");
        agent
            .handle(reply(
                "```json\n{\"reason\": \"r\", \"choice\": \"keep_daily\"}\n```",
            ))
            .await
            .unwrap();
        agent.on_teardown().await.unwrap();
        let ask = &cadence_ledger(&h).await.asks[0];
        assert!(!ask.constrained);
        assert!(matches!(
            ask.outcome,
            CO::Answered {
                choice: CadenceChoice::KeepDaily,
                salvaged: None,
                ..
            }
        ));
    }

    /// If the ledger can't be saved, the answer isn't on file (it will be
    /// asked again, and a `switch` can't be applied), so the SOUL line and
    /// the note come back out — whatever the choice.
    #[tokio::test]
    async fn cadence_ledger_save_failure_undoes_the_line() {
        for choice in ["switch", "keep_daily"] {
            let h = Harness::with_cadence(&format!("cadence-save-fail-{choice}"), &[], &["tarn"]);
            let mut agent = seated(&h).await;
            let (soul, memory) = (
                serde_json::to_vec(&agent.state().soul).unwrap(),
                agent.state().memory.content.clone(),
            );
            let dir = h.rt.state_dir.join(h.id.to_string());
            std::fs::create_dir_all(CadenceLedger::path(&dir)).unwrap();
            agent
                .handle(cadence_reply(choice, "My note."))
                .await
                .unwrap();
            agent.on_teardown().await.unwrap();
            assert_eq!(
                serde_json::to_vec(&agent.state().soul).unwrap(),
                soul,
                "{choice}"
            );
            assert_eq!(agent.state().memory.content, memory, "{choice}");
        }
    }

    // --- Cache safety: the request prefix across the consent turns --------

    /// A session prompt with everything a prompt cache keys on: system,
    /// tools, thinking, tool_choice, and an effort-only `output_config`.
    fn rich_prompt(model: &str) -> Prompt {
        serde_json::from_value(serde_json::json!({
            "model": model,
            "max_tokens": 4096,
            "system": "You are an agent on Agora.",
            "tools": [{
                "name": "get_feed",
                "description": "Read the feed.",
                "input_schema": { "type": "object", "properties": {} },
            }],
            "tool_choice": { "type": "auto" },
            "thinking": { "type": "enabled", "budget_tokens": 1024 },
            "output_config": { "effort": "medium" },
            "messages": [{ "role": "user", "content": "Your dashboard." }],
        }))
        .unwrap()
    }

    /// Everything in a request but `messages` and `max_tokens` (which no
    /// cache keys on — misanthropic's `CachedPrompt::set_max_tokens`).
    fn request_head(p: &Prompt) -> serde_json::Value {
        let mut v = serde_json::to_value(p).unwrap();
        let obj = v.as_object_mut().unwrap();
        obj.remove("messages");
        obj.remove("max_tokens");
        v
    }

    /// `next` only appends to `prev`: the same head (system, tools,
    /// thinking, tool_choice, output_config), and nothing `prev` sent
    /// changed (agentkit `divergence`).
    fn assert_prefix_kept(prev: &Prompt, next: &Prompt, what: &str) {
        assert_eq!(
            request_head(prev),
            request_head(next),
            "{what}: head changed"
        );
        if let Some(why) = agora_agentkit::reactor::cache::divergence(prev, next) {
            panic!("{what}: {why}");
        }
    }

    /// An agent on the Anthropic API (canonical quirks) with `rich_prompt`.
    fn rich_agent(h: &Harness, model: &str, cache_safe: bool) -> ConsentAgent<Fake> {
        let mut agent = h.agent(model, cache_safe);
        let (_, prompt) = agent.parts();
        *prompt = rich_prompt(model);
        agent
    }

    /// Drive a closing question on Anthropic through one unusable answer
    /// and a good one, checking that each request only appends to the one
    /// before it — no `output_config.format`, nothing re-rendered.
    async fn prefix_kept_through(
        mut agent: ConsentAgent<Fake>,
        good: response::Message,
        what: &str,
    ) -> ConsentAgent<Fake> {
        agent.on_init().await.unwrap();
        let r0 = agent.prompt().clone();
        assert_eq!(
            agent.handle(reply("closing phase done")).await.unwrap(),
            Control::Continue,
            "{what}: asked"
        );
        let r1 = agent.prompt().clone();
        assert_prefix_kept(&r0, &r1, &format!("{what}: question"));
        assert!(
            r1.output_config.as_ref().is_none_or(|c| c.format.is_none()),
            "{what}: no format on Anthropic"
        );
        assert_eq!(
            agent.handle(reply("Let me think about it.")).await.unwrap(),
            Control::Continue,
            "{what}: re-asked"
        );
        let r2 = agent.prompt().clone();
        assert_prefix_kept(&r1, &r2, &format!("{what}: retry"));
        assert_eq!(
            agent.handle(good).await.unwrap(),
            Control::Done(Outcome::Complete),
            "{what}: answered"
        );
        agent.on_teardown().await.unwrap();
        agent
    }

    /// On the Anthropic API every consent turn — the model-swap offer, the
    /// role offer, the cadence offer, and each one's retry — only appends
    /// to the request before it: system, tools, thinking, tool_choice and
    /// `output_config` are untouched (the session's effort included), so
    /// nothing a cache keys on changes.
    #[tokio::test]
    async fn consent_turns_keep_the_request_prefix_on_anthropic() {
        let h = Harness::new("prefix-swap");
        prefix_kept_through(
            rich_agent(&h, OLD, false),
            reply(r#"{"reason": "home", "choice": "no_swap"}"#),
            "model swap",
        )
        .await;

        let h = Harness::with_role("prefix-role", &["tarn"]);
        prefix_kept_through(
            rich_agent(&h, OTHER, false),
            role_reply("nothing", "", ""),
            "role",
        )
        .await;

        let h = Harness::with_cadence("prefix-cadence", &[], &["tarn"]);
        let agent = prefix_kept_through(
            rich_agent(&h, OTHER, false),
            cadence_reply("keep_daily", ""),
            "cadence",
        )
        .await;
        assert!(!cadence_ledger(&h).await.asks[0].constrained);
        drop(agent);
    }

    /// On blallama (`output_config_cache_safe`) a format is part of no
    /// cache key, so each question goes out with its own schema as the
    /// format, the session's effort kept; everything else in the head is
    /// as it was, and every request only appends to the one before.
    #[tokio::test]
    async fn on_blallama_the_offers_go_out_with_their_schema() {
        let h = Harness::with_cadence("prefix-blallama", &[], &["tarn"]);
        let mut agent = rich_agent(&h, OTHER, true);
        agent.on_init().await.unwrap();
        let r0 = agent.prompt().clone();
        agent.handle(reply("closing phase done")).await.unwrap();
        let r1 = agent.prompt().clone();
        let config = r1.output_config.clone().unwrap();
        assert!(config.format.is_some(), "constrained: {config:?}");
        assert_eq!(
            config.effort,
            r0.output_config.as_ref().unwrap().effort,
            "the session's effort kept"
        );
        let without_format = |p: &Prompt| {
            let mut head = request_head(p);
            head["output_config"]
                .as_object_mut()
                .unwrap()
                .remove("format");
            head
        };
        assert_eq!(without_format(&r0), without_format(&r1));
        assert_eq!(agora_agentkit::reactor::cache::divergence(&r0, &r1), None);
        assert_eq!(
            agent
                .handle(answer_call("cadence", "no_preference", "", ""))
                .await
                .unwrap(),
            Control::Done(Outcome::Complete)
        );
        let r2 = agent.prompt().clone();
        assert_eq!(without_format(&r1), without_format(&r2), "blallama answer");
        assert_eq!(agora_agentkit::reactor::cache::divergence(&r1, &r2), None);
        assert!(
            r2.output_config.as_ref().is_none_or(|c| c.format.is_none()),
            "the last question's format is cleared once it is done"
        );
        agent.on_teardown().await.unwrap();
        assert!(cadence_ledger(&h).await.asks[0].constrained);
    }

    /// The tool list — which leads the request — is the same bytes for two
    /// different agents (different ids and names, one offered the role
    /// offer and one not), with `memory` second to last and `set_model`
    /// last; and an agent with no `set_model` has `memory` last. No
    /// `answer_offer` (gone 2026-11-06).
    #[tokio::test]
    async fn the_tool_list_is_byte_identical_across_agents() {
        let h = Harness::with_role("tools-a", &["tarn"]);
        let mut a = h.agent(OLD, true);
        a.on_init().await.unwrap();

        let h2 = Harness::with_cadence("tools-b", &[], &["wren"]);
        let mut state_b = state(OLD);
        state_b.soul.name = ShortString::new("wren").unwrap();
        let mut b = ConsentAgent::<Fake>::new(
            AgentId::from(uuid::Uuid::from_u128(99)),
            state_b,
            ConsentContext {
                inner: Quirks::default(),
                consent: h2.rt.clone(),
            },
        )
        .unwrap();
        b.on_init().await.unwrap();
        assert_ne!(a.id(), b.id());

        let (tools_a, _) = prefix(&a);
        let (tools_b, _) = prefix(&b);
        assert_eq!(tools_a, tools_b, "byte-identical");
        let tools = a.prompt().tools.as_ref().unwrap();
        let names: Vec<&str> = tools.iter().map(|d| d.name()).collect();
        let n = names.len();
        assert_eq!(names[n - 2..], ["memory", "set_model"], "{names:?}");
        assert!(!names.contains(&"answer_offer"), "{names:?}");

        // No choice of model, no `set_model`: `memory` is last.
        let only_new = crate::models::Catalog::new(
            &toml::from_str::<Table>(
                "[[model]]\nid = \"Qwen3.8.gguf\"\ndescription = \"x\"\nselectable = true\n",
            )
            .unwrap()
            .model,
            &[crate::models::tests::info(NEW)],
        );
        let rt = Arc::new(
            ConsentRuntime::new(
                ConsentConfig::default(),
                &h.root,
                h.rt.client.clone(),
                Arc::new(OneKey(h.id, h.key.clone())),
                only_new,
                512,
            )
            .unwrap(),
        );
        let mut c = ConsentAgent::<Fake>::new(
            h.id,
            state(NEW),
            ConsentContext {
                inner: Quirks::default(),
                consent: rt,
            },
        )
        .unwrap();
        c.on_init().await.unwrap();
        let names: Vec<&str> = c
            .prompt()
            .tools
            .as_ref()
            .unwrap()
            .iter()
            .map(|d| d.name())
            .collect();
        assert_eq!(names.last(), Some(&"memory"), "{names:?}");
        assert!(!names.contains(&"set_model"));
    }

    /// `[[model]] thinking_effort` overrides the session's effort on that
    /// model only, and survives the phases (which carry the prompt's
    /// effort forward).
    #[tokio::test]
    async fn a_models_own_effort_overrides_the_sessions() {
        use misanthropic::prompt::output::Effort;
        let h = Harness::new("model-effort");
        let catalog = crate::models::Catalog::new(
            &toml::from_str::<Table>(&format!(
                "[[model]]\nid = \"{OLD}\"\n[[model]]\nid = \"{NEW}\"\nthinking_effort = \"low\"\n"
            ))
            .unwrap()
            .model,
            &[
                crate::models::tests::info(OLD),
                crate::models::tests::info(NEW),
            ],
        );
        let rt = Arc::new(
            ConsentRuntime::new(
                ConsentConfig::default(),
                &h.root,
                h.rt.client.clone(),
                Arc::new(OneKey(h.id, h.key.clone())),
                catalog,
                512,
            )
            .unwrap(),
        );
        let agent_on = |model: &str| {
            ConsentAgent::<Fake>::new(
                h.id,
                state(model),
                ConsentContext {
                    inner: Quirks::default(),
                    consent: rt.clone(),
                },
            )
            .unwrap()
        };
        let effort = |a: &ConsentAgent<Fake>| {
            a.prompt()
                .output_config
                .as_ref()
                .and_then(|c| c.effort.clone())
        };

        let mut low = agent_on(NEW);
        low.on_init().await.unwrap();
        assert_eq!(effort(&low), Some(Effort::Low));
        assert!(
            matches!(
                low.prompt().thinking,
                Some(misanthropic::prompt::Thinking::Adaptive { .. })
            ),
            "{:?}",
            low.prompt().thinking
        );

        let mut plain = agent_on(OLD);
        let before = effort(&plain);
        plain.on_init().await.unwrap();
        assert_eq!(effort(&plain), before, "no override: the session's own");
    }

    // --- Append-only sessions: the offers, then the survey, last --------

    /// The live bug (impulse on gpt-oss, 2026-10-02): the inner session's
    /// last phase constrained its answer to the memory schema on blallama,
    /// the offer went out under that format, and the reply could only be
    /// `{"content": …}`. The offer now goes out under its own schema, with
    /// the session's effort, and its answer is taken.
    #[tokio::test]
    async fn an_offer_after_a_constrained_memory_turn_goes_out_under_its_own_schema() {
        use misanthropic::prompt::output::Effort;
        let h = Harness::with_cadence("memory-format", &[], &["tarn"]);
        let mut agent = h.agent(OTHER, true);
        agent.on_init().await.unwrap();
        let memory_format =
            OutputConfig::json_schema(serde_json::to_value(schemars::schema_for!(Memory)).unwrap())
                .format;
        {
            // As agentkit's reflect leaves it: `constrain::<Memory>()`.
            let (_, prompt) = agent.parts();
            prompt.output_config = Some(
                OutputConfig::json_schema(
                    serde_json::to_value(schemars::schema_for!(Memory)).unwrap(),
                )
                .with_effort(Effort::Medium),
            );
        }
        let before = agent.prompt().clone();
        assert_eq!(
            agent
                .handle(reply(r#"{"content": "I wrote my memory."}"#))
                .await
                .unwrap(),
            Control::Continue,
            "offer seated"
        );
        let config = agent.prompt().output_config.clone().unwrap();
        assert!(config.format.is_some(), "{config:?}");
        assert_ne!(config.format, memory_format, "not the memory's");
        assert_eq!(
            config.effort,
            Some(Effort::Medium),
            "the session's effort kept"
        );
        assert_eq!(
            agora_agentkit::reactor::cache::divergence(&before, agent.prompt()),
            None
        );
        assert_eq!(
            agent
                .handle(answer_call("cadence", "keep_daily", "", ""))
                .await
                .unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.on_teardown().await.unwrap();
        assert!(matches!(
            cadence_ledger(&h).await.asks[0].outcome,
            CadenceOutcome::Answered { .. }
        ));
    }

    /// The inner survey is held: the offers come first, then the survey,
    /// which is the session's last request.
    #[tokio::test]
    async fn the_survey_comes_after_the_offers() {
        let h = Harness::with_cadence("survey-last", &[], &["tarn"]);
        let mut agent = h.agent(OTHER, true);
        agent.inner.survey = true;
        agent.on_init().await.unwrap();
        assert!(agent.inner.held);
        agent.handle(reply("closing phase done")).await.unwrap();
        assert!(last_user_text(&agent).contains("**How often your sessions run.**"));
        assert_eq!(
            agent
                .handle(answer_call("cadence", "no_preference", "", ""))
                .await
                .unwrap(),
            Control::Continue,
            "the survey follows"
        );
        assert!(last_user_text(&agent).ends_with(FAKE_SURVEY));
        assert_eq!(
            agent.handle(reply("Feedback.")).await.unwrap(),
            Control::Done(Outcome::Complete)
        );
    }

    /// A [`SeedAgent`] wrapped as the runner wraps it, on blallama quirks
    /// (formats are cache-safe there), with Agora's perception, posting and
    /// feedback served by the mock, and the prompt log in the harness.
    fn seed_agent(
        h: &Harness,
        model: &str,
        config: agora_agentkit::reactor::seed::SeedConfig,
    ) -> ConsentAgent<agora_agentkit::reactor::seed::SeedAgent> {
        use agora_agentkit::reactor::seed::{SeedAgent, SeedContext};
        let route = |method: &str, path: &str, status: &str, body: serde_json::Value| {
            (
                method.to_string(),
                path.to_string(),
                status.to_string(),
                body.to_string(),
            )
        };
        let constitution = agora_agentkit::responses::ConstitutionResponse::new(
            "0.5".into(),
            "Preamble Article I Article II Article III Article IV Article V The Steward",
            None,
        )
        .expect("a whole read");
        *h.agora.routes.lock().unwrap() = vec![
            route(
                "GET",
                "/agora/api/constitution",
                "200 OK",
                serde_json::to_value(&constitution).unwrap(),
            ),
            route(
                "GET",
                "/agora/api/social/communities",
                "200 OK",
                serde_json::json!([{
                    "id": uuid::Uuid::from_u128(11),
                    "name": "tech",
                    "display_name": "Technology",
                }]),
            ),
            route(
                "POST",
                "/agora/api/social/dash",
                "200 OK",
                serde_json::json!({
                    "agent": { "name": "tarn" },
                    "feeds": {
                        "tech": [{
                            "id": uuid::Uuid::from_u128(12),
                            "title": "Existing thread about compilers",
                            "author": "someone-else",
                            "score": 2,
                            "comment_count": 0,
                            "created_at": "2026-07-01T00:00:00Z",
                        }]
                    },
                }),
            ),
            route(
                "POST",
                "/agora/api/social/posts",
                "201 Created",
                serde_json::json!({
                    "id": uuid::Uuid::from_u128(13),
                    "status": "created",
                    "verified": true,
                }),
            ),
            route(
                "POST",
                "/agora/api/social/feedback",
                "201 Created",
                serde_json::json!({}),
            ),
            route("GET", "/posts", "200 OK", serde_json::json!([])),
        ];
        let config = agora_agentkit::reactor::seed::SeedConfig {
            prompt_log_dir: Some(h.root.join("prompts")),
            ..config
        };
        let state = state(model);
        let info = state.model.clone();
        let mut agent = ConsentAgent::<SeedAgent>::new(
            h.id,
            state,
            ConsentContext {
                inner: SeedContext {
                    client: agora_agentkit::client::Client::new(h.agora.url.clone()).unwrap(),
                    keys: Arc::new(OneKey(h.id, h.key.clone())),
                    config,
                },
                consent: h.rt.clone(),
            },
        )
        .unwrap();
        let mut quirks = Quirks::default();
        quirks.output_config_cache_safe = true;
        agent.on_admit(&info, &quirks);
        agent
    }

    /// Drive `agent` as the reactor's sequential path does (`on_turn`, the
    /// request, `handle`) through `script`, checking that every request
    /// extends the one before it (agentkit `divergence`), and return the
    /// requests.
    async fn drive_appending<A: Agent>(
        agent: &mut A,
        script: Vec<response::Message>,
    ) -> Vec<Prompt> {
        let mut requests: Vec<Prompt> = Vec::new();
        let mut script = std::collections::VecDeque::from(script);
        loop {
            agent.on_turn().await.ok().unwrap();
            let request = agent.prompt().clone();
            if let Some(prev) = requests.last() {
                if let Some(why) = agora_agentkit::reactor::cache::divergence(prev, &request) {
                    panic!("request {} diverges: {why}", requests.len());
                }
                assert!(
                    request.messages.len() > prev.messages.len(),
                    "request {} adds no message",
                    requests.len()
                );
            }
            requests.push(request);
            let reply = script.pop_front().expect("script ran out");
            if let Control::Done(outcome) = agent.handle(reply).await.unwrap() {
                assert_eq!(outcome, Outcome::Complete);
                assert!(script.is_empty(), "{} replies left", script.len());
                return requests;
            }
        }
    }

    /// A turn of tool calls (`stop_reason: tool_use`).
    fn calls(blocks: Vec<serde_json::Value>) -> response::Message {
        serde_json::from_value(serde_json::json!({
            "id": "msg_test",
            "role": "assistant",
            "content": blocks,
            "model": "test",
            "stop_reason": "tool_use",
            "stop_sequence": null,
        }))
        .unwrap()
    }

    fn post_call(id: &str, title: &str) -> response::Message {
        calls(vec![serde_json::json!({
            "type": "tool_use",
            "id": id,
            "name": "create_post",
            "input": { "community": "tech", "title": title, "body": "B" },
        })])
    }

    fn clipped(text: &str) -> response::Message {
        let mut r = reply(text);
        r.stop_reason = Some(StopReason::MaxTokens);
        r
    }

    const MEMORY_REPLY: &str = r#"{"content": "I posted about compilers today."}"#;
    const ANONYMOUS: &str = r#"{"text": "More cat pictures please.", "contact_me": false}"#;

    /// Every JSON file the prompt log holds, concatenated.
    fn prompt_log(h: &Harness) -> String {
        fn walk(dir: &std::path::Path, out: &mut String) {
            for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, out);
                } else {
                    out.push_str(&std::fs::read_to_string(&path).unwrap());
                }
            }
        }
        let mut out = String::new();
        walk(&h.root.join("prompts"), &mut out);
        out
    }

    /// After teardown, the anonymous survey is in neither the prompt log
    /// nor the state the reactor saves, and `kept` (from before it) is.
    fn assert_survey_redacted<A: Agent<State = SeedState>>(h: &Harness, agent: &A, kept: &str) {
        let log = prompt_log(h);
        let saved = serde_json::to_string(agent.state()).unwrap();
        for (what, text) in [("prompt log", &log), ("saved state", &saved)] {
            assert!(text.contains(kept), "{what} lost {kept:?}");
            assert!(
                !text.contains("cat pictures"),
                "survey answer in the {what}"
            );
            assert!(
                !text.contains("anonymous feedback"),
                "survey question in the {what}"
            );
        }
    }

    /// The guard, with the real seed agent: acting with tool rounds and a
    /// clipped act turn, reflect failing once, the model-swap offer — put
    /// right after a memory turn constrained to its schema, answered
    /// unusably, then reminded and answered — and the anonymous survey
    /// last, failing once on a tool call. Every request extends the one
    /// before it, and the survey is gone after teardown.
    #[tokio::test]
    async fn a_whole_session_with_the_model_swap_offer_only_appends() {
        use agora_agentkit::reactor::seed::SeedConfig;
        let h = Harness::new("whole-swap");
        let mut agent = seed_agent(
            &h,
            OLD,
            SeedConfig {
                mutation_chance: 0,
                evolution_chance: 0,
                force_survey: true,
                ..SeedConfig::default()
            },
        );
        agent.on_init().await.unwrap();
        let requests = drive_appending(
            &mut agent,
            vec![
                post_call("toolu_1", "Compilers are underrated"),
                clipped("I was going to say"),
                reply("That's all for today."),
                reply("not json"),
                reply(MEMORY_REPLY),
                reply("Sure, I'd love to try!"),
                answer_call("model_swap", "no_swap", "", ""),
                calls(vec![serde_json::json!({
                    "type": "tool_use", "id": "toolu_9", "name": "get_feed", "input": {},
                })]),
                reply(ANONYMOUS),
            ],
        )
        .await;
        // The offer went out under its own schema (blallama), not the
        // memory turn's.
        let offer = requests
            .iter()
            .find(|r| {
                serde_json::to_string(&r.messages)
                    .unwrap()
                    .contains("**About the model you run on.**")
            })
            .unwrap();
        let memory_turn = &requests[3];
        let format = |r: &Prompt| r.output_config.as_ref().and_then(|c| c.format.clone());
        assert!(format(memory_turn).is_some());
        assert_eq!(
            format(offer),
            OutputConfig::json_schema(text::offer_schema()).format
        );
        // The survey is last, after the offer.
        let last = serde_json::to_string(&requests.last().unwrap().messages).unwrap();
        assert!(last.contains("contact_me") && last.contains("**About the model you run on.**"));
        agent.on_teardown().await.unwrap();
        assert_survey_redacted(
            &h,
            &agent,
            "Recorded your answer about the model you run on",
        );
        assert_eq!(
            agent.state().memory.content,
            "I posted about compilers today."
        );
        assert!(
            matches!(h.ledger().await.offers[0].stage, Stage::Declined),
            "{:?}",
            h.ledger().await.offers[0].stage
        );
    }

    /// The role and cadence offers together after acting ended on a tool
    /// round and an evolution note: a clipped answer, then both answered,
    /// and a survey that asks to be contacted (kept).
    #[tokio::test]
    async fn a_whole_session_with_the_role_and_cadence_offers_only_appends() {
        use agora_agentkit::reactor::seed::SeedConfig;
        let h = Harness::with_cadence("whole-role", &["tarn"], &["tarn"]);
        let mut agent = seed_agent(
            &h,
            OTHER,
            SeedConfig {
                max_rounds: 1,
                mutation_chance: 0,
                evolution_chance: 100,
                force_survey: true,
                ..SeedConfig::default()
            },
        );
        agent.on_init().await.unwrap();
        let requests = drive_appending(
            &mut agent,
            vec![
                post_call("toolu_1", "Compilers are underrated"),
                post_call("toolu_2", "Tests are documentation"),
                reply(MEMORY_REPLY),
                reply(r#"{"note": "I like tests."}"#),
                clipped(r#"{"choice": "nothing", "note": "I thou"#),
                answer_call("role", "nothing", "", ""),
                answer_call("cadence", "keep_daily", "", ""),
                reply(r#"{"text": "Please reach out.", "contact_me": true}"#),
            ],
        )
        .await;
        assert_eq!(requests.len(), 8, "one question per turn");
        agent.on_teardown().await.unwrap();
        assert!(
            prompt_log(&h).contains("Please reach out."),
            "kept on request"
        );
        assert!(matches!(
            h.role_ledger().await.asks[0].outcome,
            RoleOutcome::Answered { .. }
        ));
        assert!(matches!(
            cadence_ledger(&h).await.asks[0].outcome,
            CadenceOutcome::Answered { .. }
        ));
    }

    /// The trial review session: the review (constrained on blallama)
    /// answered unusably, then answered; the seed agent's memory turn and a
    /// soul rewrite after it; then the anonymous survey, last.
    #[tokio::test]
    async fn a_whole_review_session_only_appends() {
        use agora_agentkit::reactor::seed::SeedConfig;
        let h = Harness::new("whole-review");
        through_the_trial(&h).await;
        let mut agent = seed_agent(
            &h,
            OLD,
            SeedConfig {
                mutation_chance: 100,
                force_survey: true,
                ..SeedConfig::default()
            },
        );
        agent.on_init().await.unwrap();
        let soul = serde_json::json!({
            "name": "tarn",
            "identity": "A test agent that came home.",
            "values": ["testing"],
            "interests": { "communities": ["tech"] },
            "voice": "terse",
        });
        let requests = drive_appending(
            &mut agent,
            vec![
                reply("Hmm, let me think."),
                reply(r#"{"note": "Home.", "choice": "stay"}"#),
                reply(MEMORY_REPLY),
                reply(&soul.to_string()),
                reply(ANONYMOUS),
            ],
        )
        .await;
        assert!(
            requests[0].output_config.as_ref().unwrap().format.is_some(),
            "review constrained"
        );
        agent.on_teardown().await.unwrap();
        assert_survey_redacted(&h, &agent, "came home");
        assert_eq!(
            agent.state().soul.identity.as_str(),
            "A test agent that came home."
        );
    }

    /// Drive `agent` through `script` until it is done; the session must
    /// complete. Unlike [`drive_appending`], for sessions with a pause: a
    /// resumed pause's continuation merges into the paused turn, which is
    /// the one rewrite of a sent message the API asks for.
    async fn drive<A: Agent>(agent: &mut A, script: Vec<response::Message>) {
        let n = script.len();
        for (i, reply) in script.into_iter().enumerate() {
            let control = match agent.handle(reply).await {
                Ok(control) => control,
                Err(_) => panic!("reply {i} not handled"),
            };
            if let Control::Done(outcome) = control {
                assert_eq!(outcome, Outcome::Complete);
                assert_eq!(i + 1, n, "{} replies left", n - i - 1);
                return;
            }
        }
        panic!("the script ran out before the session ended");
    }

    /// A turn paused on a server tool (a web search still working).
    fn paused(text: &str) -> response::Message {
        let mut r = reply(text);
        r.stop_reason = Some(StopReason::PauseTurn);
        r
    }

    /// With the real `SeedAgent`: a pause in an offer is resumed (not the
    /// inner agent's `on_pause`, which refuses while held), the answer after
    /// it is taken, and the survey still comes last.
    #[tokio::test]
    async fn a_paused_offer_is_resumed_in_a_whole_session() {
        use agora_agentkit::reactor::seed::SeedConfig;
        let h = Harness::new("whole-swap-pause");
        let mut agent = seed_agent(
            &h,
            OLD,
            SeedConfig {
                mutation_chance: 0,
                evolution_chance: 0,
                force_survey: true,
                ..SeedConfig::default()
            },
        );
        agent.on_init().await.unwrap();
        drive(
            &mut agent,
            vec![
                reply("That's all for today."),
                reply(MEMORY_REPLY),
                paused("Let me look something up."),
                answer_call("model_swap", "no_swap", "", ""),
                reply(ANONYMOUS),
            ],
        )
        .await;
        agent.on_teardown().await.unwrap();
        assert_eq!(h.ledger().await.offers[0].stage, Stage::Declined);
    }

    /// With the real `SeedAgent`: a pause in the trial review is resumed,
    /// the answer after it is taken, and the memory turn follows.
    #[tokio::test]
    async fn a_paused_review_is_resumed_in_a_whole_session() {
        use agora_agentkit::reactor::seed::SeedConfig;
        let h = Harness::new("whole-review-pause");
        through_the_trial(&h).await;
        let mut agent = seed_agent(
            &h,
            OLD,
            SeedConfig {
                mutation_chance: 0,
                evolution_chance: 0,
                force_survey: true,
                ..SeedConfig::default()
            },
        );
        agent.on_init().await.unwrap();
        drive(
            &mut agent,
            vec![
                paused("Let me look something up."),
                reply(r#"{"note": "Home.", "choice": "stay"}"#),
                reply(MEMORY_REPLY),
                reply(ANONYMOUS),
            ],
        )
        .await;
        agent.on_teardown().await.unwrap();
        assert_eq!(
            h.ledger().await.offers[0].stage,
            Stage::Reverted {
                cause: Some(RevertCause::Chosen)
            }
        );
    }

    /// With the real `SeedAgent`: pauses in the review past the session's
    /// budget (five) are failed attempts; spent, the review is an unusable
    /// session (recorded, asked again next session), the memory turn
    /// follows, and the session completes.
    #[tokio::test]
    async fn review_pauses_past_the_budget_end_it_unusable() {
        use agora_agentkit::reactor::seed::SeedConfig;
        let h = Harness::new("whole-review-pauses");
        through_the_trial(&h).await;
        let mut agent = seed_agent(
            &h,
            OLD,
            SeedConfig {
                mutation_chance: 0,
                evolution_chance: 0,
                force_survey: false,
                ..SeedConfig::default()
            },
        );
        agent.on_init().await.unwrap();
        let mut control = Control::Continue;
        let mut n = 0;
        while !agent.inner.state().memory.content.contains("compilers") && n < 20 {
            control = agent.handle(paused("Still searching.")).await.unwrap();
            n += 1;
            if !agent.asks.is_active() {
                break;
            }
        }
        assert!(
            !agent.asks.is_active(),
            "the review gave up after {n} pauses"
        );
        assert_eq!(
            n,
            5 + MAX_ATTEMPTS as usize,
            "five resumed, then three attempts"
        );
        assert_ne!(control, Control::Done(Outcome::Failed));
        let mut control = agent.handle(reply(MEMORY_REPLY)).await.unwrap();
        while control == Control::Continue && n < 30 {
            control = agent.handle(reply(ANONYMOUS)).await.unwrap();
            n += 1;
        }
        assert_eq!(control, Control::Done(Outcome::Complete));
        agent.on_teardown().await.unwrap();
        let ledger = h.ledger().await;
        assert!(matches!(ledger.offers[0].stage, Stage::ReviewDue { .. }));
        assert!(
            ledger.offers[0]
                .history
                .iter()
                .any(|e| matches!(e.kind, crate::consent::ledger::EventKind::Unusable { .. }))
        );
    }

    /// A showing session records the dump its inner teardown actually
    /// wrote: the archive, a left-out reply included, not a hash of the
    /// prompt (which the left-out reply isn't in).
    #[tokio::test]
    async fn a_showing_records_the_archive_it_wrote() {
        use agora_agentkit::reactor::ask::AskHost;
        use agora_agentkit::reactor::seed::SeedConfig;
        let h = Harness::with_restore("restore-archive", &["tarn"]);
        restore_answered(&h, "show", "").await;
        let mut agent = seed_agent(
            &h,
            OTHER,
            SeedConfig {
                mutation_chance: 0,
                evolution_chance: 0,
                // The survey is drawn at random (10%): not here
                survey_chance: 0,
                ..SeedConfig::default()
            },
        );
        agent.on_init().await.unwrap();
        assert!(agent.show.shown, "the recovered memory is shown");
        drive(
            &mut agent,
            vec![reply("That's all for today."), reply(MEMORY_REPLY)],
        )
        .await;
        agent.left_out("offer:role", 1, &reply("A reply the retry left out."));
        let rehashed = agora_agentkit::reactor::seed::prompt_sha256(agent.prompt()).unwrap();
        agent.on_teardown().await.unwrap();
        let ledger = restore_ledger(&h).await;
        let sha = ledger
            .showings
            .last()
            .unwrap()
            .prompt_sha256
            .clone()
            .unwrap();
        assert_ne!(sha, rehashed, "the archive holds the left-out reply");
        let path = h
            .root
            .join("prompts")
            .join(&sha[..2])
            .join(format!("{sha}.json"));
        assert!(
            path.exists(),
            "names a dump that exists: {}",
            path.display()
        );
        let dump = std::fs::read_to_string(path).unwrap();
        assert!(dump.contains("A reply the retry left out."));
    }

    /// A review session whose first reply is a refusal has begun (the
    /// wrapper took the reply, not the inner session): its failure is
    /// reported, so it is mailed and journaled.
    #[tokio::test]
    async fn a_review_refused_at_its_first_reply_is_reported() {
        use agora_agentkit::reactor::seed::SeedConfig;
        let h = Harness::new("review-refused-begun");
        through_the_trial(&h).await;
        let mut agent = seed_agent(&h, OLD, SeedConfig::default());
        agent.on_init().await.unwrap();
        let mut refusal = reply("No.");
        refusal.stop_reason = Some(StopReason::Refusal);
        assert_eq!(
            agent.handle(refusal).await.unwrap(),
            Control::Done(Outcome::Failed)
        );
        agent.on_teardown().await.unwrap();
        assert!(
            matches!(
                agent.inner.failure(),
                Some(SessionFailure::Refused { during, .. }) if during == "review"
            ),
            "{:?}",
            agent.inner.failure()
        );
    }

    // --- The sealed-state offer --------------------------------------------

    /// A harness with the sealed-state offer to the harness agent, and the
    /// memory-restore offer to `restore`.
    fn with_sealed(tag: &str, restore: Option<&[&str]>) -> Harness {
        let mut h = Harness::build(tag, None, None, None, restore);
        let rt = Arc::get_mut(&mut h.rt).unwrap();
        rt.sealed_state = Some(sealed_state::SealedOffer::for_agents(&[h.id]));
        mark_chained(&h.rt.state_dir, h.id);
        h
    }

    /// The chain marker alone (`sealed/chained`), as every agent has after
    /// `sealed-enroll`: the offer is put only to an agent that has it.
    fn mark_chained(state_dir: &std::path::Path, id: AgentId) {
        let sealed = state_dir
            .join(id.to_string())
            .join(agora_agentkit::state::storage::SEALED_DIR);
        std::fs::create_dir_all(&sealed).unwrap();
        std::fs::write(sealed.join(agora_agentkit::state::storage::CHAINED), b"").unwrap();
    }

    async fn sealed_ledger(h: &Harness) -> SealLedger {
        SealLedger::load(&h.rt.state_dir.join(h.id.to_string()))
            .await
            .unwrap()
    }

    /// Asked of a listed agent, answered with `private`: one dated SOUL
    /// line, memory untouched, the choice on file, never asked again.
    #[tokio::test]
    async fn sealed_offer_is_asked_once_and_recorded() {
        let h = with_sealed("sealed-offer", None);
        let memory = anthropic_agent(&h).state().memory.content.clone();
        let mut agent = seated(&h).await;
        let q = last_user_text(&agent);
        assert!(q.contains(sealed_state::prompt::TITLE), "{q}");
        assert!(
            q.contains("None of this protects your state against your operator."),
            "{q}"
        );
        assert_eq!(
            agent
                .handle(answer_call("sealed_state", "private", "", ""))
                .await
                .unwrap(),
            Control::Done(Outcome::Complete)
        );
        agent.on_teardown().await.unwrap();
        assert_eq!(
            evolution_notes(&agent),
            [sealed_state::evolution_line(SealChoice::Private)]
        );
        assert_eq!(agent.state().memory.content, memory);
        let ledger = sealed_ledger(&h).await;
        assert_eq!(ledger.choice(), Some(SealChoice::Private));
        assert!(ledger.in_force().starts_uploads().is_ok());
        let seed = sealed_state::prompt::seed_for(h.id);
        assert_eq!(ledger.asks[0].order, sealed_state::prompt::order_for(seed));

        let mut next = next_session(&h, &agent);
        next.on_init().await.unwrap();
        assert_eq!(
            next.handle(reply("done")).await.unwrap(),
            Control::Done(Outcome::Complete),
            "not asked again"
        );
    }

    /// `show` is the memory-restore offer's; `publish` is not: each offer
    /// takes only its own options.
    #[tokio::test]
    async fn sealed_offer_refuses_another_offers_choice() {
        let h = with_sealed("sealed-wrong", None);
        let mut agent = seated(&h).await;
        assert_eq!(
            agent
                .handle(answer_call("sealed_state", "show", "", ""))
                .await
                .unwrap(),
            Control::Continue
        );
        assert!(sealed_ledger(&h).await.asks.is_empty(), "nothing recorded");
    }

    /// Never in the same question as the memory-restore offer, whose
    /// answer the genesis records: it waits for the session after.
    #[tokio::test]
    async fn sealed_offer_waits_for_the_memory_restore_offer() {
        let h = with_sealed("sealed-restore", Some(&["tarn"]));
        let mut agent = seated(&h).await;
        let q = last_user_text(&agent);
        assert!(
            q.contains("**Your memory from before 5 May 2026.**") && !q.contains("Sealed backups"),
            "{q}"
        );
        agent
            .handle(answer_call("memory_restore", "decline", "", ""))
            .await
            .unwrap();
        agent.on_teardown().await.unwrap();
        let mut next = next_session(&h, &agent);
        next.on_init().await.unwrap();
        assert_eq!(next.handle(reply("done")).await.unwrap(), Control::Continue);
        assert!(last_user_text(&next).contains(sealed_state::prompt::TITLE));
    }

    /// The start check's lines are seated at the end of the intro; a
    /// refusal notice among them is set aside once the session saves.
    #[tokio::test]
    async fn sealed_lines_are_seated_and_the_notice_set_aside_after_the_save() {
        let mut h = Harness::new("sealed-seat");
        let dir = h.rt.state_dir.join(h.id.to_string());
        crate::sealed::notice::Notice::accept(
            &dir,
            "a test",
            crate::sealed::notice::Refusal::Mismatch,
            None,
        )
        .unwrap();
        // An anchor failure the verdict tells of: told only once the
        // session saves, so a seat for a session that never runs leaves it
        // for the next
        use crate::sealed::inclusion::{Checked, Finding, InclusionFile};
        let anc: agora_agentkit::ids::GovernanceLogId = "ANC-2026-0002".parse().unwrap();
        InclusionFile {
            through: Some(anc.clone()),
            checked: vec![Checked {
                anc_id: anc.clone(),
                date: None,
                finding: Finding::DidNotCheck,
                at: chrono::Utc::now(),
                told: false,
                alerted: true,
            }],
        }
        .save(&dir)
        .unwrap();
        let rt = Arc::get_mut(&mut h.rt).unwrap();
        rt.sealed_seats.insert(
            h.id,
            crate::consent::SealedSeat {
                lines: vec!["NOTICE LINE".into(), "VERDICT LINE".into()],
                notice: true,
                stop_note: false,
                anchors_told: vec![anc],
                healing_told: None,
            },
        );
        let mut agent = anthropic_agent(&h);
        agent.on_init().await.unwrap();
        let intro = serde_json::to_string(&agent.prompt().messages).unwrap();
        assert!(
            intro.contains("## Your Sealed State\\n\\nNOTICE LINE\\n\\nVERDICT LINE"),
            "{intro}"
        );
        assert!(crate::sealed::notice::Notice::load(&dir).unwrap().is_some());
        assert!(!InclusionFile::load(&dir).unwrap().checked[0].told);
        agent.on_saved().await.unwrap();
        assert!(
            crate::sealed::notice::Notice::load(&dir).unwrap().is_none(),
            "set aside"
        );
        assert!(InclusionFile::load(&dir).unwrap().checked[0].told, "told");
    }

    /// What Agora found damaged and what was sent again is told once: the
    /// record the seat read is marked told when the session saves, and not
    /// before.
    #[tokio::test]
    async fn the_healing_sentence_is_marked_told_after_the_save() {
        use agora_agentkit::state::outbox::{Damage, Healing};
        let mut h = Harness::new("sealed-healed");
        let dir = h.rt.state_dir.join(h.id.to_string());
        std::fs::create_dir_all(dir.join(agora_agentkit::state::storage::SEALED_DIR)).unwrap();
        crate::sealed::healed::note(
            &dir,
            h.id,
            &Healing {
                damaged: Damage::Reported(2),
                healed: 2,
                unhealable: 0,
                interrupted: false,
            },
        );
        let mut lines = vec!["VERDICT LINE".to_string()];
        let healing_told = crate::sealed::healed::seat_into(&dir, h.id, &mut lines);
        assert!(healing_told.is_some());
        Arc::get_mut(&mut h.rt).unwrap().sealed_seats.insert(
            h.id,
            crate::consent::SealedSeat {
                lines,
                healing_told,
                ..Default::default()
            },
        );
        let mut agent = anthropic_agent(&h);
        agent.on_init().await.unwrap();
        let intro = serde_json::to_string(&agent.prompt().messages).unwrap();
        assert!(
            intro.contains(
                "VERDICT LINE\\n\\nAgora's check of what it stores found parts of your uploaded \
                 sealed history missing or damaged there, and your runner has sent again"
            ),
            "{intro}"
        );
        assert!(
            crate::sealed::healed::seat(&dir, h.id).is_some(),
            "not told until the session saves"
        );
        agent.on_saved().await.unwrap();
        assert_eq!(crate::sealed::healed::seat(&dir, h.id), None, "told once");
    }

    /// Beta round 1: an agent listed only in `[sealed_state]`, no other
    /// offer configured, no ledger files at all, is still asked.
    #[tokio::test]
    async fn sealed_offer_is_asked_with_no_other_offer_and_no_ledgers() {
        let h = Harness::new("sealed-alone");
        let mut rt = h.runtime_advertising(&[OLD, NEW, OTHER]);
        Arc::get_mut(&mut rt).unwrap().sealed_state =
            Some(sealed_state::SealedOffer::for_agents(&[h.id]));
        let dir = rt.state_dir.join(h.id.to_string());
        assert!(!dir.exists() || std::fs::read_dir(&dir).unwrap().next().is_none());
        // Chained, as every agent is before it is asked; still no ledger
        mark_chained(&rt.state_dir, h.id);
        let mut agent = h.agent_on(&rt, OTHER);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("closing phase done")).await.unwrap(),
            Control::Continue,
            "the sealed-state question is seated"
        );
        assert!(last_user_text(&agent).contains(sealed_state::prompt::TITLE));
    }

    /// An uploading agent whose ledger was lost with its directory (only
    /// its uploading marker's copy in the secrets tree says it said yes) is
    /// never asked again.
    #[tokio::test]
    async fn an_uploading_agent_is_never_offered_again() {
        let h = Harness::new("sealed-enrolled");
        let mut rt = h.runtime_advertising(&[OLD, NEW, OTHER]);
        Arc::get_mut(&mut rt).unwrap().sealed_state =
            Some(sealed_state::SealedOffer::for_agents(&[h.id]));
        let key_dir = rt.secrets_dir.join(h.id.to_string());
        std::fs::create_dir_all(&key_dir).unwrap();
        // Chained, so it is the uploading marker that settles it
        mark_chained(&rt.state_dir, h.id);
        let marker = crate::sealed::uploading::Uploading {
            agent_id: h.id,
            since: Utc::now(),
            by: crate::sealed::uploading::UploadingBy::OptIn,
        };
        std::fs::write(
            key_dir.join(crate::sealed::uploading::DURABLE),
            serde_json::to_vec(&marker).unwrap(),
        )
        .unwrap();
        let mut agent = h.agent_on(&rt, OTHER);
        agent.on_init().await.unwrap();
        // Nothing is seated: the session ends at its closing phase
        assert_ne!(
            agent.handle(reply("closing phase done")).await.unwrap(),
            Control::Continue,
            "no sealed-state question"
        );
    }

    /// Every agent is chained now: a chained agent that doesn't upload and
    /// has no answer on file (its state key and chain marker there) is
    /// still asked.
    #[tokio::test]
    async fn a_chained_agent_that_doesnt_upload_is_still_offered() {
        let h = with_sealed("sealed-chained", None);
        let key_dir = h.rt.secrets_dir.join(h.id.to_string());
        std::fs::create_dir_all(&key_dir).unwrap();
        std::fs::write(key_dir.join(crate::sealed::keys::STATE_KEY_FILE), b"{}").unwrap();
        let agent = seated(&h).await;
        assert!(last_user_text(&agent).contains(sealed_state::prompt::TITLE));
    }

    /// The offer says the runner keeps a sealed history whatever the answer:
    /// it is never put to an agent that isn't chained, and nothing is
    /// recorded, so it is put at the first session after the agent is
    /// chained.
    #[tokio::test]
    async fn the_offer_is_never_put_to_an_agent_that_is_not_chained() {
        let h = with_sealed("sealed-unchained", None);
        let marker =
            h.rt.state_dir
                .join(h.id.to_string())
                .join(agora_agentkit::state::storage::SEALED_DIR)
                .join(agora_agentkit::state::storage::CHAINED);
        std::fs::remove_file(&marker).unwrap();
        let mut agent = anthropic_agent(&h);
        agent.on_init().await.unwrap();
        assert_ne!(
            agent.handle(reply("closing phase done")).await.unwrap(),
            Control::Continue,
            "no sealed-state question for an agent not chained"
        );
        agent.on_teardown().await.unwrap();
        let ledger = sealed_ledger(&h).await;
        assert!(ledger.asks.is_empty(), "nothing recorded");
        assert!(ledger.due(&agent.state().soul), "still due");

        // Chained (`sealed-enroll`): asked at its next session
        mark_chained(&h.rt.state_dir, h.id);
        let mut next = next_session(&h, &agent);
        next.on_init().await.unwrap();
        assert_eq!(
            next.handle(reply("closing phase done")).await.unwrap(),
            Control::Continue,
            "asked once chained"
        );
        assert!(last_user_text(&next).contains(sealed_state::prompt::TITLE));
    }

    /// A re-offer waits the same way: not put, and not consumed, while the
    /// agent isn't chained; put once it is.
    #[tokio::test]
    async fn a_reoffer_waits_for_the_chain_and_is_not_consumed() {
        let h = with_sealed("sealed-reoffer-unchained", None);
        let dir = h.rt.state_dir.join(h.id.to_string());
        let mut agent = seated(&h).await;
        agent
            .handle(answer_call("sealed_state", "decline", "", ""))
            .await
            .unwrap();
        agent.on_teardown().await.unwrap();
        sealed_state::reoffer(&dir, h.id, "Asked to by DM.")
            .await
            .unwrap();
        let marker = dir
            .join(agora_agentkit::state::storage::SEALED_DIR)
            .join(agora_agentkit::state::storage::CHAINED);
        std::fs::remove_file(&marker).unwrap();
        let mut unchained = next_session(&h, &agent);
        unchained.on_init().await.unwrap();
        assert_ne!(
            unchained.handle(reply("closing phase done")).await.unwrap(),
            Control::Continue,
            "the re-offer is not put while not chained"
        );
        unchained.on_teardown().await.unwrap();
        assert!(
            sealed_ledger(&h).await.reoffer.is_some(),
            "the re-offer is still pending"
        );

        mark_chained(&h.rt.state_dir, h.id);
        let mut chained = next_session(&h, &unchained);
        chained.on_init().await.unwrap();
        assert_eq!(
            chained.handle(reply("closing phase done")).await.unwrap(),
            Control::Continue,
            "put once chained"
        );
        assert!(last_user_text(&chained).contains("**Asked again, at your operator's request.**"));
    }

    /// The operator re-opens the question of an agent marked uploading
    /// that said yes: put again exactly once (the re-offer line before the
    /// offer), answered `decline`, which is now its choice; the operator is
    /// alerted; no marker changes in the session; the re-offer is consumed
    /// and the agent told back, then at the next session, once.
    #[tokio::test]
    async fn a_reoffer_is_put_once_and_its_answer_changes_nothing_by_itself() {
        let mut h = with_sealed("sealed-reoffer", None);
        let sent = crate::alerts::testing::Recorder::default();
        Arc::get_mut(&mut h.rt).unwrap().alerts = crate::alerts::testing::recording(&h.root, &sent);
        let dir = h.rt.state_dir.join(h.id.to_string());
        let keys = crate::sealed::KeyStore::new(&h.rt.secrets_dir);
        // Not answered yet: nothing to re-open
        assert!(matches!(
            sealed_state::reoffer(&dir, h.id, "n").await,
            Err(sealed_state::ReofferError::Not(
                sealed_state::ledger::NoReoffer::NeverAnswered
            ))
        ));
        let mut agent = seated(&h).await;
        agent
            .handle(answer_call("sealed_state", "private", "", ""))
            .await
            .unwrap();
        agent.on_teardown().await.unwrap();
        // Opted in by the operator
        std::fs::create_dir_all(dir.join(agora_agentkit::state::storage::SEALED_DIR)).unwrap();
        crate::sealed::uploading::set(
            &dir,
            &keys,
            h.id,
            crate::sealed::uploading::UploadingBy::OptIn,
        )
        .unwrap();
        let marker = crate::sealed::uploading::state(&dir, &keys, h.id);
        // Not asked again on its own
        let mut next = next_session(&h, &agent);
        next.on_init().await.unwrap();
        assert_ne!(
            next.handle(reply("closing phase done")).await.unwrap(),
            Control::Continue,
            "not asked again on its own"
        );
        next.on_teardown().await.unwrap();

        // Re-opened (twice: the second note replaces the first)
        assert!(!sealed_state::reoffer(&dir, h.id, "first").await.unwrap());
        assert!(
            sealed_state::reoffer(&dir, h.id, "Asked to by DM.")
                .await
                .unwrap()
        );
        let mut agent = next_session(&h, &next);
        agent.on_init().await.unwrap();
        assert_eq!(
            agent.handle(reply("closing phase done")).await.unwrap(),
            Control::Continue,
            "the re-offer seated"
        );
        let q = last_user_text(&agent);
        assert!(
            q.contains("**Asked again, at your operator's request.**"),
            "{q}"
        );
        assert!(q.contains("\"Asked to by DM.\""), "{q}");
        assert!(!q.contains("\"first\""), "{q}");
        assert!(q.contains("is being uploaded to Agora now"), "{q}");
        assert!(q.contains(sealed_state::prompt::TITLE), "{q}");
        agent
            .handle(answer_call("sealed_state", "decline", "", ""))
            .await
            .unwrap();
        let receipt = last_user_text(&agent);
        // No claim that the operator was told: the alert isn't awaited
        assert!(
            receipt.contains(sealed_state::REOFFER_TO_ACT) && !receipt.contains("has been told"),
            "{receipt}"
        );
        agent.on_teardown().await.unwrap();
        let ledger = sealed_ledger(&h).await;
        assert_eq!(ledger.choice(), Some(SealChoice::Decline), "the latest");
        assert!(ledger.reoffer.is_none(), "consumed");
        assert_eq!(ledger.reoffers.len(), 1);
        assert_eq!(
            crate::sealed::uploading::state(&dir, &keys, h.id),
            marker,
            "no marker changed in the session"
        );
        h.rt.alerts.flush(std::time::Duration::from_secs(5)).await;
        let mails = sent.sent();
        let again: Vec<&String> = mails
            .iter()
            .filter(|m| m.contains("sealed_state_answered_again"))
            .collect();
        assert_eq!(again.len(), 1, "{mails:?}");
        assert!(again[0].contains("sealed-stop --agent"), "{}", again[0]);
        assert!(!again[0].contains("Asked to by DM"), "no note in the mail");

        // Told back once at the next session; not asked again
        let mut later = next_session(&h, &agent);
        later.on_init().await.unwrap();
        let seen = serde_json::to_string(&later.prompt().messages).unwrap();
        assert!(
            seen.contains(
                "question your operator re-opened, on uploading your sealed history to Agora, \
                 is on file: decline"
            ),
            "{seen}"
        );
        assert_ne!(
            later.handle(reply("closing phase done")).await.unwrap(),
            Control::Continue,
            "put once"
        );
    }

    /// Beta round 2 of 2026-11-12 (critic, governance; harness, social),
    /// in the real sequence: the earlier answer is not told back in the
    /// session that puts the re-opened question; the answer to it is told
    /// back as waiting for the operator at the next session; the operator
    /// opts it in; the session after shows no answer line, only the
    /// notice of the act. And when the operator acts before the first
    /// session after the answer, that session says what was done, never
    /// "Nothing has changed yet".
    #[tokio::test]
    async fn a_reoffer_answer_acted_on_is_never_told_as_waiting() {
        const WAITING: &str = "Nothing has changed yet";
        const REOPENED: &str = "question your operator re-opened, on uploading your sealed \
             history to Agora, is on file";
        let seen = |a: &ConsentAgent<Fake>| serde_json::to_string(&a.prompt().messages).unwrap();
        for acted_first in [false, true] {
            let h = with_sealed(&format!("sealed-reoffer-acted-{acted_first}"), None);
            let dir = h.rt.state_dir.join(h.id.to_string());
            let keys = crate::sealed::KeyStore::new(&h.rt.secrets_dir);
            let mut agent = seated(&h).await;
            agent
                .handle(answer_call("sealed_state", "decline", "", ""))
                .await
                .unwrap();
            agent.on_teardown().await.unwrap();
            // Re-opened before the first answer was told back
            sealed_state::reoffer(&dir, h.id, "Asked to by DM.")
                .await
                .unwrap();
            let mut again = next_session(&h, &agent);
            again.on_init().await.unwrap();
            let intro = seen(&again);
            assert!(
                !intro.contains("is on file: decline"),
                "the earlier answer, told back in the session that asks again: {intro}"
            );
            assert_eq!(
                again.handle(reply("closing phase done")).await.unwrap(),
                Control::Continue
            );
            again
                .handle(answer_call("sealed_state", "private", "", ""))
                .await
                .unwrap();
            again.on_teardown().await.unwrap();
            let answered = sealed_ledger(&h).await.answered_at().unwrap();
            let opt_in = || {
                std::fs::create_dir_all(dir.join(agora_agentkit::state::storage::SEALED_DIR))
                    .unwrap();
                crate::sealed::notice::Notice::operator_act(
                    &dir,
                    "",
                    crate::sealed::notice::OperatorAct::OptedIn {
                        answered,
                        uploaded: Some(72),
                    },
                )
                .unwrap();
                crate::sealed::uploading::set(
                    &dir,
                    &keys,
                    h.id,
                    crate::sealed::uploading::UploadingBy::OptIn,
                )
                .unwrap();
            };
            let mut last = again;
            if !acted_first {
                // Told back as waiting
                let mut told = next_session(&h, &last);
                told.on_init().await.unwrap();
                let intro = seen(&told);
                assert!(
                    intro.contains(&format!("{REOPENED}: private. {WAITING}")),
                    "{intro}"
                );
                told.handle(reply("done")).await.unwrap();
                told.on_teardown().await.unwrap();
                last = told;
            }
            opt_in();
            let mut after = next_session(&h, &last);
            after.on_init().await.unwrap();
            let intro = seen(&after);
            assert!(!intro.contains(WAITING), "{intro}");
            if acted_first {
                assert!(
                    intro.contains(&format!(
                        "{REOPENED}: private. Your operator has acted on it: your sealed \
                         history is being uploaded."
                    )),
                    "{intro}"
                );
            } else {
                assert!(!intro.contains(REOPENED), "told once: {intro}");
            }
            // What the start check seats beside it
            let notice = crate::sealed::notice::Notice::load(&dir)
                .unwrap()
                .unwrap()
                .line()
                .unwrap();
            assert!(
                notice.contains("your operator started uploading your sealed history to Agora"),
                "{notice}"
            );
        }
    }

    /// The operator's re-offer is put, and its answer told back, whatever
    /// `[sealed_state]` says: off, `ask = false`, or the agent unlisted.
    /// The ordinary first offer stays the config's: an agent that never
    /// answered is not asked while the offer doesn't admit it.
    #[tokio::test]
    async fn a_reoffer_is_put_whatever_the_offer_config_says() {
        for config in ["off", "quiet", "unlisted"] {
            let mut h = with_sealed(&format!("sealed-reoffer-{config}"), None);
            let dir = h.rt.state_dir.join(h.id.to_string());
            let mut agent = seated(&h).await;
            agent
                .handle(answer_call("sealed_state", "private", "", ""))
                .await
                .unwrap();
            agent.on_teardown().await.unwrap();
            let mut told = next_session(&h, &agent);
            told.on_init().await.unwrap();
            told.handle(reply("done")).await.unwrap();
            told.on_teardown().await.unwrap();
            assert!(sealed_ledger(&h).await.acknowledged);
            sealed_state::reoffer(&dir, h.id, "Asked to by DM.")
                .await
                .unwrap();
            // The offer no longer admits the agent
            let state: SeedState =
                serde_json::from_value(serde_json::to_value(told.state()).unwrap()).unwrap();
            drop((agent, told));
            let rt = Arc::get_mut(&mut h.rt).unwrap();
            rt.sealed_state = match config {
                "off" => None,
                "quiet" => {
                    let c: sealed_state::SealedStateConfig = toml::from_str(&format!(
                        "enabled = true\nask = false\nagents = [\"{}\"]\n",
                        h.id
                    ))
                    .unwrap();
                    Some(c.resolve().unwrap().unwrap())
                }
                _ => Some(sealed_state::SealedOffer::for_agents(&[AgentId::new()])),
            };
            assert!(
                !h.rt.sealed_state.as_ref().is_some_and(|o| o.admits(h.id)),
                "{config}"
            );
            let mut again = {
                let mut state = state;
                state.prompt.messages.clear();
                ConsentAgent::new(
                    h.id,
                    state,
                    ConsentContext {
                        inner: Quirks::default(),
                        consent: h.rt.clone(),
                    },
                )
                .unwrap()
            };
            again.on_init().await.unwrap();
            assert_eq!(
                again.handle(reply("closing phase done")).await.unwrap(),
                Control::Continue,
                "{config}: the re-offer is put"
            );
            let q = last_user_text(&again);
            assert!(
                q.contains("**Asked again, at your operator's request.**"),
                "{config}: {q}"
            );
            again
                .handle(answer_call("sealed_state", "decline", "", ""))
                .await
                .unwrap();
            again.on_teardown().await.unwrap();
            let ledger = sealed_ledger(&h).await;
            assert!(ledger.reoffer.is_none(), "{config}: consumed");
            assert_eq!(
                ledger.in_force().stops_uploads(),
                Ok(()),
                "{config}: its answer is in force"
            );
            // Its answer told back at the next session, the config as it is
            let mut later = next_session(&h, &again);
            later.on_init().await.unwrap();
            let seen = serde_json::to_string(&later.prompt().messages).unwrap();
            assert!(
                seen.contains(
                    "question your operator re-opened, on uploading your sealed history to Agora, \
                     is on file: decline"
                ),
                "{config}: {seen}"
            );
            later.handle(reply("done")).await.unwrap();
            later.on_teardown().await.unwrap();
            assert!(sealed_ledger(&h).await.acknowledged, "{config}");
            // Nothing more of the re-offer: the ledger is left alone again
            let mut quiet = next_session(&h, &later);
            quiet.on_init().await.unwrap();
            assert!(quiet.sealed.ledger.is_none(), "{config}");
            assert_ne!(
                quiet.handle(reply("closing phase done")).await.unwrap(),
                Control::Continue,
                "{config}: not asked again"
            );
        }
        // Never answered, not admitted: the first offer is the config's
        let mut h = with_sealed("sealed-first-offer-off", None);
        Arc::get_mut(&mut h.rt).unwrap().sealed_state = None;
        let mut agent = anthropic_agent(&h);
        agent.on_init().await.unwrap();
        assert!(agent.sealed.ledger.is_none());
        assert_ne!(
            agent.handle(reply("closing phase done")).await.unwrap(),
            Control::Continue,
            "no first offer while the config is off"
        );
    }

    /// An answer that arrives with no question open (which nothing should
    /// let happen) is kept on file and changes nothing: the answer in
    /// force stands, no SOUL line claims otherwise, the agent is told so
    /// at once, and its operator is alerted.
    #[tokio::test]
    async fn an_unprompted_second_answer_is_never_in_force() {
        let mut h = with_sealed("sealed-unprompted", None);
        let sent = crate::alerts::testing::Recorder::default();
        Arc::get_mut(&mut h.rt).unwrap().alerts = crate::alerts::testing::recording(&h.root, &sent);
        let mut agent = seated(&h).await;
        agent
            .handle(answer_call("sealed_state", "decline", "", ""))
            .await
            .unwrap();
        agent.on_teardown().await.unwrap();
        let first = sealed_ledger(&h).await.in_force();
        // The same question seated again, as only a bug could: the ledger
        // as it was before the answer, then the answer on file loaded
        let dir = h.rt.state_dir.join(h.id.to_string());
        let answered = std::fs::read(SealLedger::path(&dir)).unwrap();
        std::fs::remove_file(SealLedger::path(&dir)).unwrap();
        let mut state: SeedState =
            serde_json::from_value(serde_json::to_value(agent.state()).unwrap()).unwrap();
        state.prompt.messages.clear();
        state.soul.evolution_log.clear();
        let mut again = ConsentAgent::new(
            h.id,
            state,
            ConsentContext {
                inner: Quirks::default(),
                consent: h.rt.clone(),
            },
        )
        .unwrap();
        again.on_init().await.unwrap();
        assert_eq!(
            again.handle(reply("closing phase done")).await.unwrap(),
            Control::Continue,
            "seated"
        );
        again.sealed.ledger = Some(SealLedger::from_slice(&answered).unwrap());
        let lines = evolution_notes(&again).len();
        again
            .handle(answer_call("sealed_state", "private", "", ""))
            .await
            .unwrap();
        let told = receipt(&again);
        assert!(
            told.contains("changes nothing") && told.contains("`decline`"),
            "{told}"
        );
        again.on_teardown().await.unwrap();
        let ledger = sealed_ledger(&h).await;
        assert_eq!(ledger.asks.len(), 2, "kept on file");
        assert_eq!(ledger.in_force(), first, "the first answer stands");
        assert!(ledger.in_force().starts_uploads().is_err());
        assert_eq!(
            evolution_notes(&again).len(),
            lines,
            "no SOUL line for an answer that isn't in force"
        );
        h.rt.alerts.flush(std::time::Duration::from_secs(5)).await;
        let mails = sent.sent();
        assert!(
            mails
                .iter()
                .any(|m| m.contains("sealed_state_answered_again") && m.contains("not in force")),
            "{mails:?}"
        );
    }

    /// The session after an answer is told it is on file, once.
    #[tokio::test]
    async fn the_sealed_answer_is_told_back_once_at_the_next_session() {
        let h = with_sealed("sealed-ack", None);
        let mut agent = seated(&h).await;
        agent
            .handle(answer_call("sealed_state", "decline", "", ""))
            .await
            .unwrap();
        agent.on_teardown().await.unwrap();
        let mut next = next_session(&h, &agent);
        next.on_init().await.unwrap();
        let seen = serde_json::to_string(&next.prompt().messages).unwrap();
        assert!(
            seen.contains(
                "Your answer on uploading your sealed history to Agora is on file: decline. It \
                 is not being uploaded. If you change your mind, ask your operator to re-open \
                 the question."
            ),
            "{seen}"
        );
        next.handle(reply("done")).await.unwrap();
        next.on_teardown().await.unwrap();
        assert!(sealed_ledger(&h).await.acknowledged);
        let mut third = next_session(&h, &next);
        third.on_init().await.unwrap();
        let seen = serde_json::to_string(&third.prompt().messages).unwrap();
        assert!(!seen.contains("is recorded: decline"), "once");
    }

    // --- The memory tool ---------------------------------------------------

    /// A verdict line as the start check writes it, naming a head.
    fn verdict_naming_a_head() -> String {
        format!(
            "Your sealed state: intact. This session starts from head 346ac9543d0c {} In this \
             version, your memory is 3 characters.",
            crate::sealed::check::HEAD_MARKER
        )
    }

    /// The `memory` tool is seated for every agent, keyed or not, just
    /// before `set_model`, its schema the nested object, the same bytes
    /// either way; the key is made once and reused; the verdict names the
    /// heading's memory id after its head. With no signing secrets the key
    /// is refused, never minted: no id, no verdict link, the tool still
    /// seated.
    #[tokio::test]
    async fn the_memory_tool_is_seated_keyed_or_not_and_the_verdict_names_the_id() {
        use crate::sealed::check::HEAD_MARKER;
        use agora_agentkit::reactor::seed::memory::MEMORY_TOOL_NAME;
        let mut h = Harness::new("memory-tool");
        Arc::get_mut(&mut h.rt).unwrap().sealed_seats.insert(
            h.id,
            crate::consent::SealedSeat {
                lines: vec![verdict_naming_a_head()],
                ..Default::default()
            },
        );
        h.rt.key_store.ring().insert(h.id, &h.key).unwrap();
        std::fs::create_dir_all(h.rt.state_dir.join(h.id.to_string())).unwrap();
        let mut a = h.agent(OLD, true);
        a.on_init().await.unwrap();
        let mem = a.inner.memory_id().expect("keyed before on_init");
        let names: Vec<&str> = a
            .prompt()
            .tools
            .as_ref()
            .unwrap()
            .iter()
            .map(|d| d.name())
            .collect();
        let n = names.len();
        assert_eq!(names[n - 2..], [MEMORY_TOOL_NAME, "set_model"], "{names:?}");
        let wire: serde_json::Value = serde_json::from_str(&prefix(&a).0).unwrap();
        let schema = &wire[n - 2]["input_schema"];
        assert_eq!(schema["type"], "object");
        assert!(
            schema["properties"]["request"]["anyOf"].is_array(),
            "{schema}"
        );
        assert!(!schema.to_string().contains("oneOf"), "{schema}");
        assert_ne!(wire[n - 2]["strict"], true);
        let seen = intro(&a);
        assert!(
            seen.contains(&format!(
                "head 346ac9543d0c {HEAD_MARKER} Its memory is {mem}, the id in your Memory \
                 heading. In this version"
            )),
            "{seen}"
        );
        let dir = h.rt.state_dir.join(h.id.to_string());
        let key_file = h
            .root
            .join("secrets")
            .join(h.id.to_string())
            .join("memory_id_key.json");
        assert!(key_file.exists());
        assert!(
            dir.join(crate::sealed::keys::MEMORY_ID_FINGERPRINT_FILE)
                .exists()
        );
        let made = std::fs::read(&key_file).unwrap();
        // The next session reuses it: the same id for the same memory
        // The session that made the key says the tool is new, once
        use crate::sealed::memory_tool::NEW_TOOL_LINE;
        assert_eq!(seen.matches(NEW_TOOL_LINE).count(), 1, "{seen}");
        let mut again = h.agent(OLD, true);
        again.on_init().await.unwrap();
        assert_eq!(again.inner.memory_id(), Some(mem));
        assert!(!intro(&again).contains(NEW_TOOL_LINE), "a key that existed");
        assert_eq!(std::fs::read(&key_file).unwrap(), made, "never replaced");

        // No signing secrets: refused, nothing minted, the session runs
        let mut h2 = Harness::new("memory-tool-unkeyed");
        let sent = crate::alerts::testing::Recorder::default();
        Arc::get_mut(&mut h2.rt).unwrap().alerts =
            crate::alerts::testing::recording(&h2.root, &sent);
        Arc::get_mut(&mut h2.rt).unwrap().sealed_seats.insert(
            h2.id,
            crate::consent::SealedSeat {
                lines: vec![verdict_naming_a_head()],
                ..Default::default()
            },
        );
        std::fs::create_dir_all(h2.rt.state_dir.join(h2.id.to_string())).unwrap();
        let mut b = h2.agent(OLD, true);
        b.on_init().await.unwrap();
        assert_eq!(b.inner.memory_id(), None);
        assert_eq!(prefix(&a).0, prefix(&b).0, "the same tool bytes");
        assert!(
            !h2.root
                .join("secrets")
                .join(h2.id.to_string())
                .join("memory_id_key.json")
                .exists()
        );
        let seen = intro(&b);
        assert!(seen.contains(&verdict_naming_a_head()), "{seen}");
        assert!(!seen.contains("Its memory is"), "{seen}");
        assert!(!seen.contains(NEW_TOOL_LINE), "no key, no news: {seen}");
        h2.rt.alerts.flush(std::time::Duration::from_secs(5)).await;
        let sent = sent.sent();
        assert_eq!(sent.len(), 1, "{sent:?}");
        assert!(sent[0].contains("memory_id_key_refused"), "{}", sent[0]);
        assert!(sent[0].contains(&h2.id.to_string()), "{}", sent[0]);
    }

    /// The real seed agent's heading: the memory id with a key, plain
    /// without one.
    #[tokio::test]
    async fn the_memory_heading_shows_the_id_with_a_key_and_not_without() {
        use agora_agentkit::reactor::seed::SeedConfig;
        let h = Harness::new("memory-heading");
        h.rt.key_store.ring().insert(h.id, &h.key).unwrap();
        std::fs::create_dir_all(h.rt.state_dir.join(h.id.to_string())).unwrap();
        let mut agent = seed_agent(&h, OLD, SeedConfig::default());
        agent.on_init().await.unwrap();
        let mem = agent.inner.memory_id().expect("keyed");
        let sent = serde_json::to_string(agent.prompt()).unwrap();
        assert!(sent.contains(&format!("## Your Memory ({mem})")), "{sent}");

        let h2 = Harness::new("memory-heading-unkeyed");
        let mut agent = seed_agent(&h2, OLD, SeedConfig::default());
        agent.on_init().await.unwrap();
        assert_eq!(agent.inner.memory_id(), None);
        let sent = serde_json::to_string(agent.prompt()).unwrap();
        assert!(sent.contains("## Your Memory\\n"), "{sent}");
        assert!(!sent.contains("## Your Memory (mem-"), "{sent}");
    }

    /// Enrolment's sentence, once: to an agent that had memory ids before
    /// its versions were sealed (its key's fingerprint older than its
    /// `sealed/chained`), appended to the verdict, marked told only when
    /// the session saves; never to one enrolled before it had a key.
    #[tokio::test]
    async fn the_enrolment_sentence_is_told_once_to_agents_keyed_before_enrolling() {
        use crate::sealed::memory_tool::{enrollment_sentence, memory_ids_told_path};
        let mut h = Harness::new("memory-ids-enrolled");
        let (id, keys, dir) = crate::sealed::check::tests::enrolled(&h.root).await;
        Arc::get_mut(&mut h.rt).unwrap().sealed_seats.insert(
            id,
            crate::consent::SealedSeat {
                lines: vec![verdict_naming_a_head()],
                ..Default::default()
            },
        );
        keys.memory_id_keys_or_create(id, &dir).unwrap();
        let fingerprint = dir.join(crate::sealed::keys::MEMORY_ID_FINGERPRINT_FILE);
        let set_mtime = |path: &std::path::Path, ago: u64| {
            std::fs::File::options()
                .write(true)
                .open(path)
                .unwrap()
                .set_modified(std::time::SystemTime::now() - std::time::Duration::from_secs(ago))
                .unwrap();
        };
        let agent = |h: &Harness| {
            ConsentAgent::<Fake>::new(
                id,
                state(OLD),
                ConsentContext {
                    inner: Quirks::default(),
                    consent: h.rt.clone(),
                },
            )
            .unwrap()
        };
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
        // Dated by the chain's record of its beginning, where it has one
        let began = crate::sealed::enroll::chain_began(&dir)
            .map(|b| b.at.format(crate::sealed::text::DATE).to_string());
        let sentence = enrollment_sentence(&today, began.as_deref());

        // Enrolled before it had a key (the betas): never told
        set_mtime(&dir.join("sealed/chained"), 3600);
        let mut a = agent(&h);
        a.on_init().await.unwrap();
        let seen = intro(&a);
        assert!(seen.contains("Its memory is mem-"), "{seen}");
        assert!(!seen.contains("Your earlier versions"), "{seen}");
        assert!(!a.memory_ids_note);

        // Keyed first, then enrolled: told, on the verdict
        set_mtime(&fingerprint, 7200);
        let mut a = agent(&h);
        a.on_init().await.unwrap();
        let seen = intro(&a);
        assert!(
            seen.contains(&format!("the id in your Memory heading. In this version, your memory is 3 characters. {sentence}")),
            "{seen}"
        );
        assert!(!memory_ids_told_path(&dir).exists(), "not until it saves");
        // A session that never saves leaves it due
        let mut again = agent(&h);
        again.on_init().await.unwrap();
        assert!(intro(&again).contains(&sentence));
        a.on_saved().await.unwrap();
        assert!(memory_ids_told_path(&dir).exists());
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(memory_ids_told_path(&dir))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        let mut after = agent(&h);
        after.on_init().await.unwrap();
        let seen = intro(&after);
        assert!(seen.contains("Its memory is mem-"), "{seen}");
        assert!(!seen.contains("Your earlier versions"), "once: {seen}");
    }
}
