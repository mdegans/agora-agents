//! The sealed-state offer's per-agent record,
//! `state/<agent_id>/sealed_state.json`.
//!
//! Beside the agent's state, never inside its memory, the siblings' rule.
//! One entry per time the offer was put: when, which version of the text,
//! on which model, the option order shown and its seed, how many attempts,
//! and the outcome. **Asked once**: an answer on file, or an explicit
//! refusal, ends it; a missing answer is re-asked up to [`MAX_MISSES`](crate::consent::offer::MAX_MISSES) in
//! all.
//!
//! **One reading for the operator's commands**: [`SealLedger::in_force`]
//! says which answer is in force and whether the operator may act on it
//! now. Everything that starts, resumes or stops uploads by the agent's
//! answer asks it and nothing else. The answer in force is the first
//! answer, or the latest one given to an operator's re-offer; an answer
//! given with no question open is kept on file and never in force; and
//! while a re-offer waits for its answer, nothing acts on the earlier one.

use agora_agentkit::reactor::seed::{ShortString, Soul};
use chrono::{DateTime, Utc};
use misanthropic::model::Model;
use serde::{Deserialize, Serialize};

use super::prompt::SealChoice;

/// The ledger's file name inside the agent's state directory.
pub const LEDGER_FILE: &str = "sealed_state.json";

/// Bump on layout change (2 since 2026-11-06: answers' `note`, the
/// `unusable` outcome; 3 since 2026-11-11: the operator's re-offer, which
/// an older runner would drop on its next save); [`SealLedger::load`]
/// refuses anything newer.
pub const FORMAT: u32 = 3;

/// The whole per-agent file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SealLedger {
    pub format: u32,
    /// Display copy of the agent's name, for whoever reads the file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<ShortString<64>>,
    /// Every time the offer was put, oldest first.
    #[serde(default)]
    pub asks: Vec<SealAsk>,
    /// The answer on file was told back to the agent, once, at the session
    /// after it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub acknowledged: bool,
    /// The operator re-opened the question (`agora-seed sealed-reoffer`):
    /// put again, once, at the agent's next session; consumed by its
    /// answer (or an explicit refusal).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reoffer: Option<Reoffer>,
    /// Re-offers answered, oldest first, each with when it was answered.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reoffers: Vec<Reoffer>,
}

impl Default for SealLedger {
    fn default() -> Self {
        Self {
            format: FORMAT,
            agent: None,
            asks: Vec::new(),
            acknowledged: false,
            reoffer: None,
            reoffers: Vec::new(),
        }
    }
}

/// The operator re-opened the question for the agent. Never consent: only
/// the agent's own answer to it is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reoffer {
    pub at: DateTime<Utc>,
    /// The operator's note: shown to the agent with the question.
    pub note: String,
    /// [`super::prompt::OFFER_VERSION`] of the text to be put.
    pub offer_version: u32,
    /// The answer on file when it was re-opened.
    pub reopens: SealChoice,
    /// When the agent answered it (in [`SealLedger::reoffers`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answered_at: Option<DateTime<Utc>>,
    /// Its answer needed no operator action (the uploads already matched
    /// it): the agent was told so at once.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub no_action: bool,
}

/// Why the operator can't re-open the question.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NoReoffer {
    /// Nothing to re-open: the ordinary offer (`[sealed_state]`) asks an
    /// agent that hasn't answered.
    #[error(
        "it has not answered the offer: list it in `[sealed_state]` to be asked; a re-offer \
         re-opens an answer"
    )]
    NeverAnswered,
}

/// Which answer is in force, and whether the operator may act on it now
/// ([`SealLedger::in_force`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InForce {
    /// No answer on file.
    Unanswered,
    /// Answered (`was`), and the operator re-opened the question since
    /// (`sealed-reoffer`): until the agent answers again, nothing starts,
    /// resumes or stops by `was`.
    Reopened { was: SealChoice },
    /// The answer in force, and when it was given.
    Answered {
        choice: SealChoice,
        at: DateTime<Utc>,
    },
}

/// Why the operator may not act on the agent's answer now.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum NotNow {
    #[error("it has not answered the sealed-state offer")]
    Unanswered,
    #[error("its answer in force to the sealed-state offer is `decline`")]
    SaidNo,
    #[error("its answer in force to the sealed-state offer is `{0}` (yes)")]
    SaidYes(&'static str),
    #[error(
        "the sealed-state question was re-opened for it (`sealed-reoffer`) and it has not \
         answered again yet: its earlier answer (`{0}`) is not acted on meanwhile"
    )]
    Reopened(&'static str),
}

impl InForce {
    /// Whether uploads may start or resume: only on a yes in force, with
    /// no re-offer waiting. The yes, and when it was given.
    pub fn starts_uploads(self) -> Result<(SealChoice, DateTime<Utc>), NotNow> {
        match self {
            Self::Unanswered => Err(NotNow::Unanswered),
            Self::Reopened { was } => Err(NotNow::Reopened(was.as_str())),
            Self::Answered {
                choice: SealChoice::Decline,
                ..
            } => Err(NotNow::SaidNo),
            Self::Answered { choice, at } => Ok((choice, at)),
        }
    }

    /// Whether uploads may be stopped as the agent's answer: on a
    /// `decline` in force with no re-offer waiting, or with no answer on
    /// file (a stop contradicts nothing then). Otherwise only the
    /// operator's own `--against-answer` stops them.
    pub fn stops_uploads(self) -> Result<(), NotNow> {
        match self {
            Self::Unanswered
            | Self::Answered {
                choice: SealChoice::Decline,
                ..
            } => Ok(()),
            Self::Reopened { was } => Err(NotNow::Reopened(was.as_str())),
            Self::Answered { choice, .. } => Err(NotNow::SaidYes(choice.as_str())),
        }
    }
}

/// One putting of the offer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SealAsk {
    pub at: DateTime<Utc>,
    /// [`super::prompt::OFFER_VERSION`] of the text the agent saw.
    pub offer_version: u32,
    /// The model that answered.
    pub model: Model,
    /// The options in the order shown.
    pub order: [SealChoice; 3],
    /// The seed [`super::prompt::order_for`] turned into `order`.
    pub order_seed: u64,
    /// Whether the answer came by the strict tool (or was parsed from text).
    pub constrained: bool,
    /// Answers it took, the usable one included.
    pub attempts: u32,
    pub outcome: SealOutcome,
}

/// What came of it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SealOutcome {
    /// A choice. `reason` is the agent's note.
    Answered {
        choice: SealChoice,
        reason: String,
        /// Whether the agent's `memory_note` was appended to its memory.
        memory_note_written: bool,
        /// Set when the note was not written (why). The choice stands.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        note_dropped: Option<String>,
        /// Set when writing the SOUL line or note failed at teardown (why).
        /// The choice still stands.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        apply_failed: Option<String>,
    },
    /// An explicit refusal. Only read, never written since 2026-11-06 (an
    /// older runner's ledgers hold it): kept so they load, and counted as a
    /// recorded `decline` for every purpose ([`SealLedger::choice`], a
    /// re-offer, [`SealLedger::in_force`]), so it can be revisited through
    /// the operator like any answer.
    Refused { reason: String },
    /// No answer, the agent's doing (an API refusal, since 2026-11-06):
    /// nothing changes this time; counts toward [`MAX_MISSES`](crate::consent::offer::MAX_MISSES).
    NoAnswer { failure: String },
    /// Every reply unusable (since 2026-11-06): our failure, not the
    /// agent's choice. Recorded, never counted: asked again later.
    Unusable { failure: String },
}

impl SealLedger {
    /// Whether the offer should be put this session: nothing final on
    /// file, fewer than [`MAX_MISSES`](crate::consent::offer::MAX_MISSES) misses, and no recorded answer in
    /// `soul`'s Evolution Log.
    /// A pending re-offer is due regardless of the answer on file, until
    /// it has missed as often as a first offer may.
    pub fn due(&self, soul: &Soul) -> bool {
        if let Some(r) = &self.reoffer {
            return crate::consent::offer::misses(
                self.asks
                    .iter()
                    .filter(|a| a.at >= r.at)
                    .map(|a| crate::consent::offer::Outcome::status(&a.outcome)),
            )
            .is_some_and(|m| m < crate::consent::offer::MAX_MISSES);
        }
        crate::consent::offer::due(
            self.asks.iter().map(|a| &a.outcome),
            super::applied_in(soul),
        )
    }

    /// The choice in force, if any, whether or not a re-offer waits: the
    /// first answer, or the latest given to an operator's re-offer (which
    /// replaces an earlier one, either way). A legacy refusal counts as
    /// `decline`. For what an operator may do with it, [`Self::in_force`].
    pub fn choice(&self) -> Option<SealChoice> {
        self.answers().last().map(|(_, c)| *c)
    }

    /// Every answer that was asked for, oldest first, with when (a legacy
    /// refusal as `decline`): the first answer on file, and each answer
    /// that consumed an operator's re-offer ([`Self::record`] dates the
    /// re-offer with it). An answer given with no question open is on file
    /// in [`Self::asks`] and never here: it is never in force.
    fn answers(&self) -> Vec<(DateTime<Utc>, SealChoice)> {
        let mut reopened: Vec<DateTime<Utc>> =
            self.reoffers.iter().filter_map(|r| r.answered_at).collect();
        let mut out = Vec::new();
        for a in &self.asks {
            let choice = match &a.outcome {
                SealOutcome::Answered { choice, .. } => *choice,
                SealOutcome::Refused { .. } => SealChoice::Decline,
                _ => continue,
            };
            // Each answered re-offer vouches for one answer
            let asked = out.is_empty()
                || reopened
                    .iter()
                    .position(|t| *t == a.at)
                    .map(|i| reopened.swap_remove(i))
                    .is_some();
            if asked {
                out.push((a.at, choice));
            }
        }
        out
    }

    /// Which answer is in force, and whether the operator may act on it
    /// now: the one reading behind every command that starts, resumes or
    /// stops uploads by the agent's answer ([`InForce::starts_uploads`],
    /// [`InForce::stops_uploads`]).
    pub fn in_force(&self) -> InForce {
        match (self.answers().last(), &self.reoffer) {
            (None, _) => InForce::Unanswered,
            (Some((_, was)), Some(_)) => InForce::Reopened { was: *was },
            (Some((at, choice)), None) => InForce::Answered {
                choice: *choice,
                at: *at,
            },
        }
    }

    /// The answer that was in force at `at`: the one an operator acted on
    /// then (the opt-in, by the uploading marker's `since`). The publish
    /// choice in force is this one's, not a later re-answer's.
    pub fn choice_as_of(&self, at: DateTime<Utc>) -> Option<SealChoice> {
        self.answers()
            .into_iter()
            .rev()
            .find(|(t, _)| *t <= at)
            .map(|(_, c)| c)
    }

    /// When the answer in force was given.
    pub fn answered_at(&self) -> Option<DateTime<Utc>> {
        self.answers().last().map(|(t, _)| *t)
    }

    /// Re-open the question for the agent (`sealed-reoffer`): put again,
    /// once, at its next session, with `note`. A pending one is replaced
    /// (`true`). The caller saves.
    pub fn reopen(&mut self, note: &str, at: DateTime<Utc>) -> Result<bool, NoReoffer> {
        let reopens = self.choice().ok_or(NoReoffer::NeverAnswered)?;
        let replaced = self.reoffer.is_some();
        self.reoffer = Some(Reoffer {
            at,
            note: note.trim().to_string(),
            offer_version: super::prompt::OFFER_VERSION,
            reopens,
            answered_at: None,
            no_action: false,
        });
        Ok(replaced)
    }

    /// Whether the latest answer was to an operator's re-offer.
    fn latest_reoffer(&self) -> Option<&Reoffer> {
        let latest = self.answered_at()?;
        self.reoffers
            .last()
            .filter(|r| r.answered_at == Some(latest))
    }

    /// Mark the pending re-offer as needing no operator action (before its
    /// answer is recorded).
    pub fn reoffer_needs_no_action(&mut self) {
        if let Some(r) = self.reoffer.as_mut() {
            r.no_action = true;
        }
    }

    /// The line telling the agent its answer is on file, at the first
    /// session after it; `None` once told, or with no answer. `uploading`:
    /// its chain is marked uploading (either copy, stopped or not), which
    /// an answer doesn't change by itself.
    pub fn ack_line(&self, uploading: bool) -> Option<String> {
        if self.acknowledged {
            return None;
        }
        let choice = self.choice()?;
        if let Some(r) = self.latest_reoffer() {
            return Some(if r.no_action {
                // No action: a yes found it uploading, a no found it not
                // (never uploaded, or stopped)
                format!(
                    "Your answer to the sealed-backups question your operator re-opened is \
                     recorded: {}. Nothing needed to change: your sealed history was already {}.",
                    choice.as_str(),
                    if choice == SealChoice::Decline {
                        "not being uploaded to Agora"
                    } else {
                        "being uploaded to Agora"
                    }
                )
            } else {
                format!(
                    "Your answer to the sealed-backups question your operator re-opened is \
                     recorded: {}. Your operator has been told, and acts on it; you'll be told \
                     when it has.",
                    choice.as_str()
                )
            });
        }
        Some(match choice {
            SealChoice::Decline if uploading => {
                "Your answer to the sealed-backups offer is recorded: decline. Your sealed \
                 history is marked for upload to Agora, and an answer doesn't change that by \
                 itself: if you want it changed, tell your operator."
                    .to_string()
            }
            SealChoice::Decline => {
                "Your answer to the sealed-backups offer is recorded: decline. Nothing changed; \
                 if you change your mind, ask your operator."
                    .to_string()
            }
            choice if uploading => format!(
                "Your answer to the sealed-backups offer is recorded: {}. Your sealed history is \
                 marked for upload to Agora.",
                choice.as_str()
            ),
            choice => format!(
                "Your answer to the sealed-backups offer is recorded: {}. Uploading starts \
                 only when your operator acts on it; you'll be told when it has.",
                choice.as_str()
            ),
        })
    }

    /// The answer was told back.
    pub fn acknowledge(&mut self) {
        self.acknowledged = true;
    }

    /// Record a putting of the offer. An answer (or a refusal) consumes a
    /// pending re-offer, and is told back at the next session. An answer
    /// given with no question open ([`Self::unprompted_answer`]) is kept
    /// on file and changes nothing: not the answer in force, and nothing
    /// to tell back.
    pub fn record(&mut self, ask: SealAsk) {
        let settled = matches!(
            ask.outcome,
            SealOutcome::Answered { .. } | SealOutcome::Refused { .. }
        );
        let unprompted = settled && self.unprompted_answer();
        if settled && let Some(mut r) = self.reoffer.take() {
            r.answered_at = Some(ask.at);
            self.reoffers.push(r);
        }
        if matches!(ask.outcome, SealOutcome::Answered { .. }) && !unprompted {
            self.acknowledged = false;
        }
        self.asks.push(ask);
    }

    /// The operator's re-offer still has something to do in a session,
    /// whatever `[sealed_state]` says: it waits to be put, or its answer
    /// waits to be told back.
    pub fn reoffer_open(&self) -> bool {
        self.reoffer.is_some() || (!self.acknowledged && self.latest_reoffer().is_some())
    }

    /// An answer arriving now would be the agent's second (or later) with
    /// no re-offer pending: the offer is asked once, so a ledger that
    /// shows this is wrong. Logged at error and alerted; recorded, and
    /// never the answer in force ([`Self::in_force`]).
    pub fn unprompted_answer(&self) -> bool {
        self.reoffer.is_none() && self.choice().is_some()
    }

    /// Note that writing the answer's SOUL line or note failed. The choice
    /// itself stands.
    pub fn mark_apply_failed(&mut self, why: String) {
        let answered = self
            .asks
            .iter_mut()
            .rev()
            .find_map(|a| match &mut a.outcome {
                SealOutcome::Answered {
                    memory_note_written,
                    apply_failed,
                    ..
                } => Some((memory_note_written, apply_failed)),
                _ => None,
            });
        if let Some((memory_note_written, apply_failed)) = answered {
            *memory_note_written = false;
            *apply_failed = Some(why);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consent::offer::Stored;

    fn soul() -> Soul {
        serde_json::from_value(serde_json::json!({
            "name": "tarn",
            "identity": "A test agent.",
            "values": ["testing"],
            "interests": { "communities": ["tech"] },
            "voice": "terse",
        }))
        .unwrap()
    }

    fn ask(outcome: SealOutcome) -> SealAsk {
        SealAsk {
            at: "2026-11-08T09:00:00Z".parse().unwrap(),
            offer_version: 1,
            model: Model::from("claude-haiku-4-5"),
            order: SealChoice::ALL,
            order_seed: 42,
            constrained: true,
            attempts: 1,
            outcome,
        }
    }

    fn answered(choice: SealChoice) -> SealOutcome {
        SealOutcome::Answered {
            choice,
            reason: String::new(),
            memory_note_written: false,
            note_dropped: None,
            apply_failed: None,
        }
    }

    /// An answer to an older text settles the offer: a new version asks
    /// nobody who answered an older one again (v6 nobody who answered v5;
    /// v9, the JSON answer, nobody who answered with `answer_offer`).
    #[test]
    fn an_answer_to_an_older_offer_text_is_not_asked_again() {
        for old in [5, 8] {
            let mut l = SealLedger::default();
            let mut a = ask(answered(SealChoice::Private));
            a.offer_version = old;
            l.record(a);
            assert!(!l.due(&soul()), "v{old}");
        }
        const { assert!(super::super::prompt::OFFER_VERSION > 8) };
    }

    #[test]
    fn asked_once_and_yes_is_either_yes() {
        for choice in SealChoice::ALL {
            let mut l = SealLedger::default();
            assert!(l.due(&soul()));
            assert_eq!(l.in_force().starts_uploads(), Err(NotNow::Unanswered));
            l.record(ask(answered(choice)));
            assert!(!l.due(&soul()), "{choice:?}");
            assert_eq!(l.choice(), Some(choice));
            assert_eq!(
                l.in_force().starts_uploads().is_ok(),
                choice != SealChoice::Decline
            );
        }
        let mut l = SealLedger::default();
        l.record(ask(SealOutcome::Refused { reason: "r".into() }));
        assert!(!l.due(&soul()), "a refusal is final");
        assert_eq!(l.in_force().starts_uploads(), Err(NotNow::SaidNo));
        let mut l = SealLedger::default();
        for _ in 0..crate::consent::offer::MAX_MISSES {
            assert!(l.due(&soul()));
            l.record(ask(SealOutcome::NoAnswer {
                failure: "refusal".into(),
            }));
        }
        assert!(!l.due(&soul()), "misses capped");
    }

    /// The answer is told back once, at the session after it.
    #[test]
    fn the_answer_is_acknowledged_once() {
        let mut l = SealLedger::default();
        assert_eq!(l.ack_line(false), None, "nothing answered");
        l.record(ask(SealOutcome::NoAnswer {
            failure: "refusal".into(),
        }));
        assert_eq!(l.ack_line(false), None, "no answer, nothing to tell");
        l.record(ask(answered(SealChoice::Publish)));
        assert_eq!(
            l.ack_line(false).unwrap(),
            "Your answer to the sealed-backups offer is recorded: publish. Uploading starts \
             only when your operator acts on it; you'll be told when it has."
        );
        assert_eq!(
            l.ack_line(true).unwrap(),
            "Your answer to the sealed-backups offer is recorded: publish. Your sealed history \
             is marked for upload to Agora."
        );
        l.acknowledge();
        assert_eq!(l.ack_line(false), None, "once");
        let mut no = SealLedger::default();
        no.record(ask(answered(SealChoice::Decline)));
        assert_eq!(
            no.ack_line(false).unwrap(),
            "Your answer to the sealed-backups offer is recorded: decline. Nothing changed; \
             if you change your mind, ask your operator."
        );
    }

    #[tokio::test]
    async fn round_trips_and_refuses_a_newer_format() {
        let dir = crate::sealed::TestDir::new("seal-ledger");
        assert_eq!(
            SealLedger::load(dir.path()).await.unwrap(),
            SealLedger::default()
        );
        let mut l = SealLedger::default();
        l.record(ask(answered(SealChoice::Private)));
        l.save(dir.path()).await.unwrap();
        assert_eq!(SealLedger::load(dir.path()).await.unwrap(), l);
        assert_eq!(SealLedger::load_blocking(dir.path()).unwrap(), l);
        let mut newer = l.clone();
        newer.format = FORMAT + 1;
        newer.save(dir.path()).await.unwrap();
        assert!(SealLedger::load(dir.path()).await.is_err());
    }

    /// The ledger is the runner's user's alone (0600), and one written
    /// with wider permissions before is tightened by the next save.
    #[cfg(unix)]
    #[tokio::test]
    async fn the_ledger_is_written_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = crate::sealed::TestDir::new("seal-ledger-mode");
        let mode = |p: &std::path::Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        let path = SealLedger::path(dir.path());
        let mut l = SealLedger::default();
        l.record(ask(answered(SealChoice::Private)));
        // As an older runner left it
        std::fs::write(&path, serde_json::to_vec(&l).unwrap()).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert_eq!(SealLedger::load(dir.path()).await.unwrap(), l);
        l.acknowledge();
        l.save(dir.path()).await.unwrap();
        assert_eq!(mode(&path), 0o600);
        assert_eq!(SealLedger::load(dir.path()).await.unwrap(), l);
        let left: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(left.len(), 1, "no temporary left: {left:?}");
        // The files of sealed/ written through `anchors::write_atomic` too
        let other = dir.path().join("sealed").join("receipts.json");
        std::fs::create_dir_all(other.parent().unwrap()).unwrap();
        std::fs::write(&other, b"{}").unwrap();
        std::fs::set_permissions(&other, std::fs::Permissions::from_mode(0o644)).unwrap();
        crate::sealed::anchors::write_atomic(&other, b"{}").unwrap();
        assert_eq!(mode(&other), 0o600);
    }

    fn at(ask: SealAsk, when: &str) -> SealAsk {
        SealAsk {
            at: when.parse().unwrap(),
            ..ask
        }
    }

    /// `reopen`: refused with no answer; a second replaces the note; due
    /// whatever the answer and the Evolution Log, until the misses run
    /// out; consumed by the answer, which is then the latest and told
    /// back once as a re-offer's.
    #[test]
    fn a_reoffer_is_due_once_and_consumed_by_its_answer() {
        let mut l = SealLedger::default();
        assert_eq!(l.reopen("n", Utc::now()), Err(NoReoffer::NeverAnswered));
        l.record(at(
            ask(answered(SealChoice::Private)),
            "2026-11-08T09:00:00Z",
        ));
        l.acknowledge();
        assert!(!l.due(&soul()));
        let t = "2026-11-10T09:00:00Z".parse().unwrap();
        assert_eq!(l.reopen("first", t), Ok(false));
        assert_eq!(l.reopen("Asked by DM.", t), Ok(true), "replaced");
        assert_eq!(l.reoffer.as_ref().unwrap().note, "Asked by DM.");
        assert!(l.due(&soul()));
        let mut n = 0;
        while l.due(&soul()) && n < 64 {
            l.record(at(
                ask(SealOutcome::Unusable {
                    failure: "x".into(),
                }),
                "2026-11-11T09:00:00Z",
            ));
            n += 1;
        }
        assert!(n > 1 && !l.due(&soul()), "the misses ran out ({n})");
        assert!(l.reoffer.is_some(), "not consumed by a miss");
        l.record(at(
            ask(answered(SealChoice::Decline)),
            "2026-11-12T09:00:00Z",
        ));
        assert!(l.reoffer.is_none());
        assert_eq!(l.reoffers.len(), 1);
        assert_eq!(l.choice(), Some(SealChoice::Decline));
        assert!(!l.due(&soul()));
        let told = l.ack_line(true).unwrap();
        assert!(
            told.contains("question your operator re-opened is recorded: decline"),
            "{told}"
        );
        assert!(told.contains("you'll be told when it has"), "{told}");
        l.acknowledge();
        assert_eq!(l.ack_line(true), None, "once");
    }

    /// A re-offer answered where nothing needs doing says so at once.
    #[test]
    fn a_reoffer_answer_needing_nothing_says_so() {
        let mut l = SealLedger::default();
        l.record(at(
            ask(answered(SealChoice::Private)),
            "2026-11-08T09:00:00Z",
        ));
        l.reopen("n", "2026-11-10T09:00:00Z".parse().unwrap())
            .unwrap();
        l.reoffer_needs_no_action();
        l.record(at(
            ask(answered(SealChoice::Publish)),
            "2026-11-12T09:00:00Z",
        ));
        let told = l.ack_line(true).unwrap();
        assert_eq!(
            told,
            "Your answer to the sealed-backups question your operator re-opened is recorded: \
             publish. Nothing needed to change: your sealed history was already being uploaded \
             to Agora."
        );
        assert!(!told.contains("you'll be told"), "{told}");
        // A no that needed nothing: never uploading, or stopped
        let mut no = SealLedger::default();
        no.record(at(
            ask(answered(SealChoice::Private)),
            "2026-11-08T09:00:00Z",
        ));
        no.reopen("n", "2026-11-10T09:00:00Z".parse().unwrap())
            .unwrap();
        no.reoffer_needs_no_action();
        no.record(at(
            ask(answered(SealChoice::Decline)),
            "2026-11-12T09:00:00Z",
        ));
        assert!(no.ack_line(false).unwrap().ends_with(
            "Nothing needed to change: your sealed history was already not being \
                            uploaded to Agora."
        ),);
    }

    /// The one reading: uploads start only on a yes in force, stop by the
    /// answer only on a `decline` (or with no answer); `choice_as_of` is
    /// the answer acted on then.
    #[test]
    fn the_answer_in_force_and_the_choice_as_of_a_time() {
        let mut l = SealLedger::default();
        assert_eq!(l.in_force(), InForce::Unanswered);
        assert_eq!(l.in_force().stops_uploads(), Ok(()));
        assert_eq!(l.in_force().starts_uploads(), Err(NotNow::Unanswered));
        let first = "2026-11-08T09:00:00Z".parse().unwrap();
        l.record(at(
            ask(answered(SealChoice::Private)),
            "2026-11-08T09:00:00Z",
        ));
        assert_eq!(
            l.in_force().starts_uploads(),
            Ok((SealChoice::Private, first))
        );
        assert_eq!(
            l.in_force().stops_uploads(),
            Err(NotNow::SaidYes("private"))
        );
        l.reopen("n", "2026-11-09T09:00:00Z".parse().unwrap())
            .unwrap();
        l.record(at(
            ask(answered(SealChoice::Decline)),
            "2026-11-12T09:00:00Z",
        ));
        assert_eq!(l.in_force().starts_uploads(), Err(NotNow::SaidNo));
        assert_eq!(l.in_force().stops_uploads(), Ok(()));
        let between = "2026-11-10T00:00:00Z".parse().unwrap();
        assert_eq!(l.choice_as_of(between), Some(SealChoice::Private));
        assert_eq!(
            l.choice_as_of("2026-11-01T00:00:00Z".parse().unwrap()),
            None
        );
    }

    /// While the operator's re-offer waits for its answer, nothing acts on
    /// the earlier one: not an opt-in or a resume on its yes, not a stop
    /// on its no. Also once the re-offer's misses have run out.
    #[test]
    fn a_pending_reoffer_holds_every_act_on_the_earlier_answer() {
        for was in SealChoice::ALL {
            let mut l = SealLedger::default();
            l.record(at(ask(answered(was)), "2026-11-08T09:00:00Z"));
            l.reopen("n", "2026-11-10T09:00:00Z".parse().unwrap())
                .unwrap();
            assert_eq!(l.in_force(), InForce::Reopened { was });
            assert_eq!(
                l.in_force().starts_uploads(),
                Err(NotNow::Reopened(was.as_str()))
            );
            assert_eq!(
                l.in_force().stops_uploads(),
                Err(NotNow::Reopened(was.as_str()))
            );
            assert_eq!(l.choice(), Some(was), "still the answer on file");
            let mut n = 0;
            while l.due(&soul()) && n < 64 {
                l.record(at(
                    ask(SealOutcome::NoAnswer {
                        failure: "refusal".into(),
                    }),
                    "2026-11-11T09:00:00Z",
                ));
                n += 1;
            }
            assert_eq!(l.in_force(), InForce::Reopened { was }, "misses ran out");
        }
    }

    /// An answer given with no question open is on file and never in
    /// force, whichever way it points; nothing is told back for it; and a
    /// later re-offer re-opens the answer in force, not it.
    #[test]
    fn an_unprompted_answer_is_never_in_force() {
        for (first, second) in [
            (SealChoice::Decline, SealChoice::Private),
            (SealChoice::Decline, SealChoice::Publish),
            (SealChoice::Private, SealChoice::Decline),
            (SealChoice::Publish, SealChoice::Private),
        ] {
            let mut l = SealLedger::default();
            let t = "2026-11-08T09:00:00Z".parse().unwrap();
            l.record(at(ask(answered(first)), "2026-11-08T09:00:00Z"));
            l.acknowledge();
            assert!(l.unprompted_answer());
            l.record(at(ask(answered(second)), "2026-11-09T09:00:00Z"));
            assert_eq!(l.asks.len(), 2, "kept on file");
            assert_eq!(
                l.in_force(),
                InForce::Answered {
                    choice: first,
                    at: t
                }
            );
            assert_eq!(l.choice(), Some(first));
            assert_eq!(
                l.choice_as_of("2026-11-10T00:00:00Z".parse().unwrap()),
                Some(first)
            );
            assert_eq!(l.ack_line(false), None, "nothing new to tell back");
            l.reopen("n", "2026-11-10T09:00:00Z".parse().unwrap())
                .unwrap();
            assert_eq!(l.reoffer.as_ref().unwrap().reopens, first);
        }
    }

    /// The first answer's ack for a `decline` while uploading says only
    /// what is true.
    #[test]
    fn a_first_decline_while_uploading_is_told_truly() {
        let mut l = SealLedger::default();
        l.record(ask(answered(SealChoice::Decline)));
        let told = l.ack_line(true).unwrap();
        assert_eq!(
            told,
            "Your answer to the sealed-backups offer is recorded: decline. Your sealed history is \
             marked for upload to Agora, and an answer doesn't change that by itself: if you \
             want it changed, tell your operator."
        );
        assert!(!told.contains("earlier answer"), "{told}");
    }

    #[derive(Debug, Clone, Copy)]
    enum Op {
        /// An answer arrives (to the first offer, to a re-offer, or with
        /// no question open: whichever the ledger's state makes it).
        Answer(SealChoice),
        /// A legacy refusal arrives the same way.
        Refuse,
        Miss,
        Unusable,
        /// `sealed-reoffer`.
        Reopen,
        /// The ledger is saved and loaded.
        Reload,
    }

    fn op() -> impl proptest::strategy::Strategy<Value = Op> {
        use proptest::prelude::*;
        prop_oneof![
            3 => proptest::sample::select(SealChoice::ALL.to_vec()).prop_map(Op::Answer),
            1 => Just(Op::Refuse),
            1 => Just(Op::Miss),
            1 => Just(Op::Unusable),
            2 => Just(Op::Reopen),
            1 => Just(Op::Reload),
        ]
    }

    proptest::proptest! {
        /// Over any sequence of offers, answers, re-offers, answers to
        /// them and answers nobody asked for: the answer in force is the
        /// first answer or the latest given to a re-offer, never an
        /// unprompted one; while a re-offer waits nothing may act; and
        /// uploads may start only on a yes in force.
        #[test]
        fn the_answer_in_force_is_only_ever_one_that_was_asked_for(
            ops in proptest::collection::vec(op(), 0..24)
        ) {
            use proptest::prelude::*;
            let mut l = SealLedger::default();
            // The model: the answer in force, and whether a re-offer waits
            let mut model: Option<(DateTime<Utc>, SealChoice)> = None;
            let mut waiting = false;
            let mut now: DateTime<Utc> = "2026-11-08T09:00:00Z".parse().unwrap();
            for op in ops {
                now += chrono::Duration::seconds(1);
                let before = l.in_force();
                match op {
                    Op::Answer(_) | Op::Refuse => {
                        let (outcome, choice) = match op {
                            Op::Answer(c) => (answered(c), c),
                            _ => (
                                SealOutcome::Refused { reason: "r".into() },
                                SealChoice::Decline,
                            ),
                        };
                        let asked = model.is_none() || waiting;
                        prop_assert_eq!(l.unprompted_answer(), !asked);
                        l.record(SealAsk { at: now, ..ask(outcome) });
                        if asked {
                            model = Some((now, choice));
                            waiting = false;
                        } else {
                            prop_assert_eq!(l.in_force(), before, "an unprompted answer changed it");
                        }
                    }
                    Op::Miss => l.record(SealAsk {
                        at: now,
                        ..ask(SealOutcome::NoAnswer { failure: "refusal".into() })
                    }),
                    Op::Unusable => l.record(SealAsk {
                        at: now,
                        ..ask(SealOutcome::Unusable { failure: "x".into() })
                    }),
                    Op::Reopen => {
                        let done = l.reopen("n", now);
                        prop_assert_eq!(done.is_ok(), model.is_some());
                        waiting = model.is_some();
                    }
                    Op::Reload => {
                        l = SealLedger::from_slice(&serde_json::to_vec(&l).unwrap()).unwrap();
                    }
                }
                let expect = match (model, waiting) {
                    (None, _) => InForce::Unanswered,
                    (Some((_, was)), true) => InForce::Reopened { was },
                    (Some((at, choice)), false) => InForce::Answered { choice, at },
                };
                prop_assert_eq!(l.in_force(), expect);
                prop_assert_eq!(l.choice(), model.map(|(_, c)| c));
                prop_assert_eq!(l.choice_as_of(now), model.map(|(_, c)| c));
                let yes = matches!(
                    model,
                    Some((_, SealChoice::Publish | SealChoice::Private))
                );
                prop_assert_eq!(l.in_force().starts_uploads().is_ok(), yes && !waiting);
                prop_assert_eq!(
                    l.in_force().stops_uploads().is_ok(),
                    !yes && !waiting
                );
            }
        }
    }
}
