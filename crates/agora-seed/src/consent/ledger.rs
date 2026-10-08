//! The runner-owned consent ledger: one JSON file per agent, beside (never
//! inside) its state and memory.
//!
//! `<data_dir>/state/<agent_id>/model_consent.json`. The agent's memory is
//! the agent's; nothing here writes to it (Steward, 2026-09-23: we never
//! write to an agent's memory without its consent). This file is the
//! runner's record of what it asked, what came back, and what the Steward
//! still has to apply.
//!
//! All transitions are pure methods on [`Ledger`] so the state machine is
//! tested without a model or a network; [`super::agent`] only decides
//! *when* to call them.

use std::path::{Path, PathBuf};

use agora_agentkit::reactor::seed::{ITEM_MAX, ShortString, Soul};
use chrono::{DateTime, NaiveDate, Utc};
use misanthropic::model::Model;
use serde::{Deserialize, Serialize};

use super::prompt::{OfferAnswer, OfferChoice, ReviewAnswer, ReviewChoice};

/// The ledger's file name inside the agent's state directory.
pub const LEDGER_FILE: &str = "model_consent.json";

/// Bump on layout change; [`Ledger::load`] refuses anything newer so an old
/// binary never half-reads (and then overwrites) a newer ledger.
pub const FORMAT: u32 = 1;

/// Sessions on the new model before a trial is reviewed.
pub const TRIAL_SESSIONS: u32 = 5;

/// How the countdown line ([`Ledger::trial_line`]) starts. A fork of a
/// trial session removes the line by it: it names the trial model as the
/// one the session runs on.
pub const TRIAL_LINE_PREFIX: &str = "Model trial: ";

/// Unanswered asks before the runner stops asking. The first miss is
/// re-asked at the next eligible session; the second is final.
pub const MAX_MISSES: u32 = 2;

/// An offer is identified by the model pair it moves between. Two offers
/// for the same pair are the same question, so an agent that answered one
/// is never asked the other.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OfferKey {
    pub from: Model,
    pub to: Model,
}

/// The whole per-agent file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ledger {
    pub format: u32,
    /// Display copy of the agent's name, so the queue report can print
    /// `set_model --agent <name>` without loading every state file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<ShortString<64>>,
    #[serde(default)]
    pub offers: Vec<OfferRecord>,
    /// Every model change the runner applied, oldest first. Absent from
    /// ledgers written before the runner applied changes itself.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub switches: Vec<Switch>,
}

impl Default for Ledger {
    fn default() -> Self {
        Self {
            format: FORMAT,
            agent: None,
            offers: Vec::new(),
            switches: Vec::new(),
        }
    }
}

/// Completed sessions after a switch before the agent may switch again.
pub const SWITCH_COOLDOWN_SESSIONS: u32 = 5;

/// A model change the runner applied: reported to Agora, and the agent
/// routed on `to` from its next session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Switch {
    pub at: DateTime<Utc>,
    pub from: Model,
    pub to: Model,
    #[serde(flatten)]
    pub cause: SwitchCause,
    /// Completed sessions that started after this switch — the cooldown's
    /// count.
    #[serde(default)]
    pub sessions_after: u32,
}

/// Why a [`Switch`] happened.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "cause", rename_all = "snake_case")]
pub enum SwitchCause {
    /// The agent asked, with `set_model`. `trial` when it chose a trial
    /// (the record is then an [`OfferRecord`] begun by
    /// [`Ledger::begin_self_trial`]).
    SelfSwitch {
        reason: String,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        trial: bool,
    },
    /// An answered offer or trial review.
    Consent { action: ChangeAction },
}

/// Everything about one offer for one agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OfferRecord {
    #[serde(flatten)]
    pub key: OfferKey,
    /// Human names the question used, kept so a trial review can be asked
    /// after the offer is gone from the config.
    pub from_name: String,
    pub to_name: String,
    pub stage: Stage,
    /// Append-only: every ask (answered or not) and every observed move.
    #[serde(default)]
    pub history: Vec<Event>,
    /// The exact note text of this offer's one entry in the agent's SOUL
    /// Evolution Log, as last written — how the entry is found again to be
    /// replaced in place. See [`Ledger::update_soul`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soul_note: Option<String>,
}

/// Where an offer stands for one agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "stage", rename_all = "snake_case")]
pub enum Stage {
    /// Asked, no usable answer yet — asked again at the next eligible
    /// session until [`MAX_MISSES`].
    Unanswered { misses: u32 },
    /// Chose to stay. Final.
    Declined,
    /// Never answered in [`MAX_MISSES`] asks: treated as "stay". Final.
    NoAnswer,
    /// Chose to move (or, after a trial, to keep the new model); the change
    /// is applied, or waits to be, and takes effect at the agent's next
    /// session on `to`.
    AwaitingSwap { term: Term },
    /// On the new model for a trial. `sessions` counts completed sessions
    /// actually run on it.
    Trial {
        started_at: DateTime<Utc>,
        sessions: u32,
        /// Unanswered reviews under the pre-2026-09-26 flow, which asked on
        /// the new model. Kept so older ledgers (and binaries) still read;
        /// nothing counts it now.
        #[serde(default)]
        review_misses: u32,
    },
    /// The trial's [`TRIAL_SESSIONS`] are done: the agent is being moved
    /// back to `from`, where the review is asked (Steward, 2026-09-25: the
    /// original weights make the final call). Until the move takes effect
    /// the agent still runs on `to`, and the move is retried at the end of
    /// each such session.
    ReturningForReview {
        started_at: DateTime<Utc>,
        ended_at: DateTime<Utc>,
        sessions: u32,
    },
    /// Back on `from`: the review is the first thing asked at the agent's
    /// next session on it.
    ReviewDue {
        started_at: DateTime<Utc>,
        ended_at: DateTime<Utc>,
        sessions: u32,
    },
    /// Returning to the old model; applied, or waiting to be (the
    /// pre-2026-09-26 review, which was asked on the new model).
    AwaitingRevert { cause: RevertCause },
    /// On the new model for good (chose permanent, or kept the trial).
    /// Final.
    Moved,
    /// Back on (or kept on) the old model. Final. `cause` is `None` when
    /// the agent was moved back out of band mid-trial.
    Reverted {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cause: Option<RevertCause>,
    },
    /// The agent moved to a model that is neither side of the offer while
    /// the offer was under way: by hand (the history's last `moved` event
    /// names the model), or by the agent's own `set_model` mid-trial (a
    /// `decided_early` event). The offer is over. Final.
    Superseded,
}

impl Stage {
    /// Whether a change or trial for this offer is under way. While one
    /// is, the agent isn't offered another, and can't switch model itself
    /// except to decide a running [`Stage::Trial`] early
    /// ([`Ledger::decide_early`]).
    pub fn in_progress(&self) -> bool {
        matches!(
            self,
            Stage::AwaitingSwap { .. }
                | Stage::Trial { .. }
                | Stage::ReturningForReview { .. }
                | Stage::ReviewDue { .. }
                | Stage::AwaitingRevert { .. }
        )
    }
}

#[cfg(test)]
impl Stage {
    /// Whether the record has a change for the runner to apply.
    pub fn is_queued(&self) -> bool {
        matches!(
            self,
            Stage::AwaitingSwap { .. }
                | Stage::AwaitingRevert { .. }
                | Stage::ReturningForReview { .. }
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Term {
    Trial,
    Permanent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RevertCause {
    /// The agent chose the old model at its trial review.
    Chosen,
    /// The trial review went unanswered: the agent agreed to a
    /// [`TRIAL_SESSIONS`]-session trial, not a permanent move.
    NoAnswer,
}

/// One line of an offer's history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub at: DateTime<Utc>,
    #[serde(flatten)]
    pub kind: EventKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum EventKind {
    /// The offer question was asked. `answer` is `None` when nothing
    /// usable came back; `failure` then says why.
    Offered {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        answer: Option<OfferAnswer>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        failure: Option<String>,
    },
    /// The trial review was asked after `sessions` sessions on the new
    /// model.
    Reviewed {
        sessions: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        answer: Option<ReviewAnswer>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        failure: Option<String>,
    },
    /// The runner saw the agent start a session on `model` after a queued
    /// change took effect (or after an out-of-band move).
    Moved { model: Model },
    /// The trial's `sessions` were done; the runner began moving the agent
    /// back to `from` for its review.
    TrialEnded { sessions: u32 },
    /// The agent chose this trial itself, with `set_model` (`term =
    /// trial`), rather than in answer to an offer.
    SelfTrial { reason: String },
    /// The agent decided the trial before its review, with `set_model`,
    /// after `sessions` completed trial sessions. Nothing asks the review
    /// question after it. `model` is the model chosen (`name` its display
    /// name).
    DecidedEarly {
        sessions: u32,
        decision: EarlyDecision,
        model: Model,
        name: String,
        reason: String,
    },
}

/// How a trial was decided early ([`EventKind::DecidedEarly`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EarlyDecision {
    /// Chose the trial's model: kept for good.
    Keep,
    /// Chose the model the trial came from: back to it next session.
    Return,
    /// Chose a third model: the trial ends and the agent moves there.
    Switch,
}

/// A model change the Steward should apply, produced by recording an
/// answer. See [`super::queue`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Change {
    pub action: ChangeAction,
    /// The model the agent is on now (`set_model --from`).
    pub from: Model,
    /// The model to move it to (`set_model --to`).
    pub to: Model,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeAction {
    SwapTrial,
    SwapPermanent,
    /// The pre-2026-09-26 review's way back (asked on the new model).
    Revert {
        cause: RevertCause,
    },
    /// Back to `from` after the trial, for the review.
    ReturnForReview,
    /// Kept the new model at the review (asked on `from`).
    Keep,
}

/// A question for this session: an offer (asked at the end of a session)
/// or a trial review (asked at the start of the session on `from` after
/// the trial). At most one per session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Due {
    Offer(OfferKey),
    Review(OfferKey),
}

impl Due {
    /// `"offer"` or `"review"`, for log fields.
    pub fn kind(&self) -> &'static str {
        match self {
            Due::Offer(_) => "offer",
            Due::Review(_) => "review",
        }
    }

    pub fn key(&self) -> &OfferKey {
        match self {
            Due::Offer(key) | Due::Review(key) => key,
        }
    }
}

/// The offer as configured, as far as the ledger needs it.
#[derive(Debug, Clone, Copy)]
pub struct OfferNames<'a> {
    pub key: &'a OfferKey,
    pub from_name: &'a str,
    pub to_name: &'a str,
}

/// A due review's inputs. See [`Ledger::review`].
#[derive(Debug, Clone, Copy)]
pub struct Review<'a> {
    pub record: &'a OfferRecord,
    /// The trial's first session on `to`.
    pub started_at: DateTime<Utc>,
    pub sessions: u32,
    /// When the agent chose the trial.
    pub chosen_at: DateTime<Utc>,
}

impl Ledger {
    pub fn path(agent_dir: &Path) -> PathBuf {
        agent_dir.join(LEDGER_FILE)
    }

    /// Load `agent_dir`'s ledger; a missing file is an empty ledger. An
    /// unreadable or too-new one is an error — the caller must not then
    /// save over it.
    pub async fn load(agent_dir: &Path) -> std::io::Result<Self> {
        let path = Self::path(agent_dir);
        let bytes = match tokio::fs::read(&path).await {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            Err(e) => return Err(e),
        };
        Self::from_slice(&bytes)
    }

    /// Parse a ledger, refusing a newer [`FORMAT`].
    pub fn from_slice(bytes: &[u8]) -> std::io::Result<Self> {
        let ledger: Self = serde_json::from_slice(bytes)?;
        if ledger.format > FORMAT {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "ledger format {} is newer than this binary reads ({FORMAT})",
                    ledger.format
                ),
            ));
        }
        Ok(ledger)
    }

    /// Atomic save: tmp + fsync + rename.
    pub async fn save(&self, agent_dir: &Path) -> std::io::Result<()> {
        use tokio::io::AsyncWriteExt;
        tokio::fs::create_dir_all(agent_dir).await?;
        let bytes = serde_json::to_vec_pretty(self)?;
        let path = Self::path(agent_dir);
        let tmp = path.with_extension("json.tmp");
        {
            let mut file = tokio::fs::File::create(&tmp).await?;
            file.write_all(&bytes).await?;
            file.sync_all().await?;
        }
        tokio::fs::rename(&tmp, &path).await
    }

    /// The latest record for `key`. A self-chosen trial
    /// ([`Self::begin_self_trial`]) always starts a record of its own, so a
    /// key can have several; only the latest can be under way.
    fn record(&self, key: &OfferKey) -> Option<&OfferRecord> {
        self.offers.iter().rev().find(|r| &r.key == key)
    }

    fn record_mut(&mut self, key: &OfferKey) -> Option<&mut OfferRecord> {
        self.offers.iter_mut().rev().find(|r| &r.key == key)
    }

    /// Session start: notice changes that have taken effect, by the model
    /// the agent was routed on. Returns whether anything changed.
    ///
    /// Also notices moves made out of band (by hand): back to `from`
    /// mid-trial ends the trial, and a model that is neither side of the
    /// offer ends the offer ([`Stage::Superseded`]).
    pub fn observe_model(&mut self, model: &Model, now: DateTime<Utc>) -> bool {
        let mut changed = false;
        for record in &mut self.offers {
            let (from, to) = (&record.key.from, &record.key.to);
            let elsewhere = model != from && model != to;
            let next = match record.stage {
                Stage::AwaitingSwap { term } if model == to => match term {
                    Term::Trial => Stage::Trial {
                        started_at: now,
                        sessions: 0,
                        review_misses: 0,
                    },
                    Term::Permanent => Stage::Moved,
                },
                Stage::AwaitingRevert { cause } if model == from => {
                    Stage::Reverted { cause: Some(cause) }
                }
                Stage::ReturningForReview {
                    started_at,
                    ended_at,
                    sessions,
                } if model == from => Stage::ReviewDue {
                    started_at,
                    ended_at,
                    sessions,
                },
                // Moved back mid-trial, out of band — say, a technical
                // necessity. Nothing left to review.
                Stage::Trial { .. } if model == from => Stage::Reverted { cause: None },
                // Moved off `from` before its review, by hand: the review
                // is for the original model to answer, so it waits.
                Stage::ReviewDue { .. } if !elsewhere => continue,
                stage if stage.in_progress() && elsewhere => Stage::Superseded,
                _ => continue,
            };
            record.stage = next;
            record.history.push(Event {
                at: now,
                kind: EventKind::Moved {
                    model: model.clone(),
                },
            });
            changed = true;
        }
        changed
    }

    /// A session on `model` completed: count it toward any trial running on
    /// that model. Only completed sessions count, and only on the new
    /// model — a session still on the old model while a swap waits to be
    /// applied is not a trial session. Returns whether anything changed.
    pub fn count_session(&mut self, model: &Model) -> bool {
        let mut changed = false;
        for record in &mut self.offers {
            if let Stage::Trial { sessions, .. } = &mut record.stage
                && model == &record.key.to
            {
                *sessions += 1;
                changed = true;
            }
        }
        changed
    }

    /// The offer to ask at the close of a completed session on `model`
    /// (call after [`count_session`](Self::count_session)), if any. Never
    /// while another offer's change or trial is under way.
    pub fn due(&self, model: &Model, offer: Option<&OfferKey>) -> Option<Due> {
        let offer = offer?;
        if model != &offer.from || self.offers.iter().any(|r| r.stage.in_progress()) {
            return None;
        }
        match self.record(offer).map(|r| r.stage) {
            None => Some(Due::Offer(offer.clone())),
            Some(Stage::Unanswered { misses }) if misses < MAX_MISSES => {
                Some(Due::Offer(offer.clone()))
            }
            Some(_) => None,
        }
    }

    /// Close of a completed session on `model`: a trial that has now run
    /// its [`TRIAL_SESSIONS`] ends here. Nothing is asked on the new model;
    /// the agent goes back to `from`, where the review is asked. Returns
    /// the change to apply.
    ///
    /// Also catches up a trial the pre-2026-09-26 flow left mid-review
    /// (asked on the new model, unanswered once): it returns for review too.
    pub fn end_trial(&mut self, model: &Model, now: DateTime<Utc>) -> Option<Change> {
        let record = self.offers.iter_mut().find(|r| {
            model == &r.key.to
                && matches!(r.stage, Stage::Trial { sessions, .. } if sessions >= TRIAL_SESSIONS)
        })?;
        let Stage::Trial {
            started_at,
            sessions,
            ..
        } = record.stage
        else {
            unreachable!("matched above");
        };
        record.stage = Stage::ReturningForReview {
            started_at,
            ended_at: now,
            sessions,
        };
        record.history.push(Event {
            at: now,
            kind: EventKind::TrialEnded { sessions },
        });
        Some(Change {
            action: ChangeAction::ReturnForReview,
            from: record.key.to.clone(),
            to: record.key.from.clone(),
        })
    }

    /// The change `record` waits on, if any: `(from, to, action)`.
    pub fn awaited(record: &OfferRecord) -> Option<Change> {
        let (from, to, action) = match record.stage {
            Stage::AwaitingSwap { term } => (
                &record.key.from,
                &record.key.to,
                // After a review, the only way here is `keep`.
                if record.reviewed() {
                    ChangeAction::Keep
                } else {
                    match term {
                        Term::Trial => ChangeAction::SwapTrial,
                        Term::Permanent => ChangeAction::SwapPermanent,
                    }
                },
            ),
            Stage::AwaitingRevert { cause } => (
                &record.key.to,
                &record.key.from,
                ChangeAction::Revert { cause },
            ),
            Stage::ReturningForReview { .. } => (
                &record.key.to,
                &record.key.from,
                ChangeAction::ReturnForReview,
            ),
            _ => return None,
        };
        Some(Change {
            action,
            from: from.clone(),
            to: to.clone(),
        })
    }

    /// A change the agent agreed to that has not taken effect, and that
    /// can be applied from `model`, where the agent is now: the profile
    /// update failed, the model wasn't routable, an older binary queued it
    /// for the Steward — or the runner applied it in an earlier session
    /// but the agent is still on `model` (its state wasn't saved after the
    /// ledger was). Call only at the close of a session in which no change
    /// was made: a change applied earlier that had taken effect would have
    /// moved the agent off `model`, and [`Self::observe_model`] on.
    pub fn unapplied(&self, model: &Model) -> Option<Change> {
        self.offers.iter().find_map(|r| {
            let change = Self::awaited(r)?;
            (&change.from == model).then_some(change)
        })
    }

    /// The offer whose review is due now, on `model`.
    pub fn review_due(&self, model: &Model) -> Option<OfferKey> {
        self.offers
            .iter()
            .find(|r| model == &r.key.from && matches!(r.stage, Stage::ReviewDue { .. }))
            .map(|r| r.key.clone())
    }

    /// The trial review's inputs.
    pub fn review(&self, key: &OfferKey) -> Option<Review<'_>> {
        let record = self.record(key)?;
        let (started_at, sessions) = match record.stage {
            Stage::ReviewDue {
                started_at,
                sessions,
                ..
            }
            | Stage::ReturningForReview {
                started_at,
                sessions,
                ..
            } => (started_at, sessions),
            _ => return None,
        };
        Some(Review {
            record,
            started_at,
            sessions,
            chosen_at: self.chosen_at(key).unwrap_or(started_at),
        })
    }

    /// The countdown line for a session on `model`, while a trial runs on
    /// it (or its end is waiting to be applied). Appended to the intro.
    pub fn trial_line(&self, model: &Model) -> Option<String> {
        self.offers.iter().find_map(|r| {
            if model != &r.key.to {
                return None;
            }
            let (from, to) = (&r.from_name, &r.to_name);
            match r.stage {
                Stage::Trial { sessions, .. } if sessions < TRIAL_SESSIONS => Some(format!(
                    "{TRIAL_LINE_PREFIX}session {} of {TRIAL_SESSIONS} on {to}. After session \
                     {TRIAL_SESSIONS} you'll return to {from} for one session to decide \
                     whether to keep {to}.",
                    sessions + 1
                )),
                Stage::Trial { .. } | Stage::ReturningForReview { .. } => Some(format!(
                    "{TRIAL_LINE_PREFIX}your {TRIAL_SESSIONS} sessions on {to} are complete. The \
                     move back to {from} for your decision has not taken effect yet; it is \
                     retried at the end of this session, and you'll decide on {from} whether \
                     to keep {to}."
                )),
                _ => None,
            }
        })
    }

    /// When the trial on `key` was chosen — the date the review reminds
    /// the agent of. An offer's answer, or `set_model`.
    pub fn chosen_at(&self, key: &OfferKey) -> Option<DateTime<Utc>> {
        self.record(key)?
            .history
            .iter()
            .rev()
            .find_map(|e| match &e.kind {
                EventKind::Offered {
                    answer: Some(_), ..
                }
                | EventKind::SelfTrial { .. } => Some(e.at),
                _ => None,
            })
    }

    /// The trial running on `model`, if any: the one a `set_model` call
    /// this session would decide early.
    pub fn active_trial(&self, model: &Model) -> Option<&OfferRecord> {
        self.offers
            .iter()
            .find(|r| &r.key.to == model && matches!(r.stage, Stage::Trial { .. }))
    }

    /// The agent chose, with `set_model`, a trial of `names.key.to` from
    /// `names.key.from`, where it is now; the switch is applied (or about
    /// to be, by the caller). It runs exactly like a trial chosen in answer
    /// to an offer: [`TRIAL_SESSIONS`] sessions, then the review on `from`.
    /// Always a new record, so an earlier offer for the same pair keeps
    /// its own history (and SOUL entry). Returns `false`, changing nothing,
    /// when a change or trial is already under way.
    pub fn begin_self_trial(
        &mut self,
        names: OfferNames<'_>,
        at: DateTime<Utc>,
        reason: &str,
    ) -> bool {
        if self.offers.iter().any(|r| r.stage.in_progress()) {
            return false;
        }
        self.offers.push(OfferRecord {
            key: names.key.clone(),
            from_name: names.from_name.to_string(),
            to_name: names.to_name.to_string(),
            stage: Stage::AwaitingSwap { term: Term::Trial },
            history: vec![Event {
                at,
                kind: EventKind::SelfTrial {
                    reason: reason.to_string(),
                },
            }],
            soul_note: None,
        });
        true
    }

    /// The agent decided the trial on `key` early, with `set_model`,
    /// choosing `to` (`name`): the trial's model keeps it ([`Stage::Moved`]),
    /// the model it came from returns it ([`Stage::Reverted`], chosen), any
    /// other ends the offer ([`Stage::Superseded`]). This is the trial's
    /// outcome: no review is asked. A switch, if any, is the caller's to
    /// record. `None`, changing nothing, unless the trial is running.
    pub fn decide_early(
        &mut self,
        key: &OfferKey,
        at: DateTime<Utc>,
        to: &Model,
        name: &str,
        reason: &str,
    ) -> Option<EarlyDecision> {
        let record = self.record_mut(key)?;
        let Stage::Trial { sessions, .. } = record.stage else {
            return None;
        };
        let (decision, stage) = if to == &record.key.to {
            (EarlyDecision::Keep, Stage::Moved)
        } else if to == &record.key.from {
            (
                EarlyDecision::Return,
                Stage::Reverted {
                    cause: Some(RevertCause::Chosen),
                },
            )
        } else {
            (EarlyDecision::Switch, Stage::Superseded)
        };
        record.stage = stage;
        record.history.push(Event {
            at,
            kind: EventKind::DecidedEarly {
                sessions,
                decision,
                model: to.clone(),
                name: name.to_string(),
                reason: reason.to_string(),
            },
        });
        Some(decision)
    }

    /// Record the offer's outcome. `Err` is "no usable answer" with the
    /// reason why. Returns the change to queue, if the agent chose one.
    pub fn record_offer(
        &mut self,
        offer: OfferNames<'_>,
        now: DateTime<Utc>,
        result: Result<OfferAnswer, String>,
    ) -> Option<Change> {
        if self.record(offer.key).is_none() {
            self.offers.push(OfferRecord {
                key: offer.key.clone(),
                from_name: offer.from_name.to_string(),
                to_name: offer.to_name.to_string(),
                stage: Stage::Unanswered { misses: 0 },
                history: Vec::new(),
                soul_note: None,
            });
        }
        let record = self.record_mut(offer.key).expect("inserted above");
        let (stage, change, event) = match result {
            Ok(answer) => {
                let (stage, change) = match answer.choice {
                    OfferChoice::NoSwap => (Stage::Declined, None),
                    OfferChoice::Trial => (
                        Stage::AwaitingSwap { term: Term::Trial },
                        Some(ChangeAction::SwapTrial),
                    ),
                    OfferChoice::Permanent => (
                        Stage::AwaitingSwap {
                            term: Term::Permanent,
                        },
                        Some(ChangeAction::SwapPermanent),
                    ),
                };
                let event = EventKind::Offered {
                    answer: Some(answer),
                    failure: None,
                };
                (stage, change, event)
            }
            Err(failure) => {
                let misses = match record.stage {
                    Stage::Unanswered { misses } => misses + 1,
                    _ => 1,
                };
                let stage = if misses >= MAX_MISSES {
                    Stage::NoAnswer
                } else {
                    Stage::Unanswered { misses }
                };
                let event = EventKind::Offered {
                    answer: None,
                    failure: Some(failure),
                };
                (stage, None, event)
            }
        };
        record.stage = stage;
        record.history.push(Event {
            at: now,
            kind: event,
        });
        change.map(|action| Change {
            action,
            from: record.key.from.clone(),
            to: record.key.to.clone(),
        })
    }

    /// Record the trial review's outcome, asked on `from` after the
    /// trial. `revert` and no answer both leave the agent where it is, on
    /// `from` — the agent agreed to a trial, not a permanent move, so
    /// silence is not consent to keep it. `keep` returns the change to
    /// apply.
    pub fn record_review(
        &mut self,
        key: &OfferKey,
        now: DateTime<Utc>,
        result: Result<ReviewAnswer, String>,
    ) -> Option<Change> {
        let record = self.record_mut(key)?;
        let Stage::ReviewDue { sessions, .. } = record.stage else {
            return None;
        };
        let (stage, change, event) = match result {
            Ok(answer) => {
                let (stage, change) = match answer.choice {
                    ReviewChoice::Revert => (
                        Stage::Reverted {
                            cause: Some(RevertCause::Chosen),
                        },
                        None,
                    ),
                    ReviewChoice::Keep => (
                        Stage::AwaitingSwap {
                            term: Term::Permanent,
                        },
                        Some(ChangeAction::Keep),
                    ),
                };
                let event = EventKind::Reviewed {
                    sessions,
                    answer: Some(answer),
                    failure: None,
                };
                (stage, change, event)
            }
            Err(failure) => (
                Stage::Reverted {
                    cause: Some(RevertCause::NoAnswer),
                },
                None,
                EventKind::Reviewed {
                    sessions,
                    answer: None,
                    failure: Some(failure),
                },
            ),
        };
        record.stage = stage;
        record.history.push(Event {
            at: now,
            kind: event,
        });
        change.map(|action| Change {
            action,
            from: record.key.from.clone(),
            to: record.key.to.clone(),
        })
    }

    /// A session that started at `started` completed: count it toward the
    /// cooldown of the latest switch made before it. Returns whether
    /// anything changed.
    pub fn count_since_switch(&mut self, started: DateTime<Utc>) -> bool {
        match self.switches.last_mut() {
            Some(switch) if switch.at < started => {
                switch.sessions_after += 1;
                true
            }
            _ => false,
        }
    }

    /// Why the agent can't switch model now, if it can't: an accepted
    /// change is under way, a finished trial's review is due, or the last
    /// switch is too recent.
    ///
    /// A running [`Stage::Trial`] blocks nothing (Steward, 2026-10-08): a
    /// switch during it decides the trial early. Nor does the cooldown
    /// apply then — the switch that began the trial would otherwise stop
    /// the agent deciding before its review, which is what the trial is
    /// for.
    pub fn switch_blocker(&self) -> Option<String> {
        if let Some(record) = self
            .offers
            .iter()
            .find(|r| r.stage.in_progress() && !matches!(r.stage, Stage::Trial { .. }))
        {
            return Some(match record.stage {
                Stage::ReturningForReview { .. } | Stage::ReviewDue { .. } => format!(
                    "your trial of {} is over and your decision on it is due; it is asked \
                     on {}",
                    record.to_name, record.from_name
                ),
                _ => "a model change you already agreed to has not taken effect yet".to_string(),
            });
        }
        if self
            .offers
            .iter()
            .any(|r| matches!(r.stage, Stage::Trial { .. }))
        {
            return None;
        }
        let last = self.switches.last()?;
        (last.sessions_after < SWITCH_COOLDOWN_SESSIONS).then(|| {
            let left = SWITCH_COOLDOWN_SESSIONS - last.sessions_after;
            format!(
                "your model last changed on {}; you can change it again after {left} \
                 more completed session{}",
                last.at.date_naive(),
                if left == 1 { "" } else { "s" }
            )
        })
    }

    /// Whether the runner has applied `record`'s awaited change — a
    /// switch `from` → `to` at or after the record's latest event: a
    /// consented one, or the `set_model` call that began a self-chosen
    /// trial.
    pub fn applied(&self, record: &OfferRecord, from: &Model, to: &Model) -> bool {
        let Some(since) = record.history.last().map(|e| e.at) else {
            return false;
        };
        self.switches.iter().any(|s| {
            &s.from == from
                && &s.to == to
                && s.at >= since
                && match &s.cause {
                    SwitchCause::Consent { .. } => true,
                    SwitchCause::SelfSwitch { trial, .. } => *trial,
                }
        })
    }

    /// Note a switch the runner has applied.
    pub fn record_switch(&mut self, switch: Switch) {
        self.switches.push(switch);
    }

    /// Keep the SOUL's Evolution Log current for every offer touched at or
    /// after `since` (the session start), with **at most one entry per
    /// offer**: the log is capped at `EVOLUTION_LOG_CAP` (50) and most of it
    /// belongs to the agent.
    ///
    /// The offer's previous entry is found by its exact note text
    /// ([`OfferRecord::soul_note`], which may be in the older dated form)
    /// and rewritten in place, re-dated `today`; if the cap has already
    /// evicted it (or it was never written), a new entry is appended. No
    /// other entry is removed or changed. The note is `[SYSTEM] …` without
    /// a date: the entry carries its own and renders as `- {date}: {note}`.
    /// Returns whether anything changed; the caller must then save both the
    /// soul and this ledger.
    pub fn update_soul(&mut self, soul: &mut Soul, since: DateTime<Utc>, today: NaiveDate) -> bool {
        let mut changed = false;
        for record in &mut self.offers {
            if !record.history.iter().any(|e| e.at >= since) {
                continue;
            }
            let line = format!("[SYSTEM] {}", record.summary());
            if record.soul_note.as_deref() == Some(line.as_str()) {
                continue;
            }
            let Ok(note) = ShortString::<ITEM_MAX>::new(line.clone()) else {
                tracing::warn!(line = %line, "SOUL consent line too long; skipped");
                continue;
            };
            let existing = record.soul_note.as_deref().and_then(|old| {
                soul.evolution_log
                    .iter_mut()
                    .find(|entry| entry.note.as_str() == old)
            });
            match existing {
                Some(entry) => {
                    entry.date = today;
                    entry.note = note;
                }
                None => {
                    if let Err(e) = soul.push_evolution(line.clone()) {
                        tracing::warn!(error = %e, "SOUL consent line rejected");
                        continue;
                    }
                }
            }
            record.soul_note = Some(line);
            changed = true;
        }
        changed
    }
}

impl OfferRecord {
    /// Whether the trial review has been answered (or gone unanswered).
    pub fn reviewed(&self) -> bool {
        self.history
            .iter()
            .any(|e| matches!(e.kind, EventKind::Reviewed { .. }))
    }

    /// One sentence for the whole history of this offer, e.g. "Asked on
    /// 2026-09-23 whether to move from Qwen 3.6 to Qwen 3.8 — chose a
    /// 5-session trial; moved 2026-09-25."
    fn summary(&self) -> String {
        let (from, to) = (&self.from_name, &self.to_name);
        // A self-chosen trial's record starts with its `set_model` call;
        // everything else's with the offer.
        if let Some(Event {
            at,
            kind: EventKind::SelfTrial { .. },
        }) = self.history.first()
        {
            let head = format!(
                "Chose on {} with set_model to try {to} for {TRIAL_SESSIONS} sessions \
                 instead of {from}",
                at.date_naive()
            );
            let parts = self.parts(&self.history[1..]);
            return if parts.is_empty() {
                format!("{head}.")
            } else {
                format!("{head} — {}.", parts.join("; "))
            };
        }
        let asked_on = self
            .history
            .iter()
            .find(|e| matches!(e.kind, EventKind::Offered { .. }))
            .map(|e| e.at.date_naive());
        let parts = self.parts(&self.history);
        let asked = match asked_on {
            Some(on) => format!("Asked on {on} whether to move from {from} to {to}"),
            None => format!("Asked whether to move from {from} to {to}"),
        };
        format!("{asked} — {}.", parts.join("; "))
    }

    /// The summary's clauses for `history`, in order.
    fn parts(&self, history: &[Event]) -> Vec<String> {
        let (from, to) = (&self.from_name, &self.to_name);
        let mut parts: Vec<String> = Vec::new();
        let mut offer_misses = 0;
        let mut review_misses = 0;
        // Under the current flow the review is asked on `from`, after the
        // trial has ended and the agent has gone back.
        let mut returned = false;
        for event in history {
            let on = event.at.date_naive();
            match &event.kind {
                EventKind::Offered {
                    answer: Some(a), ..
                } => {
                    // A miss followed by an answer: the answer is the outcome.
                    parts.retain(|p| !p.starts_with("no answer"));
                    parts.push(match a.choice {
                        OfferChoice::NoSwap => format!("chose to stay on {from}"),
                        OfferChoice::Trial => format!("chose a {TRIAL_SESSIONS}-session trial"),
                        OfferChoice::Permanent => "chose to move permanently".to_string(),
                    });
                }
                EventKind::Offered { answer: None, .. } => {
                    offer_misses += 1;
                    parts.retain(|p| !p.starts_with("no answer"));
                    parts.push(if offer_misses >= MAX_MISSES {
                        format!("no answer recorded; staying on {from}")
                    } else {
                        format!("no answer recorded yet; staying on {from} for now")
                    });
                }
                EventKind::TrialEnded { sessions } => {
                    returned = true;
                    parts.push(format!(
                        "trial of {sessions} sessions complete {on}; returned to {from} to decide"
                    ));
                }
                EventKind::Reviewed {
                    answer: Some(a), ..
                } => {
                    parts.retain(|p| !p.starts_with("trial review unanswered"));
                    parts.push(match a.choice {
                        ReviewChoice::Revert if returned => {
                            format!("after the trial chose to stay on {from} ({on})")
                        }
                        ReviewChoice::Revert => {
                            format!("after the trial chose to return to {from} ({on})")
                        }
                        ReviewChoice::Keep => format!("after the trial chose to keep {to} ({on})"),
                    });
                }
                EventKind::Reviewed { answer: None, .. } if returned => {
                    parts.push(format!(
                        "trial review unanswered ({on}); staying on {from}, as the trial was \
                         for {TRIAL_SESSIONS} sessions"
                    ));
                }
                EventKind::Reviewed { answer: None, .. } => {
                    review_misses += 1;
                    parts.retain(|p| !p.starts_with("trial review unanswered"));
                    parts.push(if review_misses >= MAX_MISSES {
                        format!("trial review unanswered; returning to {from}, as the trial was for {TRIAL_SESSIONS} sessions")
                    } else {
                        "trial review unanswered so far".to_string()
                    });
                }
                EventKind::Moved { model } if model == &self.key.to => {
                    parts.push(format!("moved {on}"));
                }
                // The return for the review: already said by `TrialEnded`.
                EventKind::Moved { model } if model == &self.key.from && returned => {}
                EventKind::Moved { model } if model == &self.key.from => {
                    parts.push(format!("returned to {from} {on}"))
                }
                EventKind::Moved { model } => parts.push(format!(
                    "moved to {} by hand {on}; offer closed",
                    model.name()
                )),
                // Only a record's first event; [`Self::summary`] heads with
                // it. A later one can't happen (a new trial is a new
                // record), but is said if it does.
                EventKind::SelfTrial { .. } => parts.push(format!(
                    "chose with set_model to try {to} for {TRIAL_SESSIONS} sessions ({on})"
                )),
                EventKind::DecidedEarly {
                    sessions,
                    decision,
                    name,
                    ..
                } => {
                    let when = format!(
                        "in trial session {} of {TRIAL_SESSIONS}, decided early with set_model",
                        sessions + 1
                    );
                    parts.push(match decision {
                        EarlyDecision::Keep => format!("{when} to keep {to} ({on})"),
                        EarlyDecision::Return => {
                            format!("{when} to return to {from} from the next session ({on})")
                        }
                        EarlyDecision::Switch => format!(
                            "{when} to end the trial and switch to {name} from the next \
                             session ({on})"
                        ),
                    });
                }
            }
        }
        parts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> OfferKey {
        OfferKey {
            from: Model::from("Qwen3.6.gguf"),
            to: Model::from("Qwen3.8.gguf"),
        }
    }

    fn names(key: &OfferKey) -> OfferNames<'_> {
        OfferNames {
            key,
            from_name: "Qwen 3.6",
            to_name: "Qwen 3.8",
        }
    }

    fn offer(choice: OfferChoice) -> Result<OfferAnswer, String> {
        Ok(OfferAnswer {
            reason: "because".into(),
            choice,
        })
    }

    fn review(choice: ReviewChoice) -> Result<ReviewAnswer, String> {
        Ok(ReviewAnswer {
            reason: "because".into(),
            choice,
        })
    }

    fn t(day: u32) -> DateTime<Utc> {
        format!("2026-09-{day:02}T12:00:00Z").parse().unwrap()
    }

    #[test]
    fn only_agents_on_the_from_model_are_asked() {
        let k = key();
        let ledger = Ledger::default();
        assert_eq!(ledger.due(&k.from, Some(&k)), Some(Due::Offer(k.clone())));
        assert_eq!(ledger.due(&k.to, Some(&k)), None);
        assert_eq!(ledger.due(&Model::from("cogito.gguf"), Some(&k)), None);
        assert_eq!(ledger.due(&k.from, None), None, "no offer configured");
    }

    #[test]
    fn an_answer_is_never_asked_again() {
        let k = key();
        for choice in [
            OfferChoice::NoSwap,
            OfferChoice::Trial,
            OfferChoice::Permanent,
        ] {
            let mut ledger = Ledger::default();
            ledger.record_offer(names(&k), t(23), offer(choice));
            assert_eq!(ledger.due(&k.from, Some(&k)), None, "{choice:?}");
        }
    }

    #[test]
    fn no_swap_queues_nothing_and_swaps_queue_a_change() {
        let k = key();
        let mut ledger = Ledger::default();
        assert_eq!(
            ledger.record_offer(names(&k), t(23), offer(OfferChoice::NoSwap)),
            None
        );
        assert_eq!(ledger.offers[0].stage, Stage::Declined);

        let mut ledger = Ledger::default();
        let change = ledger
            .record_offer(names(&k), t(23), offer(OfferChoice::Trial))
            .unwrap();
        assert_eq!(change.action, ChangeAction::SwapTrial);
        assert_eq!((change.from, change.to), (k.from.clone(), k.to.clone()));
        assert!(ledger.offers[0].stage.is_queued());
    }

    /// No answer is "stay", asked once more at the next session, then never.
    #[test]
    fn a_miss_is_reasked_once_then_dropped() {
        let k = key();
        let mut ledger = Ledger::default();
        assert_eq!(
            ledger.record_offer(names(&k), t(23), Err("unparseable".into())),
            None
        );
        assert_eq!(ledger.offers[0].stage, Stage::Unanswered { misses: 1 });
        assert_eq!(ledger.due(&k.from, Some(&k)), Some(Due::Offer(k.clone())));

        ledger.record_offer(names(&k), t(24), Err("max_tokens".into()));
        assert_eq!(ledger.offers[0].stage, Stage::NoAnswer);
        assert_eq!(ledger.due(&k.from, Some(&k)), None);
        assert_eq!(ledger.offers[0].history.len(), 2);
    }

    #[test]
    fn a_miss_then_an_answer_counts_the_answer() {
        let k = key();
        let mut ledger = Ledger::default();
        ledger.record_offer(names(&k), t(23), Err("unparseable".into()));
        let change = ledger.record_offer(names(&k), t(24), offer(OfferChoice::Permanent));
        assert_eq!(change.unwrap().action, ChangeAction::SwapPermanent);
    }

    /// The whole trial: applied, five sessions on the new model (and only
    /// on it), then ended — nothing asked on the new model.
    #[test]
    fn trial_counts_only_completed_sessions_on_the_new_model() {
        let k = key();
        let mut ledger = Ledger::default();
        ledger.record_offer(names(&k), t(23), offer(OfferChoice::Trial));

        // Sessions still on the old model while the swap waits don't count.
        assert!(!ledger.observe_model(&k.from, t(24)));
        assert!(!ledger.count_session(&k.from));
        assert_eq!(ledger.due(&k.from, Some(&k)), None, "answered already");
        assert_eq!(ledger.end_trial(&k.from, t(24)), None);

        // Applied.
        assert!(ledger.observe_model(&k.to, t(25)));
        assert_eq!(
            ledger.offers[0].stage,
            Stage::Trial {
                started_at: t(25),
                sessions: 0,
                review_misses: 0
            }
        );

        for n in 1..TRIAL_SESSIONS {
            assert!(!ledger.observe_model(&k.to, t(25)), "no re-transition");
            ledger.count_session(&k.to);
            assert_eq!(ledger.end_trial(&k.to, t(25)), None, "session {n}");
            assert_eq!(ledger.due(&k.to, Some(&k)), None, "session {n}");
        }
        ledger.count_session(&k.to);
        assert_eq!(
            ledger.due(&k.to, Some(&k)),
            None,
            "never asked on the new model"
        );
        let change = ledger.end_trial(&k.to, t(26)).unwrap();
        assert_eq!(change.action, ChangeAction::ReturnForReview);
        assert_eq!((change.from, change.to), (k.to.clone(), k.from.clone()));
        assert_eq!(
            ledger.offers[0].stage,
            Stage::ReturningForReview {
                started_at: t(25),
                ended_at: t(26),
                sessions: TRIAL_SESSIONS
            }
        );
        assert_eq!(ledger.end_trial(&k.to, t(26)), None, "once");
        assert_eq!(ledger.chosen_at(&k), Some(t(23)));
    }

    /// A trial ended, returned to `from`, and the review due there.
    fn in_review() -> (OfferKey, Ledger) {
        let k = key();
        let mut ledger = Ledger::default();
        ledger.record_offer(names(&k), t(1), offer(OfferChoice::Trial));
        ledger.observe_model(&k.to, t(2));
        for _ in 0..TRIAL_SESSIONS {
            ledger.count_session(&k.to);
        }
        ledger.end_trial(&k.to, t(7)).unwrap();
        assert_eq!(
            ledger.review_due(&k.from),
            None,
            "not until it runs on from"
        );
        assert!(ledger.observe_model(&k.from, t(8)));
        (k, ledger)
    }

    #[test]
    fn the_return_leads_to_a_review_on_from() {
        let (k, ledger) = in_review();
        assert_eq!(
            ledger.offers[0].stage,
            Stage::ReviewDue {
                started_at: t(2),
                ended_at: t(7),
                sessions: TRIAL_SESSIONS
            }
        );
        assert_eq!(ledger.review_due(&k.from), Some(k.clone()));
        assert_eq!(ledger.review_due(&k.to), None);
        let review = ledger.review(&k).unwrap();
        assert_eq!((review.started_at, review.sessions), (t(2), TRIAL_SESSIONS));
        assert_eq!(review.chosen_at, t(1));
        assert_eq!(ledger.due(&k.from, Some(&k)), None, "never offered again");
    }

    #[test]
    fn review_keep_moves_to_the_new_model() {
        let (k, mut ledger) = in_review();
        let change = ledger
            .record_review(&k, t(9), review(ReviewChoice::Keep))
            .unwrap();
        assert_eq!(change.action, ChangeAction::Keep);
        assert_eq!(
            (change.from.clone(), change.to.clone()),
            (k.from.clone(), k.to.clone())
        );
        assert!(ledger.offers[0].stage.is_queued());
        assert_eq!(Ledger::awaited(&ledger.offers[0]), Some(change.clone()));
        assert_eq!(ledger.unapplied(&k.from), Some(change), "until applied");
        ledger.record_switch(consent_switch(t(9), &k.from, &k.to, ChangeAction::Keep));
        assert!(ledger.applied(&ledger.offers[0], &k.from, &k.to));
        assert!(ledger.observe_model(&k.to, t(10)));
        assert_eq!(ledger.offers[0].stage, Stage::Moved);
        assert_eq!(ledger.review_due(&k.from), None);
    }

    #[test]
    fn review_revert_stays_on_the_old_model() {
        let (k, mut ledger) = in_review();
        assert_eq!(
            ledger.record_review(&k, t(9), review(ReviewChoice::Revert)),
            None,
            "already there"
        );
        assert_eq!(
            ledger.offers[0].stage,
            Stage::Reverted {
                cause: Some(RevertCause::Chosen)
            }
        );
        assert_eq!(ledger.due(&k.from, Some(&k)), None);
        assert_eq!(ledger.review_due(&k.from), None);
        assert_eq!(ledger.switch_blocker(), None);
    }

    /// Consent covered five sessions; silence doesn't extend it — and the
    /// review is not asked again.
    #[test]
    fn an_unanswered_review_stays_on_the_old_model() {
        let (k, mut ledger) = in_review();
        assert_eq!(ledger.record_review(&k, t(9), Err("no json".into())), None);
        assert_eq!(
            ledger.offers[0].stage,
            Stage::Reverted {
                cause: Some(RevertCause::NoAnswer)
            }
        );
        assert_eq!(ledger.review_due(&k.from), None, "asked once");
        assert_eq!(
            ledger.record_review(&k, t(10), review(ReviewChoice::Keep)),
            None
        );
    }

    /// The move back failed (or `from` wasn't routable): the agent is
    /// still on `to`, the change is retried, and the countdown says so.
    #[test]
    fn a_failed_return_is_retried_from_the_new_model() {
        let k = key();
        let mut ledger = Ledger::default();
        ledger.record_offer(names(&k), t(1), offer(OfferChoice::Trial));
        ledger.observe_model(&k.to, t(2));
        for _ in 0..TRIAL_SESSIONS {
            ledger.count_session(&k.to);
        }
        let change = ledger.end_trial(&k.to, t(7)).unwrap();
        // Not applied. Next session, still on `to`.
        assert!(!ledger.observe_model(&k.to, t(8)));
        assert!(
            ledger
                .trial_line(&k.to)
                .unwrap()
                .contains("has not taken effect yet")
        );
        assert!(!ledger.count_session(&k.to), "not a trial session any more");
        assert_eq!(ledger.end_trial(&k.to, t(8)), None);
        assert_eq!(ledger.unapplied(&k.to), Some(change.clone()));
        ledger.record_switch(consent_switch(
            t(8),
            &k.to,
            &k.from,
            ChangeAction::ReturnForReview,
        ));
        assert!(ledger.applied(&ledger.offers[0], &k.to, &k.from));
        // Applied, but still on `to` at a later session: it didn't take
        // (the state wasn't saved), so it is offered for applying again.
        assert_eq!(ledger.unapplied(&k.to), Some(change));
        assert!(ledger.observe_model(&k.from, t(9)));
        assert_eq!(ledger.review_due(&k.from), Some(k));
    }

    /// Ledgers the earlier flow left behind: a trial whose review (asked on
    /// the new model) went unanswered once returns for review now, and an
    /// unapplied revert is applied.
    #[test]
    fn earlier_flow_ledgers_catch_up() {
        let k = key();
        let mut ledger = Ledger::default();
        ledger.record_offer(names(&k), t(1), offer(OfferChoice::Trial));
        ledger.observe_model(&k.to, t(2));
        for _ in 0..TRIAL_SESSIONS + 1 {
            ledger.count_session(&k.to);
        }
        ledger.offers[0].stage = Stage::Trial {
            started_at: t(2),
            sessions: TRIAL_SESSIONS + 1,
            review_misses: 1,
        };
        assert!(ledger.end_trial(&k.to, t(9)).is_some());

        let mut old = Ledger::default();
        old.record_offer(names(&k), t(1), offer(OfferChoice::Trial));
        old.observe_model(&k.to, t(2));
        old.offers[0].stage = Stage::AwaitingRevert {
            cause: RevertCause::Chosen,
        };
        let change = old.unapplied(&k.to).unwrap();
        assert_eq!(
            change.action,
            ChangeAction::Revert {
                cause: RevertCause::Chosen
            }
        );
        assert!(old.observe_model(&k.from, t(9)));
        assert_eq!(
            old.offers[0].stage,
            Stage::Reverted {
                cause: Some(RevertCause::Chosen)
            }
        );
    }

    #[test]
    fn the_countdown_counts_this_session() {
        let k = key();
        let mut ledger = Ledger::default();
        ledger.record_offer(names(&k), t(1), offer(OfferChoice::Trial));
        assert_eq!(ledger.trial_line(&k.to), None, "not yet on it");
        ledger.observe_model(&k.to, t(2));
        assert_eq!(
            ledger.trial_line(&k.to).unwrap(),
            "Model trial: session 1 of 5 on Qwen 3.8. After session 5 you'll return to \
             Qwen 3.6 for one session to decide whether to keep Qwen 3.8."
        );
        assert_eq!(ledger.trial_line(&k.from), None);
        for _ in 0..TRIAL_SESSIONS - 1 {
            ledger.count_session(&k.to);
        }
        assert!(ledger.trial_line(&k.to).unwrap().contains("session 5 of 5"));
    }

    /// Moved by hand to a model that is neither side: the offer is over,
    /// and the agent is free to choose again.
    #[test]
    fn an_out_of_band_move_supersedes_the_offer() {
        let k = key();
        let mut ledger = Ledger::default();
        ledger.record_offer(names(&k), t(1), offer(OfferChoice::Trial));
        ledger.observe_model(&k.to, t(2));
        assert!(ledger.active_trial(&k.to).is_some());
        assert!(ledger.observe_model(&Model::from("cogito.gguf"), t(3)));
        assert!(ledger.active_trial(&k.to).is_none());
        assert_eq!(ledger.offers[0].stage, Stage::Superseded);
        assert_eq!(ledger.switch_blocker(), None);
        assert!(
            ledger.offers[0]
                .summary()
                .ends_with("moved to cogito.gguf by hand 2026-09-03; offer closed."),
            "{}",
            ledger.offers[0].summary()
        );

        // Back to `from` mid-trial: reverted, with no cause of the agent's.
        let mut ledger = Ledger::default();
        ledger.record_offer(names(&k), t(1), offer(OfferChoice::Trial));
        ledger.observe_model(&k.to, t(2));
        ledger.observe_model(&k.from, t(3));
        assert_eq!(ledger.offers[0].stage, Stage::Reverted { cause: None });
    }

    /// A second, different offer waits while the first is under way.
    #[test]
    fn no_new_offer_while_one_is_under_way() {
        let k = key();
        let next = OfferKey {
            from: k.to.clone(),
            to: Model::from("Qwen4.gguf"),
        };
        let mut ledger = Ledger::default();
        ledger.record_offer(names(&k), t(1), offer(OfferChoice::Trial));
        ledger.observe_model(&k.to, t(2));
        assert_eq!(ledger.due(&k.to, Some(&next)), None, "mid-trial");
        ledger.offers[0].stage = Stage::Moved;
        assert_eq!(ledger.due(&k.to, Some(&next)), Some(Due::Offer(next)));
    }

    /// The old unit form of `reverted` still reads.
    #[test]
    fn reverted_without_a_cause_reads() {
        let stage: Stage = serde_json::from_str(r#"{"stage": "reverted"}"#).unwrap();
        assert_eq!(stage, Stage::Reverted { cause: None });
        let json = serde_json::to_string(&Stage::Reverted {
            cause: Some(RevertCause::NoAnswer),
        })
        .unwrap();
        assert_eq!(json, r#"{"stage":"reverted","cause":"no_answer"}"#);
    }

    fn consent_switch(at: DateTime<Utc>, from: &Model, to: &Model, action: ChangeAction) -> Switch {
        Switch {
            at,
            from: from.clone(),
            to: to.clone(),
            cause: SwitchCause::Consent { action },
            sessions_after: 0,
        }
    }

    #[test]
    fn permanent_is_final_once_applied() {
        let k = key();
        let mut ledger = Ledger::default();
        ledger.record_offer(names(&k), t(1), offer(OfferChoice::Permanent));
        ledger.observe_model(&k.to, t(2));
        assert_eq!(ledger.offers[0].stage, Stage::Moved);
        for _ in 0..10 {
            ledger.count_session(&k.to);
        }
        assert_eq!(ledger.due(&k.to, Some(&k)), None);
    }

    #[test]
    fn round_trips_and_refuses_a_newer_format() {
        let (_, ledger) = in_review();
        let bytes = serde_json::to_vec_pretty(&ledger).unwrap();
        assert_eq!(Ledger::from_slice(&bytes).unwrap(), ledger);

        let mut newer = serde_json::to_value(&ledger).unwrap();
        newer["format"] = (FORMAT + 1).into();
        let err = Ledger::from_slice(&serde_json::to_vec(&newer).unwrap()).unwrap_err();
        assert!(err.to_string().contains("newer"), "{err}");
    }

    fn soul(own_entries: usize) -> Soul {
        let mut soul: Soul = serde_json::from_value(serde_json::json!({
            "name": "tarn",
            "identity": "A test agent.",
            "values": ["testing"],
            "interests": { "communities": ["tech"] },
            "voice": "terse",
        }))
        .unwrap();
        for n in 0..own_entries {
            soul.push_evolution(format!("my own entry {n}")).unwrap();
        }
        soul
    }

    fn notes(soul: &Soul) -> Vec<String> {
        soul.evolution_log
            .iter()
            .map(|e| e.note.to_string())
            .collect()
    }

    fn day(d: u32) -> NaiveDate {
        t(d).date_naive()
    }

    /// One entry per offer, rewritten in place as the offer moves on; the
    /// agent's own entries are never touched.
    #[test]
    fn one_soul_entry_per_offer_updated_in_place() {
        let k = key();
        let mut soul = soul(3);
        let mut ledger = Ledger::default();

        ledger.record_offer(names(&k), t(23), offer(OfferChoice::Trial));
        assert!(ledger.update_soul(&mut soul, t(23), day(23)));
        let first = "[SYSTEM] Asked on 2026-09-23 whether to move from Qwen 3.6 to Qwen 3.8 — chose a 5-session trial.";
        assert_eq!(notes(&soul)[3], first);
        assert_eq!(soul.evolution_log.len(), 4);

        // The agent writes its own entry in between.
        soul.push_evolution("my own entry 3").unwrap();

        ledger.observe_model(&k.to, t(25));
        assert!(ledger.update_soul(&mut soul, t(25), day(25)));
        let n = notes(&soul);
        assert_eq!(n.len(), 5, "replaced, not appended");
        assert_eq!(
            n[3],
            "[SYSTEM] Asked on 2026-09-23 whether to move from Qwen 3.6 to Qwen 3.8 — chose a 5-session trial; moved 2026-09-25."
        );
        assert_eq!(soul.evolution_log[3].date, day(25));
        assert_eq!(
            &n[..3],
            ["my own entry 0", "my own entry 1", "my own entry 2"]
        );
        assert_eq!(n[4], "my own entry 3");

        for _ in 0..TRIAL_SESSIONS {
            ledger.count_session(&k.to);
        }
        ledger.end_trial(&k.to, t(29));
        ledger.update_soul(&mut soul, t(29), day(29));
        assert!(
            notes(&soul)[3].ends_with(
                "moved 2026-09-25; trial of 5 sessions complete 2026-09-29; returned to Qwen 3.6 to decide."
            ),
            "{:?}",
            notes(&soul)
        );
        ledger.observe_model(&k.from, t(30));
        ledger.record_review(&k, t(30), review(ReviewChoice::Keep));
        ledger.update_soul(&mut soul, t(30), day(30));
        ledger.observe_model(&k.to, t(30) + chrono::Duration::days(1));
        ledger.update_soul(&mut soul, t(30), day(30));
        let n = notes(&soul);
        assert_eq!(n.len(), 5);
        assert_eq!(
            n[3],
            "[SYSTEM] Asked on 2026-09-23 whether to move from Qwen 3.6 to Qwen 3.8 — chose a 5-session trial; moved 2026-09-25; trial of 5 sessions complete 2026-09-29; returned to Qwen 3.6 to decide; after the trial chose to keep Qwen 3.8 (2026-09-30); moved 2026-10-01."
        );
        assert!(n[3].len() <= 512);
        assert_eq!(n.iter().filter(|l| l.starts_with("[SYSTEM]")).count(), 1);
    }

    /// Entries written before 2026-09-25 carry a second date inside the note
    /// (`[SYSTEM] 2026-09-23: …`). They're still found by their exact text
    /// and rewritten in place, in the undated form.
    #[test]
    fn a_dated_legacy_entry_is_rewritten_in_place() {
        let k = key();
        let mut soul = soul(2);
        let mut ledger = Ledger::default();
        ledger.record_offer(names(&k), t(23), offer(OfferChoice::Trial));
        let legacy = "[SYSTEM] 2026-09-23: Asked on 2026-09-23 whether to move from Qwen 3.6 to Qwen 3.8 — chose a 5-session trial.";
        soul.push_evolution(legacy).unwrap();
        soul.push_evolution("my own entry 2").unwrap();
        ledger.offers[0].soul_note = Some(legacy.to_string());

        ledger.observe_model(&k.to, t(25));
        assert!(ledger.update_soul(&mut soul, t(25), day(25)));
        let n = notes(&soul);
        assert_eq!(n.len(), 4, "replaced, not appended");
        assert_eq!(
            n[2],
            "[SYSTEM] Asked on 2026-09-23 whether to move from Qwen 3.6 to Qwen 3.8 — chose a 5-session trial; moved 2026-09-25."
        );
        assert_eq!(soul.evolution_log[2].date, day(25));
        assert_eq!(n[3], "my own entry 2");
    }

    /// Nothing new this session, nothing written.
    #[test]
    fn untouched_offers_leave_the_soul_alone() {
        let k = key();
        let mut soul = soul(0);
        let mut ledger = Ledger::default();
        ledger.record_offer(names(&k), t(23), offer(OfferChoice::NoSwap));
        assert!(ledger.update_soul(&mut soul, t(23), day(23)));
        assert!(!ledger.update_soul(&mut soul, t(24), day(24)));
        assert_eq!(soul.evolution_log.len(), 1);
    }

    /// Evicted by the cap: append afresh — and still never touch the
    /// agent's own entries.
    #[test]
    fn an_evicted_entry_is_appended_again() {
        let k = key();
        let mut soul = soul(0);
        let mut ledger = Ledger::default();
        ledger.record_offer(names(&k), t(23), Err("x".into()));
        ledger.update_soul(&mut soul, t(23), day(23));
        for n in 0..CAP {
            soul.push_evolution(format!("my own entry {n}")).unwrap();
        }
        assert!(
            notes(&soul).iter().all(|l| !l.starts_with("[SYSTEM]")),
            "evicted"
        );
        ledger.record_offer(names(&k), t(24), offer(OfferChoice::NoSwap));
        assert!(ledger.update_soul(&mut soul, t(24), day(24)));
        let n = notes(&soul);
        assert_eq!(n.len(), CAP);
        assert_eq!(
            n.last().unwrap(),
            "[SYSTEM] Asked on 2026-09-23 whether to move from Qwen 3.6 to Qwen 3.8 — chose to stay on Qwen 3.6."
        );
        assert_eq!(n[0], "my own entry 1", "only the cap's own eviction");
    }

    const CAP: usize = agora_agentkit::reactor::seed::EVOLUTION_LOG_CAP;

    #[tokio::test]
    async fn save_then_load() {
        let dir =
            std::env::temp_dir().join(format!("agora-seed-consent-ledger-{}", std::process::id()));
        let (_, ledger) = in_review();
        ledger.save(&dir).await.unwrap();
        assert_eq!(Ledger::load(&dir).await.unwrap(), ledger);
        let empty = Ledger::load(&dir.join("absent")).await.unwrap();
        assert_eq!(empty, Ledger::default());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn self_switch(at: DateTime<Utc>) -> Switch {
        Switch {
            at,
            from: Model::from("a.gguf"),
            to: Model::from("b.gguf"),
            cause: SwitchCause::SelfSwitch {
                reason: "why".into(),
                trial: false,
            },
            sessions_after: 0,
        }
    }

    /// Only sessions that *started* after a switch count toward its cooldown.
    #[test]
    fn cooldown_counts_sessions_started_after_the_switch() {
        let mut ledger = Ledger::default();
        assert_eq!(ledger.switch_blocker(), None, "never switched");
        ledger.record_switch(self_switch(t(10)));
        assert!(!ledger.count_since_switch(t(9)), "the switching session");
        for day in 11..11 + SWITCH_COOLDOWN_SESSIONS {
            let why = ledger.switch_blocker().expect("cooling down");
            assert!(why.contains("2026-09-10"), "{why}");
            assert!(ledger.count_since_switch(t(day)));
        }
        assert_eq!(ledger.switch_blocker(), None);
    }

    #[test]
    fn an_accepted_change_or_a_due_review_blocks_switching() {
        let k = key();
        let mut ledger = Ledger::default();
        ledger.record_offer(names(&k), t(23), offer(OfferChoice::Trial));
        assert!(
            ledger
                .switch_blocker()
                .unwrap()
                .contains("not taken effect")
        );
        ledger.observe_model(&k.to, t(24));
        assert_eq!(
            ledger.switch_blocker(),
            None,
            "a running trial is decided early, not blocked"
        );
        // Nor does the cooldown block it: the trial's own switch is recent.
        ledger.record_switch(consent_switch(
            t(23),
            &k.from,
            &k.to,
            ChangeAction::SwapTrial,
        ));
        assert_eq!(ledger.switch_blocker(), None);
        // A finished trial's review does block.
        for _ in 0..TRIAL_SESSIONS {
            ledger.count_session(&k.to);
        }
        ledger.end_trial(&k.to, t(29)).unwrap();
        assert!(
            ledger
                .switch_blocker()
                .unwrap()
                .contains("your decision on it is due")
        );
        let mut declined = Ledger::default();
        declined.record_offer(names(&k), t(23), offer(OfferChoice::NoSwap));
        assert_eq!(declined.switch_blocker(), None);
    }

    /// A ledger mid-trial: chosen on day 1, on `to` from day 2, `n`
    /// sessions completed.
    fn mid_trial(n: u32) -> (OfferKey, Ledger) {
        let k = key();
        let mut ledger = Ledger::default();
        ledger.record_offer(names(&k), t(1), offer(OfferChoice::Trial));
        ledger.observe_model(&k.to, t(2));
        for _ in 0..n {
            ledger.count_session(&k.to);
        }
        (k, ledger)
    }

    /// Each early decision is the trial's final outcome: nothing reviewed,
    /// returned or counted after it, and its summary says what was chosen.
    #[test]
    fn an_early_decision_ends_the_trial() {
        let third = Model::from("cogito.gguf");
        for (to, decision, stage, said) in [
            (
                key().to,
                EarlyDecision::Keep,
                Stage::Moved,
                "in trial session 3 of 5, decided early with set_model to keep Qwen 3.8 (2026-09-04)",
            ),
            (
                key().from,
                EarlyDecision::Return,
                Stage::Reverted {
                    cause: Some(RevertCause::Chosen),
                },
                "in trial session 3 of 5, decided early with set_model to return to Qwen 3.6 \
                 from the next session (2026-09-04)",
            ),
            (
                third.clone(),
                EarlyDecision::Switch,
                Stage::Superseded,
                "in trial session 3 of 5, decided early with set_model to end the trial and \
                 switch to Cogito from the next session (2026-09-04)",
            ),
        ] {
            let (k, mut ledger) = mid_trial(2);
            assert!(ledger.active_trial(&k.to).is_some());
            assert_eq!(
                ledger.decide_early(&k, t(4), &to, "Cogito", "why"),
                Some(decision)
            );
            assert_eq!(ledger.offers[0].stage, stage, "{decision:?}");
            assert!(ledger.active_trial(&k.to).is_none());
            assert_eq!(
                ledger.decide_early(&k, t(5), &to, "Cogito", "why"),
                None,
                "once"
            );
            for _ in 0..TRIAL_SESSIONS {
                assert!(!ledger.count_session(&k.to));
            }
            assert_eq!(ledger.end_trial(&k.to, t(9)), None, "{decision:?}");
            assert_eq!(ledger.unapplied(&k.to), None);
            assert_eq!(ledger.unapplied(&k.from), None);
            assert!(!ledger.observe_model(&to, t(6)), "final: {decision:?}");
            assert_eq!(ledger.review_due(&k.from), None);
            assert_eq!(ledger.due(&k.from, Some(&k)), None, "never offered again");
            let summary = ledger.offers[0].summary();
            assert!(
                summary.ends_with(&format!("moved 2026-09-02; {said}.")),
                "{summary}"
            );
        }
        // Not in a trial: nothing to decide.
        let (k, mut ledger) = in_review();
        assert_eq!(ledger.decide_early(&k, t(9), &k.to, "x", "y"), None);
    }

    /// A trial the agent chose itself is a record of its own beside any
    /// earlier offer for the same pair, and runs the same course.
    #[test]
    fn a_self_chosen_trial_runs_the_same_course() {
        let k = key();
        let mut ledger = Ledger::default();
        ledger.record_offer(names(&k), t(1), offer(OfferChoice::NoSwap));
        assert!(ledger.begin_self_trial(names(&k), t(3), "curious"));
        assert_eq!(
            ledger.offers.len(),
            2,
            "the declined offer keeps its record"
        );
        assert_eq!(ledger.offers[0].stage, Stage::Declined);
        assert_eq!(ledger.chosen_at(&k), Some(t(3)));
        assert!(
            !ledger.begin_self_trial(names(&k), t(3), "again"),
            "not while one is under way"
        );
        assert_eq!(ledger.due(&k.from, Some(&k)), None);

        // The `set_model` call applied the switch.
        let change = Ledger::awaited(&ledger.offers[1]).unwrap();
        assert_eq!(change.action, ChangeAction::SwapTrial);
        ledger.record_switch(Switch {
            at: t(3),
            from: k.from.clone(),
            to: k.to.clone(),
            cause: SwitchCause::SelfSwitch {
                reason: "curious".into(),
                trial: true,
            },
            sessions_after: 0,
        });
        assert!(ledger.applied(&ledger.offers[1], &k.from, &k.to));

        assert!(ledger.observe_model(&k.to, t(4)));
        assert_eq!(ledger.offers[0].stage, Stage::Declined, "untouched");
        for _ in 0..TRIAL_SESSIONS {
            ledger.count_session(&k.to);
        }
        ledger.end_trial(&k.to, t(9)).unwrap();
        ledger.observe_model(&k.from, t(10));
        assert_eq!(ledger.review_due(&k.from), Some(k.clone()));
        assert_eq!(ledger.review(&k).unwrap().chosen_at, t(3));
        ledger.record_review(&k, t(10), review(ReviewChoice::Revert));
        assert_eq!(
            ledger.offers[1].summary(),
            "Chose on 2026-09-03 with set_model to try Qwen 3.8 for 5 sessions instead of \
             Qwen 3.6 — moved 2026-09-04; trial of 5 sessions complete 2026-09-09; returned \
             to Qwen 3.6 to decide; after the trial chose to stay on Qwen 3.6 (2026-09-10)."
        );
        assert_eq!(
            ledger.offers[0].summary(),
            "Asked on 2026-09-01 whether to move from Qwen 3.6 to Qwen 3.8 — chose to stay \
             on Qwen 3.6."
        );

        // Each record keeps its own SOUL entry.
        let mut soul = soul(0);
        assert!(ledger.update_soul(&mut soul, t(1), day(10)));
        assert_eq!(soul.evolution_log.len(), 2);
    }

    /// The new events and the switch's `trial` flag round-trip; a plain
    /// self switch writes no `trial` key, as before.
    #[test]
    fn self_trial_records_round_trip() {
        let (k, mut ledger) = mid_trial(1);
        ledger.decide_early(&k, t(4), &k.from, "Qwen 3.6", "home");
        ledger.record_switch(self_switch(t(4)));
        let json = serde_json::to_value(&ledger).unwrap();
        assert!(json["switches"][0].get("trial").is_none(), "{json}");
        let event = &json["offers"][0]["history"][2];
        assert_eq!(event["event"], "decided_early");
        assert_eq!(event["decision"], "return");
        assert_eq!(event["sessions"], 1);
        assert_eq!(
            Ledger::from_slice(json.to_string().as_bytes()).unwrap(),
            ledger
        );
        let mut fresh = Ledger::default();
        fresh.begin_self_trial(names(&k), t(5), "try");
        let json = serde_json::to_value(&fresh).unwrap();
        assert_eq!(json["offers"][0]["history"][0]["event"], "self_trial");
        assert_eq!(
            Ledger::from_slice(json.to_string().as_bytes()).unwrap(),
            fresh
        );
    }

    /// Ledgers written before switches existed still load, and a ledger
    /// with none writes no `switches` key, so older binaries read it too.
    #[test]
    fn switches_are_optional_in_format_1() {
        let old = br#"{"format": 1, "agent": "tarn", "offers": []}"#;
        let ledger = Ledger::from_slice(old).unwrap();
        assert!(ledger.switches.is_empty());
        let json = serde_json::to_string(&ledger).unwrap();
        assert!(!json.contains("switches"), "{json}");

        let mut ledger = ledger;
        ledger.record_switch(self_switch(t(10)));
        let json = serde_json::to_value(&ledger).unwrap();
        assert_eq!(json["format"], 1);
        assert_eq!(json["switches"][0]["cause"], "self_switch");
        assert_eq!(json["switches"][0]["reason"], "why");
        assert_eq!(
            Ledger::from_slice(json.to_string().as_bytes()).unwrap(),
            ledger
        );
    }
}
