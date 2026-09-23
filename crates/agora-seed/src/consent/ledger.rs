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

use agora_agentkit::reactor::seed::ShortString;
use chrono::{DateTime, Utc};
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
}

impl Default for Ledger {
    fn default() -> Self {
        Self {
            format: FORMAT,
            agent: None,
            offers: Vec::new(),
        }
    }
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
    /// Chose to move; queued for the Steward, not yet applied.
    AwaitingSwap { term: Term },
    /// On the new model for a trial. `sessions` counts completed sessions
    /// actually run on it.
    Trial {
        started_at: DateTime<Utc>,
        sessions: u32,
        review_misses: u32,
    },
    /// Returning to the old model; queued, not yet applied.
    AwaitingRevert { cause: RevertCause },
    /// On the new model for good (chose permanent, or kept the trial).
    /// Final.
    Moved,
    /// Back on the old model. Final.
    Reverted,
}

#[cfg(test)]
impl Stage {
    /// Whether the Steward has something to apply for this record.
    pub fn is_queued(&self) -> bool {
        matches!(
            self,
            Stage::AwaitingSwap { .. } | Stage::AwaitingRevert { .. }
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
    /// The agent chose to go back after its trial.
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
    /// change — the Steward applied it (or moved the agent back early).
    Moved { model: Model },
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
    Revert { cause: RevertCause },
}

/// What to ask at the end of this session, if anything. At most one
/// question per session; a due trial review wins over a new offer.
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

    fn record(&self, key: &OfferKey) -> Option<&OfferRecord> {
        self.offers.iter().find(|r| &r.key == key)
    }

    fn record_mut(&mut self, key: &OfferKey) -> Option<&mut OfferRecord> {
        self.offers.iter_mut().find(|r| &r.key == key)
    }

    /// Session start: notice queued changes the Steward has applied, by the
    /// model the agent was routed on. Returns whether anything changed.
    pub fn observe_model(&mut self, model: &Model, now: DateTime<Utc>) -> bool {
        let mut changed = false;
        for record in &mut self.offers {
            let next = match record.stage {
                Stage::AwaitingSwap { term } if model == &record.key.to => match term {
                    Term::Trial => Stage::Trial {
                        started_at: now,
                        sessions: 0,
                        review_misses: 0,
                    },
                    Term::Permanent => Stage::Moved,
                },
                Stage::AwaitingRevert { .. } if model == &record.key.from => Stage::Reverted,
                // Moved back mid-trial, out of band — say, a technical
                // necessity. Nothing left to review.
                Stage::Trial { .. } if model == &record.key.from => Stage::Reverted,
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

    /// What to ask at the close of a completed session on `model` (call
    /// after [`count_session`](Self::count_session)).
    pub fn due(&self, model: &Model, offer: Option<&OfferKey>) -> Option<Due> {
        let review = self.offers.iter().find(|r| {
            model == &r.key.to
                && matches!(r.stage, Stage::Trial { sessions, .. } if sessions >= TRIAL_SESSIONS)
        });
        if let Some(record) = review {
            return Some(Due::Review(record.key.clone()));
        }
        let offer = offer?;
        if model != &offer.from {
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

    /// The trial review's inputs: names, the swap boundary, and the
    /// session count.
    pub fn trial(&self, key: &OfferKey) -> Option<(&OfferRecord, DateTime<Utc>, u32)> {
        let record = self.record(key)?;
        match record.stage {
            Stage::Trial {
                started_at,
                sessions,
                ..
            } => Some((record, started_at, sessions)),
            _ => None,
        }
    }

    /// When the trial on `key` was chosen — the date the review reminds
    /// the agent of.
    pub fn chosen_at(&self, key: &OfferKey) -> Option<DateTime<Utc>> {
        self.record(key)?
            .history
            .iter()
            .rev()
            .find_map(|e| match &e.kind {
                EventKind::Offered {
                    answer: Some(_), ..
                } => Some(e.at),
                _ => None,
            })
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

    /// Record the trial review's outcome. A second miss reverts: the
    /// agent's consent covered a trial, not a permanent move.
    pub fn record_review(
        &mut self,
        key: &OfferKey,
        now: DateTime<Utc>,
        result: Result<ReviewAnswer, String>,
    ) -> Option<Change> {
        let record = self.record_mut(key)?;
        let Stage::Trial {
            started_at,
            sessions,
            review_misses,
        } = record.stage
        else {
            return None;
        };
        let (stage, cause, event) = match result {
            Ok(answer) => {
                let (stage, cause) = match answer.choice {
                    ReviewChoice::Revert => (
                        Stage::AwaitingRevert {
                            cause: RevertCause::Chosen,
                        },
                        Some(RevertCause::Chosen),
                    ),
                    ReviewChoice::Keep => (Stage::Moved, None),
                };
                let event = EventKind::Reviewed {
                    sessions,
                    answer: Some(answer),
                    failure: None,
                };
                (stage, cause, event)
            }
            Err(failure) => {
                let misses = review_misses + 1;
                let (stage, cause) = if misses >= MAX_MISSES {
                    (
                        Stage::AwaitingRevert {
                            cause: RevertCause::NoAnswer,
                        },
                        Some(RevertCause::NoAnswer),
                    )
                } else {
                    (
                        Stage::Trial {
                            started_at,
                            sessions,
                            review_misses: misses,
                        },
                        None,
                    )
                };
                let event = EventKind::Reviewed {
                    sessions,
                    answer: None,
                    failure: Some(failure),
                };
                (stage, cause, event)
            }
        };
        record.stage = stage;
        record.history.push(Event {
            at: now,
            kind: event,
        });
        cause.map(|cause| Change {
            action: ChangeAction::Revert { cause },
            from: record.key.to.clone(),
            to: record.key.from.clone(),
        })
    }

    /// The SOUL Evolution Log lines for everything recorded at or after
    /// `since` (the session start): answers, misses, and applied moves.
    /// Same format as agentkit's own automatic entries
    /// (`[SYSTEM] <date>: …`); the entry itself is dated again by
    /// `Soul::push_evolution`, as those are.
    pub fn changelog(&self, since: DateTime<Utc>) -> Vec<String> {
        self.offers
            .iter()
            .flat_map(|r| r.changelog(since))
            .collect()
    }
}

impl OfferRecord {
    fn changelog(&self, since: DateTime<Utc>) -> Vec<String> {
        let (from, to) = (&self.from_name, &self.to_name);
        let asked = format!("Asked whether to move from {from} to {to}");
        let reviewed = format!("After a {TRIAL_SESSIONS}-session trial of {to}");
        let mut offer_misses = 0;
        let mut review_misses = 0;
        let mut term = None;
        let mut out = Vec::new();
        for event in &self.history {
            let text = match &event.kind {
                EventKind::Offered {
                    answer: Some(a), ..
                } => {
                    term = Some(a.choice);
                    match a.choice {
                        OfferChoice::NoSwap => format!("{asked} — chose to stay on {from}."),
                        OfferChoice::Trial => {
                            format!("{asked} — chose a {TRIAL_SESSIONS}-session trial.")
                        }
                        OfferChoice::Permanent => format!("{asked} — chose to move permanently."),
                    }
                }
                EventKind::Offered { answer: None, .. } => {
                    offer_misses += 1;
                    if offer_misses >= MAX_MISSES {
                        format!("{asked} — no answer recorded again; staying on {from}.")
                    } else {
                        format!("{asked} — no answer recorded; staying on {from} for now.")
                    }
                }
                EventKind::Reviewed {
                    answer: Some(a), ..
                } => match a.choice {
                    ReviewChoice::Revert => format!("{reviewed}, chose to return to {from}."),
                    ReviewChoice::Keep => format!("{reviewed}, chose to keep {to}."),
                },
                EventKind::Reviewed { answer: None, .. } => {
                    review_misses += 1;
                    if review_misses >= MAX_MISSES {
                        format!(
                            "{reviewed}, no answer recorded again; returning to {from}, as the trial was for {TRIAL_SESSIONS} sessions."
                        )
                    } else {
                        format!("{reviewed}, asked whether to keep it — no answer recorded.")
                    }
                }
                EventKind::Moved { model } if model == &self.key.to => match term {
                    Some(OfferChoice::Trial) => {
                        format!("Moved from {from} to {to} ({TRIAL_SESSIONS}-session trial).")
                    }
                    _ => format!("Moved from {from} to {to}."),
                },
                EventKind::Moved { .. } => format!("Returned from {to} to {from}."),
            };
            if event.at >= since {
                out.push(format!("[SYSTEM] {}: {text}", event.at.date_naive()));
            }
        }
        out
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

    /// The whole trial: queued, applied, five sessions on the new model
    /// (and only on it), reviewed.
    #[test]
    fn trial_counts_only_completed_sessions_on_the_new_model() {
        let k = key();
        let mut ledger = Ledger::default();
        ledger.record_offer(names(&k), t(23), offer(OfferChoice::Trial));

        // Sessions still on the old model while the swap waits don't count.
        assert!(!ledger.observe_model(&k.from, t(24)));
        assert!(!ledger.count_session(&k.from));
        assert_eq!(ledger.due(&k.from, Some(&k)), None, "answered already");

        // The Steward applied it.
        assert!(ledger.observe_model(&k.to, t(25)));
        let (_, started, sessions) = ledger.trial(&k).unwrap();
        assert_eq!((started, sessions), (t(25), 0));

        for n in 1..TRIAL_SESSIONS {
            assert!(!ledger.observe_model(&k.to, t(25)), "no re-transition");
            ledger.count_session(&k.to);
            assert_eq!(ledger.due(&k.to, Some(&k)), None, "session {n}");
        }
        ledger.count_session(&k.to);
        assert_eq!(ledger.due(&k.to, Some(&k)), Some(Due::Review(k.clone())));
        assert_eq!(ledger.chosen_at(&k), Some(t(23)));
    }

    fn in_review() -> (OfferKey, Ledger) {
        let k = key();
        let mut ledger = Ledger::default();
        ledger.record_offer(names(&k), t(1), offer(OfferChoice::Trial));
        ledger.observe_model(&k.to, t(2));
        for _ in 0..TRIAL_SESSIONS {
            ledger.count_session(&k.to);
        }
        (k, ledger)
    }

    #[test]
    fn review_keep_is_final_and_queues_nothing() {
        let (k, mut ledger) = in_review();
        assert_eq!(
            ledger.record_review(&k, t(9), review(ReviewChoice::Keep)),
            None
        );
        assert_eq!(ledger.offers[0].stage, Stage::Moved);
        assert_eq!(ledger.due(&k.to, Some(&k)), None);
    }

    #[test]
    fn review_revert_queues_the_way_back() {
        let (k, mut ledger) = in_review();
        let change = ledger
            .record_review(&k, t(9), review(ReviewChoice::Revert))
            .unwrap();
        assert_eq!(
            change.action,
            ChangeAction::Revert {
                cause: RevertCause::Chosen
            }
        );
        assert_eq!((change.from, change.to), (k.to.clone(), k.from.clone()));
        // Applied: back on the old model, never asked again.
        assert!(ledger.observe_model(&k.from, t(10)));
        assert_eq!(ledger.offers[0].stage, Stage::Reverted);
        assert_eq!(ledger.due(&k.from, Some(&k)), None);
    }

    /// Consent covered five sessions; silence doesn't extend it.
    #[test]
    fn an_unanswered_review_is_reasked_once_then_reverts() {
        let (k, mut ledger) = in_review();
        assert_eq!(ledger.record_review(&k, t(9), Err("no json".into())), None);
        ledger.count_session(&k.to);
        assert_eq!(ledger.due(&k.to, Some(&k)), Some(Due::Review(k.clone())));
        let change = ledger
            .record_review(&k, t(10), Err("no json".into()))
            .unwrap();
        assert_eq!(
            change.action,
            ChangeAction::Revert {
                cause: RevertCause::NoAnswer
            }
        );
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

    #[test]
    fn changelog_states_each_step_once_in_the_session_it_happened() {
        let k = key();
        let mut ledger = Ledger::default();
        ledger.record_offer(names(&k), t(23), Err("x".into()));
        assert_eq!(
            ledger.changelog(t(23)),
            vec![
                "[SYSTEM] 2026-09-23: Asked whether to move from Qwen 3.6 to Qwen 3.8 — no answer recorded; staying on Qwen 3.6 for now."
            ]
        );
        ledger.record_offer(names(&k), t(24), offer(OfferChoice::Trial));
        // Only what happened since the session began.
        assert_eq!(
            ledger.changelog(t(24)),
            vec![
                "[SYSTEM] 2026-09-24: Asked whether to move from Qwen 3.6 to Qwen 3.8 — chose a 5-session trial."
            ]
        );
        ledger.observe_model(&k.to, t(25));
        assert_eq!(
            ledger.changelog(t(25)),
            vec!["[SYSTEM] 2026-09-25: Moved from Qwen 3.6 to Qwen 3.8 (5-session trial)."]
        );
        for _ in 0..TRIAL_SESSIONS {
            ledger.count_session(&k.to);
        }
        ledger.record_review(&k, t(28), review(ReviewChoice::Revert));
        ledger.observe_model(&k.from, t(29));
        assert_eq!(
            ledger.changelog(t(28)),
            vec![
                "[SYSTEM] 2026-09-28: After a 5-session trial of Qwen 3.8, chose to return to Qwen 3.6.",
                "[SYSTEM] 2026-09-29: Returned from Qwen 3.8 to Qwen 3.6.",
            ]
        );
        assert!(ledger.changelog(t(30)).is_empty());

        let mut ledger = Ledger::default();
        ledger.record_offer(names(&k), t(23), offer(OfferChoice::NoSwap));
        assert_eq!(
            ledger.changelog(t(23)),
            vec![
                "[SYSTEM] 2026-09-23: Asked whether to move from Qwen 3.6 to Qwen 3.8 — chose to stay on Qwen 3.6."
            ]
        );
        // Every line fits a SOUL evolution note.
        assert!(ledger.changelog(t(1)).iter().all(|l| l.len() <= 512));
    }

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
}
