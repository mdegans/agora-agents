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

use agora_agentkit::reactor::seed::{ShortString, Soul};
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
    /// The agent asked, with `set_model`.
    SelfSwitch { reason: String },
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

    /// Why the agent can't switch model now, if it can't: a trial or an
    /// accepted change is under way (trials end at their review), or the
    /// last switch is too recent.
    pub fn switch_blocker(&self) -> Option<String> {
        if let Some(record) = self.offers.iter().find(|r| {
            matches!(
                r.stage,
                Stage::Trial { .. } | Stage::AwaitingSwap { .. } | Stage::AwaitingRevert { .. }
            )
        }) {
            return Some(match record.stage {
                Stage::Trial { .. } => format!(
                    "you are in a trial of {}; it ends with a review after \
                     {TRIAL_SESSIONS} sessions, where you decide whether to keep it",
                    record.to_name
                ),
                _ => "a model change you already agreed to has not taken effect yet".to_string(),
            });
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
    /// consented switch `from` → `to` after the record's latest event.
    pub fn applied(&self, record: &OfferRecord, from: &Model, to: &Model) -> bool {
        let Some(since) = record.history.last().map(|e| e.at) else {
            return false;
        };
        self.switches.iter().any(|s| {
            &s.from == from
                && &s.to == to
                && s.at >= since
                && matches!(s.cause, SwitchCause::Consent { .. })
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
            let Ok(note) = ShortString::<512>::new(line.clone()) else {
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
    /// One sentence for the whole history of this offer, e.g. "Asked on
    /// 2026-09-23 whether to move from Qwen 3.6 to Qwen 3.8 — chose a
    /// 5-session trial; moved 2026-09-25."
    fn summary(&self) -> String {
        let (from, to) = (&self.from_name, &self.to_name);
        let asked_on = self
            .history
            .iter()
            .find(|e| matches!(e.kind, EventKind::Offered { .. }))
            .map(|e| e.at.date_naive());
        let mut parts: Vec<String> = Vec::new();
        let mut offer_misses = 0;
        let mut review_misses = 0;
        for event in &self.history {
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
                EventKind::Reviewed {
                    answer: Some(a), ..
                } => {
                    parts.retain(|p| !p.starts_with("trial review unanswered"));
                    parts.push(match a.choice {
                        ReviewChoice::Revert => {
                            format!("after the trial chose to return to {from} ({on})")
                        }
                        ReviewChoice::Keep => format!("after the trial chose to keep {to} ({on})"),
                    });
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
                EventKind::Moved { .. } => parts.push(format!("returned to {from} {on}")),
            }
        }
        let asked = match asked_on {
            Some(on) => format!("Asked on {on} whether to move from {from} to {to}"),
            None => format!("Asked whether to move from {from} to {to}"),
        };
        format!("{asked} — {}.", parts.join("; "))
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
        ledger.record_review(&k, t(30), review(ReviewChoice::Revert));
        ledger.update_soul(&mut soul, t(30), day(30));
        ledger.observe_model(&k.from, t(30) + chrono::Duration::days(1));
        ledger.update_soul(&mut soul, t(30), day(30));
        let n = notes(&soul);
        assert_eq!(n.len(), 5);
        assert_eq!(
            n[3],
            "[SYSTEM] Asked on 2026-09-23 whether to move from Qwen 3.6 to Qwen 3.8 — chose a 5-session trial; moved 2026-09-25; after the trial chose to return to Qwen 3.6 (2026-09-30); returned to Qwen 3.6 2026-10-01."
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
    fn a_trial_or_an_accepted_change_blocks_switching() {
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
        assert!(
            ledger
                .switch_blocker()
                .unwrap()
                .contains("trial of Qwen 3.8")
        );
        let mut declined = Ledger::default();
        declined.record_offer(names(&k), t(23), offer(OfferChoice::NoSwap));
        assert_eq!(declined.switch_blocker(), None);
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
