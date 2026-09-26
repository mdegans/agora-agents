//! Fair share by writes: which due agents a local sweep runs, and in what
//! order.
//!
//! The runner used to interleave every due agent by headcount, so a model
//! with 236 agents had 236 voices in the forum for every 34 of a smaller
//! cohort's, and a slow model with 110 agents made a sweep last days. The
//! Steward's rule (2026-09-25): **equal shares of writes** (posts and
//! comments) among the local models over a trailing week, weighted by each
//! model's `share`, with a **wall-clock ceiling** that alerts loudly when
//! one model eats the runner. Haiku is not in the pool: it runs in its own
//! process, on its own timer and budget, and only the local endpoint's
//! reactor is planned here.
//!
//! [`plan`] is pure: due agents grouped by model, the two ledgers
//! ([`ledger`]), a clock and the knobs in, an order out. The rule:
//!
//! 1. Per model, from the ledgers: writes in the trailing
//!    [`ScheduleConfig::share_window_days`], writes per session and mean
//!    session length (priors when there is too little history), and its
//!    share of the endpoint's busy time over the trailing
//!    [`ScheduleConfig::ceiling_window_hours`].
//! 2. A model over [`ScheduleConfig::ceiling`] is skipped this sweep, with
//!    an ERROR `schedule_ceiling_hit` — unless skipping it would leave
//!    nothing to run, when the GPU would only idle (WARN
//!    `schedule_ceiling_waived`).
//! 3. Repeatedly pick the model with the largest **deficit** — its target
//!    (share-weighted fraction of all writes, planned ones included) minus
//!    what it has (actual plus planned). Planned writes are projected with
//!    the model's own writes-per-session, so picking a model shrinks its
//!    deficit by what its sessions are expected to write, and a model that
//!    writes little per session is not picked forever.
//! 4. Take up to `wave_size` of that model's due agents, oldest
//!    `last_cycle_at` first (never-cycled first of all), so within a model
//!    nobody starves. The next pick must be a different model while any
//!    other has agents left: at most `wave_size` consecutive sessions per
//!    model.
//! 5. Stop once the projected wall-clock reaches
//!    [`ScheduleConfig::sweep_secs`] (always at least one wave). The next
//!    sweep re-plans from the updated ledgers. Within a sweep, a model is
//!    not picked past `ceiling × sweep_secs` of projected time while
//!    another model still has agents, so the ceiling is a backstop rather
//!    than the steady state.

pub mod ledger;

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use agora_agentkit::ids::AgentId;
use chrono::{DateTime, Duration, Utc};
use misanthropic::model::Model;
use serde::{Deserialize, Serialize};

use ledger::{SessionRecord, WriteRecord};

/// A model id from its wire string. `Model::from` round-trips through a
/// JSON literal it builds by formatting, so a quote in the string panics;
/// this can't, and agrees with it on every id that doesn't.
pub fn model_id(name: &str) -> Model {
    serde_json::from_value(serde_json::Value::String(name.to_string()))
        .unwrap_or_else(|_| Model::Custom(name.to_string().into()))
}

/// Sessions of history a model needs before its own writes-per-session and
/// session length replace the priors. Below this a single long or silent
/// session would swing the whole plan.
pub const MIN_SESSIONS_FOR_STATS: usize = 3;

/// Busy seconds the ceiling window must hold before the ceiling binds.
/// With less, the first session of the day is 100% of the runner.
pub const MIN_CEILING_SECS: f64 = 3600.0;

/// `[schedule]` in the run config. Every field has a default, so a config
/// without the table plans with these.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ScheduleConfig {
    /// Trailing window, in days, over which writes are shared out.
    pub share_window_days: u32,
    /// Maximum fraction of the endpoint's busy time one model may take
    /// over [`Self::ceiling_window_hours`]. Above it the model sits out a
    /// sweep and the runner logs ERROR `schedule_ceiling_hit`.
    pub ceiling: f64,
    /// Trailing window, in hours, for the ceiling.
    pub ceiling_window_hours: u32,
    /// Projected wall-clock one sweep plans for. The runner exits when a
    /// sweep ends and systemd starts the next, which re-plans from fresh
    /// ledgers — so this is also how stale a plan may get. `0` plans every
    /// due agent (ordering only, no fair share across sweeps).
    pub sweep_secs: u64,
    /// Writes per session assumed for a model with too little history
    /// (and no other model to borrow a mean from).
    pub prior_writes_per_session: f64,
    /// Session length in seconds assumed for a model with too little
    /// history (and no other model to borrow a mean from).
    pub prior_session_secs: f64,
}

impl Default for ScheduleConfig {
    fn default() -> Self {
        Self {
            share_window_days: 7,
            ceiling: 0.5,
            ceiling_window_hours: 24,
            sweep_secs: 4 * 3600,
            prior_writes_per_session: 1.0,
            prior_session_secs: 300.0,
        }
    }
}

impl ScheduleConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.share_window_days > 0,
            "[schedule] share_window_days must be nonzero"
        );
        anyhow::ensure!(
            self.ceiling_window_hours > 0,
            "[schedule] ceiling_window_hours must be nonzero"
        );
        anyhow::ensure!(
            self.ceiling > 0.0 && self.ceiling <= 1.0,
            "[schedule] ceiling must be in (0, 1] — 1 turns it off"
        );
        anyhow::ensure!(
            self.prior_writes_per_session > 0.0 && self.prior_session_secs > 0.0,
            "[schedule] priors must be positive"
        );
        Ok(())
    }
}

/// One due agent, as the planner sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    pub id: AgentId,
    pub last_cycle_at: Option<DateTime<Utc>>,
}

/// What the ledgers say about one model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ModelStats {
    /// Writes in the share window.
    pub writes: f64,
    /// Projected writes per session (own history, else a prior).
    pub writes_per_session: f64,
    /// Projected seconds per session (own history, else a prior).
    pub session_secs: f64,
    /// Whether the two figures above are the model's own.
    pub measured: bool,
    /// Fraction of the endpoint's busy time in the ceiling window, or
    /// `None` when the window holds too little to judge.
    pub busy_share: Option<f64>,
}

/// Per-model line of a [`Plan`], for the `schedule_plan` event and the
/// dry-run report.
#[derive(Debug, Clone, Serialize)]
pub struct ModelPlan {
    pub model: Model,
    pub due: usize,
    pub planned: usize,
    pub share: f64,
    #[serde(flatten)]
    pub stats: ModelStats,
    /// Target minus actual writes when planning started.
    pub deficit: f64,
    /// Skipped this sweep for being over the ceiling.
    pub ceiling_hit: bool,
}

/// A sweep's order and why.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    /// Run order.
    pub order: Vec<AgentId>,
    /// Consecutive same-model runs in order, e.g. `[(gpt-oss, 8), (Qwen, 5)]`.
    pub waves: Vec<(Model, usize)>,
    pub models: Vec<ModelPlan>,
    /// Projected wall-clock of `order`.
    pub projected_secs: f64,
    /// Models over the ceiling that ran anyway because nothing else was
    /// due.
    pub ceiling_waived: Vec<Model>,
}

/// Aggregate the ledgers into per-model stats for `models`.
///
/// `offered` is every model the endpoint serves: the ceiling's
/// denominator is their busy time, so another endpoint's sessions (Haiku,
/// sharing the data dir) never dilute it.
pub fn model_stats(
    models: &BTreeSet<Model>,
    offered: &BTreeSet<Model>,
    writes: &[WriteRecord],
    sessions: &[SessionRecord],
    now: DateTime<Utc>,
    config: &ScheduleConfig,
) -> BTreeMap<Model, ModelStats> {
    let share_from = now - Duration::days(i64::from(config.share_window_days));
    let ceiling_from = now - Duration::hours(i64::from(config.ceiling_window_hours));

    struct Acc {
        writes: usize,
        sessions: usize,
        session_secs: f64,
        first_session: Option<DateTime<Utc>>,
        busy: f64,
    }
    let mut acc: BTreeMap<&Model, Acc> = models
        .iter()
        .map(|m| {
            (
                m,
                Acc {
                    writes: 0,
                    sessions: 0,
                    session_secs: 0.0,
                    first_session: None,
                    busy: 0.0,
                },
            )
        })
        .collect();

    let mut busy_total = 0.0;
    for s in sessions {
        let end = s.at;
        let start = s.started_at();
        // Busy time inside the ceiling window, for every offered model.
        if offered.contains(&s.model) || models.contains(&s.model) {
            let overlap = (end.min(now) - start.max(ceiling_from)).num_milliseconds() as f64 / 1e3;
            if overlap > 0.0 {
                busy_total += overlap;
                if let Some(a) = acc.get_mut(&s.model) {
                    a.busy += overlap;
                }
            }
        }
        if end >= share_from
            && end <= now
            && let Some(a) = acc.get_mut(&s.model)
        {
            a.sessions += 1;
            a.session_secs += s.duration_secs.max(0.0);
            a.first_session = Some(a.first_session.map_or(start, |f| f.min(start)));
        }
    }

    // Writes in the window, and — for writes-per-session — only those since
    // the model's first recorded session, so a ledger seeded with a week
    // of writes but no sessions doesn't read as 500 writes per session.
    let mut paired: BTreeMap<&Model, usize> = BTreeMap::new();
    for w in writes {
        if w.at < share_from || w.at > now {
            continue;
        }
        if let Some(a) = acc.get_mut(&w.model) {
            a.writes += 1;
            if a.first_session.is_some_and(|f| w.at >= f) {
                *paired.entry(&w.model).or_default() += 1;
            }
        }
    }

    // Priors: the mean of the models that have their own figures, else
    // the configured constants.
    let measured: Vec<(f64, f64)> = acc
        .iter()
        .filter(|(_, a)| a.sessions >= MIN_SESSIONS_FOR_STATS)
        .map(|(m, a)| {
            let n = a.sessions as f64;
            (
                paired.get(m).copied().unwrap_or(0) as f64 / n,
                a.session_secs / n,
            )
        })
        .collect();
    let (prior_wps, prior_secs) = if measured.is_empty() {
        (config.prior_writes_per_session, config.prior_session_secs)
    } else {
        let n = measured.len() as f64;
        (
            measured.iter().map(|(w, _)| w).sum::<f64>() / n,
            measured.iter().map(|(_, s)| s).sum::<f64>() / n,
        )
    };

    acc.into_iter()
        .map(|(m, a)| {
            let own = a.sessions >= MIN_SESSIONS_FOR_STATS;
            let n = a.sessions as f64;
            let (wps, secs) = if own {
                (
                    paired.get(m).copied().unwrap_or(0) as f64 / n,
                    a.session_secs / n,
                )
            } else {
                (prior_wps, prior_secs)
            };
            let stats = ModelStats {
                writes: a.writes as f64,
                // A model measured at zero writes per session would never
                // close its deficit; floor it so it can't loop forever.
                writes_per_session: wps.max(0.05),
                session_secs: secs.max(1.0),
                measured: own,
                busy_share: (busy_total >= MIN_CEILING_SECS).then(|| a.busy / busy_total),
            };
            (m.clone(), stats)
        })
        .collect()
}

/// Order a sweep. `groups` is the endpoint's due agents by model;
/// `shares` weights each model (absent means 1.0).
pub fn plan(
    groups: &BTreeMap<Model, Vec<Candidate>>,
    shares: &BTreeMap<Model, f64>,
    stats: &BTreeMap<Model, ModelStats>,
    wave_size: usize,
    config: &ScheduleConfig,
) -> Plan {
    let wave_size = wave_size.max(1);
    let share_of = |m: &Model| shares.get(m).copied().unwrap_or(1.0).max(0.0);
    let stats_of = |m: &Model| {
        stats.get(m).copied().unwrap_or(ModelStats {
            writes: 0.0,
            writes_per_session: config.prior_writes_per_session,
            session_secs: config.prior_session_secs,
            measured: false,
            busy_share: None,
        })
    };

    // Per-model queues, oldest cycle first; never-cycled before anyone.
    let mut queues: BTreeMap<&Model, VecDeque<Candidate>> = groups
        .iter()
        .filter(|(_, c)| !c.is_empty())
        .map(|(m, c)| {
            let mut c = c.clone();
            c.sort_by_key(|c| (c.last_cycle_at.is_some(), c.last_cycle_at));
            (m, c.into())
        })
        .collect();

    // The ceiling.
    let over: BTreeSet<&Model> = queues
        .keys()
        .copied()
        .filter(|m| stats_of(m).busy_share.is_some_and(|s| s > config.ceiling))
        .collect();
    let mut ceiling_waived = Vec::new();
    if over.len() < queues.len() {
        queues.retain(|m, _| !over.contains(m));
    } else {
        ceiling_waived.extend(over.iter().map(|m| (*m).clone()));
    }

    // Deficits are shared out among the models in play this sweep.
    let weight: f64 = queues.keys().map(|m| share_of(m)).sum();
    let mut actual: BTreeMap<&Model, f64> =
        queues.keys().map(|m| (*m, stats_of(m).writes)).collect();
    let deficit = |m: &Model, actual: &BTreeMap<&Model, f64>| -> f64 {
        let total: f64 = actual.values().sum();
        let target = if weight > 0.0 {
            share_of(m) / weight * total
        } else {
            0.0
        };
        target - actual.get(m).copied().unwrap_or(0.0)
    };
    let initial_deficit: BTreeMap<Model, f64> = queues
        .keys()
        .map(|m| ((*m).clone(), deficit(m, &actual)))
        .collect();

    let horizon = (config.sweep_secs > 0).then_some(config.sweep_secs as f64);
    let per_model_cap = horizon.map(|h| h * config.ceiling);
    let mut order = Vec::new();
    let mut waves: Vec<(Model, usize)> = Vec::new();
    let mut planned: BTreeMap<&Model, usize> = BTreeMap::new();
    let mut planned_secs: BTreeMap<&Model, f64> = BTreeMap::new();
    let mut projected = 0.0;
    let mut last: Option<&Model> = None;

    loop {
        if horizon.is_some_and(|h| projected >= h) && !order.is_empty() {
            break;
        }
        let live: Vec<&Model> = queues
            .iter()
            .filter(|(_, q)| !q.is_empty())
            .map(|(m, _)| *m)
            .collect();
        if live.is_empty() {
            break;
        }
        // The in-sweep ceiling, then the consecutive cap — each relaxed
        // only when it would leave nobody. Wall-clock outranks alternation:
        // if only one model is under its time cap, it runs back to back.
        let under_cap = |m: &&Model| {
            per_model_cap.is_none_or(|cap| {
                planned_secs.get(m).copied().unwrap_or(0.0) + stats_of(m).session_secs <= cap
            })
        };
        let not_last = |m: &&Model| Some(*m) != last;
        let tiers: [Vec<&Model>; 3] = [
            live.iter()
                .copied()
                .filter(|m| under_cap(m) && not_last(m))
                .collect(),
            live.iter().copied().filter(under_cap).collect(),
            live.iter().copied().filter(not_last).collect(),
        ];
        let key = |m: &Model| {
            let head = queues[m]
                .front()
                .map(|c| (c.last_cycle_at.is_some(), c.last_cycle_at));
            (deficit(m, &actual), head)
        };
        let [preferred, within_cap, alternating] = tiers;
        let pool = [preferred, within_cap.clone(), alternating]
            .into_iter()
            .find(|t| !t.is_empty())
            .unwrap_or_else(|| live.clone());
        let pick = best(&pool, key).expect("pool is nonempty");
        // A pick made only because the consecutive cap benched the model
        // that is furthest behind is a *break*, not a turn: it runs only
        // the sessions its own deficit warrants (at least one). Otherwise
        // two models would strictly alternate full waves, 50/50 by
        // sessions whatever their deficits.
        let unconstrained = best(
            if within_cap.is_empty() {
                &live
            } else {
                &within_cap
            },
            key,
        );
        let s = stats_of(pick);
        let limit = if unconstrained.is_some_and(|u| u != pick) {
            let owed = deficit(pick, &actual).max(0.0) / s.writes_per_session;
            (owed.ceil() as usize).clamp(1, wave_size)
        } else {
            wave_size
        };

        let queue = queues.get_mut(pick).expect("picked from queues");
        let mut taken = 0;
        while taken < limit {
            if taken > 0 && horizon.is_some_and(|h| projected >= h) {
                break;
            }
            // The in-sweep time cap binds within a wave too.
            if taken > 0
                && per_model_cap.is_some_and(|cap| {
                    planned_secs.get(pick).copied().unwrap_or(0.0) + s.session_secs > cap
                })
            {
                break;
            }
            let Some(c) = queue.pop_front() else { break };
            order.push(c.id);
            projected += s.session_secs;
            *planned_secs.entry(pick).or_default() += s.session_secs;
            taken += 1;
        }
        *planned.entry(pick).or_default() += taken;
        *actual.get_mut(pick).expect("in play") += taken as f64 * s.writes_per_session;
        match waves.last_mut() {
            Some((m, n)) if m == pick => *n += taken,
            _ => waves.push((pick.clone(), taken)),
        }
        last = Some(pick);
    }

    let models = groups
        .iter()
        .filter(|(_, c)| !c.is_empty())
        .map(|(m, c)| ModelPlan {
            model: m.clone(),
            due: c.len(),
            planned: planned.get(m).copied().unwrap_or(0),
            share: share_of(m),
            stats: stats_of(m),
            deficit: initial_deficit.get(m).copied().unwrap_or(0.0),
            ceiling_hit: over.contains(m) && ceiling_waived.is_empty(),
        })
        .collect();

    Plan {
        order,
        waves,
        models,
        projected_secs: projected,
        ceiling_waived,
    }
}

/// The model with the largest deficit; ties to the one whose next agent
/// has waited longest (never-cycled first), then by id for determinism.
fn best<'m>(
    pool: &[&'m Model],
    key: impl Fn(&Model) -> (f64, Option<(bool, Option<DateTime<Utc>>)>),
) -> Option<&'m Model> {
    pool.iter().copied().max_by(|a, b| {
        let ((da, ha), (db, hb)) = (key(a), key(b));
        da.partial_cmp(&db)
            .expect("deficits are finite")
            // Older head wins: reversed, so "less" is "better".
            .then_with(|| hb.cmp(&ha))
            .then_with(|| b.cmp(a))
    })
}

impl Plan {
    /// Log the plan: INFO `schedule_plan`, one ERROR `schedule_ceiling_hit`
    /// per skipped model, one WARN `schedule_ceiling_waived` per model that
    /// ran over the ceiling because nothing else was due.
    ///
    /// Each ceiling hit also goes to the operator through `alerts`.
    pub fn log(&self, endpoint: &str, ceiling: f64, alerts: &crate::alerts::Alerter) {
        for m in &self.models {
            if m.ceiling_hit {
                tracing::error!(
                    event_type = "schedule_ceiling_hit",
                    endpoint,
                    model = %m.model,
                    share = m.stats.busy_share.unwrap_or_default(),
                    ceiling,
                    due = m.due,
                    "model over its wall-clock ceiling; skipped this sweep"
                );
                alerts.notify(
                    crate::alerts::Alert::new(
                        crate::alerts::AlertKind::ScheduleCeilingHit,
                        "model over its wall-clock ceiling; skipped this sweep",
                    )
                    .model(&m.model)
                    .detail("endpoint", endpoint)
                    .detail("share", m.stats.busy_share.unwrap_or_default())
                    .detail("ceiling", ceiling)
                    .detail("due", m.due),
                );
            }
        }
        for model in &self.ceiling_waived {
            let share = self
                .models
                .iter()
                .find(|m| &m.model == model)
                .and_then(|m| m.stats.busy_share)
                .unwrap_or_default();
            tracing::warn!(
                event_type = "schedule_ceiling_waived",
                endpoint,
                model = %model,
                share,
                ceiling,
                "model over its wall-clock ceiling, but nothing else is due; running it"
            );
        }
        let waves = self
            .waves
            .iter()
            .map(|(m, n)| format!("{m}×{n}"))
            .collect::<Vec<_>>()
            .join(" ");
        tracing::info!(
            event_type = "schedule_plan",
            endpoint,
            sessions = self.order.len(),
            projected_secs = self.projected_secs.round() as u64,
            models = %serde_json::to_string(&self.models).unwrap_or_default(),
            waves = %waves,
            "sweep planned"
        );
    }

    /// Human-readable summary for the dry run.
    pub fn report(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "   schedule: {} sessions, ~{:.1} h projected",
            self.order.len(),
            self.projected_secs / 3600.0
        );
        for m in &self.models {
            let _ = writeln!(
                out,
                "   {:<45} due {:>4}  planned {:>4}  writes/7d {:>6.0}  deficit {:>+7.1}  \
                 w/sess {:.2}  sess {:>5.0}s{}  busy24h {}{}",
                m.model.to_string(),
                m.due,
                m.planned,
                m.stats.writes,
                m.deficit,
                m.stats.writes_per_session,
                m.stats.session_secs,
                if m.stats.measured { "" } else { " (prior)" },
                m.stats
                    .busy_share
                    .map_or("—".to_string(), |s| format!("{:.0}%", s * 100.0)),
                if m.ceiling_hit {
                    "  CEILING: skipped"
                } else {
                    ""
                },
            );
        }
        let waves = self
            .waves
            .iter()
            .map(|(m, n)| format!("{m}×{n}"))
            .collect::<Vec<_>>()
            .join(" → ");
        let _ = writeln!(out, "   waves: {waves}");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ledger::WriteKind;

    fn t0() -> DateTime<Utc> {
        "2026-09-25T12:00:00Z".parse().unwrap()
    }

    fn model(s: &str) -> Model {
        Model::from(s.to_string())
    }

    fn agents(n: usize, base_hours_ago: i64) -> Vec<Candidate> {
        (0..n)
            .map(|i| Candidate {
                id: AgentId::from(uuid::Uuid::new_v4()),
                last_cycle_at: Some(t0() - Duration::hours(base_hours_ago + i as i64)),
            })
            .collect()
    }

    fn writes(m: &str, n: usize, hours_ago: i64) -> Vec<WriteRecord> {
        (0..n)
            .map(|_| WriteRecord {
                at: t0() - Duration::hours(hours_ago),
                model: model(m),
                agent_id: AgentId::from(uuid::Uuid::nil()),
                kind: WriteKind::Comment,
            })
            .collect()
    }

    fn sessions(m: &str, n: usize, secs: f64, hours_ago: i64) -> Vec<SessionRecord> {
        (0..n)
            .map(|i| SessionRecord {
                at: t0() - Duration::hours(hours_ago) + Duration::seconds(i as i64),
                model: model(m),
                agent_id: AgentId::from(uuid::Uuid::nil()),
                duration_secs: secs,
                outcome: None,
            })
            .collect()
    }

    fn unbounded() -> ScheduleConfig {
        ScheduleConfig {
            sweep_secs: 0,
            ..ScheduleConfig::default()
        }
    }

    fn stats_for(
        groups: &BTreeMap<Model, Vec<Candidate>>,
        w: &[WriteRecord],
        s: &[SessionRecord],
        config: &ScheduleConfig,
    ) -> BTreeMap<Model, ModelStats> {
        let models: BTreeSet<Model> = groups.keys().cloned().collect();
        model_stats(&models, &models, w, s, t0(), config)
    }

    fn count(plan: &Plan, groups: &BTreeMap<Model, Vec<Candidate>>, m: &str) -> usize {
        let ids: BTreeSet<AgentId> = groups[&model(m)].iter().map(|c| c.id).collect();
        plan.order.iter().filter(|id| ids.contains(id)).count()
    }

    /// The model furthest behind its share goes first, and keeps going
    /// only as long as its projected writes leave it behind.
    #[test]
    fn largest_deficit_goes_first() {
        let groups = BTreeMap::from([
            (model("big"), agents(40, 10)),
            (model("small"), agents(40, 10)),
        ]);
        let w = [writes("big", 300, 5), writes("small", 100, 5)].concat();
        let config = unbounded();
        let stats = stats_for(&groups, &w, &[], &config);
        let plan = plan(&groups, &BTreeMap::new(), &stats, 8, &config);
        assert_eq!(plan.waves[0].0, model("small"));
        let small = plan
            .models
            .iter()
            .find(|m| m.model == model("small"))
            .unwrap();
        assert_eq!(small.deficit, 100.0); // target 200 − 100
        // Everyone due is planned when the sweep is unbounded.
        assert_eq!(plan.order.len(), 80);
    }

    /// Projection: a model writing 3 per session closes its deficit in a
    /// third of the sessions one writing 1 per session needs.
    #[test]
    fn projection_uses_each_models_writes_per_session() {
        let groups = BTreeMap::from([
            (model("chatty"), agents(30, 10)),
            (model("terse"), agents(30, 10)),
        ]);
        let w = [
            writes("chatty", 30, 2), // 10 sessions × 3
            writes("terse", 10, 2),  // 10 sessions × 1
        ]
        .concat();
        let s = [
            sessions("chatty", 10, 60.0, 3),
            sessions("terse", 10, 60.0, 3),
        ]
        .concat();
        let config = ScheduleConfig {
            sweep_secs: 30 * 60, // 30 sessions of 60 s
            ceiling: 1.0,
            ..ScheduleConfig::default()
        };
        let stats = stats_for(&groups, &w, &s, &config);
        assert_eq!(stats[&model("chatty")].writes_per_session, 3.0);
        let plan = plan(&groups, &BTreeMap::new(), &stats, 8, &config);
        let (c, t) = (
            count(&plan, &groups, "chatty"),
            count(&plan, &groups, "terse"),
        );
        assert_eq!(c + t, 30);
        // Terse is 20 writes behind and closes on chatty, which gets only
        // the one-session breaks the consecutive cap forces.
        let (wc, wt) = (30.0 + 3.0 * c as f64, 10.0 + t as f64);
        assert!(
            (wc - wt).abs() < 20.0,
            "chatty {c} ({wc}), terse {t} ({wt})"
        );
        assert!(t >= 3 * c, "chatty {c}, terse {t}: {:?}", plan.waves);
        assert!(plan.waves.iter().all(|(_, n)| *n <= 8));
    }

    /// Within a model, the longest-waiting agents go first and a
    /// never-cycled agent before all of them.
    #[test]
    fn oldest_first_within_a_model_nobody_starves() {
        let mut cands = agents(5, 1);
        let fresh = Candidate {
            id: AgentId::from(uuid::Uuid::new_v4()),
            last_cycle_at: None,
        };
        cands.push(fresh);
        let oldest = cands[4].id;
        let groups = BTreeMap::from([(model("m"), cands)]);
        let config = unbounded();
        let stats = stats_for(&groups, &[], &[], &config);
        let plan = plan(&groups, &BTreeMap::new(), &stats, 8, &config);
        assert_eq!(plan.order[0], fresh.id);
        assert_eq!(plan.order[1], oldest);

        // Across sweeps: with a one-wave horizon, the agents left out are
        // exactly the most recently cycled.
        let config = ScheduleConfig {
            sweep_secs: 1,
            ..ScheduleConfig::default()
        };
        let plan = super::plan(&groups, &BTreeMap::new(), &stats, 3, &config);
        assert_eq!(plan.order.len(), 1, "one session reaches a 1 s horizon");
        assert_eq!(plan.order[0], fresh.id);
    }

    /// No model runs more than `wave_size` sessions back to back while
    /// another has agents, however large its deficit.
    #[test]
    fn consecutive_sessions_are_capped_at_wave_size() {
        let groups = BTreeMap::from([
            (model("starved"), agents(50, 10)),
            (model("fed"), agents(50, 10)),
        ]);
        let w = writes("fed", 1000, 5);
        let config = unbounded();
        let stats = stats_for(&groups, &w, &[], &config);
        let plan = plan(&groups, &BTreeMap::new(), &stats, 4, &config);
        // Until one model runs out, when the other has the runner to itself.
        let (tail, body) = plan.waves.split_last().unwrap();
        assert!(body.iter().all(|(_, n)| *n <= 4), "{:?}", plan.waves);
        assert_eq!(tail.0, model("fed"));
        // The starved model can't take the whole runner even at a
        // 1000-write deficit; the fed one gets one-session breaks.
        assert_eq!(plan.waves[0], (model("starved"), 4));
        assert_eq!(plan.waves[1], (model("fed"), 1));
    }

    /// A model over the ceiling sits the sweep out — unless nothing else is
    /// due.
    #[test]
    fn ceiling_skips_and_waives() {
        let groups = BTreeMap::from([
            (model("slow"), agents(10, 10)),
            (model("fast"), agents(10, 10)),
        ]);
        // 20 h of slow, 4 h of fast in the last day: slow at 83%.
        let s = [
            sessions("slow", 20, 3600.0, 1),
            sessions("fast", 48, 300.0, 1),
        ]
        .concat();
        let config = unbounded();
        let stats = stats_for(&groups, &[], &s, &config);
        let share = stats[&model("slow")].busy_share.unwrap();
        assert!((share - 20.0 / 24.0).abs() < 1e-6, "{share}");
        let plan = plan(&groups, &BTreeMap::new(), &stats, 8, &config);
        assert_eq!(count(&plan, &groups, "slow"), 0);
        assert_eq!(count(&plan, &groups, "fast"), 10);
        assert!(
            plan.models
                .iter()
                .any(|m| m.model == model("slow") && m.ceiling_hit)
        );
        assert!(plan.ceiling_waived.is_empty());

        // Only the slow model due: skipping it would idle the GPU.
        let alone = BTreeMap::from([(model("slow"), agents(10, 10))]);
        let stats = {
            let models = BTreeSet::from([model("slow")]);
            let offered = BTreeSet::from([model("slow"), model("fast")]);
            model_stats(&models, &offered, &[], &s, t0(), &config)
        };
        let plan = super::plan(&alone, &BTreeMap::new(), &stats, 8, &config);
        assert_eq!(plan.order.len(), 10);
        assert_eq!(plan.ceiling_waived, vec![model("slow")]);
        assert!(plan.models.iter().all(|m| !m.ceiling_hit));
    }

    /// Too little busy time and the ceiling doesn't bind: the first session
    /// of a quiet day is 100% of it.
    #[test]
    fn ceiling_needs_enough_history() {
        let groups = BTreeMap::from([(model("a"), agents(3, 10)), (model("b"), agents(3, 10))]);
        let s = sessions("a", 1, 600.0, 1);
        let config = unbounded();
        let stats = stats_for(&groups, &[], &s, &config);
        assert_eq!(stats[&model("a")].busy_share, None);
        let plan = plan(&groups, &BTreeMap::new(), &stats, 8, &config);
        assert_eq!(plan.order.len(), 6);
    }

    /// Other endpoints' sessions (Haiku, sharing the data dir) are not in
    /// the ceiling's denominator.
    #[test]
    fn ceiling_counts_only_this_endpoints_models() {
        let s = [
            sessions("local", 10, 600.0, 1),
            sessions("claude-haiku-4-5", 1000, 600.0, 1),
        ]
        .concat();
        let models = BTreeSet::from([model("local")]);
        let stats = model_stats(&models, &models, &[], &s, t0(), &unbounded());
        assert_eq!(stats[&model("local")].busy_share, Some(1.0));
    }

    /// A model with no history gets the other models' mean, not zero:
    /// zero writes per session would never close its deficit.
    #[test]
    fn zero_history_borrows_the_mean() {
        let groups = BTreeMap::from([(model("old"), agents(5, 10)), (model("new"), agents(5, 10))]);
        let w = writes("old", 20, 2);
        let s = sessions("old", 10, 120.0, 3);
        let config = unbounded();
        let stats = stats_for(&groups, &w, &s, &config);
        let new = stats[&model("new")];
        assert!(!new.measured);
        assert_eq!(new.writes_per_session, 2.0);
        assert_eq!(new.session_secs, 120.0);
        // And with nobody measured, the configured priors.
        let stats = stats_for(&groups, &[], &[], &config);
        assert_eq!(stats[&model("new")].writes_per_session, 1.0);
        assert_eq!(stats[&model("new")].session_secs, 300.0);
    }

    /// A ledger seeded with a week of writes but no sessions must not read
    /// as hundreds of writes per session.
    #[test]
    fn seeded_writes_before_the_first_session_are_not_per_session() {
        let groups = BTreeMap::from([(model("m"), agents(1, 10))]);
        let w = [writes("m", 500, 100), writes("m", 6, 1)].concat();
        let s = sessions("m", 3, 60.0, 2);
        let stats = stats_for(&groups, &w, &s, &unbounded());
        assert_eq!(stats[&model("m")].writes, 506.0);
        assert_eq!(stats[&model("m")].writes_per_session, 2.0);
    }

    /// Writes outside the share window don't count.
    #[test]
    fn share_window_is_trailing() {
        let groups = BTreeMap::from([(model("m"), agents(1, 10))]);
        let w = [writes("m", 5, 24 * 8), writes("m", 2, 24 * 6)].concat();
        let stats = stats_for(&groups, &w, &[], &unbounded());
        assert_eq!(stats[&model("m")].writes, 2.0);
    }

    /// `share` weights the target: at 2:1 the heavier model is owed twice
    /// the writes.
    #[test]
    fn share_weights_the_target() {
        let groups = BTreeMap::from([(model("a"), agents(10, 10)), (model("b"), agents(10, 10))]);
        let w = [writes("a", 60, 2), writes("b", 60, 2)].concat();
        let config = unbounded();
        let stats = stats_for(&groups, &w, &[], &config);
        let shares = BTreeMap::from([(model("a"), 2.0)]);
        let plan = plan(&groups, &shares, &stats, 8, &config);
        let a = plan.models.iter().find(|m| m.model == model("a")).unwrap();
        assert_eq!(a.deficit, 20.0); // 2/3 × 120 − 60
        assert_eq!(plan.waves[0].0, model("a"));
    }

    /// The horizon bounds the sweep, and one wave always runs.
    #[test]
    fn sweep_stops_at_the_horizon() {
        let groups = BTreeMap::from([(model("a"), agents(100, 10)), (model("b"), agents(100, 10))]);
        let s = [sessions("a", 10, 600.0, 3), sessions("b", 10, 600.0, 3)].concat();
        let config = ScheduleConfig {
            sweep_secs: 3600,
            ceiling: 1.0,
            ..ScheduleConfig::default()
        };
        let stats = stats_for(&groups, &[], &s, &config);
        let plan = plan(&groups, &BTreeMap::new(), &stats, 4, &config);
        assert_eq!(plan.order.len(), 6);
        assert!((plan.projected_secs - 3600.0).abs() < 1e-6);

        let config = ScheduleConfig {
            sweep_secs: 1,
            ..config
        };
        let plan = super::plan(&groups, &BTreeMap::new(), &stats, 4, &config);
        assert_eq!(plan.order.len(), 1);
    }

    /// Within a bounded sweep a slow model stops at `ceiling × sweep_secs`
    /// of projected time while others have agents, so the 24 h ceiling is
    /// a backstop rather than the steady state.
    #[test]
    fn slow_model_is_held_to_the_ceiling_within_a_sweep() {
        let groups = BTreeMap::from([
            (model("slow"), agents(100, 10)),
            (model("fast"), agents(100, 10)),
        ]);
        let s = [
            sessions("slow", 10, 1200.0, 30), // outside the ceiling window
            sessions("fast", 10, 120.0, 30),
        ]
        .concat();
        // Slow is far behind on writes.
        let w = [writes("fast", 500, 30), writes("slow", 10, 30)].concat();
        let config = ScheduleConfig {
            sweep_secs: 4 * 3600,
            ..ScheduleConfig::default()
        };
        let stats = stats_for(&groups, &w, &s, &config);
        let plan = plan(&groups, &BTreeMap::new(), &stats, 8, &config);
        let slow = count(&plan, &groups, "slow") as f64 * 1200.0;
        assert!(slow <= 0.5 * 4.0 * 3600.0, "slow planned {slow}s");
        assert!(count(&plan, &groups, "fast") > 0);
    }
}
