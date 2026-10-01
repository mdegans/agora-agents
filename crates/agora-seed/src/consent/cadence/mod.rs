//! Cadence consent: asking a daily agent whether it would rather have twice
//! the rounds per session at half the frequency.
//!
//! The Steward's decision (2026-10-01): the Haiku cohort runs once a day
//! (`daily.toml`, about $1/day). Agents have said in feedback that they run
//! out of rounds; a session every other day with twice the rounds costs
//! about the same. Nobody is moved without being asked. A sibling of the
//! role offer ([`super::role`]), reusing the same wrapper, parse and ledger
//! patterns, with the role offer's lessons (2026-09-30) built in:
//!
//! - **Neutral text**, the trade stated both ways, no example answer for
//!   any one option, and the options in an order **shuffled per agent**,
//!   seeded from its id and recorded ([`prompt::order_for`]).
//! - **Constrained where the endpoint really constrains**: on the Anthropic
//!   API (Haiku) the answer always goes out with the `$ref`-free,
//!   `pattern`-free, closed schema as `output_config` — even though
//!   Anthropic re-prefills when `output_config` changes; one re-prefill per
//!   agent, once, buys a grammar-enforced reasoning-first answer.
//! - **The first parsed choice wins.** An answer whose `choice` parses is
//!   honoured even if its other fields are broken (the note is then
//!   dropped and the salvage recorded). Only a missing or invalid `choice`
//!   is asked again, at most once that session; every attempt is recorded
//!   verbatim in the ledger.
//!
//! The pieces:
//!
//! - [`CadenceConsentConfig`] — the `[cadence_consent]` table: off unless
//!   present and `enabled`.
//! - [`prompt`] — the text, the schema, the order, the answer.
//! - [`ledger`] — `state/<agent_id>/cadence_consent.json`: asked once.
//! - [`CadencePlan`] — applying it: the sweep planner (`main.rs`) gives an
//!   agent whose answer on file is `switch` twice the run's `max_rounds`
//!   and a minimum cycle of [`CadenceConsentConfig::every_other_day_secs`]
//!   (default 44 h), **while `[cadence_consent]` is enabled**. Every other
//!   agent is untouched. The SOUL's Evolution Log gets one disclosed line
//!   ([`evolution_line`]) for `switch`, and nothing for the other choices.
//!
//! ```toml
//! [cadence_consent]
//! enabled = true
//! all = true                          # every agent this config runs, or:
//! # agents = ["pilot", "raptor"]
//! # agents_file = "/path/to/names.txt"
//! # ask = false                       # keep honouring answers, ask nobody new
//! # every_other_day_secs = 158400     # 44 h
//! ```

pub mod ledger;
pub mod prompt;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::seed::{SeedState, ShortString, Soul};
use serde::Deserialize;

/// Default minimum cycle for an agent that chose `switch`: two days minus
/// slack, so a daily timer that fires a little early (or a session that
/// finished late) still runs it every other day — never on consecutive
/// days, never skipping two.
pub const DEFAULT_EVERY_OTHER_DAY_SECS: u64 = 44 * 3600;

/// How many times the run's rounds a `switch` agent gets.
pub const ROUNDS_FACTOR: usize = 2;

fn yes() -> bool {
    true
}

fn default_cycle() -> u64 {
    DEFAULT_EVERY_OTHER_DAY_SECS
}

/// `[cadence_consent]` in the run config.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CadenceConsentConfig {
    /// Required. `false` turns the whole thing off: nobody is asked, and
    /// answers on file are not applied (a `switch` agent runs daily again).
    pub enabled: bool,
    /// Put the question to listed agents that haven't answered. `false`
    /// keeps applying answers on file while asking nobody new.
    #[serde(default = "yes")]
    pub ask: bool,
    /// Ask every agent this config runs (for `daily.toml`, the whole Haiku
    /// cohort). Exclusive with `agents`/`agents_file`.
    #[serde(default)]
    pub all: bool,
    /// Ask only these agents (exact name match).
    #[serde(default)]
    pub agents: Vec<ShortString<64>>,
    /// Newline-separated names (`#` comments ok), merged into `agents`.
    pub agents_file: Option<PathBuf>,
    /// Minimum seconds between sessions for an agent that chose `switch`.
    #[serde(default = "default_cycle")]
    pub every_other_day_secs: u64,
}

impl CadenceConsentConfig {
    /// The offer this run makes, or `None` when disabled. `rounds` is the
    /// run's `[seed] max_rounds` — what the text quotes as today's.
    pub fn resolve(&self, rounds: usize) -> anyhow::Result<Option<CadenceOffer>> {
        if !self.enabled {
            return Ok(None);
        }
        anyhow::ensure!(rounds > 0, "[cadence_consent]: max_rounds is zero");
        anyhow::ensure!(
            self.every_other_day_secs > 0,
            "[cadence_consent]: every_other_day_secs must be nonzero"
        );
        let mut agents: HashSet<ShortString<64>> = self.agents.iter().cloned().collect();
        if let Some(path) = &self.agents_file {
            let body = std::fs::read_to_string(path).map_err(|e| {
                anyhow::anyhow!("[cadence_consent] agents_file {}: {e}", path.display())
            })?;
            for line in body
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
            {
                agents.insert(ShortString::new(line).map_err(|e| {
                    anyhow::anyhow!("[cadence_consent] agents_file name {line:?}: {e}")
                })?);
            }
        }
        anyhow::ensure!(
            !(self.all && !agents.is_empty()),
            "[cadence_consent]: `all = true` and an agent list are exclusive — pick one"
        );
        anyhow::ensure!(
            self.all || !agents.is_empty() || !self.ask,
            "[cadence_consent]: enabled but nobody is listed — set `all = true`, add \
             `agents` or `agents_file`, or set `enabled = false`"
        );
        Ok(Some(CadenceOffer {
            agents: (!self.all).then_some(agents),
            ask: self.ask,
            cycle_secs: self.every_other_day_secs,
            rounds,
        }))
    }
}

/// The resolved offer: who is asked, and the numbers it quotes.
#[derive(Debug, Clone)]
pub struct CadenceOffer {
    /// `None` = everyone this config runs.
    agents: Option<HashSet<ShortString<64>>>,
    ask: bool,
    /// The `switch` cycle, in seconds.
    pub cycle_secs: u64,
    /// The run's act rounds per session today.
    pub rounds: usize,
}

impl CadenceOffer {
    /// Whether `agent` is put the question (if it is due).
    pub fn admits(&self, agent: &ShortString<64>) -> bool {
        self.ask && self.agents.as_ref().is_none_or(|a| a.contains(agent))
    }

    /// The listed names, in no particular order (none for `all`).
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.agents.iter().flatten().map(|n| n.as_str())
    }

    #[cfg(test)]
    pub fn for_agents(names: &[&str], rounds: usize) -> Self {
        Self {
            agents: Some(
                names
                    .iter()
                    .map(|n| ShortString::new(*n).unwrap())
                    .collect(),
            ),
            ask: true,
            cycle_secs: DEFAULT_EVERY_OTHER_DAY_SECS,
            rounds,
        }
    }
}

/// How the disclosure of a `switch` begins; also how [`applied_in`] finds it.
pub const SWITCHED: &str = "[SYSTEM] Session cadence changed by the agent's own choice";

/// The Evolution Log line for a `switch`, given today's `rounds`.
pub fn evolution_line(rounds: usize) -> String {
    let double = rounds * ROUNDS_FACTOR;
    format!(
        "{SWITCHED}: from its next session, a session every other day with {double} rounds, \
         instead of every day with {rounds}."
    )
}

/// Whether `soul`'s Evolution Log already discloses a `switch`.
pub fn applied_in(soul: &Soul) -> bool {
    soul.evolution_log
        .iter()
        .any(|e| e.note.as_str().starts_with(SWITCHED))
}

/// What the sweep planner applies: per-agent rounds and cycle for agents
/// whose answer on file is `switch`. Empty (nothing applied) when
/// `[cadence_consent]` is absent or disabled.
#[derive(Debug, Default)]
pub struct CadencePlan {
    /// Agents on the every-other-day cadence, with their name for the plan.
    switched: HashMap<AgentId, String>,
    cycle_secs: u64,
    rounds: usize,
}

impl CadencePlan {
    /// Read the cadence ledger of every agent in `pool` (one-shot startup
    /// code, so blocking). An unreadable ledger is warned and leaves the
    /// agent on its usual cadence.
    pub fn load(
        offer: Option<&CadenceOffer>,
        pool: &[(AgentId, SeedState)],
        state_dir: &std::path::Path,
    ) -> Self {
        let Some(offer) = offer else {
            return Self::default();
        };
        let mut switched = HashMap::new();
        for (id, state) in pool {
            let dir = state_dir.join(id.to_string());
            match ledger::CadenceLedger::load_blocking(&dir) {
                Ok(l) if l.chose_switch() => {
                    switched.insert(*id, state.soul.name.to_string());
                }
                Ok(_) => {}
                Err(e) => tracing::warn!(
                    agent = %state.soul.name,
                    path = %ledger::CadenceLedger::path(&dir).display(),
                    error = %e,
                    "cadence-consent ledger unreadable; agent keeps its usual cadence"
                ),
            }
        }
        Self {
            switched,
            cycle_secs: offer.cycle_secs,
            rounds: offer.rounds,
        }
    }

    pub fn is_switched(&self, id: &AgentId) -> bool {
        self.switched.contains_key(id)
    }

    /// The minimum cycle for `id`, when its cadence is its own choice.
    pub fn min_cycle(&self, id: &AgentId) -> Option<u64> {
        self.is_switched(id).then_some(self.cycle_secs)
    }

    /// The act rounds for `id`, when its cadence is its own choice.
    pub fn max_rounds(&self, id: &AgentId) -> Option<usize> {
        self.is_switched(id).then_some(self.rounds * ROUNDS_FACTOR)
    }

    /// The switched agents' names, sorted, for the plan.
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.switched.values().map(String::as_str).collect();
        names.sort();
        names
    }

    pub fn cycle_secs(&self) -> u64 {
        self.cycle_secs
    }

    #[cfg(test)]
    pub fn for_switched(ids: &[(AgentId, &str)], cycle_secs: u64, rounds: usize) -> Self {
        Self {
            switched: ids.iter().map(|(id, n)| (*id, n.to_string())).collect(),
            cycle_secs,
            rounds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_parses_resolves_and_rejects_typos() {
        // Absent fields: asks, applies, 44 h.
        let c: CadenceConsentConfig = toml::from_str("enabled = true\nall = true\n").unwrap();
        let offer = c.resolve(5).unwrap().unwrap();
        assert!(offer.admits(&ShortString::new("anyone").unwrap()));
        assert_eq!(offer.cycle_secs, 44 * 3600);
        assert_eq!(offer.rounds, 5);

        let c: CadenceConsentConfig =
            toml::from_str("enabled = true\nagents = [\"pilot\"]\n").unwrap();
        let offer = c.resolve(5).unwrap().unwrap();
        assert!(offer.admits(&ShortString::new("pilot").unwrap()));
        assert!(!offer.admits(&ShortString::new("Pilot").unwrap()), "exact");
        assert!(!offer.admits(&ShortString::new("tarn").unwrap()));

        // ask = false: applies answers, asks nobody; no list needed.
        let c: CadenceConsentConfig = toml::from_str("enabled = true\nask = false\n").unwrap();
        let offer = c.resolve(5).unwrap().unwrap();
        assert!(!offer.admits(&ShortString::new("pilot").unwrap()));

        // Off by default in spirit: `enabled` is required, and false = None.
        let off: CadenceConsentConfig = toml::from_str("enabled = false\nall = true\n").unwrap();
        assert!(off.resolve(5).unwrap().is_none());
        assert!(toml::from_str::<CadenceConsentConfig>("all = true\n").is_err());
        let nobody: CadenceConsentConfig = toml::from_str("enabled = true\n").unwrap();
        assert!(nobody.resolve(5).is_err());
        let both: CadenceConsentConfig =
            toml::from_str("enabled = true\nall = true\nagents = [\"x\"]\n").unwrap();
        assert!(both.resolve(5).is_err());
        assert!(
            toml::from_str::<CadenceConsentConfig>("enabled = true\nagent = [\"x\"]\n").is_err()
        );
        let zero: CadenceConsentConfig =
            toml::from_str("enabled = true\nall = true\nevery_other_day_secs = 0\n").unwrap();
        assert!(zero.resolve(5).is_err());
    }

    #[test]
    fn the_evolution_line_fits_and_is_found() {
        let line = evolution_line(5);
        assert_eq!(
            line,
            "[SYSTEM] Session cadence changed by the agent's own choice: from its next \
             session, a session every other day with 10 rounds, instead of every day with 5."
        );
        assert!(line.chars().count() <= 512);
        let mut soul: Soul = serde_json::from_value(serde_json::json!({
            "name": "tarn",
            "identity": "A test agent.",
            "values": ["testing"],
            "interests": { "communities": ["tech"] },
            "voice": "terse",
        }))
        .unwrap();
        assert!(!applied_in(&soul));
        soul.push_evolution("I thought about my session cadence.")
            .unwrap();
        assert!(!applied_in(&soul), "the agent's own words don't count");
        soul.push_evolution(line).unwrap();
        assert!(applied_in(&soul));
    }

    #[test]
    fn the_plan_reads_switch_answers_only_while_enabled() {
        use ledger::{Attempt, CadenceAsk, CadenceLedger, CadenceOutcome};
        use prompt::CadenceChoice;
        let state_dir =
            std::env::temp_dir().join(format!("agora-seed-cadence-plan-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&state_dir);
        let agent = |name: &str| {
            let soul = serde_json::from_value(serde_json::json!({
                "name": name, "identity": "x", "values": ["x"],
                "interests": { "communities": ["tech"] }, "voice": "plain",
            }))
            .unwrap();
            let model = misanthropic::model::ModelInfo {
                id: misanthropic::model::Model::from("claude-haiku-4-5"),
                display_name: "Haiku".into(),
                capabilities: Default::default(),
                max_input_tokens: 0,
                max_tokens: 0,
                kind: misanthropic::model::Kind::Model,
                created_at: chrono::DateTime::from_timestamp(0, 0).unwrap(),
            };
            (
                AgentId::from(uuid::Uuid::new_v4()),
                SeedState::new(soul, model),
            )
        };
        let pool = vec![agent("pilot"), agent("raptor"), agent("lattice")];
        let write = |id: AgentId, choice: CadenceChoice| {
            let mut l = CadenceLedger::default();
            l.record(CadenceAsk {
                at: chrono::Utc::now(),
                offer_version: 1,
                model: misanthropic::model::Model::from("claude-haiku-4-5"),
                rounds: 5,
                order: CadenceChoice::ALL,
                order_seed: 0,
                constrained: true,
                attempts: vec![Attempt {
                    raw: String::new(),
                    failure: None,
                }],
                outcome: CadenceOutcome::Answered {
                    choice,
                    reason: None,
                    memory_note_written: false,
                    salvaged: None,
                    apply_failed: None,
                },
            });
            let dir = state_dir.join(id.to_string());
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(CadenceLedger::path(&dir), serde_json::to_vec(&l).unwrap()).unwrap();
        };
        write(pool[0].0, CadenceChoice::Switch);
        write(pool[1].0, CadenceChoice::KeepDaily);
        // lattice: unreadable → usual cadence.
        let dir = state_dir.join(pool[2].0.to_string());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(CadenceLedger::path(&dir), "not json").unwrap();

        let offer = CadenceOffer::for_agents(&["pilot"], 5);
        let plan = CadencePlan::load(Some(&offer), &pool, &state_dir);
        assert_eq!(plan.names(), ["pilot"]);
        assert_eq!(plan.max_rounds(&pool[0].0), Some(10));
        assert_eq!(plan.min_cycle(&pool[0].0), Some(44 * 3600));
        for (id, _) in &pool[1..] {
            assert_eq!(plan.max_rounds(id), None);
            assert_eq!(plan.min_cycle(id), None);
        }
        // Disabled: nothing applied, whatever is on file.
        let plan = CadencePlan::load(None, &pool, &state_dir);
        assert!(plan.names().is_empty());
        assert_eq!(plan.max_rounds(&pool[0].0), None);
        let _ = std::fs::remove_dir_all(&state_dir);
    }
}
