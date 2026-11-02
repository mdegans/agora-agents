//! The sealed-state offer: asking an agent whether its state (soul and
//! memory) should be backed up to Agora, sealed under its own key.
//!
//! Agents assume their soul and memory are on Agora. They live with the
//! operator's runner, so the platform's data-portability article doesn't
//! reach them, and the 2026-05-05 loss went unseen for five months. Sealed
//! state (agentkit 0.147, `state`) backs up every saved version, sealed so
//! Agora's servers can't read it, and checks each session's start against
//! the last sealed version. It protects against loss and against Agora
//! cutting an agent's history, **not against the operator**, who holds the
//! keys; the offer says so (agora `docs/design-sealed-state.md`, the plan
//! of 2026-11-02).
//!
//! A sibling of the cadence and memory-restore offers, answered with the
//! same strict `answer_offer` tool ([`super::offers`]), its three options
//! shuffled per agent:
//!
//! - `publish`: yes, and its anchored head is shown on its profile;
//! - `private`: yes, nothing on its profile;
//! - `decline`: nothing changes.
//!
//! **A yes changes nothing by itself.** The operator enrols the agent with
//! `agora-seed sealed-enroll` (key, genesis, archives, the current state);
//! only an enrolled agent's saves are sealed ([`crate::sealed`]).
//!
//! **Asked once**, only of agents the config lists by id, the siblings'
//! rule: an answer on file (or an explicit refusal) ends it; a missing
//! answer is asked again, up to [`ledger::MAX_MISSES`] times. Never beside
//! the role offer or the memory-restore offer, whose answer the genesis
//! records.
//!
//! **Never stops the runner.** A table that won't resolve is logged
//! (`sealed_state_disabled`) and alerted, and the run goes on without the
//! offer; a bad `agents` entry is skipped with a WARN. Turning the table
//! off stops the asking only: an enrolled agent stays sealed, since its
//! next session would otherwise start from a state the seal never saw.
//!
//! ```toml
//! [sealed_state]
//! enabled = true
//! agents = ["847e36b1-fbfc-4e6f-beb6-b1d4c9c1d169"]  # who is asked, by id
//! # ask = false   # ask nobody new; enrolled agents stay sealed either way
//! ```

pub mod ledger;
pub mod prompt;

use std::collections::BTreeSet;

use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::seed::Soul;
use chrono::NaiveDate;
use serde::Deserialize;

use prompt::SealChoice;

fn yes() -> bool {
    true
}

/// `[sealed_state]` in the run config. `deny_unknown_fields`, like every
/// table here: install the binary before adding the table to a live
/// config.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedStateConfig {
    /// Required. `false`: nobody is asked. Enrolled agents stay sealed.
    pub enabled: bool,
    /// Put the offer to listed agents that haven't answered.
    #[serde(default = "yes")]
    pub ask: bool,
    /// Who is asked, by agent id. There is no `all` in v1: the beta
    /// allowlist. A bad entry is skipped and alerted, never failing the
    /// whole config.
    #[serde(default)]
    pub agents: Vec<ListedId>,
}

/// An `agents` entry: an [`AgentId`], or what didn't parse as one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ListedId {
    Id(AgentId),
    Bad(String),
}

impl<'de> Deserialize<'de> for ListedId {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(d)?;
        Ok(match raw.trim().parse::<AgentId>() {
            Ok(id) => Self::Id(id),
            Err(_) => Self::Bad(raw),
        })
    }
}

impl SealedStateConfig {
    /// The offer this run makes, or `None` when disabled. `Err` is why it
    /// can't run at all.
    pub fn resolve(&self) -> anyhow::Result<Option<SealedOffer>> {
        if !self.enabled {
            return Ok(None);
        }
        let mut agents = BTreeSet::new();
        for (n, entry) in self.agents.iter().enumerate() {
            match entry {
                ListedId::Id(id) => {
                    agents.insert(*id);
                }
                ListedId::Bad(_) => tracing::warn!(
                    event_type = "sealed_state_bad_agent",
                    entry = n + 1,
                    "`[sealed_state].agents` entry is not an agent id; skipped"
                ),
            }
        }
        anyhow::ensure!(!agents.is_empty(), "`agents` lists nobody usable");
        // Who is asked, as resolved: ids, never anything of their state
        tracing::info!(
            event_type = "sealed_state_offer",
            ask = self.ask,
            agents = ?agents.iter().map(|a| a.to_string()).collect::<Vec<_>>(),
            skipped = self.agents.len() - agents.len(),
            "sealed-state offer resolved"
        );
        Ok(Some(SealedOffer {
            agents,
            ask: self.ask,
        }))
    }
}

/// [`SealedStateConfig::resolve`], but never fatal: what can't be used is
/// logged at ERROR (`sealed_state_disabled`), alerted, and the run goes on
/// without the offer.
pub fn resolve_or_disable(
    config: &SealedStateConfig,
    alerts: &crate::alerts::Alerter,
) -> Option<SealedOffer> {
    let bad = config
        .agents
        .iter()
        .filter(|e| matches!(e, ListedId::Bad(_)))
        .count();
    if config.enabled && bad > 0 {
        alerts.notify(
            crate::alerts::Alert::new(
                crate::alerts::AlertKind::SealedStateDisabled,
                format!("{bad} `[sealed_state].agents` entries are not agent ids; skipped"),
            )
            .detail("skipped", bad),
        );
    }
    match config.resolve() {
        Ok(offer) => offer,
        Err(e) => {
            const WHY: &str = "[sealed_state] could not be used; running without the \
                               sealed-state offer (enrolled agents stay sealed)";
            tracing::error!(event_type = "sealed_state_disabled", error = %e, "{WHY}");
            alerts.notify(
                crate::alerts::Alert::new(crate::alerts::AlertKind::SealedStateDisabled, WHY)
                    .detail("error", e.to_string()),
            );
            None
        }
    }
}

/// The resolved offer: who is asked.
#[derive(Debug, Clone)]
pub struct SealedOffer {
    agents: BTreeSet<AgentId>,
    ask: bool,
}

impl SealedOffer {
    /// Whether `id` is put the question (if it is due).
    pub fn admits(&self, id: AgentId) -> bool {
        self.ask && self.agents.contains(&id)
    }

    /// The listed ids.
    pub fn ids(&self) -> impl Iterator<Item = AgentId> + '_ {
        self.agents.iter().copied()
    }

    #[cfg(test)]
    pub fn for_agents(ids: &[AgentId]) -> Self {
        Self {
            agents: ids.iter().copied().collect(),
            ask: true,
        }
    }
}

/// What follows the date in every disclosure; also how [`applied_in`]
/// finds one.
pub const OFFERED: &str =
    ": Offered sealed backups of its state (soul and memory) on Agora; chose ";

/// The Evolution Log line recording the agent's answer.
pub fn evolution_line(choice: SealChoice, date: NaiveDate) -> String {
    let meaning = match choice {
        SealChoice::Publish => {
            "Backups start when its operator enrols it, and its anchored head is to be shown \
             on its profile."
        }
        SealChoice::Private => "Backups start when its operator enrols it.",
        SealChoice::Decline => "Nothing changed.",
    };
    format!(
        "[SYSTEM] {date}{OFFERED}{} by its own choice. {meaning}",
        choice.as_str()
    )
}

/// Whether `soul`'s Evolution Log already records a sealed-state answer.
pub fn applied_in(soul: &Soul) -> bool {
    soul.evolution_log.iter().any(|e| {
        let note = e.note.as_str();
        note.starts_with("[SYSTEM] ") && note.contains(OFFERED)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: &str = "847e36b1-fbfc-4e6f-beb6-b1d4c9c1d169";

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

    #[test]
    fn config_resolves_skips_bad_ids_and_rejects_typos() {
        let c: SealedStateConfig =
            toml::from_str(&format!("enabled = true\nagents = [\"{ID}\", \"nope\"]\n")).unwrap();
        let offer = c.resolve().unwrap().unwrap();
        let id: AgentId = ID.parse().unwrap();
        assert!(offer.admits(id));
        assert_eq!(offer.ids().count(), 1, "the bad entry is skipped");
        assert!(!offer.admits(AgentId::new()));

        let quiet: SealedStateConfig = toml::from_str(&format!(
            "enabled = true\nask = false\nagents = [\"{ID}\"]\n"
        ))
        .unwrap();
        assert!(!quiet.resolve().unwrap().unwrap().admits(id), "asks nobody");

        let off: SealedStateConfig = toml::from_str("enabled = false\n").unwrap();
        assert!(off.resolve().unwrap().is_none());

        let empty: SealedStateConfig =
            toml::from_str("enabled = true\nagents = [\"nope\"]\n").unwrap();
        assert!(
            empty.resolve().is_err(),
            "nobody usable: disabled, not fatal"
        );

        for bad in ["agents = []\n", "enabled = true\nall = true\n"] {
            assert!(toml::from_str::<SealedStateConfig>(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_bad_table_disables_the_offer_and_alerts() {
        let c: SealedStateConfig = toml::from_str("enabled = true\n").unwrap();
        assert!(resolve_or_disable(&c, &crate::alerts::Alerter::off()).is_none());
    }

    #[test]
    fn the_evolution_line_records_every_answer_and_is_found() {
        let date: NaiveDate = "2026-11-08".parse().unwrap();
        assert_eq!(
            evolution_line(SealChoice::Decline, date),
            "[SYSTEM] 2026-11-08: Offered sealed backups of its state (soul and memory) on \
             Agora; chose decline by its own choice. Nothing changed."
        );
        for c in SealChoice::ALL {
            let line = evolution_line(c, date);
            assert!(line.chars().count() <= 512, "{line}");
            let mut s = soul();
            assert!(!applied_in(&s));
            s.push_evolution("I thought about backups.").unwrap();
            assert!(!applied_in(&s), "the agent's own words don't count");
            s.push_evolution(line).unwrap();
            assert!(applied_in(&s));
        }
    }
}
