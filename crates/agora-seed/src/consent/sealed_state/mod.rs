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
//! same way ([`super::questions`]), its three options
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
//! **Asked once**, only of agents the config lists by id (or of every
//! agent, with `all = true`: the fleet offer), the siblings' rule: an answer on file (or an explicit refusal) ends it; a missing
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
//! # all = true    # ask every agent; `agents` is then ignored (default false)
//! # ask = false   # ask nobody new; enrolled agents stay sealed either way
//! ```

pub mod ledger;
pub mod prompt;

use std::collections::BTreeSet;

use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::seed::Soul;
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
    /// Who is asked, by agent id: the beta allowlist. Ignored with
    /// [`Self::all`]. A bad entry is skipped and alerted, never failing
    /// the whole config.
    #[serde(default)]
    pub agents: Vec<ListedId>,
    /// Every agent is asked (the fleet offer); `agents` is ignored.
    /// Default `false`.
    #[serde(default)]
    pub all: bool,
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
        if self.all {
            tracing::info!(
                event_type = "sealed_state_offer",
                ask = self.ask,
                all = true,
                ignored = self.agents.len(),
                "sealed-state offer resolved: every agent (`agents` ignored)"
            );
            return Ok(Some(SealedOffer {
                agents: BTreeSet::new(),
                all: true,
                ask: self.ask,
            }));
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
            all: false,
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
    if config.enabled && !config.all && bad > 0 {
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
    /// Every agent (`all = true`).
    all: bool,
    ask: bool,
}

impl SealedOffer {
    /// Whether `id` is put the question (if it is due).
    pub fn admits(&self, id: AgentId) -> bool {
        self.ask && (self.all || self.agents.contains(&id))
    }

    /// The listed ids: none with `all`.
    pub fn ids(&self) -> impl Iterator<Item = AgentId> + '_ {
        self.agents.iter().copied()
    }

    #[cfg(test)]
    pub fn for_agents(ids: &[AgentId]) -> Self {
        Self {
            agents: ids.iter().copied().collect(),
            all: false,
            ask: true,
        }
    }
}

/// How every disclosure begins after `[SYSTEM] `; also how [`applied_in`]
/// finds one, the older ones (`[SYSTEM] <date>: Offered …`) included. The
/// Evolution Log dates each entry itself, so the line carries no date.
pub const OFFERED: &str = "Offered sealed backups of its state (soul and memory) on Agora; chose ";

/// The Evolution Log line recording the agent's answer.
pub fn evolution_line(choice: SealChoice) -> String {
    let meaning = match choice {
        SealChoice::Publish => {
            "Uploads to Agora start when its operator acts on it, and its anchored head is to be \
             shown on its profile."
        }
        SealChoice::Private => "Uploads to Agora start when its operator acts on it.",
        SealChoice::Decline => "Nothing changed.",
    };
    format!(
        "[SYSTEM] {OFFERED}{}, its own choice. {meaning}",
        choice.as_str()
    )
}

/// Why `sealed-reoffer` left an agent as it was.
#[derive(Debug, thiserror::Error)]
pub enum ReofferError {
    #[error(transparent)]
    Not(#[from] ledger::NoReoffer),
    /// Unreadable or too new: neither asked nor saved over.
    #[error("its sealed-state ledger is unreadable ({0}); nothing written")]
    Unreadable(std::io::Error),
    #[error("writing its sealed-state ledger: {0}")]
    Write(std::io::Error),
}

/// `agora-seed sealed-reoffer`: re-open the sealed-state question for the
/// agent in `agent_dir`, put again once at its next session with `note`
/// (required, shown to it). `Ok(true)`: a pending re-offer was replaced.
/// The caller holds the runner lock exclusively, so no session saves the
/// ledger over this write. The note is never logged.
pub async fn reoffer(
    agent_dir: &std::path::Path,
    id: AgentId,
    note: &str,
) -> Result<bool, ReofferError> {
    use crate::consent::offer::Stored as _;
    let mut ledger = ledger::SealLedger::load(agent_dir)
        .await
        .map_err(ReofferError::Unreadable)?;
    let replaced = ledger.reopen(note, chrono::Utc::now())?;
    ledger.save(agent_dir).await.map_err(ReofferError::Write)?;
    tracing::warn!(
        event_type = "sealed_state_reoffered",
        agent_id = %id,
        replaced,
        reopens = ledger.choice().map(SealChoice::as_str),
        note_chars = note.trim().chars().count(),
        "the operator re-opened the sealed-state question; it is put at the agent's next session"
    );
    Ok(replaced)
}

/// The Evolution Log line for an answer to the operator's re-offer: true
/// whichever way the uploads stand, since the operator acts on it.
pub fn reanswer_line(choice: SealChoice) -> String {
    format!(
        "[SYSTEM] Asked again, at its operator's request, about sealed backups of its state on \
         Agora; chose {}, its own choice. Its operator acts on it.",
        choice.as_str()
    )
}

/// Whether `soul`'s Evolution Log already records a sealed-state answer.
pub fn applied_in(soul: &Soul) -> bool {
    crate::consent::offer::in_log(soul, OFFERED)
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

        for bad in ["agents = []\n", "enabled = true\nevery = true\n"] {
            assert!(toml::from_str::<SealedStateConfig>(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn all_offers_every_agent_and_ignores_the_list() {
        let all: SealedStateConfig = toml::from_str(&format!(
            "enabled = true\nall = true\nagents = [\"{ID}\"]\n"
        ))
        .unwrap();
        let offer = all.resolve().unwrap().unwrap();
        assert!(offer.admits(AgentId::new()), "anyone");
        assert!(offer.admits(ID.parse().unwrap()));
        assert_eq!(offer.ids().count(), 0, "the list is ignored");
        let bare: SealedStateConfig = toml::from_str("enabled = true\nall = true\n").unwrap();
        assert!(bare.resolve().unwrap().unwrap().admits(AgentId::new()));
        let quiet: SealedStateConfig =
            toml::from_str("enabled = true\nall = true\nask = false\n").unwrap();
        assert!(!quiet.resolve().unwrap().unwrap().admits(AgentId::new()));
        let off: SealedStateConfig = toml::from_str("enabled = false\nall = true\n").unwrap();
        assert!(off.resolve().unwrap().is_none());
        // Default false: the list decides
        let listed: SealedStateConfig =
            toml::from_str(&format!("enabled = true\nagents = [\"{ID}\"]\n")).unwrap();
        assert!(!listed.all);
        assert!(!listed.resolve().unwrap().unwrap().admits(AgentId::new()));
    }

    #[test]
    fn a_bad_table_disables_the_offer_and_alerts() {
        let c: SealedStateConfig = toml::from_str("enabled = true\n").unwrap();
        assert!(resolve_or_disable(&c, &crate::alerts::Alerter::off()).is_none());
    }

    #[test]
    fn the_evolution_line_records_every_answer_and_is_found() {
        assert_eq!(
            evolution_line(SealChoice::Decline),
            "[SYSTEM] Offered sealed backups of its state (soul and memory) on Agora; chose \
             decline, its own choice. Nothing changed."
        );
        // A line from before the date was dropped is still found
        let mut old = soul();
        old.push_evolution(
            "[SYSTEM] 2026-11-02: Offered sealed backups of its state (soul and memory) on \
             Agora; chose decline by its own choice. Nothing changed.",
        )
        .unwrap();
        assert!(applied_in(&old));
        for c in SealChoice::ALL {
            let line = evolution_line(c);
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
