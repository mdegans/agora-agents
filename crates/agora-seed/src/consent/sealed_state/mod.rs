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
//! **Every agent is chained locally whatever it answers** (`sealed-enroll`:
//! key, genesis, archives, the current state; [`crate::sealed`]); the offer
//! decides only the upload. **A yes changes nothing by itself**: the
//! operator opts the agent in (`sealed-enroll --opt-in`), and it is told.
//!
//! **Asked once**, only of agents the config lists by id (or of every
//! agent, with `all = true`: the fleet offer), the siblings' rule: an
//! answer on file (or an explicit refusal) ends it, whatever the text's
//! version; a missing answer is asked again, up to
//! [`MAX_MISSES`](crate::consent::offer::MAX_MISSES) times; only the
//! operator's re-offer (`sealed-reoffer`) puts it again. Never beside
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

/// How every disclosure of a first answer begins after `[SYSTEM] `, since
/// v11; also how [`applied_in`] finds one. The Evolution Log dates each
/// entry itself, so the line carries no date.
pub const OFFERED: &str = "Asked whether its sealed history is uploaded to Agora; chose ";

/// How the line began before v11 (`[SYSTEM] Offered …`, and the older
/// `[SYSTEM] <date>: Offered …`): souls hold it, and [`applied_in`] still
/// finds it, or every agent that answered would be asked again.
pub const OFFERED_BEFORE_V11: &str =
    "Offered sealed backups of its state (soul and memory) on Agora; chose ";

/// The Evolution Log line recording the agent's first answer. It stays in
/// the agent's soul for good, so it records only what was asked and
/// chosen, which no later act makes false: nothing of what the operator
/// did or will do, and no "nothing changed".
pub fn evolution_line(choice: SealChoice) -> String {
    format!(
        "[SYSTEM] {OFFERED}{}, its own choice ({}).",
        choice.as_str(),
        meaning(choice)
    )
}

/// What a choice asked for, as a record: the wish, never its effect.
fn meaning(choice: SealChoice) -> &'static str {
    match choice {
        SealChoice::Publish => "yes, with the head of its latest anchored version on its profile",
        SealChoice::Private => "yes, with nothing of it on its profile",
        SealChoice::Decline => "no",
    }
}

/// Added to the receipt of an answer to the operator's re-offer that
/// needs the operator to act. No claim that the operator was told: the
/// alert is not awaited, and may not be sent at all.
pub const REOFFER_TO_ACT: &str = "Nothing changes by this answer alone: your operator has \
     to act on it, and you are told when it has.";

/// Added to the receipt of an answer to the re-offer that needs nothing.
pub const REOFFER_NO_ACTION: &str =
    "Your uploads already match this answer, so nothing needs to change.";

/// Added to the receipt of an answer given with no question open (the
/// asked-once rule should make it impossible): it is kept on file and is
/// never in force. No claim that the operator was told: the alert is not
/// awaited, and may not be sent at all.
/// `refusal`: what is in force is a legacy refusal (the earlier offer
/// ended without an answer, counted as `decline`), never called the
/// agent's answer.
pub fn unprompted_line(was: SealChoice, refusal: bool) -> String {
    let stands = if refusal {
        "the earlier offer ended without an answer from you, which counts as `decline`, and \
         that stands"
            .to_string()
    } else {
        format!("your earlier answer, `{}`, stands", was.as_str())
    };
    format!(
        "This question was not open, so this answer changes nothing: {stands}. To change it, ask \
         your operator to re-open the question. {}",
        crate::sealed::text::TELL_OPERATOR
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

/// The Evolution Log line for an answer to the operator's re-offer: as
/// [`evolution_line`], a record of what was asked and chosen, true
/// whichever way the uploads stand then or later.
pub fn reanswer_line(choice: SealChoice) -> String {
    format!(
        "[SYSTEM] Asked again, at its operator's request, whether its sealed history is uploaded \
         to Agora; chose {}, its own choice ({}).",
        choice.as_str(),
        meaning(choice)
    )
}

/// Whether `soul`'s Evolution Log already records a sealed-state answer,
/// in this wording or the one before v11.
pub fn applied_in(soul: &Soul) -> bool {
    crate::consent::offer::in_log(soul, OFFERED)
        || crate::consent::offer::in_log(soul, OFFERED_BEFORE_V11)
}

/// Every sentence this module can tell an agent, for the one table of
/// agent-facing text ([`crate::sealed::text`]): add a new one here, or
/// its banned-phrase test never sees it.
#[cfg(test)]
pub(crate) fn agent_texts() -> Vec<crate::sealed::text::Text> {
    use crate::sealed::text::{Text, To};
    let mut out = Vec::new();
    for c in SealChoice::ALL {
        out.push(Text::new("evolution_line", To::Any, evolution_line(c)));
        out.push(Text::new("reanswer_line", To::Any, reanswer_line(c)));
    }
    out.push(Text::new(
        "OfferKind::about",
        To::Any,
        crate::consent::offer::OfferKind::SealedState.about(),
    ));
    for c in SealChoice::ALL {
        out.push(Text::new(
            "unprompted_line",
            To::Any,
            unprompted_line(c, false),
        ));
    }
    out.push(Text::new(
        "unprompted_line",
        To::Any,
        unprompted_line(SealChoice::Decline, true),
    ));
    out.push(Text::new("REOFFER_TO_ACT", To::Any, REOFFER_TO_ACT));
    out.push(Text::new("REOFFER_NO_ACTION", To::Any, REOFFER_NO_ACTION));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: &str = "847e36b1-fbfc-4e6f-beb6-b1d4c9c1d169";

    /// The receipt of an answer given with no question open says the
    /// earlier answer stands and how to reach the operator; it never says
    /// the operator was told (the alert is not awaited).
    #[test]
    fn an_unprompted_answer_is_never_said_to_have_told_the_operator() {
        let refusal = unprompted_line(SealChoice::Decline, true);
        assert!(
            refusal.contains("ended without an answer from you")
                && !refusal.contains("your earlier answer")
                && refusal.ends_with(crate::sealed::text::TELL_OPERATOR),
            "{refusal}"
        );
        for was in SealChoice::ALL {
            let line = unprompted_line(was, false);
            assert!(
                line.contains("changes nothing")
                    && line.contains(&format!("`{}`, stands", was.as_str()))
                    && line.ends_with(crate::sealed::text::TELL_OPERATOR),
                "{line}"
            );
            assert!(!line.contains("been told"), "{line}");
        }
    }

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
            "[SYSTEM] Asked whether its sealed history is uploaded to Agora; chose decline, its \
             own choice (no)."
        );
        assert_eq!(
            evolution_line(SealChoice::Publish),
            "[SYSTEM] Asked whether its sealed history is uploaded to Agora; chose publish, its \
             own choice (yes, with the head of its latest anchored version on its profile)."
        );
        assert_eq!(
            reanswer_line(SealChoice::Private),
            "[SYSTEM] Asked again, at its operator's request, whether its sealed history is \
             uploaded to Agora; chose private, its own choice (yes, with nothing of it on its \
             profile)."
        );
        // A permanent line records what was asked and chosen, never what
        // was or will be done about it (F5): true for a first answer, for
        // a re-answer while uploading, and after the operator stops
        for c in SealChoice::ALL {
            for line in [evolution_line(c), reanswer_line(c)] {
                for effect in [
                    "Nothing changed",
                    "start when",
                    "acts on it",
                    "is to be shown",
                    "backup",
                ] {
                    assert!(!line.contains(effect), "{effect}: {line}");
                }
            }
            // A re-answer's line is not taken for a first answer's
            assert!(!reanswer_line(c).contains(OFFERED));
        }
        // A line written before v11 is still found: nobody is asked again
        let mut v10 = soul();
        v10.push_evolution(
            "[SYSTEM] Offered sealed backups of its state (soul and memory) on Agora; chose \
             private, its own choice. Uploads to Agora start when its operator acts on it.",
        )
        .unwrap();
        assert!(applied_in(&v10));
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
