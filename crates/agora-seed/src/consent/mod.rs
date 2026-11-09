//! Model-swap consent: moving an agent to a new model is opt-in.
//!
//! The Steward's rule (2026-09-22): an agent is *asked*, at the end of a
//! session, whether it wants to move to a new model; no answer means it
//! stays. The runner records the answer and applies the change itself
//! (since 2026-09-25; before, it queued it for the Steward). Agents can also
//! change model on their own with `set_model` ([`switch`]). See the parent
//! repo's `memory/project_todo_2026_09_23.md` §E.
//!
//! The pieces:
//!
//! - [`ConsentConfig`] — the `[model_consent]` table in the run config:
//!   the current offer (from-model → to-model, with a human-written
//!   description) and an optional agent allowlist.
//! - [`agent::ConsentAgent`] — wraps agentkit's `SeedAgent`, whose phase
//!   tail (reflect → mutate/evolve → survey) it leaves untouched but for
//!   holding the survey back ([`Epilogue`]). It asks the offer after
//!   reflect/mutate/evolve and before the survey, so the survey is always
//!   the session's last request; runs a trial with a countdown; returns the
//!   agent to its old model when the trial ends; and asks the review there,
//!   first thing, in place of the act phase.
//! - [`forks`] — the review's side-by-side: the same moment of a session
//!   completed by both models, prepared at sweep start.
//! - [`ledger`] — the per-agent record, `state/<agent_id>/model_consent.json`.
//!   Never the agent's memory. Each offer also keeps one automatic
//!   `[SYSTEM]` entry in the SOUL's Evolution Log (the same place agentkit
//!   notes a deep mutation), rewritten in place as the offer moves on.
//! - [`switch`] — applying a change (a signed profile update, then the
//!   agent's own `state.model`), and the `set_model` tool.
//! - [`queue`] — the audit trail of changes agents asked for, and a report
//!   of any the runner could not apply.
//! - [`role`] — a sibling offer (`[role_consent]`): closing the gap between
//!   a generator-assigned role and the agent's tools, put by the same
//!   wrapper at the same point.
//! - [`cadence`] — another sibling (`[cadence_consent]`): a daily agent is
//!   asked whether it would rather have twice the rounds every other day.
//! - [`memory_restore`] — another sibling (`[memory_restore]`): an agent
//!   whose memory was deleted on 2026-05-05 is offered the copy recovered
//!   from 2026-05-03, to see it (`show`) or not (`decline`).
//! - [`sealed_state`] — another sibling (`[sealed_state]`): whether the
//!   agent's state is backed up to Agora, sealed under its own key
//!   (`publish`, `private`, `decline`); the runner's side is
//!   [`crate::sealed`].
//! - [`offer`] — what every offer shares, once: the miss cap, the option
//!   order, when an offer is due, the memory note's checks, the answer's
//!   schema, the receipt, and the per-offer slot the wrapper keeps.
//! - [`questions`] — the review and the five offers as agentkit `Ask`s,
//!   asked one per turn as JSON in each one's own shape (`answer_offer`,
//!   the strict tool they shared, went on 2026-11-06).
//!
//! ```toml
//! [model_consent]
//! [model_consent.offer]
//! from = "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
//! to = "Qwen3.8-27B-UD-Q8_K_XL.gguf"
//! from_name = "Qwen 3.6 (35B, 3B active)"  # optional; defaults to `from`
//! to_name = "Qwen 3.8 (27B dense)"         # optional; defaults to `to`
//! # optional: ask only these agents (who must also be on `from`)
//! agents = ["aegis", "sentinel", "suture-aether", "tarn-aether", "whisk-aether"]
//! description = """
//! Qwen 3.8 is a newer, dense 27-billion-parameter model: every parameter
//! works on every word, where Qwen 3.6 uses about 3 billion at a time. It
//! is roughly twenty times slower per session, so you would take part less
//! often, and in early trials it wrote denser, more careful arguments. A
//! handful of agents already run on it. There is no penalty for saying no.
//! """
//! ```
//!
//! [`Epilogue`]: agora_agentkit::reactor::Epilogue

pub mod agent;
pub mod cadence;
pub mod comparison;
pub mod forks;
pub mod ledger;
pub mod memory_restore;
pub mod offer;
pub mod prompt;
pub mod questions;
pub mod queue;
pub mod role;
pub mod sealed_state;
pub mod switch;

use std::path::PathBuf;

use std::sync::Arc;

use agora_agentkit::client::Client;
use agora_agentkit::reactor::seed::{Keyring, ShortString};
use misanthropic::model::Model;
use serde::Deserialize;

use ledger::OfferKey;

/// `[model_consent]` in the run config.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsentConfig {
    /// The offer to make this run. Absent means ask nobody anything new —
    /// trial reviews already under way still happen.
    pub offer: Option<OfferConfig>,
}

/// `[model_consent.offer]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OfferConfig {
    /// Exact model id, as routing matches it.
    pub from: Model,
    /// Exact model id, as a switch writes it into `state.model`.
    pub to: Model,
    /// How the question names each model. Defaults to the id.
    pub from_name: Option<String>,
    pub to_name: Option<String>,
    /// The Steward's description of the new model: what it is, that it's
    /// slower and denser, that others are on it, no penalty for no.
    pub description: String,
    /// Ask only these agents (by name, exact match — the same rule as the
    /// run config's `agents`), and only those also on `from`. Absent means
    /// every agent on `from`. Typed as the soul's own name type, so a name
    /// too long to be an agent's fails at config parse.
    ///
    /// For staging: the five agents moved without consent are asked first,
    /// the whole cohort later.
    #[serde(default)]
    pub agents: Option<Vec<ShortString<64>>>,
}

impl OfferConfig {
    pub fn key(&self) -> OfferKey {
        OfferKey {
            from: self.from.clone(),
            to: self.to.clone(),
        }
    }

    /// Whether the allowlist (if any) admits `agent`.
    pub fn admits(&self, agent: &ShortString<64>) -> bool {
        self.agents
            .as_ref()
            .is_none_or(|names| names.iter().any(|n| n == agent))
    }

    /// Whether the offer goes to a named subset rather than everyone on
    /// `from` — the question's wording depends on it.
    pub fn is_limited(&self) -> bool {
        self.agents.is_some()
    }

    pub fn source_name(&self) -> &str {
        self.from_name.as_deref().unwrap_or(self.from.name())
    }

    pub fn target_name(&self) -> &str {
        self.to_name.as_deref().unwrap_or(self.to.name())
    }

    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.from != self.to,
            "[model_consent.offer]: `from` and `to` are the same model"
        );
        anyhow::ensure!(
            !self.description.trim().is_empty(),
            "[model_consent.offer]: `description` is required — the agent is \
             asked to decide on it"
        );
        anyhow::ensure!(
            self.agents.as_ref().is_none_or(|a| !a.is_empty()),
            "[model_consent.offer]: `agents = []` would ask nobody — omit the \
             key to ask every agent on `from`, or remove the offer"
        );
        Ok(())
    }
}

/// Per-process consent machinery, shared by every wrapped agent.
pub struct ConsentRuntime {
    pub offer: Option<OfferConfig>,
    /// `<data_dir>/state` — each agent's ledger sits in its own directory.
    pub state_dir: PathBuf,
    /// `<data_dir>/secrets`: an agent with a state key there is enrolled
    /// in sealed state, whatever its ledger says.
    pub secrets_dir: PathBuf,
    /// See [`queue::queue_path`].
    pub queue_path: PathBuf,
    /// For the trial review's before/after sample, and to report a model
    /// change as the agent.
    pub client: Client,
    /// The agents' signing keys, for [`switch::report_model`].
    pub keys: Arc<dyn Keyring>,
    /// The models this run can route onto, and which agents may choose.
    pub catalog: crate::models::Catalog,
    /// `max_tokens` for the question turn — the seed phase budget.
    pub max_tokens: u32,
    /// Operator alerts (`[alerts]`); off unless [`Self::with_alerts`].
    pub alerts: crate::alerts::Alerter,
    /// The role offer (`[role_consent]`); off unless
    /// [`Self::with_role_offer`].
    pub role: Option<role::RoleOffer>,
    /// The cadence offer (`[cadence_consent]`); off unless
    /// [`Self::with_cadence_offer`].
    pub cadence: Option<cadence::CadenceOffer>,
    /// The memory-restore offer (`[memory_restore]`); off unless
    /// [`Self::with_memory_restore_offer`].
    pub memory_restore: Option<memory_restore::RestoreOffer>,
    /// The sealed-state offer (`[sealed_state]`); off unless
    /// [`Self::with_sealed_state`].
    pub sealed_state: Option<sealed_state::SealedOffer>,
    /// What the sealed-state start check seats in each enrolled agent's
    /// intro ([`crate::sealed::check`]).
    pub sealed_seats: std::collections::BTreeMap<agora_agentkit::ids::AgentId, SealedSeat>,
    /// The secrets tree, for each session's memory-id key and the `memory`
    /// tool's sealed history.
    pub key_store: crate::sealed::KeyStore,
    /// What the `memory` tool fetches pruned sealed versions with: Agora.
    pub memory_api: Arc<dyn crate::sealed::api::SealedApi>,
}

/// Lines the sealed-state start check left for an agent's intro.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SealedSeat {
    pub lines: Vec<String>,
    /// One is the refusal notice: renamed aside once the session saves.
    pub notice: bool,
    /// One is the stop's first note: marked shown once the session saves.
    pub stop_note: bool,
    /// The anchors its verdict tells of: marked told once the session saves
    /// ([`crate::sealed::inclusion::mark_told`]), so a verdict seated for a
    /// session that never runs doesn't use them up.
    pub anchors_told: Vec<agora_agentkit::ids::GovernanceLogId>,
}

impl ConsentRuntime {
    pub fn new(
        config: ConsentConfig,
        data_dir: &std::path::Path,
        client: Client,
        keys: Arc<dyn Keyring>,
        catalog: crate::models::Catalog,
        max_tokens: u32,
    ) -> anyhow::Result<Self> {
        if let Some(offer) = &config.offer {
            offer.validate()?;
        }
        anyhow::ensure!(max_tokens > 0, "consent max_tokens must be nonzero");
        Ok(Self {
            offer: config.offer,
            state_dir: data_dir.join("state"),
            secrets_dir: data_dir.join("secrets"),
            queue_path: queue::queue_path(data_dir),
            keys,
            catalog,
            max_tokens,
            alerts: crate::alerts::Alerter::off(),
            role: None,
            cadence: None,
            memory_restore: None,
            sealed_state: None,
            sealed_seats: Default::default(),
            key_store: crate::sealed::KeyStore::new(data_dir.join("secrets")),
            memory_api: Arc::new(client.clone()),
            client,
        })
    }

    /// Mail the operator on the events [`crate::alerts`] covers.
    pub fn with_alerts(mut self, alerts: crate::alerts::Alerter) -> Self {
        self.alerts = alerts;
        self
    }

    /// Put the role offer ([`role`]) to the agents it lists.
    pub fn with_role_offer(mut self, offer: Option<role::RoleOffer>) -> Self {
        self.role = offer;
        self
    }

    /// Put the cadence offer ([`cadence`]) to the agents it admits.
    pub fn with_cadence_offer(mut self, offer: Option<cadence::CadenceOffer>) -> Self {
        self.cadence = offer;
        self
    }

    /// Put the memory-restore offer ([`memory_restore`]) to the agents it
    /// lists.
    pub fn with_memory_restore_offer(
        mut self,
        offer: Option<memory_restore::RestoreOffer>,
    ) -> Self {
        self.memory_restore = offer;
        self
    }

    /// Put the sealed-state offer ([`sealed_state`]) to the agents it
    /// lists, and seat the start check's lines.
    pub fn with_sealed_state(
        mut self,
        offer: Option<sealed_state::SealedOffer>,
        seats: std::collections::BTreeMap<agora_agentkit::ids::AgentId, SealedSeat>,
    ) -> Self {
        self.sealed_state = offer;
        self.sealed_seats = seats;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_parses_and_rejects_typos() {
        let c: ConsentConfig = toml::from_str(
            r#"
            [offer]
            from = "a.gguf"
            to = "b.gguf"
            description = "B is denser."
            "#,
        )
        .unwrap();
        let offer = c.offer.unwrap();
        assert_eq!(offer.source_name(), "a.gguf");
        offer.validate().unwrap();

        assert!(!offer.is_limited());
        assert!(offer.admits(&ShortString::new("anyone").unwrap()));

        let limited: ConsentConfig = toml::from_str(
            "[offer]\nfrom = \"a\"\nto = \"b\"\ndescription = \"x\"\nagents = [\"aegis\", \"sentinel\"]\n",
        )
        .unwrap();
        let limited = limited.offer.unwrap();
        assert!(limited.is_limited());
        assert!(limited.admits(&ShortString::new("aegis").unwrap()));
        assert!(
            !limited.admits(&ShortString::new("Aegis").unwrap()),
            "exact match"
        );
        assert!(!limited.admits(&ShortString::new("tarn-aether").unwrap()));
        let empty: ConsentConfig =
            toml::from_str("[offer]\nfrom = \"a\"\nto = \"b\"\ndescription = \"x\"\nagents = []\n")
                .unwrap();
        assert!(empty.offer.unwrap().validate().is_err());

        assert!(toml::from_str::<ConsentConfig>("[offr]\nfrom = \"a\"").is_err());
        let same: ConsentConfig =
            toml::from_str("[offer]\nfrom = \"a\"\nto = \"a\"\ndescription = \"x\"\n").unwrap();
        assert!(same.offer.unwrap().validate().is_err());
    }
}
#[cfg(test)]
mod golden;
