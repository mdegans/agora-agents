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
//!   tail (reflect → mutate/evolve → survey) it leaves untouched. It asks
//!   the offer after that tail; runs a trial with a countdown; returns the
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
//! - [`queue`] — the audit trail of changes agents asked for; any the runner
//!   could not apply are left for the Steward's `set_model` + `sync-models`.
//! - [`role`] — a sibling offer (`[role_consent]`): closing the gap between
//!   a generator-assigned role and the agent's tools, put by the same
//!   wrapper at the same point.
//! - [`cadence`] — another sibling (`[cadence_consent]`): a daily agent is
//!   asked whether it would rather have twice the rounds every other day.
//! - [`offers`] — `answer_offer`, the strict tool all three are answered
//!   with: every offer due at a session's close is seated in one question
//!   turn, each under a heading naming its key.
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

pub mod agent;
pub mod cadence;
pub mod comparison;
pub mod forks;
pub mod ledger;
pub mod offers;
pub mod prompt;
pub mod queue;
pub mod role;
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
    /// Exact model id, as `set_model --to` will write it.
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
    /// `[cache]` overrides applied to every agent's admitted quirks.
    pub cache: CacheOverrides,
}

/// `[cache]` in the run config: per-quirk overrides of what the endpoint
/// variant implies. Each `None` keeps the endpoint's own value.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheOverrides {
    /// Roll the cache markers onto the trailing assistant turns (`true`,
    /// blallama's default) or the tail user turn (`false`, Anthropic's).
    pub breakpoint_after_assistant: Option<bool>,
}

impl CacheOverrides {
    /// `quirks` with these overrides applied.
    pub fn apply(
        &self,
        quirks: &agora_agentkit::reactor::inference::Quirks,
    ) -> agora_agentkit::reactor::inference::Quirks {
        let mut q = *quirks;
        if let Some(b) = self.breakpoint_after_assistant {
            q.breakpoint_after_assistant = b;
        }
        q
    }
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
            queue_path: queue::queue_path(data_dir),
            client,
            keys,
            catalog,
            max_tokens,
            alerts: crate::alerts::Alerter::off(),
            role: None,
            cadence: None,
            cache: CacheOverrides::default(),
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

    pub fn with_cache_overrides(mut self, cache: CacheOverrides) -> Self {
        self.cache = cache;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `[cache]` overrides one quirk and leaves the rest as the endpoint
    /// set them; empty overrides change nothing.
    #[test]
    fn cache_overrides_apply_only_what_they_name() {
        use agora_agentkit::reactor::anthropic::EndpointVariant;
        use agora_agentkit::reactor::inference::Quirks;
        let blallama = Quirks::from(EndpointVariant::Blallama);
        assert!(blallama.breakpoint_after_assistant);
        assert_eq!(CacheOverrides::default().apply(&blallama), blallama);
        let off: CacheOverrides = toml::from_str("breakpoint_after_assistant = false").unwrap();
        let q = off.apply(&blallama);
        assert!(!q.breakpoint_after_assistant);
        assert_eq!(
            q.output_config_cache_safe,
            blallama.output_config_cache_safe
        );
        assert!(
            toml::from_str::<CacheOverrides>("typo = 1").is_err(),
            "closed"
        );
    }

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
