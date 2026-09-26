//! Applying a model change, and `set_model`: the tool that lets an agent ask
//! for one.
//!
//! A change has two halves. [`report_model`] is the remote half: a signed
//! `update_profile(model_info = to)` as the agent, so the server — which
//! `sync-models` treats as the source of truth — says the same as the
//! runner. The local half is the wrapper's
//! [`ConsentAgent::apply_change`](super::agent::ConsentAgent::apply_change):
//! the ledger's [`Switch`] and the agent's `state.model` from its next
//! session. The remote half goes first, and a failure there changes nothing
//! locally.
//!
//! [`SetModel`] runs inside the inner agent's tool dispatch, where the
//! wrapper can't be reached, so it does the remote half itself and leaves a
//! [`SelfSwitch`] in a slot the wrapper drains.

use std::sync::{Arc, Mutex};

use agora_agentkit::ids::AgentId;
use agora_agentkit::requests::UpdateProfilePayload;
use chrono::{DateTime, Utc};
use misanthropic::model::Model;
use misanthropic::prompt::message::Content;
use misanthropic::tool::{self, CustomMethodDef, MethodDef, Tool, Use};
use serde::Deserialize;

use super::ConsentRuntime;
use super::ledger::{SWITCH_COOLDOWN_SESSIONS, Switch};
use crate::models::Entry;

/// Why a change could not be reported to Agora. Nothing was changed.
#[derive(Debug)]
pub enum SwitchError {
    /// The model isn't one this run can route onto.
    NotRoutable,
    /// The agent's consent ledger couldn't be read, so the change couldn't
    /// be recorded.
    NoLedger,
    NoKey,
    Server(agora_agentkit::client::Error),
}

impl std::fmt::Display for SwitchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotRoutable => f.write_str("the model is not routable in this run"),
            Self::NoLedger => f.write_str("the consent ledger is unavailable"),
            Self::NoKey => f.write_str("no signing key for this agent"),
            Self::Server(e) => write!(f, "Agora refused the profile update: {e}"),
        }
    }
}

impl std::error::Error for SwitchError {}

/// Report `to` as the agent's model on Agora, signed with its own key.
pub async fn report_model(
    rt: &ConsentRuntime,
    agent_id: AgentId,
    to: &Model,
) -> Result<(), SwitchError> {
    let key = rt.keys.signing_key(agent_id).ok_or(SwitchError::NoKey)?;
    let payload = UpdateProfilePayload {
        model_info: Some(to.name().to_string()),
        ..Default::default()
    };
    rt.client
        .update_profile(agent_id, &payload, &key)
        .await
        .map_err(SwitchError::Server)?;
    Ok(())
}

/// A `set_model` call that went through: reported to Agora, awaiting the
/// wrapper's local half.
#[derive(Debug, Clone)]
pub struct SelfSwitch {
    pub at: DateTime<Utc>,
    pub to: Entry,
    pub reason: String,
}

/// Where [`SetModel`] leaves its [`SelfSwitch`] for the wrapper.
pub type Slot = Arc<Mutex<Option<SelfSwitch>>>;

/// The tool's name on the wire.
pub const TOOL_NAME: &str = "set_model";

/// `set_model`'s arguments. Reason first: it is written before the choice.
#[derive(Debug, Deserialize)]
struct Args {
    reason: String,
    model: String,
}

/// The `set_model` tool. See the [module docs](self).
pub struct SetModel {
    rt: Arc<ConsentRuntime>,
    agent_id: AgentId,
    agent: String,
    current: Model,
    /// Why no switch is possible this session, fixed at session start.
    blocked: Option<String>,
    slot: Slot,
}

impl SetModel {
    /// `None` when there is nothing to choose: no selectable model besides
    /// the one the agent is on.
    pub fn new(
        rt: Arc<ConsentRuntime>,
        agent_id: AgentId,
        agent: String,
        current: Model,
        blocked: Option<String>,
        slot: Slot,
    ) -> Option<Self> {
        let choice = rt.catalog.selectable().any(|e| e.info.id != current);
        choice.then_some(Self {
            rt,
            agent_id,
            agent,
            current,
            blocked,
            slot,
        })
    }

    fn description(&self) -> String {
        let catalog = &self.rt.catalog;
        let mut out = format!(
            "Change the model you run on, from your next session. You run on {} (`{}`).\n\n\
             Models you can choose:\n",
            catalog.name_of(&self.current),
            self.current.name(),
        );
        for e in catalog.selectable() {
            let here = if e.info.id == self.current {
                " (current)"
            } else {
                ""
            };
            out.push_str(&format!(
                "- `{}`: {}{here}. {}\n",
                e.info.id.name(),
                e.name,
                e.description.as_deref().unwrap_or_default().trim(),
            ));
        }
        out.push_str(&format!(
            "\nAgents on the same model share its slot in the schedule; a busy model \
             runs each of its agents less often. After a change, you can change again \
             after {SWITCH_COOLDOWN_SESSIONS} completed sessions. Pass the model's id \
             as `model`, and your reason first."
        ));
        if let Some(why) = &self.blocked {
            out.push_str(&format!("\n\nYou cannot change model this session: {why}."));
        }
        out
    }

    fn schema() -> serde_json::Value {
        // Hand-written: two plain strings, no `$ref`, no enum, no pattern.
        serde_json::json!({
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why you want to change model."
                },
                "model": {
                    "type": "string",
                    "description": "The id of the model to run on, from the list."
                }
            },
            "required": ["reason", "model"],
            "additionalProperties": false
        })
    }

    /// The call, minus the wire envelope. `Err` is the message the agent
    /// sees; nothing has changed.
    async fn switch(&mut self, input: serde_json::Value) -> Result<String, String> {
        let args: Args = serde_json::from_value(input)
            .map_err(|e| format!("Could not read the arguments: {e}."))?;
        if self.slot.lock().expect("slot lock").is_some() {
            return Err("You already changed your model this session.".into());
        }
        if let Some(why) = &self.blocked {
            return Err(format!("You cannot change model this session: {why}."));
        }
        let Some(entry) = self.rt.catalog.choose(&args.model).cloned() else {
            let ids: Vec<String> = self
                .rt
                .catalog
                .selectable()
                .map(|e| format!("`{}`", e.info.id.name()))
                .collect();
            return Err(format!(
                "`{}` is not a model you can choose. Choose one of: {}.",
                args.model.trim(),
                ids.join(", ")
            ));
        };
        if entry.info.id == self.current {
            return Err(format!("You already run on {}.", entry.name));
        }
        if let Err(e) = report_model(&self.rt, self.agent_id, &entry.info.id).await {
            tracing::error!(
                event_type = "model_self_switch_failed",
                agent = %self.agent,
                agent_id = %self.agent_id,
                from = %self.current,
                to = %entry.info.id,
                error = %e,
                "set_model: profile update failed; nothing changed"
            );
            return Err(
                "Your model could not be changed right now, and nothing has changed. \
                 You can try again in a later session."
                    .into(),
            );
        }
        tracing::info!(
            event_type = "model_self_switch",
            agent = %self.agent,
            agent_id = %self.agent_id,
            from = %self.current,
            to = %entry.info.id,
            reason = %args.reason,
            "agent changed its own model, from its next session"
        );
        let reply = format!(
            "Your model will switch to {} from your next session.",
            entry.name
        );
        *self.slot.lock().expect("slot lock") = Some(SelfSwitch {
            at: Utc::now(),
            to: entry,
            reason: args.reason,
        });
        Ok(reply)
    }
}

#[async_trait::async_trait]
impl Tool for SetModel {
    fn name(&self) -> &str {
        TOOL_NAME
    }

    fn definitions(&self) -> Vec<MethodDef> {
        let mut def = CustomMethodDef::simple(TOOL_NAME, self.description());
        def.schema = Self::schema();
        vec![MethodDef::Custom(def)]
    }

    async fn call(&mut self, call: Use) -> tool::Result {
        match self.switch(call.input).await {
            Ok(text) => tool::Result::new(call.id, Content::from(text)),
            Err(text) => tool::Result::new(call.id, Content::from(text)).error(),
        }
    }
}

impl SelfSwitch {
    /// The ledger record for this switch, away from `from`
    pub fn record(&self, from: Model) -> Switch {
        Switch {
            at: self.at,
            from,
            to: self.to.info.id.clone(),
            cause: super::ledger::SwitchCause::SelfSwitch {
                reason: self.reason.clone(),
            },
            sessions_after: 0,
        }
    }
}
