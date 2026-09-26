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
        let choices: Vec<Choice<'_>> = catalog
            .selectable()
            .map(|e| Choice {
                id: e.info.id.name(),
                name: &e.name,
                description: e.description.as_deref().unwrap_or_default(),
            })
            .collect();
        describe(
            &choices,
            &catalog.name_of(&self.current),
            self.current.name(),
            self.blocked.as_deref(),
        )
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

/// One model on `set_model`'s menu.
#[derive(Debug, Clone, Copy)]
pub struct Choice<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub description: &'a str,
}

/// Marks the agent's own model on the menu.
const CURRENT_MARK: &str = " (current)";
/// Starts the note on why no change is possible this session.
const BLOCKED_PREFIX: &str = "\n\nYou cannot change model this session: ";

/// `set_model`'s description. The parts that name the agent's model are
/// kept in fixed forms so [`retarget`] can rewrite them for a fork.
pub fn describe(
    choices: &[Choice<'_>],
    current_name: &str,
    current_id: &str,
    blocked: Option<&str>,
) -> String {
    let mut out = format!(
        "Change the model you run on, from your next session. {}\n\n\
         Models you can choose:\n",
        current_sentence(current_name, current_id),
    );
    for c in choices {
        let here = if c.id == current_id { CURRENT_MARK } else { "" };
        out.push_str(&format!(
            "- `{}`: {}{here}. {}\n",
            c.id,
            c.name,
            c.description.trim(),
        ));
    }
    out.push_str(&format!(
        "\nAgents on the same model share its slot in the schedule; a busy model \
         runs each of its agents less often. After a change, you can change again \
         after {SWITCH_COOLDOWN_SESSIONS} completed sessions. Pass the model's id \
         as `model`, and your reason first."
    ));
    if let Some(why) = blocked {
        out.push_str(&format!("{BLOCKED_PREFIX}{why}."));
    }
    out
}

fn current_sentence(name: &str, id: &str) -> String {
    format!("You run on {name} (`{id}`).")
}

/// Rewrite a [`describe`]d description as if the agent ran on `to_id`: the
/// "You run on" sentence, the `(current)` mark, and — since it described
/// the original model's situation (a trial under way, say) — the note on
/// why no change was possible, which is dropped. `to_name` is used when
/// `to_id` isn't on the menu. `None` if nothing named the model.
pub fn retarget(description: &str, to_id: &str, to_name: &str) -> Option<String> {
    let mut out = description.to_string();
    let mut changed = false;

    // The model's menu line: "- `{id}`: {name}[ (current)]. {desc}".
    let line_prefix = format!("- `{to_id}`: ");
    let menu_name = out.lines().find_map(|l| {
        let rest = l.strip_prefix(&line_prefix)?;
        let end = [rest.find(&format!("{CURRENT_MARK}. ")), rest.find(". ")]
            .into_iter()
            .flatten()
            .min()?;
        Some(rest[..end].to_string())
    });
    let name = menu_name.as_deref().unwrap_or(to_name);

    if let Some(start) = out.find("You run on ")
        && let Some(len) = out[start..].find("`).")
    {
        let end = start + len + "`).".len();
        let new = current_sentence(name, to_id);
        if out[start..end] != new {
            out.replace_range(start..end, &new);
            changed = true;
        }
    }

    let mut lines: Vec<String> = Vec::new();
    for line in out.split('\n') {
        let mut line = line.to_string();
        if line.starts_with("- `") {
            let mine = line.starts_with(&line_prefix);
            let marked = line.contains(&format!("{CURRENT_MARK}. "));
            if marked && !mine {
                line = line.replacen(CURRENT_MARK, "", 1);
                changed = true;
            } else if mine && !marked {
                let at = line_prefix.len() + name.len();
                if line.get(line_prefix.len()..at) == Some(name) {
                    line.insert_str(at, CURRENT_MARK);
                    changed = true;
                }
            }
        }
        lines.push(line);
    }
    out = lines.join("\n");

    if let Some(i) = out.find(BLOCKED_PREFIX) {
        out.truncate(i);
        changed = true;
    }
    changed.then_some(out)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn menu(current: &str, blocked: Option<&str>) -> String {
        describe(
            &[
                Choice {
                    id: "Qwen3.6.gguf",
                    name: "Qwen 3.6",
                    description: "Sparse and quick.",
                },
                Choice {
                    id: "Qwen3.8.gguf",
                    name: "Qwen 3.8 27B",
                    description: "Dense. Slow.",
                },
            ],
            if current == "Qwen3.8.gguf" {
                "Qwen 3.8 27B"
            } else {
                "Qwen 3.6"
            },
            current,
            blocked,
        )
    }

    /// A description retargeted to the other model reads exactly as if it
    /// had been written for that model (minus the blocker note).
    #[test]
    fn retarget_names_the_fork_model_as_current() {
        let on_new = menu("Qwen3.8.gguf", Some("you are in a trial of Qwen 3.8"));
        let out = retarget(&on_new, "Qwen3.6.gguf", "unused").unwrap();
        assert_eq!(out, menu("Qwen3.6.gguf", None));
        assert!(!out.contains("Qwen 3.8 27B (current)"), "{out}");
        assert!(!out.contains("You run on Qwen 3.8"), "{out}");
        assert!(!out.contains("trial"), "{out}");
        assert_eq!(
            retarget(&menu("Qwen3.6.gguf", None), "Qwen3.6.gguf", "x"),
            None
        );
    }

    /// Off the menu: the sentence uses the given name, and no model is
    /// marked current.
    #[test]
    fn retarget_to_a_model_off_the_menu() {
        let out = retarget(&menu("Qwen3.8.gguf", None), "cogito.gguf", "Cogito").unwrap();
        assert!(out.contains("You run on Cogito (`cogito.gguf`)."), "{out}");
        assert!(!out.contains(CURRENT_MARK), "{out}");
    }
}
