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
//!
//! **Cache prefix.** The tool definitions are the first thing on the wire,
//! ahead of the system prompt, and every agent on a model shares that
//! prefix. So `set_model` is the same for every agent on a model: always
//! registered when the run has a choice to offer, always last in `tools`
//! (see [`ConsentAgent::seat_tail_tools`](super::agent::ConsentAgent)), and
//! its description names only the model and the menu. Why an agent can't
//! switch right now (a cooldown, a review session) is per-agent, so it
//! lives in the refusal a call gets, never in the description.
//!
//! **Terms** (Steward, 2026-10-08). `term` is `permanent` (a plain switch)
//! or `trial`: [`TRIAL_SESSIONS`] sessions on the new model, then the same
//! keep-or-return review an offered trial gets, asked back on the model the
//! agent came from ([`Ledger::begin_self_trial`]). **During a trial** a
//! `permanent` call decides it early ([`Ledger::decide_early`]): the model
//! the trial came from returns the agent to it, the trial's own model (the
//! one it is on) keeps it, any other ends the trial and switches there.
//! That is the trial's outcome; no review follows. A `trial` call during a
//! trial is refused. The cooldown doesn't apply during a trial (see
//! [`Ledger::switch_blocker`]); it runs from an early return or switch as
//! from any other.
//!
//! [`Ledger::begin_self_trial`]: super::ledger::Ledger::begin_self_trial
//! [`Ledger::decide_early`]: super::ledger::Ledger::decide_early
//! [`Ledger::switch_blocker`]: super::ledger::Ledger::switch_blocker

use std::sync::{Arc, Mutex};

use agora_agentkit::ids::AgentId;
use agora_agentkit::requests::UpdateProfilePayload;
use agora_agentkit::responses::inline_input_schema_for;
use chrono::{DateTime, Utc};
use misanthropic::model::Model;
use misanthropic::prompt::message::Content;
use misanthropic::tool::{self, CustomMethodDef, MethodDef, Tool, Use};
use schemars::JsonSchema;
use serde::Deserialize;

use super::ConsentRuntime;
use super::ledger::{OfferKey, SWITCH_COOLDOWN_SESSIONS, Switch, TRIAL_SESSIONS};
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

/// A `set_model` call that went through: reported to Agora (unless it kept
/// the model the agent is on), awaiting the wrapper's local half.
#[derive(Debug, Clone)]
pub struct SelfSwitch {
    pub at: DateTime<Utc>,
    pub to: Entry,
    pub reason: String,
    pub kind: SelfKind,
}

/// What a [`SelfSwitch`] was.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelfKind {
    /// A plain switch, for good.
    Permanent,
    /// The start of a self-chosen trial of `to`.
    Trial,
    /// The early decision of the trial on `key` (`to` may be the model the
    /// agent is on: a keep, with nothing reported or switched).
    Decided { key: OfferKey },
}

/// The trial running on the agent's model this session, which a
/// `set_model` call decides early.
#[derive(Debug, Clone)]
pub struct ActiveTrial {
    pub key: OfferKey,
    /// The ledger's names for both sides.
    pub from_name: String,
    pub to_name: String,
}

/// Where [`SetModel`] leaves its [`SelfSwitch`] for the wrapper.
pub type Slot = Arc<Mutex<Option<SelfSwitch>>>;

/// The tool's name on the wire.
pub const TOOL_NAME: &str = "set_model";

/// `set_model`'s arguments, and its schema ([`SetModel::schema`]). Reason
/// first: it is written before the choice.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Args {
    /// Why you want to change model.
    reason: String,
    /// The id of the model to run on, from the list.
    model: String,
    /// `permanent` to switch for good, or `trial` to try the model first.
    term: SwitchTerm,
}

/// `set_model`'s `term`. Rendered inline as a plain string enum (no doc
/// comments on the variants, which would make it a `oneOf`); the schema
/// test pins that it carries no `$ref`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum SwitchTerm {
    Permanent,
    Trial,
}

/// The `set_model` tool. See the [module docs](self).
pub struct SetModel {
    rt: Arc<ConsentRuntime>,
    agent_id: AgentId,
    agent: String,
    current: Model,
    /// Why no switch is possible this session, fixed at session start.
    /// Told to the agent only when it calls: kept out of the description,
    /// which every agent on the model shares.
    blocked: Option<String>,
    /// The trial on `current` a call would decide, fixed at session start.
    trial: Option<ActiveTrial>,
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
        trial: Option<ActiveTrial>,
        slot: Slot,
    ) -> Option<Self> {
        let choice = rt.catalog.selectable().any(|e| e.info.id != current);
        choice.then_some(Self {
            rt,
            agent_id,
            agent,
            current,
            blocked,
            trial,
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
        )
    }

    /// [`Args`]' schema, inline: two plain strings and a string enum,
    /// closed
    fn schema() -> serde_json::Value {
        inline_input_schema_for::<Args>()
    }

    /// The model `wanted` names: a selectable one, or — during a trial —
    /// either side of it, which the agent may always decide for while the
    /// run can route it, selectable or not.
    fn resolve(&self, wanted: &str) -> Option<Entry> {
        if let Some(entry) = self.rt.catalog.choose(wanted) {
            return Some(entry.clone());
        }
        let trial = self.trial.as_ref()?;
        let wanted = wanted.trim();
        [
            (&trial.key.from, &trial.from_name),
            (&trial.key.to, &trial.to_name),
        ]
        .into_iter()
        .filter_map(|(id, name)| Some((self.rt.catalog.get(id)?, name)))
        .find(|(entry, name)| {
            wanted == entry.info.id.name()
                || wanted.eq_ignore_ascii_case(&entry.name)
                || wanted.eq_ignore_ascii_case(name)
        })
        .map(|(entry, _)| entry.clone())
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
        if let (Some(trial), SwitchTerm::Trial) = (&self.trial, args.term) {
            return Err(format!(
                "You are already in a trial of {to}, and a new trial can't start during \
                 one. To decide it now, call again with `term` set to `permanent`: choose \
                 {from} to return to it, {to} to keep it, or another model to end the \
                 trial and switch to that one. Otherwise, after session {TRIAL_SESSIONS} \
                 you return to {from} for one session to decide.",
                to = trial.to_name,
                from = trial.from_name,
            ));
        }
        let Some(entry) = self.resolve(&args.model) else {
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
        let kind = match (&self.trial, args.term) {
            (Some(trial), _) => SelfKind::Decided {
                key: trial.key.clone(),
            },
            (None, SwitchTerm::Trial) => SelfKind::Trial,
            (None, SwitchTerm::Permanent) => SelfKind::Permanent,
        };
        let keep = entry.info.id == self.current;
        if keep && self.trial.is_none() {
            return Err(format!("You already run on {}.", entry.name));
        }
        // A keep changes no model: nothing to report.
        if !keep && let Err(e) = report_model(&self.rt, self.agent_id, &entry.info.id).await {
            tracing::error!(
                event_type = "model_self_switch_failed",
                agent = %self.agent,
                agent_id = %self.agent_id,
                from = %self.current,
                to = %entry.info.id,
                term = ?args.term,
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
            term = ?args.term,
            kind = ?kind,
            reason = %args.reason,
            "agent changed its own model, from its next session"
        );
        let reply = match (&kind, &self.trial) {
            (SelfKind::Decided { .. }, Some(trial)) if keep => format!(
                "Your trial is decided: you keep {} for good. Nothing else changes.",
                trial.to_name
            ),
            (SelfKind::Decided { key }, Some(trial)) if entry.info.id == key.from => format!(
                "Your trial is decided: you return to {} from your next session.",
                trial.from_name
            ),
            (SelfKind::Decided { .. }, Some(trial)) => format!(
                "Your trial of {} is over: your model will switch to {} from your next session.",
                trial.to_name, entry.name
            ),
            (SelfKind::Trial, _) => format!(
                "Your trial of {to} starts from your next session: {TRIAL_SESSIONS} sessions \
                 on it, then one session back on {from} to decide whether to keep it.",
                to = entry.name,
                from = self.rt.catalog.name_of(&self.current),
            ),
            _ => format!(
                "Your model will switch to {} from your next session.",
                entry.name
            ),
        };
        *self.slot.lock().expect("slot lock") = Some(SelfSwitch {
            at: Utc::now(),
            to: entry,
            reason: args.reason,
            kind,
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
/// Started the note on why no change was possible this session, which
/// descriptions carried until it moved to the refusal (it made the tool
/// definitions per-agent). Kept so [`retarget`] can drop it from older
/// logged prompts.
pub(crate) const BLOCKED_PREFIX: &str = "\n\nYou cannot change model this session: ";

/// `set_model`'s description. The parts that name the agent's model are
/// kept in fixed forms so [`retarget`] can rewrite them for a fork. Nothing
/// in it is per-agent: it is part of the prefix every agent on the model
/// shares.
pub fn describe(choices: &[Choice<'_>], current_name: &str, current_id: &str) -> String {
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
         as `model`, and your reason first. If you can't change model this session, \
         the call says why and when you next can, and changes nothing.\n\n\
         `term` is `permanent` to switch for good, or `trial` to try the model for \
         {TRIAL_SESSIONS} sessions: after them you return to the model you are on now \
         for one session, and decide there whether to keep the new one. During a \
         trial, a `permanent` call decides it early: the model you came from returns \
         you to it, the trial's model keeps it, and any other model ends the trial \
         and switches to that one. A trial can't start during another."
    ));
    out
}

fn current_sentence(name: &str, id: &str) -> String {
    format!("You run on {name} (`{id}`).")
}

/// Rewrite a [`describe`]d description as if the agent ran on `to_id`: the
/// "You run on" sentence, the `(current)` mark, and — since it described
/// the original model's situation (a trial under way, say) — the note on
/// why no change was possible, which is dropped (descriptions logged
/// before that note moved to the refusal still carry it). `to_name` is used when
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
                trial: self.kind == SelfKind::Trial,
            },
            sessions_after: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The schema is `Args`' own: reason before model, both required,
    /// closed, and free of `$ref`/`$defs`/`pattern`
    #[test]
    fn schema_is_derived_reason_first_closed_and_ref_free() {
        let schema = SetModel::schema();
        let rendered = schema.to_string();
        for banned in ["$ref", "$defs", "definitions", "pattern"] {
            assert!(!rendered.contains(&format!("\"{banned}\"")), "{rendered}");
        }
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["additionalProperties"], false);
        let props: Vec<&String> = schema["properties"].as_object().unwrap().keys().collect();
        assert_eq!(props, ["reason", "model", "term"]);
        assert_eq!(
            schema["required"],
            serde_json::json!(["reason", "model", "term"])
        );
        for p in ["reason", "model", "term"] {
            assert_eq!(schema["properties"][p]["type"], "string", "{p}");
        }
        assert_eq!(
            schema["properties"]["term"]["enum"],
            serde_json::json!(["permanent", "trial"])
        );
        assert_eq!(
            schema["properties"]["model"]["description"],
            "The id of the model to run on, from the list."
        );
        // And the type it decodes into refuses what the schema refuses.
        let err = serde_json::from_value::<Args>(
            serde_json::json!({"reason": "r", "model": "m", "term": "trial", "model_info": "m"}),
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown field"), "{err}");
    }

    fn menu(current: &str, blocked: Option<&str>) -> String {
        let mut out = describe(
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
        );
        // As logged before the note moved to the refusal.
        if let Some(why) = blocked {
            out.push_str(&format!("{BLOCKED_PREFIX}{why}."));
        }
        out
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
        assert!(!out.contains("you are in a trial"), "{out}");
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
