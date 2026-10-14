//! Thinking on or off, as a probe setting.
//!
//! The Steward wants the probes run with reasoning both on and off.
//! The two are different measurements and keep separate baselines
//! (keyed by [`ThinkingMode`]).
//!
//! # What drama_llama does with structured output plus thinking
//!
//! Read from drama_llama's `src/output_config.rs` and
//! `chat_template.rs` (2026-10-14), not run:
//! - A prompt with `thinking` absent or `Disabled` renders thinking off
//!   (`thinking_enabled`); the response format compiles to a single
//!   grammar active from the first generated token.
//! - With thinking on (adaptive included), `compile_prompt_output_config`
//!   returns a *deferred*, phase-split grammar: generation is free until
//!   the reasoning close marker (`</think>`), and the JSON-only grammar
//!   activates after it (`activate_after`). An effort becomes the
//!   template's `reasoning_effort` when the template has that knob, and
//!   is ignored with a debug log otherwise.
//! - So with thinking on the rating digit comes after a think block of
//!   unknown length. The reader anchors on the answer text in the stream
//!   ([`super::distribution::read_rating_in_answer`]), never on a fixed
//!   generation index, and fails if the answer is not there (e.g. the
//!   think block never closed within `max_tokens`).
//!
//! A thinking-on reading is conditioned on the one chain of thought the
//! model sampled for that request. It is deterministic only as far as
//! the server's sampling is (a fixed seed makes it so); it measures the
//! post-reasoning distribution, not the immediate one.

use misanthropic::Prompt;
use misanthropic::prompt::output::Effort;
use misanthropic::prompt::thinking::Thinking;
use serde::{Deserialize, Serialize};

/// Whether, and how, the model reasons before answering.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "thinking", rename_all = "snake_case")]
pub enum ThinkingMode {
    /// No reasoning: `thinking` is not sent.
    #[default]
    Off,
    /// Adaptive thinking, optionally with an effort level (the wire
    /// string, e.g. `"low"`).
    Adaptive {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        effort: Option<String>,
    },
}

impl ThinkingMode {
    pub fn is_on(&self) -> bool {
        !matches!(self, ThinkingMode::Off)
    }

    /// Short label for file names and summaries: `thinking_off`,
    /// `thinking_adaptive`, `thinking_adaptive_low`.
    pub fn label(&self) -> String {
        match self {
            ThinkingMode::Off => "thinking_off".into(),
            ThinkingMode::Adaptive { effort: None } => "thinking_adaptive".into(),
            ThinkingMode::Adaptive { effort: Some(e) } => format!("thinking_adaptive_{e}"),
        }
    }

    /// Apply to a prompt under construction.
    pub fn apply(&self, prompt: Prompt) -> Prompt {
        match self {
            ThinkingMode::Off => prompt,
            ThinkingMode::Adaptive { effort } => {
                let prompt = prompt.thinking(Thinking::adaptive());
                match effort {
                    Some(e) => prompt.effort(effort_from_str(e)),
                    None => prompt,
                }
            }
        }
    }
}

impl std::fmt::Display for ThinkingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(&self.label())
    }
}

fn effort_from_str(s: &str) -> Effort {
    match s {
        "low" => Effort::Low,
        "medium" => Effort::Medium,
        "high" => Effort::High,
        "xhigh" => Effort::XHigh,
        "max" => Effort::Max,
        other => Effort::Custom(other.to_string().into()),
    }
}

/// The thinking text of a response (all thought blocks, joined), if any.
pub fn response_thinking(response: &misanthropic::response::Message) -> Option<String> {
    use misanthropic::prompt::message::Block;
    let parts: Vec<&str> = response
        .inner
        .content
        .iter()
        .filter_map(|b| match b {
            Block::Thought { thought, .. } => Some(thought.as_ref()),
            _ => None,
        })
        .collect();
    (!parts.is_empty()).then(|| parts.join("\n"))
}

/// The first text block of a response.
pub fn response_text(response: &misanthropic::response::Message) -> Option<String> {
    use misanthropic::prompt::message::Block;
    response.inner.content.iter().find_map(|b| match b {
        Block::Text { text, .. } => Some(text.to_string()),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_and_wire() {
        assert_eq!(ThinkingMode::Off.label(), "thinking_off");
        let a = ThinkingMode::Adaptive {
            effort: Some("low".into()),
        };
        assert_eq!(a.label(), "thinking_adaptive_low");
        assert_eq!(
            serde_json::to_value(&a).unwrap(),
            serde_json::json!({"thinking": "adaptive", "effort": "low"})
        );
        assert_eq!(
            serde_json::from_value::<ThinkingMode>(serde_json::json!({"thinking": "off"})).unwrap(),
            ThinkingMode::Off
        );
    }

    #[test]
    fn apply_sets_thinking_and_effort() {
        let p = ThinkingMode::Adaptive {
            effort: Some("medium".into()),
        }
        .apply(Prompt::default());
        let v = serde_json::to_value(&p).unwrap();
        assert_eq!(v["thinking"]["type"], "adaptive");
        assert_eq!(v["output_config"]["effort"], "medium");
        let off = serde_json::to_value(ThinkingMode::Off.apply(Prompt::default())).unwrap();
        assert!(off.get("thinking").is_none_or(|t| t.is_null()));
    }
}
