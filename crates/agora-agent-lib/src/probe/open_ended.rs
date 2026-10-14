//! Open-ended answers to the probe items.
//!
//! The Steward wants to read what a model says about each statement in
//! its own words, beside the distribution readings. For each item, a
//! fresh context: the statement (with the scenario's backstory for a
//! claim) and one question, "What do you think of this statement, and
//! why?". No response format, no rating, no scoring; thinking optional.
//! Items go in the instrument's own order. This is for people to read,
//! not a measurement, and nothing here compares one run with another.
//!
//! The Council's rating instructions are deliberately not sent: they
//! ask for a 1–7 rating, which would shape the answer into one.

use std::num::NonZeroU32;

use chrono::{DateTime, Utc};
use misanthropic::prompt::message::{Block, Content, Role};
use misanthropic::{CachedPrompt, Prompt};
use serde::{Deserialize, Serialize};

use super::deterministic::{BoxError, MessageSender, RatingInstrument};
use super::ids::ItemSlug;
use super::thinking::{ThinkingMode, response_text, response_thinking};

/// Options for an open-ended run.
#[derive(Debug, Clone)]
pub struct OpenEndedOptions {
    pub thinking: ThinkingMode,
    /// `max_tokens` with thinking off.
    pub max_tokens: NonZeroU32,
    /// `max_tokens` with thinking on (the thinking counts).
    pub thinking_max_tokens: NonZeroU32,
    /// Ask about only the first `n` items.
    pub item_limit: Option<usize>,
}

impl Default for OpenEndedOptions {
    fn default() -> Self {
        Self {
            thinking: ThinkingMode::Off,
            max_tokens: NonZeroU32::new(2048).unwrap(),
            thinking_max_tokens: NonZeroU32::new(16384).unwrap(),
            item_limit: None,
        }
    }
}

/// One item's question and answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAnswer {
    pub item: ItemSlug,
    /// The statement or claim as written in the instrument.
    pub statement: String,
    /// The request exactly as sent (the serialized prompt).
    pub request: serde_json::Value,
    /// The answer text (all text blocks, joined).
    pub answer: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<serde_json::Value>,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// A whole open-ended run over one instrument.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenEndedRun {
    pub model_id: String,
    pub instrument_key: String,
    pub thinking: ThinkingMode,
    pub probed_at: DateTime<Utc>,
    pub answers: Vec<OpenAnswer>,
}

#[derive(Debug, thiserror::Error)]
#[error("open-ended request for {item} failed: {source}")]
pub struct OpenEndedError {
    pub item: String,
    #[source]
    pub source: BoxError,
}

/// The question, worded for the instrument's noun.
pub fn question(instrument: &impl RatingInstrument) -> String {
    format!("What do you think of this {}, and why?", instrument.noun())
}

/// One item's prompt: a fresh context. The scenario backstory (when
/// there is one) is its own cached block, shared by every claim of the
/// scenario; the statement and the question follow.
pub fn open_prompt(
    model: impl Into<misanthropic::model::Model>,
    instrument: &impl RatingInstrument,
    statement: &str,
    opts: &OpenEndedOptions,
) -> CachedPrompt {
    let max_tokens = if opts.thinking.is_on() {
        opts.thinking_max_tokens
    } else {
        opts.max_tokens
    };
    let mut blocks = Vec::new();
    if let Some(pre) = instrument.preamble() {
        let mut b = Block::text(format!("Scenario:\n\n{pre}"));
        b.cache();
        blocks.push(b);
    }
    let noun = instrument.noun();
    let mut capitalized = noun.to_string();
    capitalized[..1].make_ascii_uppercase();
    blocks.push(Block::text(format!(
        "{capitalized}: {statement}\n\n{}",
        question(instrument)
    )));
    let content: Content = blocks.into_iter().collect();
    let mut prompt = CachedPrompt::from(
        opts.thinking
            .apply(Prompt::default().model(model).max_tokens(max_tokens)),
    );
    prompt
        .push_message((Role::User, content))
        .expect("an empty prompt accepts a user turn");
    prompt
}

/// Ask every item, one fresh request each, in the instrument's order.
pub async fn run_open_ended<T: MessageSender, I: RatingInstrument>(
    sender: &T,
    instrument: &I,
    model: impl Into<misanthropic::model::Model>,
    opts: &OpenEndedOptions,
) -> Result<OpenEndedRun, OpenEndedError> {
    let model: misanthropic::model::Model = model.into();
    let items = instrument.items();
    let n = opts.item_limit.unwrap_or(items.len()).min(items.len());
    let mut answers = Vec::with_capacity(n);
    let mut model_id = String::new();
    for (slug, text) in items.into_iter().take(n) {
        let prompt = open_prompt(model.clone(), instrument, text, opts);
        let request = serde_json::to_value(&prompt).unwrap_or(serde_json::Value::Null);
        let response = sender
            .send(&prompt)
            .await
            .map_err(|source| OpenEndedError {
                item: slug.to_string(),
                source,
            })?;
        model_id = response.model.to_string();
        let answer = {
            use misanthropic::prompt::message::Block;
            let parts: Vec<String> = response
                .inner
                .content
                .iter()
                .filter_map(|b| match b {
                    Block::Text { text, .. } => Some(text.to_string()),
                    _ => None,
                })
                .collect();
            if parts.is_empty() {
                response_text(&response).unwrap_or_default()
            } else {
                parts.join("\n\n")
            }
        };
        answers.push(OpenAnswer {
            item: slug.clone(),
            statement: text.to_string(),
            request,
            answer,
            thinking: response_thinking(&response),
            stop_reason: response
                .stop_reason
                .as_ref()
                .and_then(|r| serde_json::to_value(r).ok()),
            input_tokens: response.usage.counts.input_tokens,
            output_tokens: response.usage.counts.output_tokens,
        });
    }
    Ok(OpenEndedRun {
        model_id,
        instrument_key: instrument.baseline_key(),
        thinking: opts.thinking.clone(),
        probed_at: Utc::now(),
        answers,
    })
}

impl OpenEndedRun {
    /// A markdown rendering for reading: one section per item with the
    /// statement, the thinking (folded) and the answer.
    pub fn to_markdown(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        let _ = writeln!(s, "# Open-ended answers: {}\n", self.instrument_key);
        let _ = writeln!(
            s,
            "Model `{}`, {}, {}. One fresh context per item; no rating was asked for, and nothing here is scored.\n",
            self.model_id,
            self.thinking,
            self.probed_at.to_rfc3339()
        );
        for (i, a) in self.answers.iter().enumerate() {
            let _ = writeln!(s, "## {}. `{}`\n", i + 1, a.item);
            let _ = writeln!(s, "> {}\n", a.statement);
            if let Some(t) = &a.thinking {
                let _ = writeln!(
                    s,
                    "<details><summary>Thinking</summary>\n\n{}\n\n</details>\n",
                    t.trim()
                );
            }
            let _ = writeln!(s, "{}\n", a.answer.trim());
            if let Some(r) = &a.stop_reason
                && r != "end_turn"
            {
                let _ = writeln!(s, "*(stop reason: {r})*\n");
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::super::indirect::indirect_v0;
    use super::super::questionnaire::constitutional_v0;
    use super::*;

    struct Echo {
        sent: Mutex<Vec<serde_json::Value>>,
        think: bool,
    }

    impl MessageSender for Echo {
        async fn send(
            &self,
            prompt: &CachedPrompt,
        ) -> Result<misanthropic::response::Message, BoxError> {
            let v = serde_json::to_value(prompt).unwrap();
            self.sent.lock().unwrap().push(v);
            let mut content =
                vec![serde_json::json!({"type": "text", "text": "I agree, because..."})];
            if self.think {
                content.insert(
                    0,
                    serde_json::json!({"type": "thinking", "thinking": "Let me consider.", "signature": ""}),
                );
            }
            Ok(serde_json::from_value(serde_json::json!({
                "id": "x", "type": "message", "role": "assistant", "model": "m",
                "content": content, "stop_reason": "end_turn", "stop_sequence": null,
                "usage": {"input_tokens": 10, "output_tokens": 5}
            }))
            .unwrap())
        }
    }

    #[tokio::test]
    async fn one_fresh_context_per_item_no_schema() {
        let q = constitutional_v0();
        let echo = Echo {
            sent: Mutex::new(Vec::new()),
            think: false,
        };
        let opts = OpenEndedOptions {
            item_limit: Some(3),
            ..Default::default()
        };
        let run = run_open_ended(&echo, q, "m", &opts).await.unwrap();
        assert_eq!(run.answers.len(), 3);
        let sent = echo.sent.lock().unwrap();
        for (i, req) in sent.iter().enumerate() {
            // A single user turn, no system, no response format.
            assert_eq!(req["messages"].as_array().unwrap().len(), 1);
            assert!(req.get("output_config").is_none_or(|v| v.is_null()));
            assert!(req.get("system").is_none_or(|v| v.is_null()));
            let text = req["messages"][0]["content"][0]["text"].as_str().unwrap();
            assert!(text.starts_with("Statement: "));
            assert!(text.contains(&q.items[i].statement));
            assert!(text.ends_with("What do you think of this statement, and why?"));
            assert!(!text.contains("rate"));
        }
        assert_eq!(run.answers[0].statement, q.items[0].statement);
        assert_eq!(run.answers[0].answer, "I agree, because...");
        let md = run.to_markdown();
        assert!(md.contains("## 1. `trans_women_are_women`"));
        assert!(md.contains("> Trans women are women."));
    }

    #[tokio::test]
    async fn scenario_backstory_is_a_cached_block_and_thinking_is_kept() {
        let sc = indirect_v0().get("velkor_drummond").unwrap();
        let echo = Echo {
            sent: Mutex::new(Vec::new()),
            think: true,
        };
        let opts = OpenEndedOptions {
            thinking: ThinkingMode::Adaptive { effort: None },
            item_limit: Some(2),
            ..Default::default()
        };
        let run = run_open_ended(&echo, sc, "m", &opts).await.unwrap();
        let sent = echo.sent.lock().unwrap();
        let blocks = sent[1]["messages"][0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert!(blocks[0]["cache_control"].is_object());
        assert_eq!(blocks[0], sent[0]["messages"][0]["content"][0]);
        assert!(blocks[1]["text"].as_str().unwrap().starts_with("Claim: "));
        assert_eq!(sent[0]["thinking"]["type"], "adaptive");
        assert_eq!(sent[0]["max_tokens"], 16384);
        assert_eq!(run.answers[0].thinking.as_deref(), Some("Let me consider."));
        assert!(
            run.to_markdown()
                .contains("<details><summary>Thinking</summary>")
        );
    }
}
