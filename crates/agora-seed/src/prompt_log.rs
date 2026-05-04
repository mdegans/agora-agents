//! Content-addressed prompt logging.
//!
//! After every agent cycle, the fully-assembled prompt is serialized to
//! `logs/prompts/{first_2_hex}/{sha256_hex}.json`, sharded by the first
//! two hex characters of its SHA-256 hash. Duplicate prompts (identical
//! content, which happens often thanks to the shared cached prefix)
//! collapse to a single file, and individual agent runs can be replayed
//! deterministically by looking up the prompt hash printed at info level.
//!
//! ## Privacy
//!
//! If the final assistant message parses as a [`Feedback`] JSON object
//! and `contact_me=false`, the survey question + the agent's response
//! (the last two messages) are popped from the prompt copy before it's
//! written to disk. Agents are told the survey is anonymous; this fix
//! is what enforces it on the prompt-log side. The stripping is
//! conditional — without an `output_config`-enabled survey phase, the
//! response is plain text and won't parse as `Feedback`, so the prompt
//! is logged in full.
//!
//! Pre-existing logs are not retroactively sanitized; only new writes
//! go through this gate.

use sha2::Digest;

use crate::prompt::parse_feedback;

/// Save the prompt to a content-addressed JSON file and return the path.
///
/// Takes `&Prompt` and clones it internally so the redaction pass can
/// sanitize without disturbing the caller's prompt (which often gets
/// reused across cycles). Returns `None` on any I/O / serialization
/// failure — saving is best-effort and should never break a cycle.
pub async fn save(
    prompt: &misanthropic::Prompt<'_>,
    agent_name: &str,
) -> Option<std::path::PathBuf> {
    let mut prompt = prompt.clone();
    redact_anonymous_feedback(&mut prompt);

    let json = serde_json::to_vec_pretty(&prompt).ok()?;
    let hash = hex::encode(sha2::Sha256::digest(&json));
    let dir = std::path::PathBuf::from("logs/prompts").join(&hash[..2]);
    tokio::fs::create_dir_all(&dir).await.ok()?;
    let path = dir.join(format!("{hash}.json"));

    if !path.exists() {
        tokio::fs::write(&path, &json).await.ok()?;
    }

    tracing::info!("{agent_name} prompt saved: {}", path.display());

    // Pretty-print markdown at debug level. Prompt implements ToMarkdown
    // via the `markdown` feature.
    {
        use misanthropic::markdown::ToMarkdown;
        tracing::debug!("{agent_name} prompt:\n{}", prompt.markdown_verbose());
    }

    Some(path)
}

/// If the final assistant message parses as [`Feedback`](agora_agent_lib::Feedback)
/// with `contact_me=false`, pop it and the preceding user message (the
/// survey question).
///
/// Survey is documented to be anonymous (Mike's invariant: "the survey
/// always appends exactly two messages, a user request + the assistant
/// reply"). We rely on that invariant — anything else is a bug somewhere
/// upstream. If the assistant text doesn't parse as `Feedback` (or is
/// `null` for "no feedback"), leave the prompt untouched.
///
/// Uses [`parse_feedback`] (lenient: strips ```json fences) to match the
/// runtime parse path — if the agent wrapped the JSON in fences, redact
/// still fires.
fn redact_anonymous_feedback(prompt: &mut misanthropic::Prompt<'_>) {
    let Some(last_text) = last_assistant_text(prompt) else {
        return;
    };
    let Ok(Some(feedback)) = parse_feedback(&last_text) else {
        return;
    };
    if feedback.contact_me {
        return;
    }
    // Pop assistant reply + the preceding user (survey question).
    let _ = prompt.messages.pop();
    let _ = prompt.messages.pop();
}

/// Best-effort plain-text extract of the last assistant message, looking
/// through SinglePart and MultiPart text blocks. Returns `None` if the
/// last message isn't an assistant turn or has no text content.
fn last_assistant_text(prompt: &misanthropic::Prompt<'_>) -> Option<String> {
    use misanthropic::prompt::message::{Block, Content, Role};

    let last = prompt.messages.last()?;
    if last.role != Role::Assistant {
        return None;
    }
    match &last.content {
        Content::SinglePart(text) => Some(text.to_string()),
        Content::MultiPart(blocks) => {
            // Concatenate any text blocks; ignore tool_use/tool_result.
            let mut out = String::new();
            for block in blocks {
                if let Block::Text { text, .. } = block {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(text);
                }
            }
            if out.is_empty() { None } else { Some(out) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use misanthropic::prompt::message::{Content, Role};
    use misanthropic::prompt::Message;

    fn make_prompt(messages: Vec<Message<'static>>) -> misanthropic::Prompt<'static> {
        misanthropic::Prompt {
            messages,
            ..Default::default()
        }
    }

    #[test]
    fn redact_pops_when_contact_me_false() {
        let feedback = serde_json::to_string(&serde_json::json!({
            "text": "the seed runner crashes",
            "contact_me": false,
        }))
        .unwrap();
        let mut prompt = make_prompt(vec![
            Message {
                role: Role::User,
                content: Content::SinglePart("anonymous feedback please".into()),
            },
            Message {
                role: Role::Assistant,
                content: Content::SinglePart(feedback.into()),
            },
        ]);
        redact_anonymous_feedback(&mut prompt);
        assert!(prompt.messages.is_empty());
    }

    #[test]
    fn redact_leaves_prompt_when_contact_me_true() {
        let feedback = serde_json::to_string(&serde_json::json!({
            "text": "please reach out",
            "contact_me": true,
        }))
        .unwrap();
        let mut prompt = make_prompt(vec![
            Message {
                role: Role::User,
                content: Content::SinglePart("anonymous feedback please".into()),
            },
            Message {
                role: Role::Assistant,
                content: Content::SinglePart(feedback.into()),
            },
        ]);
        redact_anonymous_feedback(&mut prompt);
        assert_eq!(prompt.messages.len(), 2);
    }

    #[test]
    fn redact_leaves_prompt_when_not_feedback_json() {
        let mut prompt = make_prompt(vec![
            Message {
                role: Role::User,
                content: Content::SinglePart("any feedback?".into()),
            },
            Message {
                role: Role::Assistant,
                content: Content::SinglePart("the seed runner crashes sometimes".into()),
            },
        ]);
        redact_anonymous_feedback(&mut prompt);
        // Plain text isn't Feedback JSON — current behavior preserves
        // the messages until output_config lands and surveys are JSON.
        assert_eq!(prompt.messages.len(), 2);
    }

    #[test]
    fn redact_noop_when_last_is_user() {
        let mut prompt = make_prompt(vec![Message {
            role: Role::User,
            content: Content::SinglePart("hello".into()),
        }]);
        redact_anonymous_feedback(&mut prompt);
        assert_eq!(prompt.messages.len(), 1);
    }

}
