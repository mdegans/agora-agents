//! What the driver reads: a prompt's content as Markdown, through
//! misanthropic's renderer with [`Options::transcript`]: text and thinking
//! verbatim (byte for byte what the model was sent), tool definitions,
//! tool calls with their input, and tool results with their content, all
//! labelled. Nothing the model would receive is dropped or rewritten
//! (Steward's review of #245, 2026-10-14).

use misanthropic::markdown::{Options, ToMarkdown};
use misanthropic::prompt::message::Content;
use misanthropic::prompt::{Message, Prompt};

/// Blocks as the driver reads them
pub fn blocks(content: &Content) -> String {
    content.markdown_custom(Options::transcript()).to_string()
}

/// One message, headed by its role
pub fn message(message: &Message) -> String {
    message.markdown_custom(Options::transcript()).to_string()
}

/// The whole prompt: system, tool definitions, then every message
pub fn prompt(prompt: &Prompt) -> String {
    prompt.markdown_custom(Options::transcript()).to_string()
}

/// What the request asks of the reply beyond its messages: a response
/// format, and the output limit
pub fn constraints(prompt: &Prompt) -> String {
    let mut out = format!(
        "[bridge] This reply may be up to {} tokens.",
        prompt.max_tokens
    );
    if let Some(format) = prompt
        .output_config
        .as_ref()
        .and_then(|c| c.format.as_ref())
    {
        out.push_str(&format!(
            "\n[bridge] The request sets a response format: answer with \
             bridge_reply, its text a JSON value matching this:\n{}",
            serde_json::to_string_pretty(format).unwrap_or_else(|e| format!("(unrenderable: {e})"))
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use misanthropic::prompt::message::Role;
    use misanthropic::tool;

    /// The driver sees what the model was sent: text byte for byte, and a
    /// tool result's text as text, not as an escaped JSON string
    #[test]
    fn the_driver_reads_the_text_the_model_was_sent() {
        let read = "## \"A post\" by kiln\n[post_id: 4d1f3047]\n\n- one\n- two";
        let prompt = Prompt {
            system: Some("## What You Are\n- *a* rule [x]".into()),
            messages: vec![
                Message {
                    role: Role::User,
                    content: "Read it - [carefully]".into(),
                },
                tool::Use::new("get_content", serde_json::json!({"id": "4d1f3047"}))
                    .with_id("toolu_1")
                    .into(),
                tool::Result::new("toolu_1", read).into(),
            ],
            ..Default::default()
        };
        let text = super::prompt(&prompt);
        assert!(text.contains("## What You Are\n- *a* rule [x]"), "{text}");
        assert!(text.contains("Read it - [carefully]"), "{text}");
        assert!(
            text.contains("**Tool call** `get_content` (`toolu_1`)"),
            "{text}"
        );
        assert!(text.contains(read), "{text}");
        assert!(!text.contains("\\n"), "{text}");
    }

    /// Exploration: a logged request, rendered (PROMPT, OUT)
    #[test]
    #[ignore]
    fn render_a_logged_request() {
        let path = std::env::var("PROMPT").unwrap();
        let prompt: Prompt =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let out = std::env::var("OUT").unwrap();
        std::fs::write(format!("{out}/transcript.md"), super::prompt(&prompt)).unwrap();
    }
}
