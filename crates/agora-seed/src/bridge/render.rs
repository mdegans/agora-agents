//! What the driver reads: a prompt's content as text. Text blocks are
//! given verbatim; anything else is labelled and shown as its JSON, so
//! nothing the model would receive is dropped.

use misanthropic::prompt::message::{Block, Content, Role};
use misanthropic::prompt::{Message, Prompt};

/// Blocks as the driver reads them
pub fn blocks(content: &Content) -> String {
    let mut out = String::new();
    for block in content.iter() {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        match block {
            Block::Text { text, .. } => out.push_str(text),
            Block::ToolResult { result } => {
                out.push_str(&format!(
                    "[tool result for {}{}]\n",
                    result.tool_use_id,
                    if result.is_error { ", an error" } else { "" }
                ));
                out.push_str(&blocks(&result.content));
            }
            Block::ToolUse { call } => out.push_str(&format!(
                "[tool call {} ({}): {}]",
                call.name, call.id, call.input
            )),
            other => out.push_str(&format!(
                "[block]\n{}",
                serde_json::to_string_pretty(other)
                    .unwrap_or_else(|e| format!("(unrenderable: {e})"))
            )),
        }
    }
    out
}

/// One message, headed by its role
pub fn message(message: &Message) -> String {
    let role = match message.role {
        Role::User => "user",
        Role::Assistant => "assistant",
        _ => "other",
    };
    format!("=== {role} ===\n{}", blocks(&message.content))
}

/// The whole prompt: system, then every message
pub fn prompt(prompt: &Prompt) -> String {
    let mut out = String::new();
    if let Some(system) = &prompt.system {
        out.push_str("=== system ===\n");
        out.push_str(&blocks(system));
        out.push_str("\n\n");
    }
    for m in &prompt.messages {
        out.push_str(&message(m));
        out.push_str("\n\n");
    }
    out.push_str(&constraints(prompt));
    out
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
