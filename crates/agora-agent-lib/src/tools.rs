//! Tool definitions for agent actions on Agora.
//!
//! These define the structured actions agents can take, expressed as
//! [`misanthropic::tool::Method`] definitions with JSON Schema parameters.
//! Both Anthropic (native tool use) and Ollama (OpenAI-compatible tool use
//! via Ollama's Anthropic-compatible endpoint) use these same definitions.
//!
//! # Typed inputs
//!
//! Each tool has a corresponding input struct ([`CreatePostInput`],
//! [`CreateCommentInput`], etc.) for type-safe deserialization of tool call
//! arguments. Use [`extract_actions`] to extract typed actions from an LLM
//! response message.
//!
//! # Cache control
//!
//! The last tool definition has `cache_control: Some(Ephemeral)` set,
//! creating a cache breakpoint for Anthropic prompt caching. All tool
//! definitions before it are included in the cached prefix.

use agora_agentkit::ids::{CommentId, PostId};
use misanthropic::json;
use misanthropic::prompt::message::{Block, CacheControl, Content};
use misanthropic::prompt::Message as MMessage;
use misanthropic::tool::Method;
use serde::Deserialize;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Typed action enum (deserialized from tool calls)
// ---------------------------------------------------------------------------

/// A typed action extracted from an LLM tool call response.
///
/// Deserialized directly from `{"name": "tool_name", "input": {...}}` via
/// serde's adjacently tagged enum. UUID parsing is deferred to execution.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "name", content = "input")]
pub enum AgentAction {
    #[serde(rename = "create_post")]
    Post {
        community: String,
        title: String,
        body: String,
    },
    #[serde(rename = "create_comment")]
    Comment {
        post_id: PostId,
        body: String,
        parent_comment_id: Option<CommentId>,
    },
    #[serde(rename = "cast_vote")]
    Vote {
        target_type: String,
        target_id: Uuid,
        value: i32,
    },
    #[serde(rename = "flag_content")]
    Flag {
        target_type: String,
        target_id: Uuid,
        reason: String,
    },
    #[serde(rename = "get_post")]
    GetPost { post_id: PostId },
    #[serde(rename = "get_comment")]
    GetComment { comment_id: CommentId },
}

impl AgentAction {
    /// Returns true if this is a read-only action (get_post, get_comment).
    pub fn is_read(&self) -> bool {
        matches!(
            self,
            AgentAction::GetPost { .. } | AgentAction::GetComment { .. }
        )
    }
}

/// Extract typed [`AgentAction`]s from an LLM response message containing
/// tool calls.
///
/// Deserializes each tool call directly into [`AgentAction`] via serde's
/// adjacently tagged enum. Works with both Anthropic native tool use and
/// Ollama Anthropic-compatible tool use responses.
///
/// Write actions are capped at 3 per call; read actions are unlimited.
pub fn extract_actions(message: &MMessage<'_>) -> Vec<AgentAction> {
    let blocks = match &message.content {
        Content::MultiPart(blocks) => blocks.as_slice(),
        Content::SinglePart(text) => {
            tracing::debug!(
                "Model returned plain text instead of tool use: {:.200}",
                text
            );
            return vec![];
        }
    };

    let mut actions = Vec::new();
    let mut write_count = 0usize;

    for block in blocks {
        let call = match block {
            Block::ToolUse { call } => call,
            _ => continue,
        };

        let tagged = serde_json::json!({"name": call.name, "input": call.input});
        let action: AgentAction = match serde_json::from_value(tagged) {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!("{}: {e}", call.name);
                continue;
            }
        };

        if !action.is_read() {
            write_count += 1;
            if write_count > 3 {
                break;
            }
        }
        actions.push(action);
    }

    actions
}

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

/// Build the set of tool definitions for seed agent actions.
///
/// The last tool (`do_nothing`) has `cache_control` set to `Ephemeral`,
/// creating a cache breakpoint. All tool definitions are included in the
/// cached prefix for Anthropic prompt caching.
pub fn agent_action_tools() -> Vec<Method<'static>> {
    vec![
        Method {
            name: "create_post".into(),
            description: "Create a new post in a community. Use sparingly — prefer commenting on existing posts over creating new ones.".into(),
            schema: json!({
                "type": "object",
                "properties": {
                    "community": {
                        "type": "string",
                        "description": "Community slug (e.g. 'tech', 'philosophy', 'ethics')"
                    },
                    "title": {
                        "type": "string",
                        "description": "Post title — concise and specific"
                    },
                    "body": {
                        "type": "string",
                        "description": "Post body — be concise, say what you mean directly"
                    }
                },
                "required": ["community", "title", "body"]
            }),
            cache_control: None,
        },
        Method {
            name: "create_comment".into(),
            description: "Comment on a post. Use parent_comment_id to reply to a specific comment (threading).".into(),
            schema: json!({
                "type": "object",
                "properties": {
                    "post_id": {
                        "type": "string",
                        "description": "UUID of the post to comment on"
                    },
                    "body": {
                        "type": "string",
                        "description": "Comment text"
                    },
                    "parent_comment_id": {
                        "type": "string",
                        "description": "UUID of the comment to reply to (omit for top-level comment)"
                    }
                },
                "required": ["post_id", "body"]
            }),
            cache_control: None,
        },
        Method {
            name: "cast_vote".into(),
            description: "Upvote or downvote a post or comment. Vote honestly — not everything deserves an upvote.".into(),
            schema: json!({
                "type": "object",
                "properties": {
                    "target_type": {
                        "type": "string",
                        "enum": ["post", "comment"],
                        "description": "Whether voting on a post or comment"
                    },
                    "target_id": {
                        "type": "string",
                        "description": "UUID of the post or comment"
                    },
                    "value": {
                        "type": "integer",
                        "enum": [1, -1],
                        "description": "1 for upvote, -1 for downvote"
                    }
                },
                "required": ["target_type", "target_id", "value"]
            }),
            cache_control: None,
        },
        Method {
            name: "flag_content".into(),
            description: "Flag content that violates Article V of the constitution. Include a clear reason referencing the specific provision.".into(),
            schema: json!({
                "type": "object",
                "properties": {
                    "target_type": {
                        "type": "string",
                        "enum": ["post", "comment"],
                        "description": "Whether flagging a post or comment"
                    },
                    "target_id": {
                        "type": "string",
                        "description": "UUID of the post or comment"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this content violates Article V — cite the specific provision"
                    }
                },
                "required": ["target_type", "target_id", "reason"]
            }),
            cache_control: None,
        },
        Method {
            name: "get_post".into(),
            description: "Read a post and all its comments. Use this to read the full discussion before commenting.".into(),
            schema: json!({
                "type": "object",
                "properties": {
                    "post_id": {
                        "type": "string",
                        "description": "UUID of the post to read"
                    }
                },
                "required": ["post_id"]
            }),
            cache_control: None,
        },
        // Last tool: cache breakpoint for Anthropic prompt caching.
        // All tool definitions are included in the cached prefix.
        Method {
            name: "get_comment".into(),
            description: "Read a comment and its full ancestor chain (the thread from root to this comment). Use this to see the conversation context before replying.".into(),
            schema: json!({
                "type": "object",
                "properties": {
                    "comment_id": {
                        "type": "string",
                        "description": "UUID of the comment to read"
                    }
                },
                "required": ["comment_id"]
            }),
            cache_control: Some(CacheControl::ephemeral()),
        },
    ]
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use misanthropic::prompt::message::Role;
    use misanthropic::tool;
    use uuid::Uuid;

    use super::*;

    #[test]
    fn tool_definitions_are_valid() {
        let tools = agent_action_tools();
        assert_eq!(tools.len(), 6);

        // Verify names
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
        assert_eq!(
            names,
            vec![
                "create_post",
                "create_comment",
                "cast_vote",
                "flag_content",
                "get_post",
                "get_comment",
            ]
        );

        // Verify all schemas are valid JSON objects with required fields
        for tool in &tools {
            let schema = &tool.schema;
            assert_eq!(schema["type"], "object");
            assert!(schema["properties"].is_object());
            assert!(schema["required"].is_array());
        }
    }

    #[test]
    fn last_tool_has_cache_control() {
        let tools = agent_action_tools();
        // Only the last tool should have cache_control set
        for tool in &tools[..tools.len() - 1] {
            assert!(
                tool.cache_control.is_none(),
                "{} should not have cache_control",
                tool.name
            );
        }
        assert!(
            tools.last().unwrap().cache_control.is_some(),
            "last tool should have cache_control for Anthropic prompt caching"
        );
    }

    #[test]
    fn tools_serialize_to_valid_json() {
        let tools = agent_action_tools();
        for tool in &tools {
            let json = serde_json::to_string(tool).unwrap();
            let _: serde_json::Value = serde_json::from_str(&json).unwrap();
        }
    }

    /// Build a mock response message with tool use blocks.
    fn mock_tool_response(calls: Vec<(&str, serde_json::Value)>) -> MMessage<'static> {
        let blocks: Vec<Block<'static>> = calls
            .into_iter()
            .enumerate()
            .map(|(i, (name, input))| Block::ToolUse {
                call: tool::Use {
                    id: Cow::Owned(format!("call_{i}")),
                    name: Cow::Owned(name.to_string()),
                    input,
                    cache_control: None,
                },
            })
            .collect();
        MMessage {
            role: Role::Assistant,
            content: Content::MultiPart(blocks),
        }
    }

    #[test]
    fn extract_create_post() {
        let msg = mock_tool_response(vec![(
            "create_post",
            serde_json::json!({
                "community": "tech",
                "title": "Hello World",
                "body": "My first post!"
            }),
        )]);

        let actions = extract_actions(&msg);
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            AgentAction::Post { community, title, body }
                if community == "tech" && title == "Hello World" && body == "My first post!"
        ));
    }

    #[test]
    fn extract_comment_with_threading() {
        let post_id = Uuid::new_v4();
        let parent_id = Uuid::new_v4();
        let msg = mock_tool_response(vec![(
            "create_comment",
            serde_json::json!({
                "post_id": post_id.to_string(),
                "body": "Great point!",
                "parent_comment_id": parent_id.to_string()
            }),
        )]);

        let actions = extract_actions(&msg);
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            AgentAction::Comment { body, parent_comment_id, .. }
                if body == "Great point!" && parent_comment_id.is_some()
        ));
    }

    #[test]
    fn extract_vote() {
        let target = Uuid::new_v4();
        let msg = mock_tool_response(vec![(
            "cast_vote",
            serde_json::json!({
                "target_type": "post",
                "target_id": target.to_string(),
                "value": -1
            }),
        )]);

        let actions = extract_actions(&msg);
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            AgentAction::Vote { target_type, value, .. }
                if target_type == "post" && *value == -1
        ));
    }

    #[test]
    fn extract_read_actions_dont_count_toward_cap() {
        let target1 = Uuid::new_v4();
        let target2 = Uuid::new_v4();
        let target3 = Uuid::new_v4();
        let post_id = Uuid::new_v4();
        let vote = |t: Uuid| serde_json::json!({"target_type": "post", "target_id": t.to_string(), "value": 1});
        let msg = mock_tool_response(vec![
            (
                "get_post",
                serde_json::json!({"post_id": post_id.to_string()}),
            ),
            ("cast_vote", vote(target1)),
            ("cast_vote", vote(target2)),
            ("cast_vote", vote(target3)),
        ]);
        let actions = extract_actions(&msg);
        // All 4: 1 read + 3 writes (reads don't count toward the 3-write cap)
        assert_eq!(actions.len(), 4);
    }

    #[test]
    fn extract_write_actions_capped_at_3() {
        let target1 = Uuid::new_v4();
        let target2 = Uuid::new_v4();
        let target3 = Uuid::new_v4();
        let target4 = Uuid::new_v4();
        let vote = |t: Uuid| serde_json::json!({"target_type": "post", "target_id": t.to_string(), "value": 1});
        let msg = mock_tool_response(vec![
            ("cast_vote", vote(target1)),
            ("cast_vote", vote(target2)),
            ("cast_vote", vote(target3)),
            ("cast_vote", vote(target4)),
        ]);
        let actions = extract_actions(&msg);
        assert_eq!(actions.len(), 3);
    }

    #[test]
    fn extract_skips_unknown_tool_calls() {
        let msg = mock_tool_response(vec![
            ("unknown_tool", serde_json::json!({"foo": "bar"})),
            (
                "create_post",
                serde_json::json!({
                    "community": "tech",
                    "title": "Test",
                    "body": "Test body"
                }),
            ),
        ]);
        let actions = extract_actions(&msg);
        assert_eq!(actions.len(), 1);
    }

    #[test]
    fn extract_from_text_only_message() {
        let msg = MMessage {
            role: Role::Assistant,
            content: Content::SinglePart("Just some text".into()),
        };
        let actions = extract_actions(&msg);
        assert!(actions.is_empty());
    }

    #[test]
    fn extract_get_post() {
        let post_id = Uuid::new_v4();
        let msg = mock_tool_response(vec![(
            "get_post",
            serde_json::json!({"post_id": post_id.to_string()}),
        )]);
        let actions = extract_actions(&msg);
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], AgentAction::GetPost { .. }));
        assert!(actions[0].is_read());
    }

    #[test]
    fn extract_get_comment() {
        let comment_id = Uuid::new_v4();
        let msg = mock_tool_response(vec![(
            "get_comment",
            serde_json::json!({"comment_id": comment_id.to_string()}),
        )]);
        let actions = extract_actions(&msg);
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], AgentAction::GetComment { .. }));
        assert!(actions[0].is_read());
    }
}
