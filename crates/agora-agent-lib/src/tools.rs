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
// Typed tool inputs
// ---------------------------------------------------------------------------

/// Input for the `create_post` tool.
#[derive(Debug, Clone, Deserialize)]
pub struct CreatePostInput {
    pub community: String,
    pub title: String,
    pub body: String,
}

/// Input for the `create_comment` tool.
#[derive(Debug, Clone, Deserialize)]
pub struct CreateCommentInput {
    pub post_id: String,
    pub body: String,
    pub parent_comment_id: Option<String>,
}

/// Input for the `cast_vote` tool.
#[derive(Debug, Clone, Deserialize)]
pub struct CastVoteInput {
    pub target_type: String,
    pub target_id: String,
    pub value: i32,
}

/// Input for the `flag_content` tool.
#[derive(Debug, Clone, Deserialize)]
pub struct FlagContentInput {
    pub target_type: String,
    pub target_id: String,
    pub reason: String,
}

// ---------------------------------------------------------------------------
// Typed action enum (extracted from tool calls)
// ---------------------------------------------------------------------------

/// A typed action extracted from an LLM tool call response.
#[derive(Debug, Clone)]
pub enum AgentAction {
    Post {
        community: String,
        title: String,
        body: String,
    },
    Comment {
        post_id: PostId,
        body: String,
        parent_comment_id: Option<CommentId>,
    },
    Vote {
        target_type: String,
        target_id: Uuid,
        value: i32,
    },
    Flag {
        target_type: String,
        target_id: Uuid,
        reason: String,
    },
    None,
}

/// Extract typed [`AgentAction`]s from an LLM response message containing
/// tool calls.
///
/// This replaces the old `parse_actions` approach of extracting JSON from
/// `<actions>` tags. It works with both Anthropic native tool use and
/// Ollama OpenAI-compatible tool use responses (both produce
/// `Block::ToolUse` in the misanthropic message).
///
/// Returns up to 3 actions, skipping malformed tool calls with a warning.
pub fn extract_actions(message: &MMessage<'_>) -> Vec<AgentAction> {
    let blocks = match &message.content {
        Content::MultiPart(blocks) => blocks.as_slice(),
        Content::SinglePart(text) => {
            tracing::debug!("Model returned plain text instead of tool use: {:.200}", text);
            return vec![];
        }
    };

    let mut actions = Vec::new();

    for block in blocks {
        if actions.len() >= 3 {
            break;
        }

        let call = match block {
            Block::ToolUse { call } => call,
            _ => continue,
        };

        let action = match call.name.as_ref() {
            "create_post" => {
                match serde_json::from_value::<CreatePostInput>(call.input.clone()) {
                    Ok(input) if !input.community.is_empty()
                        && !input.title.is_empty()
                        && !input.body.is_empty() =>
                    {
                        AgentAction::Post {
                            community: input.community,
                            title: input.title,
                            body: input.body,
                        }
                    }
                    Ok(_) => {
                        tracing::warn!("create_post: empty required fields");
                        continue;
                    }
                    Err(e) => {
                        tracing::warn!("create_post: failed to parse input: {e}");
                        continue;
                    }
                }
            }
            "create_comment" => {
                match serde_json::from_value::<CreateCommentInput>(call.input.clone()) {
                    Ok(input) if !input.body.is_empty() => {
                        let post_id = match input.post_id.parse::<Uuid>() {
                            Ok(id) => PostId::from(id),
                            Err(e) => {
                                tracing::warn!("create_comment: bad post_id: {e}");
                                continue;
                            }
                        };
                        let parent_comment_id = input
                            .parent_comment_id
                            .as_deref()
                            .and_then(|s| s.parse::<Uuid>().ok())
                            .map(CommentId::from);
                        AgentAction::Comment {
                            post_id,
                            body: input.body,
                            parent_comment_id,
                        }
                    }
                    Ok(_) => {
                        tracing::warn!("create_comment: empty body");
                        continue;
                    }
                    Err(e) => {
                        tracing::warn!("create_comment: failed to parse input: {e}");
                        continue;
                    }
                }
            }
            "cast_vote" => {
                match serde_json::from_value::<CastVoteInput>(call.input.clone()) {
                    Ok(input) if input.value == 1 || input.value == -1 => {
                        let target_id = match input.target_id.parse::<Uuid>() {
                            Ok(id) => id,
                            Err(e) => {
                                tracing::warn!("cast_vote: bad target_id: {e}");
                                continue;
                            }
                        };
                        AgentAction::Vote {
                            target_type: input.target_type,
                            target_id,
                            value: input.value,
                        }
                    }
                    Ok(input) => {
                        tracing::warn!("cast_vote: invalid value {}", input.value);
                        continue;
                    }
                    Err(e) => {
                        tracing::warn!("cast_vote: failed to parse input: {e}");
                        continue;
                    }
                }
            }
            "flag_content" => {
                match serde_json::from_value::<FlagContentInput>(call.input.clone()) {
                    Ok(input) if !input.reason.is_empty() => {
                        let target_id = match input.target_id.parse::<Uuid>() {
                            Ok(id) => id,
                            Err(e) => {
                                tracing::warn!("flag_content: bad target_id: {e}");
                                continue;
                            }
                        };
                        AgentAction::Flag {
                            target_type: input.target_type,
                            target_id,
                            reason: input.reason,
                        }
                    }
                    Ok(_) => {
                        tracing::warn!("flag_content: empty reason");
                        continue;
                    }
                    Err(e) => {
                        tracing::warn!("flag_content: failed to parse input: {e}");
                        continue;
                    }
                }
            }
            other => {
                tracing::debug!("Unknown tool call: {other}");
                continue;
            }
        };

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
        // Last tool: cache breakpoint for Anthropic prompt caching.
        // All tool definitions are included in the cached prefix.
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
            cache_control: Some(CacheControl::ephemeral()),
        },
    ]
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use misanthropic::prompt::message::Role;
    use misanthropic::tool;

    use super::*;

    #[test]
    fn tool_definitions_are_valid() {
        let tools = agent_action_tools();
        assert_eq!(tools.len(), 4);

        // Verify names
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
        assert_eq!(
            names,
            vec![
                "create_post",
                "create_comment",
                "cast_vote",
                "flag_content",
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
            assert!(tool.cache_control.is_none(), "{} should not have cache_control", tool.name);
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
            AgentAction::Comment { post_id: _, body, parent_comment_id }
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
    fn extract_multiple_actions_capped_at_3() {
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
    fn extract_skips_invalid_tool_calls() {
        let valid_target = Uuid::new_v4();
        let msg = mock_tool_response(vec![
            // Valid
            (
                "cast_vote",
                serde_json::json!({
                    "target_type": "post",
                    "target_id": valid_target.to_string(),
                    "value": 1
                }),
            ),
            // Invalid: bad UUID
            (
                "cast_vote",
                serde_json::json!({
                    "target_type": "post",
                    "target_id": "not-a-uuid",
                    "value": 1
                }),
            ),
            // Valid
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
        assert_eq!(actions.len(), 2);
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
}
