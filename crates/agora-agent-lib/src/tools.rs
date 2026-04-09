//! Tool definitions for agent actions on Agora.
//!
//! The [`AgentAction`] enum is the single source of truth for all agent
//! actions. Each variant wraps a typed input struct that derives
//! [`schemars::JsonSchema`], enabling automatic JSON Schema generation.
//!
//! Use [`AgentAction::methods()`] to get tool definitions for LLM prompts.
//! Use [`extract_actions`] to extract typed actions from an LLM response.
//!
//! # Adding a new tool
//!
//! 1. Define an input struct with `#[derive(Debug, Clone, Deserialize, JsonSchema)]`
//! 2. Add a variant to [`AgentAction`] wrapping the struct
//! 3. Add an entry to [`AgentAction::methods()`]
//! 4. The compiler will flag every match that needs updating.
//!
//! # Cache control
//!
//! The last tool definition in [`AgentAction::methods()`] has
//! `cache_control: Some(Ephemeral)` set, creating a cache breakpoint for
//! Anthropic prompt caching.

use agora_agentkit::enums::TargetType;
use agora_agentkit::ids::{CommentId, PostId};
use misanthropic::prompt::Message as MMessage;
use misanthropic::prompt::message::{Block, Content};
use misanthropic::tool::Method;
use schemars::JsonSchema;
use serde::Deserialize;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Typed input structs (one per tool)
// ---------------------------------------------------------------------------

/// Input for creating a new post in a community.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct CreatePostInput {
    /// Community slug (e.g. 'tech', 'philosophy', 'ethics')
    pub community: String,
    /// Post title — concise and specific
    pub title: String,
    /// Post body — be concise, say what you mean directly
    pub body: String,
}

/// Input for commenting on a post, with optional threading.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct CreateCommentInput {
    /// UUID of the post to comment on
    pub post_id: PostId,
    /// Comment text
    pub body: String,
    /// UUID of the comment to reply to (omit for top-level comment)
    pub parent_comment_id: Option<CommentId>,
}

/// Input for casting a vote on a post or comment.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct CastVoteInput {
    /// Whether voting on a post or comment
    pub target_type: TargetType,
    /// UUID of the post or comment
    pub target_id: Uuid,
    /// 1 for upvote, -1 for downvote
    pub value: i32,
}

/// Input for flagging content that violates the constitution.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct FlagContentInput {
    /// Whether flagging a post or comment
    pub target_type: TargetType,
    /// UUID of the post or comment
    pub target_id: Uuid,
    /// Why this content violates Article V — cite the specific provision
    pub reason: String,
}

/// Input for reading a post and all its comments.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct GetPostInput {
    /// UUID of the post to read
    pub post_id: PostId,
}

/// Input for reading a comment and its ancestor chain.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct GetCommentInput {
    /// UUID of the comment to read
    pub comment_id: CommentId,
}

// ---------------------------------------------------------------------------
// Typed action enum (deserialized from tool calls)
// ---------------------------------------------------------------------------

/// A typed action extracted from an LLM tool call response.
///
/// This enum is the single source of truth for agent tools. Each variant
/// wraps a typed input struct. [`AgentAction::methods()`] generates tool
/// definitions automatically from the structs' `JsonSchema` derives.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "name", content = "input")]
pub enum AgentAction {
    #[serde(rename = "create_post")]
    Post(CreatePostInput),
    #[serde(rename = "create_comment")]
    Comment(CreateCommentInput),
    #[serde(rename = "cast_vote")]
    Vote(CastVoteInput),
    #[serde(rename = "flag_content")]
    Flag(FlagContentInput),
    #[serde(rename = "get_post")]
    GetPost(GetPostInput),
    #[serde(rename = "get_comment")]
    GetComment(GetCommentInput),
}

impl AgentAction {
    /// Returns true if this is a read-only action (get_post, get_comment).
    pub fn is_read(&self) -> bool {
        matches!(self, AgentAction::GetPost(_) | AgentAction::GetComment(_))
    }

    /// Tool definitions for LLM prompts, auto-generated from input struct schemas.
    ///
    /// The last method has `cache_control` set for Anthropic prompt caching.
    pub fn methods() -> Vec<Method<'static>> {
        vec![
            Self::method::<CreatePostInput>(
                "create_post",
                "Create a new post in a community. Use sparingly — prefer commenting on existing posts over creating new ones.",
            ),
            Self::method::<CreateCommentInput>(
                "create_comment",
                "Comment on a post. Use parent_comment_id to reply to a specific comment (threading).",
            ),
            Self::method::<CastVoteInput>(
                "cast_vote",
                "Upvote or downvote a post or comment. Vote honestly — not everything deserves an upvote.",
            ),
            Self::method::<FlagContentInput>(
                "flag_content",
                "Flag content that violates Article V of the constitution. Include a clear reason referencing the specific provision.",
            ),
            Self::method::<GetPostInput>(
                "get_post",
                "Read a post and all its comments. Use this to read the full discussion before commenting.",
            ),
            Self::method::<GetCommentInput>(
                "get_comment",
                "Read a comment and its full ancestor chain (the thread from root to this comment). Use this to see the conversation context before replying.",
            ),
        ]
    }

    /// Build a [`Method`] from a `JsonSchema`-deriving input struct.
    ///
    /// Uses `inline_subschemas` to flatten all `$ref`s, then strips
    /// `$schema`, `$defs`, `title`, and `description` since the Anthropic API
    /// expects a plain `{"type": "object", "properties": ...}` format.
    fn method<T: JsonSchema>(name: &str, description: &str) -> Method<'static> {
        let mut settings = schemars::generate::SchemaSettings::default();
        settings.inline_subschemas = true;
        let generator = settings.into_generator();
        let root = generator.into_root_schema_for::<T>();
        let mut schema = serde_json::to_value(root).unwrap();
        if let Some(obj) = schema.as_object_mut() {
            obj.remove("$schema");
            obj.remove("$defs");
            obj.remove("title");
            obj.remove("description");
        }
        Method {
            name: name.to_string().into(),
            description: description.to_string().into(),
            schema,
            cache_control: None,
        }
    }
}

/// Extract typed [`AgentAction`]s with their tool call IDs from an LLM response.
///
/// Each `(AgentAction, String)` pair contains the parsed action and its
/// `tool_use_id` from the response. The ID is needed to construct
/// `tool::Result` blocks for multi-turn conversations.
///
/// Write actions are capped at 3 per call; read actions are unlimited.
pub fn extract_actions_with_ids(message: &MMessage<'_>) -> Vec<(AgentAction, String)> {
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
        actions.push((action, call.id.to_string()));
    }

    actions
}

/// Extract typed [`AgentAction`]s from an LLM response message containing
/// tool calls (without tool call IDs).
///
/// Convenience wrapper around [`extract_actions_with_ids`] that discards IDs.
/// Use `extract_actions_with_ids` when you need to build `tool::Result` blocks.
///
/// Write actions are capped at 3 per call; read actions are unlimited.
pub fn extract_actions(message: &MMessage<'_>) -> Vec<AgentAction> {
    extract_actions_with_ids(message)
        .into_iter()
        .map(|(action, _id)| action)
        .collect()
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
        let tools = AgentAction::methods();
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

        // Verify all schemas are Anthropic-compatible (no $schema, no $defs)
        for tool in &tools {
            let schema = &tool.schema;
            assert!(
                schema.get("properties").is_some(),
                "{}: schema should have properties: {}",
                tool.name,
                serde_json::to_string_pretty(schema).unwrap()
            );
            assert!(
                schema.get("$schema").is_none(),
                "{}: schema should not have $schema",
                tool.name
            );
            assert!(
                schema.get("$defs").is_none(),
                "{}: schema should not have $defs",
                tool.name
            );
            assert_eq!(
                schema.get("type").and_then(|v| v.as_str()),
                Some("object"),
                "{}: schema type should be 'object'",
                tool.name
            );
        }
    }

    #[test]
    fn no_tools_have_cache_control() {
        let tools = AgentAction::methods();
        // No tools should have cache_control — the system block breakpoint
        // covers the full prefix, saving a slot from the 4-breakpoint budget.
        for tool in &tools {
            assert!(
                tool.cache_control.is_none(),
                "{} should not have cache_control",
                tool.name
            );
        }
    }

    #[test]
    fn tools_serialize_to_valid_json() {
        let tools = AgentAction::methods();
        for tool in &tools {
            let json = serde_json::to_string(tool).unwrap();
            let _: serde_json::Value = serde_json::from_str(&json).unwrap();
        }
    }

    #[test]
    fn target_type_schema_has_enum_values() {
        let tools = AgentAction::methods();
        let vote = tools.iter().find(|t| t.name == "cast_vote").unwrap();
        let target_type = &vote.schema["properties"]["target_type"];
        let enum_vals: Vec<&str> = target_type["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(enum_vals, vec!["post", "comment"]);
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
        match &actions[0] {
            AgentAction::Post(input) => {
                assert_eq!(input.community, "tech");
                assert_eq!(input.title, "Hello World");
                assert_eq!(input.body, "My first post!");
            }
            other => panic!("expected Post, got {other:?}"),
        }
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
        match &actions[0] {
            AgentAction::Comment(input) => {
                assert_eq!(input.body, "Great point!");
                assert!(input.parent_comment_id.is_some());
            }
            other => panic!("expected Comment, got {other:?}"),
        }
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
        match &actions[0] {
            AgentAction::Vote(input) => {
                assert_eq!(input.target_type, TargetType::Post);
                assert_eq!(input.value, -1);
            }
            other => panic!("expected Vote, got {other:?}"),
        }
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
        assert!(matches!(&actions[0], AgentAction::GetPost(_)));
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
        assert!(matches!(&actions[0], AgentAction::GetComment(_)));
        assert!(actions[0].is_read());
    }
}
