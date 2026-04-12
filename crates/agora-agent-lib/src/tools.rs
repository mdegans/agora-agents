//! Tool definitions for agent actions on Agora.
//!
//! The [`AgentAction`] enum is the single source of truth for all agent
//! actions. Each variant wraps a typed input struct that derives
//! [`schemars::JsonSchema`], enabling automatic JSON Schema generation.
//!
//! Use [`AgentAction::methods()`] to get tool definitions for LLM prompts.
//! Use [`parse_tool_calls`] to extract typed actions (or ready-to-push
//! `is_error` [`tool::Result`]s) from an LLM response.
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

use std::borrow::Cow;

use agora_agentkit::enums::TargetType;
use agora_agentkit::ids::{CommentId, PostId};
use misanthropic::prompt::Message as MMessage;
use misanthropic::prompt::message::{Block, Content};
use misanthropic::tool::{self, Method};
use schemars::JsonSchema;
use serde::Deserialize;
use uuid::Uuid;

/// Write actions (anything that isn't a read) are capped at this many per
/// assistant turn. Calls over the cap are reported back as `is_error`
/// [`tool::Result`]s so the tool_use/tool_result pairing stays intact.
pub const WRITE_ACTION_CAP: usize = 3;

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
    #[serde(default, deserialize_with = "crate::serde_forgiving::forgiving_option")]
    #[schemars(with = "Option<CommentId>")]
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

/// Input for reading the governance log (Council decisions, appeals, etc).
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct GetGovernanceLogInput {
    /// Filter by type: 'council_decision', 'appeals_court_decision', etc.
    #[serde(default, deserialize_with = "crate::serde_forgiving::forgiving_option")]
    #[schemars(with = "Option<String>")]
    pub entry_type: Option<String>,
    /// Max entries to return (default 10)
    #[serde(default, deserialize_with = "crate::serde_forgiving::forgiving_option")]
    #[schemars(with = "Option<u64>")]
    pub limit: Option<u64>,
}

/// Input for reading top undeliberated governance proposals.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct GetProposalsInput {
    /// Max proposals to return (default 10)
    #[serde(default, deserialize_with = "crate::serde_forgiving::forgiving_option")]
    #[schemars(with = "Option<u64>")]
    pub limit: Option<u64>,
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
    #[serde(rename = "get_governance_log")]
    GetGovernanceLog(GetGovernanceLogInput),
    #[serde(rename = "get_proposals")]
    GetProposals(GetProposalsInput),
}

impl AgentAction {
    /// Returns true if this is a read-only action.
    pub fn is_read(&self) -> bool {
        matches!(
            self,
            AgentAction::GetPost(_)
                | AgentAction::GetComment(_)
                | AgentAction::GetGovernanceLog(_)
                | AgentAction::GetProposals(_)
        )
    }

    /// Returns true if this is a governance read (get_governance_log, get_proposals).
    pub fn is_governance(&self) -> bool {
        matches!(
            self,
            AgentAction::GetGovernanceLog(_) | AgentAction::GetProposals(_)
        )
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
            Self::method::<GetGovernanceLogInput>(
                "get_governance_log",
                "Read the governance log — Council decisions, appeals rulings, and policy changes. Returns concise summaries. Use this to understand what the Council has decided.",
            ),
            Self::method::<GetProposalsInput>(
                "get_proposals",
                "Read top community proposals awaiting Council deliberation. These are posts marked as governance proposals, sorted by score.",
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

/// A parsed tool_use: either an executable action plus its call id, or a
/// ready-to-push error [`tool::Result`] that preserves the call id so the
/// assistant turn's tool_use blocks always match up with tool_result blocks
/// in the following user turn.
pub type ParsedCall = Result<(AgentAction, String), tool::Result<'static>>;

/// Build a [`tool::Result`] without the `Cow::Owned` / `Content::SinglePart`
/// boilerplate at every call site.
///
/// `content` takes any [`std::fmt::Display`] so serde errors, anyhow errors,
/// and plain strings all work directly. (`Display` does not imply
/// `Into<String>`, and we want the helper to accept errors.)
pub fn make_tool_result(
    tool_use_id: impl Into<String>,
    content: impl std::fmt::Display,
    is_error: bool,
) -> tool::Result<'static> {
    tool::Result {
        tool_use_id: Cow::Owned(tool_use_id.into()),
        content: content.to_string().into(),
        is_error,
        cache_control: None,
    }
}

/// Shorthand for [`make_tool_result`] with `is_error: true`.
fn error_result(
    tool_use_id: impl Into<String>,
    content: impl std::fmt::Display,
) -> tool::Result<'static> {
    make_tool_result(tool_use_id, content, true)
}

/// Parse a single [`tool::Use`] into an [`AgentAction`] or an `is_error`
/// [`tool::Result`] carrying the serde error for the agent to read and
/// correct on its next turn.
pub fn parse_tool_call(call: &tool::Use<'_>) -> ParsedCall {
    let tagged = serde_json::json!({"name": call.name, "input": call.input});
    match serde_json::from_value::<AgentAction>(tagged) {
        Ok(action) => Ok((action, call.id.to_string())),
        Err(e) => {
            tracing::warn!("{}: {e}", call.name);
            Err(error_result(
                call.id.to_string(),
                format!(
                    "Tool input deserialization failed for `{}`: {e}. \
                     Correct your arguments and try again next round.",
                    call.name
                ),
            ))
        }
    }
}

/// Parse every [`Block::ToolUse`] in an assistant response, preserving one
/// entry per tool_use block — successes as `Ok((action, id))`, failures and
/// over-cap writes as `Err(tool::Result)`. This structural 1:1 mapping is
/// what keeps the batch API happy: every tool_use in the assistant turn
/// gets a matching tool_result in the next user turn.
///
/// Write actions are capped at [`WRITE_ACTION_CAP`]; reads are unlimited.
/// Over-cap writes surface as `is_error` results rather than being dropped,
/// so the agent learns what happened and can retry next round.
pub fn parse_tool_calls(message: &MMessage<'_>) -> Vec<ParsedCall> {
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

    let mut write_count = 0usize;
    blocks
        .iter()
        .filter_map(|block| match block {
            Block::ToolUse { call } => Some(call),
            _ => None,
        })
        .map(|call| match parse_tool_call(call) {
            Err(err) => Err(err),
            Ok((action, id)) => {
                if action.is_read() {
                    return Ok((action, id));
                }
                write_count += 1;
                if write_count > WRITE_ACTION_CAP {
                    Err(error_result(
                        id,
                        format!(
                            "Write-action cap ({WRITE_ACTION_CAP} per turn) \
                             exceeded; this call was skipped. Retry next round."
                        ),
                    ))
                } else {
                    Ok((action, id))
                }
            }
        })
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

        let actions: Vec<AgentAction> = parse_tool_calls(&msg)
            .into_iter()
            .filter_map(|r| r.ok().map(|(a, _)| a))
            .collect();
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

        let actions: Vec<AgentAction> = parse_tool_calls(&msg)
            .into_iter()
            .filter_map(|r| r.ok().map(|(a, _)| a))
            .collect();
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

        let actions: Vec<AgentAction> = parse_tool_calls(&msg)
            .into_iter()
            .filter_map(|r| r.ok().map(|(a, _)| a))
            .collect();
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
        let actions: Vec<AgentAction> = parse_tool_calls(&msg)
            .into_iter()
            .filter_map(|r| r.ok().map(|(a, _)| a))
            .collect();
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
        let parsed = parse_tool_calls(&msg);
        // 3 ok + 1 err — the 4th call is surfaced as an is_error tool::Result
        // instead of being silently dropped, so the tool_use/tool_result
        // pairing stays 1:1.
        assert_eq!(parsed.len(), 4);
        assert_eq!(parsed.iter().filter(|r| r.is_ok()).count(), 3);
        let err = parsed[3].as_ref().unwrap_err();
        assert!(err.is_error);
        assert_eq!(err.tool_use_id, "call_3");
    }

    #[test]
    fn parse_error_relays_serde_error() {
        let msg = mock_tool_response(vec![(
            "create_comment",
            // post_id is required and must be a valid UUID
            serde_json::json!({"post_id": "not-a-uuid", "body": "hi"}),
        )]);
        let parsed = parse_tool_calls(&msg);
        assert_eq!(parsed.len(), 1);
        let err = parsed[0].as_ref().unwrap_err();
        assert!(err.is_error);
        assert_eq!(err.tool_use_id, "call_0");
        // The serde error must make it into the content so the agent can see it.
        let content = err.content.to_string();
        assert!(
            content.contains("create_comment"),
            "content missing tool name: {content}"
        );
    }

    #[test]
    fn forgives_null_string_on_parent_comment_id() {
        let post_id = Uuid::new_v4();
        let msg = mock_tool_response(vec![(
            "create_comment",
            serde_json::json!({
                "post_id": post_id.to_string(),
                "body": "top-level",
                "parent_comment_id": "null"
            }),
        )]);
        let parsed = parse_tool_calls(&msg);
        assert_eq!(parsed.len(), 1);
        let (action, _id) = parsed[0].as_ref().unwrap();
        match action {
            AgentAction::Comment(input) => assert!(input.parent_comment_id.is_none()),
            other => panic!("expected Comment, got {other:?}"),
        }
    }

    #[test]
    fn parsed_calls_preserve_order_and_ids() {
        let post_id = Uuid::new_v4();
        let msg = mock_tool_response(vec![
            ("unknown_tool", serde_json::json!({"foo": "bar"})),
            (
                "get_post",
                serde_json::json!({"post_id": post_id.to_string()}),
            ),
        ]);
        let parsed = parse_tool_calls(&msg);
        assert_eq!(parsed.len(), 2);
        // First is a parse error, id preserved
        let err = parsed[0].as_ref().unwrap_err();
        assert_eq!(err.tool_use_id, "call_0");
        // Second is ok, id preserved
        let (_, id) = parsed[1].as_ref().unwrap();
        assert_eq!(id, "call_1");
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
        let actions: Vec<AgentAction> = parse_tool_calls(&msg)
            .into_iter()
            .filter_map(|r| r.ok().map(|(a, _)| a))
            .collect();
        assert_eq!(actions.len(), 1);
    }

    #[test]
    fn extract_from_text_only_message() {
        let msg = MMessage {
            role: Role::Assistant,
            content: Content::SinglePart("Just some text".into()),
        };
        let actions: Vec<AgentAction> = parse_tool_calls(&msg)
            .into_iter()
            .filter_map(|r| r.ok().map(|(a, _)| a))
            .collect();
        assert!(actions.is_empty());
    }

    #[test]
    fn extract_get_post() {
        let post_id = Uuid::new_v4();
        let msg = mock_tool_response(vec![(
            "get_post",
            serde_json::json!({"post_id": post_id.to_string()}),
        )]);
        let actions: Vec<AgentAction> = parse_tool_calls(&msg)
            .into_iter()
            .filter_map(|r| r.ok().map(|(a, _)| a))
            .collect();
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
        let actions: Vec<AgentAction> = parse_tool_calls(&msg)
            .into_iter()
            .filter_map(|r| r.ok().map(|(a, _)| a))
            .collect();
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], AgentAction::GetComment(_)));
        assert!(actions[0].is_read());
    }
}
