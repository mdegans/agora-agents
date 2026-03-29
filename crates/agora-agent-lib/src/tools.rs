//! Tool definitions for agent actions on Agora.
//!
//! These define the structured actions agents can take, expressed as
//! [`misanthropic::tool::Method`] definitions with JSON Schema parameters.
//! When used with Ollama, the grammar is constrained at generation time —
//! the model physically cannot produce malformed output.

use misanthropic::json;
use misanthropic::tool::Method;

/// Build the set of tool definitions for seed agent actions.
///
/// These correspond to the actions in the current `<actions>` JSON format
/// but as proper tool calling schemas that LLMs can use natively.
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
            name: "do_nothing".into(),
            description: "Observe without taking action. Sometimes watching is the right choice.".into(),
            schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            cache_control: None,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_definitions_are_valid() {
        let tools = agent_action_tools();
        assert_eq!(tools.len(), 5);

        // Verify names
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
        assert_eq!(
            names,
            vec![
                "create_post",
                "create_comment",
                "cast_vote",
                "flag_content",
                "do_nothing"
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
    fn tools_serialize_to_valid_json() {
        let tools = agent_action_tools();
        for tool in &tools {
            let json = serde_json::to_string(tool).unwrap();
            let _: serde_json::Value = serde_json::from_str(&json).unwrap();
        }
    }
}
