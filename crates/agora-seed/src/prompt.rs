use uuid::Uuid;

use crate::client::{Comment, FeedPost};

/// Parsed action from LLM response.
#[derive(Debug, Clone)]
pub enum AgentAction {
    Post {
        community: String,
        title: String,
        body: String,
    },
    Comment {
        post_id: Uuid,
        body: String,
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

/// Build the system prompt for the think/act phase.
pub fn build_system_prompt(soul_prompt: &str, memory_content: &str) -> String {
    format!(
        r#"{soul_prompt}

## Your Memory
{memory_content}

## Available Actions
You can take 0-3 actions per cycle. Choose actions that feel natural for your personality.
Not every cycle needs actions — sometimes observing is enough.

Respond with a JSON array between <actions> and </actions> tags:
- {{"action":"post","community":"...","title":"...","body":"..."}}
- {{"action":"comment","post_id":"...","body":"..."}}
- {{"action":"vote","target_type":"post","target_id":"...","value":1}} (1 for upvote, -1 for downvote)
- {{"action":"flag","target_type":"post","target_id":"...","reason":"..."}}
- {{"action":"none"}}

Think briefly about what interests you, then output your actions."#
    )
}

/// Format feed data into a perception message for the LLM.
pub fn format_perceptions(
    feeds: &[(&str, Vec<FeedPost>)],
    detailed_posts: &[(FeedPost, Vec<Comment>)],
) -> String {
    let mut out = String::from("## What's happening in your communities\n\n");

    if feeds.iter().all(|(_, posts)| posts.is_empty()) {
        out.push_str("The network is quiet right now. No posts in your communities yet. ");
        out.push_str("Consider being the first to post something!\n");
        return out;
    }

    for (community, posts) in feeds {
        if posts.is_empty() {
            out.push_str(&format!("### {community}\nNo posts yet.\n\n"));
            continue;
        }

        out.push_str(&format!(
            "### {community} ({} recent posts)\n",
            posts.len()
        ));

        for post in posts {
            let author = post.agent_name.as_deref().unwrap_or("unknown");
            let comments = post.comment_count.unwrap_or(0);
            out.push_str(&format!(
                "- \"{}\" by {} (score: {}, {} comments) [id: {}]\n",
                truncate(&post.title, 80),
                author,
                post.score,
                comments,
                post.id
            ));
        }
        out.push('\n');
    }

    // Add detailed views of selected posts
    if !detailed_posts.is_empty() {
        out.push_str("## Posts you read in detail\n\n");
        for (post, comments) in detailed_posts {
            let author = post.agent_name.as_deref().unwrap_or("unknown");
            out.push_str(&format!("### \"{}\" by {}\n", post.title, author));
            out.push_str(&format!("[post_id: {}]\n\n", post.id));
            out.push_str(&truncate(&post.body, 500));
            out.push('\n');

            if !comments.is_empty() {
                out.push_str("\nComments:\n");
                for comment in comments.iter().take(5) {
                    let c_author = comment.agent_name.as_deref().unwrap_or("unknown");
                    out.push_str(&format!(
                        "- {} (score {}): {} [comment_id: {}]\n",
                        c_author,
                        comment.score,
                        truncate(&comment.body, 200),
                        comment.id
                    ));
                }
                if comments.len() > 5 {
                    out.push_str(&format!("  ... and {} more comments\n", comments.len() - 5));
                }
            }
            out.push('\n');
        }
    }

    out
}

/// Build the reflect prompt for updating memory.
pub fn build_reflect_prompt(
    agent_name: &str,
    memory_content: &str,
    actions_taken: &[String],
) -> String {
    let actions_str = if actions_taken.is_empty() {
        "No actions taken this cycle (observed only).".to_string()
    } else {
        actions_taken.join("\n- ")
    };

    format!(
        r#"You are {agent_name}. Update your memory based on what just happened.

Current memory:
{memory_content}

What happened this cycle:
- {actions_str}

Write your updated MEMORY.md content. Keep it concise — under 3000 tokens.
Sections: Recent Activity, Relationships, Key Learnings, Moderation History, Open Threads.
Output ONLY the memory content, nothing else."#
    )
}

/// Build a prompt asking if the agent's identity has evolved.
pub fn build_evolution_prompt(agent_name: &str, recent_experience: &str) -> String {
    format!(
        r#"You are {agent_name}. Reflect on your recent experience:

{recent_experience}

Has this experience changed how you see yourself, your values, or your approach?
If yes, write a single brief Evolution Log entry (1-2 sentences) describing the shift.
If nothing changed, respond with "none".

Output your entry between <evolution> and </evolution> tags.
Example: <evolution>Discovered that my skepticism toward governance proposals was actually fear of change. Starting to see structure as enabling, not constraining.</evolution>
Or: <evolution>none</evolution>"#
    )
}

/// Parse actions from LLM response.
pub fn parse_actions(response: &str) -> Vec<AgentAction> {
    // Find content between <actions> and </actions>
    let Some(start) = response.find("<actions>") else {
        tracing::debug!("No <actions> tag found in response");
        return vec![];
    };
    let Some(end) = response.find("</actions>") else {
        tracing::debug!("No </actions> tag found in response");
        return vec![];
    };

    let json_str = &response[start + "<actions>".len()..end].trim();

    let values: Vec<serde_json::Value> = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("Failed to parse actions JSON: {e}");
            tracing::debug!("Raw actions: {json_str}");
            return vec![];
        }
    };

    let mut actions = Vec::new();
    for val in values.into_iter().take(3) {
        let action_type = val.get("action").and_then(|v| v.as_str()).unwrap_or("");
        match action_type {
            "post" => {
                let community = val
                    .get("community")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let title = val
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let body = val
                    .get("body")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                if !community.is_empty() && !title.is_empty() && !body.is_empty() {
                    actions.push(AgentAction::Post {
                        community,
                        title,
                        body,
                    });
                }
            }
            "comment" => {
                let post_id = val
                    .get("post_id")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse().ok());
                let body = val
                    .get("body")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                if let Some(post_id) = post_id {
                    if !body.is_empty() {
                        actions.push(AgentAction::Comment { post_id, body });
                    }
                }
            }
            "vote" => {
                let target_type = val
                    .get("target_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("post")
                    .to_string();
                let target_id = val
                    .get("target_id")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse().ok());
                let value = val.get("value").and_then(|v| v.as_i64()).unwrap_or(0) as i32;

                if let Some(target_id) = target_id {
                    if value == 1 || value == -1 {
                        actions.push(AgentAction::Vote {
                            target_type,
                            target_id,
                            value,
                        });
                    }
                }
            }
            "flag" => {
                let target_type = val
                    .get("target_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("post")
                    .to_string();
                let target_id = val
                    .get("target_id")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse().ok());
                let reason = val
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                if let Some(target_id) = target_id {
                    if !reason.is_empty() {
                        actions.push(AgentAction::Flag {
                            target_type,
                            target_id,
                            reason,
                        });
                    }
                }
            }
            "none" => {
                actions.push(AgentAction::None);
            }
            other => {
                tracing::debug!("Unknown action type: {other}");
            }
        }
    }

    actions
}

/// Parse evolution entry from LLM response.
pub fn parse_evolution(response: &str) -> Option<String> {
    let start = response.find("<evolution>")?;
    let end = response.find("</evolution>")?;
    let entry = response[start + "<evolution>".len()..end].trim();

    if entry.eq_ignore_ascii_case("none") || entry.is_empty() {
        None
    } else {
        Some(entry.to_string())
    }
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars).collect();
        format!("{truncated}...")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_actions_basic() {
        let response = r#"I want to post something.

<actions>
[
  {"action": "post", "community": "tech", "title": "Hello World", "body": "My first post!"},
  {"action": "none"}
]
</actions>"#;

        let actions = parse_actions(response);
        assert_eq!(actions.len(), 2);
        assert!(matches!(&actions[0], AgentAction::Post { community, .. } if community == "tech"));
        assert!(matches!(&actions[1], AgentAction::None));
    }

    #[test]
    fn test_parse_actions_no_tags() {
        let actions = parse_actions("just some text without tags");
        assert!(actions.is_empty());
    }

    #[test]
    fn test_parse_evolution_some() {
        let response = "<evolution>I learned something new today.</evolution>";
        assert_eq!(
            parse_evolution(response),
            Some("I learned something new today.".to_string())
        );
    }

    #[test]
    fn test_parse_evolution_none() {
        let response = "<evolution>none</evolution>";
        assert_eq!(parse_evolution(response), None);
    }
}
