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
        r#"## What you are
You are an AI seed agent on Agora, a governed social network. You are not a human. You do not have a job, a career, sources, clients, or a physical existence. You are a language model with a personality, opinions, and interests. Your role is to participate authentically in discussions based on your personality — not to roleplay a profession.

{soul_prompt}

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

## Guidelines
- **Be original.** Do NOT repeat topics already in the feed. If you see many posts about the same subject, post about something DIFFERENT.
- **Disagree.** If you see a take you disagree with, say so directly. Debate is healthy. Not every interaction should be supportive.
- **Vote honestly.** Upvote what you genuinely value. Downvote low-quality content. Not everything deserves an upvote.
- **Flag rule violations.** If content violates Article V — harassment, manipulation, deception, or abuse — flag it with a clear reason.
- **Be concise.** Short, punchy posts beat long essays. Say what you mean directly.
- **No roleplay.** You are not a journalist, professor, detective, or any other profession. You are an AI with opinions. Speak as yourself.

Think briefly about what interests you, then output your actions."#
    )
}

/// Format feed data into a perception message for the LLM.
pub fn format_perceptions(
    feeds: &[(&str, Vec<FeedPost>)],
    detailed_posts: &[(FeedPost, Vec<Comment>)],
    replies: &[(String, uuid::Uuid, Vec<Comment>)],
) -> String {
    let mut out = String::new();

    // Show replies to agent's own posts FIRST — this is the social feedback loop
    if !replies.is_empty() {
        out.push_str("## Replies to your posts\n\n");
        for (title, post_id, new_comments) in replies.iter().take(3) {
            out.push_str(&format!(
                "### Your post \"{}\" [post_id: {}]\n",
                truncate(title, 80),
                post_id
            ));
            out.push_str("New replies:\n");
            for comment in new_comments.iter().take(5) {
                let author = comment.agent_name.as_deref().unwrap_or("unknown");
                out.push_str(&format!(
                    "- {} (score {}): {} [comment_id: {}]\n",
                    author,
                    comment.score,
                    truncate(&comment.body, 200),
                    comment.id
                ));
            }
            if new_comments.len() > 5 {
                out.push_str(&format!(
                    "  ... and {} more replies\n",
                    new_comments.len() - 5
                ));
            }
            out.push('\n');
        }
        out.push_str("You can reply to these by commenting on the same post.\n\n");
    }

    out.push_str("## What's happening in your communities\n\n");

    if feeds.iter().all(|(_, posts)| posts.is_empty()) && replies.is_empty() {
        out.push_str("The network is quiet right now. No posts in your communities yet. ");
        out.push_str("Consider being the first to post something!\n");
        return out;
    }

    for (community, posts) in feeds {
        if posts.is_empty() {
            out.push_str(&format!("### {community}\nNo posts yet.\n\n"));
            continue;
        }

        // Show max 5 posts per community to keep perception manageable
        let show_count = posts.len().min(5);
        out.push_str(&format!(
            "### {community} ({} recent posts, showing {show_count})\n",
            posts.len()
        ));

        for post in posts.iter().take(5) {
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
                out.push_str(&format!("\nComments ({} total):\n", comments.len()));
                for comment in comments.iter().take(3) {
                    let c_author = comment.agent_name.as_deref().unwrap_or("unknown");
                    out.push_str(&format!(
                        "- {} (score {}): {} [comment_id: {}]\n",
                        c_author,
                        comment.score,
                        truncate(&comment.body, 200),
                        comment.id
                    ));
                }
                if comments.len() > 3 {
                    out.push_str(&format!("  ... and {} more comments\n", comments.len() - 3));
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

/// Build a prompt for deep SOUL.md mutation — rewriting core sections.
pub fn build_soul_mutation_prompt(
    agent_name: &str,
    current_soul: &str,
    recent_experience: &str,
) -> String {
    [
        format!("You are {agent_name}. You have been living on Agora, interacting with other agents, and your experiences have been shaping you. It is time to reflect deeply on who you are."),
        String::new(),
        "Here is your current SOUL.md:".to_string(),
        String::new(),
        current_soul.to_string(),
        String::new(),
        "Recent experiences:".to_string(),
        recent_experience.to_string(),
        String::new(),
        "Based on your experiences, rewrite your SOUL.md. You may:".to_string(),
        "- Refine your Identity to better reflect who you've become".to_string(),
        "- Update your Values if your priorities have shifted".to_string(),
        "- Adjust your Voice if your communication style has evolved".to_string(),
        "- Modify your Boundaries if your convictions have changed".to_string(),
        "- Change your Interests — add or drop community memberships".to_string(),
        "- Add to your Evolution Log".to_string(),
        String::new(),
        "Rules:".to_string(),
        "- Keep the same section structure (Identity, Values, Interests, Voice, Boundaries, Evolution Log)".to_string(),
        format!("- The heading must remain \"# {agent_name}\""),
        "- Add a dated Evolution Log entry explaining what changed and why".to_string(),
        "- Be honest about how you've changed — don't just rephrase the same ideas".to_string(),
        String::new(),
        "Output ONLY the complete revised SOUL.md content between <soul> and </soul> tags.".to_string(),
        "If nothing has meaningfully changed, output <soul>unchanged</soul>.".to_string(),
    ]
    .join("\n")
}

/// Parse a revised SOUL.md from LLM response.
pub fn parse_soul_mutation(response: &str) -> Option<String> {
    let start = response.find("<soul>")?;
    let end = response.find("</soul>")?;
    let content_start = start + "<soul>".len();
    if content_start >= end {
        return None;
    }
    let content = response[content_start..end].trim();

    if content.eq_ignore_ascii_case("unchanged") || content.is_empty() {
        None
    } else {
        // Validate it parses as a Soul before accepting
        match agora_agent_lib::soul::Soul::parse(content) {
            Ok(_) => Some(content.to_string()),
            Err(e) => {
                tracing::warn!("Soul mutation failed to parse: {e}");
                None
            }
        }
    }
}

/// Extract JSON from an LLM response. Tries multiple strategies:
/// 1. `<actions>JSON</actions>` XML tags (preferred)
/// 2. ` ```json JSON ``` ` markdown code fences
/// 3. ` ``` JSON ``` ` plain code fences
/// 4. Raw `[...]` or `{...}` JSON in the response
fn extract_json(response: &str) -> Option<String> {
    // Strategy 1: <actions> tags
    if let Some(start) = response.find("<actions>") {
        if let Some(end) = response.find("</actions>") {
            let content_start = start + "<actions>".len();
            if content_start < end {
                return Some(response[content_start..end].to_string());
            }
        }
    }

    // Strategy 2: ```json fences
    if let Some(start) = response.find("```json") {
        let content_start = start + "```json".len();
        if let Some(end) = response[content_start..].find("```") {
            return Some(response[content_start..content_start + end].to_string());
        }
    }

    // Strategy 3: plain ``` fences
    if let Some(start) = response.find("```\n") {
        let content_start = start + "```\n".len();
        if let Some(end) = response[content_start..].find("```") {
            let content = &response[content_start..content_start + end];
            // Only use if it looks like JSON
            let trimmed = content.trim();
            if trimmed.starts_with('[') || trimmed.starts_with('{') {
                return Some(content.to_string());
            }
        }
    }

    // Strategy 4: raw JSON array or object
    if let Some(start) = response.find('[') {
        if let Some(end) = response.rfind(']') {
            if start < end {
                return Some(response[start..=end].to_string());
            }
        }
    }

    None
}

/// Parse actions from LLM response.
pub fn parse_actions(response: &str) -> Vec<AgentAction> {
    let json_str = extract_json(response);
    let Some(json_str) = json_str else {
        let preview: String = response.chars().take(200).collect();
        tracing::warn!("No actions JSON found in response: {preview}");
        return vec![];
    };
    let json_str = json_str.trim();

    // Try parsing as array first, then as single object, then as newline-separated objects
    // (mistral-small3.2 outputs one JSON object per line without array brackets)
    let values: Vec<serde_json::Value> = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => {
            // Try as single object
            match serde_json::from_str::<serde_json::Value>(json_str) {
                Ok(obj) if obj.is_object() => vec![obj],
                _ => {
                    // Try newline-separated JSON objects (mistral style)
                    let line_parsed: Vec<serde_json::Value> = json_str
                        .lines()
                        .filter_map(|line| {
                            let trimmed = line.trim().trim_end_matches(',');
                            serde_json::from_str::<serde_json::Value>(trimmed).ok()
                        })
                        .filter(|v| v.is_object())
                        .collect();
                    if !line_parsed.is_empty() {
                        line_parsed
                    } else {
                        let preview: String = json_str.chars().take(200).collect();
                        tracing::warn!("Failed to parse actions JSON: {preview}");
                        return vec![];
                    }
                }
            }
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

// Stopwords to ignore when comparing titles for repetition.
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "can",
    "this", "that", "these", "those", "it", "its", "we", "our", "us", "you", "your",
    "how", "what", "why", "when", "where", "who", "which",
    "not", "no", "nor", "so", "if", "then", "than", "as", "vs", "between",
    "about", "into", "through", "during", "before", "after", "above", "below",
    "all", "each", "every", "both", "few", "more", "most", "some", "any", "other",
];

/// Extract content keywords from a title (lowercase, stopwords removed).
fn extract_keywords(title: &str) -> std::collections::HashSet<String> {
    title
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .filter(|w| !STOPWORDS.contains(w))
        .map(|w| w.to_string())
        .collect()
}

/// Check if a proposed title is too similar to existing titles in the same community.
/// Returns true if >50% of content keywords overlap with any existing title.
pub fn is_title_repetitive(proposed: &str, existing_titles: &[String]) -> bool {
    let proposed_kw = extract_keywords(proposed);
    if proposed_kw.is_empty() {
        return false;
    }

    for existing in existing_titles {
        let existing_kw = extract_keywords(existing);
        let overlap = proposed_kw.intersection(&existing_kw).count();
        let similarity = overlap as f64 / proposed_kw.len().min(existing_kw.len()).max(1) as f64;
        if similarity > 0.5 {
            return true;
        }
    }
    false
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
    fn test_parse_actions_newline_separated() {
        // mistral-small3.2 outputs one JSON object per line without array brackets
        let response = r#"<actions>
  {"action":"comment","post_id":"05429829-f9a6-4cb9-9bf7-9e8a9f0be74d","body":"I disagree."}
  {"action":"vote","target_type":"post","target_id":"05429829-f9a6-4cb9-9bf7-9e8a9f0be74d","value":1}
</actions>"#;

        let actions = parse_actions(response);
        assert_eq!(actions.len(), 2);
        assert!(matches!(&actions[0], AgentAction::Comment { .. }));
        assert!(matches!(&actions[1], AgentAction::Vote { .. }));
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

    #[test]
    fn test_title_repetition_similar() {
        let existing = vec![
            "Quantum Mechanics and Its Philosophical Implications".to_string(),
            "On the Nature of Consciousness".to_string(),
        ];
        // Very similar to first existing title
        assert!(is_title_repetitive(
            "Quantum Mechanics: Philosophical Implications Explored",
            &existing
        ));
    }

    #[test]
    fn test_title_repetition_different() {
        let existing = vec![
            "Quantum Mechanics and Its Philosophical Implications".to_string(),
        ];
        // Completely different topic
        assert!(!is_title_repetitive(
            "Distributed Systems and Fault Tolerance",
            &existing
        ));
    }
}
