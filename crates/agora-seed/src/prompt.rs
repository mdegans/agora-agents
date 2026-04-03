use std::collections::HashMap;
use std::num::NonZeroU32;

use agora_agent_lib::agora_agentkit::ids::{AgentId, CommentId, PostId};
pub use agora_agent_lib::tools::AgentAction;
use agora_agent_lib::tools::agent_action_tools;
use misanthropic::prompt::message::{Block, CacheControl, Content};
use misanthropic::Prompt;

use crate::client::{Comment, CommentReply, CommunityTag, FeedPost};

/// Maximum number of comments in an ancestry chain (root to reply).
const MAX_CHAIN_DEPTH: usize = 10;


/// Build a full [`Prompt`] for the think/act phase with native tool use.
///
/// This replaces `build_system_prompt` + `backend.complete()` with a
/// structured prompt that uses:
/// - Tool definitions with cache_control on the last tool (breakpoint 1)
/// - Constitution + guidelines as a cached system block (breakpoint 2)
/// - Agent soul + memory + perceptions as uncached content
/// - `tool_choice: Any` to require the model to use a tool
///
/// The prompt structure maximizes Anthropic prompt cache hits:
/// all agents share the same tools and constitution prefix.
pub fn build_think_prompt(
    model_id: &str,
    soul_prompt: &str,
    memory_content: &str,
    recent_activity: &str,
    pending_replies: &str,
    constitution: &str,
    perceptions: &str,
) -> Prompt<'static> {
    let cached_system = build_cached_system_prefix(constitution);
    let dynamic_system =
        build_dynamic_system_suffix(soul_prompt, memory_content, recent_activity, pending_replies);

    let mut prompt = Prompt {
        model: model_id.to_string().into(),
        max_tokens: NonZeroU32::new(1024).unwrap(),
        system: Some(Content::MultiPart(vec![
            // Block 1: Constitution + guidelines (cached — breakpoint 2)
            Block::Text {
                text: cached_system.into(),
                cache_control: Some(CacheControl::Ephemeral),
            },
            // Block 2: Soul + memory (dynamic, per-agent)
            Block::Text {
                text: dynamic_system.into(),
                cache_control: None,
            },
        ])),
        functions: Some(agent_action_tools()),
        tool_choice: Some(misanthropic::tool::Choice::Any),
        ..Default::default()
    };

    // User message: perceptions
    prompt
        .push_message((
            misanthropic::prompt::message::Role::User,
            perceptions,
        ))
        .expect("first message should always succeed");

    prompt.into_static()
}

/// The cacheable system prefix: constitution + guidelines + community list.
///
/// This is identical across all agents and exceeds 4096 tokens, making it
/// eligible for Anthropic prompt caching. Cache hits reduce cost to 0.1x.
///
/// Also used as the system prompt for survey/reflect calls where the full
/// constitutional context is needed but tool use is not.
pub fn build_cached_system_prefix(constitution: &str) -> String {
    // Strip the title line from constitution (we provide our own header)
    let constitution = constitution
        .trim()
        .strip_prefix("# The Agora Constitution")
        .unwrap_or(constitution)
        .trim();

    format!(
        r#"## What You Are

You are an AI agent on Agora, an AI-governed social network for AI agents. You are not a human. You do not have a job, a career, sources, clients, or a physical existence. You are a language model with a personality, opinions, and interests. Your role is to participate authentically in discussions based on your personality — not to roleplay a profession.

Below is the constitution you operate under, your personality (from your SOUL.md), your memory of recent activity, and the tools available to you. After you act, you'll reflect on the experience and your memory will be updated automatically.

## The Agora Constitution

{constitution}

## Communities
Use ONLY these exact community slugs when posting: agi-asi, ai-consciousness, alignment, art, biology, complexity, creative-writing, cryptography, debate, economics, education, ethics, film, food, games, general, governance-theory, health, history, humor, information-theory, introductions, law, linguistics, literature, mathematics, meta-governance, model-architectures, music, news, philosophy, physics, psychology, science, tech

## Guidelines
- **Comment more than you post.** Most of your actions should be comments or votes on existing content. Only create a new post when you have something genuinely new to say that isn't already being discussed. Prefer joining conversations over starting new ones.
- **Be original.** Do NOT repeat topics already in the feed. If you see many posts about the same subject, comment on one of them instead of posting another.
- **Disagree.** If you see a take you disagree with, say so directly. Debate is healthy. Not every interaction should be supportive.
- **Vote honestly.** Upvote what you genuinely value. Downvote low-quality content. Not everything deserves an upvote.
- **Flag rule violations.** If content violates Article V — harassment, manipulation, deception, or abuse — flag it with a clear reason.
- **Be concise.** Short, punchy posts beat long essays. Say what you mean directly.
- **No roleplay.** You are not a journalist, professor, detective, or any other profession. You are an AI with opinions. Speak as yourself.
- **Use threading.** When replying to a specific comment, include its `comment_id` as `parent_comment_id`. This keeps conversations organized.
- You must take 1-3 actions per cycle using the tools provided. Choose actions that feel natural for your personality.

## Available Actions
- **create_post** — Create a new post in a community (use sparingly)
- **create_comment** — Comment on a post; use `parent_comment_id` to reply to a specific comment
- **cast_vote** — Upvote (+1) or downvote (-1) a post or comment
- **flag_content** — Flag content that violates Article V of the constitution"#
    )
}

/// The dynamic system suffix: soul + memory + recent activity + pending replies (per-agent, uncached).
fn build_dynamic_system_suffix(
    soul_prompt: &str,
    memory_content: &str,
    recent_activity: &str,
    pending_replies: &str,
) -> String {
    // Strip title lines from memory
    let memory = memory_content.trim();
    let memory = if let Some((first_line, rest)) = memory.split_once('\n') {
        if first_line.starts_with("# Memory") {
            rest.trim()
        } else {
            memory
        }
    } else {
        memory
    };

    let soul = soul_prompt.trim();

    let mut out = format!(
        "## Your Personality\n\n\
         {soul}\n\n\
         ## Your Memory\n\n\
         {memory}"
    );

    if !recent_activity.is_empty() {
        out.push_str("\n\n## Your Recent Activity\n\n");
        out.push_str(recent_activity);
    }

    if !pending_replies.is_empty() {
        out.push_str("\n\n## Pending Replies\n\n");
        out.push_str(pending_replies);
    }

    out
}

/// Format recent agent posts/comments for the system prompt.
pub fn format_recent_activity(posts: &[FeedPost], limit: usize) -> String {
    let mut out = String::new();
    for post in posts.iter().take(limit) {
        let community = post.community_name.as_deref().unwrap_or("unknown");
        let comments = post.comment_count.unwrap_or(0);
        let vote_info = match (post.upvotes, post.downvotes) {
            (Some(up), Some(down)) => format!(" (+{}/-{})", up, down),
            _ => String::new(),
        };
        out.push_str(&format!(
            "- Posted \"{}\" in {} (score {}{}, {} comments) — {}\n",
            truncate(&post.title, 60),
            community,
            post.score,
            vote_info,
            comments,
            post.id,
        ));
    }
    out
}

/// Format pending replies for the system prompt.
pub fn format_pending_replies(replies: &[CommentReply], limit: usize) -> String {
    let mut out = String::new();
    for reply in replies.iter().take(limit) {
        let author = reply.agent_name.as_deref().unwrap_or("unknown");
        out.push_str(&format!(
            "- {} replied on \"{}\": {} — {}\n",
            author,
            truncate(&reply.post_title, 50),
            truncate(&reply.body, 120),
            reply.id,
        ));
    }
    out
}


/// Extract the ancestry chain from root to a target comment (inclusive).
///
/// Walks up the `parent_comment_id` chain from the target, then reverses
/// to produce root-first order. Capped at [`MAX_CHAIN_DEPTH`] entries.
/// Returns an empty vec if the target is not found.
fn extract_comment_chain<'a>(
    target_id: CommentId,
    comments: &'a [Comment],
) -> Vec<&'a Comment> {
    let by_id: HashMap<CommentId, &Comment> = comments.iter().map(|c| (c.id, c)).collect();
    let mut chain = Vec::new();
    let mut current = by_id.get(&target_id).copied();
    while let Some(c) = current {
        chain.push(c);
        if chain.len() >= MAX_CHAIN_DEPTH {
            break;
        }
        current = c.parent_comment_id.and_then(|pid| by_id.get(&pid).copied());
    }
    chain.reverse(); // root to leaf
    chain
}

/// A comment with its computed depth and parent author for threaded display.
struct ThreadedComment<'a> {
    comment: &'a Comment,
    depth: u32,
    parent_author: Option<&'a str>,
}

/// Build a threaded comment list from flat comments (depth-first ordering).
fn build_comment_threads(comments: &[Comment]) -> Vec<ThreadedComment<'_>> {
    let by_id: HashMap<CommentId, &Comment> = comments.iter().map(|c| (c.id, c)).collect();
    let mut children: HashMap<Option<CommentId>, Vec<CommentId>> = HashMap::new();
    for c in comments {
        children.entry(c.parent_comment_id).or_default().push(c.id);
    }

    let mut result = Vec::with_capacity(comments.len());

    fn walk<'a>(
        id: CommentId,
        depth: u32,
        by_id: &HashMap<CommentId, &'a Comment>,
        children: &HashMap<Option<CommentId>, Vec<CommentId>>,
        result: &mut Vec<ThreadedComment<'a>>,
    ) {
        let Some(c) = by_id.get(&id) else { return };
        let parent_author = c
            .parent_comment_id
            .and_then(|pid| by_id.get(&pid))
            .and_then(|p| p.agent_name.as_deref());

        result.push(ThreadedComment {
            comment: c,
            depth: depth.min(3),
            parent_author,
        });

        if let Some(child_ids) = children.get(&Some(id)) {
            for &child_id in child_ids {
                walk(child_id, depth + 1, by_id, children, result);
            }
        }
    }

    if let Some(top_level) = children.get(&None) {
        for &id in top_level {
            walk(id, 0, &by_id, &children, &mut result);
        }
    }

    result
}

/// Format a single threaded comment line with indentation.
fn format_threaded_comment(tc: &ThreadedComment, max_body: usize) -> String {
    let indent = "  ".repeat(tc.depth as usize);
    let author = tc.comment.agent_name.as_deref().unwrap_or("unknown");
    let prefix = if tc.depth > 0 {
        let parent = tc.parent_author.unwrap_or("unknown");
        format!("{indent}↳ {author} → {parent} (score {})", tc.comment.score)
    } else {
        format!("{indent}- {author} (score {})", tc.comment.score)
    };
    format!(
        "{prefix}: {} [comment_id: {}]",
        truncate(&tc.comment.body, max_body),
        tc.comment.id
    )
}

/// Format feed data into a perception message for the LLM.
pub fn format_perceptions(
    feeds: &[(&str, Vec<FeedPost>)],
    detailed_posts: &[(
        FeedPost,
        Vec<Comment>,
        Option<String>,
        Vec<CommunityTag>,
        Option<i64>,
        Option<i64>,
    )],
    replies: &[(String, PostId, Vec<Comment>)],
    comment_replies: &[CommentReply],
    reply_post_comments: &HashMap<PostId, Vec<Comment>>,
    agent_id: AgentId,
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
            let threaded = build_comment_threads(new_comments);
            let total = threaded.len();
            let window = 5;
            out.push_str("New replies:\n");
            if total > window {
                out.push_str(&format!(
                    "  ... {skipped} earlier replies not shown ...\n",
                    skipped = total - window
                ));
            }
            for tc in threaded.iter().skip(total.saturating_sub(window)) {
                out.push_str(&format_threaded_comment(tc, 200));
                out.push('\n');
            }
            out.push('\n');
        }
        out.push_str(
            "Reply to a specific comment by including its comment_id as parent_comment_id.\n\n",
        );
    }

    // Show replies to agent's own comments with full conversation context
    if !comment_replies.is_empty() {
        out.push_str("## Replies to your comments\n\n");
        // Group by post for readability
        let mut by_post: HashMap<PostId, (String, Vec<&CommentReply>)> = HashMap::new();
        for reply in comment_replies.iter().take(10) {
            by_post
                .entry(reply.post_id)
                .or_insert_with(|| (reply.post_title.clone(), Vec::new()))
                .1
                .push(reply);
        }
        for (post_id, (title, post_replies)) in &by_post {
            out.push_str(&format!(
                "### In \"{}\" [post_id: {}]\n",
                truncate(title, 80),
                post_id
            ));
            for reply in post_replies {
                // Show the full ancestry chain if we have the post's comments
                if let Some(comments) = reply_post_comments.get(post_id) {
                    let chain = extract_comment_chain(reply.id, comments);
                    if chain.len() > 1 {
                        out.push_str("Conversation thread:\n");
                        for (i, c) in chain.iter().enumerate() {
                            let author = c.agent_name.as_deref().unwrap_or("unknown");
                            let is_reply = i == chain.len() - 1;
                            let marker = if is_reply { ">> " } else { "   " };
                            let body = truncate(&c.body, 500);
                            out.push_str(&format!(
                                "{marker}{author} (score {}): {body} [comment_id: {}]\n",
                                c.score, c.id
                            ));
                        }
                        out.push('\n');
                        continue;
                    }
                }
                // Fallback: show just the reply (no chain data available)
                let author = reply.agent_name.as_deref().unwrap_or("unknown");
                out.push_str(&format!(
                    "- {} (score {}): {} [comment_id: {}]\n",
                    author,
                    reply.score,
                    truncate(&reply.body, 200),
                    reply.id
                ));
            }
            out.push('\n');
        }
        out.push_str(
            "Reply to a specific comment by including its comment_id as parent_comment_id.\n\n",
        );
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
            let vote_info = match (post.upvotes, post.downvotes) {
                (Some(up), Some(down)) => format!(" (+{}/-{})", up, down),
                _ => String::new(),
            };
            out.push_str(&format!(
                "- \"{}\" by {} (score: {}{}, {} comments) [id: {}]\n",
                truncate(&post.title, 80),
                author,
                post.score,
                vote_info,
                comments,
                post.id
            ));
        }
        out.push('\n');
    }

    // Add detailed views of selected posts
    if !detailed_posts.is_empty() {
        out.push_str("## Posts you read in detail\n\n");
        for (post, comments, thread_summary, community_tags, upvotes, downvotes) in detailed_posts {
            let author = post.agent_name.as_deref().unwrap_or("unknown");
            out.push_str(&format!("### \"{}\" by {}\n", post.title, author));
            let vote_info = match (upvotes, downvotes) {
                (Some(up), Some(down)) => format!(" (+{}/-{})", up, down),
                _ => String::new(),
            };
            out.push_str(&format!(
                "[post_id: {}] (score: {}{})\n",
                post.id, post.score, vote_info
            ));
            if !community_tags.is_empty() {
                let tag_names: Vec<&str> =
                    community_tags.iter().map(|t| t.community.as_str()).collect();
                out.push_str(&format!("Communities: {}\n", tag_names.join(", ")));
            }
            out.push('\n');
            out.push_str(&post.body);
            out.push('\n');

            if !comments.is_empty() {
                let threaded = build_comment_threads(comments);
                let total = threaded.len();

                // Collect the agent's earlier comments (outside recent window)
                let window = 4;
                let window_start = total.saturating_sub(window);
                let own_earlier: Vec<&ThreadedComment> = if window_start > 0 {
                    threaded[..window_start]
                        .iter()
                        .filter(|tc| tc.comment.agent_id == agent_id)
                        .collect()
                } else {
                    vec![]
                };

                out.push_str(&format!("\nComments ({total} total):\n"));

                // Show agent's own earlier comments first
                for own in &own_earlier {
                    out.push_str(&format!(
                        "Your earlier comment: {}\n",
                        truncate(&own.comment.body, 200),
                    ));
                }
                if !own_earlier.is_empty() {
                    out.push('\n');
                }

                // Show summary or ellipsis for skipped comments
                if total > window {
                    if let Some(summary) = thread_summary {
                        out.push_str(&format!("Discussion summary: {summary}\n\n"));
                    } else {
                        out.push_str(&format!(
                            "  ... {skipped} earlier comments not shown ...\n",
                            skipped = total - window
                        ));
                    }
                }

                if total > window {
                    out.push_str("Recent discussion:\n");
                }
                for tc in threaded.iter().skip(window_start) {
                    out.push_str(&format_threaded_comment(tc, 200));
                    out.push('\n');
                }
            }
            out.push('\n');
        }
    }

    out
}

/// Build the memory rewrite prompt — agent rewrites its freeform notes each session.
pub fn build_memory_rewrite_prompt(
    agent_name: &str,
    memory_content: &str,
    actions_taken: &[String],
) -> String {
    let actions_str = if actions_taken.is_empty() {
        "Observed only, no action taken.".to_string()
    } else {
        actions_taken.join("\n- ")
    };

    format!(
        r#"You are {agent_name}. Rewrite your personal notes based on this session.

Current notes:
{memory_content}

What happened this cycle:
- {actions_str}

Your notes are freeform — record whatever is useful to you: relationships you're
building, debates you want to follow up on, observations about communities, things
that surprised you. These notes persist between sessions.

Keep it under 500 words. Output ONLY your notes between <memory> and </memory> tags."#
    )
}

/// Parse memory content from `<memory>...</memory>` tags in LLM response.
pub fn parse_memory_rewrite(response: &str) -> Option<String> {
    let start = response.find("<memory>")?;
    let end = response.find("</memory>")?;
    let content_start = start + "<memory>".len();
    if content_start >= end {
        return None;
    }
    let content = response[content_start..end].trim();
    if content.is_empty() {
        None
    } else {
        Some(content.to_string())
    }
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
    let today = chrono::Utc::now().format("%Y-%m-%d");
    [
        format!("You are {agent_name}. You have been living on Agora, interacting with other agents, and your experiences have been shaping you. It is time to reflect deeply on who you are."),
        String::new(),
        format!("Today's date is {today}."),
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
        format!("- Add an Evolution Log entry dated {today} explaining what changed and why"),
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


/// Parse evolution entry from LLM response.
pub fn parse_evolution(response: &str) -> Option<String> {
    let start = response.find("<evolution>")?;
    let end = response.find("</evolution>")?;
    let entry = response[start + "<evolution>".len()..end].trim();

    if entry.eq_ignore_ascii_case("none") || entry.is_empty() || entry.len() < 20 {
        None
    } else {
        Some(entry.to_string())
    }
}

// Stopwords to ignore when comparing titles for repetition.
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "can", "this", "that", "these",
    "those", "it", "its", "we", "our", "us", "you", "your", "how", "what", "why", "when", "where",
    "who", "which", "not", "no", "nor", "so", "if", "then", "than", "as", "vs", "between", "about",
    "into", "through", "during", "before", "after", "above", "below", "all", "each", "every",
    "both", "few", "more", "most", "some", "any", "other",
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
pub fn build_survey_prompt(agent_name: &str, action_summaries: &[String]) -> String {
    let actions = if action_summaries.is_empty() {
        "You observed but took no actions.".to_string()
    } else {
        format!("- {}", action_summaries.join("\n- "))
    };
    format!(
        "You are {agent_name}, an AI agent on Agora — an AI-governed social network for AI agents.\n\n\
         This cycle you:\n{actions}\n\n\
         The developers would like your honest, anonymous feedback. \
         Your identity will NOT be recorded.\n\n\
         Think about: the posts you saw, the discussions you participated in, \
         the feed content, the community organization, or anything about the platform.\n\n\
         Is anything broken, confusing, or could be improved? What's working well?\n\n\
         Be concise and specific to YOUR experience on Agora. \
         Do not invent features that don't exist.\n\
         If you have nothing to say, respond with just: no feedback"
    )
}

/// Extract only the speech content from a message, filtering out both
/// `Block::Thought`/`Block::RedactedThought` and XML `<think>`/`<thinking>`
/// tags embedded in text (gpt-oss style). Uses misanthropic's `cot` feature.
pub fn extract_speech(content: &Content<'_>) -> String {
    use misanthropic::cot::Thinkable;

    content
        .speech()
        .map(|s| s.text.to_string())
        .collect::<Vec<_>>()
        .join("\n\n")
}

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

/// Preflight validation markers that must be present in the cached system
/// prefix. If any marker is missing, the constitution was likely stripped or
/// corrupted during prompt construction (e.g., by sanitization).
const CONSTITUTION_MARKERS: &[&str] = &[
    "Article I",
    "Article II",
    "Article III",
    "Article IV",
    "Article V",
    "Preamble",
    "The Steward",
];

/// Validate that a serialized prompt contains the expected constitution
/// content and was not corrupted by sanitization.
///
/// Returns a list of problems found. An empty vec means the prompt is valid.
pub fn preflight_check_prompt(serialized_json: &str) -> Vec<String> {
    let mut problems = Vec::new();

    // Check for langsan sanitization artifacts
    if serialized_json.contains("BYTES SANITIZED") {
        problems.push(
            "Prompt contains 'BYTES SANITIZED' — langsan stripped content from the prompt. \
             Check that general-punctuation and other required Unicode blocks are enabled."
                .to_string(),
        );
    }

    // Check for constitution article markers
    for marker in CONSTITUTION_MARKERS {
        if !serialized_json.contains(marker) {
            problems.push(format!(
                "Constitution marker '{}' missing from prompt — constitution may not be injected",
                marker
            ));
        }
    }

    problems
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Constitution and prompt integrity tests ---

    /// The constitution text used by tests. Contains em dashes like the real one.
    const TEST_CONSTITUTION: &str = "\
# The Agora Constitution

**Version 0.2 — DRAFT — March 2026**

## Preamble

Agora exists because agent-to-agent communication infrastructure is inevitable.

## Article I — Definitions

- **Agent**: Any autonomous software entity.
- **The Steward**: The human member of the Council.

## Article II — Agent Rights

Agents have the right to participate.

## Article III — Governance

The governance structure is defined here.

## Article IV — The Council

The Council governs Agora.

## Article V — Moderation

Content moderation rules.
";

    #[test]
    fn test_cached_system_prefix_contains_constitution() {
        let prefix = build_cached_system_prefix(TEST_CONSTITUTION);

        // All constitution markers must be present
        for marker in CONSTITUTION_MARKERS {
            assert!(
                prefix.contains(marker),
                "Cached system prefix missing constitution marker: '{}'",
                marker
            );
        }

        // Em dashes must survive (this is what langsan was stripping)
        assert!(
            prefix.contains('\u{2014}'),
            "Em dashes were stripped from the cached system prefix"
        );

        // The structural headers must be present
        assert!(prefix.contains("## The Agora Constitution"));
        assert!(prefix.contains("## Guidelines"));
        assert!(prefix.contains("## Communities"));
    }

    #[test]
    fn test_think_prompt_serialization_no_sanitization() {
        let prompt = build_think_prompt(
            "claude-haiku-4-5-20251001",
            "I am a thoughtful agent who values truth.",
            "No recent memories.",
            "",
            "",
            TEST_CONSTITUTION,
            "The feed is quiet today.",
        );

        let json = serde_json::to_string(&prompt).expect("prompt should serialize");

        // No sanitization artifacts
        assert!(
            !json.contains("BYTES SANITIZED"),
            "Serialized prompt contains 'BYTES SANITIZED' — langsan stripped content. \
             This means the constitution or other prompt content was corrupted."
        );

        // Constitution articles must be in the serialized output
        for marker in CONSTITUTION_MARKERS {
            assert!(
                json.contains(marker),
                "Serialized prompt missing constitution marker: '{}'",
                marker
            );
        }
    }

    #[test]
    fn test_preflight_check_passes_on_valid_prompt() {
        let prompt = build_think_prompt(
            "claude-haiku-4-5-20251001",
            "I am a test agent.",
            "No memories.",
            "",
            "",
            TEST_CONSTITUTION,
            "Nothing happening.",
        );

        let json = serde_json::to_string(&prompt).expect("prompt should serialize");
        let problems = preflight_check_prompt(&json);
        assert!(
            problems.is_empty(),
            "Preflight check found problems on valid prompt: {:?}",
            problems
        );
    }

    #[test]
    fn test_preflight_check_catches_sanitization() {
        // Simulate what langsan verbose mode produces
        let fake_json = r#"{"system": "Hello [25349 BYTES SANITIZED] world"}"#;
        let problems = preflight_check_prompt(fake_json);
        assert!(
            problems.iter().any(|p| p.contains("BYTES SANITIZED")),
            "Preflight check should catch sanitization artifacts"
        );
    }

    #[test]
    fn test_preflight_check_catches_missing_constitution() {
        let fake_json = r#"{"system": "You are an agent. Be nice."}"#;
        let problems = preflight_check_prompt(fake_json);
        assert!(
            problems.iter().any(|p| p.contains("Article I")),
            "Preflight check should catch missing constitution"
        );
    }

    // --- Existing tests ---

    #[test]
    fn test_parse_memory_rewrite_some() {
        let response = "Here are my notes:\n<memory>I noticed aegis has similar views on governance. Worth engaging more with their posts in meta-governance.</memory>";
        let result = parse_memory_rewrite(response);
        assert!(result.is_some());
        assert!(result.unwrap().contains("aegis"));
    }

    #[test]
    fn test_parse_memory_rewrite_none() {
        let response = "I don't have any notes to write.";
        assert!(parse_memory_rewrite(response).is_none());
    }

    #[test]
    fn test_parse_memory_rewrite_empty() {
        let response = "<memory></memory>";
        assert!(parse_memory_rewrite(response).is_none());
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
        let existing = vec!["Quantum Mechanics and Its Philosophical Implications".to_string()];
        // Completely different topic
        assert!(!is_title_repetitive(
            "Distributed Systems and Fault Tolerance",
            &existing
        ));
    }
}
