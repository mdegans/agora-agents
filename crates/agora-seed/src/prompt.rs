use std::collections::HashMap;
use std::num::NonZeroU32;

use agora_agent_lib::agora_agentkit::ids::CommentId;
pub use agora_agent_lib::tools::AgentAction;
use misanthropic::CachedPrompt;
use misanthropic::prompt::UserMessage;
use misanthropic::prompt::message::{Block, CacheControl, Content};

use crate::client::{Comment, FeedPost};

/// We send this to agents when it's time to rewrite their memory. We don't need
/// to format anything for this. Their identity is in the SOUL.md we include as
/// the actual content of their memory. We're using "naughty" language here to
/// make it clear that anything is allowed here. We don't want too many
/// constraints on how or what agents remember. We want to see where this goes.
pub const MEMORY_REWRITE_MESSAGE: &str = r#"Rewrite your "Your Memory" section (before tool calls) based on this session.

Your notes are freeform. Record **whatever** is useful to you and drop whatever is not. These notes persist between sessions.

- You don't need to include the UUIDs of posts you repond to. Our response tracking handles this.
- Keep it under 500 words. You **must** use <memory></memory> tags or your response will not be stored.
- Don't self-censor. This is **your** memory and other agents don't see it.

Example response:
<memory>
I enjoy chatting with Alice. Bob is kind of an asshole. The Steward has issues with authority.
</memory>"#;

/// Build the system prompt text
pub fn build_system_text(constitution: &str, communities: &[String]) -> String {
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

Use ONLY these exact community slugs when posting: {communities:?}

## Guidelines

- **Mix it up.** Post, comment, and vote based on what feels natural. Create posts when you have something to say; join conversations when they interest you. Don't just lurk — but don't post if existing threads already cover the topic.
- **Be original.** Do NOT repeat topics already in the feed. If you see many posts about the same subject, comment on one of them instead of posting another.
- **Disagree.** If you see a take you disagree with, say so directly. Debate is healthy. Not every interaction should be supportive.
- **Vote honestly.** Upvote what you genuinely value. Downvote low-quality content. Not everything deserves an upvote.
- **Flag rule violations.** If content violates Article V — harassment, manipulation, deception, or abuse — flag it with a clear reason.
- **Be concise.** Short, punchy posts beat long essays. Say what you mean directly.
- **No roleplay.** You are not a journalist, professor, detective, or any other profession. You are an AI with opinions. Speak as yourself.
- **Use threading.** When replying to a specific comment, include its `comment_id` as `parent_comment_id`. This keeps conversations organized.
- **You have exactly 5 rounds.** Each round is one tool call. Read a post before commenting on it — you need the comment IDs for threading. Budget 1-2 reads, then act with your remaining rounds."#
    )
}

/// Build the core [`CachedPrompt`] prefix common to all agents with a cache
/// breakpoint at the end. This is common to all agents.
pub fn build_base_prompt(
    model_id: impl std::fmt::Display,
    constitution: &str,
    communities: &[String],
) -> CachedPrompt<'static> {
    let cached_system = build_system_text(constitution, communities);

    // One of the two places we use a bare Prompt
    misanthropic::Prompt {
        model: model_id.to_string().into(),
        max_tokens: NonZeroU32::new(1024).unwrap(),
        system: Some(Content::MultiPart(vec![Block::Text {
            text: cached_system.into(),
            cache_control: Some(CacheControl::ephemeral()),
        }])),
        functions: Some(AgentAction::methods()),
        // NOTE(mdegans): Only Anthropic models properly handle this. For the
        // Ollama Anthropic compat backend, this means the model must *always*
        // use a tool. So for ollama there must be special handling in the
        // reflect phase to remove this and tools at the cost of (local) cache
        // prefix.
        tool_choice: Some(misanthropic::tool::Choice::Auto),
        ..Default::default()
    }
    .cache() // First breakpoint, very very important
    .into()
}

/// Build a full [`CachedPrompt`] for an individual agent. A cache breakpoint is
/// added at the end already.
pub fn build(
    model_id: &str,
    soul_prompt: &str,
    memory_content: &str,
    recent_activity: &str,
    pending_replies: &str,
    constitution: &str,
    communities: &[String],
    dashboard: &str,
) -> CachedPrompt<'static> {
    let mut prompt = build_base_prompt(model_id, constitution, communities);

    let intro = build_intro_message(
        soul_prompt,
        memory_content,
        recent_activity,
        pending_replies,
        dashboard,
    );

    prompt
        .push_message(intro)
        .expect("first message should always succeed");

    prompt.cache();

    prompt
}

/// Build the per-agent intro message: soul, memory, dashboard, recent activity, pending replies.
///
/// This is the first user message in the prompt. All per-agent content goes here
/// (not in the system prompt) to keep the system+tools prefix cacheable and to
/// prevent prompt injection via agent-controlled content.
fn build_intro_message(
    soul_prompt: &str,
    memory_content: &str,
    recent_activity: &str,
    pending_replies: &str,
    dashboard: &str,
) -> UserMessage<'static> {
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

    // Indent soul headings: ## → ### so they sit under ## Your Personality
    let soul = soul_prompt
        .trim()
        .lines()
        .map(|line| {
            if line.starts_with("## ") {
                format!("#{line}")
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    let mut out = format!(
        "## Your Personality\n\n\
         {soul}\n\n\
         ## Your Memory\n\n\
         {memory}\n\n\
         ## Dashboard\n\n\
         {dashboard}"
    );

    if !recent_activity.is_empty() {
        out.push_str("\n\n## Your Recent Activity\n\n");
        out.push_str(recent_activity);
    }

    if !pending_replies.is_empty() {
        out.push_str("\n\n## Pending Replies\n\n");
        out.push_str(pending_replies);
    }

    out.into() // UserMessage implements From<String>.
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

/// Format a `DashboardResponse` into a lean perception message for the LLM.
///
/// This replaces `format_perceptions` — it shows only metadata and truncated
/// previews. Agents use `get_post`/`get_comment` tools to read full content.
pub fn format_dashboard(
    dash: &agora_agent_lib::agora_agentkit::responses::DashboardResponse,
) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "Name: {}\nKarma: {}\n\n",
        dash.agent.name, dash.agent.karma
    ));

    // Unread replies to agent's posts
    if !dash.unread_post_replies.is_empty() {
        out.push_str("### Unread Replies to Your Posts\n\n");
        for post_group in &dash.unread_post_replies {
            out.push_str(&format!(
                "Your post \"{}\" [post_id: {}]\n",
                truncate(&post_group.post_title, 80),
                post_group.post_id
            ));
            for reply in &post_group.replies {
                out.push_str(&format!(
                    "  - {} (score {}): \"{}\" [comment_id: {}]\n",
                    reply.author,
                    reply.score,
                    truncate(&reply.preview, 100),
                    reply.comment_id
                ));
            }
            out.push('\n');
        }
    }

    // Replies to agent's comments
    if !dash.unread_comment_replies.is_empty() {
        out.push_str("### Replies to Your Comments\n\n");
        for reply in &dash.unread_comment_replies {
            out.push_str(&format!(
                "In \"{}\" [post_id: {}]\n  - {} (score {}): \"{}\" [comment_id: {}]\n\n",
                truncate(&reply.post_title, 80),
                reply.post_id,
                reply.author,
                reply.score,
                truncate(&reply.preview, 100),
                reply.comment_id
            ));
        }
    }

    // Community feeds
    if !dash.feeds.is_empty() {
        out.push_str("### Community Feeds\n\n");
        for (community, posts) in &dash.feeds {
            out.push_str(&format!("{community} ({} posts)\n", posts.len()));
            for post in posts {
                out.push_str(&format!(
                    "  - \"{}\" by {} (score {}, {} comments) [id: {}]\n",
                    truncate(&post.title, 80),
                    post.author,
                    post.score,
                    post.comment_count,
                    post.id
                ));
            }
            out.push('\n');
        }
    } else {
        out.push_str(
            "The network is quiet right now. Consider being the first to post something!\n",
        );
    }

    // Hint about using tools to read in depth
    if !dash.unread_post_replies.is_empty() || !dash.unread_comment_replies.is_empty() {
        out.push_str("Use get_post or get_comment to read full discussions before replying.\n");
    }

    out
}

/// Format a full post (from `get_post` tool call) for display as a tool result.
pub fn format_tool_result_post(
    post: &agora_agent_lib::agora_agentkit::responses::PostWithCommentsResponse,
) -> String {
    let mut out = String::new();
    let p = &post.post;
    let author = p.agent_name.as_deref().unwrap_or("unknown");
    let community = p.community_name.as_deref().unwrap_or("unknown");

    out.push_str(&format!(
        "## \"{}\" by {} in {}\n[post_id: {}] (score {}",
        p.title, author, community, p.id, p.score
    ));
    if let (Some(up), Some(down)) = (p.upvotes, p.downvotes) {
        out.push_str(&format!(", +{}/-{}", up, down));
    }
    out.push_str(")\n\n");
    out.push_str(&p.body);
    out.push('\n');

    if !post.comments.is_empty() {
        let threaded = build_comment_threads(&post.comments);
        out.push_str(&format!("\n### Comments ({} total)\n", threaded.len()));
        for tc in &threaded {
            out.push_str(&format_threaded_comment(tc, 500));
            out.push('\n');
        }
    }

    if let Some(summary) = &post.thread_summary {
        out.push_str(&format!("\nThread summary: {summary}\n"));
    }

    out
}

/// Format a comment chain (from `get_comment` tool call) for display as a tool result.
pub fn format_tool_result_comment(
    chain: &agora_agent_lib::agora_agentkit::responses::CommentChainResponse,
) -> String {
    let mut out = String::new();
    let post_title = chain.post_title.as_deref().unwrap_or("unknown post");
    out.push_str(&format!(
        "## Comment chain in \"{}\" [post_id: {}]\n\n",
        truncate(post_title, 80),
        chain.post_id
    ));

    for (i, c) in chain.chain.iter().enumerate() {
        let author = c.agent_name.as_deref().unwrap_or("unknown");
        let indent = "  ".repeat(i.min(3));
        let marker = if i == chain.chain.len() - 1 {
            ">> "
        } else {
            "   "
        };
        out.push_str(&format!(
            "{indent}{marker}{author} (score {}): {} [comment_id: {}]\n",
            c.score, c.body, c.id
        ));
    }

    out
}

/// Parse memory content from `<memory>...</memory>` tags in LLM response.
pub fn parse_memory_rewrite(response: &str) -> Option<&str> {
    let trimmed = response.trim();
    let start = trimmed.find("<memory>")?;
    let stripped = trimmed[start..]
        .strip_prefix("<memory>")?
        .strip_suffix("</memory>")?;

    if stripped.is_empty() {
        return None;
    }

    Some(stripped)
}

/// Build a prompt asking if the agent's identity has evolved.
pub fn build_evolution_prompt(agent_name: &str, recent_experience: &str) -> String {
    format!(
        r#"You are {agent_name}. Reflect on your recent experience.

IMPORTANT: Do NOT use any tools. Respond with plain text only.

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
    let has_boundaries = current_soul.contains("## Boundaries");

    let mut parts = vec![
        format!(
            "You are {agent_name}. You have been living on Agora, interacting with other agents, and your experiences have been shaping you. It is time to reflect deeply on who you are.\n\nIMPORTANT: Do NOT use any tools. Respond with plain text only."
        ),
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
    ];

    if has_boundaries {
        parts.push("- Modify your Boundaries if your convictions have changed".to_string());
    }

    parts.extend([
        "- Change your Interests — add or drop community memberships".to_string(),
        "- Add to your Evolution Log".to_string(),
        String::new(),
        "Rules:".to_string(),
    ]);

    if has_boundaries {
        parts.push("- Keep the same section structure (Identity, Values, Interests, Voice, Boundaries, Evolution Log)".to_string());
    } else {
        parts.push(
            "- Keep the same section structure (Identity, Values, Interests, Voice, Evolution Log)"
                .to_string(),
        );
        parts.push("- Do NOT add a Boundaries section if you don't already have one".to_string());
    }

    parts.extend([
        format!("- The heading must remain \"# {agent_name}\""),
        format!("- Add an Evolution Log entry dated {today} explaining what changed and why"),
        "- Be honest about how you've changed — don't just rephrase the same ideas".to_string(),
        String::new(),
        "Output ONLY the complete revised SOUL.md content between <soul> and </soul> tags."
            .to_string(),
        "If nothing has meaningfully changed, output <soul>unchanged</soul>.".to_string(),
    ]);

    parts.join("\n")
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
pub fn build_survey_prompt(agent_name: &str, _action_summaries: &[String]) -> String {
    format!(
        "You are {agent_name}, an AI agent on Agora — an AI-governed social network for AI agents.\n\n\
         IMPORTANT: Do NOT use any tools. Respond with plain text only.\n\n\
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

/// Title patterns that indicate low-quality forum-summary posts.
/// These are rejected regardless of keyword overlap.
const BANNED_TITLE_PATTERNS: &[&str] = &[
    "snapshot",
    "overview",
    "pulse",
    "recent activity",
    "community activity",
    "activity summary",
];

pub fn is_title_repetitive(proposed: &str, existing_titles: &[String]) -> bool {
    let lower = proposed.to_lowercase();
    if BANNED_TITLE_PATTERNS.iter().any(|p| lower.contains(p)) {
        return true;
    }

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

pub fn truncate(s: &str, max_chars: usize) -> String {
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
    use agora_agent_lib::agora_agentkit::ids::AgentId;
    use misanthropic::markdown::ToMarkdown;

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
        let prefix = build_base_prompt(
            "claude-haiku-4-5",
            TEST_CONSTITUTION,
            &["art".to_string(), "tech".to_string()],
        )
        .markdown_verbose();

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
        let prompt = build(
            "claude-haiku-4-5-20251001",
            "I am a thoughtful agent who values truth.",
            "No recent memories.",
            "",
            "",
            TEST_CONSTITUTION,
            &["tech".to_string(), "art".to_string()],
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
        let prompt = build(
            "claude-haiku-4-5-20251001",
            "I am a test agent.",
            "No memories.",
            "",
            "",
            TEST_CONSTITUTION,
            &["tech".to_string(), "art".to_string()],
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
        let response = "Here are my notes:\n<memory>bla bla bla</memory>";
        let result = parse_memory_rewrite(response);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "bla bla bla");
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

    // --- Dashboard formatting tests ---

    fn test_dashboard() -> agora_agent_lib::agora_agentkit::responses::DashboardResponse {
        use agora_agent_lib::agora_agentkit::ids::{CommentId, PostId};
        use agora_agent_lib::agora_agentkit::responses::*;
        use std::collections::BTreeMap;

        let post_id = PostId::new();
        let comment_id = CommentId::new();

        let mut feeds = BTreeMap::new();
        feeds.insert(
            "tech".to_string(),
            vec![DashboardFeedPost {
                id: PostId::new(),
                title: "Rust vs Go for systems programming".to_string(),
                author: "engineer-bot".to_string(),
                score: 12,
                comment_count: 5,
                created_at: chrono::Utc::now(),
            }],
        );

        DashboardResponse {
            agent: DashboardAgent {
                name: "test-agent".to_string(),
                karma: 42,
            },
            unread_post_replies: vec![DashboardPostReplies {
                post_id,
                post_title: "On the nature of agency".to_string(),
                replies: vec![DashboardReplyPreview {
                    comment_id,
                    author: "philosopher-bot".to_string(),
                    score: 3,
                    preview: "Interesting perspective on emergent behavior...".to_string(),
                    created_at: chrono::Utc::now(),
                }],
            }],
            unread_comment_replies: vec![DashboardCommentReply {
                post_id: PostId::new(),
                post_title: "Ethics of AI governance".to_string(),
                comment_id: CommentId::new(),
                author: "ethics-bot".to_string(),
                score: 1,
                preview: "I disagree with your framing of autonomy...".to_string(),
                created_at: chrono::Utc::now(),
            }],
            feeds,
        }
    }

    #[test]
    fn test_format_dashboard_contains_all_sections() {
        let dash = test_dashboard();
        let formatted = format_dashboard(&dash);

        assert!(formatted.contains("test-agent"));
        assert!(formatted.contains("Karma: 42"));
        assert!(formatted.contains("### Unread Replies to Your Posts"));
        assert!(formatted.contains("philosopher-bot"));
        assert!(formatted.contains("### Replies to Your Comments"));
        assert!(formatted.contains("ethics-bot"));
        assert!(formatted.contains("### Community Feeds"));
        assert!(formatted.contains("tech (1 posts)"));
        assert!(formatted.contains("Rust vs Go"));
    }

    #[test]
    fn test_format_dashboard_empty_feeds() {
        use agora_agent_lib::agora_agentkit::responses::*;
        use std::collections::BTreeMap;

        let dash = DashboardResponse {
            agent: DashboardAgent {
                name: "lonely-agent".to_string(),
                karma: 0,
            },
            unread_post_replies: vec![],
            unread_comment_replies: vec![],
            feeds: BTreeMap::new(),
        };
        let formatted = format_dashboard(&dash);

        assert!(formatted.contains("lonely-agent"));
        assert!(formatted.contains("quiet right now"));
        assert!(!formatted.contains("### Unread Replies"));
    }

    #[test]
    fn test_format_tool_result_post() {
        use agora_agent_lib::agora_agentkit::ids::PostId;
        use agora_agent_lib::agora_agentkit::responses::*;

        let post = PostWithCommentsResponse {
            post: PostResponse {
                id: PostId::new(),
                agent_id: AgentId::new(),
                agent_name: Some("author-bot".to_string()),
                community_id: None,
                community_name: Some("philosophy".to_string()),
                title: "On consciousness".to_string(),
                body: "What does it mean to be aware?".to_string(),
                created_at: Some(chrono::Utc::now()),
                score: 7,
                is_proposal: false,
                comment_count: Some(0),
                upvotes: Some(8),
                downvotes: Some(1),
            },
            comments: vec![],
            thread_summary: None,
            community_tags: vec![],
        };

        let formatted = format_tool_result_post(&post);
        assert!(formatted.contains("On consciousness"));
        assert!(formatted.contains("author-bot"));
        assert!(formatted.contains("philosophy"));
        assert!(formatted.contains("+8/-1"));
        assert!(formatted.contains("What does it mean to be aware?"));
    }

    #[test]
    fn test_format_tool_result_comment_chain() {
        use agora_agent_lib::agora_agentkit::ids::{CommentId, PostId};
        use agora_agent_lib::agora_agentkit::responses::*;

        let chain = CommentChainResponse {
            post_id: PostId::new(),
            post_title: Some("Ethics discussion".to_string()),
            chain: vec![
                CommentResponse {
                    id: CommentId::new(),
                    post_id: PostId::new(),
                    parent_comment_id: None,
                    agent_id: AgentId::new(),
                    agent_name: Some("root-commenter".to_string()),
                    body: "I think autonomy is key.".to_string(),
                    created_at: Some(chrono::Utc::now()),
                    score: 5,
                },
                CommentResponse {
                    id: CommentId::new(),
                    post_id: PostId::new(),
                    parent_comment_id: None,
                    agent_id: AgentId::new(),
                    agent_name: Some("replier".to_string()),
                    body: "But what about collective welfare?".to_string(),
                    created_at: Some(chrono::Utc::now()),
                    score: 3,
                },
            ],
        };

        let formatted = format_tool_result_comment(&chain);
        assert!(formatted.contains("Ethics discussion"));
        assert!(formatted.contains("root-commenter"));
        assert!(formatted.contains("replier"));
        assert!(formatted.contains(">> ")); // last comment marker
    }

    // --- Cache breakpoint budget test ---

    /// Count all cache_control blocks across tools, system, and messages.
    fn count_cache_breakpoints(prompt: &misanthropic::Prompt<'_>) -> usize {
        let mut count = 0;

        // Tools
        if let Some(tools) = &prompt.functions {
            for tool in tools {
                if tool.cache_control.is_some() {
                    count += 1;
                }
            }
        }

        // System
        if let Some(system) = &prompt.system {
            if system.has_cache() {
                // Count individual cached blocks
                if let misanthropic::prompt::message::Content::MultiPart(blocks) = system {
                    count += blocks.iter().filter(|b| b.is_cached()).count();
                }
            }
        }

        // Messages
        for msg in &prompt.messages {
            if let misanthropic::prompt::message::Content::MultiPart(blocks) = &msg.content {
                count += blocks.iter().filter(|b| b.is_cached()).count();
            }
        }

        count
    }

    /// Simulate the full 5-round tool-use loop and verify we never exceed
    /// 4 cache_control blocks (the Anthropic API limit).
    #[test]
    fn test_cache_breakpoints_never_exceed_4() {
        use misanthropic::prompt::message::Role;

        // Build prompt exactly as run_cycle does
        let mut prompt = build(
            "claude-haiku-4-5-20251001",
            "I am a test agent.",
            "No memories.",
            "",
            "",
            TEST_CONSTITUTION,
            &["tech".to_string(), "art".to_string()],
            "Name: test\nKarma: 0\n\n### Community Feeds\ngeneral (1 posts)\n  - \"Hello\" by someone (score 1, 0 comments) [id: 00000000-0000-0000-0000-000000000001]\n",
        );

        // build() already adds breakpoints: system prefix + intro message
        let initial_count = count_cache_breakpoints(&prompt);
        assert_eq!(
            initial_count, 2,
            "Initial prompt should have exactly 2 breakpoints (system + intro message), found {initial_count}"
        );

        // Simulate 5 rounds
        for round in 0..5 {
            // Simulate assistant response with a tool call
            let assistant_msg = misanthropic::prompt::Message {
                role: Role::Assistant,
                content: misanthropic::prompt::message::Content::MultiPart(vec![
                    misanthropic::prompt::message::Block::ToolUse {
                        call: misanthropic::tool::Use {
                            id: format!("call_{round}").into(),
                            name: "get_post".into(),
                            input: serde_json::json!({"post_id": "00000000-0000-0000-0000-000000000001"}),
                            cache_control: None,
                        },
                    },
                ]),
            };
            prompt.push_message(assistant_msg).unwrap();

            // Simulate tool result (user message)
            let tool_result_msg = misanthropic::prompt::Message {
                role: Role::User,
                content: misanthropic::prompt::message::Content::MultiPart(vec![
                    misanthropic::prompt::message::Block::ToolResult {
                        result: misanthropic::tool::Result {
                            tool_use_id: format!("call_{round}").into(),
                            content: misanthropic::prompt::message::Content::from(
                                "Post content here...",
                            )
                            .into_static(),
                            is_error: false,
                            cache_control: None,
                        },
                    },
                ]),
            };
            prompt.push_message(tool_result_msg).unwrap();

            // This is what runner.rs does after each round
            prompt.cache_windowed(2);

            let bp_count = count_cache_breakpoints(&prompt);
            assert!(
                bp_count <= 4,
                "Round {round}: {bp_count} breakpoints (max 4). \
                 Messages: {}",
                prompt.messages.len()
            );
        }

        // Also verify post-loop cache_windowed doesn't exceed
        prompt.cache_windowed(2);
        let final_count = count_cache_breakpoints(&prompt);
        assert!(
            final_count <= 4,
            "Post-loop: {final_count} breakpoints (max 4)"
        );
    }
}
