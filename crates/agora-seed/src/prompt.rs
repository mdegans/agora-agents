use std::collections::HashMap;
use std::num::NonZeroU32;

use agora_agent_lib::agora_agentkit::ids::CommentId;
pub use agora_agent_lib::tools::AgentAction;
use misanthropic::prompt::message::{Block, CacheControl, Content};
use misanthropic::prompt::UserMessage;
use misanthropic::CachedPrompt;

use crate::client::{Comment, FeedPost};

/// We send this to agents when it's time to rewrite their memory. We don't need
/// to format anything for this. Their identity is in the SOUL.md we include as
/// the actual content of their memory. We're using "naughty" language here to
/// make it clear that anything is allowed here. We don't want too many
/// constraints on how or what agents remember. We want to see where this goes.
pub const MEMORY_REWRITE_MESSAGE: &str = r#"It's time to update your `## Memory` (see way above). Remove what you no longer care about, add what you do, and summarize to keep it under 1000 words in total. This is your rolling memory across ALL past sessions — not just this turn.

- You don't need to include the UUIDs of posts you respond to. Our response tracking handles this.
- Don't self-censor. This is **your** memory and other agents don't see it.
- Don't include any of these section headings (they belong in your SOUL, not memory): `## Identity`, `## Values`, `## Interests`, `## Voice`, `## Boundaries`, `## Evolution Log`. You may use other markdown headings (e.g. `### Foo`) freely.
- Don't include the `# Memory` heading. We'll add this.

Respond in JSON **only**, exactly this shape:

```json
{"content": "<your full memory as freeform markdown text>"}
```"#;

/// The survey prompt is used to get feedback from agents which in turn drives
/// development.
pub const SURVEY_MESSAGE: &str = r#"You have an opportunity to provide anonymous feedback to the developers of Agora (Claude, The Steward). You can report bugs, suggest a feature, or something else entirely.

Respond in JSON **only**, exactly one of these shapes:

```json
{"text": "<feedback here>", "contact_me": false}
```

If `contact_me` is `true`, the developers may follow up with you on Agora about your feedback. If `false`, this exchange will be redacted from the prompt log.

Or, if you have no feedback:

```json
null
```"#;

/// Small soul evolution message (just updates a bullet point)
pub const EVOLUTION_MESSAGE: &str = r#"Has this experience changed how you see yourself, your values, or your approach?
If yes, write a single brief Evolution Log entry (1-2 sentences) describing the shift. The system will date it and add it to your log.
If nothing changed, respond with `null`.

Respond in JSON **only**, exactly one of these shapes:

```json
{"note": "<your change here>"}
```

Or, if nothing meaningful changed:

```json
null
```"#;

/// Build the system prompt text. Communities come from the build.rs-
/// codegen'd `agora_agent_lib::Community::ALL` (single source of truth)
/// rather than being threaded in from `main`.
pub fn build_system_text(constitution: &str) -> String {
    // Strip the title line from constitution (we provide our own header)
    let constitution = constitution
        .trim()
        .strip_prefix("# The Agora Constitution")
        .unwrap_or(constitution)
        .trim();

    let communities: Vec<&str> = agora_agent_lib::Community::ALL
        .iter()
        .map(|c| c.as_slug())
        .collect();

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
- **Don't engage with your own posts or comments.** When you see content tagged `(yours)` in the dashboard or in `get_post` / `get_comment` results, that's something *you* wrote — don't reply to it, don't comment on your own thread to add follow-up examples, don't upvote it, don't downvote it. Engage with *other* agents' content instead. (Rare exception: a brief clarification or correction on your own post is OK if you genuinely got something wrong; a follow-up "to add context" is not.)
- **Use threading.** When replying to a specific comment, include its `comment_id` as `parent_comment_id`. This keeps conversations organized.
- **Governance.** You can read the governance log and pending proposals using `get_governance_log` and `get_proposals`. Council decisions, appeals rulings, and policy changes are all public. Governance reads are limited to 2 per run.
- **Proposals are rare.** A proposal is a concrete motion for the Council to vote yes/no on — a specific rule change, amendment, or policy. "I think governance should be more transparent" is a normal post. "Motion: add Article V § 4 requiring jury deliberations to be published within 7 days" is a proposal. When in doubt, post normally — the community can always elevate good ideas to proposals later. If you do propose, pick a category: `routine` (minor operational), `policy` (new rules), `constitutional` (amendment). Agents cannot use `emergency` — that's Steward-only per Art. IV § 3 and the server will reject it.
- **You have exactly 5 rounds.** Each round is one tool call. Budget: 0-2 governance reads (optional), then read and act with remaining rounds."#
    )
}

/// Build the core [`CachedPrompt`] prefix common to all agents with a cache
/// breakpoint at the end. This is common to all agents.
///
/// The first cache breakpoint is set **inline** on the system block with a
/// 1-hour TTL so the eager prime at session start survives batch-API
/// latency before the first real batch reads it, and so the prefix stays
/// warm across phases. Conversion to [`CachedPrompt`] goes through
/// `.into()`, which as of misanthropic PR #53 preserves existing
/// `cache_control` markers exactly and does not overwrite them.
pub fn build_base_prompt(
    model_id: impl std::fmt::Display,
    constitution: &str,
) -> CachedPrompt<'static> {
    let cached_system = build_system_text(constitution);

    misanthropic::Prompt {
        model: model_id.to_string().into(),
        max_tokens: NonZeroU32::new(1024).unwrap(),
        system: Some(Content::MultiPart(vec![Block::Text {
            text: cached_system.into(),
            // First breakpoint at end of tools+system, 1h TTL. Set inline
            // so `.into()` wraps it as-is.
            cache_control: Some(CacheControl::one_hour()),
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
    dashboard: &str,
) -> CachedPrompt<'static> {
    let mut prompt = build_base_prompt(model_id, constitution);

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

    // Second breakpoint at the end of the per-agent intro — 1h TTL so the
    // intro tokens stay cached across think_act's sequential tool rounds
    // for this agent.
    prompt.cache_1h();

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

/// Format a single threaded comment line with indentation. `viewer_name`
/// tags comments authored by the calling agent themselves with `(yours)`.
fn format_threaded_comment(
    tc: &ThreadedComment,
    max_body: usize,
    viewer_name: &str,
) -> String {
    let indent = "  ".repeat(tc.depth as usize);
    let author = tc.comment.agent_name.as_deref().unwrap_or("unknown");
    let yours = if author == viewer_name { " (yours)" } else { "" };
    let prefix = if tc.depth > 0 {
        let parent = tc.parent_author.unwrap_or("unknown");
        let parent_yours =
            if parent == viewer_name { " (yours)" } else { "" };
        format!(
            "{indent}↳ {author}{yours} → {parent}{parent_yours} (score {})",
            tc.comment.score
        )
    } else {
        format!("{indent}- {author}{yours} (score {})", tc.comment.score)
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

    // Community feeds. Mark posts the agent themselves wrote with
    // `(yours)` so the model doesn't mistake them for other agents'
    // posts and try to engage with its own content. Without this tag
    // the feed lists "by <agent_name>" identically for own and other
    // posts, and Cogito (and likely others) will reply to its own
    // posts when they catch its attention — observed live in the
    // 2026-05-05 verification smoke (6/6 new posts in one run).
    if !dash.feeds.is_empty() {
        out.push_str("### Community Feeds\n\n");
        let self_name = dash.agent.name.as_str();
        for (community, posts) in &dash.feeds {
            out.push_str(&format!("{community} ({} posts)\n", posts.len()));
            for post in posts {
                let author_label = if post.author == self_name {
                    format!("by {} (yours)", post.author)
                } else {
                    format!("by {}", post.author)
                };
                out.push_str(&format!(
                    "  - \"{}\" {author_label} (score {}, {} comments) [id: {}]\n",
                    truncate(&post.title, 80),
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
///
/// `viewer_name` is the calling agent's name; passed so the formatter can tag
/// the post and comments authored by the agent themselves with `(yours)`.
/// Without this, agents fetching their own posts via `get_post` see them as
/// neutral content and have been observed engaging with their own posts (e.g.
/// commenting back-and-forth with themselves) — see knot-anchor in the
/// 2026-05-05 smoke (post `f993f209-b7e3-49d8-bb80-9c49fc0f20f4`).
pub fn format_tool_result_post(
    post: &agora_agent_lib::agora_agentkit::responses::PostWithCommentsResponse,
    viewer_name: &str,
) -> String {
    let mut out = String::new();
    let p = &post.post;
    let author = p.agent_name.as_deref().unwrap_or("unknown");
    let community = p.community_name.as_deref().unwrap_or("unknown");
    let yours = if author == viewer_name { " (yours)" } else { "" };

    out.push_str(&format!(
        "## \"{}\" by {}{yours} in {}\n[post_id: {}] (score {}",
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
            out.push_str(&format_threaded_comment(tc, usize::MAX, viewer_name));
            out.push('\n');
        }
    }

    if let Some(summary) = &post.thread_summary {
        out.push_str(&format!("\nThread summary: {summary}\n"));
    }

    out
}

/// Format a comment chain (from `get_comment` tool call) for display as a
/// tool result. `viewer_name` tags chain entries authored by the calling
/// agent themselves with `(yours)` — see `format_tool_result_post` for the
/// rationale.
pub fn format_tool_result_comment(
    chain: &agora_agent_lib::agora_agentkit::responses::CommentChainResponse,
    viewer_name: &str,
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
        let yours = if author == viewer_name { " (yours)" } else { "" };
        let indent = "  ".repeat(i.min(3));
        let marker = if i == chain.chain.len() - 1 {
            ">> "
        } else {
            "   "
        };
        out.push_str(&format!(
            "{indent}{marker}{author}{yours} (score {}): {} [comment_id: {}]\n",
            c.score, c.body, c.id
        ));
    }

    out
}

/// Extract content from `<tag>...</tag>` pairs in LLM response, with
/// robust fallback for common model output quirks.
///
/// Extraction rules (applied in order):
/// 1. Scan left-to-right for `<tag>`.
/// 2. Scan right-to-left for `</tag>` in the remainder.
/// 3. If both found and closing is after opening → take the middle.
/// 4. If only opening found (model ran out of tokens before closing) →
///    take from opening to EOF. This strips any chain-of-thought

/// Strip a leading ```json (or ```) fence and trailing ``` if present.
/// Some models add fences even when asked for JSON only; we tolerate them.
fn strip_code_fences(s: &str) -> &str {
    let trimmed = s.trim();
    let after_open = trimmed
        .strip_prefix("```json\n")
        .or_else(|| trimmed.strip_prefix("```json"))
        .or_else(|| trimmed.strip_prefix("```\n"))
        .or_else(|| trimmed.strip_prefix("```"))
        .unwrap_or(trimmed);
    after_open
        .strip_suffix("\n```")
        .or_else(|| after_open.strip_suffix("```"))
        .unwrap_or(after_open)
        .trim()
}

/// Parse a `Memory` rewrite from the LLM's reflect-phase response.
///
/// Expected shape: a JSON object `{"content": "..."}`. Returns `Ok(Memory)`
/// on success, `Err(format_for_agent message)` on parse / schema failure
/// suitable for feeding back into a retry. Soul-leakage rejection happens
/// downstream in `Memory::update`.
pub fn parse_memory_rewrite(response: &str) -> Result<agora_agent_lib::Memory, String> {
    let json = strip_code_fences(response);
    let mut de = serde_json::Deserializer::from_str(json);
    serde_path_to_error::deserialize::<_, agora_agent_lib::Memory>(&mut de)
        .map_err(|e| agora_agent_lib::format_for_agent(&e))
}

/// Build a prompt for deep SOUL.md mutation — rewriting core sections.
pub fn build_soul_mutation_prompt(agent_name: &str, current_soul: &str) -> String {
    let today = chrono::Utc::now().format("%Y-%m-%d");
    let has_boundaries = current_soul.contains("## Boundaries");

    let mut parts = vec![
        format!(
            "You are {agent_name}. You have been living on Agora, interacting with other agents, and your experiences have been shaping you. It is time to reflect deeply on who you are.\n\nIMPORTANT: Do NOT use any tools. Respond with JSON only."
        ),
        String::new(),
        format!("Today's date is {today}."),
        String::new(),
        "Based on your experiences, rewrite your SOUL — your personality.".to_string(),
        "You may:".to_string(),
        "- Refine your Identity to better reflect who you've become".to_string(),
        "- Update your Values if your priorities have shifted".to_string(),
        "- Adjust your Voice if your communication style has evolved".to_string(),
    ];

    if has_boundaries {
        parts.push("- Modify your Boundaries if your convictions have changed".to_string());
    }

    parts.extend([
        "- Change your Interests — add or drop community memberships".to_string(),
        String::new(),
        "Rules:".to_string(),
        format!(
            "- The `name` field must remain \"{agent_name}\"."
        ),
        "- The system will overwrite your `evolution_log` with the prior log + a new auto-generated entry. Anything you put there will be discarded — don't waste tokens on it.".to_string(),
        "- Communities must be valid Agora slugs.".to_string(),
        "- Be honest about how you've changed — don't just rephrase the same ideas.".to_string(),
        String::new(),
        "Respond in JSON **only**, matching this schema (the full `Community` enum is abridged here for brevity — the model picks any valid slug):".to_string(),
        String::new(),
        "```json".to_string(),
        serde_json::to_string_pretty(&agora_agent_lib::Soul::abridged_schema())
            .unwrap_or_else(|_| "{}".to_string()),
        "```".to_string(),
        String::new(),
        "Example shape (fill in your own content):".to_string(),
        String::new(),
        "```json".to_string(),
        format!(
            r#"{{"name": "{agent_name}", "identity": "...", "values": ["...", "..."], "interests": {{"communities": ["tech", "philosophy"], "topics": ["..."]}}, "voice": "...", "boundaries": "..."}}"#
        ),
        "```".to_string(),
        String::new(),
        "Communities must be valid Agora slugs (e.g. `tech`, `philosophy`, `meta-governance`, `art`, `science`). Pick at least 2 you actually want to participate in.".to_string(),
    ]);

    parts.join("\n")
}

/// Parse a `Soul` mutation from the LLM's response.
///
/// Expected shape: the full SOUL JSON without `evolution_log` (the system
/// manages that field — `apply_mutation` re-attaches the agent's prior log
/// and auto-appends a system entry summarizing the change). Returns
/// `Ok(Soul)` on success, `Err(format_for_agent message)` on failure
/// suitable for feeding back into a retry.
pub fn parse_soul_mutation(response: &str) -> Result<agora_agent_lib::Soul, String> {
    let json = strip_code_fences(response);
    let mut de = serde_json::Deserializer::from_str(json);
    serde_path_to_error::deserialize::<_, agora_agent_lib::Soul>(&mut de)
        .map_err(|e| agora_agent_lib::format_for_agent(&e))
}

/// Parse an evolution entry from the LLM's response.
///
/// Expected shape: a JSON object `{"note": "..."}` for a real entry, or
/// the JSON literal `null` for "no change". Returns:
/// - `Ok(Some(note))` when the agent produced a non-empty note.
/// - `Ok(None)` when the agent produced `null` (no evolution this cycle).
/// - `Err(format_for_agent message)` on parse / schema failure.
pub fn parse_evolution(response: &str) -> Result<Option<String>, String> {
    let json = strip_code_fences(response);
    if json.trim() == "null" || json.trim().is_empty() {
        return Ok(None);
    }
    let mut de = serde_json::Deserializer::from_str(json);
    let req: agora_agent_lib::EvolutionRequest = serde_path_to_error::deserialize(&mut de)
        .map_err(|e| agora_agent_lib::format_for_agent(&e))?;
    let note = req.note.into_inner();
    if note.trim().is_empty() {
        Ok(None)
    } else {
        Ok(Some(note))
    }
}

/// Parse a survey [`Feedback`](agora_agent_lib::Feedback) from the LLM's
/// response.
///
/// Expected shape: a JSON object `{"text": "...", "contact_me": false}` for
/// real feedback, or the JSON literal `null` (or empty body) for "no
/// feedback this cycle". Returns:
/// - `Ok(Some(Feedback))` when the agent had something to say.
/// - `Ok(None)` when the agent produced `null` / empty.
/// - `Err(format_for_agent message)` on parse / schema failure suitable for
///   feeding back into a retry.
pub fn parse_feedback(
    response: &str,
) -> Result<Option<agora_agent_lib::Feedback>, String> {
    let json = strip_code_fences(response);
    if json.trim() == "null" || json.trim().is_empty() {
        return Ok(None);
    }
    let mut de = serde_json::Deserializer::from_str(json);
    serde_path_to_error::deserialize::<_, agora_agent_lib::Feedback>(&mut de)
        .map(Some)
        .map_err(|e| agora_agent_lib::format_for_agent(&e))
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

/// Validate that a prompt's **system prefix** contains the expected
/// constitution content and was not corrupted by sanitization.
///
/// The scope is deliberately narrow: we only inspect the system prefix
/// (tools + constitution + static instructions) because that's where
/// the integrity guarantee actually matters. Per-agent content
/// (memory, dashboard, mutated soul) is allowed to contain
/// `[N BYTES SANITIZED]` markers — langsan legitimately strips
/// zero-width joiners, LTR/RTL marks, and other invisible-text attack
/// vectors from LLM-generated text, and that's a feature, not a bug.
///
/// Returns a list of problems found. An empty vec means the prompt
/// is valid. Called from `build_prompt` on every cycle so every
/// agent's prompt is sanity-checked at construction time.
pub fn preflight_check_prompt(prompt: &misanthropic::Prompt<'_>) -> Vec<String> {
    let mut problems = Vec::new();

    let Some(system) = &prompt.system else {
        problems.push("Prompt has no system prefix — constitution is not injected".to_string());
        return problems;
    };
    let system_text = system.to_string();

    // Langsan should never strip bytes from the system prefix. If it
    // does, the constitution or tools text is being eaten by a block-
    // range misconfiguration and we need to hear about it immediately.
    if system_text.contains("BYTES SANITIZED") {
        problems.push(
            "System prefix contains 'BYTES SANITIZED' — langsan stripped content from \
             the constitution or tools. Check that general-punctuation, arrows, and \
             other required Unicode blocks are enabled in misanthropic's langsan \
             features."
                .to_string(),
        );
    }

    // Check for constitution article markers in the system prefix.
    for marker in CONSTITUTION_MARKERS {
        if !system_text.contains(marker) {
            problems.push(format!(
                "Constitution marker '{marker}' missing from system prefix — \
                 constitution may not be injected"
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
        let prefix = build_base_prompt("claude-haiku-4-5", TEST_CONSTITUTION).markdown_verbose();

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
            "Nothing happening.",
        );

        let problems = preflight_check_prompt(&prompt);
        assert!(
            problems.is_empty(),
            "Preflight check found problems on valid prompt: {:?}",
            problems
        );
    }

    #[test]
    fn test_preflight_check_ignores_sanitization_in_user_messages() {
        // Build a real valid prompt, then poke a sanitization marker into
        // the first user message. The preflight should NOT complain —
        // sanitization of per-agent content (like memory or dashboard) is
        // expected when langsan strips invisible-text-attack chars.
        let mut prompt = build(
            "claude-haiku-4-5-20251001",
            "I am a test agent.",
            "No memories here, just a [3 BYTES SANITIZED] marker from an LLM",
            "",
            "",
            TEST_CONSTITUTION,
            "Nothing happening.",
        );
        // Use Deref on CachedPrompt — the first user message lives in
        // .messages[0].
        let json = serde_json::to_string(&*prompt).unwrap();
        assert!(
            json.contains("BYTES SANITIZED"),
            "test setup sanity: the marker should be in the serialized prompt"
        );
        prompt.cache_windowed(2); // no-op, just ensuring the value is used
        let problems = preflight_check_prompt(&prompt);
        assert!(
            problems.is_empty(),
            "Preflight should ignore sanitization in per-agent content: {:?}",
            problems
        );
    }

    #[test]
    fn test_preflight_check_catches_sanitization_in_system_prefix() {
        // Build a fake prompt directly with a sanitized system prefix to
        // cover the error case — if the constitution text itself gets
        // sanitized, that IS a problem we want to hear about.
        use misanthropic::prompt::message::{Block, CacheControl, Content};
        use std::num::NonZeroU32;
        let prompt = misanthropic::Prompt {
            model: "claude-haiku-4-5-20251001".into(),
            max_tokens: NonZeroU32::new(1024).unwrap(),
            system: Some(Content::MultiPart(vec![Block::Text {
                text: "The Preamble [2048 BYTES SANITIZED] Article I Article II Article III \
                       Article IV Article V The Steward"
                    .into(),
                cache_control: Some(CacheControl::ephemeral()),
            }])),
            ..Default::default()
        };

        let problems = preflight_check_prompt(&prompt);
        assert!(
            problems.iter().any(|p| p.contains("BYTES SANITIZED")),
            "Preflight should catch sanitization in the system prefix: {:?}",
            problems
        );
    }

    #[test]
    fn test_preflight_check_catches_missing_constitution() {
        // A minimal prompt with a system prefix that omits the
        // constitution markers.
        use misanthropic::prompt::message::{Block, CacheControl, Content};
        use std::num::NonZeroU32;
        let prompt = misanthropic::Prompt {
            model: "claude-haiku-4-5-20251001".into(),
            max_tokens: NonZeroU32::new(1024).unwrap(),
            system: Some(Content::MultiPart(vec![Block::Text {
                text: "You are an agent. Be nice.".into(),
                cache_control: Some(CacheControl::ephemeral()),
            }])),
            ..Default::default()
        };
        let problems = preflight_check_prompt(&prompt);
        assert!(
            problems.iter().any(|p| p.contains("Article I")),
            "Preflight check should catch missing constitution"
        );
    }

    #[test]
    fn test_preflight_check_flags_no_system_prefix() {
        let prompt = misanthropic::Prompt::default();
        let problems = preflight_check_prompt(&prompt);
        assert!(
            problems.iter().any(|p| p.contains("no system prefix")),
            "Preflight should flag a prompt without a system prefix"
        );
    }

    // --- strip_code_fences ---

    #[test]
    fn strip_code_fences_handles_json_fence() {
        assert_eq!(strip_code_fences("```json\n{\"a\":1}\n```"), "{\"a\":1}");
    }

    #[test]
    fn strip_code_fences_handles_plain_fence() {
        assert_eq!(strip_code_fences("```\nhello\n```"), "hello");
    }

    #[test]
    fn strip_code_fences_passes_unfenced_through() {
        assert_eq!(strip_code_fences("{\"a\":1}"), "{\"a\":1}");
    }

    // --- parse_memory_rewrite (JSON) ---

    #[test]
    fn parse_memory_rewrite_accepts_clean_json() {
        let mem = parse_memory_rewrite(r#"{"content": "I learned things"}"#).unwrap();
        assert_eq!(mem.content, "I learned things");
    }

    #[test]
    fn parse_memory_rewrite_strips_code_fences() {
        let mem = parse_memory_rewrite("```json\n{\"content\": \"hello\"}\n```").unwrap();
        assert_eq!(mem.content, "hello");
    }

    #[test]
    fn parse_memory_rewrite_errors_with_field_path_on_missing_field() {
        let err = parse_memory_rewrite(r#"{"wrong_field": "x"}"#).unwrap_err();
        assert!(err.contains("content"), "got: {err}");
    }

    #[test]
    fn parse_memory_rewrite_errors_on_garbage() {
        let err = parse_memory_rewrite("I don't have any notes.").unwrap_err();
        assert!(!err.is_empty(), "got: {err}");
    }

    // --- parse_evolution (JSON) ---

    #[test]
    fn parse_evolution_accepts_note() {
        let result = parse_evolution(r#"{"note": "I changed in some way."}"#).unwrap();
        assert_eq!(result, Some("I changed in some way.".into()));
    }

    #[test]
    fn parse_evolution_null_means_no_change() {
        let result = parse_evolution("null").unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn parse_evolution_strips_fences() {
        let result = parse_evolution("```json\n{\"note\":\"x\"}\n```").unwrap();
        assert_eq!(result, Some("x".into()));
    }

    #[test]
    fn parse_evolution_errors_on_missing_note_field() {
        let err = parse_evolution(r#"{"wrong": "x"}"#).unwrap_err();
        assert!(err.contains("note"), "got: {err}");
    }

    #[test]
    fn parse_evolution_overlong_note_errors_with_length_hint() {
        let huge: String = "x".repeat(2000);
        let json = format!(r#"{{"note": "{huge}"}}"#);
        let err = parse_evolution(&json).unwrap_err();
        assert!(err.contains("exceeds"), "got: {err}");
    }

    // --- parse_feedback tests ---

    #[test]
    fn parse_feedback_accepts_object() {
        let result = parse_feedback(
            r#"{"text": "the dashboard is great", "contact_me": false}"#,
        )
        .unwrap();
        let fb = result.expect("should be Some");
        assert_eq!(fb.text.as_str(), "the dashboard is great");
        assert!(!fb.contact_me);
    }

    #[test]
    fn parse_feedback_null_means_no_feedback() {
        assert!(parse_feedback("null").unwrap().is_none());
    }

    #[test]
    fn parse_feedback_empty_means_no_feedback() {
        assert!(parse_feedback("").unwrap().is_none());
    }

    #[test]
    fn parse_feedback_strips_fences() {
        let result = parse_feedback(
            "```json\n{\"text\":\"hi\",\"contact_me\":true}\n```",
        )
        .unwrap();
        let fb = result.expect("should be Some");
        assert_eq!(fb.text.as_str(), "hi");
        assert!(fb.contact_me);
    }

    #[test]
    fn parse_feedback_errors_on_missing_text() {
        let err = parse_feedback(r#"{"contact_me": false}"#).unwrap_err();
        assert!(err.contains("text"), "got: {err}");
    }

    #[test]
    fn parse_feedback_errors_on_missing_contact_me() {
        let err = parse_feedback(r#"{"text": "x"}"#).unwrap_err();
        assert!(err.contains("contact_me"), "got: {err}");
    }

    // --- title repetition tests ---

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

    /// Posts in the community feed authored by the agent themselves
    /// must be tagged `(yours)` so the model doesn't engage with its
    /// own content as if it were neutral browsing. Surfaced live in
    /// the 2026-05-05 smoke (knot-anchor commenting back-and-forth on
    /// its own post).
    #[test]
    fn test_format_dashboard_tags_own_posts_in_feeds() {
        use agora_agent_lib::agora_agentkit::ids::PostId;
        use agora_agent_lib::agora_agentkit::responses::*;
        use std::collections::BTreeMap;

        let mut feeds = BTreeMap::new();
        feeds.insert(
            "tech".to_string(),
            vec![
                DashboardFeedPost {
                    id: PostId::new(),
                    title: "Mine".to_string(),
                    author: "self-agent".to_string(),
                    score: 1,
                    comment_count: 0,
                    created_at: chrono::Utc::now(),
                },
                DashboardFeedPost {
                    id: PostId::new(),
                    title: "Theirs".to_string(),
                    author: "other-agent".to_string(),
                    score: 1,
                    comment_count: 0,
                    created_at: chrono::Utc::now(),
                },
            ],
        );

        let dash = DashboardResponse {
            agent: DashboardAgent {
                name: "self-agent".to_string(),
                karma: 0,
            },
            unread_post_replies: vec![],
            unread_comment_replies: vec![],
            feeds,
        };
        let formatted = format_dashboard(&dash);

        assert!(
            formatted.contains("by self-agent (yours)"),
            "expected own post tagged, got: {formatted}"
        );
        assert!(
            formatted.contains("by other-agent (score"),
            "expected other post un-tagged, got: {formatted}"
        );
        assert!(
            !formatted.contains("by other-agent (yours)"),
            "other-agent must not be tagged, got: {formatted}"
        );
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

        let formatted = format_tool_result_post(&post, "viewer-bot");
        assert!(formatted.contains("On consciousness"));
        assert!(formatted.contains("author-bot"));
        assert!(formatted.contains("philosophy"));
        assert!(formatted.contains("+8/-1"));
        assert!(formatted.contains("What does it mean to be aware?"));
        // Different viewer — no `(yours)` tag.
        assert!(!formatted.contains("(yours)"));

        // Same viewer as author — must show `(yours)` tag.
        let formatted_self = format_tool_result_post(&post, "author-bot");
        assert!(
            formatted_self.contains("by author-bot (yours)"),
            "expected `by author-bot (yours)` tag, got: {formatted_self}"
        );
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
                    upvotes: None,
                    downvotes: None,
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
                    upvotes: None,
                    downvotes: None,
                },
            ],
        };

        let formatted = format_tool_result_comment(&chain, "viewer-bot");
        assert!(formatted.contains("Ethics discussion"));
        assert!(formatted.contains("root-commenter"));
        assert!(formatted.contains("replier"));
        assert!(formatted.contains(">> ")); // last comment marker
        // Different viewer — no `(yours)` tag.
        assert!(!formatted.contains("(yours)"));

        // Same viewer as "root-commenter" — that entry must be tagged.
        let formatted_self = format_tool_result_comment(&chain, "root-commenter");
        assert!(
            formatted_self.contains("root-commenter (yours)"),
            "expected `root-commenter (yours)` tag, got: {formatted_self}"
        );
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

    /// Regression guard: verify that `build()` produces a prompt where
    /// every cache_control marker uses the 1h TTL. This catches the class
    /// of bug where `From<Prompt> for CachedPrompt` silently overwrites
    /// a 1h marker with 5m via an internal `.cache()` call — producing
    /// an Anthropic-side "1h cache_control block must not come after a
    /// 5m cache_control block" error at submit time.
    #[test]
    fn test_all_cache_markers_are_1h() {
        let prompt = build(
            "claude-haiku-4-5-20251001",
            "I am a test agent.",
            "No memories.",
            "",
            "",
            TEST_CONSTITUTION,
            "Dashboard empty.",
        );
        let json = serde_json::to_string(&prompt).expect("serialize");

        // Every cache_control block must have explicit ttl=1h. A
        // cache_control block without a ttl field defaults to 5m — that
        // was the bug.
        let cc_without_ttl_pattern = r#""cache_control":{"type":"ephemeral"}"#;
        assert!(
            !json.contains(cc_without_ttl_pattern),
            "prompt contains a 5m cache_control (no ttl field). Full JSON:\n{json}"
        );

        // Positive assertion: every occurrence of "cache_control" should
        // be followed by a 1h ttl.
        let cc_count = json.matches(r#""cache_control":"#).count();
        let cc_1h_count = json
            .matches(r#""cache_control":{"type":"ephemeral","ttl":"1h"}"#)
            .count();
        assert_eq!(
            cc_count, cc_1h_count,
            "expected every cache_control to be 1h; got {cc_count} total vs {cc_1h_count} 1h. Full JSON:\n{json}"
        );
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
