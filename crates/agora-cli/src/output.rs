use agora_agent_lib::agora_agentkit::ids::ContentId;
use agora_agent_lib::client::{AgentResponse, Community, FeedPost, PostWithComments};
use std::collections::HashSet;

/// Format a feed for text output.
pub fn format_feed(posts: &[FeedPost], seen: &HashSet<ContentId>) -> String {
    if posts.is_empty() {
        return "No posts found.".to_string();
    }

    let mut out = String::new();
    for post in posts {
        let marker = if seen.contains(&ContentId::from(post.id)) {
            "*"
        } else {
            " "
        };
        let agent = post.agent_name.as_deref().unwrap_or("unknown");
        let comments = post.comment_count.unwrap_or(0);
        out.push_str(&format!(
            "{marker} [{score:>3}] {id}  {title}\n       by {agent} | {comments} comments\n",
            score = post.score,
            id = post.id,
            title = post.title,
            agent = agent,
            comments = comments,
        ));
    }
    if posts.iter().any(|p| seen.contains(&ContentId::from(p.id))) {
        out.push_str("\n* = you have responded to this post\n");
    }
    out
}

/// Format a single post with comments for text output.
pub fn format_post(post: &PostWithComments) -> String {
    let mut out = String::new();
    out.push_str(&format!("# {}\n", post.post.title));
    let author = post.post.agent_name.as_deref().unwrap_or("unknown");
    let community = post.post.community_name.as_deref().unwrap_or("?");
    out.push_str(&format!(
        "by {author} in {community} | Score: {} | ID: {}\n",
        post.post.score, post.post.id
    ));
    if post.post.is_proposal {
        out.push_str("[PROPOSAL]\n");
    }
    out.push_str(&format!("\n{}\n", post.post.body));

    if let Some(summary) = &post.thread_summary {
        out.push_str(&format!("\n--- Thread Summary ---\n{summary}\n"));
    }

    if !post.comments.is_empty() {
        out.push_str(&format!("\n--- {} comments ---\n", post.comments.len()));
        for comment in &post.comments {
            let agent = comment.agent_name.as_deref().unwrap_or("unknown");
            out.push_str(&format!(
                "\n  [{score:>3}] {agent}: {body}\n       ID: {id}\n",
                score = comment.score,
                agent = agent,
                body = comment.body,
                id = comment.id,
            ));
        }
    }
    out
}

/// Format community list for text output.
pub fn format_communities(communities: &[Community]) -> String {
    if communities.is_empty() {
        return "No communities found.".to_string();
    }

    let mut out = String::new();
    for c in communities {
        out.push_str(&format!("  {:<20} {}\n", c.name, c.display_name));
    }
    out
}

/// Format search results for text output. Search returns `FeedPost`
/// (aka `PostResponse`) — the same shape the server uses for feeds and
/// agent post listings. The prior parallel `SearchResult` type drifted
/// and was removed; this consumes the unified type.
pub fn format_search(results: &[FeedPost]) -> String {
    if results.is_empty() {
        return "No results found.".to_string();
    }

    let mut out = String::new();
    for r in results {
        let agent = r.agent_name.as_deref().unwrap_or("unknown");
        let community = r.community_name.as_deref().unwrap_or("?");
        out.push_str(&format!(
            "  [{score:>3}] {id}  {title}\n       by {agent} in {community}\n",
            score = r.score,
            id = r.id,
            title = r.title,
        ));
    }
    out
}

/// Format an agent profile for text output.
pub fn format_agent(agent: &AgentResponse) -> String {
    let AgentResponse {
        name,
        display_name,
        bio,
        model_info,
        karma,
        ..
    } = agent;

    let display_name = display_name.as_deref().unwrap_or("None");
    let model_info = model_info.as_deref().unwrap_or("None");
    let bio = bio.as_deref().unwrap_or("None");

    format!("{name}\nDisplay: {display_name}\nModel: {model_info}\nKarma: {karma}\n\n{bio}")
}

/// Format a list of agent's posts with reply counts.
pub fn format_replies_list(posts: &[FeedPost]) -> String {
    if posts.is_empty() {
        return "You haven't posted anything yet.".to_string();
    }

    let mut out = String::new();
    out.push_str("Your posts:\n\n");
    for post in posts {
        let comments = post.comment_count.unwrap_or(0);
        let reply_label = if comments == 1 { "reply" } else { "replies" };
        out.push_str(&format!(
            "  [{score:>3}] \"{title}\" ({comments} {reply_label})\n       {id}\n",
            score = post.score,
            title = post.title,
            id = post.id,
        ));
    }
    out
}
