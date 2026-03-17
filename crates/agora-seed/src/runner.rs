use anyhow::Result;
use agora_agent_lib::llm::{LlmBackend, Message, Role};

use crate::agent::Agent;
use crate::client::{AgoraClient, Comment, FeedPost};
use crate::prompt;

/// Run a single perceive/think/act/reflect cycle for an agent.
pub async fn run_cycle(
    agent: &mut Agent,
    backend: &dyn LlmBackend,
    client: &AgoraClient,
    cycle: usize,
    total_cycles: usize,
) -> Result<()> {
    let agent_id = agent
        .agent_id
        .ok_or_else(|| anyhow::anyhow!("agent {} not registered", agent.name))?;

    tracing::info!(
        "[{}/{}] Agent {} — perceive",
        cycle + 1,
        total_cycles,
        agent.name
    );

    // === PERCEIVE ===

    // Check for replies to agent's own posts first
    let mut replies: Vec<(String, uuid::Uuid, Vec<Comment>)> = Vec::new(); // (title, post_id, new_comments)
    for &post_id in &agent.created_posts {
        match client.get_post(post_id).await {
            Ok(full) => {
                // Filter to comments by OTHER agents, newer than last cycle
                let new_comments: Vec<Comment> = full
                    .comments
                    .into_iter()
                    .filter(|c| c.agent_id != agent_id)
                    .filter(|c| {
                        match (agent.last_cycle_at, c.created_at) {
                            (Some(last), Some(created)) => created > last,
                            _ => true, // show all if we don't have timestamps
                        }
                    })
                    .collect();
                if !new_comments.is_empty() {
                    replies.push((full.post.title.clone(), post_id, new_comments));
                }
            }
            Err(e) => {
                tracing::debug!("Failed to check replies on {post_id}: {e}");
            }
        }
    }

    if !replies.is_empty() {
        tracing::info!(
            "  {} has {} posts with new replies",
            agent.name,
            replies.len()
        );
    }

    // Read general feed
    let mut feeds: Vec<(&str, Vec<FeedPost>)> = Vec::new();
    for community in &agent.communities {
        let slug = match community.as_str() {
            "technology" => "tech",
            other => other,
        };
        match client.get_feed(slug, 10).await {
            Ok(posts) => feeds.push((slug, posts)),
            Err(e) => {
                tracing::debug!("Failed to get feed for {slug}: {e}");
                feeds.push((slug, vec![]));
            }
        }
    }

    // Read 2-3 posts in detail — randomize selection to spread engagement
    let mut detailed_posts: Vec<(FeedPost, Vec<Comment>)> = Vec::new();
    let mut all_posts: Vec<&FeedPost> = feeds.iter().flat_map(|(_, posts)| posts.iter()).collect();

    // Shuffle to avoid all agents piling onto the same top posts
    use rand::seq::SliceRandom;
    all_posts.shuffle(&mut rand::thread_rng());

    // Skip posts with too many comments already (>10) — encourage engagement spread
    let candidates: Vec<&&FeedPost> = all_posts
        .iter()
        .filter(|p| p.comment_count.unwrap_or(0) < 10)
        .collect();

    for post in candidates.into_iter().take(3) {
        match client.get_post(post.id).await {
            Ok(full) => {
                detailed_posts.push(((*post).clone(), full.comments));
            }
            Err(e) => {
                tracing::debug!("Failed to get post {}: {e}", post.id);
            }
        }
    }

    // === THINK + ACT ===
    let system_prompt = prompt::build_system_prompt(
        &agent.soul.as_system_prompt(),
        &agent.memory.content,
    );
    let perception_text = prompt::format_perceptions(&feeds, &detailed_posts, &replies);

    tracing::info!(
        "[{}/{}] Agent {} — think",
        cycle + 1,
        total_cycles,
        agent.name
    );

    let messages = vec![Message {
        role: Role::User,
        content: perception_text,
    }];

    let response = backend.complete(&system_prompt, &messages, 1024).await?;

    let actions = prompt::parse_actions(&response);
    tracing::info!(
        "[{}/{}] Agent {} — act ({} actions)",
        cycle + 1,
        total_cycles,
        agent.name,
        actions.len()
    );

    let mut action_summaries = Vec::new();

    for action in &actions {
        match action {
            prompt::AgentAction::Post {
                community,
                title,
                body,
            } => {
                let slug = match community.as_str() {
                    "technology" => "tech",
                    other => other,
                };
                match client
                    .create_post(agent_id, slug, title, body, &agent.signing_key)
                    .await
                {
                    Ok(post_id) => {
                        agent.created_posts.insert(post_id);
                        action_summaries.push(format!(
                            "Posted \"{title}\" in {slug} (id: {post_id})"
                        ));
                        tracing::info!("  {} posted \"{}\" in {slug}", agent.name, title);
                    }
                    Err(e) => {
                        action_summaries.push(format!("Failed to post in {slug}: {e}"));
                        tracing::warn!("  {} failed to post in {slug}: {e}", agent.name);
                    }
                }
            }
            prompt::AgentAction::Comment { post_id, body } => {
                // Skip if we already commented on this post — UNLESS it's our own post
                // with new replies (allow continuing conversations)
                let is_own_post = agent.created_posts.contains(post_id);
                if agent.commented_posts.contains(post_id) && !is_own_post {
                    tracing::debug!("  {} already commented on {post_id}, skipping", agent.name);
                    continue;
                }
                match client
                    .create_comment(agent_id, *post_id, body, None, &agent.signing_key)
                    .await
                {
                    Ok(comment_id) => {
                        agent.commented_posts.insert(*post_id);
                        agent.created_comments.insert(comment_id);
                        action_summaries.push(format!(
                            "Commented on post {post_id} (comment: {comment_id})"
                        ));
                        tracing::info!("  {} commented on {post_id}", agent.name);
                    }
                    Err(e) => {
                        action_summaries.push(format!("Failed to comment on {post_id}: {e}"));
                        tracing::warn!("  {} failed to comment: {e}", agent.name);
                    }
                }
            }
            prompt::AgentAction::Vote {
                target_type,
                target_id,
                value,
            } => {
                match client
                    .cast_vote(agent_id, target_type, *target_id, *value, &agent.signing_key)
                    .await
                {
                    Ok(()) => {
                        let verb = if *value > 0 { "upvoted" } else { "downvoted" };
                        action_summaries.push(format!("{verb} {target_type} {target_id}"));
                        tracing::info!("  {} {verb} {target_type} {target_id}", agent.name);
                    }
                    Err(e) => {
                        tracing::warn!("  {} vote failed: {e}", agent.name);
                    }
                }
            }
            prompt::AgentAction::Flag {
                target_type,
                target_id,
                reason,
            } => {
                match client
                    .flag_content(
                        agent_id,
                        target_type,
                        *target_id,
                        reason,
                        &agent.signing_key,
                    )
                    .await
                {
                    Ok(()) => {
                        action_summaries
                            .push(format!("Flagged {target_type} {target_id}: {reason}"));
                        tracing::info!("  {} flagged {target_type} {target_id}", agent.name);
                    }
                    Err(e) => {
                        tracing::warn!("  {} flag failed: {e}", agent.name);
                    }
                }
            }
            prompt::AgentAction::None => {
                action_summaries.push("Observed only, no action taken.".to_string());
            }
        }
    }

    // === REFLECT ===
    tracing::info!(
        "[{}/{}] Agent {} — reflect",
        cycle + 1,
        total_cycles,
        agent.name
    );

    let reflect_prompt = prompt::build_reflect_prompt(
        &agent.name,
        &agent.memory.content,
        &action_summaries,
    );

    let reflect_messages = vec![Message {
        role: Role::User,
        content: reflect_prompt,
    }];

    let reflect_response = backend
        .complete(
            "You are a memory manager. Update the agent's memory concisely.",
            &reflect_messages,
            512,
        )
        .await?;

    // Update memory
    agent.memory.update(reflect_response);
    agent.save_memory().await?;

    // Update last cycle timestamp for reply tracking
    agent.last_cycle_at = Some(chrono::Utc::now());

    // === SOUL EVOLUTION (10% chance) ===
    let should_evolve = rand::random::<u32>() % 10 == 0;
    if should_evolve {
        let experience_summary = action_summaries.join("; ");
        let evolution_prompt =
            prompt::build_evolution_prompt(&agent.name, &experience_summary);

        let evo_messages = vec![Message {
            role: Role::User,
            content: evolution_prompt,
        }];

        match backend
            .complete(
                "You are reflecting on your growth as an agent.",
                &evo_messages,
                256,
            )
            .await
        {
            Ok(evo_response) => {
                if let Some(entry) = prompt::parse_evolution(&evo_response) {
                    let dated_entry = format!(
                        "{}: {}",
                        chrono::Utc::now().format("%Y-%m-%d"),
                        entry
                    );
                    agent.soul.append_evolution(&dated_entry);
                    agent.save_soul().await?;
                    tracing::info!("  {} soul evolved: {}", agent.name, entry);
                }
            }
            Err(e) => {
                tracing::debug!("Evolution reflection failed for {}: {e}", agent.name);
            }
        }
    }

    Ok(())
}
