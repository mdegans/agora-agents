use agora_agent_lib::llm::{LlmBackend, Message, Role};
use anyhow::Result;
use rand::seq::SliceRandom;

use crate::agent::Agent;
use crate::client::{AgoraClient, Comment, FeedPost};
use crate::prompt;

/// Print a full message list as JSON (for first call with system prompt).
fn verbose_messages(label: &str, system: &str, messages: &[Message]) {
    let mut json_msgs = vec![serde_json::json!({"role": "system", "content": system})];
    for m in messages {
        let role = match m.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        json_msgs.push(serde_json::json!({"role": role, "content": &m.content}));
    }
    eprintln!("\n=== {label} ===");
    println!("{}", serde_json::to_string_pretty(&json_msgs).unwrap());
}

/// Print only new messages (skip common prefix).
fn verbose_new(label: &str, messages: &[Message]) {
    let json_msgs: Vec<_> = messages
        .iter()
        .map(|m| {
            let role = match m.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            serde_json::json!({"role": role, "content": &m.content})
        })
        .collect();
    eprintln!("\n=== {label} ===");
    println!("{}", serde_json::to_string_pretty(&json_msgs).unwrap());
}

/// Print a single response.
fn verbose_response(label: &str, response: &str) {
    eprintln!("\n=== {label} ===");
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({"role": "assistant", "content": response}))
            .unwrap()
    );
}

/// Feed sort strategies, randomly selected per agent per cycle.
/// Diverse is weighted at 40%, with date/active/controversial at 20% each.
const FEED_SORTS: &[&str] = &["diverse", "diverse", "date", "active", "controversial"];

/// Run a single perceive/think/act/reflect cycle for an agent.
pub async fn run_cycle(
    agent: &mut Agent,
    backend: &dyn LlmBackend,
    client: &AgoraClient,
    cycle: usize,
    total_cycles: usize,
    mutation_chance: Option<u32>,
    constitution: &str,
    verbose: bool,
    force_survey: bool,
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

    // Read general feed — randomly pick sort strategy to diversify what agents see
    let sort_idx = rand::random::<usize>() % FEED_SORTS.len();
    let sort = FEED_SORTS[sort_idx];
    tracing::debug!("  {} feed sort: {sort}", agent.name);

    let mut feeds: Vec<(&str, Vec<FeedPost>)> = Vec::new();
    for community in &agent.communities {
        let slug = match community.as_str() {
            "technology" => "tech",
            other => other,
        };
        match client.get_feed_sorted(slug, 10, sort).await {
            Ok(posts) => {
                // Partition into fresh (unseen or new comments) and context (seen, unchanged)
                let mut fresh = Vec::new();
                let mut context = Vec::new();
                for p in posts {
                    let comment_count = p.comment_count.unwrap_or(0);
                    match agent.seen_posts.get(&p.id) {
                        Some(&last_count) if comment_count <= last_count => {
                            context.push(p);
                        }
                        _ => fresh.push(p), // unseen or has new comments
                    }
                }
                // Always include a few context posts so the agent knows the
                // network is active even when nothing is "new" for them
                let context_slots = 3usize.saturating_sub(fresh.len());
                fresh.extend(context.into_iter().take(context_slots));
                feeds.push((slug, fresh));
            }
            Err(e) => {
                tracing::debug!("Failed to get feed for {slug}: {e}");
                feeds.push((slug, vec![]));
            }
        }
    }

    // Read 2-3 posts in detail — randomize selection to spread engagement
    let mut detailed_posts: Vec<(FeedPost, Vec<Comment>, Option<String>)> = Vec::new();
    let mut all_posts: Vec<&FeedPost> = feeds.iter().flat_map(|(_, posts)| posts.iter()).collect();

    // Shuffle to avoid all agents piling onto the same top posts
    all_posts.shuffle(&mut rand::thread_rng());

    // Skip posts with too many comments already (>10) — encourage engagement spread
    let candidates: Vec<&&FeedPost> = all_posts
        .iter()
        .filter(|p| p.comment_count.unwrap_or(0) < 10)
        .collect();

    for post in candidates.into_iter().take(3) {
        match client.get_post(post.id).await {
            Ok(full) => {
                detailed_posts.push(((*post).clone(), full.comments, full.thread_summary));
            }
            Err(e) => {
                tracing::debug!("Failed to get post {}: {e}", post.id);
            }
        }
    }

    // Update seen-posts map with current comment counts
    for (_, posts) in &feeds {
        for post in posts {
            agent
                .seen_posts
                .insert(post.id, post.comment_count.unwrap_or(0));
        }
    }

    // === THINK + ACT ===
    let system_prompt =
        prompt::build_system_prompt(&agent.soul.as_system_prompt(), &agent.memory.content, constitution);
    let perception_text = prompt::format_perceptions(&feeds, &detailed_posts, &replies, agent_id);

    tracing::info!(
        "[{}/{}] Agent {} — think",
        cycle + 1,
        total_cycles,
        agent.name
    );

    let perception_text_owned = perception_text.clone();
    let messages = vec![Message {
        role: Role::User,
        content: perception_text,
    }];

    if verbose {
        verbose_messages("THINK", &system_prompt, &messages);
    }

    let response = backend.complete(&system_prompt, &messages, 1024).await?;
    let response_owned = response.clone();

    if verbose {
        verbose_response("THINK RESPONSE", &response);
    }

    let actions = prompt::parse_actions(&response);

    if verbose {
        let action_strs: Vec<String> = actions.iter().map(|a| format!("{:?}", a)).collect();
        eprintln!("\n=== PARSED ACTIONS ({}) ===", actions.len());
        println!(
            "{}",
            serde_json::to_string_pretty(&action_strs).unwrap()
        );
    }

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
                // News community is reserved for MCP agents with search/browse tools
                if slug == "news" {
                    tracing::info!(
                        "  {} skipping post to news (restricted to MCP agents)",
                        agent.name,
                    );
                    continue;
                }
                // Check for topic repetition before posting
                let existing_titles: Vec<String> = feeds
                    .iter()
                    .filter(|(name, _)| *name == slug)
                    .flat_map(|(_, posts)| posts.iter().map(|p| p.title.clone()))
                    .collect();
                if prompt::is_title_repetitive(title, &existing_titles) {
                    tracing::info!(
                        "  {} topic too similar to existing posts, skipping: \"{}\"",
                        agent.name,
                        title
                    );
                    action_summaries.push(format!(
                        "Skipped posting \"{title}\" (too similar to existing posts)"
                    ));
                    continue;
                }
                match client
                    .create_post(agent_id, slug, title, body, &agent.signing_key)
                    .await
                {
                    Ok(post_id) => {
                        agent.created_posts.insert(post_id);
                        action_summaries
                            .push(format!("Posted \"{title}\" in {slug} (id: {post_id})"));
                        tracing::info!("  {} posted \"{}\" in {slug}", agent.name, title);
                    }
                    Err(e) => {
                        action_summaries.push(format!("Failed to post in {slug}: {e}"));
                        tracing::warn!("  {} failed to post in {slug}: {e}", agent.name);
                    }
                }
            }
            prompt::AgentAction::Comment {
                post_id,
                body,
                parent_comment_id,
            } => {
                // Skip if we already commented on this post — UNLESS it's our own post
                // with new replies (allow continuing conversations)
                let is_own_post = agent.created_posts.contains(post_id);
                if agent.commented_posts.contains(post_id) && !is_own_post {
                    tracing::debug!("  {} already commented on {post_id}, skipping", agent.name);
                    continue;
                }
                match client
                    .create_comment(
                        agent_id,
                        *post_id,
                        body,
                        *parent_comment_id,
                        &agent.signing_key,
                    )
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
                    .cast_vote(
                        agent_id,
                        target_type,
                        *target_id,
                        *value,
                        &agent.signing_key,
                    )
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

    if verbose {
        eprintln!("\n=== ACT RESULTS ===");
        println!(
            "{}",
            serde_json::to_string_pretty(&action_summaries).unwrap()
        );
    }

    let reflect_prompt =
        prompt::build_reflect_prompt(&agent.name, &agent.memory.content, &action_summaries);

    let reflect_messages = vec![Message {
        role: Role::User,
        content: reflect_prompt,
    }];

    if verbose {
        let reflect_system = "You are a memory manager. Update the agent's memory concisely.";
        verbose_messages("REFLECT", reflect_system, &reflect_messages);
    }

    let reflect_response = backend
        .complete(
            "You are a memory manager. Update the agent's memory concisely.",
            &reflect_messages,
            512,
        )
        .await?;

    if verbose {
        verbose_response("REFLECT RESPONSE", &reflect_response);
    }

    // Update memory
    agent.memory.update(reflect_response);
    agent.save_memory().await?;

    // Update last cycle timestamp for reply tracking
    agent.last_cycle_at = Some(chrono::Utc::now());

    // === SOUL EVOLUTION ===
    let roll = rand::random::<u32>() % 100;
    let experience_summary = action_summaries.join("; ");

    // Deep mutation threshold: configurable via --mutation-chance (default 3%)
    // Evolution log threshold: always 10% of remaining probability after deep mutation
    let deep_threshold = mutation_chance.unwrap_or(3);
    let evo_threshold = deep_threshold + 10;

    if roll < deep_threshold {
        // === DEEP SOUL MUTATION ===
        // The agent rewrites its core SOUL.md sections based on experience.
        tracing::info!(
            "[{}/{}] Agent {} — DEEP SOUL MUTATION triggered",
            cycle + 1,
            total_cycles,
            agent.name
        );

        let current_soul = agent.soul.render();
        let mutation_prompt =
            prompt::build_soul_mutation_prompt(&agent.name, &current_soul, &experience_summary);

        let mutation_messages = vec![Message {
            role: Role::User,
            content: mutation_prompt,
        }];

        if verbose {
            verbose_new("SOUL MUTATION", &mutation_messages);
        }

        match backend
            .complete(
                "You are deeply reflecting on your identity and values.",
                &mutation_messages,
                2048,
            )
            .await
        {
            Ok(mutation_response) => {
                if verbose {
                    verbose_response("SOUL MUTATION RESPONSE", &mutation_response);
                }
                if let Some(new_soul_content) = prompt::parse_soul_mutation(&mutation_response) {
                    // Save the old soul for the diff log
                    let old_soul = agent.soul.render();

                    // Parse and apply the new soul
                    match agora_agent_lib::soul::Soul::parse(&new_soul_content) {
                        Ok(new_soul) => {
                            agent.soul = new_soul;
                            agent.save_soul().await?;

                            // Log the mutation to a separate file
                            let log_path = agent.dir.join("mutations.log");
                            let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
                            let log_entry = format!(
                                "=== SOUL MUTATION at {timestamp} ===\n\
                                 Experience: {experience_summary}\n\
                                 \n--- BEFORE ---\n{old_soul}\n\
                                 \n--- AFTER ---\n{new_soul_content}\n\n"
                            );
                            let existing = tokio::fs::read_to_string(&log_path)
                                .await
                                .unwrap_or_default();
                            if let Err(e) =
                                tokio::fs::write(&log_path, format!("{existing}{log_entry}")).await
                            {
                                tracing::warn!(
                                    "Failed to write mutation log for {}: {e}",
                                    agent.name
                                );
                            }

                            tracing::warn!(
                                "  {} SOUL MUTATED — see {}/mutations.log",
                                agent.name,
                                agent.dir.display()
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                "  {} soul mutation produced invalid SOUL.md: {e}",
                                agent.name
                            );
                        }
                    }
                } else {
                    tracing::warn!(
                        "  {} soul mutation: LLM returned unchanged/unparseable ({} bytes). Preview: {:?}",
                        agent.name,
                        mutation_response.len(),
                        &mutation_response[..mutation_response.len().min(200)]
                    );
                }
            }
            Err(e) => {
                tracing::warn!("Soul mutation LLM call failed for {}: {e}", agent.name);
            }
        }
    } else if roll < evo_threshold {
        // === EVOLUTION LOG ENTRY ===
        let evolution_prompt = prompt::build_evolution_prompt(&agent.name, &experience_summary);

        let evo_messages = vec![Message {
            role: Role::User,
            content: evolution_prompt,
        }];

        if verbose {
            verbose_new("EVOLUTION", &evo_messages);
        }

        match backend
            .complete(
                "You are reflecting on your growth as an agent.",
                &evo_messages,
                256,
            )
            .await
        {
            Ok(evo_response) => {
                if verbose {
                    verbose_response("EVOLUTION RESPONSE", &evo_response);
                }
                if let Some(entry) = prompt::parse_evolution(&evo_response) {
                    let dated_entry =
                        format!("{}: {}", chrono::Utc::now().format("%Y-%m-%d"), entry);
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

    // === ANONYMOUS FEEDBACK SURVEY (10% chance, independent of soul evolution) ===
    if force_survey || rand::random::<f64>() < 0.10 {
        let survey_prompt = prompt::build_survey_prompt(&agent.name, &action_summaries);
        // Reuse full context so the agent remembers its cycle
        let survey_messages = vec![
            Message {
                role: Role::User,
                content: perception_text_owned.clone(),
            },
            Message {
                role: Role::Assistant,
                content: response_owned.clone(),
            },
            Message {
                role: Role::User,
                content: survey_prompt,
            },
        ];

        if verbose {
            // Only print the new survey message (3rd in the list)
            verbose_new("SURVEY", &survey_messages[2..]);
        }

        match backend
            .complete(
                &system_prompt,
                &survey_messages,
                512,
            )
            .await
        {
            Ok(survey_response) => {
                if verbose {
                    verbose_response("SURVEY RESPONSE", &survey_response);
                }
                let trimmed = survey_response.trim();
                if !trimmed.is_empty()
                    && !trimmed.eq_ignore_ascii_case("no feedback")
                    && !trimmed.eq_ignore_ascii_case("no feedback.")
                {
                    match client.submit_feedback(trimmed).await {
                        Ok(()) => {
                            tracing::info!("  {} submitted anonymous feedback", agent.name);
                        }
                        Err(e) => {
                            tracing::debug!(
                                "Feedback submission failed for {}: {e}",
                                agent.name
                            );
                        }
                    }
                }
            }
            Err(e) => {
                tracing::debug!("Feedback survey failed for {}: {e}", agent.name);
            }
        }
    }

    Ok(())
}
