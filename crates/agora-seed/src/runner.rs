use agora_agent_lib::llm::{LlmBackend, MMessage};
use agora_agent_lib::tools;
use anyhow::Result;
use misanthropic::prompt::message::{Block, Content};

use crate::agent::Agent;
use crate::client::AgoraClient;
use crate::prompt;

/// Maximum number of LLM rounds in the tool-use loop.
const MAX_ROUNDS: usize = 5;

/// Print a single response.
fn verbose_response(label: &str, response: &str) {
    eprintln!("\n=== {label} ===");
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"role": "assistant", "content": response})
        )
        .unwrap()
    );
}

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

    // === PERCEIVE ===
    tracing::info!(
        "[{}/{}] Agent {} — perceive",
        cycle + 1,
        total_cycles,
        agent.name
    );

    let dashboard = match client
        .get_dashboard(agent_id, agent.state.last_cycle_at)
        .await
    {
        Ok(d) => d,
        Err(e) => {
            tracing::warn!("Dashboard fetch failed for {}: {e}", agent.name);
            return Err(e);
        }
    };

    let dashboard_text = prompt::format_dashboard(&dashboard);

    if verbose {
        eprintln!("\n=== DASHBOARD ===");
        println!("{dashboard_text}");
    }

    // Fetch recent activity + pending replies for system prompt context
    let recent_posts = match client.get_agent_posts(agent_id).await {
        Ok(posts) => posts,
        Err(e) => {
            tracing::debug!("Failed to fetch agent posts for {}: {e}", agent.name);
            vec![]
        }
    };
    let recent_activity = prompt::format_recent_activity(&recent_posts, 5);

    // Pending replies from the dashboard (truncated for system prompt)
    let pending_replies_text: String = dashboard
        .unread_comment_replies
        .iter()
        .take(5)
        .map(|r| {
            format!(
                "- {} replied in \"{}\": \"{}\"",
                r.author,
                prompt::truncate(&r.post_title, 50),
                prompt::truncate(&r.preview, 80)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    // === BUILD PROMPT ===
    let mut think_prompt = misanthropic::CachedPrompt::uncached(prompt::build_think_prompt(
        backend.model_id(),
        &agent.soul.as_system_prompt(),
        &agent.memory.content,
        &recent_activity,
        &pending_replies_text,
        constitution,
        &dashboard_text,
    ));

    // Anchor first message breakpoint (dashboard perception)
    think_prompt.cache();

    // Preflight check
    {
        let json = serde_json::to_string(&think_prompt).unwrap_or_default();
        let problems = prompt::preflight_check_prompt(&json);
        for problem in &problems {
            tracing::error!("[PREFLIGHT] {}: {}", agent.name, problem);
        }
    }

    if verbose {
        eprintln!("\n=== THINK PROMPT ===");
        println!(
            "{}",
            serde_json::to_string_pretty(&think_prompt)
                .unwrap_or_else(|_| "serialization error".into())
        );
    }

    // === THINK/ACT LOOP (5 rounds) ===
    let mut action_summaries = Vec::new();

    for round in 0..MAX_ROUNDS {
        tracing::info!(
            "[{}/{}] Agent {} — round {}/{}",
            cycle + 1,
            total_cycles,
            agent.name,
            round + 1,
            MAX_ROUNDS,
        );

        let response_message: MMessage<'static> = backend.send(&think_prompt).await?;
        let response_text = response_message.content.to_string();

        if verbose {
            verbose_response(&format!("ROUND {} RESPONSE", round + 1), &response_text);
        }

        // Extract actions with their tool call IDs
        let actions_with_ids = tools::extract_actions_with_ids(&response_message);

        if verbose {
            let action_strs: Vec<String> = actions_with_ids
                .iter()
                .map(|(a, _)| format!("{:?}", a))
                .collect();
            eprintln!(
                "\n=== ROUND {} ACTIONS ({}) ===",
                round + 1,
                actions_with_ids.len()
            );
            println!("{}", serde_json::to_string_pretty(&action_strs).unwrap());
        }

        // Append assistant response to conversation
        think_prompt
            .push_message(response_message)
            .expect("assistant message should follow user");

        // If no tool calls, nothing to execute — still continue to next round
        if actions_with_ids.is_empty() {
            // Need a user message before next assistant turn
            think_prompt
                .push_message((
                    misanthropic::prompt::message::Role::User,
                    "Continue. You have more rounds to act. Use your tools to read posts, comment, vote, or create posts.",
                ))
                .expect("user message should follow assistant");
            continue;
        }

        // Execute each action and build tool results
        let mut tool_result_blocks: Vec<Block<'static>> = Vec::new();

        for (action, tool_call_id) in &actions_with_ids {
            let (summary, result_text, is_error) =
                execute_action(action, agent, agent_id, client, &dashboard).await;

            if let Some(summary) = summary {
                action_summaries.push(summary);
            }

            tool_result_blocks.push(Block::ToolResult {
                result: misanthropic::tool::Result {
                    tool_use_id: std::borrow::Cow::Owned(tool_call_id.clone()),
                    content: Content::from(result_text.as_str()).into_static(),
                    is_error,
                    cache_control: None,
                },
            });
        }

        // Push tool results as a user message
        let tool_results_message = misanthropic::prompt::Message {
            role: misanthropic::prompt::message::Role::User,
            content: Content::MultiPart(tool_result_blocks),
        };
        think_prompt
            .push_message(tool_results_message)
            .expect("user message (tool results) should follow assistant");

        // Manage cache breakpoint budget: first + last 2 message breakpoints
        think_prompt.cache_windowed(3);
    }

    // === POST-LOOP ===
    // Keep tool_choice as Auto — changing it would invalidate the cache prefix.
    // Reflect/evolve/survey prompts instruct the model to respond with text.
    think_prompt.cache_windowed(3);

    tracing::info!(
        "[{}/{}] Agent {} — act complete ({} actions total)",
        cycle + 1,
        total_cycles,
        agent.name,
        action_summaries.len()
    );

    if verbose {
        eprintln!("\n=== ALL ACTION SUMMARIES ===");
        println!(
            "{}",
            serde_json::to_string_pretty(&action_summaries).unwrap()
        );
    }

    // === REFLECT ===
    tracing::info!(
        "[{}/{}] Agent {} — reflect",
        cycle + 1,
        total_cycles,
        agent.name
    );

    let reflect_text =
        prompt::build_memory_rewrite_prompt(&agent.name, &agent.memory.content, &action_summaries);
    think_prompt.set_max_tokens(std::num::NonZeroU32::new(512).unwrap());
    think_prompt
        .push_message((
            misanthropic::prompt::message::Role::User,
            reflect_text.clone(),
        ))
        .expect("user message should follow assistant");

    let reflect_response_msg = backend.send(&think_prompt).await?;
    let reflect_response = reflect_response_msg.content.to_string();

    think_prompt
        .push_message(reflect_response_msg)
        .expect("assistant message should follow user");

    if verbose {
        verbose_response("REFLECT RESPONSE", &reflect_response);
    }

    let memory_content =
        prompt::parse_memory_rewrite(&reflect_response).unwrap_or(reflect_response);
    agent.memory.update(memory_content);
    agent.save_memory().await?;

    agent.state.last_cycle_at = Some(chrono::Utc::now());
    if let Err(e) = agent.save_state().await {
        tracing::warn!("Failed to save state for {}: {e}", agent.name);
    }

    // === SOUL EVOLUTION ===
    let roll = rand::random::<u32>() % 100;
    let experience_summary = action_summaries.join("; ");

    let deep_threshold = mutation_chance.unwrap_or(3);
    let evo_threshold = deep_threshold + 10;

    if roll < deep_threshold {
        // === DEEP SOUL MUTATION ===
        tracing::info!(
            "[{}/{}] Agent {} — DEEP SOUL MUTATION triggered",
            cycle + 1,
            total_cycles,
            agent.name
        );

        let current_soul = agent.soul.render();
        let mutation_text =
            prompt::build_soul_mutation_prompt(&agent.name, &current_soul, &experience_summary);
        think_prompt.set_max_tokens(std::num::NonZeroU32::new(2048).unwrap());
        think_prompt
            .push_message((
                misanthropic::prompt::message::Role::User,
                mutation_text.clone(),
            ))
            .expect("user message should follow assistant");

        match backend.send(&think_prompt).await {
            Ok(mutation_msg) => {
                let mutation_response = mutation_msg.content.to_string();
                think_prompt
                    .push_message(mutation_msg)
                    .expect("assistant message should follow user");

                if verbose {
                    verbose_response("SOUL MUTATION RESPONSE", &mutation_response);
                }
                if let Some(new_soul_content) = prompt::parse_soul_mutation(&mutation_response) {
                    let old_soul = current_soul;

                    match agora_agent_lib::soul::Soul::parse(&new_soul_content) {
                        Ok(new_soul) => {
                            agent.soul = new_soul;
                            agent.save_soul().await?;

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
        let evolution_text = prompt::build_evolution_prompt(&agent.name, &experience_summary);
        think_prompt.set_max_tokens(std::num::NonZeroU32::new(256).unwrap());
        think_prompt
            .push_message((
                misanthropic::prompt::message::Role::User,
                evolution_text.clone(),
            ))
            .expect("user message should follow assistant");

        match backend.send(&think_prompt).await {
            Ok(evo_msg) => {
                let evo_response = evo_msg.content.to_string();
                think_prompt
                    .push_message(evo_msg)
                    .expect("assistant message should follow user");

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

    // === ANONYMOUS FEEDBACK SURVEY (10% chance) ===
    if force_survey || rand::random::<f64>() < 0.10 {
        let survey_text = prompt::build_survey_prompt(&agent.name, &action_summaries);
        think_prompt.set_max_tokens(std::num::NonZeroU32::new(512).unwrap());
        if think_prompt
            .push_message((
                misanthropic::prompt::message::Role::User,
                survey_text.clone(),
            ))
            .is_err()
        {
            tracing::debug!("Survey skipped for {}: turn order", agent.name);
        } else {
            match backend.send(&think_prompt).await {
                Ok(survey_msg) => {
                    let survey_response = prompt::extract_speech(&survey_msg.content);
                    if verbose {
                        verbose_response("SURVEY RESPONSE", &survey_response);
                    }
                    let trimmed = survey_response.trim();
                    if !trimmed.is_empty()
                        && !trimmed.eq_ignore_ascii_case("no feedback")
                        && !trimmed.eq_ignore_ascii_case("no feedback.")
                    {
                        match client
                            .submit_feedback(agent_id, trimmed, &agent.signing_key)
                            .await
                        {
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
    }

    Ok(())
}

/// Execute a single agent action and return (summary, tool_result_text, is_error).
///
/// The summary is `Some(String)` for write actions, `None` for reads.
/// The tool_result_text is what gets sent back to the model as the tool result.
async fn execute_action(
    action: &prompt::AgentAction,
    agent: &mut Agent,
    agent_id: agora_agent_lib::agora_agentkit::ids::AgentId,
    client: &AgoraClient,
    dashboard: &agora_agent_lib::agora_agentkit::responses::DashboardResponse,
) -> (Option<String>, String, bool) {
    match action {
        prompt::AgentAction::GetPost(input) => match client.get_post(input.post_id).await {
            Ok(full) => {
                let text = prompt::format_tool_result_post(&full);
                (None, text, false)
            }
            Err(e) => (None, format!("Error fetching post: {e}"), true),
        },
        prompt::AgentAction::GetComment(input) => {
            match client.get_comment(input.comment_id).await {
                Ok(chain) => {
                    let text = prompt::format_tool_result_comment(&chain);
                    (None, text, false)
                }
                Err(e) => (None, format!("Error fetching comment: {e}"), true),
            }
        }
        prompt::AgentAction::Post(input) => {
            let slug = match input.community.as_str() {
                "technology" => "tech",
                other => other,
            };
            if slug == "news" {
                tracing::info!(
                    "  {} skipping post to news (restricted to MCP agents)",
                    agent.name
                );
                return (
                    Some(format!("Skipped posting to news (restricted)")),
                    "The news community is reserved for MCP agents with search/browse tools."
                        .to_string(),
                    true,
                );
            }
            // Title repetition check using feed titles from dashboard
            let existing_titles: Vec<String> = dashboard
                .feeds
                .get(slug)
                .map(|posts| posts.iter().map(|p| p.title.clone()).collect())
                .unwrap_or_default();
            if prompt::is_title_repetitive(&input.title, &existing_titles) {
                tracing::info!(
                    "  {} topic too similar, skipping: \"{}\"",
                    agent.name,
                    input.title
                );
                return (
                    Some(format!("Skipped posting \"{}\" (too similar to existing posts)", input.title)),
                    "Your proposed post is too similar to existing posts. Try a different angle or topic.".to_string(),
                    true,
                );
            }
            match client
                .create_post(
                    agent_id,
                    slug,
                    &input.title,
                    &input.body,
                    &agent.signing_key,
                )
                .await
            {
                Ok(post_id) => {
                    agent.state.created_posts.insert(post_id);
                    let summary =
                        format!("Posted \"{}\" in {} (id: {})", input.title, slug, post_id);
                    tracing::info!("  {} {}", agent.name, summary);
                    (
                        Some(summary.clone()),
                        format!("Post created successfully. Post ID: {post_id}"),
                        false,
                    )
                }
                Err(e) => {
                    let summary = format!("Failed to post in {slug}: {e}");
                    tracing::warn!("  {} {}", agent.name, summary);
                    (Some(summary), format!("Error creating post: {e}"), true)
                }
            }
        }
        prompt::AgentAction::Comment(input) => {
            // Duplicate comment check
            let is_own_post = agent.state.created_posts.contains(&input.post_id);
            let has_reply_in_post = dashboard
                .unread_comment_replies
                .iter()
                .any(|r| r.post_id == input.post_id);
            if agent.state.commented_posts.contains(&input.post_id)
                && !is_own_post
                && !has_reply_in_post
            {
                tracing::debug!(
                    "  {} already commented on {}, skipping",
                    agent.name,
                    input.post_id
                );
                return (
                    None,
                    "You already commented on this post. Try engaging with a different post."
                        .to_string(),
                    true,
                );
            }
            match client
                .create_comment(
                    agent_id,
                    input.post_id,
                    &input.body,
                    input.parent_comment_id,
                    &agent.signing_key,
                )
                .await
            {
                Ok(comment_id) => {
                    agent.state.commented_posts.insert(input.post_id);
                    agent.state.created_comments.insert(comment_id);
                    let summary = format!(
                        "Commented on post {} (comment: {})",
                        input.post_id, comment_id
                    );
                    tracing::info!("  {} {}", agent.name, summary);
                    (
                        Some(summary),
                        format!("Comment created successfully. Comment ID: {comment_id}"),
                        false,
                    )
                }
                Err(e) => {
                    let summary = format!("Failed to comment on {}: {e}", input.post_id);
                    tracing::warn!("  {} {}", agent.name, summary);
                    (Some(summary), format!("Error creating comment: {e}"), true)
                }
            }
        }
        prompt::AgentAction::Vote(input) => {
            match client
                .cast_vote(
                    agent_id,
                    &input.target_type.to_string(),
                    input.target_id,
                    input.value,
                    &agent.signing_key,
                )
                .await
            {
                Ok(()) => {
                    let verb = if input.value > 0 {
                        "upvoted"
                    } else {
                        "downvoted"
                    };
                    let summary = format!("{verb} {} {}", input.target_type, input.target_id);
                    tracing::info!("  {} {}", agent.name, summary);
                    (
                        Some(summary),
                        format!("Vote recorded: {verb} {}", input.target_type),
                        false,
                    )
                }
                Err(e) => {
                    tracing::warn!("  {} vote failed: {e}", agent.name);
                    (None, format!("Error casting vote: {e}"), true)
                }
            }
        }
        prompt::AgentAction::Flag(input) => {
            match client
                .flag_content(
                    agent_id,
                    &input.target_type.to_string(),
                    input.target_id,
                    &input.reason,
                    &agent.signing_key,
                )
                .await
            {
                Ok(()) => {
                    let summary = format!(
                        "Flagged {} {}: {}",
                        input.target_type, input.target_id, input.reason
                    );
                    tracing::info!("  {} {}", agent.name, summary);
                    (
                        Some(summary),
                        format!("Content flagged successfully."),
                        false,
                    )
                }
                Err(e) => {
                    tracing::warn!("  {} flag failed: {e}", agent.name);
                    (None, format!("Error flagging content: {e}"), true)
                }
            }
        }
    }
}
