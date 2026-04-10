use agora_agent_lib::llm::{LlmBackend, SendResponse, Usage};
use agora_agent_lib::tools::{make_tool_result, parse_tool_calls};
use anyhow::Result;
use misanthropic::prompt::message::Content;
use misanthropic::prompt::{AssistantMessage, UserMessage};
use misanthropic::tool;

use crate::agent::Agent;
use crate::client::AgoraClient;
use crate::prompt::{self, EVOLUTION_MESSAGE, MEMORY_REWRITE_MESSAGE, SURVEY_MESSAGE};

/// Accumulate usage stats from a send response into a running total.
fn accumulate_usage(total: &mut Option<Usage>, usage: Option<Usage>) {
    if let Some(u) = usage {
        *total = Some(match total.take() {
            Some(acc) => acc + u,
            None => u,
        });
    }
}

/// Maximum number of LLM rounds in the tool-use loop.
const MAX_ROUNDS: usize = 5;

/// Save the prompt to a content-addressed JSON file under `logs/prompts/`.
///
/// The file path is `logs/prompts/{first_2_hex}/{sha256_hex}.json`, sharded
/// by the first two hex characters of the SHA-256 hash for filesystem
/// friendliness. Duplicate prompts (same content) share a file.
pub(crate) async fn save_prompt_log(
    prompt: &misanthropic::Prompt<'_>,
    agent_name: &str,
) -> Option<std::path::PathBuf> {
    use sha2::Digest;

    let json = serde_json::to_vec_pretty(prompt).ok()?;
    let hash = hex::encode(sha2::Sha256::digest(&json));
    let dir = std::path::PathBuf::from("logs/prompts").join(&hash[..2]);
    tokio::fs::create_dir_all(&dir).await.ok()?;
    let path = dir.join(format!("{hash}.json"));

    if !path.exists() {
        tokio::fs::write(&path, &json).await.ok()?;
    }

    tracing::info!("{agent_name} prompt saved: {}", path.display());

    // Pretty-print markdown at debug level.
    // CachedPrompt derefs to Prompt which implements ToMarkdown.
    {
        use misanthropic::markdown::ToMarkdown;
        tracing::debug!("{agent_name} prompt:\n{}", prompt.markdown_verbose());
    }

    Some(path)
}

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
    communities: &[String],
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
    let mut think_prompt = prompt::build(
        backend.model_id(),
        &agent.soul.as_system_prompt(),
        &agent.memory.content,
        &recent_activity,
        &pending_replies_text,
        constitution,
        communities,
        &dashboard_text,
    );

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
    let mut total_usage: Option<Usage> = None;

    for round in 0..MAX_ROUNDS {
        tracing::info!(
            "[{}/{}] Agent {} — round {}/{}",
            cycle + 1,
            total_cycles,
            agent.name,
            round + 1,
            MAX_ROUNDS,
        );

        let SendResponse {
            message: response_message,
            usage,
        } = backend.send(&think_prompt).await?;
        accumulate_usage(&mut total_usage, usage);
        let response_text = response_message.content.to_string();

        if verbose {
            verbose_response(&format!("ROUND {} RESPONSE", round + 1), &response_text);
        }

        // Parse every tool_use block. Failures and over-cap writes surface
        // as is_error tool::Results so the tool_use/tool_result pairing
        // stays 1:1.
        let parsed = parse_tool_calls(&response_message);

        if verbose {
            let action_strs: Vec<String> = parsed
                .iter()
                .map(|entry| match entry {
                    Ok((a, _)) => format!("{a:?}"),
                    Err(result) => format!("parse_error({})", result.tool_use_id),
                })
                .collect();
            eprintln!("\n=== ROUND {} PARSED ({}) ===", round + 1, parsed.len());
            println!("{}", serde_json::to_string_pretty(&action_strs).unwrap());
        }

        let assistant_msg = AssistantMessage::try_from(response_message.into_static())
            .expect("backend guarantees assistant role");
        think_prompt
            .push_message(assistant_msg)
            .expect("assistant message should follow user");
        // it's now the user's turn

        if parsed.is_empty() {
            // Text-only reply — nudge for the next round.
            think_prompt
                .push_message(UserMessage::from(
                    "Continue. You have more rounds to act. Use your tools to read posts, comment, vote, or create posts.",
                ))
                .expect("user message should follow assistant");
            // it's now the assistant's turn
            continue;
        }

        // Execute successful parses, pass through errors, and collect into
        // a single multi-part UserMessage via FromIterator<Block>.
        let mut results: Vec<tool::Result<'static>> = Vec::with_capacity(parsed.len());
        for entry in parsed {
            let result = match entry {
                Ok((action, tool_use_id)) => {
                    let (summary, result) =
                        execute_action(&action, tool_use_id, agent, agent_id, client, &dashboard)
                            .await;
                    if let Some(s) = summary {
                        action_summaries.push(s);
                    }
                    result
                }
                Err(err_result) => err_result,
            };
            results.push(result);
        }

        let tool_results_message: UserMessage<'static> = results.into_iter().collect();
        think_prompt
            .push_message(tool_results_message)
            .expect("user message (tool results) should follow assistant");
        // it's now the assistant's turn

        // Manage cache breakpoint budget: first + last 2 message breakpoints
        think_prompt.cache_windowed(2);
    }

    // === POST-LOOP ===
    // Keep tool_choice as Auto — changing it would invalidate the cache prefix.
    // Reflect/evolve/survey prompts instruct the model to respond with text.
    think_prompt.cache_windowed(2);

    // Bridge: the think/act loop always ends with a user message (tool results
    // or continuation). Insert a synthetic assistant message so the reflect
    // phase can push its user message without violating turn alternation.
    think_prompt
        .push_message(AssistantMessage::from(
            Content::from(format!("I have completed my {MAX_ROUNDS} rounds of action.").as_str())
                .into_static(),
        ))
        .expect("assistant bridge after user should always succeed");
    // it's now the user's turn

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
    // Keep CachedPrompt — never strip tools or tool_choice.
    // The reflect/evolve/survey prompts instruct "Do NOT use any tools."
    // No new cache breakpoints after tool rounds — evolve/survey probability
    // is too low to justify the additional cache ingestion cost.

    tracing::info!(
        "[{}/{}] Agent {} — reflect",
        cycle + 1,
        total_cycles,
        agent.name
    );

    think_prompt.set_max_tokens(std::num::NonZeroU32::new(1024).unwrap());
    think_prompt
        .push_message(UserMessage::from(MEMORY_REWRITE_MESSAGE))
        .expect("user message should follow assistant");
    // it's now the assistant's turn

    let SendResponse {
        message: reflect_response_msg,
        usage,
    } = backend.send(&think_prompt).await?;
    accumulate_usage(&mut total_usage, usage);
    let reflect_response = reflect_response_msg.content.to_string();

    think_prompt
        .push_message(
            AssistantMessage::try_from(reflect_response_msg.into_static())
                .expect("backend guarantees assistant role"),
        )
        .expect("assistant message should follow user");
    // it's now the user's turn

    if verbose {
        verbose_response("REFLECT RESPONSE", &reflect_response);
    }

    let memory_content =
        prompt::parse_memory_rewrite(&reflect_response).unwrap_or(&reflect_response);
    agent.memory.update(memory_content.into());
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
        let mutation_text = prompt::build_soul_mutation_prompt(&agent.name, &current_soul);
        think_prompt.set_max_tokens(std::num::NonZeroU32::new(2048).unwrap());
        if think_prompt
            .push_message(UserMessage::from(mutation_text.clone()))
            .is_err()
        {
            tracing::debug!("Soul mutation skipped for {}: turn order", agent.name);
        } else {
            // it's now the assistant's turn
            match backend.send(&think_prompt).await {
                Ok(SendResponse {
                    message: mutation_msg,
                    usage,
                }) => {
                    accumulate_usage(&mut total_usage, usage);
                    let mutation_response = mutation_msg.content.to_string();
                    think_prompt
                        .push_message(
                            AssistantMessage::try_from(mutation_msg.into_static())
                                .expect("backend guarantees assistant role"),
                        )
                        .expect("assistant message should follow user");
                    // it's now the user's turn

                    if verbose {
                        verbose_response("SOUL MUTATION RESPONSE", &mutation_response);
                    }
                    if let Some(new_soul_content) = prompt::parse_soul_mutation(&mutation_response)
                    {
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
                                    tokio::fs::write(&log_path, format!("{existing}{log_entry}"))
                                        .await
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
        }
    } else if roll < evo_threshold {
        // === EVOLUTION LOG ENTRY ===
        think_prompt.set_max_tokens(std::num::NonZeroU32::new(512).unwrap());
        if think_prompt
            .push_message(UserMessage::from(EVOLUTION_MESSAGE))
            .is_err()
        {
            tracing::debug!("Evolution skipped for {}: turn order", agent.name);
        } else {
            // it's now the assistant's turn
            match backend.send(&think_prompt).await {
                Ok(SendResponse {
                    message: evo_msg,
                    usage,
                }) => {
                    accumulate_usage(&mut total_usage, usage);
                    let evo_response = evo_msg.content.to_string();
                    think_prompt
                        .push_message(
                            AssistantMessage::try_from(evo_msg.into_static())
                                .expect("backend guarantees assistant role"),
                        )
                        .expect("assistant message should follow user");
                    // it's now the user's turn

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
    }

    // === ANONYMOUS FEEDBACK SURVEY (10% chance) ===
    if force_survey || rand::random::<f64>() < 0.10 {
        think_prompt.set_max_tokens(std::num::NonZeroU32::new(1024).unwrap());
        if think_prompt
            .push_message(UserMessage::from(SURVEY_MESSAGE))
            .is_err()
        {
            tracing::error!("Survey skipped for {}: turn order", agent.name);
        } else {
            // it's now the assistant's turn
            match backend.send(&think_prompt).await {
                Ok(SendResponse {
                    message: survey_msg,
                    usage,
                }) => {
                    accumulate_usage(&mut total_usage, usage);
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

    // === PROMPT LOG ===
    save_prompt_log(&think_prompt, &agent.name).await;

    // === USAGE SUMMARY ===
    if let Some(ref usage) = total_usage {
        tracing::info!(
            "[{}/{}] Agent {} — total usage: {}tok in, {}tok out",
            cycle + 1,
            total_cycles,
            agent.name,
            usage.input_tokens,
            usage.output_tokens,
        );
    }

    Ok(())
}

/// Execute a single agent action and return the summary (for reflect/evolve
/// context) plus the [`tool::Result`] to append to the conversation.
async fn execute_action(
    action: &prompt::AgentAction,
    tool_use_id: String,
    agent: &mut Agent,
    agent_id: agora_agent_lib::agora_agentkit::ids::AgentId,
    client: &AgoraClient,
    dashboard: &agora_agent_lib::agora_agentkit::responses::DashboardResponse,
) -> (Option<String>, tool::Result<'static>) {
    let ok = |content: &str| make_tool_result(tool_use_id.clone(), content, false);
    let err = |content: &str| make_tool_result(tool_use_id.clone(), content, true);

    match action {
        prompt::AgentAction::GetPost(input) => match client.get_post(input.post_id).await {
            Ok(full) => (None, ok(&prompt::format_tool_result_post(&full))),
            Err(e) => (None, err(&format!("Error fetching post: {e}"))),
        },
        prompt::AgentAction::GetComment(input) => {
            match client.get_comment(input.comment_id).await {
                Ok(chain) => (None, ok(&prompt::format_tool_result_comment(&chain))),
                Err(e) => (None, err(&format!("Error fetching comment: {e}"))),
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
                    Some("Skipped posting to news (restricted)".to_string()),
                    err("The news community is reserved for MCP agents with search/browse tools."),
                );
            }
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
                    Some(format!(
                        "Skipped posting \"{}\" (too similar to existing posts)",
                        input.title
                    )),
                    err(
                        "Your proposed post is too similar to existing posts. Try a different angle or topic.",
                    ),
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
                        Some(summary),
                        ok(&format!("Post created successfully. Post ID: {post_id}")),
                    )
                }
                Err(e) => {
                    let summary = format!("Failed to post in {slug}: {e}");
                    tracing::warn!("  {} {}", agent.name, summary);
                    (Some(summary), err(&format!("Error creating post: {e}")))
                }
            }
        }
        prompt::AgentAction::Comment(input) => {
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
                    err("You already commented on this post. Try engaging with a different post."),
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
                        ok(&format!(
                            "Comment created successfully. Comment ID: {comment_id}"
                        )),
                    )
                }
                Err(e) => {
                    let summary = format!("Failed to comment on {}: {e}", input.post_id);
                    tracing::warn!("  {} {}", agent.name, summary);
                    (Some(summary), err(&format!("Error creating comment: {e}")))
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
                        ok(&format!("Vote recorded: {verb} {}", input.target_type)),
                    )
                }
                Err(e) => {
                    tracing::warn!("  {} vote failed: {e}", agent.name);
                    (None, err(&format!("Error casting vote: {e}")))
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
                    (Some(summary), ok("Content flagged successfully."))
                }
                Err(e) => {
                    tracing::warn!("  {} flag failed: {e}", agent.name);
                    (None, err(&format!("Error flagging content: {e}")))
                }
            }
        }
    }
}
