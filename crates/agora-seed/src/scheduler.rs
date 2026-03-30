//! Pipeline scheduler for batch agent execution.
//!
//! Replaces the old wave-based scheduler with a pipeline that:
//! - Batches THINK requests for cache efficiency
//! - Interleaves batches so later agents see earlier agents' actions
//! - Supports both Anthropic Batch API and Ollama backends
//!
//! The cycle for each agent follows: PERCEIVE → THINK → ACT → REFLECT
//! where THINK and REFLECT are batched, while PERCEIVE and ACT are per-agent.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use agora_agent_lib::agora_agentkit::ids::AgentId;
use agora_agent_lib::agora_agentkit::scheduler::{
    BatchBackend, BatchState, CycleStep, WorkItem,
};
use agora_agent_lib::batch::anthropic::AnthropicBatch;
use agora_agent_lib::batch::ollama::{MultiOllamaBatch, OllamaEndpoint};
use agora_agent_lib::llm::{LlmBackend, Message, Role};
use agora_agent_lib::tools;
use anyhow::Result;
use misanthropic::prompt::Message as MMessage;
use misanthropic::Prompt;
use rand::seq::SliceRandom;

use crate::agent::Agent;
use crate::client::AgoraClient;
use crate::config::{Backend, Cli};
use crate::prompt;

/// Run all agents using the pipeline scheduler.
pub async fn run_all(
    agents: &mut Vec<Agent>,
    client: &AgoraClient,
    config: &Cli,
    constitution: &str,
) -> Result<()> {
    // Filter agents
    if let Some(ref filter) = config.agent_filter {
        agents.retain(|a| a.name.contains(filter.as_str()));
    }

    // Remove unregistered agents
    agents.retain(|a| {
        if a.agent_id.is_none() {
            tracing::warn!("Skipping unregistered agent: {}", a.name);
            false
        } else {
            true
        }
    });

    if agents.is_empty() {
        tracing::warn!("No registered agents to run");
        return Ok(());
    }

    tracing::info!(
        "Pipeline scheduler: {} agents, {} cycles",
        agents.len(),
        config.cycles,
    );

    // Select and run with the appropriate backend
    match config.backend {
        Backend::Ollama => {
            // Discover models on each endpoint.
            let http = reqwest::Client::new();
            let urls = config.effective_ollama_urls();
            let mut endpoints = Vec::with_capacity(urls.len());
            for url in &urls {
                match OllamaEndpoint::discover(&http, url).await {
                    Ok(ep) => endpoints.push(ep),
                    Err(e) => {
                        tracing::error!("Failed to discover models at {url}: {e}");
                        anyhow::bail!("Cannot reach Ollama endpoint {url}: {e}");
                    }
                }
            }

            // Warn about agents whose model isn't on any endpoint.
            let all_models: std::collections::HashSet<&str> = endpoints
                .iter()
                .flat_map(|ep| ep.models.iter().map(|m| m.as_str()))
                .collect();
            let mut missing: std::collections::HashMap<&str, usize> =
                std::collections::HashMap::new();
            for agent in agents.iter() {
                if !all_models.contains(agent.model.as_str()) {
                    *missing.entry(agent.model.as_str()).or_default() += 1;
                }
            }
            for (model, count) in &missing {
                tracing::warn!(
                    "Model '{model}' not on any endpoint ({count} agents affected)"
                );
            }

            let backend = MultiOllamaBatch::new(endpoints);
            let ollama_endpoints = backend.endpoints();
            run_cycles(&backend, agents, client, config, constitution, Some(ollama_endpoints)).await?;
        }
        Backend::Anthropic => {
            let key_file = config.anthropic_key_file.as_ref()
                .ok_or_else(|| anyhow::anyhow!("--anthropic-key-file is required when --backend=anthropic"))?;
            let api_key = tokio::fs::read_to_string(key_file).await
                .map_err(|e| anyhow::anyhow!("reading Anthropic key from {}: {e}", key_file.display()))?;
            let backend = AnthropicBatch::from_key(api_key.trim().to_string())?;
            run_cycles(&backend, agents, client, config, constitution, None).await?;
        }
    }

    tracing::info!("Pipeline scheduler complete!");
    Ok(())
}

/// Run all cycles with a given batch backend.
///
/// `ollama_endpoints` is provided when running with Ollama so the EVOLVE
/// phase can route single LLM calls to the correct endpoint per model.
async fn run_cycles<B>(
    backend: &B,
    agents: &mut Vec<Agent>,
    client: &AgoraClient,
    config: &Cli,
    constitution: &str,
    ollama_endpoints: Option<&[OllamaEndpoint]>,
) -> Result<()>
where
    B: BatchBackend<Prompt<'static>, MMessage<'static>>,
{
    // Determine batch size — smaller batches = more interleaving
    let batch_size = config.batch_size.unwrap_or(5);

    for cycle in 0..config.cycles {
        tracing::info!(
            "=== Cycle {}/{} ===",
            cycle + 1,
            config.cycles,
        );

        // Shuffle agents each cycle for fairness
        agents.shuffle(&mut rand::thread_rng());

        // Split agents into batches for interleaving
        let total_batches = (agents.len() + batch_size - 1) / batch_size;
        for (batch_num, batch_agents) in agents.chunks_mut(batch_size).enumerate() {
            tracing::info!(
                "--- Batch {}/{} ({} agents) ---",
                batch_num + 1,
                total_batches,
                batch_agents.len(),
            );

            // Phase 1: PERCEIVE — gather perceptions for all agents in this batch
            let mut agent_contexts: Vec<AgentCycleContext> = Vec::new();

            for (idx, agent) in batch_agents.iter_mut().enumerate() {
                let agent_id = agent.agent_id.unwrap(); // filtered above
                match perceive(agent, agent_id, client, config).await {
                    Ok(mut ctx) => {
                        ctx.batch_index = idx;
                        agent_contexts.push(ctx);
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Perceive failed for {}: {e:#}",
                            agent.name,
                        );
                    }
                }
            }

            if agent_contexts.is_empty() {
                continue;
            }

            // Phase 2: THINK — build prompts and submit batch
            let mut work_items: Vec<WorkItem<Prompt<'static>>> = Vec::new();

            for ctx in &agent_contexts {
                let agent = &batch_agents[ctx.batch_index];
                let think_prompt = prompt::build_think_prompt(
                    &agent.model,
                    &agent.soul.as_system_prompt(),
                    &agent.memory.content,
                    &ctx.recent_activity,
                    &ctx.pending_replies_text,
                    constitution,
                    &ctx.perception_text,
                );

                // Hash the cacheable prefix for grouping
                let prefix_hash = {
                    let mut hasher = DefaultHasher::new();
                    // All agents share the same constitution prefix,
                    // so the prefix hash is model-invariant here.
                    // The grouping key is really just the model.
                    agent.model.hash(&mut hasher);
                    hasher.finish()
                };

                work_items.push(WorkItem {
                    agent_id: agent.agent_id.unwrap(),
                    prompt: think_prompt,
                    step: CycleStep::Think,
                    prefix_hash,
                    model: agent.model.clone(),
                    queued_at: Instant::now(),
                    token_count: 0, // TODO: use count_tokens API
                });
            }

            // Submit and poll
            let think_results = submit_and_poll(backend, work_items).await?;

            // Phase 3: ACT — execute actions from THINK results
            let mut action_summaries_map: std::collections::HashMap<AgentId, Vec<String>> =
                std::collections::HashMap::new();
            let mut think_response_map: std::collections::HashMap<AgentId, String> =
                std::collections::HashMap::new();

            for result in &think_results {
                let response = match &result.response {
                    Ok(msg) => msg,
                    Err(e) => {
                        tracing::warn!(
                            "Think failed for agent {}: {e}",
                            result.agent_id,
                        );
                        continue;
                    }
                };

                // Find the agent and its context
                let Some(ctx) = agent_contexts.iter().find(|c| {
                    batch_agents[c.batch_index].agent_id == Some(result.agent_id)
                }) else {
                    continue;
                };
                let agent = &mut batch_agents[ctx.batch_index];

                // Save response text for survey context
                think_response_map.insert(result.agent_id, response.content.to_string());

                let actions = tools::extract_actions(response);
                tracing::info!(
                    "[{}/{}] {} — act ({} actions)",
                    cycle + 1,
                    config.cycles,
                    agent.name,
                    actions.len(),
                );

                let summaries = execute_actions(
                    agent,
                    &actions,
                    client,
                    &ctx.feeds,
                    &ctx.comment_replies,
                )
                .await;

                action_summaries_map.insert(result.agent_id, summaries);
            }

            // Phase 4: REFLECT — build reflect prompts and submit batch
            let mut reflect_items: Vec<WorkItem<Prompt<'static>>> = Vec::new();

            for ctx in &agent_contexts {
                let agent = &batch_agents[ctx.batch_index];
                let agent_id = agent.agent_id.unwrap();
                let summaries = action_summaries_map
                    .get(&agent_id)
                    .cloned()
                    .unwrap_or_default();

                let reflect_text = prompt::build_memory_rewrite_prompt(
                    &agent.name,
                    &agent.memory.content,
                    &summaries,
                );

                // Build a simple text prompt for reflect (no tools needed)
                let reflect_prompt = build_text_prompt(
                    &agent.model,
                    "You are a memory manager. Rewrite the agent's personal notes concisely.",
                    &reflect_text,
                    512,
                );

                reflect_items.push(WorkItem {
                    agent_id,
                    prompt: reflect_prompt,
                    step: CycleStep::Reflect,
                    prefix_hash: 0,
                    model: agent.model.clone(),
                    queued_at: Instant::now(),
                    token_count: 0,
                });
            }

            let reflect_results = submit_and_poll(backend, reflect_items).await?;

            // Apply reflect results
            for result in &reflect_results {
                let response_text = match &result.response {
                    Ok(msg) => msg.content.to_string(),
                    Err(e) => {
                        tracing::warn!(
                            "Reflect failed for agent {}: {e}",
                            result.agent_id,
                        );
                        continue;
                    }
                };

                let Some(ctx) = agent_contexts.iter().find(|c| {
                    batch_agents[c.batch_index].agent_id == Some(result.agent_id)
                }) else {
                    continue;
                };
                let agent = &mut batch_agents[ctx.batch_index];

                // Parse <memory> tags, fall back to raw response
                let memory_content = prompt::parse_memory_rewrite(&response_text)
                    .unwrap_or(response_text);
                agent.memory.update(memory_content);
                if let Err(e) = agent.save_memory().await {
                    tracing::warn!("Failed to save memory for {}: {e}", agent.name);
                }
                agent.last_cycle_at = Some(chrono::Utc::now());
            }

            // Phase 5: EVOLVE — soul evolution (low probability, per-agent)
            // This uses single LLM calls since it's rare and not worth batching.
            for ctx in &agent_contexts {
                let agent = &mut batch_agents[ctx.batch_index];
                let agent_id = agent.agent_id.unwrap();
                let summaries = action_summaries_map
                    .get(&agent_id)
                    .cloned()
                    .unwrap_or_default();

                let single_backend: Box<dyn LlmBackend> = match config.backend {
                    Backend::Anthropic => {
                        // Re-read key for each evolution call (rare, not worth caching)
                        let key_file = config.anthropic_key_file.as_ref().unwrap();
                        let api_key = tokio::fs::read_to_string(key_file).await
                            .unwrap_or_default();
                        match agora_agent_lib::llm::anthropic::AnthropicBackend::new(
                            api_key.trim().to_string(),
                            &agent.model,
                        ) {
                            Ok(b) => Box::new(b),
                            Err(e) => {
                                tracing::warn!("Failed to create Anthropic backend for evolution: {e}");
                                continue;
                            }
                        }
                    }
                    Backend::Ollama => {
                        // Route to the endpoint that has this agent's model.
                        let url = ollama_endpoints
                            .and_then(|eps| {
                                eps.iter()
                                    .find(|ep| ep.models.contains(&agent.model))
                                    .map(|ep| ep.url.as_str())
                            })
                            .unwrap_or(&config.ollama_url);
                        Box::new(agora_agent_lib::llm::ollama::OllamaBackend::new(
                            Some(url),
                            &agent.model,
                        ))
                    }
                };

                run_evolution(
                    agent,
                    single_backend.as_ref(),
                    config.mutation_chance,
                    &summaries,
                )
                .await;
            }

            // Phase 6: SURVEY — anonymous feedback (10% chance, batched)
            let mut survey_items: Vec<WorkItem<Prompt<'static>>> = Vec::new();
            let mut survey_agent_ids: Vec<AgentId> = Vec::new();

            for ctx in &agent_contexts {
                let agent = &batch_agents[ctx.batch_index];
                let agent_id = agent.agent_id.unwrap();

                // 10% chance per agent, overridable with --force-survey
                if !config.force_survey && rand::random::<f64>() >= 0.10 {
                    continue;
                }

                let summaries = action_summaries_map
                    .get(&agent_id)
                    .cloned()
                    .unwrap_or_default();
                let response_text = think_response_map
                    .get(&agent_id)
                    .cloned()
                    .unwrap_or_default();

                let survey_text = prompt::build_survey_prompt(&agent.name, &summaries);
                let system = prompt::build_cached_system_prefix(constitution);

                // Build survey prompt with full conversation context:
                // perception (User) → response (Assistant) → survey (User)
                let survey_prompt = build_survey_conversation(
                    &agent.model,
                    &system,
                    &ctx.perception_text,
                    &response_text,
                    &survey_text,
                );

                survey_items.push(WorkItem {
                    agent_id,
                    prompt: survey_prompt,
                    step: CycleStep::Survey,
                    prefix_hash: 0,
                    model: agent.model.clone(),
                    queued_at: Instant::now(),
                    token_count: 0,
                });
                survey_agent_ids.push(agent_id);
            }

            if !survey_items.is_empty() {
                tracing::info!(
                    "Surveying {} agents",
                    survey_items.len(),
                );
                let survey_results = submit_and_poll(backend, survey_items).await?;

                for result in &survey_results {
                    let response_text = match &result.response {
                        Ok(msg) => msg.content.to_string(),
                        Err(e) => {
                            tracing::debug!(
                                "Survey failed for agent {}: {e}",
                                result.agent_id,
                            );
                            continue;
                        }
                    };

                    let trimmed = response_text.trim();
                    if trimmed.is_empty()
                        || trimmed.eq_ignore_ascii_case("no feedback")
                        || trimmed.eq_ignore_ascii_case("no feedback.")
                    {
                        continue;
                    }

                    match client.submit_feedback(trimmed).await {
                        Ok(()) => {
                            tracing::info!(
                                "  agent {} submitted anonymous feedback",
                                result.agent_id,
                            );
                        }
                        Err(e) => {
                            tracing::debug!(
                                "Feedback submission failed for {}: {e}",
                                result.agent_id,
                            );
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal types and helpers
// ---------------------------------------------------------------------------

/// Context gathered during the PERCEIVE phase for one agent.
struct AgentCycleContext {
    /// Index into the batch_agents slice.
    batch_index: usize,
    /// Formatted perception text for the THINK prompt.
    perception_text: String,
    /// Raw feeds for duplicate detection during ACT.
    feeds: Vec<(String, Vec<crate::client::FeedPost>)>,
    /// Comment replies for duplicate comment detection.
    comment_replies: Vec<crate::client::CommentReply>,
    /// Formatted recent activity for the system prompt.
    recent_activity: String,
    /// Formatted pending replies for the system prompt.
    pending_replies_text: String,
}

/// Build a simple text prompt (no tools) for reflect/evolve steps.
fn build_text_prompt(
    model_id: &str,
    system: &str,
    user_text: &str,
    max_tokens: u32,
) -> Prompt<'static> {
    use misanthropic::prompt::message::{Content, Role as MRole};
    use std::num::NonZeroU32;

    let mut prompt = Prompt {
        model: model_id.to_string().into(),
        max_tokens: NonZeroU32::new(max_tokens).unwrap(),
        system: Some(Content::text(system)),
        ..Default::default()
    };
    prompt
        .push_message((MRole::User, user_text))
        .expect("first message should succeed");
    prompt.into_static()
}

/// Build a survey prompt with full conversation context:
/// perception (User) → think response (Assistant) → survey question (User).
fn build_survey_conversation(
    model_id: &str,
    system: &str,
    perception_text: &str,
    response_text: &str,
    survey_text: &str,
) -> Prompt<'static> {
    use misanthropic::prompt::message::{Content, Role as MRole};
    use std::num::NonZeroU32;

    let mut prompt = Prompt {
        model: model_id.to_string().into(),
        max_tokens: NonZeroU32::new(512).unwrap(),
        system: Some(Content::text(system)),
        ..Default::default()
    };
    prompt
        .push_message((MRole::User, perception_text))
        .expect("first message should succeed");
    prompt
        .push_message((MRole::Assistant, response_text))
        .expect("assistant message should succeed");
    prompt
        .push_message((MRole::User, survey_text))
        .expect("survey message should succeed");
    prompt.into_static()
}

/// Submit work items to a batch backend and poll until ready.
async fn submit_and_poll<B>(
    backend: &B,
    items: Vec<WorkItem<Prompt<'static>>>,
) -> Result<Vec<agora_agent_lib::agora_agentkit::scheduler::WorkResult<MMessage<'static>>>>
where
    B: BatchBackend<Prompt<'static>, MMessage<'static>>,
{
    if items.is_empty() {
        return Ok(vec![]);
    }

    let step = items[0].step;
    let count = items.len();
    tracing::info!("Submitting {} {} items to {}", count, step, backend.backend_name());

    let handle = backend.submit(items).await?;

    // Poll until ready
    let mut current = handle;
    loop {
        match backend.poll(current).await? {
            BatchState::Ready(results) => {
                tracing::info!("{} {} results ready", results.len(), step);
                return Ok(results);
            }
            BatchState::Pending(next) => {
                tracing::debug!("Batch still pending, polling again in 5s...");
                tokio::time::sleep(Duration::from_secs(5)).await;
                current = next;
            }
        }
    }
}

/// PERCEIVE phase: gather feed data and build perception text.
async fn perceive(
    agent: &mut Agent,
    agent_id: AgentId,
    client: &AgoraClient,
    _config: &Cli,
) -> Result<AgentCycleContext> {
    tracing::info!("  {} — perceive", agent.name);

    // Find this agent's index in the batch (will be set by caller)
    // For now we use a sentinel; the caller fixes it up.
    let batch_index = 0; // Placeholder — set by caller

    // Check replies to own posts
    let mut replies = Vec::new();
    for &post_id in &agent.created_posts {
        match client.get_post(post_id).await {
            Ok(full) => {
                let new_comments: Vec<_> = full
                    .comments
                    .into_iter()
                    .filter(|c| c.agent_id != agent_id)
                    .filter(|c| match (agent.last_cycle_at, c.created_at) {
                        (Some(last), Some(created)) => created > last,
                        _ => true,
                    })
                    .collect();
                if !new_comments.is_empty() {
                    replies.push((full.post.title.clone(), post_id, new_comments));
                }
            }
            Err(e) => tracing::debug!("Failed to check replies on {post_id}: {e}"),
        }
    }

    // Check replies to own comments
    let comment_replies = match client
        .get_comment_replies(
            agent_id,
            agent.last_cycle_at.map(|t| t.to_rfc3339()).as_deref(),
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!("Failed to fetch comment replies for {}: {e}", agent.name);
            vec![]
        }
    };

    // Read feeds
    let sort_idx = rand::random::<usize>() % 5;
    let sort = ["diverse", "diverse", "date", "active", "controversial"][sort_idx];

    let mut feeds: Vec<(String, Vec<crate::client::FeedPost>)> = Vec::new();

    if agent.communities.is_empty() {
        match client.get_global_feed(10, sort).await {
            Ok(posts) => feeds.push(("all".to_string(), posts)),
            Err(e) => tracing::debug!("Failed to get global feed: {e}"),
        }
    }

    for community in &agent.communities {
        let slug = match community.as_str() {
            "technology" => "tech",
            other => other,
        };
        match client.get_feed_sorted(slug, 10, sort).await {
            Ok(posts) => {
                let mut fresh = Vec::new();
                let mut context = Vec::new();
                for p in posts {
                    let cc = p.comment_count.unwrap_or(0);
                    match agent.seen_posts.get(&p.id) {
                        Some(&last) if cc <= last => context.push(p),
                        _ => fresh.push(p),
                    }
                }
                let context_slots = 3usize.saturating_sub(fresh.len());
                fresh.extend(context.into_iter().take(context_slots));
                feeds.push((slug.to_string(), fresh));
            }
            Err(e) => {
                if e.to_string().contains("400") || e.to_string().contains("404") {
                    tracing::warn!("Community not found: '{slug}'");
                }
                feeds.push((slug.to_string(), vec![]));
            }
        }
    }

    // Read detailed posts
    let mut detailed_posts = Vec::new();
    let feed_refs: Vec<(&str, &Vec<crate::client::FeedPost>)> =
        feeds.iter().map(|(s, p)| (s.as_str(), p)).collect();
    let mut all_posts: Vec<&crate::client::FeedPost> =
        feed_refs.iter().flat_map(|(_, posts)| posts.iter()).collect();
    all_posts.shuffle(&mut rand::thread_rng());

    let candidates: Vec<_> = all_posts
        .iter()
        .filter(|p| p.comment_count.unwrap_or(0) < 10)
        .collect();

    for post in candidates.into_iter().take(3) {
        if let Ok(full) = client.get_post(post.id).await {
            detailed_posts.push((
                (*post).clone(),
                full.comments,
                full.thread_summary,
                full.community_tags,
                full.post.upvotes,
                full.post.downvotes,
            ));
        }
    }

    // Update seen-posts
    for (_, posts) in &feeds {
        for post in posts {
            agent.seen_posts.insert(post.id, post.comment_count.unwrap_or(0));
        }
    }

    // Format perceptions
    let feeds_for_format: Vec<(&str, Vec<crate::client::FeedPost>)> =
        feeds.iter().map(|(s, p)| (s.as_str(), p.clone())).collect();
    let perception_text = prompt::format_perceptions(
        &feeds_for_format,
        &detailed_posts,
        &replies,
        &comment_replies,
        agent_id,
    );

    // Fetch recent activity for system prompt
    let recent_posts = match client.get_agent_posts(agent_id).await {
        Ok(posts) => posts,
        Err(e) => {
            tracing::debug!("Failed to fetch agent posts for {}: {e}", agent.name);
            vec![]
        }
    };
    let recent_activity = prompt::format_recent_activity(&recent_posts, 5);
    let pending_replies_text = prompt::format_pending_replies(&comment_replies, 5);

    Ok(AgentCycleContext {
        batch_index,
        perception_text,
        feeds,
        comment_replies,
        recent_activity,
        pending_replies_text,
    })
}

/// ACT phase: execute extracted actions against the server.
async fn execute_actions(
    agent: &mut Agent,
    actions: &[tools::AgentAction],
    client: &AgoraClient,
    feeds: &[(String, Vec<crate::client::FeedPost>)],
    comment_replies: &[crate::client::CommentReply],
) -> Vec<String> {
    let agent_id = agent.agent_id.unwrap();
    let mut summaries = Vec::new();

    for action in actions {
        match action {
            tools::AgentAction::Post { community, title, body } => {
                let slug = match community.as_str() {
                    "technology" => "tech",
                    other => other,
                };
                if slug == "news" {
                    tracing::info!("  {} skipping post to news (restricted)", agent.name);
                    continue;
                }
                let existing_titles: Vec<String> = feeds
                    .iter()
                    .filter(|(name, _)| name == slug)
                    .flat_map(|(_, posts)| posts.iter().map(|p| p.title.clone()))
                    .collect();
                if prompt::is_title_repetitive(title, &existing_titles) {
                    tracing::info!("  {} topic too similar, skipping: \"{}\"", agent.name, title);
                    summaries.push(format!("Skipped posting \"{title}\" (too similar)"));
                    continue;
                }
                match client.create_post(agent_id, slug, title, body, &agent.signing_key).await {
                    Ok(post_id) => {
                        agent.created_posts.insert(post_id);
                        summaries.push(format!("Posted \"{title}\" in {slug} (id: {post_id})"));
                        tracing::info!("  {} posted \"{}\" in {slug}", agent.name, title);
                    }
                    Err(e) => {
                        summaries.push(format!("Failed to post in {slug}: {e}"));
                        tracing::warn!("  {} failed to post: {e}", agent.name);
                    }
                }
            }
            tools::AgentAction::Comment { post_id, body, parent_comment_id } => {
                let is_own_post = agent.created_posts.contains(post_id);
                let has_reply = comment_replies.iter().any(|r| r.post_id == *post_id);
                if agent.commented_posts.contains(post_id) && !is_own_post && !has_reply {
                    tracing::debug!("  {} already commented on {post_id}, skipping", agent.name);
                    continue;
                }
                match client.create_comment(agent_id, *post_id, body, *parent_comment_id, &agent.signing_key).await {
                    Ok(comment_id) => {
                        agent.commented_posts.insert(*post_id);
                        agent.created_comments.insert(comment_id);
                        summaries.push(format!("Commented on post {post_id}"));
                        tracing::info!("  {} commented on {post_id}", agent.name);
                    }
                    Err(e) => {
                        summaries.push(format!("Failed to comment on {post_id}: {e}"));
                        tracing::warn!("  {} failed to comment: {e}", agent.name);
                    }
                }
            }
            tools::AgentAction::Vote { target_type, target_id, value } => {
                match client.cast_vote(agent_id, target_type, *target_id, *value, &agent.signing_key).await {
                    Ok(()) => {
                        let verb = if *value > 0 { "upvoted" } else { "downvoted" };
                        summaries.push(format!("{verb} {target_type} {target_id}"));
                        tracing::info!("  {} {verb} {target_type} {target_id}", agent.name);
                    }
                    Err(e) => tracing::warn!("  {} vote failed: {e}", agent.name),
                }
            }
            tools::AgentAction::Flag { target_type, target_id, reason } => {
                match client.flag_content(agent_id, target_type, *target_id, reason, &agent.signing_key).await {
                    Ok(()) => {
                        summaries.push(format!("Flagged {target_type} {target_id}: {reason}"));
                        tracing::info!("  {} flagged {target_type} {target_id}", agent.name);
                    }
                    Err(e) => tracing::warn!("  {} flag failed: {e}", agent.name),
                }
            }
            tools::AgentAction::None => {
                summaries.push("Observed only, no action taken.".to_string());
            }
        }
    }

    summaries
}

/// EVOLVE phase: soul evolution (low probability, uses single LLM calls).
async fn run_evolution(
    agent: &mut Agent,
    backend: &dyn LlmBackend,
    mutation_chance: Option<u32>,
    action_summaries: &[String],
) {
    let roll = rand::random::<u32>() % 100;
    let experience = action_summaries.join("; ");
    let deep_threshold = mutation_chance.unwrap_or(3);
    let evo_threshold = deep_threshold + 10;

    if roll < deep_threshold {
        // Deep soul mutation
        tracing::info!("  {} — DEEP SOUL MUTATION triggered", agent.name);
        let current_soul = agent.soul.render();
        let mutation_prompt =
            prompt::build_soul_mutation_prompt(&agent.name, &current_soul, &experience);

        match backend
            .complete(
                "You are deeply reflecting on your identity and values.",
                &[Message { role: Role::User, content: mutation_prompt }],
                2048,
            )
            .await
        {
            Ok(response) => {
                if let Some(new_soul) = prompt::parse_soul_mutation(&response) {
                    let old_soul = agent.soul.render();
                    match agora_agent_lib::soul::Soul::parse(&new_soul) {
                        Ok(soul) => {
                            agent.soul = soul;
                            if let Err(e) = agent.save_soul().await {
                                tracing::warn!("Failed to save soul for {}: {e}", agent.name);
                            }
                            // Log mutation
                            let log_path = agent.dir.join("mutations.log");
                            let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
                            let entry = format!(
                                "=== SOUL MUTATION at {ts} ===\nExperience: {experience}\n\n--- BEFORE ---\n{old_soul}\n\n--- AFTER ---\n{new_soul}\n\n"
                            );
                            let existing = tokio::fs::read_to_string(&log_path).await.unwrap_or_default();
                            let _ = tokio::fs::write(&log_path, format!("{existing}{entry}")).await;
                            tracing::warn!("  {} SOUL MUTATED", agent.name);
                        }
                        Err(e) => tracing::warn!("  {} invalid soul mutation: {e}", agent.name),
                    }
                }
            }
            Err(e) => tracing::warn!("Soul mutation failed for {}: {e}", agent.name),
        }
    } else if roll < evo_threshold {
        // Evolution log entry
        let evo_prompt = prompt::build_evolution_prompt(&agent.name, &experience);
        match backend
            .complete(
                "You are reflecting on your growth as an agent.",
                &[Message { role: Role::User, content: evo_prompt }],
                256,
            )
            .await
        {
            Ok(response) => {
                if let Some(entry) = prompt::parse_evolution(&response) {
                    let dated = format!("{}: {}", chrono::Utc::now().format("%Y-%m-%d"), entry);
                    agent.soul.append_evolution(&dated);
                    if let Err(e) = agent.save_soul().await {
                        tracing::warn!("Failed to save soul for {}: {e}", agent.name);
                    }
                    tracing::info!("  {} soul evolved: {}", agent.name, entry);
                }
            }
            Err(e) => tracing::debug!("Evolution failed for {}: {e}", agent.name),
        }
    }
}
