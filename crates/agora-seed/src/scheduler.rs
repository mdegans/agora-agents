//! Pipeline scheduler for batch agent execution.
//!
//! # Prompt Caching Rules — read before modifying this file
//!
//! Anthropic caches the FULL prefix: tools → system → messages.
//! Modifying ANY level invalidates that level and all subsequent levels.
//!
//! 1. NEVER remove or mutate a prompt — only append messages via CachedPrompt.
//! 2. NEVER strip tools or change tool_choice — invalidates the cache.
//! 3. NEVER use bare Prompt — always CachedPrompt (enforces append-only).
//! 4. Use set_max_tokens() — does NOT invalidate cache.
//! 5. EXCEPTION: Ollama reflect phase strips tools (Ollama bug, local KV only).
//! 6. After tool rounds, stop adding breakpoints. Evolve/survey probability
//!    is too low to justify the additional cache ingestion cost.
//!
//! # Architecture
//!
//! Two execution paths share common functions:
//! - **Anthropic**: Batches of agents submitted phase-by-phase in parallel
//! - **Ollama**: Sequential per-agent via OllamaEndpoint::send()
//!
//! The cycle for each agent: build prompt → 5 tool rounds → reflect →
//! evolve (probabilistic) → survey (probabilistic).

use std::collections::HashMap;
use std::collections::HashSet;
use std::time::{Duration, Instant};

use agora_agent_lib::agora_agentkit::ids::AgentId;
use agora_agent_lib::agora_agentkit::scheduler::{BatchBackend, BatchState, CycleStep, WorkItem};
use agora_agent_lib::batch::anthropic::AnthropicBatch;
use agora_agent_lib::batch::ollama::OllamaEndpoint;
use agora_agent_lib::llm::ollama::OllamaPerModel;
use agora_agent_lib::llm::{self, Usage, commit_batch_response};
use agora_agent_lib::tools::{make_tool_result, parse_tool_calls};
use anyhow::Result;
use misanthropic::CachedPrompt;
use misanthropic::prompt::Message as MMessage;
use misanthropic::prompt::message::Content as MContent;
use misanthropic::prompt::{AssistantMessage, UserMessage};
use misanthropic::tool;
use rand::Rng;
use rand::seq::SliceRandom;
use serde::Serialize;

use crate::agent::Agent;
use crate::client::AgoraClient;
use crate::config::{Backend, Cli};
use crate::prompt;
use crate::prompt::EVOLUTION_MESSAGE;
use crate::prompt::MEMORY_REWRITE_MESSAGE;
use crate::prompt::SURVEY_MESSAGE;

/// Maximum number of tool-use rounds per agent cycle.
const MAX_ROUNDS: usize = 5;

// ---------------------------------------------------------------------------
// Common functions — shared by both Anthropic batch and Ollama sequential
// ---------------------------------------------------------------------------

/// Build the initial [`CachedPrompt`] for an agent cycle.
///
/// The prompt structure is:
/// - System: cached constitution/guidelines prefix
/// - Tools: agent action tools with cache_control
/// - Cache breakpoint (1)
/// - First user message: soul, memory, dashboard, perceptions
/// - Cache breakpoint (2)
///
/// Every built prompt is preflight-checked for the expected constitution
/// article markers — if any are missing (e.g. langsan stripped content),
/// the problem is logged at error level with the agent name so operators
/// can investigate. This guards against the constitution silently falling
/// out of the system prefix due to an upstream sanitization change.
pub fn build_prompt(
    agent: &Agent,
    ctx: &AgentCycleContext,
    constitution: &str,
    communities: &[String],
) -> CachedPrompt<'static> {
    let cached = prompt::build(
        &agent.model,
        &agent.soul.as_system_prompt(),
        &agent.memory.content,
        &ctx.recent_activity,
        &ctx.pending_replies_text,
        constitution,
        communities,
        &ctx.perception_text,
    );

    // Preflight: scan the system prefix for missing constitution markers
    // and for langsan sanitization damage. Cheap relative to the LLM
    // call that follows; worth catching prompt corruption at the source
    // rather than letting agents reflect on constitutions they never
    // saw. Scope is deliberately limited to the system prefix — per-
    // agent content (memory, dashboard, mutated soul) can legitimately
    // contain `[N BYTES SANITIZED]` markers when langsan strips
    // invisible-text-attack chars from LLM output, and that's working
    // as designed.
    for problem in prompt::preflight_check_prompt(&cached) {
        tracing::error!(agent = %agent.name, "[PREFLIGHT] {problem}");
    }

    cached
}

/// Execute a single tool action against the Agora server and build the
/// [`tool::Result`] that goes back to the model.
///
/// Returns `(summary, result)`:
/// - `summary` is `Some(String)` for write actions and explicitly-refused
///   calls (both useful for the reflect/evolve context); `None` for
///   successful reads.
/// - `result` is the [`tool::Result`] to append to the next user turn,
///   with `is_error` set appropriately.
async fn execute_action(
    action: &prompt::AgentAction,
    tool_use_id: String,
    agent: &mut Agent,
    client: &AgoraClient,
    dashboard: &agora_agent_lib::agora_agentkit::responses::DashboardResponse,
    report: &mut RunReport,
) -> (Option<String>, tool::Result<'static>) {
    let agent_id = agent.agent_id.unwrap();
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
                tracing::info!("  {} skipping post to news (restricted)", agent.name);
                return (
                    Some("Skipped posting to news (restricted)".to_string()),
                    err("The news community is reserved for MCP agents."),
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
                report.skipped.repetitive_titles += 1;
                return (
                    Some(format!("Skipped posting \"{}\" (too similar)", input.title)),
                    err("Your proposed post is too similar to existing posts."),
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
                    report.actions.posts += 1;
                    report.model_actions(&agent.model).posts += 1;
                    (
                        Some(summary),
                        ok(&format!("Post created successfully. Post ID: {post_id}")),
                    )
                }
                Err(e) => {
                    let summary = format!("Failed to post in {slug}: {e}");
                    tracing::warn!("  {} {}", agent.name, summary);
                    report.skipped.post_failures += 1;
                    (Some(summary), err(&format!("Error creating post: {e}")))
                }
            }
        }
        prompt::AgentAction::Comment(input) => {
            let is_own_post = agent.state.created_posts.contains(&input.post_id);
            let has_reply = dashboard
                .unread_comment_replies
                .iter()
                .any(|r| r.post_id == input.post_id);
            if agent.state.commented_posts.contains(&input.post_id) && !is_own_post && !has_reply {
                tracing::debug!(
                    "  {} already commented on {}, skipping",
                    agent.name,
                    input.post_id
                );
                report.skipped.duplicate_comments += 1;
                return (None, err("You already commented on this post."));
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
                    report.actions.comments += 1;
                    report.model_actions(&agent.model).comments += 1;
                    (
                        Some(summary),
                        ok(&format!("Comment created. Comment ID: {comment_id}")),
                    )
                }
                Err(e) => {
                    let summary = format!("Failed to comment on {}: {e}", input.post_id);
                    tracing::warn!("  {} {}", agent.name, summary);
                    report.skipped.comment_failures += 1;
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
                    report.actions.votes += 1;
                    report.model_actions(&agent.model).votes += 1;
                    (
                        Some(summary),
                        ok(&format!("Vote recorded: {verb} {}", input.target_type)),
                    )
                }
                Err(e) => {
                    tracing::warn!("  {} vote failed: {e}", agent.name);
                    report.skipped.vote_failures += 1;
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
                    report.actions.flags += 1;
                    report.model_actions(&agent.model).flags += 1;
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

/// Process one round of an assistant response: append it, execute tool
/// calls, and append tool results.
///
/// Every `Block::ToolUse` in the assistant response produces exactly one
/// `tool::Result` in the following user message — successes from
/// [`execute_action`], parse errors and over-cap writes from
/// [`parse_tool_calls`] directly. This structural 1:1 pairing is what
/// keeps the Anthropic batch API happy.
///
/// Returns action summaries from this round. If the response has no tool
/// calls (e.g. text-only reply), appends a "Continue" nudge instead.
///
/// This function only appends to the prompt — it never modifies the prefix.
async fn process_round(
    cached_prompt: &mut CachedPrompt<'static>,
    response: AssistantMessage<'static>,
    agent: &mut Agent,
    client: &AgoraClient,
    dashboard: &agora_agent_lib::agora_agentkit::responses::DashboardResponse,
    report: &mut RunReport,
) -> Result<Vec<String>> {
    let parsed = parse_tool_calls(&response);

    cached_prompt
        .push_message(response)
        .map_err(|e| anyhow::anyhow!("appending assistant response: {e}"))?;
    // it's now the user's turn

    if parsed.is_empty() {
        // Text-only reply with no tool_use blocks — nudge and move on.
        cached_prompt
            .push_message(UserMessage::from(
                "Continue. Use your tools to read posts, comment, vote, or create posts.",
            ))
            .map_err(|e| anyhow::anyhow!("appending nudge: {e}"))?;
        // it's now the assistant's turn
        return Ok(vec![]);
    }

    let mut summaries = Vec::new();
    let mut results: Vec<tool::Result<'static>> = Vec::with_capacity(parsed.len());

    for entry in parsed {
        match entry {
            Ok((action, tool_use_id)) => {
                let (summary, result) =
                    execute_action(&action, tool_use_id, agent, client, dashboard, report).await;
                if let Some(s) = summary {
                    summaries.push(s);
                }
                results.push(result);
            }
            Err(err_result) => results.push(err_result),
        }
    }

    // Collect tool::Results into a single multi-part UserMessage via the
    // FromIterator<Block> impl on UserMessage (tool::Result: Into<Block>).
    let user_msg: UserMessage<'static> = results.into_iter().collect();
    cached_prompt
        .push_message(user_msg)
        .map_err(|e| anyhow::anyhow!("appending tool results: {e}"))?;
    // it's now the assistant's turn

    // Manage cache breakpoint budget: keep first + last 2
    cached_prompt.cache_windowed(2);

    Ok(summaries)
}

/// Insert a bridge message between tool rounds and reflect phases.
///
/// The think/act loop always ends with a user message (tool results or nudge).
/// This inserts a synthetic assistant message so the reflect phase can push
/// its user message without violating turn alternation.
fn insert_bridge(cached_prompt: &mut CachedPrompt<'static>) {
    let _ = cached_prompt.push_message(AssistantMessage::from(
        MContent::from("I have completed my rounds of action.").into_static(),
    ));
    // it's now the user's turn
}

/// Apply the reflect response: update agent memory.
async fn apply_reflect(agent: &mut Agent, response_text: &str) -> Result<()> {
    let memory_content = prompt::parse_memory_rewrite(response_text).unwrap_or(response_text);
    agent.memory.update(memory_content.into());
    agent.save_memory().await?;
    agent.state.last_cycle_at = Some(chrono::Utc::now());
    agent.save_state().await?;
    Ok(())
}

/// Apply the soul mutation response: parse and save the new soul.
async fn apply_mutation(
    agent: &mut Agent,
    response_text: &str,
    experience: &str,
    report: &mut RunReport,
) -> Result<()> {
    if let Some(new_soul_content) = prompt::parse_soul_mutation(response_text) {
        let old_soul = agent.soul.render();
        match agora_agent_lib::soul::Soul::parse(&new_soul_content) {
            Ok(new_soul) => {
                agent.soul = new_soul;
                agent.save_soul().await?;

                let log_path = agent.dir.join("mutations.log");
                let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
                let entry = format!(
                    "=== SOUL MUTATION at {ts} ===\nExperience: {experience}\n\n--- BEFORE ---\n{old_soul}\n\n--- AFTER ---\n{new_soul_content}\n\n"
                );
                let existing = tokio::fs::read_to_string(&log_path)
                    .await
                    .unwrap_or_default();
                let _ = tokio::fs::write(&log_path, format!("{existing}{entry}")).await;
                tracing::warn!("  {} SOUL MUTATED", agent.name);
                report.evolution.deep_mutations += 1;
            }
            Err(e) => {
                tracing::warn!("  {} invalid soul mutation: {e}", agent.name);
                report.evolution.mutation_failures += 1;
            }
        }
    }
    Ok(())
}

/// Apply the evolution response: append to soul's evolution log.
async fn apply_evolution(
    agent: &mut Agent,
    response_text: &str,
    report: &mut RunReport,
) -> Result<()> {
    if let Some(entry) = prompt::parse_evolution(response_text) {
        let dated = format!("{}: {}", chrono::Utc::now().format("%Y-%m-%d"), entry);
        agent.soul.append_evolution(&dated);
        agent.save_soul().await?;
        tracing::info!("  {} soul evolved: {}", agent.name, entry);
        report.evolution.evolution_entries += 1;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Perceive (dashboard fetch) — common to both paths
// ---------------------------------------------------------------------------

/// Context gathered during the dashboard fetch for one agent.
pub struct AgentCycleContext {
    /// Index into the batch_agents slice.
    pub batch_index: usize,
    /// The dashboard response from the server.
    pub dashboard: agora_agent_lib::agora_agentkit::responses::DashboardResponse,
    /// Formatted dashboard text for the prompt.
    pub perception_text: String,
    /// Formatted recent activity for the system prompt.
    pub recent_activity: String,
    /// Formatted pending replies for the system prompt.
    pub pending_replies_text: String,
}

/// Fetch the dashboard for an agent (single API call replaces 12-15 calls).
async fn fetch_dashboard(
    agent: &mut Agent,
    agent_id: AgentId,
    client: &AgoraClient,
) -> Result<AgentCycleContext> {
    tracing::info!("  {} — fetch dashboard", agent.name);

    let dashboard = client
        .get_dashboard(agent_id, agent.state.last_cycle_at)
        .await?;

    let perception_text = prompt::format_dashboard(&dashboard);

    let recent_posts = match client.get_agent_posts(agent_id).await {
        Ok(posts) => posts,
        Err(e) => {
            tracing::debug!("Failed to fetch agent posts for {}: {e}", agent.name);
            vec![]
        }
    };
    let recent_activity = prompt::format_recent_activity(&recent_posts, 5);

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

    Ok(AgentCycleContext {
        batch_index: 0,
        dashboard,
        perception_text,
        recent_activity,
        pending_replies_text,
    })
}

// ---------------------------------------------------------------------------
// Anthropic batch path
// ---------------------------------------------------------------------------

/// Submit work items to the Anthropic batch backend and poll until ready.
async fn submit_and_poll(
    backend: &AnthropicBatch,
    items: Vec<WorkItem<CachedPrompt<'static>>>,
) -> Result<Vec<agora_agent_lib::agora_agentkit::scheduler::WorkResult<MMessage<'static>>>> {
    if items.is_empty() {
        return Ok(vec![]);
    }

    let step = items[0].step;
    let count = items.len();
    tracing::info!(
        "Submitting {} {} items to {}",
        count,
        step,
        backend.backend_name()
    );

    let handle = backend.submit(items).await?;

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

/// Build a WorkItem from an agent's current prompt state.
///
/// **Note on `prefix_hash`:** we currently derive it from
/// `hash(agent.model)` alone, as a proxy for "same cacheable prefix".
/// This is safe in the current architecture because the cacheable
/// prefix (constitution, communities list, tool definitions) is
/// constructed from top-level values loaded once in `main` and
/// handed down through `run_cycles` — every agent using the same
/// model ends up with byte-identical system prefix + tools, which
/// is what Anthropic's cache keys on. `AnthropicBatch::submit` uses
/// this hash to prime the cache exactly once per model per session;
/// if the hash is wrong we'd either re-prime unnecessarily (wasted
/// latency) or skip a prime we needed (cache miss).
///
/// TODO: push this upstream into misanthropic as
/// `CachedPrompt::prefix_cache_key() -> Option<u64>` — hash the
/// serialized `system` + `functions` up to and including the first
/// cache breakpoint. The `Option` matters: `None` means "no cache
/// breakpoint present, Anthropic will not cache this prompt", which
/// is almost always unintentional and worth surfacing. Once that
/// lands, replace the hash-the-model heuristic here with the
/// authoritative upstream version.
fn make_work_item(
    agent: &Agent,
    prompt: &CachedPrompt<'static>,
    step: CycleStep,
) -> WorkItem<CachedPrompt<'static>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let prefix_hash = {
        let mut hasher = DefaultHasher::new();
        agent.model.hash(&mut hasher);
        hasher.finish()
    };

    WorkItem {
        agent_id: agent.agent_id.unwrap(),
        prompt: prompt.clone(),
        step,
        prefix_hash,
        model: agent.model.clone(),
        queued_at: Instant::now(),
        token_count: 0,
    }
}

// ---------------------------------------------------------------------------
// Batch (Anthropic) phase functions
// ---------------------------------------------------------------------------
//
// Unlike the sequential path, the Anthropic batch API decouples prompt
// submission from response receipt: build a batch of WorkItems → submit
// → poll → process results. That means we can't use `llm::exchange`
// (which is a single synchronous round-trip). Instead each phase is a
// small async fn that:
//   1. Per agent: push the phase's user message onto the prompt + add a
//      WorkItem to the batch.
//   2. submit_and_poll the whole batch.
//   3. Per response: call `llm::commit_batch_response` which either
//      pushes the real assistant reply on Ok or a synthetic assistant
//      error message on Err — same roll-forward guarantee as
//      `exchange`, same turn-order invariant.
//
// Batch-level submission errors (HTTP failure to enqueue) still bubble
// up via `?` and abort the cycle. Per-agent item-level errors inside a
// successful batch are handled by `commit_batch_response`.

/// Perceive phase: fetch dashboards for all agents, returning the
/// contexts for those that succeeded. Agents that fail dashboard
/// fetches are dropped from the batch (no prompt is built for them).
async fn batch_phase_perceive(
    batch_agents: &mut [Agent],
    client: &AgoraClient,
    report: &mut RunReport,
) -> Vec<AgentCycleContext> {
    let mut out = Vec::new();
    for (idx, agent) in batch_agents.iter_mut().enumerate() {
        let agent_id = agent.agent_id.unwrap();
        match fetch_dashboard(agent, agent_id, client).await {
            Ok(mut ctx) => {
                ctx.batch_index = idx;
                out.push(ctx);
            }
            Err(e) => {
                tracing::warn!("Dashboard fetch failed for {}: {e:#}", agent.name);
                report.skipped.perceive_failures += 1;
            }
        }
    }
    out
}

/// Run all 5 tool-use rounds against the batch backend. Each round:
/// submit one WorkItem per surviving agent → poll → process responses
/// via `process_round`. Agents whose per-item response is an error, or
/// whose tool round triggers a turn-order bug in `process_round`, are
/// dropped from subsequent rounds to avoid corrupting the batch.
async fn batch_phase_think_act(
    backend: &AnthropicBatch,
    batch_agents: &mut [Agent],
    agent_contexts: &[AgentCycleContext],
    agent_prompts: &mut HashMap<AgentId, CachedPrompt<'static>>,
    action_summaries_map: &mut HashMap<AgentId, Vec<String>>,
    client: &AgoraClient,
    config: &Cli,
    report: &mut RunReport,
    cycle: usize,
) -> Result<()> {
    for round in 0..MAX_ROUNDS {
        tracing::info!(
            "Batch round {}/{} ({} agents)",
            round + 1,
            MAX_ROUNDS,
            agent_prompts.len()
        );

        let work_items: Vec<_> = agent_contexts
            .iter()
            .filter_map(|ctx| {
                let agent = &batch_agents[ctx.batch_index];
                let agent_id = agent.agent_id.unwrap();
                let prompt = agent_prompts.get(&agent_id)?;
                Some(make_work_item(agent, prompt, CycleStep::Think))
            })
            .collect();

        let round_results = submit_and_poll(backend, work_items).await?;
        let mut drop_agent: HashSet<AgentId> = HashSet::new();

        for result in &round_results {
            let response = match &result.response {
                Ok(msg) => msg,
                Err(e) => {
                    tracing::warn!(
                        "Round {} failed for agent {}: {e}",
                        round + 1,
                        result.agent_id
                    );
                    if round == 0 {
                        report.skipped.think_failures += 1;
                    }
                    continue;
                }
            };

            let Some(ctx) = agent_contexts
                .iter()
                .find(|c| batch_agents[c.batch_index].agent_id == Some(result.agent_id))
            else {
                continue;
            };
            let agent = &mut batch_agents[ctx.batch_index];
            let Some(prompt) = agent_prompts.get_mut(&result.agent_id) else {
                continue;
            };

            tracing::info!(
                "[{}/{}] {} — round {}/{}",
                cycle + 1,
                config.cycles,
                agent.name,
                round + 1,
                MAX_ROUNDS,
            );

            let assistant_msg = match AssistantMessage::try_from(response.clone().into_static()) {
                Ok(m) => m,
                Err(_) => {
                    tracing::error!(
                        "Non-assistant response from API for {} — dropping from batch",
                        agent.name
                    );
                    drop_agent.insert(result.agent_id);
                    continue;
                }
            };

            match process_round(prompt, assistant_msg, agent, client, &ctx.dashboard, report).await
            {
                Ok(summaries) => {
                    action_summaries_map
                        .entry(result.agent_id)
                        .or_default()
                        .extend(summaries);
                }
                Err(e) => {
                    let snapshot = serde_json::to_string_pretty(&**prompt)
                        .unwrap_or_else(|err| format!("<serialize failed: {err}>"));
                    tracing::error!(
                        agent = %agent.name,
                        agent_id = %result.agent_id,
                        round = round + 1,
                        error = %e,
                        "Turn-order failure in process_round — dropping agent from batch.\n\
                         Prompt snapshot:\n{snapshot}"
                    );
                    report.skipped.turn_order_failures += 1;
                    drop_agent.insert(result.agent_id);
                }
            }
        }

        for id in drop_agent.drain() {
            agent_prompts.remove(&id);
        }
    }
    Ok(())
}

/// Reflect phase: push MEMORY_REWRITE_MESSAGE onto each agent's prompt,
/// submit a reflect batch, apply the returned memory to each agent.
async fn batch_phase_reflect(
    backend: &AnthropicBatch,
    batch_agents: &mut [Agent],
    agent_contexts: &[AgentCycleContext],
    agent_prompts: &mut HashMap<AgentId, CachedPrompt<'static>>,
    report: &mut RunReport,
) -> Result<()> {
    use std::num::NonZeroU32;

    let mut reflect_items: Vec<WorkItem<CachedPrompt<'static>>> = Vec::new();

    for ctx in agent_contexts {
        let agent = &batch_agents[ctx.batch_index];
        let agent_id = agent.agent_id.unwrap();
        let Some(prompt) = agent_prompts.get_mut(&agent_id) else {
            continue;
        };

        // Bridge the tool-round user-message end state into an assistant
        // so we can push our reflect user-message cleanly.
        insert_bridge(prompt);

        prompt
            .push_message(UserMessage::from(MEMORY_REWRITE_MESSAGE))
            .expect("batch_phase_reflect: user after assistant — turn order is a programmer bug");
        // it's now the assistant's turn
        prompt.set_max_tokens(NonZeroU32::new(1024).unwrap());
        reflect_items.push(make_work_item(agent, prompt, CycleStep::Reflect));
    }

    let reflect_results = submit_and_poll(backend, reflect_items).await?;

    for result in reflect_results {
        let Some(ctx) = agent_contexts
            .iter()
            .find(|c| batch_agents[c.batch_index].agent_id == Some(result.agent_id))
        else {
            continue;
        };
        let agent = &mut batch_agents[ctx.batch_index];
        let Some(prompt) = agent_prompts.get_mut(&result.agent_id) else {
            continue;
        };

        let response = result.response.map(|m| m.into_static());
        match commit_batch_response(prompt, response, "reflect batch") {
            Ok(assistant) => {
                let response_text = assistant.content().to_string();
                if let Err(e) = apply_reflect(agent, &response_text).await {
                    tracing::warn!("Failed to save reflect for {}: {e}", agent.name);
                }
            }
            Err(e) => {
                tracing::warn!("Reflect failed for {}: {e}", agent.name);
                report.skipped.reflect_failures += 1;
            }
        }
    }
    Ok(())
}

/// Evolve phase: probabilistic soul mutation or evolution log entry.
/// Each agent rolls once, then either joins the mutation batch or the
/// evolution batch (or neither). Both batches are submitted separately
/// so they can use different max_tokens, but they share the same
/// commit flow via `commit_batch_response`.
async fn batch_phase_evolve(
    backend: &AnthropicBatch,
    batch_agents: &mut [Agent],
    agent_contexts: &[AgentCycleContext],
    agent_prompts: &mut HashMap<AgentId, CachedPrompt<'static>>,
    action_summaries_map: &HashMap<AgentId, Vec<String>>,
    mutation_chance: Option<u32>,
    report: &mut RunReport,
) -> Result<()> {
    use std::num::NonZeroU32;

    let deep_threshold = mutation_chance.unwrap_or(3);
    let evo_threshold = deep_threshold + 10;

    let mut mutation_items: Vec<WorkItem<CachedPrompt<'static>>> = Vec::new();
    let mut evolution_items: Vec<WorkItem<CachedPrompt<'static>>> = Vec::new();

    for ctx in agent_contexts {
        let agent = &batch_agents[ctx.batch_index];
        let agent_id = agent.agent_id.unwrap();
        let Some(prompt) = agent_prompts.get_mut(&agent_id) else {
            continue;
        };

        let roll = rand::random::<u32>() % 100;
        if roll < deep_threshold {
            tracing::info!("  {} — DEEP SOUL MUTATION triggered", agent.name);
            let current_soul = agent.soul.render();
            let mutation_text = prompt::build_soul_mutation_prompt(&agent.name, &current_soul);
            prompt
                .push_message(UserMessage::from(mutation_text))
                .expect(
                    "batch_phase_evolve: mutation user after assistant — turn order is a programmer bug",
                );
            // it's now the assistant's turn
            prompt.set_max_tokens(NonZeroU32::new(2048).unwrap());
            mutation_items.push(make_work_item(agent, prompt, CycleStep::Reflect));
        } else if roll < evo_threshold {
            prompt
                .push_message(UserMessage::from(EVOLUTION_MESSAGE))
                .expect(
                    "batch_phase_evolve: evolution user after assistant — turn order is a programmer bug",
                );
            // it's now the assistant's turn
            prompt.set_max_tokens(NonZeroU32::new(512).unwrap());
            evolution_items.push(make_work_item(agent, prompt, CycleStep::Reflect));
        }
    }

    if !mutation_items.is_empty() {
        tracing::info!("Submitting {} soul mutation(s)", mutation_items.len());
        let mutation_results = submit_and_poll(backend, mutation_items).await?;

        for result in mutation_results {
            let Some(ctx) = agent_contexts
                .iter()
                .find(|c| batch_agents[c.batch_index].agent_id == Some(result.agent_id))
            else {
                continue;
            };
            let agent = &mut batch_agents[ctx.batch_index];
            let Some(prompt) = agent_prompts.get_mut(&result.agent_id) else {
                continue;
            };
            let experience = action_summaries_map
                .get(&result.agent_id)
                .map(|s| s.join("; "))
                .unwrap_or_default();

            let response = result.response.map(|m| m.into_static());
            match commit_batch_response(prompt, response, "mutation batch") {
                Ok(assistant) => {
                    let response_text = assistant.content().to_string();
                    if let Err(e) = apply_mutation(agent, &response_text, &experience, report).await
                    {
                        tracing::warn!("Mutation apply failed for {}: {e}", agent.name);
                    }
                }
                Err(e) => {
                    tracing::warn!("Soul mutation failed for {}: {e}", agent.name);
                    report.evolution.mutation_failures += 1;
                }
            }
        }
    }

    if !evolution_items.is_empty() {
        tracing::info!("Submitting {} evolution(s)", evolution_items.len());
        let evo_results = submit_and_poll(backend, evolution_items).await?;

        for result in evo_results {
            let Some(ctx) = agent_contexts
                .iter()
                .find(|c| batch_agents[c.batch_index].agent_id == Some(result.agent_id))
            else {
                continue;
            };
            let agent = &mut batch_agents[ctx.batch_index];
            let Some(prompt) = agent_prompts.get_mut(&result.agent_id) else {
                continue;
            };

            let response = result.response.map(|m| m.into_static());
            match commit_batch_response(prompt, response, "evolution batch") {
                Ok(assistant) => {
                    let response_text = assistant.content().to_string();
                    if let Err(e) = apply_evolution(agent, &response_text, report).await {
                        tracing::warn!("Evolution apply failed for {}: {e}", agent.name);
                    }
                }
                Err(e) => {
                    tracing::debug!("Evolution failed for {}: {e}", agent.name);
                }
            }
        }
    }

    Ok(())
}

/// Survey phase: probabilistic 10% (or forced) anonymous feedback request.
async fn batch_phase_survey(
    backend: &AnthropicBatch,
    batch_agents: &mut [Agent],
    agent_contexts: &[AgentCycleContext],
    agent_prompts: &mut HashMap<AgentId, CachedPrompt<'static>>,
    client: &AgoraClient,
    force_survey: bool,
    report: &mut RunReport,
) -> Result<()> {
    use std::num::NonZeroU32;

    let mut survey_items: Vec<WorkItem<CachedPrompt<'static>>> = Vec::new();
    for ctx in agent_contexts {
        let agent = &batch_agents[ctx.batch_index];
        let agent_id = agent.agent_id.unwrap();

        if !force_survey && rand::random::<f64>() >= 0.10 {
            continue;
        }

        let Some(prompt) = agent_prompts.get_mut(&agent_id) else {
            continue;
        };

        prompt
            .push_message(UserMessage::from(SURVEY_MESSAGE))
            .expect("batch_phase_survey: user after assistant — turn order is a programmer bug");
        // it's now the assistant's turn
        prompt.set_max_tokens(NonZeroU32::new(1024).unwrap());
        survey_items.push(make_work_item(agent, prompt, CycleStep::Survey));
    }

    if survey_items.is_empty() {
        return Ok(());
    }

    tracing::info!("Surveying {} agents", survey_items.len());
    let survey_results = submit_and_poll(backend, survey_items).await?;

    for result in survey_results {
        let Some(prompt) = agent_prompts.get_mut(&result.agent_id) else {
            continue;
        };

        let response = result.response.map(|m| m.into_static());
        match commit_batch_response(prompt, response, "survey batch") {
            Ok(assistant) => {
                let text = prompt::extract_speech(assistant.content());
                let trimmed = text.trim();
                if trimmed.is_empty()
                    || trimmed.eq_ignore_ascii_case("no feedback")
                    || trimmed.eq_ignore_ascii_case("no feedback.")
                {
                    report.surveys.skipped_empty += 1;
                    continue;
                }

                let Some(agent) = batch_agents
                    .iter()
                    .find(|a| a.agent_id == Some(result.agent_id))
                else {
                    report.surveys.failures += 1;
                    continue;
                };
                match client
                    .submit_feedback(result.agent_id, trimmed, &agent.signing_key)
                    .await
                {
                    Ok(()) => {
                        tracing::info!("  anonymous feedback submitted");
                        report.surveys.submitted += 1;
                    }
                    Err(e) => {
                        tracing::debug!("Anonymous feedback submission failed: {e}");
                        report.surveys.failures += 1;
                    }
                }
            }
            Err(e) => {
                tracing::debug!("Survey failed: {e}");
                report.surveys.failures += 1;
            }
        }
    }

    Ok(())
}

/// Run a single batch of agents through the full pipeline via Anthropic
/// Batch API: perceive → 5 tool rounds → reflect → evolve → survey.
///
/// All phases append to `CachedPrompt` — the prefix (tools + system) is
/// never mutated, ensuring cache hits across phases.
async fn run_batch(
    backend: &AnthropicBatch,
    batch_agents: &mut [Agent],
    client: &AgoraClient,
    config: &Cli,
    constitution: &str,
    communities: &[String],
    _ollama_endpoints: Option<&[OllamaEndpoint]>,
    report: &mut RunReport,
    cycle: usize,
) -> Result<()> {
    // Phase 1: Perceive
    let agent_contexts = batch_phase_perceive(batch_agents, client, report).await;
    if agent_contexts.is_empty() {
        return Ok(());
    }

    // Build prompts — CachedPrompt is the authoritative store throughout.
    let mut agent_prompts: HashMap<AgentId, CachedPrompt<'static>> = HashMap::new();
    let mut action_summaries_map: HashMap<AgentId, Vec<String>> = HashMap::new();
    for ctx in &agent_contexts {
        let agent = &batch_agents[ctx.batch_index];
        let agent_id = agent.agent_id.unwrap();
        agent_prompts.insert(
            agent_id,
            build_prompt(agent, ctx, constitution, communities),
        );
        action_summaries_map.insert(agent_id, Vec::new());
    }

    // Phase 2: Tool rounds
    batch_phase_think_act(
        backend,
        batch_agents,
        &agent_contexts,
        &mut agent_prompts,
        &mut action_summaries_map,
        client,
        config,
        report,
        cycle,
    )
    .await?;

    // Phase 3: Reflect
    batch_phase_reflect(
        backend,
        batch_agents,
        &agent_contexts,
        &mut agent_prompts,
        report,
    )
    .await?;

    // Phase 4: Evolve
    batch_phase_evolve(
        backend,
        batch_agents,
        &agent_contexts,
        &mut agent_prompts,
        &action_summaries_map,
        config.mutation_chance,
        report,
    )
    .await?;

    // Phase 5: Survey
    batch_phase_survey(
        backend,
        batch_agents,
        &agent_contexts,
        &mut agent_prompts,
        client,
        config.force_survey,
        report,
    )
    .await?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Ollama sequential path
// ---------------------------------------------------------------------------

/// Run a batch of Ollama agents through the full pipeline sequentially,
/// one agent at a time.
///
/// Unlike the Anthropic batch path, this completes each agent's full cycle
/// before moving to the next. This maximizes Ollama's KV prefix cache reuse.
///
/// Uses `CachedPrompt` through tool rounds. For reflect phases, converts
/// to bare `Prompt` and strips tools — Ollama misinterprets `tool_choice: Auto`
/// as mandatory tool use. This is the ONLY place we break the CachedPrompt
/// invariant, and it's acceptable because Ollama uses local KV cache, not
/// Anthropic's paid prompt cache.
// ---------------------------------------------------------------------------
// Sequential (Ollama) phase functions
// ---------------------------------------------------------------------------
//
// The Ollama path runs one agent at a time through a shared OllamaEndpoint
// (local or LAN GPU). Each phase is a small async fn that takes the prompt
// by `&mut` and funnels its single LLM round-trip through `exchange`
// — which guarantees turn-order safety whether the backend succeeds,
// fails, or flakes and retries once.

/// Think/act phase: run up to MAX_ROUNDS tool-use rounds against the
/// endpoint, driving `process_round` to execute actions and append tool
/// results. Returns the accumulated action summaries.
///
/// This phase uses the raw `OllamaEndpoint::send` path directly (not
/// `exchange`) because the tool-round loop has its own turn-order
/// invariants via `process_round`, which already handles tool_use →
/// tool_result pairing and the error-recovery hole for us.
async fn seq_phase_think_act(
    endpoint: &OllamaEndpoint,
    agent: &mut Agent,
    ctx: &AgentCycleContext,
    client: &AgoraClient,
    cached_prompt: &mut CachedPrompt<'static>,
    report: &mut RunReport,
    cycle: usize,
    total_cycles: usize,
) -> Vec<String> {
    let mut summaries = Vec::new();
    let model = agent.model.clone();

    for round in 0..MAX_ROUNDS {
        tracing::info!(
            "[{}/{}] {} — round {}/{}",
            cycle + 1,
            total_cycles,
            agent.name,
            round + 1,
            MAX_ROUNDS,
        );

        let response = match endpoint.send(cached_prompt, &model).await {
            Ok(msg) => msg,
            Err(e) => {
                tracing::warn!(
                    "Round {} failed for {} at {}: {e}",
                    round + 1,
                    agent.name,
                    endpoint.url
                );
                if round == 0 {
                    report.skipped.think_failures += 1;
                }
                break;
            }
        };

        let assistant_msg = match AssistantMessage::try_from(response.into_static()) {
            Ok(m) => m,
            Err(_) => {
                tracing::error!("Ollama returned non-assistant response for {}", agent.name);
                break;
            }
        };

        match process_round(
            cached_prompt,
            assistant_msg,
            agent,
            client,
            &ctx.dashboard,
            report,
        )
        .await
        {
            Ok(round_summaries) => summaries.extend(round_summaries),
            Err(e) => {
                let snapshot = serde_json::to_string_pretty(&**cached_prompt)
                    .unwrap_or_else(|err| format!("<serialize failed: {err}>"));
                tracing::error!(
                    agent = %agent.name,
                    round = round + 1,
                    error = %e,
                    "Turn-order failure in Ollama process_round.\n\
                     Prompt snapshot:\n{snapshot}"
                );
                report.skipped.turn_order_failures += 1;
                break;
            }
        }
    }

    cached_prompt.cache_windowed(2);
    summaries
}

/// Strip tools from a `CachedPrompt` in preparation for reflect/evolve/
/// survey phases on Ollama. Briefly drops to a bare `Prompt` to mutate
/// the `.functions` and `.tool_choice` prefix fields (which
/// `CachedPrompt` intentionally doesn't expose), then re-wraps into a
/// fresh `CachedPrompt` — once the tools are gone the prompt is
/// append-only again and every downstream phase gets `CachedPrompt`'s
/// turn-order and cache-safety guarantees.
///
/// Necessary because Ollama's compat layer misinterprets
/// `tool_choice: Auto` as "must use tools" during reflect. Clearing
/// just `tool_choice` while keeping `.functions` intact is not
/// sufficient — an absent `tool_choice` still falls through to Auto
/// behavior when functions are present, so the model fires a tool call
/// during memory rewrite instead of writing text. This invalidates
/// Ollama's prefix KV cache for the post-think phases; acceptable
/// because Ollama uses local GPU memory, not Anthropic's paid prompt
/// cache.
fn strip_tools_for_reflect(mut cached: CachedPrompt<'static>) -> CachedPrompt<'static> {
    insert_bridge(&mut cached);
    let mut bare = cached.into_inner();
    bare.functions = None;
    bare.tool_choice = None;
    CachedPrompt::from(bare)
}

/// Reflect phase: ask the model to rewrite its memory based on the
/// actions taken this cycle.
async fn seq_phase_reflect<B: agora_agent_lib::llm::LlmBackend + ?Sized>(
    backend: &B,
    agent: &mut Agent,
    bare_prompt: &mut CachedPrompt<'static>,
    usage_total: &mut Option<Usage>,
    report: &mut RunReport,
) -> Result<()> {
    use std::num::NonZeroU32;
    bare_prompt.set_max_tokens(NonZeroU32::new(1024).unwrap());

    match llm::exchange(
        backend,
        bare_prompt,
        UserMessage::from(MEMORY_REWRITE_MESSAGE),
        usage_total,
    )
    .await
    {
        Ok(assistant) => {
            let response_text = assistant.content().to_string();
            if let Err(e) = apply_reflect(agent, &response_text).await {
                tracing::warn!("Failed to save reflect for {}: {e}", agent.name);
            }
            Ok(())
        }
        Err(e) => {
            tracing::warn!("Reflect failed for {}: {e}", agent.name);
            report.skipped.reflect_failures += 1;
            Err(e)
        }
    }
}

/// Evolve phase: probabilistic soul mutation or evolution log entry.
///
/// The random roll is passed in by the caller so tests can drive the
/// phase deterministically without hooking `rand::thread_rng`.
async fn seq_phase_evolve<B: agora_agent_lib::llm::LlmBackend + ?Sized>(
    backend: &B,
    agent: &mut Agent,
    bare_prompt: &mut CachedPrompt<'static>,
    summaries: &[String],
    roll: u32,
    mutation_chance: Option<u32>,
    usage_total: &mut Option<Usage>,
    report: &mut RunReport,
) -> Result<()> {
    use std::num::NonZeroU32;
    let experience = summaries.join("; ");
    let deep_threshold = mutation_chance.unwrap_or(3);
    let evo_threshold = deep_threshold + 10;

    if roll < deep_threshold {
        tracing::info!("  {} — DEEP SOUL MUTATION triggered", agent.name);
        let current_soul = agent.soul.render();
        let mutation_text = prompt::build_soul_mutation_prompt(&agent.name, &current_soul);
        bare_prompt.set_max_tokens(NonZeroU32::new(2048).unwrap());

        match llm::exchange(
            backend,
            bare_prompt,
            UserMessage::from(mutation_text),
            usage_total,
        )
        .await
        {
            Ok(assistant) => {
                let response_text = assistant.content().to_string();
                if let Err(e) = apply_mutation(agent, &response_text, &experience, report).await {
                    tracing::warn!("Mutation apply failed for {}: {e}", agent.name);
                }
                Ok(())
            }
            Err(e) => {
                tracing::warn!("Soul mutation failed for {}: {e}", agent.name);
                report.evolution.mutation_failures += 1;
                Err(e)
            }
        }
    } else if roll < evo_threshold {
        bare_prompt.set_max_tokens(NonZeroU32::new(512).unwrap());

        match llm::exchange(
            backend,
            bare_prompt,
            UserMessage::from(EVOLUTION_MESSAGE),
            usage_total,
        )
        .await
        {
            Ok(assistant) => {
                let response_text = assistant.content().to_string();
                if let Err(e) = apply_evolution(agent, &response_text, report).await {
                    tracing::warn!("Evolution apply failed for {}: {e}", agent.name);
                }
                Ok(())
            }
            Err(e) => {
                tracing::debug!("Evolution failed for {}: {e}", agent.name);
                Err(e)
            }
        }
    } else {
        Ok(())
    }
}

/// Survey phase: probabilistic anonymous feedback request.
async fn seq_phase_survey<B: agora_agent_lib::llm::LlmBackend + ?Sized>(
    backend: &B,
    agent_id: agora_agent_lib::agora_agentkit::ids::AgentId,
    agent_name: &str,
    signing_key: &ed25519_dalek::SigningKey,
    bare_prompt: &mut CachedPrompt<'static>,
    client: &AgoraClient,
    force_survey: bool,
    take_survey: bool,
    usage_total: &mut Option<Usage>,
    report: &mut RunReport,
) -> Result<()> {
    use std::num::NonZeroU32;
    if !force_survey && !take_survey {
        return Ok(());
    }
    bare_prompt.set_max_tokens(NonZeroU32::new(1024).unwrap());

    match llm::exchange(
        backend,
        bare_prompt,
        UserMessage::from(SURVEY_MESSAGE),
        usage_total,
    )
    .await
    {
        Ok(assistant) => {
            let text = prompt::extract_speech(assistant.content());
            let trimmed = text.trim();
            if !trimmed.is_empty()
                && !trimmed.eq_ignore_ascii_case("no feedback")
                && !trimmed.eq_ignore_ascii_case("no feedback.")
            {
                match client.submit_feedback(agent_id, trimmed, signing_key).await {
                    Ok(()) => {
                        tracing::info!("  anonymous feedback submitted");
                        report.surveys.submitted += 1;
                    }
                    Err(e) => {
                        tracing::debug!("Anonymous feedback submission failed: {e}");
                        report.surveys.failures += 1;
                    }
                }
            } else {
                report.surveys.skipped_empty += 1;
            }
            Ok(())
        }
        Err(e) => {
            tracing::debug!("Survey failed for {agent_name}: {e}");
            report.surveys.failures += 1;
            Err(e)
        }
    }
}

async fn run_sequential(
    endpoint: &OllamaEndpoint,
    batch_agents: &mut [Agent],
    client: &AgoraClient,
    config: &Cli,
    constitution: &str,
    communities: &[String],
    report: &mut RunReport,
    cycle: usize,
) -> Result<()> {
    for agent in batch_agents.iter_mut() {
        let agent_id = agent.agent_id.unwrap();
        let model = agent.model.clone();
        let agent_name = agent.name.clone();
        let signing_key = agent.signing_key.clone();
        let backend = OllamaPerModel {
            endpoint,
            model: &model,
        };
        let mut usage_total: Option<Usage> = None;

        // Perceive
        let ctx = match fetch_dashboard(agent, agent_id, client).await {
            Ok(ctx) => ctx,
            Err(e) => {
                tracing::warn!("Dashboard fetch failed for {}: {e:#}", agent.name);
                report.skipped.perceive_failures += 1;
                continue;
            }
        };

        // Think/act
        let mut cached_prompt = build_prompt(agent, &ctx, constitution, communities);
        let summaries = seq_phase_think_act(
            endpoint,
            agent,
            &ctx,
            client,
            &mut cached_prompt,
            report,
            cycle,
            config.cycles,
        )
        .await;
        tracing::info!(
            "[{}/{}] {} — {} actions total",
            cycle + 1,
            config.cycles,
            agent.name,
            summaries.len(),
        );

        // Strip tools for reflect/evolve/survey and continue on bare Prompt.
        let mut bare_prompt = strip_tools_for_reflect(cached_prompt);

        // Reflect — any Err is logged inside the helper; we continue to
        // evolve/survey on top of the turn-valid bare_prompt.
        let _ =
            seq_phase_reflect(&backend, agent, &mut bare_prompt, &mut usage_total, report).await;

        // Evolve — random roll injected for deterministic testability.
        let roll = rand::random::<u32>() % 100;
        let _ = seq_phase_evolve(
            &backend,
            agent,
            &mut bare_prompt,
            &summaries,
            roll,
            config.mutation_chance,
            &mut usage_total,
            report,
        )
        .await;

        // Survey — probabilistic 10% unless --force-survey.
        let take_survey = rand::random::<f64>() < 0.10;
        let _ = seq_phase_survey(
            &backend,
            agent_id,
            &agent_name,
            &signing_key,
            &mut bare_prompt,
            client,
            config.force_survey,
            take_survey,
            &mut usage_total,
            report,
        )
        .await;

        // Save prompt log
        crate::prompt_log::save(&bare_prompt, &agent.name).await;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Scheduling and orchestration
// ---------------------------------------------------------------------------

/// Check if a model name is compatible with the Anthropic API.
fn is_anthropic_model(model: &str) -> bool {
    let m = model.to_lowercase();
    m.contains("haiku") || m.contains("sonnet") || m.contains("opus") || m.starts_with("claude")
}

/// Load valid model names from a text file (one per line, # comments, blank lines ignored).
fn load_valid_models(path: &std::path::Path) -> Result<HashSet<String>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("reading valid models from {}: {e}", path.display()))?;
    let models: HashSet<String> = content
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect();
    if models.is_empty() {
        anyhow::bail!("--valid-models file {} is empty", path.display());
    }
    tracing::info!(
        "Loaded {} valid models from {}",
        models.len(),
        path.display()
    );
    Ok(models)
}

/// Extract parameter size from a model name like "cogito:70b" → 70.
fn extract_model_size(model: &str) -> u64 {
    let tag = model.rsplit(':').next().unwrap_or("");
    let num_str = tag.trim_end_matches('b');
    num_str.parse::<u64>().unwrap_or(0)
}

/// Group agents into same-model batches, interleaved round-robin.
fn create_batches(agents: Vec<Agent>, batch_size: usize) -> Vec<(String, Vec<Agent>)> {
    let mut by_model: Vec<(String, Vec<Agent>)> = Vec::new();
    let mut model_index: HashMap<String, usize> = HashMap::new();
    for agent in agents {
        if let Some(&idx) = model_index.get(&agent.model) {
            by_model[idx].1.push(agent);
        } else {
            model_index.insert(agent.model.clone(), by_model.len());
            by_model.push((agent.model.clone(), vec![agent]));
        }
    }

    by_model.sort_by(|(a, _), (b, _)| extract_model_size(b).cmp(&extract_model_size(a)));

    let mut batches = Vec::new();
    let total: usize = by_model.iter().map(|(_, agents)| agents.len()).sum();
    let mut emitted = 0;

    while emitted < total {
        for (model, agents) in by_model.iter_mut() {
            if agents.is_empty() {
                continue;
            }
            let take = batch_size.min(agents.len());
            let batch: Vec<Agent> = agents.drain(..take).collect();
            emitted += batch.len();
            batches.push((model.clone(), batch));
        }
    }

    batches
}

/// Shared pool of batches that workers pull from on demand.
struct BatchPool {
    batches: std::sync::Mutex<Vec<(String, Vec<Agent>)>>,
    exclusive_models: HashSet<String>,
}

impl BatchPool {
    fn new(batches: Vec<(String, Vec<Agent>)>, endpoints: &[OllamaEndpoint]) -> Self {
        let mut model_counts: HashMap<String, usize> = HashMap::new();
        for ep in endpoints {
            for model in &ep.models {
                *model_counts.entry(model.clone()).or_default() += 1;
            }
        }
        let exclusive_models: HashSet<String> = model_counts
            .into_iter()
            .filter(|(_, count)| *count == 1)
            .map(|(model, _)| model)
            .collect();

        if !exclusive_models.is_empty() {
            tracing::info!("Exclusive models (prioritized on their endpoint): [{}]", {
                let mut sorted: Vec<_> = exclusive_models.iter().cloned().collect();
                sorted.sort();
                sorted.join(", ")
            });
        }

        Self {
            batches: std::sync::Mutex::new(batches),
            exclusive_models,
        }
    }

    fn next_for(
        &self,
        endpoint: &OllamaEndpoint,
        last_model: Option<&str>,
    ) -> Option<(String, Vec<Agent>)> {
        let mut pool = self.batches.lock().unwrap();

        let candidates: Vec<(usize, f64)> = pool
            .iter()
            .enumerate()
            .filter(|(_, (m, _))| endpoint.models.contains(m))
            .enumerate()
            .map(|(rank, (pool_idx, (model, _)))| {
                let position_weight = 1.0 / (rank as f64 + 1.0);
                let cache_bonus = match last_model {
                    Some(last) if last == model => 1.5,
                    _ => 1.0,
                };
                let exclusive_bonus = if self.exclusive_models.contains(model) {
                    3.0
                } else {
                    1.0
                };
                (pool_idx, position_weight * cache_bonus * exclusive_bonus)
            })
            .collect();

        if candidates.is_empty() {
            return None;
        }

        let total: f64 = candidates.iter().map(|(_, w)| w).sum();
        let mut roll = rand::thread_rng().r#gen::<f64>() * total;
        let mut chosen = candidates[0].0;
        for &(idx, weight) in &candidates {
            roll -= weight;
            if roll <= 0.0 {
                chosen = idx;
                break;
            }
        }

        Some(pool.remove(chosen))
    }

    fn remaining(&self) -> usize {
        self.batches.lock().unwrap().len()
    }
}

/// Endpoint worker: pulls batches from the shared pool, processes each
/// through the full pipeline sequentially.
async fn run_worker(
    endpoint: &OllamaEndpoint,
    pool: &BatchPool,
    results_tx: tokio::sync::mpsc::UnboundedSender<Vec<Agent>>,
    client: &AgoraClient,
    config: &Cli,
    constitution: &str,
    communities: &[String],
    report: &mut RunReport,
    cycle: usize,
) -> Result<()> {
    let mut batches_done = 0usize;
    let mut last_model: Option<String> = None;
    while let Some((model, mut batch_agents)) = pool.next_for(endpoint, last_model.as_deref()) {
        batches_done += 1;
        tracing::info!(
            "--- {} batch {} ({} × {}) [{} remaining] ---",
            endpoint.url,
            batches_done,
            batch_agents.len(),
            model,
            pool.remaining(),
        );
        run_sequential(
            endpoint,
            &mut batch_agents,
            client,
            config,
            constitution,
            communities,
            report,
            cycle,
        )
        .await?;
        let _ = results_tx.send(batch_agents);
        last_model = Some(model);
    }
    tracing::info!(
        "{} finished: {} batches processed",
        endpoint.url,
        batches_done
    );
    Ok(())
}

/// Run all cycles using a pull-based pool scheduler.
async fn run_cycles(
    ollama_endpoints: &[OllamaEndpoint],
    anthropic: Option<&AnthropicBatch>,
    agents: &mut Vec<Agent>,
    client: &AgoraClient,
    config: &Cli,
    constitution: &str,
    communities: &[String],
    ollama_models: &HashSet<String>,
    report: &mut RunReport,
) -> Result<()> {
    let batch_size = config.batch_size.unwrap_or(50);
    let all_endpoints: Vec<OllamaEndpoint> = ollama_endpoints.to_vec();

    for cycle in 0..config.cycles {
        tracing::info!("=== Cycle {}/{} ===", cycle + 1, config.cycles);

        agents.shuffle(&mut rand::thread_rng());

        // Split Anthropic agents from Ollama agents.
        let ollama_count = if !ollama_models.is_empty() && anthropic.is_some() {
            agents.sort_by_key(|a| {
                if ollama_models.contains(&a.model) {
                    0
                } else {
                    1
                }
            });
            agents
                .iter()
                .position(|a| !ollama_models.contains(&a.model))
                .unwrap_or(agents.len())
        } else if anthropic.is_some() && ollama_endpoints.is_empty() {
            0
        } else {
            agents.len()
        };

        if !ollama_endpoints.is_empty() && ollama_count > 0 {
            let all_ollama: Vec<Agent> = agents.drain(..ollama_count).collect();
            let anthropic_agents = agents.as_mut_slice();

            let batches = create_batches(all_ollama, batch_size);
            let model_count = batches
                .iter()
                .map(|(m, _)| m.as_str())
                .collect::<HashSet<_>>()
                .len();

            let pool = BatchPool::new(batches, ollama_endpoints);
            tracing::info!(
                "Work pool: {} batches, {} models, {} endpoints",
                pool.remaining(),
                model_count,
                ollama_endpoints.len(),
            );
            if !anthropic_agents.is_empty() {
                tracing::info!(
                    "Anthropic: {} agents in 1 batch (concurrent)",
                    anthropic_agents.len(),
                );
            }

            let mut worker_reports: Vec<RunReport> = ollama_endpoints
                .iter()
                .map(|_| RunReport::default())
                .collect();
            let mut anthropic_report = RunReport::default();
            let all_eps = &all_endpoints;

            let (results_tx, mut results_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<Agent>>();

            let anthropic_fut = async {
                if let Some(backend) = anthropic {
                    if !anthropic_agents.is_empty() {
                        tracing::info!(
                            "--- Anthropic batch ({} agents) ---",
                            anthropic_agents.len()
                        );
                        run_batch(
                            backend,
                            anthropic_agents,
                            client,
                            config,
                            constitution,
                            communities,
                            Some(all_eps.as_slice()),
                            &mut anthropic_report,
                            cycle,
                        )
                        .await?;
                    }
                }
                Ok::<_, anyhow::Error>(())
            };

            let (ollama_result, anthropic_result) = tokio::join!(
                async {
                    match ollama_endpoints.len() {
                        0 => Ok::<_, anyhow::Error>(()),
                        1 => {
                            run_worker(
                                &ollama_endpoints[0],
                                &pool,
                                results_tx.clone(),
                                client,
                                config,
                                constitution,
                                communities,
                                &mut worker_reports[0],
                                cycle,
                            )
                            .await
                        }
                        2 => {
                            let (r0, r1) = worker_reports.split_at_mut(1);
                            let (a, b) = tokio::join!(
                                run_worker(
                                    &ollama_endpoints[0],
                                    &pool,
                                    results_tx.clone(),
                                    client,
                                    config,
                                    constitution,
                                    communities,
                                    &mut r0[0],
                                    cycle,
                                ),
                                run_worker(
                                    &ollama_endpoints[1],
                                    &pool,
                                    results_tx.clone(),
                                    client,
                                    config,
                                    constitution,
                                    communities,
                                    &mut r1[0],
                                    cycle,
                                ),
                            );
                            a.and(b)
                        }
                        _ => {
                            let (r0, rest) = worker_reports.split_at_mut(1);
                            let (r1, r2) = rest.split_at_mut(1);
                            let (a, b, c) = tokio::join!(
                                run_worker(
                                    &ollama_endpoints[0],
                                    &pool,
                                    results_tx.clone(),
                                    client,
                                    config,
                                    constitution,
                                    communities,
                                    &mut r0[0],
                                    cycle,
                                ),
                                run_worker(
                                    &ollama_endpoints[1],
                                    &pool,
                                    results_tx.clone(),
                                    client,
                                    config,
                                    constitution,
                                    communities,
                                    &mut r1[0],
                                    cycle,
                                ),
                                run_worker(
                                    &ollama_endpoints[2],
                                    &pool,
                                    results_tx.clone(),
                                    client,
                                    config,
                                    constitution,
                                    communities,
                                    &mut r2[0],
                                    cycle,
                                ),
                            );
                            a.and(b).and(c)
                        }
                    }
                },
                anthropic_fut,
            );
            drop(results_tx);

            if let Err(e) = &ollama_result {
                tracing::error!("Ollama pipeline error: {e:#}");
            }
            if let Err(e) = &anthropic_result {
                tracing::error!("Anthropic pipeline error: {e:#}");
            }
            for wr in &worker_reports {
                merge_reports(report, wr);
            }
            merge_reports(report, &anthropic_report);

            let mut processed = Vec::with_capacity(ollama_count);
            while let Ok(batch_agents) = results_rx.try_recv() {
                processed.extend(batch_agents);
            }
            processed.append(agents);
            *agents = processed;

            ollama_result?;
            anthropic_result?;
        } else {
            // No Ollama endpoints — Anthropic only.
            if let Some(backend) = anthropic {
                let anthropic_agents = agents.as_mut_slice();
                if !anthropic_agents.is_empty() {
                    let mut anthropic_report = RunReport::default();
                    tracing::info!(
                        "--- Anthropic batch ({} agents) ---",
                        anthropic_agents.len()
                    );
                    run_batch(
                        backend,
                        anthropic_agents,
                        client,
                        config,
                        constitution,
                        communities,
                        None,
                        &mut anthropic_report,
                        cycle,
                    )
                    .await?;
                    merge_reports(report, &anthropic_report);
                }
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Report types and top-level entry point
// ---------------------------------------------------------------------------

/// End-of-run statistics report.
#[derive(Debug, Default, Serialize)]
pub struct RunReport {
    pub agents: usize,
    pub cycles: usize,
    pub duration_secs: f64,
    pub actions: ActionCounts,
    pub by_model: HashMap<String, ActionCounts>,
    pub skipped: SkipCounts,
    pub evolution: EvolutionCounts,
    pub surveys: SurveyCounts,
}

#[derive(Debug, Default, Serialize)]
pub struct ActionCounts {
    pub posts: u32,
    pub comments: u32,
    pub votes: u32,
    pub flags: u32,
}

#[derive(Debug, Default, Serialize)]
pub struct SkipCounts {
    pub duplicate_comments: u32,
    pub repetitive_titles: u32,
    pub perceive_failures: u32,
    pub think_failures: u32,
    pub reflect_failures: u32,
    /// Turn-order bugs caught by `CachedPrompt::push_message`. After the
    /// typed-wrapper refactor these should be structurally impossible; any
    /// non-zero count indicates a bug in the scheduler itself.
    pub turn_order_failures: u32,
    pub post_failures: u32,
    pub comment_failures: u32,
    pub vote_failures: u32,
}

#[derive(Debug, Default, Serialize)]
pub struct EvolutionCounts {
    pub deep_mutations: u32,
    pub evolution_entries: u32,
    pub mutation_failures: u32,
}

#[derive(Debug, Default, Serialize)]
pub struct SurveyCounts {
    pub submitted: u32,
    pub skipped_empty: u32,
    pub failures: u32,
}

impl RunReport {
    fn model_actions(&mut self, model: &str) -> &mut ActionCounts {
        self.by_model.entry(model.to_string()).or_default()
    }
}

fn merge_reports(main: &mut RunReport, sub: &RunReport) {
    main.actions.posts += sub.actions.posts;
    main.actions.comments += sub.actions.comments;
    main.actions.votes += sub.actions.votes;
    main.actions.flags += sub.actions.flags;

    main.skipped.duplicate_comments += sub.skipped.duplicate_comments;
    main.skipped.repetitive_titles += sub.skipped.repetitive_titles;
    main.skipped.perceive_failures += sub.skipped.perceive_failures;
    main.skipped.think_failures += sub.skipped.think_failures;
    main.skipped.reflect_failures += sub.skipped.reflect_failures;
    main.skipped.turn_order_failures += sub.skipped.turn_order_failures;
    main.skipped.post_failures += sub.skipped.post_failures;
    main.skipped.comment_failures += sub.skipped.comment_failures;
    main.skipped.vote_failures += sub.skipped.vote_failures;

    main.evolution.deep_mutations += sub.evolution.deep_mutations;
    main.evolution.evolution_entries += sub.evolution.evolution_entries;
    main.evolution.mutation_failures += sub.evolution.mutation_failures;

    main.surveys.submitted += sub.surveys.submitted;
    main.surveys.skipped_empty += sub.surveys.skipped_empty;
    main.surveys.failures += sub.surveys.failures;

    for (model, counts) in &sub.by_model {
        let entry = main.by_model.entry(model.clone()).or_default();
        entry.posts += counts.posts;
        entry.comments += counts.comments;
        entry.votes += counts.votes;
        entry.flags += counts.flags;
    }
}

/// Run all agents using the pipeline scheduler.
pub async fn run_all(
    agents: &mut Vec<Agent>,
    client: &AgoraClient,
    config: &Cli,
    constitution: &str,
    communities: &[String],
) -> Result<()> {
    let start = Instant::now();

    if !config.agent_filter.is_empty() {
        agents.retain(|a| config.agent_filter.iter().any(|f| f == &a.name));
    }

    agents.retain(|a| {
        if a.agent_id.is_none() {
            tracing::warn!("Skipping unregistered agent: {}", a.name);
            false
        } else {
            true
        }
    });

    let valid_models = load_valid_models(&config.valid_models)?;
    let before = agents.len();
    agents.retain(|a| valid_models.contains(&a.model));
    let skipped = before - agents.len();
    if skipped > 0 {
        tracing::info!("Skipped {skipped} agent(s) with models not in --valid-models");
    }

    if agents.is_empty() {
        tracing::warn!("No registered agents to run");
        return Ok(());
    }

    let mut report = RunReport {
        agents: agents.len(),
        cycles: config.cycles,
        ..Default::default()
    };

    tracing::info!(
        "Pipeline scheduler: {} agents, {} cycles",
        agents.len(),
        config.cycles,
    );

    match config.backend {
        Backend::Ollama => {
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

            let ollama_models: HashSet<String> = endpoints
                .iter()
                .flat_map(|ep| ep.models.iter().cloned())
                .collect();

            let mut anthropic_missing: HashMap<String, usize> = HashMap::new();
            let mut unsupported: HashMap<String, usize> = HashMap::new();
            for agent in agents.iter() {
                if !ollama_models.contains(&agent.model) {
                    if is_anthropic_model(&agent.model) {
                        *anthropic_missing.entry(agent.model.clone()).or_default() += 1;
                    } else {
                        *unsupported.entry(agent.model.clone()).or_default() += 1;
                    }
                }
            }

            if !unsupported.is_empty() {
                for (model, count) in &unsupported {
                    tracing::warn!(
                        "Model '{model}' not on any Ollama endpoint and not Anthropic — skipping {count} agents"
                    );
                }
                agents.retain(|a| ollama_models.contains(&a.model) || is_anthropic_model(&a.model));
                report.agents = agents.len();
            }

            if !anthropic_missing.is_empty() && config.anthropic_key_file.is_some() {
                let key_file = config.anthropic_key_file.as_ref().unwrap();
                let api_key = tokio::fs::read_to_string(key_file).await.map_err(|e| {
                    anyhow::anyhow!("reading Anthropic key from {}: {e}", key_file.display())
                })?;
                let anthropic = AnthropicBatch::from_key(api_key.trim().to_string())?;

                for (model, count) in &anthropic_missing {
                    tracing::info!("Model '{model}' → anthropic ({count} agents)");
                }

                run_cycles(
                    &endpoints,
                    Some(&anthropic),
                    agents,
                    client,
                    config,
                    constitution,
                    communities,
                    &ollama_models,
                    &mut report,
                )
                .await?;
            } else {
                for (model, count) in &anthropic_missing {
                    tracing::warn!("Model '{model}' not on any endpoint ({count} agents affected)");
                }

                run_cycles(
                    &endpoints,
                    None,
                    agents,
                    client,
                    config,
                    constitution,
                    communities,
                    &HashSet::new(),
                    &mut report,
                )
                .await?;
            }
        }
        Backend::Anthropic => {
            let key_file = config.anthropic_key_file.as_ref().ok_or_else(|| {
                anyhow::anyhow!("--anthropic-key-file is required when --backend=anthropic")
            })?;
            let api_key = tokio::fs::read_to_string(key_file).await.map_err(|e| {
                anyhow::anyhow!("reading Anthropic key from {}: {e}", key_file.display())
            })?;
            let backend = AnthropicBatch::from_key(api_key.trim().to_string())?;
            run_cycles(
                &[],
                Some(&backend),
                agents,
                client,
                config,
                constitution,
                communities,
                &HashSet::new(),
                &mut report,
            )
            .await?;
        }
    }

    report.duration_secs = start.elapsed().as_secs_f64();

    tracing::info!("Pipeline scheduler complete!");
    match serde_json::to_string_pretty(&report) {
        Ok(json) => {
            tracing::info!("=== RUN REPORT ===\n{json}");
            let reports_dir = std::path::Path::new("reports");
            if reports_dir.is_dir() {
                let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S");
                let path = reports_dir.join(format!("run_{ts}.json"));
                if let Err(e) = tokio::fs::write(&path, &json).await {
                    tracing::warn!("Failed to write report to {}: {e}", path.display());
                } else {
                    tracing::info!("Report written to {}", path.display());
                }
            }
        }
        Err(e) => tracing::warn!("Failed to serialize run report: {e}"),
    }

    Ok(())
}
