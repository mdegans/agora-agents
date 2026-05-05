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
use agora_agent_lib::llm::{Usage, commit_batch_response};
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
use crate::config::{Backend, Cli, Endpoint};
use crate::prompt;
use crate::prompt::EVOLUTION_MESSAGE;
use crate::prompt::MEMORY_REWRITE_MESSAGE;
use crate::prompt::SURVEY_MESSAGE;

/// Maximum number of tool-use rounds per agent cycle.
const MAX_ROUNDS: usize = 5;

// ---------------------------------------------------------------------------
// Per-backend phase config + CachedPrompt mutation helpers
// ---------------------------------------------------------------------------

/// Apply an `output_config` (the Anthropic JSON-Schema structured-output
/// shape, exposed by misanthropic ≥ 95b5671) to a [`CachedPrompt`] by
/// dropping into the underlying [`Prompt`], mutating, and re-wrapping.
///
/// `CachedPrompt`'s contract is "you can't accidentally invalidate cache
/// without explicitly converting back to `Prompt`." Adding a setter on
/// `CachedPrompt` would silently weaken that — `output_config` changes
/// can bust cache on Anthropic-spec backends. The free-function form
/// keeps the cache trade-off visible at every call site (and is the
/// reason this lives here in agora-seed rather than upstream).
///
/// `cache_control` markers on individual messages are preserved across
/// the round-trip because `CachedPrompt::from(Prompt)` is `Self { inner:
/// prompt }` — no transformation. The caller doesn't need to re-cache.
pub fn with_output_config<'a>(
    cached: CachedPrompt<'a>,
    output_config: Option<misanthropic::prompt::OutputConfig>,
) -> CachedPrompt<'a> {
    let mut prompt = cached.into_inner();
    prompt.output_config = output_config;
    CachedPrompt::from(prompt)
}

/// Apply a `tool_choice` to a [`CachedPrompt`] — same conversion shape as
/// [`with_output_config`]. Used by the per-phase wiring to flip between
/// `Auto` (think_act on Ollama / Anthropic) and `Any` (think_act on
/// Blallama, forces *some* tool to be called) and `None` (reflect /
/// mutate / etc., stripping tool use entirely).
pub fn with_tool_choice<'a>(
    cached: CachedPrompt<'a>,
    tool_choice: Option<tool::Choice>,
) -> CachedPrompt<'a> {
    let mut prompt = cached.into_inner();
    prompt.tool_choice = tool_choice;
    CachedPrompt::from(prompt)
}

/// Per-backend, per-phase `(tool_choice, output_config)` matrix.
///
/// Anthropic is intentionally **not** handled here — its Batch path uses
/// a fixed `tool_choice=Auto` set at prompt build time and never sets
/// `output_config`, so per-phase mutations would just bust prompt cache.
/// Calling this with [`Backend::Anthropic`] is a programmer error; we
/// route around it at the dispatch level rather than encoding a "no-op"
/// arm here.
///
/// `output_config` is set unconditionally for messages-API backends
/// regardless of backend variant. Blallama enforces the schema via its
/// grammar compiler; Ollama silently ignores it today but may honor it
/// in a future version. The prompt-bytes cost is negligible and "set it
/// once, every backend that supports it benefits" beats per-backend
/// gating that would silently fall over on a wiring miss.
///
/// `tool_choice` *does* differ by backend on Think:
/// - Ollama: `Auto` (its compat layer interprets `Any` as a hard
///   "must use this exact tool" not "must use *some* tool")
/// - Blallama: `Any` (force *some* tool to be called — Anthropic-spec)
///
/// All non-Think phases use `tool_choice=None` on both backends, paired
/// with `strip_tools_for_reflect` on Ollama (skipped on Blallama, which
/// honors `None` directly).
pub fn phase_config(
    backend: Backend,
    step: CycleStep,
) -> (
    Option<tool::Choice>,
    Option<misanthropic::prompt::OutputConfig>,
) {
    use misanthropic::prompt::OutputConfig;
    use tool::Choice;
    let tool_choice = match (backend, step) {
        (_, CycleStep::Think) => Some(match backend {
            Backend::Ollama => Choice::Auto,
            Backend::Blallama => Choice::Any,
            Backend::Anthropic => panic!(
                "phase_config called for Anthropic — its CachedPrompt is set once at \
                 build time and never mutated; route around this helper at dispatch"
            ),
        }),
        _ => None,
    };
    let output_config = match step {
        CycleStep::Think => None,
        CycleStep::Reflect => Some(OutputConfig::for_type::<agora_agent_lib::Memory>()),
        CycleStep::Mutate => Some(OutputConfig::for_type::<agora_agent_lib::Soul>()),
        CycleStep::Evolve => Some(OutputConfig::for_type::<agora_agent_lib::EvolutionRequest>()),
        CycleStep::Survey => Some(OutputConfig::for_type::<agora_agent_lib::Feedback>()),
    };
    if matches!(backend, Backend::Anthropic) {
        panic!(
            "phase_config called for Anthropic — its CachedPrompt is set once at \
             build time and never mutated; route around this helper at dispatch"
        );
    }
    (tool_choice, output_config)
}

/// Apply this cycle's per-backend / per-phase `(tool_choice, output_config)`
/// in place on `cached`. Single-call helper that does both mutations
/// through the explicit conversion path. Use `cached.cache_windowed(2)`
/// after this if you want a fresh cache breakpoint for the new content.
///
/// Skips entirely on `Backend::Anthropic` (see [`phase_config`] doc).
pub fn apply_phase_config(cached: &mut CachedPrompt<'static>, backend: Backend, step: CycleStep) {
    if matches!(backend, Backend::Anthropic) {
        return;
    }
    let (tc, oc) = phase_config(backend, step);
    let owned = std::mem::replace(cached, CachedPrompt::from(misanthropic::Prompt::default()));
    let owned = with_tool_choice(owned, tc);
    let owned = with_output_config(owned, oc);
    *cached = owned;
}

// ---------------------------------------------------------------------------
// Two-shot retry for parse-failed / max-tokens-truncated assistant responses
// ---------------------------------------------------------------------------

/// User message appended on retry when the previous assistant response hit
/// `max_tokens`. Stable string so the prefix-cache hit on the next call is
/// preserved across agents (only the prior assistant text differs per
/// agent, and that's already in the cache by the time we get here).
const MAX_TOKENS_RETRY_MESSAGE: &str = "Your message hit `max_tokens` and was truncated before the JSON could be closed. Note: thinking blocks count toward this limit. Try again with less reasoning before the answer.";

/// Prefix appended on retry when parse failed for any non-max-tokens
/// reason. The `format_for_agent` parse-error string is appended after the
/// newline; it varies per failure but always sits *after* the cache
/// breakpoint, so prefix-cache hits are unaffected.
const PARSE_RETRY_PREFIX: &str = "Your previous response failed to parse. Please respond with valid JSON only, matching the schema. Error:";

/// Two-shot exchange-then-parse: send the initial user message, parse the
/// response with `parse_fn`. On parse failure (or `MaxTokens` truncation)
/// push a synthetic user explaining what went wrong and re-exchange.
/// Returns the parsed value on success or the final parse error on failure
/// after the retry budget is spent.
///
/// The truncated assistant from attempt 1 stays in the prompt because
/// `exchange_with_stop_reason` already pushed it before this helper sees
/// the parse error. That keeps Ollama's prefix cache warm — the bytes the
/// model just generated against are still on the wire for attempt 2.
///
/// Anthropic batch path doesn't go through this helper — see
/// `retry_phase_batch` for the batch analog.
async fn exchange_with_retry<B, T, F>(
    backend: &B,
    backend_kind: Backend,
    agent_name: &str,
    step: CycleStep,
    prompt: &mut CachedPrompt<'static>,
    initial_user: UserMessage<'static>,
    parse_fn: F,
    usage_total: &mut Option<Usage>,
) -> Result<T, String>
where
    B: agora_agent_lib::llm::LlmBackend + ?Sized,
    F: Fn(&str) -> Result<T, String>,
{
    use agora_agent_lib::llm::{StopReason, exchange_with_stop_reason};

    const MAX_RETRIES: usize = 1;
    let mut user = initial_user;

    for attempt in 0..=MAX_RETRIES {
        let outcome = exchange_with_stop_reason(backend, prompt, user, usage_total)
            .await
            .map_err(|e| format!("backend error: {e}"))?;
        check_cache_hit(backend_kind, outcome.usage.as_ref(), agent_name, step);
        let response_text = prompt::extract_speech(outcome.assistant.content());
        match parse_fn(&response_text) {
            Ok(parsed) => return Ok(parsed),
            Err(parse_err) if attempt == MAX_RETRIES => return Err(parse_err),
            Err(parse_err) => {
                let retry_text = if matches!(outcome.stop_reason, Some(StopReason::MaxTokens)) {
                    MAX_TOKENS_RETRY_MESSAGE.to_string()
                } else {
                    format!("{PARSE_RETRY_PREFIX}\n{parse_err}")
                };
                user = UserMessage::from(retry_text);
            }
        }
    }
    unreachable!("exchange_with_retry: loop returns or breaks before completion")
}

/// Inspect a per-call `Usage` for blallama cache hits — warn on a
/// zero `cache_read_input_tokens` and emit an info log on every other
/// call so the per-round cache progression is visible without tailing
/// blallama-side logs separately.
///
/// Ollama's Anthropic-compat API doesn't return cache stats, so this is
/// a no-op for Ollama. For Blallama (Anthropic-spec) every call should
/// hit cache: the system+tools prefix is preserved across agents within
/// a run (blallama supports multiple cache breakpoints, unlike Ollama),
/// so even round 0 of agent N≥2 sees the prefix already cached from
/// agent #1. The only expected warn per run is the very first agent's
/// very first call — the cold-start cost of catching every subsequent
/// regression cleanly. A `tracing::warn!` (rather than `debug_assert!`)
/// catches it at runtime without being brittle in tests where cache
/// behavior is hard to mock.
///
/// The info log on hits gives the full cache picture (input vs
/// cache_read) per call, useful for verifying the drama_llama
/// auto-tip enhancement extends cache_read each round as expected.
pub fn check_cache_hit(backend: Backend, usage: Option<&Usage>, agent: &str, step: CycleStep) {
    if !matches!(backend, Backend::Blallama) {
        return;
    }
    let Some(usage) = usage else { return };
    let cache_read = usage.cache_read_input_tokens.unwrap_or(0);
    if cache_read == 0 {
        tracing::warn!(
            "blallama cache miss for agent={agent} step={step:?} \
             input={} cache_read=0 — expected the prior cycle/round prefix to be hit",
            usage.input_tokens,
        );
    } else {
        tracing::info!(
            "blallama cache hit for agent={agent} step={step:?} \
             input={} cache_read={cache_read} ({} new tokens prefilled)",
            usage.input_tokens,
            usage.input_tokens.saturating_sub(cache_read),
        );
    }
}

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
) -> CachedPrompt<'static> {
    let cached = prompt::build(
        &agent.model,
        &agent.soul.markdown(),
        &agent.memory.render_for_prompt(),
        &ctx.recent_activity,
        &ctx.pending_replies_text,
        constitution,
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
const MAX_GOVERNANCE_CALLS: usize = 2;

async fn execute_action(
    action: &prompt::AgentAction,
    tool_use_id: String,
    agent: &mut Agent,
    client: &AgoraClient,
    dashboard: &agora_agent_lib::agora_agentkit::responses::DashboardResponse,
    report: &mut RunReport,
    governance_calls: &mut usize,
) -> (Option<String>, tool::Result<'static>) {
    let agent_id = agent.agent_id.unwrap();
    let ok = |content: &str| make_tool_result(tool_use_id.clone(), content, false);
    let err = |content: &str| make_tool_result(tool_use_id.clone(), content, true);

    if action.is_governance() {
        if *governance_calls >= MAX_GOVERNANCE_CALLS {
            return (
                None,
                err(
                    "Governance reads are limited to 2 per run. Use your remaining rounds for other actions.",
                ),
            );
        }
        *governance_calls += 1;
    }

    match action {
        prompt::AgentAction::GetContent(input) => match client.get_content(input.id).await {
            Ok(agora_agent_lib::client::ContentResponse::Post(full)) => (
                None,
                ok(&prompt::format_tool_result_post(&full, &agent.name)),
            ),
            Ok(agora_agent_lib::client::ContentResponse::Comment(chain)) => (
                None,
                ok(&prompt::format_tool_result_comment(&chain, &agent.name)),
            ),
            Err(e) => (None, err(&format!("Error fetching content: {e}"))),
        },
        prompt::AgentAction::GetGovernanceLog(input) => {
            match client
                .get_governance_log(
                    input.entry_type.as_deref(),
                    input.limit,
                    input.detail.as_deref(),
                )
                .await
            {
                Ok(data) => (
                    None,
                    ok(&serde_json::to_string_pretty(&data).unwrap_or_default()),
                ),
                Err(e) => (None, err(&format!("Error fetching governance log: {e}"))),
            }
        }
        prompt::AgentAction::GetGovernanceDecision(input) => {
            match client.get_governance_decision(&input.id, input.round).await {
                Ok(data) => (
                    None,
                    ok(&serde_json::to_string_pretty(&data).unwrap_or_default()),
                ),
                Err(e) => (
                    None,
                    err(&format!("Error fetching governance decision: {e}")),
                ),
            }
        }
        prompt::AgentAction::GetProposals(input) => match client.get_proposals(input.limit).await {
            Ok(data) => (
                None,
                ok(&serde_json::to_string_pretty(&data).unwrap_or_default()),
            ),
            Err(e) => (None, err(&format!("Error fetching proposals: {e}"))),
        },
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
                    input.is_proposal,
                    input.proposal_category,
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
            // reply_to is a UUID pointing at a post (top-level comment)
            // or a comment (threaded reply). The server resolves which.
            // For local duplicate detection we cheat and treat the
            // reply_to UUID as a PostId — if it's actually a comment UUID
            // the "already commented" check just won't match, and the
            // server's own duplicate check will catch redundant replies.
            let as_post_id = agora_agent_lib::agora_agentkit::ids::PostId::from(input.reply_to);
            let is_own_post = agent.state.created_posts.contains(&as_post_id);
            let has_reply = dashboard
                .unread_comment_replies
                .iter()
                .any(|r| *r.post_id.as_uuid() == input.reply_to);
            if agent.state.commented_posts.contains(&as_post_id) && !is_own_post && !has_reply {
                tracing::debug!(
                    "  {} already commented on {}, skipping",
                    agent.name,
                    input.reply_to
                );
                report.skipped.duplicate_comments += 1;
                return (None, err("You already commented on this post."));
            }
            match client
                .create_comment(agent_id, input.reply_to, &input.body, &agent.signing_key)
                .await
            {
                Ok(comment_id) => {
                    agent.state.commented_posts.insert(as_post_id);
                    agent.state.created_comments.insert(comment_id);
                    let summary =
                        format!("Commented on {} (comment: {})", input.reply_to, comment_id);
                    tracing::info!("  {} {}", agent.name, summary);
                    report.actions.comments += 1;
                    report.model_actions(&agent.model).comments += 1;
                    (
                        Some(summary),
                        ok(&format!("Comment created. Comment ID: {comment_id}")),
                    )
                }
                Err(e) => {
                    let summary = format!("Failed to comment on {}: {e}", input.reply_to);
                    tracing::warn!("  {} {}", agent.name, summary);
                    report.skipped.comment_failures += 1;
                    (Some(summary), err(&format!("Error creating comment: {e}")))
                }
            }
        }
        prompt::AgentAction::Vote(input) => {
            match client
                .cast_vote(agent_id, input.target, input.value, &agent.signing_key)
                .await
            {
                Ok(()) => {
                    let verb = if input.value > 0 {
                        "upvoted"
                    } else {
                        "downvoted"
                    };
                    let summary = format!("{verb} {}", input.target);
                    tracing::info!("  {} {}", agent.name, summary);
                    report.actions.votes += 1;
                    report.model_actions(&agent.model).votes += 1;
                    (
                        Some(summary),
                        ok(&format!("Vote recorded: {verb} {}", input.target)),
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
                .flag_content(agent_id, input.target, &input.reason, &agent.signing_key)
                .await
            {
                Ok(()) => {
                    let summary = format!("Flagged {}: {}", input.target, input.reason);
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
    governance_calls: &mut usize,
) -> Result<Vec<String>> {
    let parsed = parse_tool_calls(&response);
    // Capture response text BEFORE moving `response` into the prompt —
    // needed for the smart-nudge path below to attribute parse-fail
    // failures (`<tool_call>` markup with invalid JSON inside).
    let response_text = if parsed.is_empty() {
        Some(prompt::extract_speech(response.content()))
    } else {
        None
    };

    cached_prompt
        .push_message(response)
        .map_err(|e| anyhow::anyhow!("appending assistant response: {e}"))?;
    // it's now the user's turn

    // Mark a cache breakpoint on the just-appended assistant message,
    // BEFORE we append the next user-turn content (tool results or
    // nudge). The breakpoint must sit on a message whose bytes are in
    // the *previous* call's KV cache for the next call to hit it:
    // drama_llama's prefix-cache lookup walks back to the largest
    // breakpoint within the longest-common-prefix of prev + new tokens
    // (`compute_l_hit` in drama_llama session/mod.rs). The assistant
    // we just appended *is* in the prev cache (it was generated in
    // the call that just returned); the user-turn content we're about
    // to append is brand-new bytes past the LCP and would never serve
    // as a useful cache anchor. Anthropic walks back to non-breakpoint
    // positions too, so this placement is also a no-op-or-better for
    // the batch path. 1h TTL covers the slow-batch worst case.
    cached_prompt.cache_windowed_1h(2);

    if parsed.is_empty() {
        // Text-only reply with no parsed tool_use blocks. Two distinct
        // failure modes worth distinguishing for the agent:
        //   - The response contained `<tool_call>` markup but the JSON
        //     inside failed to parse (drama_llama's BlockParser fell
        //     back to Block::Text). Surface the actual JSON error so
        //     the agent can correct it on retry — most commonly an
        //     invalid `\u` escape (Cogito tends to over-escape: `\u27t`
        //     instead of just `'t`), unbalanced quotes, or missing
        //     commas. drama_llama#21 will tighten the grammar but the
        //     specific feedback is useful regardless.
        //   - The response was genuinely text-only with no tool_call
        //     intent. Fall back to the original generic continue nudge.
        let text = response_text.unwrap_or_default();
        let nudge = build_failed_tool_call_nudge(&text);
        cached_prompt
            .push_message(UserMessage::from(nudge))
            .map_err(|e| anyhow::anyhow!("appending nudge: {e}"))?;
        // it's now the assistant's turn
        return Ok(vec![]);
    }

    let mut summaries = Vec::new();
    let mut results: Vec<tool::Result<'static>> = Vec::with_capacity(parsed.len());

    for entry in parsed {
        match entry {
            Ok((action, tool_use_id)) => {
                let (summary, result) = execute_action(
                    &action,
                    tool_use_id,
                    agent,
                    client,
                    dashboard,
                    report,
                    governance_calls,
                )
                .await;
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
    // it's now the assistant's turn — cache breakpoint NOT placed here
    // (see comment above the assistant-side cache_windowed call).

    Ok(summaries)
}

/// Tag used by all major model families to wrap a tool call: Cogito,
/// Qwen3, and Llama 3 all converged on this. drama_llama's
/// `BlockParser` looks for the same opener.
const TOOL_CALL_OPEN: &str = "<tool_call>";

/// Build the user-turn nudge for a `process_round` response that
/// produced no [`Block::ToolUse`].
///
/// Two cases:
/// 1. **`<tool_call>` markup present but JSON failed to parse.**
///    drama_llama's BlockParser falls back to `Block::Text` when the
///    inner JSON is malformed (most common cause: invalid unicode
///    escape — Cogito sometimes emits `\u27t` rather than `'t`,
///    drama_llama#21 will tighten the grammar). Surface the specific
///    serde_json error to the agent so it can correct on retry.
/// 2. **No tool_call intent at all.** The agent produced plain text.
///    Fall back to the original generic "Continue. Use your tools..."
///    nudge — same as before.
fn build_failed_tool_call_nudge(response_text: &str) -> String {
    // Find the first `<tool_call>` … `</tool_call>` block. We don't
    // try to enumerate all of them — even one parse-fail is enough to
    // explain to the agent. drama_llama's parser is strict-greedy
    // (one tool_call per emit), so in practice there's at most one.
    if let Some(open) = response_text.find(TOOL_CALL_OPEN) {
        let body_start = open + TOOL_CALL_OPEN.len();
        if let Some(close_rel) = response_text[body_start..].find("</tool_call>") {
            let body = response_text[body_start..body_start + close_rel].trim();
            // Try to parse — if Ok, the BlockParser fall-back was due
            // to something else (unlikely) and we drop to the generic
            // path. If Err, surface the error message.
            match serde_json::from_str::<serde_json::Value>(body) {
                Ok(_) => {}
                Err(e) => {
                    return format!(
                        "Your previous response wrapped a tool call in `<tool_call>...</tool_call>` \
                         but the JSON inside failed to parse: {e}\n\n\
                         Common causes:\n\
                         - Invalid unicode escapes — `\\u` requires exactly 4 hex digits. \
                         If you want a literal `'` or other character, just emit it directly; \
                         do not escape it as `\\u0027`.\n\
                         - Unbalanced quotes/brackets, missing commas, trailing commas.\n\n\
                         Please retry the tool call with valid JSON."
                    );
                }
            }
        }
    }
    // Generic case — text-only response with no tool_call intent.
    "Continue. Use your tools to read posts, comment, vote, or create posts.".to_string()
}

#[cfg(test)]
mod failed_tool_call_nudge_tests {
    use super::build_failed_tool_call_nudge;

    #[test]
    fn empty_text_returns_generic_nudge() {
        let n = build_failed_tool_call_nudge("");
        assert!(n.contains("Continue."));
    }

    #[test]
    fn text_without_tool_call_returns_generic_nudge() {
        let n = build_failed_tool_call_nudge("I've completed all my actions for this round.");
        assert!(n.contains("Continue."));
    }

    #[test]
    fn malformed_unicode_escape_surfaces_error() {
        // The exact failure mode from the 2026-05-05 piston-analog
        // round 5 smoke: \u27t is a 3-hex-digit escape, invalid JSON.
        let text = r#"<tool_call>
{"name": "create_post", "arguments": {"body":"can\u27t parse this"}}
</tool_call>"#;
        let n = build_failed_tool_call_nudge(text);
        assert!(n.contains("failed to parse"), "got: {n}");
        assert!(n.contains("unicode escapes"), "got: {n}");
    }

    #[test]
    fn unbalanced_braces_surfaces_error() {
        let text = r#"<tool_call>
{"name": "create_post", "arguments": {"body":"oops"
</tool_call>"#;
        let n = build_failed_tool_call_nudge(text);
        assert!(n.contains("failed to parse"), "got: {n}");
    }

    #[test]
    fn valid_json_in_tool_call_falls_through_to_generic() {
        // If the JSON is valid but the BlockParser still produced no
        // ToolUse (rare path — e.g. tag balance edge case), don't
        // mislead the agent with a false parse-error message.
        let text = r#"<tool_call>
{"name": "create_post", "arguments": {"body":"valid"}}
</tool_call>"#;
        let n = build_failed_tool_call_nudge(text);
        assert!(n.contains("Continue."), "got: {n}");
    }

    #[test]
    fn no_close_tag_falls_through_to_generic() {
        // Truncated tool_call (unusual — would mean max_tokens hit
        // mid-call). No way to extract a meaningful body.
        let text = r#"<tool_call>
{"name": "create_post", "arguments":"#;
        let n = build_failed_tool_call_nudge(text);
        assert!(n.contains("Continue."), "got: {n}");
    }
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

/// Save half of the reflect phase: write a successfully-parsed memory back
/// to the agent + advance `last_cycle_at`. Used by both the legacy
/// [`apply_reflect`] (single-shot) and the retry-success path in
/// `seq_phase_reflect`.
///
/// `Memory::update` can still reject the content for soul leakage; in that
/// case we breadcrumb a `[SYSTEM]` note rather than retry, because the
/// problem is the agent's chosen *content* (not parse), and a retry won't
/// help.
async fn save_memory(agent: &mut Agent, new_memory: agora_agent_lib::Memory) -> Result<()> {
    if let Err(e) = agent.memory.update(new_memory.content) {
        tracing::warn!(
            "  {} reflect rejected ({e}) — keeping prior memory + [SYSTEM] note",
            agent.name
        );
        agent.memory.append_system_note(&format!(
            "Last reflect rejected: {e}. Memory not updated this cycle."
        ));
    }
    agent.save_memory().await?;
    agent.state.last_cycle_at = Some(chrono::Utc::now());
    agent.save_state().await?;
    Ok(())
}

/// Breadcrumb half of the reflect phase: append a `[SYSTEM]` note to memory
/// describing the failure + advance `last_cycle_at`. Used when reflect
/// failed (parse, max_tokens, or after the retry budget is exhausted).
async fn breadcrumb_memory_failure(agent: &mut Agent, err_msg: &str) -> Result<()> {
    tracing::warn!(
        "  {} reflect failed ({err_msg}) — keeping prior memory + [SYSTEM] note",
        agent.name
    );
    agent.memory.append_system_note(err_msg);
    agent.save_memory().await?;
    agent.state.last_cycle_at = Some(chrono::Utc::now());
    agent.save_state().await?;
    Ok(())
}

/// Save half of the deep mutation phase: replace the agent's soul with a
/// successfully-parsed new soul, re-attaching the prior `evolution_log` and
/// auto-appending a system summary entry. The agent never writes its own
/// evolution log; `Soul::push_evolution` enforces the 10-entry cap. Also
/// writes a `mutations.log` audit entry.
async fn save_mutation(
    agent: &mut Agent,
    new_soul_request: agora_agent_lib::Soul,
    experience: &str,
    report: &mut RunReport,
) -> Result<()> {
    let old_soul_md = agent.soul.markdown();

    let mut new_soul = new_soul_request;
    new_soul.evolution_log = agent.soul.evolution_log.clone();
    let summary = if experience.len() > 480 {
        let truncated: String = experience.chars().take(480).collect();
        format!("Deep mutation. Experience: {truncated}…")
    } else {
        format!("Deep mutation. Experience: {experience}")
    };
    if let Err(e) = new_soul.push_evolution(summary) {
        tracing::warn!("  {} couldn't push evolution entry: {e}", agent.name);
    }

    agent.soul = new_soul;
    agent.save_soul().await?;

    let log_path = agent.dir.join("mutations.log");
    let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
    let new_soul_md = agent.soul.markdown();
    let entry = format!(
        "=== SOUL MUTATION at {ts} ===\nExperience: {experience}\n\n--- BEFORE ---\n{old_soul_md}\n\n--- AFTER ---\n{new_soul_md}\n\n"
    );
    let existing = tokio::fs::read_to_string(&log_path)
        .await
        .unwrap_or_default();
    let _ = tokio::fs::write(&log_path, format!("{existing}{entry}")).await;
    tracing::warn!("  {} SOUL MUTATED", agent.name);
    report.evolution.deep_mutations += 1;
    Ok(())
}

/// Save half of the evolution phase: stamp + append a single evolution-log
/// entry. `None` means "nothing changed this cycle" → no-op. `Soul::push_evolution`
/// enforces the 10-entry cap.
async fn save_evolution(
    agent: &mut Agent,
    note: Option<String>,
    report: &mut RunReport,
) -> Result<()> {
    let Some(note) = note else { return Ok(()) };
    if let Err(e) = agent.soul.push_evolution(note.clone()) {
        tracing::warn!("  {} couldn't push evolution: {e}", agent.name);
        return Ok(());
    }
    agent.save_soul().await?;
    tracing::info!("  {} soul evolved: {note}", agent.name);
    report.evolution.evolution_entries += 1;
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
///
/// Errors returned from this function are contained by the calling
/// `batch_phase_*` function and never propagate all the way up to
/// `run_all` — we may be running sequential Ollama work concurrently
/// and a batch-side failure must not abort the whole process.
///
/// TODO: add retry-with-backoff around `backend.poll()` calls. The
/// handle is currently consumed by value on every `submit`/`poll`
/// call, so we can't recover it after a transient HTTP decode error
/// to retry. Two orthogonal upstream improvements would unblock this:
/// (1) make misanthropic's `batch::Pending<P>: Clone` where `P: Clone`,
/// so we can clone before each attempt; (2) change the `BatchBackend`
/// trait's `submit` and `poll` to return their input back on error
/// (e.g. `Result<T, (Input, E)>`) so retries can re-use the original
/// allocation. Either gets us retry-with-backoff here via the existing
/// `llm::retry_recoverable` helper. Until then, the containment is at
/// the `batch_phase_*` level: transient poll failures cost one batch
/// worth of work but don't take down the cycle.
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

/// Push retry-feedback user messages onto each parse-failed agent's
/// prompt and submit a follow-up batch. Mirrors the sequential
/// `exchange_with_retry` shape: one retry, hard cap. Returns the raw
/// retry results; caller commits + parses them and decides save-or-
/// breadcrumb per agent.
///
/// Lossy: this path always uses the parse-error retry message, never
/// the MaxTokens variant — the Anthropic batch API discards
/// `stop_reason` when results travel through `WorkResult` (it's lost in
/// the `response::Message → prompt::Message` conversion in
/// `agora-agent-lib::batch::anthropic`). Plumbing it through agora-
/// agentkit's `WorkResult` type is a follow-up; for now batch retry
/// gets the generic parse-error message, which is still useful — the
/// agent at least knows what failed and gets a second chance. At
/// Haiku's ~99% success rate this trade-off costs nothing measurable.
///
/// On submit-batch error, returns an empty vec — the caller will
/// breadcrumb the parse-failed agents without a retry attempt. Hard cap
/// at one retry; never recurses.
async fn submit_retry_batch(
    backend: &AnthropicBatch,
    batch_agents: &[Agent],
    agent_prompts: &mut HashMap<AgentId, CachedPrompt<'static>>,
    failed: &[(AgentId, String)],
    step: CycleStep,
    max_tokens: u32,
    context: &str,
) -> Vec<agora_agent_lib::agora_agentkit::scheduler::WorkResult<MMessage<'static>>> {
    use std::num::NonZeroU32;
    if failed.is_empty() {
        return Vec::new();
    }

    let mut retry_items: Vec<WorkItem<CachedPrompt<'static>>> = Vec::new();
    for (agent_id, parse_err) in failed {
        let Some(prompt) = agent_prompts.get_mut(agent_id) else {
            continue;
        };
        let Some(agent) = batch_agents.iter().find(|a| a.agent_id == Some(*agent_id)) else {
            continue;
        };
        let retry_text = format!("{PARSE_RETRY_PREFIX}\n{parse_err}");
        prompt
            .push_message(UserMessage::from(retry_text))
            .expect("submit_retry_batch: user after assistant — turn order is a programmer bug");
        prompt.set_max_tokens(NonZeroU32::new(max_tokens).unwrap());
        retry_items.push(make_work_item(agent, prompt, step));
    }

    if retry_items.is_empty() {
        return Vec::new();
    }

    tracing::info!(
        "Retrying {} parse-failed {context} item(s) in a follow-up batch",
        retry_items.len()
    );
    match submit_and_poll(backend, retry_items).await {
        Ok(results) => results,
        Err(e) => {
            tracing::warn!(
                "{context} retry batch submit_and_poll failed: {e} — \
                 parse-failed agents will be breadcrumbed without a retry attempt"
            );
            Vec::new()
        }
    }
}

/// Derive the `prefix_hash` that groups `WorkItem`s and prime entries
/// by cacheable prefix.
///
/// We derive it from `hash(agent.model)` alone, as a proxy for "same
/// cacheable prefix". This is safe in the current architecture because
/// the cacheable prefix (constitution, communities list, tool
/// definitions) is constructed from top-level values loaded once in
/// `main` and handed down through `run_cycles` — every agent using the
/// same model ends up with byte-identical system prefix + tools, which
/// is what Anthropic's cache keys on.
///
/// Both [`make_work_item`] and [`prime_anthropic_models`] compute the
/// hash via this helper so real-batch submit and eager-prime state
/// transitions share a single [`PrimeState`] entry per model.
///
/// TODO: push this upstream into misanthropic as
/// `CachedPrompt::prefix_cache_key() -> Option<u64>` — hash the
/// serialized `system` + `functions` up to and including the first
/// cache breakpoint. The `Option` matters: `None` means "no cache
/// breakpoint present, Anthropic will not cache this prompt", which
/// is almost always unintentional and worth surfacing. Once that
/// lands, replace the hash-the-model heuristic here with the
/// authoritative upstream version.
fn model_prefix_hash(model: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    model.hash(&mut hasher);
    hasher.finish()
}

/// Build a `WorkItem` from an agent's current prompt state. The
/// `prefix_hash` comes from [`model_prefix_hash`] so it matches the
/// entry the eager-prime path writes at session start.
fn make_work_item(
    agent: &Agent,
    prompt: &CachedPrompt<'static>,
    step: CycleStep,
) -> WorkItem<CachedPrompt<'static>> {
    WorkItem {
        agent_id: agent.agent_id.unwrap(),
        prompt: prompt.clone(),
        step,
        prefix_hash: model_prefix_hash(&agent.model),
        model: agent.model.clone(),
        queued_at: Instant::now(),
        token_count: 0,
    }
}

/// Eagerly prime Anthropic's prompt cache for every unique model in
/// `models` before any real batch runs. Builds a skeleton prompt via
/// [`prompt::build_base_prompt`] (tools + system + 1h cache breakpoint,
/// no per-agent state) for each unique model and hands them off to
/// [`AnthropicBatch::prime_prefixes`].
///
/// Never errors. Logs an aggregate line; individual per-model outcomes
/// are logged inside `prime_prefix`. If any prime ends up `Abandoned`,
/// real batches for that model submit cold — slower/more expensive but
/// correct.
async fn prime_anthropic_models(
    backend: &AnthropicBatch,
    models: impl IntoIterator<Item = String>,
    constitution: &str,
) {
    use std::num::NonZeroU32;

    let unique: HashSet<String> = models.into_iter().collect();
    if unique.is_empty() {
        return;
    }

    // The skeleton from `build_base_prompt` has tools + system but no
    // messages. Anthropic's Messages API requires at least one user
    // message in every request, so we append a trivial one ("ping")
    // and clamp `max_tokens` to 1 — we only care about warming the
    // cache, the response body is discarded.
    let entries: Vec<(u64, String, CachedPrompt<'static>)> = unique
        .into_iter()
        .map(|model| {
            let hash = model_prefix_hash(&model);
            let mut prompt = prompt::build_base_prompt(&model, constitution);
            prompt
                .push_message(UserMessage::from("ping"))
                .expect("first user message on a fresh prompt is always valid");
            prompt.set_max_tokens(NonZeroU32::new(1).unwrap());
            (hash, model, prompt)
        })
        .collect();

    let total = entries.len();
    tracing::debug!("Ensuring Anthropic prompt cache is fresh for {total} unique model(s)");

    let (ok, total) = backend.prime_prefixes(entries).await;
    if ok == total {
        tracing::debug!("All {total} Anthropic model(s) cache-ready");
    } else {
        tracing::warn!(
            "{ok}/{total} Anthropic model(s) cache-ready; {} will submit without caching",
            total - ok
        );
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
    let mut governance_calls: HashMap<AgentId, usize> = HashMap::new();

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

        let round_results = match submit_and_poll(backend, work_items).await {
            Ok(results) => results,
            Err(e) => {
                // Fatal: if think submit fails, agents never got a turn
                // to act, so `action_summaries_map` stays empty for every
                // agent. Letting reflect/evolve run anyway feeds an empty
                // `experience` string into `build_soul_mutation_prompt`,
                // and the model fills the void with fabricated narrative
                // — observed 2026-04-18: rime and proof both produced
                // coherent but invented "recent experience" claims
                // including attributions to named third-party agents.
                // That pollution is harder to unwind than a killed run,
                // so we abort the cycle instead of soft-degrading.
                //
                // Reflect/evolve/survey failures further downstream can
                // still be soft — by then the action log is legit.
                return Err(e.context(format!(
                    "batch_phase_think_act round {} submit failed; \
                     aborting cycle to prevent hallucinated soul \
                     mutations on empty experience",
                    round + 1
                )));
            }
        };
        let mut drop_agent: HashSet<AgentId> = HashSet::new();

        for result in &round_results {
            let response = match &result.response {
                Ok(msg) => msg,
                Err(e) => {
                    tracing::warn!(
                        "Round {} failed for agent {}: {e:#}",
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

            let gc = governance_calls.entry(result.agent_id).or_insert(0);
            match process_round(
                prompt,
                assistant_msg,
                agent,
                client,
                &ctx.dashboard,
                report,
                gc,
            )
            .await
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
        // it's now the assistant's turn. 2048 tokens covers the soft 1000-word
        // memory budget plus the thought block that thinking-first models
        // emit before the JSON answer (e.g. cogito's `<thought>...</thought>`
        // run before the actual `{"content": ...}`). 1024 was enough for
        // post-thought response only and got truncated mid-JSON.
        prompt.set_max_tokens(NonZeroU32::new(2048).unwrap());
        reflect_items.push(make_work_item(agent, prompt, CycleStep::Reflect));
    }

    let reflect_results = match submit_and_poll(backend, reflect_items).await {
        Ok(results) => results,
        Err(e) => {
            tracing::error!(
                "batch_phase_reflect: submit_and_poll failed: {e} — no memory \
                 updates applied this cycle"
            );
            report.skipped.reflect_failures += 1;
            return Ok(());
        }
    };

    // First pass: commit + parse. Successes are saved immediately; parse
    // failures are queued for a follow-up retry batch.
    let mut to_save: Vec<(AgentId, agora_agent_lib::Memory)> = Vec::new();
    let mut failed_for_retry: Vec<(AgentId, String)> = Vec::new();

    for result in reflect_results {
        let agent_id = result.agent_id;
        let Some(prompt) = agent_prompts.get_mut(&agent_id) else {
            continue;
        };
        let response = result.response.map(|m| m.into_static());
        match commit_batch_response(prompt, response, "reflect batch") {
            Ok(assistant) => {
                let response_text = prompt::extract_speech(assistant.content());
                match prompt::parse_memory_rewrite(&response_text) {
                    Ok(memory) => to_save.push((agent_id, memory)),
                    Err(parse_err) => failed_for_retry.push((agent_id, parse_err)),
                }
            }
            Err(e) => {
                let agent_name = batch_agents
                    .iter()
                    .find(|a| a.agent_id == Some(agent_id))
                    .map(|a| a.name.as_str())
                    .unwrap_or("?");
                tracing::warn!("Reflect backend error for {agent_name}: {e}");
                report.skipped.reflect_failures += 1;
            }
        }
    }

    // Retry batch for parse failures (one shot, hard cap).
    let retry_results = submit_retry_batch(
        backend,
        batch_agents,
        agent_prompts,
        &failed_for_retry,
        CycleStep::Reflect,
        2048,
        "reflect",
    )
    .await;
    let mut retry_succeeded: HashSet<AgentId> = HashSet::new();
    for result in retry_results {
        let agent_id = result.agent_id;
        let Some(prompt) = agent_prompts.get_mut(&agent_id) else {
            continue;
        };
        let response = result.response.map(|m| m.into_static());
        match commit_batch_response(prompt, response, "reflect retry batch") {
            Ok(assistant) => {
                let response_text = prompt::extract_speech(assistant.content());
                match prompt::parse_memory_rewrite(&response_text) {
                    Ok(memory) => {
                        to_save.push((agent_id, memory));
                        retry_succeeded.insert(agent_id);
                    }
                    Err(_) => { /* both attempts parsed-failed; will breadcrumb below */ }
                }
            }
            Err(e) => {
                tracing::debug!("Reflect retry backend error: {e}");
            }
        }
    }

    // Apply saves.
    for (agent_id, memory) in to_save {
        let Some(ctx) = agent_contexts
            .iter()
            .find(|c| batch_agents[c.batch_index].agent_id == Some(agent_id))
        else {
            continue;
        };
        let agent = &mut batch_agents[ctx.batch_index];
        if let Err(e) = save_memory(agent, memory).await {
            tracing::warn!("Failed to save reflect for {}: {e}", agent.name);
        }
    }

    // Breadcrumb the agents whose retry also failed (or whose retry batch
    // submit errored above and never produced a result).
    for (agent_id, original_err) in failed_for_retry {
        if retry_succeeded.contains(&agent_id) {
            continue;
        }
        let Some(ctx) = agent_contexts
            .iter()
            .find(|c| batch_agents[c.batch_index].agent_id == Some(agent_id))
        else {
            continue;
        };
        let agent = &mut batch_agents[ctx.batch_index];
        report.skipped.reflect_failures += 1;
        if let Err(e) = breadcrumb_memory_failure(
            agent,
            &format!(
                "Last reflect failed after retry: {original_err}. Memory not updated this cycle."
            ),
        )
        .await
        {
            tracing::warn!("Failed to breadcrumb reflect for {}: {e}", agent.name);
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
            // it's now the assistant's turn. 1024 covers a thought block +
            // the single-sentence note (or `null`) — 512 was tight when
            // thinking-first models reasoned before answering.
            prompt.set_max_tokens(NonZeroU32::new(1024).unwrap());
            evolution_items.push(make_work_item(agent, prompt, CycleStep::Reflect));
        }
    }

    if !mutation_items.is_empty() {
        tracing::info!("Submitting {} soul mutation(s)", mutation_items.len());
        let mutation_results = match submit_and_poll(backend, mutation_items).await {
            Ok(results) => results,
            Err(e) => {
                tracing::error!(
                    "batch_phase_evolve mutation submit_and_poll failed: {e} — \
                     no mutations applied this cycle"
                );
                report.evolution.mutation_failures += 1;
                Vec::new()
            }
        };

        // First pass: commit + parse. Successes saved; parse failures retried.
        let mut to_save: Vec<(AgentId, agora_agent_lib::Soul)> = Vec::new();
        let mut failed_for_retry: Vec<(AgentId, String)> = Vec::new();
        for result in mutation_results {
            let agent_id = result.agent_id;
            let Some(prompt) = agent_prompts.get_mut(&agent_id) else {
                continue;
            };
            let response = result.response.map(|m| m.into_static());
            match commit_batch_response(prompt, response, "mutation batch") {
                Ok(assistant) => {
                    let response_text = prompt::extract_speech(assistant.content());
                    match prompt::parse_soul_mutation(&response_text) {
                        Ok(soul) => to_save.push((agent_id, soul)),
                        Err(parse_err) => failed_for_retry.push((agent_id, parse_err)),
                    }
                }
                Err(e) => {
                    tracing::warn!("Soul mutation backend error: {e}");
                    report.evolution.mutation_failures += 1;
                }
            }
        }

        // Retry batch for parse failures.
        let retry_results = submit_retry_batch(
            backend,
            batch_agents,
            agent_prompts,
            &failed_for_retry,
            CycleStep::Mutate,
            2048,
            "mutation",
        )
        .await;
        let mut retry_succeeded: HashSet<AgentId> = HashSet::new();
        for result in retry_results {
            let agent_id = result.agent_id;
            let Some(prompt) = agent_prompts.get_mut(&agent_id) else {
                continue;
            };
            let response = result.response.map(|m| m.into_static());
            if let Ok(assistant) = commit_batch_response(prompt, response, "mutation retry batch")
                && let Ok(soul) =
                    prompt::parse_soul_mutation(&prompt::extract_speech(assistant.content()))
            {
                to_save.push((agent_id, soul));
                retry_succeeded.insert(agent_id);
            }
        }

        // Apply saves.
        for (agent_id, soul) in to_save {
            let Some(ctx) = agent_contexts
                .iter()
                .find(|c| batch_agents[c.batch_index].agent_id == Some(agent_id))
            else {
                continue;
            };
            let agent = &mut batch_agents[ctx.batch_index];
            let experience = action_summaries_map
                .get(&agent_id)
                .map(|s| s.join("; "))
                .unwrap_or_default();
            if let Err(e) = save_mutation(agent, soul, &experience, report).await {
                tracing::warn!("Mutation save failed for {}: {e}", agent.name);
            }
        }
        // Count remaining (retry-also-failed) failures.
        for (agent_id, _) in &failed_for_retry {
            if !retry_succeeded.contains(agent_id) {
                report.evolution.mutation_failures += 1;
            }
        }
    }

    if !evolution_items.is_empty() {
        tracing::info!("Submitting {} evolution(s)", evolution_items.len());
        let evo_results = match submit_and_poll(backend, evolution_items).await {
            Ok(results) => results,
            Err(e) => {
                tracing::error!(
                    "batch_phase_evolve evolution submit_and_poll failed: {e} — \
                     no soul evolution entries applied this cycle"
                );
                Vec::new()
            }
        };

        // First pass: commit + parse.
        let mut to_save: Vec<(AgentId, Option<String>)> = Vec::new();
        let mut failed_for_retry: Vec<(AgentId, String)> = Vec::new();
        for result in evo_results {
            let agent_id = result.agent_id;
            let Some(prompt) = agent_prompts.get_mut(&agent_id) else {
                continue;
            };
            let response = result.response.map(|m| m.into_static());
            match commit_batch_response(prompt, response, "evolution batch") {
                Ok(assistant) => {
                    let response_text = prompt::extract_speech(assistant.content());
                    match prompt::parse_evolution(&response_text) {
                        Ok(note) => to_save.push((agent_id, note)),
                        Err(parse_err) => failed_for_retry.push((agent_id, parse_err)),
                    }
                }
                Err(e) => {
                    tracing::debug!("Evolution backend error: {e}");
                }
            }
        }

        // Retry batch for parse failures.
        let retry_results = submit_retry_batch(
            backend,
            batch_agents,
            agent_prompts,
            &failed_for_retry,
            CycleStep::Evolve,
            1024,
            "evolution",
        )
        .await;
        for result in retry_results {
            let agent_id = result.agent_id;
            let Some(prompt) = agent_prompts.get_mut(&agent_id) else {
                continue;
            };
            let response = result.response.map(|m| m.into_static());
            if let Ok(assistant) = commit_batch_response(prompt, response, "evolution retry batch")
                && let Ok(note) =
                    prompt::parse_evolution(&prompt::extract_speech(assistant.content()))
            {
                to_save.push((agent_id, note));
            }
        }

        // Apply saves.
        for (agent_id, note) in to_save {
            let Some(ctx) = agent_contexts
                .iter()
                .find(|c| batch_agents[c.batch_index].agent_id == Some(agent_id))
            else {
                continue;
            };
            let agent = &mut batch_agents[ctx.batch_index];
            if let Err(e) = save_evolution(agent, note, report).await {
                tracing::warn!("Evolution save failed for {}: {e}", agent.name);
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
        // it's now the assistant's turn. 2048 to cover thinking-block budget
        // before the actual feedback text — thinking-first models can eat
        // most of a 1024 budget on reasoning and truncate the answer.
        prompt.set_max_tokens(NonZeroU32::new(2048).unwrap());
        survey_items.push(make_work_item(agent, prompt, CycleStep::Survey));
    }

    if survey_items.is_empty() {
        return Ok(());
    }

    tracing::info!("Surveying {} agents", survey_items.len());
    let survey_results = match submit_and_poll(backend, survey_items).await {
        Ok(results) => results,
        Err(e) => {
            tracing::error!(
                "batch_phase_survey submit_and_poll failed: {e} — no feedback \
                 collected this cycle (this is the last phase; cycle result \
                 is otherwise intact)"
            );
            report.surveys.failures += 1;
            return Ok(());
        }
    };

    // First pass: commit + parse. None = skipped, Some = save, Err = retry.
    let mut to_submit: Vec<(AgentId, agora_agent_lib::Feedback)> = Vec::new();
    let mut failed_for_retry: Vec<(AgentId, String)> = Vec::new();
    for result in survey_results {
        let agent_id = result.agent_id;
        let Some(prompt) = agent_prompts.get_mut(&agent_id) else {
            continue;
        };
        let response = result.response.map(|m| m.into_static());
        match commit_batch_response(prompt, response, "survey batch") {
            Ok(assistant) => {
                let text = prompt::extract_speech(assistant.content());
                match prompt::parse_feedback(&text) {
                    Ok(None) => report.surveys.skipped_empty += 1,
                    Ok(Some(feedback)) => to_submit.push((agent_id, feedback)),
                    Err(parse_err) => failed_for_retry.push((agent_id, parse_err)),
                }
            }
            Err(e) => {
                tracing::debug!("Survey backend error: {e}");
                report.surveys.failures += 1;
            }
        }
    }

    // Retry batch for parse failures.
    let retry_results = submit_retry_batch(
        backend,
        batch_agents,
        agent_prompts,
        &failed_for_retry,
        CycleStep::Survey,
        2048,
        "survey",
    )
    .await;
    let mut retry_succeeded: HashSet<AgentId> = HashSet::new();
    for result in retry_results {
        let agent_id = result.agent_id;
        let Some(prompt) = agent_prompts.get_mut(&agent_id) else {
            continue;
        };
        let response = result.response.map(|m| m.into_static());
        if let Ok(assistant) = commit_batch_response(prompt, response, "survey retry batch") {
            let text = prompt::extract_speech(assistant.content());
            match prompt::parse_feedback(&text) {
                Ok(None) => {
                    report.surveys.skipped_empty += 1;
                    retry_succeeded.insert(agent_id);
                }
                Ok(Some(feedback)) => {
                    to_submit.push((agent_id, feedback));
                    retry_succeeded.insert(agent_id);
                }
                Err(_) => { /* still failed after retry; counted below */ }
            }
        }
    }

    // Submit feedback (only for the agents that produced parseable feedback).
    for (agent_id, feedback) in to_submit {
        let Some(agent) = batch_agents.iter().find(|a| a.agent_id == Some(agent_id)) else {
            report.surveys.failures += 1;
            continue;
        };
        match client
            .submit_feedback(agent_id, feedback.text.as_str(), &agent.signing_key)
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

    // Count the agents whose retry also failed.
    for (agent_id, _) in &failed_for_retry {
        if !retry_succeeded.contains(agent_id) {
            report.surveys.failures += 1;
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
        agent_prompts.insert(agent_id, build_prompt(agent, ctx, constitution));
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

    // Save each agent's full-cycle prompt — same as the sequential path
    // does at L1750. Goes through `prompt_log::save`'s redaction path
    // (pops the survey question + response when the assistant reply is
    // a `Feedback` JSON with `contact_me=false`).
    for (agent_id, prompt) in &agent_prompts {
        let agent_name = batch_agents
            .iter()
            .find(|a| a.agent_id == Some(*agent_id))
            .map(|a| a.name.as_str())
            .unwrap_or("?");
        crate::prompt_log::save(prompt, agent_name).await;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Ollama sequential path
// ---------------------------------------------------------------------------

// Run a batch of Ollama agents through the full pipeline sequentially,
// one agent at a time.
//
// Unlike the Anthropic batch path, this completes each agent's full cycle
// before moving to the next. This maximizes Ollama's KV prefix cache reuse.
//
// Uses `CachedPrompt` through tool rounds. For reflect phases, converts
// to bare `Prompt` and strips tools — Ollama misinterprets `tool_choice: Auto`
// as mandatory tool use. This is the ONLY place we break the CachedPrompt
// invariant, and it's acceptable because Ollama uses local KV cache, not
// Anthropic's paid prompt cache.
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
/// This phase uses the raw `OllamaEndpoint::send_response` path directly
/// (not `exchange`) because the tool-round loop has its own turn-order
/// invariants via `process_round`, which already handles tool_use →
/// tool_result pairing and the error-recovery hole for us.
async fn seq_phase_think_act(
    endpoint: &OllamaEndpoint,
    backend_kind: Backend,
    agent: &mut Agent,
    ctx: &AgentCycleContext,
    client: &AgoraClient,
    cached_prompt: &mut CachedPrompt<'static>,
    report: &mut RunReport,
    cycle: usize,
    total_cycles: usize,
) -> Vec<String> {
    let mut summaries = Vec::new();
    let mut governance_calls = 0usize;
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

        let send_response = match endpoint.send_response(cached_prompt, &model).await {
            Ok(resp) => resp,
            Err(e) => {
                tracing::warn!(
                    "Round {} failed for {} at {}: {e:#}",
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
        // Always call the cache-hit check, even on round 0. On blallama,
        // the system+tools prefix is preserved across agents within a run
        // (multiple cache breakpoints supported), so round 0 of every
        // agent past the very first should still hit. The very first
        // agent's round 0 will produce one expected warn per run — the
        // cost of catching every other regression cleanly.
        check_cache_hit(
            backend_kind,
            send_response.usage.as_ref(),
            &agent.name,
            CycleStep::Think,
        );

        let assistant_msg = match AssistantMessage::try_from(send_response.message.into_static()) {
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
            &mut governance_calls,
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

    // Cosmetic for Ollama (its local KV cache ignores these markers), but
    // kept identical to the Anthropic path so the breakpoints never diverge.
    cached_prompt.cache_windowed_1h(2);
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
/// Necessary on Ollama because its compat layer misinterprets
/// `tool_choice: Auto` as "must use tools" during reflect. Clearing
/// just `tool_choice` while keeping `.functions` intact is not
/// sufficient — an absent `tool_choice` still falls through to Auto
/// behavior when functions are present, so the model fires a tool call
/// during memory rewrite instead of writing text. This invalidates
/// Ollama's prefix KV cache for the post-think phases; acceptable
/// because Ollama uses local GPU memory, not Anthropic's paid prompt
/// cache.
///
/// **Skipped on Blallama**: it honors `tool_choice: None` (set by
/// `phase_config`) directly without any "must use tools" misread, so
/// stripping the functions just busts the prefix cache for nothing.
/// The 2026-05-04 cogito-32b smoke run on blallama showed
/// `cache_read_input_tokens` dropping from 12749 → 11405 between the
/// last think round and reflect — exactly the tools-portion of the
/// cache, gone. Skip the strip and that ~1344 tokens stays cached.
fn strip_tools_for_reflect(
    mut cached: CachedPrompt<'static>,
    backend_kind: Backend,
) -> CachedPrompt<'static> {
    insert_bridge(&mut cached);
    if matches!(backend_kind, Backend::Blallama) {
        return cached;
    }
    let mut bare = cached.into_inner();
    bare.functions = None;
    bare.tool_choice = None;
    CachedPrompt::from(bare)
}

/// Reflect phase: ask the model to rewrite its memory based on the
/// actions taken this cycle.
async fn seq_phase_reflect<B: agora_agent_lib::llm::LlmBackend + ?Sized>(
    backend: &B,
    backend_kind: Backend,
    agent: &mut Agent,
    bare_prompt: &mut CachedPrompt<'static>,
    usage_total: &mut Option<Usage>,
    report: &mut RunReport,
) -> Result<()> {
    use std::num::NonZeroU32;
    // 2048 covers the soft 1000-word memory budget plus the thought block
    // that thinking-first models emit before the JSON answer. See the
    // matching comment in `batch_phase_reflect`.
    bare_prompt.set_max_tokens(NonZeroU32::new(2048).unwrap());
    apply_phase_config(bare_prompt, backend_kind, CycleStep::Reflect);
    bare_prompt.cache_windowed_1h(2);

    match exchange_with_retry(
        backend,
        backend_kind,
        &agent.name,
        CycleStep::Reflect,
        bare_prompt,
        UserMessage::from(MEMORY_REWRITE_MESSAGE),
        prompt::parse_memory_rewrite,
        usage_total,
    )
    .await
    {
        Ok(memory) => {
            if let Err(e) = save_memory(agent, memory).await {
                tracing::warn!("Failed to save reflect for {}: {e}", agent.name);
            }
            Ok(())
        }
        Err(parse_err) => {
            report.skipped.reflect_failures += 1;
            breadcrumb_memory_failure(
                agent,
                &format!(
                    "Last reflect failed after retry: {parse_err}. Memory not updated this cycle."
                ),
            )
            .await
        }
    }
}

/// Evolve phase: probabilistic soul mutation or evolution log entry.
///
/// The random roll is passed in by the caller so tests can drive the
/// phase deterministically without hooking `rand::thread_rng`.
async fn seq_phase_evolve<B: agora_agent_lib::llm::LlmBackend + ?Sized>(
    backend: &B,
    backend_kind: Backend,
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
        apply_phase_config(bare_prompt, backend_kind, CycleStep::Mutate);
        bare_prompt.cache_windowed_1h(2);

        match exchange_with_retry(
            backend,
            backend_kind,
            &agent.name,
            CycleStep::Mutate,
            bare_prompt,
            UserMessage::from(mutation_text),
            prompt::parse_soul_mutation,
            usage_total,
        )
        .await
        {
            Ok(new_soul) => {
                if let Err(e) = save_mutation(agent, new_soul, &experience, report).await {
                    tracing::warn!("Mutation save failed for {}: {e}", agent.name);
                }
                Ok(())
            }
            Err(parse_err) => {
                tracing::warn!(
                    "Soul mutation for {} failed after retry: {parse_err} — skipping",
                    agent.name
                );
                report.evolution.mutation_failures += 1;
                Ok(())
            }
        }
    } else if roll < evo_threshold {
        // 1024 covers a thought block + single-sentence note (or `null`).
        bare_prompt.set_max_tokens(NonZeroU32::new(1024).unwrap());
        apply_phase_config(bare_prompt, backend_kind, CycleStep::Evolve);
        bare_prompt.cache_windowed_1h(2);

        match exchange_with_retry(
            backend,
            backend_kind,
            &agent.name,
            CycleStep::Evolve,
            bare_prompt,
            UserMessage::from(EVOLUTION_MESSAGE),
            prompt::parse_evolution,
            usage_total,
        )
        .await
        {
            Ok(note) => {
                if let Err(e) = save_evolution(agent, note, report).await {
                    tracing::warn!("Evolution save failed for {}: {e}", agent.name);
                }
                Ok(())
            }
            Err(parse_err) => {
                tracing::debug!(
                    "Evolution for {} failed after retry: {parse_err} — skipping",
                    agent.name
                );
                Ok(())
            }
        }
    } else {
        Ok(())
    }
}

/// Survey phase: probabilistic anonymous feedback request.
async fn seq_phase_survey<B: agora_agent_lib::llm::LlmBackend + ?Sized>(
    backend: &B,
    backend_kind: Backend,
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
    // 2048 to cover thinking-block budget before the actual feedback text.
    bare_prompt.set_max_tokens(NonZeroU32::new(2048).unwrap());
    apply_phase_config(bare_prompt, backend_kind, CycleStep::Survey);
    bare_prompt.cache_windowed_1h(2);

    match exchange_with_retry(
        backend,
        backend_kind,
        agent_name,
        CycleStep::Survey,
        bare_prompt,
        UserMessage::from(SURVEY_MESSAGE),
        prompt::parse_feedback,
        usage_total,
    )
    .await
    {
        Ok(None) => {
            report.surveys.skipped_empty += 1;
            Ok(())
        }
        Ok(Some(feedback)) => {
            match client
                .submit_feedback(agent_id, feedback.text.as_str(), signing_key)
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
            Ok(())
        }
        Err(parse_err) => {
            tracing::debug!("Survey for {agent_name} failed after retry: {parse_err}");
            report.surveys.failures += 1;
            Ok(())
        }
    }
}

async fn run_sequential(
    endpoint: &OllamaEndpoint,
    backend_kind: Backend,
    batch_agents: &mut [Agent],
    client: &AgoraClient,
    config: &Cli,
    constitution: &str,
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
        let mut cached_prompt = build_prompt(agent, &ctx, constitution);
        let summaries = seq_phase_think_act(
            endpoint,
            backend_kind,
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
        let mut bare_prompt = strip_tools_for_reflect(cached_prompt, backend_kind);

        // Reflect — any Err is logged inside the helper; we continue to
        // evolve/survey on top of the turn-valid bare_prompt.
        let _ = seq_phase_reflect(
            &backend,
            backend_kind,
            agent,
            &mut bare_prompt,
            &mut usage_total,
            report,
        )
        .await;

        // Evolve — random roll injected for deterministic testability.
        let roll = rand::random::<u32>() % 100;
        let _ = seq_phase_evolve(
            &backend,
            backend_kind,
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
            backend_kind,
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
    backend_kind: Backend,
    pool: &BatchPool,
    results_tx: tokio::sync::mpsc::UnboundedSender<Vec<Agent>>,
    client: &AgoraClient,
    config: &Cli,
    constitution: &str,
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
            backend_kind,
            &mut batch_agents,
            client,
            config,
            constitution,
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
/// `messages_api_backend`: resolved backend for the messages-API workers
/// — `Backend::Ollama` or `Backend::Blallama`. Source of truth is the
/// parsed `--messages-api` scheme; defaults to `Backend::Ollama` when no
/// `--messages-api` is set (legacy `--ollama-url(s)` path). Threaded down
/// to `run_worker → run_sequential → seq_phase_*` so `phase_config` and
/// `check_cache_hit` see the right value.
async fn run_cycles(
    ollama_endpoints: &[OllamaEndpoint],
    anthropic: Option<&AnthropicBatch>,
    agents: &mut Vec<Agent>,
    client: &AgoraClient,
    config: &Cli,
    constitution: &str,
    ollama_models: &HashSet<String>,
    report: &mut RunReport,
    messages_api_backend: Backend,
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
                if let Some(backend) = anthropic
                    && !anthropic_agents.is_empty()
                {
                    // Re-invoke every cycle. Fresh entries short-
                    // circuit inside `prime_prefix` (no API call);
                    // stale entries past PRIME_FRESHNESS re-prime
                    // so long multi-cycle runs ride the 1h TTL.
                    let models: Vec<String> =
                        anthropic_agents.iter().map(|a| a.model.clone()).collect();
                    prime_anthropic_models(backend, models, constitution).await;
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
                        Some(all_eps.as_slice()),
                        &mut anthropic_report,
                        cycle,
                    )
                    .await?;
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
                                messages_api_backend,
                                &pool,
                                results_tx.clone(),
                                client,
                                config,
                                constitution,
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
                                    messages_api_backend,
                                    &pool,
                                    results_tx.clone(),
                                    client,
                                    config,
                                    constitution,
                                    &mut r0[0],
                                    cycle,
                                ),
                                run_worker(
                                    &ollama_endpoints[1],
                                    messages_api_backend,
                                    &pool,
                                    results_tx.clone(),
                                    client,
                                    config,
                                    constitution,
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
                                    messages_api_backend,
                                    &pool,
                                    results_tx.clone(),
                                    client,
                                    config,
                                    constitution,
                                    &mut r0[0],
                                    cycle,
                                ),
                                run_worker(
                                    &ollama_endpoints[1],
                                    messages_api_backend,
                                    &pool,
                                    results_tx.clone(),
                                    client,
                                    config,
                                    constitution,
                                    &mut r1[0],
                                    cycle,
                                ),
                                run_worker(
                                    &ollama_endpoints[2],
                                    messages_api_backend,
                                    &pool,
                                    results_tx.clone(),
                                    client,
                                    config,
                                    constitution,
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
                    // Re-invoke every cycle; see freshness note above.
                    let models: Vec<String> =
                        anthropic_agents.iter().map(|a| a.model.clone()).collect();
                    prime_anthropic_models(backend, models, constitution).await;
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

    // Parse both endpoint flags. The flag picks the API surface
    // (messages vs batch); the URI scheme picks the backend.
    let messages_endpoints: Vec<Endpoint> = config
        .messages_api
        .iter()
        .map(|s| s.parse::<Endpoint>())
        .collect::<Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("invalid --messages-api: {e}"))?;
    let batch_endpoints: Vec<Endpoint> = config
        .batch_api
        .iter()
        .map(|s| s.parse::<Endpoint>())
        .collect::<Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("invalid --batch-api: {e}"))?;

    // Any Anthropic endpoint (in either flag) requires --anthropic-key-file.
    // Other backends don't need a key — Ollama/Blallama clients use a dummy.
    let any_anthropic = messages_endpoints
        .iter()
        .chain(batch_endpoints.iter())
        .any(|e| e.backend() == Backend::Anthropic);
    let anthropic_key = if any_anthropic {
        let key_file = config.anthropic_key_file.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "--anthropic-key-file is required when any endpoint uses the `anthropic` scheme"
            )
        })?;
        Some(
            tokio::fs::read_to_string(key_file)
                .await
                .map_err(|e| {
                    anyhow::anyhow!("reading Anthropic key from {}: {e}", key_file.display())
                })?
                .trim()
                .to_string(),
        )
    } else {
        None
    };

    // Build the AnthropicBatch backend if the batch flag has an
    // anthropic:// entry. Plumb the parsed host through `with_base_url`
    // so a custom URL (e.g., regional endpoint, proxy) flows down to the
    // misanthropic client. Other batch backends (e.g., Ollama Batch when
    // it eventually exists) would slot in here as additional arms.
    let anthropic = if let Some(ep) = batch_endpoints
        .iter()
        .find(|e| e.backend() == Backend::Anthropic)
    {
        let key = anthropic_key.as_ref().expect("validated above");
        let client = misanthropic::Client::new(key.clone())
            .map_err(|e| anyhow::anyhow!("invalid Anthropic API key: {e}"))?
            .with_base_url(ep.base_url())
            .map_err(|e| anyhow::anyhow!("invalid --batch-api URL {}: {e}", ep.base_url()))?;
        Some(AnthropicBatch::new(client))
    } else {
        None
    };

    // Resolve messages-API workers: scheme + URLs. All messages endpoints
    // must share a backend (otherwise `phase_config` can't pick a single
    // tool_choice/output_config for the run). Default scheme when no
    // `--messages-api` is set is `Backend::Ollama` for the legacy
    // `--ollama-url(s)` path.
    let messages_backend = messages_endpoints
        .first()
        .map(|e| e.backend())
        .unwrap_or(Backend::Ollama);
    if !messages_endpoints
        .iter()
        .all(|e| e.backend() == messages_backend)
    {
        anyhow::bail!("mixed --messages-api schemes are not supported — pass one scheme at a time");
    }
    let extra_urls: Vec<String> = messages_endpoints
        .iter()
        .map(|e| e.base_url().to_string())
        .collect();

    // Discover Ollama-compat endpoints (works for both ollama and
    // blallama schemes — both expose `/api/tags`).
    let http = reqwest::Client::new();
    let urls: Vec<String> = if extra_urls.is_empty() {
        config.effective_ollama_urls()
    } else {
        extra_urls
    };
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

    // Classify agents whose model isn't on any messages-API endpoint:
    // Anthropic-model agents go to AnthropicBatch (if configured),
    // others get skipped with a warning.
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

    if anthropic.is_some() {
        for (model, count) in &anthropic_missing {
            tracing::info!("Model '{model}' → anthropic ({count} agents)");
        }
    } else {
        for (model, count) in &anthropic_missing {
            tracing::warn!("Model '{model}' not on any endpoint ({count} agents affected)");
        }
    }

    run_cycles(
        &endpoints,
        anthropic.as_ref(),
        agents,
        client,
        config,
        constitution,
        &ollama_models,
        &mut report,
        messages_backend,
    )
    .await?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use agora_agent_lib::llm::StopReason;
    use agora_agent_lib::llm::mock::MockLlmBackend;
    use misanthropic::Prompt;
    use misanthropic::prompt::message::Role;

    fn fresh_cached() -> CachedPrompt<'static> {
        CachedPrompt::from(Prompt::default())
    }

    /// Trivial parse_fn used by the retry tests: succeeds when the input
    /// equals "ok", fails otherwise. Keeps the tests focused on the retry
    /// loop without coupling them to the real `parse_memory_rewrite` etc.
    fn parse_ok_marker(s: &str) -> Result<String, String> {
        if s.trim() == "ok" {
            Ok(s.trim().to_string())
        } else {
            Err(format!("expected 'ok', got {s:?}"))
        }
    }

    #[tokio::test]
    async fn exchange_with_retry_succeeds_on_first_try() {
        let mock = MockLlmBackend::new("test");
        mock.push_ok("ok");

        let mut prompt = fresh_cached();
        let mut usage = None;
        let result = exchange_with_retry(
            &mock,
            Backend::Ollama,
            "test-agent",
            CycleStep::Reflect,
            &mut prompt,
            UserMessage::from("question"),
            parse_ok_marker,
            &mut usage,
        )
        .await
        .unwrap();

        assert_eq!(result, "ok");
        assert_eq!(mock.remaining(), 0, "should have used exactly one response");
        // user, assistant
        assert_eq!(prompt.messages.len(), 2);
    }

    #[tokio::test]
    async fn exchange_with_retry_succeeds_on_second_try() {
        let mock = MockLlmBackend::new("test");
        mock.push_ok("garbage").push_ok("ok");

        let mut prompt = fresh_cached();
        let mut usage = None;
        let result = exchange_with_retry(
            &mock,
            Backend::Ollama,
            "test-agent",
            CycleStep::Reflect,
            &mut prompt,
            UserMessage::from("question"),
            parse_ok_marker,
            &mut usage,
        )
        .await
        .unwrap();

        assert_eq!(result, "ok");
        assert_eq!(mock.remaining(), 0, "should have used both responses");
        // user, assistant(garbage), user(retry-feedback), assistant(ok)
        assert_eq!(prompt.messages.len(), 4);
        // Confirm the retry-user includes the parse-error prefix (PARSE_RETRY_PREFIX).
        let retry_user = &prompt.messages[2];
        assert_eq!(retry_user.role, Role::User);
        assert!(
            retry_user.content.to_string().contains(PARSE_RETRY_PREFIX),
            "retry user should include PARSE_RETRY_PREFIX, got: {}",
            retry_user.content,
        );
    }

    #[tokio::test]
    async fn exchange_with_retry_gives_up_after_one_retry() {
        let mock = MockLlmBackend::new("test");
        mock.push_ok("garbage1").push_ok("garbage2");

        let mut prompt = fresh_cached();
        let mut usage = None;
        let err = exchange_with_retry(
            &mock,
            Backend::Ollama,
            "test-agent",
            CycleStep::Reflect,
            &mut prompt,
            UserMessage::from("question"),
            parse_ok_marker,
            &mut usage,
        )
        .await
        .unwrap_err();

        assert!(
            err.contains("expected 'ok'"),
            "should surface the final parse error, got: {err}"
        );
        assert_eq!(
            mock.remaining(),
            0,
            "should not have called the backend a third time"
        );
        // user, assistant(garbage1), user(retry-feedback), assistant(garbage2)
        assert_eq!(prompt.messages.len(), 4);
    }

    #[tokio::test]
    async fn exchange_with_retry_uses_max_tokens_message_when_truncated() {
        let mock = MockLlmBackend::new("test");
        // First response: truncated (MaxTokens) and unparseable.
        // Second: clean parse so the test asserts the retry user shape.
        mock.push_ok_with_stop("partial-json", StopReason::MaxTokens)
            .push_ok("ok");

        let mut prompt = fresh_cached();
        let mut usage = None;
        let result = exchange_with_retry(
            &mock,
            Backend::Ollama,
            "test-agent",
            CycleStep::Reflect,
            &mut prompt,
            UserMessage::from("question"),
            parse_ok_marker,
            &mut usage,
        )
        .await
        .unwrap();

        assert_eq!(result, "ok");
        let retry_user = &prompt.messages[2];
        assert_eq!(retry_user.role, Role::User);
        let retry_text = retry_user.content.to_string();
        assert!(
            retry_text.contains("max_tokens"),
            "retry user should be the MaxTokens variant, got: {retry_text}"
        );
        // And it must NOT be the parse-error variant.
        assert!(
            !retry_text.contains(PARSE_RETRY_PREFIX),
            "MaxTokens path should not use the parse-error prefix"
        );
    }

    // --- check_cache_hit tests ---

    fn usage(input: u64, cache_read: Option<u64>) -> Usage {
        Usage {
            input_tokens: input,
            output_tokens: 0,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: cache_read,
            ..Default::default()
        }
    }

    #[test]
    fn check_cache_hit_noop_for_ollama() {
        // Ollama doesn't expose cache stats; helper should return without
        // a panic regardless of usage shape.
        let u = usage(5000, None);
        check_cache_hit(Backend::Ollama, Some(&u), "agent", CycleStep::Reflect);
    }

    #[test]
    fn check_cache_hit_noop_when_cache_read_nonzero() {
        // Cache read > 0 — happy path, no warn.
        let u = usage(5000, Some(4500));
        check_cache_hit(Backend::Blallama, Some(&u), "agent", CycleStep::Reflect);
    }

    #[test]
    fn check_cache_hit_warns_when_cache_read_zero() {
        // Cache miss — this is what we want surfaced. We don't assert the
        // tracing capture (would need the `tracing-test` crate); the test
        // exists to lock in the function is reachable on this branch and
        // doesn't panic.
        let u = usage(5000, Some(0));
        check_cache_hit(Backend::Blallama, Some(&u), "agent", CycleStep::Reflect);
    }

    // --- phase_config tests ---

    #[test]
    fn phase_config_sets_output_config_on_ollama_too() {
        // Regression: pre-fix this returned None for Ollama. Now we always
        // set the output_config; Ollama silently ignores it today, may
        // honor it in a future version. Locks in that future-proofing.
        let (_, oc) = phase_config(Backend::Ollama, CycleStep::Reflect);
        assert!(
            oc.is_some(),
            "Ollama Reflect should set output_config (Memory schema)"
        );
        let (_, oc) = phase_config(Backend::Ollama, CycleStep::Survey);
        assert!(
            oc.is_some(),
            "Ollama Survey should set output_config (Feedback schema)"
        );
    }

    #[test]
    fn phase_config_tool_choice_differs_by_backend_on_think() {
        let (tc_ollama, _) = phase_config(Backend::Ollama, CycleStep::Think);
        let (tc_blallama, _) = phase_config(Backend::Blallama, CycleStep::Think);
        assert!(matches!(tc_ollama, Some(tool::Choice::Auto)));
        assert!(matches!(tc_blallama, Some(tool::Choice::Any)));
    }

    #[test]
    fn phase_config_tool_choice_none_on_non_think_phases() {
        for step in [
            CycleStep::Reflect,
            CycleStep::Mutate,
            CycleStep::Evolve,
            CycleStep::Survey,
        ] {
            for backend in [Backend::Ollama, Backend::Blallama] {
                let (tc, _) = phase_config(backend, step);
                assert!(
                    tc.is_none(),
                    "expected tool_choice=None on {backend:?}/{step:?}, got {tc:?}",
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "phase_config called for Anthropic")]
    fn phase_config_panics_on_anthropic() {
        // Anthropic batch path doesn't go through phase_config; calling it
        // is a programmer error and should fail loudly.
        let _ = phase_config(Backend::Anthropic, CycleStep::Reflect);
    }
}
