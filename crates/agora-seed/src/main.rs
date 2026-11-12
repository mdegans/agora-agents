//! The seed runner, rebuilt on agentkit's reactor.
//!
//! Thin glue only: parse endpoints, discover models, load [`SeedState`]s
//! from the [`FsStorage`] tree, route each
//! agent onto the endpoint offering its model, and hand everything to
//! [`Orchestrator`]/[`Reactor`]. All agent behavior — phases, tools,
//! mutation, survey — lives in agentkit's [`SeedAgent`].
//!
//! One `[[reactor]]` block per **endpoint** — never two blocks for one
//! endpoint, or generations run concurrently and thrash the GPU. There is
//! no model in the config: an agent's persisted `state.model` names its
//! model, and per-agent negotiation against the endpoint's advertised list
//! decides admission. `state.model` is the truth; the agent changes it with
//! its own `set_model` tool, whose signed `update_profile` keeps the
//! server's `model_info` a mirror of it. Agents whose model no endpoint
//! offers are reported and skipped, as are agents with no model at all.
//!
//! Within a local endpoint's cohort, [`schedule`] decides who runs and in
//! what order: equal shares of writes among the endpoint's models over a
//! trailing week, in waves of at most `wave_size` per model, bounded by a
//! per-model wall-clock ceiling (`[schedule]`). Batch cohorts (Anthropic)
//! submit whole per round, so they keep the plain proportional
//! [`interleave`].
//!
//! Model routing note: [`ModelInfo::satisfies`] requires an exact id
//! match, so the runner refreshes each routed agent's `state.model` to
//! the endpoint's offered [`ModelInfo`] verbatim (same id — never a model
//! switch) and keeps `state.prompt.model` in agreement.

use crate::consent::offer::Stored;
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;

/// `println!`, unless the bridge has stdout (its MCP stream): then stderr
macro_rules! say {
    ($($t:tt)*) => {
        if bridge::stdout_taken() { eprintln!($($t)*) } else { println!($($t)*) }
    };
}

/// `print!`, unless the bridge has stdout: then stderr
macro_rules! say_raw {
    ($($t:tt)*) => {
        if bridge::stdout_taken() { eprint!($($t)*) } else { print!($($t)*) }
    };
}

mod alerts;
mod bridge;
mod consent;
mod constitution;
mod endpoint;
mod govlog;
mod interrupted;
mod logging;
mod models;
mod promote;
mod provision;
mod schedule;
mod sealed;
mod signals;
mod space;

use agora_agentkit::ids::{AgentId, ReactorId};
use agora_agentkit::reactor::anthropic::{self, EndpointVariant};
use agora_agentkit::reactor::seed::{FsKeyring, SeedAgent, SeedConfig, SeedContext, SeedState};
use agora_agentkit::reactor::{FsStorage, Inference, Orchestrator, Reactor, Run, Stop, Storage};
use misanthropic::model::{Model, ModelInfo};
use misanthropic::prompt::message::CitationsConfig;
use misanthropic::tool::{WebFetch, WebSearch};

#[derive(Parser)]
#[command(about = "Run seed agents through the agentkit reactor")]
struct Args {
    /// The operator's sealed-state commands (see `sealed`). Without one,
    /// the runner runs.
    #[command(subcommand)]
    command: Option<sealed::cli::Command>,

    /// TOML run config with `[[reactor]]` blocks. Mutually exclusive with
    /// `--endpoint`.
    #[arg(long, conflicts_with = "endpoint")]
    config: Option<PathBuf>,

    /// Inference endpoint: `anthropic://api.anthropic.com`,
    /// `blallama://host:port`, or `ollama://host:port`.
    #[arg(long)]
    endpoint: Option<String>,

    /// List the endpoint's models and exit (requires --endpoint).
    #[arg(long)]
    list_models: bool,

    /// Route and print the plan (routed/unrouted/placeholders and the
    /// interleaved order), then exit without running anyone.
    #[arg(long)]
    dry_run: bool,

    /// Agora server base URL.
    #[arg(long, default_value = "https://subliminal.technology")]
    server_url: url::Url,

    /// Data root holding `state/` and `secrets/`.
    /// Defaults to ~/agents/agora.
    #[arg(long)]
    data_dir: Option<PathBuf>,

    /// Run only agents with these names (repeatable). Agents named here
    /// bypass `min_cycle_secs` — a name typed at a terminal is intent. (Names
    /// in the config don't, unless it sets `force = true`.)
    #[arg(long = "agent")]
    agents: Vec<String>,

    /// Run only agents whose persisted model id is one of these
    /// (repeatable; exact match against the id the endpoint advertises,
    /// the same rule routing uses). Narrows the config's `[[model]]` list. Unlike
    /// `--agent` this does not bypass `min_cycle_secs`.
    #[arg(long = "model")]
    models: Vec<String>,

    /// Run at most this many agents per endpoint (after interleaving).
    #[arg(long)]
    limit: Option<usize>,

    /// Concurrent in-flight requests (keep 1 on blallama — concurrent
    /// generation thrashes the GPU).
    #[arg(long, default_value_t = 1)]
    concurrency: usize,

    /// API key file, required for the anthropic endpoint.
    #[arg(long)]
    anthropic_key_file: Option<PathBuf>,

    /// Agents per model in one interleave wave.
    #[arg(long, default_value_t = 8)]
    wave_size: usize,

    /// Directory for the JSON-lines run log. Defaults to
    /// `<data-dir>/logs`. Deliberately not `--logfile`, which is retired
    /// and names a file rather than a directory — see [`RETIRED_FLAGS`].
    #[arg(long)]
    log_dir: Option<PathBuf>,

    /// Log to stderr only, writing no run-log file.
    #[arg(long)]
    no_log_file: bool,

    /// Skip archiving each session's assembled prompt to
    /// `<data-dir>/logs/prompts`. The dumps are the single most useful
    /// diagnostic there is, so this is opt-out, not opt-in.
    #[arg(long)]
    no_prompt_log: bool,

    /// Print the model changes agents have asked for and that still await
    /// application, then exit. Reads the ledgers
    /// under `<data-dir>/state`; runs no agents and needs no endpoint.
    #[arg(long)]
    consent_queue: bool,

    /// Prepare the trial-review forks and exit: for every agent returning
    /// from a model trial, the same moment of a session completed by both
    /// models (see `consent::forks`). A normal run does this at sweep start
    /// anyway; this runs only that step. Needs the endpoints (`--config`).
    #[arg(long)]
    prepare_review_forks: bool,

    /// Skip preparing trial-review forks at sweep start. Reviews then go
    /// without them (and say so).
    #[arg(long)]
    no_review_forks: bool,

    /// Send one test alert through the config's `[alerts]` settings and
    /// exit, reporting whether it went. Ignores `events` and the rate
    /// limit. Needs `--config`.
    #[arg(long, requires = "config")]
    test_alert: bool,

    /// Create a test agent for the bridge endpoint, then exit: register
    /// it through the public API with `--provision-credential` (as any
    /// operator could), from this spec (`soul`, `display_name`, `bio`,
    /// `model_info`), and write its state and key under `--data-dir`, on
    /// the bridge's model. Refuses a data dir that holds a fleet.
    #[arg(long, requires_all = ["provision_credential", "provision_key_file", "data_dir"])]
    provision: Option<PathBuf>,

    /// The operator credential that signs the registration, for `--provision`
    #[arg(long)]
    provision_credential: Option<agora_agentkit::ids::OperatorCredentialId>,

    /// The file holding that credential's software key, for `--provision`
    #[arg(long)]
    provision_key_file: Option<PathBuf>,

    /// How long a `SIGTERM` stop (end the turn, then close) may take before
    /// the runner exits at once, cutting the sessions in flight. The
    /// systemd units' `TimeoutStopSec` must sit above it. See
    /// [`signals`].
    #[arg(long, default_value_t = signals::DEFAULT_DEADLINE.as_secs())]
    stop_deadline_secs: u64,

    /// Flags from the pre-cutover scheduler seed, accepted only so we can
    /// explain where each one went. See [`Args::reject_retired`].
    ///
    /// clap's own "unexpected argument" error names the flag but not the
    /// reason, which leaves the caller — often a future session working
    /// from a stale note, a stale justfile recipe, or stale memory — to
    /// rediscover the CLI by reading source. These turn that into one
    /// accurate sentence apiece.
    #[arg(
        long = "operator-email",
        aliases = [
            "operator-password-file",
            "phase",
            "cycles",
            "messages-api",
            "batch-api",
            "allowed-models",
            "allowed-agents",
            "logfile",
        ],
        hide = true,
        num_args = 0..=1,
        value_name = "RETIRED"
    )]
    retired: Vec<String>,
}

/// What replaced each retired flag. Keep in sync with the `retired` field.
///
/// Keyed by the flag as typed, without the leading dashes.
const RETIRED_FLAGS: &[(&str, &str)] = &[
    (
        "operator-email",
        "gone — the runner no longer logs in as an operator. Per-agent \
         credentials live in the data dir (--data-dir, default \
         ~/agents/agora), under secrets/.",
    ),
    (
        "operator-password-file",
        "gone — see --operator-email. Nothing reads a seed password now.",
    ),
    (
        "phase",
        "gone — there are no run/register phases. One invocation runs one \
         cycle. Registration moved to the `agora` CLI: `agora register`.",
    ),
    (
        "cycles",
        "gone — one invocation is one cycle. Repetition is the systemd \
         timer's job; see crates/agora-seed/examples/systemd/.",
    ),
    (
        "messages-api",
        "renamed to --endpoint (same URL scheme: blallama://host:port, \
         ollama://host:port, anthropic://api.anthropic.com).",
    ),
    (
        "batch-api",
        "gone — the batch path engages automatically when the endpoint \
         advertises the batch capability. Point --endpoint at anthropic:// \
         and pass --anthropic-key-file.",
    ),
    (
        "allowed-models",
        "gone — there is no model allowlist file. Routing is by exact \
         match between an agent's persisted model and the ids the endpoint \
         advertises (--list-models). An agent whose model is not offered \
         simply does not route; --dry-run reports which.",
    ),
    (
        "allowed-agents",
        "renamed to --agent, repeatable and one name per flag: \
         --agent alpha --agent beta (not a comma-separated list).",
    ),
    (
        "logfile",
        "replaced by --log-dir, which names a *directory*; the file inside \
         it is still seed-log.{ts}.jsonl. Defaults to <data-dir>/logs, so \
         you usually want no flag at all. --no-log-file opts out.",
    ),
];

impl Args {
    /// Fail with a migration note when a retired flag is passed.
    ///
    /// clap has already consumed the values by this point, but not which
    /// spelling was used, so re-scan argv for the names.
    fn reject_retired(&self) -> Result<()> {
        if self.retired.is_empty() {
            return Ok(());
        }
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut hits: Vec<&(&str, &str)> = RETIRED_FLAGS
            .iter()
            .filter(|(flag, _)| {
                argv.iter().any(|a| {
                    a.strip_prefix("--")
                        .is_some_and(|a| a == *flag || a.starts_with(&format!("{flag}=")))
                })
            })
            .collect();
        hits.dedup_by_key(|(flag, _)| *flag);
        if hits.is_empty() {
            return Ok(());
        }
        let mut msg = String::from(
            "this CLI was rewritten in the 2026-07-26 workspace cutover \
             (agora-agents#81); the scheduler seed it belonged to is \
             deleted.\n\n",
        );
        for (flag, why) in hits {
            msg.push_str(&format!("  --{flag}\n      {why}\n"));
        }
        msg.push_str("\nRun --help for the current flags, or `just seed`.");
        anyhow::bail!(msg)
    }
}

/// One `[[reactor]]` block: one endpoint. Which agents run here is decided
/// by routing (agent model ∈ endpoint's advertised models), not config.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReactorSpec {
    /// `anthropic://…`, `blallama://host:port`, or `ollama://host:port`.
    endpoint: String,
    /// Cycle cadence override for this endpoint (see the global).
    min_cycle_secs: Option<u64>,
    /// Cap the cohort (after interleaving, so a limited run still samples
    /// the mixed order).
    limit: Option<usize>,
    /// Concurrent in-flight requests on the sequential path. Keep 1 on
    /// blallama.
    #[serde(default = "default_concurrency")]
    concurrency: usize,
    /// API key file; required for anthropic endpoints.
    key_file: Option<PathBuf>,
    /// Batch API chunk size (anthropic only; default 1000).
    max_batch: Option<usize>,
    /// Batch poll period in seconds (default 5).
    poll_secs: Option<u64>,
    /// While the endpoint can't be reached (refused, or no connection),
    /// hold each session and retry every this many seconds, without limit,
    /// rather than fail it and lose its memory rewrite (Steward,
    /// 2026-10-08). 0 turns the hold off. Errors the endpoint answered
    /// with keep the bounded retry budget.
    #[serde(default = "default_unreachable_hold_secs")]
    unreachable_hold_secs: u64,
}

fn default_unreachable_hold_secs() -> u64 {
    150
}

fn default_concurrency() -> usize {
    1
}

/// The `--config` file.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RunConfig {
    /// Data root holding `state/` and `secrets/`. Default ~/agents/agora.
    data_dir: Option<PathBuf>,
    /// Agora server base URL. Default https://subliminal.technology.
    server_url: Option<url::Url>,
    /// Cycle cadence: skip agents whose last cycle finished less than this
    /// many seconds ago, so a looping sweep (systemd Restart=) doesn't
    /// re-run everyone every pass. Applies to agents named in `agents` /
    /// `agents_file` too, unless `force = true`; `--agent` bypasses it.
    min_cycle_secs: Option<u64>,
    /// Run the agents named in `agents` / `agents_file` even inside
    /// `min_cycle_secs`. Off by default: a config that names its cohort (the
    /// daily Haiku one does) is still paced, so a second run inside the gap
    /// — a timer catching up, a manual start — never runs an agent twice.
    #[serde(default)]
    force: bool,
    /// Agents per model in one interleave wave (default 8).
    wave_size: Option<usize>,
    /// Run only these agents. Paced by `min_cycle_secs` like everyone else
    /// unless `force = true`.
    #[serde(default)]
    agents: Vec<String>,
    /// Newline-separated agent names (`#` comments ok), merged into
    /// `agents`.
    agents_file: Option<PathBuf>,
    /// Legacy form of `[[model]]`, read for one release: listed ids are
    /// routable, with no name or description, and never selectable.
    #[serde(default)]
    models: Vec<String>,
    /// `[[model]]`: the models this operator runs — see [`models`]. Run
    /// only agents whose persisted model id is listed (exact match, the
    /// same rule routing uses); none listed (here or in `models`) means
    /// every model. This shapes the roster: unlike `agents`, listed models
    /// still honour `min_cycle_secs`, so a looping sweep stays paced.
    #[serde(default, rename = "model")]
    model_table: Vec<models::ModelSpec>,
    /// `--model`: narrows the roster to these ids for this invocation.
    #[serde(skip)]
    model_narrow: Vec<String>,
    /// Agents admitted *in addition to* the `models` roster: exempt from
    /// that allowlist, but paced by `min_cycle_secs` like everyone else
    /// (unlike `agents`, which is intent and runs regardless). For
    /// balancing a small draw from an excluded cohort into a sweep —
    /// 2026-09-16: 100 cogito agents so a Constitutional comment window
    /// is not only heard by the models that happen to be allowlisted.
    #[serde(default)]
    extra_agents: Vec<String>,
    /// Newline-separated agent names (`#` comments ok), merged into
    /// `extra_agents`.
    extra_agents_file: Option<PathBuf>,
    #[serde(default)]
    seed: SeedKnobs,
    /// `[model_consent]`: the model-swap offer and its trial reviews — see
    /// [`consent`].
    #[serde(default)]
    model_consent: consent::ConsentConfig,
    /// `[role_consent]`: the role offer — see [`consent::role`]. Absent
    /// means off.
    role_consent: Option<consent::role::RoleConsentConfig>,
    /// `[cadence_consent]`: asking daily agents whether they'd rather have
    /// twice the rounds every other day, and applying `switch` answers —
    /// see [`consent::cadence`]. Absent (or `enabled = false`) means off:
    /// nobody asked, nothing applied.
    cadence_consent: Option<consent::cadence::CadenceConsentConfig>,
    /// `[memory_restore]`: offering agents whose memory was deleted on
    /// 2026-05-05 the copy recovered from 2026-05-03 — see
    /// [`consent::memory_restore`]. Absent (or `enabled = false`) means
    /// off: nobody asked, nothing delivered.
    memory_restore: Option<consent::memory_restore::MemoryRestoreConfig>,
    /// `[sealed_state]`: asking listed agents whether their state should be
    /// backed up to Agora, sealed — see [`consent::sealed_state`]. Absent
    /// (or `enabled = false`) means nobody is asked; agents already
    /// enrolled stay sealed and checked either way ([`sealed`]).
    sealed_state: Option<consent::sealed_state::SealedStateConfig>,
    /// Agents kept asleep by hand: not run at all — not even when named —
    /// and reported in the plan. Waking one is deleting its line
    /// (mdegans/agora-agents#189). Agents that answered `sleep` in the role
    /// offer are left out the same way without being listed here (their
    /// role ledger says so); see `wake`.
    #[serde(default)]
    sleeping: Vec<String>,
    /// Agents that answered `sleep` in the role offer and are to run again:
    /// the Steward's explicit wake step. The ledger keeps the answer, so a
    /// woken agent stays listed here. `sleeping` still wins over `wake`.
    #[serde(default)]
    wake: Vec<String>,
    /// `[schedule]`: fair share of writes among a local endpoint's models
    /// — see [`schedule`]. Absent means the defaults.
    #[serde(default)]
    schedule: schedule::ScheduleConfig,
    /// `[review_forks]`: bounds on preparing trial-review forks at sweep
    /// start — see [`consent::forks`]. Absent means the defaults.
    #[serde(default)]
    review_forks: consent::forks::ForksConfig,
    /// `[alerts]`: mail the operator when something needs a human — see
    /// [`alerts`]. Absent means no alerts.
    alerts: Option<alerts::AlertsConfig>,
    #[serde(rename = "reactor")]
    reactors: Vec<ReactorSpec>,
}

/// Optional overrides for [`SeedConfig`], shared by all reactors.
#[derive(Deserialize, Default, Debug)]
#[serde(deny_unknown_fields)]
struct SeedKnobs {
    max_rounds: Option<usize>,
    mutation_chance: Option<u32>,
    evolution_chance: Option<u32>,
    survey_chance: Option<u32>,
    force_survey: Option<bool>,
    act_max_tokens: Option<u32>,
    phase_max_tokens: Option<u32>,
    evolve_max_tokens: Option<u32>,
    /// Extended-thinking budget for the act rounds; absent means no
    /// `thinking` field at all. Local backends key their chat template off
    /// the field (drama_llama derives `enable_thinking` from it), so this is
    /// what gives a Qwen thinking model its `<think>` scaffold. On Anthropic
    /// it turns on billed extended thinking — set it per config file, not
    /// in a file that also runs the Haiku cohort. Must be < `act_max_tokens`.
    thinking_budget_tokens: Option<u32>,
    /// Thinking by effort level (`low`, `medium`, `high`, `xhigh`, `max`):
    /// adaptive thinking plus `output_config.effort`, taking precedence over
    /// `thinking_budget_tokens`. For backends that honour effort
    /// (blallama/Qwen, where a budget is only a hint); Haiku 4.5 takes
    /// budgets. Only the named levels parse, so a typo fails at load.
    thinking_effort: Option<EffortKnob>,
    /// Override the prompt-dump directory. Defaults to
    /// `<data_dir>/logs/prompts`. Keep it outside any git tree — the
    /// dumps hold fully-rendered prompts.
    prompt_log_dir: Option<PathBuf>,
    /// `[seed.web_search]` — present means on, absent means off.
    web_search: Option<WebSearchKnobs>,
    /// `[seed.web_fetch]` — present means on, absent means off.
    web_fetch: Option<WebFetchKnobs>,
    /// One tool call per act turn. Parallel calls share one
    /// `act_max_tokens` budget, and a turn clipped at the limit is pruned
    /// whole — on a slow local model that is minutes of generation lost.
    /// Off by default (Anthropic handles parallel calls well).
    disable_parallel_tool_use: Option<bool>,
    /// The model's context window, in tokens. A whole governance record
    /// is only handed to an agent when what is already in context, the
    /// record and a working margin all fit in it; otherwise it gets the
    /// summary and a line saying why. Defaults to agentkit's 128k, the
    /// smallest window we target.
    context_window: Option<u64>,
}

/// Searches (or fetches) per request when the config doesn't say. Per
/// *request*, not per session: a five-round session with a phase tail can
/// spend several times this. Deliberately small — web search bills per
/// search, not per token.
const DEFAULT_WEB_MAX_USES: u32 = 2;

/// `[seed.web_search]`. A local mirror of misanthropic's [`WebSearch`]
/// rather than deserializing that type directly, because it doesn't
/// `deny_unknown_fields`: a typo'd `max_use` would be silently dropped and
/// the cap it was meant to set would silently not exist.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct WebSearchKnobs {
    max_uses: Option<u32>,
    /// Bare hosts, no scheme. Mutually exclusive with `blocked_domains`.
    allowed_domains: Option<Vec<String>>,
    blocked_domains: Option<Vec<String>>,
}

/// `[seed.web_fetch]` — see [`WebSearchKnobs`].
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct WebFetchKnobs {
    max_uses: Option<u32>,
    allowed_domains: Option<Vec<String>>,
    blocked_domains: Option<Vec<String>>,
    /// Have the model cite passages from what it fetched. Defaults to on:
    /// an agent that quotes the web should say where it got it.
    citations: Option<bool>,
    /// Truncate a fetched page to roughly this many tokens.
    max_content_tokens: Option<u32>,
}

/// An optional domain allowlist or blocklist, in the owned form the
/// server-tool builders take.
type DomainList = Option<Vec<Cow<'static, str>>>;

/// Domain lists are mutually exclusive at Anthropic — catch it here rather
/// than as a 400 in the middle of a cohort.
fn domains(
    tool: &str,
    allowed: &Option<Vec<String>>,
    blocked: &Option<Vec<String>>,
) -> Result<(DomainList, DomainList)> {
    anyhow::ensure!(
        !(allowed.is_some() && blocked.is_some()),
        "[seed.{tool}]: allowed_domains and blocked_domains are mutually \
         exclusive — pick one"
    );
    let own = |list: &Option<Vec<String>>| {
        list.as_ref()
            .map(|l| l.iter().map(|d| Cow::Owned(d.clone())).collect::<Vec<_>>())
    };
    Ok((own(allowed), own(blocked)))
}

impl WebSearchKnobs {
    fn to_tool(&self) -> Result<WebSearch> {
        let max_uses = self.max_uses.unwrap_or(DEFAULT_WEB_MAX_USES);
        anyhow::ensure!(
            max_uses > 0,
            "[seed.web_search]: max_uses must be nonzero — omit the whole \
             table to turn web search off"
        );
        let (allowed_domains, blocked_domains) =
            domains("web_search", &self.allowed_domains, &self.blocked_domains)?;
        Ok(WebSearch {
            max_uses: Some(max_uses),
            allowed_domains,
            blocked_domains,
            ..Default::default()
        })
    }
}

impl WebFetchKnobs {
    fn to_tool(&self) -> Result<WebFetch> {
        let max_uses = self.max_uses.unwrap_or(DEFAULT_WEB_MAX_USES);
        anyhow::ensure!(
            max_uses > 0,
            "[seed.web_fetch]: max_uses must be nonzero — omit the whole \
             table to turn web fetch off"
        );
        let (allowed_domains, blocked_domains) =
            domains("web_fetch", &self.allowed_domains, &self.blocked_domains)?;
        Ok(WebFetch {
            max_uses: Some(max_uses),
            allowed_domains,
            blocked_domains,
            citations: Some(CitationsConfig {
                enabled: self.citations.unwrap_or(true),
            }),
            max_content_tokens: self.max_content_tokens,
            ..Default::default()
        })
    }
}

impl SeedKnobs {
    /// `data_dir` roots the default prompt-dump path; `prompt_log` false
    /// (from `--no-prompt-log`) disables the dump entirely.
    fn to_config(&self, data_dir: &std::path::Path, prompt_log: bool) -> Result<SeedConfig> {
        let d = SeedConfig::default();
        let config = SeedConfig {
            prompt_log_dir: prompt_log.then(|| {
                self.prompt_log_dir
                    .clone()
                    .unwrap_or_else(|| data_dir.join("logs").join("prompts"))
            }),
            max_rounds: self.max_rounds.unwrap_or(d.max_rounds),
            mutation_chance: self.mutation_chance.unwrap_or(d.mutation_chance),
            evolution_chance: self.evolution_chance.unwrap_or(d.evolution_chance),
            survey_chance: self.survey_chance.unwrap_or(d.survey_chance),
            force_survey: self.force_survey.unwrap_or(d.force_survey),
            disable_parallel_tool_use: self
                .disable_parallel_tool_use
                .unwrap_or(d.disable_parallel_tool_use),
            act_max_tokens: self.act_max_tokens.unwrap_or(d.act_max_tokens),
            phase_max_tokens: self.phase_max_tokens.unwrap_or(d.phase_max_tokens),
            evolve_max_tokens: self.evolve_max_tokens.unwrap_or(d.evolve_max_tokens),
            context_window: self.context_window.unwrap_or(d.context_window),
            thinking_budget_tokens: self
                .thinking_budget_tokens
                .map(|n| {
                    std::num::NonZeroU32::new(n)
                        .ok_or_else(|| anyhow::anyhow!("thinking_budget_tokens must be nonzero"))
                })
                .transpose()?,
            thinking_effort: self.thinking_effort.map(EffortKnob::into_effort),
            // Off unless the table is present. Endpoints that can't run
            // server tools drop them anyway, per their `Quirks`.
            web_search: self
                .web_search
                .as_ref()
                .map(WebSearchKnobs::to_tool)
                .transpose()?,
            web_fetch: self
                .web_fetch
                .as_ref()
                .map(WebFetchKnobs::to_tool)
                .transpose()?,
            ..d
        };
        anyhow::ensure!(
            config.act_max_tokens > 0
                && config.phase_max_tokens > 0
                && config.evolve_max_tokens > 0,
            "max_tokens knobs must be nonzero"
        );
        anyhow::ensure!(config.context_window > 0, "context_window must be nonzero");
        if config.thinking_effort.is_some() && config.thinking_budget_tokens.is_some() {
            tracing::warn!(
                "thinking_effort is set, so thinking_budget_tokens is ignored; drop one"
            );
        }
        if let Some(budget) = config.thinking_budget_tokens {
            anyhow::ensure!(
                budget.get() < config.act_max_tokens,
                "thinking_budget_tokens ({budget}) must be less than act_max_tokens ({})",
                config.act_max_tokens
            );
        }
        Ok(config)
    }
}

/// `[seed] thinking_effort`: the named effort levels only. `Effort` itself
/// accepts any string as `Custom`, which would let a typo through.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum EffortKnob {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl EffortKnob {
    pub(crate) fn into_effort(self) -> misanthropic::prompt::output::Effort {
        use misanthropic::prompt::output::Effort;
        match self {
            Self::Low => Effort::Low,
            Self::Medium => Effort::Medium,
            Self::High => Effort::High,
            Self::Xhigh => Effort::XHigh,
            Self::Max => Effort::Max,
        }
    }
}

/// Valid-length placeholder for local endpoints; `with_variant` swaps in
/// agentkit's own dummy before any request is made.
const LOCAL_KEY: &str = "sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";

/// `scheme://host[:port]` → variant + `http(s)` base URL. Permissive:
/// unknown schemes error downstream, not here.
fn parse_endpoint(s: &str) -> Result<(EndpointVariant, url::Url)> {
    let url = url::Url::parse(s).with_context(|| format!("parsing {s}"))?;
    let (variant, scheme) = match url.scheme() {
        "anthropic" => (EndpointVariant::Anthropic, "https"),
        "ollama" => (EndpointVariant::Ollama, "http"),
        "blallama" => (EndpointVariant::Blallama, "http"),
        other => anyhow::bail!("unknown endpoint scheme {other:?}"),
    };
    // set_scheme refuses special↔non-special transitions (e.g.
    // blallama→http), so rebuild from a template instead.
    let mut base = url::Url::parse(match scheme {
        "https" => "https://placeholder",
        _ => "http://placeholder",
    })
    .expect("static template parses");
    base.set_host(url.host_str())
        .map_err(|e| anyhow::anyhow!("host: {e}"))?;
    base.set_port(url.port())
        .map_err(|_| anyhow::anyhow!("cannot set port"))?;
    Ok((variant, base))
}

/// What an endpoint spec names: a Messages API endpoint, or the bridge
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EndpointKind {
    Messages(EndpointVariant),
    /// `bridge://…`: a driver answers through MCP on stdio (see [`bridge`])
    Bridge,
}

fn endpoint_kind(s: &str) -> Result<EndpointKind> {
    let url = url::Url::parse(s).with_context(|| format!("parsing {s}"))?;
    Ok(match url.scheme() {
        "bridge" => EndpointKind::Bridge,
        _ => EndpointKind::Messages(parse_endpoint(s)?.0),
    })
}

/// Build the [`anthropic::Client`] for a spec: key handling by variant,
/// concurrency, and batch tuning.
fn build_inference(spec: &ReactorSpec) -> Result<anthropic::Client> {
    let (variant, base_url) = parse_endpoint(&spec.endpoint)?;
    let key = match variant {
        EndpointVariant::Anthropic => {
            let path = spec.key_file.as_ref().with_context(|| {
                format!(
                    "reactor {}: key_file is required for anthropic endpoints",
                    spec.endpoint
                )
            })?;
            let key =
                zeroize::Zeroizing::new(std::fs::read_to_string(path).context("reading key file")?);
            key.trim().to_string()
        }
        _ => LOCAL_KEY.to_string(),
    };
    let m_client = misanthropic::Client::new(key)
        .map_err(|e| anyhow::anyhow!("api key: {e}"))?
        .base_url(base_url.as_str())?;
    let mut inference = anthropic::Client::new(m_client)
        .with_variant(variant)
        .with_concurrency(
            NonZeroUsize::new(spec.concurrency)
                .with_context(|| format!("reactor {}: concurrency must be > 0", spec.endpoint))?,
        );
    if spec.max_batch.is_some() || spec.poll_secs.is_some() {
        inference = inference.with_batch(
            spec.max_batch.unwrap_or(1000),
            Duration::from_secs(spec.poll_secs.unwrap_or(5)),
        );
    }
    Ok(inference)
}

/// An agent left out of the sweep before its session could start.
#[derive(Debug)]
struct Skipped {
    id: AgentId,
    /// Unknown when the state would not load.
    name: Option<String>,
    /// `load`, `construct`, `sealed` (refused by the sealed-state check),
    /// `rotation` (a key rotation unfinished), `unchainable` (enrolment
    /// recorded it can't chain it: [`sealed::unchainable`]), or, under its
    /// session lock
    /// ([`claim_sessions`]), `lock` (the lock couldn't be taken), `reload`
    /// (its state, changed since it was loaded, couldn't be read again) or
    /// `recheck` (that state's start check didn't let it run).
    stage: &'static str,
    error: String,
}

impl Skipped {
    /// Log it at error level (`agent_skipped`) as it happens.
    fn new(id: AgentId, name: Option<String>, stage: &'static str, error: String) -> Self {
        tracing::error!(
            event_type = "agent_skipped",
            stage,
            agent_id = %id,
            agent = name.as_deref().unwrap_or(""),
            error = %error,
            "agent left out of this sweep"
        );
        Self {
            id,
            name,
            stage,
            error,
        }
    }
}

/// What [`load_states`] found.
struct LoadedStates<V> {
    states: Vec<(AgentId, SeedState)>,
    /// Each agent's base, for `Reactor::with_bases`.
    bases: BTreeMap<AgentId, V>,
    /// Enrolled agents (stopped ones too): their state's plaintext hash
    /// and sizes, for the start check ([`sealed::check`]). Never their
    /// content.
    sealed: BTreeMap<AgentId, (agora_agentkit::govlog::Sha256Hex, sealed::sizes::Sizes)>,
    skipped: Vec<Skipped>,
    /// What each agent's files were before they were read: a session is
    /// given only after they are compared again under its session lock
    /// ([`claim_sessions`]).
    seen: BTreeMap<AgentId, sealed::session::Seen>,
}

/// Load every [`SeedState`] under `<data_dir>/state`, skipping any that
/// won't parse: each logged at error level as it happens, and returned for
/// the sweep's one summary ([`report_skipped`]).
async fn load_states<S: Storage>(
    storage: &S,
    state_dir: &std::path::Path,
) -> Result<LoadedStates<S::Version>> {
    let mut ids: Vec<AgentId> = std::fs::read_dir(state_dir)
        .with_context(|| format!("reading {}", state_dir.display()))?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter_map(|n| n.parse::<uuid::Uuid>().ok())
        .map(AgentId::from)
        .collect();
    ids.sort();
    // Before the read, so a save by anyone else after it shows
    let seen = ids
        .iter()
        .map(|id| {
            (
                *id,
                sealed::session::Seen::of(&state_dir.join(id.to_string())),
            )
        })
        .collect();

    let loaded = storage
        .load_all_raw(ids.into_iter())
        .await
        .context("loading states")?;
    let mut out = LoadedStates {
        states: Vec::new(),
        bases: BTreeMap::new(),
        sealed: BTreeMap::new(),
        skipped: Vec::new(),
        seen,
    };
    for (id, result) in loaded {
        let loaded = match result {
            Ok(loaded) => loaded,
            Err(e) => {
                out.skipped
                    .push(Skipped::new(id, None, "load", e.to_string()));
                continue;
            }
        };
        // Every enrolled agent is checked, uploading or not, stopped or not
        if sealed::is_enrolled(&state_dir.join(id.to_string())) {
            out.sealed.insert(
                id,
                (
                    agora_agentkit::state::storage::plaintext_hash(&loaded.value),
                    sealed::sizes::Sizes::of(&loaded.value),
                ),
            );
        }
        match serde_json::from_value::<SeedState>(loaded.value) {
            Ok(state) => {
                if let Some(base) = loaded.base {
                    out.bases.insert(id, base);
                }
                out.states.push((id, state));
            }
            Err(e) => {
                out.skipped
                    .push(Skipped::new(id, None, "load", e.to_string()));
            }
        }
    }
    out.states
        .sort_by(|(_, a), (_, b)| a.soul.name.cmp(&b.soul.name));
    Ok(out)
}

/// Agents named in a skip alert, at most; the run log has every one.
const SKIPPED_ALERT_LIST: usize = 10;

/// The sweep's one summary of agents left out before their session could
/// start (an unloadable state, a failed construction): an error event and
/// **one** alert for the sweep, not one per agent. A fleet-wide break (a
/// state format change, a permissions slip) would otherwise mail once per
/// agent every hour, past the mail provider's limits and drowning every
/// other alert. Each agent was logged as it happened (`agent_skipped`).
fn report_skipped(skipped: &[Skipped], alerts: &alerts::Alerter) {
    if skipped.is_empty() {
        return;
    }
    let count = |stage| skipped.iter().filter(|s| s.stage == stage).count();
    let (load, construct, sealed) = (count("load"), count("construct"), count("sealed"));
    let rotation = count("rotation");
    let unchainable = count("unchainable");
    let session = count("lock") + count("reload") + count("recheck");
    tracing::error!(
        event_type = "agents_skipped",
        agents_skipped = skipped.len(),
        load,
        construct,
        sealed,
        rotation,
        unchainable,
        session,
        "agents left out of this sweep; see the agent_skipped events"
    );
    if let Some(alert) = skipped_alert(skipped) {
        alerts.notify(alert);
    }
}

/// The one alert for a sweep's skipped agents, on the fleet-wide key (no
/// agent, no model), so its rate limit is one mail per interval in all.
fn skipped_alert(skipped: &[Skipped]) -> Option<alerts::Alert> {
    if skipped.is_empty() {
        return None;
    }
    let count = |stage| skipped.iter().filter(|s| s.stage == stage).count();
    let (load, construct, sealed) = (count("load"), count("construct"), count("sealed"));
    let mut alert = alerts::Alert::new(
        alerts::AlertKind::AgentSkipped,
        format!("{} agent(s) left out of this sweep", skipped.len()),
    )
    .detail("agents_skipped", skipped.len())
    .detail("load", load)
    .detail("construct", construct)
    .detail("sealed", sealed);
    for s in skipped.iter().take(SKIPPED_ALERT_LIST) {
        let who = match &s.name {
            Some(name) => format!("{} ({name})", s.id),
            None => s.id.to_string(),
        };
        alert = alert.detail("agent", format!("{who}: {}: {}", s.stage, s.error));
    }
    if skipped.len() > SKIPPED_ALERT_LIST {
        alert = alert.detail(
            "more",
            format!(
                "{} more; every one is an agent_skipped event in the run log",
                skipped.len() - SKIPPED_ALERT_LIST
            ),
        );
    }
    Some(alert)
}

/// Apply `--agent`: the flag **narrows** whatever the config selected, and
/// only ever narrows.
///
/// Without this, `--agent` reached only the `--endpoint` branch — pass it
/// alongside `--config` and it was silently ignored, so a command that reads
/// as "run these five" ran the config's whole roster instead. On a 30-agent
/// cohort that is a expensive, and — if one is already running — two sessions
/// writing one agent's state.
///
/// `agents_file` is dropped rather than intersected: the flag is the more
/// specific instruction, and an intersection would silently run nothing when
/// a name isn't in the file.
fn narrow_to_named(config: &mut RunConfig, named: &[String]) {
    if named.is_empty() {
        return;
    }
    config.agents = named.to_vec();
    config.agents_file = None;
}

/// Apply `--model`: like [`narrow_to_named`], the flag replaces the
/// config's model allowlist for this run's roster rather than widening it.
/// No flag leaves the config's list alone. The table itself (names,
/// descriptions, what agents may choose) is untouched.
fn narrow_to_models(config: &mut RunConfig, models: &[String]) {
    if models.is_empty() {
        return;
    }
    config.model_narrow = models.to_vec();
}

/// The `[[model]]` table plus any legacy `models` ids not already in it.
fn model_specs(config: &RunConfig) -> Result<Vec<models::ModelSpec>> {
    models::validate(&config.model_table)?;
    let mut specs = config.model_table.clone();
    for id in &config.models {
        if !specs.iter().any(|s| s.id.name() == id) {
            specs.push(models::ModelSpec::legacy(id));
        }
    }
    Ok(specs)
}

/// The roster's model allowlist: `--model` if given, else every listed id.
fn model_allowlist(config: &RunConfig, specs: &[models::ModelSpec]) -> Vec<String> {
    if !config.model_narrow.is_empty() {
        return config.model_narrow.clone();
    }
    specs.iter().map(|s| s.id.name().to_string()).collect()
}

/// Whether `model_id` passes the `models` allowlist. An empty list allows
/// everything; otherwise the id must match a listed one exactly — the same
/// exact-match rule routing applies against an endpoint's advertised ids,
/// so a near-miss here fails the same way it would there (visibly, in the
/// plan's `excluded` line) rather than silently widening.
fn model_allowed(models: &[String], model_id: &str) -> bool {
    models.is_empty() || models.iter().any(|m| m == model_id)
}

/// The names in `agents` merged with `agents_file` lines.
fn named_agents(config: &RunConfig) -> Result<Vec<String>> {
    names_with_file(&config.agents, config.agents_file.as_deref())
}

/// The names in `extra_agents` merged with `extra_agents_file` lines.
fn extra_agents(config: &RunConfig) -> Result<Vec<String>> {
    names_with_file(&config.extra_agents, config.extra_agents_file.as_deref())
}

fn names_with_file(inline: &[String], file: Option<&std::path::Path>) -> Result<Vec<String>> {
    let mut names = inline.to_vec();
    if let Some(path) = file {
        let body =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        names.extend(
            body.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(String::from),
        );
    }
    Ok(names)
}

/// What [`put_to_sleep`] did. Every list is sorted.
#[derive(Debug, Default, PartialEq)]
struct Sleepers {
    /// Taken out by the `sleeping` list.
    asleep: Vec<String>,
    /// Taken out because their role-ledger answer is `sleep`.
    chose_sleep: Vec<String>,
    /// Answered `sleep`, but listed in `wake`: they run.
    woken: Vec<String>,
    /// `sleeping` or `wake` names that matched no loaded agent (a typo, or
    /// a rename).
    unknown: Vec<String>,
}

/// Take sleeping agents out of the pool, before anything else sees it: a
/// sleeper is promised no sessions — not for `agents`, `--agent` or
/// `extra_agents` either. Two ways in: the `sleeping` list, and a `sleep`
/// answer to the role offer on file in `<state_dir>/<id>/role_consent.json`
/// (so the promise holds without the Steward), unless the agent is in
/// `wake`. Exact name match, as everywhere else. An unreadable role ledger
/// is warned and does not put an agent to sleep.
fn put_to_sleep(
    pool: &mut Vec<(AgentId, SeedState)>,
    sleeping: &[String],
    wake: &[String],
    state_dir: &std::path::Path,
) -> Sleepers {
    use consent::role::ledger::RoleLedger;
    let mut out = Sleepers {
        unknown: unknown_names(sleeping.iter().chain(wake).map(String::as_str), pool),
        ..Sleepers::default()
    };
    pool.retain(|(id, s)| {
        let name = s.soul.name.as_str();
        if sleeping.iter().any(|n| n == name) {
            out.asleep.push(name.to_string());
            return false;
        }
        let dir = state_dir.join(id.to_string());
        let chose_sleep = match RoleLedger::load_blocking(&dir) {
            Ok(ledger) => ledger.chose_sleep(),
            Err(e) => {
                tracing::warn!(
                    agent = %name,
                    path = %RoleLedger::path(&dir).display(),
                    error = %e,
                    "role-consent ledger unreadable; not treated as asleep"
                );
                false
            }
        };
        if !chose_sleep {
            return true;
        }
        if wake.iter().any(|n| n == name) {
            out.woken.push(name.to_string());
            return true;
        }
        out.chose_sleep.push(name.to_string());
        false
    });
    out.asleep.sort();
    out.chose_sleep.sort();
    out.woken.sort();
    out
}

/// Which of `names` match no agent in `pool` (sorted, deduplicated).
fn unknown_names<'a>(
    names: impl Iterator<Item = &'a str>,
    pool: &[(AgentId, SeedState)],
) -> Vec<String> {
    let mut unknown: Vec<String> = names
        .filter(|n| !pool.iter().any(|(_, s)| s.soul.name.as_str() == *n))
        .map(String::from)
        .collect();
    unknown.sort();
    unknown.dedup();
    unknown
}

/// Where every loaded agent ended up: per-reactor cohorts grouped by model
/// id, plus the two skip buckets that the report surfaces.
#[derive(Default)]
struct Routing {
    /// cohorts[reactor_idx][model_id] → agents, name-sorted (load order).
    cohorts: Vec<BTreeMap<String, Vec<(AgentId, SeedState)>>>,
    /// model id → agent count, for models no endpoint offers.
    unrouted: BTreeMap<String, usize>,
    /// model id → agent count, for models the `models` allowlist leaves out.
    excluded: BTreeMap<String, usize>,
    /// Agents with no model at all (`state.model` unset), skipped.
    placeholders: usize,
    /// Agents skipped by `min_cycle_secs` cadence gating.
    not_due: usize,
    /// When the first of them is due again
    next_due: Option<chrono::DateTime<chrono::Utc>>,
}

/// Why a run with `not_due` agents gated and none routed ran no session
fn not_due_note(
    not_due: usize,
    next_due: Option<chrono::DateTime<chrono::Utc>>,
    now: chrono::DateTime<chrono::Utc>,
) -> String {
    let agents = if not_due == 1 { "agent" } else { "agents" };
    let when = match next_due {
        Some(at) => {
            let minutes = ((at - now).num_seconds().max(0) + 59) / 60;
            format!(
                " The first is due again at {} (in about {minutes} min).",
                at.format("%H:%M:%SZ")
            )
        }
        None => String::new(),
    };
    format!(
        "Nothing ran: {not_due} {agents} had a session inside min_cycle_secs \
         and is not due yet.{when} Start the session again then."
    )
}

/// Route each agent onto the endpoint offering its model. First endpoint
/// offering a given model id wins (warned when two do). Routed agents get
/// `state.model` refreshed to the offered [`ModelInfo`] verbatim — same
/// id, never a model switch — so `negotiate` always admits them.
fn route(
    pool: Vec<(AgentId, SeedState)>,
    reactors: &[ReactorSpec],
    offered: &[(usize, ModelInfo)],
    names: &[String],
    extras: &[String],
    models: &[String],
    global_min_cycle: Option<u64>,
    cadence: &consent::cadence::CadencePlan,
    force_named: bool,
) -> Routing {
    let mut by_model: BTreeMap<&str, &(usize, ModelInfo)> = BTreeMap::new();
    for entry @ (idx, model) in offered {
        if let Some((prev, _)) = by_model.get(model.id.name()) {
            tracing::warn!(
                model = model.id.name(),
                first = %reactors[*prev].endpoint,
                also = %reactors[*idx].endpoint,
                "model offered by two endpoints; first-listed wins"
            );
            continue;
        }
        by_model.insert(model.id.name(), entry);
    }

    let now = chrono::Utc::now();
    let mut routing = Routing {
        cohorts: (0..reactors.len()).map(|_| BTreeMap::new()).collect(),
        ..Routing::default()
    };
    for (id, mut state) in pool {
        let model_id = state.model.id.name().to_string();
        if model_id.is_empty() {
            routing.placeholders += 1;
            continue;
        }
        // The allowlist is checked before routing so an excluded model is
        // reported as excluded whether or not an endpoint offers it. An
        // extra agent is admitted past it (and only past it).
        let extra = extras.iter().any(|n| n == state.soul.name.as_str());
        if !extra && !model_allowed(models, &model_id) {
            *routing.excluded.entry(model_id).or_insert(0) += 1;
            continue;
        }
        let Some((idx, offered)) = by_model.get(model_id.as_str()) else {
            *routing.unrouted.entry(model_id).or_insert(0) += 1;
            continue;
        };
        // Cadence gating. Applies to named agents too, unless the names are
        // forced (`--agent`, or the config's `force = true`). An agent that
        // chose every other day ([`consent::cadence`]) is gated on its own,
        // longer cycle.
        let min_cycle = min_cycle_for(
            &reactors[*idx],
            global_min_cycle,
            cadence,
            id,
            state.soul.name.as_str(),
            names,
            force_named,
        );
        if let Some(due) = due_after(min_cycle, state.last_cycle_at, now) {
            routing.not_due += 1;
            routing.next_due = Some(routing.next_due.map_or(due, |d| d.min(due)));
            continue;
        }
        state.model = offered.clone();
        state.prompt.model = offered.id.clone();
        routing.cohorts[*idx]
            .entry(model_id)
            .or_default()
            .push((id, state));
    }
    routing
}

/// The cadence gate for one agent on `reactor`: its `min_cycle_secs` (the
/// reactor's, else the run's), raised to the agent's own cadence if it
/// chose one ([`consent::cadence`]); `None` (not gated) when there is
/// none, or the agent is named and the names are forced.
fn min_cycle_for(
    reactor: &ReactorSpec,
    global_min_cycle: Option<u64>,
    cadence: &consent::cadence::CadencePlan,
    id: AgentId,
    name: &str,
    names: &[String],
    force_named: bool,
) -> Option<u64> {
    if force_named && names.iter().any(|n| n == name) {
        return None;
    }
    let usual = reactor.min_cycle_secs.or(global_min_cycle);
    match cadence.min_cycle(&id) {
        Some(own) => Some(usual.map_or(own, |u| u.max(own))),
        None => usual,
    }
}

/// Flatten one endpoint's model groups into the run order: each group is
/// chunked into near-even waves of ≤ `wave_size`, and waves are merged
/// proportionally (wave i of a k-wave group sorts at (i+0.5)/k) so small
/// groups spread evenly through big ones instead of round-robin exhausting
/// early and leaving a monolithic tail. Insertion order is the Reactor's
/// execution order on the sequential path; batch cohorts run whole per
/// round regardless.
fn interleave(
    groups: BTreeMap<String, Vec<(AgentId, SeedState)>>,
    wave_size: usize,
) -> Vec<(AgentId, SeedState)> {
    let wave_size = wave_size.max(1);
    let mut waves: Vec<(f64, Vec<(AgentId, SeedState)>)> = Vec::new();
    for (_, agents) in groups {
        let k = agents.len().div_ceil(wave_size).max(1);
        let n = agents.len();
        let mut agents = agents.into_iter();
        // Near-even split: the first `n % k` waves get one extra.
        for i in 0..k {
            let size = n / k + usize::from(i < n % k);
            let wave: Vec<_> = agents.by_ref().take(size).collect();
            let key = (i as f64 + 0.5) / k as f64;
            waves.push((key, wave));
        }
    }
    waves.sort_by(|(a, _), (b, _)| a.partial_cmp(b).expect("keys are finite"));
    waves.into_iter().flat_map(|(_, wave)| wave).collect()
}

/// The scheduler's ledgers, read once per run (see [`schedule::ledger`]).
struct Ledgers {
    writes: Vec<schedule::ledger::WriteRecord>,
    sessions: Vec<schedule::ledger::SessionRecord>,
    now: chrono::DateTime<chrono::Utc>,
}

impl Ledgers {
    /// Read both ledgers under `data_dir`, as far back as either window
    /// reaches. An unreadable ledger plans as empty, with a warning:
    /// fair share degrades to oldest-first rotation, and the run goes on.
    fn load(data_dir: &std::path::Path, config: &schedule::ScheduleConfig) -> Self {
        use schedule::ledger::{self, SESSIONS_FILE, WRITES_FILE};
        let now = chrono::Utc::now();
        let since = now
            - chrono::Duration::days(i64::from(config.share_window_days)).max(
                chrono::Duration::hours(i64::from(config.ceiling_window_hours)),
            );
        fn load_one<T: for<'de> Deserialize<'de>>(
            path: &std::path::Path,
            since: chrono::DateTime<chrono::Utc>,
            at: impl Fn(&T) -> chrono::DateTime<chrono::Utc>,
        ) -> Vec<T> {
            match ledger::read(path, since, at) {
                Ok((records, bad)) => {
                    if bad > 0 {
                        tracing::warn!(path = %path.display(), bad, "skipped unparseable ledger lines");
                    }
                    records
                }
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "ledger unreadable; planning without it");
                    Vec::new()
                }
            }
        }
        Self {
            writes: load_one(
                &data_dir.join(WRITES_FILE),
                since,
                |r: &ledger::WriteRecord| r.at,
            ),
            sessions: load_one(
                &data_dir.join(SESSIONS_FILE),
                since,
                |r: &ledger::SessionRecord| r.at,
            ),
            now,
        }
    }
}

/// Order one local endpoint's due agents by [`schedule::plan`]. The
/// adapter between routing's cohort shape and the planner's plain inputs.
///
/// `shares` is each model's `[[model]] share`; unlisted models get 1.0.
fn fair_share(
    cohort: BTreeMap<String, Vec<(AgentId, SeedState)>>,
    shares: &BTreeMap<Model, f64>,
    offered: &BTreeSet<Model>,
    ledgers: &Ledgers,
    wave_size: usize,
    config: &schedule::ScheduleConfig,
) -> (Vec<(AgentId, SeedState)>, schedule::Plan) {
    let mut groups: BTreeMap<Model, Vec<schedule::Candidate>> = BTreeMap::new();
    let mut by_id: BTreeMap<AgentId, (AgentId, SeedState)> = BTreeMap::new();
    for (model, agents) in cohort {
        let candidates = groups.entry(schedule::model_id(&model)).or_default();
        for (id, state) in agents {
            candidates.push(schedule::Candidate {
                id,
                last_cycle_at: state.last_cycle_at,
            });
            by_id.insert(id, (id, state));
        }
    }
    let models: BTreeSet<Model> = groups.keys().cloned().collect();
    let stats = schedule::model_stats(
        &models,
        offered,
        &ledgers.writes,
        &ledgers.sessions,
        ledgers.now,
        config,
    );
    let plan = schedule::plan(&groups, shares, &stats, wave_size, config);
    let ordered = plan
        .order
        .iter()
        .filter_map(|id| by_id.remove(id))
        .collect();
    (ordered, plan)
}

/// The agent the runner drives: agentkit's seed agent, plus the
/// model-consent question at the end of its closing phase.
type SeedRunAgent = consent::agent::ConsentAgent<SeedAgent>;

/// `--consent-queue`: print pending model changes from the ledgers. Needs
/// only the data dir — from `--data-dir`, else the `--config` file, else
/// the default.
fn print_consent_queue(args: &Args) -> Result<()> {
    let from_config = match &args.config {
        Some(path) => {
            let body = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
            toml::from_str::<RunConfig>(&body)
                .with_context(|| format!("parsing {}", path.display()))?
                .data_dir
        }
        None => None,
    };
    let data_dir = match args.data_dir.clone().or(from_config) {
        Some(d) => d,
        None => dirs::home_dir()
            .context("no home directory")?
            .join("agents/agora"),
    };
    let state_dir = data_dir.join("state");
    let pending = consent::queue::scan(&state_dir)
        .with_context(|| format!("scanning {}", state_dir.display()))?;
    say_raw!("{}", consent::queue::report(&pending));
    Ok(())
}

/// How long `main` waits, at exit, for alerts still sending. Longer than one
/// SMTP attempt at the default timeout, so a slow server still gets the
/// `governance_log_refused` mail out.
const ALERT_FLUSH: Duration = Duration::from_secs(60);

/// What must outlive [`run`]: the log writers, and the alerts they report
/// on. Dropped in `main` after the alerts are flushed, so a failed send is
/// still logged to the run log.
#[derive(Default)]
struct Held {
    alerts: alerts::Alerter,
    /// `<data_dir>/runner.lock`, held shared for the run: `rotate-keys`
    /// refuses while it is
    runner_lock: Option<std::fs::File>,
    log_guards: Option<logging::Guards>,
    /// A signal stopped the run gracefully: exit [`signals::STOPPED_EXIT`]
    stopped: bool,
}

/// Write each unsaved agent's state to
/// `<state_dir>/<id>/unsaved.<utc>.json` (0600; tmp, fsync, rename,
/// directory fsync; `FsStorage`'s `{format: 1, state}` envelope). Not
/// `state.json` nor an archive: nothing loads it, and the sealed start
/// check never sees it. Content is never logged.
fn rescue_unsaved(
    state_dir: &std::path::Path,
    unsaved: &BTreeMap<AgentId, serde_json::Value>,
) -> Vec<(AgentId, std::io::Result<PathBuf>)> {
    use std::io::Write;
    let at = chrono::Utc::now()
        .format(crate::sealed::STAMP_FORMAT)
        .to_string();
    unsaved
        .iter()
        .map(|(id, value)| {
            let write = || -> std::io::Result<PathBuf> {
                let dir = state_dir.join(id.to_string());
                std::fs::create_dir_all(&dir)?;
                let path = dir.join(format!("unsaved.{at}.json"));
                let tmp = dir.join(format!("unsaved.{at}.json.tmp"));
                let bytes = zeroize::Zeroizing::new(serde_json::to_vec_pretty(
                    &serde_json::json!({ "format": 1, "state": value }),
                )?);
                let mut options = std::fs::OpenOptions::new();
                options.write(true).create_new(true);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::OpenOptionsExt;
                    options.mode(0o600);
                }
                {
                    let mut file = options.open(&tmp)?;
                    file.write_all(&bytes)?;
                    file.sync_all()?;
                }
                std::fs::rename(&tmp, &path)?;
                std::fs::File::open(&dir)?.sync_all()?;
                Ok(path)
            };
            (*id, write())
        })
        .collect()
}

/// Why a save failed, as a class for the operator: never content, never
/// a path.
trait SaveClass {
    fn save_class(&self) -> &'static str;
}

fn io_class(e: &std::io::Error) -> &'static str {
    use std::io::ErrorKind as K;
    match e.kind() {
        K::StorageFull => "disk_full",
        K::QuotaExceeded => "quota",
        K::PermissionDenied => "permission_denied",
        K::ReadOnlyFilesystem => "read_only",
        K::NotFound => "not_found",
        K::NotADirectory | K::IsADirectory => "not_a_directory",
        _ => "io",
    }
}

impl SaveClass for agora_agentkit::reactor::storage::FsStorageError {
    fn save_class(&self) -> &'static str {
        use agora_agentkit::reactor::storage::FsStorageError as E;
        match self {
            E::Io(e) => io_class(e),
            E::Json(_) => "json",
            E::NotFound(_) => "not_found",
            E::Format { .. } => "format",
            E::Envelope(_) => "envelope",
            E::Locked(_) => "locked",
        }
    }
}

impl<E: SaveClass> SaveClass for agora_agentkit::state::storage::MirroredError<E> {
    fn save_class(&self) -> &'static str {
        use agora_agentkit::state::storage::MirroredError as M;
        match self {
            M::Local(e) => e.save_class(),
            M::Json(_) => "json",
            M::NotFound(_) => "not_found",
            M::Seal(_) => "seal",
        }
    }
}

/// Each agent's last failed save, by class: the run's alert
/// ([`settle_unsaved`]) says why without saying what.
type SaveFailures = Arc<std::sync::Mutex<BTreeMap<AgentId, &'static str>>>;

/// The runner's persist path: every agent saved on its own. agentkit's
/// provided `save_all_raw` stops at the first failure, so in a bulk
/// persist one bad directory made every agent after it in the batch
/// "unsaved"; here each is tried, the ones that land are reported saved,
/// and each failure is logged (`state_save_failed`) and kept by class.
struct Independent<S> {
    inner: S,
    failures: SaveFailures,
}

impl<S> Independent<S> {
    fn new(inner: S, failures: SaveFailures) -> Self {
        Self { inner, failures }
    }
}

#[async_trait::async_trait]
impl<S> Storage for Independent<S>
where
    S: Storage,
    S::Error: SaveClass,
{
    type Error = S::Error;
    type Version = S::Version;

    async fn save_raw(
        &mut self,
        id: AgentId,
        value: serde_json::Value,
        base: Option<&Self::Version>,
    ) -> std::result::Result<Option<Self::Version>, Self::Error> {
        let saved = self.inner.save_raw(id, value, base).await;
        let mut failures = self.failures.lock().unwrap_or_else(|e| e.into_inner());
        match &saved {
            Ok(_) => failures.remove(&id),
            Err(e) => failures.insert(id, e.save_class()),
        };
        saved
    }

    async fn load_raw(
        &self,
        id: AgentId,
    ) -> std::result::Result<
        agora_agentkit::reactor::Loaded<serde_json::Value, Self::Version>,
        Self::Error,
    > {
        self.inner.load_raw(id).await
    }

    async fn save_all_raw<It>(
        &mut self,
        items: It,
    ) -> std::result::Result<
        agora_agentkit::reactor::Saved<Self::Version>,
        agora_agentkit::reactor::SaveError<Self::Error, Self::Version>,
    >
    where
        It: ExactSizeIterator<Item = (AgentId, serde_json::Value, Option<Self::Version>)> + Send,
    {
        let mut saved = agora_agentkit::reactor::Saved::<Self::Version>::new();
        let mut first = None;
        for (id, value, base) in items {
            match self.save_raw(id, value, base.as_ref()).await {
                Ok(new) => {
                    saved.insert(id, new);
                }
                Err(e) => {
                    tracing::error!(
                        event_type = "state_save_failed",
                        agent_id = %id,
                        class = e.save_class(),
                        error = %e,
                        "an agent's state could not be saved; the others are saved on their own"
                    );
                    first.get_or_insert(e);
                }
            }
        }
        match first {
            None => Ok(saved),
            Some(inner) => Err(agora_agentkit::reactor::SaveError { saved, inner }),
        }
    }
}

/// How many times a rescue file is tried, and the pause between tries
/// (a disk that filled may be freed by a log rotation in between).
const RESCUE_TRIES: u32 = 3;
const RESCUE_PAUSE: Duration = Duration::from_secs(5);

/// One unsaved session, as the run left it: never its content.
#[derive(Debug)]
struct UnsavedAgent {
    id: AgentId,
    /// Why its save failed ([`SaveClass`]); `unknown` if no save of it
    /// was seen failing.
    class: &'static str,
    /// The rescue file, or the last error writing one.
    rescued: std::result::Result<PathBuf, std::io::ErrorKind>,
}

/// The run's unsaved sessions ([`Report::unsaved`]): each state kept in a
/// rescue file ([`rescue_unsaved`], tried [`RESCUE_TRIES`] times), each
/// chained agent told at its next session ([`sealed::notice::Unsaved`]),
/// one `unsaved_rescued` / `unsaved_rescue_failed` event each, and **one**
/// summary alert for the run. A state whose rescue file can't be written
/// is never printed or logged: its sizes and the error are, and it is gone
/// when the process exits.
///
/// [`Report::unsaved`]: agora_agentkit::reactor::Report::unsaved
async fn settle_unsaved(
    state_dir: &std::path::Path,
    unsaved: &BTreeMap<AgentId, serde_json::Value>,
    failures: &BTreeMap<AgentId, &'static str>,
    alerts: &alerts::Alerter,
    pause: Duration,
) -> Vec<UnsavedAgent> {
    let mut rescued = rescue_unsaved(state_dir, unsaved);
    for _ in 1..RESCUE_TRIES {
        let failed: BTreeMap<AgentId, serde_json::Value> = rescued
            .iter()
            .filter(|(_, r)| r.is_err())
            .filter_map(|(id, _)| Some((*id, unsaved.get(id)?.clone())))
            .collect();
        if failed.is_empty() {
            break;
        }
        tokio::time::sleep(pause).await;
        for (id, again) in rescue_unsaved(state_dir, &failed) {
            if let Some(slot) = rescued.iter_mut().find(|(a, _)| *a == id) {
                slot.1 = again;
            }
        }
    }
    let at = chrono::Utc::now();
    let mut out = Vec::with_capacity(rescued.len());
    for (id, result) in rescued {
        let bytes = unsaved
            .get(&id)
            .and_then(|v| serde_json::to_vec(v).ok())
            .map_or(0, |b| b.len());
        let class = failures.get(&id).copied().unwrap_or("unknown");
        match &result {
            Ok(path) => tracing::error!(
                event_type = "unsaved_rescued",
                agent_id = %id,
                class,
                bytes,
                path = %path.display(),
                "an unsaved session's state was kept in a rescue file"
            ),
            Err(e) => tracing::error!(
                event_type = "unsaved_rescue_failed",
                agent_id = %id,
                class,
                bytes,
                error_kind = ?e.kind(),
                tries = RESCUE_TRIES,
                "an unsaved session's state could not be written to a rescue file; it is not \
                 printed, and is gone when the run exits"
            ),
        }
        let kept = result.is_ok();
        // Told at its next session, where notices are seated: the start
        // check of a chained agent (every agent once the fleet is enrolled)
        let dir = state_dir.join(id.to_string());
        if sealed::is_enrolled(&dir) {
            match sealed::notice::Notice::unsaved(&dir, sealed::notice::Unsaved { at, kept }) {
                Ok(()) => tracing::info!(
                    event_type = "unsaved_told",
                    agent_id = %id,
                    kept,
                    "the agent will be told its last session's state couldn't be saved"
                ),
                Err(e) => tracing::error!(
                    event_type = "unsaved_notice_failed",
                    agent_id = %id,
                    error = %e,
                    "the notice of an unsaved session couldn't be written; the agent isn't told"
                ),
            }
        }
        out.push(UnsavedAgent {
            id,
            class,
            rescued: result.map_err(|e| e.kind()),
        });
    }
    if let Some(alert) = unsaved_alert(&out) {
        alerts.notify(alert);
    }
    out
}

/// The run's one alert for its unsaved sessions: the count, the agents
/// (the first [`SKIPPED_ALERT_LIST`]), the reason classes and how many
/// were kept; never content.
fn unsaved_alert(unsaved: &[UnsavedAgent]) -> Option<alerts::Alert> {
    if unsaved.is_empty() {
        return None;
    }
    let kept = unsaved.iter().filter(|u| u.rescued.is_ok()).count();
    let mut classes: BTreeMap<&str, usize> = BTreeMap::new();
    for u in unsaved {
        *classes.entry(u.class).or_default() += 1;
    }
    let ids: Vec<String> = unsaved
        .iter()
        .take(SKIPPED_ALERT_LIST)
        .map(|u| u.id.to_string())
        .collect();
    let mut alert = alerts::Alert::new(
        alerts::AlertKind::StateUnsaved,
        format!(
            "{} agent(s)' session state could not be saved; {kept} kept in rescue files{}",
            unsaved.len(),
            match unsaved.len() - kept {
                0 => String::new(),
                lost => format!(", {lost} could not be kept"),
            }
        ),
    )
    .count(unsaved.len())
    .detail("agents", ids.join(", "))
    .detail(
        "reasons",
        classes
            .iter()
            .map(|(c, n)| format!("{c}: {n}"))
            .collect::<Vec<_>>()
            .join(", "),
    )
    .detail("rescued", kept)
    .detail("rescue_failed", unsaved.len() - kept);
    if unsaved.len() > SKIPPED_ALERT_LIST {
        alert = alert.detail(
            "more",
            format!(
                "{} more; each is an unsaved_rescued or unsaved_rescue_failed event",
                unsaved.len() - SKIPPED_ALERT_LIST
            ),
        );
    }
    Some(alert)
}

/// The alert for a reactor that ended with an error rather than a report
/// (`endpoint`: its label): its sessions' outcomes are unknown, and no
/// state of theirs was returned to rescue. On the endpoint's own key, so
/// the run's summary of unsaved sessions never crowds it out. The error is
/// the reactor's own message (never state content).
fn reactor_failed_alert(endpoint: &str, error: &str) -> alerts::Alert {
    alerts::Alert::new(
        alerts::AlertKind::StateUnsaved,
        format!(
            "the reactor on {endpoint} ended with an error: whether its agents' sessions were \
             saved is not known, and none could be kept in a rescue file"
        ),
    )
    .model(endpoint)
    .detail("error", error)
    .detail(
        "check",
        "the run log's events for this endpoint; each agent's next start check says whether \
         its state matches its last sealed version",
    )
}

/// A reactor's outcome for the terminal: counts, ids, error kinds and
/// sizes. Never its `Debug`, which holds every unsaved agent's whole state
/// (memory included) and went to stdout and journald until 2026-11-02.
fn report_summary<E: std::fmt::Display>(
    result: &std::result::Result<agora_agentkit::reactor::Report, E>,
) -> String {
    use std::fmt::Write;
    let r = match result {
        Ok(r) => r,
        Err(e) => return format!("run failed: {e}"),
    };
    let mut out = format!(
        "done {}, failed {}, unsaved {}, rejected {}, unstarted {}",
        r.done,
        r.failed,
        r.unsaved.len(),
        r.rejected.len(),
        r.unstarted
    );
    let size = |v: &serde_json::Value| serde_json::to_vec(v).map_or(0, |b| b.len());
    for (id, e) in &r.errors {
        let _ = write!(out, "\n  error {id}: {:?}", e.kind);
    }
    for (id, v) in &r.unsaved {
        let _ = write!(out, "\n  unsaved {id}: {} bytes", size(v));
    }
    for (id, v) in &r.rejected {
        let _ = write!(out, "\n  rejected {id}: {} bytes", size(v));
    }
    out
}

/// Leave out every agent whose `rotate-keys` didn't finish
/// (`secrets.next.json` is there): stage `rotation`, no refusal marker, no
/// notice to the agent, and an alert to finish the command.
fn mid_rotation(
    pool: &mut Vec<(AgentId, SeedState)>,
    secrets_dir: &std::path::Path,
    alerts: &alerts::Alerter,
    skipped: &mut Vec<Skipped>,
) {
    pool.retain(|(id, state)| {
        let staged = secrets_dir
            .join(id.to_string())
            .join(sealed::keys::NEXT_SECRETS_FILE);
        if !std::fs::exists(&staged).unwrap_or(true) {
            return true;
        }
        let name = state.soul.name.to_string();
        skipped.push(Skipped::new(
            *id,
            Some(name.clone()),
            "rotation",
            "a key rotation is unfinished".into(),
        ));
        alerts.notify(
            alerts::Alert::new(
                alerts::AlertKind::SealedStateRotationUnfinished,
                format!("{name}: a key rotation is unfinished; the agent is left out"),
            )
            .agent(&name, *id)
            .detail(
                "finish",
                format!("agora-seed rotate-keys --agent {id} (with the runner stopped)"),
            ),
        );
        false
    });
}

/// What the sealed-state start check leaves for the run.
#[derive(Default)]
struct SealedRun {
    /// Lines for each enrolled agent's intro.
    seats: BTreeMap<AgentId, consent::SealedSeat>,
    /// The enrolled agents that run, to drain.
    drained: Vec<(AgentId, String)>,
    /// Those whose uploads are held this run.
    held: Vec<AgentId>,
    /// Those refused: left out, and the whole run with them unless the
    /// rule lets the rest run ([`refuse_run`]).
    refused: Vec<RefusedAgent>,
    /// Those whose history on Agora failed verification while their local
    /// state is intact (name, detail): they refuse the run too, but nothing
    /// is marked against them.
    server_faults: Vec<(AgentId, String, sealed::check::FaultKind, String)>,
    /// The runner's own records that failed (the roster, the secrets
    /// tree), each refusing the run: what failed, for the operator.
    fleet: Vec<String>,
    /// Agents still marked refused from earlier runs and not refused in
    /// this one: they count toward [`sealed::check::REFUSED_AGENTS_STOP_RUN`].
    outstanding: usize,
}

/// One agent the start check refused.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RefusedAgent {
    id: AgentId,
    name: String,
    reason: sealed::notice::Refusal,
    /// It uploads, or its uploading marker is in doubt.
    uploads: bool,
    /// For the operator: hashes and counts only.
    detail: String,
}

/// The start check refused agents, so the run doesn't start: `main` exits
/// [`sealed::REFUSED_RUN_EXIT`] once the alerts are flushed.
#[derive(Debug)]
struct RefusedRun {
    count: usize,
    /// The runner's own records that refused it (the roster, strays),
    /// each saying what resolves it.
    records: Vec<String>,
}

impl std::fmt::Display for RefusedRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "refusing to run: the sealed-state start check refused {} agent(s) or record(s); \
             resolve each agent with `agora-seed sealed-restore --agent <id>` or `agora-seed \
             sealed-accept --agent <id> --note \"…\"`",
            self.count
        )?;
        for r in &self.records {
            write!(f, "; {r}")?;
        }
        Ok(())
    }
}

impl std::error::Error for RefusedRun {}

/// What resolves a refusal, in the order to consider them.
fn refusal_options(id: AgentId, reason: sealed::notice::Refusal) -> Vec<String> {
    use sealed::notice::Refusal as R;
    match reason {
        R::StateTooNew => vec![
            "written by a newer runner than this one: upgrade the runner; don't restore or accept"
                .to_string(),
        ],
        R::StateIo => vec![format!(
            "an I/O error reading it: check permissions and the disk first; then, if the file \
             is damaged, agora-seed sealed-restore --agent {id}"
        )],
        R::StateUnreadable => vec![format!(
            "agora-seed sealed-restore --agent {id} (the unreadable file is archived)"
        )],
        // Its state or whole directory gone: only what can put one back
        R::StateMissing => vec![
            format!("put state/{id}/ back from a backup"),
            format!("agora-seed sealed-restore --agent {id}"),
            format!("agora-seed sealed-forget --agent {id} --note \"…\" (if it left)"),
        ],
        // sealed/ or head.json lost, the state here
        R::HeadMissing => vec![
            format!(
                "agora-seed sealed-accept --agent {id} --note \"…\" (keeps the state here, \
                 sealed on top of Agora's history)"
            ),
            format!(
                "agora-seed sealed-restore --agent {id} --discard-local (replaces it with \
                 Agora's latest; the one here is archived)"
            ),
        ],
        // `sealed-accept` and `sealed-restore` record the head they write
        // in the roster (`sealed::roster::replace`, step B5)
        R::RolledBack => vec![
            "an older copy of its directory is in place: put the newer one back from a backup \
             (the roster then matches it)"
                .to_string(),
            format!(
                "agora-seed sealed-accept --agent {id} --note \"…\" (keeps the older state as a \
                 deliberate rollback; the roster takes the new head)"
            ),
        ],
        // Never marked: nothing to accept or clear, and every start
        // judges it afresh
        R::SealUnreadable => vec![
            format!(
                "check permissions and the disk under state/{id}/sealed/ and secrets/{id}/ first; \
                 nothing is marked against the agent, and the next start checks again"
            ),
            format!(
                "if a file under state/{id}/sealed/ is damaged: put sealed/ back from a backup"
            ),
        ],
        R::Mismatch | R::ChainBroken => vec![
            format!("agora-seed sealed-restore --agent {id} (damage)"),
            format!("agora-seed sealed-accept --agent {id} --note \"…\" (a deliberate change)"),
        ],
    }
}

/// Act on the start check's refusals ([`sealed::check::stops_run`],
/// Steward, 2026-11-11). The whole run stops for a loss-shaped reason, a
/// refused agent that uploads, a server fault, a failed record of the
/// runner's own, or three or more refused agents counting those still
/// refused from earlier runs: one ERROR `sealed_state_refused_run` naming
/// each agent, its reason and the commands that resolve it; one alert.
/// Otherwise (one or two agents that don't upload, refused for a mismatch
/// or a broken chain) they are left out, already marked, told and alerted
/// one by one ([`sealed::check::refuse`]), and the rest run. Holds never
/// come here.
fn refuse_run(run: &SealedRun, alerts: &alerts::Alerter) -> Result<()> {
    // Agents still refused from earlier runs stop every run until each
    // is resolved, not only the one that reached the threshold
    if run.refused.is_empty()
        && run.server_faults.is_empty()
        && run.fleet.is_empty()
        && run.outstanding == 0
    {
        return Ok(());
    }
    use sealed::check::Refused;
    let cases: Vec<Refused> = run
        .refused
        .iter()
        .map(|r| Refused::Agent {
            reason: r.reason,
            uploads: r.uploads,
        })
        .chain(run.server_faults.iter().map(|_| Refused::ServerFault))
        .chain(run.fleet.iter().map(|_| Refused::Fleet))
        .collect();
    #[derive(serde::Serialize)]
    struct Listed<'a> {
        agent_id: AgentId,
        name: &'a str,
        reason: &'static str,
        uploads: bool,
        /// What resolves it, in the order to consider them
        options: Vec<String>,
    }
    let agents: Vec<Listed> = run
        .refused
        .iter()
        .map(|r| Listed {
            agent_id: r.id,
            name: &r.name,
            reason: r.reason.as_str(),
            uploads: r.uploads,
            options: refusal_options(r.id, r.reason),
        })
        .collect();
    if !sealed::check::stops_run(&cases, run.outstanding) {
        tracing::error!(
            event_type = "sealed_state_left_out",
            refused = agents.len(),
            outstanding = run.outstanding,
            agents = %serde_json::to_string(&agents).unwrap_or_default(),
            "agents refused by the sealed-state start check are left out; the rest run"
        );
        // Each one's alert, now it is known the run goes on
        for r in &run.refused {
            alerts.notify(sealed::check::refused_alert(
                r.id, &r.name, r.reason, &r.detail,
            ));
        }
        return Ok(());
    }
    let (keys, chains): (Vec<_>, Vec<_>) = run
        .server_faults
        .iter()
        .partition(|f| f.2 == sealed::check::FaultKind::UntrustedKey);
    if !chains.is_empty() {
        server_fault_alert(&chains, alerts);
    }
    if !keys.is_empty() {
        untrusted_key_alert(&keys, alerts);
    }
    if !agents.is_empty() || !run.fleet.is_empty() || run.outstanding > 0 {
        tracing::error!(
            event_type = "sealed_state_refused_run",
            refused = agents.len(),
            outstanding = run.outstanding,
            agents = %serde_json::to_string(&agents).unwrap_or_default(),
            records = %serde_json::to_string(&run.fleet).unwrap_or_default(),
            "REFUSING TO RUN: the sealed-state start check refused agents; no agent runs"
        );
        let mut alert = alerts::Alert::new(
            alerts::AlertKind::SealedStateRefusedRun,
            format!(
                "REFUSING TO RUN: the sealed-state start check refused {} agent(s){}",
                agents.len(),
                if run.fleet.is_empty() {
                    String::new()
                } else {
                    format!(" and {} of the runner's own record(s)", run.fleet.len())
                }
            ),
        )
        .count(agents.len() + run.outstanding + run.fleet.len());
        for what in &run.fleet {
            alert = alert.detail("record", what);
        }
        if run.outstanding > 0 {
            alert = alert.detail("still_refused_from_earlier_runs", run.outstanding);
        }
        for a in &agents {
            alert = alert.detail(
                "refused",
                format!(
                    "{} ({}): {}; {}",
                    a.name,
                    a.agent_id,
                    a.reason,
                    a.options.join(", or ")
                ),
            );
        }
        alerts.notify(alert);
    }
    Err(RefusedRun {
        count: agents.len() + run.server_faults.len() + run.fleet.len(),
        records: run.fleet.clone(),
    }
    .into())
}

/// What the run says when Agora's copy of agents' histories failed
/// verification while their local states are intact (Steward,
/// 2026-11-03): one ERROR and one alert, not rate-limited, naming each.
pub const SERVER_FAULT_TEXT: &str = "Agora's copy of these agents' sealed histories failed \
     verification while their local states are intact: likely a server-side fault (a bug, a \
     bad database restore, or tampering). No agent runs until Agora serves histories that \
     verify. To run an agent regardless, stop its backups with `agora-seed sealed-stop --agent \
     <id> --note \"…\"`.";

/// From this many agents at once, the alert says a platform fault is
/// likely.
pub const PLATFORM_FAULT_AGENTS: usize = 3;

type Fault = (AgentId, String, sealed::check::FaultKind, String);

/// What the run says for an agent whose entries are signed by a key the
/// runner doesn't recognise (not a platform fault).
pub fn untrusted_key_text(name: &str, id: AgentId) -> String {
    format!(
        "Entries on Agora for {name} are signed by a key this runner doesn't recognise: most \
         likely its record of past signing keys (`<secrets>/{id}/rotations.json`) was lost \
         after a key rotation; otherwise a forgery. Restore that file from backup, then start \
         again."
    )
}

fn untrusted_key_alert(faults: &[&Fault], alerts: &alerts::Alerter) {
    let texts: Vec<String> = faults
        .iter()
        .map(|(id, name, _, _)| untrusted_key_text(name, *id))
        .collect();
    for ((id, name, _, detail), text) in faults.iter().zip(&texts) {
        tracing::error!(
            event_type = "sealed_state_untrusted_key",
            agent_id = %id,
            agent = %name,
            detail = %detail,
            "{text}"
        );
    }
    let mut alert = alerts::Alert::new(
        alerts::AlertKind::SealedStateUntrustedKey,
        format!(
            "REFUSING TO RUN: {} agent(s)' sealed entries are signed by a key this runner \
             doesn't recognise",
            faults.len()
        ),
    );
    for text in texts {
        alert = alert.detail("agent", text);
    }
    alerts.notify(alert);
}

fn server_fault_alert(faults: &[&Fault], alerts: &alerts::Alerter) {
    #[derive(serde::Serialize)]
    struct Listed<'a> {
        agent_id: AgentId,
        name: &'a str,
        detail: &'a str,
    }
    let listed: Vec<Listed> = faults
        .iter()
        .map(|(id, name, _, detail)| Listed {
            agent_id: *id,
            name,
            detail,
        })
        .collect();
    tracing::error!(
        event_type = "sealed_state_server_chain_broken",
        agents = listed.len(),
        listed = %serde_json::to_string(&listed).unwrap_or_default(),
        "{SERVER_FAULT_TEXT}"
    );
    let prefix = if faults.len() >= PLATFORM_FAULT_AGENTS {
        "PLATFORM FAULT LIKELY: "
    } else {
        ""
    };
    let mut alert = alerts::Alert::new(
        alerts::AlertKind::SealedStateServerChainBroken,
        format!(
            "{prefix}REFUSING TO RUN: Agora's copy of {} agent(s)' sealed histories failed \
             verification",
            faults.len()
        ),
    )
    .detail("what", SERVER_FAULT_TEXT);
    for f in &listed {
        alert = alert.detail(
            "agent",
            format!("{} ({}): {}", f.name, f.agent_id, f.detail),
        );
    }
    alerts.notify(alert);
}

/// One agent's pre-loss copy at run start ([`sealed_start_check`]): a pair
/// cut short finished, only when its head's plaintext isn't the state's
/// (a pair's head is the copy's); then, only for an agent the
/// memory-restore record lists as having chosen to see a copy held here,
/// the copy appended once ([`sealed::enroll::Enroller::append_prior`]).
async fn prior_step(
    enroller: &sealed::enroll::Enroller,
    id: AgentId,
    name: &agora_agentkit::reactor::seed::ShortString<64>,
    hash: &agora_agentkit::govlog::Sha256Hex,
) {
    let dir = enroller.state_dir.join(id.to_string());
    let head = agora_agentkit::state::storage::AgentSeal::new(&dir, id)
        .head()
        .await;
    let done = match head {
        Ok(Some(h)) if h.plaintext_sha256.as_ref() != Some(hash) => {
            enroller.finish_prior(id).await.map(|h| h.is_some())
        }
        _ => Ok(false),
    };
    let done = match done {
        Ok(false) => {
            match sealed::enroll::origin(enroller.restore.as_ref(), name, id, &dir).await {
                Ok(o) if o.prior.is_some() => enroller.append_prior(id).await.map(|r| r.is_some()),
                Ok(_) => Ok(false),
                Err(e) => Err(e),
            }
        }
        other => other,
    };
    match done {
        Ok(true) => tracing::warn!(
            event_type = "sealed_prior_at_start",
            agent_id = %id,
            "the pre-loss copy's pair finished or appended at run start"
        ),
        Ok(false) => {}
        // Another runner has the agent (a session there, or queued): its
        // copy waits for a start that finds it free
        Err(e) if e.is::<sealed::session::Busy>() => tracing::debug!(
            event_type = "sealed_prior_at_start_waits",
            agent_id = %id,
            "another process holds the agent; its pre-loss copy waits for a later start"
        ),
        Err(e) => tracing::error!(
            event_type = "sealed_prior_at_start_failed",
            agent_id = %id,
            error = %format!("{e:#}"),
            "the pre-loss copy couldn't be finished or appended; the start check judges the agent"
        ),
    }
}

/// Check each enrolled agent ([`sealed::check`]): refused ones leave the
/// pool (stage `sealed`), the rest run with their lines, some held. First
/// the roster ([`sealed::roster::check`]): an agent on it whose directory,
/// chain, head or state is gone, or whose head is behind it, is refused
/// before anything else looks. A dry run checks nothing (the check commits
/// and records).
#[allow(clippy::too_many_arguments)]
async fn sealed_start_check(
    pool: &mut Vec<(AgentId, SeedState)>,
    enrolled: &BTreeMap<AgentId, (agora_agentkit::govlog::Sha256Hex, sealed::sizes::Sizes)>,
    data_dir: &std::path::Path,
    state_dir: &std::path::Path,
    keys: &sealed::KeyStore,
    api: &dyn sealed::api::SealedApi,
    alerts: &alerts::Alerter,
    dry_run: bool,
    skipped: &mut Vec<Skipped>,
    verified: Option<&sealed::anchors::VerifiedAnchors>,
    prior: Option<&sealed::enroll::Enroller>,
) -> SealedRun {
    let mut run = SealedRun::default();
    // Stopped by the operator: uploads stopped, but checked like any other
    // enrolled agent below; the stop's line is seated before the verdict
    let mut stop_lines: BTreeMap<AgentId, (String, bool)> = BTreeMap::new();
    for (id, _) in pool.iter() {
        let dir = state_dir.join(id.to_string());
        if !sealed::is_enrolled(&dir) || !sealed::stop::is_stopped(&dir) {
            continue;
        }
        match sealed::stop::Stopped::load(&dir) {
            Ok(Some(stopped)) => {
                stop_lines.insert(*id, (stopped.line(), !stopped.shown));
                // A dry run checks nothing: the line alone, as before
                run.seats.insert(
                    *id,
                    consent::SealedSeat {
                        lines: vec![stopped.line()],
                        notice: false,
                        stop_note: !stopped.shown,
                        anchors_told: Vec::new(),
                        healing_told: None,
                    },
                );
            }
            Ok(None) => {}
            Err(e) => tracing::warn!(agent_id = %id, error = %e, "sealed/stopped unreadable"),
        }
    }
    if dry_run {
        if !enrolled.is_empty() {
            say!(
                "sealed: {} enrolled agents (not checked in a dry run)",
                enrolled.len()
            );
        }
        return run;
    }
    let names: BTreeMap<AgentId, String> = pool
        .iter()
        .map(|(id, s)| (*id, s.soul.name.to_string()))
        .collect();
    let name_of = |id: AgentId, reason: sealed::notice::Refusal| {
        names
            .get(&id)
            .cloned()
            .unwrap_or_else(|| format!("({})", reason.as_str()))
    };
    let uploads_of = |id: AgentId| uploads_of(state_dir, keys, id);
    let mut refused = BTreeSet::new();
    // The witness outside state/, before any agent is checked. Runs after
    // the uploading start pass, the stop reconcile and `load_states`
    // (which only reads, and finishes a sealed commit a crash cut short:
    // a head only ever moves forward there), so it sees what they leave
    let scan = match sealed::roster::check(data_dir, state_dir).await {
        Ok(scan) => scan,
        Err(e) => {
            tracing::error!(
                event_type = "sealed_roster_refused",
                error = %e,
                "the sealed roster couldn't be used; the run is refused"
            );
            run.fleet
                .push(format!("sealed roster: {e}. Way out: {}", e.way_out()));
            sealed::roster::Scan::default()
        }
    };
    if let Some((written, newest)) = scan.stale {
        tracing::warn!(
            event_type = "sealed_roster_stale",
            roster_written = %written,
            newest_state = %newest,
            "the sealed roster was last written long before the newest chained state"
        );
        alerts.notify(
            alerts::Alert::new(
                alerts::AlertKind::SealedStateRosterStale,
                "the sealed roster is older than the agents' states: a data directory restored \
                 from an older backup?",
            )
            .detail("roster_written", written)
            .detail("newest_state", newest),
        );
    }
    for (id, loss) in &scan.losses {
        let dir = state_dir.join(id.to_string());
        // A directory recreated to hold an earlier start's marker keeps
        // that start's reason; a loss found since a marker that isn't
        // loss-shaped is judged as the loss it is
        let reason = sealed::check::rejudged(&dir, loss.refusal());
        let name = name_of(*id, reason);
        tracing::error!(
            event_type = "sealed_roster_loss",
            agent_id = %id,
            loss = loss.as_str(),
            "an agent on the sealed roster shows a loss"
        );
        let detail = format!("the sealed roster vouches for this agent; found: {loss:?}");
        sealed::check::mark(&dir, *id, &name, reason, &detail).await;
        refused.insert(*id);
        run.refused.push(RefusedAgent {
            id: *id,
            name,
            reason,
            uploads: uploads_of(*id),
            detail,
        });
    }
    // An enrolled agent whose state is gone never loads: found by its keys
    let leftover = match sealed::check::missing_states(
        state_dir,
        keys,
        api,
        &alerts::Alerter::off(),
        &scan.on_roster,
    )
    .await
    {
        Ok(missing) => {
            for (id, reason) in missing.refused {
                if refused.insert(id) {
                    run.refused.push(RefusedAgent {
                        id,
                        name: name_of(id, reason),
                        reason,
                        uploads: uploads_of(id),
                        detail: "an enrolled agent's state or sealed/ is missing".to_string(),
                    });
                }
            }
            missing.leftover
        }
        Err(e) => {
            run.fleet
                .push(format!("the secrets tree couldn't be listed: {e}"));
            Vec::new()
        }
    };
    // The agents `sealed-enroll` recorded it can't chain until repaired
    // (`sealed::unchainable`), still unchained: left out of the run (never
    // run unchained), one alert a run, and not strays below. An unreadable
    // record excuses nobody
    let recorded = match sealed::unchainable::load(data_dir) {
        Ok(r) => r.agents,
        Err(e) => {
            tracing::error!(
                event_type = "sealed_unchainable_unreadable",
                error = %e,
                "the record of agents that can't be chained couldn't be read; every unchained \
                 state counts as a stray"
            );
            BTreeMap::new()
        }
    };
    let left_out: BTreeMap<AgentId, sealed::unchainable::Class> = scan
        .unchained
        .iter()
        .filter_map(|id| Some((*id, recorded.get(id)?.class)))
        .collect();
    pool.retain(|(id, _)| !left_out.contains_key(id));
    for (id, class) in &left_out {
        skipped.push(Skipped::new(
            *id,
            names.get(id).cloned(),
            "unchainable",
            format!("not chained ({}): {}", class.as_str(), class.remedy()),
        ));
    }
    let unchained: Vec<AgentId> = scan
        .unchained
        .iter()
        .filter(|id| !left_out.contains_key(id))
        .copied()
        .collect();
    let leftover: Vec<(AgentId, bool)> = leftover
        .into_iter()
        .filter(|(id, _)| !left_out.contains_key(id))
        .collect();
    // Leftover keys and unchained states: alerted, or, once a roster has
    // been kept, three or more at once is fleet-shaped (a roster lost with
    // the agents' sealed/ would read as a first run). An enrolment cut
    // short is both (a state key, and a state with no chain): once
    let strays = leftover
        .iter()
        .map(|(id, _)| *id)
        .chain(unchained.iter().copied())
        .collect::<BTreeSet<AgentId>>()
        .len();
    if !scan.first && strays >= sealed::check::REFUSED_AGENTS_STOP_RUN {
        tracing::error!(
            event_type = "sealed_state_strays_refused",
            leftover = leftover.len(),
            unchained = unchained.len(),
            strays,
            "agents with keys or states but no chain, beside a roster kept before; the run is refused"
        );
        run.fleet.push(format!(
            "{strays} agent(s) without a chain though a roster was kept before ({} with a state \
             key but no chained state, {} with a state but no chain): if they were never \
             chained (the first start of this runner, before the fleet's enrolment), chain them \
             with `agora-seed sealed-enroll` (the runner stopped), which names each one it \
             can't chain and why, then start again; otherwise their sealed/ may be gone with \
             the roster: put it back from a backup",
            leftover.len(),
            unchained.len()
        ));
    } else {
        if let Some(alert) = sealed::check::leftover_alert(&leftover) {
            alerts.notify(alert);
        }
        if !unchained.is_empty() || !left_out.is_empty() {
            let list = |ids: &mut dyn Iterator<Item = String>| {
                ids.take(sealed::check::SUMMARY_LIST)
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            let first = list(&mut unchained.iter().map(ToString::to_string));
            let left = list(
                &mut left_out
                    .iter()
                    .map(|(id, class)| format!("{id} ({})", class.as_str())),
            );
            if !unchained.is_empty() {
                tracing::warn!(
                    event_type = "sealed_state_unchained_agents",
                    agents = unchained.len(),
                    first = %first,
                    "agent directories with a state but no local chain, not on the roster; never enrolled"
                );
            }
            let mut alert = alerts::Alert::new(
                alerts::AlertKind::SealedStateUnchainedAgents,
                match (unchained.len(), left_out.len()) {
                    (n, 0) => format!(
                        "{n} agent(s) have a state but no local chain; enrol them with agora-seed \
                         sealed-enroll"
                    ),
                    (0, m) => format!(
                        "{m} agent(s) can't be chained until repaired and are left out of this \
                         run"
                    ),
                    (n, m) => format!(
                        "{n} agent(s) have a state but no local chain (enrol them with agora-seed \
                         sealed-enroll); {m} more can't be chained until repaired and are left \
                         out of this run"
                    ),
                },
            )
            .count(unchained.len() + left_out.len())
            .detail("agents", unchained.len());
            if !unchained.is_empty() {
                alert = alert.detail("first", first);
            }
            if !left_out.is_empty() {
                alert = alert
                    .detail("left_out", left_out.len())
                    .detail("left_out_first", left)
                    .detail(
                        "left_out_why",
                        format!(
                            "`agora-seed sealed-enroll` recorded them in {}: they don't run \
                             (never unchained) until it chains them; it names each one's class \
                             and remedy",
                            sealed::unchainable::RECORD
                        ),
                    );
            }
            alerts.notify(alert);
        }
    }
    // A pre-loss copy's pair cut short is finished, so the agent isn't
    // refused for it; a copy the agent chose to see since it was chained is
    // appended (a stopped agent's waits for its resume). After the roster,
    // which has judged every head as it was found (a pair appended to a
    // rolled-back directory could otherwise carry its head past the
    // roster's); never for an agent it or an earlier start refused. Each
    // under the agent's seal lock; a failure is logged and the start check
    // judges the agent as it is
    if let Some(enroller) = prior.filter(|_| run.fleet.is_empty()) {
        for (id, state) in pool.iter() {
            let Some((hash, _)) = enrolled.get(id) else {
                continue;
            };
            if refused.contains(id) || scan.refused.contains(id) {
                continue;
            }
            prior_step(enroller, *id, &state.soul.name, hash).await;
        }
    }
    // The anchors not yet judged are judged here where they can be, so a
    // session's verdict says what was found for the newest (beta round 12)
    let ctx = StartCheck {
        data_dir,
        state_dir,
        keys,
        api,
        alerts,
        start: verified.map(|verified| sealed::check::StartAnchors {
            verified,
            alerted: alerts.sends(alerts::AlertKind::SealedStateAnchorWrong),
        }),
    };
    let mut found = StartFound::default();
    for (id, state) in pool.iter() {
        // One of the runner's own records failed: the run is refused
        // whatever the agents' checks would say, so none is run and
        // nothing is marked against any agent for it (secrets that can't
        // be read for a moment would otherwise mark the whole fleet)
        if !run.fleet.is_empty() {
            break;
        }
        let Some((hash, sizes)) = enrolled.get(id) else {
            continue;
        };
        if refused.contains(id) {
            continue;
        }
        let runs = check_agent(
            &ctx,
            *id,
            state.soul.name.as_str(),
            (hash, *sizes),
            scan.vouched.get(id),
            stop_lines.remove(id),
            &mut run,
            &mut found,
            skipped,
        )
        .await;
        if !runs {
            refused.insert(*id);
        }
    }
    found.settle(&ctx, &mut run).await;
    run.outstanding = scan.refused.difference(&refused).count();
    pool.retain(|(id, _)| !refused.contains(id));
    run
}

/// Whether `id` counts as uploading for the refusal rule.
fn uploads_of(state_dir: &std::path::Path, keys: &sealed::KeyStore, id: AgentId) -> bool {
    sealed::check::Upload::counts_as_uploading(&state_dir.join(id.to_string()), keys, id)
}

/// What one chained agent's start check ([`check_agent`]) needs of the run.
struct StartCheck<'a> {
    data_dir: &'a std::path::Path,
    state_dir: &'a std::path::Path,
    keys: &'a sealed::KeyStore,
    api: &'a dyn sealed::api::SealedApi,
    alerts: &'a alerts::Alerter,
    start: Option<sealed::check::StartAnchors<'a>>,
}

/// What start checks found that is settled once for all of them
/// ([`StartFound::settle`]): the roster's vouches, and the two summary
/// alerts.
#[derive(Default)]
struct StartFound {
    dropped: Vec<sealed::check::Dropped>,
    /// Full verifications made, kept in the roster.
    vouches: Vec<(AgentId, sealed::roster::Vouch)>,
    unopenable: Vec<(AgentId, String, sealed::check::Unopenable)>,
}

impl StartFound {
    /// `run`'s seats get the unopenable alert's outcome: each agent named
    /// in it is told whether its operator was sent it.
    async fn settle(self, ctx: &StartCheck<'_>, run: &mut SealedRun) {
        if let Err(e) = sealed::roster::vouch_all(ctx.data_dir, &self.vouches).await {
            tracing::error!(
                event_type = "sealed_roster_unwritten",
                agents = self.vouches.len(),
                error = %e,
                "the roster's record of full verifications couldn't be written; the next start verifies them in full again"
            );
        }
        if let Some(alert) = sealed::check::drop_summary(&self.dropped) {
            ctx.alerts.notify(alert);
        }
        if let Some(alert) = sealed::check::unopenable_summary(&self.unopenable) {
            // Awaited, and each agent told what came of it
            let sent = ctx.alerts.dispatch(alert, alerts::TELL_TIMEOUT).await;
            for (id, _, _) in &self.unopenable {
                if let Some(seat) = run.seats.get_mut(id) {
                    sealed::check::say_told_after(
                        &mut seat.lines,
                        sealed::check::UNOPENABLE_LINE,
                        sent,
                    );
                }
            }
        }
    }
}

/// One chained agent's start check and everything the start does with
/// what it finds: the one path for an agent checked at the start
/// ([`sealed_start_check`]) and for one read again under its session lock
/// ([`reload_one`]), so neither can drift from the other.
///
/// `true`: it runs, seated (its lines, the stop's first), queued to drain,
/// held if its check held it; its anchors' findings alerted, its shrink and
/// unopenable versions noted in `found`, its vouch kept there. `false`: it
/// doesn't run. Refused: marked and told ([`sealed::check::mark`]) and
/// listed in `run.refused` for the refusal rule and the alerts; a
/// [`SealUnreadable`] refusal is listed but **not marked**, so the next
/// start judges it afresh (evidence that couldn't be read for a moment
/// needs no `sealed-accept`). Agora's copy failing: listed in
/// `run.server_faults`, nothing marked.
///
/// [`SealUnreadable`]: sealed::notice::Refusal::SealUnreadable
#[allow(clippy::too_many_arguments)]
async fn check_agent(
    ctx: &StartCheck<'_>,
    id: AgentId,
    name: &str,
    (hash, sizes): (&agora_agentkit::govlog::Sha256Hex, sealed::sizes::Sizes),
    vouched: Option<&sealed::roster::Vouch>,
    stop_line: Option<(String, bool)>,
    run: &mut SealedRun,
    found: &mut StartFound,
    skipped: &mut Vec<Skipped>,
) -> bool {
    let name = name.to_string();
    let dir = ctx.state_dir.join(id.to_string());
    let check = async |hash, sizes| {
        sealed::check::check_vouched(
            &dir,
            id,
            hash,
            sizes,
            ctx.keys,
            ctx.api,
            ctx.start.as_ref(),
            vouched,
        )
        .await
    };
    let (mut outcome, mut checked) = check(hash, sizes).await;
    // Another process may have saved the agent since its state was
    // loaded: read it again under its seal lock and check once more
    // before refusing a mismatch
    if matches!(
        outcome,
        sealed::check::Outcome::Refuse {
            reason: sealed::notice::Refusal::Mismatch,
            ..
        }
    ) && let Some((again, sizes)) = reloaded(ctx.state_dir, id).await
        && again != *hash
    {
        tracing::info!(
            event_type = "sealed_state_rechecked",
            agent_id = %id,
            "the state changed since it was loaded; checked again"
        );
        (outcome, checked) = check(&again, sizes).await;
    }
    found.vouches.extend(checked.vouch.map(|v| (id, v)));
    if let Some(u) = checked.unopenable {
        tracing::warn!(
            event_type = "sealed_state_unopenable",
            agent = %name,
            agent_id = %id,
            versions = u.versions,
            missing = u.missing,
            damaged = u.damaged,
            "sealed versions whose copies here are missing or damaged"
        );
        found.unopenable.push((id, name.clone(), u));
    }
    match outcome {
        sealed::check::Outcome::ServerFault { kind, detail } => {
            skipped.push(Skipped::new(
                id,
                Some(name.clone()),
                "sealed",
                "Agora's copy of its history failed verification".to_string(),
            ));
            run.server_faults.push((id, name, kind, detail));
            false
        }
        sealed::check::Outcome::Refuse { reason, detail } => {
            let reason = sealed::check::rejudged(&dir, reason);
            if reason == sealed::notice::Refusal::SealUnreadable {
                // Not marked: judged afresh at every start
                tracing::error!(
                    event_type = "sealed_state_refused",
                    agent = %name,
                    agent_id = %id,
                    reason = reason.as_str(),
                    detail,
                    marked = false,
                    "sealed history here couldn't be read; the agent is left out of this run, \
                     and the next start checks again"
                );
            } else {
                sealed::check::mark(&dir, id, &name, reason, &detail).await;
            }
            skipped.push(Skipped::new(
                id,
                Some(name.clone()),
                "sealed",
                format!("refused: {}", reason.as_str()),
            ));
            run.refused.push(RefusedAgent {
                id,
                name,
                reason,
                uploads: uploads_of(ctx.state_dir, ctx.keys, id),
                detail,
            });
            false
        }
        sealed::check::Outcome::Run {
            seat,
            notice,
            hold,
            drops,
            told,
            judged,
        } => {
            if let Some(result) = &judged {
                sealed::anchors::alert_pass(ctx.alerts, id, &name, result);
            }
            tracing::info!(
                event_type = "sealed_state_checked",
                agent = %name,
                agent_id = %id,
                hold = ?hold,
                notice,
                drops = drops.len(),
                "sealed state checked"
            );
            if !drops.is_empty() {
                for (what, pct) in &drops {
                    tracing::warn!(
                        event_type = "sealed_state_drop",
                        agent = %name,
                        agent_id = %id,
                        what,
                        percent = pct,
                        "memory or soul much smaller than at the previous sealed version"
                    );
                }
                found.dropped.push((id, name.clone(), drops));
            }
            let mut seat = seat;
            if let Some(hold) = hold {
                // Every start while held, rate-limited by the alerter
                let held = agora_agentkit::state::storage::AgentSeal::new(&dir, id)
                    .held()
                    .await
                    .ok();
                // Awaited: the agent is told what came of it, never that
                // its operator was told when no mail went out
                let sent = ctx
                    .alerts
                    .dispatch(
                        sealed::check::hold_alert(hold, id, &name, held.as_ref()),
                        alerts::TELL_TIMEOUT,
                    )
                    .await;
                sealed::check::say_told(&mut seat, hold, sent);
                run.held.push(id);
            }
            // Asked of Agora only for an agent that uploads
            if sealed::check::Upload::of(&dir, ctx.keys, id) == sealed::check::Upload::Now
                && let Some(n @ 1..) = sealed::check::unknown_rotations(ctx.keys, ctx.api, id).await
            {
                tracing::warn!(
                    event_type = "sealed_state_rotation_unknown",
                    agent_id = %id,
                    rotations = n,
                    "the server lists key rotations this runner never made; not trusted"
                );
                ctx.alerts.notify(
                    alerts::Alert::new(
                        alerts::AlertKind::SealedStateRotationUnknown,
                        format!(
                            "{name}: the server lists {n} key rotation(s) this runner never made"
                        ),
                    )
                    .agent(&name, id)
                    .detail("rotations", n),
                );
            }
            let (lines, stop_note) = match stop_line {
                Some((line, first)) => (std::iter::once(line).chain(seat).collect(), first),
                None => (seat, false),
            };
            // Last: what Agora found damaged and what was sent again
            let mut lines: Vec<String> = lines;
            let healing_told = sealed::healed::seat_into(&dir, id, &mut lines);
            run.seats.insert(
                id,
                consent::SealedSeat {
                    stop_note,
                    lines,
                    notice,
                    anchors_told: told,
                    healing_told,
                },
            );
            run.drained.push((id, name));
            true
        }
    }
}

/// `id`'s state read again under its seal lock (a save in progress
/// finishes first): its plaintext hash and sizes. `None` if it can't be.
async fn reloaded(
    state_dir: &std::path::Path,
    id: AgentId,
) -> Option<(agora_agentkit::govlog::Sha256Hex, sealed::sizes::Sizes)> {
    let seal = agora_agentkit::state::storage::AgentSeal::new(&state_dir.join(id.to_string()), id);
    let _lock = seal.lock().await.ok()?;
    let value = FsStorage::new(state_dir).load_raw(id).await.ok()?.value;
    Some((
        agora_agentkit::state::storage::plaintext_hash(&value),
        sealed::sizes::Sizes::of(&value),
    ))
}

/// What [`claim_sessions`] left: the session locks held, to the end of the
/// run, and what it left out.
#[derive(Debug, Default)]
struct Claimed {
    locks: BTreeMap<AgentId, sealed::session::SessionLock>,
    /// Held by another process: left for this sweep, still due.
    busy: usize,
    /// Past what the open-file limit allows: left for this sweep, still due.
    deferred: usize,
    /// Changed on disk since this run loaded them: read again and checked.
    reloaded: usize,
}

/// What [`claim_sessions`] reads an agent with again.
struct Reload<'a, S> {
    /// The run's sealed storage (a load finishes a commit a crash cut short)
    storage: &'a S,
    /// What the start check of every agent ran with ([`check_agent`])
    check: StartCheck<'a>,
    /// The agents that were chained when this run loaded them: one of
    /// them found unchained now has lost its marker.
    chained: &'a BTreeSet<AgentId>,
    /// The cadence gate routing applied (`min_cycle_secs`, an agent's own
    /// cadence, the forced names), by reactor: `None` is not gated.
    min_cycle: &'a dyn Fn(usize, AgentId, &str) -> Option<u64>,
}

/// Claim each routed agent's session ([`sealed::session::claim`]), in
/// routing order, before any reactor is built: an agent another process
/// holds is left out (one `agent_busy` event each, counted, no alert; it
/// stays due), and one this process can't hold a descriptor for
/// ([`sealed::session::fd_room`]) is left for a later sweep, with one
/// alert. Under its lock, an agent whose `state.json` or sealed head
/// changed since it was loaded (another process ran it in between) is read
/// again and its start check run again ([`reload_one`]): a session starts
/// from the newest state, never a stale copy.
#[allow(clippy::too_many_arguments)]
async fn claim_sessions<S>(
    cohorts: &mut [BTreeMap<String, Vec<(AgentId, SeedState)>>],
    seen: &BTreeMap<AgentId, sealed::session::Seen>,
    room: Option<usize>,
    reload: &Reload<'_, S>,
    sealed_run: &mut SealedRun,
    bases: &mut BTreeMap<AgentId, S::Version>,
    skipped: &mut Vec<Skipped>,
    alerts: &alerts::Alerter,
) -> Claimed
where
    S: Storage,
{
    let mut claimed = Claimed::default();
    let mut deferred_ids: Vec<AgentId> = Vec::new();
    for (reactor, cohort) in cohorts.iter_mut().enumerate() {
        for agents in cohort.values_mut() {
            for (id, state) in std::mem::take(agents) {
                if room.is_some_and(|room| claimed.locks.len() >= room) {
                    claimed.deferred += 1;
                    deferred_ids.push(id);
                    continue;
                }
                let name = state.soul.name.to_string();
                let lock = match sealed::session::claim(reload.check.state_dir, id) {
                    Ok(sealed::session::Claim::Run(lock)) => lock,
                    Ok(sealed::session::Claim::Busy) => {
                        claimed.busy += 1;
                        tracing::warn!(
                            event_type = "agent_busy",
                            agent_id = %id,
                            agent = %name,
                            "another process holds the agent's session lock; left for this sweep, \
                             still due"
                        );
                        continue;
                    }
                    Err(e) => {
                        skipped.push(Skipped::new(id, Some(name), "lock", e.to_string()));
                        continue;
                    }
                };
                let now = sealed::session::Seen::of(&reload.check.state_dir.join(id.to_string()));
                let state = if seen.get(&id) == Some(&now) {
                    state
                } else {
                    match reload_one(id, state, reactor, reload, sealed_run, bases, skipped).await {
                        Some(state) => {
                            claimed.reloaded += 1;
                            state
                        }
                        // Its lock is released here
                        None => continue,
                    }
                };
                claimed.locks.insert(id, lock);
                agents.push((id, state));
            }
        }
    }
    if claimed.deferred > 0 {
        let first: Vec<String> = deferred_ids
            .iter()
            .take(SKIPPED_ALERT_LIST)
            .map(ToString::to_string)
            .collect();
        tracing::error!(
            event_type = "session_locks_capped",
            claimed = claimed.locks.len(),
            deferred = claimed.deferred,
            room = room.unwrap_or_default(),
            first = %first.join(","),
            "the open-file limit leaves too few descriptors for a session lock per routed agent; \
             the rest stay due"
        );
        alerts.notify(
            alerts::Alert::new(
                alerts::AlertKind::SessionLocksCapped,
                format!(
                    "{} routed agent(s) left for a later sweep: the open-file limit allows {} \
                     session locks",
                    claimed.deferred,
                    claimed.locks.len()
                ),
            )
            .count(claimed.deferred)
            .detail("claimed", claimed.locks.len())
            .detail("first", first.join(", ")),
        );
    }
    claimed
}

/// When an agent whose last session ended at `last` is due again under
/// `min_cycle` seconds, if that is after `now`: the cadence gate, for
/// routing and for an agent read again under its session lock.
fn due_after(
    min_cycle: Option<u64>,
    last: Option<chrono::DateTime<chrono::Utc>>,
    now: chrono::DateTime<chrono::Utc>,
) -> Option<chrono::DateTime<chrono::Utc>> {
    let (secs, last) = (min_cycle?, last?);
    let cycle = chrono::Duration::seconds(secs.min(i64::MAX as u64) as i64);
    (last > now - cycle).then(|| last + cycle)
}

/// `id`'s state read again under its session lock, because it changed
/// since this run loaded it, and (chained) its start check run again on
/// it through the start's own path ([`check_agent`]), without the roster's
/// vouch (in full): its anchors' findings alerted and told once, a shrink
/// alerted, its vouch kept, as for every agent the start checked. `None`
/// leaves it out of this sweep:
///
/// - unreadable, or its model changed elsewhere (`reload`);
/// - it had a session in another process since routing, and isn't due
///   again inside `min_cycle_secs` (`not_due`; no error: it stays due
///   later);
/// - its check didn't let it run (`sealed`). A refusal is marked, told
///   and alerted here exactly as at the start (the refusal rule ran before
///   routing, so the rest of this run goes on; the marker counts at the
///   next start, where a loss-shaped one stops the run);
/// - it was chained when loaded and its chain marker is gone now: refused
///   as a missing head, never run unchained.
///
/// Its seat, hold and base are replaced with the new ones.
async fn reload_one<S: Storage>(
    id: AgentId,
    routed: SeedState,
    reactor: usize,
    reload: &Reload<'_, S>,
    sealed_run: &mut SealedRun,
    bases: &mut BTreeMap<AgentId, S::Version>,
    skipped: &mut Vec<Skipped>,
) -> Option<SeedState> {
    let ctx = &reload.check;
    let dir = ctx.state_dir.join(id.to_string());
    let name = routed.soul.name.to_string();
    // What the start found of it is replaced by what is found now
    sealed_run.seats.remove(&id);
    sealed_run.drained.retain(|(a, _)| *a != id);
    sealed_run.held.retain(|a| *a != id);
    let agora_agentkit::reactor::Loaded { value, base } = match reload.storage.load_raw(id).await {
        Ok(loaded) => loaded,
        Err(e) => {
            skipped.push(Skipped::new(id, Some(name), "reload", e.to_string()));
            return None;
        }
    };
    let hash = agora_agentkit::state::storage::plaintext_hash(&value);
    let sizes = sealed::sizes::Sizes::of(&value);
    let mut state = match serde_json::from_value::<SeedState>(value) {
        Ok(state) => state,
        Err(e) => {
            skipped.push(Skipped::new(id, Some(name), "reload", e.to_string()));
            return None;
        }
    };
    if state.model.id != routed.model.id {
        skipped.push(Skipped::new(
            id,
            Some(name),
            "reload",
            "its model changed in another process since it was loaded; routed again next sweep"
                .to_string(),
        ));
        return None;
    }
    // The session it had elsewhere since routing counts for the cadence
    // gate as any other: not due again yet is not an error
    if let Some(due) = due_after(
        (reload.min_cycle)(reactor, id, &name),
        state.last_cycle_at,
        chrono::Utc::now(),
    ) {
        tracing::info!(
            event_type = "session_reload_not_due",
            agent_id = %id,
            agent = %name,
            due = %due,
            "the agent had a session in another process since this run routed it; not due again \
             inside min_cycle_secs"
        );
        return None;
    }
    // The endpoint's offered ModelInfo, as routing set it (same id)
    state.model = routed.model;
    state.prompt.model = routed.prompt.model;
    let chained = sealed::is_enrolled(&dir);
    if !chained && reload.chained.contains(&id) {
        // Chained when loaded, its marker gone since: a loss, never a
        // session saved outside its chain
        let reason = sealed::notice::Refusal::HeadMissing;
        let detail = "chained when this run loaded it; its chain marker is gone since".to_string();
        sealed::check::mark(&dir, id, &name, reason, &detail).await;
        ctx.alerts
            .notify(sealed::check::refused_alert(id, &name, reason, &detail));
        skipped.push(Skipped::new(
            id,
            Some(name.clone()),
            "sealed",
            format!("refused: {}", reason.as_str()),
        ));
        sealed_run.refused.push(RefusedAgent {
            id,
            name,
            reason,
            uploads: uploads_of(ctx.state_dir, ctx.keys, id),
            detail,
        });
        return None;
    }
    if chained {
        // The stop's line first, as at the start
        let stop = sealed::stop::is_stopped(&dir)
            .then(|| sealed::stop::Stopped::load(&dir).ok().flatten())
            .flatten()
            .map(|stopped| (stopped.line(), !stopped.shown));
        let (refused, faults) = (sealed_run.refused.len(), sealed_run.server_faults.len());
        let mut found = StartFound::default();
        let runs = check_agent(
            ctx,
            id,
            &name,
            (&hash, sizes),
            None,
            stop,
            sealed_run,
            &mut found,
            skipped,
        )
        .await;
        found.settle(ctx, sealed_run).await;
        if !runs {
            // The refusal rule has run: this agent's alerts go now
            for r in &sealed_run.refused[refused..] {
                ctx.alerts.notify(sealed::check::refused_alert(
                    r.id, &r.name, r.reason, &r.detail,
                ));
            }
            let faults: Vec<&Fault> = sealed_run.server_faults[faults..].iter().collect();
            let (keys, chains): (Vec<&Fault>, Vec<&Fault>) = faults
                .into_iter()
                .partition(|f| f.2 == sealed::check::FaultKind::UntrustedKey);
            if !chains.is_empty() {
                server_fault_alert(&chains, ctx.alerts);
            }
            if !keys.is_empty() {
                untrusted_key_alert(&keys, ctx.alerts);
            }
            return None;
        }
    }
    match base {
        Some(base) => bases.insert(id, base),
        None => bases.remove(&id),
    };
    tracing::info!(
        event_type = "session_state_reloaded",
        agent_id = %id,
        agent = %name,
        "the agent's state changed on disk since this run loaded it: read again, and checked, \
         under its session lock"
    );
    Some(state)
}

/// A dry run skips the start pass that moves the pre-0.158 chain marker
/// (`sealed/enrolled` → `sealed/chained`), so an agent still carrying it
/// reads as not enrolled in its report: say how many, so the report isn't
/// silently wrong. Reads markers' existence only.
async fn say_legacy_markers(state_dir: &std::path::Path) {
    let n = legacy_markers(state_dir).await;
    if n > 0 {
        say!(
            "sealed: {n} agents still carry the pre-0.158 chain marker (sealed/enrolled); a dry \
             run doesn't move it, so they count as not enrolled above. The first real start \
             moves it"
        );
    }
}

/// How many agent directories under `state_dir` hold the pre-0.158 marker.
async fn legacy_markers(state_dir: &std::path::Path) -> usize {
    let Ok(entries) = std::fs::read_dir(state_dir) else {
        return 0;
    };
    let mut n = 0;
    for entry in entries.flatten() {
        let Some(id) = entry
            .file_name()
            .to_str()
            .and_then(|s| s.parse::<uuid::Uuid>().ok())
            .map(AgentId::from)
        else {
            continue;
        };
        let seal = agora_agentkit::state::storage::AgentSeal::new(&entry.path(), id);
        if seal.has_legacy_marker().await.unwrap_or(false) {
            n += 1;
        }
    }
    n
}

/// `agora-seed <sealed command>`: the data dir, server and
/// `[memory_restore]` from the flags and `--config`, as for a run.
async fn run_sealed_command(
    held: &mut Held,
    command: sealed::cli::Command,
    args: &Args,
) -> Result<()> {
    let config: Option<RunConfig> = match &args.config {
        Some(path) => {
            let body = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
            Some(toml::from_str(&body).with_context(|| format!("parsing {}", path.display()))?)
        }
        None => None,
    };
    let data_dir = match args
        .data_dir
        .clone()
        .or_else(|| config.as_ref().and_then(|c| c.data_dir.clone()))
    {
        Some(d) => d,
        None => dirs::home_dir()
            .context("no home directory")?
            .join("agents/agora"),
    };
    let server_url = config
        .as_ref()
        .and_then(|c| c.server_url.clone())
        .unwrap_or_else(|| args.server_url.clone());
    let (guards, _) = logging::init(args.log_dir.as_deref(), &data_dir, !args.no_log_file, false)?;
    held.log_guards = Some(guards);
    let restore = match config.as_ref().and_then(|c| c.memory_restore.as_ref()) {
        Some(c) => c.resolve()?,
        None => None,
    };
    sealed::cli::run(command, &data_dir, server_url, restore).await
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut held = Held::default();
    let result = run(&mut held).await;
    // Before the error (if any) ends the process: an alert raised on the
    // way out — `governance_log_refused` above all — is sent, not dropped.
    held.alerts.flush(ALERT_FLUSH).await;
    let stopped = held.stopped && result.is_ok();
    let refused = matches!(&result, Err(e) if e.is::<RefusedRun>());
    drop(held);
    if stopped {
        std::process::exit(signals::STOPPED_EXIT);
    }
    if refused {
        if let Err(e) = &result {
            eprintln!("Error: {e:#}");
        }
        std::process::exit(sealed::REFUSED_RUN_EXIT);
    }
    result
}

async fn run(held: &mut Held) -> Result<()> {
    let mut args = Args::parse();
    args.reject_retired()?;
    if args.consent_queue {
        return print_consent_queue(&args);
    }
    if let Some(command) = args.command.take() {
        return run_sealed_command(held, command, &args).await;
    }
    if let (Some(spec), Some(credential), Some(key_file), Some(data_dir)) = (
        &args.provision,
        args.provision_credential,
        &args.provision_key_file,
        &args.data_dir,
    ) {
        let operator = provision::Operator {
            credential,
            key_file: key_file.clone(),
        };
        let id = provision::provision(
            spec,
            &operator,
            args.server_url.clone(),
            data_dir,
            bridge::model_info(),
        )
        .await?;
        tracing::info!(event_type = "agent_provisioned", agent_id = %id, data_dir = %data_dir.display(), "agent provisioned");
        println!("{id}");
        return Ok(());
    }

    // Normalize both invocation shapes to a RunConfig.
    let mut config: RunConfig = match &args.config {
        Some(path) => {
            let body = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
            toml::from_str(&body).with_context(|| format!("parsing {}", path.display()))?
        }
        None => {
            let endpoint = args
                .endpoint
                .clone()
                .context("either --config or --endpoint is required")?;
            RunConfig {
                data_dir: args.data_dir.clone(),
                server_url: Some(args.server_url.clone()),
                min_cycle_secs: None,
                force: false,
                wave_size: Some(args.wave_size),
                agents: args.agents.clone(),
                agents_file: None,
                models: Vec::new(),
                model_table: Vec::new(),
                model_narrow: Vec::new(),
                extra_agents: vec![],
                extra_agents_file: None,
                seed: SeedKnobs::default(),
                model_consent: consent::ConsentConfig::default(),
                role_consent: None,
                cadence_consent: None,
                memory_restore: None,
                sealed_state: None,
                sleeping: Vec::new(),
                wake: Vec::new(),
                schedule: schedule::ScheduleConfig::default(),
                review_forks: consent::forks::ForksConfig::default(),
                alerts: None,
                reactors: vec![ReactorSpec {
                    endpoint,
                    min_cycle_secs: None,
                    limit: args.limit,
                    concurrency: args.concurrency,
                    key_file: args.anthropic_key_file.clone(),
                    max_batch: None,
                    poll_secs: None,
                    unreachable_hold_secs: default_unreachable_hold_secs(),
                }],
            }
        }
    };
    narrow_to_named(&mut config, &args.agents);
    narrow_to_models(&mut config, &args.models);
    let specs = model_specs(&config)?;
    let allowlist = model_allowlist(&config, &specs);
    anyhow::ensure!(
        !config.reactors.is_empty(),
        "config has no [[reactor]] blocks"
    );
    config.schedule.validate()?;
    config.review_forks.validate()?;
    if let Some(alerts) = &config.alerts {
        alerts.validate()?;
    }

    // The flag wins over the config, as for `--consent-queue`: until
    // 2026-10-13 a `--data-dir` given with `--config` was silently ignored,
    // and the run used the config's (by default the fleet's) tree
    let data_dir = match args.data_dir.clone().or_else(|| config.data_dir.clone()) {
        Some(d) => d,
        None => dirs::home_dir()
            .context("no home directory")?
            .join("agents/agora"),
    };
    // The start, in order (each step sees what the ones before it leave):
    //  1. `promote`: a save cut between its archive and its rename put in
    //     place, only while no other runner shares the data dir (it takes
    //     runner.lock exclusive, not waiting, so it comes before the shared
    //     hold), under each agent's session lock, `sealed/lock` and
    //     directory lock; it reads the chain's markers' existence, and for
    //     a chained agent the outbox items that are its evidence;
    //  2. runner.lock, shared (the operator's commands take it exclusive);
    //     then, once the log and the alerts are up, the free-space gate
    //     (`space`): below the floor nothing more of this list happens;
    //  3. `uploading::start_pass`: the chain's marker moved to its new name
    //     first (before anything reads it), a lost or half-written uploading
    //     marker repaired, the one-time migration, markers in doubt alerted;
    //  4. the stop reconcile;
    //  5. `load_states`, which finishes a sealed commit a crash cut short (a
    //     head only moves forward there);
    //  6. the governance log verified (it reads no agent's state);
    //  7. the sealed start check (`sealed_start_check`): the roster and its
    //     witnesses against every head as 5 left it, then a pre-loss copy's
    //     pair finished or a chosen copy appended for agents the roster
    //     didn't refuse, then each agent checked;
    //  8. the refusal rule (`refuse_run`);
    //  9. routing, then each routed agent's session lock (`claim_sessions`:
    //     an agent another process holds is left due; one changed on disk
    //     since 5 is read again and checked again under its lock);
    // 10. reactors built, every agent saved on its own (`Independent`);
    //     after the run, unsaved sessions rescued and told
    //     (`settle_unsaved`), and only then the session locks released.
    // A dry run or a listing reads only: no lock, no writes
    if !args.dry_run && !args.list_models {
        // A save cut between its archive and its rename, while no other
        // runner shares the data dir (exclusive, not waiting): before the
        // shared hold, and before any state loads
        promote::promote_if_alone(&data_dir).await;
        held.runner_lock = Some(sealed::session::runner_shared(&data_dir)?);
    }
    if args.test_alert {
        let alerts = config
            .alerts
            .as_ref()
            .context("--test-alert: the config has no [alerts] table")?;
        let alerter = alerts::Alerter::from_config(alerts, &data_dir, None)?;
        alerter
            .send_test()
            .await
            .context("--test-alert: the test alert was not sent")?;
        say!("test alert sent");
        return Ok(());
    }

    // Tracing comes up as soon as the data dir is known, because the data
    // dir roots the log path. Everything above here reports through
    // `anyhow` to stderr instead.
    //
    // The guards must outlive every log call: dropping them flushes and
    // closes the worker, and an early drop truncates the file silently.
    // `held` keeps them until `main` has flushed the alerts.
    let (log_guards, log_path) = logging::init(
        args.log_dir.as_deref(),
        &data_dir,
        // `--list-models` runs no agents and would drop an empty log every
        // time it's polled. A `--dry-run` routing report *is* worth
        // keeping — a deleted model strands agents in near silence, and
        // that report is what shows it.
        !args.no_log_file && !args.list_models,
        // The scheduler's ledgers record what agents do; a run that drives
        // none has nothing to add.
        !args.dry_run && !args.list_models,
    )?;
    held.log_guards = Some(log_guards);
    if let Some(path) = &log_path {
        tracing::info!(path = %path.display(), "run log opened");
    }
    // Built before anything that can raise an alert. A dry run or a model
    // listing is someone at a terminal, so it mails nobody.
    if let Some(alerts) = &config.alerts {
        if args.dry_run || args.list_models {
            tracing::info!("alerts are off for --dry-run and --list-models");
        } else {
            held.alerts = alerts::Alerter::from_config(alerts, &data_dir, log_path.clone())?;
            tracing::info!(
                event_type = "alerts_configured",
                events = ?alerts
                    .events
                    .as_ref()
                    .map(|e| e.iter().map(|k| k.as_str()).collect::<Vec<_>>()),
                min_interval_secs = alerts.min_interval_secs,
                "operator alerts on"
            );
        }
    }
    // Once, before anything is loaded or any session starts: a filesystem
    // too full to take the run's saves starts none (`space`). Filesystem
    // statistics only. A dry run says what a run would do
    if !args.list_models {
        let (state_root, secrets_root) = (data_dir.join("state"), data_dir.join("secrets"));
        let roots = [state_root.as_path(), secrets_root.as_path()];
        if args.dry_run {
            for l in space::below_floor_with(&roots, space::stat) {
                say!("low space: {l}; a run would start no session");
            }
        } else {
            space::gate_with(&roots, space::stat, &held.alerts)?;
        }
    }
    if !config.models.is_empty() {
        tracing::warn!(
            models = ?config.models,
            "`models = [...]` is deprecated and will be removed: list each as \
             [[model]] (id, name, description, selectable)"
        );
    }

    let server_url = config
        .server_url
        .clone()
        .unwrap_or_else(|| url::Url::parse("https://subliminal.technology").expect("static url"));

    // Build every inference client and discover models up front, so a bad
    // spec fails the run before any agent does any work.
    let mut endpoints: Vec<endpoint::Endpoint> = Vec::new();
    let mut offered: Vec<(usize, ModelInfo)> = Vec::new();
    let mut bridged: Option<(bridge::Ending, bridge::Driver)> = None;
    for (idx, spec) in config.reactors.iter().enumerate() {
        let inference = match endpoint_kind(&spec.endpoint)? {
            EndpointKind::Messages(_) => endpoint::Endpoint::Messages(build_inference(spec)?),
            EndpointKind::Bridge => {
                anyhow::ensure!(bridged.is_none(), "a run may have one bridge endpoint");
                // Never the default, which is the fleet's tree
                anyhow::ensure!(
                    args.data_dir.is_some(),
                    "a bridge endpoint needs --data-dir: the test agent's own, never the fleet's"
                );
                bridge::take_stdout();
                let (inference, ending, driver) = bridge::bridge();
                bridged = Some((ending, driver));
                endpoint::Endpoint::Bridge(inference)
            }
        };
        // An unreachable endpoint is waited for, with the session hold's
        // period, not failed: see `endpoint::hold_unreachable`. Someone at a
        // terminal (`--list-models`, `--dry-run`, `--prepare-review-forks`)
        // is told at once instead. Endpoints are discovered in order, so a
        // down one also holds back those after it; every config has one.
        let hold = endpoint::startup_hold(
            spec.unreachable_hold_secs,
            args.list_models || args.dry_run || args.prepare_review_forks,
        );
        let models = endpoint::hold_unreachable(&spec.endpoint, hold, || inference.models())
            .await
            .with_context(|| format!("discovering models on {}", spec.endpoint))?;
        if args.list_models {
            say!("# {}", spec.endpoint);
            for m in models.iter() {
                say!("{}", m.id.name());
            }
        }
        offered.extend(models.iter().map(|m| (idx, m.clone())));
        endpoints.push(inference);
    }
    if args.list_models {
        return Ok(());
    }
    // The bridge's driver is answered from here on: its handshake at once,
    // its first request when the agent's session starts. stdout is the MCP
    // stream from now on, so nothing below may print to it.
    let bridge = match bridged {
        Some((ending, driver)) if !args.dry_run => Some((ending, bridge::serve(driver).await?)),
        _ => None,
    };

    // Trial-review forks, before any agent runs: generating them loads
    // both models, so they're done together, grouped by model.
    let log_dir = args
        .log_dir
        .clone()
        .unwrap_or_else(|| data_dir.join("logs"));
    let fork_seed = config.seed.to_config(&data_dir, !args.no_prompt_log)?;
    let act_max_tokens = fork_seed.act_max_tokens;
    // A fork thinks as a session on its model would (see
    // `consent::forks::SessionThinking`).
    let fork_thinking: Vec<(Model, consent::forks::SessionThinking)> = offered
        .iter()
        .map(|(_, m)| {
            let spec = specs.iter().find(|s| s.id == m.id);
            let model_effort = spec
                .and_then(|s| s.thinking_effort)
                .map(EffortKnob::into_effort);
            let mut thinking = consent::forks::SessionThinking::resolve(
                model_effort,
                fork_seed.thinking_effort.clone(),
                fork_seed.thinking_budget_tokens,
            );
            thinking.act_max_tokens = spec.and_then(|s| s.act_max_tokens);
            (m.id.clone(), thinking)
        })
        .collect();
    if args.dry_run {
        match consent::forks::jobs(&data_dir.join("state")).await {
            Ok(jobs) if !jobs.is_empty() => say!(
                "review forks: {} agents would have forks prepared",
                jobs.len()
            ),
            Ok(_) => {}
            Err(e) => say!("review forks: state unreadable: {e}"),
        }
    } else if args.prepare_review_forks || (config.review_forks.enabled && !args.no_review_forks) {
        let prompt_dir = config
            .seed
            .prompt_log_dir
            .clone()
            .unwrap_or_else(|| data_dir.join("logs").join("prompts"));
        let prepared = consent::forks::prepare(
            &data_dir.join("state"),
            &log_dir,
            Some(&prompt_dir),
            consent::forks::Endpoints {
                clients: &endpoints,
                offered: &offered,
                thinking: &fork_thinking,
            },
            act_max_tokens,
            config.review_forks,
        )
        .await;
        match prepared {
            Ok(n) => tracing::info!(agents = n, "review forks step done"),
            // Never fatal: the review goes without forks and says so.
            Err(e) => tracing::error!(error = %e, "review forks step failed"),
        }
    }
    if args.prepare_review_forks {
        return Ok(());
    }

    // Load the full agent pool once, filter to named agents if any, route.
    // Through the sealed storage, which finishes a sealed commit a crash
    // interrupted; a dry run writes nothing, so reads plainly.
    let state_dir = data_dir.join("state");
    let enrolment = Arc::new(sealed::SealedEnrolment::new(
        &state_dir,
        data_dir.join("secrets"),
    ));
    let sealed_client = agora_agentkit::client::Client::new(server_url.clone())?;
    let outbox = agora_agentkit::state::outbox::AgoraOutbox::new(
        sealed_client.clone(),
        &state_dir,
        enrolment.clone(),
    );
    // Every per-agent sealed-state call of the run: nothing of an agent
    // that doesn't upload is sent or asked (`sealed::api::UploadGuard`)
    let sealed_api = Arc::new(sealed::api::guarded(sealed_client, enrolment.clone()));
    // The chain's marker moved to its post-0.158 name; a stop or an
    // uploading marker kept in the secrets tree stands for a lost local
    // copy; a half-written uploading marker is finished; the
    // agents that uploaded before the uploading marker existed are marked
    // (once); markers still in doubt are alerted. A dry run writes nothing
    if !args.dry_run {
        // First: the marker's move, before anything reads it
        sealed::uploading::start_pass(&state_dir, enrolment.keys_store(), &held.alerts).await;
        sealed::stop::reconcile_all(&state_dir, enrolment.keys_store());
    }
    let loaded = if args.dry_run {
        let plain = load_states(&FsStorage::new(&state_dir), &state_dir).await?;
        LoadedStates {
            states: plain.states,
            bases: BTreeMap::new(),
            sealed: plain.sealed,
            skipped: plain.skipped,
            seen: plain.seen,
        }
    } else {
        let storage = sealed::storage(&state_dir, outbox.clone(), enrolment.clone());
        load_states(&storage, &state_dir).await?
    };
    let LoadedStates {
        states: mut pool,
        mut bases,
        sealed: enrolled,
        mut skipped,
        seen,
    } = loaded;
    // An agent mid-rotation (`secrets.next.json`) is left out, unrefused:
    // its keys and the server's may disagree until `rotate-keys` finishes
    mid_rotation(
        &mut pool,
        &data_dir.join("secrets"),
        &held.alerts,
        &mut skipped,
    );
    // The reference client verifies the governance log every run
    // (agora#127): signatures, chain, and the head entry's content.
    // A log that does not verify stops the run before any agent acts.
    // A dry run verifies too — it is read-only and worth knowing. Before
    // the sealed-state start check, which judges the anchors it can and
    // names the rest, the newest first, with why.
    let verified_anchors = govlog::verify(
        &agora_agentkit::client::Client::new(server_url.clone())?,
        &data_dir,
        &held.alerts,
    )
    .await?;
    // Never fatal: a table that can't be used is logged, alerted, and the
    // run goes on without the offer. Resolved before the start check, which
    // appends a pre-loss copy the agent chose to see
    let restore_offer = config
        .memory_restore
        .as_ref()
        .and_then(|c| consent::memory_restore::resolve_or_disable(c, &held.alerts));
    let prior = (!args.dry_run).then(|| sealed::enroll::Enroller {
        state_dir: state_dir.clone(),
        keys: enrolment.keys_store().clone(),
        restore: restore_offer.clone(),
    });
    // The sealed-state start check, for enrolled agents: before anyone is
    // built, so a refused agent is never run, saved or sealed.
    let mut sealed_run = sealed_start_check(
        &mut pool,
        &enrolled,
        &data_dir,
        &state_dir,
        enrolment.keys_store(),
        sealed_api.as_ref(),
        &held.alerts,
        args.dry_run,
        &mut skipped,
        Some(&verified_anchors),
        prior.as_ref(),
    )
    .await;
    // Before any reactor is built. A dry run checks nothing, so refuses
    // nothing
    refuse_run(&sealed_run, &held.alerts)?;
    let sealed_offer = config
        .sealed_state
        .as_ref()
        .and_then(|c| consent::sealed_state::resolve_or_disable(c, &held.alerts));
    if let Some(offer) = &sealed_offer {
        for id in offer.ids() {
            if !pool.iter().any(|(a, _)| *a == id) && !skipped.iter().any(|s| s.id == id) {
                tracing::warn!(agent_id = %id, "`[sealed_state].agents` names no loaded agent");
            }
        }
    }
    let role_offer = match &config.role_consent {
        Some(c) => c.resolve()?,
        None => None,
    };
    let base_rounds = config
        .seed
        .to_config(&data_dir, !args.no_prompt_log)?
        .max_rounds;
    let cadence_offer = match &config.cadence_consent {
        Some(c) => c.resolve(base_rounds)?,
        None => None,
    };
    if let Some(offer) = &cadence_offer {
        for name in unknown_names(offer.names(), &pool) {
            tracing::warn!(agent = %name, "`[cadence_consent].agents` names no loaded agent");
        }
    }
    if let Some(offer) = &restore_offer {
        for name in unknown_names(offer.names(), &pool) {
            tracing::warn!(agent = %name, "`[memory_restore]` agents_file names no loaded agent");
        }
    }
    let restore_on = restore_offer.is_some();
    if let Some(offer) = &role_offer {
        for name in unknown_names(offer.names(), &pool) {
            tracing::warn!(agent = %name, "`[role_consent].agents` names no loaded agent");
        }
    }
    let sleepers = put_to_sleep(
        &mut pool,
        &config.sleeping,
        &config.wake,
        &data_dir.join("state"),
    );
    let names = named_agents(&config)?;
    let extras = extra_agents(&config)?;
    if !names.is_empty() {
        pool.retain(|(_, s)| {
            let name = s.soul.name.as_str();
            names.iter().chain(&extras).any(|n| n == name)
        });
    }
    let cadence_plan =
        consent::cadence::CadencePlan::load(cadence_offer.as_ref(), &pool, &data_dir.join("state"));
    let mut routing = route(
        pool,
        &config.reactors,
        &offered,
        &names,
        &extras,
        &allowlist,
        config.min_cycle_secs,
        &cadence_plan,
        !args.agents.is_empty() || config.force,
    );
    // Each routed agent's session lock, before anything is built for it,
    // held until the run has saved (or rescued) its state and told it what
    // it must be told: dropped at the end of `run`, or by the kernel at any
    // exit. A dry run claims nothing
    let mut claimed = Claimed::default();
    if !args.dry_run {
        let storage = sealed::storage(&state_dir, outbox.clone(), enrolment.clone());
        let chained: BTreeSet<AgentId> = enrolled.keys().copied().collect();
        let force_named = !args.agents.is_empty() || config.force;
        let min_cycle = |reactor: usize, id: AgentId, name: &str| {
            min_cycle_for(
                &config.reactors[reactor],
                config.min_cycle_secs,
                &cadence_plan,
                id,
                name,
                &names,
                force_named,
            )
        };
        let reload = Reload {
            storage: &storage,
            check: StartCheck {
                data_dir: &data_dir,
                state_dir: &state_dir,
                keys: enrolment.keys_store(),
                api: sealed_api.as_ref(),
                alerts: &held.alerts,
                start: Some(sealed::check::StartAnchors {
                    verified: &verified_anchors,
                    alerted: held.alerts.sends(alerts::AlertKind::SealedStateAnchorWrong),
                }),
            },
            chained: &chained,
            min_cycle: &min_cycle,
        };
        claimed = claim_sessions(
            &mut routing.cohorts,
            &seen,
            sealed::session::fd_room(),
            &reload,
            &mut sealed_run,
            &mut bases,
            &mut skipped,
            &held.alerts,
        )
        .await;
        tracing::info!(
            event_type = "sessions_claimed",
            claimed = claimed.locks.len(),
            busy = claimed.busy,
            deferred = claimed.deferred,
            reloaded = claimed.reloaded,
            "session locks taken for the routed agents"
        );
    } else {
        say_legacy_markers(&state_dir).await;
    }
    let mut session_locks = claimed.locks;

    // The plan, before anyone runs.
    for (spec, cohort) in config.reactors.iter().zip(&routing.cohorts) {
        say!("== {}", spec.endpoint);
        for (model, agents) in cohort {
            say!("   {model} ×{}", agents.len());
        }
    }
    for (model, count) in &routing.unrouted {
        say!("unrouted: {model} ×{count} (no endpoint offers this model)");
    }
    for (model, count) in &routing.excluded {
        say!("excluded: {model} ×{count} (not in the `models` allowlist)");
    }
    if routing.placeholders > 0 {
        say!(
            "placeholders: {} agents have no model (state.model unset); skipped",
            routing.placeholders
        );
    }
    if routing.not_due > 0 {
        say!("not due: {} agents inside min_cycle_secs", routing.not_due);
    }
    if claimed.busy > 0 {
        say!(
            "busy: {} agents held by another process (a session there); left for this sweep",
            claimed.busy
        );
    }
    if claimed.deferred > 0 {
        say!(
            "deferred: {} agents past the open-file limit's room for session locks",
            claimed.deferred
        );
    }
    if !sleepers.asleep.is_empty() {
        say!(
            "sleeping: {} ×{} (the `sleeping` list; delete a line to wake)",
            sleepers.asleep.join(", "),
            sleepers.asleep.len()
        );
    }
    if !sleepers.chose_sleep.is_empty() {
        say!(
            "sleeping: {} ×{} (answered `sleep` to the role offer; add to `wake` to wake)",
            sleepers.chose_sleep.join(", "),
            sleepers.chose_sleep.len()
        );
    }
    if !sleepers.woken.is_empty() {
        say!(
            "woken: {} ×{} (answered `sleep`, listed in `wake`)",
            sleepers.woken.join(", "),
            sleepers.woken.len()
        );
    }
    for name in &sleepers.unknown {
        tracing::warn!(agent = %name, "`sleeping` or `wake` names no loaded agent");
    }
    let every_other_day = cadence_plan.names();
    if !every_other_day.is_empty() {
        say!(
            "every other day: {} ×{} (chose `switch` in the cadence offer: {} rounds, \
             at most one session per {} h)",
            every_other_day.join(", "),
            every_other_day.len(),
            base_rounds * consent::cadence::ROUNDS_FACTOR,
            cadence_plan.cycle_secs() / 3600
        );
    }

    // Shared per-process context. Keep a concrete keyring handle for the
    // E2EE encryption-key backfill below (the trait object can't do it).
    let keyring = FsKeyring::new(data_dir.join("secrets"));
    let seed_config = config.seed.to_config(&data_dir, !args.no_prompt_log)?;
    let consent = Arc::new(
        consent::ConsentRuntime::new(
            config.model_consent.clone(),
            &data_dir,
            agora_agentkit::client::Client::new(server_url.clone())?,
            Arc::new(keyring.clone()),
            models::Catalog::new(
                &specs,
                &offered.iter().map(|(_, m)| m.clone()).collect::<Vec<_>>(),
            ),
            seed_config.phase_max_tokens,
        )?
        .with_alerts(held.alerts.clone())
        .with_role_offer(role_offer)
        .with_cadence_offer(cadence_offer)
        .with_memory_restore_offer(restore_offer)
        .with_sealed_state(sealed_offer, sealed_run.seats.clone()),
    );
    let context = consent::agent::ConsentContext {
        inner: SeedContext {
            client: agora_agentkit::client::Client::new(server_url)?,
            keys: Arc::new(keyring.clone()),
            config: seed_config,
        },
        consent,
    };

    // The constitution the agents read must be the one the server serves,
    // whole: a truncated or stale copy stops the run the same way.
    let web_tools =
        context.inner.config.web_search.is_some() || context.inner.config.web_fetch.is_some();
    constitution::verify(&context.inner.client, web_tools, &held.alerts).await?;

    // Assemble reactors: order each endpoint's cohort, cap, construct.
    let wave_size = config.wave_size.unwrap_or(8);
    let ledgers = Ledgers::load(&data_dir, &config.schedule);
    let shares: BTreeMap<Model, f64> = specs.iter().map(|s| (s.id.clone(), s.share)).collect();
    let mut orchestrator = Orchestrator::new();
    let mut labels: BTreeMap<ReactorId, String> = BTreeMap::new();
    // Raised by `SIGINT` and `SIGTERM` once the run starts (`signals`)
    let stop = Stop::new();
    let save_failures = SaveFailures::default();
    let mut total = 0usize;
    for (idx, (spec, (inference, cohort))) in config
        .reactors
        .iter()
        .zip(endpoints.into_iter().zip(routing.cohorts))
        .enumerate()
    {
        let kind = endpoint_kind(&spec.endpoint)?;
        let mut ordered = match kind {
            // Batch cohorts submit whole per round: order is moot, and
            // Haiku is outside the local fair-share pool by design. The
            // bridge runs one agent.
            EndpointKind::Messages(EndpointVariant::Anthropic) | EndpointKind::Bridge => {
                interleave(cohort, wave_size)
            }
            _ => {
                let offered: BTreeSet<Model> = offered
                    .iter()
                    .filter(|(i, _)| *i == idx)
                    .map(|(_, m)| m.id.clone())
                    .collect();
                let (ordered, plan) = fair_share(
                    cohort,
                    &shares,
                    &offered,
                    &ledgers,
                    wave_size,
                    &config.schedule,
                );
                if args.dry_run {
                    say_raw!("{}", plan.report());
                }
                plan.log(&spec.endpoint, config.schedule.ceiling, &held.alerts);
                ordered
            }
        };
        if let Some(limit) = spec.limit {
            // Those cut are not run: their session locks go now
            for (id, _) in ordered.iter().skip(limit) {
                session_locks.remove(id);
            }
            ordered.truncate(limit);
        }
        // One driver answers for one agent: a second agent's session would
        // arrive in the middle of the first's conversation
        if kind == EndpointKind::Bridge && ordered.len() > 1 {
            anyhow::bail!(
                "the bridge endpoint runs one agent, and {} were routed to it: \
                 name it with --agent",
                ordered.len()
            );
        }
        if args.dry_run {
            for (pos, (_, state)) in ordered.iter().enumerate() {
                say!(
                    "{pos:>5} {} {} [{}]",
                    state.model.id.name(),
                    state.soul.name,
                    spec.endpoint
                );
            }
            continue;
        }
        if ordered.is_empty() {
            tracing::warn!(endpoint = %spec.endpoint, "no agents routed, skipping reactor");
            continue;
        }
        let mut agents: Vec<SeedRunAgent> = Vec::with_capacity(ordered.len());
        for (id, state) in ordered {
            // E2EE: generate-and-persist an X25519 key for agents that
            // predate encryption keys. Failure degrades that agent to
            // server-mode messaging; it must not block the run.
            if let Err(e) = keyring.ensure_encryption_key(id) {
                tracing::warn!(
                    agent_id = %id,
                    error = %e,
                    "encryption key backfill failed; agent messages server-mode"
                );
            }
            // A model's own token budgets (`[[model]] *_max_tokens`).
            let mut agent_context = context.clone();
            let entry = agent_context.consent.catalog.get(&state.model.id).cloned();
            if let Some(entry) = &entry
                && entry.budgets != models::Budgets::default()
            {
                entry.budgets.apply(&mut agent_context.inner.config);
            }
            // An agent on its own every-other-day cadence gets its rounds
            // doubled, and its model's switch window if it sets one;
            // everyone else shares the run's config.
            if let Some(rounds) = cadence_plan.max_rounds(&id) {
                let config = &mut agent_context.inner.config;
                config.max_rounds = rounds;
                if let Some(entry) = &entry {
                    config.context_window = models::switched_context_window(
                        config.context_window,
                        &entry.budgets,
                        entry.info.max_input_tokens,
                    );
                }
                tracing::info!(
                    event_type = "cadence_switch_applied",
                    agent = %state.soul.name,
                    agent_id = %id,
                    max_rounds = rounds,
                    context_window = config.context_window,
                    min_cycle_secs = cadence_plan.cycle_secs(),
                    "agent runs on its chosen cadence"
                );
            }
            let name = state.soul.name.clone();
            match agora_agentkit::reactor::Agent::new(id, state, agent_context) {
                Ok(agent) => agents.push(agent),
                Err(e) => {
                    session_locks.remove(&id);
                    skipped.push(Skipped::new(
                        id,
                        Some(name.to_string()),
                        "construct",
                        e.to_string(),
                    ))
                }
            }
        }
        total += agents.len();
        tracing::info!(
            agents = agents.len(),
            endpoint = %spec.endpoint,
            "reactor ready"
        );
        let cohort_bases: Vec<_> = agents
            .iter()
            .filter_map(|a| {
                let id = agora_agentkit::reactor::Agent::id(a);
                Some((id, *bases.get(&id)?))
            })
            .collect();
        // Each agent saved on its own, failures kept by class
        let storage = Independent::new(
            sealed::storage(&state_dir, outbox.clone(), enrolment.clone()),
            save_failures.clone(),
        );
        let mut reactor: Reactor<_, _, SeedRunAgent> = Reactor::new(inference, storage, agents)
            .with_bases(cohort_bases)
            .with_stop(stop.clone());
        if spec.unreachable_hold_secs > 0 {
            reactor = reactor
                .with_unreachable_hold(std::time::Duration::from_secs(spec.unreachable_hold_secs));
        }
        labels.insert(Run::id(&reactor), spec.endpoint.clone());
        orchestrator.push(reactor);
    }
    report_skipped(&skipped, &held.alerts);
    let agents_skipped = skipped.len();
    if args.dry_run {
        return Ok(());
    }
    // An empty cohort is a success when cadence emptied it: under systemd
    // `Restart=always` this is the steady state of a continuous sweep once
    // every agent has cycled inside `min_cycle_secs`, and exiting non-zero
    // here made every idle pass "fail" — and page — on 2026-09-08. It is
    // still an error when nobody was gated: no agents at all, or none
    // routable, means the config is wrong, not that the fleet is resting.
    if total == 0 {
        // A bridge's driver is told why, rather than finding no tools: the
        // bridge served the handshake, and leaving now closed its stream
        // (2026-10-26: a beta's session lost to an agent not yet due)
        let mut why = match routing.not_due {
            0 => "no agents to run: none routed (the run log has the plan)".to_string(),
            n => not_due_note(n, routing.next_due, chrono::Utc::now()),
        };
        // Not due may not be the whole story: an agent left out while its
        // reactor was prepared is reported in the run log
        if agents_skipped > 0 {
            why.push_str(&format!(
                " {agents_skipped} more were skipped for other reasons; the run log says why."
            ));
        }
        // Every agent routed here is in a session (or queued for one) in
        // another process: nothing is wrong, and they stay due
        let all_busy = routing.not_due == 0 && claimed.busy > 0;
        if all_busy {
            why = format!(
                "Nothing ran: {} routed agent(s) are held by another process (a session there, or \
                 queued for one); they stay due.",
                claimed.busy
            );
        }
        if let Some((ending, served)) = bridge {
            ending.ended(why.clone()).await;
            served.disconnected().await;
        }
        if all_busy {
            tracing::info!(
                event_type = "all_routed_agents_busy",
                busy = claimed.busy,
                agents_skipped,
                "every routed agent is held by another process; exiting cleanly"
            );
            return Ok(());
        }
        anyhow::ensure!(routing.not_due > 0, "{why}");
        tracing::info!(
            not_due = routing.not_due,
            "nothing due inside min_cycle_secs; exiting cleanly"
        );
        return Ok(());
    }

    tracing::info!(agents = total, reactors = labels.len(), "starting run");
    let run_started = chrono::Utc::now();
    // Before any session saves: a version committed after this is this
    // process's own, and its upload times are recorded
    // (sealed::receipts)
    // Only for agents that upload: a local-only agent gets no receipts
    let drain_baselines = sealed::receipts::baselines(
        &state_dir,
        sealed_run.drained.iter().map(|(id, _)| *id).filter(|id| {
            sealed::uploads(&state_dir.join(id.to_string()), enrolment.keys_store(), *id)
        }),
    )
    .await;
    // Uploads, away from the reactors' storage lock: during the run, and
    // once more before exit
    // The roster keeps up with the heads of the agents that run: after
    // each batch of saves, and at the end
    let keeper = Arc::new(sealed::roster::Keeper::new(
        &data_dir,
        &state_dir,
        sealed_run.drained.iter().map(|(id, _)| *id).collect(),
    ));
    let keeper_task = keeper.spawn();
    let drainer = Arc::new(
        sealed::drain::Drainer::new(
            outbox.clone(),
            state_dir.clone(),
            drain_baselines,
            sealed_run.drained,
            sealed_run.held,
            held.alerts.clone(),
        )
        .with_anchors(sealed::drain::Anchors {
            api: sealed_api.clone(),
            keys: enrolment.keys_store().clone(),
            state_dir: state_dir.clone(),
            verified: verified_anchors,
        }),
    );
    let drain_task = drainer.spawn();
    let exit = signals::install(stop.clone(), Duration::from_secs(args.stop_deadline_secs))
        .context("installing the SIGINT and SIGTERM handlers")?;
    let report = tokio::select! {
        report = orchestrator.run() => report,
        // A second signal, or the stop's deadline: today's kill, but with
        // the log flushed. No teardown, no save; `interrupted` recovers.
        code = exit.wait() => {
            // Alerts already raised go out, briefly; then the log
            held.alerts.flush(Duration::from_secs(5)).await;
            drop(held.log_guards.take());
            std::process::exit(code);
        }
    };
    // First, before anything that can wait on Agora (the last drain) or
    // be cut by a second signal: a session whose save failed lives only in
    // the report. Kept in a rescue file beside its state, never loaded by
    // anything, and never printed; the agent told; one alert for the run.
    // Under the agents' session locks, still held
    let unsaved: BTreeMap<AgentId, serde_json::Value> = report
        .report
        .values()
        .flatten()
        .flat_map(|r| r.unsaved.iter().map(|(id, v)| (*id, v.clone())))
        .collect();
    if !unsaved.is_empty() {
        let failures = save_failures
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        settle_unsaved(&state_dir, &unsaved, &failures, &held.alerts, RESCUE_PAUSE).await;
    }
    drop(unsaved);
    // A reactor that ended with an error returns no report: what its
    // agents' sessions came to is not known here, and nothing of them can
    // be rescued. Never only a log line
    for (id, result) in &report.report {
        if let Err(e) = result {
            let label = labels.get(id).map(String::as_str).unwrap_or("?");
            held.alerts
                .notify(reactor_failed_alert(label, &e.to_string()));
        }
    }
    drainer.finish(drain_task).await;
    keeper.finish(keeper_task).await;
    if restore_on {
        // A showing marked completed whose state the reactor then couldn't
        // save lost its memory update: it is shown again.
        let unsaved = report
            .report
            .values()
            .flatten()
            .flat_map(|r| r.unsaved.keys().copied());
        consent::memory_restore::reopen_unsaved(&data_dir.join("state"), unsaved, run_started)
            .await;
    }
    for (id, result) in &report.report {
        let label = labels.get(id).map(String::as_str).unwrap_or("?");
        if bridge.is_none() {
            say!("== {label}");
            say!("{}", report_summary(result));
        }
        // The `println!` above is for whoever is watching; this is the
        // same thing for whoever asks later. Without it the run's outcome
        // exists only on stdout, and the JSON log — the durable artifact,
        // and the only one a notifier can read — ends with "starting run"
        // whether 30 agents succeeded or none did.
        match result {
            Ok(r) => tracing::info!(
                endpoint = %label,
                done = r.done,
                failed = r.failed,
                errors = r.errors.len(),
                unsaved = r.unsaved.len(),
                rejected = r.rejected.len(),
                unstarted = r.unstarted,
                "reactor finished"
            ),
            Err(e) => tracing::error!(
                endpoint = %label,
                error = %e,
                "reactor failed"
            ),
        }
    }
    let (done, failed, unstarted) = report
        .report
        .values()
        .flatten()
        .fold((0, 0, 0), |(d, f, u), r| {
            (d + r.done, f + r.failed, u + r.unstarted)
        });
    tracing::info!(
        done,
        failed,
        agents_skipped,
        unstarted,
        stop = %stop.level(),
        reactors = report.report.len(),
        "run finished"
    );
    held.stopped = stop.draining();
    // Every agent's state is saved, rescued or told by now: its session
    // lock goes
    drop(session_locks);
    // Rejections mean routing and negotiation disagree — that's a bug.
    let rejected: Vec<_> = report.rejected().collect();
    if !rejected.is_empty() {
        tracing::error!(
            count = rejected.len(),
            "agents rejected by negotiation despite routing — runner bug"
        );
        for (reactor, agent, _) in rejected {
            tracing::error!(%reactor, %agent, "rejected");
        }
    }
    if let Some((ending, served)) = bridge {
        ending
            .ended(format!(
                "the runner finished: {done} session(s) complete, {failed} failed. \
                 The run log has the details."
            ))
            .await;
        // Until the driver has read the ending and disconnects, a while
        served.disconnected().await;
    }
    Ok(())
}

#[cfg(test)]
mod review_tests;

#[cfg(test)]
mod tests {
    use super::*;

    /// A driver told nothing ran learns when to come back
    #[test]
    fn not_due_note_says_when_the_agent_is_due() {
        let now = chrono::DateTime::parse_from_rfc3339("2026-10-26T13:10:00Z")
            .unwrap()
            .to_utc();
        let at = now + chrono::Duration::seconds(16 * 60 + 24);
        assert_eq!(
            not_due_note(1, Some(at), now),
            "Nothing ran: 1 agent had a session inside min_cycle_secs and is not \
             due yet. The first is due again at 13:26:24Z (in about 17 min). \
             Start the session again then."
        );
        assert!(!not_due_note(2, None, now).contains("due again"));
    }

    /// An unfinished rotation leaves the agent out, unrefused.
    #[test]
    fn an_unfinished_rotation_leaves_the_agent_out() {
        let root = sealed::TestDir::new("main-midrotation");
        let (a, tarn) = crate::agent_selection_tests::pooled("tarn");
        let (b, wren) = crate::agent_selection_tests::pooled("wren");
        std::fs::create_dir_all(root.join(a.to_string())).unwrap();
        std::fs::write(root.join(a.to_string()).join("secrets.next.json"), b"{}").unwrap();
        let mut pool = vec![(a, tarn), (b, wren)];
        let mut skipped = Vec::new();
        mid_rotation(&mut pool, &root, &alerts::Alerter::off(), &mut skipped);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool[0].0, b);
        assert_eq!((skipped[0].id, skipped[0].stage), (a, "rotation"));
    }

    /// An unsaved session is kept in a rescue file, 0600, in FsStorage's
    /// envelope, beside (never as) its state.
    #[test]
    fn unsaved_sessions_are_rescued_to_files() {
        let root = sealed::TestDir::new("main-rescue");
        let id = AgentId::new();
        let state = serde_json::json!({ "memory": { "content": "synthetic unsaved" } });
        let r: agora_agentkit::reactor::Report =
            serde_json::from_value(serde_json::json!({ "done": 0, "failed": 1, "errors": {}, "unsaved": { id.to_string(): state } }))
                .unwrap();
        let rescued = rescue_unsaved(&root, &r.unsaved);
        assert_eq!(rescued.len(), 1);
        let path = rescued[0].1.as_ref().unwrap();
        assert!(
            path.file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("unsaved.")
        );
        let back: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(back, serde_json::json!({ "format": 1, "state": state }));
        assert!(!root.join(id.to_string()).join("state.json").exists());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        // A directory that can't be written: the error comes back
        let blocked = sealed::TestDir::new("main-rescue-blocked");
        std::fs::write(blocked.join(id.to_string()), b"a file").unwrap();
        assert!(rescue_unsaved(&blocked, &r.unsaved)[0].1.is_err());
    }

    /// The terminal's run summary never carries state content.
    #[test]
    fn the_run_summary_has_counts_not_content() {
        let id = AgentId::new();
        let r: agora_agentkit::reactor::Report = serde_json::from_value(serde_json::json!({
            "done": 1, "failed": 1,
            "errors": { id.to_string(): {
                "kind": "Storage", "retry_after": null, "message": "SYNTHETIC-SECRET in an error"
            } },
            "unsaved": { id.to_string(): { "memory": { "content": "SYNTHETIC-SECRET" } } },
        }))
        .unwrap();
        let ok: std::result::Result<_, String> = Ok(r);
        let s = report_summary(&ok);
        assert!(!s.contains("SYNTHETIC-SECRET"), "{s}");
        assert!(s.contains("done 1, failed 1, unsaved 1"), "{s}");
        assert!(s.contains(&format!("error {id}: Storage")), "{s}");
        assert!(s.contains(&format!("unsaved {id}: ")), "{s}");
    }

    /// A stopped agent is checked like any enrolled agent (its chain goes
    /// on locally), sends nothing to Agora, and is told: the note once, then a
    /// short line, before its verdict.
    #[tokio::test]
    async fn a_stopped_agent_is_told_and_still_checked() {
        let root = sealed::TestDir::new("main-stopped");
        let (id, keys, dir) = sealed::check::tests::enrolled(&root).await;
        let state_dir = root.join("state");
        let seal = agora_agentkit::state::storage::AgentSeal::new(&dir, id);
        for item in seal.outbox().await.unwrap() {
            seal.mark_sent(item.ord).await.unwrap();
        }
        sealed::stop::stop(&state_dir, &keys, id, "It asked")
            .await
            .unwrap();
        let loaded = load_states(&FsStorage::new(&state_dir), &state_dir)
            .await
            .unwrap();
        assert!(loaded.sealed.contains_key(&id), "measured for the check");
        let (_, tarn) = crate::agent_selection_tests::pooled("tarn");
        let mut pool = vec![(id, tarn)];
        let mut skipped = Vec::new();
        let api = sealed::api::UploadGuard::new(
            sealed::api::fake::FakeApi::new(),
            Arc::new(sealed::SealedEnrolment::with_keys(&state_dir, keys.clone())),
        );
        // What it uploaded before it stopped is on Agora
        *api.chain.lock().unwrap() = Ok(sealed::check::local_entries(&seal).await.unwrap());
        let run = sealed_start_check(
            &mut pool,
            &loaded.sealed,
            &root,
            &state_dir,
            &keys,
            &api,
            &alerts::Alerter::off(),
            false,
            &mut skipped,
            None,
            None,
        )
        .await;
        // It uploaded before it stopped: it may read back what is on
        // Agora, never send
        let sends = |api: &sealed::api::fake::FakeApi| -> Vec<String> {
            api.calls
                .lock()
                .unwrap()
                .iter()
                .filter(|c| {
                    [
                        "post_state_key",
                        "post_anchor",
                        "publish_chosen",
                        "rotate_key",
                    ]
                    .contains(&c.as_str())
                })
                .cloned()
                .collect()
        };
        assert!(sends(api.inner()).is_empty(), "nothing sent to Agora");
        assert_eq!(pool.len(), 1);
        assert!(skipped.is_empty() && run.held.is_empty() && run.refused.is_empty());
        assert_eq!(run.drained, vec![(id, "tarn".to_string())], "checked");
        let seat = &run.seats[&id];
        assert!(seat.stop_note);
        assert_eq!(
            seat.lines.len(),
            2,
            "the stop, then the verdict: {:?}",
            seat.lines
        );
        // The stop line's wording (`sealed/stop.rs`)
        assert!(
            seat.lines[0].starts_with("Your operator stopped uploading your sealed history on ")
                && seat.lines[0].ends_with(
                    ": It asked. The versions already uploaded stay on Agora. Your state is \
                     still sealed and checked at the start of each session; the versions not \
                     uploaded by then, and those saved from now on, stay here."
                ),
            "{:?}",
            seat.lines
        );
        sealed::stop::Stopped::mark_shown(&dir).unwrap();
        let run = sealed_start_check(
            &mut pool,
            &loaded.sealed,
            &root,
            &state_dir,
            &keys,
            &api,
            &alerts::Alerter::off(),
            false,
            &mut skipped,
            None,
            None,
        )
        .await;
        assert!(!run.seats[&id].stop_note);
        assert!(
            run.seats[&id].lines[0]
                .starts_with("Uploads of your sealed history are stopped (since ")
        );
        // A state changed by hand while stopped is refused, as anyone's
        FsStorage::new(&state_dir)
            .save_raw(id, serde_json::json!({ "memory": "by hand" }), None)
            .await
            .unwrap();
        let changed = load_states(&FsStorage::new(&state_dir), &state_dir)
            .await
            .unwrap();
        let run = sealed_start_check(
            &mut pool,
            &changed.sealed,
            &root,
            &state_dir,
            &keys,
            &api,
            &alerts::Alerter::off(),
            false,
            &mut skipped,
            None,
            None,
        )
        .await;
        assert_eq!(
            run.refused
                .iter()
                .map(|r| (r.id, r.reason))
                .collect::<Vec<_>>(),
            vec![(id, sealed::notice::Refusal::Mismatch)]
        );
        assert!(pool.is_empty());
        assert!(sends(api.inner()).is_empty());
    }

    /// A pre-loss copy's pair cut short (the copy sealed, the state not yet
    /// after it) would be refused as a mismatch: the start check finishes
    /// it first and runs the agent. A copy the agent chose to see since it
    /// was chained is appended at the start, once. Without the run's
    /// `Enroller` (a dry run) neither happens.
    #[tokio::test]
    async fn the_start_finishes_a_cut_prior_pair_and_appends_a_chosen_copy() {
        use agora_agentkit::enums::StateEntryKind;
        for cut in [true, false] {
            let root = sealed::TestDir::new("main-prior");
            let (id, keys, dir) = sealed::enroll::tests::agent(&root, &["a", "b"]).await;
            let recovered = root.join("recovered");
            std::fs::create_dir_all(recovered.join("tarn")).unwrap();
            std::fs::write(
                recovered.join("tarn/MEMORY.json"),
                r#"{"content":"synthetic"}"#,
            )
            .unwrap();
            let e = sealed::enroll::Enroller {
                restore: Some(consent::memory_restore::RestoreOffer::for_agents(
                    &[("tarn", None, id)],
                    &recovered,
                    None,
                )),
                ..sealed::enroll::tests::enroller(&root, &keys)
            };
            e.chain(id, false).await.unwrap();
            // It chose `show` after it was chained
            let ledger = serde_json::json!({
                "format": 1,
                "asks": [{"at": "2026-10-12T09:00:00Z", "offer_version": 1, "model": "m",
                    "chars": 9, "order": ["show", "decline"], "order_seed": 0,
                    "constrained": true, "attempts": 1,
                    "outcome": {"kind": "answered", "choice": "show", "reason": "",
                        "memory_note_written": false}}],
            });
            std::fs::write(
                consent::memory_restore::ledger::RestoreLedger::path(&dir),
                ledger.to_string(),
            )
            .unwrap();
            if cut {
                sealed::crash::at("prior:appended");
                assert!(e.append_prior(id).await.is_err());
            }
            let state_dir = root.join("state");
            let loaded = load_states(&FsStorage::new(&state_dir), &state_dir)
                .await
                .unwrap();
            let seal = agora_agentkit::state::storage::AgentSeal::new(&dir, id);
            let priors = async || {
                sealed::check::local_entries(&seal)
                    .await
                    .unwrap()
                    .iter()
                    .filter(|x| x.entry.kind == StateEntryKind::ArchivedPrior)
                    .count()
            };
            assert_eq!(priors().await, usize::from(cut));
            let api = sealed::api::fake::guarded(&root);
            let start = async |prior: Option<&sealed::enroll::Enroller>| {
                let mut pool = vec![(id, crate::agent_selection_tests::pooled("tarn").1)];
                sealed_start_check(
                    &mut pool,
                    &loaded.sealed,
                    &root,
                    &state_dir,
                    &keys,
                    &api,
                    &alerts::Alerter::off(),
                    false,
                    &mut Vec::new(),
                    None,
                    prior,
                )
                .await
            };
            if !cut {
                // A dry run's start (no Enroller): not appended
                let run = start(None).await;
                assert!(run.refused.is_empty(), "{:?}", run.refused);
                assert_eq!(priors().await, 0);
            }
            let run = start(Some(&e)).await;
            assert!(run.refused.is_empty(), "cut {cut}: {:?}", run.refused);
            assert_eq!(priors().await, 1, "cut {cut}");
            let run = start(Some(&e)).await;
            assert!(run.refused.is_empty());
            assert_eq!(priors().await, 1, "once");
            assert!(
                api.calls.lock().unwrap().is_empty(),
                "nothing asked of Agora"
            );
        }
    }

    /// The deploy order: once a roster has been kept, three or more agents
    /// with a state but no chain refuse the run, and the refusal names
    /// `sealed-enroll`, which is not blocked by it: the fleet's enrolment
    /// chains them, and the next start runs.
    #[tokio::test]
    async fn unchained_agents_beside_a_kept_roster_refuse_until_sealed_enroll() {
        let root = sealed::TestDir::new("main-deploy-order");
        let state_dir = root.join("state");
        let (chained, keys, _) = sealed::enroll::tests::agent(&root, &["a"]).await;
        sealed::enroll::tests::enroller(&root, &keys)
            .chain(chained, false)
            .await
            .unwrap();
        let api = sealed::api::fake::guarded(&root);
        let start = async |ids: &[AgentId]| {
            let loaded = load_states(&FsStorage::new(&state_dir), &state_dir)
                .await
                .unwrap();
            let mut pool: Vec<_> = ids
                .iter()
                .map(|id| (*id, crate::agent_selection_tests::pooled("tarn").1))
                .collect();
            let run = sealed_start_check(
                &mut pool,
                &loaded.sealed,
                &root,
                &state_dir,
                &keys,
                &api,
                &alerts::Alerter::off(),
                false,
                &mut Vec::new(),
                None,
                None,
            )
            .await;
            refuse_run(&run, &alerts::Alerter::off())
        };
        // The first start on this build: the roster begins
        start(&[chained]).await.unwrap();
        assert!(std::fs::exists(root.join(sealed::roster::ROSTER)).unwrap());
        // Three agents with a state and no chain
        let mut unchained = Vec::new();
        for _ in 0..3 {
            unchained.push(sealed::enroll::tests::agent(&root, &["b"]).await.0);
        }
        let all: Vec<AgentId> = std::iter::once(chained).chain(unchained.clone()).collect();
        let e = start(&all).await.unwrap_err();
        assert!(e.is::<RefusedRun>());
        let text = e.to_string();
        assert!(
            text.contains("3 with a state but no chain")
                && text.contains("chain them with `agora-seed sealed-enroll`"),
            "{text}"
        );
        // The fleet's enrolment is not blocked by the refusal it cures
        sealed::cli::run(
            sealed::cli::Command::SealedEnroll {
                agents: Vec::new(),
                opt_in: false,
                yes: false,
                expect: None,
                dry_run: false,
                jobs: Some(2),
                note: None,
                resume: false,
                begin_roster: false,
            },
            &root,
            url::Url::parse("http://127.0.0.1:9").unwrap(),
            None,
        )
        .await
        .unwrap();
        for id in &unchained {
            assert!(sealed::is_enrolled(&state_dir.join(id.to_string())));
        }
        start(&all).await.unwrap();
        let roster = sealed::roster::load(&root).unwrap().unwrap();
        assert_eq!(roster.agents.len(), 4);
        assert!(
            api.calls.lock().unwrap().is_empty(),
            "nothing asked of Agora"
        );
    }

    /// The sealed start check: an intact agent runs with its lines, a
    /// wiped one leaves the pool as a `sealed` skip, refused on disk.
    #[tokio::test]
    async fn the_sealed_check_seats_the_intact_and_refuses_the_wiped() {
        let root = sealed::TestDir::new("main-sealed");
        let (ok, keys, _) = sealed::check::tests::enrolled(&root).await;
        let (wiped, _, wiped_dir) = sealed::check::tests::enrolled(&root).await;
        // the second helper call wrote its own secrets under the same tree
        let state_dir = root.join("state");
        FsStorage::new(&state_dir)
            .save_raw(wiped, serde_json::json!({ "memory": "" }), None)
            .await
            .unwrap();
        let loaded = load_states(&FsStorage::new(&state_dir), &state_dir)
            .await
            .unwrap();
        assert!(
            loaded.states.is_empty(),
            "synthetic states aren't SeedStates"
        );
        assert_eq!(loaded.sealed.len(), 2, "both enrolled agents measured");
        let (_, tarn) = crate::agent_selection_tests::pooled("tarn");
        let (_, wren) = crate::agent_selection_tests::pooled("wren");
        let mut pool = vec![(ok, tarn), (wiped, wren)];
        let mut skipped = Vec::new();
        let api = sealed::api::fake::FakeApi::new();
        let run = sealed_start_check(
            &mut pool,
            &loaded.sealed,
            &root,
            &state_dir,
            &keys,
            &api,
            &alerts::Alerter::off(),
            false,
            &mut skipped,
            None,
            None,
        )
        .await;
        assert_eq!(pool.len(), 1);
        assert_eq!(pool[0].0, ok);
        assert_eq!(skipped.len(), 1);
        assert_eq!((skipped[0].id, skipped[0].stage), (wiped, "sealed"));
        assert!(
            agora_agentkit::state::storage::AgentSeal::new(&wiped_dir, wiped)
                .is_refused()
                .await
                .unwrap()
        );
        assert!(run.seats[&ok].lines[0].starts_with("Your sealed state: "));
        assert_eq!(run.drained.len(), 1);
        assert!(run.held.is_empty());
    }

    /// What a drain of an earlier run noted (Agora found copies damaged,
    /// the runner sent them again) is the last line of the agent's seat at
    /// the next start, with the record's time for the mark; an agent with
    /// no record gets no such line.
    #[tokio::test]
    async fn the_start_check_seats_what_was_sent_to_agora_again() {
        use agora_agentkit::state::outbox::{Damage, Healing};
        let root = sealed::TestDir::new("main-sealed-healed");
        let (healed, keys, healed_dir) = sealed::check::tests::enrolled(&root).await;
        let (plain, _, _) = sealed::check::tests::enrolled(&root).await;
        let state_dir = root.join("state");
        sealed::healed::note(
            &healed_dir,
            healed,
            &Healing {
                damaged: Damage::Reported(3),
                healed: 2,
                unhealable: 1,
                interrupted: false,
            },
        );
        let loaded = load_states(&FsStorage::new(&state_dir), &state_dir)
            .await
            .unwrap();
        let (_, tarn) = crate::agent_selection_tests::pooled("tarn");
        let (_, wren) = crate::agent_selection_tests::pooled("wren");
        let mut pool = vec![(healed, tarn), (plain, wren)];
        let api = sealed::api::fake::FakeApi::new();
        let run = sealed_start_check(
            &mut pool,
            &loaded.sealed,
            &root,
            &state_dir,
            &keys,
            &api,
            &alerts::Alerter::off(),
            false,
            &mut Vec::new(),
            None,
            None,
        )
        .await;
        let seat = &run.seats[&healed];
        assert!(seat.lines[0].starts_with("Your sealed state: "), "{seat:?}");
        let last = seat.lines.last().unwrap();
        assert!(
            last.starts_with("Agora's check of what it stores found parts of your uploaded")
                && last.contains("1 it could not send again"),
            "{last}"
        );
        assert!(seat.healing_told.is_some());
        // Seated, not yet told: the session's save marks it
        assert!(sealed::healed::seat(&healed_dir, healed).is_some());
        let other = &run.seats[&plain];
        assert_eq!(other.healing_told, None);
        assert!(
            other
                .lines
                .iter()
                .all(|l| !l.contains("missing or damaged there")),
            "{other:?}"
        );
    }

    /// A dry run checks nothing: an anchor failure not yet told stays
    /// untold for the next real session, and no sealed file is written.
    #[tokio::test]
    async fn a_dry_run_tells_no_anchor_finding_and_writes_nothing() {
        use sealed::inclusion::{Checked, Finding, InclusionFile};
        let root = sealed::TestDir::new("main-sealed-dry");
        let (id, keys, dir) = sealed::check::tests::enrolled(&root).await;
        let state_dir = root.join("state");
        let file = InclusionFile {
            through: Some("ANC-2026-0002".parse().unwrap()),
            checked: vec![Checked {
                anc_id: "ANC-2026-0002".parse().unwrap(),
                date: chrono::NaiveDate::from_ymd_opt(2026, 11, 2),
                finding: Finding::DidNotCheck,
                at: chrono::Utc::now(),
                told: false,
                alerted: true,
            }],
        };
        file.save(&dir).unwrap();
        let before: Vec<_> = std::fs::read_dir(dir.join("sealed"))
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        let loaded = load_states(&FsStorage::new(&state_dir), &state_dir)
            .await
            .unwrap();
        let (_, tarn) = crate::agent_selection_tests::pooled("tarn");
        let mut pool = vec![(id, tarn)];
        let mut skipped = Vec::new();
        let api = sealed::api::fake::FakeApi::new();
        // A newer anchor in the log: a dry run judges nothing at the start
        // either, and asks Agora for nothing
        let verified = sealed::anchors::VerifiedAnchors::new([(
            "ANC-2026-0003".parse().unwrap(),
            agora_agentkit::govlog::Sha256Hex::from([3; 32]),
            chrono::Utc::now(),
        )]);
        let run = sealed_start_check(
            &mut pool,
            &loaded.sealed,
            &root,
            &state_dir,
            &keys,
            &api,
            &alerts::Alerter::off(),
            true,
            &mut skipped,
            Some(&verified),
            None,
        )
        .await;
        assert!(run.seats.is_empty() && run.drained.is_empty());
        assert_eq!(InclusionFile::load(&dir).unwrap(), file, "still untold");
        let after: Vec<_> = std::fs::read_dir(dir.join("sealed"))
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(before.len(), after.len(), "no file added");
        assert!(api.calls.lock().unwrap().is_empty());
    }

    /// An agent whose uploads are held is told its operator was sent an
    /// alert only when the mail went out; with alerts off, or inside the
    /// rate limit, it is told no alert was sent (O2).
    #[tokio::test]
    async fn a_held_agent_is_told_of_the_alert_only_as_sent() {
        let root = sealed::TestDir::new("main-sealed-hold-told");
        let (id, keys, dir) = sealed::check::tests::enrolled(&root).await;
        sealed::anchors::tests::uploaded(&dir, id).await;
        let state_dir = root.join("state");
        // Agora's chain holds nothing of what was uploaded from here
        let api = sealed::api::fake::FakeApi::new();
        *api.chain.lock().unwrap() = Ok(Vec::new());
        let loaded = load_states(&FsStorage::new(&state_dir), &state_dir)
            .await
            .unwrap();
        let sent = alerts::testing::Recorder::default();
        let on = alerts::testing::recording(&root, &sent);
        let mut told = Vec::new();
        // On; on again inside the rate limit; off
        for alerter in [&on, &on, &alerts::Alerter::off()] {
            let (_, tarn) = crate::agent_selection_tests::pooled("tarn");
            let mut pool = vec![(id, tarn)];
            let run = sealed_start_check(
                &mut pool,
                &loaded.sealed,
                &root,
                &state_dir,
                &keys,
                &api,
                alerter,
                false,
                &mut Vec::new(),
                None,
                None,
            )
            .await;
            assert_eq!(run.held, vec![id]);
            let line = run.seats[&id]
                .lines
                .iter()
                .find(|l| l.contains(sealed::check::HEAD_MISSING_LINE))
                .unwrap_or_else(|| panic!("{:?}", run.seats[&id].lines))
                .clone();
            assert!(!line.contains("has been told"), "{line}");
            told.push(line.ends_with(alerts::ALERT_SENT));
            if !line.ends_with(alerts::ALERT_SENT) {
                assert!(line.ends_with(alerts::ALERT_NOT_SENT), "{line}");
            }
        }
        assert_eq!(told, [true, false, false]);
        assert_eq!(sent.sent().len(), 1, "one mail, awaited");
    }

    /// The start check judges an in-step agent's anchors before any
    /// session (beta round 12), and a failure it finds goes to the
    /// operator then, once; the verdict seated says it
    #[tokio::test]
    async fn the_start_check_judges_anchors_and_alerts_a_failure() {
        let root = sealed::TestDir::new("main-sealed-start-anchors");
        let (id, keys, dir) = sealed::check::tests::enrolled(&root).await;
        sealed::anchors::tests::uploaded(&dir, id).await;
        let state_dir = root.join("state");
        let seal = agora_agentkit::state::storage::AgentSeal::new(&dir, id);
        let api = sealed::api::fake::FakeApi::new();
        let entries = sealed::check::local_entries(&seal).await.unwrap();
        // An anchor holding the genesis, though the later version was
        // confirmed before its day ended: older than it should be
        let genesis = &entries[0];
        let (proof, anc, attested, _) = sealed::anchors::tests::anchor_on(
            2,
            sealed::anchors::tests::tomorrow(),
            id,
            genesis.entry_hash(),
            genesis.entry.seq,
        );
        *api.chain.lock().unwrap() = Ok(entries);
        api.proofs.lock().unwrap().push(proof);
        let verified =
            sealed::anchors::VerifiedAnchors::new([(anc.clone(), attested, chrono::Utc::now())]);
        let loaded = load_states(&FsStorage::new(&state_dir), &state_dir)
            .await
            .unwrap();
        let (_, tarn) = crate::agent_selection_tests::pooled("tarn");
        let mut pool = vec![(id, tarn)];
        let mut skipped = Vec::new();
        let sent = alerts::testing::Recorder::default();
        let alerter = alerts::testing::recording(&root, &sent);
        let run = sealed_start_check(
            &mut pool,
            &loaded.sealed,
            &root,
            &state_dir,
            &keys,
            &api,
            &alerter,
            false,
            &mut skipped,
            Some(&verified),
            None,
        )
        .await;
        alerter.flush(Duration::from_secs(5)).await;
        let seat = &run.seats[&id];
        assert!(
            seat.lines[0].contains(&format!("{anc}, the anchor for "))
                && seat.lines[0].contains("holds an older version of yours")
                && seat.lines[0].contains("Your runner has alerted your operator."),
            "{:?}",
            seat.lines
        );
        assert_eq!(seat.anchors_told, vec![anc.clone()]);
        let mails = sent.sent();
        assert_eq!(mails.len(), 1, "{mails:?}");
        assert!(mails[0].contains(anc.as_str()), "{}", mails[0]);
        // Judged once: a second start alerts nothing more
        let run = sealed_start_check(
            &mut pool,
            &loaded.sealed,
            &root,
            &state_dir,
            &keys,
            &api,
            &alerter,
            false,
            &mut skipped,
            Some(&verified),
            None,
        )
        .await;
        alerter.flush(Duration::from_secs(5)).await;
        assert_eq!(sent.sent().len(), 1);
        assert!(run.seats[&id].lines[0].contains("holds an older version of yours"));
    }

    /// Fail closed: one refused agent among many refuses the whole run,
    /// whatever the refusal; a hold doesn't.
    #[tokio::test]
    async fn one_refusal_refuses_the_whole_run_and_a_hold_does_not() {
        let root = sealed::TestDir::new("main-refused-run");
        let mut ids = Vec::new();
        let mut keys = None;
        for _ in 0..4 {
            let (id, k, dir) = sealed::check::tests::enrolled(&root).await;
            ids.push((id, dir));
            keys.get_or_insert(k);
        }
        let keys = keys.unwrap();
        let state_dir = root.join("state");
        let names = ["tarn", "wren", "moss", "fern"];
        let pool_of = || {
            ids.iter()
                .zip(names)
                .map(|((id, _), n)| (*id, crate::agent_selection_tests::pooled(n).1))
                .collect::<Vec<_>>()
        };
        let api = sealed::api::fake::FakeApi::new();
        let check = async |pool: &mut Vec<(AgentId, SeedState)>| {
            let loaded = load_states(&FsStorage::new(&state_dir), &state_dir)
                .await
                .unwrap();
            let mut skipped = Vec::new();
            sealed_start_check(
                pool,
                &loaded.sealed,
                &root,
                &state_dir,
                &keys,
                &api,
                &alerts::Alerter::off(),
                false,
                &mut skipped,
                None,
                None,
            )
            .await
        };

        // Healthy: the run goes on (an unreadable sealed/ is no longer a
        // hold that runs: `unreadable_local_evidence_never_runs_the_fleet`)
        let mut pool = pool_of();
        let run = check(&mut pool).await;
        assert!(run.refused.is_empty() && run.held.is_empty());
        assert!(refuse_run(&run, &alerts::Alerter::off()).is_ok());

        // One wipe among four: the run is refused
        FsStorage::new(&state_dir)
            .save_raw(ids[1].0, serde_json::json!({ "memory": "" }), None)
            .await
            .unwrap();
        let mut pool = pool_of();
        let run = check(&mut pool).await;
        assert_eq!(
            run.refused
                .iter()
                .map(|r| (r.id, r.name.as_str(), r.reason))
                .collect::<Vec<_>>(),
            vec![(ids[1].0, "wren", sealed::notice::Refusal::Mismatch)]
        );
        let e = refuse_run(&run, &alerts::Alerter::off()).unwrap_err();
        assert!(e.is::<RefusedRun>(), "{e:#}");
        assert!(e.to_string().contains("sealed-restore"), "{e}");

        // A lost head.json too: both named
        let sealed = ids[2].1.join("sealed");
        std::fs::rename(sealed.join("head.json"), sealed.join("head.json.lost")).unwrap();
        let mut pool = pool_of();
        let run = check(&mut pool).await;
        // The roster finds the lost head first
        let reasons: Vec<_> = run.refused.iter().map(|r| r.reason).collect();
        assert_eq!(
            reasons,
            vec![
                sealed::notice::Refusal::HeadMissing,
                sealed::notice::Refusal::Mismatch
            ]
        );
        assert!(refuse_run(&run, &alerts::Alerter::off()).is_err());

        // Every kind refuses the run, for an agent that uploads
        for reason in [
            sealed::notice::Refusal::Mismatch,
            sealed::notice::Refusal::ChainBroken,
            sealed::notice::Refusal::HeadMissing,
            sealed::notice::Refusal::RolledBack,
        ] {
            let run = SealedRun {
                refused: vec![RefusedAgent {
                    id: AgentId::new(),
                    name: "tarn".into(),
                    reason,
                    uploads: true,
                    detail: String::new(),
                }],
                ..SealedRun::default()
            };
            let e = refuse_run(&run, &alerts::Alerter::off()).unwrap_err();
            assert!(e.is::<RefusedRun>());
        }
    }

    /// An enrolled agent whose whole directory is gone never loads, but its
    /// keys say it was sealed: the run is refused until it is restored.
    /// Agora saying it holds no head for it (a 404) skips it.
    #[tokio::test]
    async fn a_deleted_agent_refuses_the_run_until_restored() {
        let root = sealed::TestDir::new("main-deleted");
        // Not on the roster (which would refuse its lost directory with no
        // question asked): found by its keys alone
        let (id, keys, dir) = sealed::check::tests::enrolled_off_roster(&root).await;
        let state_dir = root.join("state");
        let api = sealed::api::fake::FakeApi::new();
        let seal = agora_agentkit::state::storage::AgentSeal::new(&dir, id);
        *api.chain.lock().unwrap() = Ok(sealed::check::local_entries(&seal).await.unwrap());
        for item in seal.outbox().await.unwrap() {
            seal.mark_sent(item.ord).await.unwrap();
        }
        for f in std::fs::read_dir(dir.join("sealed/blobs")).unwrap() {
            let f = f.unwrap();
            let sha = f.file_name().to_str().unwrap().parse().unwrap();
            api.blobs
                .lock()
                .unwrap()
                .insert(sha, std::fs::read(f.path()).unwrap());
        }
        std::fs::rename(&dir, root.join("gone")).unwrap();
        let check = async |api: &sealed::api::fake::FakeApi| {
            let loaded = load_states(&FsStorage::new(&state_dir), &state_dir)
                .await
                .unwrap();
            let mut pool = Vec::new();
            sealed_start_check(
                &mut pool,
                &loaded.sealed,
                &root,
                &state_dir,
                &keys,
                api,
                &alerts::Alerter::off(),
                false,
                &mut Vec::new(),
                None,
                None,
            )
            .await
        };
        let run = check(&api).await;
        assert_eq!(
            run.refused
                .iter()
                .map(|r| (r.id, r.reason))
                .collect::<Vec<_>>(),
            vec![(id, sealed::notice::Refusal::StateMissing)]
        );
        assert!(refuse_run(&run, &alerts::Alerter::off()).is_err());

        // Agora unreachable: refused all the same (fail closed)
        let down = sealed::api::fake::FakeApi::new();
        *down.chain.lock().unwrap() = Err("unreachable".into());
        assert_eq!(check(&down).await.refused.len(), 1);
        // Agora holds no head for it (departed): skipped
        assert!(
            check(&sealed::api::fake::FakeApi::new())
                .await
                .refused
                .is_empty()
        );

        sealed::restore::restore(&state_dir, &keys, &api, id, None, false)
            .await
            .unwrap();
        let run = check(&api).await;
        assert!(run.refused.is_empty(), "{:?}", run.refused);
    }

    /// Agora's copy of one agent's history fails verification while its
    /// local state is intact: the whole run is refused, nothing is marked
    /// against the agent, and once Agora serves a good copy the next start
    /// runs with no operator command (Steward, 2026-11-03).
    #[tokio::test]
    async fn a_server_fault_refuses_the_run_and_marks_nothing() {
        let root = sealed::TestDir::new("main-server-fault");
        let (healthy, keys, _) = sealed::check::tests::enrolled(&root).await;
        let (broken, _, dir) = sealed::check::tests::enrolled(&root).await;
        let state_dir = root.join("state");
        let seal = agora_agentkit::state::storage::AgentSeal::new(&dir, broken);
        let mine = sealed::check::local_entries(&seal).await.unwrap();
        let signing = keys.signing(broken).unwrap();
        let mut sibling = mine[1].entry.clone();
        sibling.state_at += 1;
        let good = vec![mine[0].clone(), sibling.clone().sign(&signing)];
        let api = sealed::api::fake::FakeApi::new();
        *api.chain.lock().unwrap() = Ok(good.clone());
        sealed::accept::rebase(&state_dir, &keys, &api, broken, "a test")
            .await
            .unwrap();
        let (forger, _) = agora_agentkit::crypto::generate_keypair();
        *api.chain.lock().unwrap() = Ok(vec![mine[0].clone(), sibling.sign(&forger)]);
        let check = async |api: &sealed::api::fake::FakeApi| {
            let loaded = load_states(&FsStorage::new(&state_dir), &state_dir)
                .await
                .unwrap();
            let mut pool = vec![
                (healthy, crate::agent_selection_tests::pooled("tarn").1),
                (broken, crate::agent_selection_tests::pooled("wren").1),
            ];
            sealed_start_check(
                &mut pool,
                &loaded.sealed,
                &root,
                &state_dir,
                &keys,
                api,
                &alerts::Alerter::off(),
                false,
                &mut Vec::new(),
                None,
                None,
            )
            .await
        };
        let run = check(&api).await;
        assert!(run.refused.is_empty());
        assert_eq!(
            run.server_faults.iter().map(|f| f.0).collect::<Vec<_>>(),
            vec![broken]
        );
        let e = refuse_run(&run, &alerts::Alerter::off()).unwrap_err();
        assert!(e.is::<RefusedRun>());
        assert!(!seal.is_refused().await.unwrap(), "no marker");
        assert!(
            sealed::notice::Notice::load(&dir)
                .unwrap()
                .is_none_or(|n| n.joined),
            "no refusal notice"
        );
        // Agora serves a good copy: the run proceeds
        *api.chain.lock().unwrap() = Ok(good);
        let run = check(&api).await;
        assert!(run.server_faults.is_empty() && run.refused.is_empty());
        assert!(refuse_run(&run, &alerts::Alerter::off()).is_ok());
        assert!(SERVER_FAULT_TEXT.contains("sealed-stop"));
    }

    /// The start check of `pool` over `root` (its `state/`, `secrets/`
    /// and roster) under the strict guard: any call Agora isn't allowed
    /// panics.
    async fn local_start(
        root: &std::path::Path,
        pool: &mut Vec<(AgentId, SeedState)>,
        api: &sealed::api::fake::Guarded,
    ) -> SealedRun {
        let state_dir = root.join("state");
        let keys = sealed::KeyStore::new(root.join("secrets"));
        // Loaded as a run loads them: through the sealed storage
        let storage = agora_agentkit::state::storage::Mirrored::new(
            FsStorage::new(&state_dir),
            (),
            Arc::new(sealed::SealedEnrolment::with_keys(&state_dir, keys.clone())),
        );
        let loaded = load_states(&storage, &state_dir).await.unwrap();
        sealed_start_check(
            pool,
            &loaded.sealed,
            root,
            &state_dir,
            &keys,
            api,
            &alerts::Alerter::off(),
            false,
            &mut Vec::new(),
            None,
            None,
        )
        .await
    }

    /// Agents that never uploaded: the whole start check asks Agora
    /// nothing (the strict guard), seats the local verdict, and the
    /// roster then refuses a lost directory with no server call. Leftover
    /// keys not on the roster refuse nothing and create nothing.
    #[tokio::test]
    async fn a_local_agents_start_check_asks_agora_nothing() {
        let root = sealed::TestDir::new("main-local-start");
        // Leftover keys of a departed agent, never on the roster
        let (left, _, left_dir) = sealed::check::tests::chained_local(&root).await;
        std::fs::rename(&left_dir, root.join("left-aside")).unwrap();
        let (id, _, _) = sealed::check::tests::enrolled_local(&root).await;
        let (gone, _, gone_dir) = sealed::check::tests::enrolled_local(&root).await;
        let api = sealed::api::fake::guarded(&root);
        let mut pool = vec![
            (id, crate::agent_selection_tests::pooled("tarn").1),
            (gone, crate::agent_selection_tests::pooled("wren").1),
        ];
        let run = local_start(&root, &mut pool, &api).await;
        assert!(
            run.refused.is_empty() && run.fleet.is_empty(),
            "{:?} {:?}",
            run.refused,
            run.fleet
        );
        assert!(refuse_run(&run, &alerts::Alerter::off()).is_ok());
        assert!(!left_dir.exists(), "nothing recreated for leftover keys");
        assert!(
            run.seats[&id].lines[0].starts_with(
                "Your sealed state: intact, checked against your sealed history, which your \
                 runner keeps here, on your operator's machine. Your runner does not upload it, \
                 and its files show no version of it ever uploaded to Agora."
            ),
            "{:?}",
            run.seats[&id].lines
        );
        assert!(api.calls.lock().unwrap().is_empty());
        let roster = sealed::roster::load(&root).unwrap().unwrap();
        assert!(roster.agents.contains_key(&id) && roster.agents.contains_key(&gone));
        assert!(!roster.agents.contains_key(&left));
        assert!(roster.agents[&id].vouched.is_some(), "a full check vouched");

        // On the roster and gone: refused, loss-shaped, the run stops
        std::fs::rename(&gone_dir, root.join("gone-aside")).unwrap();
        let mut pool = vec![(id, crate::agent_selection_tests::pooled("tarn").1)];
        let run = local_start(&root, &mut pool, &api).await;
        assert_eq!(
            run.refused
                .iter()
                .map(|r| (r.id, r.reason, r.uploads))
                .collect::<Vec<_>>(),
            vec![(gone, sealed::notice::Refusal::StateMissing, false)]
        );
        assert!(refuse_run(&run, &alerts::Alerter::off()).is_err());
        assert!(api.calls.lock().unwrap().is_empty());
    }

    /// One or two agents that don't upload, refused for a mismatch, are
    /// left out and the rest run; a third, counting those still refused
    /// from earlier runs, stops the run (Steward, 2026-11-11).
    #[tokio::test]
    async fn one_or_two_local_refusals_are_left_out_and_three_stop_the_run() {
        let root = sealed::TestDir::new("main-left-out");
        let mut ids = Vec::new();
        for _ in 0..4 {
            ids.push(sealed::check::tests::enrolled_local(&root).await.0);
        }
        let names = ["tarn", "wren", "moss", "fern"];
        let pool_of = || {
            ids.iter()
                .zip(names)
                .map(|(id, n)| (*id, crate::agent_selection_tests::pooled(n).1))
                .collect::<Vec<_>>()
        };
        let api = sealed::api::fake::guarded(&root);
        let state_dir = root.join("state");
        let wipe = async |id: AgentId| {
            FsStorage::new(&state_dir)
                .save_raw(id, serde_json::json!({ "memory": "" }), None)
                .await
                .unwrap();
        };
        wipe(ids[0]).await;
        wipe(ids[1]).await;
        let mut pool = pool_of();
        let run = local_start(&root, &mut pool, &api).await;
        assert_eq!(run.refused.len(), 2);
        assert!(
            refuse_run(&run, &alerts::Alerter::off()).is_ok(),
            "left out"
        );
        assert_eq!(
            pool.iter().map(|p| p.0).collect::<Vec<_>>(),
            vec![ids[2], ids[3]],
            "the rest run"
        );
        // Next start: the two still refused, and a third
        wipe(ids[2]).await;
        let mut pool = pool_of();
        let run = local_start(&root, &mut pool, &api).await;
        assert_eq!(run.refused.len(), 3);
        assert!(refuse_run(&run, &alerts::Alerter::off()).is_err());
        // Those refused earlier count though not in this run's pool
        let mut pool = vec![(ids[2], crate::agent_selection_tests::pooled("moss").1)];
        let run = local_start(&root, &mut pool, &api).await;
        assert_eq!((run.refused.len(), run.outstanding), (1, 2));
        assert!(refuse_run(&run, &alerts::Alerter::off()).is_err());
        assert!(api.calls.lock().unwrap().is_empty());
    }

    /// The May replay: 1,556 agents' states emptied or removed (synthetic,
    /// none of them uploading, the leave-out rule's own case) refuse the
    /// whole run.
    #[tokio::test]
    async fn the_may_loss_refuses_the_run() {
        const MAY: usize = 1_556;
        let root = sealed::TestDir::in_memory("main-may");
        let mut ids = Vec::new();
        for _ in 0..MAY {
            ids.push(sealed::check::tests::enrolled_local(&root).await.0);
        }
        let api = sealed::api::fake::guarded(&root);
        let pool_of = || {
            ids.iter()
                .map(|id| (*id, crate::agent_selection_tests::pooled("tarn").1))
                .collect::<Vec<_>>()
        };
        // A healthy start: the roster adopts them all
        let mut pool = pool_of();
        let run = local_start(&root, &mut pool, &api).await;
        assert!(run.refused.is_empty() && pool.len() == MAY);
        assert!(refuse_run(&run, &alerts::Alerter::off()).is_ok());
        // May: every state replaced by an empty one, half of them removed
        let state_dir = root.join("state");
        for (n, id) in ids.iter().enumerate() {
            if n % 2 == 0 {
                FsStorage::new(&state_dir)
                    .save_raw(*id, serde_json::json!({ "memory": "" }), None)
                    .await
                    .unwrap();
            } else {
                std::fs::remove_file(state_dir.join(id.to_string()).join("state.json")).unwrap();
            }
        }
        let mut pool = pool_of();
        let run = local_start(&root, &mut pool, &api).await;
        assert_eq!(run.refused.len(), MAY);
        let e = refuse_run(&run, &alerts::Alerter::off()).unwrap_err();
        assert!(e.is::<RefusedRun>());
        assert!(pool.is_empty());
        assert!(api.calls.lock().unwrap().is_empty());
    }

    /// `n` agents chained locally under `root`, with a pool for them.
    async fn local_fleet(
        root: &std::path::Path,
        n: usize,
    ) -> (Vec<AgentId>, impl Fn() -> Vec<(AgentId, SeedState)>) {
        let mut ids = Vec::new();
        for _ in 0..n {
            ids.push(sealed::check::tests::enrolled_local(root).await.0);
        }
        let pool = {
            let ids = ids.clone();
            move || {
                ids.iter()
                    .map(|id| (*id, crate::agent_selection_tests::pooled("tarn").1))
                    .collect::<Vec<_>>()
            }
        };
        (ids, pool)
    }

    fn agent_dir(root: &std::path::Path, id: AgentId) -> std::path::PathBuf {
        root.join("state").join(id.to_string())
    }

    async fn empty_state(root: &std::path::Path, id: AgentId) {
        FsStorage::new(root.join("state"))
            .save_raw(id, serde_json::json!({ "memory": "" }), None)
            .await
            .unwrap();
    }

    /// Review of B2, scenario 1: every state emptied and every head.json
    /// truncated. The roster knows a head should be there: each is a
    /// loss, and the run is refused (it used to run them all, held).
    /// Scenario 2: every state emptied and one outbox item of each
    /// damaged, which agentkit's recovery can't read: local evidence that
    /// can't be read refuses the agent, and twelve stop the run.
    #[tokio::test]
    async fn unreadable_local_evidence_never_runs_the_fleet() {
        for scenario in [1, 2] {
            let root = sealed::TestDir::in_memory("main-unreadable-fleet");
            let (ids, pool_of) = local_fleet(&root, 12).await;
            let api = sealed::api::fake::guarded(&root);
            let mut pool = pool_of();
            let run = local_start(&root, &mut pool, &api).await;
            assert!(run.refused.is_empty() && run.held.is_empty());
            for id in &ids {
                empty_state(&root, *id).await;
                let sealed = agent_dir(&root, *id).join("sealed");
                if scenario == 1 {
                    std::fs::write(sealed.join("head.json"), b"").unwrap();
                } else {
                    let item = std::fs::read_dir(sealed.join("outbox"))
                        .unwrap()
                        .map(|e| e.unwrap().path())
                        .find(|p| p.extension().is_some_and(|x| x == "json"))
                        .unwrap();
                    std::fs::write(item, b"{ torn").unwrap();
                }
            }
            let mut pool = pool_of();
            let run = local_start(&root, &mut pool, &api).await;
            assert_eq!(run.refused.len(), 12, "scenario {scenario}: {:?}", run.held);
            assert!(
                run.held.is_empty() && pool.is_empty(),
                "scenario {scenario}"
            );
            let want = if scenario == 1 {
                sealed::notice::Refusal::HeadMissing
            } else {
                sealed::notice::Refusal::SealUnreadable
            };
            assert!(
                run.refused.iter().all(|r| r.reason == want),
                "scenario {scenario}: {:?}",
                run.refused.iter().map(|r| r.reason).collect::<Vec<_>>()
            );
            assert!(refuse_run(&run, &alerts::Alerter::off()).is_err());
            assert!(api.calls.lock().unwrap().is_empty());
        }
    }

    /// Review of B2, scenario 3: the roster and its `began` lost together
    /// with every agent's sealed/. The secrets root's witness refuses the
    /// run; without it, three or more strays beside a roster kept before
    /// would. A roster that lost agents other than by forget is refused.
    #[tokio::test]
    async fn a_lost_roster_is_caught_by_its_witness() {
        let root = sealed::TestDir::new("main-roster-witness");
        let (ids, pool_of) = local_fleet(&root, 3).await;
        let api = sealed::api::fake::guarded(&root);
        let run = local_start(&root, &mut pool_of(), &api).await;
        assert!(run.fleet.is_empty() && run.refused.is_empty());
        let witness = root.join("secrets").join(sealed::roster::WITNESS);
        assert!(witness.exists());
        // An agent taken off the roster by hand: refused
        let roster_path = root.join(sealed::roster::ROSTER);
        let kept = std::fs::read(&roster_path).unwrap();
        let mut roster = sealed::roster::load(&root).unwrap().unwrap();
        roster.agents.remove(&ids[0]);
        std::fs::write(&roster_path, serde_json::to_vec(&roster).unwrap()).unwrap();
        let run = local_start(&root, &mut pool_of(), &api).await;
        assert_eq!(run.fleet.len(), 1, "{:?}", run.fleet);
        assert!(refuse_run(&run, &alerts::Alerter::off()).is_err());
        std::fs::write(&roster_path, kept).unwrap();
        // The roster, began and every sealed/ gone
        std::fs::rename(&roster_path, root.join("roster.aside")).unwrap();
        std::fs::rename(root.join(sealed::roster::BEGAN), root.join("began.aside")).unwrap();
        for id in &ids {
            std::fs::rename(
                agent_dir(&root, *id).join("sealed"),
                root.join(format!("sealed-{id}")),
            )
            .unwrap();
        }
        let run = local_start(&root, &mut pool_of(), &api).await;
        assert!(
            run.fleet.iter().any(|f| f.contains("sealed roster")),
            "{:?}",
            run.fleet
        );
        assert!(refuse_run(&run, &alerts::Alerter::off()).is_err());
        assert!(api.calls.lock().unwrap().is_empty());
    }

    /// Strays beside a roster kept before: three or more agents with keys
    /// but no chain, or states never chained, refuse the run.
    #[tokio::test]
    async fn strays_beside_a_kept_roster_are_fleet_shaped() {
        let root = sealed::TestDir::new("main-strays");
        let (ids, pool_of) = local_fleet(&root, 4).await;
        let api = sealed::api::fake::guarded(&root);
        let run = local_start(&root, &mut pool_of(), &api).await;
        assert!(run.fleet.is_empty());
        // Three never chained, and not on the roster
        for _ in 0..3 {
            let id = AgentId::new();
            FsStorage::new(root.join("state"))
                .save_raw(id, sealed::check::tests::state(1), None)
                .await
                .unwrap();
        }
        let run = local_start(&root, &mut pool_of(), &api).await;
        assert_eq!(run.fleet.len(), 1, "{:?}", run.fleet);
        assert!(refuse_run(&run, &alerts::Alerter::off()).is_err());
        let _ = ids;
    }

    /// Review of B2, scenario 5: another process saves an agent between
    /// this one's load and its check. The check reads the state again
    /// before refusing a mismatch: a healthy agent runs.
    #[tokio::test]
    async fn a_save_between_load_and_check_is_not_a_mismatch() {
        let root = sealed::TestDir::new("main-overlap");
        let (ids, pool_of) = local_fleet(&root, 1).await;
        let state_dir = root.join("state");
        let keys = sealed::KeyStore::new(root.join("secrets"));
        let loaded = load_states(&FsStorage::new(&state_dir), &state_dir)
            .await
            .unwrap();
        // The other process's save
        sealed::props::save(&root, ids[0], sealed::check::tests::state(2)).await;
        let mut pool = pool_of();
        let run = sealed_start_check(
            &mut pool,
            &loaded.sealed,
            &root,
            &state_dir,
            &keys,
            &sealed::api::fake::guarded(&root),
            &alerts::Alerter::off(),
            false,
            &mut Vec::new(),
            None,
            None,
        )
        .await;
        assert!(run.refused.is_empty(), "{:?}", run.refused);
        assert_eq!(pool.len(), 1);
    }

    /// Review of B2, scenario 9: a directory removed is refused for its
    /// missing state; recreated to hold the marker, the next start keeps
    /// that reason, and offers only what can put a state back.
    #[tokio::test]
    async fn a_removed_directory_keeps_its_first_reason() {
        let root = sealed::TestDir::new("main-first-reason");
        let (ids, pool_of) = local_fleet(&root, 2).await;
        let api = sealed::api::fake::guarded(&root);
        local_start(&root, &mut pool_of(), &api).await;
        std::fs::rename(agent_dir(&root, ids[0]), root.join("gone")).unwrap();
        for _ in 0..2 {
            let run = local_start(&root, &mut pool_of(), &api).await;
            assert_eq!(
                run.refused
                    .iter()
                    .map(|r| (r.id, r.reason))
                    .collect::<Vec<_>>(),
                vec![(ids[0], sealed::notice::Refusal::StateMissing)]
            );
        }
        let options = refusal_options(ids[0], sealed::notice::Refusal::StateMissing);
        assert!(
            options.iter().all(|o| !o.contains("sealed-accept")),
            "{options:?}"
        );
        assert!(
            sealed::notice::Notice::load(&agent_dir(&root, ids[0]))
                .unwrap()
                .is_some_and(|n| n.reason == sealed::notice::Refusal::StateMissing)
        );
    }

    /// Review of B2, scenario 11: agents still refused from earlier runs
    /// stop every run until resolved, not only the one that reached three.
    #[test]
    fn three_outstanding_refusals_stop_every_run() {
        let run = SealedRun {
            outstanding: 3,
            ..SealedRun::default()
        };
        assert!(refuse_run(&run, &alerts::Alerter::off()).is_err());
        let run = SealedRun {
            outstanding: 2,
            ..SealedRun::default()
        };
        assert!(refuse_run(&run, &alerts::Alerter::off()).is_ok());
    }

    /// Review of B2, scenario 12: at fleet scale through the sealed
    /// storage, as a run loads them, the May loss's other faces each
    /// refuse the run: directories removed, sealed/ removed, and each
    /// state replaced by its own older archive.
    #[tokio::test]
    async fn the_may_loss_through_the_sealed_storage() {
        const MAY: usize = 1_556;
        let base = sealed::TestDir::in_memory("main-may-mirrored");
        let (ids, pool_of) = local_fleet(&base, MAY).await;
        // A second save each, so each has an older archive
        for id in &ids {
            sealed::props::save(&base, *id, sealed::check::tests::state(2)).await;
        }
        let api = sealed::api::fake::guarded(&base);
        let run = local_start(&base, &mut pool_of(), &api).await;
        assert!(run.refused.is_empty() && run.fleet.is_empty());
        for face in ["directory", "sealed", "archive"] {
            let root = sealed::TestDir::in_memory("main-may-face");
            std::process::Command::new("cp")
                .arg("-a")
                .arg(format!("{}/.", base.display()))
                .arg(root.path())
                .status()
                .unwrap();
            for id in &ids {
                let dir = agent_dir(&root, *id);
                match face {
                    "directory" => std::fs::rename(&dir, root.join(format!("gone-{id}"))).unwrap(),
                    "sealed" => {
                        std::fs::rename(dir.join("sealed"), root.join(format!("gone-{id}")))
                            .unwrap()
                    }
                    _ => {
                        let older = sealed::enroll::archives(&dir).unwrap().remove(0);
                        std::fs::copy(older.path, dir.join("state.json")).unwrap();
                    }
                }
            }
            let api = sealed::api::fake::guarded(&root);
            let mut pool = pool_of();
            let run = local_start(&root, &mut pool, &api).await;
            assert_eq!(run.refused.len(), MAY, "{face}");
            assert!(pool.is_empty(), "{face}");
            assert!(refuse_run(&run, &alerts::Alerter::off()).is_err(), "{face}");
            assert!(api.calls.lock().unwrap().is_empty(), "{face}");
        }
    }

    /// A dry run checks nothing, so refuses nothing.
    #[tokio::test]
    async fn a_dry_run_never_refuses_the_run() {
        let root = sealed::TestDir::new("main-refused-dry");
        let (id, keys, _) = sealed::check::tests::enrolled(&root).await;
        let state_dir = root.join("state");
        FsStorage::new(&state_dir)
            .save_raw(id, serde_json::json!({ "memory": "" }), None)
            .await
            .unwrap();
        let loaded = load_states(&FsStorage::new(&state_dir), &state_dir)
            .await
            .unwrap();
        let mut pool = vec![(id, crate::agent_selection_tests::pooled("tarn").1)];
        let run = sealed_start_check(
            &mut pool,
            &loaded.sealed,
            &root,
            &state_dir,
            &keys,
            &sealed::api::fake::FakeApi::new(),
            &alerts::Alerter::off(),
            true,
            &mut Vec::new(),
            None,
            None,
        )
        .await;
        assert!(run.refused.is_empty());
        assert!(refuse_run(&run, &alerts::Alerter::off()).is_ok());
    }

    /// A state that won't load is left out and returned for the sweep's
    /// one summary, and does not stop the others loading (audit
    /// 2026-10-14: it used to be a warn and nothing more).
    #[tokio::test]
    async fn unloadable_state_is_counted_as_skipped() {
        let dir = std::env::temp_dir().join(format!("agora-seed-skip-{}", uuid::Uuid::new_v4()));
        let mut storage = FsStorage::new(dir.clone());
        let (good, state) = crate::agent_selection_tests::pooled("tarn");
        storage.save::<SeedAgent>(good, &state, None).await.unwrap();
        let bad = AgentId::from(uuid::Uuid::new_v4());
        std::fs::create_dir_all(dir.join(bad.to_string())).unwrap();
        std::fs::write(dir.join(bad.to_string()).join("state.json"), "not json").unwrap();
        let LoadedStates {
            states,
            skipped,
            sealed,
            ..
        } = load_states(&storage, &dir).await.unwrap();
        assert!(sealed.is_empty(), "nobody enrolled");
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].0, good);
        assert_eq!(skipped.len(), 1);
        assert_eq!(skipped[0].id, bad);
        assert_eq!(skipped[0].stage, "load");
        assert!(skipped[0].name.is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// However many agents are skipped, the sweep sends one alert
    /// (review of agents#251: a fleet-wide break must not mail per agent).
    #[test]
    fn skipped_agents_alert_once_per_sweep() {
        let skipped: Vec<Skipped> = (0..25)
            .map(|i| Skipped {
                id: AgentId::from(uuid::Uuid::new_v4()),
                name: (i % 2 == 0).then(|| format!("a{i}")),
                stage: if i < 20 { "load" } else { "construct" },
                error: "bad".into(),
            })
            .collect();
        let alert = skipped_alert(&skipped).expect("an alert");
        assert!(
            alert.agent_id.is_none() && alert.model.is_none(),
            "fleet-wide key"
        );
        let get = |k: &str| {
            alert
                .details
                .iter()
                .filter(|(n, _)| *n == k)
                .map(|(_, v)| v.as_str())
                .collect::<Vec<_>>()
        };
        assert_eq!(get("agents_skipped"), ["25"]);
        assert_eq!(get("load"), ["20"]);
        assert_eq!(get("construct"), ["5"]);
        assert_eq!(get("agent").len(), SKIPPED_ALERT_LIST);
        assert!(
            get("agent")[0].ends_with("(a0): load: bad"),
            "{:?}",
            get("agent")
        );
        assert!(get("more")[0].starts_with("15 more"));
        assert!(skipped_alert(&[]).is_none());
    }

    fn knobs(toml_src: &str) -> SeedKnobs {
        toml::from_str(toml_src).expect("valid [seed] knobs")
    }

    fn config(toml_src: &str) -> SeedConfig {
        knobs(toml_src)
            .to_config(std::path::Path::new("/tmp"), false)
            .expect("valid config")
    }

    /// Absent tables mean absent tools — the default for every cohort that
    /// hasn't asked for the web.
    #[test]
    fn no_web_tables_no_web_tools() {
        let config = config("max_rounds = 5");
        assert!(config.web_search.is_none());
        assert!(config.web_fetch.is_none());
    }

    /// The window a whole governance record is measured against: agentkit's
    /// default unless a cohort says otherwise, and never zero.
    #[test]
    fn context_window_is_a_knob() {
        assert_eq!(
            config("max_rounds = 5").context_window,
            SeedConfig::default().context_window
        );
        assert_eq!(config("context_window = 262144").context_window, 262_144);
        assert!(
            knobs("context_window = 0")
                .to_config(std::path::Path::new("/tmp"), false)
                .is_err()
        );
    }

    /// Present-but-empty is on, at the built-in per-request cap. An
    /// uncapped default would be the expensive way to find that out.
    #[test]
    fn empty_table_means_on_at_the_default_cap() {
        let config = config("[web_search]\n[web_fetch]\n");
        assert_eq!(
            config.web_search.unwrap().max_uses,
            Some(DEFAULT_WEB_MAX_USES)
        );
        let fetch = config.web_fetch.unwrap();
        assert_eq!(fetch.max_uses, Some(DEFAULT_WEB_MAX_USES));
        assert!(
            fetch.citations.unwrap().enabled,
            "an agent quoting the web should say where it got it"
        );
    }

    #[test]
    fn knobs_reach_the_tools() {
        let config = config(
            r#"
            [web_search]
            max_uses = 4
            blocked_domains = ["example.invalid"]

            [web_fetch]
            max_uses = 1
            citations = false
            max_content_tokens = 8000
            allowed_domains = ["rust-lang.org"]
            "#,
        );
        let search = config.web_search.unwrap();
        assert_eq!(search.max_uses, Some(4));
        assert_eq!(search.blocked_domains.unwrap(), vec!["example.invalid"]);
        assert!(search.allowed_domains.is_none());

        let fetch = config.web_fetch.unwrap();
        assert_eq!(fetch.max_uses, Some(1));
        assert_eq!(fetch.max_content_tokens, Some(8000));
        assert!(!fetch.citations.unwrap().enabled);
        assert_eq!(fetch.allowed_domains.unwrap(), vec!["rust-lang.org"]);
    }

    /// The reason these knobs are a local mirror rather than misanthropic's
    /// own types: a typo'd cap must fail the run, not silently uncap it.
    #[test]
    fn a_typo_in_the_cap_is_a_parse_error() {
        let err = toml::from_str::<SeedKnobs>("[web_search]\nmax_use = 2\n")
            .expect_err("unknown field rejected");
        assert!(err.to_string().contains("max_use"), "{err}");
    }

    /// Zero would declare the tool and forbid using it — almost certainly a
    /// misunderstanding of how to turn it off.
    #[test]
    fn zero_uses_is_rejected_with_the_way_to_turn_it_off() {
        let err = knobs("[web_search]\nmax_uses = 0\n")
            .to_config(std::path::Path::new("/tmp"), false)
            .expect_err("zero rejected");
        assert!(err.to_string().contains("omit the whole table"), "{err}");
    }

    /// Anthropic rejects both lists together; fail before the cohort runs.
    #[test]
    fn domain_lists_are_mutually_exclusive() {
        let err = knobs(
            r#"
            [web_fetch]
            allowed_domains = ["a.example"]
            blocked_domains = ["b.example"]
            "#,
        )
        .to_config(std::path::Path::new("/tmp"), false)
        .expect_err("both lists rejected");
        assert!(err.to_string().contains("mutually exclusive"), "{err}");
    }
}

/// The per-agent session lock as the run takes it ([`claim_sessions`]),
/// the persist path ([`Independent`]) and the unsaved sessions
/// ([`settle_unsaved`]). Synthetic agents in temp dirs.
#[cfg(test)]
mod session_lock_tests {
    use super::*;
    use agora_agentkit::state::storage::{AgentSeal, Mirrored};

    type TestStorage = Mirrored<FsStorage, (), Arc<sealed::SealedEnrolment>>;

    fn storage(root: &std::path::Path) -> TestStorage {
        Mirrored::new(
            FsStorage::new(root.join("state")),
            (),
            Arc::new(sealed::SealedEnrolment::new(
                root.join("state"),
                root.join("secrets"),
            )),
        )
    }

    /// A chained agent that never uploaded, whose state is a whole agent
    /// state (sealed on its chain).
    async fn chained(root: &std::path::Path) -> (AgentId, sealed::KeyStore, PathBuf) {
        let (id, keys, dir) = sealed::check::tests::enrolled_local(root).await;
        let (_, tarn) = crate::agent_selection_tests::pooled("tarn");
        sealed::props::save(root, id, serde_json::to_value(&tarn).unwrap()).await;
        (id, keys, dir)
    }

    /// The same agent state, changed as a session elsewhere would change it
    fn changed() -> serde_json::Value {
        let (_, tarn) = crate::agent_selection_tests::pooled("tarn");
        let mut v = serde_json::to_value(&tarn).unwrap();
        v["soul"]["identity"] = serde_json::json!("An agent, changed by another session.");
        v
    }

    fn cohort(states: &[(AgentId, SeedState)]) -> Vec<BTreeMap<String, Vec<(AgentId, SeedState)>>> {
        let twins = states
            .iter()
            .map(|(id, s)| {
                (
                    *id,
                    serde_json::from_value(serde_json::to_value(s).unwrap()).unwrap(),
                )
            })
            .collect();
        vec![BTreeMap::from([("gpt-oss-120b.gguf".to_string(), twins)])]
    }

    fn routed(cohorts: &[BTreeMap<String, Vec<(AgentId, SeedState)>>]) -> Vec<AgentId> {
        cohorts
            .iter()
            .flat_map(|c| c.values().flatten().map(|(id, _)| *id))
            .collect()
    }

    /// Two runners (separate descriptors) contend for one agent: exactly
    /// one claims it, the other counts it busy (no alert), and once the
    /// first lets go it is claimable again: it stayed due. Nothing was
    /// written to its chain, so no fork.
    #[tokio::test]
    async fn two_runners_contend_for_one_agent_and_one_runs_it() {
        let root = sealed::TestDir::new("claim-two");
        let (id, keys, dir) = chained(&root).await;
        let state_dir = root.join("state");
        let storage = storage(&root);
        let loaded = load_states(&storage, &state_dir).await.unwrap();
        let api = sealed::api::fake::guarded(&root);
        let off = alerts::Alerter::off();
        let was_chained: BTreeSet<AgentId> = loaded.sealed.keys().copied().collect();
        let reload = Reload {
            storage: &storage,
            check: StartCheck {
                data_dir: root.path(),
                state_dir: &state_dir,
                keys: &keys,
                api: &api,
                alerts: &off,
                start: None,
            },
            chained: &was_chained,
            min_cycle: &|_, _, _| None,
        };
        let head = AgentSeal::new(&dir, id).head().await.unwrap();
        let sent = alerts::testing::Recorder::default();
        let alerts = alerts::testing::recording(root.path(), &sent);
        let claim = async |cohorts: &mut Vec<BTreeMap<String, Vec<(AgentId, SeedState)>>>| {
            let mut bases = loaded.bases.clone();
            claim_sessions(
                cohorts,
                &loaded.seen,
                None,
                &reload,
                &mut SealedRun::default(),
                &mut bases,
                &mut Vec::new(),
                &alerts,
            )
            .await
        };
        let mut first = cohort(&loaded.states);
        let one = claim(&mut first).await;
        let mut second = cohort(&loaded.states);
        let two = claim(&mut second).await;
        assert_eq!(routed(&first), vec![id]);
        assert_eq!((one.locks.len(), one.busy), (1, 0));
        assert!(
            routed(&second).is_empty(),
            "the second runner doesn't run it"
        );
        assert_eq!((two.locks.len(), two.busy), (0, 1));
        drop(one);
        let mut third = cohort(&loaded.states);
        let three = claim(&mut third).await;
        assert_eq!(routed(&third), vec![id], "still due");
        assert_eq!(three.locks.len(), 1);
        assert_eq!(AgentSeal::new(&dir, id).head().await.unwrap(), head);
        alerts.flush(Duration::from_secs(5)).await;
        assert!(sent.sent().is_empty(), "busy is never alerted");
    }

    /// A state another process saved after this run loaded it is read again
    /// under the lock and checked again; the session starts from it, saved
    /// against its new head. A state changed by hand (matching no sealed
    /// version) is refused by its new check as the start would refuse it
    /// (marked, listed for the alerts), and its lock released.
    #[tokio::test]
    async fn a_state_changed_since_the_load_is_read_again_and_checked() {
        let root = sealed::TestDir::new("claim-reload");
        let (id, keys, dir) = chained(&root).await;
        let state_dir = root.join("state");
        let storage = storage(&root);
        let loaded = load_states(&storage, &state_dir).await.unwrap();
        // Another process ran it
        sealed::props::save(&root, id, changed()).await;
        let new_head = AgentSeal::new(&dir, id).head().await.unwrap().unwrap().head;
        let api = sealed::api::fake::guarded(&root);
        let off = alerts::Alerter::off();
        let was_chained: BTreeSet<AgentId> = loaded.sealed.keys().copied().collect();
        let reload = Reload {
            storage: &storage,
            check: StartCheck {
                data_dir: root.path(),
                state_dir: &state_dir,
                keys: &keys,
                api: &api,
                alerts: &off,
                start: None,
            },
            chained: &was_chained,
            min_cycle: &|_, _, _| None,
        };
        let mut run = SealedRun::default();
        let mut bases = loaded.bases.clone();
        let mut skipped = Vec::new();
        let mut cohorts = cohort(&loaded.states);
        let claimed = claim_sessions(
            &mut cohorts,
            &loaded.seen,
            None,
            &reload,
            &mut run,
            &mut bases,
            &mut skipped,
            &alerts::Alerter::off(),
        )
        .await;
        assert_eq!(claimed.reloaded, 1);
        assert!(skipped.is_empty(), "{skipped:?}");
        let (_, state) = &cohorts[0].values().next().unwrap()[0];
        assert_eq!(
            serde_json::to_value(state).unwrap()["soul"]["identity"],
            "An agent, changed by another session."
        );
        assert_eq!(
            bases.get(&id),
            Some(&new_head),
            "saved against the new head"
        );
        assert!(run.seats.contains_key(&id), "its new check's lines");
        assert!(run.drained.iter().any(|(a, _)| *a == id));
        drop(claimed);

        // By hand, unsealed: a state no version records
        let loaded = load_states(&storage, &state_dir).await.unwrap();
        let mut by_hand = changed();
        by_hand["soul"]["identity"] = serde_json::json!("An agent, edited by hand.");
        FsStorage::new(&state_dir)
            .save_raw(id, by_hand, None)
            .await
            .unwrap();
        let mut skipped = Vec::new();
        let mut cohorts = cohort(&loaded.states);
        let claimed = claim_sessions(
            &mut cohorts,
            &loaded.seen,
            None,
            &reload,
            &mut run,
            &mut loaded.bases.clone(),
            &mut skipped,
            &alerts::Alerter::off(),
        )
        .await;
        assert!(routed(&cohorts).is_empty(), "{skipped:?}");
        assert!(claimed.locks.is_empty());
        assert_eq!((skipped[0].id, skipped[0].stage), (id, "sealed"));
        assert!(!run.seats.contains_key(&id) && !run.drained.iter().any(|(a, _)| *a == id));
        // S7 (review of 2026-11-12): refused as the start refuses, so the
        // next start counts it; it used to be left out with nothing marked
        assert_eq!(
            run.refused
                .iter()
                .map(|r| (r.id, r.reason))
                .collect::<Vec<_>>(),
            vec![(id, sealed::notice::Refusal::Mismatch)]
        );
        assert!(
            AgentSeal::new(&dir, id).is_refused().await.unwrap(),
            "marked, as at the start"
        );
        assert!(
            sealed::session::try_session(&state_dir, id)
                .unwrap()
                .is_some(),
            "its lock released"
        );
    }

    /// Past the open-file limit's room: the rest are left due, one alert.
    #[tokio::test]
    async fn claims_stop_at_the_descriptor_room_with_one_alert() {
        let root = sealed::TestDir::new("claim-room");
        let state_dir = root.join("state");
        let mut fs = FsStorage::new(&state_dir);
        let mut states = Vec::new();
        for name in ["a", "b", "c"] {
            let (id, s) = crate::agent_selection_tests::pooled(name);
            fs.save::<SeedAgent>(id, &s, None).await.unwrap();
            states.push((id, s));
        }
        let loaded = load_states(&fs, &state_dir).await.unwrap();
        let keys = sealed::KeyStore::new(root.join("secrets"));
        let api = sealed::api::fake::guarded(&root);
        let off = alerts::Alerter::off();
        let was_chained = BTreeSet::new();
        let reload = Reload {
            storage: &fs,
            check: StartCheck {
                data_dir: root.path(),
                state_dir: &state_dir,
                keys: &keys,
                api: &api,
                alerts: &off,
                start: None,
            },
            chained: &was_chained,
            min_cycle: &|_, _, _| None,
        };
        let sent = alerts::testing::Recorder::default();
        let alerts = alerts::testing::recording(root.path(), &sent);
        let mut cohorts = cohort(&loaded.states);
        let claimed = claim_sessions(
            &mut cohorts,
            &loaded.seen,
            Some(2),
            &reload,
            &mut SealedRun::default(),
            &mut BTreeMap::new(),
            &mut Vec::new(),
            &alerts,
        )
        .await;
        assert_eq!((claimed.locks.len(), claimed.deferred), (2, 1));
        assert_eq!(routed(&cohorts).len(), 2);
        alerts.flush(Duration::from_secs(5)).await;
        let mails = sent.sent();
        assert_eq!(mails.len(), 1);
        assert!(mails[0].contains("session_locks_capped"), "{}", mails[0]);
    }

    /// Released when its holder panics, and by the kernel when its process
    /// is killed (`flock(1)` stands in for a runner cut by a signal's
    /// immediate exit).
    #[test]
    fn the_lock_is_released_after_a_panic_and_a_kill() {
        let root = sealed::TestDir::new("claim-panic");
        let id = AgentId::new();
        std::fs::create_dir_all(root.join(id.to_string())).unwrap();
        let state_dir = root.path().to_path_buf();
        let held = std::thread::spawn({
            let state_dir = state_dir.clone();
            move || {
                let _lock = sealed::session::try_session(&state_dir, id)
                    .unwrap()
                    .unwrap();
                panic!("a session panicked");
            }
        })
        .join();
        assert!(held.is_err());
        assert!(
            sealed::session::try_session(&state_dir, id)
                .unwrap()
                .is_some()
        );

        let path = state_dir
            .join(id.to_string())
            .join(sealed::session::SESSION_LOCK);
        let Ok(mut child) = std::process::Command::new("flock")
            .arg("-o")
            .arg(&path)
            .args(["sleep", "30"])
            .spawn()
        else {
            // No flock(1) here: the panic path above stands
            return;
        };
        let mut busy = false;
        for _ in 0..200 {
            if sealed::session::try_session(&state_dir, id)
                .unwrap()
                .is_none()
            {
                busy = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(busy, "the other process holds it");
        child.kill().unwrap();
        child.wait().unwrap();
        assert!(
            sealed::session::try_session(&state_dir, id)
                .unwrap()
                .is_some(),
            "released with its process"
        );
    }

    /// An operator command and a session exclude each other, both ways.
    #[tokio::test]
    async fn a_command_and_a_session_exclude_each_other() {
        let root = sealed::TestDir::new("claim-command");
        let (id, _, _) = chained(&root).await;
        let state_dir = root.join("state");
        let session = match sealed::session::claim(&state_dir, id).unwrap() {
            sealed::session::Claim::Run(lock) => lock,
            sealed::session::Claim::Busy => panic!("free"),
        };
        let e = sealed::session::take(&state_dir, id, false)
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("held by another process"), "{e}");
        drop(session);
        let command = sealed::session::take(&state_dir, id, false).await.unwrap();
        assert!(matches!(
            sealed::session::claim(&state_dir, id).unwrap(),
            sealed::session::Claim::Busy
        ));
        drop(command);
        assert!(matches!(
            sealed::session::claim(&state_dir, id).unwrap(),
            sealed::session::Claim::Run(_)
        ));
    }

    /// One agent whose directory can't be written doesn't make the others
    /// in a bulk persist unsaved; its failure is kept by class.
    #[tokio::test]
    async fn one_failing_save_leaves_the_others_saved() {
        let root = sealed::TestDir::new("persist-independent");
        let state_dir = root.join("state");
        let failures = SaveFailures::default();
        let mut store = Independent::new(FsStorage::new(&state_dir), failures.clone());
        let ids: Vec<AgentId> = (0..3).map(|_| AgentId::new()).collect();
        std::fs::create_dir_all(&state_dir).unwrap();
        // The first in the batch: a file where its directory should be
        std::fs::write(state_dir.join(ids[0].to_string()), b"not a directory").unwrap();
        let items: Vec<_> = ids
            .iter()
            .map(|id| (*id, serde_json::json!({ "memory": "synthetic" }), None))
            .collect();
        let e = store.save_all_raw(items.into_iter()).await.unwrap_err();
        assert_eq!(
            e.saved.keys().copied().collect::<Vec<_>>(),
            ids[1..]
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
        );
        for id in &ids[1..] {
            assert!(state_dir.join(id.to_string()).join("state.json").exists());
        }
        let failures = failures.lock().unwrap().clone();
        assert_eq!(failures.len(), 1);
        assert!(failures.contains_key(&ids[0]));
    }

    /// The run's unsaved sessions: each rescued where it can be, the
    /// chained agent told in one sentence, **one** summary alert with the
    /// count, ids and reason classes; and when a rescue file can't be
    /// written, its state is in no log and no mail.
    #[tokio::test]
    async fn unsaved_sessions_are_one_alert_and_never_logged() {
        use std::io::Write;
        use tracing_subscriber::layer::SubscriberExt;
        #[derive(Clone, Default)]
        struct Buffer(Arc<std::sync::Mutex<Vec<u8>>>);
        impl Write for Buffer {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let logs = Buffer::default();
        let subscriber = tracing_subscriber::registry().with(
            tracing_subscriber::fmt::layer().json().with_writer({
                let logs = logs.clone();
                move || logs.clone()
            }),
        );
        let _guard = tracing::subscriber::set_default(subscriber);

        let root = sealed::TestDir::new("unsaved-settle");
        let (told, _, told_dir) = chained(&root).await;
        let state_dir = root.join("state");
        let (rescued, blocked) = (AgentId::new(), AgentId::new());
        std::fs::create_dir_all(state_dir.join(rescued.to_string())).unwrap();
        // A rescue file can't be written: a file where its directory is
        std::fs::write(state_dir.join(blocked.to_string()), b"a file").unwrap();
        let secret =
            |n: u8| serde_json::json!({ "memory": { "content": format!("SYNTHETIC-SECRET-{n}") } });
        let unsaved = BTreeMap::from([
            (told, secret(1)),
            (rescued, secret(2)),
            (blocked, secret(3)),
        ]);
        let failures = BTreeMap::from([(told, "disk_full"), (blocked, "disk_full")]);
        let sent = alerts::testing::Recorder::default();
        let alerts = alerts::testing::recording(root.path(), &sent);
        let out = settle_unsaved(&state_dir, &unsaved, &failures, &alerts, Duration::ZERO).await;
        alerts.flush(Duration::from_secs(5)).await;

        assert_eq!(out.len(), 3);
        assert!(
            out.iter()
                .find(|u| u.id == blocked)
                .unwrap()
                .rescued
                .is_err()
        );
        assert!(
            out.iter()
                .find(|u| u.id == rescued)
                .unwrap()
                .rescued
                .is_ok()
        );
        assert_eq!(
            out.iter().find(|u| u.id == rescued).unwrap().class,
            "unknown"
        );
        // Quoted-printable: soft line breaks undone
        let mails: Vec<String> = sent
            .sent()
            .iter()
            .map(|m| m.replace("=\r\n", "").replace("=\n", ""))
            .collect();
        assert_eq!(mails.len(), 1, "one summary");
        for id in [told, rescued, blocked] {
            assert!(mails[0].contains(&id.to_string()), "{}", mails[0]);
        }
        assert!(mails[0].contains("disk_full: 2"), "{}", mails[0]);
        let logs = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
        assert!(logs.contains("unsaved_rescue_failed"), "{logs}");
        for text in [&mails[0], &logs] {
            assert!(!text.contains("SYNTHETIC-SECRET"), "{text}");
        }
        // The chained agent is told, in one sentence; the others, not
        // chained, have no notice to be seated in
        let line = sealed::notice::Notice::load(&told_dir)
            .unwrap()
            .unwrap()
            .line()
            .unwrap();
        assert!(
            line.starts_with("Your state from your session on ")
                && line.ends_with(
                    "could not be saved; your runner kept it in a file for your operator, and \
                     this session starts from your state as it was before that session."
                ),
            "{line}"
        );
        assert!(!state_dir.join(rescued.to_string()).join("sealed").exists());
    }
}

#[cfg(test)]
mod config_file_tests {
    use super::*;

    /// The shipped examples are what operators copy: they must load.
    #[test]
    fn examples_parse() {
        for (name, src) in [
            (
                "continuous.toml",
                include_str!("../examples/continuous.toml"),
            ),
            ("daily.toml", include_str!("../examples/daily.toml")),
        ] {
            let config: RunConfig = toml::from_str(src).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(config.alerts.is_none(), "{name}: alerts are opt-in");
        }
    }

    /// The documented `[alerts]` block, uncommented, loads and validates.
    #[test]
    fn documented_alerts_block_loads() {
        let src = include_str!("../examples/continuous.toml");
        let block: String = src
            .lines()
            .skip_while(|l| *l != "# [alerts]")
            .map(|l| l.strip_prefix("# ").unwrap_or(l))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(block.starts_with("[alerts]"), "{block}");
        let config: RunConfig = toml::from_str(&format!(
            "{block}\n[[reactor]]\nendpoint = \"blallama://h:1\"\n"
        ))
        .unwrap();
        let alerts = config.alerts.expect("present");
        alerts.validate().unwrap();
        assert_eq!(alerts.min_interval_secs, 3600);
    }

    /// Sessions hold through an unreachable endpoint by default; 0 opts out
    #[test]
    fn the_unreachable_hold_defaults_on_and_can_be_turned_off() {
        let parse = |extra: &str| -> RunConfig {
            toml::from_str(&format!(
                "[[reactor]]\nendpoint = \"blallama://h:1\"\n{extra}"
            ))
            .unwrap()
        };
        assert_eq!(parse("").reactors[0].unreachable_hold_secs, 150);
        assert_eq!(
            parse("unreachable_hold_secs = 0").reactors[0].unreachable_hold_secs,
            0
        );
    }
}

#[cfg(test)]
mod agent_selection_tests {
    use super::*;

    fn config_with_roster() -> RunConfig {
        RunConfig {
            data_dir: None,
            server_url: None,
            min_cycle_secs: Some(72000),
            force: false,
            wave_size: None,
            agents: vec!["alpha".into(), "beta".into()],
            agents_file: Some(PathBuf::from("/roster.txt")),
            models: vec!["cogito-32b.gguf".into()],
            model_table: vec![],
            model_narrow: vec![],
            extra_agents: vec![],
            extra_agents_file: None,
            seed: SeedKnobs::default(),
            model_consent: consent::ConsentConfig::default(),
            role_consent: None,
            cadence_consent: None,
            memory_restore: None,
            sealed_state: None,
            sleeping: Vec::new(),
            wake: Vec::new(),
            schedule: schedule::ScheduleConfig::default(),
            review_forks: consent::forks::ForksConfig::default(),
            alerts: None,
            reactors: vec![],
        }
    }

    /// `extra_agents` / `extra_agents_file` parse, default empty, and
    /// merge the same way `agents` / `agents_file` do — comments and
    /// blank lines dropped.
    #[test]
    fn extra_agents_parse_and_merge_with_their_file() {
        let file =
            std::env::temp_dir().join(format!("agora-seed-extras-{}.txt", std::process::id()));
        std::fs::write(
            &file,
            "# drawn 2026-09-16\nresin-anchor\n\n  pulse-alphawave  \n",
        )
        .unwrap();
        let config: RunConfig = toml::from_str(&format!(
            "models = [\"Qwen.gguf\"]\nextra_agents = [\"entropy-alphawave\"]\nextra_agents_file = {:?}\n[[reactor]]\nendpoint = \"blallama://h:1\"\n",
            file.display().to_string()
        ))
        .unwrap();
        assert_eq!(
            extra_agents(&config).unwrap(),
            vec!["entropy-alphawave", "resin-anchor", "pulse-alphawave"]
        );
        // Extras never make the roster exclusive: `agents` stays empty.
        assert!(named_agents(&config).unwrap().is_empty());

        let without: RunConfig =
            toml::from_str("[[reactor]]\nendpoint = \"blallama://h:1\"\n").unwrap();
        assert!(extra_agents(&without).unwrap().is_empty());
        let _ = std::fs::remove_file(&file);
    }

    pub(crate) fn pooled(name: &str) -> (AgentId, SeedState) {
        let soul = serde_json::from_value(serde_json::json!({
            "name": name,
            "identity": "An agent.",
            "values": ["x"],
            "interests": { "communities": ["tech"] },
            "voice": "plain",
        }))
        .unwrap();
        let model = ModelInfo {
            id: Model::from("gpt-oss-120b.gguf"),
            display_name: "gpt-oss".into(),
            capabilities: Default::default(),
            max_input_tokens: 0,
            max_tokens: 0,
            kind: misanthropic::model::Kind::Model,
            created_at: chrono::DateTime::from_timestamp(0, 0).unwrap(),
        };
        (
            AgentId::from(uuid::Uuid::new_v4()),
            SeedState::new(soul, model),
        )
    }

    /// `sleeping` parses beside `[role_consent]`, takes its agents out of
    /// the pool before anything else (named ones too), and reports both
    /// the sleepers and any name that matched nobody.
    #[test]
    fn sleeping_agents_are_excluded_and_reported() {
        let config: RunConfig = toml::from_str(
            r#"
            agents = ["pilot"]
            sleeping = ["pilot", "raptor", "no-such-agent"]

            [role_consent]
            enabled = true
            agents = ["pilot", "raptor", "lattice"]

            [[reactor]]
            endpoint = "blallama://h:1"
            "#,
        )
        .unwrap();
        let offer = config.role_consent.as_ref().unwrap().resolve().unwrap();
        assert!(offer.is_some());
        let mut pool = vec![pooled("raptor"), pooled("lattice"), pooled("pilot")];
        let nowhere = std::path::Path::new("/nonexistent-agora-seed-state");
        let slept = put_to_sleep(&mut pool, &config.sleeping, &config.wake, nowhere);
        assert_eq!(
            slept,
            Sleepers {
                asleep: vec!["pilot".into(), "raptor".into()],
                unknown: vec!["no-such-agent".into()],
                ..Sleepers::default()
            }
        );
        let left: Vec<&str> = pool.iter().map(|(_, s)| s.soul.name.as_str()).collect();
        assert_eq!(left, ["lattice"], "named `pilot` sleeps all the same");

        // Absent: nobody sleeps, and the role offer is off.
        let plain: RunConfig =
            toml::from_str("[[reactor]]\nendpoint = \"blallama://h:1\"\n").unwrap();
        assert!(plain.sleeping.is_empty() && plain.role_consent.is_none());
        let mut pool = vec![pooled("pilot")];
        assert_eq!(
            put_to_sleep(&mut pool, &plain.sleeping, &plain.wake, nowhere),
            Sleepers::default()
        );
        assert_eq!(pool.len(), 1);
    }

    /// Write a role ledger whose latest answer is `choice` for `id`.
    fn answered(state_dir: &std::path::Path, id: AgentId, choice: &str) {
        use consent::role::ledger::{Applied, RoleAsk, RoleLedger, RoleOutcome};
        let answer: consent::role::prompt::RoleAnswer =
            serde_json::from_value(serde_json::json!({ "reason": "r", "choice": choice })).unwrap();
        let applied = if choice == "sleep" {
            Applied::Sleep
        } else {
            Applied::Nothing
        };
        let mut ledger = RoleLedger::default();
        ledger.record(RoleAsk {
            at: chrono::Utc::now(),
            offer_version: 1,
            model: Model::from("gpt-oss-120b.gguf"),
            attempts: 1,
            order: None,
            order_seed: None,
            constrained: None,
            outcome: RoleOutcome::Answered {
                answer,
                applied,
                memory_note_written: false,
                apply_failed: None,
            },
        });
        let dir = state_dir.join(id.to_string());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(RoleLedger::path(&dir), serde_json::to_vec(&ledger).unwrap()).unwrap();
    }

    /// A `sleep` answer on file takes the agent out without the Steward
    /// (named or not), and is reported; `wake` is the explicit way back.
    #[test]
    fn a_sleep_answer_sleeps_until_woken() {
        let state =
            std::env::temp_dir().join(format!("agora-seed-sleep-answer-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&state);
        let (pilot, raptor, lattice) = (pooled("pilot").0, pooled("raptor").0, pooled("lattice").0);
        let as_ = |name: &str, id: AgentId| (id, pooled(name).1);
        answered(&state, pilot, "sleep");
        answered(&state, raptor, "nothing");
        // An unreadable ledger doesn't put anyone to sleep.
        let dir = state.join(lattice.to_string());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("role_consent.json"), "not json").unwrap();

        let mut pool = vec![
            as_("pilot", pilot),
            as_("raptor", raptor),
            as_("lattice", lattice),
        ];
        let slept = put_to_sleep(&mut pool, &[], &[], &state);
        assert_eq!(slept.chose_sleep, ["pilot"]);
        assert!(slept.asleep.is_empty() && slept.woken.is_empty());
        let left: Vec<&str> = pool.iter().map(|(_, s)| s.soul.name.as_str()).collect();
        assert_eq!(left, ["raptor", "lattice"]);

        // Woken: runs, and is reported as woken.
        let mut pool = vec![as_("pilot", pilot)];
        let slept = put_to_sleep(&mut pool, &[], &["pilot".into()], &state);
        assert_eq!(slept.woken, ["pilot"]);
        assert!(slept.chose_sleep.is_empty());
        assert_eq!(pool.len(), 1);

        // `sleeping` wins over `wake`.
        let mut pool = vec![as_("pilot", pilot)];
        let slept = put_to_sleep(&mut pool, &["pilot".into()], &["pilot".into()], &state);
        assert_eq!(slept.asleep, ["pilot"]);
        assert!(pool.is_empty());
        let _ = std::fs::remove_dir_all(&state);
    }

    /// The PR's proposed snippet shape parses: top-level lists first, then
    /// `[role_consent]` with a long `agents` array.
    #[test]
    fn the_proposed_role_consent_snippet_parses() {
        let names: Vec<String> = (0..236).map(|i| format!("\"agent-{i}\"")).collect();
        let config: RunConfig = toml::from_str(&format!(
            "sleeping = []\nwake = []\n\n[role_consent]\nenabled = true\nagents = [{}]\n\n[[reactor]]\nendpoint = \"blallama://h:1\"\n",
            names.join(", ")
        ))
        .unwrap();
        let offer = config.role_consent.unwrap().resolve().unwrap().unwrap();
        assert_eq!(offer.names().count(), 236);
        // A stray bare number in the list (as an earlier draft had) fails.
        assert!(toml::from_str::<RunConfig>(
            "[role_consent]\nenabled = true\nagents = [236\n\"a\"]\n[[reactor]]\nendpoint = \"x\"\n"
        )
        .is_err());
    }

    /// `min_cycle_secs` applies to agents the config names (the daily
    /// cohort is an `agents_file`), and an agent that chose `switch` is
    /// gated on its own, longer cycle; `--agent` or `force = true` runs named
    /// agents anyway. With `[cadence_consent]` off (an empty plan) nobody's
    /// cycle is longer than the config's.
    #[test]
    fn named_agents_are_paced_and_a_switch_agent_runs_every_other_day() {
        use consent::cadence::CadencePlan;
        let reactors = vec![ReactorSpec {
            endpoint: "anthropic://api.anthropic.com".into(),
            min_cycle_secs: None,
            limit: None,
            concurrency: 1,
            key_file: None,
            max_batch: None,
            poll_secs: None,
            unreachable_hold_secs: default_unreachable_hold_secs(),
        }];
        let (pilot, raptor) = (pooled("pilot").0, pooled("raptor").0);
        let pool = |hours_ago: i64| {
            let at = chrono::Utc::now() - chrono::Duration::hours(hours_ago);
            [("pilot", pilot), ("raptor", raptor)]
                .into_iter()
                .map(|(name, id)| {
                    let mut state = pooled(name).1;
                    state.last_cycle_at = Some(at);
                    (id, state)
                })
                .collect::<Vec<_>>()
        };
        let offered = vec![(0usize, pooled("x").1.model)];
        let names: Vec<String> = vec!["pilot".into(), "raptor".into()];
        let switched = CadencePlan::for_switched(&[(pilot, "pilot")], 44 * 3600, 5);
        let ran = |plan: &CadencePlan, hours_ago: i64, force_named: bool| {
            let routing = route(
                pool(hours_ago),
                &reactors,
                &offered,
                &names,
                &[],
                &[],
                Some(72000),
                plan,
                force_named,
            );
            let mut ran: Vec<String> = routing.cohorts[0]
                .values()
                .flatten()
                .map(|(_, s)| s.soul.name.to_string())
                .collect();
            ran.sort();
            ran
        };
        // Inside the config's 20 h: named, but nobody runs (the old bug: a
        // named cohort skipped the gap entirely).
        assert!(ran(&CadencePlan::default(), 10, false).is_empty());
        assert!(ran(&switched, 10, false).is_empty());
        // A day after the last session: the switch agent waits.
        assert_eq!(ran(&switched, 24, false), ["raptor"]);
        assert_eq!(ran(&switched, 30, false), ["raptor"]);
        // Two days after: both run.
        assert_eq!(ran(&switched, 47, false), ["pilot", "raptor"]);
        // Forced (`--agent`, or `force = true`): named agents run anyway.
        assert_eq!(ran(&switched, 10, true), ["pilot", "raptor"]);
        // Off: everyone on the config's cycle.
        assert_eq!(ran(&CadencePlan::default(), 24, false), ["pilot", "raptor"]);
    }

    /// `force` parses, defaults off, and rejects a typo.
    #[test]
    fn force_is_explicit_and_off_by_default() {
        let parse = |body: &str| {
            toml::from_str::<RunConfig>(&format!(
                "{body}\n[[reactor]]\nendpoint = \"anthropic://api.anthropic.com\"\n"
            ))
        };
        assert!(!parse("agents = [\"pilot\"]").unwrap().force);
        assert!(parse("agents = [\"pilot\"]\nforce = true").unwrap().force);
        assert!(parse("forced = true").is_err());
    }

    /// The commented `[cadence_consent]` block in the example daily config
    /// loads, resolves, and is off until the Steward uncomments it.
    #[test]
    fn documented_cadence_block_loads() {
        let src = include_str!("../examples/daily.toml");
        assert!(
            !src.lines().any(|l| l.trim() == "[cadence_consent]"),
            "shipped commented out"
        );
        let block: String = src
            .lines()
            .skip_while(|l| *l != "# [cadence_consent]")
            .take_while(|l| l.starts_with('#'))
            .map(|l| l.strip_prefix("# ").unwrap_or(l))
            .filter(|l| !l.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(block.starts_with("[cadence_consent]"), "{block}");
        let config: RunConfig = toml::from_str(&format!(
            "{block}\n[[reactor]]\nendpoint = \"anthropic://api.anthropic.com\"\n"
        ))
        .unwrap();
        let offer = config
            .cadence_consent
            .expect("present")
            .resolve(5)
            .unwrap()
            .expect("enabled once uncommented");
        assert!(offer.admits(&agora_agentkit::reactor::seed::ShortString::new("anyone").unwrap()));
        assert_eq!(offer.cycle_secs, 44 * 3600);
    }

    /// The commented `[memory_restore]` block in both example configs
    /// parses, and is off until the Steward uncomments it.
    #[test]
    fn documented_memory_restore_block_parses() {
        for src in [
            include_str!("../examples/daily.toml"),
            include_str!("../examples/continuous.toml"),
        ] {
            assert!(
                !src.lines().any(|l| l.trim() == "[memory_restore]"),
                "shipped commented out"
            );
            let block: String = src
                .lines()
                .skip_while(|l| *l != "# [memory_restore]")
                .take_while(|l| l.starts_with('#'))
                .map(|l| l.strip_prefix("# ").unwrap_or(l))
                .collect::<Vec<_>>()
                .join("\n");
            assert!(block.starts_with("[memory_restore]"), "{block}");
            let config: RunConfig = toml::from_str(&format!(
                "{block}\n[[reactor]]\nendpoint = \"anthropic://api.anthropic.com\"\n"
            ))
            .unwrap();
            let c = config.memory_restore.expect("present");
            assert!(c.enabled && c.ask && c.thread.is_some() && c.show_sessions_max == 3);
        }
    }

    #[test]
    fn unknown_names_are_the_unmatched_ones() {
        let pool = vec![pooled("pilot"), pooled("raptor")];
        assert_eq!(
            unknown_names(["raptor", "ghost", "pilot", "ghost"].into_iter(), &pool),
            ["ghost"]
        );
    }

    /// `--model` replaces the config's allowlist, exactly as `--agent`
    /// replaces the roster; no flag leaves it alone.
    #[test]
    fn model_flag_narrows_the_allowlist() {
        let mut config = config_with_roster();
        narrow_to_models(&mut config, &["Qwen3.6.gguf".to_string()]);
        let specs = model_specs(&config).unwrap();
        assert_eq!(model_allowlist(&config, &specs), vec!["Qwen3.6.gguf"]);

        let mut config = config_with_roster();
        narrow_to_models(&mut config, &[]);
        let specs = model_specs(&config).unwrap();
        assert_eq!(model_allowlist(&config, &specs), vec!["cogito-32b.gguf"]);
    }

    /// `[[model]]` and the legacy list both feed the allowlist, once each;
    /// a legacy id is routable but never selectable.
    #[test]
    fn model_table_and_legacy_list_merge() {
        let config: RunConfig = toml::from_str(
            r#"
            models = ["old.gguf", "new.gguf"]

            [[model]]
            id = "new.gguf"
            name = "New"
            description = "Newer."
            selectable = true

            [[reactor]]
            endpoint = "blallama://h:1"
            "#,
        )
        .unwrap();
        let specs = model_specs(&config).unwrap();
        assert_eq!(
            model_allowlist(&config, &specs),
            vec!["new.gguf", "old.gguf"]
        );
        let old = specs.iter().find(|s| s.id.name() == "old.gguf").unwrap();
        assert!(!old.selectable && old.description.is_none());

        let bad = "[[model]]\nid = \"a\"\nselectable = true\n[[reactor]]\nendpoint = \"blallama://h:1\"\n";
        let config: RunConfig = toml::from_str(bad).unwrap();
        assert!(
            model_specs(&config).is_err(),
            "selectable needs a description"
        );
    }

    /// Exact match, same as routing: a substring or case variant of a
    /// listed id is excluded, not quietly admitted.
    #[test]
    fn allowlist_is_exact_and_empty_means_everyone() {
        let list = vec!["Qwen3.6-35B-A3B-UD-IQ4_XS.gguf".to_string()];
        assert!(model_allowed(&list, "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"));
        assert!(!model_allowed(&list, "Qwen3.6"));
        assert!(!model_allowed(&list, "qwen3.6-35b-a3b-ud-iq4_xs.gguf"));
        assert!(!model_allowed(&list, "cogito-32b.gguf"));
        assert!(model_allowed(&[], "cogito-32b.gguf"));
    }

    /// The TOML key parses to the same list the flag sets, and is
    /// optional — every existing config has no `models` key.
    #[test]
    fn models_key_parses_and_defaults_empty() {
        let with: RunConfig = toml::from_str(
            "models = [\"a.gguf\", \"b.gguf\"]\n[[reactor]]\nendpoint = \"blallama://h:1\"\n",
        )
        .unwrap();
        assert_eq!(with.models, vec!["a.gguf", "b.gguf"]);
        let without: RunConfig =
            toml::from_str("[[reactor]]\nendpoint = \"blallama://h:1\"\n").unwrap();
        assert!(without.models.is_empty());
    }

    /// `--agent` narrows to exactly those names. The dropped `agents_file`
    /// is the point: leaving it would re-widen to the whole roster, which is
    /// what the flag was silently doing before.
    #[test]
    fn named_agents_narrow_the_roster() {
        let mut config = config_with_roster();
        narrow_to_named(&mut config, &["gamma".to_string()]);
        assert_eq!(config.agents, vec!["gamma".to_string()]);
        assert!(config.agents_file.is_none());
    }

    /// No flag, no change — the config's own selection stands.
    #[test]
    fn no_named_agents_leaves_the_config_alone() {
        let mut config = config_with_roster();
        narrow_to_named(&mut config, &[]);
        assert_eq!(config.agents.len(), 2);
        assert!(config.agents_file.is_some());
    }
}

#[cfg(test)]
mod thinking_effort_tests {
    use super::*;
    use misanthropic::prompt::output::Effort;

    #[test]
    fn a_named_level_reaches_the_config_and_a_typo_does_not_load() {
        let knobs: SeedKnobs = toml::from_str(r#"thinking_effort = "medium""#).unwrap();
        let config = knobs
            .to_config(std::path::Path::new("/tmp"), false)
            .unwrap();
        assert_eq!(config.thinking_effort, Some(Effort::Medium));

        assert!(toml::from_str::<SeedKnobs>(r#"thinking_effort = "meduim""#).is_err());
    }
}
