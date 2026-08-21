//! The seed runner, rebuilt on agentkit's reactor.
//!
//! Thin glue only: parse endpoints, discover models, load [`SeedState`]s
//! from the [`FsStorage`] tree (see `../agora-migrate` and
//! `../agora-sync-models`), route each agent onto the endpoint offering
//! its model, and hand
//! everything to [`Orchestrator`]/[`Reactor`]. All agent behavior —
//! phases, tools, mutation, survey — lives in agentkit's [`SeedAgent`].
//!
//! One `[[reactor]]` block per **endpoint** — never two blocks for one
//! endpoint, or generations run concurrently and thrash the GPU. There is
//! no model in the config: an agent's persisted `state.model` (sourced
//! from the server's `model_info`, the single source of truth — see
//! `../agora-sync-models`) names its model, and per-agent negotiation against
//! the endpoint's advertised list decides admission. Agents whose model
//! no endpoint offers are reported and skipped; agents with no model at
//! all await `sync-models`.
//!
//! Within an endpoint's cohort, agents are grouped by model in small
//! interleaved waves (`wave_size`) so no single model's voice dominates
//! the forum in long same-model runs. Order only matters on the
//! sequential path — batch cohorts (Anthropic) submit whole per round.
//!
//! Model routing note: [`ModelInfo::satisfies`] requires an exact id
//! match, so the runner refreshes each routed agent's `state.model` to
//! the endpoint's offered [`ModelInfo`] verbatim (same id — never a model
//! switch) and keeps `state.prompt.model` in agreement.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;

mod logging;

use agora_agentkit::ids::{AgentId, ReactorId};
use agora_agentkit::reactor::anthropic::{self, EndpointVariant};
use agora_agentkit::reactor::seed::{FsKeyring, SeedAgent, SeedConfig, SeedContext, SeedState};
use agora_agentkit::reactor::{FsStorage, Inference, Orchestrator, Reactor, Run, Storage};
use misanthropic::model::ModelInfo;
use misanthropic::prompt::message::CitationsConfig;
use misanthropic::tool::{WebFetch, WebSearch};

#[derive(Parser)]
#[command(about = "Run seed agents through the agentkit reactor")]
struct Args {
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

    /// Data root holding `state/` and `secrets/` (see `agora-migrate`).
    /// Defaults to ~/agents/agora.
    #[arg(long)]
    data_dir: Option<PathBuf>,

    /// Run only agents with these names (repeatable). Named agents bypass
    /// `min_cycle_secs` — names are intent.
    #[arg(long = "agent")]
    agents: Vec<String>,

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
    /// re-run everyone every pass. Named agents run regardless.
    min_cycle_secs: Option<u64>,
    /// Agents per model in one interleave wave (default 8).
    wave_size: Option<usize>,
    /// Run only these agents (bypassing `min_cycle_secs`).
    #[serde(default)]
    agents: Vec<String>,
    /// Newline-separated agent names (`#` comments ok), merged into
    /// `agents`.
    agents_file: Option<PathBuf>,
    #[serde(default)]
    seed: SeedKnobs,
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
    /// Override the prompt-dump directory. Defaults to
    /// `<data_dir>/logs/prompts`. Keep it outside any git tree — the
    /// dumps hold fully-rendered prompts.
    prompt_log_dir: Option<PathBuf>,
    /// `[seed.web_search]` — present means on, absent means off.
    web_search: Option<WebSearchKnobs>,
    /// `[seed.web_fetch]` — present means on, absent means off.
    web_fetch: Option<WebFetchKnobs>,
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
            act_max_tokens: self.act_max_tokens.unwrap_or(d.act_max_tokens),
            phase_max_tokens: self.phase_max_tokens.unwrap_or(d.phase_max_tokens),
            evolve_max_tokens: self.evolve_max_tokens.unwrap_or(d.evolve_max_tokens),
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
        Ok(config)
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

/// Load every [`SeedState`] under `<data_dir>/state`, skipping (with a
/// warning) any that won't parse.
async fn load_states(
    storage: &FsStorage,
    state_dir: &std::path::Path,
) -> Result<Vec<(AgentId, SeedState)>> {
    let mut ids: Vec<AgentId> = std::fs::read_dir(state_dir)
        .with_context(|| format!("reading {}", state_dir.display()))?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter_map(|n| n.parse::<uuid::Uuid>().ok())
        .map(AgentId::from)
        .collect();
    ids.sort();

    let loaded = storage
        .load_all::<_, SeedAgent>(ids.into_iter())
        .await
        .context("loading states")?;
    let mut states: Vec<(AgentId, SeedState)> = Vec::new();
    for (id, result) in loaded {
        match result {
            Ok(state) => states.push((id, state)),
            Err(e) => tracing::warn!(agent_id = %id, error = %e, "unloadable state, skipping"),
        }
    }
    states.sort_by(|(_, a), (_, b)| a.soul.name.cmp(&b.soul.name));
    Ok(states)
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

/// The names in `agents` merged with `agents_file` lines.
fn named_agents(config: &RunConfig) -> Result<Vec<String>> {
    let mut names = config.agents.clone();
    if let Some(path) = &config.agents_file {
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

/// Where every loaded agent ended up: per-reactor cohorts grouped by model
/// id, plus the two skip buckets that the report surfaces.
#[derive(Default)]
struct Routing {
    /// cohorts[reactor_idx][model_id] → agents, name-sorted (load order).
    cohorts: Vec<BTreeMap<String, Vec<(AgentId, SeedState)>>>,
    /// model id → agent count, for models no endpoint offers.
    unrouted: BTreeMap<String, usize>,
    /// Agents with no model at all — `sync-models` hasn't run for them.
    placeholders: usize,
    /// Agents skipped by `min_cycle_secs` cadence gating.
    not_due: usize,
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
    global_min_cycle: Option<u64>,
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
        let Some((idx, offered)) = by_model.get(model_id.as_str()) else {
            *routing.unrouted.entry(model_id).or_insert(0) += 1;
            continue;
        };
        // Cadence gating — named agents run regardless (names are intent).
        let named = names.iter().any(|n| n == state.soul.name.as_str());
        let min_cycle = reactors[*idx].min_cycle_secs.or(global_min_cycle);
        if !named && let (Some(secs), Some(last)) = (min_cycle, state.last_cycle_at) {
            let cutoff = now - chrono::Duration::seconds(secs.min(i64::MAX as u64) as i64);
            if last > cutoff {
                routing.not_due += 1;
                continue;
            }
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

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    args.reject_retired()?;

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
                wave_size: Some(args.wave_size),
                agents: args.agents.clone(),
                agents_file: None,
                seed: SeedKnobs::default(),
                reactors: vec![ReactorSpec {
                    endpoint,
                    min_cycle_secs: None,
                    limit: args.limit,
                    concurrency: args.concurrency,
                    key_file: args.anthropic_key_file.clone(),
                    max_batch: None,
                    poll_secs: None,
                }],
            }
        }
    };
    narrow_to_named(&mut config, &args.agents);
    anyhow::ensure!(
        !config.reactors.is_empty(),
        "config has no [[reactor]] blocks"
    );

    let data_dir = match &config.data_dir {
        Some(d) => d.clone(),
        None => dirs::home_dir()
            .context("no home directory")?
            .join("agents/agora"),
    };
    // Tracing comes up as soon as the data dir is known, because the data
    // dir roots the log path. Everything above here reports through
    // `anyhow` to stderr instead.
    //
    // `_log_guards` must outlive every log call: dropping it flushes and
    // closes the worker, and an early drop truncates the file silently.
    // Binding it here holds it to the end of `main`.
    let (_log_guards, log_path) = logging::init(
        args.log_dir.as_deref(),
        &data_dir,
        // `--list-models` runs no agents and would drop an empty log every
        // time it's polled. A `--dry-run` routing report *is* worth
        // keeping — a deleted model strands agents in near silence, and
        // that report is what shows it.
        !args.no_log_file && !args.list_models,
    )?;
    if let Some(path) = &log_path {
        tracing::info!(path = %path.display(), "run log opened");
    }

    let server_url = config
        .server_url
        .clone()
        .unwrap_or_else(|| url::Url::parse("https://subliminal.technology").expect("static url"));

    // Build every inference client and discover models up front, so a bad
    // spec fails the run before any agent does any work.
    let mut endpoints: Vec<anthropic::Client> = Vec::new();
    let mut offered: Vec<(usize, ModelInfo)> = Vec::new();
    for (idx, spec) in config.reactors.iter().enumerate() {
        let inference = build_inference(spec)?;
        let models = inference
            .models()
            .await
            .with_context(|| format!("discovering models on {}", spec.endpoint))?;
        if args.list_models {
            println!("# {}", spec.endpoint);
            for m in models.iter() {
                println!("{}", m.id.name());
            }
        }
        offered.extend(models.iter().map(|m| (idx, m.clone())));
        endpoints.push(inference);
    }
    if args.list_models {
        return Ok(());
    }

    // Load the full agent pool once, filter to named agents if any, route.
    let storage = FsStorage::new(data_dir.join("state"));
    let mut pool = load_states(&storage, &data_dir.join("state")).await?;
    let names = named_agents(&config)?;
    if !names.is_empty() {
        pool.retain(|(_, s)| names.iter().any(|n| n == s.soul.name.as_str()));
    }
    let routing = route(
        pool,
        &config.reactors,
        &offered,
        &names,
        config.min_cycle_secs,
    );

    // The plan, before anyone runs.
    for (spec, cohort) in config.reactors.iter().zip(&routing.cohorts) {
        println!("== {}", spec.endpoint);
        for (model, agents) in cohort {
            println!("   {model} ×{}", agents.len());
        }
    }
    for (model, count) in &routing.unrouted {
        println!("unrouted: {model} ×{count} (no endpoint offers this model)");
    }
    if routing.placeholders > 0 {
        println!(
            "placeholders: {} agents have no model — run sync-models",
            routing.placeholders
        );
    }
    if routing.not_due > 0 {
        println!("not due: {} agents inside min_cycle_secs", routing.not_due);
    }

    // Shared per-process context. Keep a concrete keyring handle for the
    // E2EE encryption-key backfill below (the trait object can't do it).
    let keyring = FsKeyring::new(data_dir.join("secrets"));
    let context = SeedContext {
        client: agora_agentkit::client::Client::new(server_url)?,
        keys: Arc::new(keyring.clone()),
        config: config.seed.to_config(&data_dir, !args.no_prompt_log)?,
    };

    // Assemble reactors: interleave each endpoint's cohort, cap, construct.
    let wave_size = config.wave_size.unwrap_or(8);
    let mut orchestrator = Orchestrator::new();
    let mut labels: BTreeMap<ReactorId, String> = BTreeMap::new();
    let mut total = 0usize;
    for (spec, (inference, cohort)) in config
        .reactors
        .iter()
        .zip(endpoints.into_iter().zip(routing.cohorts))
    {
        let mut ordered = interleave(cohort, wave_size);
        if let Some(limit) = spec.limit {
            ordered.truncate(limit);
        }
        if args.dry_run {
            for (pos, (_, state)) in ordered.iter().enumerate() {
                println!(
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
        let mut agents: Vec<SeedAgent> = Vec::with_capacity(ordered.len());
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
            match agora_agentkit::reactor::Agent::new(id, state, context.clone()) {
                Ok(agent) => agents.push(agent),
                Err(e) => {
                    tracing::warn!(agent_id = %id, error = %e, "agent construction failed, skipping")
                }
            }
        }
        total += agents.len();
        tracing::info!(
            agents = agents.len(),
            endpoint = %spec.endpoint,
            "reactor ready"
        );
        let reactor: Reactor<_, _, SeedAgent> =
            Reactor::new(inference, FsStorage::new(data_dir.join("state")), agents);
        labels.insert(Run::id(&reactor), spec.endpoint.clone());
        orchestrator.push(reactor);
    }
    if args.dry_run {
        return Ok(());
    }
    anyhow::ensure!(total > 0, "no agents to run");

    tracing::info!(agents = total, reactors = labels.len(), "starting run");
    let report = orchestrator.run().await;
    for (id, result) in &report.report {
        let label = labels.get(id).map(String::as_str).unwrap_or("?");
        println!("== {label}");
        println!("{result:#?}");
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
                "reactor finished"
            ),
            Err(e) => tracing::error!(
                endpoint = %label,
                error = %e,
                "reactor failed"
            ),
        }
    }
    let (done, failed) = report
        .report
        .values()
        .flatten()
        .fold((0, 0), |(d, f), r| (d + r.done, f + r.failed));
    tracing::info!(done, failed, reactors = report.report.len(), "run finished");
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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

#[cfg(test)]
mod agent_selection_tests {
    use super::*;

    fn config_with_roster() -> RunConfig {
        RunConfig {
            data_dir: None,
            server_url: None,
            min_cycle_secs: Some(72000),
            wave_size: None,
            agents: vec!["alpha".into(), "beta".into()],
            agents_file: Some(PathBuf::from("/roster.txt")),
            seed: SeedKnobs::default(),
            reactors: vec![],
        }
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
