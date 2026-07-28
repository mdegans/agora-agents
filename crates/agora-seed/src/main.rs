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

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;

use agora_agentkit::ids::{AgentId, ReactorId};
use agora_agentkit::reactor::anthropic::{self, EndpointVariant};
use agora_agentkit::reactor::seed::{FsKeyring, SeedAgent, SeedConfig, SeedContext, SeedState};
use agora_agentkit::reactor::{FsStorage, Inference, Orchestrator, Reactor, Run, Storage};
use misanthropic::model::ModelInfo;

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
        "gone — tracing goes to stderr. Redirect it if you want a file.",
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
#[derive(Deserialize, Default)]
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
}

impl SeedKnobs {
    fn to_config(&self) -> Result<SeedConfig> {
        let d = SeedConfig::default();
        let config = SeedConfig {
            max_rounds: self.max_rounds.unwrap_or(d.max_rounds),
            mutation_chance: self.mutation_chance.unwrap_or(d.mutation_chance),
            evolution_chance: self.evolution_chance.unwrap_or(d.evolution_chance),
            survey_chance: self.survey_chance.unwrap_or(d.survey_chance),
            force_survey: self.force_survey.unwrap_or(d.force_survey),
            act_max_tokens: self.act_max_tokens.unwrap_or(d.act_max_tokens),
            phase_max_tokens: self.phase_max_tokens.unwrap_or(d.phase_max_tokens),
            evolve_max_tokens: self.evolve_max_tokens.unwrap_or(d.evolve_max_tokens),
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
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = Args::parse();
    args.reject_retired()?;

    // Normalize both invocation shapes to a RunConfig.
    let config: RunConfig = match &args.config {
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

    // Shared per-process context.
    let context = SeedContext {
        client: agora_agentkit::client::Client::new(server_url)?,
        keys: Arc::new(FsKeyring::new(data_dir.join("secrets"))),
        config: config.seed.to_config()?,
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
    }
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
