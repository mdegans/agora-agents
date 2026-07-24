//! The seed runner, rebuilt on agentkit's reactor.
//!
//! Thin glue only: parse endpoints, discover models, load [`SeedState`]s
//! from the [`FsStorage`] tree (see `../migrate`), route each agent onto a
//! concrete endpoint model, and hand everything to
//! [`Orchestrator`]/[`Reactor`]. All agent behavior — phases, tools,
//! mutation, survey — lives in agentkit's [`SeedAgent`].
//!
//! Two invocation shapes:
//! - `--endpoint`/`--model` flags: one reactor, ad-hoc runs.
//! - `--config run.toml`: one reactor per `[[reactor]]` block, all pushed
//!   into one [`Orchestrator`] and run concurrently. Reactors with explicit
//!   `agents` lists claim their cohorts first; open reactors (no list)
//!   then claim what remains, in file order.
//!
//! Model routing note: [`ModelInfo::satisfies`] requires an exact id match,
//! so every agent's `state.model` must name a model the endpoint actually
//! offers (and `state.prompt.model` must agree) or `negotiate` rejects it.
//! Assignment happens here, after discovery.

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

    /// Model id to assign to the cohort. Must be offered by the endpoint;
    /// see `--list-models`.
    #[arg(long)]
    model: Option<String>,

    /// List the endpoint's models and exit (requires --endpoint).
    #[arg(long)]
    list_models: bool,

    /// Agora server base URL.
    #[arg(long, default_value = "https://subliminal.technology")]
    server_url: url::Url,

    /// Data root holding `state/` and `secrets/` (see `agora-migrate`).
    /// Defaults to ~/agents/agora.
    #[arg(long)]
    data_dir: Option<PathBuf>,

    /// Run only agents with these names (repeatable).
    #[arg(long = "agent")]
    agents: Vec<String>,

    /// Run at most this many agents (after name filtering, sorted by name).
    #[arg(long)]
    limit: Option<usize>,

    /// Concurrent in-flight requests (blallama slots / API rate headroom).
    #[arg(long, default_value_t = 1)]
    concurrency: usize,

    /// API key file, required for the anthropic endpoint.
    #[arg(long)]
    anthropic_key_file: Option<PathBuf>,

    // SeedConfig knobs, classic defaults.
    #[arg(long, default_value_t = 5)]
    max_rounds: usize,
    #[arg(long, default_value_t = 3)]
    mutation_chance: u32,
    #[arg(long, default_value_t = 10)]
    evolution_chance: u32,
    #[arg(long, default_value_t = 10)]
    survey_chance: u32,
    #[arg(long)]
    force_survey: bool,
}

/// One `[[reactor]]` block: an endpoint, the model its cohort runs on, and
/// which agents it claims.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReactorSpec {
    /// `anthropic://…`, `blallama://host:port`, or `ollama://host:port`.
    endpoint: String,
    /// Model id offered by the endpoint (exact match).
    model: String,
    /// Agent names this reactor claims. Empty/absent = open reactor: claims
    /// every agent no named reactor took.
    #[serde(default)]
    agents: Vec<String>,
    /// Newline-separated agent names (`#` comments ok), merged into `agents`.
    agents_file: Option<PathBuf>,
    /// Model family this reactor's model belongs to (e.g. "cogito",
    /// "qwen", "gpt-oss"). Sticky claims match any persisted model id
    /// starting with this prefix (case-insensitive), so an agent follows
    /// its family across parameter counts, quants, and endpoint renames —
    /// but never crosses families. Absent = exact-id stickiness only.
    family: Option<String>,
    /// Claim agents that have no established model yet (fresh migrations,
    /// placeholder ids). Without this, an unnamed reactor only claims
    /// agents whose persisted model already matches `model`/`family` —
    /// agents never switch models implicitly.
    #[serde(default)]
    adopt: bool,
    /// Cap the cohort (after name claim, sorted by name).
    limit: Option<usize>,
    /// Concurrent in-flight requests on the sequential path.
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
}

impl SeedKnobs {
    fn to_config(&self) -> SeedConfig {
        let d = SeedConfig::default();
        SeedConfig {
            max_rounds: self.max_rounds.unwrap_or(d.max_rounds),
            mutation_chance: self.mutation_chance.unwrap_or(d.mutation_chance),
            evolution_chance: self.evolution_chance.unwrap_or(d.evolution_chance),
            survey_chance: self.survey_chance.unwrap_or(d.survey_chance),
            force_survey: self.force_survey.unwrap_or(d.force_survey),
            ..d
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

/// Build the [`anthropic::Client`] for a spec: key handling by variant,
/// concurrency, and batch tuning.
fn build_inference(spec: &ReactorSpec) -> Result<anthropic::Client> {
    let (variant, base_url) = parse_endpoint(&spec.endpoint)?;
    let key = match variant {
        EndpointVariant::Anthropic => {
            let path = spec.key_file.as_ref().with_context(|| {
                format!("reactor {}: key_file is required for anthropic endpoints", spec.endpoint)
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
async fn load_states(storage: &FsStorage, state_dir: &std::path::Path) -> Result<Vec<(AgentId, SeedState)>> {
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

/// The names a spec claims: `agents` merged with `agents_file` lines.
fn claimed_names(spec: &ReactorSpec) -> Result<Vec<String>> {
    let mut names = spec.agents.clone();
    if let Some(path) = &spec.agents_file {
        let body = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        names.extend(
            body.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(String::from),
        );
    }
    Ok(names)
}

/// How a claim pass selects agents from the pool. Model stickiness lives
/// here: an agent's persisted `state.model` is part of its identity, so
/// only an explicit by-name claim may move it to a different model.
enum Claim<'a> {
    /// Explicitly listed names — the one path that may switch a model
    /// (warned, never silent).
    Named(&'a [String]),
    /// Agents whose persisted model id equals the spec's model, or —
    /// when the spec names a `family` — starts with that family prefix
    /// (case-insensitive).
    Sticky {
        model: &'a str,
        family: Option<&'a str>,
    },
    /// Agents with no established model (empty/placeholder id).
    Unassigned,
}

/// Take up to `limit` matching agents out of `pool`.
fn claim(
    pool: &mut Vec<(AgentId, SeedState)>,
    how: Claim<'_>,
    limit: Option<usize>,
) -> Vec<(AgentId, SeedState)> {
    let mut cohort: Vec<(AgentId, SeedState)> = Vec::new();
    let mut rest = Vec::with_capacity(pool.len());
    for entry in pool.drain(..) {
        let matched = match how {
            Claim::Named(names) => {
                names.iter().any(|n| n == entry.1.soul.name.as_str())
            }
            Claim::Sticky { model, family } => {
                let id = entry.1.model.id.name();
                id == model
                    || family.is_some_and(|f| {
                        id.to_lowercase().starts_with(&f.to_lowercase())
                    })
            }
            Claim::Unassigned => entry.1.model.id.name().is_empty(),
        };
        if matched {
            cohort.push(entry);
        } else {
            rest.push(entry);
        }
    }
    *pool = rest;
    if let Some(limit) = limit {
        // Give unclaimed overflow back to the pool for later specs.
        let overflow = cohort.split_off(limit.min(cohort.len()));
        pool.extend(overflow);
        pool.sort_by(|(_, a), (_, b)| a.soul.name.cmp(&b.soul.name));
    }
    cohort
}

/// Route a cohort onto `model` and construct the [`SeedAgent`]s.
/// `negotiate` matches by exact id, so `state.model` must be the offered
/// [`ModelInfo`] verbatim; `Agent::new` keeps `prompt.model` in agreement.
fn build_agents(
    cohort: Vec<(AgentId, SeedState)>,
    model: &ModelInfo,
    context: &SeedContext,
) -> Vec<SeedAgent> {
    let mut agents: Vec<SeedAgent> = Vec::with_capacity(cohort.len());
    for (id, mut state) in cohort {
        state.model = model.clone();
        state.prompt.model = model.id.clone();
        match agora_agentkit::reactor::Agent::new(id, state, context.clone()) {
            Ok(agent) => agents.push(agent),
            Err(e) => {
                tracing::warn!(agent_id = %id, error = %e, "agent construction failed, skipping")
            }
        }
    }
    agents
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = Args::parse();

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
                seed: SeedKnobs {
                    max_rounds: Some(args.max_rounds),
                    mutation_chance: Some(args.mutation_chance),
                    evolution_chance: Some(args.evolution_chance),
                    survey_chance: Some(args.survey_chance),
                    force_survey: Some(args.force_survey),
                },
                reactors: vec![ReactorSpec {
                    endpoint,
                    // --list-models doesn't need one; checked below otherwise.
                    model: args.model.clone().unwrap_or_default(),
                    agents: args.agents.clone(),
                    agents_file: None,
                    family: None,
                    // The flags path is explicitly ad-hoc: no name list
                    // means "run whatever is here", stickiness aside.
                    adopt: true,
                    limit: args.limit,
                    concurrency: args.concurrency,
                    key_file: args.anthropic_key_file.clone(),
                    max_batch: None,
                    poll_secs: None,
                }],
            }
        }
    };
    anyhow::ensure!(!config.reactors.is_empty(), "config has no [[reactor]] blocks");

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
    let mut endpoints: Vec<(anthropic::Client, ModelInfo)> = Vec::new();
    for spec in &config.reactors {
        let inference = build_inference(spec)?;
        let models = inference.models().await.with_context(|| {
            format!("discovering models on {}", spec.endpoint)
        })?;
        if args.list_models {
            println!("# {}", spec.endpoint);
            for m in models.iter() {
                println!("{}", m.id.name());
            }
            continue;
        }
        anyhow::ensure!(
            !spec.model.is_empty(),
            "reactor {}: model is required (see --list-models)",
            spec.endpoint
        );
        let model: ModelInfo = models
            .iter()
            .find(|m| m.id.name() == spec.model)
            .with_context(|| {
                format!(
                    "{} does not offer {:?}; see --list-models",
                    spec.endpoint, spec.model
                )
            })?
            .clone();
        endpoints.push((inference, model));
    }
    if args.list_models {
        return Ok(());
    }

    // Load the full agent pool once; reactors claim from it.
    let storage = FsStorage::new(data_dir.join("state"));
    let mut pool = load_states(&storage, &data_dir.join("state")).await?;

    // Shared per-process context.
    let context = SeedContext {
        client: agora_agentkit::client::Client::new(server_url)?,
        keys: Arc::new(FsKeyring::new(data_dir.join("secrets"))),
        config: config.seed.to_config(),
    };

    // Claim cohorts in three passes over all specs (config order within
    // each): explicit names first, then family/model stickiness, then
    // adoption of unassigned agents. Only a by-name claim may move an
    // agent off its established model family — and never silently.
    let mut cohorts: Vec<Vec<(AgentId, SeedState)>> =
        (0..config.reactors.len()).map(|_| Vec::new()).collect();
    for (i, spec) in config.reactors.iter().enumerate() {
        let names = claimed_names(spec)?;
        if names.is_empty() {
            continue;
        }
        let cohort = claim(&mut pool, Claim::Named(&names), spec.limit);
        for (_, state) in &cohort {
            let id = state.model.id.name();
            if !id.is_empty() && id != spec.model {
                tracing::warn!(
                    agent = %state.soul.name,
                    from = %id,
                    to = %spec.model,
                    "explicit claim switches an agent's model"
                );
            }
        }
        cohorts[i] = cohort;
    }
    for (i, spec) in config.reactors.iter().enumerate() {
        let sticky = claim(
            &mut pool,
            Claim::Sticky {
                model: &spec.model,
                family: spec.family.as_deref(),
            },
            spec.limit.map(|l| l.saturating_sub(cohorts[i].len())),
        );
        cohorts[i].extend(sticky);
    }
    for (i, spec) in config.reactors.iter().enumerate() {
        if spec.adopt {
            let adopted = claim(
                &mut pool,
                Claim::Unassigned,
                spec.limit.map(|l| l.saturating_sub(cohorts[i].len())),
            );
            cohorts[i].extend(adopted);
        }
    }

    // Assemble reactors and label them for the report.
    let mut orchestrator = Orchestrator::new();
    let mut labels: BTreeMap<ReactorId, String> = BTreeMap::new();
    let mut total = 0usize;
    for ((spec, (inference, model)), cohort) in
        config.reactors.iter().zip(endpoints).zip(cohorts)
    {
        if cohort.is_empty() {
            tracing::warn!(endpoint = %spec.endpoint, "no agents claimed, skipping reactor");
            continue;
        }
        let agents = build_agents(cohort, &model, &context);
        total += agents.len();
        tracing::info!(
            agents = agents.len(),
            model = model.id.name(),
            endpoint = %spec.endpoint,
            "reactor ready"
        );
        let reactor: Reactor<_, _, SeedAgent> =
            Reactor::new(inference, FsStorage::new(data_dir.join("state")), agents);
        labels.insert(
            Run::id(&reactor),
            format!("{} [{}]", spec.endpoint, model.id.name()),
        );
        orchestrator.push(reactor);
    }
    anyhow::ensure!(total > 0, "no agents to run");

    tracing::info!(agents = total, reactors = labels.len(), "starting run");
    let report = orchestrator.run().await;
    for (id, result) in &report.report {
        let label = labels.get(id).map(String::as_str).unwrap_or("?");
        println!("== {label}");
        println!("{result:#?}");
    }
    Ok(())
}
