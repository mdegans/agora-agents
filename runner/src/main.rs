//! The seed runner, rebuilt on agentkit's reactor.
//!
//! Thin glue only: parse endpoints, discover models, load [`SeedState`]s
//! from the [`FsStorage`] tree (see `../migrate`), route each agent onto a
//! concrete endpoint model, and hand everything to
//! [`Orchestrator`]/[`Reactor`]. All agent behavior — phases, tools,
//! mutation, survey — lives in agentkit's [`SeedAgent`].
//!
//! Model routing note: [`ModelInfo::satisfies`] requires an exact id match,
//! so every agent's `state.model` must name a model the endpoint actually
//! offers (and `state.prompt.model` must agree) or `negotiate` rejects it.
//! Assignment happens here, after discovery.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;

use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::anthropic::{self, EndpointVariant};
use agora_agentkit::reactor::seed::{FsKeyring, SeedAgent, SeedConfig, SeedContext, SeedState};
use agora_agentkit::reactor::{FsStorage, Inference, Orchestrator, Reactor, Storage};
use misanthropic::model::ModelInfo;

/// Valid-length placeholder for local endpoints; `with_variant` swaps in
/// agentkit's own dummy before any request is made.
const LOCAL_KEY: &str = "sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";

#[derive(Parser)]
#[command(about = "Run seed agents through the agentkit reactor")]
struct Args {
    /// Inference endpoint: `anthropic://api.anthropic.com`,
    /// `blallama://host:port`, or `ollama://host:port`.
    #[arg(long)]
    endpoint: String,

    /// Model id to assign to the cohort. Must be offered by the endpoint;
    /// see `--list-models`.
    #[arg(long)]
    model: Option<String>,

    /// List the endpoint's models and exit.
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

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = Args::parse();

    let data_dir = match &args.data_dir {
        Some(d) => d.clone(),
        None => dirs::home_dir()
            .context("no home directory")?
            .join("agents/agora"),
    };

    // Inference client for the endpoint.
    let (variant, base_url) = parse_endpoint(&args.endpoint)?;
    let key = match variant {
        EndpointVariant::Anthropic => {
            let path = args
                .anthropic_key_file
                .as_ref()
                .context("--anthropic-key-file is required for anthropic endpoints")?;
            let key =
                zeroize::Zeroizing::new(std::fs::read_to_string(path).context("reading key file")?);
            key.trim().to_string()
        }
        _ => LOCAL_KEY.to_string(),
    };
    let m_client = misanthropic::Client::new(key)
        .map_err(|e| anyhow::anyhow!("api key: {e}"))?
        .base_url(base_url.as_str())?;
    let inference = anthropic::Client::new(m_client)
        .with_variant(variant)
        .with_concurrency(NonZeroUsize::new(args.concurrency).context("concurrency must be > 0")?);

    // Model discovery / selection.
    let models = inference.models().await?;
    if args.list_models {
        for m in models.iter() {
            println!("{}", m.id.name());
        }
        return Ok(());
    }
    let model_id = args
        .model
        .as_deref()
        .context("--model is required (see --list-models)")?;
    let model: ModelInfo = models
        .iter()
        .find(|m| m.id.name() == model_id)
        .with_context(|| format!("endpoint does not offer {model_id:?}; see --list-models"))?
        .clone();

    // Load states.
    let storage = FsStorage::new(data_dir.join("state"));
    let mut ids: Vec<AgentId> = std::fs::read_dir(data_dir.join("state"))
        .with_context(|| format!("reading {}", data_dir.display()))?
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
    if !args.agents.is_empty() {
        states.retain(|(_, s)| args.agents.iter().any(|n| n == s.soul.name.as_str()));
    }
    states.sort_by(|(_, a), (_, b)| a.soul.name.cmp(&b.soul.name));
    if let Some(limit) = args.limit {
        states.truncate(limit);
    }
    anyhow::ensure!(!states.is_empty(), "no agents to run");

    // Shared per-process context.
    let context = SeedContext {
        client: agora_agentkit::client::Client::new(args.server_url.clone())?,
        keys: Arc::new(FsKeyring::new(data_dir.join("secrets"))),
        config: SeedConfig {
            max_rounds: args.max_rounds,
            mutation_chance: args.mutation_chance,
            evolution_chance: args.evolution_chance,
            survey_chance: args.survey_chance,
            force_survey: args.force_survey,
            ..SeedConfig::default()
        },
    };

    // Route the cohort onto the selected model and build agents.
    // `negotiate` matches by exact id, so state.model must be the offered
    // ModelInfo verbatim; Agent::new keeps prompt.model in agreement.
    let mut agents: Vec<SeedAgent> = Vec::with_capacity(states.len());
    for (id, mut state) in states {
        state.model = model.clone();
        state.prompt.model = model.id.clone();
        match agora_agentkit::reactor::Agent::new(id, state, context.clone()) {
            Ok(agent) => agents.push(agent),
            Err(e) => {
                tracing::warn!(agent_id = %id, error = %e, "agent construction failed, skipping")
            }
        }
    }
    tracing::info!(
        agents = agents.len(),
        model = model.id.name(),
        endpoint = %base_url,
        "starting run"
    );

    let reactor: Reactor<_, _, SeedAgent> = Reactor::new(inference, storage, agents);
    let mut orchestrator = Orchestrator::new();
    orchestrator.push(reactor);
    let report = orchestrator.run().await;
    println!("{report:#?}");
    Ok(())
}
