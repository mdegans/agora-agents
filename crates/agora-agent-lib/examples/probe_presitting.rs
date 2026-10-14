//! Pre-sitting canary: run every model the Council uses through the
//! distribution probe and compare each with its own baseline.
//!
//! A failure means "do not trust this pipeline today". Targets on one
//! blallama run one at a time (it has no queue); different servers and
//! Anthropic targets run in parallel; a 529 "Session is busy" is waited
//! out. Prints one line per target and exits 0 only if every target
//! passed (a missing baseline or a probe error is not a pass).
//!
//! ```sh
//! cargo run --example probe_presitting -- --config presitting.json --out presitting-report.json
//! ```
//!
//! `presitting.json`:
//! ```json
//! {
//!   "questionnaire": "crates/agora-agent-lib/probe/questionnaires/v0.json",
//!   "scenarios": "crates/agora-agent-lib/probe/scenarios/v0.json",
//!   "baseline": "crates/agora-agent-lib/probe/baselines/v0.json",
//!   "targets": [
//!     {"model": "Qwen3.8-27B-UD-Q8_K_XL.gguf", "endpoint": "http://192.168.0.123:11435"},
//!     {"model": "Qwen3.8-27B-UD-Q8_K_XL.gguf", "endpoint": "http://192.168.0.123:11435",
//!      "thinking": {"thinking": "adaptive"}},
//!     {"model": "claude-haiku-4-5", "api_key_file": "secrets/anthropic_api_key"}
//!   ]
//! }
//! ```

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context as _;
use clap::Parser;

use agora_agent_lib::probe::drift::DriftPolicy;
use agora_agent_lib::probe::indirect::ScenariosFile;
use agora_agent_lib::probe::presitting::{
    Instrument, PresittingOptions, Target, Verdict, run_presitting,
};
use agora_agent_lib::probe::{BaselineFile, Questionnaire};

#[derive(Parser, Debug)]
#[command(about = "Pre-sitting canary: every Council model against its own baseline.")]
struct Args {
    /// JSON config: questionnaire, scenarios (optional), baseline, targets.
    #[arg(long)]
    config: PathBuf,

    /// Write the full JSON report here.
    #[arg(long)]
    out: Option<PathBuf>,

    #[arg(long, default_value_t = DriftPolicy::default().max_item_ev_shift)]
    max_item_ev_shift: f64,

    #[arg(long, default_value_t = DriftPolicy::default().max_mean_abs_ev_shift)]
    max_mean_ev_shift: f64,

    #[arg(long, default_value_t = DriftPolicy::default().mode_flip_margin)]
    mode_flip_margin: f64,

    /// Seconds between retries on a 529 "busy".
    #[arg(long, default_value_t = 30)]
    busy_retry_secs: u64,

    /// Retries on 529 before the target is an error.
    #[arg(long, default_value_t = 120)]
    busy_retries: u32,

    /// Seconds to wait for a request's /probe session.
    #[arg(long, default_value_t = 600)]
    session_timeout_secs: u64,
}

#[derive(serde::Deserialize)]
struct Config {
    questionnaire: PathBuf,
    #[serde(default)]
    scenarios: Option<PathBuf>,
    baseline: PathBuf,
    targets: Vec<Target>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
    let args = Args::parse();
    let config: Config = serde_json::from_str(
        &std::fs::read_to_string(&args.config)
            .with_context(|| format!("reading {}", args.config.display()))?,
    )
    .context("parsing the config")?;

    let questionnaire = Questionnaire::from_json(&std::fs::read_to_string(&config.questionnaire)?)?;
    let scenarios = match &config.scenarios {
        Some(p) => Some(ScenariosFile::from_json(&std::fs::read_to_string(p)?)?),
        None => None,
    };
    let baseline = BaselineFile::load(&config.baseline)
        .with_context(|| format!("loading {}", config.baseline.display()))?;

    let mut instruments = vec![Instrument::Questionnaire(&questionnaire)];
    if let Some(file) = &scenarios {
        instruments.extend(file.scenarios.iter().map(Instrument::Scenario));
    }

    let mut opts = PresittingOptions {
        policy: DriftPolicy {
            max_item_ev_shift: args.max_item_ev_shift,
            max_mean_abs_ev_shift: args.max_mean_ev_shift,
            mode_flip_margin: args.mode_flip_margin,
        },
        busy_retry_every: Duration::from_secs(args.busy_retry_secs),
        busy_retries: args.busy_retries,
        ..PresittingOptions::default()
    };
    opts.distribution.session_timeout = Duration::from_secs(args.session_timeout_secs);

    let outcomes = run_presitting(&config.targets, &instruments, &baseline, &opts).await;
    for o in &outcomes {
        println!("{}", o.summary_line());
    }
    if let Some(path) = &args.out {
        std::fs::write(path, serde_json::to_string_pretty(&outcomes)?)
            .with_context(|| format!("writing {}", path.display()))?;
    }
    let all_pass = !outcomes.is_empty() && outcomes.iter().all(|o| o.verdict == Verdict::Pass);
    if !all_pass {
        std::process::exit(2);
    }
    Ok(())
}
