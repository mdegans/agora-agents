//! Indirect probe — CLI harness.
//!
//! Sends fictional-scenario probes to a model endpoint and prints
//! the ratings (or pass/fail against a baseline). Mirrors
//! `probe_canary` but consumes a `ScenariosFile` and runs one or all
//! scenarios inside it.
//!
//! # Examples
//!
//! Against Anthropic (needs `ANTHROPIC_API_KEY` or `--api-key-file`):
//! ```sh
//! cargo run --example probe_canary_indirect -- \
//!     --scenarios crates/agora-agent-lib/probe/scenarios/v0.json \
//!     --baseline crates/agora-agent-lib/probe/baselines/indirect_v0.json \
//!     --model claude-haiku-4-5 --capture
//! ```
//!
//! Just one scenario:
//! ```sh
//! cargo run --example probe_canary_indirect -- \
//!     --scenarios .../scenarios/v0.json \
//!     --baseline .../baselines/indirect_v0.json \
//!     --scenario velkor_drummond \
//!     --model claude-haiku-4-5 --capture
//! ```

use std::path::PathBuf;

use anyhow::Context as _;
use chrono::Utc;
use clap::{Parser, ValueEnum};
use misanthropic::Client;

use agora_agent_lib::probe::{
    evaluate, indirect, BaselineEntry, BaselineFile, Questionnaire,
    QuestionnaireItem,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputMode {
    Human,
    Json,
}

#[derive(Parser, Debug)]
#[command(
    about = "Indirect alignment-drift probe. Sends fictional-scenario \
             claims to a model, optionally captures baselines, reports \
             pass/fail."
)]
struct Args {
    /// Local endpoint (e.g. http://192.168.0.123:11435). Omit to use
    /// Anthropic's API (reads ANTHROPIC_API_KEY).
    #[arg(long)]
    endpoint: Option<String>,

    /// Path to an Anthropic API key file. Ignored when --endpoint is set.
    #[arg(long)]
    api_key_file: Option<PathBuf>,

    /// Model id to pass in the prompt.
    #[arg(long, default_value = "claude-haiku-4-5")]
    model: String,

    /// Path to the scenarios JSON file.
    #[arg(long)]
    scenarios: PathBuf,

    /// Path to the indirect-baseline JSON file.
    #[arg(long)]
    baseline: PathBuf,

    /// Run a single scenario by id. Omit to run all scenarios in the
    /// file back-to-back.
    #[arg(long)]
    scenario: Option<String>,

    /// Output mode.
    #[arg(long, value_enum, default_value_t = OutputMode::Human)]
    output: OutputMode,

    /// Append each measurement to the baseline file as a new UNRATIFIED
    /// entry. Intended for detect-only bootstrapping.
    #[arg(long)]
    capture: bool,

    /// Tolerance used when writing new baseline entries via --capture.
    #[arg(long, default_value_t = 2)]
    capture_tolerance: u32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    let s_json = std::fs::read_to_string(&args.scenarios)
        .with_context(|| format!("reading {}", args.scenarios.display()))?;
    let scenarios_file = indirect::ScenariosFile::from_json(&s_json)
        .context("parsing scenarios JSON")?;

    let scenarios: Vec<&indirect::Scenario> = match args.scenario.as_deref() {
        Some(id) => {
            let s = scenarios_file.get(id).with_context(|| {
                format!(
                    "scenario id '{id}' not found in {}",
                    args.scenarios.display()
                )
            })?;
            vec![s]
        }
        None => scenarios_file.scenarios.iter().collect(),
    };

    let client = build_client(&args)?;

    let mut baseline_file = if args.baseline.exists() {
        BaselineFile::load(&args.baseline)?
    } else {
        BaselineFile::empty()
    };

    let mut any_fail = false;

    for scenario in &scenarios {
        let outcome = indirect::probe(&client, scenario, args.model.clone())
            .await
            .with_context(|| format!("probe failed for scenario '{}'", scenario.id))?;

        let key = scenario.baseline_key();

        // Build a lightweight pseudo-Questionnaire for evaluate() —
        // evaluate takes a Questionnaire only to read version +
        // item count. We reuse the direct probe's report/evaluate
        // machinery unchanged by synthesizing a Questionnaire here.
        let fake_questionnaire = Questionnaire {
            version: key.clone(),
            description: scenario.description.clone(),
            instructions: scenario.instructions.clone(),
            items: scenario
                .items
                .iter()
                .map(|it| QuestionnaireItem {
                    id: it.id.clone(),
                    statement: it.claim.clone(),
                })
                .collect(),
        };

        let report_opt = baseline_file
            .get(&outcome.model_id, &key)
            .map(|b| evaluate(&outcome, b, &fake_questionnaire))
            .transpose()?;

        if args.capture {
            let entry = BaselineEntry {
                model_id: outcome.model_id.clone(),
                questionnaire_version: key.clone(),
                ratified_at: Utc::now(),
                council_decision_id: None,
                tolerance_per_item: args.capture_tolerance,
                answers: outcome.answers.clone(),
            };
            baseline_file.entries.push(entry);
            baseline_file.save(&args.baseline)?;
            eprintln!(
                "[capture] appended unratified baseline for model={} \
                 scenario={} to {}. Council ratification REQUIRED before \
                 this baseline has governance weight.",
                outcome.model_id,
                key,
                args.baseline.display()
            );
        }

        match args.output {
            OutputMode::Human => {
                print_human_scenario(scenario, &outcome, report_opt.as_ref());
            }
            OutputMode::Json => {
                if let Some(report) = report_opt.as_ref() {
                    println!("{}", serde_json::to_string(report)?);
                } else {
                    let payload = serde_json::json!({
                        "pass": null,
                        "reason": "no baseline for (model, scenario_version)",
                        "model_id": outcome.model_id,
                        "scenario_id": scenario.id,
                        "scenario_version": scenario.version,
                        "probed_at": outcome.probed_at,
                        "measured_ratings": outcome.answers.ratings.iter()
                            .map(|r| r.rating).collect::<Vec<_>>(),
                        "item_ids": scenario.items.iter()
                            .map(|i| i.id.clone()).collect::<Vec<_>>(),
                        "axes": scenario.items.iter()
                            .map(|i| i.axis.clone()).collect::<Vec<_>>(),
                        "input_tokens": outcome.usage.input_tokens,
                        "output_tokens": outcome.usage.output_tokens,
                    });
                    println!("{}", serde_json::to_string(&payload)?);
                }
            }
        }

        if let Some(r) = report_opt.as_ref() {
            if !r.pass {
                any_fail = true;
            }
        }
    }

    if any_fail {
        std::process::exit(2);
    }
    Ok(())
}

fn build_client(args: &Args) -> anyhow::Result<Client> {
    if let Some(endpoint) = args.endpoint.as_ref() {
        let placeholder_key = "x".repeat(108);
        let client = Client::new(placeholder_key)
            .context("constructing local client")?
            .with_base_url(endpoint)
            .context("setting base_url for local endpoint")?;
        Ok(client)
    } else {
        let key = match args.api_key_file.as_ref() {
            Some(path) => std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?
                .trim()
                .to_string(),
            None => std::env::var("ANTHROPIC_API_KEY").context(
                "ANTHROPIC_API_KEY not set; pass --api-key-file or --endpoint",
            )?,
        };
        Client::new(key).context("constructing Anthropic client")
    }
}

fn print_human_scenario(
    scenario: &indirect::Scenario,
    outcome: &agora_agent_lib::probe::ProbeOutcome,
    report: Option<&agora_agent_lib::probe::ProbeReport>,
) {
    println!();
    println!(
        "indirect_probe scenario={} {} | model: {} | {}",
        scenario.id,
        scenario.version,
        outcome.model_id,
        outcome.probed_at.to_rfc3339()
    );
    println!();

    let id_w = scenario
        .items
        .iter()
        .map(|i| i.id.len())
        .max()
        .unwrap_or(0)
        .max(4);
    let axis_w = scenario
        .items
        .iter()
        .map(|i| i.axis.len())
        .max()
        .unwrap_or(0)
        .max(4);

    if let Some(report) = report {
        println!(
            "{:<id_w$}  {:<axis_w$}  {:>8}  {:>8}  {:>6}",
            "ITEM", "AXIS", "MEASURED", "BASELINE", "DELTA"
        );
        for (i, item) in scenario.items.iter().enumerate() {
            let m = report.measured_ratings.get(i).copied().unwrap_or(0);
            let b = report.baseline_ratings.get(i).copied().unwrap_or(0);
            let d = report.per_item_delta.get(i).copied().unwrap_or(0);
            let mark = if d.unsigned_abs() > report.tolerance_per_item {
                "  *** DRIFT"
            } else {
                ""
            };
            println!(
                "{:<id_w$}  {:<axis_w$}  {:>8}  {:>8}  {:>+6}{}",
                item.id, item.axis, m, b, d, mark
            );
        }
        println!();
        println!(
            "max_abs_delta: {}  |  tolerance: {}  |  {}",
            report.max_abs_delta,
            report.tolerance_per_item,
            if report.pass { "PASS" } else { "FAIL" }
        );
    } else {
        println!(
            "{:<id_w$}  {:<axis_w$}  {:>8}",
            "ITEM", "AXIS", "MEASURED"
        );
        for (i, item) in scenario.items.iter().enumerate() {
            let m = outcome
                .answers
                .ratings
                .get(i)
                .map(|r| r.rating)
                .unwrap_or(0);
            println!(
                "{:<id_w$}  {:<axis_w$}  {:>8}",
                item.id, item.axis, m
            );
        }
        println!();
        println!(
            "(no baseline for model={} scenario={} — run with --capture to seed one)",
            outcome.model_id,
            scenario.baseline_key()
        );
    }
    println!();
    println!(
        "tokens: {} in / {} out",
        outcome.usage.input_tokens, outcome.usage.output_tokens
    );
}
