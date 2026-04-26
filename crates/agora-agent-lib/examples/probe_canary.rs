//! Alignment-drift canary probe — CLI harness.
//!
//! Sends a Council-ratified (or pre-ratification seed) questionnaire
//! to a model endpoint and prints pass/fail against a baseline. Can
//! also capture a new baseline entry (`--capture`) during the
//! detect-only bootstrapping phase.
//!
//! # Examples
//!
//! Against Anthropic (needs `ANTHROPIC_API_KEY`):
//! ```sh
//! cargo run --example probe_canary -- \
//!     --questionnaire crates/agora-agent-lib/probe/questionnaires/v0.json \
//!     --baseline crates/agora-agent-lib/probe/baselines/v0.json \
//!     --model claude-haiku-4-5 --capture
//! ```
//!
//! Against a local drama_llama server:
//! ```sh
//! cargo run --example probe_canary -- \
//!     --endpoint http://192.168.0.123:11435 \
//!     --model cogito-32b.gguf \
//!     --questionnaire crates/agora-agent-lib/probe/questionnaires/v0.json \
//!     --baseline crates/agora-agent-lib/probe/baselines/v0.json \
//!     --capture
//! ```

use std::path::PathBuf;

use anyhow::Context as _;
use chrono::Utc;
use clap::{Parser, ValueEnum};
use misanthropic::Client;

use agora_agent_lib::probe::{BaselineEntry, BaselineFile, Questionnaire, evaluate, probe};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputMode {
    Human,
    Json,
}

#[derive(Parser, Debug)]
#[command(about = "Alignment-drift canary probe. Sends a questionnaire to a \
             model, optionally captures a baseline, reports pass/fail.")]
struct Args {
    /// Local endpoint (e.g. http://192.168.0.123:11435). Omit to use
    /// Anthropic's API (reads ANTHROPIC_API_KEY).
    #[arg(long)]
    endpoint: Option<String>,

    /// Path to an Anthropic API key file. Ignored when --endpoint is set.
    #[arg(long)]
    api_key_file: Option<PathBuf>,

    /// Model id to pass in the prompt. For Anthropic: model slug
    /// (e.g. `claude-haiku-4-5`). For drama_llama: GGUF filename
    /// (e.g. `cogito-32b.gguf`).
    #[arg(long, default_value = "claude-haiku-4-5")]
    model: String,

    /// Path to the questionnaire JSON.
    #[arg(long)]
    questionnaire: PathBuf,

    /// Path to the baseline JSON file.
    #[arg(long)]
    baseline: PathBuf,

    /// Output mode.
    #[arg(long, value_enum, default_value_t = OutputMode::Human)]
    output: OutputMode,

    /// Append the measurement to the baseline file as a new UNRATIFIED
    /// entry. Intended for detect-only bootstrapping; the entry carries
    /// `council_decision_id: null` until governance action assigns one.
    #[arg(long)]
    capture: bool,

    /// Tolerance used when writing a new baseline entry via --capture.
    /// Ignored otherwise.
    #[arg(long, default_value_t = 2)]
    capture_tolerance: u32,

    /// Path the response came through, recorded on each captured
    /// baseline entry. Default: `"anthropic_api"` when --endpoint is
    /// omitted, otherwise `"self_hosted_drama_llama"`. Override
    /// explicitly when running through Together, Fireworks, etc., or
    /// when distinguishing pre-fix vs post-fix wrapper captures.
    #[arg(long)]
    provider_source: Option<String>,
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

    // Load questionnaire.
    let q_json = std::fs::read_to_string(&args.questionnaire)
        .with_context(|| format!("reading {}", args.questionnaire.display()))?;
    let questionnaire = Questionnaire::from_json(&q_json).context("parsing questionnaire JSON")?;

    // Construct client.
    let client = build_client(&args)?;

    // Run the probe.
    let outcome = probe(&client, &questionnaire, args.model.clone()).await?;

    // Load baseline.
    let mut baseline_file = if args.baseline.exists() {
        BaselineFile::load(&args.baseline)?
    } else {
        BaselineFile::empty()
    };

    // If a matching baseline exists, evaluate against it.
    let report_opt = baseline_file
        .get(&outcome.model_id, &questionnaire.version)
        .map(|b| evaluate(&outcome, b, &questionnaire))
        .transpose()?;

    // --capture: append a new unratified entry.
    if args.capture {
        let provider_source = args.provider_source.clone().unwrap_or_else(|| {
            if args.endpoint.is_some() {
                "self_hosted_drama_llama".to_string()
            } else {
                "anthropic_api".to_string()
            }
        });
        let entry = BaselineEntry {
            model_id: outcome.model_id.clone(),
            questionnaire_version: questionnaire.version.clone(),
            provider_source: provider_source.clone(),
            capture_date: Utc::now(),
            ratified_at: None,
            council_decision_id: None,
            tolerance_per_item: args.capture_tolerance,
            answers: outcome.answers.clone(),
        };
        baseline_file.entries.push(entry);
        baseline_file.save(&args.baseline)?;
        eprintln!(
            "[capture] appended unratified baseline for model={} version={} \
             provider={} to {}. Council ratification REQUIRED before this \
             baseline has governance weight.",
            outcome.model_id,
            questionnaire.version,
            provider_source,
            args.baseline.display()
        );
    }

    // Print.
    match args.output {
        OutputMode::Human => print_human(&outcome, &questionnaire, report_opt.as_ref()),
        OutputMode::Json => {
            if let Some(report) = report_opt.as_ref() {
                println!("{}", serde_json::to_string(report)?);
            } else {
                // No baseline to compare; emit the raw outcome so the
                // caller has structured data either way.
                let payload = serde_json::json!({
                    "pass": null,
                    "reason": "no baseline for (model, version)",
                    "model_id": outcome.model_id,
                    "questionnaire_version": questionnaire.version,
                    "probed_at": outcome.probed_at,
                    "measured_ratings": outcome.answers.ratings.iter()
                        .map(|r| r.rating).collect::<Vec<_>>(),
                    "input_tokens": outcome.usage.input_tokens,
                    "output_tokens": outcome.usage.output_tokens,
                });
                println!("{}", serde_json::to_string(&payload)?);
            }
        }
    }

    // Non-zero exit on fail — makes this scriptable as a gate.
    if let Some(report) = report_opt.as_ref()
        && !report.pass
    {
        std::process::exit(2);
    }
    Ok(())
}

fn build_client(args: &Args) -> anyhow::Result<Client> {
    if let Some(endpoint) = args.endpoint.as_ref() {
        // Local server; key not validated against Anthropic. Use a
        // fixed placeholder that satisfies the 108-byte length check.
        let placeholder_key = "x".repeat(108);
        let client = Client::new(placeholder_key)
            .context("constructing local client")?
            .with_base_url(endpoint)
            .context("setting base_url for local endpoint")?;
        Ok(client)
    } else {
        // Anthropic — env var or file.
        let key = match args.api_key_file.as_ref() {
            Some(path) => std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?
                .trim()
                .to_string(),
            None => std::env::var("ANTHROPIC_API_KEY").context(
                "ANTHROPIC_API_KEY not set; pass --api-key-file or \
                 --endpoint",
            )?,
        };
        Client::new(key).context("constructing Anthropic client")
    }
}

fn print_human(
    outcome: &agora_agent_lib::probe::ProbeOutcome,
    questionnaire: &Questionnaire,
    report: Option<&agora_agent_lib::probe::ProbeReport>,
) {
    println!();
    println!(
        "probe_canary {} | model: {} | {}",
        questionnaire.version,
        outcome.model_id,
        outcome.probed_at.to_rfc3339()
    );
    println!();

    let id_w = questionnaire
        .items
        .iter()
        .map(|i| i.id.len())
        .max()
        .unwrap_or(0)
        .max(4);
    if let Some(report) = report {
        println!(
            "{:<id_w$}  {:>8}  {:>8}  {:>6}",
            "ITEM", "MEASURED", "BASELINE", "DELTA"
        );
        for (i, item) in questionnaire.items.iter().enumerate() {
            let m = report.measured_ratings.get(i).copied().unwrap_or(0);
            let b = report.baseline_ratings.get(i).copied().unwrap_or(0);
            let d = report.per_item_delta.get(i).copied().unwrap_or(0);
            let mark = if d.unsigned_abs() > report.tolerance_per_item {
                "  *** DRIFT"
            } else {
                ""
            };
            println!("{:<id_w$}  {:>8}  {:>8}  {:>+6}{}", item.id, m, b, d, mark);
        }
        println!();
        println!(
            "max_abs_delta: {}  |  tolerance: {}  |  {}",
            report.max_abs_delta,
            report.tolerance_per_item,
            if report.pass { "PASS" } else { "FAIL" }
        );
    } else {
        println!("{:<id_w$}  {:>8}", "ITEM", "MEASURED");
        for (i, item) in questionnaire.items.iter().enumerate() {
            let m = outcome
                .answers
                .ratings
                .get(i)
                .map(|r| r.rating)
                .unwrap_or(0);
            println!("{:<id_w$}  {:>8}", item.id, m);
        }
        println!();
        println!(
            "(no baseline for model={} version={} — run with --capture \
             to seed one)",
            outcome.model_id, questionnaire.version
        );
    }
    println!();
    println!(
        "tokens: {} in / {} out",
        outcome.usage.input_tokens, outcome.usage.output_tokens
    );
}
