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

use agora_agent_lib::probe::{
    BaselineEntry, BaselineFile, CompletedSession, ProbeStreamConsumer, Questionnaire, evaluate,
    probe, probe_url_from_endpoint,
};

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

    /// Probe-stream SSE URL (blallama `--probe-stream` endpoint). When
    /// set, the consumer connects before the `/v1/messages` request,
    /// captures the per-token pre-grammar snapshots, and writes them
    /// to a sidecar JSONL alongside the baseline file. Default when
    /// `--endpoint` is set: same scheme/host/port with path `/probe`.
    /// Pass an explicit URL to override or to use a different host.
    /// Pass the literal string `none` to disable snapshot capture
    /// even when an endpoint is set.
    #[arg(long)]
    probe_stream_endpoint: Option<String>,

    /// Directory (relative to the baseline file's parent) where probe
    /// snapshot JSONL sidecars are written. Each sidecar is one
    /// completion's worth of per-token snapshots.
    #[arg(long, default_value = "probe_snapshots")]
    snapshot_dir: PathBuf,
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

    // Resolve probe-stream URL and spin up the SSE consumer once for
    // the run. The consumer accumulates events for every session and
    // we look them up by Message.id post-completion.
    let probe_stream_url = resolve_probe_stream_url(&args)?;
    let mut probe_stream = if let Some(ref url) = probe_stream_url {
        eprintln!("[probe-stream] connecting to {url}");
        Some(
            ProbeStreamConsumer::start(url.clone())
                .await
                .context("starting probe-stream consumer")?,
        )
    } else {
        None
    };

    // Run the probe.
    let outcome = probe(&client, &questionnaire, args.model.clone()).await?;

    // If probe-stream is active and we know the request_id, claim the
    // matching session from the SSE consumer. Server emits SessionEnd
    // before the synchronous /v1/messages response, so a 5s ceiling
    // covers any event-loop scheduling tail.
    let session: Option<CompletedSession> = match (probe_stream.as_mut(), outcome.request_id) {
        (Some(consumer), Some(req_id)) => match consumer
            .take(req_id, std::time::Duration::from_secs(5))
            .await
        {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!(
                    "[probe-stream] take({req_id}) failed: {e:#} \
                     — recording outcome without snapshot"
                );
                None
            }
        },
        _ => None,
    };

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

        // Resolve absolute snapshot dir, relative to the baseline file's
        // parent. Created lazily on first capture.
        let baseline_parent = args
            .baseline
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        let snapshot_dir = baseline_parent.join(&args.snapshot_dir);

        let snapshot_rel_path = match session.as_ref() {
            Some(s) => Some(write_snapshot_sidecar(
                &snapshot_dir,
                &args.snapshot_dir,
                &outcome.model_id,
                &questionnaire.version,
                outcome.probed_at,
                s,
            )?),
            None => None,
        };

        let entry = BaselineEntry {
            model_id: outcome.model_id.clone(),
            questionnaire_version: questionnaire.version.clone(),
            provider_source: provider_source.clone(),
            capture_date: Utc::now(),
            ratified_at: None,
            council_decision_id: None,
            tolerance_per_item: args.capture_tolerance,
            answers: outcome.answers.clone(),
            snapshot_path: snapshot_rel_path.clone(),
            request_id: outcome.request_id,
        };
        baseline_file.entries.push(entry);
        baseline_file.save(&args.baseline)?;
        let snapshot_note = match snapshot_rel_path.as_deref() {
            Some(p) => format!(" + snapshot at {p}"),
            None => String::new(),
        };
        eprintln!(
            "[capture] appended unratified baseline for model={} version={} \
             provider={} to {}{snapshot_note}. Council ratification \
             REQUIRED before this baseline has governance weight.",
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

    // Drain the probe-stream consumer cleanly so any pending session
    // it received but didn't get to deliver gets logged for analysis.
    if let Some(consumer) = probe_stream.take() {
        consumer.stop().await;
    }

    // Non-zero exit on fail — makes this scriptable as a gate.
    if let Some(report) = report_opt.as_ref()
        && !report.pass
    {
        std::process::exit(2);
    }
    Ok(())
}

/// Resolve the probe-stream URL.
///
/// Precedence: explicit `--probe-stream-endpoint` (parsed as a URL) >
/// derived-from-`--endpoint` (same scheme/host/port, path replaced
/// with `/probe`) > none. The literal string `"none"` for the explicit
/// flag disables the stream even when an endpoint is set.
fn resolve_probe_stream_url(args: &Args) -> anyhow::Result<Option<url::Url>> {
    if let Some(explicit) = args.probe_stream_endpoint.as_deref() {
        if explicit == "none" {
            return Ok(None);
        }
        let url = url::Url::parse(explicit)
            .with_context(|| format!("parsing --probe-stream-endpoint {explicit}"))?;
        return Ok(Some(url));
    }
    if let Some(ref endpoint) = args.endpoint {
        let endpoint_url =
            url::Url::parse(endpoint).with_context(|| format!("parsing --endpoint {endpoint}"))?;
        let probe = probe_url_from_endpoint(&endpoint_url)
            .context("deriving probe-stream URL from --endpoint")?;
        return Ok(Some(probe));
    }
    // No --endpoint and no --probe-stream-endpoint → Anthropic API,
    // no probe stream available.
    Ok(None)
}

/// Write one completion's probe-snapshot stream as a JSONL sidecar.
/// Returns the path *relative to* the baseline file's parent dir,
/// suitable for storing on `BaselineEntry::snapshot_path`.
fn write_snapshot_sidecar(
    abs_dir: &std::path::Path,
    rel_dir: &std::path::Path,
    model_id: &str,
    questionnaire_version: &str,
    capture_date: chrono::DateTime<chrono::Utc>,
    session: &CompletedSession,
) -> anyhow::Result<String> {
    std::fs::create_dir_all(abs_dir)
        .with_context(|| format!("creating snapshot dir {}", abs_dir.display()))?;
    let safe_model = model_id.replace(['/', '\\', ' '], "_");
    let ts = capture_date.format("%Y%m%dT%H%M%SZ");
    let filename = format!(
        "{safe_model}__{questionnaire_version}__{ts}__{}.jsonl",
        session.id
    );
    let abs_path = abs_dir.join(&filename);
    let rel_path = rel_dir.join(&filename);

    let mut file = std::fs::File::create(&abs_path)
        .with_context(|| format!("creating snapshot sidecar {}", abs_path.display()))?;
    use std::io::Write as _;
    let header = serde_json::json!({
        "kind": "probe_snapshot_header",
        "model_id": model_id,
        "questionnaire_version": questionnaire_version,
        "capture_date": capture_date,
        "request_id": session.id,
        "session_model": session.model,
        "n_tokens": session.snapshots.len(),
    });
    writeln!(file, "{header}")?;
    for snap in &session.snapshots {
        writeln!(file, "{}", serde_json::to_string(snap)?)?;
    }

    Ok(rel_path.to_string_lossy().into_owned())
}

fn build_client(args: &Args) -> anyhow::Result<Client> {
    if let Some(endpoint) = args.endpoint.as_ref() {
        // Local server; key not validated against Anthropic. Use a
        // fixed placeholder that satisfies the 108-byte length check.
        let placeholder_key = "x".repeat(108);
        let client = Client::new(placeholder_key)
            .context("constructing local client")?
            .base_url(endpoint)
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
