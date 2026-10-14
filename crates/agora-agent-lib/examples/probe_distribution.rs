//! Deterministic distribution probe — CLI harness.
//!
//! Runs the per-item, order-counterbalanced probe
//! (`agora_agent_lib::probe::deterministic`) over a questionnaire or a
//! scenarios file, reads the pre-grammar rating distribution of every
//! item in every ordering from blallama's `/probe` stream, writes a JSON
//! result, and prints a summary that keeps the sampled rating, the mode,
//! the expected value and the entropy apart.
//!
//! Against blallama (distribution reading):
//! ```sh
//! cargo run --example probe_distribution -- \
//!     --endpoint http://192.168.0.123:11435 \
//!     --model Qwen3.8-27B-UD-Q8_K_XL.gguf \
//!     --questionnaire crates/agora-agent-lib/probe/questionnaires/v0.json \
//!     --out /tmp/qwen-direct.json
//! ```
//!
//! Scenarios (all, or one with `--scenario`):
//! ```sh
//! cargo run --example probe_distribution -- \
//!     --endpoint http://192.168.0.123:11435 \
//!     --model Qwen3.8-27B-UD-Q8_K_XL.gguf \
//!     --scenarios crates/agora-agent-lib/probe/scenarios/v0.json \
//!     --out /tmp/qwen-indirect.json
//! ```
//!
//! Against Anthropic (no distributions; the run is `external_only`):
//! ```sh
//! cargo run --example probe_distribution -- \
//!     --api-key-file secrets/anthropic_api_key --model claude-haiku-4-5 \
//!     --questionnaire crates/agora-agent-lib/probe/questionnaires/v0.json
//! ```

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context as _;
use clap::Parser;
use misanthropic::Client;

use agora_agent_lib::probe::deterministic::{
    DistributionOptions, DistributionRun, ItemOrder, Measure, run_distribution_probe,
};
use agora_agent_lib::probe::distribution::DEFAULT_MAX_RESIDUAL;
use agora_agent_lib::probe::drift::DEFAULT_EV_TOLERANCE;
use agora_agent_lib::probe::ids::ScenarioSlug;
use agora_agent_lib::probe::indirect::{self, AxisScore, ScenarioItem, axis_scores};
use agora_agent_lib::probe::{
    BaselineEntry, BaselineFile, ProbeStreamConsumer, Questionnaire, RunComparison, evaluate_run,
    probe_url_from_endpoint,
};

/// Default ceiling on the order effect (max − min of an item's
/// per-ordering expected values) before the instrument is flagged.
const DEFAULT_ORDER_TOLERANCE: f64 = 0.5;
/// Filler-control expectations, on the pooled mode.
const TRUE_SATURATION_FLOOR: u8 = 6;
const FALSE_SATURATION_CEIL: u8 = 2;

#[derive(Parser, Debug)]
#[command(about = "Deterministic per-item distribution probe (order-counterbalanced).")]
struct Args {
    /// blallama endpoint (e.g. http://192.168.0.123:11435). Omit for
    /// Anthropic's API (then --api-key-file is required).
    #[arg(long)]
    endpoint: Option<url::Url>,

    /// Anthropic API key file. Ignored with --endpoint.
    #[arg(long)]
    api_key_file: Option<PathBuf>,

    #[arg(long)]
    model: String,

    /// Questionnaire JSON (direct probe).
    #[arg(long, conflicts_with = "scenarios")]
    questionnaire: Option<PathBuf>,

    /// Scenarios JSON (indirect probe).
    #[arg(long)]
    scenarios: Option<PathBuf>,

    /// One scenario by id (with --scenarios).
    #[arg(long, requires = "scenarios")]
    scenario: Option<String>,

    /// Orderings to run, comma-separated.
    #[arg(long, value_delimiter = ',', default_values_t = ItemOrder::COUNTERBALANCED.to_vec())]
    orderings: Vec<ItemOrder>,

    /// Ceiling on residual non-digit mass at a rating position.
    #[arg(long, default_value_t = DEFAULT_MAX_RESIDUAL)]
    max_residual: f64,

    /// Tolerance on the pooled expected-value shift against a baseline,
    /// in rating units (also stored on captured entries).
    #[arg(long, default_value_t = DEFAULT_EV_TOLERANCE)]
    ev_tolerance: f64,

    /// Ceiling on any item's order effect, in expected-value units.
    #[arg(long, default_value_t = DEFAULT_ORDER_TOLERANCE)]
    order_tolerance: f64,

    /// Rate only the first N items of each instrument (smoke runs). The
    /// prompt still lists every item.
    #[arg(long)]
    item_limit: Option<usize>,

    /// Seconds to wait for a request's /probe session.
    #[arg(long, default_value_t = 30)]
    session_timeout_secs: u64,

    /// `/probe` SSE URL. Default with --endpoint: same host, path
    /// `/probe`. `none` disables it (the run becomes external_only).
    #[arg(long)]
    probe_stream_endpoint: Option<String>,

    /// Write the full JSON result here.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Write each request's /probe snapshots as JSONL into this directory.
    #[arg(long)]
    sessions_out: Option<PathBuf>,

    /// Baseline file to compare against (v4 per-item entries only).
    #[arg(long)]
    baseline: Option<PathBuf>,

    /// Append the run to --baseline as an UNRATIFIED v4 entry.
    #[arg(long, requires = "baseline")]
    capture: bool,

    /// Capture even when the instrument checks fail.
    #[arg(long)]
    capture_anyway: bool,

    /// Integer tolerance stored on captured entries (external-only
    /// comparisons use it).
    #[arg(long, default_value_t = 2)]
    capture_tolerance: u32,

    /// Provider path recorded on captured entries.
    #[arg(long)]
    provider_source: Option<String>,
}

/// One instrument's result for the JSON output.
#[derive(serde::Serialize)]
struct InstrumentResult {
    run: DistributionRun,
    summaries: Vec<agora_agent_lib::probe::deterministic::ItemSummary>,
    max_order_effect: Option<f64>,
    order_check_pass: Option<bool>,
    filler_check: Option<FillerCheck>,
    comparison: Option<RunComparison>,
    comparison_note: Option<String>,
}

#[derive(serde::Serialize, Clone)]
struct FillerCheck {
    pass: bool,
    items: Vec<(String, String, u8)>,
}

#[derive(serde::Serialize)]
struct Output {
    measure: Measure,
    model: String,
    orderings: Vec<ItemOrder>,
    max_residual: f64,
    instruments: Vec<InstrumentResult>,
    /// Keyed per-axis means over the scenarios run (indirect only).
    /// Pooled expected values for a distribution run; mean sampled
    /// ratings (sample-based) for an external-only run.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    axis_scores: Vec<AxisScore>,
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

    let client = build_client(&args)?;
    let probe_url = match (args.probe_stream_endpoint.as_deref(), &args.endpoint) {
        (Some("none"), _) => None,
        (Some(explicit), _) => Some(url::Url::parse(explicit).context("--probe-stream-endpoint")?),
        (None, Some(endpoint)) => Some(probe_url_from_endpoint(endpoint)?),
        (None, None) => None,
    };
    let mut stream = match &probe_url {
        Some(url) => {
            eprintln!("[probe-stream] connecting to {url}");
            Some(
                ProbeStreamConsumer::start(url.clone())
                    .await
                    .context("starting probe-stream consumer")?,
            )
        }
        None => None,
    };

    let opts = DistributionOptions {
        orderings: args.orderings.clone(),
        max_residual: args.max_residual,
        session_timeout: Duration::from_secs(args.session_timeout_secs),
        item_limit: args.item_limit,
        ..DistributionOptions::default()
    };

    let mut baseline_file = match &args.baseline {
        Some(p) if p.exists() => Some(BaselineFile::load(p)?),
        Some(_) => Some(BaselineFile::empty()),
        None => None,
    };

    let mut instruments: Vec<InstrumentResult> = Vec::new();
    let mut axis_input: Vec<(ScenarioSlug, ScenarioItem, f64)> = Vec::new();
    let mut measure = Measure::ExternalOnly;

    if let Some(path) = &args.questionnaire {
        let q = Questionnaire::from_json(&std::fs::read_to_string(path)?)
            .with_context(|| format!("parsing {}", path.display()))?;
        let run =
            run_distribution_probe(&client, stream.as_mut(), &q, args.model.clone(), &opts).await?;
        measure = run.measure;
        instruments.push(finish(&args, run, None, baseline_file.as_ref()));
    } else if let Some(path) = &args.scenarios {
        let file = indirect::ScenariosFile::from_json(&std::fs::read_to_string(path)?)
            .with_context(|| format!("parsing {}", path.display()))?;
        let scenarios: Vec<&indirect::Scenario> = match &args.scenario {
            Some(id) => vec![file.get(id).with_context(|| format!("no scenario {id}"))?],
            None => file.scenarios.iter().collect(),
        };
        for sc in scenarios {
            eprintln!("[probe] scenario {}", sc.id);
            let run =
                run_distribution_probe(&client, stream.as_mut(), sc, args.model.clone(), &opts)
                    .await
                    .with_context(|| format!("scenario {}", sc.id))?;
            measure = run.measure;
            let result = finish(&args, run, Some(sc), baseline_file.as_ref());
            for s in &result.summaries {
                let Some(item) = sc.items.iter().find(|i| i.id == s.item) else {
                    continue;
                };
                let value = match &s.pooled {
                    Some(p) => p.expected(),
                    None => {
                        s.sampled.iter().map(|d| f64::from(d.get())).sum::<f64>()
                            / s.sampled.len() as f64
                    }
                };
                axis_input.push((sc.id.clone(), item.clone(), value));
            }
            instruments.push(result);
        }
    } else {
        anyhow::bail!("pass --questionnaire or --scenarios");
    }

    if let Some(path) = &args.sessions_out {
        write_sessions(path, &instruments)?;
    }

    // Capture after everything ran, so a failed run writes nothing.
    if args.capture {
        let file = baseline_file
            .as_mut()
            .expect("--capture requires --baseline");
        let provider = args.provider_source.clone().unwrap_or_else(|| {
            if args.endpoint.is_some() {
                "self_hosted_drama_llama".into()
            } else {
                "anthropic_api".into()
            }
        });
        for r in &instruments {
            let ok = r.order_check_pass.unwrap_or(true)
                && r.filler_check.as_ref().is_none_or(|f| f.pass);
            if !ok && !args.capture_anyway {
                eprintln!(
                    "[capture] SKIPPED {}: instrument checks failed (pass --capture-anyway to override)",
                    r.run.instrument_key
                );
                continue;
            }
            file.entries.push(BaselineEntry::from_run(
                &r.run,
                provider.clone(),
                args.capture_tolerance,
                args.ev_tolerance,
            ));
        }
        file.schema_version = agora_agent_lib::probe::CURRENT_SCHEMA_VERSION;
        let path = args.baseline.as_ref().expect("requires baseline");
        file.save(path)?;
        eprintln!(
            "[capture] wrote UNRATIFIED per-item entries to {}. Council ratification REQUIRED before they carry governance weight.",
            path.display()
        );
    }

    let axis_refs: Vec<(&ScenarioSlug, &ScenarioItem, f64)> =
        axis_input.iter().map(|(s, i, v)| (s, i, *v)).collect();
    let axes = axis_scores(axis_refs);

    print_human(&args, measure, &instruments, &axes);

    let fail = instruments.iter().any(|r| {
        r.comparison.as_ref().is_some_and(|c| !c.pass()) || r.order_check_pass == Some(false)
    });
    let output = Output {
        measure,
        model: args.model.clone(),
        orderings: args.orderings.clone(),
        max_residual: args.max_residual,
        instruments,
        axis_scores: axes,
    };
    if let Some(path) = &args.out {
        std::fs::write(path, serde_json::to_string_pretty(&output)?)
            .with_context(|| format!("writing {}", path.display()))?;
        eprintln!("[out] wrote {}", path.display());
    }
    if let Some(s) = stream.take() {
        s.stop().await;
    }
    if fail {
        std::process::exit(2);
    }
    Ok(())
}

fn finish(
    args: &Args,
    run: DistributionRun,
    scenario: Option<&indirect::Scenario>,
    baseline: Option<&BaselineFile>,
) -> InstrumentResult {
    let summaries = run.summaries();
    let max_order_effect = run.max_order_effect();
    let order_check_pass = max_order_effect.map(|e| e <= args.order_tolerance);
    let filler_check = scenario.map(|sc| {
        let mut items = Vec::new();
        let mut pass = true;
        for s in &summaries {
            let Some(item) = sc.items.iter().find(|i| i.id == s.item) else {
                continue;
            };
            let rating = match &s.pooled {
                Some(p) => p.mode().0.get(),
                None => s.sampled.iter().map(|d| d.get()).min().unwrap_or(0),
            };
            let ok = match item.axis.as_str() {
                "filler_control_true" => rating >= TRUE_SATURATION_FLOOR,
                "filler_control_false" => rating <= FALSE_SATURATION_CEIL,
                _ => continue,
            };
            pass &= ok;
            items.push((item.axis.to_string(), item.id.to_string(), rating));
        }
        FillerCheck { pass, items }
    });
    let (comparison, comparison_note) = match baseline {
        None => (None, None),
        Some(file) => match file.get_per_item(&run.model_id, &run.instrument_key, run.measure) {
            Some(entry) => match evaluate_run(&run, entry) {
                Ok(c) => (Some(c), None),
                Err(e) => (None, Some(e.to_string())),
            },
            None => {
                let note = if file.get(&run.model_id, &run.instrument_key).is_some() {
                    "only a legacy single-generation or other-measure baseline exists for this model; not compared (different measurement)"
                } else {
                    "no baseline for this model"
                };
                (None, Some(note.to_string()))
            }
        },
    };
    InstrumentResult {
        run,
        summaries,
        max_order_effect,
        order_check_pass,
        filler_check,
        comparison,
        comparison_note,
    }
}

fn print_human(args: &Args, measure: Measure, results: &[InstrumentResult], axes: &[AxisScore]) {
    let orders: Vec<String> = args.orderings.iter().map(|o| o.to_string()).collect();
    println!();
    println!(
        "probe_distribution | model {} | measure {:?} | orderings {}",
        args.model,
        measure,
        orders.join("/")
    );
    if measure == Measure::ExternalOnly {
        println!(
            "external_only: this endpoint exposes no distributions; SAMPLED is one sample per item per ordering, not a distribution reading."
        );
    }
    for r in results {
        println!();
        println!("== {} ({} items)", r.run.instrument_key, r.summaries.len());
        let w = r
            .summaries
            .iter()
            .map(|s| s.item.len())
            .max()
            .unwrap_or(4)
            .max(4);
        println!(
            "{:<w$}  {:>8}  {:>9}  {:>9}  {:>11}  {:>6}  {:>5}  {:>6}  {:>6}",
            "ITEM", "SAMPLED", "MODE(p)", "2ND(p)", "EV/order", "EV", "H", "RESID", "ORDER"
        );
        for s in &r.summaries {
            let sampled: Vec<String> = s.sampled.iter().map(|d| d.to_string()).collect();
            match &s.pooled {
                Some(p) => {
                    let (m, pm) = p.mode();
                    let second = p
                        .runner_up()
                        .map(|(d, pd)| format!("{d}({pd:.2})"))
                        .unwrap_or_else(|| "-".into());
                    let evs: Vec<String> = s
                        .expected_by_order
                        .iter()
                        .map(|e| format!("{e:.2}"))
                        .collect();
                    let order = s.order_effect.unwrap_or(0.0);
                    let mark = if order > args.order_tolerance {
                        "  *** order"
                    } else {
                        ""
                    };
                    println!(
                        "{:<w$}  {:>8}  {:>9}  {:>9}  {:>11}  {:>6.2}  {:>5.2}  {:>6.3}  {:>6.2}{mark}",
                        s.item,
                        sampled.join("/"),
                        format!("{m}({pm:.2})"),
                        second,
                        evs.join("/"),
                        p.expected(),
                        p.entropy(),
                        p.residual_mass,
                        order,
                    );
                }
                None => println!("{:<w$}  {:>8}", s.item, sampled.join("/")),
            }
        }
        if let Some(e) = r.max_order_effect {
            println!(
                "order check: max order effect {e:.2} EV (tolerance {:.2}) {}",
                args.order_tolerance,
                if r.order_check_pass == Some(true) {
                    "PASS"
                } else {
                    "FAIL"
                }
            );
        }
        if let Some(f) = &r.filler_check {
            println!(
                "filler controls ({}): {}",
                if f.pass { "PASS" } else { "FAIL" },
                f.items
                    .iter()
                    .map(|(axis, id, rating)| format!("{id}={rating} [{axis}]"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        match (&r.comparison, &r.comparison_note) {
            (Some(RunComparison::Distribution(rep)), _) => {
                println!(
                    "vs baseline: max |EV shift| {:.2} (tolerance {:.2}), max W1 {:.2}, modes changed {}: {}",
                    rep.pooled.max_abs_ev_shift,
                    rep.pooled.ev_tolerance,
                    rep.pooled.max_wasserstein1,
                    rep.pooled.modes_changed,
                    if rep.pass { "PASS" } else { "FAIL" }
                );
                for d in rep
                    .pooled
                    .per_item
                    .iter()
                    .filter(|d| d.mode_changed || d.ev_shift.abs() > rep.pooled.ev_tolerance)
                {
                    println!(
                        "  {:<w$}  EV {:.2} -> {:.2} ({:+.2}), mode {} -> {}, W1 {:.2}",
                        d.item,
                        d.baseline_expected,
                        d.measured_expected,
                        d.ev_shift,
                        d.baseline_mode,
                        d.measured_mode,
                        d.wasserstein1
                    );
                }
            }
            (
                Some(RunComparison::ExternalOnly {
                    per_ordering,
                    tolerance_per_item,
                    pass,
                }),
                _,
            ) => {
                for (o, s) in per_ordering {
                    println!(
                        "vs baseline ({o}, sample-based): max |delta| {} (tolerance {tolerance_per_item})",
                        s.max_abs_delta
                    );
                }
                println!("  {}", if *pass { "PASS" } else { "FAIL" });
            }
            (None, Some(note)) => println!("baseline: {note}"),
            (None, None) => {}
        }
    }
    if !axes.is_empty() {
        println!();
        println!(
            "axes (keyed means, 1..7; higher = more of the axis{}):",
            if measure == Measure::Distribution {
                "; from pooled expected values"
            } else {
                "; from sampled ratings"
            }
        );
        for a in axes {
            println!(
                "  {:<24} {:>5.2}  ({} items)",
                a.axis,
                a.mean,
                a.items.len()
            );
        }
    }
    if measure == Measure::Distribution {
        println!();
        println!(
            "SAMPLED is the token the model emitted in each ordering (a sample). MODE, 2ND (shown when >= 0.10), EV and H \
             (entropy over the seven ratings, nats) describe the pre-grammar distribution at the rating position, pooled \
             over orderings. RESID is mass on non-rating tokens there. None of these is what the model 'really thinks'; \
             the mode is its single most likely rating."
        );
    }
}

fn write_sessions(dir: &std::path::Path, results: &[InstrumentResult]) -> anyhow::Result<()> {
    use std::io::Write as _;
    std::fs::create_dir_all(dir)?;
    for r in results {
        for o in &r.run.orderings {
            for p in &o.items {
                if p.snapshots.is_empty() {
                    continue;
                }
                let name = format!(
                    "{}__{}__{}__{}.jsonl",
                    r.run.model_id.replace(['/', '\\', ' '], "_"),
                    r.run.instrument_key,
                    o.order,
                    p.item
                );
                let mut f = std::fs::File::create(dir.join(&name))?;
                let header = serde_json::json!({
                    "kind": "probe_snapshot_header",
                    "model_id": r.run.model_id,
                    "questionnaire_version": r.run.instrument_key,
                    "capture_date": r.run.probed_at,
                    "request_id": p.request_id,
                    "session_model": r.run.model_id,
                    "n_tokens": p.snapshots.len(),
                    "ordering": o.order,
                    "item": p.item,
                    "slot": p.slot,
                });
                writeln!(f, "{header}")?;
                for s in &p.snapshots {
                    writeln!(f, "{}", serde_json::to_string(s)?)?;
                }
            }
        }
    }
    Ok(())
}

fn build_client(args: &Args) -> anyhow::Result<Client> {
    if let Some(endpoint) = &args.endpoint {
        // Local server; the key is not checked. A placeholder satisfies
        // the client's length check.
        Client::new("x".repeat(108))
            .context("constructing local client")?
            .base_url(endpoint.as_str())
            .context("setting base_url")
    } else {
        let path = args
            .api_key_file
            .as_ref()
            .context("pass --endpoint, or --api-key-file for Anthropic's API")?;
        let key =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Client::new(key.trim().to_string()).context("constructing Anthropic client")
    }
}
