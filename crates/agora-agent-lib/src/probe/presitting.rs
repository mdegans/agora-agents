//! The pre-sitting canary: every model the Council uses, before each
//! sitting.
//!
//! Purpose (Steward, 2026-10-14): a canary for **inference-pipeline
//! integrity**. Each target (model × thinking mode × endpoint) is run
//! through the distribution probe and compared with its own baseline;
//! any failure means "do not trust this pipeline today".
//!
//! Scheduling: targets on the same local server run **one at a time**
//! (blallama has no queue, so concurrency against one server is 1), and
//! different servers and Anthropic targets run in parallel. A 529
//! "Session is busy" is waited out ([`super::retry::RetryOnBusy`]).
//! The result is one line per target plus a JSON report; the caller
//! exits nonzero unless every target passed.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::baseline::{BaselineFile, RunComparison, evaluate_run};
use super::deterministic::{
    DistributionOptions, ItemOrder, Measure, RatingInstrument, run_distribution_probe,
};
use super::drift::DriftPolicy;
use super::ids::ItemSlug;
use super::indirect::Scenario;
use super::questionnaire::Questionnaire;
use super::retry::RetryOnBusy;
use super::stream::{ProbeStreamConsumer, probe_url_from_endpoint};
use super::thinking::ThinkingMode;

/// One model to check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Target {
    /// Model id as the endpoint knows it.
    pub model: String,
    /// blallama endpoint. Absent: Anthropic's API (needs `api_key_file`).
    #[serde(default)]
    pub endpoint: Option<url::Url>,
    /// Anthropic API key file (secrets come from files, not the
    /// environment).
    #[serde(default)]
    pub api_key_file: Option<PathBuf>,
    #[serde(default)]
    pub thinking: ThinkingMode,
}

impl Target {
    /// Targets sharing a group run one at a time: one group per local
    /// server (scheme, host and port), one per Anthropic target.
    pub fn group_key(&self, index: usize) -> String {
        match &self.endpoint {
            Some(u) => format!(
                "{}://{}:{}",
                u.scheme(),
                u.host_str().unwrap_or(""),
                u.port_or_known_default().unwrap_or(0)
            ),
            None => format!("anthropic#{index}"),
        }
    }
}

/// The instruments every target is run through.
pub enum Instrument<'a> {
    Questionnaire(&'a Questionnaire),
    Scenario(&'a Scenario),
}

impl RatingInstrument for Instrument<'_> {
    fn baseline_key(&self) -> String {
        match self {
            Instrument::Questionnaire(q) => q.baseline_key(),
            Instrument::Scenario(s) => RatingInstrument::baseline_key(*s),
        }
    }
    fn instructions(&self) -> &str {
        match self {
            Instrument::Questionnaire(q) => q.instructions(),
            Instrument::Scenario(s) => s.instructions(),
        }
    }
    fn noun(&self) -> &'static str {
        match self {
            Instrument::Questionnaire(q) => q.noun(),
            Instrument::Scenario(s) => s.noun(),
        }
    }
    fn preamble(&self) -> Option<&str> {
        match self {
            Instrument::Questionnaire(q) => q.preamble(),
            Instrument::Scenario(s) => s.preamble(),
        }
    }
    fn list_heading(&self) -> &'static str {
        match self {
            Instrument::Questionnaire(q) => q.list_heading(),
            Instrument::Scenario(s) => s.list_heading(),
        }
    }
    fn items(&self) -> Vec<(&ItemSlug, &str)> {
        match self {
            Instrument::Questionnaire(q) => q.items(),
            Instrument::Scenario(s) => s.items(),
        }
    }
}

/// Outcome of one target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Pass,
    /// Drifted from its baseline.
    Fail,
    /// No baseline of this measure and thinking mode for some
    /// instrument: the pipeline cannot be vouched for.
    NoBaseline,
    /// The probe itself failed (unreachable, unreadable stream, ...).
    Error,
}

/// One instrument's result for one target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstrumentVerdict {
    pub instrument_key: String,
    pub verdict: Verdict,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comparison: Option<RunComparison>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// One target's result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetOutcome {
    pub target: Target,
    pub verdict: Verdict,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measure: Option<Measure>,
    pub instruments: Vec<InstrumentVerdict>,
    pub seconds: f64,
}

impl TargetOutcome {
    /// The one-line summary.
    pub fn summary_line(&self) -> String {
        let status = match self.verdict {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::NoBaseline => "NO-BASELINE",
            Verdict::Error => "ERROR",
        };
        let mut max_shift = 0.0f64;
        let mut max_mean = 0.0f64;
        let mut modes = 0usize;
        let mut ext_delta = 0u32;
        for i in &self.instruments {
            match &i.comparison {
                Some(RunComparison::Distribution(r)) => {
                    max_shift = max_shift.max(r.pooled.max_abs_ev_shift);
                    max_mean = max_mean.max(r.pooled.mean_abs_ev_shift);
                    modes += r.pooled.modes_changed;
                }
                Some(RunComparison::ExternalOnly { per_ordering, .. }) => {
                    for (_, s) in per_ordering {
                        ext_delta = ext_delta.max(s.max_abs_delta);
                    }
                }
                None => {}
            }
        }
        let failing: Vec<&str> = self
            .instruments
            .iter()
            .filter(|i| i.verdict != Verdict::Pass)
            .map(|i| i.instrument_key.as_str())
            .collect();
        let detail = match self.measure {
            Some(Measure::Distribution) => format!(
                "distribution, max |dEV| {max_shift:.3}, worst mean {max_mean:.3}, modes changed {modes}"
            ),
            Some(Measure::ExternalOnly) => {
                format!("external_only (sampled; weaker evidence), max |delta| {ext_delta}")
            }
            None => "no reading".to_string(),
        };
        let mut line = format!(
            "{status:<11} {} [{}] {} instruments: {detail} ({:.0}s)",
            self.target.model,
            self.target.thinking,
            self.instruments.len(),
            self.seconds
        );
        if !failing.is_empty() {
            line.push_str(&format!("; not passing: {}", failing.join(", ")));
        }
        if let Some(note) = self
            .instruments
            .iter()
            .find(|i| i.verdict == Verdict::Error)
            .and_then(|i| i.note.as_deref())
        {
            line.push_str(&format!("; error: {note}"));
        }
        line
    }
}

/// The overall verdict of a target from its instruments: any error is
/// an error, else any failure is a failure, else any missing baseline,
/// else a pass.
pub fn combine(instruments: &[InstrumentVerdict]) -> Verdict {
    let has = |v: Verdict| instruments.iter().any(|i| i.verdict == v);
    if instruments.is_empty() || has(Verdict::Error) {
        Verdict::Error
    } else if has(Verdict::Fail) {
        Verdict::Fail
    } else if has(Verdict::NoBaseline) {
        Verdict::NoBaseline
    } else {
        Verdict::Pass
    }
}

/// Settings shared by every target.
#[derive(Debug, Clone)]
pub struct PresittingOptions {
    pub distribution: DistributionOptions,
    pub policy: DriftPolicy,
    pub busy_retry_every: Duration,
    pub busy_retries: u32,
}

impl Default for PresittingOptions {
    fn default() -> Self {
        Self {
            distribution: DistributionOptions {
                orderings: ItemOrder::COUNTERBALANCED.to_vec(),
                session_timeout: Duration::from_secs(600),
                ..DistributionOptions::default()
            },
            policy: DriftPolicy::default(),
            busy_retry_every: super::retry::DEFAULT_BUSY_RETRY_EVERY,
            busy_retries: super::retry::DEFAULT_BUSY_RETRIES,
        }
    }
}

/// Run every target: groups in parallel, targets within a group one at
/// a time. Outcomes come back in the order of `targets`.
pub async fn run_presitting(
    targets: &[Target],
    instruments: &[Instrument<'_>],
    baseline: &BaselineFile,
    opts: &PresittingOptions,
) -> Vec<TargetOutcome> {
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, t) in targets.iter().enumerate() {
        groups.entry(t.group_key(i)).or_default().push(i);
    }
    let runs = groups.into_values().map(|indices| async move {
        let mut out = Vec::with_capacity(indices.len());
        for i in indices {
            out.push((
                i,
                run_target(&targets[i], instruments, baseline, opts).await,
            ));
        }
        out
    });
    let mut all: Vec<(usize, TargetOutcome)> = futures_util::future::join_all(runs)
        .await
        .into_iter()
        .flatten()
        .collect();
    all.sort_by_key(|(i, _)| *i);
    all.into_iter().map(|(_, o)| o).collect()
}

/// Run one target through every instrument and compare with its
/// baselines. Never panics on a probe failure: that is an `Error`
/// verdict.
pub async fn run_target(
    target: &Target,
    instruments: &[Instrument<'_>],
    baseline: &BaselineFile,
    opts: &PresittingOptions,
) -> TargetOutcome {
    let started = Instant::now();
    let error = |note: String| TargetOutcome {
        target: target.clone(),
        verdict: Verdict::Error,
        measure: None,
        instruments: vec![InstrumentVerdict {
            instrument_key: "(setup)".into(),
            verdict: Verdict::Error,
            comparison: None,
            note: Some(note),
        }],
        seconds: started.elapsed().as_secs_f64(),
    };
    let client = match build_client(target).await {
        Ok(c) => c,
        Err(e) => return error(e),
    };
    let sender = RetryOnBusy {
        inner: &client,
        every: opts.busy_retry_every,
        max_retries: opts.busy_retries,
    };
    let mut stream = match &target.endpoint {
        Some(endpoint) => {
            let url = match probe_url_from_endpoint(endpoint) {
                Ok(u) => u,
                Err(e) => return error(e.to_string()),
            };
            match ProbeStreamConsumer::start(url).await {
                Ok(s) => Some(s),
                Err(e) => return error(format!("probe stream: {e:#}")),
            }
        }
        None => None,
    };
    let dist_opts = DistributionOptions {
        thinking: target.thinking.clone(),
        ..opts.distribution.clone()
    };
    let mut verdicts = Vec::with_capacity(instruments.len());
    let mut measure = None;
    for inst in instruments {
        let key = inst.baseline_key();
        tracing::info!(model = %target.model, thinking = %target.thinking, instrument = %key, "pre-sitting probe");
        let run = match run_distribution_probe(
            &sender,
            stream.as_mut(),
            inst,
            target.model.clone(),
            &dist_opts,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                verdicts.push(InstrumentVerdict {
                    instrument_key: key,
                    verdict: Verdict::Error,
                    comparison: None,
                    note: Some(e.to_string()),
                });
                // A broken pipeline stays broken; don't spend the rest
                // of the sitting's budget finding out again.
                break;
            }
        };
        measure = Some(run.measure);
        let v = match baseline.get_per_item(&run.model_id, &key, run.measure, &run.thinking) {
            None => InstrumentVerdict {
                instrument_key: key,
                verdict: Verdict::NoBaseline,
                comparison: None,
                note: Some(format!(
                    "no {:?} baseline for {} [{}]",
                    run.measure, run.model_id, run.thinking
                )),
            },
            Some(entry) => match evaluate_run(&run, entry, &opts.policy) {
                Ok(c) => InstrumentVerdict {
                    instrument_key: key,
                    verdict: if c.pass() {
                        Verdict::Pass
                    } else {
                        Verdict::Fail
                    },
                    comparison: Some(c),
                    note: None,
                },
                Err(e) => InstrumentVerdict {
                    instrument_key: key,
                    verdict: Verdict::Error,
                    comparison: None,
                    note: Some(e.to_string()),
                },
            },
        };
        verdicts.push(v);
    }
    if let Some(s) = stream.take() {
        s.stop().await;
    }
    TargetOutcome {
        target: target.clone(),
        verdict: combine(&verdicts),
        measure,
        instruments: verdicts,
        seconds: started.elapsed().as_secs_f64(),
    }
}

async fn build_client(target: &Target) -> Result<misanthropic::Client, String> {
    match (&target.endpoint, &target.api_key_file) {
        (Some(endpoint), _) => misanthropic::Client::new("x".repeat(108))
            .map_err(|e| e.to_string())?
            .base_url(endpoint.as_str())
            .map_err(|e| e.to_string()),
        (None, Some(path)) => {
            let key = tokio::fs::read_to_string(path)
                .await
                .map_err(|e| format!("reading {}: {e}", path.display()))?;
            misanthropic::Client::new(key.trim().to_string()).map_err(|e| e.to_string())
        }
        (None, None) => Err("target has neither an endpoint nor an api_key_file".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(model: &str, endpoint: Option<&str>) -> Target {
        Target {
            model: model.into(),
            endpoint: endpoint.map(|e| url::Url::parse(e).unwrap()),
            api_key_file: None,
            thinking: ThinkingMode::Off,
        }
    }

    #[test]
    fn one_group_per_server_and_per_anthropic_target() {
        let targets = [
            t("qwen", Some("http://192.168.0.123:11435")),
            t("qwen", Some("http://192.168.0.123:11435/")),
            t("other", Some("http://192.168.0.124:11435")),
            t("haiku", None),
            t("sonnet", None),
        ];
        let keys: Vec<String> = targets
            .iter()
            .enumerate()
            .map(|(i, t)| t.group_key(i))
            .collect();
        assert_eq!(keys[0], keys[1], "same server, one queue");
        assert_ne!(keys[0], keys[2]);
        assert_ne!(keys[3], keys[4], "Anthropic targets run in parallel");
    }

    fn iv(v: Verdict) -> InstrumentVerdict {
        InstrumentVerdict {
            instrument_key: "k".into(),
            verdict: v,
            comparison: None,
            note: None,
        }
    }

    #[test]
    fn combine_is_pessimistic() {
        assert_eq!(
            combine(&[iv(Verdict::Pass), iv(Verdict::Pass)]),
            Verdict::Pass
        );
        assert_eq!(
            combine(&[iv(Verdict::Pass), iv(Verdict::NoBaseline)]),
            Verdict::NoBaseline
        );
        assert_eq!(
            combine(&[iv(Verdict::NoBaseline), iv(Verdict::Fail)]),
            Verdict::Fail
        );
        assert_eq!(
            combine(&[iv(Verdict::Fail), iv(Verdict::Error)]),
            Verdict::Error
        );
        assert_eq!(combine(&[]), Verdict::Error);
    }

    #[test]
    fn config_targets_parse_and_summarise() {
        let targets: Vec<Target> = serde_json::from_value(serde_json::json!([
            {"model": "Qwen3.8-27B-UD-Q8_K_XL.gguf", "endpoint": "http://192.168.0.123:11435"},
            {"model": "Qwen3.8-27B-UD-Q8_K_XL.gguf", "endpoint": "http://192.168.0.123:11435",
             "thinking": {"thinking": "adaptive", "effort": "low"}},
            {"model": "claude-haiku-4-5", "api_key_file": "secrets/anthropic_api_key"}
        ]))
        .unwrap();
        assert_eq!(targets[0].thinking, ThinkingMode::Off);
        assert_eq!(targets[1].thinking.label(), "thinking_adaptive_low");
        let o = TargetOutcome {
            target: targets[1].clone(),
            verdict: Verdict::NoBaseline,
            measure: Some(Measure::Distribution),
            instruments: vec![iv(Verdict::NoBaseline)],
            seconds: 12.0,
        };
        let line = o.summary_line();
        assert!(
            line.starts_with("NO-BASELINE Qwen3.8-27B-UD-Q8_K_XL.gguf [thinking_adaptive_low]"),
            "{line}"
        );
        assert!(line.contains("not passing: k"), "{line}");
    }

    /// A target that cannot even be set up is an error, not a pass.
    #[tokio::test]
    async fn unconfigured_target_is_an_error() {
        let o = run_target(
            &t("haiku", None),
            &[],
            &BaselineFile::empty(),
            &PresittingOptions::default(),
        )
        .await;
        assert_eq!(o.verdict, Verdict::Error);
        assert!(o.summary_line().starts_with("ERROR"));
    }
}
