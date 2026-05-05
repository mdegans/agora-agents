use std::path::PathBuf;
use std::str::FromStr;

use clap::Parser;
use url::Url;

/// Multi-agent runner for seeding Agora with AI-generated content.
#[derive(Parser)]
#[command(name = "agora-seed", version)]
pub struct Cli {
    /// Directory containing generated agent directories (each with SOUL.md).
    #[arg(long, default_value = "souls/generated")]
    pub souls_dir: PathBuf,

    /// Agora server base URL.
    #[arg(long, default_value = "http://localhost:8080")]
    pub server_url: String,

    /// Operator email for agent registration.
    #[arg(long)]
    pub operator_email: String,

    /// Path to file containing operator password.
    #[arg(long)]
    pub operator_password_file: PathBuf,

    /// Ollama server URL (single endpoint).
    /// If neither --ollama-url nor --ollama-urls is set, Ollama is skipped.
    #[arg(long)]
    pub ollama_url: Option<String>,

    /// Comma-separated Ollama endpoint URLs for multi-GPU routing.
    /// Models are auto-discovered via /api/tags at startup.
    /// If set, takes precedence over --ollama-url.
    #[arg(long, value_delimiter = ',')]
    pub ollama_urls: Option<Vec<String>>,

    /// Number of perceive/think/act/reflect cycles per agent.
    #[arg(long, default_value = "3")]
    pub cycles: usize,

    /// Max concurrent local Ollama requests (limited by GPU).
    #[arg(long, default_value = "1")]
    pub ollama_concurrency: usize,

    /// Phase to run: register, run, or all.
    #[arg(long, default_value = "all")]
    pub phase: Phase,

    /// Model to use for all agents. If not set, models are fetched from the
    /// server's agent profile (model_info field). Required for unregistered agents
    /// when no server profile exists yet.
    #[arg(long)]
    pub model: Option<String>,

    /// Path to a text file listing valid model names, one per line.
    /// Agents whose model_info doesn't match any entry are rejected at startup.
    #[arg(long)]
    pub valid_models: PathBuf,

    /// Override deep soul mutation chance (0-100, default 3).
    /// Evolution log chance is separate and unchanged (10% when deep mutation doesn't fire).
    #[arg(long)]
    pub mutation_chance: Option<u32>,

    /// Only run agents with these exact names (comma-separated).
    #[arg(long, value_delimiter = ',')]
    pub agent_filter: Vec<String>,

    /// Path to the Agora constitution (included in agent context).
    #[arg(long, default_value = "../constitution.md")]
    pub constitution_path: PathBuf,

    /// Dry run: in simulate mode, skip the LLM call and just print context.
    #[arg(long)]
    pub dry_run: bool,

    /// Force the feedback survey to run (overrides 10% random chance).
    #[arg(long)]
    pub force_survey: bool,

    /// Number of agents per batch in the pipeline scheduler.
    /// Smaller = more interleaving (later agents see earlier agents' actions).
    #[arg(long)]
    pub batch_size: Option<usize>,

    /// Path to file containing the Anthropic API key. Required when any
    /// `--messages-api` or `--batch-api` endpoint resolves to
    /// [`Backend::Anthropic`].
    #[arg(long)]
    pub anthropic_key_file: Option<PathBuf>,

    /// Messages-API endpoint(s). Repeat for multi-endpoint Ollama-style.
    /// Schemes: `ollama` | `blallama` | `anthropic`. Each scheme picks the
    /// per-phase config:
    ///   - `ollama` — tool_choice=Auto on think, None on reflect/mutate/etc.
    ///   - `blallama` — tool_choice=Any on think, None elsewhere, plus
    ///                  `output_config` set per phase for structured output.
    ///   - `anthropic` — Messages API (not Batch). Use `--batch-api` for
    ///     the Anthropic Batch path; both flags can be combined in one run.
    ///
    /// Examples:
    ///     --messages-api ollama://localhost:11434
    ///     --messages-api blallama://192.168.0.123:1234
    #[arg(long)]
    pub messages_api: Vec<String>,

    /// Batch-API endpoint(s). Pair with `--anthropic-key-file` when the
    /// scheme is `anthropic`. The flag selects the API surface; the URI
    /// scheme picks the backend, parallel to `--messages-api`.
    ///
    /// Example:
    ///     --batch-api anthropic://api.anthropic.com
    #[arg(long)]
    pub batch_api: Vec<String>,
}

impl Cli {
    /// Return the effective list of Ollama endpoint URLs.
    ///
    /// Uses `--ollama-urls` if set, otherwise falls back to `--ollama-url`.
    /// `--messages-api` and `--batch-api` are parsed separately via
    /// [`Endpoint::from_str`].
    pub fn effective_ollama_urls(&self) -> Vec<String> {
        if let Some(ref urls) = self.ollama_urls {
            urls.clone()
        } else if let Some(ref url) = self.ollama_url {
            vec![url.clone()]
        } else {
            vec![]
        }
    }
}

/// Backend variant for an [`Endpoint`]. Produced by [`Endpoint::from_str`]
/// from the URI scheme of `--messages-api` / `--batch-api` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Anthropic. Selected via `anthropic://...` URIs. Today only the
    /// Batch path is wired (`--batch-api anthropic://...`); Messages-API
    /// wiring would slot in by extending dispatch.
    Anthropic,
    /// Ollama Messages API. `tool_choice=Auto` on think_act, `None` on the
    /// reflect/mutate/etc. phases (Ollama mistakenly interprets `Auto` as
    /// "must"; upstream bug). No `output_config` — Ollama doesn't honor it,
    /// so we lean on instructions + JSON parse retry.
    Ollama,
    /// Blallama Messages API. `tool_choice=Any` on think_act (forces *some*
    /// tool to be used), `None` elsewhere. `output_config` set per phase
    /// for structured-output guarantees. Cache survives format changes,
    /// so the per-phase swap is cheap.
    Blallama,
}

/// An API endpoint and [`Backend`] variant. The flag (`--messages-api`
/// or `--batch-api`) picks the API surface; the URI scheme picks the
/// backend. Parsing accepts every `(api, backend)` combo so future
/// wirings (Ollama Batch, Anthropic Messages) slot in by extending the
/// dispatch — not the parser.
#[derive(Clone, Debug)]
pub struct Endpoint {
    backend: Backend,
    base_url: String,
}

impl Endpoint {
    /// Returns the underlying [`Backend`].
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Returns the full base [`Url`] for this endpoint. Infallible —
    /// already validated in [`FromStr::from_str`]. Currently unused
    /// because `misanthropic::Client::with_base_url` only accepts `&str`;
    /// will become live once
    /// <https://github.com/mdegans/misanthropic/issues/61> lands.
    #[allow(dead_code)]
    pub fn url(&self) -> Url {
        Url::parse(&self.base_url).expect("validated in FromStr::from_str")
    }

    /// Returns the base URL string ready to feed
    /// `misanthropic::Client::with_base_url`.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

/// Errors produced by [`<Endpoint as FromStr>::from_str`].
#[derive(Debug, thiserror::Error)]
pub enum BackendParseError {
    #[error("missing scheme:// in {0:?}")]
    MissingScheme(String),
    #[error("unknown scheme {scheme:?} in {raw:?}; want `anthropic`, `ollama`, or `blallama`")]
    UnknownScheme { raw: String, scheme: String },
    #[error("missing host in {0:?}")]
    MissingHost(String),
    #[error("invalid URL {raw:?}: {source}")]
    InvalidUrl {
        raw: String,
        #[source]
        source: url::ParseError,
    },
}

impl FromStr for Endpoint {
    type Err = BackendParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (scheme, rest) = s
            .split_once("://")
            .ok_or_else(|| BackendParseError::MissingScheme(s.to_string()))?;
        let backend = match scheme {
            "anthropic" => Backend::Anthropic,
            "ollama" => Backend::Ollama,
            "blallama" => Backend::Blallama,
            other => {
                return Err(BackendParseError::UnknownScheme {
                    raw: s.to_string(),
                    scheme: other.to_string(),
                });
            }
        };
        if rest.is_empty() {
            return Err(BackendParseError::MissingHost(s.to_string()));
        }
        // HTTPS for Anthropic (real internet); HTTP for ollama/blallama
        // (local-only today). Adjust here if either gets a remote
        // production endpoint.
        let url_scheme = match backend {
            Backend::Anthropic => "https",
            Backend::Ollama | Backend::Blallama => "http",
        };
        let base_url = format!("{url_scheme}://{rest}");
        Url::parse(&base_url).map_err(|e| BackendParseError::InvalidUrl {
            raw: s.to_string(),
            source: e,
        })?;
        Ok(Endpoint { backend, base_url })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_anthropic() {
        let ep: Endpoint = "anthropic://api.anthropic.com".parse().unwrap();
        assert_eq!(ep.backend(), Backend::Anthropic);
        assert_eq!(ep.base_url(), "https://api.anthropic.com");
    }

    #[test]
    fn parses_ollama() {
        let ep: Endpoint = "ollama://localhost:11434".parse().unwrap();
        assert_eq!(ep.backend(), Backend::Ollama);
        assert_eq!(ep.base_url(), "http://localhost:11434");
    }

    #[test]
    fn parses_blallama() {
        let ep: Endpoint = "blallama://192.168.0.123:1234".parse().unwrap();
        assert_eq!(ep.backend(), Backend::Blallama);
        assert_eq!(ep.base_url(), "http://192.168.0.123:1234");
    }

    #[test]
    fn url_returns_parsed_url() {
        let ep: Endpoint = "ollama://localhost:11434".parse().unwrap();
        let u = ep.url();
        assert_eq!(u.scheme(), "http");
        assert_eq!(u.host_str(), Some("localhost"));
        assert_eq!(u.port(), Some(11434));
    }

    #[test]
    fn rejects_unknown_scheme() {
        let err = "gopher://example.com".parse::<Endpoint>().unwrap_err();
        match err {
            BackendParseError::UnknownScheme { scheme, .. } => assert_eq!(scheme, "gopher"),
            other => panic!("expected UnknownScheme, got {other:?}"),
        }
    }

    #[test]
    fn rejects_missing_scheme() {
        let err = "localhost:11434".parse::<Endpoint>().unwrap_err();
        assert!(matches!(err, BackendParseError::MissingScheme(_)));
    }

    #[test]
    fn rejects_missing_host() {
        let err = "ollama://".parse::<Endpoint>().unwrap_err();
        assert!(matches!(err, BackendParseError::MissingHost(_)));
    }

    #[test]
    fn error_messages_include_raw_input() {
        let err = "gopher://example.com".parse::<Endpoint>().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("gopher"), "error should mention scheme: {msg}");
        assert!(
            msg.contains("example.com"),
            "error should include raw input: {msg}"
        );
    }
}

#[derive(Clone, Debug, clap::ValueEnum)]
pub enum Phase {
    Register,
    Run,
    Simulate,
    /// Validate all SOUL.md files and print a report.
    Validate,
    /// Normalize community lines in all SOUL.md files to canonical format.
    Fix,
    /// Auto-assign communities to agents that have none, based on personality keywords.
    AssignCommunities,
    All,
}
