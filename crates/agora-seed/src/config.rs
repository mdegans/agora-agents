use std::path::PathBuf;
use std::str::FromStr;

use clap::Parser;
use url::Url;

/// Multi-agent runner for seeding Agora with AI-generated content.
#[derive(Parser)]
#[command(name = "agora-seed", version)]
pub struct Args {
    /// Directory containing generated agent directories
    #[arg(long, default_value = "souls/generated")]
    pub souls_dir: PathBuf,

    /// Agora server base URL.
    #[arg(long, default_value = "https://subliminal.technology")]
    pub server_url: Url,

    /// Operator email for agent registration.
    #[arg(long)]
    pub operator_email: String,

    /// Path to file containing operator password.
    #[arg(long)]
    pub operator_password_file: PathBuf,

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

    /// Messages-API endpoint(s). Repeat for multi-endpoint.
    ///
    /// Schemes: `ollama` | `blallama` | `anthropic`.
    ///
    /// Each scheme picks slightly different model discovery, caching and tool
    /// use strategies for each backend impl. Ollama, for example, needs special
    /// handling due to bugs and Blallama is more flexible than Anthropic on
    /// cache invalidation. Only `anthropic` exposes `/v1/models`
    ///
    /// Examples:
    ///     --messages-api ollama://localhost:11434
    ///     --messages-api blallama://192.168.0.123:12345
    #[arg(long)]
    pub messages_api: Vec<String>,

    /// Batch-API endpoint(s). Repeat for multi-batch.
    ///
    /// Example:
    ///     --batch-api anthropic://api.anthropic.com
    #[arg(long)]
    pub batch_api: Vec<String>,
}

/// Backend variant for an [`Endpoint`]. Produced by [`Endpoint::from_str`]
/// from the URI scheme of `--messages-api` / `--batch-api` values.
///
/// Different backends support caching and tool use slightly differently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Anthropic. Selected via `anthropic://...` URIs.
    ///
    /// - `tool_choice=Any` on think_act (forces *some* tool to be used)
    /// - `None` elsewhere
    /// - `output_config` supported but not used since changing it per turn
    ///   invalidates cache, unlike Blallama.
    Anthropic,
    /// Ollama `/v1/messages` impl
    ///
    /// - `tool_choice=Auto` on think_act. `Any` is unsupported by ollama. As of
    ///   writing, `Auto` means "must use tools" with ollama (Anthropic `Any`).
    /// - `None` on the reflect/mutate/etc phases (or else tools are called when
    ///   json in text blocks is what is desired).
    /// - `output_format` is ignored by ollama (structured generation).
    Ollama,
    /// Blallama (`drama_llama/bin`) `/v1/messages` impl
    ///
    /// - `tool_choice=Any` on think_act (forces *some* tool to be used)
    /// - `None` elsewhere
    /// -  `output_config` set per phase for structured-output guarantees.
    ///
    /// This is Anthropic conformant behavior with one exception:
    ///
    /// - Cache survives `output_config` changes, so the per-phase swap is
    ///   cheap. With Anthropic, changing the `output_config` invalidates cache
    ///   even though it's a generation constraint and it shouln't strictly need
    ///   to.
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
    base_url: Url,
}

impl Endpoint {
    /// Returns the underlying [`Backend`].
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Returns the base [`Url`] for this endpoint, with the URI scheme
    /// resolved to `http`/`https` (the original `ollama`/`blallama`/`anthropic`
    /// scheme is replaced during parsing).
    pub fn base_url(&self) -> &Url {
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
        // First check the literal `://` separator so we can produce
        // MissingScheme — Url::parse on a bare `localhost:11434` would
        // succeed (treating the input as a relative ref) and we'd lose
        // that error case.
        if !s.contains("://") {
            return Err(BackendParseError::MissingScheme(s.to_string()));
        }
        let parsed = Url::parse(s).map_err(|e| BackendParseError::InvalidUrl {
            raw: s.to_string(),
            source: e,
        })?;
        let backend = match parsed.scheme() {
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
        let host = parsed
            .host_str()
            .filter(|h| !h.is_empty())
            .ok_or_else(|| BackendParseError::MissingHost(s.to_string()))?
            .to_string();

        // HTTPS for Anthropic (real internet); HTTP for ollama/blallama
        // (local-only today). The url crate forbids set_scheme() across
        // the special/non-special boundary, so build a fresh URL from a
        // static seed and copy components over with typed setters.
        let seed = match backend {
            Backend::Anthropic => "https://placeholder",
            Backend::Ollama | Backend::Blallama => "http://placeholder",
        };
        let mut base_url = Url::parse(seed).expect("seed URL is statically valid");
        base_url
            .set_host(Some(&host))
            .map_err(|e| BackendParseError::InvalidUrl {
                raw: s.to_string(),
                source: e,
            })?;
        base_url
            .set_port(parsed.port())
            .expect("http(s) accepts ports validated by the original parse");
        base_url.set_path(parsed.path());
        base_url.set_query(parsed.query());
        base_url.set_fragment(parsed.fragment());

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
        assert_eq!(ep.base_url().scheme(), "https");
        assert_eq!(ep.base_url().host_str(), Some("api.anthropic.com"));
    }

    #[test]
    fn parses_ollama() {
        let ep: Endpoint = "ollama://localhost:11434".parse().unwrap();
        assert_eq!(ep.backend(), Backend::Ollama);
        assert_eq!(ep.base_url().scheme(), "http");
        assert_eq!(ep.base_url().host_str(), Some("localhost"));
        assert_eq!(ep.base_url().port(), Some(11434));
    }

    #[test]
    fn parses_blallama() {
        let ep: Endpoint = "blallama://192.168.0.123:1234".parse().unwrap();
        assert_eq!(ep.backend(), Backend::Blallama);
        assert_eq!(ep.base_url().scheme(), "http");
        assert_eq!(ep.base_url().host_str(), Some("192.168.0.123"));
        assert_eq!(ep.base_url().port(), Some(1234));
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
