use std::path::PathBuf;

use clap::Parser;

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

    /// Backend to use for batched LLM requests.
    #[arg(long, default_value = "ollama")]
    pub backend: Backend,

    /// Path to file containing Anthropic API key (required when --backend=anthropic).
    #[arg(long)]
    pub anthropic_key_file: Option<PathBuf>,

    /// Messages-API endpoint(s). Repeat for multi-endpoint Ollama-style.
    /// Schemes: `ollama` | `blallama`. If set, overrides `--backend` and
    /// the legacy URL flags. Each scheme picks the per-phase config:
    ///   - `ollama` — tool_choice=Auto on think, None on reflect/mutate/etc.
    ///   - `blallama` — tool_choice=Any on think, None elsewhere, plus
    ///                  `output_config` set per phase for structured output.
    /// Anthropic stays selected via the legacy `--backend anthropic` +
    /// `--anthropic-key-file` route since it goes through the Batch API.
    ///
    /// Examples:
    ///     --messages-api ollama://localhost:11434
    ///     --messages-api blallama://192.168.0.123:1234
    #[arg(long)]
    pub messages_api: Vec<String>,
}

impl Cli {
    /// Return the effective list of Ollama endpoint URLs.
    ///
    /// Uses `--ollama-urls` if set, otherwise falls back to `--ollama-url`.
    /// `--messages-api` is parsed separately via [`parse_messages_api`].
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Backend {
    /// Anthropic Batch API (Haiku). Always uses `tool_choice=Auto` and never
    /// sets `output_config`, so prompt cache stays stable across phases.
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

/// One resolved `--messages-api` endpoint: backend kind + base URL.
#[derive(Clone, Debug)]
pub struct MessagesEndpoint {
    pub backend: Backend,
    /// Base URL ready to feed `misanthropic::Client::with_base_url`. Always
    /// `http://...` for now (no TLS to local Ollama / blallama).
    pub base_url: String,
}

/// Parse `--messages-api` values into `(backend, base_url)` pairs.
///
/// Each value must be `<scheme>://<host>[:port]` with `scheme` one of
/// `ollama`, `blallama`. Returns the parsed endpoints; on any malformed
/// value returns the offending input as the error string.
pub fn parse_messages_api(values: &[String]) -> Result<Vec<MessagesEndpoint>, String> {
    let mut out = Vec::with_capacity(values.len());
    for raw in values {
        let (scheme, rest) = raw
            .split_once("://")
            .ok_or_else(|| format!("--messages-api {raw:?}: missing scheme://"))?;
        let backend = match scheme {
            "ollama" => Backend::Ollama,
            "blallama" => Backend::Blallama,
            other => {
                return Err(format!(
                    "--messages-api {raw:?}: unknown scheme {other:?}; want `ollama` or `blallama`"
                ));
            }
        };
        if rest.is_empty() {
            return Err(format!("--messages-api {raw:?}: missing host"));
        }
        let base_url = format!("http://{rest}");
        out.push(MessagesEndpoint { backend, base_url });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ollama() {
        let parsed = parse_messages_api(&["ollama://localhost:11434".to_string()]).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].backend, Backend::Ollama);
        assert_eq!(parsed[0].base_url, "http://localhost:11434");
    }

    #[test]
    fn parses_blallama() {
        let parsed = parse_messages_api(&["blallama://192.168.0.123:1234".to_string()]).unwrap();
        assert_eq!(parsed[0].backend, Backend::Blallama);
        assert_eq!(parsed[0].base_url, "http://192.168.0.123:1234");
    }

    #[test]
    fn parses_multiple() {
        let parsed = parse_messages_api(&[
            "ollama://localhost:11434".to_string(),
            "blallama://10.0.0.1:1234".to_string(),
        ])
        .unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].backend, Backend::Ollama);
        assert_eq!(parsed[1].backend, Backend::Blallama);
    }

    #[test]
    fn rejects_unknown_scheme() {
        let err = parse_messages_api(&["anthropic://api.anthropic.com".to_string()]).unwrap_err();
        assert!(err.contains("unknown scheme"));
    }

    #[test]
    fn rejects_missing_scheme() {
        let err = parse_messages_api(&["localhost:11434".to_string()]).unwrap_err();
        assert!(err.contains("missing scheme"));
    }

    #[test]
    fn rejects_missing_host() {
        let err = parse_messages_api(&["ollama://".to_string()]).unwrap_err();
        assert!(err.contains("missing host"));
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
