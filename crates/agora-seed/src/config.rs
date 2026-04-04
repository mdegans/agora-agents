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

    /// Only run agents whose name contains this substring.
    #[arg(long)]
    pub agent_filter: Option<String>,

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
}

impl Cli {
    /// Return the effective list of Ollama endpoint URLs.
    ///
    /// Uses `--ollama-urls` if set, otherwise falls back to `--ollama-url`.
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

#[derive(Clone, Debug, clap::ValueEnum)]
pub enum Backend {
    Ollama,
    Anthropic,
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
