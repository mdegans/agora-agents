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

    /// Local Ollama server URL (for models like llama3.1:8b, qwen2.5:7b, mistral:7b).
    #[arg(long, default_value = "http://localhost:11434")]
    pub ollama_url: String,

    /// Remote Ollama server URL (for offloading to another machine).
    /// Agents whose model.txt matches --remote-ollama-models will use this URL.
    #[arg(long)]
    pub remote_ollama_url: Option<String>,

    /// Comma-separated list of model names that should run on the remote Ollama server.
    #[arg(long, default_value = "gpt-oss:20b")]
    pub remote_ollama_models: String,

    /// Max concurrent requests to the remote Ollama server.
    #[arg(long, default_value = "1")]
    pub remote_ollama_concurrency: usize,

    /// Number of perceive/think/act/reflect cycles per agent.
    #[arg(long, default_value = "3")]
    pub cycles: usize,

    /// Max concurrent local Ollama requests (limited by GPU).
    #[arg(long, default_value = "1")]
    pub ollama_concurrency: usize,

    /// Phase to run: register, run, or all.
    #[arg(long, default_value = "all")]
    pub phase: Phase,

    /// Override model for all agents (for testing).
    #[arg(long)]
    pub model_override: Option<String>,
}

impl Cli {
    /// Returns the set of model names that should use the remote Ollama URL.
    pub fn remote_models(&self) -> Vec<String> {
        self.remote_ollama_models
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Check if a model should run on the remote Ollama server.
    pub fn is_remote_model(&self, model: &str) -> bool {
        self.remote_ollama_url.is_some()
            && self.remote_models().iter().any(|m| m == model)
    }
}

#[derive(Clone, Debug, clap::ValueEnum)]
pub enum Phase {
    Register,
    Run,
    All,
}
