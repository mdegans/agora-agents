use anyhow::Context;
use std::path::Path;

/// Agora constitution (must be at this path to build).
pub const CONSTITUTION: &str = include_str!("../../../constitution.md");

pub(crate) fn init_logging() {
    if let Err(e) = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::from_default_env()),
        )
        .try_init()
    {
        tracing::error!(e)
    }
}

pub(crate) async fn read_file_stripped(path: impl AsRef<Path>) -> anyhow::Result<String> {
    Ok(tokio::fs::read_to_string(path.as_ref())
        .await
        .with_context(|| format!("Reading file: {}", path.as_ref().display()))?
        .trim()
        .to_string())
}
