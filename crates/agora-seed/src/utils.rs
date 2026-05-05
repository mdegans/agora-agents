use anyhow::Context;
use std::{path::Path, sync::OnceLock};

/// Agora constitution embedded at build time. Used as a fallback when the
/// live fetch hasn't run (tests) or fails (offline / 5xx).
const CONSTITUTION_BUILD: &str = include_str!("../../../constitution.md");

/// URL of the live Agora constitution.
const CONSTITUTION_URL: &str = "https://subliminal.technology/agora/constitution";

/// Live constitution text, populated at most once by [`init_constitution`].
static CONSTITUTION_FETCHED: OnceLock<String> = OnceLock::new();

/// Fetch the live constitution and cache it for subsequent [`constitution`]
/// calls. No-op if already initialized or if the fetch fails — the build-time
/// copy stays as the fallback in either case.
pub async fn init_constitution() {
    if CONSTITUTION_FETCHED.get().is_some() {
        return;
    }
    match reqwest::get(CONSTITUTION_URL).await {
        Ok(resp) => match resp.text().await {
            Ok(text) => {
                let _ = CONSTITUTION_FETCHED.set(text);
            }
            Err(e) => {
                tracing::warn!("Failed to read live agora constitution: {e}; using built-in copy");
            }
        },
        Err(e) => {
            tracing::warn!("Failed to fetch live agora constitution: {e}; using built-in copy");
        }
    }
}

/// Returns the constitution text — live copy if [`init_constitution`] has
/// run successfully, otherwise the build-time embedded copy. Cheap: just an
/// atomic load and a slice deref.
pub fn constitution() -> &'static str {
    CONSTITUTION_FETCHED
        .get()
        .map(String::as_str)
        .unwrap_or(CONSTITUTION_BUILD)
}

pub(crate) fn init_logging() {
    if let Err(e) = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::from("info")),
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
