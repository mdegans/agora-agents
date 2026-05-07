use anyhow::Context;
use tracing_appender::non_blocking;

use std::{fs::File, path::Path, sync::OnceLock};

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

pub(crate) fn init_logging(
    logfile: Option<impl AsRef<Path>>,
) -> anyhow::Result<(non_blocking::WorkerGuard, non_blocking::WorkerGuard)> {
    use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

    // logfile defaults to `seed-log.{ts}.json`
    let logfile = if let Some(path) = logfile {
        File::create(path)?
    } else {
        let ts = chrono::Utc::now().timestamp();
        File::create(format!("seed-log.{ts}.jsonl"))?
    };
    let (stderr, stderr_guard) = non_blocking(std::io::stderr());
    let (logfile, logfile_guard) = non_blocking(logfile);

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::from("info"));

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().json().with_writer(stderr))
        .with(
            fmt::layer()
                .json()
                .with_line_number(true)
                .with_writer(logfile),
        )
        .init();

    Ok((stderr_guard, logfile_guard))
}

pub(crate) async fn read_file_stripped(path: impl AsRef<Path>) -> anyhow::Result<String> {
    Ok(tokio::fs::read_to_string(path.as_ref())
        .await
        .with_context(|| format!("Reading file: {}", path.as_ref().display()))?
        .trim()
        .to_string())
}
