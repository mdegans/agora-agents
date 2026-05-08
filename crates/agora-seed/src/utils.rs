use agora_agent_lib::info_payload;
use anyhow::Context;
use serde::Serialize;
use tokio::process::Command;
use tracing_appender::non_blocking;

use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::OnceLock,
};

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

    // Default lands under `logs/` so the auto-named jsonl doesn't trip the
    // assert_clean preflight. `logs/` is gitignored in the agents repo. When
    // invoked via the parent justfile cwd is `agents/` so this resolves to
    // `agents/logs/seed-log.{ts}.jsonl`.
    let logfile = if let Some(path) = logfile {
        File::create(path)?
    } else {
        std::fs::create_dir_all("logs").context("creating logs/ directory")?;
        let ts = chrono::Utc::now().timestamp();
        File::create(format!("logs/seed-log.{ts}.jsonl"))?
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

// Credit to Claude Opus 4.7
pub async fn walk_up_to_git(start: impl AsRef<Path>) -> anyhow::Result<PathBuf> {
    let start = tokio::fs::canonicalize(start.as_ref())
        .await
        .with_context(|| format!("canonicalizing {}", start.as_ref().display()))?;

    for ancestor in start.ancestors() {
        if tokio::fs::try_exists(ancestor.join(".git")).await? {
            return Ok(ancestor.to_path_buf());
        }
    }

    anyhow::bail!(
        "no .git directory or file found in any parent of {}",
        start.display()
    );
}

// Credit to Claude Opus 4.7
pub async fn assert_clean(start: impl AsRef<Path>) -> anyhow::Result<()> {
    let repo = walk_up_to_git(start).await?;

    let output = Command::new("git")
        .arg("-C")
        .arg(&repo)
        .args(["status", "--porcelain", "--", "."])
        .output()
        .await
        .context("running git status")?;

    if !output.status.success() {
        anyhow::bail!(
            "git status failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    if !output.stdout.is_empty() {
        anyhow::bail!(
            r#"agents repo has uncommitted changes:

```stdout
{}
```

Commit before launching a seed run so this run's provenance is anchored.
This includes SOUL.json mutations from prior runs — those need to be committed too, otherwise this run's mutations will overwrite them."#,
            String::from_utf8_lossy(&output.stdout).trim_end()
        );
    }

    let branch_output = Command::new("git")
        .arg("-C")
        .arg(&repo)
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .await
        .context("running git rev-parse --abbrev-ref HEAD")?;
    if !branch_output.status.success() {
        anyhow::bail!(
            "git rev-parse failed: {}",
            String::from_utf8_lossy(&branch_output.stderr)
        );
    }
    let branch = String::from_utf8_lossy(&branch_output.stdout)
        .trim()
        .to_string();
    if branch != "main" {
        anyhow::bail!(
            "agents repo is on branch {branch:?}, not `main`. Switch to main \
             before launching a seed run so this run's mutations are \
             attributable to canonical history.\n\nA detached HEAD reports as \
             \"HEAD\" and is also rejected."
        );
    }

    let sha = tokio::process::Command::new("git")
        .arg("-C")
        .arg(&repo)
        .args(["rev-parse", "HEAD"])
        .output()
        .await?;
    let git_sha = String::from_utf8(sha.stdout)?.trim().to_string();
    #[derive(Serialize)]
    struct GitStatusPorcelain<'a> {
        git_sha: &'a str,
    }
    info_payload!(GitStatusPorcelain { git_sha: &git_sha });

    Ok(())
}
