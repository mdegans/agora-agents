//! Tracing setup: pretty to stderr for whoever is watching, JSON lines to
//! a file so a run is still queryable tomorrow.
//!
//! Restores the sink the pre-cutover scheduler seed had (agora-agents#91).
//! Both writers are non-blocking — `tracing-appender` hands each off to a
//! worker thread, so a slow terminal or a slow disk never stalls an agent
//! mid-session.
//!
//! The file lands under the **data dir** (`--data-dir`, default
//! `~/agents/agora/logs/`), never the repo and never the CWD. Two reasons:
//! a dev `cargo run` and the installed binary would otherwise fight over
//! one path, and a run's logs quote fully-rendered prompts — SOUL, memory,
//! dashboard — which must never be one `git add -A` away from a public
//! repo. The prompt dumps ([`agora_agentkit`]'s `prompt_log`) sit beside
//! them under `logs/prompts/` for the same reason.
//!
//! `seed-log.{unix_ts}.jsonl` is the historical filename and is kept
//! deliberately: the archive going back to 2026-04 is named this way, and
//! anything that reads it globs for exactly this shape.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
// Both the module and the same-named constructor function — they live in
// different namespaces, and a `{self, WorkerGuard}` import would bring in
// only the module, leaving the calls below unresolved.
use tracing_appender::non_blocking;
use tracing_appender::non_blocking::WorkerGuard;

/// Live handles for the non-blocking writers.
///
/// **Keep this alive for the whole process.** Each guard flushes its
/// worker on drop; dropping one early truncates that sink silently, which
/// is exactly the failure you don't notice until you go looking for the
/// log that explains a bad run.
#[must_use = "dropping the guards truncates the log"]
pub struct Guards {
    _stderr: WorkerGuard,
    _file: Option<WorkerGuard>,
}

/// Where the file sink will write, given the resolved directory.
fn log_path(dir: &Path) -> PathBuf {
    let ts = chrono::Utc::now().timestamp();
    dir.join(format!("seed-log.{ts}.jsonl"))
}

/// Install the subscriber. `dir` overrides the default
/// `<data_dir>/logs`; `to_file` false leaves stderr as the only sink.
///
/// Returns the guards and the log path, if a file was opened. Tracing is
/// not up yet while this runs, so the path is returned for the caller to
/// announce rather than logged here — a line naming the log file is the
/// one thing a reader needs *before* the log exists.
pub fn init(
    dir: Option<&Path>,
    data_dir: &Path,
    to_file: bool,
) -> Result<(Guards, Option<PathBuf>)> {
    use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

    let (stderr, stderr_guard) = non_blocking(std::io::stderr());
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::from("info"));

    let file = if to_file {
        let dir = dir.map_or_else(|| data_dir.join("logs"), Path::to_path_buf);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating log directory {}", dir.display()))?;
        let path = log_path(&dir);
        let handle = std::fs::File::create(&path)
            .with_context(|| format!("creating log file {}", path.display()))?;
        Some((handle, path))
    } else {
        None
    };

    // Split so the file layer is only built when there is a file; a
    // `None` layer would still need a writer.
    let (file_layer, file_guard, path) = match file {
        Some((handle, path)) => {
            let (writer, guard) = non_blocking(handle);
            let layer = fmt::layer()
                .json()
                .with_line_number(true)
                .with_writer(writer);
            (Some(layer), Some(guard), Some(path))
        }
        None => (None, None, None),
    };

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_writer(stderr))
        .with(file_layer)
        .init();

    Ok((
        Guards {
            _stderr: stderr_guard,
            _file: file_guard,
        },
        path,
    ))
}
