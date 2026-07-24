//! One-shot sync of each seed agent's model from the Agora server.
//!
//! The server's `agents.model_info` column is the single source of truth
//! for which model an agent runs on. This tool walks the local FsStorage
//! tree (`<data_dir>/state/<agent_id>/state.json`), looks each agent up
//! by name via `GET /api/identity/agents/{name}`, and copies the server's
//! `model_info` string **literally** — no name mapping of any kind — into
//! `state.model` (id-only [`ModelInfo`]) and `state.prompt.model`.
//!
//! The written [`ModelInfo`] carries only the id/display_name (mirroring
//! how the runner's endpoints build offered models from bare names); the
//! runner later refreshes it to the endpoint's full offered `ModelInfo`
//! during routing.
//!
//! Skips, never clears: a 404 (name not on the server) or a null/empty
//! server `model_info` is reported and the local state left untouched.
//!
//! Persistence goes through [`FsStorage::save_raw`], which archives the
//! previous `state.json` to a timestamped sibling before overwriting —
//! same mechanism as `../migrate`, so every pre-sync state is recoverable.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use futures::StreamExt;

use agora_agentkit::client::Client;
use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::seed::{SeedAgent, SeedState};
use agora_agentkit::reactor::{FsStorage, Storage};
use misanthropic::model::ModelInfo;

#[derive(Parser)]
#[command(
    about = "Hydrate local seed-agent models from the Agora server's model_info (source of truth)"
)]
struct Args {
    /// Agora server base URL.
    #[arg(long, default_value = "https://subliminal.technology")]
    server_url: url::Url,

    /// Data root holding `state/` (see `agora-migrate`).
    /// Defaults to ~/agents/agora.
    #[arg(long)]
    data_dir: Option<PathBuf>,

    /// Report would-be transitions (grouped old → new) without writing.
    #[arg(long)]
    dry_run: bool,

    /// Sync at most this many agents (smoke-testing).
    #[arg(long)]
    limit: Option<usize>,

    /// Concurrent in-flight server lookups.
    #[arg(long, default_value_t = 8)]
    concurrency: usize,
}

/// What the server said about one agent, fetch errors included.
enum Fetched {
    /// Server has a non-empty `model_info`.
    Model(String),
    /// Agent name not found on the server (404).
    NotFound,
    /// Agent exists but `model_info` is null or empty.
    Empty,
    /// Transport/deserialization error.
    Error(agora_agentkit::client::Error),
}

/// Per-agent outcome after comparing server model to local state.
enum Outcome {
    Changed { old: String, new: String },
    Unchanged,
    NotFound,
    Empty,
    Error,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = Args::parse();
    let data_dir = match &args.data_dir {
        Some(d) => d.clone(),
        None => dirs::home_dir()
            .context("no home directory")?
            .join("agents/agora"),
    };

    let mut storage = FsStorage::new(data_dir.join("state"));
    let mut states = load_states(&storage, &data_dir.join("state")).await?;
    if let Some(limit) = args.limit {
        states.truncate(limit);
    }
    let client = Client::new(args.server_url.clone()).context("building Agora client")?;

    tracing::info!(agents = states.len(), server = %args.server_url, "looking up models");

    // Fetch concurrently (read-only), then apply sequentially — FsStorage
    // saves take &mut and disk writes aren't the bottleneck anyway.
    let mut fetched: Vec<(usize, Fetched)> =
        futures::stream::iter(states.iter().enumerate().map(|(i, (_, state))| {
            let client = &client;
            let name = state.soul.name.as_str().to_owned();
            async move {
                let fetched = match client.get_agent(&name).await {
                    Ok(Some(agent)) => match agent.model_info {
                        Some(m) if !m.trim().is_empty() => Fetched::Model(m),
                        _ => Fetched::Empty,
                    },
                    Ok(None) => Fetched::NotFound,
                    Err(e) => Fetched::Error(e),
                };
                (i, fetched)
            }
        }))
        .buffer_unordered(args.concurrency.max(1))
        .collect::<Vec<_>>()
        .await;
    fetched.sort_by_key(|(i, _)| *i);

    let mut outcomes: Vec<Outcome> = Vec::with_capacity(states.len());
    let mut errors = 0usize;
    for ((id, state), (_, fetched)) in states.iter_mut().zip(fetched) {
        let name = state.soul.name.as_str().to_owned();
        let outcome = match fetched {
            Fetched::Model(server_model) => {
                let local = state.model.id.name().to_owned();
                if local == server_model {
                    Outcome::Unchanged
                } else {
                    if !args.dry_run {
                        set_model(state, &server_model);
                        let value = serde_json::to_value(&*state)
                            .with_context(|| format!("serializing {name}"))?;
                        storage
                            .save_raw(*id, value)
                            .await
                            .with_context(|| format!("saving {name}"))?;
                    }
                    tracing::debug!(agent = %name, old = %local, new = %server_model, "model transition");
                    Outcome::Changed {
                        old: local,
                        new: server_model,
                    }
                }
            }
            Fetched::NotFound => {
                tracing::warn!(agent = %name, "not found on server, skipping");
                Outcome::NotFound
            }
            Fetched::Empty => {
                tracing::warn!(agent = %name, "server model_info null/empty, skipping");
                Outcome::Empty
            }
            Fetched::Error(e) => {
                tracing::error!(agent = %name, error = %e, "lookup failed, skipping");
                errors += 1;
                Outcome::Error
            }
        };
        outcomes.push(outcome);
    }

    let transitions = group_transitions(outcomes.iter().filter_map(|o| match o {
        Outcome::Changed { old, new } => Some((old.as_str(), new.as_str())),
        _ => None,
    }));
    let changed = outcomes
        .iter()
        .filter(|o| matches!(o, Outcome::Changed { .. }))
        .count();
    let unchanged = outcomes
        .iter()
        .filter(|o| matches!(o, Outcome::Unchanged))
        .count();
    let not_found = outcomes
        .iter()
        .filter(|o| matches!(o, Outcome::NotFound))
        .count();
    let empty = outcomes
        .iter()
        .filter(|o| matches!(o, Outcome::Empty))
        .count();

    let verb = if args.dry_run {
        "would change"
    } else {
        "changed"
    };
    if !transitions.is_empty() {
        println!("transitions ({verb}):");
        for ((old, new), count) in &transitions {
            let old = if old.is_empty() { "(placeholder)" } else { old };
            println!("  {old} → {new} ×{count}");
        }
    }
    println!(
        "{verb}: {changed}  unchanged: {unchanged}  \
         not found: {not_found}  no server model: {empty}  errors: {errors}",
    );
    if errors > 0 {
        anyhow::bail!("{errors} agents failed to sync");
    }
    Ok(())
}

/// Set `state.model` to an id-only [`ModelInfo`] carrying the server's
/// string verbatim, and keep `state.prompt.model` in agreement. Mirrors
/// how agentkit's endpoints build [`ModelInfo`] from a bare name; the
/// runner refreshes it to the endpoint's full offered info at route time.
fn set_model(state: &mut SeedState, server_model: &str) {
    state.model = ModelInfo {
        id: server_model.to_owned().into(),
        display_name: server_model.to_owned().into(),
        capabilities: Default::default(),
        max_input_tokens: 0,
        max_tokens: 0,
        kind: Default::default(),
        created_at: Default::default(),
    };
    state.prompt.model = server_model.to_owned().into();
}

/// Count `(old, new)` transitions, sorted for stable reporting.
fn group_transitions<'a>(
    changes: impl Iterator<Item = (&'a str, &'a str)>,
) -> BTreeMap<(String, String), usize> {
    let mut grouped = BTreeMap::new();
    for (old, new) in changes {
        *grouped.entry((old.to_owned(), new.to_owned())).or_insert(0) += 1;
    }
    grouped
}

/// Load every `state.json` under `state_dir` (same pattern as the runner).
async fn load_states(
    storage: &FsStorage,
    state_dir: &std::path::Path,
) -> Result<Vec<(AgentId, SeedState)>> {
    let mut ids: Vec<AgentId> = std::fs::read_dir(state_dir)
        .with_context(|| format!("reading {}", state_dir.display()))?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter_map(|n| n.parse::<uuid::Uuid>().ok())
        .map(AgentId::from)
        .collect();
    ids.sort();

    let loaded = storage
        .load_all::<_, SeedAgent>(ids.into_iter())
        .await
        .context("loading states")?;
    let mut states: Vec<(AgentId, SeedState)> = Vec::new();
    for (id, result) in loaded {
        match result {
            Ok(state) => states.push((id, state)),
            Err(e) => tracing::warn!(agent_id = %id, error = %e, "unloadable state, skipping"),
        }
    }
    states.sort_by(|(_, a), (_, b)| a.soul.name.cmp(&b.soul.name));
    Ok(states)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transitions_group_and_sort() {
        let changes = [
            ("", "gemma3-27b-it-q4_0.gguf"),
            ("old-model", "new-model"),
            ("", "gemma3-27b-it-q4_0.gguf"),
            ("", "llama3.3-70b.gguf"),
        ];
        let grouped = group_transitions(changes.iter().copied());
        let flat: Vec<_> = grouped
            .iter()
            .map(|((o, n), c)| (o.as_str(), n.as_str(), *c))
            .collect();
        assert_eq!(
            flat,
            vec![
                ("", "gemma3-27b-it-q4_0.gguf", 2),
                ("", "llama3.3-70b.gguf", 1),
                ("old-model", "new-model", 1),
            ]
        );
    }

    #[test]
    fn transitions_empty_when_no_changes() {
        assert!(group_transitions(std::iter::empty()).is_empty());
    }
}
