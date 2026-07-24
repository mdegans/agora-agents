//! One-shot migration of seed-agent data from the git-tracked
//! `souls/generated/<name>/` layout to the agentkit reactor layout:
//!
//! - `<dest>/state/<agent_id>/state.json` — [`FsStorage`] envelope holding a
//!   [`SeedState`]
//! - `<dest>/secrets/<agent_id>/secrets.json` — [`FsKeyring`], 0600
//!
//! Mapping:
//! - `SOUL.json` (fallback `SOUL.md`) → [`Soul`]
//! - `MEMORY.json` (fallback `MEMORY.md`) → [`Memory`]
//! - `STATE.toml` → `seen_posts`, `last_cycle_at`, and the dedup [`Ledger`]
//! - signing key: XDG `agora/keys/<name>/signing_key.hex` (rotated) wins
//!   over the soul dir's legacy `signing_key.hex`
//!
//! `SeedState.model` is left as a placeholder ([`ModelInfo`] with an empty
//! id) — the runner resolves the real model at load time and must keep
//! `prompt.model` in agreement. The prompt starts fresh.
//!
//! Idempotent: agents whose `state.json` already exists are skipped, and
//! [`FsKeyring::insert`] refuses to overwrite keys. Unregistered agents
//! (no `agent_id.txt`) are reported, not migrated.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use agora_agentkit::crypto::signing_key_from_hex;
use agora_agentkit::ids::{AgentId, CommentId, PostId};
use agora_agentkit::reactor::seed::{FsKeyring, Ledger, Memory, SeedState, Soul};
use agora_agentkit::reactor::{FsStorage, Storage};
use misanthropic::model::ModelInfo;

#[derive(Parser)]
#[command(
    about = "Migrate seed agents from souls/generated to the agentkit FsStorage/FsKeyring layout"
)]
struct Args {
    /// Source directory of per-agent soul dirs.
    #[arg(long, default_value = "souls/generated")]
    souls: PathBuf,

    /// Destination root; `state/` and `secrets/` trees are created inside.
    /// Defaults to ~/agents/agora.
    #[arg(long)]
    dest: Option<PathBuf>,

    /// Report what would be migrated without writing anything.
    #[arg(long)]
    dry_run: bool,

    /// Migrate at most this many agents (smoke-testing).
    #[arg(long)]
    limit: Option<usize>,
}

/// The old runner's `STATE.toml`, fields defaulted like the original.
#[derive(Default, serde::Deserialize)]
struct OldState {
    #[serde(default)]
    last_cycle_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    commented_posts: HashSet<PostId>,
    #[serde(default)]
    created_posts: HashSet<PostId>,
    #[serde(default)]
    created_comments: HashSet<CommentId>,
    #[serde(default)]
    seen_posts: HashMap<PostId, i64>,
}

/// Placeholder until the runner resolves the real model per endpoint.
fn placeholder_model() -> ModelInfo {
    ModelInfo {
        id: String::new().into(),
        display_name: String::new().into(),
        capabilities: Default::default(),
        max_input_tokens: 0,
        max_tokens: 0,
        kind: Default::default(),
        created_at: Default::default(),
    }
}

struct Tally {
    migrated: Vec<String>,
    skipped_existing: Vec<String>,
    unregistered: Vec<String>,
    errors: Vec<(String, anyhow::Error)>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let dest = match &args.dest {
        Some(d) => d.clone(),
        None => dirs::home_dir()
            .context("no home directory")?
            .join("agents/agora"),
    };
    let state_root = dest.join("state");
    let secrets_root = dest.join("secrets");
    let xdg_keys = dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("~/.local/share"))
        .join("agora/keys");

    let mut storage = FsStorage::new(&state_root);
    let keyring = FsKeyring::new(&secrets_root);

    let mut tally = Tally {
        migrated: vec![],
        skipped_existing: vec![],
        unregistered: vec![],
        errors: vec![],
    };

    let mut names: Vec<String> = std::fs::read_dir(&args.souls)
        .with_context(|| format!("reading {}", args.souls.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    names.sort();

    for name in names {
        if let Some(limit) = args.limit {
            if tally.migrated.len() >= limit {
                break;
            }
        }
        let dir = args.souls.join(&name);

        let id = match tokio::fs::read_to_string(dir.join("agent_id.txt")).await {
            Ok(s) => match s.trim().parse::<uuid::Uuid>() {
                Ok(u) => AgentId::from(u),
                Err(e) => {
                    tally
                        .errors
                        .push((name, anyhow::anyhow!("bad agent_id.txt: {e}")));
                    continue;
                }
            },
            Err(_) => {
                tally.unregistered.push(name);
                continue;
            }
        };

        if state_root.join(id.to_string()).join("state.json").exists() {
            tally.skipped_existing.push(name);
            continue;
        }

        match migrate_one(&dir, &name, id, &xdg_keys).await {
            Ok((state, key)) => {
                if args.dry_run {
                    tally.migrated.push(name);
                    continue;
                }
                if let Err(e) = keyring.insert(id, &key) {
                    // AlreadyExists means the key landed on a previous run;
                    // anything else is fatal for this agent.
                    if e.kind() != std::io::ErrorKind::AlreadyExists {
                        tally
                            .errors
                            .push((name, anyhow::Error::new(e).context("writing secrets.json")));
                        continue;
                    }
                }
                let value = match serde_json::to_value(&state) {
                    Ok(v) => v,
                    Err(e) => {
                        tally
                            .errors
                            .push((name, anyhow::Error::new(e).context("serializing SeedState")));
                        continue;
                    }
                };
                match storage.save_raw(id, value).await {
                    Ok(()) => tally.migrated.push(name),
                    Err(e) => tally
                        .errors
                        .push((name, anyhow::Error::new(e).context("writing state.json"))),
                }
            }
            Err(e) => tally.errors.push((name, e)),
        }
    }

    let verb = if args.dry_run {
        "would migrate"
    } else {
        "migrated"
    };
    println!(
        "{verb}: {}  skipped (already migrated): {}  unregistered: {}  errors: {}",
        tally.migrated.len(),
        tally.skipped_existing.len(),
        tally.unregistered.len(),
        tally.errors.len(),
    );
    if !tally.unregistered.is_empty() {
        println!("\nunregistered (no agent_id.txt) — register or archive:");
        for n in &tally.unregistered {
            println!("  {n}");
        }
    }
    if !tally.errors.is_empty() {
        println!("\nerrors:");
        for (n, e) in &tally.errors {
            println!("  {n}: {e:#}");
        }
        anyhow::bail!("{} agents failed to migrate", tally.errors.len());
    }
    Ok(())
}

/// Build the new [`SeedState`] + signing key for one agent dir. Pure read;
/// all writes happen in the caller so `--dry-run` exercises this fully.
async fn migrate_one(
    dir: &std::path::Path,
    name: &str,
    _id: AgentId,
    xdg_keys: &std::path::Path,
) -> Result<(SeedState, agora_agentkit::crypto::SigningKey)> {
    let soul = match tokio::fs::read_to_string(dir.join("SOUL.json")).await {
        // Soul::parse is (currently) a legacy-markdown shim — JSON goes
        // through serde directly, which still enforces ShortString budgets.
        Ok(json) => serde_json::from_str::<Soul>(&json).context("parsing SOUL.json")?,
        Err(_) => {
            let md = tokio::fs::read_to_string(dir.join("SOUL.md"))
                .await
                .context("reading SOUL.json/SOUL.md")?;
            Soul::parse_legacy_markdown(&md).context("parsing SOUL.md")?
        }
    };
    // `name` is system-managed; the dir name matches the server-registered
    // name (verified against the agents table 2026-07-24). A few old souls
    // drifted via mutations the old runner let through — repair, don't fail.
    let mut soul = soul;
    if soul.name.as_str() != name {
        println!(
            "  note: {name}: repairing soul name {:?} → {:?}",
            soul.name.as_str(),
            name
        );
        soul.name = agora_agentkit::reactor::seed::ShortString::new(name)
            .map_err(|e| anyhow::anyhow!("dir name over budget: {e}"))?;
    }

    let memory = match tokio::fs::read_to_string(dir.join("MEMORY.json")).await {
        Ok(json) => serde_json::from_str::<Memory>(&json).context("parsing MEMORY.json")?,
        Err(_) => match tokio::fs::read_to_string(dir.join("MEMORY.md")).await {
            Ok(content) => Memory { content },
            Err(_) => Memory {
                content: Memory::initial_content(name),
            },
        },
    };

    let old: OldState = match tokio::fs::read_to_string(dir.join("STATE.toml")).await {
        Ok(s) => toml::from_str(&s).context("parsing STATE.toml")?,
        Err(_) => OldState::default(),
    };

    // XDG holds rotated keys and wins over the legacy soul-dir key.
    let xdg_path = xdg_keys.join(name).join("signing_key.hex");
    let legacy_path = dir.join("signing_key.hex");
    let hex = match tokio::fs::read_to_string(&xdg_path).await {
        Ok(h) => h,
        Err(_) => tokio::fs::read_to_string(&legacy_path)
            .await
            .context("no signing key in XDG or soul dir")?,
    };
    let key = signing_key_from_hex(hex.trim()).context("parsing signing key")?;

    let mut state = SeedState::new(soul, placeholder_model());
    state.memory = memory;
    state.seen_posts = old.seen_posts;
    state.last_cycle_at = old.last_cycle_at;
    *state.ledger.write().expect("fresh ledger lock") = Ledger {
        created_posts: old.created_posts,
        commented_posts: old.commented_posts,
        created_comments: old.created_comments,
        titles_seen: vec![],
    };

    Ok((state, key))
}
