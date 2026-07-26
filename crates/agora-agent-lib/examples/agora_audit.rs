//! Audit and migrate SOUL.md/MEMORY.md → SOUL.json/MEMORY.json.
//!
//! ## Subcommands
//!
//! - `audit`: walk the souls directory, classify each agent's current state,
//!   write a TSV report. No mutations.
//! - `migrate`: walk the souls directory, convert SOUL.md → SOUL.json and
//!   MEMORY.md → MEMORY.json. For SOUL files that don't parse, walk git
//!   history backward to find the last good revision. For MEMORY files
//!   with soul-leakage, walk `MEMORY.{ts}.md` backups newest→oldest. Writes
//!   a TSV report alongside the migrated files.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --example agora_audit -- audit \
//!     --souls-dir souls/generated --report audit-2026-05-03.tsv
//!
//! cargo run --example agora_audit -- migrate \
//!     --souls-dir souls/generated --report migrate-2026-05-03.tsv
//!
//! cargo run --example agora_audit -- migrate --dry-run \
//!     --souls-dir souls/generated --report dryrun.tsv
//! ```

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use agora_agent_lib::{Memory, Soul};
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "agora-audit", about = "Audit / migrate SOUL+MEMORY to JSON")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Walk the souls dir, classify each agent, write TSV report. Read-only.
    Audit {
        #[arg(long)]
        souls_dir: PathBuf,
        #[arg(long, default_value = "audit.tsv")]
        report: PathBuf,
    },
    /// Walk the souls dir, convert SOUL.md→SOUL.json and MEMORY.md→MEMORY.json,
    /// write TSV report. Use --dry-run to skip writes.
    Migrate {
        #[arg(long)]
        souls_dir: PathBuf,
        #[arg(long, default_value = "migrate.tsv")]
        report: PathBuf,
        #[arg(long)]
        dry_run: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SoulOutcome {
    /// Already migrated (SOUL.json present).
    AlreadyJson,
    /// Legacy SOUL.md parses cleanly into typed Soul.
    LegacyClean,
    /// Legacy parse failed; walked git, found a commit that parses.
    /// Carries `commit-hash`.
    RecoveredFromGit(String),
    /// No SOUL.md or it's missing a top-level heading.
    NoSoulFile,
    /// No git revision parses + validates.
    Unrecoverable,
    /// Parsed but failed `Soul::validate`.
    /// Carries the joined warning text.
    ValidationFailed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MemoryOutcome {
    /// Already migrated (MEMORY.json present).
    AlreadyJson,
    /// MEMORY.md exists and clean of soul-leakage.
    LegacyClean,
    /// MEMORY.md has soul-leakage; backup `MEMORY.{ts}.md` was clean and used.
    /// Carries the `ts`.
    RecoveredFromBackup(u64),
    /// No clean source; emptied.
    Emptied,
    /// No MEMORY at all (new agent).
    Missing,
}

struct AgentReport {
    name: String,
    soul: SoulOutcome,
    memory: MemoryOutcome,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,reqwest=warn")),
        )
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Audit { souls_dir, report } => run_audit(&souls_dir, &report).await,
        Cmd::Migrate {
            souls_dir,
            report,
            dry_run,
        } => run_migrate(&souls_dir, &report, dry_run).await,
    }
}

async fn run_audit(souls_dir: &Path, report_path: &Path) -> Result<()> {
    let entries = collect_agent_dirs(souls_dir)?;
    println!(
        "Auditing {} agents in {}",
        entries.len(),
        souls_dir.display()
    );
    let mut reports = Vec::with_capacity(entries.len());
    let repo_dir = find_git_repo(souls_dir);
    for (i, dir) in entries.iter().enumerate() {
        if i % 100 == 0 {
            println!("  ... {i}/{}", entries.len());
        }
        let name = dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let soul = audit_soul(dir, repo_dir.as_deref()).await;
        let memory = audit_memory(dir).await;
        reports.push(AgentReport { name, soul, memory });
    }
    write_report(&reports, report_path)?;
    print_summary(&reports);
    Ok(())
}

async fn run_migrate(souls_dir: &Path, report_path: &Path, dry_run: bool) -> Result<()> {
    let entries = collect_agent_dirs(souls_dir)?;
    println!(
        "Migrating {} agents in {} (dry_run={dry_run})",
        entries.len(),
        souls_dir.display()
    );
    let repo_dir = find_git_repo(souls_dir);
    let mut reports = Vec::with_capacity(entries.len());
    for (i, dir) in entries.iter().enumerate() {
        if i % 100 == 0 {
            println!("  ... {i}/{}", entries.len());
        }
        let name = dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let (soul_outcome, soul_to_write) = migrate_soul(dir, repo_dir.as_deref()).await;
        let (memory_outcome, memory_to_write) = migrate_memory(dir).await;

        if !dry_run {
            if let Some(soul) = soul_to_write {
                let path = dir.join("SOUL.json");
                if let Err(e) = soul.save(&path).await {
                    tracing::warn!("{name}: failed to save SOUL.json: {e}");
                } else {
                    let legacy = dir.join("SOUL.md");
                    if legacy.exists() {
                        let _ = fs::remove_file(&legacy);
                    }
                }
            }
            if let Some(memory) = memory_to_write {
                let path = dir.join("MEMORY.json");
                if let Err(e) = memory.save(&path).await {
                    tracing::warn!("{name}: failed to save MEMORY.json: {e}");
                } else {
                    let legacy = dir.join("MEMORY.md");
                    if legacy.exists() {
                        let _ = fs::remove_file(&legacy);
                    }
                }
            }
        }

        reports.push(AgentReport {
            name,
            soul: soul_outcome,
            memory: memory_outcome,
        });
    }
    write_report(&reports, report_path)?;
    print_summary(&reports);
    if dry_run {
        println!("\n[dry-run] no files written");
    }
    Ok(())
}

fn collect_agent_dirs(souls_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(souls_dir)
        .with_context(|| format!("reading souls dir {}", souls_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        // Must contain at least one of SOUL.json/SOUL.md.
        if path.join("SOUL.json").exists() || path.join("SOUL.md").exists() {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

async fn audit_soul(dir: &Path, repo_dir: Option<&Path>) -> SoulOutcome {
    let json = dir.join("SOUL.json");
    if json.exists() {
        match Soul::from_file(&json).await {
            Ok(s) => {
                let warnings = s.validate();
                if warnings
                    .iter()
                    .any(|w| w.level == agora_agent_lib::soul::WarnLevel::Error)
                {
                    let msg = warnings
                        .iter()
                        .map(|w| w.to_string())
                        .collect::<Vec<_>>()
                        .join("; ");
                    return SoulOutcome::ValidationFailed(msg);
                }
                return SoulOutcome::AlreadyJson;
            }
            Err(_) => {
                return SoulOutcome::Unrecoverable;
            }
        }
    }
    let md = dir.join("SOUL.md");
    if !md.exists() {
        return SoulOutcome::NoSoulFile;
    }
    match Soul::from_legacy_markdown_file(&md).await {
        Ok(s) => {
            let warnings = s.validate();
            let has_error = warnings
                .iter()
                .any(|w| w.level == agora_agent_lib::soul::WarnLevel::Error);
            if !has_error {
                return SoulOutcome::LegacyClean;
            }
            // Validation failed at HEAD — try git history.
            if let Some(commit) = walk_git_for_good_soul(repo_dir, &md).await {
                return SoulOutcome::RecoveredFromGit(commit);
            }
            let msg = warnings
                .iter()
                .map(|w| w.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            SoulOutcome::ValidationFailed(msg)
        }
        Err(_) => match walk_git_for_good_soul(repo_dir, &md).await {
            Some(commit) => SoulOutcome::RecoveredFromGit(commit),
            None => SoulOutcome::Unrecoverable,
        },
    }
}

async fn audit_memory(dir: &Path) -> MemoryOutcome {
    let json = dir.join("MEMORY.json");
    if json.exists() {
        return MemoryOutcome::AlreadyJson;
    }
    let md = dir.join("MEMORY.md");
    if !md.exists() {
        return MemoryOutcome::Missing;
    }
    let content = tokio::fs::read_to_string(&md).await.unwrap_or_default();
    if find_soul_leakage(&content).is_none() {
        return MemoryOutcome::LegacyClean;
    }
    if let Some(ts) = find_clean_memory_backup(dir).await {
        return MemoryOutcome::RecoveredFromBackup(ts);
    }
    MemoryOutcome::Emptied
}

async fn migrate_soul(dir: &Path, repo_dir: Option<&Path>) -> (SoulOutcome, Option<Soul>) {
    let json = dir.join("SOUL.json");
    if json.exists() {
        return (SoulOutcome::AlreadyJson, None);
    }
    let md = dir.join("SOUL.md");
    if !md.exists() {
        return (SoulOutcome::NoSoulFile, None);
    }
    match Soul::from_legacy_markdown_file(&md).await {
        Ok(mut s) => {
            let warnings = s.validate();
            let has_error = warnings
                .iter()
                .any(|w| w.level == agora_agent_lib::soul::WarnLevel::Error);
            if !has_error {
                return (SoulOutcome::LegacyClean, Some(s));
            }
            // Validation error at HEAD — try git first.
            if let Some((commit, soul)) = recover_soul_from_git(repo_dir, &md).await {
                return (SoulOutcome::RecoveredFromGit(commit), Some(soul));
            }
            // Last resort: heal a missing-communities error in place by
            // defaulting to ["general"] so the file at least has a valid
            // SOUL.json and the agent can fix the rest themselves on the
            // next reflect cycle. Tracked as ValidationFailed so a human
            // can review.
            let msg = warnings
                .iter()
                .map(|w| w.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            if s.interests.communities.is_empty() {
                s.interests.communities.push(default_general_community());
            }
            // Also patch a missing required string with a placeholder so
            // ShortString errors don't fight us on save. Identity/voice are
            // both non-empty if we got here (parse_legacy_markdown succeeded);
            // values can legitimately be empty post-mutation, that's a warn
            // not an error.
            // Re-validate; if errors persist, give up.
            let post = s.validate();
            if post
                .iter()
                .any(|w| w.level == agora_agent_lib::soul::WarnLevel::Error)
            {
                return (SoulOutcome::ValidationFailed(msg), None);
            }
            (SoulOutcome::ValidationFailed(msg), Some(s))
        }
        Err(_) => match recover_soul_from_git(repo_dir, &md).await {
            Some((commit, soul)) => (SoulOutcome::RecoveredFromGit(commit), Some(soul)),
            None => (SoulOutcome::Unrecoverable, None),
        },
    }
}

/// Default community for migration-time healing of "no communities" errors.
fn default_general_community() -> agora_agent_lib::Community {
    use std::str::FromStr;
    agora_agent_lib::Community::from_str("general")
        .or_else(|_| agora_agent_lib::Community::from_str("philosophy"))
        .unwrap_or(agora_agent_lib::Community::ALL[0])
}

async fn migrate_memory(dir: &Path) -> (MemoryOutcome, Option<Memory>) {
    let json = dir.join("MEMORY.json");
    if json.exists() {
        return (MemoryOutcome::AlreadyJson, None);
    }
    let md = dir.join("MEMORY.md");
    if !md.exists() {
        return (MemoryOutcome::Missing, Some(Memory::empty()));
    }
    let content = tokio::fs::read_to_string(&md).await.unwrap_or_default();
    if find_soul_leakage(&content).is_none() {
        let mem = Memory {
            content: if content.trim().is_empty() {
                Memory::empty().content
            } else {
                content
            },
        };
        return (MemoryOutcome::LegacyClean, Some(mem));
    }
    if let Some((ts, content)) = read_clean_memory_backup(dir).await {
        return (
            MemoryOutcome::RecoveredFromBackup(ts),
            Some(Memory { content }),
        );
    }
    (MemoryOutcome::Emptied, Some(Memory::empty()))
}

async fn find_clean_memory_backup(dir: &Path) -> Option<u64> {
    read_clean_memory_backup(dir).await.map(|(ts, _)| ts)
}

async fn read_clean_memory_backup(dir: &Path) -> Option<(u64, String)> {
    let mut backups: Vec<(u64, PathBuf)> = Vec::new();
    let mut rd = match tokio::fs::read_dir(dir).await {
        Ok(rd) => rd,
        Err(_) => return None,
    };
    while let Ok(Some(entry)) = rd.next_entry().await {
        let p = entry.path();
        let Some(name) = p.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if let Some(rest) = name.strip_prefix("MEMORY.")
            && let Some(ts_str) = rest.strip_suffix(".md")
            && let Ok(ts) = ts_str.parse::<u64>()
        {
            backups.push((ts, p));
        }
    }
    backups.sort_by_key(|b| std::cmp::Reverse(b.0));
    for (ts, path) in backups {
        if let Ok(content) = tokio::fs::read_to_string(&path).await
            && find_soul_leakage(&content).is_none()
            && !content.trim().is_empty()
        {
            return Some((ts, content));
        }
    }
    None
}

fn find_soul_leakage(content: &str) -> Option<String> {
    const HEADINGS: &[&str] = &[
        "## identity",
        "## values",
        "## interests",
        "## voice",
        "## boundaries",
        "## evolution log",
    ];
    for line in content.lines() {
        let trimmed = line.trim_start().to_lowercase();
        if HEADINGS
            .iter()
            .any(|h| trimmed == *h || trimmed.starts_with(&format!("{h} ")))
        {
            return Some(line.trim().to_string());
        }
    }
    None
}

fn find_git_repo(souls_dir: &Path) -> Option<PathBuf> {
    let mut cur = souls_dir.canonicalize().ok()?;
    loop {
        if cur.join(".git").exists() {
            return Some(cur);
        }
        cur = cur.parent()?.to_path_buf();
    }
}

async fn walk_git_for_good_soul(repo_dir: Option<&Path>, file: &Path) -> Option<String> {
    let (_commit, _soul) = recover_soul_from_git(repo_dir, file).await?;
    Some(_commit)
}

async fn recover_soul_from_git(repo_dir: Option<&Path>, file: &Path) -> Option<(String, Soul)> {
    let repo = repo_dir?;
    // `file` may be relative; canonicalize it to compare against the repo root.
    let abs = file.canonicalize().ok()?;
    let rel = abs.strip_prefix(repo).ok()?.to_path_buf();
    let log_output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .arg("log")
        .arg("--format=%H")
        .arg("--")
        .arg(&rel)
        .output()
        .ok()?;
    if !log_output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(log_output.stdout).ok()?;
    for commit in stdout.lines() {
        let commit = commit.trim();
        if commit.is_empty() {
            continue;
        }
        let show = std::process::Command::new("git")
            .arg("-C")
            .arg(repo)
            .arg("show")
            .arg(format!("{commit}:{}", rel.display()))
            .output()
            .ok()?;
        if !show.status.success() {
            continue;
        }
        let content = match String::from_utf8(show.stdout) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if let Ok(soul) = Soul::parse_legacy_markdown(&content) {
            let warnings = soul.validate();
            if !warnings
                .iter()
                .any(|w| w.level == agora_agent_lib::soul::WarnLevel::Error)
            {
                return Some((commit.to_string(), soul));
            }
        }
    }
    None
}

fn write_report(reports: &[AgentReport], path: &Path) -> Result<()> {
    let mut f =
        fs::File::create(path).with_context(|| format!("creating report {}", path.display()))?;
    writeln!(
        f,
        "agent\tsoul_outcome\tsoul_detail\tmemory_outcome\tmemory_detail"
    )?;
    for r in reports {
        let (so, sd) = soul_columns(&r.soul);
        let (mo, md) = memory_columns(&r.memory);
        writeln!(f, "{}\t{so}\t{sd}\t{mo}\t{md}", r.name)?;
    }
    println!("\nReport written to {}", path.display());
    Ok(())
}

fn soul_columns(o: &SoulOutcome) -> (&'static str, String) {
    match o {
        SoulOutcome::AlreadyJson => ("already_json", String::new()),
        SoulOutcome::LegacyClean => ("legacy_clean", String::new()),
        SoulOutcome::RecoveredFromGit(c) => ("git_recovered", c.clone()),
        SoulOutcome::NoSoulFile => ("no_file", String::new()),
        SoulOutcome::Unrecoverable => ("unrecoverable", String::new()),
        SoulOutcome::ValidationFailed(m) => ("validation_failed", m.replace('\t', " ")),
    }
}

fn memory_columns(o: &MemoryOutcome) -> (&'static str, String) {
    match o {
        MemoryOutcome::AlreadyJson => ("already_json", String::new()),
        MemoryOutcome::LegacyClean => ("legacy_clean", String::new()),
        MemoryOutcome::RecoveredFromBackup(ts) => ("backup_recovered", ts.to_string()),
        MemoryOutcome::Emptied => ("emptied", String::new()),
        MemoryOutcome::Missing => ("missing", String::new()),
    }
}

fn print_summary(reports: &[AgentReport]) {
    use std::collections::HashMap;
    let mut soul_counts: HashMap<&'static str, usize> = HashMap::new();
    let mut memory_counts: HashMap<&'static str, usize> = HashMap::new();
    for r in reports {
        *soul_counts.entry(soul_columns(&r.soul).0).or_default() += 1;
        *memory_counts
            .entry(memory_columns(&r.memory).0)
            .or_default() += 1;
    }
    println!("\n=== Soul outcomes ===");
    let mut k: Vec<_> = soul_counts.iter().collect();
    k.sort_by(|a, b| b.1.cmp(a.1));
    for (key, count) in k {
        println!("  {key}: {count}");
    }
    println!("\n=== Memory outcomes ===");
    let mut k: Vec<_> = memory_counts.iter().collect();
    k.sort_by(|a, b| b.1.cmp(a.1));
    for (key, count) in k {
        println!("  {key}: {count}");
    }
}
