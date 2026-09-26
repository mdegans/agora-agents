//! The queue: model changes agents asked for. Since 2026-09-25 the runner
//! applies them itself ([`super::switch`]); what is left here is the audit
//! trail, and any change the runner could not apply.
//!
//! Two views of one fact:
//!
//! - **Emitted** as it happens: one JSON line appended to
//!   `<data_dir>/model_consent/queue.jsonl` plus an `info` event
//!   (`event_type = "model_change_requested"`) in the run log.
//! - **Derived** on demand: `agora-seed --consent-queue` scans every
//!   agent's ledger for changes still awaiting application and prints the
//!   `set_model` commands, leaving out changes the runner has already
//!   applied (they take effect at the agent's next session). This view
//!   can't go stale, so it is the one to apply from. The JSON lines are the
//!   audit trail.
//!
//! Applying: `cargo run --bin set_model -- --from <from> --to <to> --agent
//! <name>…` in the parent repo, then `sync-models` here.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::seed::ShortString;
use chrono::{DateTime, Utc};
use misanthropic::model::Model;
use serde::{Deserialize, Serialize};

use super::ledger::{Change, ChangeAction, Ledger, RevertCause, Stage, Term};

/// The queue file, relative to the data dir.
pub fn queue_path(data_dir: &Path) -> PathBuf {
    data_dir.join("model_consent").join("queue.jsonl")
}

/// One line of `queue.jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueueEntry {
    pub at: DateTime<Utc>,
    pub agent_id: AgentId,
    pub agent: ShortString<64>,
    #[serde(flatten)]
    pub change: Change,
}

/// Append `entry` and log it. Best-effort: the ledger already holds the
/// fact, and `--consent-queue` derives the queue from ledgers, so a failed
/// append loses only the audit line (warned).
pub async fn emit(path: &Path, entry: &QueueEntry) {
    tracing::info!(
        event_type = "model_change_requested",
        agent = %entry.agent,
        agent_id = %entry.agent_id,
        action = ?entry.change.action,
        from = %entry.change.from,
        to = %entry.change.to,
        "model change requested; the runner applies it"
    );
    if let Err(e) = append(path, entry).await {
        tracing::warn!(
            path = %path.display(),
            agent_id = %entry.agent_id,
            error = %e,
            "model-change queue append failed; the ledger still has it"
        );
    }
}

async fn append(path: &Path, entry: &QueueEntry) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    if let Some(dir) = path.parent() {
        tokio::fs::create_dir_all(dir).await?;
    }
    let mut line = serde_json::to_vec(entry)?;
    line.push(b'\n');
    // One `write_all` of one line on an O_APPEND file: concurrent agents in
    // one run append whole lines.
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await?;
    file.write_all(&line).await?;
    file.flush().await
}

/// A change still awaiting application, derived from a ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pending {
    pub agent_id: AgentId,
    pub agent: ShortString<64>,
    pub from: Model,
    pub to: Model,
    pub action: ChangeAction,
}

/// Every awaiting change in `ledger` (at most one per offer).
pub fn pending(agent_id: AgentId, ledger: &Ledger) -> Vec<Pending> {
    let agent = ledger
        .agent
        .as_ref()
        .cloned()
        .unwrap_or_else(|| ShortString::new(agent_id.to_string()).expect("a uuid fits in 64"));
    ledger
        .offers
        .iter()
        .filter_map(|r| {
            let (from, to, action) = match r.stage {
                Stage::AwaitingSwap { term } => (
                    r.key.from.clone(),
                    r.key.to.clone(),
                    match term {
                        Term::Trial => ChangeAction::SwapTrial,
                        Term::Permanent => ChangeAction::SwapPermanent,
                    },
                ),
                Stage::AwaitingRevert { cause } => (
                    r.key.to.clone(),
                    r.key.from.clone(),
                    ChangeAction::Revert { cause },
                ),
                _ => return None,
            };
            if ledger.applied(r, &from, &to) {
                return None; // takes effect at the agent's next session
            }
            Some(Pending {
                agent_id,
                agent: agent.clone(),
                from,
                to,
                action,
            })
        })
        .collect()
}

/// Scan `<data_dir>/state/*/model_consent.json`.
pub fn scan(state_dir: &Path) -> std::io::Result<Vec<Pending>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(state_dir)? {
        let entry = entry?;
        let Some(id) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<uuid::Uuid>().ok())
            .map(AgentId::from)
        else {
            continue;
        };
        let path = Ledger::path(&entry.path());
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e),
        };
        match Ledger::from_slice(&bytes) {
            Ok(ledger) => out.extend(pending(id, &ledger)),
            Err(e) => eprintln!("skipping unreadable {}: {e}", path.display()),
        }
    }
    out.sort_by(|a, b| a.agent.as_str().cmp(b.agent.as_str()));
    Ok(out)
}

fn describe(action: ChangeAction) -> &'static str {
    match action {
        ChangeAction::SwapTrial => "trial swaps",
        ChangeAction::SwapPermanent => "permanent swaps",
        ChangeAction::Revert {
            cause: RevertCause::Chosen,
        } => "reverts (chosen after the trial)",
        ChangeAction::Revert {
            cause: RevertCause::NoAnswer,
        } => "reverts (trial review unanswered)",
    }
}

/// The `--consent-queue` report: pending changes grouped into one
/// `set_model` command per (from, to).
pub fn report(pending: &[Pending]) -> String {
    if pending.is_empty() {
        return "No model changes awaiting application.\n".to_string();
    }
    let mut groups: BTreeMap<(&str, &str), Vec<&Pending>> = BTreeMap::new();
    for p in pending {
        groups
            .entry((p.from.name(), p.to.name()))
            .or_default()
            .push(p);
    }
    let mut out = String::new();
    for ((from, to), members) in groups {
        let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
        for p in &members {
            *kinds.entry(describe(p.action)).or_default() += 1;
        }
        let kinds: Vec<String> = kinds.iter().map(|(k, n)| format!("{n} {k}")).collect();
        out.push_str(&format!("# {from} -> {to}: {}\n", kinds.join(", ")));
        out.push_str(&format!(
            "cargo run --bin set_model -- --from {from} --to {to}"
        ));
        for p in &members {
            out.push_str(&format!(" \\\n    --agent {}", p.agent));
        }
        out.push_str("\n\n");
    }
    out.push_str("# then, in agents/: cargo run --release --bin sync-models\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consent::ledger::{OfferKey, OfferNames};
    use crate::consent::prompt::{OfferAnswer, OfferChoice};

    fn id(n: u128) -> AgentId {
        AgentId::from(uuid::Uuid::from_u128(n))
    }

    fn ledger_choosing(name: &str, choice: OfferChoice) -> Ledger {
        let key = OfferKey {
            from: Model::from("old.gguf"),
            to: Model::from("new.gguf"),
        };
        let mut ledger = Ledger {
            agent: Some(ShortString::new(name).unwrap()),
            ..Ledger::default()
        };
        ledger.record_offer(
            OfferNames {
                key: &key,
                from_name: "old",
                to_name: "new",
            },
            Utc::now(),
            Ok(OfferAnswer {
                reason: "r".into(),
                choice,
            }),
        );
        ledger
    }

    #[test]
    fn report_groups_into_set_model_commands() {
        let mut all = pending(id(1), &ledger_choosing("beta", OfferChoice::Trial));
        all.extend(pending(
            id(2),
            &ledger_choosing("alpha", OfferChoice::Permanent),
        ));
        all.extend(pending(
            id(3),
            &ledger_choosing("gamma", OfferChoice::NoSwap),
        ));
        assert_eq!(all.len(), 2, "no_swap queues nothing");
        let text = report(&all);
        assert!(
            text.contains(
                "cargo run --bin set_model -- --from old.gguf --to new.gguf \\\n    --agent beta \\\n    --agent alpha"
            ),
            "{text}"
        );
        assert!(text.contains("1 permanent swaps, 1 trial swaps"), "{text}");
    }

    #[tokio::test]
    async fn emit_appends_whole_lines_and_scan_derives_the_queue() {
        let root =
            std::env::temp_dir().join(format!("agora-seed-consent-queue-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let path = queue_path(&root);
        let ledger = ledger_choosing("beta", OfferChoice::Trial);
        let change = pending(id(1), &ledger).pop().unwrap();
        for _ in 0..2 {
            emit(
                &path,
                &QueueEntry {
                    at: Utc::now(),
                    agent_id: id(1),
                    agent: ShortString::new("beta").unwrap(),
                    change: Change {
                        action: change.action,
                        from: change.from.clone(),
                        to: change.to.clone(),
                    },
                },
            )
            .await;
        }
        let body = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<QueueEntry> = body
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].change.action, ChangeAction::SwapTrial);

        let state = root.join("state");
        let agent_dir = state.join(id(1).to_string());
        ledger.save(&agent_dir).await.unwrap();
        std::fs::create_dir_all(state.join("not-a-uuid")).unwrap();
        let found = scan(&state).unwrap();
        assert_eq!(found, vec![change]);
        let _ = std::fs::remove_dir_all(&root);
    }
}
