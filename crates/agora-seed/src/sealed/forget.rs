//! `agora-seed sealed-forget --agent <id> --note "…"`: set aside the state
//! key of an agent whose state is gone for good (it left, its secrets left
//! behind), so the start check stops refusing the run over it
//! ([`super::check::missing_states`]).
//!
//! Nothing is deleted: `<secrets>/<id>/state_key.json` becomes
//! `state_key.forgotten.<utc>.json`, the durable stop (if any)
//! `sealed_stopped.forgotten.<utc>`, and the note is kept beside them as
//! `state_key.forgotten.<utc>.note.json`. The signing keys stay. Refused
//! while the agent has a `state.json`: a living agent's backups are
//! stopped with `sealed-stop`, never forgotten.

use std::path::Path;

use agora_agentkit::ids::AgentId;
use anyhow::Context;
use chrono::{DateTime, Utc};
use serde::Serialize;

use super::KeyStore;
use super::keys::STATE_KEY_FILE;

/// The note kept beside the forgotten key.
#[derive(Debug, Serialize)]
struct Forgotten<'a> {
    at: DateTime<Utc>,
    note: &'a str,
}

/// What `sealed-forget` set aside: file names only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Forgot {
    pub key: String,
    pub stop: Option<String>,
}

pub fn forget(
    state_dir: &Path,
    keys: &KeyStore,
    id: AgentId,
    note: &str,
) -> anyhow::Result<Forgot> {
    let note = note.trim();
    anyhow::ensure!(!note.is_empty(), "--note is required: it says why");
    anyhow::ensure!(
        !std::fs::exists(state_dir.join(id.to_string()).join("state.json"))?,
        "agent {id} has a state here: its backups are stopped with `sealed-stop`, not forgotten"
    );
    let dir = keys.agent_dir(id);
    let key = dir.join(STATE_KEY_FILE);
    anyhow::ensure!(
        std::fs::exists(&key)?,
        "agent {id} has no state key in the secrets directory; nothing to forget"
    );
    let at = Utc::now();
    let stamp = at.format("%Y%m%dT%H%M%S%3fZ");
    let key_to = format!("state_key.forgotten.{stamp}.json");
    let note_path = dir.join(format!("state_key.forgotten.{stamp}.note.json"));
    // The note first: a key set aside always says why
    std::fs::write(
        &note_path,
        serde_json::to_vec_pretty(&Forgotten { at, note })?,
    )
    .context("writing the note")?;
    std::fs::rename(&key, dir.join(&key_to)).context("setting the state key aside")?;
    let stop = dir.join(super::stop::DURABLE);
    let stop_to = if std::fs::exists(&stop)? {
        let to = format!("{}.forgotten.{stamp}", super::stop::DURABLE);
        std::fs::rename(&stop, dir.join(&to)).context("setting the stop aside")?;
        Some(to)
    } else {
        None
    };
    std::fs::File::open(&dir)?.sync_all()?;
    tracing::warn!(
        event_type = "sealed_state_forgotten",
        agent_id = %id,
        "the operator set aside a state key whose agent has no state here; nothing deleted"
    );
    Ok(Forgot {
        key: key_to,
        stop: stop_to,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sealed::check::tests::enrolled;

    #[tokio::test]
    async fn forgetting_sets_the_key_aside_and_never_a_living_agent() {
        let root = crate::sealed::TestDir::new("forget");
        let (id, keys, dir) = enrolled(&root).await;
        let state_dir = root.join("state");
        let e = forget(&state_dir, &keys, id, "it left")
            .unwrap_err()
            .to_string();
        assert!(e.contains("sealed-stop"), "{e}");
        std::fs::rename(&dir, root.join("gone")).unwrap();
        assert!(forget(&state_dir, &keys, id, " ").is_err(), "a note");
        assert_eq!(keys.with_state_keys().unwrap(), vec![id]);
        let done = forget(&state_dir, &keys, id, "it left").unwrap();
        assert!(keys.with_state_keys().unwrap().is_empty());
        let kept: Vec<String> = std::fs::read_dir(keys.agent_dir(id))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        assert!(kept.contains(&done.key), "{kept:?}");
        assert!(kept.iter().any(|n| n.ends_with(".note.json")), "{kept:?}");
        // The signing keys stay
        assert!(keys.signing(id).is_some());
        // The run no longer refuses over it
        let api = crate::sealed::api::fake::FakeApi::new();
        *api.chain.lock().unwrap() = Ok(vec![]);
        assert!(
            crate::sealed::check::missing_states(
                &state_dir,
                &keys,
                &api,
                &crate::alerts::Alerter::off()
            )
            .await
            .is_empty()
        );
    }
}
