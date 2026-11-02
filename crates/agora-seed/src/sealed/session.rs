//! What every operator command holds while it works on an agent: proof
//! that no session is in progress, and the agent's `sealed/lock` for the
//! whole command. Writes to `sealed/` then go through agentkit's
//! `AgentSeal::prepare` and `commit` under that one lock ([`append`],
//! [`genesis`]).

use std::path::Path;

use agora_agentkit::enums::StateEntryKind;
use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::FsStorage;
use agora_agentkit::state::GenesisOrigin;
use agora_agentkit::state::codec::{self, KeyframePolicy};
use agora_agentkit::state::storage::{
    AgentSeal, Append, SealError, SealKeys, SealLock, plaintext_hash,
};
use agora_agentkit::state::verify::HeadRef;
use chrono::{DateTime, Utc};

/// Refuse while `id` has a session journal: a session is in progress, or
/// one was interrupted (its journal stays until the agent's next session
/// completes). With `refused_ok`, a journal older than the agent's
/// refusal is let pass: a refused agent runs no session, so it can't clear
/// one.
pub fn no_session(state_dir: &Path, id: AgentId, refused_ok: bool) -> anyhow::Result<()> {
    let journal = FsStorage::new(state_dir).journal(id);
    let modified = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    let Some(at) = modified(journal.path()) else {
        return Ok(());
    };
    if refused_ok {
        let refused = state_dir
            .join(id.to_string())
            .join(agora_agentkit::state::storage::SEALED_DIR)
            .join("refused");
        if modified(&refused).is_some_and(|r| r >= at) {
            return Ok(());
        }
    }
    anyhow::bail!(
        "agent {id} has a session in progress, or an interrupted one whose journal ({}) stays \
         until its next session completes; run this when it has no session",
        journal.path().display()
    )
}

/// [`no_session`], then the agent's `sealed/lock`, held until dropped.
pub async fn take(state_dir: &Path, id: AgentId, refused_ok: bool) -> anyhow::Result<SealLock> {
    no_session(state_dir, id, refused_ok)?;
    let lock = AgentSeal::new(&state_dir.join(id.to_string()), id)
        .lock()
        .await?;
    // A session that began while we waited
    no_session(state_dir, id, refused_ok)?;
    Ok(lock)
}

/// `AgentSeal::append_state`, under a lock the caller holds.
pub async fn append(
    seal: &AgentSeal,
    lock: &SealLock,
    keys: &SealKeys,
    policy: &KeyframePolicy,
    kind: StateEntryKind,
    state: &serde_json::Value,
    state_at: DateTime<Utc>,
) -> Result<HeadRef, SealError> {
    let head = seal.head().await?.ok_or(SealError::NoHead)?;
    let plaintext = codec::plaintext(state);
    let prepared = seal
        .prepare(
            lock,
            keys,
            policy,
            Append {
                kind,
                plaintext: &plaintext,
                plaintext_sha256: (kind == StateEntryKind::Snapshot).then(|| plaintext_hash(state)),
                state_at,
                parent: Some(head.head),
            },
        )
        .await?;
    seal.commit(lock, &prepared).await
}

/// `AgentSeal::genesis`, under a lock the caller holds.
pub async fn genesis(
    seal: &AgentSeal,
    lock: &SealLock,
    keys: &SealKeys,
    origin: &GenesisOrigin,
    state_at: DateTime<Utc>,
) -> Result<HeadRef, SealError> {
    let plaintext = agora_agentkit::govlog::canonical_json(&serde_json::to_value(origin)?);
    let prepared = seal
        .prepare(
            lock,
            keys,
            &KeyframePolicy::default(),
            Append {
                kind: StateEntryKind::Genesis,
                plaintext: &plaintext,
                plaintext_sha256: None,
                state_at,
                parent: None,
            },
        )
        .await?;
    seal.commit(lock, &prepared).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_journal_blocks_the_commands_unless_older_than_a_refusal() {
        let root = crate::sealed::TestDir::new("session-guard");
        let id = AgentId::new();
        let dir = root.join(id.to_string());
        std::fs::create_dir_all(&dir).unwrap();
        assert!(no_session(&root, id, false).is_ok());
        let journal = FsStorage::new(&*root).journal(id);
        std::fs::write(journal.path(), b"{}\n").unwrap();
        let e = no_session(&root, id, false).unwrap_err().to_string();
        assert!(e.contains("session in progress"), "{e}");
        assert!(no_session(&root, id, true).is_err(), "not refused");
        std::thread::sleep(std::time::Duration::from_millis(20));
        AgentSeal::new(&dir, id).refuse("mismatch").await.unwrap();
        assert!(
            no_session(&root, id, true).is_ok(),
            "older than the refusal"
        );
        assert!(no_session(&root, id, false).is_err());
        assert!(take(&root, id, false).await.is_err());
        let _lock = take(&root, id, true).await.unwrap();
    }
}
