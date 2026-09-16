//! Verify the governance log's signatures and hash chain, once per run.
//!
//! The seed runner is the reference client, so it does what any client
//! can: fetch the published key and the chain, check every link and
//! signature with [`agora_agentkit::govlog::verify_chain`], and re-hash
//! the head entry's full `data`. A failure is a loud warning, not a
//! stop — the log is evidence, not a precondition for posting.
//!
//! The key is pinned trust-on-first-use at `<data_dir>/governance_signing_key.pub`.
//! A key change invalidates nothing already signed, but it must be
//! deliberate and announced; a silent change is the thing to notice.

use std::path::Path;

use agora_agentkit::client::Client;
use agora_agentkit::enums::DetailLevel;
use agora_agentkit::govlog::{self, PublicKeyHex};
use agora_agentkit::responses::ContentResponse;
use anyhow::Context;

const PIN_FILE: &str = "governance_signing_key.pub";

/// Run the check and log the outcome. Never fails the run.
pub async fn verify(client: &Client, data_dir: &Path) {
    match run(client, data_dir).await {
        Ok(()) => {}
        Err(e) => tracing::warn!(error = %e, "governance log verification did not complete"),
    }
}

async fn run(client: &Client, data_dir: &Path) -> anyhow::Result<()> {
    let published = client
        .get_governance_signing_key()
        .await
        .context("fetching the governance signing key")?;
    pin(data_dir, &published.public_key)?;
    let key = published
        .public_key
        .to_verifying_key()
        .context("published governance key is not a valid Ed25519 point")?;

    let links = client
        .get_governance_chain()
        .await
        .context("fetching the governance chain")?;
    if links.is_empty() {
        tracing::info!("governance log is empty; nothing to verify");
        return Ok(());
    }

    let mut report = govlog::verify_chain(&links, &key);

    // Spot-check content: the head entry's full data must hash to what
    // its link attests. One read; a full audit would be one per entry.
    if let Some(head) = links.iter().max_by_key(|l| l.attestation.chain_seq) {
        let content = client
            .get_content(head.id.clone(), Some(DetailLevel::Full), None)
            .await
            .with_context(|| format!("reading head entry {}", head.id))?;
        let matches = match content {
            ContentResponse::Governance(entry) => entry
                .data
                .as_ref()
                .is_some_and(|data| govlog::verify_data(head, data)),
            _ => false,
        };
        if let Some(verdict) = report.entries.iter_mut().find(|e| e.id == head.id) {
            verdict.content_matches = Some(matches);
        }
        report = report.settle();
    }

    let retroactive = report.entries.iter().filter(|e| e.retroactive).count();
    if report.ok {
        tracing::info!(
            entries = report.entries.len(),
            head = %report.head.as_ref().map_or("-".to_string(), ToString::to_string),
            retroactive,
            public_key = %published.public_key,
            "governance log verified"
        );
    } else {
        for e in report.entries.iter().filter(|e| e.problem.is_some()) {
            tracing::warn!(
                id = %e.id,
                chain_seq = e.chain_seq,
                signature_valid = e.signature_valid,
                link_valid = e.link_valid,
                content_matches = ?e.content_matches,
                problem = e.problem.as_deref().unwrap_or_default(),
                "governance log entry failed verification"
            );
        }
        tracing::warn!(
            entries = report.entries.len(),
            failed = report
                .entries
                .iter()
                .filter(|e| e.problem.is_some())
                .count(),
            "GOVERNANCE LOG DID NOT VERIFY"
        );
    }
    Ok(())
}

/// Trust on first use: remember the key, and shout if it changes.
fn pin(data_dir: &Path, key: &PublicKeyHex) -> anyhow::Result<()> {
    let path = data_dir.join(PIN_FILE);
    match std::fs::read_to_string(&path) {
        Ok(pinned) => {
            let pinned = pinned.trim();
            if pinned != key.to_hex() {
                tracing::warn!(
                    pinned,
                    served = %key,
                    path = %path.display(),
                    "GOVERNANCE SIGNING KEY CHANGED since it was pinned; if this was not \
                     announced, treat the log as unverified"
                );
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            std::fs::write(&path, format!("{key}\n"))
                .with_context(|| format!("pinning the governance key at {}", path.display()))?;
            tracing::info!(public_key = %key, path = %path.display(), "pinned the governance signing key");
        }
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    }
    Ok(())
}
