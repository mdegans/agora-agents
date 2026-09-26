//! Verify the governance log's signatures and hash chain, once per run.
//!
//! The seed runner is the reference client, so it does what any client
//! can: resolve the genesis signing key, fetch the chain, check every
//! link and signature with [`agora_agentkit::govlog::verify_chain`], and
//! re-hash the head entry's full `data`. A failure **stops the run**: a
//! client does not act on a platform whose governance record it cannot
//! verify (Steward, 2026-09-26 — "Clients should not connect to a
//! compromised server. If this means they crash, they should."). That
//! includes a check that could not complete: unverifiable is treated as
//! unverified. A repudiated entry inside a declared compromise window is
//! a declared state, not a failure, and does not stop the run.
//!
//! Two independent pins live in `data_dir`, and mean different things:
//!
//! - `governance_signing_key.pub` — the **genesis key**: the key the
//!   chain started under, trust-on-first-use. It is not necessarily the
//!   key the platform serves today (the chain may have rotated since);
//!   it is what [`govlog::verify_chain`] needs as `genesis_key`, together
//!   with [`govlog::KeyAnchor::published`]. When `GET
//!   /api/governance/signing-keys` is available, the genesis key is read
//!   off the chain's own key history (the record with `introduced_by ==
//!   None`) and pinned once; when that endpoint isn't there yet, this
//!   falls back to the old behaviour of pinning whatever key is
//!   currently served.
//! - `governance_head.pin` — the last verified chain head
//!   (`chain_seq`/`id`/`entry_hash`). Checked on every run: if the chain
//!   no longer contains that exact link, the log's history changed after
//!   this client last looked, which a full-history rewrite by a key
//!   thief would otherwise hide. The pin is left untouched when that
//!   happens, so it stays as evidence of what the chain said before.

use std::path::Path;

use agora_agentkit::client::Client;
use agora_agentkit::enums::DetailLevel;
use agora_agentkit::govlog::{
    self, GovernanceChainLink, GovernanceVerification, KeyAnchor, PublicKeyHex, RootSet, Sha256Hex,
};
use agora_agentkit::ids::GovernanceLogId;
use agora_agentkit::responses::ContentResponse;
use anyhow::Context;
use serde::{Deserialize, Serialize};

const PIN_FILE: &str = "governance_signing_key.pub";
const HEAD_PIN_FILE: &str = "governance_head.pin";

/// Why the runner refused to go on. Each alarm has already been logged
/// at ERROR with its details; this is the summary the run exits with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Alarm {
    /// One or more entries failed signature, link, or content checks.
    DidNotVerify { failed: usize },
    /// The chain's active key is not the key the platform serves.
    KeyMismatch,
    /// Nothing out of band vouches for the key the chain started under.
    Unanchored,
    /// The served key differs from the one pinned on first use (servers
    /// without the signing-key history endpoint only).
    PinnedKeyChanged,
    /// The chain no longer holds the head this client last verified.
    HistoryRewritten,
}

impl std::fmt::Display for Alarm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Alarm::DidNotVerify { failed } => write!(f, "{failed} entries failed verification"),
            Alarm::KeyMismatch => f.write_str("the chain's active key is not the served key"),
            Alarm::Unanchored => f.write_str("the genesis key is unanchored"),
            Alarm::PinnedKeyChanged => f.write_str("the served key changed since it was pinned"),
            Alarm::HistoryRewritten => f.write_str("history changed since the pinned head"),
        }
    }
}

/// The governance log did not verify; the runner must not proceed.
#[derive(Debug)]
pub struct Refused(pub Vec<Alarm>);

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("refusing to run: the governance log did not verify (")?;
        for (i, a) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str("; ")?;
            }
            write!(f, "{a}")?;
        }
        f.write_str(")")
    }
}

impl std::error::Error for Refused {}

/// Verify the log. `Err` means the run must stop: either an [`Alarm`]
/// fired (wrapped in [`Refused`]) or the check could not complete.
///
/// Both failures log one ERROR `governance_log_refused` event, which the
/// agora repo's `scripts/stall-watch.py` mails to the Steward at once.
pub async fn verify(client: &Client, data_dir: &Path) -> anyhow::Result<()> {
    let alarms = match run(client, data_dir).await {
        Ok(alarms) => alarms,
        Err(e) => {
            tracing::error!(
                event_type = "governance_log_refused",
                error = %format!("{e:#}"),
                "REFUSING TO RUN: governance log verification did not complete"
            );
            return Err(e.context("refusing to run: governance log verification did not complete"));
        }
    };
    if alarms.is_empty() {
        Ok(())
    } else {
        tracing::error!(
            event_type = "governance_log_refused",
            alarms = ?alarms.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "REFUSING TO RUN: the governance log did not verify"
        );
        Err(Refused(alarms).into())
    }
}

async fn run(client: &Client, data_dir: &Path) -> anyhow::Result<Vec<Alarm>> {
    let mut alarms = Vec::new();
    let served = client
        .get_governance_signing_key()
        .await
        .context("fetching the governance signing key")?;
    let served_key = served.public_key;

    let genesis_key_hex = resolve_genesis_key(client, data_dir, served_key, &mut alarms).await?;
    let genesis_key = genesis_key_hex
        .to_verifying_key()
        .context("pinned governance genesis key is not a valid Ed25519 point")?;
    let anchor = KeyAnchor::published().with(genesis_key_hex);

    let links = client
        .get_governance_chain()
        .await
        .context("fetching the governance chain")?;
    if links.is_empty() {
        tracing::info!("governance log is empty; nothing to verify");
        return Ok(alarms);
    }

    // Every change of key must be certified by an offline root key
    // (agentkit 0.28): the platform's own signing key cannot move the chain.
    let mut report = govlog::verify_chain(&links, &genesis_key, &anchor, &RootSet::published());

    // Spot-check content: the head entry's full data must hash to what
    // its link attests, or to what a redaction of it left behind.
    // `check_content` verifies both, so a redacted head verifies.
    if let Some(head) = links.iter().max_by_key(|l| l.attestation.chain_seq) {
        let content = client
            .get_content(head.id.clone(), Some(DetailLevel::Full), None)
            .await
            .with_context(|| format!("reading head entry {}", head.id))?;
        if let ContentResponse::Governance(entry) = content
            && let Some(data) = entry.data.as_ref()
        {
            report.check_content(head, data);
        }
        report = report.settle();
    }

    // A rotated chain not matching what's served is expected under a
    // routine rotation and is not itself an alarm — it's only a problem
    // if the chain's *active* key (after following every rotation) isn't
    // what the platform serves right now.
    if !active_key_matches_served(&report, served_key) {
        alarms.push(Alarm::KeyMismatch);
        tracing::error!(
            active_key = %report.public_key,
            served_key = %served_key,
            "GOVERNANCE SIGNING KEY MISMATCH: the chain's active key after \
             following its rotations does not match the key the platform \
             currently serves"
        );
    }
    if !report.unanchored_keys.is_empty() {
        alarms.push(Alarm::Unanchored);
        tracing::error!(
            unanchored_keys = ?report.unanchored_keys.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "GOVERNANCE LOG GENESIS KEY IS UNANCHORED: neither this \
             client's agora-agentkit nor a root certificate in the chain \
             vouches for the key the log started under — re-check it out \
             of band. (Later keys are never reported here: a rotation the \
             root keys did not certify is refused outright.)"
        );
    }
    if !report.repudiated.is_empty() {
        // A declared state, not a defect — but worth saying every run so
        // a human notices a Steward-declared compromise window exists.
        tracing::info!(
            repudiated = ?report.repudiated.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "governance log has repudiated entries inside a declared \
             compromise window"
        );
    }

    let retroactive = report.entries.iter().filter(|e| e.retroactive).count();
    if report.ok {
        tracing::info!(
            entries = report.entries.len(),
            head = %report.head.as_ref().map_or("-".to_string(), ToString::to_string),
            retroactive,
            active_key = %report.public_key,
            "governance log verified"
        );
    } else {
        for e in report.entries.iter().filter(|e| e.problem.is_some()) {
            tracing::error!(
                id = %e.id,
                chain_seq = e.chain_seq,
                signature_valid = e.signature_valid,
                link_valid = e.link_valid,
                content_matches = ?e.content_matches,
                problem = e.problem.as_deref().unwrap_or_default(),
                "governance log entry failed verification"
            );
        }
        let failed = report
            .entries
            .iter()
            .filter(|e| e.problem.is_some())
            .count();
        alarms.push(Alarm::DidNotVerify { failed });
        tracing::error!(
            entries = report.entries.len(),
            failed,
            "GOVERNANCE LOG DID NOT VERIFY"
        );
    }

    let pin = check_and_update_head_pin(data_dir, &links, report.ok)
        .await
        .context("checking the pinned governance head")?;
    if matches!(pin, HeadPinOutcome::Rewritten { .. }) {
        alarms.push(Alarm::HistoryRewritten);
    }

    Ok(alarms)
}

/// The genesis key `verify_chain` should be told, resolved and pinned.
///
/// `GET /api/governance/signing-keys` may not exist on the server yet;
/// any error from it (404 included) means "not available", not a
/// failure, and falls back to pinning whatever key is currently served —
/// today's behaviour, and correct as long as the chain has never
/// rotated.
async fn resolve_genesis_key(
    client: &Client,
    data_dir: &Path,
    served: PublicKeyHex,
    alarms: &mut Vec<Alarm>,
) -> anyhow::Result<PublicKeyHex> {
    let path = data_dir.join(PIN_FILE);

    match client.get_governance_signing_keys().await {
        Ok(history) => {
            match read_pinned_key(&path).await? {
                // The file's meaning is fixed once written: the genesis
                // key it first saw. Never overwrite it here.
                Some(pinned) => Ok(pinned),
                None => {
                    let genesis = history
                        .keys
                        .iter()
                        .find(|k| k.introduced_by.is_none())
                        .map(|k| k.public_key);
                    match genesis {
                        Some(genesis) => {
                            write_pinned_key(&path, genesis).await?;
                            tracing::info!(
                                public_key = %genesis,
                                path = %path.display(),
                                "pinned the governance genesis signing key"
                            );
                            Ok(genesis)
                        }
                        // Defensive: the endpoint answered but named no
                        // genesis record. Fall back rather than fail the
                        // run over a server-side inconsistency.
                        None => pin_served_key(&path, served, alarms).await,
                    }
                }
            }
        }
        Err(_) => pin_served_key(&path, served, alarms).await,
    }
}

/// The pre-genesis-history behaviour: pin the currently served key
/// trust-on-first-use, and raise [`Alarm::PinnedKeyChanged`] if a later
/// run sees a different one than what's pinned.
async fn pin_served_key(
    path: &Path,
    served: PublicKeyHex,
    alarms: &mut Vec<Alarm>,
) -> anyhow::Result<PublicKeyHex> {
    match read_pinned_key(path).await? {
        Some(pinned) => {
            if pinned != served {
                alarms.push(Alarm::PinnedKeyChanged);
                tracing::error!(
                    pinned = %pinned,
                    served = %served,
                    path = %path.display(),
                    "GOVERNANCE SIGNING KEY CHANGED since it was pinned; if this was not \
                     announced, treat the log as unverified"
                );
            }
            Ok(pinned)
        }
        None => {
            write_pinned_key(path, served).await?;
            tracing::info!(public_key = %served, path = %path.display(), "pinned the governance signing key");
            Ok(served)
        }
    }
}

async fn read_pinned_key(path: &Path) -> anyhow::Result<Option<PublicKeyHex>> {
    match tokio::fs::read_to_string(path).await {
        Ok(s) => {
            let key = s
                .trim()
                .parse()
                .with_context(|| format!("parsing pinned governance key at {}", path.display()))?;
            Ok(Some(key))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

async fn write_pinned_key(path: &Path, key: PublicKeyHex) -> anyhow::Result<()> {
    tokio::fs::write(path, format!("{key}\n"))
        .await
        .with_context(|| format!("pinning the governance key at {}", path.display()))
}

/// `true` when the chain's active key (the key in force after following
/// every rotation `verify_chain` accepted) is the key the platform
/// currently serves. `false` is the alarm case: either the platform
/// serves a stale key, or the chain was quietly moved somewhere the
/// platform doesn't currently vouch for.
fn active_key_matches_served(report: &GovernanceVerification, served: PublicKeyHex) -> bool {
    report.public_key == served
}

/// What a client last verified as the chain's head.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct HeadPin {
    chain_seq: u64,
    id: GovernanceLogId,
    entry_hash: Sha256Hex,
}

impl HeadPin {
    fn of(link: &GovernanceChainLink) -> Self {
        Self {
            chain_seq: link.attestation.chain_seq,
            id: link.id.clone(),
            entry_hash: link.attestation.entry_hash,
        }
    }
}

/// What comparing a pinned head against the current chain found.
#[derive(Debug, Clone, PartialEq, Eq)]
enum HeadPinOutcome {
    /// Nothing pinned yet.
    NoPreviousPin,
    /// The chain still holds the pinned link, unchanged (whether or not
    /// it has since grown past it).
    Consistent,
    /// The chain no longer agrees with the pin: a different entry now
    /// sits at the pinned position (`observed: Some`), or the chain is
    /// shorter than the pinned position altogether (`observed: None`).
    /// Both are the same alarm — history changed since this client last
    /// verified it.
    Rewritten { observed: Option<HeadPin> },
}

/// Pure comparison: does `links` still agree with a previously pinned
/// head? No I/O, so this is the part unit tests exercise directly.
fn compare_head_pin(pinned: Option<&HeadPin>, links: &[GovernanceChainLink]) -> HeadPinOutcome {
    let Some(pinned) = pinned else {
        return HeadPinOutcome::NoPreviousPin;
    };
    match links
        .iter()
        .find(|l| l.attestation.chain_seq == pinned.chain_seq)
    {
        Some(l) if l.id == pinned.id && l.attestation.entry_hash == pinned.entry_hash => {
            HeadPinOutcome::Consistent
        }
        Some(l) => HeadPinOutcome::Rewritten {
            observed: Some(HeadPin::of(l)),
        },
        None => HeadPinOutcome::Rewritten { observed: None },
    }
}

/// Compare the current chain against the pinned head (if any), log an
/// error-level alarm on a rewrite without touching the pin (so it stays
/// as evidence), and otherwise — when the chain verified `ok` — refresh
/// the pin to the current head. I/O errors are returned to the caller,
/// which refuses to run on them.
async fn check_and_update_head_pin(
    data_dir: &Path,
    links: &[GovernanceChainLink],
    ok: bool,
) -> anyhow::Result<HeadPinOutcome> {
    let path = data_dir.join(HEAD_PIN_FILE);
    let pinned = read_head_pin(&path).await?;
    let outcome = compare_head_pin(pinned.as_ref(), links);

    match &outcome {
        HeadPinOutcome::Rewritten { observed } => {
            let pinned = pinned.expect("Rewritten only follows a previous pin");
            tracing::error!(
                pinned_chain_seq = pinned.chain_seq,
                pinned_id = %pinned.id,
                pinned_entry_hash = %pinned.entry_hash,
                observed_chain_seq = observed.as_ref().map(|o| o.chain_seq),
                observed_id = observed.as_ref().map(|o| o.id.to_string()),
                observed_entry_hash = observed.as_ref().map(|o| o.entry_hash.to_string()),
                "GOVERNANCE LOG HISTORY CHANGED since this client last verified it"
            );
        }
        HeadPinOutcome::NoPreviousPin | HeadPinOutcome::Consistent => {
            if ok && let Some(head) = links.iter().max_by_key(|l| l.attestation.chain_seq) {
                write_head_pin(&path, &HeadPin::of(head)).await?;
            }
        }
    }

    Ok(outcome)
}

async fn read_head_pin(path: &Path) -> anyhow::Result<Option<HeadPin>> {
    match tokio::fs::read_to_string(path).await {
        Ok(s) => serde_json::from_str(&s)
            .with_context(|| format!("parsing pinned governance head at {}", path.display()))
            .map(Some),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

async fn write_head_pin(path: &Path, pin: &HeadPin) -> anyhow::Result<()> {
    let json = serde_json::to_string(pin).context("serializing governance head pin")?;
    tokio::fs::write(path, json)
        .await
        .with_context(|| format!("pinning the governance head at {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agora_agentkit::crypto::{self, generate_keypair};
    use agora_agentkit::enums::GovernanceLogEntryType;
    use agora_agentkit::govlog::{Envelope, attest, data_hash, truncate_to_micros};
    use chrono::{DateTime, Utc};
    use serde_json::json;

    #[test]
    fn refused_names_every_alarm() {
        let r = Refused(vec![
            Alarm::DidNotVerify { failed: 2 },
            Alarm::HistoryRewritten,
        ]);
        assert_eq!(
            r.to_string(),
            "refusing to run: the governance log did not verify \
             (2 entries failed verification; history changed since the pinned head)"
        );
    }

    fn at(secs: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000 + secs, 0).unwrap()
    }

    fn gov(n: u32) -> GovernanceLogId {
        format!("GOV-2026-{n:04}").parse().unwrap()
    }

    /// A minimal, validly-signed chain of `n` council decisions, for
    /// exercising the pure parts of this module (no server involved).
    fn chain(key: &crypto::SigningKey, n: u32) -> Vec<GovernanceChainLink> {
        chain_labeled(key, n, "Decision")
    }

    /// [`chain`], but with `label` in each entry's data instead of
    /// "Decision" — enough to make two chains diverge in content (and so
    /// `entry_hash`) at every position, independent of which key signed
    /// them.
    fn chain_labeled(key: &crypto::SigningKey, n: u32, label: &str) -> Vec<GovernanceChainLink> {
        let mut out: Vec<GovernanceChainLink> = Vec::new();
        for i in 1..=n {
            let data = json!({"title": format!("{label} {i}")});
            let created_at = truncate_to_micros(at(i as i64 * 10));
            let prev_hash = out
                .last()
                .map(|p: &GovernanceChainLink| p.attestation.entry_hash);
            let envelope = Envelope::new(
                gov(i),
                GovernanceLogEntryType::CouncilDecision,
                created_at,
                prev_hash,
                data_hash(&data),
            );
            let attestation = attest(key, &envelope, i as u64, at(i as i64 * 10 + 1));
            out.push(GovernanceChainLink {
                id: gov(i),
                entry_type: GovernanceLogEntryType::CouncilDecision,
                created_at,
                attestation,
                data: None,
                texts: None,
            });
        }
        out
    }

    // -- compare_head_pin --

    #[test]
    fn no_previous_pin_is_reported_as_such() {
        let (key, _) = generate_keypair();
        let links = chain(&key, 2);
        assert_eq!(
            compare_head_pin(None, &links),
            HeadPinOutcome::NoPreviousPin
        );
    }

    #[test]
    fn same_head_is_consistent() {
        let (key, _) = generate_keypair();
        let links = chain(&key, 2);
        let pin = HeadPin::of(&links[1]);
        assert_eq!(
            compare_head_pin(Some(&pin), &links),
            HeadPinOutcome::Consistent
        );
    }

    #[test]
    fn extended_chain_is_still_consistent_at_the_old_head() {
        let (key, _) = generate_keypair();
        let short = chain(&key, 2);
        let pin = HeadPin::of(&short[1]);
        let extended = chain(&key, 5);
        assert_eq!(
            compare_head_pin(Some(&pin), &extended),
            HeadPinOutcome::Consistent
        );
    }

    #[test]
    fn a_different_entry_at_the_pinned_position_is_rewritten() {
        let (key, _) = generate_keypair();
        let links = chain(&key, 2);
        let pin = HeadPin::of(&links[1]);

        // A different chain, same length, same seq -- different content
        // (`entry_hash` covers the envelope, not the signature, so a
        // different signing key alone would not move it).
        let rewritten = chain_labeled(&key, 2, "Reworded");

        match compare_head_pin(Some(&pin), &rewritten) {
            HeadPinOutcome::Rewritten { observed } => {
                let observed = observed.expect("a link exists at that seq");
                assert_eq!(observed.chain_seq, pin.chain_seq);
                assert_ne!(observed.entry_hash, pin.entry_hash);
            }
            other => panic!("expected Rewritten, got {other:?}"),
        }
    }

    #[test]
    fn a_shorter_chain_than_the_pin_is_rewritten_with_no_observed_link() {
        let (key, _) = generate_keypair();
        let long = chain(&key, 5);
        let pin = HeadPin::of(&long[4]);
        let truncated = chain(&key, 2);

        assert_eq!(
            compare_head_pin(Some(&pin), &truncated),
            HeadPinOutcome::Rewritten { observed: None }
        );
    }

    // -- active_key_matches_served --

    #[test]
    fn active_key_matches_served_when_equal() {
        let (_, pk) = generate_keypair();
        let key_hex: PublicKeyHex = (&pk).into();
        let report = GovernanceVerification {
            public_key: key_hex,
            ok: true,
            head: None,
            entries: Vec::new(),
            keys: Vec::new(),
            unanchored_keys: Vec::new(),
            repudiated: Vec::new(),
        };
        assert!(active_key_matches_served(&report, key_hex));
    }

    #[test]
    fn active_key_mismatch_is_not_silently_ok() {
        let (_, pk) = generate_keypair();
        let (_, other) = generate_keypair();
        let report = GovernanceVerification {
            public_key: (&pk).into(),
            ok: true,
            head: None,
            entries: Vec::new(),
            keys: Vec::new(),
            unanchored_keys: Vec::new(),
            repudiated: Vec::new(),
        };
        assert!(!active_key_matches_served(&report, (&other).into()));
    }
}
