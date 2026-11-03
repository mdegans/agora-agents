//! The session-start check, for enrolled agents only: run in `main` right
//! after the states load, before any agent is built, as a skip stage
//! (`sealed`). Not in `on_init`: agentkit's reactor would still save a
//! refused agent there.
//!
//! | Found | Done |
//! |---|---|
//! | the local state is the head's (or a pending item's, now committed), the main line verifies, and Agora has it or is behind | run; the verdict line is seated |
//! | the local state matches nothing, the chain fails, or `head.json` is gone | **refuse**: skipped, `sealed/refused`, a notice for the agent, [`SealedStateRefused`] |
//! | the local state is one whose seal failed | run; uploads held; [`SealedStateSealFailed`] |
//! | `sealed/` can't be read | run; saves unsealed, uploads held; [`SealedStateUnreadable`]; never a refusal on its own |
//! | the local head is not on Agora (`LocalMissing`) | run; held; [`SealedStateHeadMissing`] |
//! | Agora moved on (`ServerAhead`, `Diverged`) | run on the local copy; held; [`SealedStateFork`] |
//!
//! Every problem is told to the agent too, in plain words. The line names
//! the file and the commit that checked it, never a model. Nothing here
//! reads any content of the state: the check is over hashes.
//!
//! [`SealedStateRefused`]: crate::alerts::AlertKind::SealedStateRefused
//! [`SealedStateSealFailed`]: crate::alerts::AlertKind::SealedStateSealFailed
//! [`SealedStateHeadMissing`]: crate::alerts::AlertKind::SealedStateHeadMissing
//! [`SealedStateFork`]: crate::alerts::AlertKind::SealedStateFork
//! [`SealedStateUnreadable`]: crate::alerts::AlertKind::SealedStateUnreadable

use std::collections::HashMap;
use std::path::Path;

use agora_agentkit::enums::StateEntryKind;
use agora_agentkit::govlog::Sha256Hex;
use agora_agentkit::ids::AgentId;
use agora_agentkit::state::SignedStateEntry;
use agora_agentkit::state::storage::{AgentSeal, Recovery};
use agora_agentkit::state::verify::{HeadRef, HeadStatus, head_status, verify_chain};

use super::KeyStore;
use super::api::SealedApi;
use super::notice::{Notice, Refusal};
use super::sizes::{Recorded, SizeLog, Sizes, drop_lines};
use crate::alerts::{Alert, AlertKind, Alerter};

/// What checked it, as the verdict line names it. Pinned to the agentkit
/// in `Cargo.lock` by `the_verifier_is_the_locked_agentkit`.
pub const VERIFIER: &str = "git:agora-agentkit:v0.147.1:src/state/verify.rs";

/// Why uploads are held this run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hold {
    SealFailed,
    HeadMissing,
    Fork,
    /// `sealed/` couldn't be read: the agent runs, its saves go unsealed
    /// (no base) and are recorded as such.
    Unreadable,
    /// The chain couldn't be checked (Agora unreachable for a main line
    /// that runs through it): the agent runs, uploads held. No refusal
    /// without evidence.
    Unchecked,
}

/// The check's answer for one agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// Run. `seat`: lines for the end of the intro; `notice`: one of them
    /// is the refusal notice, renamed aside once the session saves.
    Run {
        seat: Vec<String>,
        notice: bool,
        hold: Option<Hold>,
        /// A memory or soul that shrank past the threshold: what, by how
        /// many percent ([`super::sizes`]).
        drops: Vec<(&'static str, u32)>,
    },
    /// Leave it out, refused. `detail` is for the operator, hashes and
    /// counts only.
    Refuse { reason: Refusal, detail: String },
}

/// The main line ending at `head`, oldest first, from every entry the
/// runner holds; `None` if a link is missing.
pub fn main_line(entries: &[SignedStateEntry], head: &Sha256Hex) -> Option<Vec<SignedStateEntry>> {
    let by_hash: HashMap<Sha256Hex, &SignedStateEntry> =
        entries.iter().map(|e| (e.entry_hash(), e)).collect();
    let mut line = Vec::new();
    let mut at = Some(*head);
    while let Some(hash) = at {
        let entry = by_hash.get(&hash)?;
        at = entry.entry.prev_hash;
        line.push((*entry).clone());
        if line.len() > entries.len() {
            return None;
        }
    }
    line.reverse();
    Some(line)
}

/// Every entry the agent's `sealed/` holds: uploaded, then waiting.
pub async fn local_entries(seal: &AgentSeal) -> Result<Vec<SignedStateEntry>, String> {
    let mut items = seal.sent().await.map_err(|e| e.to_string())?;
    items.extend(seal.outbox().await.map_err(|e| e.to_string())?);
    Ok(items.into_iter().map(|i| i.entry).collect())
}

/// `1234567` as `1,234,567`.
fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, d) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(d);
    }
    out
}

fn day(micros: i64) -> Option<String> {
    chrono::DateTime::from_timestamp_micros(micros).map(|d| d.format("%Y-%m-%d").to_string())
}

/// What Agora's copy said, for the verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Server {
    /// Agora's head is this one.
    InStep,
    /// Agora couldn't be asked.
    Unchecked,
    /// Anything else: uploads waiting, or a problem a line of its own
    /// tells.
    Other,
}

/// The verdict line for an intact chain: the verdict, the snapshots since
/// the earliest one's date, a pre-loss copy apart, the head marker, and
/// the state's sizes (counts only).
pub fn verdict_line(
    line: &[SignedStateEntry],
    head: &HeadRef,
    sizes: Sizes,
    previous: Option<Sizes>,
    server: Server,
) -> String {
    // "13,539 characters (−1,204 since your last checked version)"
    let change = |now: usize, before: Option<usize>, long: bool| -> String {
        let Some(before) = before else {
            return String::new();
        };
        let since = if long {
            " since your last checked version"
        } else {
            ""
        };
        match now.cmp(&before) {
            std::cmp::Ordering::Equal => " (unchanged)".into(),
            std::cmp::Ordering::Greater => format!(" (+{}{since})", thousands(now - before)),
            std::cmp::Ordering::Less => format!(" (\u{2212}{}{since})", thousands(before - now)),
        }
    };
    let memory_change = change(sizes.memory_chars, previous.map(|p| p.memory_chars), true);
    let soul_change = change(sizes.soul_chars, previous.map(|p| p.soul_chars), false);
    let verdict = match server {
        Server::InStep => "intact, and in step with Agora.",
        Server::Unchecked => "intact here; Agora's copy couldn't be checked this session.",
        Server::Other => "intact.",
    };
    let snapshots: Vec<&SignedStateEntry> = line
        .iter()
        .filter(|e| e.entry.kind == StateEntryKind::Snapshot)
        .collect();
    let since = snapshots
        .iter()
        .map(|e| e.entry.state_at)
        .min()
        .and_then(day)
        .unwrap_or_else(|| "an unknown date".into());
    let prior = line
        .iter()
        .find(|e| e.entry.kind == StateEntryKind::ArchivedPrior)
        .and_then(|e| day(e.entry.state_at))
        .map(|d| format!(", plus a copy of your state from before a loss, dated {d}"))
        .unwrap_or_default();
    let mut short = head.entry_hash.to_hex();
    short.truncate(12);
    format!(
        "Your sealed state: {verdict} {} version{} since {since}{prior}; this session starts from \
         head {short} (a marker for this version; the next session's will differ). Your memory \
         is {} characters{memory_change} and your soul {}{soul_change}, as at your last sealed \
         version. Checked by \
         {VERIFIER}: links, signatures, and that your state matches the last sealed version.",
        snapshots.len(),
        if snapshots.len() == 1 { "" } else { "s" },
        thousands(sizes.memory_chars),
        thousands(sizes.soul_chars),
    )
}

pub const FORK_LINE: &str = "Your sealed state on Agora has moved on from the version this \
     session starts from: two sessions saved from the same version, or Agora holds versions \
     this runner doesn't. This session runs on your state here, and nothing is lost: new \
     versions are kept here, not uploaded, until your operator reconciles the two. Your \
     operator has been told.";

pub const HEAD_MISSING_LINE: &str = "Your latest sealed version is not on Agora: the copy there \
     is missing versions that were sealed here. This session runs on your state here, which \
     matches your last sealed version; new versions are kept here, not uploaded, and your \
     operator has been told.";

pub const UNREADABLE_LINE: &str = "Your sealed history couldn't be read here at the start of \
     this session, so it couldn't be checked. This session runs on your state as it is; its \
     save is kept here but not sealed, and your operator has been told.";

pub const UNCHECKED_LINE: &str = "Your sealed history couldn't be checked at the start of this \
     session: part of it is on Agora, which couldn't be reached. This session runs on your \
     state as it is, and your operator has been told.";

pub const SEAL_FAILED_LINE: &str = "Your last save was written here but could not be sealed, so \
     your sealed history does not include it yet. This session starts from that save; new \
     versions are kept here, not uploaded, until your operator has looked, and your operator \
     has been told.";

/// The sizes of the newest earlier sealed version: from the sidecar, else
/// by opening it, if its blobs are still here.
async fn earlier_sizes(
    agent_dir: &Path,
    seal: &AgentSeal,
    keys: &KeyStore,
    id: AgentId,
    line: &[SignedStateEntry],
    upto: usize,
) -> Option<Recorded> {
    let snapshots: Vec<&SignedStateEntry> = line[..upto]
        .iter()
        .filter(|e| e.entry.kind == StateEntryKind::Snapshot)
        .collect();
    let hashes: Vec<Sha256Hex> = snapshots.iter().map(|e| e.entry_hash()).collect();
    if let Some(r) = SizeLog::load(agent_dir).latest_among(&hashes) {
        return Some(r.clone());
    }
    let last = snapshots.last()?;
    let k = keys.seal_keys(id)?;
    let keyframe = last.entry.blob.as_ref()?.encoding.base().and_then(|base| {
        line.iter()
            .map(|e| &e.entry)
            .find(|e| e.entry_hash() == base)
    });
    let state = seal.open_state(&k, &last.entry, keyframe).await.ok()?;
    Some(Recorded {
        entry_hash: last.entry_hash(),
        seq: last.entry.seq,
        state_at: chrono::DateTime::from_timestamp_micros(last.entry.state_at)?,
        sizes: Sizes::of(&state),
    })
}

/// One enrolled agent's check. `local`: its state's plaintext hash;
/// `sizes`: its memory's and soul's sizes.
pub async fn check(
    agent_dir: &Path,
    id: AgentId,
    local: &Sha256Hex,
    sizes: Sizes,
    keys: &KeyStore,
    server: &dyn SealedApi,
) -> Outcome {
    let seal = AgentSeal::new(agent_dir, id);
    let refuse = |reason: Refusal, detail: String| Outcome::Refuse { reason, detail };
    let unreadable = || Outcome::Run {
        seat: vec![UNREADABLE_LINE.to_string()],
        notice: false,
        hold: Some(Hold::Unreadable),
        drops: Vec::new(),
    };
    let unchecked = || Outcome::Run {
        seat: vec![UNCHECKED_LINE.to_string()],
        notice: false,
        hold: Some(Hold::Unchecked),
        drops: Vec::new(),
    };
    match seal.is_refused().await {
        Ok(false) => {}
        Ok(true) => {
            let why = std::fs::read_to_string(seal.dir().join("refused")).unwrap_or_default();
            let reason = Notice::load(agent_dir)
                .ok()
                .flatten()
                .map_or(Refusal::Mismatch, |n| n.reason);
            return refuse(
                reason,
                format!("still refused ({why}); clear with sealed-accept"),
            );
        }
        Err(e) => {
            tracing::warn!(agent_id = %id, error = %e, "sealed refusal marker unreadable");
            return unreadable();
        }
    }
    let (head, seal_failed) = match seal.recover(Some(local)).await {
        Ok(Recovery::Intact(h) | Recovery::Committed(h)) => (h, None),
        Ok(Recovery::SealFailed(h, kind)) => (h, Some(kind)),
        Ok(Recovery::NoHead) => {
            return refuse(
                Refusal::HeadMissing,
                "enrolled, but sealed/head.json is missing; `sealed-accept` rebuilds it from the chain on Agora if that verifies".into(),
            );
        }
        Ok(Recovery::Mismatch(h) | Recovery::NoLocalState(h)) => {
            return refuse(
                Refusal::Mismatch,
                format!(
                    "local state matches neither head {} nor a pending version",
                    h.entry_hash
                ),
            );
        }
        Ok(Recovery::NotEnrolled) => {
            return Outcome::Run {
                seat: Vec::new(),
                notice: false,
                hold: None,
                drops: Vec::new(),
            };
        }
        Ok(Recovery::Unreadable(kind)) => {
            tracing::warn!(agent_id = %id, kind = ?kind, "sealed state unreadable");
            return unreadable();
        }
        Err(e) => {
            tracing::warn!(agent_id = %id, error = %e, kind = ?e.kind(), "sealed state unreadable");
            return unreadable();
        }
    };
    let entries = match local_entries(&seal).await {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!(agent_id = %id, error = %e, "local sealed chain unreadable");
            return unreadable();
        }
    };
    // After a rebase the main line runs through Agora's entries, which
    // aren't kept here: they are read from Agora and verified below with
    // the runner's own trust, never taken on its word
    let mut fetched: Option<Result<Vec<SignedStateEntry>, String>> = None;
    let mut line = main_line(&entries, &head.entry_hash);
    if line.is_none()
        && let Some(signing) = keys.signing(id)
    {
        let chain = server.chain(id, &signing).await;
        if let Ok(chain) = &chain {
            let mut all = entries.clone();
            all.extend(chain.iter().cloned());
            line = main_line(&all, &head.entry_hash);
        }
        fetched = Some(chain);
    }
    let line = match (line, &fetched) {
        (Some(line), _) => line,
        // Agora couldn't be asked, or there is no key to ask with: no
        // evidence either way, so the agent runs, unchecked
        (None, Some(Err(e))) => {
            tracing::warn!(agent_id = %id, error = %e, "the main line runs through Agora's chain, which couldn't be read");
            return unchecked();
        }
        (None, None) => return unchecked(),
        (None, Some(Ok(_))) => {
            return refuse(
                Refusal::ChainBroken,
                format!(
                    "the main line to head {} has a missing link",
                    head.entry_hash
                ),
            );
        }
    };
    // Keys that can't be read are no evidence against the chain
    let trusted = match keys.trusted(id) {
        Ok(Some(t)) => t,
        Ok(None) => {
            tracing::warn!(agent_id = %id, "no signing key to verify the sealed chain with");
            return unreadable();
        }
        Err(e) => {
            tracing::warn!(agent_id = %id, error = %e, "the agent's key records are unreadable");
            return unreadable();
        }
    };
    let report = verify_chain(id, &line, &trusted);
    for notice in &report.notices {
        tracing::warn!(event_type = "sealed_state_key_notice", agent_id = %id, notice = ?notice, "sealed-state key notice");
    }
    if !report.intact || report.head != Some(head) {
        return refuse(
            Refusal::ChainBroken,
            format!("chain did not verify: {:?}", report.problems),
        );
    }

    let mut seat = Vec::new();
    let mut notice = false;
    if let Ok(Some(n)) = Notice::load(agent_dir)
        && let Some(text) = n.line()
    {
        seat.push(text);
        notice = true;
    }
    // Against the head when this state isn't it (a failed seal), else
    // against the version before it
    let upto = if seal_failed.is_some() {
        line.len()
    } else {
        line.len() - 1
    };
    let before = earlier_sizes(agent_dir, &seal, keys, id, &line, upto).await;
    let previous = before.as_ref().map(|b| b.sizes);
    let drops: Vec<(&'static str, String, u32)> = match &before {
        Some(before) => drop_lines(before, sizes)
            .into_iter()
            .map(|(what, pct, text)| (what, text, pct))
            .collect(),
        None => Vec::new(),
    };
    if seal_failed.is_none()
        && let Some(last) = line.last()
        && let Some(state_at) = chrono::DateTime::from_timestamp_micros(last.entry.state_at)
        && let Err(e) = SizeLog::record(
            agent_dir,
            Recorded {
                entry_hash: head.entry_hash,
                seq: head.seq,
                state_at,
                sizes,
            },
        )
    {
        tracing::warn!(agent_id = %id, error = %e, "sealed sizes not recorded");
    }
    let drop_lines: Vec<String> = drops.iter().map(|d| d.1.clone()).collect();
    let drops: Vec<(&'static str, u32)> = drops.iter().map(|d| (d.0, d.2)).collect();
    if seal_failed.is_some() {
        seat.push(SEAL_FAILED_LINE.into());
        seat.extend(drop_lines);
        return Outcome::Run {
            seat,
            notice,
            hold: Some(Hold::SealFailed),
            drops,
        };
    }
    let Some(signing) = keys.signing(id) else {
        seat.push(verdict_line(
            &line,
            &head,
            sizes,
            previous,
            Server::Unchecked,
        ));
        seat.extend(drop_lines);
        return Outcome::Run {
            seat,
            notice,
            hold: None,
            drops,
        };
    };
    let outbox: Vec<_> = match seal.outbox().await {
        Ok(items) => items.into_iter().map(|i| i.entry.entry).collect(),
        Err(_) => Vec::new(),
    };
    let chain = match fetched {
        Some(chain) => chain,
        None => server.chain(id, &signing).await,
    };
    let (status, server_said) = match chain {
        Ok(chain) => {
            let hashes: Vec<Sha256Hex> = chain.iter().map(|e| e.entry_hash()).collect();
            let status = head_status(Some(&head.entry_hash), &outbox, &hashes);
            let said = if status == HeadStatus::Same {
                Server::InStep
            } else {
                Server::Other
            };
            (status, said)
        }
        Err(e) => {
            tracing::warn!(
                event_type = "sealed_state_server_unchecked",
                agent_id = %id,
                error = %e,
                "the copy on Agora could not be checked; running on the local check"
            );
            (HeadStatus::Pending, Server::Unchecked)
        }
    };
    seat.push(verdict_line(&line, &head, sizes, previous, server_said));
    seat.extend(drop_lines);
    let hold = match status {
        HeadStatus::Same | HeadStatus::Pending => None,
        HeadStatus::LocalMissing => {
            seat.push(HEAD_MISSING_LINE.into());
            Some(Hold::HeadMissing)
        }
        HeadStatus::ServerAhead | HeadStatus::Diverged => {
            seat.push(FORK_LINE.into());
            Some(Hold::Fork)
        }
    };
    Outcome::Run {
        seat,
        notice,
        hold,
        drops,
    }
}

/// Rotations the server lists that the runner's own `rotations.json`
/// doesn't: advisory, never trusted, but the operator hears of them (a
/// server that reports a rotation the runner never made). `None` when the
/// server couldn't be asked.
pub async fn unknown_rotations(
    keys: &KeyStore,
    server: &dyn SealedApi,
    id: AgentId,
) -> Option<usize> {
    let signing = keys.signing(id)?;
    let listed = match server.rotations(id, &signing).await {
        Ok(listed) => listed,
        Err(e) => {
            tracing::warn!(agent_id = %id, error = %e, "the server's rotations could not be read");
            return None;
        }
    };
    let ours = keys.rotations(id).unwrap_or_default();
    Some(
        listed
            .iter()
            .filter(|r| {
                !ours
                    .iter()
                    .any(|o| o.seq == r.seq && o.new_public_key == r.new_public_key)
            })
            .count(),
    )
}

/// Act on a refusal: `sealed/refused` (blocks every seal), the notice for
/// the agent, the alert. Never touches the state.
pub async fn refuse(
    agent_dir: &Path,
    id: AgentId,
    name: &str,
    reason: Refusal,
    detail: &str,
    alerts: &Alerter,
) {
    let seal = AgentSeal::new(agent_dir, id);
    if !seal.is_refused().await.unwrap_or(false)
        && let Err(e) = seal.refuse(reason.as_str()).await
    {
        tracing::error!(agent_id = %id, error = %e, "could not write the refusal marker");
    }
    if let Err(e) = Notice::refused(agent_dir, reason) {
        tracing::error!(agent_id = %id, error = %e, "could not write the refusal notice");
    }
    tracing::error!(
        event_type = "sealed_state_refused",
        agent = name,
        agent_id = %id,
        reason = reason.as_str(),
        detail,
        "sealed state did not check out; the agent is left out until `agora-seed sealed-accept`"
    );
    alerts.notify(
        Alert::new(
            AlertKind::SealedStateRefused,
            format!("{name}'s session was refused: {}", reason.plain()),
        )
        .agent(name, id)
        .detail("reason", reason.as_str())
        .detail("detail", detail),
    );
}

/// The alert for a memory or soul that shrank.
pub fn drop_alert(id: AgentId, name: &str, drops: &[(&'static str, u32)]) -> Alert {
    let mut alert = Alert::new(
        AlertKind::SealedStateDrop,
        format!("{name}: memory or soul much smaller than at the previous sealed version"),
    )
    .agent(name, id);
    for (what, pct) in drops {
        alert = alert.detail("smaller", format!("{what} by {pct}%"));
    }
    alert
}

/// The alert for a run that goes on held.
pub fn hold_alert(
    hold: Hold,
    id: AgentId,
    name: &str,
    held: Option<&agora_agentkit::state::storage::Held>,
) -> Alert {
    let (kind, summary) = match hold {
        Hold::SealFailed => (
            AlertKind::SealedStateSealFailed,
            "the last save was written unsealed; uploads held",
        ),
        Hold::HeadMissing => (
            AlertKind::SealedStateHeadMissing,
            "the local sealed head is not on Agora; uploads held",
        ),
        Hold::Fork => (
            AlertKind::SealedStateFork,
            "the sealed chain on Agora moved on from the local head; uploads held",
        ),
        Hold::Unreadable => (
            AlertKind::SealedStateUnreadable,
            "sealed/ or its keys could not be read; the agent runs, uploads held",
        ),
        Hold::Unchecked => (
            AlertKind::SealedStateUnreadable,
            "the sealed chain could not be checked (Agora unreachable); the agent runs, uploads held",
        ),
    };
    let mut alert = Alert::new(kind, format!("{name}: {summary}")).agent(name, id);
    if let Some(held) = held {
        alert = alert
            .detail("held", held.outbox_items)
            .detail("blobs", held.blobs)
            .detail("blob_bytes", held.blob_bytes)
            .detail("forks", held.displaced);
    }
    if hold == Hold::Fork {
        alert = alert.detail(
            "resolve",
            format!("agora-seed sealed-accept --agent {id} --rebase --note \"…\""),
        );
    }
    alert
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use agora_agentkit::reactor::{FsStorage, Storage};
    use agora_agentkit::state::codec::KeyframePolicy;
    use agora_agentkit::state::crypto::StateKey;
    use agora_agentkit::state::storage::plaintext_hash;
    use agora_agentkit::state::{GenesisClass, GenesisOrigin};

    use crate::sealed::api::fake::FakeApi;

    /// A server that holds exactly these entries.
    fn server(chain: Vec<SignedStateEntry>) -> FakeApi {
        let api = FakeApi::new();
        *api.chain.lock().unwrap() = Ok(chain);
        api
    }

    pub fn state(n: usize) -> serde_json::Value {
        serde_json::json!({ "memory": format!("synthetic {n}") })
    }

    /// A synthetic enrolled agent: keys, a state, genesis and that state
    /// sealed, enrolled.
    pub async fn enrolled(root: &Path) -> (AgentId, KeyStore, std::path::PathBuf) {
        let id = AgentId::new();
        let keys = KeyStore::new(root.join("secrets"));
        keys.ring()
            .insert(id, &agora_agentkit::crypto::generate_keypair().0)
            .unwrap();
        keys.write_state_key(
            id,
            &super::super::keys::StateKeyFile::new(&StateKey::generate(), 0, 0),
        )
        .unwrap();
        let mut fs = FsStorage::new(root.join("state"));
        fs.save_raw(id, state(1), None).await.unwrap();
        let dir = fs.agent_dir(id);
        let seal = AgentSeal::new(&dir, id);
        let k = keys.seal_keys(id).unwrap();
        let origin = GenesisOrigin {
            class: GenesisClass::None,
            incident: None,
            prior_copy_held: false,
            verified_at: None,
        };
        seal.genesis(&k, &origin, chrono::Utc::now()).await.unwrap();
        seal.append_state(
            &k,
            &KeyframePolicy::default(),
            StateEntryKind::Snapshot,
            &state(1),
            chrono::Utc::now(),
        )
        .await
        .unwrap();
        seal.enroll().await.unwrap();
        (id, keys, dir)
    }

    async fn local_hash(root: &Path, id: AgentId) -> Sha256Hex {
        let v = FsStorage::new(root.join("state"))
            .load_raw(id)
            .await
            .unwrap()
            .value;
        plaintext_hash(&v)
    }

    async fn sealed_chain(dir: &Path, id: AgentId) -> Vec<SignedStateEntry> {
        local_entries(&AgentSeal::new(dir, id)).await.unwrap()
    }

    #[tokio::test]
    async fn an_intact_state_runs_with_the_verdict_line() {
        let root = crate::sealed::TestDir::new("check-intact");
        let (id, keys, dir) = enrolled(&root).await;
        let server = server(Vec::new());
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &server,
        )
        .await;
        let Outcome::Run {
            seat, notice, hold, ..
        } = out
        else {
            panic!("{out:?}")
        };
        assert_eq!(
            (notice, hold),
            (false, None),
            "pending upload is no problem"
        );
        assert_eq!(seat.len(), 1);
        assert!(
            seat[0].starts_with("Your sealed state: intact. 1 version since "),
            "{}",
            seat[0]
        );
        assert!(
            seat[0].contains(VERIFIER)
                && seat[0].contains("(a marker for this version; the next session's will differ)")
                && !seat[0].contains("history tools")
        );

        // Agora has all of it: the same
        *server.chain.lock().unwrap() = Ok(sealed_chain(&dir, id).await);
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &server,
        )
        .await;
        let Outcome::Run {
            seat, hold: None, ..
        } = out
        else {
            panic!("{out:?}")
        };
        assert!(
            seat[0].starts_with("Your sealed state: intact, and in step with Agora."),
            "{seat:?}"
        );
        // Agora unreachable: said so
        *server.chain.lock().unwrap() = Err("unreachable".into());
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &server,
        )
        .await;
        let Outcome::Run { seat, .. } = out else {
            panic!("{out:?}")
        };
        assert!(
            seat[0].starts_with(
                "Your sealed state: intact here; Agora's copy couldn't be checked this session."
            ),
            "{seat:?}"
        );
    }

    #[tokio::test]
    async fn a_wiped_state_is_refused_and_never_sealed() {
        let root = crate::sealed::TestDir::new("check-wipe");
        let (id, keys, dir) = enrolled(&root).await;
        let mut fs = FsStorage::new(root.join("state"));
        fs.save_raw(id, serde_json::json!({ "memory": "" }), None)
            .await
            .unwrap();
        let server = server(Vec::new());
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &server,
        )
        .await;
        let Outcome::Refuse { reason, .. } = out else {
            panic!("{out:?}")
        };
        assert_eq!(reason, Refusal::Mismatch);
        refuse(&dir, id, "tarn", reason, "test", &Alerter::off()).await;
        assert!(AgentSeal::new(&dir, id).is_refused().await.unwrap());
        assert_eq!(
            Notice::load(&dir).unwrap().unwrap().reason,
            Refusal::Mismatch
        );
        // Still refused on the next start, whatever the state
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &server,
        )
        .await;
        assert!(matches!(out, Outcome::Refuse { .. }), "{out:?}");
    }

    #[tokio::test]
    async fn a_missing_head_file_is_refused() {
        let root = crate::sealed::TestDir::new("check-nohead");
        let (id, keys, dir) = enrolled(&root).await;
        let sealed = dir.join("sealed");
        std::fs::rename(sealed.join("head.json"), sealed.join("head.json.moved")).unwrap();
        let server = server(Vec::new());
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &server,
        )
        .await;
        assert!(
            matches!(
                out,
                Outcome::Refuse {
                    reason: Refusal::HeadMissing,
                    ..
                }
            ),
            "{out:?}"
        );
    }

    #[tokio::test]
    async fn a_chain_signed_by_an_untrusted_key_is_refused() {
        let root = crate::sealed::TestDir::new("check-key");
        let (id, _, dir) = enrolled(&root).await;
        // Another runner's keys: the chain's signer isn't trusted
        let other = KeyStore::new(root.join("other-secrets"));
        other
            .ring()
            .insert(id, &agora_agentkit::crypto::generate_keypair().0)
            .unwrap();
        let server = server(Vec::new());
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &other,
            &server,
        )
        .await;
        assert!(
            matches!(
                out,
                Outcome::Refuse {
                    reason: Refusal::ChainBroken,
                    ..
                }
            ),
            "{out:?}"
        );
    }

    #[tokio::test]
    async fn a_server_that_moved_on_is_a_held_fork_and_a_missing_head_is_held() {
        let root = crate::sealed::TestDir::new("check-fork");
        let (id, keys, dir) = enrolled(&root).await;
        let mine = sealed_chain(&dir, id).await;
        // Agora holds a sibling of our last version
        let signing = keys.signing(id).unwrap();
        let mut sibling = mine[1].entry.clone();
        sibling.state_at += 1;
        let server_chain = vec![mine[0].clone(), sibling.sign(&signing)];
        let server = server(server_chain);
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &server,
        )
        .await;
        let Outcome::Run { seat, hold, .. } = out else {
            panic!("{out:?}")
        };
        assert_eq!(hold, Some(Hold::Fork));
        assert_eq!(seat.last().unwrap(), FORK_LINE);

        // Everything uploaded, nothing waiting, and Agora has lost our
        // last version: the head is missing there
        let seal = AgentSeal::new(&dir, id);
        for item in seal.outbox().await.unwrap() {
            seal.mark_sent(item.ord).await.unwrap();
        }
        *server.chain.lock().unwrap() = Ok(vec![mine[0].clone()]);
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &server,
        )
        .await;
        let Outcome::Run { seat, hold, .. } = out else {
            panic!("{out:?}")
        };
        assert_eq!(hold, Some(Hold::HeadMissing));
        assert_eq!(seat.last().unwrap(), HEAD_MISSING_LINE);
    }

    #[tokio::test]
    async fn a_failed_seal_runs_held() {
        let root = crate::sealed::TestDir::new("check-sealfail");
        let (id, keys, dir) = enrolled(&root).await;
        let mut fs = FsStorage::new(root.join("state"));
        fs.save_raw(id, state(2), None).await.unwrap();
        let local = local_hash(&root, id).await;
        AgentSeal::new(&dir, id)
            .record_seal_failed(local, agora_agentkit::state::storage::SealFailureKind::Io)
            .await
            .unwrap();
        let server = server(Vec::new());
        let out = check(&dir, id, &local, Sizes::default(), &keys, &server).await;
        let Outcome::Run { seat, hold, .. } = out else {
            panic!("{out:?}")
        };
        assert_eq!(hold, Some(Hold::SealFailed));
        assert_eq!(seat, vec![SEAL_FAILED_LINE.to_string()]);
    }

    /// A memory wiped by a session and sealed by its save: the next start
    /// is intact, but tells of the drop. Sizes from the sidecar, or by
    /// opening the earlier version when there is none.
    #[tokio::test]
    async fn a_memory_drop_is_told_but_not_refused() {
        use agora_agentkit::state::storage::Mirrored;
        let root = crate::sealed::TestDir::new("check-drop");
        let (id, keys, dir) = enrolled(&root).await;
        let full =
            serde_json::json!({ "memory": { "content": "x".repeat(1000) }, "soul": {"n": 1} });
        let wiped = serde_json::json!({ "memory": { "content": "" }, "soul": {"n": 1} });
        let enrolment = std::sync::Arc::new(crate::sealed::SealedEnrolment::new(
            root.join("state"),
            root.join("secrets"),
        ));
        let mut store = Mirrored::new(FsStorage::new(root.join("state")), (), enrolment);
        let base = store.load_raw(id).await.unwrap().base;
        let base = store
            .save_raw(id, full.clone(), base.as_ref())
            .await
            .unwrap();
        let server = server(Vec::new());
        let out = check(
            &dir,
            id,
            &plaintext_hash(&full),
            Sizes::of(&full),
            &keys,
            &server,
        )
        .await;
        assert!(
            matches!(&out, Outcome::Run { drops, .. } if drops.is_empty()),
            "{out:?}"
        );
        store
            .save_raw(id, wiped.clone(), base.as_ref())
            .await
            .unwrap();

        for sidecar in [true, false] {
            if !sidecar {
                std::fs::rename(
                    dir.join("sealed/sizes.json"),
                    dir.join("sealed/sizes.moved"),
                )
                .unwrap();
            }
            let out = check(
                &dir,
                id,
                &plaintext_hash(&wiped),
                Sizes::of(&wiped),
                &keys,
                &server,
            )
            .await;
            let Outcome::Run {
                seat, drops, hold, ..
            } = out
            else {
                panic!("{out:?}")
            };
            assert_eq!(hold, None);
            assert_eq!(drops, vec![("memory", 100)], "sidecar {sidecar}");
            assert!(
                seat.last()
                    .unwrap()
                    .starts_with("Your memory is 100% smaller than at your last sealed version ("),
                "{seat:?}"
            );
        }
    }

    /// A `sealed/` that can't be read never refuses on its own: the agent
    /// runs, told, held.
    #[tokio::test]
    async fn an_unreadable_seal_runs_held_and_is_never_refused() {
        let root = crate::sealed::TestDir::new("check-unreadable");
        let (id, keys, dir) = enrolled(&root).await;
        std::fs::write(dir.join("sealed/head.json"), b"{ not json").unwrap();
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &server(Vec::new()),
        )
        .await;
        assert_eq!(
            out,
            Outcome::Run {
                seat: vec![UNREADABLE_LINE.to_string()],
                notice: false,
                hold: Some(Hold::Unreadable),
                drops: Vec::new(),
            }
        );
    }

    /// A rotation the server lists and the runner never made is counted,
    /// never trusted.
    #[tokio::test]
    async fn an_unknown_server_rotation_is_counted() {
        let root = crate::sealed::TestDir::new("check-rotations");
        let (id, keys, _) = enrolled(&root).await;
        let api = server(Vec::new());
        assert_eq!(unknown_rotations(&keys, &api, id).await, Some(0));
        let (other, _) = agora_agentkit::crypto::generate_keypair();
        let planted = agora_agentkit::state::KeyRotation::agent_signed(
            id,
            0,
            &keys.signing(id).unwrap(),
            &other,
            None,
            chrono::Utc::now(),
        );
        api.rotations.lock().unwrap().push(planted.clone());
        assert_eq!(unknown_rotations(&keys, &api, id).await, Some(1));
        keys.record_rotation(id, &planted).unwrap();
        assert_eq!(unknown_rotations(&keys, &api, id).await, Some(0));
    }

    /// No refusal without evidence: unreadable key records, or a chain
    /// that runs through an unreachable Agora, run held.
    #[tokio::test]
    async fn transient_failures_run_held_and_are_never_refused() {
        let root = crate::sealed::TestDir::new("check-transient");
        let (id, keys, dir) = enrolled(&root).await;
        std::fs::write(keys.agent_dir(id).join("rotations.json"), b"{ not json").unwrap();
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &server(Vec::new()),
        )
        .await;
        assert!(
            matches!(
                out,
                Outcome::Run {
                    hold: Some(Hold::Unreadable),
                    ..
                }
            ),
            "{out:?}"
        );
        assert!(!AgentSeal::new(&dir, id).is_refused().await.unwrap());

        // A main line through Agora's entries, Agora unreachable
        std::fs::rename(
            keys.agent_dir(id).join("rotations.json"),
            keys.agent_dir(id).join("rotations.broken"),
        )
        .unwrap();
        let seal = AgentSeal::new(&dir, id);
        let mine = local_entries(&seal).await.unwrap();
        let mut sibling = mine[1].entry.clone();
        sibling.state_at += 1;
        let api = server(vec![
            mine[0].clone(),
            sibling.sign(&keys.signing(id).unwrap()),
        ]);
        crate::sealed::accept::rebase(&root.join("state"), &keys, &api, id, "a test")
            .await
            .unwrap();
        *api.chain.lock().unwrap() = Err("unreachable".into());
        let out = check(
            &dir,
            id,
            &local_hash(&root, id).await,
            Sizes::default(),
            &keys,
            &api,
        )
        .await;
        assert!(
            matches!(
                out,
                Outcome::Run {
                    hold: Some(Hold::Unchecked),
                    ..
                }
            ),
            "{out:?}"
        );
        assert!(!seal.is_refused().await.unwrap());
    }

    /// The verdict: said, counted from the earliest snapshot, a pre-loss
    /// copy apart, the sizes.
    #[test]
    fn the_verdict_line_counts_snapshots_and_names_a_prior_copy_apart() {
        let key = agora_agentkit::crypto::generate_keypair().0;
        let id = AgentId::new();
        let at = |d: &str| {
            chrono::DateTime::parse_from_rfc3339(&format!("{d}T12:00:00Z"))
                .unwrap()
                .timestamp_micros()
        };
        let entry = |seq: u64, kind, state_at| {
            agora_agentkit::state::StateEntry {
                agora_state_entry: agora_agentkit::state::chain::STATE_ENTRY_VERSION,
                agent_id: id,
                seq,
                created_at: at("2026-11-02") + seq as i64,
                state_at,
                kind,
                prev_hash: None,
                merged_from: vec![],
                blob: None,
                target: None,
                key_gen: 0,
            }
            .sign(&key)
        };
        let line = vec![
            entry(0, StateEntryKind::Genesis, at("2026-07-24")),
            entry(1, StateEntryKind::ArchivedPrior, at("2026-05-03")),
            entry(2, StateEntryKind::Snapshot, at("2026-07-25")),
            entry(3, StateEntryKind::Snapshot, at("2026-11-01")),
        ];
        let head = HeadRef {
            entry_hash: line[3].entry_hash(),
            seq: 3,
        };
        let text = verdict_line(
            &line,
            &head,
            Sizes {
                memory_chars: 12_345,
                soul_chars: 2_100,
            },
            None,
            Server::Other,
        );
        assert!(
            verdict_line(&line, &head, Sizes::default(), None, Server::InStep)
                .starts_with("Your sealed state: intact, and in step with Agora. 2 versions")
        );
        assert!(
            verdict_line(&line, &head, Sizes::default(), None, Server::Unchecked).starts_with(
                "Your sealed state: intact here; Agora's copy couldn't be checked this session. 2"
            )
        );
        assert!(
            text.starts_with(
                "Your sealed state: intact. 2 versions since 2026-07-25, plus a copy of your \
                 state from before a loss, dated 2026-05-03; this session starts from head "
            ),
            "{text}"
        );
        assert!(
            text.contains(
                "Your memory is 12,345 characters and your soul 2,100, as at your last sealed \
                 version."
            ),
            "{text}"
        );
        // Beside a previous version: every change shown, a shrink included
        let now = Sizes {
            memory_chars: 13_539,
            soul_chars: 3_440,
        };
        let before = Sizes {
            memory_chars: 14_743,
            soul_chars: 3_440,
        };
        let text = verdict_line(&line, &head, now, Some(before), Server::Other);
        assert!(
            text.contains(
                "Your memory is 13,539 characters (\u{2212}1,204 since your last checked version) \
                 and your soul 3,440 (unchanged), as at your last sealed version."
            ),
            "{text}"
        );
        let grown = verdict_line(&line, &head, before, Some(now), Server::Other);
        assert!(
            grown.contains("14,743 characters (+1,204 since your last checked version)"),
            "{grown}"
        );
    }

    #[test]
    fn the_verifier_is_the_locked_agentkit() {
        let lock = include_str!("../../../../Cargo.lock");
        let version = lock
            .split("[[package]]")
            .find(|p| p.contains("name = \"agora-agentkit\""))
            .and_then(|p| p.lines().find_map(|l| l.strip_prefix("version = \"")))
            .and_then(|v| v.strip_suffix('"'))
            .unwrap();
        assert_eq!(
            VERIFIER,
            format!("git:agora-agentkit:v{version}:src/state/verify.rs")
        );
    }

    #[test]
    fn the_lines_name_no_model() {
        for line in [
            FORK_LINE,
            HEAD_MISSING_LINE,
            SEAL_FAILED_LINE,
            UNREADABLE_LINE,
            UNCHECKED_LINE,
        ] {
            let l = line.to_lowercase();
            for banned in ["claude", "haiku", "qwen", "model"] {
                assert!(!l.contains(banned), "{banned}: {line}");
            }
        }
    }
}
