//! `agora-seed rotate-keys --agent <id> [--signing] [--encryption]`.
//!
//! In this order, so no step can lose a key:
//!
//! 0. **pre-flight**, nothing written and nothing sent: every entry that
//!    will be re-signed is read and must be signed by one of the agent's
//!    keys, and the chain to the head must verify as it will after the
//!    rotation (the new key generated in memory, the rotation simulated).
//!    For an agent Agora holds a chain of, Agora's entries complete the
//!    line (`sent/` may be partial after a restore). Anything that would
//!    stop the command midway stops it here instead;
//! 1. the new secrets are written to `secrets.next.json` (or, on a retry,
//!    the ones already there are reused), and the old keys are appended to
//!    `past_keys.json`;
//! 2. the server takes the change: an agent-signed rotation of the Ed25519
//!    key (signed by the old key, with the new key's proof; its `seq` the
//!    next after `rotations.json`'s), recorded in `rotations.json` once
//!    taken, then the new X25519 key, registered under the current signing
//!    key;
//! 3. an enrolled agent's state key is re-wrapped to the new X25519 key
//!    with the next `wrap_seq`: the record is kept in
//!    `state_key.wrap.json` **before** it is posted, so a rerun posts the
//!    same bytes (the server refuses a different record at the same
//!    `(key_gen, wrap_seq)`), then `state_key.json` names its `wrap_seq`;
//! 4. every entry kept here that Agora doesn't hold is re-signed with the
//!    new key ([`resign_local`]): an agent that doesn't upload, and an
//!    uploader's unsent `.tmp`/`.displaced`/`.abandoned` items alike;
//! 5. only then `secrets.next.json` becomes `secrets.json`; the replaced
//!    file is kept as `secrets.<utc>.json`.
//!
//! A failure at 2 to 4 leaves `secrets.json` as it was, and a rerun picks
//! up `secrets.next.json`: the pre-flight runs again, so once its cause is
//! fixed the rerun completes. `rotate-keys --status` says which step an
//! unfinished rotation is at and what blocks it ([`status`]).
//! `secrets.next.json` may hold the only copy of the key Agora takes for
//! the agent: nothing here ever suggests removing it. Old keys are never
//! deleted. Until step 5 the runner leaves the agent out of every run
//! (`main.rs`'s `mid_rotation`): that is what makes a chain half re-signed
//! safe.
//!
//! **Three kinds of agent** ([`Reach`]):
//!
//! - **uploading**: as above; the rotation names Agora's head, which must
//!   be the local head with nothing waiting.
//! - **stopped after uploading**: Agora holds its chain to the last
//!   version it acknowledged (`sealed/sent/`), and may be read. The
//!   rotation names Agora's head, which must be the newest version in
//!   `sent/`; the versions sealed here since (never sent) are re-signed
//!   with the new key, as below, and the re-wrap is posted (a wrap of the
//!   key, not state: without it the blobs already on Agora could not be
//!   opened after an X25519 rotation).
//! - **never uploaded**: nothing of its chain is on Agora, so the rotation
//!   names no head (`old_key_last: None`, as for an agent not enrolled),
//!   and the old key's trust then covers no entry. Every entry kept here
//!   is re-signed with the new key, and the re-wrap is kept locally
//!   (`state_key.wrap.json`, `state_key.json`'s `wrap_seq`), never
//!   posted: the opt-in posts the current wrap.
//!
//! Re-signing ([`resign_local`]) is sound because an entry's hash covers
//! the entry, not its signature (agentkit `StateEntry::entry_hash`):
//! hashes, links and the head are unchanged, and the chain verifies under
//! the runner's recorded keys (the old key trusted through the head Agora
//! holds, or through no entry; the new one current), as Agora verifies an
//! upload at a later opt-in or resume (each entry against the agent's
//! current key). Entries Agora acknowledged (`sent/*.json`) keep the old
//! key's signature, which Agora's copy carries.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use agora_agentkit::crypto::{SigningKey, VerifyingKey};
use agora_agentkit::govlog::Sha256Hex;
use agora_agentkit::ids::AgentId;
use agora_agentkit::state::storage::{AgentSeal, OutboxItem};
use agora_agentkit::state::verify::{HeadRef, TrustedKeys, verify_chain};
use agora_agentkit::state::{KeyRotation, SignedStateEntry};
use anyhow::Context;
use chrono::Utc;

use super::KeyStore;
use super::api::SealedApi;
use super::keys::{SecretsFile, StateKeyFile};
use super::version::Reach;

/// What changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rotated {
    pub signing: bool,
    pub encryption: bool,
    /// The state key was re-wrapped to the new X25519 key.
    pub rewrapped: bool,
    /// `secrets.next.json` from an earlier attempt was used.
    pub resumed: bool,
    /// The re-wrap was kept locally only (`state_key.wrap.json`): the
    /// agent never uploaded, and its opt-in posts it.
    pub wrap_kept_local: bool,
    /// Signed entries re-signed with the new key; ones already re-signed
    /// by an earlier attempt not counted.
    pub resigned: usize,
}

/// What the pre-flight established, for the steps after it.
struct Plan {
    /// The head the old key's trust ends at (`old_key_last`).
    last: Option<HeadRef>,
    /// Entries Agora holds: never re-signed (its copy carries the old
    /// key's signature).
    on_agora: HashSet<Sha256Hex>,
    /// Agora's entries, to complete a local line.
    server: Vec<SignedStateEntry>,
}

/// A file holding signed entries that a rotation may rewrite.
struct Held {
    path: PathBuf,
    entries: Vec<SignedStateEntry>,
}

/// Every signed entry kept in `seal`'s directory, by file: outbox items of
/// any kind (`.json`, `.tmp`, `.displaced`, `.abandoned`), `sent/` items
/// (`.json`, `.lost`), and the copies a resolution kept under
/// `sealed/lost/<utc>/` (its `items/` and its `INTENT`/`DONE` record).
/// Read-only. Oldest first, by the ord the item names begin with.
async fn held_entries(seal: &AgentSeal) -> anyhow::Result<Vec<Held>> {
    let mut files = item_files(&seal.dir().join("outbox"))?;
    files.extend(item_files(&seal.dir().join("sent"))?);
    files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    for (dir, _) in super::lost::resolutions(seal)? {
        files.extend(item_files(&dir.join("items"))?);
    }
    let mut out = Vec::new();
    for path in files {
        let item: OutboxItem = serde_json::from_slice(
            &std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
        )
        .map_err(|e| {
            anyhow::anyhow!(
                "{} isn't an outbox item ({:?})",
                path.display(),
                e.classify()
            )
        })?;
        out.push(Held {
            path,
            entries: vec![item.entry],
        });
    }
    for (dir, _) in super::lost::resolutions(seal)? {
        for (path, entries) in super::lost::record_entries(&dir)? {
            out.push(Held { path, entries });
        }
    }
    Ok(out)
}

/// The main line to `head`: from the entries here, completed by Agora's
/// (`server`) where it runs through them. `Err` names the missing link.
async fn line_to(
    seal: &AgentSeal,
    head: &HeadRef,
    server: &[SignedStateEntry],
) -> anyhow::Result<Vec<SignedStateEntry>> {
    let local = super::check::local_entries(seal)
        .await
        .map_err(anyhow::Error::msg)?;
    if let Some(line) = super::check::main_line(&local, &head.entry_hash) {
        return Ok(line);
    }
    let all = super::check::merge_server(&local, server).map_err(|hash| {
        anyhow::anyhow!("Agora's copy of entry {hash} differs from the one held here")
    })?;
    super::check::main_line(&all, &head.entry_hash).with_context(|| {
        format!(
            "the main line to head {} (seq {}) has a missing link{}. Put the missing \
             sealed/sent or sealed/outbox files back from a backup of the agent's directory, \
             then run this again",
            head.entry_hash,
            head.seq,
            if server.is_empty() {
                " here, and nothing of the agent is on Agora to complete it"
            } else {
                ", here and on Agora"
            }
        )
    })
}

/// Step 0: everything that could stop the command after the server took
/// the rotation, checked before anything is written or sent (see the
/// module's docs). `rotate`: the signing key changes; `new`: the key it
/// changes to (generated in memory, or `secrets.next.json`'s).
#[allow(clippy::too_many_arguments)]
async fn preflight(
    agent_dir: &Path,
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
    reach: Reach,
    old: &SigningKey,
    new: &SigningKey,
    rotate: bool,
) -> anyhow::Result<Plan> {
    let mut plan = Plan {
        last: None,
        on_agora: HashSet::new(),
        server: Vec::new(),
    };
    if !super::is_enrolled(agent_dir) || !rotate {
        return Ok(plan);
    }
    let seal = AgentSeal::new(agent_dir, id);
    if reach.on_agora() {
        plan.server = super::api::chain(api, id, old)
            .await
            .map_err(anyhow::Error::msg)
            .context("reading the agent's chain on Agora")?;
    }
    plan.on_agora
        .extend(plan.server.iter().map(|e| e.entry_hash()));
    plan.on_agora
        .extend(seal.sent().await?.iter().map(|i| i.entry.entry_hash()));
    let server_head = plan.server.last().map(|e| HeadRef {
        entry_hash: e.entry_hash(),
        seq: e.entry.seq,
    });
    let head = seal.head().await?.map(|h| h.head);
    match reach {
        Reach::Local => {}
        Reach::Uploads => {
            let waiting = seal.outbox().await?.len();
            anyhow::ensure!(
                waiting == 0,
                "pending uploads: {waiting} sealed version(s) wait in sealed/outbox. Let the \
                 runner upload them, or run this with the runner stopped after a drain"
            );
            anyhow::ensure!(
                head == server_head,
                "the sealed head here and Agora's differ: resolve that first (the start check \
                 says how)"
            );
            plan.last = server_head;
        }
        Reach::Stopped => plan.last = server_head,
    }
    let Some(head) = head else {
        anyhow::bail!(
            "agent {id}'s sealed/head.json is missing: `sealed-accept` rebuilds it first"
        );
    };
    let line = line_to(&seal, &head, &plan.server).await?;
    if let Some(last) = plan.last {
        // Stopped: Agora's head must be on the line here, and what follows
        // it unsent
        anyhow::ensure!(
            line.iter().any(|e| e.entry_hash() == last.entry_hash),
            "Agora's head for agent {id} ({} at seq {}) is not on the sealed line here: \
             resolve that first (the start check says how)",
            last.entry_hash,
            last.seq
        );
    }
    // Every entry to be re-signed is the agent's own
    let new_vk = new.verifying_key();
    let mut signers = vec![old.verifying_key(), new_vk];
    signers.extend(keys.past_keys(id)?.signing_public());
    for held in held_entries(&seal).await? {
        for e in &held.entries {
            if !plan.on_agora.contains(&e.entry_hash()) {
                anyhow::ensure!(
                    signers.iter().any(|k| e.verify(k)),
                    "{} holds an entry (seq {}, {}) signed by none of the agent's keys: it \
                     would not be re-signed, and the chain would not verify after the \
                     rotation. Nothing was written. Set that file aside (or put back the right \
                     one from a backup), then run this again",
                    held.path.display(),
                    e.entry.seq,
                    e.entry_hash()
                );
            }
        }
    }
    // The line as it will be, under the trust as it will be
    let trusted = trusted_after(keys, id, old, new, plan.last)?;
    let simulated: Vec<SignedStateEntry> = line
        .iter()
        .map(|e| {
            if plan.on_agora.contains(&e.entry_hash()) || e.verify(&new_vk) {
                e.clone()
            } else {
                e.entry.clone().sign(new)
            }
        })
        .collect();
    let report = verify_chain(id, &simulated, &trusted);
    anyhow::ensure!(
        report.intact && report.head == Some(head),
        "agent {id}'s chain would not verify after the rotation ({:?}). Nothing was written",
        report.problems
    );
    Ok(plan)
}

/// The keys the runner will trust once the rotation to `new` is recorded:
/// `rotations.json` as it is if it already ends at `new`, else with the
/// rotation simulated (naming `last`).
fn trusted_after(
    keys: &KeyStore,
    id: AgentId,
    old: &SigningKey,
    new: &SigningKey,
    last: Option<HeadRef>,
) -> anyhow::Result<TrustedKeys> {
    let new_vk = new.verifying_key();
    let mut rotations = keys.rotations(id)?;
    let recorded = rotations
        .iter()
        .max_by_key(|r| r.seq)
        .is_some_and(|r| r.new_public_key == agora_agentkit::govlog::PublicKeyHex::from(&new_vk));
    if !recorded && old.verifying_key() != new_vk {
        let seq = rotations.iter().map(|r| r.seq + 1).max().unwrap_or(0);
        rotations.push(KeyRotation::agent_signed(
            id,
            seq,
            old,
            new,
            last,
            Utc::now(),
        ));
    }
    let first = match rotations.iter().min_by_key(|r| r.seq) {
        Some(r) => r
            .old_public_key
            .to_verifying_key()
            .map_err(|e| anyhow::anyhow!("rotations.json: {e}"))?,
        None => new_vk,
    };
    let trusted = TrustedKeys::new(first).with_rotations(&rotations);
    anyhow::ensure!(
        trusted.current() == new_vk,
        "the recorded rotations don't lead to the new key ({:?})",
        trusted.notices()
    );
    Ok(trusted)
}

pub async fn rotate_keys(
    state_dir: &Path,
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
    signing: bool,
    encryption: bool,
) -> anyhow::Result<Rotated> {
    anyhow::ensure!(
        signing || encryption,
        "name what to rotate: --signing, --encryption, or both"
    );
    let agent_dir = state_dir.join(id.to_string());
    // An uploading marker in doubt: the runner can't tell whether Agora
    // holds a chain of the agent, so nothing is sent or written until its
    // operator settles it
    if super::check::Upload::of(&agent_dir, keys, id) == super::check::Upload::InDoubt {
        anyhow::bail!(
            "agent {id}'s uploading marker is in doubt (sealed/uploading and its copy in the \
             secrets tree disagree, or can't be read): nothing was sent and nothing written. \
             Check its answer on file (`sealed_state.json`), then settle it: `agora-seed \
             sealed-stop --agent {id} --note \"…\"` if it is not to upload, or `agora-seed \
             sealed-enroll --opt-in --agent {id} --yes` if its answer in force is yes; then \
             run this again. Never copy a marker by hand"
        );
    }
    // No session in progress: one would sign with a key retired under it.
    // An enrolled agent's lock is held for the whole command.
    super::session::no_session(state_dir, id, false)?;
    let enrolled = super::is_enrolled(&agent_dir);
    // The session lock either way (`take` holds it for a chained one)
    let _lock = if enrolled {
        (
            Some(super::session::take(state_dir, id, false).await?),
            None,
        )
    } else {
        (None, Some(super::session::agent_lock(state_dir, id)?))
    };
    let reach = super::version::reach(&agent_dir, keys, id);
    let current = keys
        .secrets(id)?
        .with_context(|| format!("agent {id} has no secrets.json"))?;
    let old_signing = current.signing()?;
    let old_encryption = current.encryption()?;

    // The new keys: an earlier attempt's, or made now, in memory until the
    // pre-flight passes
    let (next, resumed) = match keys.next_secrets(id)? {
        Some(next) => (next, true),
        None => {
            let new_signing = if signing {
                agora_agentkit::crypto::generate_keypair().0
            } else {
                old_signing.clone()
            };
            let new_encryption = if encryption {
                Some(agora_agentkit::envelope::generate_encryption_keypair().0)
            } else {
                old_encryption.clone()
            };
            (
                SecretsFile::new(&new_signing, new_encryption.as_ref()),
                false,
            )
        }
    };
    let new_signing = next.signing()?;
    let new_encryption = next.encryption()?;
    let mut done = Rotated {
        signing: new_signing.to_bytes() != old_signing.to_bytes(),
        encryption: new_encryption.as_ref().map(|k| k.to_bytes())
            != old_encryption.as_ref().map(|k| k.to_bytes()),
        resumed,
        ..Rotated::default()
    };

    // 0. Pre-flight: nothing written, nothing sent
    let plan = preflight(
        &agent_dir,
        keys,
        api,
        id,
        reach,
        &old_signing,
        &new_signing,
        done.signing,
    )
    .await?;

    // 1. The new keys, on disk before anything else
    if !resumed {
        keys.write_next_secrets(id, &next)
            .context("writing secrets.next.json")?;
    }
    keys.retire(
        id,
        done.signing.then_some(&old_signing),
        if done.encryption {
            old_encryption.as_ref()
        } else {
            None
        },
    )
    .context("recording the old keys in past_keys.json")?;

    // 2. The server
    let new_public = agora_agentkit::govlog::PublicKeyHex::from(&new_signing.verifying_key());
    let recorded = keys.rotations(id)?;
    // An earlier attempt got the signing rotation taken and recorded, and
    // stopped after it: never a second rotation to the same key
    let signing_done = done.signing
        && recorded
            .iter()
            .max_by_key(|r| r.seq)
            .is_some_and(|r| r.new_public_key == new_public);
    if signing_done && keys.pending_rotation(id)?.is_some() {
        let seq = recorded.iter().map(|r| r.seq).max().unwrap_or(0);
        keys.settle_pending_rotation(id, seq)?;
    }
    if done.signing && !signing_done {
        let seq = recorded.iter().map(|r| r.seq + 1).max().unwrap_or(0);
        // The rotation an earlier attempt signed and sent, if it was this one
        let rotation = match keys.pending_rotation(id)? {
            Some(r) if r.new_public_key == new_public && r.seq == seq => r,
            _ => {
                // The old key signs nothing past the head Agora confirms
                let r = KeyRotation::agent_signed(
                    id,
                    seq,
                    &old_signing,
                    &new_signing,
                    plan.last,
                    Utc::now(),
                );
                keys.write_pending_rotation(id, &r)
                    .context("keeping the signed rotation before sending it")?;
                r
            }
        };
        // An earlier attempt's answer may have been lost: if the server
        // already takes the new key, the rotation was taken
        let taken = resumed
            && api
                .signer_registered(id, &new_signing)
                .await
                .map_err(anyhow::Error::msg)
                .context("asking the server which signing key it holds")?;
        if taken {
            tracing::warn!(
                event_type = "sealed_rotation_already_taken",
                agent_id = %id,
                seq,
                "the server already holds the new signing key: an earlier attempt's answer was lost"
            );
        } else if let Err(e) = api.rotate_key(id, &old_signing, &rotation).await {
            if e.starts_with(super::api::CONFLICT) {
                anyhow::bail!(
                    "Agora's sealed head moved on from the one this rotation names ({e}): \
                     secrets.json is unchanged. Let the runner upload what waits, stop it, \
                     and run this again"
                );
            }
            return Err(anyhow::Error::msg(e).context(
                "the server refused the signing-key rotation; secrets.json is unchanged",
            ));
        }
        // Ours to verify the chain by, never the server's
        keys.record_rotation(id, &rotation)
            .context("recording the rotation in rotations.json")?;
        keys.settle_pending_rotation(id, seq)?;
    }
    super::crash::point("rotate:recorded")?;
    if done.encryption
        && let Some(secret) = &new_encryption
    {
        let public = agora_agentkit::envelope::EncryptionPublicKey::from(secret);
        api.register_encryption_key(id, &new_signing, &public)
            .await
            .map_err(anyhow::Error::msg)
            .context("the server refused the new encryption key; secrets.json is unchanged")?;
    }

    // 3. The state key, re-wrapped
    if done.encryption
        && let Some(secret) = &new_encryption
        && let Some(file) = keys.state_key_file(id)?
    {
        done.wrap_kept_local = !rewrap(keys, api, id, &file, secret, &new_signing, reach).await?;
        done.rewrapped = true;
    }

    // 4. Everything here Agora doesn't hold, re-signed
    if enrolled && done.signing {
        let seal = AgentSeal::new(&agent_dir, id);
        let mut signers = vec![old_signing.verifying_key()];
        signers.extend(keys.past_keys(id)?.signing_public());
        done.resigned = resign_local(&seal, &new_signing, &signers, &plan.on_agora).await?;
        let trusted = keys
            .trusted_ending_at(id, &new_signing.verifying_key())
            .context("the recorded rotations")?;
        if let Some(head) = seal.head().await? {
            let line = line_to(&seal, &head.head, &plan.server).await?;
            let report = verify_chain(id, &line, &trusted);
            anyhow::ensure!(
                report.intact && report.head == Some(head.head),
                "the re-signed chain doesn't verify under the new key ({:?}): secrets.json is \
                 unchanged, secrets.next.json holds the new key Agora takes, and the agent \
                 stays out of runs. `rotate-keys --agent {id} --status` says what blocks it; \
                 run this again once that is fixed",
                report.problems
            );
        }
    }
    super::crash::point("rotate:resigned")?;

    // 5. Only now
    keys.promote_next_secrets(id)
        .context("replacing secrets.json")?;
    tracing::warn!(
        event_type = "sealed_keys_rotated",
        agent_id = %id,
        signing = done.signing,
        encryption = done.encryption,
        rewrapped = done.rewrapped,
        wrap_kept_local = done.wrap_kept_local,
        resigned = done.resigned,
        resumed = done.resumed,
        "agent keys rotated; the old ones are in past_keys.json"
    );
    Ok(done)
}

/// Where an unfinished rotation stands (`rotate-keys --status`): read-only,
/// no server write (a stopped or uploading agent's chain is read, to say
/// whether the line would verify).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Status {
    /// `secrets.next.json` is there: a rotation is unfinished.
    pub unfinished: bool,
    /// The signing rotation: not begun, sent with no answer recorded
    /// (`rotation.pending.json`), or recorded in `rotations.json`.
    pub signing: &'static str,
    /// The re-wrap: kept in `state_key.wrap.json` for the new X25519 key,
    /// and whether `state_key.json` names it.
    pub wrap: &'static str,
    /// Entries here, not held by Agora, still signed by the old key.
    pub to_resign: usize,
    /// What stops a rerun from finishing, if anything.
    pub blocked: Option<String>,
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.unfinished {
            return f.write_str("no rotation is unfinished (no secrets.next.json)");
        }
        write!(
            f,
            "a rotation is unfinished; secrets.next.json holds the new keys (keep it). signing \
             rotation: {}; re-wrap: {}; entries still to re-sign: {}; {}",
            self.signing,
            self.wrap,
            self.to_resign,
            match &self.blocked {
                Some(why) => format!("blocked: {why}"),
                None => "nothing blocks it: run `rotate-keys` again with the same flags".into(),
            }
        )
    }
}

pub async fn status(
    state_dir: &Path,
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
) -> anyhow::Result<Status> {
    let Some(next) = keys.next_secrets(id)? else {
        return Ok(Status {
            unfinished: false,
            signing: "",
            wrap: "",
            to_resign: 0,
            blocked: None,
        });
    };
    let agent_dir = state_dir.join(id.to_string());
    let reach = super::version::reach(&agent_dir, keys, id);
    let old = keys
        .secrets(id)?
        .with_context(|| format!("agent {id} has no secrets.json"))?;
    let (old_signing, new_signing) = (old.signing()?, next.signing()?);
    let new_public = agora_agentkit::govlog::PublicKeyHex::from(&new_signing.verifying_key());
    let rotate = old_signing.to_bytes() != new_signing.to_bytes();
    let signing = if !rotate {
        "not rotated"
    } else if keys
        .rotations(id)?
        .iter()
        .max_by_key(|r| r.seq)
        .is_some_and(|r| r.new_public_key == new_public)
    {
        "recorded (Agora took it)"
    } else if keys.pending_rotation(id)?.is_some() {
        "sent, no answer recorded (the rerun asks Agora which key it holds)"
    } else {
        "not sent yet"
    };
    let wrap = match (
        next.encryption()?,
        keys.wrap_record(id)?,
        keys.state_key_file(id)?,
    ) {
        (Some(secret), Some(r), Some(file))
            if r.wrapped_to_pub == super::keys::x25519_public_hex(&secret) =>
        {
            if file.wrap_seq == r.wrap_seq {
                "kept, and named by state_key.json"
            } else {
                "kept, not yet posted or named"
            }
        }
        _ => "not made yet (or no encryption rotation)",
    };
    let new_vk = new_signing.verifying_key();
    let mut to_resign = 0;
    let mut blocked = None;
    match preflight(
        &agent_dir,
        keys,
        api,
        id,
        reach,
        &old_signing,
        &new_signing,
        rotate,
    )
    .await
    {
        Ok(plan) => {
            if super::is_enrolled(&agent_dir) && rotate {
                let seal = AgentSeal::new(&agent_dir, id);
                for held in held_entries(&seal).await? {
                    to_resign += held
                        .entries
                        .iter()
                        .filter(|e| !plan.on_agora.contains(&e.entry_hash()) && !e.verify(&new_vk))
                        .count();
                }
            }
        }
        Err(e) => blocked = Some(format!("{e:#}")),
    }
    Ok(Status {
        unfinished: true,
        signing,
        wrap,
        to_resign,
        blocked,
    })
}

/// Re-wrap the state key in `file` to `secret`'s public key, signed by
/// `signing`: the record kept in `state_key.wrap.json` first (or the one an
/// earlier attempt kept, reused), then posted unless the agent never
/// uploaded, then `state_key.json` names its `wrap_seq`. The seq is the
/// one after the highest Agora holds when Agora may be read (it may be
/// ahead of `state_key.json`), else after `state_key.json`'s. `Ok(true)`:
/// posted (or already on Agora); `Ok(false)`: kept locally for the opt-in.
async fn rewrap(
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
    file: &StateKeyFile,
    secret: &agora_agentkit::envelope::EncryptionSecretKey,
    signing: &SigningKey,
    reach: Reach,
) -> anyhow::Result<bool> {
    let local_only = !reach.on_agora();
    let state_key = file.state_key()?;
    // An earlier attempt's: the same key, to this X25519 key, signed by
    // this signing key, at or past the current wrap
    let kept = super::keys::kept_wrap(
        keys,
        id,
        file,
        file.wrap_seq,
        secret,
        &signing.verifying_key(),
    )?;
    let record = match kept {
        Some(r) => r,
        None => {
            // Past the current wrap and whatever Agora holds of this
            // generation
            let held = if local_only {
                Vec::new()
            } else {
                api.state_keys(id, signing)
                    .await
                    .map_err(anyhow::Error::msg)
                    .context("reading the agent's state key records on Agora")?
            };
            super::keys::new_wrap(keys, id, file, file.wrap_seq + 1, secret, signing, &held)
                .context("re-wrapping the state key")?
        }
    };
    super::crash::point("rotate:wrap_kept")?;
    if file.wrap_seq == record.wrap_seq {
        // Done by an earlier attempt, state_key.json written after it
        return Ok(!local_only);
    }
    if !local_only {
        if let Err(e) = api.post_state_key(id, signing, &record).await {
            // Taken by an earlier attempt whose answer was lost: these bytes
            let held = api
                .state_keys(id, signing)
                .await
                .map_err(anyhow::Error::msg)
                .context("asking Agora which state key records it holds")?;
            anyhow::ensure!(
                held.contains(&record),
                "the server refused the re-wrapped state key ({e}); secrets.json is unchanged"
            );
            tracing::warn!(
                event_type = "sealed_rewrap_already_taken",
                agent_id = %id,
                key_gen = record.key_gen,
                wrap_seq = record.wrap_seq,
                "Agora already holds this re-wrap: an earlier attempt's answer was lost"
            );
        }
        super::crash::point("rotate:wrap_posted")?;
    }
    keys.write_state_key(
        id,
        &StateKeyFile::new(&state_key, file.key_gen, record.wrap_seq),
    )?;
    Ok(!local_only)
}

/// Every signed entry kept in `seal`'s directory that Agora doesn't hold
/// (`on_agora`), re-signed with `new`, in order (oldest first): see
/// [`held_entries`] for the files. Each file is rewritten whole (a unique
/// temporary, fsync, rename over it, fsync of its directory), so there is
/// never a moment without it. An entry already signed by `new` is left as
/// it is, so a rerun after a cut finishes the rest; one signed by none of
/// `signers` (the agent's old keys) is refused, never signed over (the
/// pre-flight refuses it before anything is written). Returns how many
/// entries were re-signed.
pub async fn resign_local(
    seal: &AgentSeal,
    new: &SigningKey,
    signers: &[VerifyingKey],
    on_agora: &HashSet<Sha256Hex>,
) -> anyhow::Result<usize> {
    let new_vk = new.verifying_key();
    let resign = |path: &Path, entry: &mut SignedStateEntry| {
        if entry.verify(&new_vk) || on_agora.contains(&entry.entry_hash()) {
            return Ok(false);
        }
        anyhow::ensure!(
            signers.iter().any(|k| entry.verify(k)),
            "{} holds an entry (seq {}, {}) signed by none of the agent's keys: not \
             re-signing it. secrets.json is unchanged and the agent stays out of runs",
            path.display(),
            entry.entry.seq,
            entry.entry_hash()
        );
        *entry = entry.entry.clone().sign(new);
        Ok(true)
    };
    let mut count = 0;
    for held in held_entries(seal).await? {
        // A resolution's record: rewritten by its own module
        if matches!(
            held.path.file_name().and_then(|n| n.to_str()),
            Some("INTENT" | "DONE")
        ) {
            continue;
        }
        let mut item: OutboxItem = serde_json::from_slice(&std::fs::read(&held.path)?)?;
        if resign(&held.path, &mut item.entry)? {
            super::write_private(&held.path, &serde_json::to_vec_pretty(&item)?)
                .with_context(|| format!("rewriting {}", held.path.display()))?;
            count += 1;
            super::crash::point("rotate:resigned_one")?;
        }
    }
    for (dir, _) in super::lost::resolutions(seal)? {
        count += super::lost::resign_record(&dir, |path, entry| resign(path, entry))?;
    }
    Ok(count)
}

/// The files under `dir` that hold one outbox item each, sorted by name:
/// every regular file but a writer's temporary (`*.writing`).
fn item_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).with_context(|| format!("reading {}", dir.display())),
    };
    let mut out = Vec::new();
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let ext = path.extension().and_then(|x| x.to_str());
        if entry.file_type()?.is_file()
            && matches!(
                ext,
                Some("json" | "tmp" | "displaced" | "abandoned" | "lost")
            )
        {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::sealed::api::fake::FakeApi;
    use crate::sealed::check::{self, Outcome, tests::enrolled};
    use agora_agentkit::reactor::{FsStorage, Storage};
    use agora_agentkit::state::crypto::StateKey;
    use agora_agentkit::state::storage::plaintext_hash;

    /// Everything sealed is on the fake Agora: nothing waits to upload.
    async fn uploaded(dir: &std::path::Path, id: AgentId, api: &FakeApi) {
        let seal = agora_agentkit::state::storage::AgentSeal::new(dir, id);
        let chain = check::local_entries(&seal).await.unwrap();
        for item in seal.outbox().await.unwrap() {
            seal.mark_sent(item.ord).await.unwrap();
        }
        *api.chain.lock().unwrap() = Ok(chain);
    }

    #[tokio::test]
    async fn new_keys_are_written_first_and_secrets_replaced_only_after_the_server() {
        let root = crate::sealed::TestDir::new("rotate");
        let (id, keys, dir) = enrolled(&root).await;
        keys.ring().ensure_encryption_key(id).unwrap();
        let old_signing = keys.signing(id).unwrap();
        let old_enc = keys.encryption(id).unwrap();
        let api = FakeApi::new();
        *api.public_key.lock().unwrap() = Some(old_signing.verifying_key());
        uploaded(&dir, id, &api).await;

        let state_dir = root.join("state");
        assert!(
            rotate_keys(&state_dir, &keys, &api, id, false, false)
                .await
                .is_err()
        );

        // The server refuses: nothing replaced, the new keys kept
        *api.fail.lock().unwrap() = Some("rotate_key");
        assert!(
            rotate_keys(&state_dir, &keys, &api, id, true, true)
                .await
                .is_err()
        );
        assert_eq!(keys.signing(id).unwrap().to_bytes(), old_signing.to_bytes());
        let staged = keys.next_secrets(id).unwrap().unwrap().signing().unwrap();
        let past = keys.past_keys(id).unwrap();
        assert_eq!(past.signing_public(), vec![old_signing.verifying_key()]);
        assert_eq!(past.encryption.len(), 1);

        // A retry resumes with the same new keys
        *api.fail.lock().unwrap() = None;
        api.calls.lock().unwrap().clear();
        let done = rotate_keys(&state_dir, &keys, &api, id, true, true)
            .await
            .unwrap();
        assert_eq!(
            done,
            Rotated {
                signing: true,
                encryption: true,
                rewrapped: true,
                resumed: true,
                ..Rotated::default()
            }
        );
        assert_eq!(
            *api.calls.lock().unwrap(),
            [
                // The pre-flight's read, and the re-wrap's seq from Agora
                "chain",
                "signer_registered",
                "rotate_key",
                "register_encryption_key",
                "state_keys",
                "post_state_key"
            ],
            "rotation, then the X25519 key under the new signer, then the re-wrap"
        );
        assert_eq!(keys.signing(id).unwrap().to_bytes(), staged.to_bytes());
        assert!(keys.next_secrets(id).unwrap().is_none());
        let new_enc = keys.encryption(id).unwrap();
        assert_ne!(new_enc.to_bytes(), old_enc.to_bytes());
        // The re-wrap opens with the new X25519 key to the same state key
        let record = api.keys.lock().unwrap().last().unwrap().clone();
        assert_eq!(record.wrap_seq, 1);
        let file = keys.state_key_file(id).unwrap().unwrap();
        assert_eq!(file.wrap_seq, 1);
        assert_eq!(
            StateKey::unwrap(&record, &new_enc).unwrap().expose(),
            file.state_key().unwrap().expose()
        );
        // Old keys are still on disk, never deleted; the rotation is ours
        assert_eq!(keys.past_keys(id).unwrap().signing.len(), 1);
        let rotations = keys.rotations(id).unwrap();
        assert_eq!(rotations.len(), 1);
        assert_eq!(rotations[0].seq, 0);
        assert!(rotations[0].verify_agent_signed().is_ok());
        let backups = std::fs::read_dir(keys.agent_dir(id))
            .unwrap()
            .filter(|e| {
                let n = e.as_ref().unwrap().file_name();
                let n = n.to_string_lossy();
                n.starts_with("secrets.2") && n.ends_with(".json")
            })
            .count();
        assert_eq!(backups, 1);

        // The chain signed by the old key still verifies at the next start
        let local = plaintext_hash(
            &FsStorage::new(root.join("state"))
                .load_raw(id)
                .await
                .unwrap()
                .value,
        );
        let out = check::check(&dir, id, &local, Default::default(), &keys, &api).await;
        assert!(matches!(out, Outcome::Run { hold: None, .. }), "{out:?}");
    }

    /// The server took the rotation but its answer was lost: the retry
    /// finds the new key registered, records the same signed rotation, and
    /// never rotates twice.
    #[tokio::test]
    async fn a_rotation_whose_answer_was_lost_is_recorded_on_retry() {
        let root = crate::sealed::TestDir::new("rotate-lost");
        let (id, keys, dir) = enrolled(&root).await;
        let state_dir = root.join("state");
        let api = FakeApi::new();
        *api.public_key.lock().unwrap() = Some(keys.signing(id).unwrap().verifying_key());
        uploaded(&dir, id, &api).await;
        *api.fail.lock().unwrap() = Some("rotate_key_reply");
        assert!(
            rotate_keys(&state_dir, &keys, &api, id, true, false)
                .await
                .is_err()
        );
        let sent = keys.pending_rotation(id).unwrap().unwrap();
        assert!(
            keys.rotations(id).unwrap().is_empty(),
            "not recorded without an answer"
        );

        *api.fail.lock().unwrap() = None;
        api.calls.lock().unwrap().clear();
        let done = rotate_keys(&state_dir, &keys, &api, id, true, false)
            .await
            .unwrap();
        assert!(done.signing && done.resumed);
        assert_eq!(
            *api.calls.lock().unwrap(),
            ["chain", "signer_registered"],
            "no second rotation (the pre-flight reads the chain)"
        );
        assert_eq!(
            keys.rotations(id).unwrap(),
            vec![sent.clone()],
            "the one sent"
        );
        assert!(keys.pending_rotation(id).unwrap().is_none());
        assert_eq!(
            agora_agentkit::govlog::PublicKeyHex::from(&keys.signing(id).unwrap().verifying_key()),
            sent.new_public_key
        );
        assert!(keys.trusted(id).unwrap().is_some());
    }

    /// Never while the agent has a session journal.
    #[tokio::test]
    async fn rotation_refuses_while_a_session_is_in_progress() {
        let root = crate::sealed::TestDir::new("rotate-session");
        let (id, keys, _) = enrolled(&root).await;
        let state_dir = root.join("state");
        let journal = FsStorage::new(&state_dir).journal(id);
        std::fs::write(journal.path(), b"{}\n").unwrap();
        let e = rotate_keys(&state_dir, &keys, &FakeApi::new(), id, true, false)
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("session in progress"), "{e}");
        assert!(keys.next_secrets(id).unwrap().is_none(), "nothing written");
    }

    /// The signing rotation taken and recorded, the encryption key
    /// refused: the retry finishes the encryption step and never signs a
    /// second rotation.
    #[tokio::test]
    async fn a_half_done_rotation_resumes_after_the_signing_step() {
        let root = crate::sealed::TestDir::new("rotate-half");
        let (id, keys, dir) = enrolled(&root).await;
        keys.ring().ensure_encryption_key(id).unwrap();
        let state_dir = root.join("state");
        let api = FakeApi::new();
        *api.public_key.lock().unwrap() = Some(keys.signing(id).unwrap().verifying_key());
        uploaded(&dir, id, &api).await;
        *api.fail.lock().unwrap() = Some("register_encryption_key");
        assert!(
            rotate_keys(&state_dir, &keys, &api, id, true, true)
                .await
                .is_err()
        );
        assert_eq!(keys.rotations(id).unwrap().len(), 1);

        *api.fail.lock().unwrap() = None;
        api.calls.lock().unwrap().clear();
        let done = rotate_keys(&state_dir, &keys, &api, id, true, true)
            .await
            .unwrap();
        assert!(done.signing && done.encryption && done.rewrapped);
        assert_eq!(
            *api.calls.lock().unwrap(),
            [
                "chain",
                "register_encryption_key",
                "state_keys",
                "post_state_key"
            ],
            "no second rotation (the pre-flight's read, the re-wrap's seq)"
        );
        let rotations = keys.rotations(id).unwrap();
        assert_eq!(rotations.len(), 1);
        assert!(keys.pending_rotation(id).unwrap().is_none());
        assert!(
            keys.trusted(id).unwrap().is_some(),
            "ends at the current key"
        );
    }

    /// An enrolled agent rotates only at the head Agora confirms: versions
    /// waiting to upload stop it, and a moved head is a clear refusal.
    #[tokio::test]
    async fn a_rotation_names_the_head_agora_confirms() {
        let root = crate::sealed::TestDir::new("rotate-head");
        let (id, keys, dir) = enrolled(&root).await;
        let state_dir = root.join("state");
        let seal = agora_agentkit::state::storage::AgentSeal::new(&dir, id);
        let api = FakeApi::new();
        *api.public_key.lock().unwrap() = Some(keys.signing(id).unwrap().verifying_key());
        let e = rotate_keys(&state_dir, &keys, &api, id, true, false)
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("pending uploads"), "{e}");
        assert!(keys.rotations(id).unwrap().is_empty());
        let chain = crate::sealed::check::local_entries(&seal).await.unwrap();
        for item in seal.outbox().await.unwrap() {
            seal.mark_sent(item.ord).await.unwrap();
        }
        *api.chain.lock().unwrap() = Ok(chain.clone());
        *api.fail.lock().unwrap() = Some("rotate_key_conflict");
        let e = rotate_keys(&state_dir, &keys, &api, id, true, false)
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("head moved on"), "{e}");
        *api.fail.lock().unwrap() = None;
        rotate_keys(&state_dir, &keys, &api, id, true, false)
            .await
            .unwrap();
        let r = &keys.rotations(id).unwrap()[0];
        assert_eq!(
            r.old_key_last.map(|h| h.entry_hash),
            Some(chain.last().unwrap().entry_hash())
        );
    }

    /// More versions sealed here, never sent.
    async fn saves(dir: &std::path::Path, id: AgentId, keys: &KeyStore, n: usize) {
        use agora_agentkit::enums::StateEntryKind;
        use agora_agentkit::state::codec::KeyframePolicy;
        let seal = AgentSeal::new(dir, id);
        let k = keys.seal_keys(id).unwrap();
        for i in 0..n {
            seal.append_state(
                &k,
                &KeyframePolicy::default(),
                StateEntryKind::Snapshot,
                &check::tests::state(10 + i),
                Utc::now(),
            )
            .await
            .unwrap();
        }
    }

    /// The local main line to the head, verified under the runner's own
    /// recorded keys (`secrets.json`'s key current).
    async fn verifies(dir: &std::path::Path, id: AgentId, keys: &KeyStore) -> usize {
        let seal = AgentSeal::new(dir, id);
        let head = seal.head().await.unwrap().unwrap().head;
        let entries = check::local_entries(&seal).await.unwrap();
        let line = check::main_line(&entries, &head.entry_hash).unwrap();
        let report = verify_chain(id, &line, &keys.trusted(id).unwrap().unwrap());
        assert!(report.intact, "{:?}", report.problems);
        assert_eq!(report.head, Some(head));
        line.len()
    }

    /// An agent that never uploaded: the identity key rotated with Agora
    /// naming no head, its local chain re-signed with the new key (hashes
    /// and head unchanged), the re-wrap kept locally. Under the strict
    /// guard: nothing of its chain is asked or sent.
    #[tokio::test]
    async fn a_local_only_agent_is_rotated_and_its_chain_resigned() {
        let root = crate::sealed::TestDir::new("rotate-local");
        let (id, keys, dir) = check::tests::enrolled_local(&root).await;
        keys.ring().ensure_encryption_key(id).unwrap();
        saves(&dir, id, &keys, 2).await;
        let seal = AgentSeal::new(&dir, id);
        let before = check::local_entries(&seal).await.unwrap();
        let head = seal.head().await.unwrap().unwrap();
        let old = keys.signing(id).unwrap().verifying_key();
        let api = crate::sealed::api::fake::guarded(&root);
        let done = rotate_keys(&root.join("state"), &keys, &api, id, true, true)
            .await
            .unwrap();
        assert!(done.signing && done.encryption && done.rewrapped && done.wrap_kept_local);
        assert_eq!(done.resigned, before.len());
        assert_eq!(
            *api.calls.lock().unwrap(),
            ["rotate_key", "register_encryption_key"],
            "the identity only"
        );
        let r = &keys.rotations(id).unwrap()[0];
        assert_eq!(r.old_key_last, None);
        let new = keys.signing(id).unwrap().verifying_key();
        assert_ne!(new, old);
        let after = check::local_entries(&seal).await.unwrap();
        assert_eq!(
            after.iter().map(|e| e.entry_hash()).collect::<Vec<_>>(),
            before.iter().map(|e| e.entry_hash()).collect::<Vec<_>>(),
            "the same entries"
        );
        assert!(after.iter().all(|e| e.verify(&new)));
        assert_eq!(
            seal.head().await.unwrap().unwrap(),
            head,
            "the head unchanged"
        );
        assert_eq!(verifies(&dir, id, &keys).await, before.len());
        // The re-wrap: kept, never posted, named by state_key.json
        let wrap = keys.wrap_record(id).unwrap().unwrap();
        assert_eq!(wrap.wrap_seq, 1);
        assert!(wrap.verify(&new));
        assert_eq!(keys.state_key_file(id).unwrap().unwrap().wrap_seq, 1);
        assert!(api.keys.lock().unwrap().is_empty());
        let enc = keys.encryption(id).unwrap();
        assert_eq!(
            StateKey::unwrap(&wrap, &enc).unwrap().expose(),
            keys.state_key_file(id)
                .unwrap()
                .unwrap()
                .state_key()
                .unwrap()
                .expose()
        );
    }

    /// Cut at every step of a local-only rotation: `secrets.json` is the
    /// old key until the end (the runner leaves the agent out while
    /// `secrets.next.json` is there), and the rerun finishes, with one
    /// rotation sent and a chain that verifies.
    #[tokio::test]
    async fn a_cut_local_only_rotation_finishes_on_rerun() {
        for point in [
            "rotate:recorded",
            "rotate:wrap_kept",
            "rotate:resigned_one",
            "rotate:resigned",
        ] {
            let root = crate::sealed::TestDir::new("rotate-local-cut");
            let (id, keys, dir) = check::tests::enrolled_local(&root).await;
            keys.ring().ensure_encryption_key(id).unwrap();
            saves(&dir, id, &keys, 2).await;
            let n = check::local_entries(&AgentSeal::new(&dir, id))
                .await
                .unwrap()
                .len();
            let old = keys.signing(id).unwrap().to_bytes();
            let api = crate::sealed::api::fake::guarded(&root);
            crate::sealed::crash::at(point);
            let e = rotate_keys(&root.join("state"), &keys, &api, id, true, true)
                .await
                .unwrap_err();
            assert!(
                format!("{e:#}").contains("simulated crash"),
                "{point}: {e:#}"
            );
            assert_eq!(keys.signing(id).unwrap().to_bytes(), old, "{point}");
            assert!(
                keys.next_secrets(id).unwrap().is_some(),
                "{point}: left out"
            );
            if point == "rotate:resigned_one" {
                let next = keys.next_secrets(id).unwrap().unwrap().signing().unwrap();
                let entries = check::local_entries(&AgentSeal::new(&dir, id))
                    .await
                    .unwrap();
                let resigned = entries
                    .iter()
                    .filter(|e| e.verify(&next.verifying_key()))
                    .count();
                assert!(resigned > 0 && resigned < n, "half done: {resigned} of {n}");
            }
            let done = rotate_keys(&root.join("state"), &keys, &api, id, true, true)
                .await
                .unwrap();
            assert!(done.resumed && done.rewrapped, "{point}");
            assert_eq!(keys.rotations(id).unwrap().len(), 1, "{point}");
            assert_eq!(
                api.calls
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|c| *c == "rotate_key")
                    .count(),
                1,
                "{point}: one rotation sent"
            );
            assert!(keys.next_secrets(id).unwrap().is_none());
            assert_eq!(verifies(&dir, id, &keys).await, n, "{point}");
            assert_eq!(
                keys.state_key_file(id).unwrap().unwrap().wrap_seq,
                1,
                "{point}"
            );
        }
    }

    /// An entry signed by none of the agent's keys is never signed over:
    /// refused, the agent left out.
    #[tokio::test]
    async fn a_forged_entry_is_never_resigned() {
        let root = crate::sealed::TestDir::new("rotate-forged");
        let (id, keys, dir) = check::tests::enrolled_local(&root).await;
        saves(&dir, id, &keys, 1).await;
        let seal = AgentSeal::new(&dir, id);
        let item = seal.outbox().await.unwrap().pop().unwrap();
        let path = dir
            .join("sealed/outbox")
            .join(format!("{:020}.json", item.ord));
        let mut forged = item.clone();
        forged.entry = item
            .entry
            .entry
            .clone()
            .sign(&agora_agentkit::crypto::generate_keypair().0);
        std::fs::write(&path, serde_json::to_vec(&forged).unwrap()).unwrap();
        let api = crate::sealed::api::fake::guarded(&root);
        let e = rotate_keys(&root.join("state"), &keys, &api, id, true, false)
            .await
            .unwrap_err();
        assert!(
            format!("{e:#}").contains("signed by none of the agent's keys"),
            "{e:#}"
        );
        assert!(
            keys.next_secrets(id).unwrap().is_none(),
            "the pre-flight: nothing written"
        );
        assert!(keys.past_keys(id).unwrap().signing.is_empty());
        assert!(api.calls.lock().unwrap().is_empty(), "nothing sent");
        assert_eq!(
            std::fs::read(&path).unwrap(),
            serde_json::to_vec(&forged).unwrap(),
            "untouched"
        );
    }

    /// An uploader cut after its re-wrap was kept, and after it was posted
    /// but before state_key.json: the rerun posts the same bytes (or finds
    /// them on Agora when the server refuses a second post), never a
    /// second wrap at the same (key_gen, wrap_seq).
    #[tokio::test]
    async fn a_cut_rewrap_posts_the_same_bytes() {
        for point in ["rotate:wrap_kept", "rotate:wrap_posted"] {
            let root = crate::sealed::TestDir::new("rotate-rewrap-cut");
            let (id, keys, dir) = enrolled(&root).await;
            keys.ring().ensure_encryption_key(id).unwrap();
            let api = FakeApi::new();
            *api.public_key.lock().unwrap() = Some(keys.signing(id).unwrap().verifying_key());
            uploaded(&dir, id, &api).await;
            crate::sealed::crash::at(point);
            assert!(
                rotate_keys(&root.join("state"), &keys, &api, id, false, true)
                    .await
                    .is_err()
            );
            let kept = keys.wrap_record(id).unwrap().unwrap();
            assert_eq!(keys.state_key_file(id).unwrap().unwrap().wrap_seq, 0);
            // The real server refuses a second post of a record it holds
            if point == "rotate:wrap_posted" {
                *api.fail.lock().unwrap() = Some("post_state_key");
            }
            let done = rotate_keys(&root.join("state"), &keys, &api, id, false, true)
                .await
                .unwrap();
            assert!(done.rewrapped && !done.wrap_kept_local, "{point}");
            let posted = api.keys.lock().unwrap().clone();
            assert!(
                !posted.is_empty() && posted.iter().all(|r| *r == kept),
                "{point}"
            );
            assert_eq!(keys.state_key_file(id).unwrap().unwrap().wrap_seq, 1);
            assert_eq!(keys.wrap_record(id).unwrap().unwrap(), kept);
        }
    }

    /// Stopped after uploading: Agora holds its chain to the last version
    /// it acknowledged. The rotation names that head; what was sealed here
    /// since is re-signed, what Agora holds is not; the re-wrap is posted.
    /// Through the strict guard: reads and key maintenance only.
    #[tokio::test]
    async fn a_stopped_agent_rotates_at_agoras_head() {
        let root = crate::sealed::TestDir::new("rotate-stopped");
        let (id, keys, dir) = enrolled(&root).await;
        keys.ring().ensure_encryption_key(id).unwrap();
        let api = crate::sealed::api::fake::guarded(&root);
        *api.public_key.lock().unwrap() = Some(keys.signing(id).unwrap().verifying_key());
        uploaded(&dir, id, &api).await;
        let on_agora = api.chain.lock().unwrap().clone().unwrap();
        crate::sealed::stop::stop(&root.join("state"), &keys, id, "It asked")
            .await
            .unwrap();
        saves(&dir, id, &keys, 2).await;
        let old = keys.signing(id).unwrap().verifying_key();
        let done = rotate_keys(&root.join("state"), &keys, &api, id, true, true)
            .await
            .unwrap();
        assert_eq!(done.resigned, 2);
        assert!(done.rewrapped && !done.wrap_kept_local);
        let r = &keys.rotations(id).unwrap()[0];
        assert_eq!(
            r.old_key_last.map(|h| h.entry_hash),
            Some(on_agora.last().unwrap().entry_hash())
        );
        let seal = AgentSeal::new(&dir, id);
        let sent = seal.sent().await.unwrap();
        assert_eq!(
            sent.iter().map(|i| i.entry.clone()).collect::<Vec<_>>(),
            on_agora,
            "Agora's copy untouched"
        );
        assert!(sent.iter().all(|i| i.entry.verify(&old)));
        let new = keys.signing(id).unwrap().verifying_key();
        assert!(
            seal.outbox()
                .await
                .unwrap()
                .iter()
                .all(|i| i.entry.verify(&new))
        );
        verifies(&dir, id, &keys).await;
        assert!(
            api.calls
                .lock()
                .unwrap()
                .iter()
                .any(|c| c == "post_state_key")
        );
    }

    /// An Agora that takes blobs and entries as the real one checks them:
    /// each entry signed by the agent's current key and linked to the one
    /// before. What it took, in order.
    pub(crate) async fn accepting(
        key: agora_agentkit::crypto::VerifyingKey,
    ) -> (
        url::Url,
        std::sync::Arc<std::sync::Mutex<Vec<agora_agentkit::state::SignedStateEntry>>>,
    ) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        type Body = agora_agentkit::requests::SignedRequest<
            agora_agentkit::requests::PostStateEntryPayload,
        >;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let took = std::sync::Arc::new(std::sync::Mutex::new(Vec::<
            agora_agentkit::state::SignedStateEntry,
        >::new()));
        let chain = took.clone();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                let chain = chain.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 8192];
                    let end = loop {
                        let n = sock.read(&mut chunk).await.unwrap_or(0);
                        if n == 0 {
                            return;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                            break i + 4;
                        }
                    };
                    let head = String::from_utf8_lossy(&buf[..end]).to_string();
                    let len: usize = head
                        .lines()
                        .find_map(|l| {
                            let (k, v) = l.split_once(':')?;
                            k.eq_ignore_ascii_case("content-length")
                                .then(|| v.trim().parse().ok())?
                        })
                        .unwrap_or(0);
                    while buf.len() < end + len {
                        let n = sock.read(&mut chunk).await.unwrap_or(0);
                        if n == 0 {
                            break;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                    }
                    let body = &buf[end..];
                    let line = head.lines().next().unwrap_or_default().to_string();
                    let (status, reply) =
                        if line.contains(" /agora/api/state/blobs/") && line.starts_with("PUT ") {
                            let sha = line
                                .split_whitespace()
                                .nth(1)
                                .unwrap()
                                .rsplit('/')
                                .next()
                                .unwrap()
                                .to_string();
                            (
                                "200 OK",
                                serde_json::json!({
                                    "id": uuid::Uuid::new_v4(),
                                    "sha256": sha,
                                    "size_bytes": body.len(),
                                }),
                            )
                        } else if line.starts_with("POST /agora/api/state/entries") {
                            let entry = serde_json::from_slice::<Body>(body).unwrap().payload.entry;
                            let mut chain = chain.lock().unwrap();
                            let linked = match chain.last() {
                                None => entry.entry.seq == 0 && entry.entry.prev_hash.is_none(),
                                Some(p) => {
                                    entry.entry.seq == p.entry.seq + 1
                                        && entry.entry.prev_hash == Some(p.entry_hash())
                                }
                            };
                            if entry.verify(&key) && linked {
                                chain.push(entry.clone());
                                (
                                    "200 OK",
                                    serde_json::json!({
                                        "id": uuid::Uuid::new_v4(),
                                        "head": {
                                            "entry_hash": entry.entry_hash(),
                                            "seq": entry.entry.seq,
                                            "updated_at": Utc::now(),
                                        },
                                    }),
                                )
                            } else {
                                (
                                    "422 Unprocessable Entity",
                                    serde_json::json!({ "error": "bad signature or link" }),
                                )
                            }
                        } else {
                            ("404 Not Found", serde_json::json!({ "error": line }))
                        };
                    let reply = reply.to_string();
                    let resp = format!(
                        "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: \
                         {}\r\nconnection: close\r\n\r\n{reply}",
                        reply.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });
        (format!("http://{addr}/").parse().unwrap(), took)
    }

    /// Rotated while local only, then opted in: the whole re-signed chain
    /// uploads, each entry checked against the current key as Agora does.
    #[tokio::test]
    async fn a_resigned_chain_uploads_in_full_after_an_opt_in() {
        use agora_agentkit::state::outbox::{AgoraOutbox, Drained};
        let root = crate::sealed::TestDir::new("rotate-upload");
        let (id, keys, dir) = check::tests::enrolled_local(&root).await;
        saves(&dir, id, &keys, 3).await;
        let api = crate::sealed::api::fake::guarded(&root);
        rotate_keys(&root.join("state"), &keys, &api, id, true, false)
            .await
            .unwrap();
        let local = check::local_entries(&AgentSeal::new(&dir, id))
            .await
            .unwrap();
        // Before the rotation, Agora would have refused the first entry
        let (url, took) = accepting(keys.signing(id).unwrap().verifying_key()).await;
        crate::sealed::uploading::set(
            &dir,
            &keys,
            id,
            crate::sealed::uploading::UploadingBy::OptIn,
        )
        .unwrap();
        let enrolment = std::sync::Arc::new(crate::sealed::SealedEnrolment::new(
            root.join("state"),
            root.join("secrets"),
        ));
        let outbox = AgoraOutbox::new(
            agora_agentkit::client::Client::new(url).unwrap(),
            root.join("state"),
            enrolment,
        );
        let Drained::Ran(report) = outbox.drain(id).await.unwrap() else {
            panic!("not drained")
        };
        assert_eq!(report.uploaded, local.len());
        assert_eq!(*took.lock().unwrap(), local, "every entry, in order");
        let report = verify_chain(
            id,
            &took.lock().unwrap(),
            &keys.trusted(id).unwrap().unwrap(),
        );
        assert!(report.intact, "{:?}", report.problems);
    }

    /// Every file under `dir`, with its bytes.
    fn tree(dir: &std::path::Path) -> std::collections::BTreeMap<std::path::PathBuf, Vec<u8>> {
        let mut out = std::collections::BTreeMap::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    stack.push(p);
                } else if p
                    .file_name()
                    .is_some_and(|n| n != "lock" && !n.to_string_lossy().ends_with(".lock"))
                {
                    out.insert(p.clone(), std::fs::read(&p).unwrap());
                }
            }
        }
        out
    }

    /// A stopped agent whose `sent/` is partial or empty (as after a past
    /// restore from Agora): Agora's entries complete the line, and the
    /// rotation finishes.
    #[tokio::test]
    async fn a_stopped_agent_with_a_partial_sent_dir_rotates() {
        for keep in [1usize, 0] {
            let root = crate::sealed::TestDir::new("rotate-stopped-partial");
            let (id, keys, dir) = enrolled(&root).await;
            let api = crate::sealed::api::fake::guarded(&root);
            *api.public_key.lock().unwrap() = Some(keys.signing(id).unwrap().verifying_key());
            uploaded(&dir, id, &api).await;
            crate::sealed::stop::stop(&root.join("state"), &keys, id, "It asked")
                .await
                .unwrap();
            saves(&dir, id, &keys, 2).await;
            let mut sent: Vec<_> = std::fs::read_dir(dir.join("sealed/sent"))
                .unwrap()
                .map(|e| e.unwrap().path())
                .collect();
            sent.sort();
            for p in sent.iter().skip(keep) {
                std::fs::remove_file(p).unwrap();
            }
            let done = rotate_keys(&root.join("state"), &keys, &api, id, true, false)
                .await
                .unwrap_or_else(|e| panic!("sent kept {keep}: {e:#}"));
            assert_eq!(done.resigned, 2, "sent kept {keep}");
            assert!(keys.next_secrets(id).unwrap().is_none());
        }
    }

    /// A missing link in a chain kept here (a never-uploaded agent), and
    /// an empty `sent/` with nothing on Agora to complete it: refused by the
    /// pre-flight, with no write and no server call.
    #[tokio::test]
    async fn a_broken_local_line_is_refused_before_anything() {
        let root = crate::sealed::TestDir::new("rotate-broken-line");
        let (id, keys, dir) = check::tests::enrolled_local(&root).await;
        saves(&dir, id, &keys, 2).await;
        let seal = AgentSeal::new(&dir, id);
        let item = &seal.outbox().await.unwrap()[1];
        let path = dir
            .join("sealed/outbox")
            .join(format!("{:020}.json", item.ord));
        std::fs::rename(&path, root.join("moved")).unwrap();
        let before = (tree(&root.join("state")), tree(&root.join("secrets")));
        let api = crate::sealed::api::fake::guarded(&root);
        let e = format!(
            "{:#}",
            rotate_keys(&root.join("state"), &keys, &api, id, true, true)
                .await
                .unwrap_err()
        );
        assert!(
            e.contains("missing link") && e.contains("from a backup"),
            "{e}"
        );
        assert!(!e.contains("secrets.next.json"), "{e}");
        assert_eq!(
            (tree(&root.join("state")), tree(&root.join("secrets"))),
            before,
            "nothing written"
        );
        assert!(api.calls.lock().unwrap().is_empty(), "nothing sent");
    }

    /// Already stuck mid-rotation (the server took it; then a file went
    /// missing): `--status` says where it is and what blocks it, and once
    /// the file is back the rerun finishes. The new key is never lost.
    #[tokio::test]
    async fn a_stuck_rotation_is_shown_and_finishes_once_fixed() {
        let root = crate::sealed::TestDir::new("rotate-stuck");
        let (id, keys, dir) = check::tests::enrolled_local(&root).await;
        saves(&dir, id, &keys, 2).await;
        let api = crate::sealed::api::fake::guarded(&root);
        let state_dir = root.join("state");
        assert!(
            !status(&state_dir, &keys, &api, id)
                .await
                .unwrap()
                .unfinished
        );
        crate::sealed::crash::at("rotate:recorded");
        assert!(
            rotate_keys(&state_dir, &keys, &api, id, true, false)
                .await
                .is_err()
        );
        let seal = AgentSeal::new(&dir, id);
        let item = &seal.outbox().await.unwrap()[1];
        let path = dir
            .join("sealed/outbox")
            .join(format!("{:020}.json", item.ord));
        std::fs::rename(&path, root.join("moved")).unwrap();
        let st = status(&state_dir, &keys, &api, id).await.unwrap();
        assert!(st.unfinished);
        assert_eq!(st.signing, "recorded (Agora took it)");
        let why = st.blocked.clone().expect("blocked");
        assert!(why.contains("missing link"), "{why}");
        assert!(st.to_string().contains("keep it"), "{st}");
        assert!(
            rotate_keys(&state_dir, &keys, &api, id, true, false)
                .await
                .is_err()
        );
        let next = keys.next_secrets(id).unwrap().unwrap().signing().unwrap();
        // Fixed: the rerun finishes, with the key Agora took
        std::fs::rename(root.join("moved"), &path).unwrap();
        let st = status(&state_dir, &keys, &api, id).await.unwrap();
        assert!(st.blocked.is_none(), "{st}");
        assert!(st.to_resign > 0);
        rotate_keys(&state_dir, &keys, &api, id, true, false)
            .await
            .unwrap();
        assert_eq!(keys.signing(id).unwrap().to_bytes(), next.to_bytes());
        verifies(&dir, id, &keys).await;
    }

    /// An uploader's unsent items (a prepared `.tmp`) are re-signed too: a
    /// later commit of an old-key item would not verify.
    #[tokio::test]
    async fn an_uploaders_unsent_items_are_resigned() {
        let root = crate::sealed::TestDir::new("rotate-uploader-tmp");
        let (id, keys, dir) = enrolled(&root).await;
        let api = FakeApi::new();
        *api.public_key.lock().unwrap() = Some(keys.signing(id).unwrap().verifying_key());
        uploaded(&dir, id, &api).await;
        let seal = AgentSeal::new(&dir, id);
        let sent = seal.sent().await.unwrap();
        // A prepared item Agora doesn't hold, signed by the old key
        let mut tmp = sent.last().unwrap().clone();
        tmp.ord = 99;
        tmp.entry.entry.seq += 100;
        let path = dir.join("sealed/outbox/00000000000000000099.tmp");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        tmp.entry = tmp.entry.entry.clone().sign(&keys.signing(id).unwrap());
        std::fs::write(&path, serde_json::to_vec(&tmp).unwrap()).unwrap();
        let done = rotate_keys(&root.join("state"), &keys, &api, id, true, false)
            .await
            .unwrap();
        assert_eq!(done.resigned, 1);
        let back: OutboxItem = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert!(
            back.entry
                .verify(&keys.signing(id).unwrap().verifying_key())
        );
        assert!(seal.sent().await.unwrap().iter().all(|i| sent.contains(i)));
    }

    /// Agora already holds a record at the next seq (state_key.json behind
    /// it, the wrap file gone): the re-wrap takes the seq after Agora's.
    #[tokio::test]
    async fn the_rewrap_seq_follows_agora() {
        let root = crate::sealed::TestDir::new("rotate-rewrap-seq");
        let (id, keys, dir) = enrolled(&root).await;
        let enc = keys.ring().ensure_encryption_key(id).unwrap();
        let api = FakeApi::new();
        let signing = keys.signing(id).unwrap();
        *api.public_key.lock().unwrap() = Some(signing.verifying_key());
        uploaded(&dir, id, &api).await;
        let key = keys
            .state_key_file(id)
            .unwrap()
            .unwrap()
            .state_key()
            .unwrap();
        let other = agora_agentkit::envelope::EncryptionPublicKey::from(&enc);
        api.keys
            .lock()
            .unwrap()
            .push(key.wrap(id, 0, 1, &other, &signing).unwrap());
        rotate_keys(&root.join("state"), &keys, &api, id, false, true)
            .await
            .unwrap();
        assert_eq!(keys.state_key_file(id).unwrap().unwrap().wrap_seq, 2);
    }
}
