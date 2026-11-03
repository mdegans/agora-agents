//! An agent's last sealed version, opened: for `sealed-restore` (which
//! writes it back) and `sealed-accept` (which compares it with the state it
//! accepts). Both take it the way the start check would: the main line to
//! `head.json`'s head, from the entries here and, when it runs through
//! them, Agora's, verified with the runner's own trust
//! ([`KeyStore::trusted`]), never on the server's word.
//!
//! A blob that isn't here (agentkit keeps only the current keyframe's and
//! the outbox's, `prune_blobs`) is fetched from Agora and checked against
//! its entry's `BlobRef` before it is opened (agentkit's
//! `open_entry_blob`); nothing fetched is written to disk. The plaintext
//! is never logged: errors carry hashes, kinds and counts only.

use agora_agentkit::crypto::SigningKey;
use agora_agentkit::enums::StateEntryKind;
use agora_agentkit::govlog::Sha256Hex;
use agora_agentkit::ids::AgentId;
use agora_agentkit::state::chain::BlobEncoding;
use agora_agentkit::state::codec;
use agora_agentkit::state::crypto::open_entry_blob;
use agora_agentkit::state::storage::{AgentSeal, HeadFile, SealKeys};
use agora_agentkit::state::verify::verify_chain;
use agora_agentkit::state::{SignedStateEntry, StateEntry};
use anyhow::Context;

use super::KeyStore;
use super::api::SealedApi;
use super::check::{local_entries, main_line};

/// The last sealed version, opened.
pub struct LastSealed {
    /// `head.json` as found.
    pub head: HeadFile,
    /// The main line to the head, oldest first, verified.
    pub line: Vec<SignedStateEntry>,
    /// The newest snapshot on it: the head itself, unless the head is
    /// another kind.
    pub snapshot: SignedStateEntry,
    /// Its state. Never logged.
    pub state: serde_json::Value,
}

/// A sealed blob: from `sealed/blobs/` if it is still here, else from
/// Agora. Unchecked: [`open_entry_blob`] checks it against the entry.
async fn blob(
    seal: &AgentSeal,
    api: &dyn SealedApi,
    id: AgentId,
    signing: &SigningKey,
    sha: &Sha256Hex,
) -> anyhow::Result<Vec<u8>> {
    match seal.read_blob(sha).await {
        Ok(bytes) => Ok(bytes),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => api
            .blob(id, signing, sha)
            .await
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("fetching blob {sha} from Agora")),
        Err(e) => Err(e).with_context(|| format!("reading blob {sha}")),
    }
}

/// One entry's compressed-then-decompressed plaintext, against `dict`.
/// The decompression (brotli, up to a whole state) runs on a blocking
/// thread.
async fn plaintext(
    seal: &AgentSeal,
    api: &dyn SealedApi,
    keys: &SealKeys,
    entry: &StateEntry,
    dict: Option<zeroize::Zeroizing<Vec<u8>>>,
) -> anyhow::Result<zeroize::Zeroizing<Vec<u8>>> {
    let sha = entry
        .blob
        .as_ref()
        .map(|b| b.sha256)
        .context("the entry has no blob")?;
    anyhow::ensure!(
        entry.key_gen == keys.key_gen,
        "entry seq {} is sealed under state key generation {}, and this runner holds {}",
        entry.seq,
        entry.key_gen,
        keys.key_gen
    );
    let sealed = blob(seal, api, entry.agent_id, &keys.signing, &sha).await?;
    let compressed = open_entry_blob(&keys.state_key, entry, &sealed)
        .with_context(|| format!("opening blob {sha} (entry seq {})", entry.seq))?;
    let seq = entry.seq;
    tokio::task::spawn_blocking(move || {
        codec::decompress(&compressed, dict.as_deref().map(|d| &d[..]))
            .with_context(|| format!("decompressing blob {sha} (entry seq {seq})"))
    })
    .await
    .context("the decompression task failed")?
}

/// Open `entry`, a snapshot on `line`: its keyframe first when it is a
/// delta.
pub async fn open(
    seal: &AgentSeal,
    api: &dyn SealedApi,
    keys: &SealKeys,
    line: &[SignedStateEntry],
    entry: &StateEntry,
) -> anyhow::Result<serde_json::Value> {
    let encoding = entry
        .blob
        .as_ref()
        .map(|b| b.encoding)
        .context("the entry has no blob")?;
    let dict = match encoding {
        BlobEncoding::Keyframe => None,
        BlobEncoding::Delta { base } => {
            let keyframe = line
                .iter()
                .map(|e| &e.entry)
                .find(|e| e.entry_hash() == base)
                .with_context(|| format!("the delta's keyframe {base} is not on the main line"))?;
            Some(plaintext(seal, api, keys, keyframe, None).await?)
        }
    };
    let bytes = plaintext(seal, api, keys, entry, dict).await?;
    tokio::task::spawn_blocking(move || {
        let mut envelope: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| anyhow::anyhow!("the sealed plaintext isn't JSON ({:?})", e.classify()))?;
        envelope
            .get_mut("state")
            .map(serde_json::Value::take)
            .context("the sealed plaintext holds no state")
    })
    .await
    .context("the parsing task failed")?
}

/// The main line to `head`, oldest first: from the entries here, and
/// through Agora's when it runs through them (ours winning), verified
/// with the runner's own trust.
pub async fn verified_line(
    seal: &AgentSeal,
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
    head: &HeadFile,
) -> anyhow::Result<Vec<SignedStateEntry>> {
    let entries = local_entries(seal).await.map_err(anyhow::Error::msg)?;
    let line = match main_line(&entries, &head.head.entry_hash) {
        Some(line) => line,
        // Through Agora's entries (after a rebase or a restore)
        None => {
            let signing = keys.signing(id).context("no signing key for the agent")?;
            let chain = api
                .chain(id, &signing)
                .await
                .map_err(anyhow::Error::msg)
                .context("reading the agent's chain on Agora")?;
            let all = super::check::merge_server(&entries, &chain).map_err(|hash| {
                anyhow::anyhow!("Agora's copy of entry {hash} differs from the one held here")
            })?;
            main_line(&all, &head.head.entry_hash).with_context(|| {
                format!(
                    "the main line to head {} has a missing link",
                    head.head.entry_hash
                )
            })?
        }
    };
    let trusted = keys
        .trusted(id)?
        .context("no signing key to verify the sealed chain with")?;
    let report = verify_chain(id, &line, &trusted);
    anyhow::ensure!(
        report.intact && report.head == Some(head.head),
        "the sealed chain did not verify: {:?}",
        report.problems
    );
    Ok(line)
}

/// The main line to `head.json`'s head, verified, and its newest snapshot
/// opened. `Err` says why not (no head, a link missing, a chain that
/// doesn't verify, a blob that can't be had), in hashes and counts.
pub async fn last_sealed(
    seal: &AgentSeal,
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
) -> anyhow::Result<LastSealed> {
    let head = seal
        .head()
        .await
        .context("reading sealed/head.json")?
        .context("sealed/head.json is missing")?;
    let k = keys
        .seal_keys(id)
        .context("no sealing keys for the agent")?;
    let line = verified_line(seal, keys, api, id, &head).await?;
    let snapshot = line
        .iter()
        .rev()
        .find(|e| e.entry.kind == StateEntryKind::Snapshot)
        .cloned()
        .context("the main line holds no snapshot")?;
    let state = open(seal, api, &k, &line, &snapshot.entry).await?;
    Ok(LastSealed {
        head,
        line,
        snapshot,
        state,
    })
}
