//! An agent's last sealed version, opened: for `sealed-restore` (which
//! writes it back) and `sealed-accept` (which compares it with the state it
//! accepts). Both take it the way the start check would: the main line to
//! `head.json`'s head, from the entries here and, when it runs through
//! them, Agora's, verified with the runner's own trust
//! ([`KeyStore::trusted`]), never on the server's word.
//!
//! A blob that isn't here (agentkit keeps only the current keyframe's and
//! the outbox's, `remove_unnamed_blobs`) is fetched from Agora and checked against
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
use super::api::{BlobError, SealedApi};
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

/// A sealed blob: from the first of `seals` whose `blobs/` holds it (the
/// agent's `sealed/`, then any `sealed/lost/<utc>/` copies), else from
/// Agora. Unchecked: [`open_entry_blob`] checks it against the entry.
/// Nothing fetched is written anywhere.
///
/// A fetch is bounded ([`FETCH_TIMEOUT`] a try) and retried only when
/// trying again could help ([`BlobError::retry`]: a 5xx, a timeout, a
/// connection; never a 404 or another 4xx). A blob that couldn't be had
/// is remembered in `failed` and not asked for again until
/// [`FAILED_FOR`] has passed.
async fn blob(
    seals: &[&AgentSeal],
    api: &dyn SealedApi,
    id: AgentId,
    signing: &SigningKey,
    sha: &Sha256Hex,
    failed: &mut FailedBlobs,
) -> anyhow::Result<Vec<u8>> {
    for seal in seals {
        match seal.read_blob(sha).await {
            Ok(bytes) => return Ok(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e).with_context(|| format!("reading blob {sha}")),
        }
    }
    if failed.unreachable() {
        anyhow::bail!("{UNREACHABLE} (blob {sha} not asked for)");
    }
    if let Some(why) = failed.recent(sha) {
        anyhow::bail!(
            "blob {sha} couldn't be fetched from Agora a moment ago ({why}); not asked again yet"
        );
    }
    // Marked failed before each try and cleared on an answer: a fetch the
    // caller's deadline cancels mid-try is remembered all the same
    let mut last = BlobError::transient("not tried");
    for wait in FETCH_RETRIES {
        tokio::time::sleep(wait).await;
        failed.trying(*sha);
        last = match tokio::time::timeout(FETCH_TIMEOUT, api.blob(id, signing, sha)).await {
            Ok(Ok(bytes)) => {
                failed.answered(sha, None);
                return Ok(bytes);
            }
            Ok(Err(e)) => e,
            Err(_) => BlobError::transient(format!("no answer in {FETCH_TIMEOUT:?}")),
        };
        if !last.retry {
            // Agora answered: reachable, this blob just isn't to be had
            failed.answered(sha, Some(last.message.clone()));
            break;
        }
        failed.record(*sha, last.message.clone());
        if failed.unreachable() {
            break;
        }
    }
    Err(anyhow::Error::new(last)).with_context(|| format!("fetching blob {sha} from Agora"))
}

/// The waits before each try of a blob fetch: three tries in all, for a
/// failure that may pass.
const FETCH_RETRIES: [std::time::Duration; 3] = [
    std::time::Duration::ZERO,
    std::time::Duration::from_millis(200),
    std::time::Duration::from_millis(800),
];

/// The longest one try of a fetch from Agora may take (agentkit's client
/// sets no timeout of its own). A blob's three tries and their waits come
/// to 25 s, inside the history's 30 s deadline.
pub const FETCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(8);

/// How long a fetch that failed is remembered and not tried again.
pub const FAILED_FOR: std::time::Duration = std::time::Duration::from_secs(5 * 60);

/// Fetches from Agora that failed lately, by blob hash, and whether Agora
/// is answering at all. A blob that failed isn't tried again until
/// [`FAILED_FOR`] has passed; after [`BREAKER_AFTER`] tries in a row with
/// no answer (timeouts, 5xx, connections), Agora is taken as unreachable
/// for [`FAILED_FOR`] and nothing more is fetched. Hold one for a session:
/// one outage then costs a session one bounded wait.
#[derive(Debug, Default, Clone)]
pub struct FailedBlobs {
    failed: std::collections::HashMap<Sha256Hex, (tokio::time::Instant, String)>,
    /// Tries in a row without an answer, and when the last began.
    unanswered: u32,
    since: Option<tokio::time::Instant>,
}

/// Tries in a row without an answer before Agora is taken as unreachable.
pub const BREAKER_AFTER: u32 = 3;

/// What a version not fetched while Agora is taken as unreachable says.
pub const UNREACHABLE: &str = "Agora couldn't be reached just now";

impl FailedBlobs {
    /// Why `sha` failed, if it did within [`FAILED_FOR`].
    fn recent(&self, sha: &Sha256Hex) -> Option<&str> {
        self.failed
            .get(sha)
            .filter(|(at, _)| at.elapsed() < FAILED_FOR)
            .map(|(_, why)| why.as_str())
    }

    /// Whether Agora is taken as unreachable just now.
    pub fn unreachable(&self) -> bool {
        self.unanswered >= BREAKER_AFTER && self.since.is_some_and(|t| t.elapsed() < FAILED_FOR)
    }

    /// A try begins: counted as unanswered, and the blob as failed, until
    /// [`Self::answered`] says otherwise.
    fn trying(&mut self, sha: Sha256Hex) {
        let now = tokio::time::Instant::now();
        // Past the breaker's window, the count starts over
        if self.since.is_some_and(|t| t.elapsed() >= FAILED_FOR) {
            self.unanswered = 0;
        }
        self.unanswered += 1;
        self.since = Some(now);
        self.failed
            .entry(sha)
            .or_insert_with(|| (now, "no answer: the fetch was cut off".into()));
    }

    /// Agora answered: with the blob (`None`), or with a refusal not worth
    /// retrying, which is remembered.
    fn answered(&mut self, sha: &Sha256Hex, refused: Option<String>) {
        self.unanswered = 0;
        match refused {
            None => {
                self.failed.remove(sha);
            }
            Some(why) => self.record(*sha, why),
        }
    }

    fn record(&mut self, sha: Sha256Hex, why: String) {
        self.failed.insert(sha, (tokio::time::Instant::now(), why));
    }

    /// How many are remembered.
    pub fn count(&self) -> usize {
        self.failed.len()
    }
}

/// The agent's chain as Agora serves it ([`super::api::chain`]), asked
/// once a session: a chain that came back is kept for the session (what
/// it adds since is the runner's own, held here too), and one that didn't
/// isn't asked for again until [`FAILED_FOR`] has passed. Each ask is
/// bounded by [`FETCH_TIMEOUT`].
#[derive(Debug, Default, Clone)]
pub struct ChainCache(Option<(tokio::time::Instant, Result<Vec<SignedStateEntry>, String>)>);

impl ChainCache {
    pub async fn get(
        &mut self,
        api: &dyn SealedApi,
        id: AgentId,
        signing: &SigningKey,
    ) -> Result<Vec<SignedStateEntry>, String> {
        match &self.0 {
            Some((_, Ok(chain))) => return Ok(chain.clone()),
            Some((at, Err(e))) if at.elapsed() < FAILED_FOR => return Err(e.clone()),
            _ => {}
        }
        let got =
            match tokio::time::timeout(FETCH_TIMEOUT, super::api::chain(api, id, signing)).await {
                Ok(got) => got,
                Err(_) => Err(format!("no answer in {FETCH_TIMEOUT:?}")),
            };
        self.0 = Some((tokio::time::Instant::now(), got.clone()));
        got
    }

    /// Whether a chain is held.
    pub fn held(&self) -> bool {
        matches!(self.0, Some((_, Ok(_))))
    }
}

/// A keyframe's plaintext, the dictionary its deltas open against.
type Dict = std::sync::Arc<zeroize::Zeroizing<Vec<u8>>>;

/// One entry's compressed-then-decompressed plaintext, against `dict`.
/// The decompression (brotli, up to a whole state) runs on a blocking
/// thread.
async fn plaintext(
    seals: &[&AgentSeal],
    api: &dyn SealedApi,
    keys: &SealKeys,
    entry: &StateEntry,
    dict: Option<Dict>,
    failed: &mut FailedBlobs,
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
    let sealed = blob(seals, api, entry.agent_id, &keys.signing, &sha, failed).await?;
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
    Opener::new(vec![seal], api, keys).open(line, entry).await
}

/// Opens one agent's sealed versions, one after another: blobs looked for
/// in each of its seals in turn, then on Agora, and the last keyframe
/// opened kept for the deltas after it (consecutive versions mostly share
/// one). Read-only: nothing is written, fetched blobs included.
pub struct Opener<'a> {
    seals: Vec<&'a AgentSeal>,
    api: &'a dyn SealedApi,
    keys: &'a SealKeys,
    keyframe: Option<(Sha256Hex, Dict)>,
    /// Fetches that failed lately: not tried again ([`FailedBlobs`]).
    pub failed: FailedBlobs,
}

impl<'a> Opener<'a> {
    pub fn new(seals: Vec<&'a AgentSeal>, api: &'a dyn SealedApi, keys: &'a SealKeys) -> Self {
        Self {
            seals,
            api,
            keys,
            keyframe: None,
            failed: FailedBlobs::default(),
        }
    }

    /// Open `entry`, sealed on `line` (whose entries hold its keyframe):
    /// the value it sealed, a state for a snapshot.
    pub async fn open(
        &mut self,
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
            BlobEncoding::Delta { base } => Some(self.keyframe(line, base).await?),
        };
        let bytes = plaintext(
            &self.seals,
            self.api,
            self.keys,
            entry,
            dict,
            &mut self.failed,
        )
        .await?;
        tokio::task::spawn_blocking(move || {
            let mut envelope: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
                anyhow::anyhow!("the sealed plaintext isn't JSON ({:?})", e.classify())
            })?;
            envelope
                .get_mut("state")
                .map(serde_json::Value::take)
                .context("the sealed plaintext holds no state")
        })
        .await
        .context("the parsing task failed")?
    }

    async fn keyframe(
        &mut self,
        line: &[SignedStateEntry],
        base: Sha256Hex,
    ) -> anyhow::Result<Dict> {
        // On its line, cached or not
        let keyframe = line
            .iter()
            .map(|e| &e.entry)
            .find(|e| e.entry_hash() == base)
            .with_context(|| format!("the delta's keyframe {base} is not on its line"))?;
        if let Some((hash, dict)) = &self.keyframe
            && *hash == base
        {
            return Ok(dict.clone());
        }
        let dict: Dict = std::sync::Arc::new(
            plaintext(
                &self.seals,
                self.api,
                self.keys,
                keyframe,
                None,
                &mut self.failed,
            )
            .await?,
        );
        self.keyframe = Some((base, dict.clone()));
        Ok(dict)
    }
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
    verified_line_cached(seal, keys, api, id, head, &mut ChainCache::default()).await
}

/// [`verified_line`], asking Agora for the chain through `chain` (a
/// session's [`ChainCache`]).
pub async fn verified_line_cached(
    seal: &AgentSeal,
    keys: &KeyStore,
    api: &dyn SealedApi,
    id: AgentId,
    head: &HeadFile,
    chain: &mut ChainCache,
) -> anyhow::Result<Vec<SignedStateEntry>> {
    let entries = local_entries(seal).await.map_err(anyhow::Error::msg)?;
    let line = match main_line(&entries, &head.head.entry_hash) {
        Some(line) => line,
        // Through Agora's entries (after a rebase or a restore)
        None => {
            let signing = keys.signing(id).context("no signing key for the agent")?;
            let chain = chain
                .get(api, id, &signing)
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
