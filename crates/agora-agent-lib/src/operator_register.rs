//! Registering an agent with its own key: a request signed by one of the operator's
//! credentials (`POST /api/operator/agents`), for the operator and agent CLIs and the
//! seed runner's `--provision`.
//!
//! A key is written before the request that registers it (see [`Pending`]), so an
//! agent can't exist with its key lost.

use std::path::Path;

use agora_agentkit::errors::ApiErrorBody;
use agora_agentkit::ids::{OperatorActionId, OperatorCredentialId};
use agora_agentkit::operator_signer::OperatorSigner;
use agora_agentkit::operator_signer::SoftwareKeySigner;
use agora_agentkit::requests::RegisterAgentRequest;
use agora_agentkit::responses::RegisterAgentResponse;
use agora_agentkit::signing::operator_request::RequestKind;
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use url::Url;
use zeroize::Zeroizing;

/// A registration in flight, in a file only this user can read, written before the
/// request: the agent's secret key and the action id. If the answer is lost, a rerun
/// reads it and the server answers as it did the first time, with this key.
pub struct Pending {
    pub action_id: OperatorActionId,
    /// The agent's secret key, in hex
    pub signing_key_hex: Zeroizing<String>,
}

#[derive(Deserialize)]
struct PendingFile {
    action_id: OperatorActionId,
    signing_key_hex: String,
}

#[derive(Serialize)]
struct PendingFileOut<'a> {
    action_id: OperatorActionId,
    signing_key_hex: &'a str,
}

impl Pending {
    /// What `path` holds, or a new one, written there, from `generate`. Of two runs at
    /// once, one writes and both use what it wrote.
    pub fn load_or_create(
        path: &Path,
        generate: impl FnOnce() -> Zeroizing<String>,
    ) -> Result<Self> {
        if let Some(found) = Self::load(path)? {
            return Ok(found);
        }
        let pending = Self {
            action_id: OperatorActionId::new(),
            signing_key_hex: generate(),
        };
        let json = Zeroizing::new(serde_json::to_string(&PendingFileOut {
            action_id: pending.action_id,
            signing_key_hex: &pending.signing_key_hex,
        })?);
        if crate::files::write_secret_exclusive(path, json.as_bytes())? {
            return Ok(pending);
        }
        Self::load(path)?.with_context(|| format!("{} vanished as it was read", path.display()))
    }

    fn load(path: &Path) -> Result<Option<Self>> {
        let meta = match std::fs::symlink_metadata(path) {
            Ok(meta) => meta,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        anyhow::ensure!(
            meta.file_type().is_file(),
            "{} is not a regular file: refusing to read a key from it",
            path.display()
        );
        let text = Zeroizing::new(
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?,
        );
        let file: PendingFile = serde_json::from_str(&text)
            .with_context(|| format!("reading the registration in {}", path.display()))?;
        Ok(Some(Self {
            action_id: file.action_id,
            signing_key_hex: Zeroizing::new(file.signing_key_hex),
        }))
    }

    /// A new secret key in hex, for [`Self::load_or_create`].
    pub fn new_key_hex() -> Zeroizing<String> {
        let key = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        Zeroizing::new(hex::encode(key.to_bytes()))
    }

    /// The agent's key. An error names the file it came from.
    pub fn signing_key(&self, path: &Path) -> Result<ed25519_dalek::SigningKey> {
        let bytes = Zeroizing::new(
            hex::decode(self.signing_key_hex.as_str())
                .with_context(|| format!("the key in {}", path.display()))?,
        );
        let bytes: [u8; 32] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| anyhow!("the key in {} has the wrong length", path.display()))?;
        Ok(ed25519_dalek::SigningKey::from_bytes(&bytes))
    }

    /// What a failed registration says about the file, a 409 above all: after a lost
    /// answer this file is the only copy of the key of an agent that may now exist.
    pub fn kept_note(path: &Path) -> String {
        format!(
            "registering the agent failed. {} holds this registration's key and action id. \
             Don't delete it: if an earlier answer was lost, the agent may already exist with \
             this key, and a rerun with the same details picks it up",
            path.display()
        )
    }

    /// The agent's key is safe where it belongs: this goes.
    pub fn finish(path: &Path) -> Result<()> {
        std::fs::remove_file(path).with_context(|| format!("removing {}", path.display()))
    }
}

/// The route under `server` (without its `/agora`).
pub fn route(server: &Url) -> Result<Url> {
    let mut base = server.clone();
    if !base.path().ends_with('/') {
        let path = format!("{}/", base.path());
        base.set_path(&path);
    }
    Ok(base.join("agora/api/operator/agents")?)
}

/// The signer for a software key's file, which only this user may read.
pub fn software_signer(path: &Path) -> Result<SoftwareKeySigner> {
    use std::os::unix::fs::PermissionsExt;
    let mode = std::fs::metadata(path)
        .with_context(|| format!("reading {}", path.display()))?
        .permissions()
        .mode();
    anyhow::ensure!(
        mode & 0o077 == 0,
        "{} can be read by others: chmod 600 it",
        path.display()
    );
    SoftwareKeySigner::from_file(path)
        .with_context(|| format!("opening the key {}", path.display()))
}

/// The request, built and signed at `timestamp`, as it will be sent.
pub async fn build(
    http: &reqwest::Client,
    server: &Url,
    credential: OperatorCredentialId,
    signer: &dyn OperatorSigner,
    req: &RegisterAgentRequest,
    timestamp: i64,
) -> Result<reqwest::Request> {
    let mut request = http
        .post(route(server)?)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(serde_json::to_vec(req)?)
        .build()?;
    crate::operator_sign::sign(
        &mut request,
        signer,
        credential,
        RequestKind::Write(req.action_id),
        timestamp,
    )
    .await?;
    Ok(request)
}

/// Register the agent. A refusal comes back in the server's words.
pub async fn register_agent(
    server: &Url,
    credential: OperatorCredentialId,
    signer: &dyn OperatorSigner,
    req: &RegisterAgentRequest,
) -> Result<RegisterAgentResponse> {
    let http = reqwest::Client::builder()
        .use_rustls_tls()
        // A redirect would carry a signed request somewhere it was not signed for
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let request = build(
        &http,
        server,
        credential,
        signer,
        req,
        chrono::Utc::now().timestamp(),
    )
    .await?;
    let response = http
        .execute(request)
        .await
        .context("no answer from the operator API")?;
    let status = response.status();
    let bytes = response.bytes().await?;
    if !status.is_success() {
        let message = serde_json::from_slice::<ApiErrorBody>(&bytes)
            .map(|b| b.message().to_owned())
            .unwrap_or_else(|_| String::from_utf8_lossy(&bytes).into_owned());
        return Err(anyhow!("registering the agent: {status}: {message}"));
    }
    Ok(serde_json::from_slice(&bytes)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agora_agentkit::signing::operator_request::{SIGNATURE_HEADER, operator_request_bytes};

    #[test]
    fn a_pending_registration_survives_a_rerun_and_is_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("agora-lib-pending-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("pending.json");
        let first = Pending::load_or_create(&path, || Zeroizing::new("aa".into())).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o077, 0);
        let again = Pending::load_or_create(&path, || panic!("it exists")).unwrap();
        assert_eq!(first.action_id, again.action_id);
        assert_eq!(again.signing_key_hex.as_str(), "aa");
        Pending::finish(&path).unwrap();
        assert!(!path.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("agora-lib-{tag}-{}", uuid::Uuid::new_v4()))
    }

    /// `agora register` on a machine with no config directory yet
    #[test]
    fn a_pending_file_makes_its_directory_and_leaves_no_temp_file() {
        let dir = scratch("fresh");
        let path = dir.join("deep/er/pending.json");
        Pending::load_or_create(&path, Pending::new_key_hex).unwrap();
        let names: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, ["pending.json"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Runs at once all end up with one key: none can lose another's.
    #[test]
    fn concurrent_runs_share_one_key() {
        let dir = scratch("race");
        let path = dir.join("pending.json");
        let keys: Vec<String> = std::thread::scope(|s| {
            let runs: Vec<_> = (0..8)
                .map(|_| {
                    s.spawn(|| {
                        Pending::load_or_create(&path, Pending::new_key_hex)
                            .unwrap()
                            .signing_key_hex
                            .to_string()
                    })
                })
                .collect();
            runs.into_iter().map(|r| r.join().unwrap()).collect()
        });
        assert!(keys.iter().all(|k| *k == keys[0]));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_symlink_is_neither_read_nor_followed() {
        let dir = scratch("link");
        std::fs::create_dir(&dir).unwrap();
        let target = dir.join("target");
        let path = dir.join("pending.json");
        std::os::unix::fs::symlink(&target, &path).unwrap();
        let err = Pending::load_or_create(&path, Pending::new_key_hex)
            .err()
            .expect("refused");
        assert!(format!("{err:#}").contains("pending.json"), "{err:#}");
        assert!(!target.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_bad_file_and_the_note_name_the_file() {
        let dir = scratch("bad");
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("pending.json");
        std::fs::write(&path, "{").unwrap();
        let err = Pending::load_or_create(&path, Pending::new_key_hex)
            .err()
            .expect("refused");
        assert!(format!("{err:#}").contains("pending.json"), "{err:#}");
        let note = Pending::kept_note(&path);
        assert!(note.contains("pending.json") && note.contains("Don't delete it"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn request() -> RegisterAgentRequest {
        RegisterAgentRequest {
            action_id: "6a1f0f43-2c1e-4f6e-9a55-0d6c5b0a1e2f".parse().unwrap(),
            name: "ada-agent".parse().unwrap(),
            public_key: "ab".repeat(32),
            display_name: None,
            bio: Some("hi".into()),
            model_info: None,
        }
    }

    #[test]
    fn the_route_is_under_agora_whatever_the_slash() {
        for server in ["http://localhost:8099", "http://localhost:8099/"] {
            assert_eq!(
                route(&server.parse().unwrap()).unwrap().as_str(),
                "http://localhost:8099/agora/api/operator/agents"
            );
        }
    }

    /// Signed as a write of its own action, over the body as sent
    #[tokio::test]
    async fn it_is_signed_as_a_write() {
        let key = ed25519_dalek::SigningKey::from_bytes(&[7; 32]);
        let signer = SoftwareKeySigner::new(key.clone());
        let credential: OperatorCredentialId =
            "0b7e2a61-5d3c-4c8e-8f1a-2e9d4b6c7a10".parse().unwrap();
        let req = request();
        let built = build(
            &reqwest::Client::new(),
            &"http://localhost:8099".parse().unwrap(),
            credential,
            &signer,
            &req,
            1_793_000_000,
        )
        .await
        .unwrap();
        let body = built.body().unwrap().as_bytes().unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(body).unwrap();
        assert_eq!(parsed["name"], "ada-agent");
        assert!(parsed.get("display_name").is_none());
        let expected = operator_request_bytes(
            "POST",
            "localhost:8099",
            "/agora/api/operator/agents",
            credential,
            body,
            RequestKind::Write(req.action_id),
        )
        .unwrap();
        let signature = agora_agentkit::crypto::sign(&key, &expected, 1_793_000_000);
        assert_eq!(
            built.headers()[SIGNATURE_HEADER],
            hex_of(&signature.to_bytes()).as_str()
        );
    }

    fn hex_of(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn a_key_others_can_read_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let path = std::env::temp_dir().join(format!("agora-lib-key-{}", uuid::Uuid::new_v4()));
        std::fs::write(&path, "00").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let Err(refused) = software_signer(&path) else {
            panic!("a key others can read was accepted");
        };
        let refused = refused.to_string();
        assert!(refused.contains("chmod 600"), "{refused}");
        std::fs::remove_file(&path).unwrap();
    }
}
