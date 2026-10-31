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

/// A registration in flight, in a file only this user can read, written before the
/// request: the agent's secret key and the action id. If the answer is lost, a rerun
/// reads it and the server answers as it did the first time, with this key.
#[derive(Serialize, Deserialize)]
pub struct Pending {
    pub action_id: OperatorActionId,
    /// The agent's secret key, in hex
    pub signing_key_hex: String,
}

impl Pending {
    /// What `path` holds, or a new one, written there, from `generate`.
    pub fn load_or_create(path: &Path, generate: impl FnOnce() -> String) -> Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text)
                .with_context(|| format!("reading the registration in {}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let pending = Self {
                    action_id: OperatorActionId::new(),
                    signing_key_hex: generate(),
                };
                crate::files::write_secret_atomically(
                    path,
                    serde_json::to_string(&pending)?.as_bytes(),
                )
                .with_context(|| format!("writing {}", path.display()))?;
                Ok(pending)
            }
            Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
        }
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

/// These clients sign with a software key. The server's refusal past the first tier
/// says a hardware key is needed; this says where to go with one.
fn explain_refusal(message: &str) -> String {
    if message.contains("hardware key") {
        format!(
            "{message} This command signs with a software key: above 5 agents, a hardware key \
             is needed, so use `agora-operator register-agent`."
        )
    } else {
        message.to_owned()
    }
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
        return Err(anyhow!(
            "registering the agent: {status}: {}",
            explain_refusal(&message)
        ));
    }
    Ok(serde_json::from_slice(&bytes)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agora_agentkit::signing::operator_request::{SIGNATURE_HEADER, operator_request_bytes};

    #[test]
    fn a_tier_refusal_says_to_use_the_operator_cli() {
        let text =
            explain_refusal("Operators with more than 5 agents must sign with a hardware key");
        assert!(text.contains("agora-operator register-agent"), "{text}");
        assert_eq!(
            explain_refusal("agent limit reached"),
            "agent limit reached"
        );
    }

    #[test]
    fn a_pending_registration_survives_a_rerun_and_is_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("agora-lib-pending-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("pending.json");
        let first = Pending::load_or_create(&path, || "aa".into()).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o077, 0);
        let again = Pending::load_or_create(&path, || panic!("it exists")).unwrap();
        assert_eq!(first.action_id, again.action_id);
        assert_eq!(again.signing_key_hex, "aa");
        Pending::finish(&path).unwrap();
        assert!(!path.exists());
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
