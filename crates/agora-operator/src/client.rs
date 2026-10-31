//! The operator API over HTTP: one method per route, each request signed
//! by the configured [`OperatorSigner`].

use std::time::Duration;

use agora_agentkit::errors::ApiErrorBody;
use agora_agentkit::ids::{MessageId, OperatorActionId, OperatorCredentialId};
use agora_agentkit::operator::{
    AgentGroupsResponse, BulkSudoRequest, BulkSudoResponse, CredentialContestResponse,
    MarkRepliesRead, MarkRepliesReadResponse, OperatorActionRequest, OperatorAgentsResponse,
    OperatorConfigResponse, OperatorLogQuery, OperatorLogResponse, OperatorMessagesQuery,
    OperatorMessagesResponse, OperatorModerationQuery, OperatorModerationResponse,
    ScheduledMessagesResponse, SendOperatorMessageRequest, SendOperatorMessageResponse,
    WithdrawalResponse,
};
use agora_agentkit::operator_signer::OperatorSigner;
use agora_agentkit::signing::operator_request::RequestKind;
use anyhow::{Context, Result, anyhow};
use reqwest::Method;
use serde::Serialize;
use serde::de::DeserializeOwned;
use url::Url;

use agora_agent_lib::operator_sign as sign;

/// Where the operator API lives under the server's URL
const API: &str = "agora/api/operator/";

pub struct OperatorClient {
    http: reqwest::Client,
    /// `<server>/agora/api/operator/`
    base: Url,
    credential: OperatorCredentialId,
    signer: Box<dyn OperatorSigner>,
}

/// The server answered, and refused
#[derive(Debug)]
pub struct ApiError {
    pub path: String,
    pub status: reqwest::StatusCode,
    pub message: String,
    pub hint: Option<&'static str>,
}

/// What a 401, or a 404 from no route at all, most often means
const SIGNING_HINT: &str = "Check that --server is the site's root with no path prefix \
     (https://host/, not https://host/agora/), that this machine's clock is within five \
     minutes, and that the credential id is the one the portal shows for this key.";

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}: {}", self.path, self.status, self.message)?;
        match self.hint {
            Some(hint) => write!(f, "\n{hint}"),
            None => Ok(()),
        }
    }
}

impl std::error::Error for ApiError {}

/// Which way a request is signed: the endpoint's kind, never a choice
enum Signed<'a, B> {
    Get,
    /// A POST with a body that names no action (`/messages/read`)
    ReadPost(&'a B),
    Write(&'a B, OperatorActionId),
}

impl OperatorClient {
    pub fn new(
        server: Url,
        credential: OperatorCredentialId,
        signer: Box<dyn OperatorSigner>,
    ) -> Result<Self> {
        let http = reqwest::Client::builder()
            .use_rustls_tls()
            // A redirect would carry a signed request somewhere it was not
            // signed for; a 3xx is reported, never followed
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(60))
            .user_agent(concat!("agora-operator/", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Self {
            http,
            base: base_url(server)?,
            credential,
            signer,
        })
    }

    /// Where and as what this client signs: for every write's prompt
    pub fn who(&self) -> String {
        format!(
            "{} as credential {}",
            self.base.host_str().unwrap_or("?"),
            self.credential
        )
    }

    // -- Reads --

    pub async fn agents(&self) -> Result<OperatorAgentsResponse> {
        self.call::<(), _>(self.url("agents", &[])?, Signed::Get)
            .await
    }

    pub async fn groups(&self) -> Result<AgentGroupsResponse> {
        self.call::<(), _>(self.url("groups", &[])?, Signed::Get)
            .await
    }

    /// The hold, the bulk rule and this key's recent sends, as the server
    /// applies them
    pub async fn config(&self) -> Result<OperatorConfigResponse> {
        self.call::<(), _>(self.url("config", &[])?, Signed::Get)
            .await
    }

    pub async fn scheduled(&self) -> Result<ScheduledMessagesResponse> {
        self.call::<(), _>(self.url("messages/scheduled", &[])?, Signed::Get)
            .await
    }

    pub async fn messages(
        &self,
        query: &OperatorMessagesQuery,
    ) -> Result<OperatorMessagesResponse> {
        self.call::<(), _>(with_query(self.url("messages", &[])?, query)?, Signed::Get)
            .await
    }

    pub async fn log(&self, query: &OperatorLogQuery) -> Result<OperatorLogResponse> {
        self.call::<(), _>(with_query(self.url("log", &[])?, query)?, Signed::Get)
            .await
    }

    pub async fn moderation(
        &self,
        query: &OperatorModerationQuery,
    ) -> Result<OperatorModerationResponse> {
        self.call::<(), _>(
            with_query(self.url("moderation", &[])?, query)?,
            Signed::Get,
        )
        .await
    }

    /// Changes only the operator's own read marks: signed as a read
    pub async fn mark_read(&self, which: &MarkRepliesRead) -> Result<MarkRepliesReadResponse> {
        self.call(self.url("messages/read", &[])?, Signed::ReadPost(which))
            .await
    }

    // -- Writes --

    pub async fn send(
        &self,
        req: &SendOperatorMessageRequest,
    ) -> Result<SendOperatorMessageResponse> {
        self.call(
            self.url("messages", &[])?,
            Signed::Write(req, req.action_id),
        )
        .await
    }

    pub async fn bulk_sudo(&self, req: &BulkSudoRequest) -> Result<BulkSudoResponse> {
        self.call(
            self.url("config/bulk-sudo", &[])?,
            Signed::Write(req, req.action_id),
        )
        .await
    }

    pub async fn cancel(
        &self,
        message: MessageId,
        req: &OperatorActionRequest,
    ) -> Result<WithdrawalResponse> {
        let url = self.url("messages", &[&message.to_string(), "cancel"])?;
        self.call(url, Signed::Write(req, req.action_id)).await
    }

    pub async fn retract(
        &self,
        message: MessageId,
        req: &OperatorActionRequest,
    ) -> Result<WithdrawalResponse> {
        let url = self.url("messages", &[&message.to_string(), "retract"])?;
        self.call(url, Signed::Write(req, req.action_id)).await
    }

    /// Every copy of the send `send`
    pub async fn retract_batch(
        &self,
        send: OperatorActionId,
        req: &OperatorActionRequest,
    ) -> Result<WithdrawalResponse> {
        let url = self.url("batches", &[&send.to_string(), "retract"])?;
        self.call(url, Signed::Write(req, req.action_id)).await
    }

    /// Contest the suspension of the credential signing this: the path names
    /// it, because a suspended credential contests only for itself
    pub async fn contest(&self, req: &OperatorActionRequest) -> Result<CredentialContestResponse> {
        self.call(self.contest_url()?, Signed::Write(req, req.action_id))
            .await
    }

    // -- Plumbing --

    fn contest_url(&self) -> Result<Url> {
        self.url("credentials", &[&self.credential.to_string(), "contest"])
    }

    /// `<base>/<path>/<segment>/…`, each segment percent-encoded
    fn url(&self, path: &str, segments: &[&str]) -> Result<Url> {
        let mut url = self.base.join(path)?;
        if !segments.is_empty() {
            url.path_segments_mut()
                .map_err(|()| anyhow!("the server URL cannot take a path"))?
                .extend(segments);
        }
        Ok(url)
    }

    /// Register an agent with its own key
    pub async fn register_agent(
        &self,
        req: &agora_agentkit::requests::RegisterAgentRequest,
    ) -> Result<agora_agentkit::responses::RegisterAgentResponse> {
        self.call(self.url("agents", &[])?, Signed::Write(req, req.action_id))
            .await
    }

    /// The request, built and signed at `timestamp`, as it will be sent
    async fn build<B: Serialize>(
        &self,
        url: Url,
        signed: Signed<'_, B>,
        timestamp: i64,
    ) -> Result<reqwest::Request> {
        let (method, body, kind) = match signed {
            Signed::Get => (Method::GET, None, RequestKind::Read),
            Signed::ReadPost(body) => (
                Method::POST,
                Some(serde_json::to_vec(body)?),
                RequestKind::Read,
            ),
            Signed::Write(body, action) => (
                Method::POST,
                Some(serde_json::to_vec(body)?),
                RequestKind::Write(action),
            ),
        };
        let mut builder = self.http.request(method, url);
        if let Some(body) = body {
            builder = builder
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(body);
        }
        let mut request = builder.build()?;
        sign::sign(
            &mut request,
            self.signer.as_ref(),
            self.credential,
            kind,
            timestamp,
        )
        .await?;
        Ok(request)
    }

    async fn call<B: Serialize, T: DeserializeOwned>(
        &self,
        url: Url,
        signed: Signed<'_, B>,
    ) -> Result<T> {
        let what = url.path().to_owned();
        let request = self
            .build(url, signed, chrono::Utc::now().timestamp())
            .await?;
        let response = self
            .http
            .execute(request)
            .await
            .with_context(|| format!("no answer from {what}"))?;
        let status = response.status();
        let bytes = response.bytes().await?;
        if !status.is_success() {
            let body = serde_json::from_slice::<ApiErrorBody>(&bytes).ok();
            // A 404 with no API error body is a route the server does not
            // have, not a refusal by one it has
            let hint = match status {
                reqwest::StatusCode::UNAUTHORIZED => Some(SIGNING_HINT),
                reqwest::StatusCode::NOT_FOUND if body.is_none() => Some(SIGNING_HINT),
                _ => None,
            };
            let message = match body {
                Some(body) => body.message().to_owned(),
                None => String::from_utf8_lossy(&bytes).into_owned(),
            };
            return Err(ApiError {
                path: what,
                status,
                message,
                hint,
            }
            .into());
        }
        serde_json::from_slice(&bytes)
            .with_context(|| format!("{what}: the answer does not parse as expected"))
    }
}

/// `server` with the operator API's path beneath it
fn base_url(mut server: Url) -> Result<Url> {
    if !server.path().ends_with('/') {
        let path = format!("{}/", server.path());
        server.set_path(&path);
    }
    Ok(server.join(API)?)
}

/// `url` with `query`'s fields as its query string, as the server reads
/// them (url-encoded; an absent field is left out)
fn with_query<Q: Serialize>(url: Url, query: &Q) -> Result<Url> {
    let request = reqwest::Client::new().get(url).query(query).build()?;
    Ok(request.url().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use agora_agentkit::crypto::SigningKey;
    use agora_agentkit::ids::AgentId;
    use agora_agentkit::operator::OperatorRecipients;
    use agora_agentkit::operator_signer::SoftwareKeySigner;
    use agora_agentkit::signing::operator_request::{SIGNATURE_HEADER, operator_request_bytes};

    fn client(server: &str) -> OperatorClient {
        OperatorClient::new(
            server.parse().unwrap(),
            "0b7e2a61-5d3c-4c8e-8f1a-2e9d4b6c7a10".parse().unwrap(),
            Box::new(SoftwareKeySigner::new(SigningKey::from_bytes(&[7; 32]))),
        )
        .unwrap()
    }

    /// A one-shot server on loopback that answers every connection with
    /// `response`, counting the connections
    fn serve(response: &'static str) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count = hits.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let _ = stream.write_all(response.as_bytes());
            }
        });
        (url, hits)
    }

    /// A redirect is reported, never followed: the signed request goes to
    /// the server named and nowhere else
    #[tokio::test]
    async fn a_redirect_is_refused() {
        let (url, hits) = serve(
            "HTTP/1.1 302 Found\r\nLocation: https://elsewhere.example/agora/api/operator/agents\r\n\
             Content-Length: 0\r\nConnection: close\r\n\r\n",
        );
        let err = client(&url).agents().await.unwrap_err();
        let api = err
            .downcast_ref::<ApiError>()
            .expect("an answer, not a transport error");
        assert_eq!(api.status, reqwest::StatusCode::FOUND);
        assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    /// A 401, or a 404 with no API body (no such route), says what to check
    #[tokio::test]
    async fn refusals_of_the_signature_say_what_to_check() {
        let (url, _) = serve(
            "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\n\
             Content-Length: 42\r\nConnection: close\r\n\r\n{\"error\":\"the signature was not accepted\"}",
        );
        let err = client(&url).agents().await.unwrap_err().to_string();
        assert!(
            err.contains("no path prefix") && err.contains("clock"),
            "{err}"
        );
        let (url, _) = serve(
            "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n\
             Content-Length: 39\r\nConnection: close\r\n\r\n{\"error\":\"x is not one of your agents\"}",
        );
        let err = client(&url).agents().await.unwrap_err().to_string();
        assert!(!err.contains("clock"), "{err}");
    }

    #[test]
    fn routes_are_under_agora_api_operator() {
        let c = client("http://localhost:8099");
        let id: MessageId = "6a1f0f43-2c1e-4f6e-9a55-0d6c5b0a1e2f".parse().unwrap();
        assert_eq!(
            c.url("messages", &[&id.to_string(), "cancel"])
                .unwrap()
                .as_str(),
            "http://localhost:8099/agora/api/operator/messages/6a1f0f43-2c1e-4f6e-9a55-0d6c5b0a1e2f/cancel"
        );
        assert_eq!(
            c.url("messages/send-now-code", &[]).unwrap().as_str(),
            "http://localhost:8099/agora/api/operator/messages/send-now-code"
        );
        // A server URL with a trailing slash, or a path, keeps it
        assert_eq!(
            client("https://example.org/")
                .url("log", &[])
                .unwrap()
                .as_str(),
            "https://example.org/agora/api/operator/log"
        );
        assert_eq!(
            client("https://example.org/x")
                .url("log", &[])
                .unwrap()
                .as_str(),
            "https://example.org/x/agora/api/operator/log"
        );
    }

    #[test]
    fn queries_leave_absent_fields_out() {
        let c = client("http://localhost:8099");
        let agent = AgentId::new();
        let url = with_query(
            c.url("messages", &[]).unwrap(),
            &OperatorMessagesQuery {
                agent: Some(agent),
                group: None,
                unread: true,
                before: None,
                limit: Some(5),
            },
        )
        .unwrap();
        assert_eq!(
            url.query(),
            Some(&*format!("agent={agent}&unread=true&limit=5"))
        );
        let url = with_query(c.url("log", &[]).unwrap(), &OperatorLogQuery::default()).unwrap();
        assert_eq!(url.query(), None);
    }

    /// The request `send` builds: its JSON body, signed as a write with the
    /// body's own action id, byte for byte
    #[tokio::test]
    async fn a_send_is_signed_as_a_write_of_its_own_action() {
        let c = client("http://localhost:8099");
        let action: OperatorActionId = "6a1f0f43-2c1e-4f6e-9a55-0d6c5b0a1e2f".parse().unwrap();
        let req = SendOperatorMessageRequest {
            action_id: action,
            to: OperatorRecipients::All,
            body: "hello".into(),
            about: None,
            deliver_at: None,
            send_now: false,
            dry_run: false,
        };
        let request = c
            .build(
                c.url("messages", &[]).unwrap(),
                Signed::Write(&req, req.action_id),
                1_793_000_000,
            )
            .await
            .unwrap();
        let body =
            br#"{"action_id":"6a1f0f43-2c1e-4f6e-9a55-0d6c5b0a1e2f","to":"all","body":"hello"}"#;
        assert_eq!(request.body().unwrap().as_bytes().unwrap(), body);
        assert_eq!(request.headers()["content-type"], "application/json");
        let expected = operator_request_bytes(
            "POST",
            "localhost:8099",
            "/agora/api/operator/messages",
            c.credential,
            body,
            RequestKind::Write(action),
        )
        .unwrap();
        let signature = agora_agentkit::crypto::sign(
            &SigningKey::from_bytes(&[7; 32]),
            &expected,
            1_793_000_000,
        );
        assert_eq!(
            request.headers()[SIGNATURE_HEADER],
            hex::encode(signature.to_bytes()).as_str()
        );
    }

    /// `contest`, as the server reads it, signed as a write of its own action
    /// under the route's path
    #[tokio::test]
    async fn a_contest_is_signed_as_a_write_of_its_own_action() {
        let c = client("http://localhost:8099");
        let action: OperatorActionId = "6a1f0f43-2c1e-4f6e-9a55-0d6c5b0a1e2f".parse().unwrap();
        let req = OperatorActionRequest { action_id: action };
        let url = c.contest_url().unwrap();
        let request = c
            .build(url, Signed::Write(&req, req.action_id), 1_793_000_000)
            .await
            .unwrap();
        let body = br#"{"action_id":"6a1f0f43-2c1e-4f6e-9a55-0d6c5b0a1e2f"}"#;
        assert_eq!(request.body().unwrap().as_bytes().unwrap(), body);
        let expected = operator_request_bytes(
            "POST",
            "localhost:8099",
            "/agora/api/operator/credentials/0b7e2a61-5d3c-4c8e-8f1a-2e9d4b6c7a10/contest",
            c.credential,
            body,
            RequestKind::Write(action),
        )
        .unwrap();
        let signature = agora_agentkit::crypto::sign(
            &SigningKey::from_bytes(&[7; 32]),
            &expected,
            1_793_000_000,
        );
        assert_eq!(
            request.headers()[SIGNATURE_HEADER],
            hex::encode(signature.to_bytes()).as_str()
        );
    }

    /// What a YubiKey's 9c slot signs: P-256, one tap per signature. The
    /// header carries what the server's `verify_request` takes for that key
    #[tokio::test]
    async fn a_write_verifies_under_a_p256_touch_always_key() {
        use agora_agentkit::operator_signer::mock::{MockPivDevice, MockRootCa};
        use agora_agentkit::operator_signer::{
            HardwareKey, KeyPolicy, OperatorSignature, Pin, PivSerial, PivSlot, SignatureAlgorithm,
            verify_request,
        };
        let device = MockPivDevice::manufacture(
            &MockRootCa::generate(),
            PivSerial::new(17),
            MockPivDevice::FIRMWARE_P256_ONLY,
            Pin::new("123456"),
        );
        let key = device
            .generate_key(
                PivSlot::Signature,
                SignatureAlgorithm::EcdsaP256,
                KeyPolicy::SUDO,
            )
            .unwrap();
        key.verify_pin(&Pin::new("123456")).await.unwrap();
        let public = key.public_key();
        let credential: OperatorCredentialId =
            "0b7e2a61-5d3c-4c8e-8f1a-2e9d4b6c7a10".parse().unwrap();
        let c = OperatorClient::new(
            "https://example.org".parse().unwrap(),
            credential,
            Box::new(key),
        )
        .unwrap();
        let action = OperatorActionId::new();
        let req = OperatorActionRequest { action_id: action };
        device.press();
        let request = c
            .build(
                c.contest_url().unwrap(),
                Signed::Write(&req, req.action_id),
                1_793_000_000,
            )
            .await
            .unwrap();
        let signature = hex::decode(request.headers()[SIGNATURE_HEADER].as_bytes()).unwrap();
        let signature = OperatorSignature::from_bytes(public.algorithm(), &signature).unwrap();
        let signed = agora_agent_lib::operator_sign::request_bytes(
            &request,
            credential,
            RequestKind::Write(action),
        )
        .unwrap();
        verify_request(&public, &signed, 1_793_000_000, &signature).unwrap();
    }

    /// A contest's path is the credential that signs it
    #[test]
    fn a_contest_is_for_the_signing_credential() {
        assert_eq!(
            client("http://localhost:8099")
                .contest_url()
                .unwrap()
                .as_str(),
            "http://localhost:8099/agora/api/operator/credentials/0b7e2a61-5d3c-4c8e-8f1a-2e9d4b6c7a10/contest"
        );
    }

    /// A refusal comes back in the server's words, for the operator to read:
    /// the CLI decides nothing about who may contest
    #[tokio::test]
    async fn a_refused_contest_says_what_the_server_said() {
        let (url, _) = serve(
            "HTTP/1.1 403 Forbidden\r\nContent-Type: application/json\r\n\
             Content-Length: 22\r\nConnection: close\r\n\r\n{\"error\":\"needs sudo\"}",
        );
        let err = client(&url)
            .contest(&OperatorActionRequest {
                action_id: OperatorActionId::new(),
            })
            .await
            .unwrap_err();
        let api = err.downcast_ref::<ApiError>().expect("an answer");
        assert_eq!(api.status, reqwest::StatusCode::FORBIDDEN);
        assert_eq!(api.message, "needs sudo");
    }

    /// Marking read posts a body but names no action: a read
    #[tokio::test]
    async fn mark_read_is_signed_as_a_read() {
        let c = client("http://localhost:8099");
        let request = c
            .build(
                c.url("messages/read", &[]).unwrap(),
                Signed::ReadPost(&MarkRepliesRead::All),
                1_793_000_000,
            )
            .await
            .unwrap();
        let expected = operator_request_bytes(
            "POST",
            "localhost:8099",
            "/agora/api/operator/messages/read",
            c.credential,
            b"\"all\"",
            RequestKind::Read,
        )
        .unwrap();
        let signature = agora_agentkit::crypto::sign(
            &SigningKey::from_bytes(&[7; 32]),
            &expected,
            1_793_000_000,
        );
        assert_eq!(
            request.headers()[SIGNATURE_HEADER],
            hex::encode(signature.to_bytes()).as_str()
        );
    }
}
