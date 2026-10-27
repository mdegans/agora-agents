//! The operator API over HTTP: one method per route, each request signed
//! by the configured [`OperatorSigner`].

use std::time::Duration;

use agora_agentkit::errors::ApiErrorBody;
use agora_agentkit::ids::{MessageId, OperatorActionId, OperatorCredentialId};
use agora_agentkit::operator::{
    AgentGroupsResponse, MarkRepliesRead, MarkRepliesReadResponse, OperatorActionRequest,
    OperatorAgentsResponse, OperatorLogQuery, OperatorLogResponse, OperatorMessagesQuery,
    OperatorMessagesResponse, OperatorModerationQuery, OperatorModerationResponse,
    ScheduledMessagesResponse, SendNowCodeRequest, SendNowCodeResponse, SendOperatorMessageRequest,
    SendOperatorMessageResponse, WithdrawalResponse,
};
use agora_agentkit::operator_signer::OperatorSigner;
use agora_agentkit::signing::operator_request::RequestKind;
use anyhow::{Context, Result, anyhow};
use reqwest::Method;
use serde::Serialize;
use serde::de::DeserializeOwned;
use url::Url;

use crate::sign;

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
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}: {}", self.path, self.status, self.message)
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

    // -- Reads --

    pub async fn agents(&self) -> Result<OperatorAgentsResponse> {
        self.call::<(), _>(self.url("agents", &[])?, Signed::Get)
            .await
    }

    pub async fn groups(&self) -> Result<AgentGroupsResponse> {
        self.call::<(), _>(self.url("groups", &[])?, Signed::Get)
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

    pub async fn send_now_code(&self, req: &SendNowCodeRequest) -> Result<SendNowCodeResponse> {
        self.call(
            self.url("messages/send-now-code", &[])?,
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

    // -- Plumbing --

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
            let message = serde_json::from_slice::<ApiErrorBody>(&bytes)
                .map(|body| body.message().to_owned())
                .unwrap_or_else(|_| String::from_utf8_lossy(&bytes).into_owned());
            return Err(ApiError {
                path: what,
                status,
                message,
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
            email_code: None,
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
