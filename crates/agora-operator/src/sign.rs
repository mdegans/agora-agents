//! Signing a request as an operator credential: agentkit's
//! `operator_request_bytes` over the request exactly as it will be sent,
//! signed through an [`OperatorSigner`], into the three headers.

use agora_agentkit::ids::OperatorCredentialId;
use agora_agentkit::operator_signer::OperatorSigner;
use agora_agentkit::signing::operator_request::{
    CREDENTIAL_ID_HEADER, RequestKind, SIGNATURE_HEADER, TIMESTAMP_HEADER, operator_request_bytes,
};
use agora_agentkit::signing::signed_get::host_of;
use anyhow::{Context, Result};
use reqwest::header::HeaderValue;

/// The bytes `request`'s signature covers: its method, host, path and
/// query, `credential`, its body, and for a write the action id
pub fn request_bytes(
    request: &reqwest::Request,
    credential: OperatorCredentialId,
    kind: RequestKind,
) -> Result<Vec<u8>> {
    let url = request.url();
    let body = match request.body() {
        None => &[][..],
        Some(body) => body
            .as_bytes()
            .context("an operator request's body is sent whole, never streamed")?,
    };
    Ok(operator_request_bytes(
        request.method().as_str(),
        &host_of(url),
        &url[url::Position::BeforePath..url::Position::AfterQuery],
        credential,
        body,
        kind,
    )?)
}

/// Sign `request` at `timestamp` (Unix seconds), adding the headers
pub async fn sign(
    request: &mut reqwest::Request,
    signer: &dyn OperatorSigner,
    credential: OperatorCredentialId,
    kind: RequestKind,
    timestamp: i64,
) -> Result<()> {
    let bytes = request_bytes(request, credential, kind)?;
    let signature =
        agora_agentkit::operator_signer::sign_request(signer, &bytes, timestamp).await?;
    let headers = request.headers_mut();
    headers.insert(
        CREDENTIAL_ID_HEADER,
        HeaderValue::from_str(&credential.to_string())?,
    );
    headers.insert(TIMESTAMP_HEADER, HeaderValue::from(timestamp));
    headers.insert(
        SIGNATURE_HEADER,
        HeaderValue::from_str(&signature.to_hex())?,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use agora_agentkit::crypto::{self, SigningKey};
    use agora_agentkit::ids::OperatorActionId;
    use agora_agentkit::operator_signer::SoftwareKeySigner;

    const TIME: i64 = 1_793_000_000;

    fn key() -> SigningKey {
        SigningKey::from_bytes(&[7; 32])
    }

    fn credential() -> OperatorCredentialId {
        "0b7e2a61-5d3c-4c8e-8f1a-2e9d4b6c7a10".parse().unwrap()
    }

    fn header<'a>(request: &'a reqwest::Request, name: &str) -> &'a str {
        request.headers()[name].to_str().unwrap()
    }

    /// The signature header is the key's own signature over agentkit's
    /// bytes, built here from literals: byte for byte
    fn assert_signed_over(request: &reqwest::Request, expected: &[u8]) {
        let signature = crypto::sign(&key(), expected, TIME);
        assert_eq!(
            header(request, SIGNATURE_HEADER),
            hex::encode(signature.to_bytes())
        );
        assert_eq!(header(request, TIMESTAMP_HEADER), TIME.to_string());
        assert_eq!(
            header(request, CREDENTIAL_ID_HEADER),
            credential().to_string()
        );
        assert!(crypto::verify(
            &key().verifying_key(),
            expected,
            TIME,
            &signature
        ));
    }

    #[tokio::test]
    async fn a_write_signs_its_body_and_action() {
        let action: OperatorActionId = "6a1f0f43-2c1e-4f6e-9a55-0d6c5b0a1e2f".parse().unwrap();
        let body =
            br#"{"action_id":"6a1f0f43-2c1e-4f6e-9a55-0d6c5b0a1e2f","to":"all","body":"hi"}"#;
        let mut request = reqwest::Client::new()
            .post("http://LocalHost:8099/agora/api/operator/messages")
            .body(body.to_vec())
            .build()
            .unwrap();
        let signer = SoftwareKeySigner::new(key());
        sign(
            &mut request,
            &signer,
            credential(),
            RequestKind::Write(action),
            TIME,
        )
        .await
        .unwrap();
        let expected = operator_request_bytes(
            "POST",
            "localhost:8099",
            "/agora/api/operator/messages",
            credential(),
            body,
            RequestKind::Write(action),
        )
        .unwrap();
        assert_signed_over(&request, &expected);
        // The body sent is the body signed
        assert_eq!(request.body().unwrap().as_bytes().unwrap(), body);
    }

    #[tokio::test]
    async fn a_read_signs_its_query_as_sent_and_no_action() {
        let mut request = reqwest::Client::new()
            .get("https://example.org/agora/api/operator/log")
            .query(&[("before", "2026-10-27T12:00:00Z"), ("limit", "5")])
            .build()
            .unwrap();
        let signer = SoftwareKeySigner::new(key());
        sign(&mut request, &signer, credential(), RequestKind::Read, TIME)
            .await
            .unwrap();
        let expected = operator_request_bytes(
            "GET",
            "example.org",
            "/agora/api/operator/log?before=2026-10-27T12%3A00%3A00Z&limit=5",
            credential(),
            b"",
            RequestKind::Read,
        )
        .unwrap();
        assert_signed_over(&request, &expected);
    }

    #[tokio::test]
    async fn a_signature_made_as_a_read_is_not_a_writes() {
        let action = OperatorActionId::new();
        let request = reqwest::Client::new()
            .post("http://localhost:8099/agora/api/operator/messages/read")
            .body("\"all\"")
            .build()
            .unwrap();
        let read = request_bytes(&request, credential(), RequestKind::Read).unwrap();
        let write = request_bytes(&request, credential(), RequestKind::Write(action)).unwrap();
        assert_ne!(read, write);
    }
}
