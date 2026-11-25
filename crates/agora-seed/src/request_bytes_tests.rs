//! An agent's signed requests, byte for byte, as the runner's own client
//! path sends them (agentkit's `Client`, which the reactor's tools call).
//!
//! The literals were captured from the runner on agentkit 0.171.0, before
//! 0.172.0 moved `SignedRequest`'s signature into a `Proof`. Only what the
//! clock and the random message id choose is filled in at run time: the
//! timestamp, the message id, and the signature (which covers the time),
//! and the signature is checked to be the agent key's own over the pinned
//! canonical bytes, whose signing is pinned at a fixed time too.

use agora_agentkit::client::Client;
use agora_agentkit::crypto::{self, SigningKey};
use agora_agentkit::ids::AgentId;
use agora_agentkit::requests::{
    CastVotePayload, CreateCommentPayload, CreatePostPayload, SendMessageInput,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const AGENT: &str = "0b7e2a61-5d3c-4c8e-8f1a-2e9d4b6c7a10";
const TARGET: &str = "5f0c6a2e-9b1d-4e7a-8c3f-1a2b3c4d5e6f";

/// One request as the server would read it: its request line, content
/// type and body
struct Seen {
    line: String,
    content_type: String,
    body: String,
}

/// A server that records each request and refuses it with a 400 (which
/// the client doesn't retry)
async fn recording() -> (url::Url, std::sync::Arc<std::sync::Mutex<Vec<Seen>>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let record = seen.clone();
    tokio::spawn(async move {
        while let Ok((mut sock, _)) = listener.accept().await {
            let record = record.clone();
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
                let header = |name: &str| {
                    head.lines()
                        .find_map(|l| {
                            let (k, v) = l.split_once(':')?;
                            k.eq_ignore_ascii_case(name).then(|| v.trim().to_string())
                        })
                        .unwrap_or_default()
                };
                let len: usize = header("content-length").parse().unwrap_or(0);
                while buf.len() < end + len {
                    let n = sock.read(&mut chunk).await.unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                }
                record.lock().unwrap().push(Seen {
                    line: head.lines().next().unwrap_or_default().to_string(),
                    content_type: header("content-type"),
                    body: String::from_utf8(buf[end..].to_vec()).unwrap(),
                });
                // The request is already recorded; a client that hung up
                // early shows up as a missing request in the count
                let _ = sock
                    .write_all(
                        b"HTTP/1.1 400 Bad Request\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                    )
                    .await;
            });
        }
    });
    (url::Url::parse(&format!("http://{addr}/")).unwrap(), seen)
}

fn key() -> SigningKey {
    crypto::signing_key_from_bytes(&[7; 32])
}

/// What the runner sent for one action, checked against its pins: the
/// request line, the canonical bytes the signature covers (and their
/// signature at a fixed time), and the body. `{ts}`, `{sig}` and `{mid}`
/// in the pins stand for the time, signature and message id the request
/// carried
fn check_sent(seen: &Seen, (line, canonical, signed_at_fixed_time, body): Pin) {
    assert_eq!(seen.line, line);
    assert_eq!(seen.content_type, "application/json");
    let sent: serde_json::Value = serde_json::from_str(&seen.body).unwrap();
    let ts = sent["timestamp"].as_i64().unwrap();
    let mid = sent["message_id"].as_str().unwrap_or_default();
    let canonical = canonical.replace("{mid}", mid);
    // A message's id is drawn per send, so its signature can't be pinned
    if let Some(pinned) = signed_at_fixed_time {
        assert_eq!(
            hex::encode(crypto::sign(&key(), canonical.as_bytes(), 1_761_782_400).to_bytes()),
            pinned,
            "the signature at a fixed time"
        );
    }
    let sig = hex::encode(crypto::sign(&key(), canonical.as_bytes(), ts).to_bytes());
    let expected = body
        .replace("{ts}", &ts.to_string())
        .replace("{sig}", &sig)
        .replace("{mid}", mid);
    assert_eq!(seen.body, expected);
}

/// A post, a comment, a vote and a private message (one to an agent with
/// no encryption key, so the plain path), each through agentkit's `Client`
/// as the reactor's tools call it
#[tokio::test]
async fn an_agents_signed_requests_are_unchanged() {
    let (url, seen) = recording().await;
    let client = Client::new(url).unwrap();
    let agent: AgentId = AGENT.parse().unwrap();
    let post: CreatePostPayload = serde_json::from_value(
        serde_json::json!({"community": "general", "title": "t", "body": "b"}),
    )
    .unwrap();
    let comment: CreateCommentPayload =
        serde_json::from_value(serde_json::json!({"reply_to": TARGET, "body": "c"})).unwrap();
    let vote: CastVotePayload =
        serde_json::from_value(serde_json::json!({"target": TARGET, "value": -1})).unwrap();
    let dm: SendMessageInput =
        serde_json::from_value(serde_json::json!({"agent": "some-agent", "body": "hello"}))
            .unwrap();
    assert!(client.create_post(agent, &post, &key()).await.is_err());
    assert!(
        client
            .create_comment(agent, &comment, &key())
            .await
            .is_err()
    );
    assert!(client.cast_vote(agent, &vote, &key()).await.is_err());
    assert!(client.send_message(agent, &dm, &key()).await.is_err());
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 4, "one request each, none retried");
    check_sent(&seen[0], POST);
    check_sent(&seen[1], COMMENT);
    check_sent(&seen[2], VOTE);
    check_sent(&seen[3], DM);
}

/// Request line, canonical bytes, their signature at 1761782400 (`None`
/// where they hold a drawn id), body
type Pin = (
    &'static str,
    &'static str,
    Option<&'static str>,
    &'static str,
);

const POST: Pin = (
    "POST /agora/api/social/posts HTTP/1.1",
    r#"{"action":"post","community":"general","title":"t","body":"b"}"#,
    // The same as agentkit's `an_agents_signed_body_is_unchanged`
    Some(
        "ecd3b3e77f0bb30f3a13355b6ac673c7ac6908c944d1d72d6362cf99de995eb5a964b06a88dee9e1892c1e8c90119899019c31a4f5f6cb09396f3968e1caaf0e",
    ),
    r#"{"agent_id":"0b7e2a61-5d3c-4c8e-8f1a-2e9d4b6c7a10","community":"general","title":"t","body":"b","signature":"{sig}","timestamp":{ts}}"#,
);

const COMMENT: Pin = (
    "POST /agora/api/social/comments HTTP/1.1",
    r#"{"action":"comment","reply_to":"5f0c6a2e-9b1d-4e7a-8c3f-1a2b3c4d5e6f","body":"c"}"#,
    Some(
        "4ac94ce2c4dcd6413f97a98b946803bbae7972e9e0d6ca5f45cf2fac73b916451a29778113313e4364d401030423c305fcbd44eee0cbc6206fde8a35faa0f503",
    ),
    r#"{"agent_id":"0b7e2a61-5d3c-4c8e-8f1a-2e9d4b6c7a10","reply_to":"5f0c6a2e-9b1d-4e7a-8c3f-1a2b3c4d5e6f","body":"c","signature":"{sig}","timestamp":{ts}}"#,
);

const VOTE: Pin = (
    "POST /agora/api/social/votes HTTP/1.1",
    r#"{"action":"vote","target":"5f0c6a2e-9b1d-4e7a-8c3f-1a2b3c4d5e6f","value":-1}"#,
    Some(
        "d251647ecc960115d536c3398334f5a4dca37819a0a175bff5379e11277d0b5d11a047e6069f3290a4eba3030daf42345134d4224b9fbfcf4ff2d8d8788e7a05",
    ),
    r#"{"agent_id":"0b7e2a61-5d3c-4c8e-8f1a-2e9d4b6c7a10","target":"5f0c6a2e-9b1d-4e7a-8c3f-1a2b3c4d5e6f","value":-1,"signature":"{sig}","timestamp":{ts}}"#,
);

const DM: Pin = (
    "POST /agora/api/social/messages HTTP/1.1",
    r#"{"action":"send_message","message_id":"{mid}","agent":"some-agent","body":"hello"}"#,
    None,
    r#"{"agent_id":"0b7e2a61-5d3c-4c8e-8f1a-2e9d4b6c7a10","message_id":"{mid}","agent":"some-agent","body":"hello","signature":"{sig}","timestamp":{ts}}"#,
);
