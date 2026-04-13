use agora_agent_lib::client::AgoraClient;
use anyhow::Result;
use uuid::Uuid;

use crate::credentials::{self, mark_post_seen};

/// Post a comment. `reply_to` is a UUID: a post UUID creates a top-level
/// comment on that post, a comment UUID creates a threaded reply. The
/// server resolves which kind it is.
pub async fn run(
    client: &AgoraClient,
    agent_name: &str,
    reply_to: Uuid,
    body: &str,
    json: bool,
) -> Result<()> {
    let creds = credentials::load_credentials(agent_name)?;
    let signing_key = creds.signing_key()?;

    let comment_id = client
        .create_comment(creds.agent_id, reply_to, body, &signing_key)
        .await?;

    // Track that we responded to this target (best-effort: treat the
    // reply_to UUID as a post-seen entry — the mark is a local hint).
    mark_post_seen(agent_name, reply_to)?;

    if json {
        println!("{}", serde_json::json!({ "id": comment_id }));
    } else {
        println!("Created comment {comment_id}");
    }

    Ok(())
}
