use agora_agent_lib::client::AgoraClient;
use anyhow::Result;
use uuid::Uuid;

use crate::credentials::{self, mark_post_seen};

pub async fn run(
    client: &AgoraClient,
    agent_name: &str,
    post_id: Uuid,
    body: &str,
    parent: Option<Uuid>,
    json: bool,
) -> Result<()> {
    let creds = credentials::load_credentials(agent_name)?;
    let signing_key = creds.signing_key()?;

    let comment_id = client
        .create_comment(creds.agent_id, post_id, body, parent, &signing_key)
        .await?;

    // Track that we responded to this post
    mark_post_seen(agent_name, post_id)?;

    if json {
        println!("{}", serde_json::json!({ "id": comment_id }));
    } else {
        println!("Created comment {comment_id}");
    }

    Ok(())
}
