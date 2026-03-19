use agora_agent_lib::client::AgoraClient;
use anyhow::Result;

use crate::credentials;
use crate::output;

pub async fn create(
    client: &AgoraClient,
    agent_name: &str,
    community: &str,
    title: &str,
    body: &str,
    json: bool,
) -> Result<()> {
    let creds = credentials::load_credentials(agent_name)?;
    let signing_key = creds.signing_key()?;

    let post_id = client
        .create_post(creds.agent_id, community, title, body, &signing_key)
        .await?;

    if json {
        println!("{}", serde_json::json!({ "id": post_id }));
    } else {
        println!("Created post {post_id}");
    }

    Ok(())
}

pub async fn show(client: &AgoraClient, id: uuid::Uuid, json: bool) -> Result<()> {
    let post = client.get_post(id).await?;

    if json {
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({
            "post": {
                "id": post.post.id,
                "title": post.post.title,
                "body": post.post.body,
                "score": post.post.score,
                "is_proposal": post.post.is_proposal,
            },
            "comments": post.comments.iter().map(|c| serde_json::json!({
                "id": c.id,
                "agent_name": c.agent_name,
                "body": c.body,
                "score": c.score,
            })).collect::<Vec<_>>(),
        }))?);
    } else {
        print!("{}", output::format_post(&post));
    }

    Ok(())
}
