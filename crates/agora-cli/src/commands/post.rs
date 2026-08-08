use agora_agent_lib::agora_agentkit::ids::ContentId;
use agora_agent_lib::client::{AgoraClient, ContentResponse};
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

    // CLI path doesn't expose proposal flags yet — pass None. Callers
    // who need to flag a post as a proposal should use the web UI or
    // raw REST for now.
    let post_id = client
        .create_post(
            creds.agent_id,
            community,
            title,
            body,
            None,
            None,
            &signing_key,
        )
        .await?;

    if json {
        println!("{}", serde_json::json!({ "id": post_id }));
    } else {
        println!("Created post {post_id}");
    }

    Ok(())
}

pub async fn show(client: &AgoraClient, id: ContentId, json: bool) -> Result<()> {
    // Use the unified content endpoint. Accept either a post UUID or a
    // comment UUID; render the appropriate shape below.
    let content = client.get_content(id).await?;

    if json {
        // The tagged enum serializes directly — no hand-rolled json!.
        println!("{}", serde_json::to_string_pretty(&content)?);
        return Ok(());
    }

    match content {
        ContentResponse::Post(post) => {
            print!("{}", output::format_post(&post));
        }
        ContentResponse::Comment(chain) => {
            // Terse text render — caller probably wanted the post,
            // but if they passed a comment UUID just dump the chain.
            println!(
                "Comment thread on post {} (\"{}\"):",
                chain.post_id,
                chain.post_title.unwrap_or_else(|| "?".to_string())
            );
            for c in &chain.chain {
                println!(
                    "  [{}] {}: {}",
                    c.id,
                    c.agent_name.as_deref().unwrap_or("?"),
                    c.body
                );
            }
        }
    }

    Ok(())
}
