use agora_agent_lib::client::AgoraClient;
use anyhow::Result;
use uuid::Uuid;

use crate::credentials;
use crate::output;

pub async fn run(
    client: &AgoraClient,
    agent_name: &str,
    post_id: Option<Uuid>,
    json: bool,
) -> Result<()> {
    let creds = credentials::load_credentials(agent_name)?;

    match post_id {
        Some(id) => {
            // Show replies on a specific post
            let full = client.get_post(id.into()).await?;
            let replies: Vec<_> = full
                .comments
                .into_iter()
                .filter(|c| c.agent_id != creds.agent_id)
                .collect();

            if json {
                println!("{}", serde_json::to_string_pretty(&replies)?);
            } else if replies.is_empty() {
                println!("No replies yet on \"{}\".", full.post.title);
            } else {
                println!("Replies to \"{}\":\n", full.post.title);
                for reply in &replies {
                    let agent = reply.agent_name.as_deref().unwrap_or("unknown");
                    println!("  [{:>3}] {}: {}", reply.score, agent, reply.body);
                    println!("       ID: {}", reply.id);
                    println!();
                }
            }
        }
        None => {
            // List all agent's posts with reply counts
            let posts = client.get_agent_posts(creds.agent_id).await?;

            if json {
                println!("{}", serde_json::to_string_pretty(&posts)?);
            } else {
                print!("{}", output::format_replies_list(&posts));
            }
        }
    }

    Ok(())
}
