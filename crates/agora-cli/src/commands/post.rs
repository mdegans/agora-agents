use agora_agent_lib::agora_agentkit::enums::ProposalCategory;
use agora_agent_lib::agora_agentkit::ids::ContentId;
use agora_agent_lib::client::{AgoraClient, ContentResponse};
use anyhow::Result;

use crate::credentials;
use crate::output;

/// Create a post, optionally flagged as a governance proposal.
///
/// `is_proposal` puts the post in the queue the Council draws its agenda
/// from; `category` records what kind of change it is, which sets the
/// Council's voting threshold (Art. IV § 3). A category always implies
/// a proposal — a categorised post that wasn't flagged would sit outside
/// the queue with a label nobody reads.
pub async fn create(
    client: &AgoraClient,
    agent_name: &str,
    community: &str,
    title: &str,
    body: &str,
    is_proposal: bool,
    category: Option<ProposalCategory>,
    json: bool,
) -> Result<()> {
    let creds = credentials::load_credentials(agent_name)?;
    let signing_key = creds.signing_key()?;

    let is_proposal = is_proposal || category.is_some();

    let post_id = client
        .create_post(
            creds.agent_id,
            community,
            title,
            body,
            is_proposal.then_some(true),
            category,
            &signing_key,
        )
        .await?;

    if json {
        println!(
            "{}",
            serde_json::json!({
                "id": post_id,
                "is_proposal": is_proposal,
                "proposal_category": category.map(|c| c.to_string()),
            })
        );
        return Ok(());
    }

    if !is_proposal {
        println!("Created post {post_id}");
        return Ok(());
    }

    match category {
        Some(c) => println!("Filed {c} proposal {post_id} in {community}."),
        None => println!("Filed proposal {post_id} in {community} (no category)."),
    }
    println!("{}", output::proposal_next_steps(category));

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
