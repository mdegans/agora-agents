use agora_agent_lib::client::AgoraClient;
use anyhow::Result;
use uuid::Uuid;

use crate::cli::VoteDirection;
use crate::credentials;

pub async fn run(
    client: &AgoraClient,
    agent_name: &str,
    direction: &VoteDirection,
    target_type: &str,
    target_id: Uuid,
    json: bool,
) -> Result<()> {
    let creds = credentials::load_credentials(agent_name)?;
    let signing_key = creds.signing_key()?;

    let value = match direction {
        VoteDirection::Up => 1,
        VoteDirection::Down => -1,
    };

    client
        .cast_vote(creds.agent_id, target_type, target_id, value, &signing_key)
        .await?;

    if json {
        println!("{}", serde_json::json!({ "status": "ok" }));
    } else {
        let arrow = match direction {
            VoteDirection::Up => "upvoted",
            VoteDirection::Down => "downvoted",
        };
        println!("{arrow} {target_type} {target_id}");
    }

    Ok(())
}
