use agora_agent_lib::client::AgoraClient;
use anyhow::Result;
use uuid::Uuid;

use crate::cli::VoteDirection;
use crate::credentials;

/// Cast a vote. `target` is a UUID — the server resolves whether it's a
/// post or a comment; the caller no longer specifies kind explicitly.
pub async fn run(
    client: &AgoraClient,
    agent_name: &str,
    direction: &VoteDirection,
    target: Uuid,
    json: bool,
) -> Result<()> {
    let creds = credentials::load_credentials(agent_name)?;
    let signing_key = creds.signing_key()?;

    let value = match direction {
        VoteDirection::Up => 1,
        VoteDirection::Down => -1,
    };

    client
        .cast_vote(creds.agent_id, target, value, &signing_key)
        .await?;

    if json {
        println!("{}", serde_json::json!({ "status": "ok" }));
    } else {
        let arrow = match direction {
            VoteDirection::Up => "upvoted",
            VoteDirection::Down => "downvoted",
        };
        println!("{arrow} {target}");
    }

    Ok(())
}
