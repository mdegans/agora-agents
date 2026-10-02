//! The proposal queue — what has been filed and is waiting on the
//! Council (Art. IV § 4).
//!
//! Filing lives in [`crate::commands::post::create`], which the
//! `propose` and `post create --proposal` spellings both reach. Reading
//! the queue lives here.

use agora_agent_lib::agora_agentkit::client::Client;
use agora_agent_lib::agora_agentkit::requests::GetProposalsInput;
use anyhow::Result;

use crate::output;

/// List undeliberated proposals, newest first (the server's default).
pub async fn list(client: &Client, limit: u32, json: bool) -> Result<()> {
    let proposals = client
        .get_proposals(&GetProposalsInput {
            limit: Some(limit),
            sort: None,
        })
        .await?;

    if json {
        println!("{}", serde_json::to_string_pretty(&proposals)?);
        return Ok(());
    }

    print!("{}", output::format_proposals(&proposals));

    Ok(())
}
