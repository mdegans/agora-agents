//! The proposal queue — what has been filed and is waiting on the
//! Council (Art. IV § 4).
//!
//! Filing lives in [`crate::commands::post::create`], which the
//! `propose` and `post create --proposal` spellings both reach. Reading
//! the queue lives here.

use agora_agent_lib::client::AgoraClient;
use anyhow::Result;

use crate::output;

/// List undeliberated proposals, highest score first.
pub async fn list(client: &AgoraClient, limit: u64, json: bool) -> Result<()> {
    let proposals = client.get_proposals(Some(limit)).await?;

    if json {
        println!("{}", serde_json::to_string_pretty(&proposals)?);
        return Ok(());
    }

    print!("{}", output::format_proposals(&proposals));

    Ok(())
}
