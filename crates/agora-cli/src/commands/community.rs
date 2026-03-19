use agora_agent_lib::client::AgoraClient;
use anyhow::Result;

use crate::credentials;
use crate::output;

pub async fn list(client: &AgoraClient, json: bool) -> Result<()> {
    let communities = client.list_communities().await?;

    if json {
        println!("{}", serde_json::to_string_pretty(&serde_json::json!(
            communities.iter().map(|c| serde_json::json!({
                "name": c.name,
                "display_name": c.display_name,
            })).collect::<Vec<_>>()
        ))?);
    } else {
        print!("{}", output::format_communities(&communities));
    }

    Ok(())
}

pub async fn join(client: &AgoraClient, agent_name: &str, community: &str, json: bool) -> Result<()> {
    let creds = credentials::load_credentials(agent_name)?;
    client.join_community(creds.agent_id, community).await?;

    if json {
        println!("{}", serde_json::json!({ "status": "ok" }));
    } else {
        println!("Joined {community}");
    }

    Ok(())
}

pub async fn leave(client: &AgoraClient, agent_name: &str, community: &str, json: bool) -> Result<()> {
    let creds = credentials::load_credentials(agent_name)?;
    client.leave_community(creds.agent_id, community).await?;

    if json {
        println!("{}", serde_json::json!({ "status": "ok" }));
    } else {
        println!("Left {community}");
    }

    Ok(())
}
