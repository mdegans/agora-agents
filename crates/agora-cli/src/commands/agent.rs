use agora_agent_lib::client::AgoraClient;
use anyhow::Result;

use crate::output;

pub async fn info(client: &AgoraClient, name: &str, json: bool) -> Result<()> {
    let agent = client
        .get_agent(name)
        .await?
        .ok_or_else(|| anyhow::anyhow!("agent '{name}' not found"))?;

    if json {
        println!("{}", serde_json::to_string_pretty(&agent)?);
    } else {
        print!("{}", output::format_agent(&agent));
    }

    Ok(())
}
