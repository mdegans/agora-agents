use agora_agent_lib::agora_agentkit::client::Client;
use agora_agent_lib::agora_agentkit::requests::SearchInput;
use anyhow::Result;

use crate::output;

pub async fn run(client: &Client, query: &str, community: Option<&str>, json: bool) -> Result<()> {
    let found = client
        .search(&SearchInput {
            community: community.map(str::to_string),
            ..SearchInput::new(query)
        })
        .await?;

    if json {
        println!("{}", serde_json::to_string_pretty(&found)?);
    } else {
        print!("{}", output::format_search(&found));
    }

    Ok(())
}
