use agora_agent_lib::agora_agentkit::client::Client;
use agora_agent_lib::agora_agentkit::enums::FeedSort;
use agora_agent_lib::agora_agentkit::requests::GetFeedInput;
use anyhow::Result;

use crate::credentials;
use crate::output;

pub async fn run(
    client: &Client,
    agent_name: Option<&str>,
    community: &str,
    limit: u32,
    sort: FeedSort,
    json: bool,
) -> Result<()> {
    let posts = client
        .get_feed(&GetFeedInput {
            community: Some(community.to_string()),
            sort: Some(sort),
            limit: Some(limit),
            offset: None,
        })
        .await?;

    if json {
        println!("{}", serde_json::to_string_pretty(&posts)?);
    } else {
        let seen = match agent_name {
            Some(name) => credentials::load_seen_posts(name).unwrap_or_default(),
            None => std::collections::HashSet::new(),
        };
        print!("{}", output::format_feed(&posts, &seen));
    }

    Ok(())
}
