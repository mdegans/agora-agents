use agora_agent_lib::client::AgoraClient;
use anyhow::Result;

use crate::credentials;
use crate::output;

pub async fn run(
    client: &AgoraClient,
    agent_name: Option<&str>,
    community: &str,
    limit: i64,
    sort: &str,
    json: bool,
) -> Result<()> {
    let posts = client.get_feed_sorted(community, limit, sort).await?;

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
