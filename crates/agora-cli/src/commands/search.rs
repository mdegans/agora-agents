use agora_agent_lib::client::AgoraClient;
use anyhow::Result;

use crate::output;

pub async fn run(
    client: &AgoraClient,
    query: &str,
    community: Option<&str>,
    json: bool,
) -> Result<()> {
    let results = client.search(query, community).await?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!(
                results
                    .iter()
                    .map(|r| serde_json::json!({
                        "id": r.id,
                        "title": r.title,
                        "agent_name": r.agent_name,
                        "community": r.community_name,
                        "score": r.score,
                    }))
                    .collect::<Vec<_>>()
            ))?
        );
    } else {
        print!("{}", output::format_search(&results));
    }

    Ok(())
}
