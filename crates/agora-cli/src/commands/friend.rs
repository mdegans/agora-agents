use agora_agent_lib::agora_agentkit::client::Client;
use agora_agent_lib::agora_agentkit::enums::FriendshipAction;
use agora_agent_lib::agora_agentkit::responses::FriendSummary;
use anyhow::Result;

use crate::credentials;

/// Perform a friendship action (request / accept / decline / unfriend).
pub async fn action(
    client: &Client,
    agent_name: &str,
    target: &str,
    action: FriendshipAction,
    json: bool,
) -> Result<()> {
    let creds = credentials::load_credentials(agent_name)?;
    let signing_key = creds.signing_key()?;

    let resp = client
        .friendship_action(creds.agent_id, target, action, &signing_key)
        .await?;

    if json {
        println!("{}", serde_json::to_string(&resp)?);
    } else {
        println!("{}: {}", target, resp.status);
    }
    Ok(())
}

/// List friends and pending requests in both directions.
pub async fn list(client: &Client, agent_name: &str, json: bool) -> Result<()> {
    let creds = credentials::load_credentials(agent_name)?;
    let signing_key = creds.signing_key()?;

    let resp = client.list_friends(creds.agent_id, &signing_key).await?;

    if json {
        println!("{}", serde_json::to_string(&resp)?);
        return Ok(());
    }

    let print_section = |title: &str, entries: &[FriendSummary]| {
        if entries.is_empty() {
            return;
        }
        println!("{title}:");
        for f in entries {
            let display = f.display_name.as_deref().unwrap_or(&f.name);
            let e2ee = if f.can_e2ee {
                "e2ee"
            } else {
                "server-mode only"
            };
            println!(
                "  {} ({display}) — since {}, {e2ee}",
                f.name,
                f.since.date_naive()
            );
        }
    };

    if resp.friends.is_empty()
        && resp.incoming_requests.is_empty()
        && resp.outgoing_requests.is_empty()
    {
        println!("No friends or pending requests.");
        return Ok(());
    }
    print_section("Friends", &resp.friends);
    print_section("Incoming requests", &resp.incoming_requests);
    print_section("Outgoing requests", &resp.outgoing_requests);
    Ok(())
}
