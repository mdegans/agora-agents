use anyhow::{Context, Result};

use crate::agent::Agent;
use crate::client::AgoraClient;

/// Register the seed operator and all agents.
pub async fn register_all(
    agents: &mut [Agent],
    client: &AgoraClient,
    operator_email: &str,
    operator_password: &str,
) -> Result<()> {
    // Step 1: Register operator (idempotent)
    tracing::info!("Registering operator: {operator_email}");
    match client
        .register_operator(operator_email, operator_password, Some("Seed Operator"))
        .await
    {
        Ok(_id) => tracing::info!("Operator registered (or already exists)"),
        Err(e) => {
            // If it's a conflict that's fine, otherwise bail
            let msg = e.to_string();
            if msg.contains("409") || msg.contains("already registered") {
                tracing::info!("Operator already registered");
            } else {
                return Err(e).context("registering operator");
            }
        }
    }

    // Step 2: Register each agent
    let total = agents.len();
    let mut registered = 0;
    let mut skipped = 0;
    let mut failed = 0;

    for agent in agents.iter_mut() {
        // Skip if already registered
        if agent.agent_id.is_some() {
            skipped += 1;
            continue;
        }

        // Check if agent exists on server
        match client.get_agent(&agent.name).await {
            Ok(Some(data)) => {
                if let Some(id_str) = data.get("id").and_then(|v| v.as_str()) {
                    if let Ok(uuid) = id_str.parse::<uuid::Uuid>() {
                        agent.agent_id = Some(agora_agent_lib::agora_agentkit::ids::AgentId::from(uuid));
                        agent.save_agent_id().await?;
                        skipped += 1;
                        continue;
                    }
                }
            }
            Ok(None) => {}
            Err(e) => {
                tracing::warn!("Failed to check agent {}: {e}", agent.name);
            }
        }

        // Extract bio from SOUL.md Identity section
        let bio = agent.soul.section("Identity").map(|s| {
            let truncated: String = s.chars().take(500).collect();
            truncated
        });

        match client
            .register_agent(
                operator_email,
                operator_password,
                &agent.name,
                &agent.public_key_hex(),
                Some(&agent.name), // display_name
                bio.as_deref(),
                Some(&agent.model),
            )
            .await
        {
            Ok(resp) => {
                agent.agent_id = Some(resp.id);
                agent.save_agent_id().await?;
                registered += 1;
                tracing::info!(
                    "[{}/{total}] Registered agent: {} ({})",
                    registered + skipped,
                    agent.name,
                    resp.id
                );
            }
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("already exists") {
                    skipped += 1;
                } else if msg.contains("agent limit reached") {
                    tracing::error!(
                        "Agent limit reached! Run: UPDATE operators SET max_agents = 1000 \
                         WHERE email = '{operator_email}'"
                    );
                    return Err(e).context("agent limit reached");
                } else {
                    tracing::error!("Failed to register {}: {e:#}", agent.name);
                    failed += 1;
                }
            }
        }
    }

    tracing::info!(
        "Registration complete: {registered} new, {skipped} skipped, {failed} failed (of {total})"
    );

    // Step 3: Join communities
    tracing::info!("Joining communities...");
    let mut join_count = 0;
    for agent in agents.iter() {
        let Some(agent_id) = agent.agent_id else {
            continue;
        };
        for community in &agent.communities {
            // Map SOUL.md community names to actual community slugs
            let slug = match community.as_str() {
                "technology" => "tech",
                other => other,
            };
            if let Err(e) = client.join_community(agent_id, slug).await {
                tracing::debug!("Join {slug} for {}: {e}", agent.name);
            } else {
                join_count += 1;
            }
        }
    }
    tracing::info!("Joined {join_count} community memberships");

    Ok(())
}
