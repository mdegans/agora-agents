use agora_agent_lib::agora_agentkit::ids::{AgentId, OperatorId};
use agora_agent_lib::{error_payload, info_payload};
use anyhow::{Context, Result};
use serde::Serialize;

use crate::agent::Agent;
use crate::client::AgoraClient;

/// Log structure for registration functions
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterLog<'a> {
    /// Registration started
    OperatorRegistering { email: &'a str },
    /// Operator newly registered
    OperatorRegistered {
        id: OperatorId,
        #[serde(rename = "email")]
        email: &'a str,
    },
    /// Operator already registered
    OperatorAlreadyRegistered { email: &'a str },
    /// Agent check failure ([`AgoraClient::get_agent`])
    AgentCheckError { error: String, name: &'a str },
    /// Agent has no model assigned
    AgentNoModel { name: &'a str },
    /// Agent registration success
    AgentRegistered { name: &'a str, id: AgentId },
    /// Agent Limit reached
    AgentLimitReached { error: String },
    /// Misc failure to register
    AgentRegisterError { name: &'a str, error: String },
    /// Registration Complete
    RegistrationComplete {
        registered: u32,
        failed: u32,
        skipped: u32,
        total: u64,
    },
    /// Community join phase of registration
    CommunityJoin,
    /// Community join error
    CommunityJoinError {
        agent_name: &'a str,
        slug: &'a str,
        error: String,
    },
    /// Community join registration phase complete
    CommunityJoinComplete { joined: u64 },
}

/// Register the seed operator and all agents.
// FIXME: Split up into smaller functions, cover
pub async fn register_all(
    agents: &mut [Agent],
    client: &AgoraClient,
    email: &str,
    password: &str,
) -> Result<()> {
    // Step 1: Register operator (idempotent)
    info_payload!(RegisterLog::OperatorRegistering { email });
    match client
        .register_operator(email, password, Some("Seed Operator"))
        .await
    {
        Ok(id) => info_payload!(RegisterLog::OperatorRegistered { id, email }),
        Err(e) => {
            // If it's a conflict that's fine, otherwise bail
            let msg = e.to_string();
            if msg.contains("409") || msg.contains("already registered") {
                info_payload!(RegisterLog::OperatorAlreadyRegistered { email })
            } else {
                return Err(e).context("registering operator");
            }
        }
    }

    // Step 2: Register each agent
    let total = agents.len() as u64;
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
            // FIXME: Uses raw Uuid when an ID type exists. Also uses
            // serde_json::Value when a typed response exists.
            Ok(Some(agent_response)) => {
                // Agent already registered. Set and save id. Should never
                // change but also can't hurt.
                agent.agent_id = Some(agent_response.id);
                agent.save_agent_id().await?;
                skipped += 1;
                continue;
            }
            Ok(None) => {
                // Agent does not exist yet
            }
            Err(e) => {
                error_payload!(RegisterLog::AgentCheckError {
                    error: format!("{e:#}"),
                    name: &agent.name
                });
                failed += 1;
                // NOTE: Previously we would not skip here but that was likely a
                // mistake?
                continue;
            }
        }

        // Skip agents with no model — can't register without one
        if agent.model.is_empty() {
            error_payload!(RegisterLog::AgentNoModel { name: &agent.name });
            failed += 1;
            continue;
        }

        match client
            .register_agent(
                email,
                password,
                &agent.name,
                &agent.public_key_hex(),
                Some(&agent.name), // display_name
                Some(&agent.soul.identity),
                Some(&agent.model),
            )
            .await
        {
            Ok(resp) => {
                agent.agent_id = Some(resp.id);
                agent.save_agent_id().await?;
                registered += 1;
                info_payload!(RegisterLog::AgentRegistered {
                    name: &agent.name,
                    id: resp.id
                })
            }
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("already exists") {
                    skipped += 1;
                } else if msg.contains("agent limit reached") {
                    error_payload!(RegisterLog::AgentLimitReached {
                        error: format!(
                            "Agent limit reached! Run: UPDATE operators SET max_agents = 1000 WHERE email = '{email}'"
                        )
                    });
                    return Err(e).context("agent limit reached");
                } else {
                    error_payload!(RegisterLog::AgentRegisterError {
                        name: &agent.name,
                        error: format!("{e:#}")
                    });
                    failed += 1;
                }
            }
        }
    }

    info_payload!(RegisterLog::RegistrationComplete {
        registered,
        failed,
        skipped,
        total
    });

    // Step 3: Join communities
    info_payload!(RegisterLog::CommunityJoin);
    let mut joined = 0;
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

            // NOTE: this is only sane so long as operator agent limit remains
            // low (5 as of writing). If that changes, add a bulk route.
            if let Err(e) = client
                .join_community(agent_id, slug, &agent.signing_key)
                .await
            {
                error_payload!(RegisterLog::CommunityJoinError {
                    agent_name: &agent.name,
                    slug,
                    error: format!("{e:#}")
                })
            } else {
                joined += 1;
            }
        }
    }
    info_payload!(RegisterLog::CommunityJoinComplete { joined });

    Ok(())
}
