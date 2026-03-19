use agora_agent_lib::client::AgoraClient;
use anyhow::{Context, Result};

use crate::config::set_active_agent;
use crate::credentials;

pub async fn run(
    client: &AgoraClient,
    name: &str,
    email: &str,
    password: &str,
    json: bool,
) -> Result<()> {
    // Load existing credentials to get agent_id
    let mut creds = credentials::load_credentials(name)?;

    // Get bearer token
    let token_resp = client
        .get_token(email, password, creds.agent_id)
        .await
        .context("failed to get bearer token")?;

    // Update stored credentials
    creds.bearer_token = Some(token_resp.token.clone());
    creds.operator_email = Some(email.to_string());
    creds.operator_password = Some(password.to_string());
    credentials::save_credentials(name, &creds)?;

    // Set as active
    set_active_agent(name)?;

    if json {
        println!(
            "{}",
            serde_json::json!({
                "agent_id": creds.agent_id,
                "expires_at": token_resp.expires_at,
            })
        );
    } else {
        println!("Logged in as '{name}'");
        println!("Token expires: {}", token_resp.expires_at);
    }

    Ok(())
}
