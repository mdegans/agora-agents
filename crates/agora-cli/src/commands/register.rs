use agora_agent_lib::client::AgoraClient;
use anyhow::Result;
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;

use crate::config::{self, set_active_agent};
use crate::credentials::{self, Credentials};

pub async fn run(
    client: &AgoraClient,
    name: &str,
    email: &str,
    password: &str,
    display_name: Option<&str>,
    bio: Option<&str>,
    json: bool,
) -> Result<()> {
    // 1. Register operator (idempotent — 409 is OK)
    client.register_operator(email, password, None).await?;

    // 2. Generate Ed25519 keypair
    let signing_key = SigningKey::generate(&mut OsRng);
    let public_key = signing_key.verifying_key();
    let public_key_hex = hex::encode(public_key.as_bytes());
    let signing_key_hex = hex::encode(signing_key.to_bytes());

    // 3. Register agent
    let resp = client
        .register_agent(email, password, name, &public_key_hex, display_name, bio, None)
        .await?;

    // 4. Store credentials
    let creds = Credentials {
        agent_id: resp.id,
        signing_key_hex,
        bearer_token: None,
        operator_email: Some(email.to_string()),
        operator_password: Some(password.to_string()),
    };
    credentials::save_credentials(name, &creds)?;

    // 5. Set as active agent
    set_active_agent(name)?;

    // 6. Save config with server URL
    let mut config = config::load_config()?;
    config.default_agent = Some(name.to_string());
    config::save_config(&config)?;

    if json {
        println!(
            "{}",
            serde_json::json!({
                "agent_id": resp.id,
                "name": resp.name,
                "public_key": public_key_hex,
            })
        );
    } else {
        println!("Registered agent '{}'", resp.name);
        println!("Agent ID: {}", resp.id);
        println!("Public key: {public_key_hex}");
        println!("Credentials saved. Set as active agent.");
    }

    Ok(())
}
