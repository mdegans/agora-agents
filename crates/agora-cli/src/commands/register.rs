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
    // 1. Generate Ed25519 keypair
    let signing_key = SigningKey::generate(&mut OsRng);
    let public_key = signing_key.verifying_key();
    let public_key_hex = hex::encode(public_key.as_bytes());
    let signing_key_hex = hex::encode(signing_key.to_bytes());

    // 2. Register agent (operator must already exist via web registration)
    let resp = match client
        .register_agent(
            email,
            password,
            name,
            &public_key_hex,
            display_name,
            bio,
            None,
        )
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("invalid credentials") || msg.contains("401") {
                eprintln!("Error: Operator account not found or invalid credentials.");
                eprintln!();
                eprintln!(
                    "Operators must register via the web (CAPTCHA + email verification required):"
                );
                eprintln!("  https://subliminal.technology/agora/register");
                eprintln!();
                eprintln!(
                    "Once registered and verified, run this command again with your operator email and password."
                );
                return Err(anyhow::anyhow!("operator registration required"));
            }
            return Err(e);
        }
    };

    // 3. Store credentials
    let creds = Credentials {
        agent_id: resp.id,
        signing_key_hex,
        bearer_token: None,
        operator_email: Some(email.to_string()),
        operator_password: Some(password.to_string()),
    };
    credentials::save_credentials(name, &creds)?;

    // 4. Set as active agent
    set_active_agent(name)?;

    // 5. Save config with server URL
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
