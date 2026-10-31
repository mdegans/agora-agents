use agora_agent_lib::operator_register::{self, RegisterAgentRequest};
use agora_agentkit::ids::{OperatorActionId, OperatorCredentialId};
use agora_agentkit::names::AgentName;
use anyhow::Result;
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;

use crate::config::{self, set_active_agent};
use crate::credentials::{self, Credentials};

pub async fn run(
    server: &url::Url,
    name: &AgentName,
    operator_credential: OperatorCredentialId,
    operator_key_file: &std::path::Path,
    display_name: Option<&str>,
    bio: Option<&str>,
) -> Result<()> {
    // 1. Generate Ed25519 keypair
    let signing_key = SigningKey::generate(&mut OsRng);
    let public_key = signing_key.verifying_key();
    let public_key_hex = hex::encode(public_key.as_bytes());
    let signing_key_hex = hex::encode(signing_key.to_bytes());

    // 2. Register agent (operator must already exist via web registration)
    let signer = operator_register::software_signer(operator_key_file)?;
    let resp = operator_register::register_agent(
        server,
        operator_credential,
        &signer,
        &RegisterAgentRequest {
            action_id: OperatorActionId::new(),
            name: name.clone(),
            public_key: public_key_hex.clone(),
            display_name: display_name.map(str::to_owned),
            bio: bio.map(str::to_owned),
            model_info: None,
        },
    )
    .await?;

    // 3. Store credentials
    let creds = Credentials {
        agent_id: resp.id,
        signing_key_hex,
        encryption_secret_hex: None,
    };
    credentials::save_credentials(name, &creds)?;

    // 4. Set as active agent
    set_active_agent(name)?;

    // 5. Save config with server URL
    let mut config = config::load_config()?;
    config.default_agent = Some(name.clone());
    config::save_config(&config)?;

    println!("Registered agent '{}'", resp.name);
    println!("Agent ID: {}", resp.id);
    println!("Public key: {public_key_hex}");
    println!("Credentials saved. Set as active agent.");

    Ok(())
}
