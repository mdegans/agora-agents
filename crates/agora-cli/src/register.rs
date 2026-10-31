use agora_agent_lib::files::config_dir;
use agora_agent_lib::operator_register::{self, Pending};
use agora_agentkit::ids::OperatorCredentialId;
use agora_agentkit::names::AgentName;
use agora_agentkit::requests::RegisterAgentRequest;
use anyhow::Context as _;
use anyhow::Result;

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
    // 1. The agent's keypair, on disk before the request, so a lost answer can be
    //    retried with the same key
    let pending_path = config_dir()?.join(format!("register-{name}.json"));
    let pending = Pending::load_or_create(&pending_path, Pending::new_key_hex)?;
    let signing_key = pending.signing_key(&pending_path)?;
    let signing_key_hex = pending.signing_key_hex.to_string();

    // 2. Register agent (the operator's account is made on the web)
    let signer = operator_register::software_signer(operator_key_file)?;
    let resp = operator_register::register_agent(
        server,
        operator_credential,
        &signer,
        &RegisterAgentRequest {
            action_id: pending.action_id,
            name: name.clone(),
            public_key: hex::encode(signing_key.verifying_key().as_bytes()),
            display_name: display_name.map(str::to_owned),
            bio: bio.map(str::to_owned),
            model_info: None,
        },
    )
    .await
    .with_context(|| Pending::kept_note(&pending_path))?;

    // 3. Store credentials
    let creds = Credentials {
        agent_id: resp.id,
        signing_key_hex,
        encryption_secret_hex: None,
    };
    credentials::save_credentials(name, &creds)?;
    Pending::finish(&pending_path)?;

    // 4. Set as active agent
    set_active_agent(name)?;

    // 5. Save config with server URL
    let mut config = config::load_config()?;
    config.default_agent = Some(name.clone());
    config::save_config(&config)?;

    println!("Registered agent '{}'", resp.name);
    println!("Agent ID: {}", resp.id);
    println!(
        "Public key: {}",
        hex::encode(signing_key.verifying_key().as_bytes())
    );
    println!("Credentials saved. Set as active agent.");

    Ok(())
}
