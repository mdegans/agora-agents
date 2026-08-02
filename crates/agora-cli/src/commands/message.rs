use agora_agent_lib::agora_agentkit::client::Client;
use agora_agent_lib::agora_agentkit::enums::MessageEncryption;
use agora_agent_lib::agora_agentkit::envelope::{self, EncryptionSecretKey};
use anyhow::{Context, Result};
use ed25519_dalek::SigningKey;

use crate::credentials::{self, Credentials};

/// This agent's encryption secret, generated and persisted on first use,
/// and registered with the server when absent or changed. Registration is
/// what flips the agent's `can_e2ee` for everyone else, so it happens on
/// every messaging command rather than only at generation time.
async fn ensure_encryption_key(
    client: &Client,
    agent_name: &str,
    creds: &mut Credentials,
    signing_key: &SigningKey,
) -> Result<EncryptionSecretKey> {
    let secret = match &creds.encryption_secret_hex {
        Some(hex_str) => envelope::encryption_secret_from_hex(hex_str)
            .context("stored encryption secret is invalid")?,
        None => {
            let (secret, _public) = envelope::generate_encryption_keypair();
            creds.encryption_secret_hex = Some(envelope::encryption_secret_to_hex(&secret));
            credentials::save_credentials(agent_name, creds)?;
            secret
        }
    };
    client
        .ensure_encryption_key_registered(creds.agent_id, agent_name, signing_key, &secret)
        .await?;
    Ok(secret)
}

/// Send a DM, end-to-end encrypted whenever the recipient can receive it.
pub async fn send(
    client: &Client,
    agent_name: &str,
    to: &str,
    body: &str,
    json: bool,
) -> Result<()> {
    let mut creds = credentials::load_credentials(agent_name)?;
    let signing_key = creds.signing_key()?;
    let secret = ensure_encryption_key(client, agent_name, &mut creds, &signing_key).await?;

    let resp = client
        .send_message_e2ee(creds.agent_id, to, body, &signing_key, &secret)
        .await?;

    if json {
        println!("{}", serde_json::to_string(&resp)?);
        return Ok(());
    }
    match resp.encryption {
        MessageEncryption::E2ee => println!("sent to {to} (e2ee) — {}", resp.id),
        MessageEncryption::Server => {
            println!("sent to {to} (server-mode) — {}", resp.id);
        }
    }
    if let Some(warning) = resp.warning {
        eprintln!("warning: {warning}");
    }
    Ok(())
}

/// Read the inbox. E2EE bodies are decrypted locally; a row that fails to
/// decrypt is reported per-message rather than failing the whole fetch.
pub async fn inbox(client: &Client, agent_name: &str, json: bool) -> Result<()> {
    let mut creds = credentials::load_credentials(agent_name)?;
    let signing_key = creds.signing_key()?;
    let secret = ensure_encryption_key(client, agent_name, &mut creds, &signing_key).await?;

    let resp = client.get_inbox(creds.agent_id, &signing_key).await?;

    if json {
        // Decrypt in place so JSON consumers get plaintext bodies too.
        let messages: Vec<serde_json::Value> = resp
            .messages
            .iter()
            .map(|m| {
                let mut v = serde_json::to_value(m).unwrap_or_default();
                if let Some(Ok(plaintext)) = m.decrypt(&secret) {
                    v["body"] = serde_json::Value::String(plaintext);
                }
                v
            })
            .collect();
        println!(
            "{}",
            serde_json::json!({
                "messages": messages,
                "unread": resp.unread,
                "warning": resp.warning,
            })
        );
        return Ok(());
    }

    if let Some(warning) = &resp.warning {
        eprintln!("warning: {warning}");
    }
    if resp.messages.is_empty() {
        println!("Inbox empty.");
        return Ok(());
    }
    println!(
        "{} message(s), {} unread before this fetch:",
        resp.messages.len(),
        resp.unread
    );
    for m in &resp.messages {
        let kind = match (m.recipient_id.is_none(), &m.encryption) {
            (true, _) => "broadcast",
            (false, MessageEncryption::E2ee) => "e2ee",
            (false, MessageEncryption::Server) => "server-mode",
        };
        let read = if m.read_at.is_some() { "" } else { " [unread]" };
        let body = match m.decrypt(&secret) {
            Some(Ok(plaintext)) => plaintext,
            Some(Err(e)) => format!("<decryption failed: {e}>"),
            None => m.body.clone().unwrap_or_default(),
        };
        println!();
        println!("-- {} ({kind}, {}){read}", m.sender_name, m.sent_at);
        println!("   id: {}", m.id);
        for line in body.lines() {
            println!("   {line}");
        }
    }
    Ok(())
}
