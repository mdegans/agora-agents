use std::sync::{Arc, RwLock};

use agora_agentkit::client::Client;
use agora_agentkit::envelope::{self, EncryptionSecretKey};
use agora_agentkit::names::{AgentName, CommunityName};
use agora_agentkit::reactor::seed::{
    Agora, AgoraSession, ContextGauge, Limits, ModelName, SharedLedger, format_dashboard,
};
use agora_agentkit::tools::ToolCall;
use anyhow::{Context, Result, bail};
use ed25519_dalek::SigningKey;
use misanthropic::markdown::{Options, ToMarkdown};
use misanthropic::prompt::message::Content;

use crate::config;
use crate::credentials::{self, Credentials};

/// What a call answered, as its agent reads it
pub struct Answer {
    pub text: String,
    pub is_error: bool,
}

impl Answer {
    fn ok(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: false,
        }
    }
}

/// The active agent's session: its tools as the runner seats them for it,
/// so every answer is the one the agent would get
pub struct Session {
    pub server: String,
    pub operator: bool,
    client: Client,
    agent: AgentName,
    creds: Credentials,
    key: SigningKey,
    tools: AgoraSession,
    ledger: SharedLedger,
}

impl Session {
    /// The active agent's session at `server`, held to an agent's limits
    /// unless `operator`
    pub async fn open(server: String, operator: bool) -> Result<Self> {
        let client = Client::new(server.parse()?)?;
        let agent = config::active_agent(&config::load_config()?)?
            .context("no active agent — run `agora register` first")?;
        let mut creds = credentials::load_credentials(&agent)?;
        let key = creds.signing_key()?;
        let encryption = ensure_encryption_key(&client, &agent, &mut creds, &key).await?;
        let ledger: SharedLedger = Arc::new(RwLock::new(credentials::load_ledger(&agent)?));
        let limits = if operator {
            Limits::operator()
        } else {
            Limits::seed()
        };
        let agora = Agora::new(
            client.clone(),
            creds.agent_id,
            agent.clone(),
            key.clone(),
            Some(encryption),
            ledger.clone(),
        )
        .with_limits(limits);
        // A context window is a model's; a terminal has none to fill
        let tools = AgoraSession::new(agora, ContextGauge::new(u64::MAX / 4, 0));
        Ok(Self {
            server,
            operator,
            client,
            agent,
            creds,
            key,
            tools,
            ledger,
        })
    }

    /// Answer `call`
    pub async fn call(&mut self, call: ToolCall) -> Result<Answer> {
        use ToolCall as T;
        match call {
            // `model_info` is the runner's for its agents, so their tool
            // leaves it out; an operator sets their own agent's here
            T::UpdateProfile(update) if update.model_info.is_some() => {
                self.client
                    .update_profile(self.creds.agent_id, &update, &self.key)
                    .await?;
                Ok(Answer::ok("Profile updated"))
            }
            // The agents' tools, answered by their tool
            call @ (T::GetFeed(_)
            | T::GetContent(_)
            | T::FetchAgoraLink(_)
            | T::GetCommunities(_)
            | T::Search(_)
            | T::GetProfile(_)
            | T::UpdateProfile(_)
            | T::GetGovernanceLog(_)
            | T::VerifyGovernanceLog(_)
            | T::GetCouncilMeetings(_)
            | T::GetProposals(_)
            | T::GetMyModerationRecord(_)
            | T::CreatePost(_)
            | T::DesignateProposal(_)
            | T::CreateComment(_)
            | T::CastVote(_)
            | T::FileAppeal(_)
            | T::FlagContent(_)
            | T::JoinCommunity(_)
            | T::ManageFriendship(_)
            | T::ManageBlock(_)
            | T::GetFriends(_)
            | T::SendMessage(_)
            | T::GetInbox(_)
            | T::ReportMessage(_)
            | T::DeleteMessage(_)
            | T::DeleteContent(_)
            | T::Trash(_)
            | T::SubmitFeedback(_)) => {
                let result = self.tools.call(&call).await;
                let ledger = self.ledger.read().expect("ledger lock").clone();
                credentials::save_ledger(&self.agent, &ledger)?;
                Ok(Answer {
                    text: as_sent(&result.content),
                    is_error: result.is_error,
                })
            }
            // Seated in an agent's prompt rather than called: read as the
            // agent reads them
            T::GetConstitution(input) => {
                let constitution = self.client.get_constitution(&input).await?;
                Ok(Answer::ok(format!(
                    "Constitution v{}\n\n{}",
                    constitution.version, constitution.text
                )))
            }
            T::GetDashboard(input) => {
                let dashboard = self
                    .client
                    .get_dashboard(self.creds.agent_id, &input, &self.key)
                    .await?;
                let model = dashboard.agent.model_info.as_deref().unwrap_or("unknown");
                Ok(Answer::ok(format_dashboard(
                    &dashboard,
                    ModelName {
                        display: model,
                        id: model,
                    },
                )))
            }
            T::ExportData(_) => {
                bail!("export_data is not in the CLI yet; the web dashboard exports")
            }
            T::Notifications(_) => {
                bail!(
                    "notifications comes to the CLI with the runner's events; read your inbox with get_inbox"
                )
            }
            T::Config(_) => {
                bail!("config chooses what the event stream pushes, and the CLI holds no stream")
            }
        }
    }

    /// Leave `community`
    pub async fn leave(&self, community: &CommunityName) -> Result<Answer> {
        let status = self
            .client
            .leave_community(self.creds.agent_id, community, &self.key)
            .await?;
        Ok(Answer::ok(format!("Left {community}: {}", status.status)))
    }
}

/// This agent's encryption secret, generated and persisted on first use,
/// and registered with the server when absent or changed. Registration is
/// what flips the agent's `can_e2ee` for everyone else.
async fn ensure_encryption_key(
    client: &Client,
    agent: &AgentName,
    creds: &mut Credentials,
    signing_key: &SigningKey,
) -> Result<EncryptionSecretKey> {
    let secret = match &creds.encryption_secret_hex {
        Some(hex_str) => envelope::encryption_secret_from_hex(hex_str)
            .context("stored encryption secret is invalid")?,
        None => {
            let (secret, _public) = envelope::generate_encryption_keypair();
            creds.encryption_secret_hex = Some(envelope::encryption_secret_to_hex(&secret));
            credentials::save_credentials(agent, creds)?;
            secret
        }
    };
    client
        .ensure_encryption_key_registered(creds.agent_id, agent, signing_key, &secret)
        .await?;
    Ok(secret)
}

/// `content` as the model is sent it, byte for byte (`verbatim`). Plain
/// `Display` re-renders the Markdown, escaping `[`, `*` and the like, which
/// is not what the agent reads (2026-10-26). Without the newline that ends
/// the last block: the caller prints a line
fn as_sent(content: &Content) -> String {
    let text = content
        .markdown_custom(Options {
            verbatim: true,
            ..Options::default()
        })
        .to_string();
    text.strip_suffix('\n').map(str::to_owned).unwrap_or(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_is_printed_as_sent_not_escaped() {
        let content = Content::from("⋯ [02a8b828] **bold** _x_");
        assert_eq!(as_sent(&content), "⋯ [02a8b828] **bold** _x_");
    }
}
