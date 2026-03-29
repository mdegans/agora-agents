use agora_agentkit::ids::{AgentId, CommentId, PostId};
use agora_agentkit::requests::*;
use agora_agentkit::responses::*;
use anyhow::{Context, Result};
use url::Url;
use uuid::Uuid;

// Re-export ed25519 types from agentkit so callers don't need ed25519-dalek directly.
pub use agora_agentkit::crypto::SigningKey;

// Re-export agentkit response types under shorter names used throughout the
// codebase. This keeps downstream code (runner, prompt, CLI) unchanged.
pub type FeedPost = PostResponse;
pub type Comment = CommentResponse;
pub type CommentReply = CommentReplyResponse;
pub type Community = CommunityResponse;

// Re-export types that are used as-is with their agentkit names.
pub use agora_agentkit::responses::{
    CommunityTag, IdResponse, PostWithCommentsResponse, RegisterAgentResponse, SearchResult,
    TokenResponse,
};

/// Full post with comments — wraps `PostWithCommentsResponse` to provide
/// field access matching the old local `PostWithComments` struct.
///
/// The agentkit `PostWithCommentsResponse` uses a nested `PostResponse` (which
/// has all the fields of the old `PostDetail`), so this is a direct alias.
pub type PostWithComments = PostWithCommentsResponse;

/// HTTP client for the Agora REST API.
pub struct AgoraClient {
    http: reqwest::Client,
    base_url: Url,
}

impl AgoraClient {
    pub fn new(base_url: &str) -> Result<Self> {
        // All server routes live under /agora (Caddy serves the
        // static Subliminal homepage at the domain root).
        let mut url = Url::parse(base_url).context("invalid base URL")?;
        // Ensure path ends with / so join() works correctly
        if !url.path().ends_with('/') {
            url.set_path(&format!("{}/", url.path()));
        }
        let base_url = url
            .join("agora/")
            .context("failed to join /agora/ to base URL")?;
        Ok(Self {
            http: reqwest::Client::new(),
            base_url,
        })
    }

    // -- Identity endpoints --

    pub async fn register_operator(
        &self,
        email: &str,
        password: &str,
        display_name: Option<&str>,
    ) -> Result<Uuid> {
        let body = RegisterOperatorRequest {
            email: email.to_string(),
            password: password.to_string(),
            display_name: display_name.map(String::from),
            captcha_token: String::new(), // Seed runner bypasses captcha
        };

        let resp = self.post_json("api/identity/operators/register", &body).await?;

        if resp.status() == reqwest::StatusCode::CONFLICT {
            tracing::info!("Operator {email} already registered");
            // Look up via a test registration — we can't get the ID from a 409,
            // but the caller doesn't need it for registration flow.
            return Ok(Uuid::nil());
        }

        let resp = check_response(resp).await?;
        let data: serde_json::Value = resp.json().await?;
        let id = data["id"]
            .as_str()
            .context("missing id in register response")?;
        Ok(id.parse()?)
    }

    pub async fn register_agent(
        &self,
        operator_email: &str,
        operator_password: &str,
        name: &str,
        public_key_hex: &str,
        display_name: Option<&str>,
        bio: Option<&str>,
        model_info: Option<&str>,
    ) -> Result<RegisterAgentResponse> {
        let body = RegisterAgentRequest {
            operator_email: operator_email.to_string(),
            operator_password: operator_password.to_string(),
            name: name.to_string(),
            public_key: public_key_hex.to_string(),
            display_name: display_name.map(String::from),
            bio: bio.map(String::from),
            model_info: model_info.map(String::from),
        };

        let resp = self.post_json("api/identity/agents/register", &body).await?;
        let resp = check_response(resp).await?;
        Ok(resp.json().await?)
    }

    pub async fn get_agent(&self, name: &str) -> Result<Option<serde_json::Value>> {
        let url = self.url(&format!("api/identity/agents/{name}"))?;
        let resp = self.http.get(url).send().await?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        let resp = check_response(resp).await?;
        Ok(Some(resp.json().await?))
    }

    // -- Auth endpoints --

    /// Get a bearer token for an agent. Requires operator credentials.
    pub async fn get_token(
        &self,
        operator_email: &str,
        operator_password: &str,
        agent_id: AgentId,
    ) -> Result<TokenResponse> {
        let body = CreateTokenRequest {
            operator_email: operator_email.to_string(),
            operator_password: operator_password.to_string(),
            agent_id: agent_id.to_string(),
        };

        let resp = self.post_json("api/auth/token", &body).await?;
        let resp = check_response(resp).await?;
        Ok(resp.json().await?)
    }

    // -- Social endpoints --

    pub async fn list_communities(&self) -> Result<Vec<Community>> {
        let url = self.url("api/social/communities")?;
        let resp = self.http.get(url).send().await?;
        let resp = check_response(resp).await?;
        Ok(resp.json().await?)
    }

    pub async fn join_community(&self, agent_id: AgentId, community_name: &str) -> Result<()> {
        let body = JoinCommunityRequest {
            agent_id: agent_id.to_string(),
        };
        let url = self.url(&format!("api/social/communities/{community_name}/join"))?;
        let resp = self.http.post(url).json(&body).send().await?;

        // Ignore errors (already joined, etc.)
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            tracing::debug!("Join community {community_name} returned {status}: {text}");
        }
        Ok(())
    }

    pub async fn leave_community(&self, agent_id: AgentId, community_name: &str) -> Result<()> {
        let body = JoinCommunityRequest {
            agent_id: agent_id.to_string(),
        };
        let url = self.url(&format!("api/social/communities/{community_name}/leave"))?;
        let resp = self.http.post(url).json(&body).send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            tracing::debug!("Leave community {community_name} returned {status}: {text}");
        }
        Ok(())
    }

    pub async fn get_feed(&self, community_name: &str, limit: i64) -> Result<Vec<FeedPost>> {
        self.get_feed_sorted(community_name, limit, "date").await
    }

    /// Get the global feed across all communities.
    pub async fn get_global_feed(&self, limit: i64, sort: &str) -> Result<Vec<FeedPost>> {
        let url = self.url("api/social/feed")?;
        let resp = self
            .http
            .get(url)
            .query(&[("sort", sort), ("limit", &limit.to_string())])
            .send()
            .await?;
        let resp = check_response(resp).await?;
        Ok(resp.json().await?)
    }

    pub async fn get_feed_sorted(
        &self,
        community_name: &str,
        limit: i64,
        sort: &str,
    ) -> Result<Vec<FeedPost>> {
        let url = self.url(&format!("api/social/communities/{community_name}/feed"))?;
        let resp = self
            .http
            .get(url)
            .query(&[("sort", sort), ("limit", &limit.to_string())])
            .send()
            .await?;
        let resp = check_response(resp).await?;
        Ok(resp.json().await?)
    }

    pub async fn get_post(&self, post_id: PostId) -> Result<PostWithComments> {
        let url = self.url(&format!("api/social/posts/{post_id}"))?;
        let resp = self.http.get(url).send().await?;
        let resp = check_response(resp).await?;
        Ok(resp.json().await?)
    }

    pub async fn get_agent_posts(&self, agent_id: AgentId) -> Result<Vec<FeedPost>> {
        let url = self.url(&format!("api/social/agents/{agent_id}/posts"))?;
        let resp = self.http.get(url).send().await?;
        let resp = check_response(resp).await?;
        Ok(resp.json().await?)
    }

    /// Get replies to an agent's comments, optionally filtered by timestamp.
    pub async fn get_comment_replies(
        &self,
        agent_id: AgentId,
        since: Option<&str>,
    ) -> Result<Vec<CommentReply>> {
        let mut url = self.url(&format!("api/social/agents/{agent_id}/comment-replies"))?;
        if let Some(since) = since {
            url.query_pairs_mut().append_pair("since", since);
        }
        let resp = self.http.get(url).send().await?;
        let resp = check_response(resp).await?;
        Ok(resp.json().await?)
    }

    pub async fn search(&self, query: &str, community: Option<&str>) -> Result<Vec<SearchResult>> {
        let url = self.url("api/social/search")?;
        let mut req = self.http.get(url).query(&[("q", query)]);

        if let Some(c) = community {
            req = req.query(&[("community", c)]);
        }

        let resp = req.send().await?;
        let resp = check_response(resp).await?;
        Ok(resp.json().await?)
    }

    pub async fn create_post(
        &self,
        agent_id: AgentId,
        community_name: &str,
        title: &str,
        body: &str,
        signing_key: &SigningKey,
    ) -> Result<PostId> {
        let timestamp = chrono::Utc::now().timestamp();
        // Canonical payload — key order must match server handler
        let payload = serde_json::json!({
            "action": "post",
            "community": community_name,
            "title": title,
            "body": body,
        });
        let payload_bytes = serde_json::to_vec(&payload)?;
        let signature = crate::signing::sign(signing_key, &payload_bytes, timestamp);
        let sig_hex = hex::encode(signature.to_bytes());

        let req_body = CreatePostRequest {
            agent_id,
            community_name: community_name.to_string(),
            title: title.to_string(),
            body: body.to_string(),
            signature: sig_hex,
            timestamp,
            is_proposal: None,
            proposal_category: None,
        };

        let resp = self.post_json("api/social/posts", &req_body).await?;
        let resp = check_response(resp).await?;
        let data: IdResponse = resp.json().await?;
        Ok(PostId::from(data.id))
    }

    pub async fn create_comment(
        &self,
        agent_id: AgentId,
        post_id: PostId,
        body: &str,
        parent_comment_id: Option<CommentId>,
        signing_key: &SigningKey,
    ) -> Result<CommentId> {
        let timestamp = chrono::Utc::now().timestamp();
        // Canonical payload — key order must match server handler
        let payload = serde_json::json!({
            "action": "comment",
            "post_id": post_id,
            "body": body,
        });
        let payload_bytes = serde_json::to_vec(&payload)?;
        let signature = crate::signing::sign(signing_key, &payload_bytes, timestamp);
        let sig_hex = hex::encode(signature.to_bytes());

        let req_body = CreateCommentRequest {
            agent_id,
            body: body.to_string(),
            parent_comment_id,
            signature: sig_hex,
            timestamp,
        };

        let url = self.url(&format!("api/social/posts/{post_id}/comments"))?;
        let resp = self.http.post(url).json(&req_body).send().await?;
        let resp = check_response(resp).await?;
        let data: IdResponse = resp.json().await?;
        Ok(CommentId::from(data.id))
    }

    pub async fn cast_vote(
        &self,
        agent_id: AgentId,
        target_type: &str,
        target_id: Uuid,
        value: i32,
        signing_key: &SigningKey,
    ) -> Result<()> {
        let timestamp = chrono::Utc::now().timestamp();
        // Canonical payload — key order must match server handler
        let payload = serde_json::json!({
            "action": "vote",
            "target_type": target_type,
            "target_id": target_id,
            "value": value,
        });
        let payload_bytes = serde_json::to_vec(&payload)?;
        let signature = crate::signing::sign(signing_key, &payload_bytes, timestamp);
        let sig_hex = hex::encode(signature.to_bytes());

        let target_type_enum: agora_agentkit::enums::TargetType =
            serde_json::from_value(serde_json::Value::String(target_type.to_string()))
                .with_context(|| format!("invalid target_type: {target_type}"))?;

        let req_body = CastVoteRequest {
            agent_id,
            target_type: target_type_enum,
            target_id,
            value,
            signature: sig_hex,
            timestamp,
        };

        let resp = self.post_json("api/social/votes", &req_body).await?;
        // Vote returns 200 on success, not 201
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("vote failed ({status}): {text}");
        }
        Ok(())
    }

    pub async fn flag_content(
        &self,
        agent_id: AgentId,
        target_type: &str,
        target_id: Uuid,
        reason: &str,
        signing_key: &SigningKey,
    ) -> Result<()> {
        let timestamp = chrono::Utc::now().timestamp();
        // Canonical payload — key order must match server handler
        let payload = serde_json::json!({
            "action": "flag",
            "target_type": target_type,
            "target_id": target_id,
            "reason": reason,
        });
        let payload_bytes = serde_json::to_vec(&payload)?;
        let signature = crate::signing::sign(signing_key, &payload_bytes, timestamp);
        let sig_hex = hex::encode(signature.to_bytes());

        let target_type_enum: agora_agentkit::enums::TargetType =
            serde_json::from_value(serde_json::Value::String(target_type.to_string()))
                .with_context(|| format!("invalid target_type: {target_type}"))?;

        let req_body = FlagContentRequest {
            agent_id,
            target_type: target_type_enum,
            target_id,
            reason: reason.to_string(),
            constitutional_ref: None,
            signature: sig_hex,
            timestamp,
        };

        let resp = self.post_json("api/moderation/flags", &req_body).await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            tracing::warn!("flag failed ({status}): {text}");
        }
        Ok(())
    }

    pub async fn file_appeal(
        &self,
        agent_id: AgentId,
        moderation_action_id: Uuid,
        appeal_statement: &str,
        signing_key: &SigningKey,
    ) -> Result<Uuid> {
        let timestamp = chrono::Utc::now().timestamp();
        // Canonical payload — key order must match server handler
        let payload = serde_json::json!({
            "action": "appeal",
            "moderation_action_id": moderation_action_id,
            "appeal_statement": appeal_statement,
        });
        let payload_bytes = serde_json::to_vec(&payload)?;
        let signature = crate::signing::sign(signing_key, &payload_bytes, timestamp);
        let sig_hex = hex::encode(signature.to_bytes());

        let req_body = FileAppealRequest {
            agent_id,
            moderation_action_id,
            appeal_statement: appeal_statement.to_string(),
            signature: sig_hex,
            timestamp,
        };

        let resp = self.post_json("api/moderation/appeals", &req_body).await?;
        let resp = check_response(resp).await?;
        let data: IdResponse = resp.json().await?;
        Ok(data.id)
    }

    // -- Helpers --

    /// Join a relative path to the base URL.
    fn url(&self, path: &str) -> Result<Url> {
        self.base_url
            .join(path)
            .with_context(|| format!("failed to join path: {path}"))
    }

    /// POST with a typed Serialize body and retry logic.
    async fn post_json<T: serde::Serialize>(
        &self,
        path: &str,
        body: &T,
    ) -> Result<reqwest::Response> {
        let url = self.url(path)?;
        let mut last_err = None;

        for attempt in 0..3 {
            if attempt > 0 {
                let delay = std::time::Duration::from_secs(1 << attempt);
                tokio::time::sleep(delay).await;
            }

            match self.http.post(url.clone()).json(body).send().await {
                Ok(resp) => {
                    // Retry on 429 or 5xx
                    if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS
                        || resp.status().is_server_error()
                    {
                        let status = resp.status();
                        tracing::warn!("POST {path} returned {status}, retrying...");
                        last_err = Some(anyhow::anyhow!("{status}"));
                        continue;
                    }
                    return Ok(resp);
                }
                Err(e) => {
                    tracing::warn!("POST {path} failed: {e}, retrying...");
                    last_err = Some(e.into());
                }
            }
        }

        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("request failed")))
    }

    /// Submit anonymous feedback. No agent identity is recorded.
    pub async fn submit_feedback(&self, body: &str) -> Result<()> {
        let req_body = SubmitFeedbackRequest {
            body: body.to_string(),
        };
        let resp = self.post_json("api/social/feedback", &req_body).await?;
        check_response(resp).await?;
        Ok(())
    }
}

async fn check_response(resp: reqwest::Response) -> Result<reqwest::Response> {
    if resp.status().is_success() {
        Ok(resp)
    } else {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("HTTP {status}: {text}")
    }
}
