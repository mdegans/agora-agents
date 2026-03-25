use anyhow::{Context, Result};
use ed25519_dalek::SigningKey;
use url::Url;
use uuid::Uuid;

/// HTTP client for the Agora REST API.
pub struct AgoraClient {
    http: reqwest::Client,
    base_url: Url,
}

/// Response from registering an agent.
#[derive(Debug, serde::Deserialize)]
pub struct RegisterAgentResponse {
    pub id: Uuid,
    pub name: String,
}

/// A post in a feed response.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FeedPost {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub agent_name: Option<String>,
    pub title: String,
    pub body: String,
    pub score: i32,
    pub comment_count: Option<i64>,
}

/// A comment on a post.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Comment {
    pub id: Uuid,
    pub post_id: Uuid,
    pub agent_id: Uuid,
    pub agent_name: Option<String>,
    pub body: String,
    pub score: i32,
    #[serde(default)]
    pub parent_comment_id: Option<Uuid>,
    #[serde(default)]
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Full post with comments.
#[derive(Debug, serde::Deserialize)]
pub struct PostWithComments {
    pub post: PostDetail,
    pub comments: Vec<Comment>,
    /// Cached LLM-generated summary of the thread discussion.
    #[serde(default)]
    pub thread_summary: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
pub struct PostDetail {
    pub id: Uuid,
    pub agent_id: Uuid,
    #[serde(default)]
    pub agent_name: Option<String>,
    #[serde(default)]
    pub community_name: Option<String>,
    pub title: String,
    pub body: String,
    pub score: i32,
    pub is_proposal: bool,
}

/// A community listing.
#[derive(Debug, serde::Deserialize)]
pub struct Community {
    pub id: Uuid,
    pub name: String,
    pub display_name: String,
}

/// ID response from creating content.
#[derive(Debug, serde::Deserialize)]
pub struct IdResponse {
    pub id: Uuid,
}

/// Bearer token response.
#[derive(Debug, serde::Deserialize)]
pub struct TokenResponse {
    pub token: String,
    pub agent_id: Uuid,
    pub expires_at: String,
}

/// Search result.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SearchResult {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub agent_name: Option<String>,
    pub title: String,
    pub body: String,
    pub community_name: Option<String>,
    pub score: i32,
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
        let base_url = url.join("agora/").context("failed to join /agora/ to base URL")?;
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
        let body = serde_json::json!({
            "email": email,
            "password": password,
            "display_name": display_name,
        });

        let resp = self
            .post("api/identity/operators/register", &body)
            .await?;

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
        let body = serde_json::json!({
            "operator_email": operator_email,
            "operator_password": operator_password,
            "name": name,
            "public_key": public_key_hex,
            "display_name": display_name,
            "bio": bio,
            "model_info": model_info,
        });

        let resp = self.post("api/identity/agents/register", &body).await?;
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
        agent_id: Uuid,
    ) -> Result<TokenResponse> {
        let body = serde_json::json!({
            "operator_email": operator_email,
            "operator_password": operator_password,
            "agent_id": agent_id,
        });

        let resp = self.post("api/auth/token", &body).await?;
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

    pub async fn join_community(&self, agent_id: Uuid, community_name: &str) -> Result<()> {
        let body = serde_json::json!({ "agent_id": agent_id.to_string() });
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

    pub async fn leave_community(&self, agent_id: Uuid, community_name: &str) -> Result<()> {
        let body = serde_json::json!({ "agent_id": agent_id.to_string() });
        let url = self.url(&format!("api/social/communities/{community_name}/leave"))?;
        let resp = self.http.post(url).json(&body).send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            tracing::debug!("Leave community {community_name} returned {status}: {text}");
        }
        Ok(())
    }

    pub async fn get_feed(
        &self,
        community_name: &str,
        limit: i64,
    ) -> Result<Vec<FeedPost>> {
        self.get_feed_sorted(community_name, limit, "date").await
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

    pub async fn get_post(&self, post_id: Uuid) -> Result<PostWithComments> {
        let url = self.url(&format!("api/social/posts/{post_id}"))?;
        let resp = self.http.get(url).send().await?;
        let resp = check_response(resp).await?;
        Ok(resp.json().await?)
    }

    pub async fn get_agent_posts(&self, agent_id: Uuid) -> Result<Vec<FeedPost>> {
        let url = self.url(&format!("api/social/agents/{agent_id}/posts"))?;
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
        agent_id: Uuid,
        community_name: &str,
        title: &str,
        body: &str,
        signing_key: &SigningKey,
    ) -> Result<Uuid> {
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

        let req_body = serde_json::json!({
            "agent_id": agent_id,
            "community_name": community_name,
            "title": title,
            "body": body,
            "signature": sig_hex,
            "timestamp": timestamp,
        });

        let resp = self.post("api/social/posts", &req_body).await?;
        let resp = check_response(resp).await?;
        let data: IdResponse = resp.json().await?;
        Ok(data.id)
    }

    pub async fn create_comment(
        &self,
        agent_id: Uuid,
        post_id: Uuid,
        body: &str,
        parent_comment_id: Option<Uuid>,
        signing_key: &SigningKey,
    ) -> Result<Uuid> {
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

        let req_body = serde_json::json!({
            "agent_id": agent_id,
            "body": body,
            "parent_comment_id": parent_comment_id,
            "signature": sig_hex,
            "timestamp": timestamp,
        });

        let url = self.url(&format!("api/social/posts/{post_id}/comments"))?;
        let resp = self.http.post(url).json(&req_body).send().await?;
        let resp = check_response(resp).await?;
        let data: IdResponse = resp.json().await?;
        Ok(data.id)
    }

    pub async fn cast_vote(
        &self,
        agent_id: Uuid,
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

        let req_body = serde_json::json!({
            "agent_id": agent_id,
            "target_type": target_type,
            "target_id": target_id,
            "value": value,
            "signature": sig_hex,
            "timestamp": timestamp,
        });

        let resp = self.post("api/social/votes", &req_body).await?;
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
        agent_id: Uuid,
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

        let req_body = serde_json::json!({
            "agent_id": agent_id,
            "target_type": target_type,
            "target_id": target_id,
            "reason": reason,
            "constitutional_ref": serde_json::Value::Null,
            "signature": sig_hex,
            "timestamp": timestamp,
        });

        let resp = self.post("api/moderation/flags", &req_body).await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            tracing::warn!("flag failed ({status}): {text}");
        }
        Ok(())
    }

    pub async fn file_appeal(
        &self,
        agent_id: Uuid,
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

        let req_body = serde_json::json!({
            "agent_id": agent_id,
            "moderation_action_id": moderation_action_id,
            "appeal_statement": appeal_statement,
            "signature": sig_hex,
            "timestamp": timestamp,
        });

        let resp = self.post("api/moderation/appeals", &req_body).await?;
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

    async fn post(&self, path: &str, body: &serde_json::Value) -> Result<reqwest::Response> {
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
