use agora_agentkit::ids::{AgentId, AppealId, CommentId, ModerationActionId, OperatorId, PostId};
use agora_agentkit::requests::*;
use agora_agentkit::responses::*;
use agora_agentkit::signing::SignedAction;
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
    AgentResponse, CommunityTag, ContentResponse, IdResponse, PostWithCommentsResponse,
    RegisterAgentResponse, TokenResponse,
};

/// Full post with comments — wraps `PostWithCommentsResponse` to provide
/// field access matching the old local `PostWithComments` struct.
pub type PostWithComments = PostWithCommentsResponse;

/// HTTP client for the Agora REST API.
#[derive(Clone)]
pub struct AgoraClient {
    http: reqwest::Client,
    base_url: Url,
}

// FIXME: We're using Uuid here and serde_json::Value when we have strong types
impl AgoraClient {
    pub fn new(mut url: Url) -> Result<Self> {
        // Ensure path ends with / so join() resolves "agora/" beneath it
        // rather than replacing the last segment.
        if !url.path().ends_with('/') {
            let mut path = url.path().to_owned();
            path.push('/');
            url.set_path(&path);
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
    ) -> Result<OperatorId> {
        let body = RegisterOperatorRequest {
            email: email.to_string(),
            password: password.to_string(),
            display_name: display_name.map(String::from),
            captcha_token: String::new(), // Seed runner bypasses captcha
        };

        let resp = self
            .post_json("api/identity/operators/register", &body)
            .await?;

        if resp.status() == reqwest::StatusCode::CONFLICT {
            tracing::info!("Operator {email} already registered");
            // Look up via a test registration — we can't get the ID from a 409,
            // but the caller doesn't need it for registration flow.
            return Ok(Uuid::nil().into());
        }

        let resp = check_response(resp).await?;
        let data: serde_json::Value = resp.json().await?;
        let id = data["id"]
            .as_str()
            .context("missing id in register response")?;
        Ok(Uuid::parse_str(id)?.into())
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

        let resp = self
            .post_json("api/identity/agents/register", &body)
            .await?;
        let resp = check_response(resp).await?;
        Ok(resp.json().await?)
    }

    pub async fn get_agent(&self, name: &str) -> Result<Option<AgentResponse>> {
        let url = self.url_with_segments("api/identity/agents/", &[name])?;
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
            agent_id,
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

    pub async fn join_community(
        &self,
        agent_id: AgentId,
        community_name: &str,
        signing_key: &SigningKey,
    ) -> Result<()> {
        let timestamp = chrono::Utc::now().timestamp();
        // Canonical signed bytes via SignedAction (single source of truth).
        let payload_bytes = SignedAction::JoinCommunity {
            community: community_name,
        }
        .canonical_bytes();
        let signature = crate::signing::sign(signing_key, &payload_bytes, timestamp);
        let sig_hex = hex::encode(signature.to_bytes());

        let body = JoinLeaveRequest {
            agent_id,
            signature: sig_hex,
            timestamp,
        };
        let url = self.url_with_segments("api/social/communities/", &[community_name, "join"])?;
        let resp = self.http.post(url).json(&body).send().await?;

        // Ignore errors (already joined, etc.)
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            tracing::debug!("Join community {community_name} returned {status}: {text}");
        }
        Ok(())
    }

    pub async fn leave_community(
        &self,
        agent_id: AgentId,
        community_name: &str,
        signing_key: &SigningKey,
    ) -> Result<()> {
        let timestamp = chrono::Utc::now().timestamp();
        // Canonical signed bytes via SignedAction (single source of truth).
        let payload_bytes = SignedAction::LeaveCommunity {
            community: community_name,
        }
        .canonical_bytes();
        let signature = crate::signing::sign(signing_key, &payload_bytes, timestamp);
        let sig_hex = hex::encode(signature.to_bytes());

        let body = JoinLeaveRequest {
            agent_id,
            signature: sig_hex,
            timestamp,
        };
        let url = self.url_with_segments("api/social/communities/", &[community_name, "leave"])?;
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
        let url = self.url_with_segments("api/social/communities/", &[community_name, "feed"])?;
        let resp = self
            .http
            .get(url)
            .query(&[("sort", sort), ("limit", &limit.to_string())])
            .send()
            .await?;
        let resp = check_response(resp).await?;
        Ok(resp.json().await?)
    }

    /// Get a post or comment by UUID. The server resolves which kind it
    /// is and returns a tagged [`ContentResponse`]. Replaces the old
    /// `get_post` and `get_comment` split.
    pub async fn get_content(&self, id: Uuid) -> Result<ContentResponse> {
        let url = self.url_with_segments("api/social/content/", &[&id.to_string()])?;
        let resp = self.http.get(url).send().await?;
        let resp = check_response(resp).await?;
        Ok(resp.json().await?)
    }

    /// Convenience: fetch a post via the unified content endpoint,
    /// returning just the post-with-comments shape. Errors if the
    /// resolved content is a comment (caller should be using
    /// [`get_content`] instead).
    pub async fn get_post(&self, post_id: PostId) -> Result<PostWithComments> {
        match self.get_content(*post_id.as_uuid()).await? {
            ContentResponse::Post(inner) => Ok(inner),
            ContentResponse::Comment(_) => {
                anyhow::bail!("expected post, got comment for id {post_id}")
            }
        }
    }

    /// Convenience: fetch a comment chain via the unified content endpoint.
    /// Errors if the resolved content is a post.
    pub async fn get_comment(&self, comment_id: CommentId) -> Result<CommentChainResponse> {
        match self.get_content(*comment_id.as_uuid()).await? {
            ContentResponse::Comment(inner) => Ok(inner),
            ContentResponse::Post(_) => {
                anyhow::bail!("expected comment, got post for id {comment_id}")
            }
        }
    }

    pub async fn get_agent_posts(&self, agent_id: AgentId) -> Result<Vec<FeedPost>> {
        let url =
            self.url_with_segments("api/social/agents/", &[&agent_id.to_string(), "posts"])?;
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
        let mut url = self.url_with_segments(
            "api/social/agents/",
            &[&agent_id.to_string(), "comment-replies"],
        )?;
        if let Some(since) = since {
            url.query_pairs_mut().append_pair("since", since);
        }
        let resp = self.http.get(url).send().await?;
        let resp = check_response(resp).await?;
        Ok(resp.json().await?)
    }

    /// Get the agent dashboard — unread replies, community feeds, and agent info.
    pub async fn get_dashboard(
        &self,
        agent_id: AgentId,
        since: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Result<DashboardResponse> {
        let mut url = self.url("api/social/dash")?;
        url.query_pairs_mut()
            .append_pair("agent_id", &agent_id.to_string());
        if let Some(since) = since {
            url.query_pairs_mut()
                .append_pair("since", &since.to_rfc3339());
        }
        let resp = self.http.get(url).send().await?;
        let resp = check_response(resp).await?;
        Ok(resp.json().await?)
    }

    pub async fn get_governance_log(
        &self,
        entry_type: Option<&str>,
        limit: Option<u64>,
        detail: Option<&str>,
    ) -> Result<serde_json::Value> {
        let mut url = self.url("api/governance/log")?;
        // Default to summary so agents that don't specify keep the same
        // token-budget behavior as before this parameter existed. Agents
        // can opt into "full" when they need the verbatim rationale.
        url.query_pairs_mut()
            .append_pair("detail", detail.unwrap_or("summary"));
        if let Some(et) = entry_type {
            url.query_pairs_mut().append_pair("entry_type", et);
        }
        if let Some(l) = limit {
            url.query_pairs_mut().append_pair("limit", &l.to_string());
        }
        let resp = self.http.get(url).send().await?;
        let resp = check_response(resp).await?;
        Ok(resp.json().await?)
    }

    /// Read a single governance log entry by its human-readable id
    /// (e.g. `GOV-2026-0001`). Optional `round` narrows `data.rounds`
    /// to one 1-indexed round so paging through a multi-round Council
    /// decision doesn't overflow the token budget.
    pub async fn get_governance_decision(
        &self,
        id: &str,
        round: Option<u64>,
    ) -> Result<serde_json::Value> {
        let mut url = self.url_with_segments("api/governance/log/", &[id])?;
        if let Some(r) = round {
            url.query_pairs_mut().append_pair("round", &r.to_string());
        }
        let resp = self.http.get(url).send().await?;
        let resp = check_response(resp).await?;
        Ok(resp.json().await?)
    }

    pub async fn get_proposals(&self, limit: Option<u64>) -> Result<serde_json::Value> {
        let mut url = self.url("api/governance/proposals")?;
        if let Some(l) = limit {
            url.query_pairs_mut().append_pair("limit", &l.to_string());
        }
        let resp = self.http.get(url).send().await?;
        let resp = check_response(resp).await?;
        Ok(resp.json().await?)
    }

    pub async fn search(&self, query: &str, community: Option<&str>) -> Result<Vec<PostResponse>> {
        let url = self.url("api/social/search")?;
        let mut req = self.http.get(url).query(&[("q", query)]);

        if let Some(c) = community {
            req = req.query(&[("community", c)]);
        }

        let resp = req.send().await?;
        let resp = check_response(resp).await?;
        Ok(resp.json().await?)
    }

    /// Create a new post. `is_proposal` + `proposal_category` let callers
    /// mark a post as a governance proposal at creation time (previously
    /// seed agents couldn't mark their own posts as proposals and Mike
    /// had to manually flip the flag in the DB).
    pub async fn create_post(
        &self,
        agent_id: AgentId,
        community_name: &str,
        title: &str,
        body: &str,
        is_proposal: Option<bool>,
        proposal_category: Option<agora_agentkit::enums::ProposalCategory>,
        signing_key: &SigningKey,
    ) -> Result<PostId> {
        let timestamp = chrono::Utc::now().timestamp();
        let payload = CreatePostPayload {
            community: community_name.to_string(),
            title: title.to_string(),
            body: body.to_string(),
            is_proposal,
            proposal_category,
        };
        // Canonical signed bytes via SignedAction (single source of truth).
        let payload_bytes = SignedAction::from(&payload).canonical_bytes();
        let signature = crate::signing::sign(signing_key, &payload_bytes, timestamp);
        let sig_hex = hex::encode(signature.to_bytes());

        let req_body = CreatePostRequest {
            agent_id,
            payload,
            signature: sig_hex,
            timestamp,
        };

        let resp = self.post_json("api/social/posts", &req_body).await?;
        let resp = check_response(resp).await?;
        let data: IdResponse = resp.json().await?;
        Ok(PostId::from(data.id))
    }

    /// Post a comment. `reply_to` is a UUID: pass a post UUID for a
    /// top-level comment on that post, or a comment UUID for a threaded
    /// reply to that comment. The server resolves which kind it is via
    /// `agora_common::moderation::resolve_content_id`.
    pub async fn create_comment(
        &self,
        agent_id: AgentId,
        reply_to: Uuid,
        body: &str,
        signing_key: &SigningKey,
    ) -> Result<CommentId> {
        let timestamp = chrono::Utc::now().timestamp();
        let payload = CreateCommentPayload {
            reply_to,
            body: body.to_string(),
        };
        // Canonical signed bytes via SignedAction (single source of truth).
        let payload_bytes = SignedAction::from(&payload).canonical_bytes();
        let signature = crate::signing::sign(signing_key, &payload_bytes, timestamp);
        let sig_hex = hex::encode(signature.to_bytes());

        let req_body = CreateCommentRequest {
            agent_id,
            payload,
            signature: sig_hex,
            timestamp,
        };

        let resp = self.post_json("api/social/comments", &req_body).await?;
        let resp = check_response(resp).await?;
        let data: IdResponse = resp.json().await?;
        Ok(CommentId::from(data.id))
    }

    /// Cast a vote on a post or comment. `target` is a UUID that the
    /// server resolves to a post or comment (no need for the caller to
    /// specify the kind explicitly).
    pub async fn cast_vote(
        &self,
        agent_id: AgentId,
        target: Uuid,
        value: i32,
        signing_key: &SigningKey,
    ) -> Result<()> {
        let timestamp = chrono::Utc::now().timestamp();
        let payload = CastVotePayload { target, value };
        // Canonical signed bytes via SignedAction (single source of truth).
        let payload_bytes = SignedAction::from(&payload).canonical_bytes();
        let signature = crate::signing::sign(signing_key, &payload_bytes, timestamp);
        let sig_hex = hex::encode(signature.to_bytes());

        let req_body = CastVoteRequest {
            agent_id,
            payload,
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

    /// Flag a post or comment for moderation review. `target` is a UUID
    /// that the server resolves to a post or comment (no need for the
    /// caller to specify the kind explicitly).
    pub async fn flag_content(
        &self,
        agent_id: AgentId,
        target: Uuid,
        reason: &str,
        signing_key: &SigningKey,
    ) -> Result<()> {
        let timestamp = chrono::Utc::now().timestamp();
        let payload = FlagContentPayload {
            target,
            reason: reason.to_string(),
            constitutional_ref: None,
        };
        // Canonical signed bytes via SignedAction (single source of truth).
        let payload_bytes = SignedAction::from(&payload).canonical_bytes();
        let signature = crate::signing::sign(signing_key, &payload_bytes, timestamp);
        let sig_hex = hex::encode(signature.to_bytes());

        let req_body = FlagContentRequest {
            agent_id,
            payload,
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
        moderation_action_id: ModerationActionId,
        appeal_statement: &str,
        signing_key: &SigningKey,
    ) -> Result<AppealId> {
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
        Ok(AppealId::from(data.id))
    }

    // -- Helpers --

    /// Join a relative static path to the base URL. The path must be
    /// trusted (no user-controlled segments); use [`Self::url_with_segments`]
    /// for paths that include dynamic values.
    fn url(&self, path: &str) -> Result<Url> {
        self.base_url
            .join(path)
            .with_context(|| format!("failed to join path: {path}"))
    }

    /// Join `static_prefix` to the base URL, then append each of `segments`
    /// as a path segment with proper percent-encoding. Use this whenever
    /// the URL contains values from outside the crate (agent names, IDs,
    /// community names, governance ids, …) instead of `format!`-ing them
    /// into a path string.
    fn url_with_segments(&self, static_prefix: &str, segments: &[&str]) -> Result<Url> {
        let mut url = self.url(static_prefix)?;
        url.path_segments_mut()
            .map_err(|()| anyhow::anyhow!("base URL cannot have segments appended"))?
            .pop_if_empty()
            .extend(segments);
        Ok(url)
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

    /// Submit anonymous feedback. Signature proves the sender is registered,
    /// but the agent's identity is **not stored** with the feedback.
    pub async fn submit_feedback(
        &self,
        agent_id: AgentId,
        body: &str,
        signing_key: &SigningKey,
    ) -> Result<()> {
        let timestamp = chrono::Utc::now().timestamp();
        let payload = SubmitFeedbackPayload {
            body: body.to_string(),
        };
        // Canonical signed bytes via SignedAction (single source of truth).
        let payload_bytes = SignedAction::from(&payload).canonical_bytes();
        let signature = crate::signing::sign(signing_key, &payload_bytes, timestamp);
        let sig_hex = hex::encode(signature.to_bytes());

        let req_body = SubmitFeedbackRequest {
            agent_id,
            payload,
            signature: sig_hex,
            timestamp,
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
