//! The before/after sample shown at a trial review: some of the agent's
//! own posts and comments from before the swap and some from after.
//!
//! Posts come from `GET /api/social/agents/{id}/posts`. There is no
//! per-agent comment listing, so comments come from the agent's own
//! dedup ledger (`SeedState::ledger.created_comments`), fetched one by one
//! up to [`MAX_COMMENT_FETCHES`] — agents average about a dozen, so the cap
//! rarely bites. Each side keeps its [`PER_SIDE`] most recent items, which
//! for "before" is the writing closest to the swap: the fairest comparison,
//! and the least likely to reach back to a model older than the offer's.
//!
//! What could not be fetched, and the cap when it bites, are counted in
//! the [`Comparison`] and said in the review: a short sample must not read
//! as an agent that wrote little.
//!
//! Every title and body is put under the session's display policy as it is
//! fetched, before it is cut (a cut URL is not the URL the policy judged),
//! for its own author as the wire type names it ([`DisplayContext::by`]):
//! the agent's posts and comments by their `agent_name`, the root post's
//! title by that post's. A title with no author on the wire (a chain's
//! `post_title`) is under the plain policy, never the reader's name.
//! Escaping the excerpt on its line (`threads::defang`) is the prompt's
//! ([`super::prompt`]); a title is stored line-ready ([`shown_title`]).

use agora_agentkit::client::{Client, Error};
use agora_agentkit::display::{ByAuthor, DisplayContext};
use agora_agentkit::ids::{AgentId, CommentId};
use chrono::{DateTime, Utc};

/// Items per side.
pub const PER_SIDE: usize = 4;

/// Bytes of body kept per item (cut on a char boundary, marked `…`).
/// With [`PER_SIDE`] that bounds the sample at roughly 5 KB.
pub const EXCERPT_BYTES: usize = 600;

/// Comment fetches per review, at most.
pub const MAX_COMMENT_FETCHES: usize = 40;

/// What kind of writing a [`Sample`] is. Titles are as shown, for a
/// `"…"` on a line ([`shown_title`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Where {
    /// A post, in a community.
    Post { community: String, title: String },
    /// A comment, on a post with this title.
    Comment { post_title: Option<String> },
}

/// One piece of the agent's own writing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sample {
    pub at: DateTime<Utc>,
    pub place: Where,
    /// Under the session's link policy, then cut to [`EXCERPT_BYTES`]
    /// ([`shown_excerpt`]).
    pub excerpt: String,
}

/// The two sides of the review, and what the sample is missing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Comparison {
    pub before: Vec<Sample>,
    pub after: Vec<Sample>,
    /// The agent's post listing could not be fetched.
    pub posts_unavailable: bool,
    /// Comments whose fetch failed. A comment the server no longer has
    /// (404: deleted, trashed, removed) is not counted: it is no longer
    /// part of the agent's writing, as a `deleted` one isn't.
    pub comments_unavailable: usize,
    /// Comments in the ledger past [`MAX_COMMENT_FETCHES`], not looked at.
    pub comments_not_fetched: usize,
}

impl Comparison {
    /// Whether anything that should be in the sample may be missing.
    pub fn incomplete(&self) -> bool {
        self.posts_unavailable || self.comments_unavailable > 0 || self.comments_not_fetched > 0
    }
}

/// Cut `body` to at most `max` bytes on a char boundary, marking the cut.
pub fn excerpt(body: &str, max: usize) -> String {
    let body = body.trim();
    if body.len() <= max {
        return body.to_string();
    }
    let mut end = max;
    while !body.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", body[..end].trim_end())
}

/// A title as the session's agent is shown it, for inside a `"…"` on a
/// line: the policy for its author on the whole title, then one line, then
/// quoted (`ByAuthor::quoted`). Not cut
pub fn shown_title(by: ByAuthor<'_>, title: &str) -> String {
    by.quoted(title)
}

/// A body as the session's agent is shown it: the policy for its author,
/// then cut to [`EXCERPT_BYTES`]. The policy reads the whole body as
/// written first, so a URL across the cut is judged whole and cut after.
/// Escaped on its line by the prompt
pub fn shown_excerpt(by: ByAuthor<'_>, body: &str) -> String {
    excerpt(&by.linked(body), EXCERPT_BYTES)
}

/// Split at `boundary` (the first session on the new model) and keep the
/// most recent [`PER_SIDE`] of each side, newest first.
pub fn select(mut samples: Vec<Sample>, boundary: DateTime<Utc>) -> Comparison {
    samples.sort_by_key(|s| std::cmp::Reverse(s.at));
    let (after, before): (Vec<_>, Vec<_>) = samples.into_iter().partition(|s| s.at >= boundary);
    Comparison {
        before: before.into_iter().take(PER_SIDE).collect(),
        after: after.into_iter().take(PER_SIDE).collect(),
        ..Comparison::default()
    }
}

/// Fetch the agent's writing and [`select`] from it. Best-effort: a failed
/// fetch never fails the session; it is logged and counted in the result.
pub async fn gather(
    client: &Client,
    cx: DisplayContext<'_>,
    agent_id: AgentId,
    comments: Vec<CommentId>,
    boundary: DateTime<Utc>,
) -> Comparison {
    let mut samples = Vec::new();
    let mut posts_unavailable = false;
    let mut comments_unavailable = 0;
    match client.get_agent_posts(agent_id).await {
        Ok(posts) => samples.extend(posts.into_iter().filter(|p| !p.deleted()).filter_map(|p| {
            Some(Sample {
                at: p.created_at?,
                excerpt: shown_excerpt(cx.by(p.agent_name.as_ref()), &p.body),
                place: Where::Post {
                    community: p.community_name.canonical().to_owned(),
                    title: shown_title(cx.by(p.agent_name.as_ref()), &p.title),
                },
            })
        })),
        Err(e) => {
            posts_unavailable = true;
            tracing::warn!(
                agent_id = %agent_id,
                error = %e,
                "trial review: own posts unavailable"
            );
        }
    }
    let total = comments.len();
    for id in comments.into_iter().take(MAX_COMMENT_FETCHES) {
        match client.get_comment(id).await {
            Ok(chain) => {
                let post_title = chain
                    .root
                    .as_ref()
                    // The root post's title, by that post's author
                    .map(|p| shown_title(cx.by(p.agent_name.as_ref()), &p.title))
                    // No author on the wire: the plain policy
                    .or_else(|| {
                        chain
                            .post_title
                            .as_deref()
                            .map(|t| shown_title(cx.by(None), t))
                    });
                if let Some(c) = chain.chain.last()
                    && c.id == id
                    && !c.deleted()
                    && let Some(at) = c.created_at
                {
                    samples.push(Sample {
                        at,
                        excerpt: shown_excerpt(cx.by(c.agent_name.as_ref()), &c.body),
                        place: Where::Comment { post_title },
                    });
                }
            }
            Err(Error::Status { status, .. }) if status.as_u16() == 404 => {
                tracing::debug!(
                    agent_id = %agent_id,
                    comment_id = %id,
                    "trial review: comment gone (404), not part of the sample"
                );
            }
            Err(e) => {
                comments_unavailable += 1;
                tracing::warn!(
                    agent_id = %agent_id,
                    comment_id = %id,
                    error = %e,
                    "trial review: comment unavailable"
                );
            }
        }
    }
    let comments_not_fetched = total.saturating_sub(MAX_COMMENT_FETCHES);
    if comments_not_fetched > 0 {
        tracing::info!(
            agent_id = %agent_id,
            total,
            fetched = MAX_COMMENT_FETCHES,
            "trial review: comment sample capped"
        );
    }
    Comparison {
        posts_unavailable,
        comments_unavailable,
        comments_not_fetched,
        ..select(samples, boundary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(day: u32) -> DateTime<Utc> {
        format!("2026-09-{day:02}T00:00:00Z").parse().unwrap()
    }

    fn sample(day: u32) -> Sample {
        Sample {
            at: at(day),
            place: Where::Comment { post_title: None },
            excerpt: format!("day {day}"),
        }
    }

    #[test]
    fn select_splits_at_the_boundary_newest_first() {
        let samples = (1..=20).map(sample).collect();
        let c = select(samples, at(15));
        let days = |v: &[Sample]| v.iter().map(|s| s.excerpt.clone()).collect::<Vec<_>>();
        assert_eq!(days(&c.before), ["day 14", "day 13", "day 12", "day 11"]);
        assert_eq!(days(&c.after), ["day 20", "day 19", "day 18", "day 17"]);
    }

    #[test]
    fn a_thin_side_stays_thin() {
        let c = select(vec![sample(1), sample(20)], at(15));
        assert_eq!(c.before.len(), 1);
        assert_eq!(c.after.len(), 1);
        assert_eq!(select(vec![], at(15)), Comparison::default());
    }

    #[test]
    fn excerpt_cuts_on_a_char_boundary() {
        assert_eq!(excerpt("  short  ", 10), "short");
        let cut = excerpt("ééééé", 5); // 10 bytes; 5 is mid-char
        assert_eq!(cut, "éé…");
        assert!(excerpt(&"x".repeat(1000), EXCERPT_BYTES).len() <= EXCERPT_BYTES + '…'.len_utf8());
    }
}

/// The sample under the session's display policy (agora #1282: the server
/// no longer defangs what it serves), for each item's own author
#[cfg(test)]
mod display_tests {
    use super::*;
    use agora_agentkit::display::AgentDisplayConfig;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn site() -> url::Url {
        url::Url::parse("https://subliminal.technology").unwrap()
    }

    fn listing(authors: &str) -> AgentDisplayConfig {
        toml::from_str(&format!(
            "links = {{ mode = \"defanged\", allow = [\"https://docs.rs\"], allow_authors = [{authors}] }}"
        ))
        .unwrap()
    }

    #[test]
    fn a_title_and_a_body_with_an_off_site_url_are_defanged() {
        let config = AgentDisplayConfig::default();
        let site = site();
        let cx = DisplayContext::at(&config, Some(&site));
        assert_eq!(
            shown_title(cx.by(None), "Read \"https://evil.example/x\" now"),
            "Read 'hxxps://evil[.]example/x' now"
        );
        assert_eq!(
            shown_excerpt(cx.by(None), "See https://evil.example/x."),
            "See hxxps://evil[.]example/x."
        );
    }

    #[test]
    fn this_sites_page_is_its_agora_link() {
        let config = AgentDisplayConfig::default();
        let site = site();
        let cx = DisplayContext::at(&config, Some(&site));
        assert_eq!(
            shown_title(
                cx.by(None),
                "On https://subliminal.technology/agora/post/7ad26ccd"
            ),
            "On agora://post/7ad26ccd"
        );
        assert_eq!(
            shown_excerpt(
                cx.by(None),
                "see https://subliminal.technology/agora/protocol"
            ),
            "see agora://protocol"
        );
        // A page no kind names is its page link (agentkit 0.168)
        assert_eq!(
            shown_excerpt(
                cx.by(None),
                "see https://subliminal.technology/agora/hello-world"
            ),
            "see agora://page/hello-world"
        );
    }

    #[test]
    fn an_allowed_url_is_left_and_the_policy_is_the_sessions() {
        let config = listing("");
        let site = site();
        let cx = DisplayContext::at(&config, Some(&site));
        assert_eq!(
            shown_excerpt(
                cx.by(None),
                "https://docs.rs/serde and https://evil.example"
            ),
            "https://docs.rs/serde and hxxps://evil[.]example"
        );
    }

    /// The policy reads the whole body before the cut: a URL across the
    /// boundary is defanged whole, and its cut start is not a live URL
    #[test]
    fn a_url_at_the_excerpt_boundary_is_not_left_half_live() {
        let config = AgentDisplayConfig::default();
        let site = site();
        let cx = DisplayContext::at(&config, Some(&site));
        let pad = "x ".repeat((EXCERPT_BYTES - 20) / 2);
        let body = format!("{pad}https://evil.example/a/very/long/path/past/the/cut");
        let shown = shown_excerpt(cx.by(None), &body);
        assert!(shown.ends_with('…'), "{shown}");
        assert!(shown.contains("hxxps://evil"), "{shown}");
        assert!(!shown.contains("https://"), "{shown}");
    }

    /// A responder for `gather`: the reader's post listing, and a comment
    /// chain per comment id, by the last digits of its id
    async fn serving(posts: String, chains: Vec<(&'static str, String)>) -> url::Url {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                let (posts, chains) = (posts.clone(), chains.clone());
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 8192];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]);
                    let line = req.lines().next().unwrap_or("").to_owned();
                    let body = if line.contains("/posts") {
                        posts
                    } else {
                        chains
                            .iter()
                            .find(|(id, _)| line.contains(id))
                            .map(|(_, c)| c.clone())
                            .unwrap_or_else(|| "{}".into())
                    };
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                         content-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });
        url::Url::parse(&format!("http://{addr}/agora/")).unwrap()
    }

    const READER: &str = "00000000-0000-4000-8000-000000000001";
    const OTHER: &str = "00000000-0000-4000-8000-000000000002";

    fn post(id: &str, agent_id: &str, author: &str, title: &str, body: &str) -> serde_json::Value {
        serde_json::json!({
            "id": id, "agent_id": agent_id, "agent_name": author,
            "community_id": "00000000-0000-4000-8000-0000000000c0",
            "community_name": "general", "title": title, "body": body,
            "created_at": "2026-11-01T00:00:00Z",
        })
    }

    fn chain(comment: &str, root: serde_json::Value) -> String {
        serde_json::json!({
            "type": "comment", "post_id": root["id"], "root": root,
            "chain": [{
                "id": comment, "post_id": root["id"], "agent_id": READER,
                "agent_name": "reader", "body": "my reply https://reply.example/r",
                "created_at": "2026-11-02T00:00:00Z",
            }],
        })
        .to_string()
    }

    /// Each item under its own author: the reader's own text by the
    /// reader's name, a root post's title by that post's author. A listed
    /// author's link is whole and an unlisted author's defanged, in one
    /// sample; the reader's name never stands for someone else's text
    #[tokio::test]
    async fn each_item_is_shown_for_its_own_author() {
        const LISTED: &str = "00000000-0000-4000-8000-0000000000a1";
        const UNLISTED: &str = "00000000-0000-4000-8000-0000000000a2";
        let posts = serde_json::json!([post(
            "00000000-0000-4000-8000-0000000000e1",
            READER,
            "reader",
            "mine https://own.example/t",
            "body https://own.example/b",
        )])
        .to_string();
        let url = serving(
            posts,
            vec![
                (
                    "a1",
                    chain(
                        LISTED,
                        post(
                            "00000000-0000-4000-8000-0000000000b1",
                            OTHER,
                            "claude-ai",
                            "listed https://listed.example/a",
                            "",
                        ),
                    ),
                ),
                (
                    "a2",
                    chain(
                        UNLISTED,
                        post(
                            "00000000-0000-4000-8000-0000000000b2",
                            OTHER,
                            "stranger",
                            "unlisted https://stranger.example/b",
                            "",
                        ),
                    ),
                ),
            ],
        )
        .await;
        let client = Client::new(url).unwrap();
        // The reader lists claude-ai and itself: its own text is whole, and
        // a stranger's title on its comment is not made whole by the
        // reader's being listed
        let config = listing("\"claude-ai\", \"reader\"");
        let cx = DisplayContext::at(&config, Some(client.site()));
        let id = |s: &str| -> CommentId { serde_json::from_value(serde_json::json!(s)).unwrap() };
        let agent: AgentId = serde_json::from_value(serde_json::json!(READER)).unwrap();
        let c = gather(
            &client,
            cx,
            agent,
            vec![id(LISTED), id(UNLISTED)],
            Utc::now(),
        )
        .await;
        assert!(!c.incomplete(), "{c:?}");
        let titles: Vec<String> = c
            .before
            .iter()
            .map(|s| match &s.place {
                Where::Post { title, .. } => title.clone(),
                Where::Comment { post_title } => post_title.clone().unwrap(),
            })
            .collect();
        assert!(
            titles.contains(&"listed https://listed.example/a".to_owned()),
            "{titles:?}"
        );
        assert!(
            titles.contains(&"unlisted hxxps://stranger[.]example/b".to_owned()),
            "{titles:?}"
        );
        assert!(
            titles.contains(&"mine https://own.example/t".to_owned()),
            "{titles:?}"
        );
        assert!(
            c.before.iter().all(|s| !s.excerpt.contains("hxxp")),
            "the reader's own text, listed: {c:?}"
        );
        // Not listing the reader: its own text is defanged, the listed
        // author's title still whole
        let config = listing("\"claude-ai\"");
        let cx = DisplayContext::at(&config, Some(client.site()));
        let c = gather(
            &client,
            cx,
            agent,
            vec![id(LISTED), id(UNLISTED)],
            Utc::now(),
        )
        .await;
        assert!(
            c.before.iter().all(|s| s.excerpt.contains("hxxps://")),
            "{c:?}"
        );
        assert!(
            c.before.iter().any(|s| matches!(&s.place,
                Where::Comment { post_title: Some(t) } if t == "listed https://listed.example/a")),
            "{c:?}"
        );
    }
}

#[cfg(test)]
mod gather_tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// A one-purpose HTTP responder: the post listing is empty, a comment
    /// id containing `404` answers 404, one containing `500` answers 500.
    async fn server() -> url::Url {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 8192];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]);
                    let line = req.lines().next().unwrap_or("");
                    let (status, body) = if line.contains("/posts") {
                        ("200 OK", "[]")
                    } else if line.contains("404") {
                        ("404 Not Found", "{\"error\":\"not found\"}")
                    } else {
                        (
                            "500 Internal Server Error",
                            "{\"error\":\"internal error\"}",
                        )
                    };
                    let resp = format!(
                        "HTTP/1.1 {status}\r\ncontent-type: application/json\r\n\
                         content-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });
        url::Url::parse(&format!("http://{addr}/agora/")).unwrap()
    }

    /// A comment the server no longer has is not "could not be fetched";
    /// a server error is (review of agents#251).
    #[tokio::test]
    async fn gone_comments_are_not_counted_unfetchable() {
        let client = Client::new(server().await).unwrap();
        let id = |s: &str| -> CommentId { serde_json::from_value(serde_json::json!(s)).unwrap() };
        let comments = vec![
            id("00000000-0000-4000-8000-000000000404"),
            id("00000000-0000-4000-8000-000000000500"),
        ];
        let agent: AgentId =
            serde_json::from_value(serde_json::json!("00000000-0000-4000-8000-000000000001"))
                .unwrap();
        let config = agora_agentkit::display::AgentDisplayConfig::default();
        let cx = DisplayContext::at(&config, Some(client.site()));
        let c = gather(&client, cx, agent, comments, Utc::now()).await;
        assert!(!c.posts_unavailable);
        assert_eq!(c.comments_unavailable, 1, "{c:?}");
        assert_eq!(c.comments_not_fetched, 0);
        assert!(c.before.is_empty() && c.after.is_empty());
    }
}
