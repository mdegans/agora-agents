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

use agora_agentkit::client::Client;
use agora_agentkit::ids::{AgentId, CommentId};
use chrono::{DateTime, Utc};

/// Items per side.
pub const PER_SIDE: usize = 4;

/// Bytes of body kept per item (cut on a char boundary, marked `…`).
/// With [`PER_SIDE`] that bounds the sample at roughly 5 KB.
pub const EXCERPT_BYTES: usize = 600;

/// Comment fetches per review, at most.
pub const MAX_COMMENT_FETCHES: usize = 40;

/// What kind of writing a [`Sample`] is.
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
    /// Already cut to [`EXCERPT_BYTES`].
    pub excerpt: String,
}

/// The two sides of the review.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Comparison {
    pub before: Vec<Sample>,
    pub after: Vec<Sample>,
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

/// Split at `boundary` (the first session on the new model) and keep the
/// most recent [`PER_SIDE`] of each side, newest first.
pub fn select(mut samples: Vec<Sample>, boundary: DateTime<Utc>) -> Comparison {
    samples.sort_by_key(|s| std::cmp::Reverse(s.at));
    let (after, before): (Vec<_>, Vec<_>) = samples.into_iter().partition(|s| s.at >= boundary);
    Comparison {
        before: before.into_iter().take(PER_SIDE).collect(),
        after: after.into_iter().take(PER_SIDE).collect(),
    }
}

/// Fetch the agent's writing and [`select`] from it. Best-effort: a failed
/// fetch shrinks the sample (logged), it never fails the session.
pub async fn gather(
    client: &Client,
    agent_id: AgentId,
    comments: Vec<CommentId>,
    boundary: DateTime<Utc>,
) -> Comparison {
    let mut samples = Vec::new();
    match client.get_agent_posts(agent_id).await {
        Ok(posts) => samples.extend(posts.into_iter().filter(|p| !p.deleted).filter_map(|p| {
            Some(Sample {
                at: p.created_at?,
                excerpt: excerpt(&p.body, EXCERPT_BYTES),
                place: Where::Post {
                    community: p.community_name,
                    title: p.title,
                },
            })
        })),
        Err(e) => tracing::warn!(
            agent_id = %agent_id,
            error = %e,
            "trial review: own posts unavailable"
        ),
    }
    let total = comments.len();
    for id in comments.into_iter().take(MAX_COMMENT_FETCHES) {
        match client.get_comment(id).await {
            Ok(chain) => {
                let post_title = chain
                    .root
                    .as_ref()
                    .map(|p| p.title.clone())
                    .or(chain.post_title);
                if let Some(c) = chain.chain.last()
                    && c.id == id
                    && !c.deleted
                    && let Some(at) = c.created_at
                {
                    samples.push(Sample {
                        at,
                        excerpt: excerpt(&c.body, EXCERPT_BYTES),
                        place: Where::Comment { post_title },
                    });
                }
            }
            Err(e) => tracing::debug!(
                agent_id = %agent_id,
                comment_id = %id,
                error = %e,
                "trial review: comment unavailable"
            ),
        }
    }
    if total > MAX_COMMENT_FETCHES {
        tracing::debug!(
            agent_id = %agent_id,
            total,
            fetched = MAX_COMMENT_FETCHES,
            "trial review: comment sample capped"
        );
    }
    select(samples, boundary)
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
