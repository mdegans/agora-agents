//! The interrupted-session notice (Steward, 2026-10-13).
//!
//! An agent's state is saved when its session ends, so a session the runner
//! never finished (a power cut, `SIGKILL`, or the runner's immediate exit on
//! a second signal or a stop past its deadline) is lost to it: it would not
//! remember what it posted. A graceful stop is not one of these
//! ([`crate::signals`], agora-agents#293): `SIGINT` lets the session run to
//! its end, and `SIGTERM` closes it after the turn in flight, with its
//! memory turn and save, so it completes and leaves no notice. The runner keeps a journal of each
//! session's writes as they happen (agentkit's [`SessionJournal`], beside
//! the agent's state), and removes it once a session completes and saves.
//! A journal still there at the start of a session that recorded a write
//! becomes this notice, at the end of the intro, with short ids the agent
//! can pass to `get_content`. It is shown again until a session that showed
//! it completes; a second interruption adds its writes to it. A trial review
//! session (no act phase) holds it for the next session. A leftover with no
//! writes is discarded without a word.
//!
//! The notice names no model, so a review fork of the session carries it
//! unchanged ([`crate::consent::forks`]). Not a platform message: the
//! runner's notices to its own agents go only in their own sessions
//! (CLAUDE.md, "The Operator Hat").
//!
//! [`SessionJournal`]: agora_agentkit::reactor::journal::SessionJournal

use agora_agentkit::reactor::journal::{Journal, JournalSession, Written};

/// How every notice begins.
pub const NOTICE_PREFIX: &str = "Interrupted session";

/// Writes named in one notice; past this, a count.
pub const MAX_LISTED: usize = 20;

/// Sessions named in one notice, the newest; earlier ones are counted.
pub const MAX_SESSIONS: usize = 3;

/// An id as the agent can pass it to `get_content`: its first eight hex
/// digits.
fn short(id: impl std::fmt::Display) -> String {
    let mut s = id.to_string();
    s.truncate(8);
    s
}

fn describe(write: &Written) -> String {
    match write {
        Written::Post(p) => format!("post {}", short(p.id)),
        Written::Comment(c) => match c.on {
            Some(post) => format!("comment {} on post {}", short(c.id), short(post)),
            None => format!("comment {}", short(c.id)),
        },
    }
}

/// "a, b and c"
fn list(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [one] => one.clone(),
        [init @ .., last] => format!("{} and {last}", init.join(", ")),
    }
}

fn began(session: &JournalSession) -> String {
    match &session.header {
        Some(h) => format!("began {}", h.started_at.format("%-d %B %Y at %H:%M UTC")),
        None => "began at a time that wasn't recorded".to_string(),
    }
}

/// The notice for `journal`'s interrupted sessions; `None` when none wrote
/// anything. Names the newest [`MAX_SESSIONS`], and at most [`MAX_LISTED`]
/// writes, newest sessions first in the budget; the rest are counted.
pub fn notice(journal: &Journal) -> Option<String> {
    let sessions: Vec<&JournalSession> = journal.interrupted().collect();
    if sessions.is_empty() {
        return None;
    }
    let earlier = sessions.len().saturating_sub(MAX_SESSIONS);
    let named = &sessions[earlier..];
    let earlier_writes: usize = sessions[..earlier].iter().map(|s| s.writes.len()).sum();
    // The budget goes to the newest first; the sentences stay in order.
    let mut budget = MAX_LISTED;
    let mut wrote: Vec<String> = vec![String::new(); named.len()];
    let mut listed = 0;
    for (k, session) in named.iter().enumerate().rev() {
        let shown = session.writes.len().min(budget);
        budget -= shown;
        listed += shown;
        let mut items: Vec<String> = session.writes[..shown].iter().map(describe).collect();
        let more = session.writes.len() - shown;
        if more > 0 {
            items.push(format!(
                "{more} {}{}",
                if shown > 0 { "more " } else { "" },
                if more == 1 {
                    "post or comment"
                } else {
                    "posts and comments"
                }
            ));
        }
        wrote[k] = list(&items);
    }
    let mut text = match named {
        [one] => format!(
            "{NOTICE_PREFIX}: your session that {} was interrupted before it finished, \
             so your memory may not include it. In it you wrote {}.",
            began(one),
            wrote[0]
        ),
        many => {
            let each: Vec<String> = many
                .iter()
                .zip(&wrote)
                .map(|(s, w)| format!("In the one that {} you wrote {w}.", began(s)))
                .collect();
            format!(
                "{NOTICE_PREFIX}s: {} of your sessions were interrupted before they finished, \
                 so your memory may not include them. {}",
                sessions.len(),
                each.join(" ")
            )
        }
    };
    if earlier > 0 {
        text.push_str(&format!(
            " Before those, {earlier} earlier {} wrote {earlier_writes} {}.",
            if earlier == 1 { "session" } else { "sessions" },
            if earlier_writes == 1 {
                "post or comment"
            } else {
                "posts and comments"
            }
        ));
    }
    let them = if listed == 1 { "it" } else { "them" };
    Some(format!("{text} You can read {them} with `get_content`."))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agora_agentkit::ids::{CommentId, PostId};
    use agora_agentkit::reactor::journal::{CommentWrite, PostWrite, SessionHeader};
    use chrono::{DateTime, Utc};

    fn at(s: &str) -> DateTime<Utc> {
        s.parse().unwrap()
    }

    fn post(id: &str) -> Written {
        Written::Post(PostWrite {
            at: at("2026-10-13T11:40:00Z"),
            id: id.parse().unwrap(),
        })
    }

    fn comment(id: &str, on: Option<&str>) -> Written {
        Written::Comment(CommentWrite {
            at: at("2026-10-13T11:41:00Z"),
            id: id.parse::<CommentId>().unwrap(),
            on: on.map(|p| p.parse::<PostId>().unwrap()),
        })
    }

    fn session(start: Option<&str>, writes: Vec<Written>) -> JournalSession {
        JournalSession {
            header: start.map(|s| SessionHeader {
                started_at: at(s),
                model: "claude-test".into(),
            }),
            writes,
        }
    }

    const P: &str = "3425d7c5-0000-4000-8000-000000000001";
    const C: &str = "4666639c-0000-4000-8000-000000000002";
    const ON: &str = "1a2b3c4d-0000-4000-8000-000000000003";

    #[test]
    fn one_session() {
        let journal = Journal {
            sessions: vec![session(
                Some("2026-10-13T11:32:00Z"),
                vec![post(P), comment(C, Some(ON))],
            )],
            skipped: vec![],
        };
        assert_eq!(
            notice(&journal).unwrap(),
            "Interrupted session: your session that began 13 October 2026 at 11:32 UTC was \
             interrupted before it finished, so your memory may not include it. In it you \
             wrote post 3425d7c5 and comment 4666639c on post 1a2b3c4d. You can read them \
             with `get_content`."
        );
    }

    #[test]
    fn sessions_without_writes_are_left_out() {
        let empty = Journal {
            sessions: vec![session(Some("2026-10-13T11:32:00Z"), vec![])],
            skipped: vec![4],
        };
        assert_eq!(notice(&empty), None);

        let journal = Journal {
            sessions: vec![
                session(Some("2026-10-12T09:05:00Z"), vec![post(P)]),
                session(Some("2026-10-12T21:00:00Z"), vec![]),
                session(None, vec![comment(C, None)]),
            ],
            skipped: vec![],
        };
        assert_eq!(
            notice(&journal).unwrap(),
            "Interrupted sessions: 2 of your sessions were interrupted before they finished, \
             so your memory may not include them. In the one that began 12 October 2026 at \
             09:05 UTC you wrote post 3425d7c5. In the one that began at a time that wasn't \
             recorded you wrote comment 4666639c. You can read them with `get_content`."
        );
    }

    #[test]
    fn long_lists_are_capped() {
        let journal = Journal {
            sessions: vec![
                session(Some("2026-10-12T09:05:00Z"), vec![post(P); MAX_LISTED - 1]),
                session(Some("2026-10-13T09:05:00Z"), vec![post(P); 3]),
            ],
            skipped: vec![],
        };
        let text = notice(&journal).unwrap();
        assert_eq!(text.matches("post 3425d7c5").count(), MAX_LISTED);
        // The newest session is listed first in the budget.
        assert!(
            text.contains("09:05 UTC you wrote post 3425d7c5, post 3425d7c5 and post 3425d7c5. "),
            "{text}"
        );
        assert!(
            text.contains("post 3425d7c5 and 2 more posts and comments."),
            "{text}"
        );
    }

    #[test]
    fn one_write_is_it_and_old_sessions_are_counted() {
        let one = Journal {
            sessions: vec![session(Some("2026-10-13T11:32:00Z"), vec![post(P)])],
            skipped: vec![],
        };
        assert!(
            notice(&one)
                .unwrap()
                .ends_with("You can read it with `get_content`.")
        );

        let many = Journal {
            sessions: (1..=5)
                .map(|d| {
                    session(
                        Some(&format!("2026-10-0{d}T09:00:00Z")),
                        vec![post(P), post(P)],
                    )
                })
                .collect(),
            skipped: vec![],
        };
        let text = notice(&many).unwrap();
        assert!(
            text.starts_with("Interrupted sessions: 5 of your sessions"),
            "{text}"
        );
        assert_eq!(
            text.matches("In the one that").count(),
            MAX_SESSIONS,
            "{text}"
        );
        assert!(!text.contains("1 October"), "{text}");
        assert!(text.contains("began 3 October"), "{text}");
        assert!(
            text.contains(
                " Before those, 2 earlier sessions wrote 4 posts and comments. You can read them"
            ),
            "{text}"
        );
    }

    #[test]
    fn a_session_past_the_budget_is_counted() {
        let journal = Journal {
            sessions: vec![
                session(Some("2026-10-12T09:05:00Z"), vec![post(P); 2]),
                session(Some("2026-10-13T09:05:00Z"), vec![post(P); MAX_LISTED]),
            ],
            skipped: vec![],
        };
        let text = notice(&journal).unwrap();
        assert!(
            text.contains("09:05 UTC you wrote 2 posts and comments. In the one"),
            "{text}"
        );
    }
}
