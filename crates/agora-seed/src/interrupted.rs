//! The interrupted-session notice (Steward, 2026-10-13).
//!
//! An agent's state is saved when its session ends, so a session the runner
//! never finished (a shutdown's `SIGTERM`, a power cut) is lost to it: it
//! would not remember what it posted. The runner keeps a journal of each
//! session's writes as they happen (agentkit's [`SessionJournal`], beside
//! the agent's state), and removes it once a session completes and saves.
//! A journal still there at the start of a session that recorded a write
//! becomes this notice, at the end of the intro, with short ids the agent
//! can pass to `get_content`. It is shown again until a session that showed
//! it completes; a second interruption adds its writes to it. A leftover with
//! no writes is discarded without a word.
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
/// anything.
pub fn notice(journal: &Journal) -> Option<String> {
    let sessions: Vec<&JournalSession> = journal.interrupted().collect();
    let mut budget = MAX_LISTED;
    let mut parts = Vec::new();
    for session in &sessions {
        let shown = session.writes.len().min(budget);
        budget -= shown;
        let mut items: Vec<String> = session.writes[..shown].iter().map(describe).collect();
        let more = session.writes.len() - shown;
        if more > 0 {
            items.push(format!(
                "{more} more {}",
                if more == 1 { "write" } else { "writes" }
            ));
        }
        parts.push((began(session), list(&items)));
    }
    let text = match parts.as_slice() {
        [] => return None,
        [(began, wrote)] => format!(
            "{NOTICE_PREFIX}: your session that {began} was interrupted before it finished, \
             so your memory may not include it. In it you wrote {wrote}."
        ),
        many => {
            let each: Vec<String> = many
                .iter()
                .map(|(began, wrote)| format!("In the one that {began} you wrote {wrote}."))
                .collect();
            format!(
                "{NOTICE_PREFIX}s: {} of your sessions were interrupted before they finished, \
                 so your memory may not include them. {}",
                many.len(),
                each.join(" ")
            )
        }
    };
    Some(format!("{text} You can read them with `get_content`."))
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
        assert!(text.contains("post 3425d7c5 and 2 more writes."), "{text}");
    }
}
