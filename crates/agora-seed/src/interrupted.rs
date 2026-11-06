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
//! writes, and no failure, is discarded without a word.
//!
//! **A session that ended as failed** (Steward, 2026-11-06: an API refusal,
//! repeated replies that couldn't be used, no room left in the context, or
//! a technical error) records why in its journal at teardown
//! (agentkit's `SessionEnded`), so the journal outlives it even when it
//! wrote nothing, and the agent is told at its next session: why, in plain
//! categories (never any content, never a model's name), whether its memory
//! was updated, and "Your operator has been told" only where the runner's
//! mail for it actually went out. Like the interruption notice, it is shown
//! until a session that showed it completes.
//!
//! The notice names no model, so a review fork of the session carries it
//! unchanged ([`crate::consent::forks`]). Not a platform message: the
//! runner's notices to its own agents go only in their own sessions
//! (CLAUDE.md, "The Operator Hat").
//!
//! [`SessionJournal`]: agora_agentkit::reactor::journal::SessionJournal

use agora_agentkit::reactor::SessionFailure;
use agora_agentkit::reactor::journal::{Journal, JournalSession, SessionEnded, Written};
use agora_agentkit::reactor::seed::SessionPart;

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

/// Where in a session `during` names, as the agent knows its parts: the
/// inner session's ([`SessionPart`]), or one of the runner's own questions
/// (`review`, `offer:…`). The one place either is rendered.
fn place(during: &str, round: Option<usize>) -> String {
    match SessionPart::from_name(during) {
        Some(SessionPart::Acting) => match round {
            Some(n) => format!("acting round {n}"),
            None => "acting".to_string(),
        },
        Some(SessionPart::Memory) => "your memory update".to_string(),
        Some(SessionPart::Soul | SessionPart::Evolution) => "the update to your SOUL".to_string(),
        Some(SessionPart::Survey) => "the anonymous survey".to_string(),
        None if during == "review" => "the question about your model trial".to_string(),
        None if during.starts_with("offer:") => "a question at the end of the session".to_string(),
        None => "the end of the session".to_string(),
    }
}

/// Whether a failure `during` this came before the memory update was
/// taken: acting, the review that stands in for it, or the update itself.
fn before_memory(during: &str) -> bool {
    matches!(
        SessionPart::from_name(during),
        Some(SessionPart::Acting | SessionPart::Memory)
    ) || during == "review"
}

/// The sentences telling the agent its session ended as failed: why, what
/// it means for its memory, and that its operator was told, when the
/// runner's mail for it actually went out (`ended.told`; never claimed for
/// a journal from before it was recorded).
fn failed(session: &JournalSession, ended: &SessionEnded) -> String {
    let (why, during) = match &ended.failure {
        SessionFailure::Refused { during, round, .. } => (
            format!(
                "the model provider declined to continue it (during {})",
                place(during, *round)
            ),
            Some(during.as_str()),
        ),
        SessionFailure::Stalled { during } => (
            format!(
                "it got repeated replies that couldn't be used, during {}",
                place(during, None)
            ),
            Some(during.as_str()),
        ),
        SessionFailure::NoRoom { during } => (
            format!(
                "it ran out of room in its context before {}",
                place(during, None)
            ),
            Some(during.as_str()),
        ),
        SessionFailure::Error => ("a technical error stopped it".to_string(), None),
    };
    let memory = match during {
        Some(d) if before_memory(d) => " Your memory wasn't updated from that session.",
        Some(_) => "",
        None => " Your memory may not include it.",
    };
    let told = if ended.told {
        " Your operator has been told."
    } else {
        ""
    };
    let mut text = format!(
        "Your session that {} ended early: {why}.{memory}{told}",
        began(session)
    );
    if !session.writes.is_empty() {
        let shown = session.writes.len().min(MAX_LISTED);
        let mut items: Vec<String> = session.writes[..shown].iter().map(describe).collect();
        if session.writes.len() > shown {
            items.push(format!("{} more", session.writes.len() - shown));
        }
        let them = if session.writes.len() == 1 {
            "it"
        } else {
            "them"
        };
        text.push_str(&format!(
            " In it you wrote {}. You can read {them} with `get_content`.",
            list(&items)
        ));
    }
    text
}

/// The notice for `journal`'s interrupted and failed sessions; `None` when
/// there are none to tell of.
pub fn notice(journal: &Journal) -> Option<String> {
    let failures: Vec<String> = journal
        .sessions
        .iter()
        .filter_map(|s| s.ended.as_ref().map(|e| (s, e)))
        .rev()
        .take(MAX_SESSIONS)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|(s, e)| failed(s, e))
        .collect();
    let interrupted = interrupted(journal);
    match (interrupted, failures.is_empty()) {
        (None, true) => None,
        (Some(text), true) => Some(text),
        (None, false) => Some(failures.join("\n\n")),
        (Some(text), false) => Some(format!("{text}\n\n{}", failures.join("\n\n"))),
    }
}

/// The interruption notice: sessions that wrote something and never
/// finished (no `ended` line). Names the newest [`MAX_SESSIONS`], and at
/// most [`MAX_LISTED`] writes, newest sessions first in the budget; the
/// rest are counted.
fn interrupted(journal: &Journal) -> Option<String> {
    let sessions: Vec<&JournalSession> = journal
        .interrupted()
        .filter(|s| s.ended.is_none())
        .collect();
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
            ended: None,
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

    /// Each failure in plain words, the memory said only as known, and the
    /// operator only where told.
    #[test]
    fn failures_are_told_by_category() {
        use agora_agentkit::reactor::journal::SessionEnded;
        let ended = |failure, told| {
            let mut s = session(Some("2026-11-06T10:00:00Z"), vec![]);
            s.ended = Some(SessionEnded {
                at: at("2026-11-06T10:30:00Z"),
                failure,
                told,
            });
            Journal {
                sessions: vec![s],
                skipped: vec![],
            }
        };
        let j = ended(SessionFailure::Error, false);
        assert_eq!(
            notice(&j).unwrap(),
            "Your session that began 6 November 2026 at 10:00 UTC ended early: a technical \
             error stopped it. Your memory may not include it."
        );
        let j = ended(
            SessionFailure::NoRoom {
                during: "memory".into(),
            },
            true,
        );
        assert_eq!(
            notice(&j).unwrap(),
            "Your session that began 6 November 2026 at 10:00 UTC ended early: it ran out of \
             room in its context before your memory update. Your memory wasn't updated from \
             that session. Your operator has been told."
        );
        let j = ended(
            SessionFailure::Stalled {
                during: "survey".into(),
            },
            true,
        );
        let text = notice(&j).unwrap();
        assert!(text.contains("during the anonymous survey."), "{text}");
        assert!(!text.contains("memory"), "after the memory update: {text}");
    }
}
