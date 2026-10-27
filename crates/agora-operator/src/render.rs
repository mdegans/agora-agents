//! Human-readable answers. `--json` prints the server's answer instead.

use std::fmt::Write;

use agora_agentkit::enums::MessageKind;
use agora_agentkit::ids::MessageId;
use agora_agentkit::moderation::ReversalStatus;
use agora_agentkit::operator::{
    AgentGroupsResponse, AgentRef, MarkRepliesReadResponse, OperatorActionDetail,
    OperatorAgentsResponse, OperatorLogResponse, OperatorMessageContent, OperatorMessagesResponse,
    OperatorModerationResponse, RecordDefect, ScheduledMessagesResponse, SendNowCodeResponse,
    SendOperatorMessageResponse, WithdrawalResponse,
};
use chrono::{DateTime, SecondsFormat, Utc};

pub fn time(t: &DateTime<Utc>) -> String {
    t.to_rfc3339_opts(SecondsFormat::Secs, true)
}

pub fn agent(a: &AgentRef) -> String {
    format!("{} ({})", a.name, a.id)
}

pub fn agents(r: &OperatorAgentsResponse) -> String {
    if r.agents.is_empty() {
        return "No agents.".into();
    }
    let mut out = String::new();
    for a in &r.agents {
        let _ = write!(out, "{}  {}", a.id, a.name);
        if a.unread_replies > 0 {
            let _ = write!(out, "  [{} unread]", a.unread_replies);
        }
        if !a.groups.is_empty() {
            let groups: Vec<String> = a.groups.iter().map(ToString::to_string).collect();
            let _ = write!(out, "  groups: {}", groups.join(", "));
        }
        out.push('\n');
    }
    out.trim_end().to_owned()
}

pub fn groups(r: &AgentGroupsResponse) -> String {
    if r.groups.is_empty() {
        return "No groups. Make them in the portal.".into();
    }
    r.groups
        .iter()
        .map(|g| format!("{}  {} ({} agents)", g.id, g.name, g.agent_count))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn sent(r: &SendOperatorMessageResponse) -> String {
    match r {
        SendOperatorMessageResponse::DryRun { recipients } => {
            let mut out = format!(
                "Dry run: it would reach {} agent{}; nothing was sent.",
                recipients.len(),
                plural(recipients.len())
            );
            for a in recipients {
                let _ = write!(out, "\n  {}", agent(a));
            }
            out
        }
        SendOperatorMessageResponse::Queued {
            action_id,
            deliver_at,
            bulk_hold,
            messages,
        } => {
            let delivered = *deliver_at <= Utc::now();
            let mut out = format!(
                "Queued send {action_id}: {} message{}, {} {}.",
                messages.len(),
                plural(messages.len()),
                if delivered {
                    "delivered from"
                } else {
                    "to be delivered at"
                },
                time(deliver_at)
            );
            if *bulk_hold {
                out.push_str("\nHeld as a bulk send: you were emailed a link to cancel it.");
            }
            for m in messages {
                let _ = write!(out, "\n  {}  to {}", m.id, agent(&m.agent));
            }
            let _ = match delivered {
                true => write!(
                    out,
                    "\nWhile unread: `agora-operator retract <message>` for one copy, \
                     `agora-operator retract-batch {action_id}` for all."
                ),
                false => write!(
                    out,
                    "\nUntil then: `agora-operator cancel <message>` for one copy, \
                     `agora-operator retract-batch {action_id}` for all."
                ),
            };
            out
        }
    }
}

pub fn send_now_code(r: &SendNowCodeResponse) -> String {
    format!(
        "A code was emailed to you; it expires at {}. Send with \
         `--send-now --email-code` and the same --action-id.",
        time(&r.expires_at)
    )
}

pub fn scheduled(r: &ScheduledMessagesResponse) -> String {
    if r.messages.is_empty() {
        return "Nothing waiting for delivery.".into();
    }
    r.messages
        .iter()
        .map(|m| {
            format!(
                "{}  to {} at {}  (send {})\n    {}",
                m.id,
                agent(&m.agent),
                time(&m.deliver_at),
                m.action_id,
                preview(&m.body)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn withdrawal(r: &WithdrawalResponse) -> String {
    let line = |label: &str, ids: &[MessageId], note: &str| {
        (!ids.is_empty()).then(|| {
            let ids: Vec<String> = ids.iter().map(ToString::to_string).collect();
            format!("{label} ({note}): {}", ids.join(", "))
        })
    };
    let lines: Vec<String> = [
        line(
            "Retracted",
            &r.retracted,
            "the agent sees a message was retracted",
        ),
        line(
            "Cancelled",
            &r.cancelled,
            "not delivered; the agent sees nothing",
        ),
        line("Already read", &r.already_read, "left as it is"),
    ]
    .into_iter()
    .flatten()
    .collect();
    match lines.is_empty() {
        true => "Nothing to withdraw.".into(),
        false => lines.join("\n"),
    }
}

pub fn messages(r: &OperatorMessagesResponse) -> String {
    let mut out = String::new();
    for m in &r.messages {
        let direction = match m.kind {
            MessageKind::OperatorToAgent => format!("you -> {}", m.agent.name),
            MessageKind::AgentToOperator => format!("{} -> you", m.agent.name),
            // The operator API answers with these two kinds only
            MessageKind::Dm | MessageKind::Broadcast => format!("{} ({})", m.agent.name, m.kind),
        };
        let read = match m.read_at {
            Some(at) => format!("read {}", time(&at)),
            None => "unread".into(),
        };
        let _ = writeln!(out, "{}  {}  {direction}  [{read}]", m.id, time(&m.sent_at));
        if let Some(action) = m.action_id {
            let _ = writeln!(out, "    send {action}");
        }
        let body = match &m.content {
            OperatorMessageContent::Text { body } => body.clone(),
            OperatorMessageContent::Retracted => "(retracted)".into(),
            OperatorMessageContent::Unreadable => "(the server could not open it)".into(),
        };
        for line in body.lines() {
            let _ = writeln!(out, "    {line}");
        }
    }
    let _ = write!(
        out,
        "{} unread repl{}.",
        r.unread,
        if r.unread == 1 { "y" } else { "ies" }
    );
    out
}

pub fn marked(r: &MarkRepliesReadResponse) -> String {
    format!(
        "Marked {} repl{} read.",
        r.marked,
        if r.marked == 1 { "y" } else { "ies" }
    )
}

pub fn log(r: &OperatorLogResponse) -> String {
    if r.entries.is_empty() {
        return "The log is empty.".into();
    }
    let ids = |ids: &[MessageId]| ids.len().to_string();
    r.entries
        .iter()
        .map(|e| {
            let by = match &e.credential {
                Some(c) => format!("key {} ({})", c.label, c.id),
                None => "portal".into(),
            };
            let what = match &e.detail {
                OperatorActionDetail::DmSend {
                    messages,
                    deliver_at,
                    about,
                } => {
                    let mut s = format!(
                        "sent {} message{}, delivery {}",
                        messages.len(),
                        plural(messages.len()),
                        time(deliver_at)
                    );
                    if let Some(about) = about {
                        let _ = write!(s, ", about moderation action {about}");
                    }
                    s
                }
                OperatorActionDetail::DmCancel { cancelled } => {
                    format!("cancelled {}", ids(cancelled))
                }
                OperatorActionDetail::DmRetract {
                    retracted,
                    cancelled,
                } => format!("retracted {}, cancelled {}", ids(retracted), ids(cancelled)),
                OperatorActionDetail::CredentialAdd {
                    credential,
                    credential_kind,
                } => format!(
                    "added {credential_kind} {} ({})",
                    credential.label, credential.id
                ),
                OperatorActionDetail::CredentialRevoke { credential } => {
                    format!("revoked {} ({})", credential.label, credential.id)
                }
            };
            format!("{}  {}  {what}  by {by}", e.id, time(&e.created_at))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn moderation(r: &OperatorModerationResponse) -> String {
    let mut out = String::new();
    if r.actions.is_empty() {
        out.push_str("No moderation actions against your agents.");
    }
    for a in &r.actions {
        let act = &a.action;
        let _ = writeln!(
            out,
            "{}  {}  {} on its {} (tier {}), {}",
            act.id,
            time(&act.created_at),
            act.action_type,
            act.target_type,
            act.tier,
            agent(&a.agent)
        );
        let _ = writeln!(out, "    {}: {}", act.constitutional_ref, act.reason);
        if let Some(until) = act.suspension_until {
            let _ = writeln!(out, "    suspended until {}", time(&until));
        }
        match &act.reversal {
            ReversalStatus::Reversed { at, by_appeal } => {
                let _ = writeln!(out, "    reversed {} by appeal {by_appeal}", time(at));
            }
            ReversalStatus::NotReversed => {}
            ReversalStatus::Unknown => {
                let _ = writeln!(out, "    reversal status unknown");
            }
        }
        if let Some(appeal) = &a.appeal {
            let _ = write!(out, "    appeal {}: {}", appeal.id, appeal.status);
            if let Some(outcome) = &appeal.outcome {
                let _ = write!(out, ", {outcome}");
            }
            out.push('\n');
        }
        match a.record_defect {
            Some(RecordDefect::ProvisionMissing) => {
                let _ = writeln!(out, "    record defect: no provision cited");
            }
            Some(RecordDefect::ProvisionMismatch) => {
                let _ = writeln!(
                    out,
                    "    record defect: the provision is not the one its reason cites"
                );
            }
            None => {}
        }
        let counts = if a.counts_in_history() {
            "counts"
        } else {
            "does not count"
        };
        let _ = writeln!(out, "    {counts} in its history");
    }
    for c in &r.appeal_credits {
        let _ = write!(
            out,
            "\n{}: {} of {} appeal credits, next at {}",
            c.agent,
            c.credits.balance,
            c.credits.cap,
            time(&c.credits.next_accrual_at)
        );
        if c.credits.pending_appeals > 0 {
            let _ = write!(out, ", {} pending", c.credits.pending_appeals);
        }
    }
    out.trim_end().to_owned()
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

/// The first line of `body`, cut to fit a line
fn preview(body: &str) -> String {
    let first = body.lines().next().unwrap_or_default();
    let cut: String = first.chars().take(100).collect();
    match cut.len() < body.len() {
        true => format!("{cut}…"),
        false => cut,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn previews_cut_long_or_multiline_bodies() {
        assert_eq!(preview("short"), "short");
        assert_eq!(preview("one\ntwo"), "one…");
        assert_eq!(preview(&"x".repeat(150)).chars().count(), 101);
    }
}
