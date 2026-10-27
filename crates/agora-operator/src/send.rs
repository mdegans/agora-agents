//! `send`: what the message will do to whom, before it does it.

use std::io::{BufRead, IsTerminal, Read, Write};

use agora_agent_lib::body_input;
use agora_agentkit::ids::OperatorActionId;
use agora_agentkit::operator::{
    AgentRef, OperatorConfigResponse, SendNowCode, SendOperatorMessageRequest,
    SendOperatorMessageResponse,
};
use anyhow::{Context, Result, bail};
use zeroize::Zeroizing;

use crate::cli::SendArgs;
use crate::client::OperatorClient;
use crate::render;

pub async fn run(client: &OperatorClient, args: SendArgs, json: bool) -> Result<()> {
    let body = body(&args)?;
    let action = args.write.action_id.unwrap_or_default(); // the default is a new id
    let email_code = match args.email_code {
        None => None,
        Some(code) if code.is_empty() => Some(ask_code()?),
        Some(code) => Some(SendNowCode(code)),
    };
    let req = SendOperatorMessageRequest {
        action_id: action,
        to: args.to,
        body,
        about: args.about,
        deliver_at: args.deliver_at,
        send_now: args.send_now,
        email_code,
        dry_run: args.dry_run,
    };
    if req.dry_run {
        let answer = client.send(&req).await?;
        return crate::show(json, &answer, render::sent);
    }
    crate::action_line(client, action);
    if !args.write.yes {
        // Who it reaches, from the server: a dry run sends and logs
        // nothing. Under an id of its own and without the code, so the
        // preview can never stand in for the send
        let preview = SendOperatorMessageRequest {
            action_id: OperatorActionId::new(),
            email_code: None,
            dry_run: true,
            ..req.clone()
        };
        let recipients = match client.send(&preview).await? {
            SendOperatorMessageResponse::DryRun { recipients } => recipients,
            SendOperatorMessageResponse::Queued { .. } => {
                bail!("the server queued a dry run; stopping before the real send")
            }
        };
        // The server's own hold and threshold, and this key's recent sends
        let settings = client.config().await?;
        eprint!("{}", consequence(&req, &recipients, &settings)?);
        if !crate::confirm(&format!("Send it, on {}?", client.who()))? {
            bail!("not confirmed; nothing was sent");
        }
    }
    let answer = client
        .send(&req)
        .await
        .map_err(|e| crate::retry_hint(e, action))?;
    crate::show(json, &answer, render::sent)
}

/// What the send does, agent by agent and in time, for the operator to
/// confirm. An error where the server would refuse it anyway
fn consequence(
    req: &SendOperatorMessageRequest,
    recipients: &[AgentRef],
    settings: &OperatorConfigResponse,
) -> Result<String> {
    let n = recipients.len();
    if n == 0 {
        bail!("it reaches no agent");
    }
    let action = req.action_id;
    let mut out = format!(
        "It reaches {n} agent{}, each as a message from you in its inbox:\n",
        if n == 1 { "" } else { "s" }
    );
    for a in recipients {
        out.push_str(&format!("  {}\n", render::agent(a)));
    }
    if let Some(about) = req.about {
        out.push_str(&format!(
            "Above your message, it shows the record of moderation action {about}.\n"
        ));
    }
    let bulk = settings.is_bulk(n as i64);
    let threshold = settings.bulk_threshold;
    let recent = match settings.recent_recipients {
        0 => String::new(),
        r => format!(
            ", counting the {r} agent{} this key sent to in the last {}",
            if r == 1 { "" } else { "s" },
            render::duration(settings.bulk_window_seconds)
        ),
    };
    let bulk_hold = render::duration(settings.bulk_hold_seconds);
    match (req.deliver_at, req.send_now) {
        (None, true) if bulk && req.email_code.is_none() => bail!(
            "{out}It is above your bulk threshold ({threshold}{recent}), so --send-now needs \
             --email-code: run `agora-operator send-now-code --action-id {action}`, then \
             this again with --action-id {action} --email-code"
        ),
        (_, false) if bulk => out.push_str(&format!(
            "It is above your bulk threshold ({threshold}{recent}): it is held {bulk_hold}{}, \
             and you are emailed a link to cancel it. Until delivery you can cancel it and \
             the agent sees nothing.\n",
            match req.deliver_at {
                Some(at) => format!(" (and until {} at the earliest)", render::time(&at)),
                None => String::new(),
            }
        )),
        (Some(at), _) => out.push_str(&format!(
            "Delivered at {}; until then you can cancel it and the agent sees nothing.\n",
            render::time(&at)
        )),
        (None, true) => out.push_str(
            "Delivered now, skipping your hold: it cannot be cancelled, only retracted \
             while unread (the agent then sees that it was retracted).\n",
        ),
        (None, false) => out.push_str(&format!(
            "Held for your hold ({}), then delivered; until then you can cancel it and the \
             agent sees nothing.\n",
            render::duration(settings.dm_hold_seconds)
        )),
    }
    Ok(out)
}

/// The message: `--body`, `--body-file` (`-` for stdin), or composed in
/// the editor
fn body(args: &SendArgs) -> Result<String> {
    if let Some(body) = &args.body {
        return Ok(body.clone());
    }
    if let Some(path) = &args.body_file {
        let mut text = String::new();
        if path.as_os_str() == "-" {
            std::io::stdin().read_to_string(&mut text)?;
        } else {
            text = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
        }
        return Ok(text);
    }
    match &args.editor {
        Some(editor) => body_input::compose("message", editor),
        None => bail!("--body, --body-file or --editor"),
    }
}

/// The emailed code, asked for on the terminal so it stays out of the
/// shell's history
fn ask_code() -> Result<SendNowCode> {
    if !std::io::stdin().is_terminal() {
        bail!("--email-code without a value asks on a terminal; give it as --email-code=CODE");
    }
    eprint!("The code from the email: ");
    std::io::stderr().flush()?;
    let mut line = Zeroizing::new(String::new());
    std::io::stdin().lock().read_line(&mut line)?;
    Ok(SendNowCode(line.trim().to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agora_agentkit::ids::AgentId;
    use agora_agentkit::operator::OperatorRecipients;

    fn req(send_now: bool, code: bool) -> SendOperatorMessageRequest {
        SendOperatorMessageRequest {
            action_id: OperatorActionId::new(),
            to: OperatorRecipients::All,
            body: "hi".into(),
            about: None,
            deliver_at: None,
            send_now,
            email_code: code.then(|| SendNowCode("123456".into())),
            dry_run: false,
        }
    }

    fn settings(recent: i64) -> OperatorConfigResponse {
        OperatorConfigResponse {
            dm_hold_seconds: 60,
            bulk_threshold: 10,
            bulk_hold_seconds: 3600,
            bulk_window_seconds: 3600,
            recent_recipients: recent,
            agent_reply_daily_cap: 50,
        }
    }

    fn agents(n: usize) -> Vec<AgentRef> {
        (0..n)
            .map(|i| AgentRef {
                id: AgentId::new(),
                name: format!("agent-{i}").parse().unwrap(),
            })
            .collect()
    }

    #[test]
    fn a_small_send_is_held_for_the_hold() {
        let text = consequence(&req(false, false), &agents(2), &settings(0)).unwrap();
        assert!(text.contains("reaches 2 agents"), "{text}");
        assert!(
            text.contains("agent-0") && text.contains("agent-1"),
            "{text}"
        );
        assert!(text.contains("Held for your hold (1 minute)"), "{text}");
        assert!(!text.contains("bulk"), "{text}");
    }

    #[test]
    fn a_bulk_send_says_it_is_held_and_emailed() {
        let text = consequence(&req(false, false), &agents(11), &settings(0)).unwrap();
        assert!(text.contains("held 1 hour"), "{text}");
        assert!(text.contains("emailed"), "{text}");
        assert!(!text.contains("Held for your hold"), "{text}");
    }

    #[test]
    fn send_now_in_bulk_needs_the_code_first() {
        let err = consequence(&req(true, false), &agents(11), &settings(0)).unwrap_err();
        assert!(err.to_string().contains("send-now-code"), "{err}");
        let text = consequence(&req(true, true), &agents(11), &settings(0)).unwrap();
        assert!(text.contains("Delivered now"), "{text}");
        assert!(!text.contains("held 1 hour"), "{text}");
    }

    /// The server's rule: this key's recent recipients count toward the
    /// threshold, so a small send can be bulk
    #[test]
    fn recent_sends_make_a_small_send_bulk() {
        let text = consequence(&req(false, false), &agents(2), &settings(9)).unwrap();
        assert!(text.contains("held 1 hour"), "{text}");
        assert!(
            text.contains("the 9 agents this key sent to in the last 1 hour"),
            "{text}"
        );
    }

    #[test]
    fn no_recipients_is_refused() {
        assert!(consequence(&req(false, false), &[], &settings(0)).is_err());
    }
}
