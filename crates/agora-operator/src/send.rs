//! `send`: what the message will do to whom, before it does it.

use std::io::{IsTerminal, Read};

use agora_agent_lib::body_input;
use agora_agent_lib::secret_input::{self, SecretSource};
use agora_agentkit::ids::OperatorActionId;
use agora_agentkit::operator::{
    AgentRef, OperatorConfigResponse, OperatorRecipients, SendNowCode, SendOperatorMessageRequest,
    SendOperatorMessageResponse,
};
use anyhow::{Context, Result, bail};

use crate::cli::SendArgs;
use crate::client::OperatorClient;
use crate::render;

pub async fn run(client: &OperatorClient, args: SendArgs, json: bool) -> Result<()> {
    let code_source = code_source(&args)?;
    let body = body(&args)?;
    let action = args.write.action_id.unwrap_or_default(); // the default is a new id
    let email_code = match code_source {
        None => None,
        Some(source) => {
            let mut code = secret_input::read_secret("The code from the email", source)?;
            // Moved, not copied: `SendNowCode` wipes it when dropped
            Some(SendNowCode(std::mem::take(&mut *code)))
        }
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
    // Who it reaches, from the server, even with --yes: a dry run sends and
    // logs nothing. Under an id of its own and without the code, so the
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
    check_count(
        &req.to,
        recipients.len(),
        &settings,
        args.expect,
        args.write.yes,
    )?;
    let text = consequence(&req, &recipients, &settings)?;
    if !args.write.yes {
        eprint!("{}", headline_bold(&text, std::io::stderr().is_terminal()));
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

/// Where the emailed code is read from, if one is wanted
fn code_source(args: &SendArgs) -> Result<Option<SecretSource>> {
    match (args.email_code, args.email_code_stdin) {
        (_, true)
            if args
                .body_file
                .as_ref()
                .is_some_and(|p| p.as_os_str() == "-") =>
        {
            bail!("--email-code-stdin and --body-file - both want stdin; give the body another way")
        }
        (_, true) => Ok(Some(SecretSource::Stdin)),
        (true, false) => Ok(Some(SecretSource::Prompt)),
        (false, false) => Ok(None),
    }
}

/// Whether to whom it goes needs `--expect`: a send to everyone or a group,
/// or to more agents than the bulk threshold. An `all` meant as "my few
/// test agents" reached 1,707 agents on 2026-10-27
fn needs_expect(to: &OperatorRecipients, n: usize, settings: &OperatorConfigResponse) -> bool {
    matches!(to, OperatorRecipients::All | OperatorRecipients::Group(_))
        || i64::try_from(n).map_or(true, |n| n > settings.bulk_threshold)
}

/// `--expect`, checked against the dry run's count of `n` recipients: any
/// send that names one must reach exactly that many, and one that
/// [`needs_expect`] cannot go with `--yes` alone
fn check_count(
    to: &OperatorRecipients,
    n: usize,
    settings: &OperatorConfigResponse,
    expect: Option<usize>,
    yes: bool,
) -> Result<()> {
    let agents = |n: usize| format!("{n} agent{}", if n == 1 { "" } else { "s" });
    match expect {
        Some(e) if e != n => bail!(
            "it reaches {}, not the {} --expect says; nothing was sent. If {} is meant, \
             pass --expect {n}",
            agents(n),
            e,
            agents(n)
        ),
        Some(_) => Ok(()),
        None if yes && needs_expect(to, n, settings) => bail!(
            "it reaches {} ({}); with --yes that needs --expect {n}. Nothing was sent",
            agents(n),
            match to {
                OperatorRecipients::All => "all your agents".to_owned(),
                OperatorRecipients::Group(_) => "a group".to_owned(),
                _ => format!("above your bulk threshold of {}", settings.bulk_threshold),
            }
        ),
        None => Ok(()),
    }
}

/// `text` with its first line in bold, on a terminal
fn headline_bold(text: &str, terminal: bool) -> String {
    match (terminal, text.split_once('\n')) {
        (true, Some((first, rest))) => format!("\x1b[1m{first}\x1b[0m\n{rest}"),
        _ => text.to_owned(),
    }
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
    // The count first and loudest, before a list that may scroll it away
    let mut out = format!(
        "IT REACHES {n} AGENT{}.\nEach gets it as a message from you in its inbox:\n",
        if n == 1 { "" } else { "S" }
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
             the emailed code: run `agora-operator send-now-code --action-id {action}`, then \
             this again with --action-id {action} --email-code (or --email-code-stdin)"
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

#[cfg(test)]
mod tests {
    use super::*;
    use agora_agentkit::ids::{AgentGroupId, AgentId};

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
        assert!(text.starts_with("IT REACHES 2 AGENTS.\n"), "{text}");
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

    /// The count leads, in caps, and in bold on a terminal only
    #[test]
    fn the_count_is_the_first_line() {
        let text = consequence(&req(false, false), &agents(1), &settings(0)).unwrap();
        assert!(text.starts_with("IT REACHES 1 AGENT.\n"), "{text}");
        let bold = headline_bold(&text, true);
        assert!(
            bold.starts_with("\x1b[1mIT REACHES 1 AGENT.\x1b[0m\n"),
            "{bold}"
        );
        assert_eq!(headline_bold(&text, false), text);
    }

    fn group() -> OperatorRecipients {
        OperatorRecipients::Group(AgentGroupId::new())
    }

    fn some(n: usize) -> OperatorRecipients {
        OperatorRecipients::Agents((0..n).map(|_| AgentId::new()).collect())
    }

    /// The 2026-10-27 incident: `--to all --yes` meant six agents and
    /// addressed 1,707
    #[test]
    fn all_with_yes_needs_expect_and_names_the_count() {
        let err = check_count(&OperatorRecipients::All, 1707, &settings(0), None, true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("reaches 1707 agents"), "{err}");
        assert!(err.contains("--expect 1707"), "{err}");
        // Even when "all" is few
        assert!(check_count(&OperatorRecipients::All, 6, &settings(0), None, true).is_err());
        assert!(check_count(&OperatorRecipients::All, 6, &settings(0), Some(6), true).is_ok());
    }

    #[test]
    fn a_group_with_yes_needs_expect() {
        let err = check_count(&group(), 3, &settings(0), None, true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("--expect 3"), "{err}");
        assert!(check_count(&group(), 3, &settings(0), Some(3), true).is_ok());
    }

    #[test]
    fn above_the_threshold_with_yes_needs_expect() {
        let err = check_count(&some(11), 11, &settings(0), None, true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("bulk threshold of 10"), "{err}");
        assert!(err.contains("--expect 11"), "{err}");
        assert!(check_count(&some(11), 11, &settings(0), Some(11), true).is_ok());
        // At the threshold, a named list goes with --yes alone
        assert!(check_count(&some(10), 10, &settings(0), None, true).is_ok());
    }

    #[test]
    fn a_wrong_expect_is_refused_with_or_without_yes() {
        for yes in [true, false] {
            let err = check_count(&OperatorRecipients::All, 1707, &settings(0), Some(6), yes)
                .unwrap_err()
                .to_string();
            assert!(err.contains("reaches 1707 agents, not the 6"), "{err}");
            assert!(err.contains("--expect 1707"), "{err}");
            assert!(check_count(&some(2), 2, &settings(0), Some(3), yes).is_err());
        }
    }

    /// Asked on the terminal, the confirmation (count first) is the guard
    #[test]
    fn asked_on_the_terminal_expect_is_optional() {
        assert!(check_count(&OperatorRecipients::All, 1707, &settings(0), None, false).is_ok());
        assert!(check_count(&group(), 3, &settings(0), None, false).is_ok());
    }

    #[test]
    fn stdin_cannot_carry_both_body_and_code() {
        use clap::Parser;
        let parse = |line: &str| {
            let cli = crate::cli::Cli::try_parse_from(
                std::iter::once("agora-operator").chain(line.split_whitespace()),
            )
            .unwrap();
            let crate::cli::Command::Send(send) = cli.command else {
                panic!("expected send");
            };
            send
        };
        let both = parse("send --to all --body-file - --send-now --email-code-stdin");
        let err = code_source(&both).unwrap_err().to_string();
        assert!(err.contains("both want stdin"), "{err}");
        let prompt = parse("send --to all --body-file - --send-now --email-code");
        assert_eq!(code_source(&prompt).unwrap(), Some(SecretSource::Prompt));
        let stdin = parse("send --to all --body x --send-now --email-code-stdin");
        assert_eq!(code_source(&stdin).unwrap(), Some(SecretSource::Stdin));
        let none = parse("send --to all --body x");
        assert_eq!(code_source(&none).unwrap(), None);
    }
}
