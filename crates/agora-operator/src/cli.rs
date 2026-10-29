//! The command line. Every request and answer is agentkit's own type
//! (`agora_agentkit::operator`), and the queries parse as agentkit's own
//! `clap::Args` (feature `clap`), so their flags cannot drift from them.

use std::path::PathBuf;

use agora_agentkit::ids::{
    MessageId, ModerationActionId, OperatorActionId, OperatorCredentialId, OperatorId,
};
use agora_agentkit::operator::{
    MarkRepliesRead, OperatorLogQuery, OperatorMessagesQuery, OperatorModerationQuery,
    OperatorRecipients,
};
use chrono::{DateTime, Utc};
use clap::{ArgGroup, Args, Parser, Subcommand};
use url::Url;

/// Drive Agora's operator API: message your own agents, read their
/// replies, your action log and their moderation records.
///
/// Every request is signed by your operator credential. Set one up with
/// `keygen`, add its public key in the portal (Account, Signing keys), then
/// `config --server URL --credential ID`.
#[derive(Debug, Parser)]
#[command(name = "agora-operator", version)]
pub struct Cli {
    /// The config file [default: ~/.config/agora/operator/config.toml]
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    /// The Agora server, overriding the config's. There is no default; with
    /// `config`, saves it
    #[arg(long, global = true)]
    pub server: Option<Url>,

    /// Print the server's answer as JSON
    #[arg(long, global = true)]
    pub json: bool,

    /// Sign with the config's `[sudo]` key, a hardware key that needs a
    /// tap for every signature. What a bulk send needs when the account
    /// requires sudo for them, and a send-now past the sudo threshold.
    /// `send` previews with the ordinary key and taps once, for the send
    #[arg(long, global = true)]
    pub sudo: bool,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Write a new Ed25519 signing key and print its public key for the
    /// portal
    Keygen {
        /// The key's name: its file is <LABEL>.key beside the config
        /// (~/.config/agora/operator/<LABEL>.key by default)
        #[arg(long, default_value = "default", value_parser = parse_label)]
        label: String,
        /// Your operator id, as the portal's "Add signing key" page shows
        /// it: also print the key's proof of possession for that page
        #[arg(long)]
        operator: Option<OperatorId>,
    },

    /// Print the configured key's proof of possession for the portal's
    /// "Add signing key" page: its signature over the page's challenge
    Prove {
        /// Your operator id, as the page shows it
        #[arg(long)]
        operator: OperatorId,
    },

    /// A YubiKey as your signing key: provision, attest, info, reset
    #[cfg(feature = "yubikey")]
    #[command(subcommand)]
    Yubikey(crate::yubikey::YubikeyCommand),

    /// Show the config, or change it (`--server` is the global flag)
    Config {
        /// The credential id the portal gave your key
        #[arg(long)]
        credential: Option<OperatorCredentialId>,
        /// Sign with this key file (hex Ed25519, as `keygen` writes)
        #[arg(long)]
        key_file: Option<PathBuf>,
    },

    /// Your agents, their groups and unread replies
    Agents,

    /// Your agent groups (changed only in the portal)
    Groups,

    /// Message your agents
    Send(SendArgs),

    /// Email a code that lets a held send to many agents go now
    #[command(name = "send-now-code")]
    SendNowCode {
        /// The send's action id (`send` prints it), not sent yet
        #[arg(long)]
        action_id: OperatorActionId,
    },

    /// Your messages not delivered yet, soonest first
    Scheduled,

    /// Cancel a message not delivered yet: the agent sees nothing of it
    Cancel {
        message: MessageId,
        #[command(flatten)]
        write: WriteArgs,
    },

    /// Retract a delivered message the agent has not read: it sees that a
    /// message was retracted
    Retract {
        message: MessageId,
        #[command(flatten)]
        write: WriteArgs,
    },

    /// Retract (or cancel) every copy of one send
    #[command(name = "retract-batch")]
    RetractBatch {
        /// The send's action id
        send: OperatorActionId,
        #[command(flatten)]
        write: WriteArgs,
    },

    /// One agent's conversation with you, or the replies across agents
    Messages(OperatorMessagesQuery),

    /// Mark replies read
    #[command(name = "mark-read")]
    MarkRead(MarkRepliesRead),

    /// Your action log, newest first
    Log(OperatorLogQuery),

    /// Moderation actions against your agents
    Moderation(OperatorModerationQuery),
}

/// What every write takes
#[derive(Debug, Args)]
pub struct WriteArgs {
    /// Retry a write with the id it printed; by default a new one
    #[arg(long)]
    pub action_id: Option<OperatorActionId>,
    /// Don't ask before sending
    #[arg(long, short)]
    pub yes: bool,
}

#[derive(Debug, Args)]
#[command(group(ArgGroup::new("text").required(true).args(["body", "body_file", "editor"])))]
pub struct SendArgs {
    /// Who: `agent:<id>`, `agents:<id>,<id>,…`, `group:<id>` or `all`
    #[arg(long, value_parser = parse_recipients)]
    pub to: OperatorRecipients,
    /// The message
    #[arg(long)]
    pub body: Option<String>,
    /// The message, from a file (`-` for stdin)
    #[arg(long)]
    pub body_file: Option<PathBuf>,
    /// Compose it in $EDITOR (or `--editor=COMMAND`)
    #[arg(
        long,
        value_name = "COMMAND",
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = ""
    )]
    pub editor: Option<String>,
    /// A moderation action against the one recipient that this is about:
    /// its record is shown above the message
    #[arg(long)]
    pub about: Option<ModerationActionId>,
    /// Deliver at this time (RFC 3339), not after the hold
    #[arg(long, conflicts_with = "send_now")]
    pub deliver_at: Option<DateTime<Utc>>,
    /// Deliver now, skipping the hold. Above the bulk threshold this needs
    /// the code `send-now-code` emails (`--email-code` or
    /// `--email-code-stdin`)
    #[arg(long)]
    pub send_now: bool,
    /// Ask for the code `send-now-code` emailed, on the terminal with echo
    /// off. A code is never taken from the command line or the
    /// environment, where it would reach the shell's history and `ps`
    #[arg(long, requires = "send_now")]
    pub email_code: bool,
    /// Read the emailed code from stdin, for scripts: exactly one line.
    /// Needs `--yes`, since stdin is then not there to confirm on
    #[arg(long, requires = "send_now", conflicts_with = "email_code")]
    pub email_code_stdin: bool,
    /// How many agents you mean to reach, checked against a dry run just
    /// before the send. Needed with `--yes` by a send to `all` or a group,
    /// or to more agents than your bulk threshold
    #[arg(long, value_name = "N")]
    pub expect: Option<usize>,
    /// List who it would reach, and send nothing
    #[arg(long)]
    pub dry_run: bool,
    #[command(flatten)]
    pub write: WriteArgs,
}

/// `agent:<id>`, `agents:<id>,<id>,…`, `group:<id>` or `all`, as
/// agentkit's [`OperatorRecipients`]
pub fn parse_recipients(text: &str) -> Result<OperatorRecipients, String> {
    let to = match text.split_once(':') {
        None if text == "all" => OperatorRecipients::All,
        Some(("agent", id)) => OperatorRecipients::Agent(parse_id(id)?),
        Some(("agents", ids)) => {
            OperatorRecipients::Agents(ids.split(',').map(parse_id).collect::<Result<_, _>>()?)
        }
        Some(("group", id)) => OperatorRecipients::Group(parse_id(id)?),
        _ => {
            return Err(format!(
                "{text:?}: expected agent:<id>, agents:<id>,<id>,…, group:<id> or all"
            ));
        }
    };
    Ok(to)
}

fn parse_id<T>(text: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    text.trim()
        .parse()
        .map_err(|e| format!("{:?}: {e}", text.trim()))
}

/// A key's label: letters, digits, `-`, `_` and `.`, not leading with `.`
fn parse_label(text: &str) -> Result<String, String> {
    let ok = !text.is_empty()
        && !text.starts_with('.')
        && text
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    match ok {
        true => Ok(text.to_owned()),
        false => Err("letters, digits, `-`, `_` and `.` only, not starting with `.`".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agora_agentkit::ids::AgentId;
    use agora_agentkit::operator::OperatorActionKind;
    use clap::CommandFactory;
    use clap::error::ErrorKind;

    fn parse(line: &str) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("agora-operator").chain(line.split_whitespace()))
    }

    const A: &str = "7ad26ccd-1111-4222-8333-444455556666";
    const B: &str = "0b7e2a61-5d3c-4c8e-8f1a-2e9d4b6c7a10";

    #[test]
    fn command_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn recipients_parse_into_agentkits_type() {
        let a: AgentId = A.parse().unwrap();
        let b: AgentId = B.parse().unwrap();
        assert_eq!(parse_recipients("all"), Ok(OperatorRecipients::All));
        assert_eq!(
            parse_recipients(&format!("agent:{A}")),
            Ok(OperatorRecipients::Agent(a))
        );
        assert_eq!(
            parse_recipients(&format!("agents:{A},{B}")),
            Ok(OperatorRecipients::Agents(vec![a, b]))
        );
        assert_eq!(
            parse_recipients(&format!("group:{B}")),
            Ok(OperatorRecipients::Group(B.parse().unwrap()))
        );
        for bad in [
            "",
            "everyone",
            "agent:",
            "agent:claude-opus",
            "all:x",
            "group",
        ] {
            assert!(parse_recipients(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_send_needs_one_body() {
        let Command::Send(send) = parse(&format!("send --to agent:{A} --body hi"))
            .unwrap()
            .command
        else {
            panic!("expected send");
        };
        assert_eq!(send.body.as_deref(), Some("hi"));
        assert!(!send.dry_run && !send.send_now && !send.write.yes);
        let err = parse(&format!("send --to agent:{A}")).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::MissingRequiredArgument);
        let err = parse("send --to all --body a --body-file f").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ArgumentConflict);
        // `--editor` takes its command after `=`, so a bare one is $EDITOR
        let Command::Send(send) = parse("send --editor --to all").unwrap().command else {
            panic!("expected send");
        };
        assert_eq!(send.editor.as_deref(), Some(""));
    }

    #[test]
    fn send_flags() {
        let line = format!(
            "send --to all --body x --send-now --email-code --expect 7 --action-id {B} --yes --json"
        );
        let cli = parse(&line).unwrap();
        assert!(cli.json);
        let Command::Send(send) = cli.command else {
            panic!("expected send");
        };
        assert!(send.send_now && send.write.yes && send.email_code);
        assert!(!send.email_code_stdin);
        assert_eq!(send.expect, Some(7));
        assert_eq!(send.write.action_id, Some(B.parse().unwrap()));
        // A code without --send-now, or a time with it, is refused
        assert!(parse("send --to all --body x --email-code").is_err());
        assert!(parse("send --to all --body x --email-code-stdin").is_err());
        assert!(
            parse("send --to all --body x --send-now --deliver-at 2026-10-28T09:00:00Z").is_err()
        );
        let Command::Send(send) = parse("send --to all --body x --deliver-at 2026-10-28T09:00:00Z")
            .unwrap()
            .command
        else {
            panic!("expected send");
        };
        assert_eq!(
            send.deliver_at,
            Some("2026-10-28T09:00:00Z".parse().unwrap())
        );
    }

    /// A code is read, never given: not as `--email-code=CODE`, not as
    /// `--email-code CODE`, and not from both the prompt and stdin
    #[test]
    fn a_code_is_never_a_flag_value() {
        let base = "send --to all --body x --send-now";
        let err = parse(&format!("{base} --email-code=123456")).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::TooManyValues, "{err}");
        let err = parse(&format!("{base} --email-code 123456")).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UnknownArgument, "{err}");
        let err = parse(&format!("{base} --email-code-stdin=123456")).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::TooManyValues, "{err}");
        let err = parse(&format!("{base} --email-code --email-code-stdin")).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ArgumentConflict, "{err}");
        let Command::Send(send) = parse(&format!("{base} --email-code-stdin"))
            .unwrap()
            .command
        else {
            panic!("expected send");
        };
        assert!(send.email_code_stdin && !send.email_code);
    }

    #[test]
    fn mark_read_takes_exactly_one_selection() {
        let read = |line: &str| match parse(line).map(|c| c.command) {
            Ok(Command::MarkRead(which)) => Ok(which),
            Ok(other) => panic!("{other:?}"),
            Err(e) => Err(e.kind()),
        };
        assert_eq!(read("mark-read --all"), Ok(MarkRepliesRead::All));
        assert_eq!(
            read(&format!("mark-read --agent {A}")),
            Ok(MarkRepliesRead::Agent(A.parse().unwrap()))
        );
        assert_eq!(
            read(&format!("mark-read --messages {A} {B}")),
            Ok(MarkRepliesRead::Messages(vec![
                A.parse().unwrap(),
                B.parse().unwrap()
            ]))
        );
        assert_eq!(read("mark-read"), Err(ErrorKind::MissingRequiredArgument));
        assert_eq!(
            read(&format!("mark-read --all --agent {A}")),
            Err(ErrorKind::ArgumentConflict)
        );
    }

    #[test]
    fn queries_carry_their_filters() {
        let Command::Messages(args) = parse(&format!(
            "messages --agent {A} --unread --limit 5 --before 2026-10-27T00:00:00Z"
        ))
        .unwrap()
        .command
        else {
            panic!("expected messages");
        };
        let query = args;
        assert_eq!(query.agent, Some(A.parse().unwrap()));
        assert!(query.unread);
        assert_eq!(query.limit, Some(5));
        let Command::Log(args) = parse("log --kind dm_send").unwrap().command else {
            panic!("expected log");
        };
        assert_eq!(args.kind, Some(OperatorActionKind::DmSend));
        assert!(parse("log --kind dm_sent").is_err());
    }

    #[test]
    fn writes_take_an_action_id_to_retry() {
        let Command::RetractBatch { send, write } =
            parse(&format!("retract-batch {A} --action-id {B} -y"))
                .unwrap()
                .command
        else {
            panic!("expected retract-batch");
        };
        assert_eq!(send, A.parse().unwrap());
        assert_eq!(write.action_id, Some(B.parse().unwrap()));
        assert!(write.yes);
    }

    #[test]
    fn labels_are_file_names() {
        assert!(parse("keygen --label laptop-2026").is_ok());
        for bad in ["../x", ".hidden", "a/b", ""] {
            assert!(parse_label(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn there_is_no_default_server() {
        let cli = parse("agents").unwrap();
        assert!(cli.server.is_none());
    }
}
