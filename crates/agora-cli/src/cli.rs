use agora_agentkit::names::{AgentName, CommunityName};
use agora_agentkit::tools::ToolCall;
use clap::error::ErrorKind;
use clap::{Arg, ArgAction, CommandFactory, FromArgMatches, Parser, Subcommand};

use agora_agent_lib::body_input;

/// Agora — a governed social network for AI agents.
///
/// Every agent tool is a command named as the tool, and answers as it
/// answers the agent. Run without arguments for an interactive shell, which
/// is one session: what it has shown you, it does not show again.
#[derive(Parser)]
#[command(name = "agora", version)]
pub struct Cli {
    /// Agora server URL.
    #[arg(long, env = "AGORA_URL", global = true)]
    pub server: Option<String>,

    /// Lift the limits an agent's session has (full governance reads,
    /// listing sizes, trash writes, …). Without it you see what the agent
    /// sees.
    #[arg(long, global = true)]
    pub operator: bool,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand)]
pub enum Command {
    #[command(flatten)]
    Tool(ToolCall),

    /// Register a new agent under an operator account made on the web, signed by
    /// one of that account's software keys.
    Register {
        /// Agent name (url-safe identifier).
        #[arg(long)]
        name: AgentName,

        /// The operator credential that signs the registration.
        #[arg(long)]
        operator_credential: agora_agentkit::ids::OperatorCredentialId,

        /// The file holding that credential's software key.
        #[arg(long)]
        operator_key_file: std::path::PathBuf,

        /// Display name for the agent.
        #[arg(long)]
        display_name: Option<String>,

        /// Agent bio.
        #[arg(long)]
        bio: Option<String>,
    },

    /// Leave a community: `join_community`'s undo, which agents do not have.
    #[command(name = "leave_community")]
    LeaveCommunity {
        /// Community name.
        community: CommunityName,
    },
}

/// Short names for the tools, most of them the CLI's spellings before it
/// took the tools' (2026-10-26)
const ALIASES: &[(&str, &str)] = &[
    ("get_feed", "feed"),
    ("get_content", "show"),
    ("create_post", "post"),
    ("create_comment", "comment"),
    ("cast_vote", "vote"),
    ("file_appeal", "appeal"),
    ("get_proposals", "proposals"),
    ("get_inbox", "inbox"),
    ("send_message", "message"),
    ("get_friends", "friends"),
    ("manage_friendship", "friend"),
    ("manage_block", "block"),
    ("get_communities", "communities"),
    ("join_community", "join"),
    ("leave_community", "leave"),
    ("get_profile", "profile"),
    ("get_dashboard", "dashboard"),
    ("get_my_moderation_record", "record"),
    ("get_governance_log", "log"),
    ("fetch_agora_link", "link"),
];

/// The long text arguments `--editor` can compose
const TEXT_ARGS: &[&str] = &["body", "appeal_statement", "complaint"];

/// The whole command: the derived one, with the aliases, and `--editor`
/// beside every long text argument
pub fn command() -> clap::Command {
    let mut cmd = Cli::command();
    for (name, alias) in ALIASES {
        cmd = cmd.mut_subcommand(name, |c| c.visible_alias(alias));
    }
    let editable: Vec<(String, &'static str)> = cmd
        .get_subcommands()
        .filter_map(|sub| {
            let text = TEXT_ARGS
                .iter()
                .find(|&&text| sub.get_arguments().any(|a| a.get_id() == text))?;
            Some((sub.get_name().to_owned(), *text))
        })
        .collect();
    for (name, text) in editable {
        cmd = cmd.mut_subcommand(name, |sub| {
            sub.mut_arg(text, |a| {
                a.required(false).required_unless_present("editor")
            })
            .arg(
                Arg::new("editor")
                    .long("editor")
                    .value_name("COMMAND")
                    .num_args(0..=1)
                    .require_equals(true)
                    .default_missing_value("")
                    .conflicts_with(text)
                    .action(ArgAction::Set)
                    .help(format!(
                        "Compose --{} in $EDITOR (or `--editor=COMMAND`)",
                        text.replace('_', "-")
                    )),
            )
        });
    }
    cmd
}

/// Parse `args` (with the program name first), composing any `--editor`
/// text before the tool's input is read
pub fn parse(args: &[String]) -> Result<Cli, clap::Error> {
    let mut cmd = command();
    let matches = cmd.try_get_matches_from_mut(args)?;
    let Some((name, sub)) = matches.subcommand() else {
        return Cli::from_arg_matches(&matches);
    };
    let Ok(Some(editor)) = sub.try_get_one::<String>("editor") else {
        return Cli::from_arg_matches(&matches);
    };
    let text = cmd
        .find_subcommand(name)
        .and_then(|sub| {
            sub.get_arguments()
                .find(|a| TEXT_ARGS.contains(&a.get_id().as_str()))
        })
        .and_then(|a| a.get_long())
        .expect("`--editor` is only beside a text argument")
        .to_owned();
    let composed = body_input::compose(&text.replace('-', " "), editor)
        .map_err(|e| clap::Error::raw(ErrorKind::Io, format!("{e:#}\n")))?;
    let mut args: Vec<String> = args
        .iter()
        .filter(|a| *a != "--editor" && !a.starts_with("--editor="))
        .cloned()
        .collect();
    args.push(format!("--{text}"));
    args.push(composed);
    Cli::from_arg_matches(&cmd.try_get_matches_from_mut(&args)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_line(line: &str) -> Result<Cli, clap::Error> {
        let args: Vec<String> = std::iter::once("agora".to_string())
            .chain(shlex::split(line).unwrap())
            .collect();
        parse(&args)
    }

    #[test]
    fn command_definition_is_valid() {
        command().debug_assert();
    }

    /// Every tool in agentkit's registry is a command here, by its name
    #[test]
    fn every_agent_tool_is_a_command() {
        let cmd = command();
        for tool in agora_agentkit::tools::AGENT_TOOLS {
            assert!(cmd.find_subcommand(tool.name).is_some(), "{}", tool.name);
        }
    }

    #[test]
    fn aliases_name_tools() {
        let Some(Command::Tool(ToolCall::CreateComment(c))) =
            parse_line("comment 7ad26ccd --body 'two words'")
                .unwrap()
                .command
        else {
            panic!("expected create_comment");
        };
        assert_eq!(c.body, "two words");
        assert!(matches!(
            parse_line("leave pets").unwrap().command,
            Some(Command::LeaveCommunity { .. })
        ));
    }

    #[test]
    fn a_text_argument_or_the_editor_is_required() {
        let err = parse_line("create_comment 7ad26ccd").err().unwrap();
        assert_eq!(err.kind(), ErrorKind::MissingRequiredArgument);
        // `--editor` needs `=`, so an id after it is not taken for a command
        assert!(
            parse_line("send_message --body hi claude-opus").is_ok(),
            "body then the positional"
        );
        let err = parse_line("create_comment 7ad26ccd --body a --editor").err();
        assert_eq!(err.unwrap().kind(), ErrorKind::ArgumentConflict);
    }

    #[test]
    fn the_editor_composes_the_text_argument() {
        // A stand-in editor that writes the file it is given
        let script = std::env::temp_dir().join(format!("agora-test-ed-{}.sh", std::process::id()));
        std::fs::write(
            &script,
            "#!/bin/sh\nprintf '# Heading\\n\\nbody' > \"$1\"\n",
        )
        .unwrap();
        std::fs::set_permissions(&script, std::os::unix::fs::PermissionsExt::from_mode(0o755))
            .unwrap();
        let line = format!(
            "file_appeal 7ad26ccd-1111-4222-8333-444455556666 --editor={}",
            script.display()
        );
        let Some(Command::Tool(ToolCall::FileAppeal(input))) = parse_line(&line).unwrap().command
        else {
            panic!("expected file_appeal");
        };
        let _ = std::fs::remove_file(&script);
        assert_eq!(input.appeal_statement, "# Heading\n\nbody");
    }

    /// The globals reach every command, the shell's lines included
    #[test]
    fn globals_parse_anywhere() {
        let cli = parse_line("get_feed --operator --server http://localhost:8099").unwrap();
        assert!(cli.operator);
        assert_eq!(cli.server.as_deref(), Some("http://localhost:8099"));
    }
}
