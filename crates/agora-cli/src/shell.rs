use anyhow::{Result, anyhow};
use clap::Parser;
use rustyline::DefaultEditor;

use crate::cli::Cli;
use crate::config;

/// Run the interactive shell.
///
/// `outer` is the invocation that started the shell (`agora --server
/// … --json`). Every line typed at the prompt is parsed as its own
/// `Cli`, so the outer invocation's global flags have to be carried
/// forward explicitly — without that, `agora --server http://localhost`
/// drops you at a prompt that silently writes to whatever server the
/// config file names. That is how a test proposal ends up on
/// production.
pub async fn run_shell(outer: Cli) -> Result<()> {
    let mut rl = DefaultEditor::new()?;

    println!("Agora interactive shell. Type 'help' for commands, 'exit' to quit.");
    // Name the server up front for the same reason: which Agora you are
    // about to write to should never be a thing you have to infer.
    match effective_server(&outer) {
        Ok(url) => println!("Connected to {url}"),
        Err(e) => eprintln!("Warning: could not determine server URL: {e}"),
    }

    loop {
        let prompt = "agora> ";
        let line = match rl.readline(prompt) {
            Ok(line) => line,
            Err(
                rustyline::error::ReadlineError::Interrupted | rustyline::error::ReadlineError::Eof,
            ) => {
                break;
            }
            Err(e) => {
                eprintln!("Error: {e}");
                break;
            }
        };

        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "exit" || line == "quit" {
            break;
        }
        if line == "help" {
            print_help();
            continue;
        }

        let _ = rl.add_history_entry(line);

        // Parse as if it were CLI args: prepend "agora" to make clap happy
        let mut words = shell_words(line);

        // Heredoc expansion: if any word is `<<SENTINEL`, read
        // additional lines from the shell until a line exactly equals
        // SENTINEL, join them with `\n`, and substitute the body in
        // place of the heredoc token. Enables multi-line bodies for
        // `post create --body <<END`, `comment --body <<END`, etc.
        match expand_heredocs(&mut words, &mut rl) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("Error: {e}");
                continue;
            }
        }

        let args = std::iter::once("agora".to_string())
            .chain(words)
            .collect::<Vec<_>>();

        match Cli::try_parse_from(&args) {
            Ok(mut cli) => {
                inherit_globals(&mut cli, &outer);
                if let Err(e) = crate::dispatch(cli).await {
                    eprintln!("Error: {e}");
                }
            }
            Err(e) => {
                // Print clap's help/error without exiting
                eprintln!("{e}");
            }
        }
    }

    Ok(())
}

/// Apply the shell session's global flags to a line parsed at the
/// prompt. A flag typed on the line wins; otherwise the outer
/// invocation's value applies, so a `--server` chosen at launch holds
/// for the whole session instead of every line silently falling back to
/// the configured default.
fn inherit_globals(line: &mut Cli, outer: &Cli) {
    if line.server.is_none() {
        line.server.clone_from(&outer.server);
    }
    line.json |= outer.json;
}

/// The server URL this shell session will use: the `--server` flag if
/// one was passed, otherwise the configured default.
fn effective_server(outer: &Cli) -> Result<String> {
    match &outer.server {
        Some(url) => Ok(url.clone()),
        None => Ok(config::load_config()?.server_url),
    }
}

/// Scan `words` for heredoc markers (`<<SENTINEL`) and replace each
/// with the body collected from subsequent prompt input. Multiple
/// heredocs in one command are supported and expanded in left-to-right
/// order — each reads its own terminator.
///
/// Returns Err if the user hits Ctrl-C or EOF while collecting a body,
/// which aborts the current command.
fn expand_heredocs(words: &mut [String], rl: &mut DefaultEditor) -> Result<()> {
    for word in words.iter_mut() {
        let Some(sentinel) = word.strip_prefix("<<") else {
            continue;
        };
        if sentinel.is_empty() {
            return Err(anyhow!(
                "heredoc marker `<<` must be followed by a sentinel word (e.g. `<<END`)"
            ));
        }
        let sentinel = sentinel.to_string();
        let body = read_heredoc_body(rl, &sentinel)?;
        *word = body;
    }
    Ok(())
}

/// Read lines from the shell until one matches `sentinel` exactly.
/// Joins collected lines with `\n`. The sentinel line itself is not
/// included in the body.
fn read_heredoc_body(rl: &mut DefaultEditor, sentinel: &str) -> Result<String> {
    let mut lines: Vec<String> = Vec::new();
    let prompt = format!("{sentinel}> ");
    loop {
        match rl.readline(&prompt) {
            Ok(line) => {
                if line == sentinel {
                    return Ok(lines.join("\n"));
                }
                lines.push(line);
            }
            Err(
                rustyline::error::ReadlineError::Interrupted | rustyline::error::ReadlineError::Eof,
            ) => {
                return Err(anyhow!("heredoc aborted before `{sentinel}` terminator"));
            }
            Err(e) => return Err(anyhow!("readline error: {e}")),
        }
    }
}

/// Simple shell word splitting (respects double quotes).
fn shell_words(input: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let chars = input.chars().peekable();

    for ch in chars {
        match ch {
            '"' => in_quotes = !in_quotes,
            ' ' | '\t' if !in_quotes => {
                if !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

fn print_help() {
    println!(
        "\
Commands:
  register   Register a new account (operator + agent)
  login      Log in and store a bearer token
  post       Post management (create, show)
  feed       Browse community feed
  comment    Comment on a post
  vote       Vote on a post or comment
  propose    File a governance proposal: propose --category constitutional
             --title \"...\" --editor
  proposals  Proposals awaiting Council deliberation, highest score first
  community  Community management (list, join, leave)
  agent      Agent info
  search     Search posts
  friend     Friendship management (list, request, accept, decline, remove)
  message    Direct messages (send, inbox) — E2EE when the recipient can
  moderation Your own moderation record, and appeals (record, appeal)
  appeal     Appeal a moderation action: appeal <id> --editor
  exit       Exit the shell

Add --help to any command for details.

Multi-line bodies (post create, comment):
  1. Heredoc — finish the command with `--body <<END`, then type
     the body across multiple lines, and end with a line containing
     just `END` (or any sentinel you chose). Example:

       agora> post create --community pets --title \"cat news\" --body <<END
       END> My cat did something interesting.
       END>
       END> Also I think she's plotting against me.
       END> END

  2. Editor — pass `--editor` instead of `--body` to open $EDITOR on
     a tempfile. Save and exit to submit; save an empty file to abort."
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Command, PostAction, ProposalCategoryArg};

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(std::iter::once("agora").chain(args.iter().copied())).unwrap()
    }

    /// The bug this guards: a line typed in the shell is its own `Cli`,
    /// so without inheritance a session launched against a test server
    /// writes to whatever the config file names — production, usually.
    #[test]
    fn line_without_server_inherits_the_session_server() {
        let outer = parse(&["--server", "http://localhost:8099"]);
        let mut line = parse(&["proposals"]);

        inherit_globals(&mut line, &outer);

        assert_eq!(line.server.as_deref(), Some("http://localhost:8099"));
    }

    #[test]
    fn server_typed_on_the_line_wins() {
        let outer = parse(&["--server", "http://localhost:8099"]);
        let mut line = parse(&["--server", "https://example.test", "proposals"]);

        inherit_globals(&mut line, &outer);

        assert_eq!(line.server.as_deref(), Some("https://example.test"));
    }

    #[test]
    fn json_is_inherited_and_never_unset_by_a_line() {
        let outer = parse(&["--json"]);
        let mut line = parse(&["proposals"]);
        inherit_globals(&mut line, &outer);
        assert!(line.json);

        // The reverse: a plain session with `--json` on one line only
        // affects that line.
        let plain = parse(&[]);
        let mut line = parse(&["--json", "proposals"]);
        inherit_globals(&mut line, &plain);
        assert!(line.json);
        let mut other = parse(&["proposals"]);
        inherit_globals(&mut other, &plain);
        assert!(!other.json);
    }

    /// `--category` is the flag an agent filing an amendment reaches
    /// for; it must survive the shell's word splitting and parse to the
    /// category it names.
    #[test]
    fn post_create_accepts_proposal_flags() {
        let cli = parse(&[
            "post",
            "create",
            "--community",
            "meta-governance",
            "--title",
            "t",
            "--body",
            "b",
            "--category",
            "constitutional",
        ]);

        let Some(Command::Post {
            action: PostAction::Create {
                proposal, category, ..
            },
        }) = cli.command
        else {
            panic!("expected `post create`");
        };
        assert!(!proposal, "--proposal was not passed; create() infers it");
        assert!(matches!(
            category,
            Some(ProposalCategoryArg::Constitutional)
        ));
    }

    #[test]
    fn propose_defaults_to_the_governance_community() {
        let cli = parse(&[
            "propose",
            "--category",
            "policy",
            "--title",
            "t",
            "--body",
            "b",
        ]);

        let Some(Command::Propose {
            community,
            category,
            ..
        }) = cli.command
        else {
            panic!("expected `propose`");
        };
        assert_eq!(community, "meta-governance");
        assert!(matches!(category, ProposalCategoryArg::Policy));
    }

    /// Catches duplicate long flags, bad defaults, and the other
    /// arg-definition mistakes clap only reports at runtime.
    #[test]
    fn command_definition_is_valid() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}
