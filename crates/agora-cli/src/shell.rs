use anyhow::{Result, anyhow};
use clap::Parser;
use rustyline::DefaultEditor;

use crate::cli::Cli;

/// Run the interactive shell.
pub async fn run_shell() -> Result<()> {
    let mut rl = DefaultEditor::new()?;

    println!("Agora interactive shell. Type 'help' for commands, 'exit' to quit.");

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
            Ok(cli) => {
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
