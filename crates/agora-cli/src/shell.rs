use anyhow::Result;
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
        let args = std::iter::once("agora".to_string())
            .chain(shell_words(line))
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

/// Simple shell word splitting (respects double quotes).
fn shell_words(input: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
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
  exit       Exit the shell

Add --help to any command for details."
    );
}
