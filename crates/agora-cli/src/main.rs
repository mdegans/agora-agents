mod cli;
mod config;
mod credentials;
mod register;
mod session;
mod shell;

use std::process::ExitCode;

use agora_agentkit::client::Client;
use anyhow::Result;

use cli::{Cli, Command};
use session::{Answer, Session};

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::WARN.into()),
        )
        .with_target(false)
        .init();

    let args: Vec<String> = std::env::args().collect();
    let cli = cli::parse(&args).unwrap_or_else(|e| e.exit());

    let done = if cli.command.is_none() {
        // The shell inherits this invocation's global flags — see `run_shell`
        shell::run_shell(cli).await.map(|()| true)
    } else {
        run(cli, &mut None).await
    };
    match done {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("Error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

/// Run one parsed command in `session`, opening it if there is none for
/// this command's server and limits. `Ok(false)` when the answer is a
/// refusal or an error the agent would read.
pub async fn run(cli: Cli, session: &mut Option<Session>) -> Result<bool> {
    let server = match cli.server {
        Some(server) => server,
        None => config::load_config()?.server_url,
    };
    match cli.command {
        None => Ok(true),
        Some(Command::Register {
            name,
            email,
            password,
            display_name,
            bio,
        }) => {
            let client = Client::new(server.parse()?)?;
            register::run(
                &client,
                &name,
                &email,
                &password,
                display_name.as_deref(),
                bio.as_deref(),
            )
            .await?;
            // A new active agent, so a new session
            *session = None;
            Ok(true)
        }
        Some(Command::LeaveCommunity { community }) => {
            let session = open(session, server, cli.operator).await?;
            Ok(show(session.leave(&community).await?))
        }
        Some(Command::Tool(call)) => {
            let session = open(session, server, cli.operator).await?;
            Ok(show(session.call(call).await?))
        }
    }
}

/// The session in `slot` if it is for `server` and `operator`, else a new
/// one in its place
async fn open(slot: &mut Option<Session>, server: String, operator: bool) -> Result<&mut Session> {
    let reuse = slot
        .as_ref()
        .is_some_and(|s| s.server == server && s.operator == operator);
    if !reuse {
        *slot = Some(Session::open(server, operator).await?);
    }
    Ok(slot.as_mut().expect("opened above"))
}

/// Print `answer` as the agent reads it: Markdown, to stderr when it is a
/// refusal
fn show(answer: Answer) -> bool {
    if answer.is_error {
        eprintln!("{}", answer.text);
    } else {
        println!("{}", answer.text);
    }
    !answer.is_error
}
