//! Graceful stop by signal (agora-agents#293; Steward, 2026-10-26 and
//! 2026-10-27: "SIGINT exits after the current session, SIGTERM after the
//! current turn, and a second signal exits at once").
//!
//! | Signal | Effect |
//! |---|---|
//! | `SIGINT` | Drain: no new session starts; each session in flight runs to its end (teardown, saves, ledgers), then the runner exits [`STOPPED_EXIT`]. No deadline. |
//! | `SIGTERM` | End the turn: each session in flight finishes the turn in flight (its model call returns, its tool results are seated), then closes the way a spent round budget closes it (the agent is told why, and its memory turn and the rest of the tail follow) and is saved, complete. A session that hadn't had its first turn has it first. No new session starts. Exits [`STOPPED_EXIT`], within the deadline (`--stop-deadline-secs`, [`DEFAULT_DEADLINE`]); past it, at once with [`DEADLINE_EXIT`]. |
//! | `SIGTERM` while draining | The drain becomes an end-turn, with its deadline. |
//! | `SIGINT` while draining, or either after `SIGTERM` | Exit at once, `128 + signal`: today's kill, which [`crate::interrupted`] recovers from. |
//!
//! `SIGTERM` after `SIGINT` raises the stop rather than cutting it, so a
//! shutdown or `systemctl stop` arriving during a drain still lets the turns
//! finish. A repeat of the same signal, or anything after `SIGTERM`, is
//! someone meaning "now".
//!
//! The reactor reads the level at session and turn boundaries
//! ([`agora_agentkit::reactor::Stop`]); this module only raises it and keeps
//! the deadline. A session cut by the immediate exit has no teardown and no
//! save, as before: its journal becomes the interrupted-session notice next
//! session.
//!
//! The handlers are installed just before the run starts. Before then the
//! default disposition stands (an immediate kill), and no session is open.

use std::time::Duration;

use agora_agentkit::reactor::{Stop, StopLevel};
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::oneshot;

/// How long a `SIGTERM` stop may take before the runner exits at once. A
/// turn on the local models takes 19 s at the median and 3 min at p90, the
/// tail after a stop up to five more (memory, soul or evolution, offers,
/// survey), and the slowest turns of October 2026 took 20 to 50 min. Half
/// an hour lets nearly every stop finish and bounds one stuck on a backend
/// (blallama "Session is busy" for an hour, 2026-10-25). The units'
/// `TimeoutStopSec` sits above it.
pub const DEFAULT_DEADLINE: Duration = Duration::from_secs(30 * 60);

/// The exit status of a run a signal stopped gracefully, either way. Not 0:
/// under `Restart=always` a 0 restarts the unit after its cooldown, so a
/// drain sent with `systemctl kill` would quietly become a restart. The
/// units list it in `RestartPreventExitStatus=` and `SuccessExitStatus=`
/// (review of agora-agents#293).
pub const STOPPED_EXIT: i32 = 3;

/// The exit status when the deadline passes, cutting the sessions in
/// flight. Not a success to systemd, so the failure mail says so.
pub const DEADLINE_EXIT: i32 = 124;

/// How long [`Exit`]'s receiver gets to exit (flushing the log) before the
/// signal task exits the process itself
const EXIT_GRACE: Duration = Duration::from_secs(10);

/// A signal the runner answers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    Int,
    Term,
}

impl Signal {
    fn name(self) -> &'static str {
        match self {
            Signal::Int => "SIGINT",
            Signal::Term => "SIGTERM",
        }
    }

    /// The shell's status for a process this signal killed
    fn exit_code(self) -> i32 {
        128 + match self {
            Signal::Int => 2,
            Signal::Term => 15,
        }
    }
}

/// What a signal does, given how far the stop has gone
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    Raise(StopLevel),
    Exit(i32),
}

fn on_signal(level: StopLevel, signal: Signal) -> Action {
    match (level, signal) {
        (StopLevel::Run, Signal::Int) => Action::Raise(StopLevel::Drain),
        (StopLevel::Run | StopLevel::Drain, Signal::Term) => Action::Raise(StopLevel::EndTurn),
        (StopLevel::Drain | StopLevel::EndTurn, _) => Action::Exit(signal.exit_code()),
    }
}

/// Resolves with the status to exit with at once: a second signal, or the
/// deadline. Never resolves for a stop that finishes in time.
pub struct Exit(oneshot::Receiver<i32>);

impl Exit {
    pub async fn wait(self) -> i32 {
        match self.0.await {
            Ok(code) => code,
            // The signal task is gone (it never is, but then nothing will
            // ever ask for an exit).
            Err(_) => std::future::pending().await,
        }
    }
}

/// Answer `SIGINT` and `SIGTERM` from now on by raising `stop` (see the
/// [module docs](self)), with `deadline` for a `SIGTERM` stop.
pub fn install(stop: Stop, deadline: Duration) -> std::io::Result<Exit> {
    let mut int = signal(SignalKind::interrupt())?;
    let mut term = signal(SignalKind::terminate())?;
    let (tx, rx) = oneshot::channel();
    tokio::spawn(async move {
        let mut cutoff: Option<tokio::time::Instant> = None;
        let code = loop {
            let received = tokio::select! {
                _ = int.recv() => Signal::Int,
                _ = term.recv() => Signal::Term,
                () = async { tokio::time::sleep_until(cutoff.expect("guarded")).await },
                    if cutoff.is_some() =>
                {
                    tracing::error!(
                        event_type = "stop_deadline_passed",
                        deadline_secs = deadline.as_secs(),
                        exit_code = DEADLINE_EXIT,
                        "the stop took too long; exiting now, cutting the sessions in flight"
                    );
                    break DEADLINE_EXIT;
                }
            };
            match on_signal(stop.level(), received) {
                Action::Raise(level) => {
                    stop.request(level);
                    if level == StopLevel::EndTurn && cutoff.is_none() {
                        cutoff = Some(tokio::time::Instant::now() + deadline);
                    }
                    tracing::warn!(
                        event_type = "stop_requested",
                        signal = received.name(),
                        level = %level,
                        deadline_secs = cutoff.map(|_| deadline.as_secs()),
                        "stopping: {}",
                        match level {
                            StopLevel::Drain => "no new session starts; those in flight finish",
                            _ => "no new session starts; those in flight close after this turn",
                        }
                    );
                }
                Action::Exit(code) => {
                    tracing::warn!(
                        event_type = "stop_immediate",
                        signal = received.name(),
                        exit_code = code,
                        "a second signal: exiting now, cutting the sessions in flight"
                    );
                    break code;
                }
            }
        };
        let _ = tx.send(code);
        // Whoever holds `Exit` exits; should it not get to run, this does.
        tokio::time::sleep(EXIT_GRACE).await;
        std::process::exit(code);
    });
    Ok(Exit(rx))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signals_escalate_and_a_second_exits() {
        use Action::*;
        use StopLevel::*;
        assert_eq!(on_signal(Run, Signal::Int), Raise(Drain));
        assert_eq!(on_signal(Run, Signal::Term), Raise(EndTurn));
        // A shutdown during a drain lets the turns finish
        assert_eq!(on_signal(Drain, Signal::Term), Raise(EndTurn));
        assert_eq!(on_signal(Drain, Signal::Int), Exit(130));
        assert_eq!(on_signal(EndTurn, Signal::Int), Exit(130));
        assert_eq!(on_signal(EndTurn, Signal::Term), Exit(143));
        assert_ne!(DEADLINE_EXIT, Signal::Term.exit_code(), "a cut is mailed");
    }
}
