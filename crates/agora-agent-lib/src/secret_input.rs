//! Secrets a CLI takes from its operator: the emailed send-now code now,
//! a YubiKey's PIN and PUK next.
//!
//! A secret is read from one of two places, and never from anywhere else:
//!
//! - [`SecretSource::Prompt`]: the controlling terminal (`/dev/tty`) with
//!   echo off, so it works even when stdin carries something else (a
//!   message body). The normal way.
//! - [`SecretSource::Stdin`]: one line on stdin, for scripts. Refused when
//!   stdin is a terminal, where it would echo.
//!
//! **Never argv, never the environment.** A command line reaches the
//! shell's history and every user's `ps`, and an environment variable is
//! inherited by every child process and readable in `/proc/<pid>/environ`.
//! This is also why the CLIs drive a YubiKey through a library rather than
//! shelling out to `ykman`: `ykman piv access change-pin --pin … --new-pin …`
//! (and its `--management-key`, `--puk`) take the secrets on argv.
//!
//! The secret is held in [`Zeroizing`] from the first byte read: a fixed
//! buffer, then a `String` allocated once at its final size, so no
//! reallocation leaves a copy behind.
//!
//! After a prompted read the clipboard is cleared, since a code is usually
//! pasted. That needs a clipboard crate, which is behind the `clipboard`
//! feature (off by default) and not written yet; see [`clear_clipboard`].

use std::io::{IsTerminal, Read};

use anyhow::{Context, Result, bail};
use zeroize::Zeroizing;

/// The longest secret read, in bytes. Codes, PINs and PUKs are far shorter;
/// anything longer is refused rather than cut
pub const SECRET_MAX_BYTES: usize = 256;

/// Where [`read_secret`] reads from
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecretSource {
    /// The terminal, with echo off
    Prompt,
    /// One line on stdin, for scripts
    Stdin,
}

/// Read one secret, `prompt` naming it (`"The code from the email"`).
///
/// Surrounding spaces and tabs are trimmed; an empty secret, more than one
/// line, a control character or more than [`SECRET_MAX_BYTES`] is refused
pub fn read_secret(prompt: &str, source: SecretSource) -> Result<Zeroizing<String>> {
    match source {
        SecretSource::Prompt => {
            let secret = prompt_no_echo(prompt)?;
            clear_clipboard()?;
            Ok(secret)
        }
        SecretSource::Stdin => {
            let stdin = std::io::stdin();
            if stdin.is_terminal() {
                bail!(
                    "{prompt}: stdin is a terminal, where it would echo; use the prompt \
                     instead, or pipe it in"
                );
            }
            read_secret_from(stdin.lock(), false).with_context(|| format!("{prompt}, on stdin"))
        }
    }
}

/// Read a secret from `r`: up to the first newline when `stop_at_newline`
/// (a terminal), else to the end, which must hold one line (stdin)
pub fn read_secret_from(mut r: impl Read, stop_at_newline: bool) -> Result<Zeroizing<String>> {
    // One more than the limit, and room for "\r\n", to tell "too long" apart
    let mut buf = Zeroizing::new([0u8; SECRET_MAX_BYTES + 3]);
    let mut len = 0;
    loop {
        if len == buf.len() {
            bail!("longer than {SECRET_MAX_BYTES} bytes");
        }
        match r.read(&mut buf[len..len + 1]) {
            Ok(0) => break,
            Ok(_) => {
                len += 1;
                if stop_at_newline && buf[len - 1] == b'\n' {
                    break;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e.into()),
        }
    }
    let mut line = &buf[..len];
    if let Some(rest) = line.strip_suffix(b"\n") {
        line = rest.strip_suffix(b"\r").unwrap_or(rest);
    }
    if line.contains(&b'\n') {
        bail!("more than one line; give exactly one");
    }
    let line = line.trim_ascii();
    if line.is_empty() {
        bail!("nothing given");
    }
    if line.iter().any(|b| b.is_ascii_control()) {
        bail!("a control character in it");
    }
    if line.len() > SECRET_MAX_BYTES {
        bail!("longer than {SECRET_MAX_BYTES} bytes");
    }
    let text = std::str::from_utf8(line).map_err(|_| anyhow::anyhow!("not UTF-8"))?;
    let mut secret = Zeroizing::new(String::with_capacity(text.len()));
    secret.push_str(text);
    Ok(secret)
}

/// `prompt` on the terminal, and a line read back with echo off
#[cfg(unix)]
fn prompt_no_echo(prompt: &str) -> Result<Zeroizing<String>> {
    use nix::sys::termios::{LocalFlags, SetArg, tcgetattr, tcsetattr};
    use std::io::Write;

    let mut tty = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/tty")
        .context("no terminal to ask on; pipe it in on stdin instead")?;
    let original = tcgetattr(&tty).context("reading the terminal's settings")?;

    /// Echo back on, however the read ends
    struct Restore<'a>(&'a std::fs::File, nix::sys::termios::Termios);
    impl Drop for Restore<'_> {
        fn drop(&mut self) {
            let _ = tcsetattr(self.0, SetArg::TCSAFLUSH, &self.1);
        }
    }

    let mut quiet = original.clone();
    quiet.local_flags.remove(LocalFlags::ECHO);
    // The newline still shows, so the next output starts on its own line
    quiet.local_flags.insert(LocalFlags::ECHONL);
    write!(tty, "{prompt}: ")?;
    tty.flush()?;
    let restore = Restore(&tty, original);
    tcsetattr(&tty, SetArg::TCSAFLUSH, &quiet).context("turning the terminal's echo off")?;
    let secret = read_secret_from(&tty, true);
    drop(restore);
    secret.with_context(|| prompt.to_owned())
}

#[cfg(not(unix))]
fn prompt_no_echo(prompt: &str) -> Result<Zeroizing<String>> {
    bail!("{prompt}: no echo-off prompt on this platform; pipe it in on stdin instead")
}

/// Clear the clipboard after a prompted read: a pasted secret stays there
/// otherwise.
///
/// Not written yet: it needs a clipboard crate, and dependencies need the
/// Steward's approval. Without the feature, nothing is cleared
#[cfg(feature = "clipboard")]
pub fn clear_clipboard() -> Result<()> {
    // Intended, with `arboard`: clear only when the clipboard still holds
    // the secret (compare `get_text()` with it, then `clear()`), so a copy
    // made since is not lost. Over SSH the clipboard is the client's and
    // cannot be reached this way.
    todo!("clear the clipboard once the clipboard crate is approved")
}

/// Without the `clipboard` feature, nothing to clear with
#[cfg(not(feature = "clipboard"))]
pub fn clear_clipboard() -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stdin(text: &[u8]) -> Result<Zeroizing<String>> {
        read_secret_from(text, false)
    }

    #[test]
    fn one_line_is_read_with_or_without_its_newline() {
        assert_eq!(stdin(b"123456\n").unwrap().as_str(), "123456");
        assert_eq!(stdin(b"123456").unwrap().as_str(), "123456");
        assert_eq!(stdin(b"123456\r\n").unwrap().as_str(), "123456");
        assert_eq!(stdin(b"  123456\t\n").unwrap().as_str(), "123456");
    }

    #[test]
    fn anything_but_one_line_is_refused() {
        for bad in [
            &b""[..],
            b"\n",
            b"   \n",
            b"123456\n\n",
            b"123456\n654321\n",
            b"123\n456",
            b"12\x0034",
            b"12\x1b[A34",
            b"\xff\xfe\n",
        ] {
            assert!(stdin(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn too_long_is_refused_not_cut() {
        let max = vec![b'7'; SECRET_MAX_BYTES];
        assert_eq!(stdin(&max).unwrap().len(), SECRET_MAX_BYTES);
        let long = vec![b'7'; SECRET_MAX_BYTES + 1];
        assert!(stdin(&long).is_err());
        let longer = vec![b'7'; 10 * SECRET_MAX_BYTES];
        assert!(stdin(&longer).is_err());
    }

    /// A terminal read stops at the newline and leaves the rest unread
    #[test]
    fn a_terminal_read_stops_at_the_newline() {
        let mut input = &b"123456\nnext\n"[..];
        assert_eq!(
            read_secret_from(&mut input, true).unwrap().as_str(),
            "123456"
        );
        assert_eq!(input, b"next\n");
    }

    /// The child half of [`read_secret_reads_real_stdin`]: run only when
    /// that test starts this binary again with its stdin piped
    #[test]
    fn stdin_child() {
        if std::env::var_os("AGORA_SECRET_INPUT_CHILD").is_none() {
            return;
        }
        match read_secret("the code", SecretSource::Stdin) {
            Ok(secret) => println!("CHILD-OK {}", secret.len()),
            Err(e) => println!("CHILD-ERR {e:#}"),
        }
    }

    /// `read_secret` on the process's own stdin, piped as a script would
    #[test]
    fn read_secret_reads_real_stdin() {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let run = |input: &[u8]| {
            let mut child = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "secret_input::tests::stdin_child", "--nocapture"])
                .env("AGORA_SECRET_INPUT_CHILD", "1")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .unwrap();
            child.stdin.take().unwrap().write_all(input).unwrap();
            let out = child.wait_with_output().unwrap();
            String::from_utf8(out.stdout).unwrap()
        };
        let ok = run(b"123456\n");
        assert!(ok.contains("CHILD-OK 6"), "{ok}");
        let two = run(b"123456\n654321\n");
        assert!(
            two.contains("CHILD-ERR the code, on stdin: more than one line"),
            "{two}"
        );
        let none = run(b"");
        assert!(
            none.contains("CHILD-ERR the code, on stdin: nothing given"),
            "{none}"
        );
    }
}
