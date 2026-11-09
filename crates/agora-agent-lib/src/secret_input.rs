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
//! The secret is held in [`Zeroizing`]: from stdin, from the first byte
//! read (a fixed buffer, then a `String` allocated once at its final size,
//! so no reallocation leaves a copy behind); from the prompt, from the
//! moment `rpassword` hands it over.
//!
//! After a prompted read the clipboard is cleared if it still holds the
//! secret, since a code is usually pasted (feature `clipboard`, on in
//! `agora-operator`; see [`clear_clipboard`]).

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
/// line, a control character or more than [`SECRET_MAX_BYTES`] is refused.
///
/// What is and isn't wiped: stdin is read unbuffered into a wiped buffer.
/// The prompt's line comes from `rpassword`, whose `SafeString` grows by
/// reallocation, so a secret longer than 8 bytes can leave a freed, unwiped
/// copy of its first part on the heap before it reaches us
pub fn read_secret(prompt: &str, source: SecretSource) -> Result<Zeroizing<String>> {
    match source {
        SecretSource::Prompt => {
            let secret = prompt_no_echo(prompt)?;
            clear_clipboard(&secret);
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
            // Not through `stdin.lock()`: std's stdin is a global 8 KiB
            // `BufReader` that would keep the secret, unwiped, for the life
            // of the process. An unbuffered duplicate of fd 0 reads straight
            // into our wiped buffer
            let fd = unbuffered_stdin(&stdin).context("duplicating stdin")?;
            read_secret_from(fd, false).with_context(|| format!("{prompt}, on stdin"))
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

/// Stdin as a `File` of its own (a `dup` of fd 0), read without std's
/// buffer
#[cfg(unix)]
fn unbuffered_stdin(stdin: &std::io::Stdin) -> std::io::Result<std::fs::File> {
    use std::os::fd::AsFd;
    Ok(std::fs::File::from(stdin.as_fd().try_clone_to_owned()?))
}

#[cfg(windows)]
fn unbuffered_stdin(stdin: &std::io::Stdin) -> std::io::Result<std::fs::File> {
    use std::os::windows::io::AsHandle;
    Ok(std::fs::File::from(stdin.as_handle().try_clone_to_owned()?))
}

/// `prompt` on the terminal (`/dev/tty` on unix, so stdin stays free),
/// and a line read back with echo off, checked as [`read_secret_from`]
/// checks stdin's
fn prompt_no_echo(prompt: &str) -> Result<Zeroizing<String>> {
    let line = Zeroizing::new(
        rpassword::prompt_password(format!("{prompt}: "))
            .context("no terminal to ask on; pipe it in on stdin instead")?,
    );
    read_secret_from(line.as_bytes(), false).with_context(|| prompt.to_owned())
}

/// Clear the clipboard after a prompted read if it still holds `secret`,
/// which was most likely pasted. Anything copied since is left alone.
///
/// Never fails: with no clipboard to reach (a headless box, an SSH session,
/// where the clipboard is the client's) it does nothing, and says so only
/// in a debug log.
///
/// Limits: reading the clipboard to compare leaves unwiped copies of
/// whatever it held inside arboard and wl-clipboard-rs (only our own copy
/// is wiped), and a clipboard manager's history keeps the code regardless
#[cfg(feature = "clipboard")]
pub fn clear_clipboard(secret: &str) {
    let mut clipboard = match arboard::Clipboard::new() {
        Ok(c) => c,
        Err(e) => {
            tracing::debug!(error = %e, "no clipboard to clear");
            return;
        }
    };
    let held = match clipboard.get_text() {
        Ok(text) => Zeroizing::new(text),
        Err(e) => {
            tracing::debug!(error = %e, "clipboard not read; not cleared");
            return;
        }
    };
    if held.trim() != secret {
        return;
    }
    if let Err(e) = clipboard.clear() {
        tracing::debug!(error = %e, "clipboard not cleared");
    }
}

/// Without the `clipboard` feature, nothing to clear with
#[cfg(not(feature = "clipboard"))]
pub fn clear_clipboard(_secret: &str) {}

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
