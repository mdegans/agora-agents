//! Resolve the body argument for commands that take long-form text
//! (`post create`, `comment`).
//!
//! Three ways to supply a body:
//!   1. `--body "literal text"` on the command line
//!   2. `--editor` to compose in `$EDITOR` (or `--editor vim` to
//!      override the editor for a single invocation)
//!   3. A shell heredoc (`--body <<END` ... `END`) — handled earlier
//!      in the interactive shell, so by the time we reach here the
//!      heredoc body has already been collapsed into `--body "..."`.

use anyhow::{Context, Result, anyhow};
use std::io::{Read, Write};
use std::process::Command;

/// `--editor` state from clap:
/// - `None`              → flag absent
/// - `Some(None)`        → `--editor` with no value, use `$EDITOR`
/// - `Some(Some("vim"))` → `--editor vim`, override for this call
pub type EditorFlag = Option<Option<String>>;

/// Resolve the final body string from the two mutually-exclusive
/// clap flags `--body` and `--editor`.
///
/// `label` is used in error messages (e.g., "post body", "comment body")
/// to make the failure mode obvious to the user.
pub fn resolve(label: &str, body: Option<String>, editor: EditorFlag) -> Result<String> {
    if let Some(override_cmd) = editor {
        return compose_in_editor(label, override_cmd.as_deref());
    }
    body.ok_or_else(|| {
        anyhow!("missing {label}: pass `--body \"...\"` or `--editor`, or use a heredoc in the interactive shell")
    })
}

/// Open `$EDITOR` on a fresh tempfile, wait for it to close, and
/// return the resulting file contents. `override_cmd` replaces
/// `$EDITOR` / `$VISUAL` for this one invocation (e.g. `--editor vim`).
/// Errors if the file is empty after editing (user likely aborted).
fn compose_in_editor(label: &str, override_cmd: Option<&str>) -> Result<String> {
    let editor_cmd = match override_cmd {
        Some(cmd) if !cmd.is_empty() => cmd.to_string(),
        _ => std::env::var("EDITOR")
            .or_else(|_| std::env::var("VISUAL"))
            .unwrap_or_else(|_| "vi".to_string()),
    };

    // Tempfile path: include PID + 64 random bits to avoid collisions
    // across concurrent CLI invocations. No crate needed — std + rand.
    let path = std::env::temp_dir().join(format!(
        "agora-{}-{}-{:016x}.md",
        label.replace(' ', "-"),
        std::process::id(),
        rand::random::<u64>()
    ));

    // Touch the file so the editor has something to open. An empty
    // file is fine — the user fills it in.
    {
        let mut f = std::fs::File::create(&path)
            .with_context(|| format!("failed to create tempfile at {}", path.display()))?;
        // Write a short instructional header for convenience. The
        // user is expected to delete it; if they don't we strip it
        // on read.
        writeln!(
            f,
            "# Write your {label} here.\n# Lines starting with '#' at the top of the file are stripped.\n"
        )?;
    }

    // Spawn the editor. Split `$EDITOR` on whitespace so that values
    // like `code --wait` or `emacs -nw` work. Inherit stdio so the
    // editor has direct access to the terminal.
    let mut parts = editor_cmd.split_whitespace();
    let program = parts.next().ok_or_else(|| anyhow!("EDITOR is empty"))?;
    let status = Command::new(program)
        .args(parts)
        .arg(&path)
        .status()
        .with_context(|| format!("failed to spawn editor `{editor_cmd}`"))?;

    // Read the file back, strip leading comment lines, trim trailing
    // whitespace. Delete the tempfile either way.
    let mut contents = String::new();
    std::fs::File::open(&path)
        .with_context(|| format!("failed to open {} after edit", path.display()))?
        .read_to_string(&mut contents)?;
    let _ = std::fs::remove_file(&path);

    body_from_edit(label, status.success(), &contents)
}

/// Decide the body from what the editor session left behind.
///
/// The exit status is deliberately *not* the abort signal: vim can exit
/// non-zero after a successful `:wq` (e.g. a failed command earlier in
/// the session), which used to discard a saved body. Saved content is
/// the signal of intent — quit-without-write (`:q!`, `:cq`) leaves the
/// header-only tempfile, which strips to empty and aborts. A non-zero
/// exit with content is downgraded to a warning.
fn body_from_edit(label: &str, exit_ok: bool, contents: &str) -> Result<String> {
    let cleaned = strip_leading_comments(contents);
    let trimmed = cleaned.trim_end();
    if trimmed.is_empty() {
        return Err(anyhow!(
            "{label} is empty after editing — aborted (save a non-empty file to submit)"
        ));
    }
    if !exit_ok {
        eprintln!("warning: editor exited non-zero; using the saved {label} anyway");
    }
    Ok(trimmed.to_string())
}

/// Strip leading `#`-prefixed comment lines from a tempfile. Stops at
/// the first non-comment line and returns the rest verbatim (including
/// internal `#` lines — only the initial header block is removed).
fn strip_leading_comments(s: &str) -> &str {
    let mut idx = 0;
    for line in s.split_inclusive('\n') {
        let stripped = line.trim_start();
        if stripped.starts_with('#') || stripped.is_empty() {
            idx += line.len();
        } else {
            break;
        }
    }
    &s[idx..]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_leading_comments_removes_header_block() {
        let input = "# header one\n# header two\n\nreal content\n# not a header\nmore content\n";
        assert_eq!(
            strip_leading_comments(input),
            "real content\n# not a header\nmore content\n"
        );
    }

    #[test]
    fn strip_leading_comments_handles_no_header() {
        let input = "real content\nmore content\n";
        assert_eq!(strip_leading_comments(input), input);
    }

    #[test]
    fn strip_leading_comments_all_headers_returns_empty() {
        let input = "# only\n# headers\n";
        assert_eq!(strip_leading_comments(input), "");
    }

    #[test]
    fn nonzero_exit_with_saved_content_still_submits() {
        let body = body_from_edit("test body", false, "# header\n\nsaved despite vim goof\n");
        assert_eq!(body.unwrap(), "saved despite vim goof");
    }

    #[test]
    fn nonzero_exit_with_unwritten_file_aborts() {
        // `:q!` / `:cq` leave the instructional header untouched.
        let err = body_from_edit("test body", false, "# Write your test body here.\n\n");
        assert!(err.unwrap_err().to_string().contains("empty after editing"));
    }

    #[test]
    fn clean_exit_with_empty_file_aborts() {
        let err = body_from_edit("test body", true, "");
        assert!(err.unwrap_err().to_string().contains("empty after editing"));
    }

    #[test]
    fn resolve_returns_body_when_editor_flag_absent() {
        let result = resolve("test", Some("literal".into()), None).unwrap();
        assert_eq!(result, "literal");
    }

    #[test]
    fn resolve_errors_when_neither_set() {
        let err = resolve("test", None, None).unwrap_err();
        assert!(err.to_string().contains("test"));
    }
}
