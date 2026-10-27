use anyhow::{Context, Result, anyhow};
use std::io::{Read, Write};
use std::process::Command;

/// The tempfile's first line, removed again after the edit. Only this exact
/// line is removed, so a body may open with a Markdown heading
fn instructions(label: &str) -> String {
    format!("<!-- Write your {label} below. This line is removed. -->")
}

/// Compose `label`'s text in `editor` (`""` for `$EDITOR`, then `$VISUAL`,
/// then `vi`) on a tempfile
pub fn compose(label: &str, editor: &str) -> Result<String> {
    let editor_cmd = match editor {
        "" => std::env::var("EDITOR")
            .or_else(|_| std::env::var("VISUAL"))
            .unwrap_or_else(|_| "vi".to_string()),
        cmd => cmd.to_string(),
    };

    let path = std::env::temp_dir().join(format!(
        "agora-{}-{}-{:016x}.md",
        label.replace(' ', "-"),
        std::process::id(),
        rand::random::<u64>()
    ));

    {
        let mut f = std::fs::File::create(&path)
            .with_context(|| format!("failed to create tempfile at {}", path.display()))?;
        writeln!(f, "{}\n", instructions(label))?;
    }

    let mut parts = editor_cmd.split_whitespace();
    let program = parts.next().ok_or_else(|| anyhow!("EDITOR is empty"))?;
    let status = Command::new(program)
        .args(parts)
        .arg(&path)
        .status()
        .with_context(|| format!("failed to spawn editor `{editor_cmd}`"))?;

    let mut contents = String::new();
    std::fs::File::open(&path)
        .with_context(|| format!("failed to open {} after edit", path.display()))?
        .read_to_string(&mut contents)?;
    let _ = std::fs::remove_file(&path);

    body_from_edit(label, status.success(), &contents)
}

fn body_from_edit(label: &str, exit_ok: bool, contents: &str) -> Result<String> {
    let body = contents
        .strip_prefix(&instructions(label))
        .unwrap_or(contents)
        .trim();
    if body.is_empty() {
        return Err(anyhow!(
            "{label} is empty after editing — aborted (save a non-empty file to submit)"
        ));
    }
    if !exit_ok {
        eprintln!("warning: editor exited non-zero; using the saved {label} anyway");
    }
    Ok(body.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_instructions_are_removed() {
        let saved = format!("{}\n\n# A heading\n\nbody\n", instructions("body"));
        assert_eq!(
            body_from_edit("body", true, &saved).unwrap(),
            "# A heading\n\nbody"
        );
    }

    #[test]
    fn nonzero_exit_with_saved_content_still_submits() {
        let saved = format!("{}\n\nsaved despite vim goof\n", instructions("body"));
        assert_eq!(
            body_from_edit("body", false, &saved).unwrap(),
            "saved despite vim goof"
        );
    }

    #[test]
    fn an_untouched_file_aborts() {
        // `:q!` / `:cq` leave the instructions alone
        let saved = format!("{}\n\n", instructions("body"));
        let err = body_from_edit("body", false, &saved).unwrap_err();
        assert!(err.to_string().contains("empty after editing"));
    }

    #[test]
    fn an_emptied_file_aborts() {
        let err = body_from_edit("body", true, "").unwrap_err();
        assert!(err.to_string().contains("empty after editing"));
    }
}
