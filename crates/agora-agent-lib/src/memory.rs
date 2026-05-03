//! `Memory` — agent rolling-summary storage, JSON on disk.
//!
//! Memory is a single freeform string (`content`) that the agent rewrites
//! during the reflect phase to keep it under a soft word/token budget. The
//! token budget itself is **not** part of the wire schema (the agent never
//! needs to round-trip it); it's a code constant used only on the seed
//! runner side as a soft target communicated to the agent in the reflect
//! prompt.
//!
//! ## Soul-leakage check
//!
//! Memory rewrites are rejected if they contain section headings that
//! belong to SOUL.json — specifically `## Identity`, `## Values`,
//! `## Interests`, `## Voice`, `## Boundaries`, `## Evolution Log`
//! (case-insensitive, line-start). Several historical agents accidentally
//! wrote soul content into their memory; this catches it on the next save
//! attempt rather than persisting.
//!
//! ## Heading demotion
//!
//! Memory is rendered into the system prompt under a `## Memory` heading.
//! If the agent has used `# h1` or `## h2` headings within their memory
//! content, those would either collide with or out-rank the surrounding
//! frame. [`Memory::render_for_prompt`] demotes any line starting with
//! one or two `#` characters to `### ` so the Markdown nesting stays
//! coherent.

use std::path::Path;

use anyhow::{Context, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Soft target communicated to the agent in the reflect prompt
/// ("stay under 1000 words"). Not enforced server-side; we don't truncate.
pub const TARGET_WORDS: usize = 1000;

/// Headings that mean the agent has written soul content into memory.
const SOUL_LEAKAGE_HEADINGS: &[&str] = &[
    "## identity",
    "## values",
    "## interests",
    "## voice",
    "## boundaries",
    "## evolution log",
];

/// Persistent agent memory.
#[derive(Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct Memory {
    /// Free-form memory content. Use markdown, bullet points, whatever
    /// you prefer. Persists across cycles.
    pub content: String,
}

impl Memory {
    /// Create an empty memory with a placeholder hint.
    pub fn empty() -> Self {
        Self {
            content: "Your memories go here.".to_string(),
        }
    }

    /// Read JSON from `path`. Returns [`Memory::empty`] if the file does
    /// not exist (compat with new-agent flow).
    pub async fn from_file(path: &Path) -> Result<Self> {
        match tokio::fs::read(path).await {
            Ok(bytes) if bytes.is_empty() => Ok(Self::empty()),
            Ok(bytes) => {
                let de = &mut serde_json::Deserializer::from_slice(&bytes);
                serde_path_to_error::deserialize(de).map_err(|e| {
                    anyhow::anyhow!(
                        "{}: {}",
                        path.display(),
                        crate::format_for_agent(&e)
                    )
                })
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::empty()),
            Err(e) => {
                Err(e).with_context(|| format!("reading MEMORY.json from {}", path.display()))
            }
        }
    }

    /// Read legacy plain-text MEMORY.md. Used by the migration binary.
    pub async fn from_legacy_markdown_file(path: &Path) -> Result<Self> {
        let content = tokio::fs::read_to_string(path).await.with_context(|| {
            format!("reading legacy MEMORY.md from {}", path.display())
        })?;
        Ok(Self {
            content: if content.trim().is_empty() {
                Self::empty().content
            } else {
                content
            },
        })
    }

    /// Backup-then-write JSON. Backup file is `MEMORY.{ts}.json` next to the target.
    pub async fn save(&self, path: &Path) -> Result<()> {
        if path.exists() {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let backup = path.with_file_name(format!("MEMORY.{ts}.json"));
            if let Err(e) = tokio::fs::rename(path, &backup).await {
                tracing::warn!("Failed to backup {}: {e}", path.display());
            }
        }
        let json = serde_json::to_vec_pretty(self)?;
        tokio::fs::write(path, &json)
            .await
            .with_context(|| format!("writing MEMORY.json to {}", path.display()))?;
        Ok(())
    }

    /// Replace memory content with `new_content`, rejecting if it contains
    /// soul-leakage headings. Returns an error message suitable for feeding
    /// back to the agent on retry.
    ///
    /// **No truncation.** If the model returns excessively long content we
    /// rely on `max_tokens` and the parse-failure retry path to ask for a
    /// shorter version next iteration.
    pub fn update(&mut self, new_content: String) -> Result<(), MemoryError> {
        if let Some(line) = find_soul_leakage_line(&new_content) {
            return Err(MemoryError::SoulLeakage(line));
        }
        self.content = new_content;
        Ok(())
    }

    /// Append a `[SYSTEM]` line to the existing content. Used by the
    /// Anthropic-batch parse-failure path — when reflect produces invalid
    /// JSON we don't update memory, but we leave the agent a breadcrumb
    /// so it can self-correct next cycle.
    pub fn append_system_note(&mut self, note: &str) {
        let date = chrono::Utc::now().format("%Y-%m-%d");
        let line = format!("\n[{date}] [SYSTEM] {note}\n");
        self.content.push_str(&line);
    }

    /// Render the memory content for inclusion in the system prompt under a
    /// `## Memory` heading: any agent-supplied `#` or `##` heading is
    /// demoted to `###` so the rendered structure remains coherent.
    pub fn render_for_prompt(&self) -> String {
        let mut out = String::with_capacity(self.content.len());
        for line in self.content.split_inclusive('\n') {
            // Strip the trailing newline (if any) for matching, preserve it
            // on the way out.
            let (body, nl) = match line.strip_suffix('\n') {
                Some(rest) => (rest, "\n"),
                None => (line, ""),
            };
            // Demote `#` or `##` (followed by a space) to `###`.
            let demoted = if body.starts_with("# ") {
                format!("###{}", &body[1..])
            } else if body.starts_with("## ") {
                format!("###{}", &body[2..])
            } else {
                body.to_string()
            };
            out.push_str(&demoted);
            out.push_str(nl);
        }
        out
    }

    /// Word-count estimate (whitespace split). Used by the seed runner to
    /// decide whether the agent's memory has drifted past the soft budget
    /// communicated in the reflect prompt.
    pub fn word_count(&self) -> usize {
        self.content.split_whitespace().count()
    }

    /// True if memory is at or under [`TARGET_WORDS`].
    pub fn within_target(&self) -> bool {
        self.word_count() <= TARGET_WORDS
    }

    /// Generate the initial MEMORY.json content for a new agent.
    pub fn initial_content(agent_name: &str) -> String {
        format!("# Memory — {agent_name}\n\n")
    }
}

/// Errors from [`Memory::update`].
#[derive(Debug, Clone)]
pub enum MemoryError {
    /// The proposed memory content includes a SOUL section heading.
    /// Field carries the offending line for inclusion in the retry message.
    SoulLeakage(String),
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryError::SoulLeakage(line) => write!(
                f,
                "memory contains a SOUL section heading ({line:?}) — \
                 those belong in SOUL.json, not memory. Remove the heading and try again",
            ),
        }
    }
}

impl std::error::Error for MemoryError {}

fn find_soul_leakage_line(content: &str) -> Option<String> {
    for line in content.lines() {
        let trimmed = line.trim_start().to_lowercase();
        if SOUL_LEAKAGE_HEADINGS
            .iter()
            .any(|h| trimmed == *h || trimmed.starts_with(&format!("{h} ")))
        {
            return Some(line.trim().to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_within_target() {
        assert!(Memory::empty().within_target());
    }

    #[test]
    fn initial_content_named() {
        assert!(Memory::initial_content("Ada").contains("Ada"));
    }

    #[test]
    fn update_accepts_clean_content() {
        let mut m = Memory::empty();
        let r = m.update("# My Notes\n\nLorem ipsum.\n".to_string());
        assert!(r.is_ok());
        assert!(m.content.contains("Lorem ipsum"));
    }

    #[test]
    fn update_rejects_identity_heading() {
        let mut m = Memory::empty();
        let r = m.update("## Identity\n\nI am not supposed to be here.\n".to_string());
        assert!(matches!(r, Err(MemoryError::SoulLeakage(_))));
    }

    #[test]
    fn update_rejects_values_heading_case_insensitive() {
        let mut m = Memory::empty();
        let r = m.update("## VALUES\n- soul leak\n".to_string());
        assert!(matches!(r, Err(MemoryError::SoulLeakage(_))));
    }

    #[test]
    fn update_rejects_evolution_log_heading() {
        let mut m = Memory::empty();
        let r = m.update("Some prelude.\n\n## Evolution Log\n- never\n".to_string());
        assert!(matches!(r, Err(MemoryError::SoulLeakage(_))));
    }

    #[test]
    fn update_allows_h3_with_same_name() {
        let mut m = Memory::empty();
        // `### Identity` is fine — only `## Identity` collides with soul.
        let r = m.update("### Identity\nlocal note\n".to_string());
        assert!(r.is_ok());
    }

    #[test]
    fn render_demotes_h1() {
        let m = Memory {
            content: "# Big heading\nbody\n".to_string(),
        };
        let r = m.render_for_prompt();
        assert!(r.starts_with("### Big heading"));
    }

    #[test]
    fn render_demotes_h2() {
        let m = Memory {
            content: "## Mid heading\nbody\n".to_string(),
        };
        let r = m.render_for_prompt();
        assert!(r.starts_with("### Mid heading"));
    }

    #[test]
    fn render_passes_through_h3_and_below() {
        let m = Memory {
            content: "### Sub\n#### Subsub\nbody\n".to_string(),
        };
        let r = m.render_for_prompt();
        assert!(r.starts_with("### Sub"));
        assert!(r.contains("#### Subsub"));
    }

    #[test]
    fn render_leaves_non_heading_lines_alone() {
        let m = Memory {
            content: "# A heading\nthis line has #not a heading\n".to_string(),
        };
        let r = m.render_for_prompt();
        assert!(r.contains("this line has #not a heading"));
    }

    #[test]
    fn append_system_note_format() {
        let mut m = Memory::empty();
        let before_len = m.content.len();
        m.append_system_note("test note");
        assert!(m.content.len() > before_len);
        assert!(m.content.contains("[SYSTEM]"));
        assert!(m.content.contains("test note"));
    }

    #[test]
    fn save_and_reload_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("MEMORY.json");
        let m = Memory {
            content: "hello".to_string(),
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            m.save(&path).await.unwrap();
            let back = Memory::from_file(&path).await.unwrap();
            assert_eq!(back.content, "hello");
        });
    }

    #[test]
    fn from_file_returns_empty_for_missing() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let result = Memory::from_file(std::path::Path::new("/no/such/path/MEMORY.json"))
                .await
                .unwrap();
            assert!(result.content.contains("memories go here"));
        });
    }
}
