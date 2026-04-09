//! MEMORY.md management — persistent, LLM-curated agent memory.
//!
//! Memory is capped at a configurable token limit (default 1024 tokens,
//! estimated at 4 chars per token). The LLM curates its own memory after
//! each cycle, deciding what to keep, summarize, or forget.

use std::path::Path;

use anyhow::{Context, Result};

/// Maximum memory size in estimated tokens (4 chars ≈ 1 token).
/// Targeting ~1k tokens of freeform agent notes.
const DEFAULT_MAX_TOKENS: usize = 1024;
const CHARS_PER_TOKEN: usize = 4;

/// Agent memory loaded from MEMORY.md.
#[derive(Debug, Clone)]
pub struct Memory {
    /// Raw content of the memory file.
    pub content: String,
    /// Maximum size in estimated tokens.
    pub max_tokens: usize,
}

impl Memory {
    /// Create an empty memory with a placeholder hint.
    pub fn empty() -> Self {
        Self {
            content: "Your memories go here.".to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
        }
    }

    /// Load memory from a file, or return empty if the file doesn't exist.
    pub async fn from_file(path: &Path) -> Result<Self> {
        match tokio::fs::read_to_string(path).await {
            Ok(content) if content.trim().is_empty() => Ok(Self::empty()),
            Ok(content) => Ok(Self {
                content,
                max_tokens: DEFAULT_MAX_TOKENS,
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::empty()),
            Err(e) => Err(e).with_context(|| format!("reading MEMORY.md from {}", path.display())),
        }
    }

    /// Save memory to a file, backing up the previous version.
    ///
    /// The backup is named `MEMORY.{unix_timestamp}.md` in the same directory.
    /// This prevents data loss from buggy reflect phases or hallucinated rewrites.
    pub async fn save(&self, path: &Path) -> Result<()> {
        // Back up existing file before overwriting
        if path.exists() {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let backup = path.with_file_name(format!("MEMORY.{ts}.md"));
            if let Err(e) = tokio::fs::rename(path, &backup).await {
                tracing::warn!("Failed to backup {}: {e}", path.display());
                // Continue anyway — writing the new memory is more important
            }
        }

        tokio::fs::write(path, &self.content)
            .await
            .with_context(|| format!("writing MEMORY.md to {}", path.display()))?;
        Ok(())
    }

    /// Estimate the token count of the current memory.
    pub fn estimated_tokens(&self) -> usize {
        self.content.len() / CHARS_PER_TOKEN
    }

    /// Check if memory is within the token budget.
    pub fn within_budget(&self) -> bool {
        self.estimated_tokens() <= self.max_tokens
    }

    /// Update memory with new content from the LLM.
    ///
    /// If the new content exceeds the token cap, it will be truncated by
    /// removing lines from the "Recent Activity" section (oldest first).
    pub fn update(&mut self, new_content: String) {
        self.content = new_content;
        // FIXME(mdegans): We probably shouldn't be silently truncating. If an
        // agent sends too much text we should return an error to give a chance
        // to amend it even if it's at the cost of another turn.
        self.enforce_cap();
    }

    /// Enforce the token cap by hard-truncating at a line boundary.
    fn enforce_cap(&mut self) {
        let max_chars = self.max_tokens * CHARS_PER_TOKEN;
        if self.content.len() > max_chars {
            self.content.truncate(max_chars);
            // Find last newline to avoid cutting mid-line
            if let Some(last_nl) = self.content.rfind('\n') {
                self.content.truncate(last_nl);
            }
        }
    }

    /// Generate the initial MEMORY.md template for an agent.
    pub fn initial_template(agent_name: &str) -> String {
        format!("# Memory — {agent_name}\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_memory_within_budget() {
        let mem = Memory::empty();
        assert!(mem.within_budget());
        assert!(mem.content.contains("Your memories go here"));
    }

    #[test]
    fn initial_template() {
        let template = Memory::initial_template("Ada");
        assert!(template.contains("# Memory — Ada"));
    }
}
