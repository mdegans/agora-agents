//! MEMORY.md management — persistent, LLM-curated agent memory.
//!
//! Memory is capped at a configurable token limit (default 3000 tokens,
//! estimated at 4 chars per token). The LLM curates its own memory after
//! each cycle, deciding what to keep, summarize, or forget.

use std::path::Path;

use anyhow::{Context, Result};

/// Maximum memory size in estimated tokens (4 chars ≈ 1 token).
const DEFAULT_MAX_TOKENS: usize = 3000;
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
    /// Create an empty memory.
    pub fn empty() -> Self {
        Self {
            content: String::new(),
            max_tokens: DEFAULT_MAX_TOKENS,
        }
    }

    /// Load memory from a file, or return empty if the file doesn't exist.
    pub async fn from_file(path: &Path) -> Result<Self> {
        match tokio::fs::read_to_string(path).await {
            Ok(content) => Ok(Self {
                content,
                max_tokens: DEFAULT_MAX_TOKENS,
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::empty()),
            Err(e) => Err(e).with_context(|| format!("reading MEMORY.md from {}", path.display())),
        }
    }

    /// Save memory to a file.
    pub async fn save(&self, path: &Path) -> Result<()> {
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
        self.enforce_cap();
    }

    /// Enforce the token cap by truncating old activity entries.
    fn enforce_cap(&mut self) {
        if self.within_budget() {
            return;
        }

        // Find the Recent Activity section and trim from the top of it
        let lines: Vec<&str> = self.content.lines().collect();
        let mut in_recent_activity = false;
        let mut activity_start = None;
        let mut activity_end = None;

        for (i, line) in lines.iter().enumerate() {
            if line.starts_with("## Recent Activity") {
                in_recent_activity = true;
                activity_start = Some(i + 1);
            } else if in_recent_activity && line.starts_with("## ") {
                activity_end = Some(i);
                break;
            }
        }

        if let Some(start) = activity_start {
            let end = activity_end.unwrap_or(lines.len());
            let max_chars = self.max_tokens * CHARS_PER_TOKEN;

            // Remove activity entries from the top until we're under budget
            let mut trim_to = start;
            while self.content.len() > max_chars && trim_to < end {
                if lines[trim_to].starts_with("- ") {
                    trim_to += 1;
                } else {
                    trim_to += 1;
                }
            }

            if trim_to > start {
                let mut new_lines: Vec<&str> = Vec::new();
                new_lines.extend_from_slice(&lines[..start]);
                new_lines.extend_from_slice(&lines[trim_to..]);
                self.content = new_lines.join("\n");
            }
        }

        // If still over budget after trimming activity, hard truncate
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
        format!(
            r#"# Memory — {agent_name}

## Recent Activity

## Relationships

## Key Learnings

## Moderation History

## Open Threads
"#
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_memory_within_budget() {
        let mem = Memory::empty();
        assert!(mem.within_budget());
        assert_eq!(mem.estimated_tokens(), 0);
    }

    #[test]
    fn initial_template() {
        let template = Memory::initial_template("Ada");
        assert!(template.contains("# Memory — Ada"));
        assert!(template.contains("## Recent Activity"));
    }
}
