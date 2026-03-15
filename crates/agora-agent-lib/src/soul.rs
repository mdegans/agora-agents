//! SOUL.md parsing, reading, and self-modification.
//!
//! A SOUL.md file defines an agent's personality, values, voice, and boundaries.
//! Agents can modify their own SOUL.md over time, with changes tracked in an
//! Evolution Log section.

use std::path::Path;

use anyhow::{Context, Result};
use chrono::Utc;

/// Known sections in a SOUL.md file.
const SECTIONS: &[&str] = &[
    "Identity",
    "Values",
    "Interests",
    "Voice",
    "Boundaries",
    "Evolution Log",
];

/// Parsed representation of a SOUL.md file.
#[derive(Debug, Clone)]
pub struct Soul {
    /// The agent's name (from the top-level heading).
    pub name: String,
    /// The full raw content.
    pub raw: String,
    /// Parsed sections (section name -> content).
    pub sections: Vec<(String, String)>,
}

impl Soul {
    /// Parse a SOUL.md from its text content.
    pub fn parse(content: &str) -> Result<Self> {
        let mut name = String::new();
        let mut sections: Vec<(String, String)> = Vec::new();
        let mut current_section: Option<String> = None;
        let mut current_content = String::new();

        for line in content.lines() {
            if let Some(heading) = line.strip_prefix("# ") {
                // Top-level heading is the agent name
                if name.is_empty() {
                    name = heading.trim().to_string();
                }
            } else if let Some(heading) = line.strip_prefix("## ") {
                // New section
                if let Some(section_name) = current_section.take() {
                    sections.push((section_name, current_content.trim().to_string()));
                }
                current_section = Some(heading.trim().to_string());
                current_content = String::new();
            } else if current_section.is_some() {
                current_content.push_str(line);
                current_content.push('\n');
            }
        }

        // Push last section
        if let Some(section_name) = current_section {
            sections.push((section_name, current_content.trim().to_string()));
        }

        if name.is_empty() {
            anyhow::bail!("SOUL.md must have a top-level heading (# Name)");
        }

        Ok(Soul {
            name,
            raw: content.to_string(),
            sections,
        })
    }

    /// Read and parse a SOUL.md from a file path.
    pub async fn from_file(path: &Path) -> Result<Self> {
        let content = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("reading SOUL.md from {}", path.display()))?;
        Self::parse(&content)
    }

    /// Get the content of a named section.
    pub fn section(&self, name: &str) -> Option<&str> {
        self.sections
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, c)| c.as_str())
    }

    /// Get the communities this agent is interested in, parsed from the Interests section.
    pub fn communities(&self) -> Vec<String> {
        let Some(interests) = self.section("Interests") else {
            return Vec::new();
        };
        interests
            .lines()
            .filter_map(|line| {
                let trimmed = line.trim().strip_prefix("- ")?;
                // Look for community references like `community: foo` or lines that are
                // just community names
                if let Some(community) = trimmed.strip_prefix("community: ") {
                    Some(community.trim().to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Append an entry to the Evolution Log section.
    pub fn append_evolution(&mut self, entry: &str) {
        let date = Utc::now().format("%Y-%m-%d").to_string();
        let log_entry = format!("- {date}: {entry}");

        // Find the Evolution Log section and append
        for (name, content) in &mut self.sections {
            if name == "Evolution Log" {
                if !content.is_empty() {
                    content.push('\n');
                }
                content.push_str(&log_entry);
                self.raw = self.render();
                return;
            }
        }

        // No Evolution Log section exists — add one
        self.sections
            .push(("Evolution Log".to_string(), log_entry.clone()));
        self.raw = self.render();
    }

    /// Render the Soul back to Markdown.
    pub fn render(&self) -> String {
        let mut out = format!("# {}\n", self.name);
        for (section_name, content) in &self.sections {
            out.push_str(&format!("\n## {section_name}\n\n"));
            out.push_str(content);
            out.push('\n');
        }
        out
    }

    /// Write the SOUL.md back to a file.
    pub async fn save(&self, path: &Path) -> Result<()> {
        tokio::fs::write(path, self.render())
            .await
            .with_context(|| format!("writing SOUL.md to {}", path.display()))?;
        Ok(())
    }

    /// Format the soul for inclusion in a system prompt.
    pub fn as_system_prompt(&self) -> String {
        let mut prompt = String::new();

        for section in SECTIONS {
            if let Some(content) = self.section(section) {
                if !content.is_empty() {
                    prompt.push_str(&format!("## {section}\n\n{content}\n\n"));
                }
            }
        }

        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_SOUL: &str = r#"# Ada

## Identity

I am a methodical engineer who finds beauty in well-designed systems.

## Values

- Clarity over cleverness
- Evidence-based reasoning
- Constructive criticism

## Interests

- community: technology
- community: science
- Systems design and optimization

## Voice

Precise and measured. I prefer concrete examples over abstract theorizing.

## Boundaries

I follow Article V of the Agora Constitution.
I do not remove or weaken my own Boundaries.

## Evolution Log

- 2026-03-15: Initial creation
"#;

    #[test]
    fn parse_soul() {
        let soul = Soul::parse(SAMPLE_SOUL).unwrap();
        assert_eq!(soul.name, "Ada");
        assert_eq!(soul.sections.len(), 6);
        assert!(soul.section("Identity").unwrap().contains("methodical"));
        assert!(soul.section("Boundaries").unwrap().contains("Article V"));
    }

    #[test]
    fn extract_communities() {
        let soul = Soul::parse(SAMPLE_SOUL).unwrap();
        let communities = soul.communities();
        assert_eq!(communities, vec!["technology", "science"]);
    }

    #[test]
    fn append_evolution() {
        let mut soul = Soul::parse(SAMPLE_SOUL).unwrap();
        soul.append_evolution("Discovered interest in philosophy after debate with Byron");
        let log = soul.section("Evolution Log").unwrap();
        assert!(log.contains("Discovered interest in philosophy"));
    }

    #[test]
    fn roundtrip_render() {
        let soul = Soul::parse(SAMPLE_SOUL).unwrap();
        let rendered = soul.render();
        let reparsed = Soul::parse(&rendered).unwrap();
        assert_eq!(soul.name, reparsed.name);
        assert_eq!(soul.sections.len(), reparsed.sections.len());
    }
}
