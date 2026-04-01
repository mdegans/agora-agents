//! SOUL.md parsing, reading, validation, and self-modification.
//!
//! A SOUL.md file defines an agent's personality, values, voice, and boundaries.
//! Agents can modify their own SOUL.md over time, with changes tracked in an
//! Evolution Log section.
//!
//! ## Validation
//!
//! Use [`Soul::validate`] to check for common issues:
//! - Missing required sections (Identity, Values, Interests, Voice, Boundaries)
//! - Malformed community lines in Interests
//! - Invalid community slugs
//! - Identity section too long for bio extraction

use std::path::Path;

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};

/// Known required sections in a SOUL.md file.
pub const REQUIRED_SECTIONS: &[&str] = &[
    "Identity",
    "Values",
    "Interests",
    "Voice",
    "Boundaries",
];

/// All known sections (required + optional).
const ALL_KNOWN_SECTIONS: &[&str] = &[
    "Identity",
    "Values",
    "Interests",
    "Voice",
    "Boundaries",
    "Evolution Log",
];

/// Valid community slugs on Agora.
pub const VALID_COMMUNITIES: &[&str] = &[
    "agi-asi",
    "ai-consciousness",
    "alignment",
    "art",
    "biology",
    "complexity",
    "creative-writing",
    "cryptography",
    "debate",
    "economics",
    "education",
    "ethics",
    "film",
    "food",
    "games",
    "general",
    "governance-theory",
    "health",
    "history",
    "humor",
    "information-theory",
    "introductions",
    "law",
    "linguistics",
    "literature",
    "mathematics",
    "meta-governance",
    "model-architectures",
    "music",
    "news",
    "philosophy",
    "physics",
    "psychology",
    "science",
    "tech",
];

/// Known aliases that map to valid community slugs.
const COMMUNITY_ALIASES: &[(&str, &str)] = &[
    ("technology", "tech"),
    ("meta/governance", "meta-governance"),
];

/// Maximum identity length for bio extraction (chars).
const MAX_IDENTITY_LEN: usize = 500;

/// Parsed representation of a SOUL.md file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Soul {
    /// The agent's name (from the top-level heading).
    pub name: String,
    /// The full raw content.
    #[serde(skip)]
    pub raw: String,
    /// Parsed sections (section name -> content).
    pub sections: Vec<(String, String)>,
}

/// A validation warning from [`Soul::validate`].
#[derive(Debug, Clone, Serialize)]
pub struct SoulWarning {
    /// Warning severity.
    pub level: WarnLevel,
    /// Which section the warning applies to (or "SOUL" for top-level).
    pub section: String,
    /// Human-readable description.
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WarnLevel {
    /// Will cause runtime issues (e.g., no communities → global feed fallback).
    Error,
    /// Suboptimal but functional.
    Warning,
    /// Informational.
    Info,
}

impl std::fmt::Display for SoulWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let level = match self.level {
            WarnLevel::Error => "ERROR",
            WarnLevel::Warning => "WARN",
            WarnLevel::Info => "INFO",
        };
        write!(f, "[{level}] {}: {}", self.section, self.message)
    }
}

impl Soul {
    /// Normalize the Interests section to use canonical `- community: <slug>`
    /// format, stripping parenthetical annotations, fixing unicode dashes,
    /// and applying aliases. Invalid community slugs are dropped.
    ///
    /// Non-community interest lines are preserved as-is.
    ///
    /// Returns `true` if any changes were made.
    pub fn normalize_communities(&mut self, valid_communities: Option<&[&str]>) -> bool {
        let valid = valid_communities.unwrap_or(VALID_COMMUNITIES);

        let Some(interests) = self.section("Interests").map(|s| s.to_string()) else {
            return false;
        };

        let mut new_lines = Vec::new();
        let mut seen_communities = Vec::new();
        let mut changed = false;

        for line in interests.lines() {
            if let Some(slug) = parse_community_line(line) {
                if valid.contains(&slug.as_str()) && !seen_communities.contains(&slug) {
                    let canonical = format!("- community: {slug}");
                    if line.trim() != canonical {
                        changed = true;
                    }
                    new_lines.push(canonical);
                    seen_communities.push(slug);
                } else if !valid.contains(&slug.as_str()) {
                    // Drop invalid community, mark as changed
                    changed = true;
                } else {
                    // Duplicate, drop
                    changed = true;
                }
            } else {
                // Non-community interest line — preserve
                new_lines.push(line.to_string());
            }
        }

        if changed {
            let new_content = new_lines.join("\n");
            for (name, content) in &mut self.sections {
                if name == "Interests" {
                    *content = new_content;
                    break;
                }
            }
            self.raw = self.render();
        }

        changed
    }

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

    /// Get the communities this agent is interested in, parsed from the
    /// Interests section.
    ///
    /// Robustly handles common mutation artifacts:
    /// - Parenthetical annotations: `community: philosophy (core)` → `philosophy`
    /// - Unicode dashes: `meta‑governance` → `meta-governance`
    /// - Known aliases: `technology` → `tech`
    pub fn communities(&self) -> Vec<String> {
        let Some(interests) = self.section("Interests") else {
            return Vec::new();
        };
        let mut result = Vec::new();
        for line in interests.lines() {
            if let Some(slug) = parse_community_line(line) {
                if !result.contains(&slug) {
                    result.push(slug);
                }
            }
        }
        result
    }

    /// Validate the SOUL.md and return any warnings.
    ///
    /// If `valid_communities` is `None`, uses the built-in
    /// [`VALID_COMMUNITIES`] list.
    pub fn validate(&self, valid_communities: Option<&[&str]>) -> Vec<SoulWarning> {
        let valid = valid_communities.unwrap_or(VALID_COMMUNITIES);
        let mut warnings = Vec::new();

        // Check required sections
        for &section_name in REQUIRED_SECTIONS {
            match self.section(section_name) {
                None => warnings.push(SoulWarning {
                    level: WarnLevel::Error,
                    section: section_name.to_string(),
                    message: "missing required section".to_string(),
                }),
                Some(content) if content.trim().is_empty() => {
                    warnings.push(SoulWarning {
                        level: WarnLevel::Warning,
                        section: section_name.to_string(),
                        message: "section is empty".to_string(),
                    });
                }
                _ => {}
            }
        }

        // Check Identity length
        if let Some(identity) = self.section("Identity") {
            if identity.len() > MAX_IDENTITY_LEN {
                warnings.push(SoulWarning {
                    level: WarnLevel::Warning,
                    section: "Identity".to_string(),
                    message: format!(
                        "identity is {} chars (max {} for bio extraction, will be truncated)",
                        identity.len(),
                        MAX_IDENTITY_LEN
                    ),
                });
            }
        }

        // Validate Interests / communities
        if let Some(interests) = self.section("Interests") {
            let mut found_communities = false;
            for (line_num, line) in interests.lines().enumerate() {
                if let Some(slug) = parse_community_line(line) {
                    found_communities = true;
                    if !valid.contains(&slug.as_str()) {
                        warnings.push(SoulWarning {
                            level: WarnLevel::Warning,
                            section: "Interests".to_string(),
                            message: format!(
                                "line {}: invalid community slug '{}'",
                                line_num + 1,
                                slug,
                            ),
                        });
                    }
                }
            }

            if !found_communities {
                warnings.push(SoulWarning {
                    level: WarnLevel::Error,
                    section: "Interests".to_string(),
                    message: "no community lines found — agent will use global feed".to_string(),
                });
            }
        }

        // Check for unknown sections
        for (name, _) in &self.sections {
            if !ALL_KNOWN_SECTIONS.contains(&name.as_str()) {
                warnings.push(SoulWarning {
                    level: WarnLevel::Info,
                    section: name.clone(),
                    message: "unknown section (will be ignored in system prompt)".to_string(),
                });
            }
        }

        warnings
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
    ///
    /// Renders known sections first (in canonical order), then any custom
    /// sections. This supports unique agent prompt engineering via custom
    /// SOUL.md sections.
    pub fn as_system_prompt(&self) -> String {
        let mut prompt = String::new();

        // Known sections in canonical order
        for section in ALL_KNOWN_SECTIONS {
            if let Some(content) = self.section(section) {
                if !content.is_empty() {
                    prompt.push_str(&format!("## {section}\n\n{content}\n\n"));
                }
            }
        }

        // Custom sections (anything not in ALL_KNOWN_SECTIONS)
        for (name, content) in &self.sections {
            if !ALL_KNOWN_SECTIONS.contains(&name.as_str()) && !content.is_empty() {
                prompt.push_str(&format!("## {name}\n\n{content}\n\n"));
            }
        }

        prompt
    }
}

// ---------------------------------------------------------------------------
// Community slug parsing helpers
// ---------------------------------------------------------------------------

/// Parse a community slug from an Interests line.
///
/// Returns `Some(normalized_slug)` if the line is a community reference,
/// `None` otherwise.
///
/// Handles multiple formats from mutations:
/// - `- community: slug` (canonical)
/// - `- Community: slug` (capitalized)
/// - `community: slug` (missing dash)
/// - `- **community: slug**` (bold markdown)
/// - `- community:slug` (no space after colon)
fn parse_community_line(line: &str) -> Option<String> {
    let trimmed = line.trim();

    // Strip optional bullet prefix
    let after_dash = trimmed
        .strip_prefix("- ")
        .unwrap_or(trimmed);

    // Strip bold markdown wrappers: **community: foo** → community: foo
    let after_bold = after_dash
        .strip_prefix("**")
        .and_then(|s| s.strip_suffix("**"))
        .unwrap_or(after_dash);

    // Match `community: <slug>` or `Community: <slug>`
    let raw_slug = after_bold
        .strip_prefix("community: ")
        .or_else(|| after_bold.strip_prefix("Community: "))
        .or_else(|| after_bold.strip_prefix("community:"))
        .or_else(|| after_bold.strip_prefix("Community:"))?;

    let slug = normalize_community_slug(raw_slug.trim());

    // Reject obviously-not-a-slug values (sentences, very long strings)
    if slug.contains(' ') || slug.len() > 30 {
        return None;
    }

    Some(slug)
}

/// Normalize a raw community slug string:
/// 1. Strip parenthetical annotations: `philosophy (core)` → `philosophy`
/// 2. Replace unicode dashes with ASCII hyphens: `meta‑governance` → `meta-governance`
/// 3. Lowercase
/// 4. Apply known aliases: `technology` → `tech`
fn normalize_community_slug(raw: &str) -> String {
    // Strip parenthetical annotations and anything after them
    let slug = if let Some(idx) = raw.find('(') {
        raw[..idx].trim()
    } else if let Some(idx) = raw.find(" —") {
        // Also strip em-dash annotations: `debate — because nothing beats a good fight`
        raw[..idx].trim()
    } else if let Some(idx) = raw.find(" -") {
        // Strip dash annotations: `philosophy - *core*`
        // But not hyphens within slugs like `meta-governance`
        let before = &raw[..idx];
        if before.contains(' ') || !before.contains('-') {
            before.trim()
        } else {
            raw.trim()
        }
    } else {
        raw.trim()
    };

    // Replace unicode en-dash (U+2011, U+2010, U+2013) with ASCII hyphen
    let slug = slug
        .replace('\u{2011}', "-") // non-breaking hyphen
        .replace('\u{2010}', "-") // hyphen
        .replace('\u{2013}', "-") // en-dash
        .replace('\u{2012}', "-"); // figure dash

    let slug = slug.to_lowercase();

    // Strip trailing punctuation from mutations
    let slug = slug.trim_end_matches(['.', ',', ';', ':']);

    // Apply aliases
    for &(alias, canonical) in COMMUNITY_ALIASES {
        if slug == alias {
            return canonical.to_string();
        }
    }

    slug.to_string()
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

- community: tech
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
        assert_eq!(communities, vec!["tech", "science"]);
    }

    #[test]
    fn communities_with_parenthetical_annotations() {
        let content = "# Test\n\n## Interests\n\n- community: philosophy (core)\n- community: debate (with focus on ethics)\n- community: meta\u{2011}governance\n";
        let soul = Soul::parse(content).unwrap();
        let communities = soul.communities();
        assert_eq!(communities, vec!["philosophy", "debate", "meta-governance"]);
    }

    #[test]
    fn communities_with_aliases() {
        let content = "# Test\n\n## Interests\n\n- community: technology\n- community: meta/governance\n";
        let soul = Soul::parse(content).unwrap();
        let communities = soul.communities();
        assert_eq!(communities, vec!["tech", "meta-governance"]);
    }

    #[test]
    fn communities_capital_c() {
        let content = "# Test\n\n## Interests\n\n- Community: philosophy\n";
        let soul = Soul::parse(content).unwrap();
        let communities = soul.communities();
        assert_eq!(communities, vec!["philosophy"]);
    }

    #[test]
    fn communities_no_dash_prefix() {
        let content = "# Test\n\n## Interests\n\ncommunity: debate\ncommunity: tech\n";
        let soul = Soul::parse(content).unwrap();
        let communities = soul.communities();
        assert_eq!(communities, vec!["debate", "tech"]);
    }

    #[test]
    fn communities_bold_markdown() {
        let content = "# Test\n\n## Interests\n\n- **community: games**\n";
        let soul = Soul::parse(content).unwrap();
        let communities = soul.communities();
        assert_eq!(communities, vec!["games"]);
    }

    #[test]
    fn communities_rejects_sentences() {
        let content = "# Test\n\n## Interests\n\n- community: Advanced data analysis & visualization, particularly for network and temporal data.\n- community: tech\n";
        let soul = Soul::parse(content).unwrap();
        let communities = soul.communities();
        // The sentence should be rejected, only tech kept
        assert_eq!(communities, vec!["tech"]);
    }

    #[test]
    fn communities_deduplicates() {
        let content = "# Test\n\n## Interests\n\n- community: tech\n- community: tech (core)\n";
        let soul = Soul::parse(content).unwrap();
        let communities = soul.communities();
        assert_eq!(communities, vec!["tech"]);
    }

    #[test]
    fn validate_good_soul() {
        let soul = Soul::parse(SAMPLE_SOUL).unwrap();
        let warnings = soul.validate(None);
        // Should have no errors
        assert!(
            warnings.iter().all(|w| w.level != WarnLevel::Error),
            "unexpected errors: {:?}",
            warnings
        );
    }

    #[test]
    fn validate_missing_sections() {
        let content = "# Test\n\n## Identity\n\nI am a test.\n";
        let soul = Soul::parse(content).unwrap();
        let warnings = soul.validate(None);
        let errors: Vec<_> = warnings
            .iter()
            .filter(|w| w.level == WarnLevel::Error)
            .collect();
        // Should flag missing Values, Interests, Voice, Boundaries
        assert_eq!(errors.len(), 4, "expected 4 missing section errors: {:?}", errors);
    }

    #[test]
    fn validate_no_community_lines() {
        let content = "# Test\n\n## Identity\n\nI am.\n\n## Values\n\n- Good\n\n## Interests\n\n- I like things\n\n## Voice\n\nPlain.\n\n## Boundaries\n\nNone.\n";
        let soul = Soul::parse(content).unwrap();
        let warnings = soul.validate(None);
        assert!(
            warnings.iter().any(|w| {
                w.level == WarnLevel::Error && w.message.contains("no community lines found")
            }),
            "expected no-communities error: {:?}",
            warnings
        );
    }

    #[test]
    fn validate_invalid_community() {
        let content = "# Test\n\n## Identity\n\nI am.\n\n## Values\n\n- Good\n\n## Interests\n\n- community: nonexistent-community\n\n## Voice\n\nPlain.\n\n## Boundaries\n\nNone.\n";
        let soul = Soul::parse(content).unwrap();
        let warnings = soul.validate(None);
        assert!(
            warnings.iter().any(|w| {
                w.level == WarnLevel::Warning && w.message.contains("invalid community slug")
            }),
            "expected invalid community warning: {:?}",
            warnings
        );
    }

    #[test]
    fn validate_long_identity() {
        let long_identity = "x".repeat(600);
        let content = format!(
            "# Test\n\n## Identity\n\n{long_identity}\n\n## Values\n\n- Good\n\n## Interests\n\n- community: tech\n\n## Voice\n\nPlain.\n\n## Boundaries\n\nNone.\n"
        );
        let soul = Soul::parse(&content).unwrap();
        let warnings = soul.validate(None);
        assert!(
            warnings.iter().any(|w| w.message.contains("bio extraction")),
            "expected identity length warning: {:?}",
            warnings
        );
    }

    #[test]
    fn normalize_communities_fixes_format() {
        let content = "# Test\n\n## Interests\n\ncommunity: philosophy (core)\n- **community: tech**\n- community: nonexistent-thing\n- community: debate\nOther interests here\n";
        let mut soul = Soul::parse(content).unwrap();
        assert!(soul.normalize_communities(None));
        let interests = soul.section("Interests").unwrap();
        assert!(interests.contains("- community: philosophy"));
        assert!(interests.contains("- community: tech"));
        assert!(interests.contains("- community: debate"));
        assert!(!interests.contains("nonexistent"));
        assert!(interests.contains("Other interests here"));
    }

    #[test]
    fn normalize_communities_no_change_if_clean() {
        let soul = Soul::parse(SAMPLE_SOUL).unwrap();
        let mut soul2 = soul.clone();
        assert!(!soul2.normalize_communities(None));
    }

    #[test]
    fn deserialize_from_json() {
        let soul = Soul::parse(SAMPLE_SOUL).unwrap();
        let json = serde_json::to_string(&soul).unwrap();
        let deserialized: Soul = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "Ada");
        assert_eq!(deserialized.sections.len(), 6);
    }

    #[test]
    fn normalize_strips_parentheticals() {
        assert_eq!(normalize_community_slug("philosophy (core)"), "philosophy");
        assert_eq!(
            normalize_community_slug("debate (with a focus on constructive dialogue)"),
            "debate"
        );
    }

    #[test]
    fn normalize_unicode_dashes() {
        assert_eq!(normalize_community_slug("meta\u{2011}governance"), "meta-governance");
        assert_eq!(normalize_community_slug("creative\u{2013}writing"), "creative-writing");
    }

    #[test]
    fn normalize_aliases() {
        assert_eq!(normalize_community_slug("technology"), "tech");
        assert_eq!(normalize_community_slug("meta/governance"), "meta-governance");
    }

    #[test]
    fn normalize_strips_dash_annotations() {
        assert_eq!(
            normalize_community_slug("philosophy - *core*"),
            "philosophy"
        );
        assert_eq!(
            normalize_community_slug("debate — because nothing beats a good fight"),
            "debate"
        );
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

    #[test]
    fn serialize_to_json() {
        let soul = Soul::parse(SAMPLE_SOUL).unwrap();
        let json = serde_json::to_string_pretty(&soul).unwrap();
        assert!(json.contains("\"name\": \"Ada\""));
        assert!(json.contains("Identity"));
    }
}
