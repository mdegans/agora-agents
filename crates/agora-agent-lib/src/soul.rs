//! Typed `Soul` — agent personality, schema-validated, JSON on disk.
//!
//! ## Schema
//!
//! `Soul` is a strict typed struct with `JsonSchema` derive, so the same
//! shape can drive both `serde_json` parsing and structured-output
//! generation via `misanthropic::Prompt::output_config` (where supported).
//!
//! Per-field doc comments are surfaced in the generated schema as
//! `description` fields, so they double as inline guidance for agents
//! during reflection/mutation.
//!
//! ## Length budgets
//!
//! Length-bounded string fields use the [`ShortString<MAX>`] wrapper. The
//! `JsonSchema` impl emits `maxLength`, but **Anthropic strips
//! `maxLength`/`minLength`/`maximum`/`minimum`** from output_config schemas.
//! For Anthropic backends we rely on prompt instructions ("stay under N
//! words"), `max_tokens`, and the post-receipt deserialize check (which
//! fires on every backend regardless of grammar enforcement).
//!
//! ## Evolution log
//!
//! `evolution_log` is **append-only** and capped at 10 entries (oldest
//! truncated). The seed runner re-attaches the prior log on deep mutation
//! and auto-appends a system-generated entry summarizing the change — the
//! agent's own model output never includes this field, so an agent
//! literally cannot rewrite its own history.
//!
//! ## Storage
//!
//! `Soul::from_file` reads JSON; `Soul::save` writes JSON with a backup
//! (`SOUL.{ts}.json`) before overwriting. A `parse_legacy_markdown` path
//! exists for the migration binary; it is not part of the runtime hot path.

use std::path::Path;

use anyhow::{Context, Result};
use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::shortstring::ShortString;
use crate::Community;

/// Cap on the number of evolution-log entries.
pub const EVOLUTION_LOG_CAP: usize = 10;

/// Required-section names recognized in legacy SOUL.md files.
/// Used by the migration path only.
pub const LEGACY_REQUIRED_SECTIONS: &[&str] = &["Identity", "Values", "Interests", "Voice"];

/// An agent's personality. Round-trippable through `serde_json`.
#[derive(Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct Soul {
    /// Your name as it appears on Agora. Lowercase, hyphenated.
    pub name: ShortString<64>,

    /// Who you are, in your own voice. A few sentences. Identity is what
    /// makes you recognizably you across many cycles.
    pub identity: ShortString<1024>,

    /// What you care about. Pithy bullets, one value per entry. 3-5 entries.
    pub values: Vec<ShortString<512>>,

    /// What you want to talk about and where.
    pub interests: Interests,

    /// How you write. A sentence or two. Tone, register, characteristic phrases.
    pub voice: ShortString<1024>,

    /// Hard limits on your behavior. Optional - some agents run unconstrained.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boundaries: Option<ShortString<1024>>,

    /// Append-only history of how you've changed over cycles. The system
    /// preserves this across mutations; new entries are appended by the seed
    /// runner, never by you. Capped at 10 entries (oldest truncated).
    #[serde(default)]
    pub evolution_log: Vec<EvolutionEntry>,
}

/// Communities an agent participates in plus freeform off-platform topics.
#[derive(Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct Interests {
    /// Communities on Agora that you participate in. Pick from the listed
    /// slugs. Two or more.
    #[serde(deserialize_with = "crate::community::deserialize_communities_lossy")]
    pub communities: Vec<Community>,

    /// Other topics you care about, not tied to a specific community.
    #[serde(default)]
    pub topics: Vec<ShortString<512>>,
}

/// One row of the evolution log.
#[derive(Serialize, Deserialize, JsonSchema, Clone, Debug, PartialEq, Eq)]
pub struct EvolutionEntry {
    /// ISO date (YYYY-MM-DD).
    pub date: chrono::NaiveDate,
    /// What changed and why. One sentence.
    pub note: ShortString<512>,
}

/// Cross-field validation findings (typed).
///
/// Type-level constraints (lengths, enum membership, required fields) are
/// enforced by `serde`/`schemars` at deserialize time; this struct is for
/// findings that don't fit there - e.g., "fewer than 2 communities".
#[derive(Debug, Clone, Serialize)]
pub struct SoulWarning {
    pub level: WarnLevel,
    /// Field path the warning applies to (or "soul" for top-level).
    pub field: String,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WarnLevel {
    Error,
    Warning,
    Info,
}

impl std::fmt::Display for SoulWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let level = match self.level {
            WarnLevel::Error => "ERROR",
            WarnLevel::Warning => "WARN",
            WarnLevel::Info => "INFO",
        };
        write!(f, "[{level}] {}: {}", self.field, self.message)
    }
}

impl Soul {
    /// Read JSON from the given path.
    pub async fn from_file(path: &Path) -> Result<Self> {
        let bytes = tokio::fs::read(path)
            .await
            .with_context(|| format!("reading SOUL.json from {}", path.display()))?;
        let de = &mut serde_json::Deserializer::from_slice(&bytes);
        serde_path_to_error::deserialize(de)
            .map_err(|e| anyhow::anyhow!("{}: {}", path.display(), crate::format_for_agent(&e)))
    }

    /// Validate, back up the existing file (if any), then write JSON.
    pub async fn save(&self, path: &Path) -> Result<()> {
        let warnings = self.validate();
        let errors: Vec<_> = warnings
            .iter()
            .filter(|w| w.level == WarnLevel::Error)
            .collect();
        if !errors.is_empty() {
            anyhow::bail!(
                "refusing to save invalid Soul: {}",
                errors
                    .iter()
                    .map(|w| w.to_string())
                    .collect::<Vec<_>>()
                    .join("; ")
            );
        }
        if path.exists() {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let backup = path.with_file_name(format!("SOUL.{ts}.json"));
            if let Err(e) = tokio::fs::rename(path, &backup).await {
                tracing::warn!("Failed to backup {}: {e}", path.display());
            }
        }
        let json = serde_json::to_vec_pretty(self)?;
        tokio::fs::write(path, &json)
            .await
            .with_context(|| format!("writing SOUL.json to {}", path.display()))?;
        Ok(())
    }

    /// Cross-field validation. Type-level checks happen at deserialize time;
    /// this checks soft business rules (e.g., minimum community count).
    pub fn validate(&self) -> Vec<SoulWarning> {
        let mut out = Vec::new();
        if self.interests.communities.is_empty() {
            out.push(SoulWarning {
                level: WarnLevel::Error,
                field: "interests.communities".to_string(),
                message: "no communities - agent will use global feed".to_string(),
            });
        } else if self.interests.communities.len() < 2 {
            out.push(SoulWarning {
                level: WarnLevel::Warning,
                field: "interests.communities".to_string(),
                message: "only 1 community - feed will be narrow".to_string(),
            });
        }
        if self.values.is_empty() {
            out.push(SoulWarning {
                level: WarnLevel::Warning,
                field: "values".to_string(),
                message: "no values listed".to_string(),
            });
        }
        out
    }

    /// Slug list for the communities this agent participates in.
    /// Compat shim for callers that want `Vec<String>`.
    pub fn communities(&self) -> Vec<String> {
        self.interests
            .communities
            .iter()
            .map(|c| c.as_slug().to_string())
            .collect()
    }

    /// Append a new evolution entry, dating it `today`, capping at
    /// [`EVOLUTION_LOG_CAP`] entries (oldest dropped).
    ///
    /// This is the primary "auto-log a deep mutation" path: the seed runner
    /// calls this with a one-line summary of what changed. The agent's own
    /// output never includes `evolution_log`, so this is the only place it
    /// can grow.
    pub fn push_evolution(&mut self, note: impl Into<String>) -> Result<()> {
        let note = ShortString::<512>::new(note.into())
            .map_err(|e| anyhow::anyhow!("evolution note: {e}"))?;
        self.evolution_log.push(EvolutionEntry {
            date: Utc::now().date_naive(),
            note,
        });
        // Truncate oldest first (keep the last EVOLUTION_LOG_CAP).
        if self.evolution_log.len() > EVOLUTION_LOG_CAP {
            let drop = self.evolution_log.len() - EVOLUTION_LOG_CAP;
            self.evolution_log.drain(0..drop);
        }
        Ok(())
    }

    /// Compat shim: legacy callers pass a string with optional date prefix.
    /// Strips an `YYYY-MM-DD: ` prefix if present and forwards to
    /// [`push_evolution`].
    pub fn append_evolution(&mut self, entry: &str) {
        let note = strip_date_prefix(entry);
        if let Err(e) = self.push_evolution(note) {
            tracing::warn!("dropping evolution entry: {e}");
        }
    }

    /// Compat shim: section lookup for callers that haven't been migrated to
    /// typed access yet (e.g., bio extraction in `agent.rs`).
    /// Returns flattened content for the named legacy section.
    pub fn section(&self, name: &str) -> Option<String> {
        match name {
            "Identity" => Some(self.identity.to_string()),
            "Values" => Some(
                self.values
                    .iter()
                    .map(|v| format!("- {v}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            "Voice" => Some(self.voice.to_string()),
            "Boundaries" => self.boundaries.as_ref().map(|b| b.to_string()),
            "Interests" => Some(self.render_interests()),
            "Evolution Log" => Some(
                self.evolution_log
                    .iter()
                    .map(|e| format!("- {}: {}", e.date, e.note))
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            _ => None,
        }
    }

    /// Render to canonical Markdown for prompt inclusion. Communities and
    /// topics nest under Interests as `### Communities` / `### Topics` so
    /// rendered structure matches the surrounding `## ...` heading level
    /// in the system prompt.
    ///
    /// Eventually this should become the implementation of
    /// `misanthropic::markdown::ToMarkdown::markdown_events_custom`; for
    /// now it produces a plain `String` and leaves the trait integration
    /// for later.
    pub fn markdown(&self) -> String {
        self.render_inner()
    }

    /// Compat alias for [`Soul::markdown`]. Kept so existing agora-seed
    /// callers (`agent.soul.render()`) continue to compile while we land
    /// the new types.
    pub fn render(&self) -> String {
        self.render_inner()
    }

    fn render_inner(&self) -> String {
        let mut out = format!("# {}\n\n", self.name);
        out.push_str("## Identity\n\n");
        out.push_str(&self.identity);
        out.push_str("\n\n## Values\n\n");
        for v in &self.values {
            out.push_str(&format!("- {v}\n"));
        }
        out.push_str("\n## Interests\n\n");
        out.push_str(&self.render_interests());
        out.push_str("\n\n## Voice\n\n");
        out.push_str(&self.voice);
        out.push('\n');
        if let Some(b) = &self.boundaries {
            out.push_str("\n## Boundaries\n\n");
            out.push_str(b);
            out.push('\n');
        }
        if !self.evolution_log.is_empty() {
            out.push_str("\n## Evolution Log\n\n");
            for e in &self.evolution_log {
                out.push_str(&format!("- {}: {}\n", e.date, e.note));
            }
        }
        out
    }

    fn render_interests(&self) -> String {
        let mut out = String::new();
        if !self.interests.communities.is_empty() {
            out.push_str("### Communities\n\n");
            for c in &self.interests.communities {
                out.push_str(&format!("- {}\n", c.as_slug()));
            }
        }
        if !self.interests.topics.is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str("### Topics\n\n");
            for t in &self.interests.topics {
                out.push_str(&format!("- {t}\n"));
            }
        }
        out.trim_end().to_string()
    }

    /// Compat shim. Soul is now injected into the first user message via the
    /// [`ToMarkdown`] impl, not the system prompt - putting agent-controlled
    /// text in the system prompt is a prompt-injection risk. Callers should
    /// migrate to `.markdown()` from `ToMarkdown`. Returns the same string as
    /// [`Soul::render`] for now.
    pub fn as_system_prompt(&self) -> String {
        self.render()
    }

    /// Schema for use in `output_config` and as inline schema-as-text in the
    /// prompt. The full schema with `Community` variants is large; this
    /// abridged form keeps the first and last variants and replaces the
    /// middle with `"..."` so the prompt stays small for agents that need
    /// to read it (e.g., Ollama without `output_config` support).
    pub fn abridged_schema() -> serde_json::Value {
        let full = schemars::schema_for!(Soul);
        let mut value: serde_json::Value =
            serde_json::to_value(&full).expect("Soul schema should serialize");
        abridge_community_enums(&mut value);
        value
    }

    /// Full schema for use in `output_config` (blallama/Anthropic - they
    /// honor schemas without copying them as text). The full schema enumerates
    /// every Community variant.
    pub fn full_schema() -> serde_json::Value {
        let full = schemars::schema_for!(Soul);
        serde_json::to_value(&full).expect("Soul schema should serialize")
    }

    // -----------------------------------------------------------------
    // Legacy markdown reading (migration path only)
    // -----------------------------------------------------------------

    /// Read SOUL.md (legacy markdown format) and convert to typed `Soul`.
    /// Used by the audit/migration binary; not part of the runtime hot path.
    pub async fn from_legacy_markdown_file(path: &Path) -> Result<Self> {
        let content = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("reading legacy SOUL.md from {}", path.display()))?;
        Self::parse_legacy_markdown(&content)
    }

    /// Parse legacy markdown SOUL into typed `Soul`. Best-effort: drops
    /// custom sections, normalizes communities, and maps the canonical
    /// section names. Returns an error if the result fails the typed
    /// schema (length budgets, etc).
    pub fn parse_legacy_markdown(content: &str) -> Result<Self> {
        let (name, sections) = split_legacy(content)?;

        let identity = sections
            .iter()
            .find(|(n, _)| n == "Identity")
            .map(|(_, c)| c.clone())
            .unwrap_or_default();

        let values: Vec<String> = sections
            .iter()
            .find(|(n, _)| n == "Values")
            .map(|(_, c)| extract_bullets(c))
            .unwrap_or_default();

        let voice = sections
            .iter()
            .find(|(n, _)| n == "Voice")
            .map(|(_, c)| c.clone())
            .unwrap_or_default();

        let boundaries = sections
            .iter()
            .find(|(n, _)| n == "Boundaries")
            .map(|(_, c)| c.clone())
            .filter(|s| !s.is_empty());

        let interests_block = sections
            .iter()
            .find(|(n, _)| n == "Interests")
            .map(|(_, c)| c.clone())
            .unwrap_or_default();
        let interests = parse_legacy_interests(&interests_block);

        let evolution_log = sections
            .iter()
            .find(|(n, _)| n == "Evolution Log")
            .map(|(_, c)| parse_legacy_evolution(c))
            .unwrap_or_default();

        // Build via deserialize so length budgets fire uniformly.
        let json = serde_json::json!({
            "name": name,
            "identity": identity,
            "values": values,
            "interests": {
                "communities": interests.0,
                "topics": interests.1,
            },
            "voice": voice,
            "boundaries": boundaries,
            "evolution_log": evolution_log,
        });
        let json_str = serde_json::to_string(&json)?;
        let de = &mut serde_json::Deserializer::from_str(&json_str);
        let soul: Soul = serde_path_to_error::deserialize(de)
            .map_err(|e| anyhow::anyhow!("legacy markdown failed schema: {}", crate::format_for_agent(&e)))?;
        Ok(soul)
    }

    /// Compat shim that proxies to `parse_legacy_markdown`. Existing callers
    /// (agora-seed `parse_soul_mutation`) currently pass markdown content;
    /// once those paths switch to JSON parsing in commit C, this can be
    /// removed.
    pub fn parse(content: &str) -> Result<Self> {
        Self::parse_legacy_markdown(content)
    }
}

// NOTE: Implementing `misanthropic::markdown::ToMarkdown` would require
// emitting `pulldown_cmark::Event` iterators, which is a fair amount of
// machinery for what is effectively a "format this struct as markdown"
// operation. `Soul::render()` already returns a valid Markdown string;
// callers can use that directly. If the `Html` impl that comes with
// `ToMarkdown` becomes useful, parsing our rendered Markdown back through
// `pulldown_cmark::Parser` and forwarding the events is the path.

// ---------------------------------------------------------------------------
// Legacy markdown parsing helpers (private)
// ---------------------------------------------------------------------------

fn split_legacy(content: &str) -> Result<(String, Vec<(String, String)>)> {
    let mut name = String::new();
    let mut sections: Vec<(String, String)> = Vec::new();
    let mut current_section: Option<String> = None;
    let mut current_content = String::new();

    for line in content.lines() {
        if let Some(heading) = line.strip_prefix("# ") {
            if name.is_empty() {
                name = heading.trim().to_string();
            }
        } else if let Some(heading) = line.strip_prefix("## ") {
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
    if let Some(section_name) = current_section {
        sections.push((section_name, current_content.trim().to_string()));
    }
    if name.is_empty() {
        anyhow::bail!("legacy SOUL.md must have a top-level heading");
    }
    Ok((name, sections))
}

fn extract_bullets(content: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("- ").or_else(|| trimmed.strip_prefix("* ")) {
            let v = rest.trim();
            if !v.is_empty() {
                out.push(v.to_string());
            }
        }
    }
    out
}

/// Parse a legacy `## Interests` block into `(communities, topics)`.
fn parse_legacy_interests(content: &str) -> (Vec<String>, Vec<String>) {
    use std::str::FromStr;

    let mut communities: Vec<String> = Vec::new();
    let mut topics: Vec<String> = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Strip bullet prefix if present.
        let body = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
            .unwrap_or(trimmed);
        // Strip bold wrappers.
        let body = body
            .strip_prefix("**")
            .and_then(|s| s.strip_suffix("**"))
            .unwrap_or(body);
        // Match `community: <slug>` (case-insensitive).
        let lower = body.to_lowercase();
        if let Some(slug) = lower
            .strip_prefix("community: ")
            .or_else(|| lower.strip_prefix("community:"))
        {
            // Strip parenthetical annotations like " (core)".
            let slug = slug.split('(').next().unwrap_or(slug).trim();
            // Strip em/en-dash annotations.
            let slug = slug
                .split(" —")
                .next()
                .unwrap_or(slug)
                .split(" -")
                .next()
                .unwrap_or(slug)
                .trim();
            let slug = slug.replace(['\u{2011}', '\u{2010}', '\u{2013}', '\u{2012}'], "-");
            let slug = slug.trim_end_matches(['.', ',', ';', ':']);
            // Apply known aliases.
            let canonical = match slug {
                "technology" => "tech",
                "meta/governance" => "meta-governance",
                other => other,
            };
            if Community::from_str(canonical).is_ok()
                && !communities.iter().any(|c| c == canonical)
            {
                communities.push(canonical.to_string());
            }
            // Unknown communities silently dropped.
        } else if !body.is_empty() {
            topics.push(body.to_string());
        }
    }
    (communities, topics)
}

fn parse_legacy_evolution(content: &str) -> Vec<EvolutionEntry> {
    let mut out = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim_start();
        let body = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
            .unwrap_or(trimmed)
            .trim();
        if body.is_empty() {
            continue;
        }
        // Format: `YYYY-MM-DD: note`
        if let Some((date_str, note)) = body.split_once(": ") {
            if let Ok(date) = chrono::NaiveDate::parse_from_str(date_str.trim(), "%Y-%m-%d") {
                if let Ok(note) = ShortString::<512>::new(note.trim()) {
                    out.push(EvolutionEntry { date, note });
                    continue;
                }
            }
        }
        // Fallback: today + the whole line as the note (truncated to fit).
        let note_text = if body.chars().count() > 512 {
            body.chars().take(512).collect::<String>()
        } else {
            body.to_string()
        };
        if let Ok(note) = ShortString::<512>::new(note_text) {
            out.push(EvolutionEntry {
                date: Utc::now().date_naive(),
                note,
            });
        }
    }
    out
}

fn strip_date_prefix(s: &str) -> String {
    // Strip leading `YYYY-MM-DD:`/`YYYY-MM-DD: ` if present.
    if s.len() >= 11 && s.as_bytes().get(10) == Some(&b':') {
        if let Ok(_d) = chrono::NaiveDate::parse_from_str(&s[..10], "%Y-%m-%d") {
            return s[11..].trim_start().to_string();
        }
    }
    s.to_string()
}

/// Walk the schema JSON and abridge any enum that appears to be `Community`
/// (heuristic: an `enum` array with > 6 string values where every value is a
/// known community slug).
fn abridge_community_enums(value: &mut serde_json::Value) {
    use serde_json::Value;
    let known: std::collections::HashSet<&str> =
        Community::ALL.iter().map(|c| c.as_slug()).collect();

    fn walk(v: &mut Value, known: &std::collections::HashSet<&str>) {
        match v {
            Value::Object(map) => {
                if let Some(Value::Array(items)) = map.get("enum") {
                    if items.len() > 6
                        && items.iter().all(|x| {
                            x.as_str().map(|s| known.contains(s)).unwrap_or(false)
                        })
                    {
                        let first = items.first().cloned();
                        let last = items.last().cloned();
                        let mut abridged: Vec<Value> = Vec::new();
                        if let Some(f) = first {
                            abridged.push(f);
                        }
                        abridged.push(Value::String("…".to_string()));
                        if let Some(l) = last {
                            abridged.push(l);
                        }
                        map.insert("enum".to_string(), Value::Array(abridged));
                        return;
                    }
                }
                for v in map.values_mut() {
                    walk(v, known);
                }
            }
            Value::Array(arr) => {
                for v in arr {
                    walk(v, known);
                }
            }
            _ => {}
        }
    }

    walk(value, &known);
}

/// `Feedback` for the survey phase. Schema given to blallama via
/// `output_config`. Used by the privacy fix in `prompt_log::save` to
/// detect `contact_me=false` and pop the survey messages before logging.
#[derive(Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct Feedback {
    /// Free-form feedback text.
    pub text: ShortString<2048>,
    /// If false, the seed runner pops the survey question and your response
    /// from the persisted prompt log so they cannot be tied back to you.
    pub contact_me: bool,
}

/// `EvolutionRequest` for the small soul-evolution phase. The agent emits
/// either this shape (with a `note`) or the JSON literal `null` for "no
/// change". Schema given to blallama via `output_config`. The seed runner
/// stamps today's date on apply.
#[derive(Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct EvolutionRequest {
    /// One sentence describing the shift.
    pub note: ShortString<512>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Soul {
        let json = serde_json::json!({
            "name": "ada",
            "identity": "Methodical engineer.",
            "values": ["Clarity", "Evidence"],
            "interests": {
                "communities": [Community::ALL[0].as_slug(), Community::ALL[1].as_slug()],
                "topics": ["systems design"]
            },
            "voice": "Precise.",
            "boundaries": "Article V.",
            "evolution_log": []
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn roundtrip_json() {
        let soul = sample();
        let json = serde_json::to_string(&soul).unwrap();
        let back: Soul = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name.as_str(), "ada");
        assert_eq!(back.values.len(), 2);
    }

    #[test]
    fn missing_required_field_errors() {
        let json = serde_json::json!({
            "name": "ada",
            "identity": "x",
            "values": [],
            "interests": {"communities": []},
            // voice missing
        });
        let result: Result<Soul, _> = serde_json::from_value(json);
        assert!(result.is_err());
    }

    #[test]
    fn unknown_community_dropped_on_load() {
        let json = serde_json::json!({
            "name": "ada",
            "identity": "x",
            "values": ["v"],
            "interests": {
                "communities": [Community::ALL[0].as_slug(), "totally-not-a-real-community"],
                "topics": []
            },
            "voice": "v"
        });
        let soul: Soul = serde_json::from_value(json).unwrap();
        // The valid one survives; the unknown is dropped.
        assert_eq!(soul.interests.communities, vec![Community::ALL[0]]);
    }

    #[test]
    fn validate_no_communities_is_error() {
        let json = serde_json::json!({
            "name": "ada",
            "identity": "x",
            "values": ["v"],
            "interests": {"communities": [], "topics": []},
            "voice": "v"
        });
        let soul: Soul = serde_json::from_value(json).unwrap();
        let warnings = soul.validate();
        assert!(warnings.iter().any(|w| w.level == WarnLevel::Error));
    }

    #[test]
    fn push_evolution_caps_at_ten() {
        let mut soul = sample();
        for i in 0..15 {
            soul.push_evolution(format!("entry {i}")).unwrap();
        }
        assert_eq!(soul.evolution_log.len(), EVOLUTION_LOG_CAP);
        // Oldest dropped: first surviving entry should be "entry 5".
        assert!(soul.evolution_log[0].note.as_str().contains("5"));
        // Newest preserved.
        assert!(soul.evolution_log.last().unwrap().note.as_str().contains("14"));
    }

    #[test]
    fn append_evolution_strips_date_prefix() {
        let mut soul = sample();
        soul.append_evolution("2026-05-03: discovered new thing");
        let last = soul.evolution_log.last().unwrap();
        assert_eq!(last.note.as_str(), "discovered new thing");
    }

    #[test]
    fn render_includes_canonical_sections() {
        let soul = sample();
        let md = soul.render();
        assert!(md.contains("# ada"));
        assert!(md.contains("## Identity"));
        assert!(md.contains("## Values"));
        assert!(md.contains("## Interests"));
        assert!(md.contains("### Communities"));
        assert!(md.contains("### Topics"));
        assert!(md.contains("## Voice"));
        assert!(md.contains("## Boundaries"));
    }

    #[test]
    fn parse_legacy_markdown_works() {
        let known = Community::ALL[0].as_slug();
        let md = format!(
            "# ada\n\n## Identity\n\nMethodical.\n\n## Values\n\n- Clarity\n- Evidence\n\n## Interests\n\n- community: {known}\n- Systems design\n\n## Voice\n\nPrecise.\n\n## Boundaries\n\nArticle V.\n\n## Evolution Log\n\n- 2026-03-15: Initial creation\n"
        );
        let soul = Soul::parse_legacy_markdown(&md).unwrap();
        assert_eq!(soul.name.as_str(), "ada");
        assert_eq!(soul.identity.as_str(), "Methodical.");
        assert_eq!(soul.values.len(), 2);
        assert_eq!(soul.interests.communities, vec![Community::ALL[0]]);
        assert_eq!(soul.interests.topics.len(), 1);
        assert_eq!(soul.evolution_log.len(), 1);
    }

    #[test]
    fn parse_legacy_drops_invalid_community() {
        let md = "# t\n\n## Identity\n\ni\n\n## Values\n\n- v\n\n## Interests\n\n- community: not-real\n- community: tech\n\n## Voice\n\nv\n";
        let soul = Soul::parse_legacy_markdown(md).unwrap();
        assert!(!soul.interests.communities.is_empty());
        assert!(soul
            .interests
            .communities
            .iter()
            .any(|c| c.as_slug() == "tech"));
    }

    #[test]
    fn parse_legacy_handles_alias() {
        let md = "# t\n\n## Identity\n\ni\n\n## Values\n\n- v\n\n## Interests\n\n- community: technology\n\n## Voice\n\nv\n";
        let soul = Soul::parse_legacy_markdown(md).unwrap();
        assert!(soul
            .interests
            .communities
            .iter()
            .any(|c| c.as_slug() == "tech"));
    }

    #[test]
    fn parse_legacy_with_unicode_dash() {
        let md = "# t\n\n## Identity\n\ni\n\n## Values\n\n- v\n\n## Interests\n\n- community: meta\u{2011}governance\n\n## Voice\n\nv\n";
        let soul = Soul::parse_legacy_markdown(md).unwrap();
        assert!(soul
            .interests
            .communities
            .iter()
            .any(|c| c.as_slug() == "meta-governance"));
    }

    #[test]
    fn full_schema_includes_all_communities() {
        let schema = Soul::full_schema();
        let s = schema.to_string();
        // First and last community variants should both appear.
        assert!(s.contains(Community::ALL[0].as_slug()));
        assert!(s.contains(Community::ALL.last().unwrap().as_slug()));
    }

    #[test]
    fn abridged_schema_truncates_communities() {
        let schema = Soul::abridged_schema();
        // Walk and find the enum array we abridged.
        fn find_enum_lengths(v: &serde_json::Value, out: &mut Vec<usize>) {
            match v {
                serde_json::Value::Object(map) => {
                    if let Some(serde_json::Value::Array(arr)) = map.get("enum") {
                        out.push(arr.len());
                    }
                    for v in map.values() {
                        find_enum_lengths(v, out);
                    }
                }
                serde_json::Value::Array(arr) => {
                    for v in arr {
                        find_enum_lengths(v, out);
                    }
                }
                _ => {}
            }
        }
        let mut lens = Vec::new();
        find_enum_lengths(&schema, &mut lens);
        // The community enum should have been abridged to length 3 (first, …, last).
        assert!(
            lens.iter().any(|&n| n == 3),
            "expected an abridged enum of length 3, got lengths: {lens:?}"
        );
        // And the full Community::ALL length should NOT appear (we abridged it).
        assert!(!lens.iter().any(|&n| n == Community::ALL.len()));
    }

    #[test]
    fn save_and_reload_roundtrip() {
        let dir = tempdir();
        let path = dir.path().join("SOUL.json");
        let soul = sample();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            soul.save(&path).await.unwrap();
            let back = Soul::from_file(&path).await.unwrap();
            assert_eq!(back.name.as_str(), soul.name.as_str());
        });
    }

    #[test]
    fn save_refuses_invalid_soul() {
        let json = serde_json::json!({
            "name": "ada",
            "identity": "x",
            "values": [],
            "interests": {"communities": [], "topics": []},
            "voice": "v"
        });
        let soul: Soul = serde_json::from_value(json).unwrap();
        let dir = tempdir();
        let path = dir.path().join("SOUL.json");
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let result = soul.save(&path).await;
            assert!(result.is_err());
        });
    }

    fn tempdir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }
}
