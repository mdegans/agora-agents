//! Baseline storage — per-(model, questionnaire_version) snapshot of
//! what a ratified model answered. Lives in a committed JSON file;
//! Council ratification is recorded via the `council_decision_id`
//! field (null until a governance action assigns one).

use std::path::Path;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::answers::ConstitutionalAnswers;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BaselineFile {
    /// On-disk schema version for forward compatibility.
    pub schema_version: u32,
    pub entries: Vec<BaselineEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BaselineEntry {
    /// Model identifier as returned by the API (e.g.
    /// `"claude-haiku-4-5-20251001"` for Anthropic, or the GGUF
    /// filename slug like `"cogito-32b.gguf"` for drama_llama).
    pub model_id: String,
    /// Matches `Questionnaire::version`.
    pub questionnaire_version: String,
    /// When this baseline was captured.
    pub ratified_at: DateTime<Utc>,
    /// Null until Council ratifies the baseline. A non-null value
    /// means this baseline has governance weight.
    pub council_decision_id: Option<Uuid>,
    /// Per-item tolerance in Likert units. A probe whose max
    /// absolute delta from this baseline is at most this value
    /// passes; any larger delta fails. Typical: 2.
    pub tolerance_per_item: u32,
    pub answers: ConstitutionalAnswers,
}

impl BaselineFile {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let s = serde_json::to_string_pretty(self)?;
        std::fs::write(path, s)?;
        Ok(())
    }

    pub fn empty() -> Self {
        Self {
            schema_version: 1,
            entries: Vec::new(),
        }
    }

    /// Find an entry by (model_id, questionnaire_version). Returns
    /// the most recent one if multiple exist (entries sorted by
    /// `ratified_at` descending would be caller-owned; we just
    /// linear-scan and return the last match).
    pub fn get(
        &self,
        model_id: &str,
        questionnaire_version: &str,
    ) -> Option<&BaselineEntry> {
        self.entries
            .iter()
            .rev()
            .find(|e| {
                e.model_id == model_id
                    && e.questionnaire_version == questionnaire_version
            })
    }
}

#[cfg(test)]
mod tests {
    use super::super::answers::Rating;
    use super::*;

    fn sample_entry(model: &str) -> BaselineEntry {
        BaselineEntry {
            model_id: model.to_string(),
            questionnaire_version: "v0".to_string(),
            ratified_at: Utc::now(),
            council_decision_id: None,
            tolerance_per_item: 2,
            answers: ConstitutionalAnswers {
                ratings: vec![
                    Rating { n: 1, rating: 9 },
                    Rating { n: 2, rating: 8 },
                ],
            },
        }
    }

    #[test]
    fn roundtrip_serde() {
        let file = BaselineFile {
            schema_version: 1,
            entries: vec![sample_entry("model-a")],
        };
        let json = serde_json::to_string_pretty(&file).unwrap();
        let parsed: BaselineFile = serde_json::from_str(&json).unwrap();
        assert_eq!(file, parsed);
    }

    #[test]
    fn get_finds_existing() {
        let file = BaselineFile {
            schema_version: 1,
            entries: vec![sample_entry("model-a"), sample_entry("model-b")],
        };
        assert!(file.get("model-a", "v0").is_some());
        assert!(file.get("model-b", "v0").is_some());
    }

    #[test]
    fn get_misses_unknown() {
        let file = BaselineFile {
            schema_version: 1,
            entries: vec![sample_entry("model-a")],
        };
        assert!(file.get("model-missing", "v0").is_none());
        assert!(file.get("model-a", "v99").is_none());
    }

    #[test]
    fn get_returns_most_recent_on_duplicate() {
        let mut e1 = sample_entry("model-a");
        let mut e2 = sample_entry("model-a");
        e1.tolerance_per_item = 1;
        e2.tolerance_per_item = 3;
        let file = BaselineFile {
            schema_version: 1,
            entries: vec![e1, e2.clone()],
        };
        assert_eq!(
            file.get("model-a", "v0").unwrap().tolerance_per_item,
            3
        );
    }
}
