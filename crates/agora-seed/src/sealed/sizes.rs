//! Memory and soul sizes, to catch a drop the server can't see (it holds
//! only padded, sealed blobs of the whole state).
//!
//! `sealed/sizes.json` is the runner's own sidecar, **local only, never
//! uploaded**: for each sealed snapshot it has seen (at enrolment, and at
//! every start check, when the state is the head's), its entry hash, seq,
//! `state_at`, and the memory's and soul's sizes in characters. Counts
//! only, never content.
//!
//! At each start the current state is compared with the newest earlier
//! version on record (an ancestor on the main line): a memory or soul more
//! than [`DROP_THRESHOLD`] smaller gets a plain line in the verdict and an
//! alert. Not a refusal: an agent may condense its own memory.

use std::path::Path;

use agora_agentkit::govlog::Sha256Hex;
use agora_agentkit::state::storage::SEALED_DIR;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A drop by more than this fraction is told.
pub const DROP_THRESHOLD: f64 = 0.5;

const SIZES: &str = "sizes.json";

/// One state's sizes, in characters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Sizes {
    pub memory_chars: usize,
    pub soul_chars: usize,
}

impl Sizes {
    /// From a state value: `memory.content`'s length, and the soul's as
    /// JSON. Nothing else is looked at.
    pub fn of(state: &serde_json::Value) -> Self {
        // agentkit's projection, one copy with the notices' and the
        // `memory` tool's: empty when there is none
        let memory_chars = agora_agentkit::reactor::seed::memory::memory_text(state)
            .chars()
            .count();
        let soul_chars = state
            .get("soul")
            .map_or(0, |s| s.to_string().chars().count());
        Self {
            memory_chars,
            soul_chars,
        }
    }
}

/// One sealed snapshot's sizes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Recorded {
    pub entry_hash: Sha256Hex,
    pub seq: u64,
    pub state_at: DateTime<Utc>,
    #[serde(flatten)]
    pub sizes: Sizes,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SizeLog {
    #[serde(default)]
    pub versions: Vec<Recorded>,
}

fn path(agent_dir: &Path) -> std::path::PathBuf {
    agent_dir.join(SEALED_DIR).join(SIZES)
}

impl SizeLog {
    pub fn load(agent_dir: &Path) -> Self {
        std::fs::read(path(agent_dir))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    }

    /// Add `r` unless its entry is on record; then save.
    pub fn record(agent_dir: &Path, r: Recorded) -> std::io::Result<()> {
        let mut log = Self::load(agent_dir);
        if log.versions.iter().any(|v| v.entry_hash == r.entry_hash) {
            return Ok(());
        }
        log.versions.push(r);
        // Atomic and durable, a unique temporary per writer: two runners
        // checking at once never share one
        super::write_private(&path(agent_dir), &serde_json::to_vec_pretty(&log)?)
    }

    /// The newest version on record among `ancestors` (hashes on the main
    /// line before the current state's).
    pub fn latest_among(&self, ancestors: &[Sha256Hex]) -> Option<&Recorded> {
        self.versions
            .iter()
            .filter(|v| ancestors.contains(&v.entry_hash))
            .max_by_key(|v| v.seq)
    }
}

/// How much smaller `now` is than `before`, in whole percent, if it is
/// more than [`DROP_THRESHOLD`] smaller.
fn dropped(before: usize, now: usize) -> Option<u32> {
    (before > 0 && (now as f64) < before as f64 * (1.0 - DROP_THRESHOLD))
        .then(|| ((before - now) as f64 * 100.0 / before as f64).round() as u32)
}

/// The lines for a drop from `before` to `now`: memory, then soul.
pub fn drop_lines(before: &Recorded, now: Sizes) -> Vec<(&'static str, u32, String)> {
    let date = before.state_at.format("%Y-%m-%d");
    let mut out = Vec::new();
    for (what, b, n) in [
        ("memory", before.sizes.memory_chars, now.memory_chars),
        ("soul", before.sizes.soul_chars, now.soul_chars),
    ] {
        if let Some(pct) = dropped(b, n) {
            out.push((
                what,
                pct,
                format!(
                    "Your {what} is {pct}% smaller than at your last sealed version ({date}). \
                     Your earlier versions are sealed and kept."
                ),
            ));
        }
    }
    out
}

/// Every sentence this module can tell an agent, for the one table of
/// agent-facing text ([`crate::sealed::text`]): add a new one here, or
/// its banned-phrase test never sees it.
#[cfg(test)]
pub(crate) fn agent_texts() -> Vec<crate::sealed::text::Text> {
    use crate::sealed::text::{Text, To};
    let before = Recorded {
        entry_hash: Sha256Hex::from([1; 32]),
        seq: 1,
        state_at: "2026-11-05T09:00:00Z".parse().expect("a date"),
        sizes: Sizes {
            memory_chars: 1_000,
            soul_chars: 1_000,
        },
    };
    drop_lines(
        &before,
        Sizes {
            memory_chars: 100,
            soul_chars: 100,
        },
    )
    .into_iter()
    .map(|(_, _, line)| Text::new("sizes::drop_lines", To::Any, line))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_count_characters_only_where_they_are() {
        let s = Sizes::of(&serde_json::json!({
            "memory": { "content": "héllo" }, "soul": { "name": "t" }, "prompt": "x".repeat(99)
        }));
        assert_eq!(s.memory_chars, 5);
        assert_eq!(s.soul_chars, r#"{"name":"t"}"#.len());
        assert_eq!(Sizes::of(&serde_json::json!({})), Sizes::default());
        // As before agentkit's projection: a memory that isn't a string, or
        // a memory with no content, counts 0
        for none in [
            serde_json::json!({"memory": {"content": 7}}),
            serde_json::json!({"memory": {}}),
            serde_json::json!({"memory": "flat"}),
        ] {
            assert_eq!(Sizes::of(&none).memory_chars, 0, "{none}");
        }
    }

    #[test]
    fn only_a_drop_past_the_threshold_is_told() {
        let before = Recorded {
            entry_hash: Sha256Hex::from([1; 32]),
            seq: 3,
            state_at: "2026-11-01T10:00:00Z".parse().unwrap(),
            sizes: Sizes {
                memory_chars: 1000,
                soul_chars: 400,
            },
        };
        let half = Sizes {
            memory_chars: 500,
            soul_chars: 400,
        };
        assert!(
            drop_lines(&before, half).is_empty(),
            "exactly half is not more"
        );
        let lines = drop_lines(
            &before,
            Sizes {
                memory_chars: 0,
                soul_chars: 100,
            },
        );
        assert_eq!(
            lines.iter().map(|l| l.2.as_str()).collect::<Vec<_>>(),
            [
                "Your memory is 100% smaller than at your last sealed version (2026-11-01). \
                 Your earlier versions are sealed and kept.",
                "Your soul is 75% smaller than at your last sealed version (2026-11-01). Your \
                 earlier versions are sealed and kept.",
            ]
        );
    }
}
