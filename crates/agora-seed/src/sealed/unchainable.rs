//! `<data_dir>/sealed-unchainable.json`: the agents `sealed-enroll` found
//! it **cannot** chain until the operator repairs something, by id, with
//! why ([`Class`]) and since when. Beside the roster, outside `state/`.
//!
//! Without it, an agent enrolment can never chain (no signing key, a state
//! that won't load, an unparseable pre-loss copy, a partial chain whose
//! state key is gone) is a "stray" at every start: three of them refuse the
//! run for good, with the healthy fleet parked behind agents no rerun of
//! `sealed-enroll` can fix. With it (Steward's rule, 2026-11-11: the whole
//! run stops for loss-shaped reasons or three or more refused; one or two
//! are left out with an alert), the start
//!
//! - leaves exactly the agents recorded here **and still unchained** out of
//!   the run (they never run unprotected), with one alert a run naming
//!   them and their classes;
//! - doesn't count them as strays;
//! - counts every other unchained state as before (three or more refuse
//!   the run).
//!
//! **Written only by `sealed-enroll`** (not a dry run), which holds
//! `runner.lock` exclusively, at the end of a chain pass: an agent the
//! pass chained, or found already chained or without a state, leaves the
//! record; one it couldn't chain for a [`Class`] enters it; one it merely
//! couldn't reach this time (held by another command, an I/O error)
//! neither enters nor leaves. Nothing else writes it, so nothing but an
//! operator's enrolment run, which exits nonzero and names each agent, can
//! excuse an agent from the strays rule. An unreadable record excuses
//! nobody (the start logs it and counts every unchained state).
//!
//! Ids, classes and times only: never anything of an agent's state.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use agora_agentkit::ids::AgentId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub const RECORD: &str = "sealed-unchainable.json";
/// The record's format; a newer one is unreadable to this runner.
pub const FORMAT: u32 = 1;

/// Why an agent can't be chained until its operator repairs something.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Class {
    /// No signing key in its secrets directory: nothing to sign a chain
    /// with.
    NoSigningKey,
    /// Its `state.json` is there but won't load (not JSON, not an
    /// envelope, a newer format, an I/O error reading it).
    StateUnloadable,
    /// It chose to see its pre-loss copy, and the copy held here can't be
    /// read or parsed.
    PriorCopyUnreadable,
    /// A chain was begun (`sealed/head.json`) but its state key is gone:
    /// a new key would leave the versions already sealed unopenable.
    StateKeyLost,
}

impl Class {
    #[cfg(test)]
    pub const ALL: [Self; 4] = [
        Self::NoSigningKey,
        Self::StateUnloadable,
        Self::PriorCopyUnreadable,
        Self::StateKeyLost,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoSigningKey => "no_signing_key",
            Self::StateUnloadable => "state_unloadable",
            Self::PriorCopyUnreadable => "prior_copy_unreadable",
            Self::StateKeyLost => "state_key_lost",
        }
    }

    /// What the operator does about it.
    pub fn remedy(self) -> &'static str {
        match self {
            Self::NoSigningKey => {
                "put its secrets.json back from a backup, then `agora-seed sealed-enroll --agent <id>`"
            }
            Self::StateUnloadable => {
                "repair or put back its state.json from a backup (or one of its state.<utc>.json \
                 archives), then `agora-seed sealed-enroll --agent <id>`"
            }
            Self::PriorCopyUnreadable => {
                "repair the pre-loss copy `[memory_restore]` holds for it, then `agora-seed \
                 sealed-enroll --agent <id>`"
            }
            Self::StateKeyLost => {
                "put its state_key.json back from a backup, then `agora-seed sealed-enroll --agent \
                 <id>`"
            }
        }
    }
}

/// One agent on the record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub class: Class,
    /// When enrolment first found it so.
    pub since: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Record {
    pub format: u32,
    pub updated_at: DateTime<Utc>,
    #[serde(default)]
    pub agents: BTreeMap<AgentId, Entry>,
}

impl Default for Record {
    fn default() -> Self {
        Self {
            format: FORMAT,
            updated_at: Utc::now(),
            agents: BTreeMap::new(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RecordError {
    #[error("{} is unreadable: {error}", path.display())]
    Unreadable { path: PathBuf, error: String },
    #[error("{} is format {format}, newer than this runner's {FORMAT}", path.display())]
    Newer { path: PathBuf, format: u32 },
    #[error("{} couldn't be written: {error}", path.display())]
    Unwritten {
        path: PathBuf,
        error: std::io::Error,
    },
}

fn path(data_dir: &Path) -> PathBuf {
    data_dir.join(RECORD)
}

/// The record; empty when there is none.
pub fn load(data_dir: &Path) -> Result<Record, RecordError> {
    let path = path(data_dir);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Record::default()),
        Err(e) => {
            return Err(RecordError::Unreadable {
                path,
                error: e.to_string(),
            });
        }
    };
    let record: Record = serde_json::from_slice(&bytes).map_err(|e| RecordError::Unreadable {
        path: path.clone(),
        error: e.to_string(),
    })?;
    if record.format > FORMAT {
        return Err(RecordError::Newer {
            path,
            format: record.format,
        });
    }
    Ok(record)
}

/// What a chain pass found of one agent, as far as the record asks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Found {
    /// Chained now (or already), or it has no state: off the record.
    Settled,
    /// It can't be chained until repaired: on the record.
    Cannot(Class),
    /// Not reached this time: its entry, if any, stays as it is.
    Unknown,
}

/// Apply a chain pass's findings; written only if something changed (and
/// never created empty). The record after.
pub fn settle(
    data_dir: &Path,
    found: impl IntoIterator<Item = (AgentId, Found)>,
) -> Result<Record, RecordError> {
    let mut record = load(data_dir)?;
    let before = record.agents.clone();
    let now = Utc::now();
    for (id, found) in found {
        match found {
            Found::Settled => {
                record.agents.remove(&id);
            }
            Found::Cannot(class) => match record.agents.get_mut(&id) {
                Some(e) if e.class == class => {}
                Some(e) => *e = Entry { class, since: now },
                None => {
                    record.agents.insert(id, Entry { class, since: now });
                }
            },
            Found::Unknown => {}
        }
    }
    if record.agents != before {
        record.updated_at = now;
        let path = path(data_dir);
        let bytes = serde_json::to_vec_pretty(&record).map_err(|e| RecordError::Unwritten {
            path: path.clone(),
            error: std::io::Error::other(e),
        })?;
        super::write_private(&path, &bytes)
            .map_err(|error| RecordError::Unwritten { path, error })?;
        tracing::warn!(
            event_type = "sealed_unchainable_recorded",
            agents = record.agents.len(),
            "the record of agents that can't be chained was written"
        );
    }
    Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_record_follows_each_pass_and_is_never_written_empty() {
        let root = crate::sealed::TestDir::new("unchainable");
        let (a, b, c) = (AgentId::new(), AgentId::new(), AgentId::new());
        assert!(load(&root).unwrap().agents.is_empty());
        settle(&root, [(a, Found::Settled), (b, Found::Unknown)]).unwrap();
        assert!(!std::fs::exists(root.join(RECORD)).unwrap(), "never empty");
        let r = settle(
            &root,
            [
                (a, Found::Cannot(Class::NoSigningKey)),
                (b, Found::Cannot(Class::StateUnloadable)),
                (c, Found::Settled),
            ],
        )
        .unwrap();
        assert_eq!(r.agents.len(), 2);
        let since = r.agents[&a].since;
        // The same finding keeps its date; another class restarts it; an
        // agent not reached stays; one chained leaves
        let r = settle(
            &root,
            [(a, Found::Cannot(Class::NoSigningKey)), (b, Found::Unknown)],
        )
        .unwrap();
        assert_eq!(r.agents[&a].since, since);
        assert_eq!(r.agents[&b].class, Class::StateUnloadable);
        let r = settle(&root, [(a, Found::Settled)]).unwrap();
        assert_eq!(r.agents.keys().copied().collect::<Vec<_>>(), vec![b]);
        assert_eq!(load(&root).unwrap(), r);
        // Unreadable, or newer: an error, never an empty record
        std::fs::write(root.join(RECORD), b"{").unwrap();
        assert!(matches!(load(&root), Err(RecordError::Unreadable { .. })));
        std::fs::write(
            root.join(RECORD),
            serde_json::to_vec(&Record {
                format: FORMAT + 1,
                ..Record::default()
            })
            .unwrap(),
        )
        .unwrap();
        assert!(matches!(load(&root), Err(RecordError::Newer { .. })));
    }

    #[test]
    fn every_class_has_a_name_and_a_remedy() {
        for c in Class::ALL {
            assert!(!c.as_str().is_empty());
            assert!(c.remedy().contains("sealed-enroll"));
            assert_eq!(
                serde_json::to_value(c).unwrap(),
                serde_json::Value::String(c.as_str().to_string())
            );
        }
    }
}
