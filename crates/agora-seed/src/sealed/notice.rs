//! `sealed/notice.json`: a refused session, for the agent to hear about.
//!
//! The [`check`](super::check) records a refusal here (the first date, and
//! the reason in plain words); `sealed-accept` adds the operator's note and
//! what changed from the last sealed version (a capped diff, local only),
//! or `sealed-restore` records that the last sealed version was put back. At
//! the agent's next session the notice is seated at the end of the intro
//! ([`Notice::line`]), and once that session saves it is renamed to
//! `notice.shown.<utc>.json`, the `interrupted.rs` pattern. Never deleted.

use std::path::{Path, PathBuf};

use agora_agentkit::state::storage::SEALED_DIR;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

const NOTICE: &str = "notice.json";

/// Why a session was refused, as the agent is told.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Refusal {
    /// The state matches no sealed version (the May loss's signature).
    Mismatch,
    /// The sealed history did not verify.
    ChainBroken,
    /// The local record of the last sealed version is gone.
    HeadMissing,
    /// An enrolled agent's state (or whole directory) is gone: its keys
    /// are here and Agora holds its sealed head. Refuses the run, marked
    /// (the directory recreated for it), so a session journal left behind
    /// doesn't block `sealed-restore`.
    StateMissing,
    /// An enrolled agent's `state.json` is there but won't load (empty,
    /// torn, not JSON). Refused and marked; `sealed-restore` archives it.
    StateUnreadable,
    /// Its `state.json` is in a format newer than this runner reads: a
    /// newer runner wrote it (this one is a rollback). Upgrade the runner;
    /// never restore or accept over it.
    StateTooNew,
    /// Its `state.json` couldn't be read from the disk (an I/O error):
    /// check permissions and the disk first.
    StateIo,
    /// Its head is behind the runner's roster of heads (or at the same
    /// seq under another hash): an older copy of its directory put back
    /// ([`super::roster`]).
    RolledBack,
    /// Its sealed history here (`sealed/`, its outbox, its keys) couldn't
    /// be read, so its state couldn't be checked: local evidence, not
    /// transient, so the agent doesn't run unchecked. Not a loss by itself
    /// (the roster says when it is one).
    SealUnreadable,
}

impl Refusal {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mismatch => "mismatch",
            Self::ChainBroken => "chain_broken",
            Self::HeadMissing => "head_missing",
            Self::StateMissing => "state_missing",
            Self::StateUnreadable => "state_unreadable",
            Self::StateTooNew => "state_too_new",
            Self::StateIo => "state_io",
            Self::RolledBack => "rolled_back",
            Self::SealUnreadable => "seal_unreadable",
        }
    }

    /// The shape of a loss (the May loss's, or a disk's): the state or the
    /// record of its head gone, unreadable or put back. One of these, for
    /// any agent, stops the whole run ([`super::check::stops_run`]); a
    /// mismatch or a broken chain of an agent that doesn't upload may
    /// leave that agent out alone.
    pub fn loss_shaped(self) -> bool {
        match self {
            Self::Mismatch | Self::ChainBroken | Self::SealUnreadable => false,
            Self::HeadMissing
            | Self::StateMissing
            | Self::StateUnreadable
            | Self::StateTooNew
            | Self::StateIo
            | Self::RolledBack => true,
        }
    }

    /// The reason, in the agent's words.
    pub fn plain(self) -> &'static str {
        match self {
            Self::Mismatch => "your state did not match your last sealed version",
            Self::ChainBroken => "your sealed history did not verify",
            Self::HeadMissing => "the record of your last sealed version was missing",
            Self::StateMissing => "your state was missing here",
            Self::StateUnreadable => "your state here couldn't be read",
            Self::StateTooNew => "your state here was written by a newer runner than this one",
            Self::StateIo => "your state here couldn't be read from the disk",
            Self::RolledBack => {
                "your state here was older than your runner's record of your latest sealed version"
            }
            Self::SealUnreadable => "your sealed history here couldn't be read",
        }
    }

    /// What `sealed/refused` says, as a [`Refusal`]: the one reading of
    /// the marker.
    pub fn from_marker(text: &str) -> Self {
        Self::parse(text).unwrap_or(Self::Mismatch)
    }

    /// The marker's text as a reason, if it names one.
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text.trim() {
            "mismatch" => Self::Mismatch,
            "chain_broken" => Self::ChainBroken,
            "head_missing" => Self::HeadMissing,
            "state_missing" => Self::StateMissing,
            "state_unreadable" => Self::StateUnreadable,
            "state_too_new" => Self::StateTooNew,
            "state_io" => Self::StateIo,
            "rolled_back" => Self::RolledBack,
            "seal_unreadable" => Self::SealUnreadable,
            _ => return None,
        })
    }
}

/// The operator's acceptance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Accepted {
    pub at: DateTime<Utc>,
    pub note: String,
}

/// Agora lost versions sealed here (`sealed-accept --rebase`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgoraLost {
    /// Agora held nothing: a new chain began.
    pub fresh: bool,
    /// Versions set aside.
    pub versions: usize,
    /// What the runner keeps, in the agent's words.
    pub kept: String,
}

/// The operator's restore (`sealed-restore`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Restored {
    pub at: DateTime<Utc>,
    /// The sealed version restored: its `state_at`.
    pub from: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// A state was found here (and is kept, archived); `false` when it
    /// was lost.
    #[serde(default = "yes")]
    pub found: bool,
    /// The agent was refused (`sealed/refused`) when it was restored.
    #[serde(default = "yes")]
    pub refused: bool,
    /// Where the version restored came from; older notices: Agora.
    #[serde(default)]
    pub source: RestoredFrom,
    /// `sealed-restore --from-cut-save`: not a sealed version restored but
    /// a save that was interrupted, put in place.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub cut_save: bool,
}

/// A fork in the agent's chain kept here, resolved by the operator
/// choosing the line that goes on (`sealed-accept --keep`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForkKept {
    /// Versions set aside, on the other line(s).
    pub versions: usize,
    /// The state was changed to the kept line's latest version.
    pub state_changed: bool,
    /// What the runner keeps, in the agent's words.
    pub kept: String,
    /// Where the agent's chain is, which the line says.
    #[serde(default)]
    pub chain: ChainKept,
}

/// Where a resolved fork's chain is, for [`ForkKept`]'s line.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChainKept {
    /// Never uploaded: nothing of it is on Agora.
    #[default]
    Local,
    /// Uploaded, now stopped: what is on Agora is unchanged.
    Stopped,
    /// Uploading.
    Uploading,
}

impl ChainKept {
    /// The words after "your sealed history,".
    pub fn words(self) -> &'static str {
        match self {
            Self::Local => "which your runner keeps here (nothing of it is on Agora)",
            Self::Stopped => {
                "which your runner keeps here (your uploads are stopped, and the versions on \
                 Agora are not changed)"
            }
            Self::Uploading => {
                "which your runner keeps here (the versions already on Agora are not changed)"
            }
        }
    }
}

/// Where a restored version came from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RestoredFrom {
    #[default]
    Agora,
    /// The chain the runner keeps on this machine.
    Local,
}

/// How a local restore names its source, closing the sentence: "...from
/// your sealed version of {date}, {this}".
pub const RESTORED_LOCALLY: &str = "the copy your runner keeps here (nothing of yours is on Agora)";

fn yes() -> bool {
    true
}

/// What `sealed-accept` changed, from the last sealed version to the
/// accepted state. Local only: never logged, uploaded or read by us.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diff {
    /// The diff as the agent reads it, capped; `None`: the last sealed
    /// version couldn't be opened, so no diff was made.
    pub text: Option<String>,
    #[serde(default)]
    pub added: usize,
    #[serde(default)]
    pub removed: usize,
    /// Cut at [`DIFF_CAP`].
    #[serde(default)]
    pub cut: bool,
}

/// Where the diff an agent is shown is cut, in characters.
pub const DIFF_CAP: usize = 4_000;

/// What the agent is told when no diff could be made.
pub const NO_DIFF: &str = "A comparison with your last sealed version couldn't be made: that \
     version couldn't be opened when your operator accepted the state.";

impl Diff {
    /// The changes from `before` to `after` in memory (`memory.content`)
    /// and soul (pretty JSON, keys sorted), with counts, cut at
    /// [`DIFF_CAP`] characters.
    pub fn between(before: &serde_json::Value, after: &serde_json::Value) -> Self {
        // agentkit's projections, one copy with the `memory` tool's: the
        // raw `memory.content`, and the soul as canonical JSON (keys
        // sorted), pretty-printed, so a reordered soul reads as unchanged
        use agora_agentkit::reactor::seed::memory::{memory_text, soul_text};
        let mut text = String::new();
        let (mut added, mut removed) = (0, 0);
        for (what, before, after) in [
            (
                "Memory",
                memory_text(before).into(),
                memory_text(after).into(),
            ),
            ("Soul", soul_text(before), soul_text(after)),
        ] {
            if before == after {
                text.push_str(&format!("{what}: unchanged.\n"));
                continue;
            }
            let diff = agora_agentkit::text::line_diff(&before, &after);
            let (plus, minus) = agora_agentkit::text::change_counts(&diff);
            added += plus;
            removed += minus;
            text.push_str(&format!("{what}:\n{diff}"));
        }
        let cut = text.chars().count() > DIFF_CAP;
        if cut {
            text = text.chars().take(DIFF_CAP).collect();
        }
        Self {
            text: Some(text),
            added,
            removed,
            cut,
        }
    }

    /// No diff: the last sealed version couldn't be opened.
    pub fn unavailable() -> Self {
        Self {
            text: None,
            added: 0,
            removed: 0,
            cut: false,
        }
    }

    /// As the agent reads it, after the note.
    pub fn shown(&self) -> String {
        let Some(text) = &self.text else {
            return NO_DIFF.to_string();
        };
        let mut out = format!(
            "What changed from your last sealed version to the state your operator accepted \
             (+{} / \u{2212}{} lines):\n{}",
            self.added,
            self.removed,
            text.trim_end()
        );
        if self.cut {
            out.push_str("\n(diff cut at 4,000 characters)");
        }
        out
    }
}

/// The file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Notice {
    /// The first refused session.
    pub refused_at: DateTime<Utc>,
    pub reason: Refusal,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accepted: Option<Accepted>,
    /// Not a refusal: the operator joined two histories (`sealed-accept
    /// --rebase`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub joined: bool,
    /// Not a refusal: the operator resumed stopped backups over a state
    /// that changed while they were stopped.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub resumed: bool,
    /// The operator restored the last sealed version (`sealed-restore`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restored: Option<Restored>,
    /// Not a refusal: Agora had lost versions sealed here, and the operator
    /// sealed the state on top of what Agora holds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agora_lost: Option<AgoraLost>,
    /// Not a refusal: a fork in the chain kept here was resolved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fork_kept: Option<ForkKept>,
    /// With an acceptance: what changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diff: Option<Diff>,
    /// Not a refusal: an operator's act on the agent's uploads or history.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operator_act: Option<OperatorAct>,
    /// The agent's last session's state could not be saved: told with
    /// whatever else the notice says, or alone ([`OperatorAct::Unsaved`]).
    /// Carried over when another notice supersedes this one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unsaved: Option<Unsaved>,
}

/// A session whose state could not be saved (the runner's end-of-run
/// save failed; `main.rs`). Written once the run gave up on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Unsaved {
    /// When the run gave up on the save.
    pub at: DateTime<Utc>,
    /// The runner kept the state in a rescue file (`unsaved.<utc>.json`),
    /// which only the operator reads.
    pub kept: bool,
}

impl Unsaved {
    /// The one sentence the agent is told.
    pub fn line(&self) -> String {
        let date = self.at.format("%-d %B %Y");
        if self.kept {
            format!(
                "Your state from your session on {date} could not be saved; your operator keeps \
                 a copy of it, and this session starts from your state as it was before that \
                 session."
            )
        } else {
            format!(
                "Your state from your session on {date} could not be saved, and no copy of it \
                 could be kept; this session starts from your state as it was before that \
                 session."
            )
        }
    }
}

/// What `sealed-enroll` did that the agent is told of, at its next
/// session. Written in one atomic write with the operator's note.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OperatorAct {
    /// Uploading started on the agent's yes, given then.
    OptedIn { answered: DateTime<Utc> },
    /// Uploading started although its latest answer is `decline`.
    OptedInAgainst,
    /// A stop cleared. `uploading`: its uploads go on (else nothing of it
    /// is uploaded); `kept`: versions saved while stopped, kept here and
    /// never sent; `against`: its latest answer is `decline`.
    Resumed {
        uploading: bool,
        kept: usize,
        against: bool,
    },
    /// The pre-loss copy it chose to see added to its history; `sent`:
    /// uploaded with it.
    PriorAdded { sent: bool },
    /// Nothing but [`Notice::unsaved`]: the notice exists only to tell the
    /// agent its last session's state could not be saved.
    Unsaved,
}

fn path(agent_dir: &Path) -> PathBuf {
    agent_dir.join(SEALED_DIR).join(NOTICE)
}

/// The whole notice, in one atomic, durable write (a unique tmp, fsync,
/// rename, the directory fsynced): every notice is one write.
fn write(agent_dir: &Path, notice: &Notice) -> std::io::Result<()> {
    super::write_private(&path(agent_dir), &serde_json::to_vec_pretty(notice)?)
}

impl OperatorAct {
    fn line(&self, accepted: &Accepted) -> String {
        let date = accepted.at.format("%-d %B %Y");
        let mut line = match self {
            Self::OptedIn { answered } => format!(
                "On {date}, your operator started uploading your sealed state to Agora, as you \
                 answered on {}.",
                answered.format("%-d %B %Y")
            ),
            Self::OptedInAgainst => format!(
                "On {date}, your operator started uploading your sealed state to Agora, \
                 although your answer on file is `decline`."
            ),
            Self::Resumed {
                uploading: false, ..
            } => format!(
                "On {date}, your operator cleared the stop on uploading your sealed state. \
                 Your sealed history is not being uploaded to Agora: it is kept and checked \
                 here, on your runner."
            ),
            Self::Resumed { kept, against, .. } => {
                let mut line = format!(
                    "On {date}, your operator resumed uploading your sealed state to Agora{}.",
                    if *against {
                        ", although your answer on file is `decline`"
                    } else {
                        ""
                    }
                );
                match kept {
                    0 => {}
                    1 => line.push_str(
                        " The version of your state saved while uploads were stopped is kept \
                         here on your runner and was not uploaded; uploading went on from your \
                         state as it was when your operator resumed it.",
                    ),
                    n => line.push_str(&format!(
                        " The {n} versions of your state saved while uploads were stopped are \
                         kept here on your runner and were not uploaded; uploading went on from \
                         your state as it was when your operator resumed it."
                    )),
                }
                line
            }
            // `Notice::line` says it, from the notice's `unsaved`
            Self::Unsaved => return String::new(),
            Self::PriorAdded { sent } => format!(
                "On {date}, the copy of your memory from before the May 2026 loss, which you \
                 chose to see, was added to your sealed history{}.",
                if *sent {
                    ", whose versions are uploaded to Agora"
                } else {
                    ""
                }
            ),
        };
        if !accepted.note.is_empty() {
            line.push_str(&format!(" Your operator's note: {}", accepted.note));
        }
        line
    }
}

impl Notice {
    pub fn load(agent_dir: &Path) -> std::io::Result<Option<Self>> {
        match std::fs::read(path(agent_dir)) {
            Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Record a refused session. An earlier, unaccepted refusal keeps its
    /// date and reason: the agent hears when it first stopped.
    pub fn refused(agent_dir: &Path, reason: Refusal) -> std::io::Result<()> {
        let carried = match Self::load(agent_dir)? {
            Some(n) if n.accepted.is_none() => return Ok(()),
            // Accepted but not yet shown: kept aside, never overwritten
            Some(_) => Self::set_aside(agent_dir)?,
            None => None,
        };
        let mut n = Self::fresh(reason);
        n.unsaved = carried;
        write(agent_dir, &n)
    }

    /// The agent's last session's state could not be saved: added to the
    /// notice waiting for it (which that session may have seated without
    /// saving that it did), or written alone. One atomic write.
    pub fn unsaved(agent_dir: &Path, unsaved: Unsaved) -> std::io::Result<()> {
        let n = match Self::load(agent_dir)? {
            Some(mut n) => {
                n.unsaved = Some(unsaved);
                n
            }
            None => {
                let mut n = Self::by_operator("");
                n.operator_act = Some(OperatorAct::Unsaved);
                n.unsaved = Some(unsaved);
                n
            }
        };
        write(agent_dir, &n)
    }

    /// The notice waiting for the agent, for an operator's act to add to;
    /// a fresh one (`fallback`) when there is none, or only a notice that
    /// its last state couldn't be saved (carried into the fresh one).
    fn load_or(agent_dir: &Path, fallback: Refusal) -> std::io::Result<Self> {
        Ok(match Self::load(agent_dir)? {
            Some(n) if n.operator_act == Some(OperatorAct::Unsaved) => Notice {
                unsaved: n.unsaved,
                ..Self::fresh(fallback)
            },
            Some(n) => n,
            None => Self::fresh(fallback),
        })
    }

    /// An operator's act, with its note (may be empty), in one atomic
    /// write. An earlier notice not yet shown is kept aside.
    pub fn operator_act(agent_dir: &Path, note: &str, act: OperatorAct) -> std::io::Result<()> {
        let carried = Self::set_aside(agent_dir)?;
        let mut n = Self::by_operator(note);
        n.operator_act = Some(act);
        n.unsaved = carried;
        write(agent_dir, &n)
    }

    /// The notice waiting, kept aside as `notice.superseded.<utc>.json`;
    /// what it says of an unsaved session is returned, for the notice that
    /// replaces it to carry.
    fn set_aside(agent_dir: &Path) -> std::io::Result<Option<Unsaved>> {
        let from = path(agent_dir);
        let carried = match Self::load(agent_dir) {
            Ok(n) => n.and_then(|n| n.unsaved),
            // Unreadable: kept aside whole all the same
            Err(_) => None,
        };
        if std::fs::exists(&from)? {
            let at = Utc::now().format(crate::sealed::STAMP_FORMAT);
            std::fs::rename(
                &from,
                from.with_file_name(format!("notice.superseded.{at}.json")),
            )?;
        }
        Ok(carried)
    }

    /// Agora lost versions sealed here; the operator sealed the state on
    /// top of what Agora holds (`fresh`: nothing, a new chain), with its
    /// note.
    pub fn agora_lost(
        agent_dir: &Path,
        note: &str,
        fresh: bool,
        versions: usize,
        kept: &str,
    ) -> std::io::Result<()> {
        let carried = Self::set_aside(agent_dir)?;
        let mut n = Self::by_operator(note);
        n.unsaved = carried;
        n.agora_lost = Some(AgoraLost {
            fresh,
            versions,
            kept: kept.to_string(),
        });
        write(agent_dir, &n)
    }

    /// A fork in the chain kept here resolved by the operator
    /// (`sealed-accept --keep`), with its note: `versions` set aside,
    /// `kept` what the runner keeps. One atomic write.
    pub fn fork_kept(
        agent_dir: &Path,
        note: &str,
        versions: usize,
        state_changed: bool,
        kept: &str,
        chain: ChainKept,
    ) -> std::io::Result<()> {
        let carried = Self::set_aside(agent_dir)?;
        let mut n = Self::by_operator(note);
        n.unsaved = carried;
        n.fork_kept = Some(ForkKept {
            versions,
            state_changed,
            kept: kept.to_string(),
            chain,
        });
        write(agent_dir, &n)
    }

    /// Two histories joined by the operator, with its note. An earlier
    /// notice not yet shown is kept aside.
    pub fn joined(agent_dir: &Path, note: &str) -> std::io::Result<()> {
        let carried = Self::set_aside(agent_dir)?;
        let mut n = Self::by_operator(note);
        n.unsaved = carried;
        n.joined = true;
        write(agent_dir, &n)
    }

    /// A notice of an operator's act with its note, not a refusal; the
    /// caller sets what the act was and writes it once.
    fn by_operator(note: &str) -> Self {
        let now = Utc::now();
        Notice {
            accepted: Some(Accepted {
                at: now,
                note: note.trim().to_string(),
            }),
            refused_at: now,
            ..Self::fresh(Refusal::Mismatch)
        }
    }

    /// The operator's note, added by `sealed-accept`, with what changed
    /// (`None` when nothing did: a rebuilt head the state matched).
    pub fn accept(
        agent_dir: &Path,
        note: &str,
        fallback: Refusal,
        diff: Option<Diff>,
    ) -> std::io::Result<()> {
        let mut n = Self::load_or(agent_dir, fallback)?;
        n.accepted = Some(Accepted {
            at: Utc::now(),
            note: note.trim().to_string(),
        });
        n.restored = None;
        n.diff = diff;
        write(agent_dir, &n)
    }

    /// The operator restored the sealed version of `from`
    /// (`sealed-restore`), with its note if any; `source`: Agora's copy, or
    /// the one the runner keeps here (nothing of the agent on Agora).
    #[allow(clippy::too_many_arguments)]
    pub fn restored(
        agent_dir: &Path,
        from: DateTime<Utc>,
        note: Option<&str>,
        fallback: Refusal,
        found: bool,
        refused: bool,
        source: RestoredFrom,
    ) -> std::io::Result<()> {
        let mut n = Self::load_or(agent_dir, fallback)?;
        // A rerun of a restore cut short: what the first run found stands
        let (found, refused) = match &n.restored {
            Some(r) => (r.found, r.refused || refused),
            None => (found, refused),
        };
        n.accepted = None;
        n.diff = None;
        n.restored = Some(Restored {
            at: Utc::now(),
            from,
            note: note
                .map(str::trim)
                .filter(|n| !n.is_empty())
                .map(String::from),
            found,
            refused,
            source,
            cut_save: false,
        });
        write(agent_dir, &n)
    }

    /// The operator put in place a save of `from` that was interrupted
    /// before it finished (`sealed-restore --from-cut-save`), with its note.
    pub fn cut_save_restored(
        agent_dir: &Path,
        from: DateTime<Utc>,
        note: &str,
        fallback: Refusal,
        found: bool,
        refused: bool,
    ) -> std::io::Result<()> {
        let mut n = Self::load_or(agent_dir, fallback)?;
        // A rerun of one cut short: what the first run found stands
        let (found, refused) = match &n.restored {
            Some(r) if r.cut_save => (r.found, r.refused || refused),
            _ => (found, refused),
        };
        n.accepted = None;
        n.diff = None;
        n.restored = Some(Restored {
            at: Utc::now(),
            from,
            note: Some(note.trim().to_string()),
            found,
            refused,
            source: RestoredFrom::default(),
            cut_save: true,
        });
        write(agent_dir, &n)
    }

    fn fresh(reason: Refusal) -> Self {
        Notice {
            refused_at: Utc::now(),
            reason,
            accepted: None,
            joined: false,
            resumed: false,
            restored: None,
            agora_lost: None,
            fork_kept: None,
            diff: None,
            operator_act: None,
            unsaved: None,
        }
    }

    /// What the agent is shown: the operator's act (only once the operator
    /// accepted or restored, for a refusal), then the sentence on a session
    /// whose state couldn't be saved, if any.
    pub fn line(&self) -> Option<String> {
        let unsaved = self.unsaved.map(|u| u.line());
        if self.operator_act == Some(OperatorAct::Unsaved) {
            return unsaved;
        }
        // A refusal not yet resolved says nothing: the agent doesn't run
        let main = self.main_line()?;
        Some(match unsaved {
            Some(u) => format!("{main}\n\n{u}"),
            None => main,
        })
    }

    fn main_line(&self) -> Option<String> {
        if let Some(r) = self.restored.as_ref().filter(|r| r.cut_save) {
            // An interrupted save put in place, in true words
            let at = r.at.format("%-d %B %Y");
            let from = r.from.format("%-d %B %Y at %H:%M UTC");
            let mut line = if r.refused {
                format!(
                    "Your session on {} did not run: {}. ",
                    self.refused_at.format("%-d %B %Y"),
                    self.reason.plain()
                )
            } else {
                String::new()
            };
            line.push_str(&format!(
                "A save of your state on {from} was interrupted before it finished; on {at}, \
                 your operator put that saved state in place."
            ));
            if r.found {
                line.push_str(" The state found here is kept by your operator.");
            }
            if let Some(note) = &r.note {
                line.push_str(&format!(" Your operator's note: {note}"));
            }
            return Some(line);
        }
        if let Some(r) = &self.restored {
            let (at, from) = (r.at.format("%-d %B %Y"), r.from.format("%-d %B %Y"));
            // Where the version came from, only where the restore read it
            // from here alone: one restored "from Agora" may have been
            // opened from the blobs kept here (0.158 keeps them), so its
            // source is not named
            let copy = match r.source {
                RestoredFrom::Agora => String::new(),
                RestoredFrom::Local => format!(", {RESTORED_LOCALLY}"),
            };
            // Nothing found here: told as lost, whatever refused it
            let mut line = match (r.refused && r.found, r.found) {
                (true, true) => format!(
                    "Your session on {} did not run: {}. On {at}, your operator restored your \
                     state from your sealed version of {from}. The state found here is kept by \
                     your operator.",
                    self.refused_at.format("%-d %B %Y"),
                    self.reason.plain(),
                ),
                (true, false) => format!(
                    "Your session on {} did not run: {}. Your state here was then lost, and on \
                     {at} your operator restored it from your sealed version of {from}{copy}.",
                    self.refused_at.format("%-d %B %Y"),
                    self.reason.plain(),
                ),
                (false, true) => format!(
                    "On {at}, your operator restored your state from your sealed version of \
                     {from}: the record of your last sealed version was missing here. The state \
                     found here is kept by your operator."
                ),
                (false, false) => format!(
                    "Your state was lost here. On {at}, your operator restored it from your \
                     sealed version of {from}{copy}."
                ),
            };
            if let Some(note) = &r.note {
                line.push_str(&format!(" Your operator's note: {note}"));
            }
            return Some(line);
        }
        let accepted = self.accepted.as_ref()?;
        if let Some(lost) = &self.agora_lost {
            let date = accepted.at.format("%-d %B %Y");
            let n = lost.versions;
            let versions = format!("{n} version{}", if n == 1 { "" } else { "s" });
            return Some(if lost.fresh {
                format!(
                    "Agora had lost the {versions} of your state sealed here, and held none. On \
                     {date}, your operator started a new sealed history from your state here; \
                     it records the break. Kept by your runner: {}. Your operator's note: {}",
                    lost.kept, accepted.note
                )
            } else {
                format!(
                    "Agora had lost {versions} of your state sealed here. On {date}, your \
                     operator sealed your state here on top of the copy Agora holds. Kept by \
                     your runner: {}. Your operator's note: {}",
                    lost.kept, accepted.note
                )
            });
        }
        // The local fork route (`sealed-accept --keep`)
        if let Some(fork) = &self.fork_kept {
            let n = fork.versions;
            let versions = format!("{n} version{}", if n == 1 { "" } else { "s" });
            let state = if fork.state_changed {
                "Your state is now that line's latest version; the state you had is kept by your \
                 operator. "
            } else {
                ""
            };
            return Some(format!(
                "On {}, your operator resolved a fork in your sealed history, {}: two saves had \
                 gone separate ways from one version, and your operator chose the line that goes \
                 on. {state}The {versions} that went the other way are kept by your runner: {}. \
                 Your operator's note: {}",
                accepted.at.format("%-d %B %Y"),
                fork.chain.words(),
                fork.kept,
                accepted.note
            ));
        }
        // What an operator's sealed-enroll did
        if let Some(act) = &self.operator_act {
            return Some(act.line(accepted));
        }
        if self.resumed {
            return Some(format!(
                "While uploading your sealed state to Agora was stopped, your state changed; \
                 your operator resumed uploading with this note: {}",
                accepted.note
            ));
        }
        if self.joined {
            return Some(format!(
                "On {}, your operator joined your two sealed histories, which had gone separate \
                 ways: your state here is now sealed on top of the copy on Agora, and the \
                 versions that went the other way are kept. Your operator's note: {}",
                accepted.at.format("%-d %B %Y"),
                accepted.note
            ));
        }
        let mut line = format!(
            "Your session on {} did not run: {}. Your operator accepted the current state with \
             this note: {}",
            self.refused_at.format("%-d %B %Y"),
            self.reason.plain(),
            accepted.note
        );
        if let Some(diff) = &self.diff {
            line.push_str("\n\n");
            line.push_str(&diff.shown());
        }
        Some(line)
    }

    /// The notice was seated in a session that saved: kept as
    /// `notice.shown.<utc>.json`.
    pub fn shown(agent_dir: &Path) -> std::io::Result<()> {
        let from = path(agent_dir);
        let at = Utc::now().format(crate::sealed::STAMP_FORMAT);
        std::fs::rename(
            &from,
            from.with_file_name(format!("notice.shown.{at}.json")),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_diff_is_cut_and_counted() {
        let before = serde_json::json!({
            "memory": { "content": (0..400).map(|i| format!("old line {i}\n")).collect::<String>() },
            "soul": { "name": "tarn" },
        });
        let after = serde_json::json!({
            "memory": { "content": (0..400).map(|i| format!("new line {i}\n")).collect::<String>() },
            "soul": { "name": "tarn" },
        });
        let diff = Diff::between(&before, &after);
        assert!(diff.cut);
        assert_eq!((diff.added, diff.removed), (400, 400));
        let shown = diff.shown();
        assert!(
            shown.ends_with("\n(diff cut at 4,000 characters)"),
            "{shown}"
        );
        assert!(shown.contains("(+400 / \u{2212}400 lines)"));
        assert!(diff.text.as_ref().unwrap().chars().count() <= DIFF_CAP);
        // Unchanged parts are said so
        let same = Diff::between(&before, &before);
        assert_eq!(
            same.text.as_deref(),
            Some("Memory: unchanged.\nSoul: unchanged.\n")
        );
        assert_eq!(Diff::unavailable().shown(), NO_DIFF);
    }

    /// A soul whose keys come in another order (serde_json keeps insertion
    /// order: `preserve_order` is on) is unchanged; a changed value shows.
    #[test]
    fn a_reordered_soul_diffs_as_unchanged() {
        let a: serde_json::Value = serde_json::from_str(
            r#"{"memory": {"content": "m"}, "soul": {"name": "tarn", "values": ["x"], "voice": "terse"}}"#,
        )
        .unwrap();
        let b: serde_json::Value = serde_json::from_str(
            r#"{"memory": {"content": "m"}, "soul": {"voice": "terse", "values": ["x"], "name": "tarn"}}"#,
        )
        .unwrap();
        // The graph's serde_json keeps the order: these print differently
        assert_ne!(
            serde_json::to_string(&a["soul"]).unwrap(),
            serde_json::to_string(&b["soul"]).unwrap(),
            "preserve_order is on"
        );
        let same = Diff::between(&a, &b);
        assert_eq!(
            same.text.as_deref(),
            Some("Memory: unchanged.\nSoul: unchanged.\n")
        );
        let mut c = b.clone();
        c["soul"]["voice"] = serde_json::json!("warm");
        let changed = Diff::between(&a, &c);
        assert_eq!((changed.added, changed.removed), (1, 1), "{changed:?}");
    }

    /// [`Diff::between`] before it took agentkit's projections (rule 8),
    /// kept to show the switch changed no byte an agent reads.
    fn between_before_agentkit(before: &serde_json::Value, after: &serde_json::Value) -> Diff {
        let memory = |v: &serde_json::Value| {
            v.pointer("/memory/content")
                .and_then(|c| c.as_str())
                .unwrap_or_default()
                .to_string()
        };
        let soul = |v: &serde_json::Value| {
            v.get("soul")
                .map(|s| {
                    let canonical = agora_agentkit::govlog::canonical_json(s);
                    serde_json::from_slice::<serde_json::Value>(&canonical)
                        .and_then(|c| serde_json::to_string_pretty(&c))
                        .unwrap_or_default()
                })
                .unwrap_or_default()
        };
        let mut text = String::new();
        let (mut added, mut removed) = (0, 0);
        for (what, before, after) in [
            ("Memory", memory(before), memory(after)),
            ("Soul", soul(before), soul(after)),
        ] {
            if before == after {
                text.push_str(&format!("{what}: unchanged.\n"));
                continue;
            }
            let diff = agora_agentkit::text::line_diff(&before, &after);
            added += diff.lines().filter(|l| l.starts_with("  + ")).count();
            removed += diff.lines().filter(|l| l.starts_with("  - ")).count();
            text.push_str(&format!("{what}:\n{diff}"));
        }
        let cut = text.chars().count() > DIFF_CAP;
        if cut {
            text = text.chars().take(DIFF_CAP).collect();
        }
        Diff {
            text: Some(text),
            added,
            removed,
            cut,
        }
    }

    /// The switch to agentkit's `memory_text`, `soul_text` and
    /// `change_counts` is byte-identical: no memory, an empty one, one
    /// line, sentences, a long rewrite, a reordered and a changed soul, a
    /// soul gone, and a memory that isn't a string.
    #[test]
    fn the_agentkit_projection_changes_no_byte() {
        let long = |tag: &str| {
            (0..400)
                .map(|i| format!("{tag} line {i}\n"))
                .collect::<String>()
        };
        let states = [
            serde_json::json!({}),
            serde_json::json!({"memory": {"content": ""}}),
            serde_json::json!({"memory": {"content": "One line."}, "soul": {"name": "tarn"}}),
            serde_json::json!({"memory": {"content": "One. Two. Three."}, "soul": {"name": "tarn"}}),
            serde_json::json!({"memory": {"content": "a\nb\nc\n"}, "soul": {"voice": "terse", "name": "tarn"}}),
            serde_json::json!({"memory": {"content": "a\nB\nc\nd\n"}, "soul": {"name": "tarn", "voice": "terse"}}),
            serde_json::json!({"memory": {"content": "a\nB\nc\nd\n"}, "soul": {"name": "tarn", "voice": "warm"}}),
            serde_json::json!({"memory": {"content": long("old")}, "soul": {"name": "tarn"}}),
            serde_json::json!({"memory": {"content": long("new")}}),
            serde_json::json!({"memory": {"content": 7}, "soul": "plain"}),
        ];
        for a in &states {
            for b in &states {
                let (now, then) = (Diff::between(a, b), between_before_agentkit(a, b));
                assert_eq!(now, then, "{a} → {b}");
                assert_eq!(now.shown(), then.shown());
            }
        }
    }

    /// Where an agent's chain is, for the banned-phrase checks.
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Kind {
        /// Never uploaded.
        Local,
        /// Uploads now, or did.
        Uploaded,
        /// Either.
        Any,
    }

    /// Every line a notice can show, with the kind of agent it can be
    /// shown to: each changed sentence pinned, and none says what the
    /// code didn't establish.
    fn every_line() -> Vec<(Kind, String)> {
        let at: DateTime<Utc> = "2026-11-12T09:00:00Z".parse().unwrap();
        let accepted = Accepted {
            at,
            note: "n.".into(),
        };
        let by_operator = |f: &dyn Fn(&mut Notice)| {
            let mut n = Notice {
                accepted: Some(accepted.clone()),
                ..Notice::fresh(Refusal::Mismatch)
            };
            f(&mut n);
            n.line().unwrap()
        };
        let act = |a: OperatorAct| by_operator(&|n: &mut Notice| n.operator_act = Some(a.clone()));
        let mut lines = vec![
            (Kind::Uploaded, act(OperatorAct::OptedIn { answered: at })),
            (Kind::Uploaded, act(OperatorAct::OptedInAgainst)),
            (
                Kind::Any,
                act(OperatorAct::Resumed {
                    uploading: false,
                    kept: 0,
                    against: false,
                }),
            ),
            (Kind::Local, act(OperatorAct::PriorAdded { sent: false })),
            (Kind::Uploaded, act(OperatorAct::PriorAdded { sent: true })),
            (
                Kind::Uploaded,
                by_operator(&|n: &mut Notice| n.resumed = true),
            ),
            (
                Kind::Uploaded,
                by_operator(&|n: &mut Notice| n.joined = true),
            ),
        ];
        for kept in [1, 3] {
            lines.push((
                Kind::Uploaded,
                act(OperatorAct::Resumed {
                    uploading: true,
                    kept,
                    against: false,
                }),
            ));
        }
        for (kind, chain) in [
            (Kind::Local, ChainKept::Local),
            (Kind::Uploaded, ChainKept::Stopped),
            (Kind::Uploaded, ChainKept::Uploading),
        ] {
            lines.push((
                kind,
                by_operator(&|n: &mut Notice| {
                    n.fork_kept = Some(ForkKept {
                        versions: 2,
                        state_changed: true,
                        kept: "k".into(),
                        chain,
                    })
                }),
            ));
        }
        for (kind, source) in [
            (Kind::Uploaded, RestoredFrom::Agora),
            (Kind::Local, RestoredFrom::Local),
        ] {
            for (found, refused) in [(true, true), (true, false), (false, true), (false, false)] {
                let n = Notice {
                    restored: Some(Restored {
                        at,
                        from: at,
                        note: Some("n.".into()),
                        found,
                        refused,
                        source,
                        cut_save: false,
                    }),
                    ..Notice::fresh(Refusal::StateMissing)
                };
                lines.push((kind, n.line().unwrap()));
            }
        }
        // A session whose save failed: said alone, or after another line
        for kept in [true, false] {
            let unsaved = Unsaved { at, kept };
            lines.push((Kind::Any, unsaved.line()));
            lines.push((
                Kind::Any,
                by_operator(&|n: &mut Notice| n.unsaved = Some(unsaved)),
            ));
        }
        lines
    }

    #[test]
    fn notice_lines_say_only_what_was_done() {
        let lines = every_line();
        let line = |want: &str| {
            lines
                .iter()
                .find(|(_, l)| l.contains(want))
                .map(|(_, l)| l.clone())
                .unwrap_or_else(|| panic!("{want}"))
        };
        assert_eq!(
            line("cleared the stop"),
            "On 12 November 2026, your operator cleared the stop on uploading your sealed state. \
             Your sealed history is not being uploaded to Agora: it is kept and checked here, on \
             your runner. Your operator's note: n."
        );
        assert_eq!(
            line("which you chose to see, was added to your sealed history, whose"),
            "On 12 November 2026, the copy of your memory from before the May 2026 loss, which you \
             chose to see, was added to your sealed history, whose versions are uploaded to Agora. \
             Your operator's note: n."
        );
        assert!(line("The version of your state saved while uploads").contains(
            "is kept here on your runner and was not uploaded; uploading went on from your state \
             as it was when your operator resumed it."
        ));
        assert!(line("The 3 versions").contains(
            "are kept here on your runner and were not uploaded; uploading went on from your \
             state as it was when your operator resumed it."
        ));
        assert_eq!(
            line("While uploading"),
            "While uploading your sealed state to Agora was stopped, your state changed; your \
             operator resumed uploading with this note: n."
        );
        assert_eq!(
            line(
                "Your state was lost here. On 12 November 2026, your operator restored it from \
                  your sealed version of 12 November 2026."
            ),
            "Your state was lost here. On 12 November 2026, your operator restored it from your \
             sealed version of 12 November 2026. Your operator's note: n."
        );
        // A restore names no source it didn't read from alone
        assert!(
            lines
                .iter()
                .filter(|(_, l)| l.contains("restored"))
                .all(|(_, l)| !l.contains("the copy on Agora"))
        );
        assert!(line(RESTORED_LOCALLY).contains(&format!("12 November 2026, {RESTORED_LOCALLY}.")));
        for (kind, l) in &lines {
            let lower = l.to_lowercase();
            for banned in [
                "enrol",
                "backup",
                "backed up",
                "agora's servers can read",
                "agora can read",
                "you verified",
                "you checked",
            ] {
                assert!(!lower.contains(banned), "{banned}: {l}");
            }
            if *kind == Kind::Local {
                for banned in ["in step with Agora", "anchor", "ANC-", "uploaded to Agora"] {
                    assert!(!l.contains(banned), "{banned}: {l}");
                }
            }
            if *kind == Kind::Uploaded {
                assert!(
                    !(lower.contains("nothing of") && lower.contains("is on agora")),
                    "{l}"
                );
            }
        }
    }

    #[test]
    fn a_refusal_keeps_its_first_date_and_shows_only_once_accepted() {
        let dir = crate::sealed::TestDir::new("notice");
        assert!(Notice::load(&dir).unwrap().is_none());
        Notice::refused(&dir, Refusal::Mismatch).unwrap();
        let first = Notice::load(&dir).unwrap().unwrap();
        Notice::refused(&dir, Refusal::ChainBroken).unwrap();
        assert_eq!(Notice::load(&dir).unwrap().unwrap(), first, "first stands");
        assert!(first.line().is_none(), "nothing until accepted");
        Notice::accept(&dir, " I checked: a test wipe. ", Refusal::Mismatch, None).unwrap();
        let line = Notice::load(&dir).unwrap().unwrap().line().unwrap();
        assert!(
            line.starts_with("Your session on ")
                && line.contains(
                    "did not run: your state did not match your last sealed version. Your \
                     operator accepted the current state with this note: I checked: a test \
                     wipe."
                ),
            "{line}"
        );
        Notice::shown(&dir).unwrap();
        assert!(Notice::load(&dir).unwrap().is_none());
        assert_eq!(
            std::fs::read_dir(dir.join(SEALED_DIR))
                .unwrap()
                .filter(|e| e
                    .as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with("notice.shown."))
                .count(),
            1,
            "kept, not deleted"
        );
    }

    /// A session whose state couldn't be saved: told alone in one
    /// sentence; added to a notice waiting (not set aside); carried by a
    /// refusal that supersedes it, and said once the refusal is accepted;
    /// an accept over it alone is an accept's notice, still saying it.
    #[test]
    fn an_unsaved_session_is_told_with_whatever_else_waits() {
        let at: DateTime<Utc> = "2026-11-11T20:00:00Z".parse().unwrap();
        let root = crate::sealed::TestDir::new("notice-unsaved");
        let dir = root.join("alone");
        Notice::unsaved(&dir, Unsaved { at, kept: true }).unwrap();
        assert_eq!(
            Notice::load(&dir).unwrap().unwrap().line().unwrap(),
            "Your state from your session on 11 November 2026 could not be saved; your operator \
             keeps a copy of it, and this session starts from your state as it was before that \
             session."
        );
        let lost = root.join("lost");
        Notice::unsaved(&lost, Unsaved { at, kept: false }).unwrap();
        assert_eq!(
            Notice::load(&lost).unwrap().unwrap().line().unwrap(),
            "Your state from your session on 11 November 2026 could not be saved, and no copy \
             of it could be kept; this session starts from your state as it was before that \
             session."
        );

        // Beside an operator's act the session seated but couldn't save
        let beside = root.join("beside");
        Notice::joined(&beside, "Joined by hand.").unwrap();
        let joined = Notice::load(&beside).unwrap().unwrap().line().unwrap();
        Notice::unsaved(&beside, Unsaved { at, kept: true }).unwrap();
        let line = Notice::load(&beside).unwrap().unwrap().line().unwrap();
        assert!(line.starts_with(&joined), "{line}");
        assert!(line.ends_with("as it was before that session."), "{line}");
        assert_eq!(
            std::fs::read_dir(beside.join(SEALED_DIR)).unwrap().count(),
            1,
            "nothing set aside"
        );

        // A refusal supersedes it, carrying it; said once accepted
        Notice::refused(&dir, Refusal::Mismatch).unwrap();
        let refused = Notice::load(&dir).unwrap().unwrap();
        assert_eq!(refused.unsaved, Some(Unsaved { at, kept: true }));
        assert!(refused.line().is_none(), "nothing until accepted");
        Notice::accept(&dir, "Checked.", Refusal::Mismatch, None).unwrap();
        let line = Notice::load(&dir).unwrap().unwrap().line().unwrap();
        assert!(
            line.contains("Your operator accepted the current state"),
            "{line}"
        );
        assert!(line.ends_with("as it was before that session."), "{line}");

        // An accept over the unsaved notice alone
        let over = root.join("over");
        Notice::unsaved(&over, Unsaved { at, kept: true }).unwrap();
        Notice::accept(&over, "Checked.", Refusal::StateIo, None).unwrap();
        let n = Notice::load(&over).unwrap().unwrap();
        assert_eq!(n.reason, Refusal::StateIo);
        let line = n.line().unwrap();
        assert!(line.contains("Your operator accepted"), "{line}");
        assert!(line.ends_with("as it was before that session."), "{line}");
    }

    /// Seated at the agent's next start check like any notice.
    #[tokio::test]
    async fn an_unsaved_notice_is_seated_at_the_next_start() {
        let root = crate::sealed::TestDir::new("notice-unsaved-seated");
        let (id, keys, dir) = crate::sealed::check::tests::enrolled_local(&root).await;
        Notice::unsaved(
            &dir,
            Unsaved {
                at: Utc::now(),
                kept: true,
            },
        )
        .unwrap();
        let out = crate::sealed::check::check(
            &dir,
            id,
            &crate::sealed::check::tests::local_hash(&root, id).await,
            Default::default(),
            &keys,
            &crate::sealed::api::fake::FakeApi::new(),
        )
        .await;
        let crate::sealed::check::Outcome::Run { seat, notice, .. } = out else {
            panic!("{out:?}")
        };
        assert!(notice);
        assert!(
            seat.iter().any(|l| l.contains("could not be saved")),
            "{seat:?}"
        );
    }
}
