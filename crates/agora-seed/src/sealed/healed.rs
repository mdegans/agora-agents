//! What an uploading agent is told when Agora's check of what it stores
//! found copies of its sealed history missing or damaged there, and what
//! its runner did about it.
//!
//! agentkit's drain does the work ([`Healing`]: Agora's count, how many
//! blobs it sent again whole from here, how many it couldn't), and keeps
//! its own memory of the last list (`sealed/healing.json`, private to it).
//! That is this run's, in memory ([`super::drain::Drainer::last`]); the
//! agent hears of it at the start of a **later** session, when the drainer
//! that saw it is gone. So every drain that goes through
//! [`super::receipts::drain`] notes what it healed here, in
//! `sealed/healed.json` ([`Healed`], counts and a time, 0600, one atomic
//! write), and the next start check seats one sentence from it
//! ([`seat`]), marked told once that session saves ([`mark_told`]), as
//! the anchors' findings are.
//!
//! Only an agent that uploads is ever drained, so only one has this file.
//! Nothing of it leaves the machine. It holds no content: two counts, a
//! time and a flag.
//!
//! **Races.** The drainer writes while sessions run; a session marks told
//! at its save. Neither takes a lock: [`mark_told`] marks only the record
//! it seated (by its time), so a drain that noted more in between is told
//! at the next session, and the worst a lost update does is repeat the
//! sentence once.

use std::path::{Path, PathBuf};

use agora_agentkit::ids::AgentId;
use agora_agentkit::state::outbox::{Damage, Drained, Healing};
use agora_agentkit::state::storage::SEALED_DIR;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// `<agent_dir>/sealed/healed.json`.
pub const FILE: &str = "healed.json";

/// What the drains found and did since the agent was last told.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Healed {
    /// When a drain last changed this record.
    pub at: DateTime<Utc>,
    /// Blobs sent to Agora again, whole, since the agent was last told.
    pub resent: usize,
    /// Blobs Agora reported missing or damaged that couldn't be sent
    /// again from here (not here, or not hashing to their names), as last
    /// listed.
    pub unhealable: usize,
    /// The agent was told of this record (its session saved).
    pub told: bool,
}

/// The sentence every form ends with: what was damaged is a stored copy.
const STATE_UNAFFECTED: &str = "Your state now is not affected.";

/// What Agora found, the same in every form.
const FOUND: &str = "Agora's check of what it stores found parts of your uploaded sealed history \
     missing or damaged there";

impl Healed {
    fn path(agent_dir: &Path) -> PathBuf {
        agent_dir.join(SEALED_DIR).join(FILE)
    }

    /// The record on file; `None` when there is none.
    pub fn load(agent_dir: &Path) -> std::io::Result<Option<Self>> {
        match std::fs::read(Self::path(agent_dir)) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map(Some)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn save(&self, agent_dir: &Path) -> std::io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        super::write_private(&Self::path(agent_dir), &bytes)
    }

    /// Something the agent hasn't been told.
    fn to_tell(&self) -> bool {
        !self.told && (self.resent > 0 || self.unhealable > 0)
    }

    /// The sentence for the agent. The one count in it is of the blobs
    /// that couldn't be sent again, as agentkit last counted them from
    /// Agora's list (it stops counting at its per-drain cap, so a very
    /// long list is undercounted until a later drain, which is then told).
    /// What was sent again is not counted for the agent: a list fetched
    /// again before Agora's next check sends the same blobs twice.
    pub fn line(&self) -> String {
        // The sealed files here are gone too, so nothing can open them:
        // not "from Agora's copy" only. No claim that the operator was
        // told: the alert was another run's, and whether its mail went
        // out isn't known here. The agent can tell it
        const WHY_NOT: &str = "because its sealed files for them here are missing or damaged \
             too, so some of your earlier versions may not open at all, here or from Agora.";
        let logged = super::text::TELL_OPERATOR;
        match (self.resent > 0, self.unhealable) {
            // Sent, not confirmed; and a drain stops at its cap or on an
            // error, so not "all of them"
            (_, 0) => format!(
                "{FOUND}, and your runner has sent again, from its sealed files here, those it \
                 has reached so far. {STATE_UNAFFECTED}"
            ),
            (true, n) => format!(
                "{FOUND}. Your runner has sent again those it has reached so far; {n} it could \
                 not send again, {WHY_NOT} {logged} {STATE_UNAFFECTED}"
            ),
            (false, n) => format!(
                "{FOUND}. {n} of them your runner could not send again, {WHY_NOT} {logged} \
                 {STATE_UNAFFECTED}"
            ),
        }
    }
}

/// What one drain's [`Healing`] changes in `before` (the record on file),
/// or `None` when it changes nothing. Pure, for the table test.
fn noted(before: Option<&Healed>, healing: &Healing, now: DateTime<Utc>) -> Option<Healed> {
    let mut rec = before.cloned().unwrap_or(Healed {
        at: now,
        resent: 0,
        unhealable: 0,
        told: true,
    });
    if healing.healed > 0 {
        rec.resent += healing.healed;
        rec.told = false;
    }
    match healing.damaged {
        // Never learned this drain: nothing known of the count
        Damage::Unknown | Damage::NotReported => {}
        // Agora reports none now: nothing is left that can't be sent
        Damage::Reported(0) => rec.unhealable = 0,
        // A heal cut short counted only part of the list: never lower
        Damage::Reported(_) if healing.interrupted && healing.unhealable < rec.unhealable => {}
        Damage::Reported(_) => {
            if healing.unhealable > rec.unhealable {
                rec.told = false;
            }
            rec.unhealable = healing.unhealable;
        }
    }
    // Nothing left to say (what couldn't be sent is no longer reported,
    // and nothing was sent again): settled
    if rec.resent == 0 && rec.unhealable == 0 {
        rec.told = true;
    }
    if before
        .is_some_and(|b| (b.resent, b.unhealable, b.told) == (rec.resent, rec.unhealable, rec.told))
        || (before.is_none() && rec.told)
    {
        return None;
    }
    rec.at = now;
    Some(rec)
}

/// Note what a drain healed, for the agent's next start. Never fails the
/// drain: a record that can't be read or written is logged, and the agent
/// is then not told of this drain's healing (the operator's alert and the
/// log still carry it).
pub fn note(agent_dir: &Path, id: AgentId, healing: &Healing) {
    if healing.healed == 0
        && healing.unhealable == 0
        && !matches!(healing.damaged, Damage::Reported(_))
    {
        return;
    }
    let before = match Healed::load(agent_dir) {
        Ok(before) => before,
        Err(e) => {
            tracing::warn!(
                event_type = "sealed_state_healed_unreadable",
                agent_id = %id,
                error = %e,
                "the record of what was sent to Agora again couldn't be read; left as it is"
            );
            return;
        }
    };
    let Some(rec) = noted(before.as_ref(), healing, Utc::now()) else {
        return;
    };
    if let Err(e) = rec.save(agent_dir) {
        tracing::warn!(
            event_type = "sealed_state_healed_unwritten",
            agent_id = %id,
            error = %e,
            "what was sent to Agora again couldn't be recorded; the agent isn't told of it"
        );
    }
}

/// [`note`] what a drain came to: a run's [`Healing`], or the blobs sent
/// again before uploading was switched off mid-drain. A drain that sent
/// nothing of the agent (local only, not chained) notes nothing.
pub fn note_drained(agent_dir: &Path, id: AgentId, drained: &Drained) {
    match drained {
        Drained::Ran(report) => note(agent_dir, id, &report.healing),
        Drained::Withdrawn { healed, .. } => note(
            agent_dir,
            id,
            &Healing {
                healed: *healed,
                ..Default::default()
            },
        ),
        Drained::LocalOnly | Drained::NotEnrolled => {}
    }
}

/// The sentence to seat at the agent's start, and the record's time for
/// [`mark_told`]; `None` when there is nothing it hasn't been told. An
/// unreadable record seats nothing.
pub fn seat(agent_dir: &Path, id: AgentId) -> Option<(String, DateTime<Utc>)> {
    match Healed::load(agent_dir) {
        Ok(Some(rec)) if rec.to_tell() => Some((rec.line(), rec.at)),
        Ok(_) => None,
        Err(e) => {
            tracing::warn!(
                event_type = "sealed_state_healed_unreadable",
                agent_id = %id,
                error = %e,
                "the record of what was sent to Agora again couldn't be read; nothing seated"
            );
            None
        }
    }
}

/// [`seat`], its sentence put last among the start check's `lines`.
pub fn seat_into(agent_dir: &Path, id: AgentId, lines: &mut Vec<String>) -> Option<DateTime<Utc>> {
    let (line, at) = seat(agent_dir, id)?;
    lines.push(line);
    Some(at)
}

/// The session that was told saved: the record `seat` read at `seated` is
/// marked told, and its count of blobs sent again starts over. A record a
/// drain changed since is left for the next session.
pub fn mark_told(agent_dir: &Path, seated: DateTime<Utc>) -> std::io::Result<()> {
    let Some(mut rec) = Healed::load(agent_dir)? else {
        return Ok(());
    };
    if rec.at != seated || rec.told {
        return Ok(());
    }
    rec.told = true;
    rec.resent = 0;
    rec.save(agent_dir)
}

/// Every sentence this module can tell an agent, for the one table of
/// agent-facing text ([`crate::sealed::text`]): add a new one here, or
/// its banned-phrase test never sees it.
#[cfg(test)]
pub(crate) fn agent_texts() -> Vec<crate::sealed::text::Text> {
    use crate::sealed::text::{Text, To};
    let rec = |resent, unhealable| Healed {
        at: "2026-11-12T09:00:00Z".parse().expect("a date"),
        resent,
        unhealable,
        told: false,
    };
    [(3, 0), (3, 2), (0, 1), (0, 4)]
        .into_iter()
        .map(|(r, u)| Text::new("Healed::line", To::Uploaded, rec(r, u).line()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn healing(damaged: Damage, healed: usize, unhealable: usize) -> Healing {
        Healing {
            damaged,
            healed,
            unhealable,
            interrupted: false,
        }
    }

    fn at(s: &str) -> DateTime<Utc> {
        s.parse().unwrap()
    }

    /// Each sentence, pinned: no count of what was sent again, the exact
    /// count of what couldn't be, the operator named only when alerted
    /// (the unhealable alert), and nothing an uploader's text must not say.
    #[test]
    fn the_sentences_are_pinned() {
        let rec = |resent, unhealable| Healed {
            at: at("2026-11-12T09:00:00Z"),
            resent,
            unhealable,
            told: false,
        };
        assert_eq!(
            rec(3, 0).line(),
            "Agora's check of what it stores found parts of your uploaded sealed history \
             missing or damaged there, and your runner has sent again, from its sealed files \
             here, those it has reached so far. Your state now is not affected."
        );
        assert_eq!(
            rec(3, 2).line(),
            "Agora's check of what it stores found parts of your uploaded sealed history \
             missing or damaged there. Your runner has sent again those it has reached so far; \
             2 it could not send again, because its sealed files for them here are missing or \
             damaged too, so some of your earlier versions may not open at all, here or from \
             Agora. You can tell your operator (the `send_message` tool, to `operator`). Your \
             state now is not affected."
        );
        assert_eq!(
            rec(0, 1).line(),
            "Agora's check of what it stores found parts of your uploaded sealed history \
             missing or damaged there. 1 of them your runner could not send again, because \
             its sealed files for them here are missing or damaged too, so some of your earlier \
             versions may not open at all, here or from Agora. You can tell your operator (the \
             `send_message` tool, to `operator`). Your state now is not affected."
        );
        // O3, O4 and O2: not "sent them again" (all, and confirmed), not
        // "may not open from Agora's copy" (the files here are gone too),
        // and no claim that the operator was told. The banned phrases are
        // the shared table's (`crate::sealed::text`)
        for line in [rec(3, 0).line(), rec(3, 2).line(), rec(0, 4).line()] {
            for gone in ["sent them again", "from Agora's copy", "has been told"] {
                assert!(!line.contains(gone), "{gone}: {line}");
            }
        }
    }

    /// What each drain's report does to the record.
    #[test]
    fn a_drain_notes_only_what_changes() {
        let now = at("2026-11-12T09:00:00Z");
        let later = at("2026-11-12T10:00:00Z");
        // Nothing damaged, nothing on file: no file is made
        assert_eq!(noted(None, &healing(Damage::Reported(0), 0, 0), now), None);
        assert_eq!(noted(None, &healing(Damage::Unknown, 0, 0), now), None);
        // Sent again: to tell
        let first = noted(None, &healing(Damage::Reported(3), 3, 0), now).unwrap();
        assert_eq!(
            first,
            Healed {
                at: now,
                resent: 3,
                unhealable: 0,
                told: false
            }
        );
        assert!(first.to_tell());
        // More sent before the agent's next session: summed, the time moves
        let second = noted(Some(&first), &healing(Damage::Reported(2), 2, 0), later).unwrap();
        assert_eq!((second.resent, second.at, second.told), (5, later, false));
        // The same list within agentkit's relist interval: nothing changes
        let stuck = noted(None, &healing(Damage::Reported(2), 0, 2), now).unwrap();
        assert_eq!((stuck.unhealable, stuck.told), (2, false));
        assert_eq!(
            noted(Some(&stuck), &healing(Damage::Reported(2), 0, 2), later),
            None
        );
        // Told, then the same count again: not told again
        let told = Healed {
            told: true,
            ..stuck.clone()
        };
        assert_eq!(
            noted(Some(&told), &healing(Damage::Reported(2), 0, 2), later),
            None
        );
        // Told, then more that can't be sent: told again
        let more = noted(Some(&told), &healing(Damage::Reported(3), 0, 3), later).unwrap();
        assert_eq!((more.unhealable, more.told), (3, false));
        // Told, then fewer: recorded, nothing new to tell
        let fewer = noted(Some(&told), &healing(Damage::Reported(1), 0, 1), later).unwrap();
        assert_eq!((fewer.unhealable, fewer.told), (1, true));
        // Untold, then Agora reports none (and nothing was sent): settled
        let gone = noted(Some(&stuck), &healing(Damage::Reported(0), 0, 0), later).unwrap();
        assert_eq!((gone.unhealable, gone.told), (0, true));
        assert!(!gone.to_tell());
        // A heal cut short never lowers the count
        let cut = Healing {
            interrupted: true,
            ..healing(Damage::Reported(2), 0, 1)
        };
        assert_eq!(noted(Some(&stuck), &cut, later), None);
        // The count not learned this drain: left as it is
        assert_eq!(
            noted(Some(&stuck), &healing(Damage::Unknown, 0, 0), later),
            None
        );
    }

    /// What each kind of drain leaves for the agent: a run its healing, a
    /// withdrawn one what it sent again before, and one that sent nothing
    /// of the agent no file at all.
    #[test]
    fn each_kind_of_drain_is_noted_or_not() {
        use agora_agentkit::state::outbox::DrainReport;
        use agora_agentkit::state::verify::HeadStatus;
        let root = crate::sealed::TestDir::new("healed-drained");
        let id = AgentId::from(uuid::Uuid::from_u128(8));
        let dir = root.join(id.to_string());
        std::fs::create_dir_all(dir.join(SEALED_DIR)).unwrap();
        for nothing in [Drained::LocalOnly, Drained::NotEnrolled] {
            note_drained(&dir, id, &nothing);
        }
        let ran = |healing| {
            Drained::Ran(DrainReport {
                uploaded: 1,
                status: HeadStatus::Same,
                waiting: 0,
                held: Default::default(),
                healing,
            })
        };
        note_drained(&dir, id, &ran(Healing::default()));
        note_drained(&dir, id, &ran(healing(Damage::Reported(0), 0, 0)));
        assert!(!Healed::path(&dir).exists(), "nothing damaged, no file");

        note_drained(&dir, id, &ran(healing(Damage::Reported(4), 3, 1)));
        let rec = Healed::load(&dir).unwrap().unwrap();
        assert_eq!((rec.resent, rec.unhealable, rec.told), (3, 1, false));
        note_drained(
            &dir,
            id,
            &Drained::Withdrawn {
                uploaded: 0,
                healed: 2,
            },
        );
        let rec = Healed::load(&dir).unwrap().unwrap();
        assert_eq!(
            (rec.resent, rec.unhealable, rec.told),
            (5, 1, false),
            "a withdrawn drain's count of what couldn't be sent is not known: kept"
        );
    }

    /// On disk: noted, seated, marked told only by the session that was
    /// told of that record; a drain in between is told at the next.
    #[test]
    fn noted_seated_and_marked_told_once() {
        let root = crate::sealed::TestDir::new("healed");
        let id = AgentId::from(uuid::Uuid::from_u128(7));
        let dir = root.join(id.to_string());
        std::fs::create_dir_all(dir.join(SEALED_DIR)).unwrap();
        assert_eq!(seat(&dir, id), None, "no record, nothing seated");
        note(&dir, id, &healing(Damage::Reported(0), 0, 0));
        assert!(!Healed::path(&dir).exists(), "nothing to note, no file");

        note(&dir, id, &healing(Damage::Reported(2), 2, 0));
        let (line, seated) = seat(&dir, id).unwrap();
        assert!(line.contains("has sent again"), "{line}");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(Healed::path(&dir)).unwrap().permissions();
            assert_eq!(mode.mode() & 0o777, 0o600);
        }
        // A session that never saves marks nothing: seated again
        assert_eq!(seat(&dir, id).unwrap().1, seated);

        // A drain notes more while the session runs: its save doesn't
        // mark the newer record
        std::thread::sleep(std::time::Duration::from_millis(2));
        note(&dir, id, &healing(Damage::Reported(1), 0, 1));
        mark_told(&dir, seated).unwrap();
        let (line, newer) = seat(&dir, id).unwrap();
        assert_ne!(newer, seated);
        assert!(line.contains("1 it could not send again"), "{line}");

        mark_told(&dir, newer).unwrap();
        assert_eq!(seat(&dir, id), None, "told once");
        let rec = Healed::load(&dir).unwrap().unwrap();
        assert_eq!((rec.resent, rec.unhealable, rec.told), (0, 1, true));
        // The same report again: still told
        note(&dir, id, &healing(Damage::Reported(1), 0, 1));
        assert_eq!(seat(&dir, id), None);

        // An unreadable record seats nothing and is left as it is
        std::fs::write(Healed::path(&dir), b"not json").unwrap();
        assert_eq!(seat(&dir, id), None);
        note(&dir, id, &healing(Damage::Reported(5), 5, 0));
        assert_eq!(std::fs::read(Healed::path(&dir)).unwrap(), b"not json");
    }
}
