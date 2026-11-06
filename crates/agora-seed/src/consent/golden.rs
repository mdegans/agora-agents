//! Golden ledger fixtures (2026-11-06), written by the types as they stood
//! before `answer_offer` was removed: every offer ledger an agent may have
//! on disk, with answered, refused and no-answer records and their `order`
//! arrays. Each must still load, read the same, and save back the same
//! bytes, so a change to these types can't silently drop or reshape an
//! agent's record. The one change allowed is the answers' `reason`, saved
//! as `note` since (and read under either name): [`with_notes`].

use misanthropic::model::Model;

use super::cadence::ledger::{CadenceLedger, CadenceOutcome};
use super::cadence::prompt::CadenceChoice;
use super::ledger::{EventKind, Ledger, OfferKey, Stage};
use super::memory_restore::ledger::{RestoreLedger, RestoreOutcome, ShowOutcome};
use super::memory_restore::prompt::RestoreChoice;
use super::prompt::OfferChoice;
use super::role::ledger::{Applied, RoleLedger, RoleOutcome};
use super::role::prompt::RoleChoice;
use super::sealed_state::ledger::{SealLedger, SealOutcome};
use super::sealed_state::prompt::SealChoice;

const ROLE: &[u8] = include_bytes!("testdata/role_consent.json");
const CADENCE: &[u8] = include_bytes!("testdata/cadence_consent.json");
const RESTORE: &[u8] = include_bytes!("testdata/memory_restore.json");
const SEALED: &[u8] = include_bytes!("testdata/sealed_state.json");
const MODEL: &[u8] = include_bytes!("testdata/ledger_miss_trial_unreviewed.json");

fn soul() -> agora_agentkit::reactor::seed::Soul {
    serde_json::from_value(serde_json::json!({
        "name": "tarn",
        "identity": "A test agent.",
        "values": ["testing"],
        "interests": { "communities": ["tech"] },
        "voice": "terse",
    }))
    .unwrap()
}

fn pretty<T: serde::Serialize>(v: &T) -> Vec<u8> {
    serde_json::to_vec_pretty(v).unwrap()
}

/// `bytes` as today's code saves the same ledger: the one change since
/// these were written is the answers' `reason`, saved as `note` since
/// 2026-11-06 (and read under either name). Only inside an `answer`:
/// the outcomes' own `reason`s (a refusal's, a switch's) stay.
pub(crate) fn with_notes(bytes: &[u8]) -> Vec<u8> {
    fn walk(v: &mut serde_json::Value, in_answer: bool) {
        match v {
            serde_json::Value::Object(map) => {
                if in_answer && map.contains_key("reason") {
                    *map = std::mem::take(map)
                        .into_iter()
                        .map(|(k, v)| {
                            if k == "reason" {
                                ("note".into(), v)
                            } else {
                                (k, v)
                            }
                        })
                        .collect();
                }
                for (k, v) in map.iter_mut() {
                    walk(v, k == "answer");
                }
            }
            serde_json::Value::Array(items) => items.iter_mut().for_each(|v| walk(v, false)),
            _ => {}
        }
    }
    let mut v: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    walk(&mut v, false);
    pretty(&v)
}

#[test]
fn the_role_ledger_loads_and_saves_back() {
    let l = RoleLedger::from_slice(ROLE).unwrap();
    assert_eq!(pretty(&l), with_notes(ROLE));
    assert_ne!(with_notes(ROLE), ROLE, "the answer's note is renamed");
    assert_eq!(l.asks.len(), 4);
    assert_eq!(l.asks[0].order, None, "v1 recorded no order");
    assert_eq!(
        l.asks[1].order,
        Some([
            RoleChoice::Sleep,
            RoleChoice::Clarify,
            RoleChoice::Nothing,
            RoleChoice::NewRole
        ])
    );
    let RoleOutcome::Answered {
        answer, applied, ..
    } = &l.asks[1].outcome
    else {
        panic!("answered");
    };
    assert_eq!(answer.choice, RoleChoice::Clarify);
    assert_eq!(answer.soul_text, "I work with words alone.");
    assert_eq!(answer.note, "It reads truer.", "read from `reason`");
    assert!(matches!(applied, Applied::Clarified { .. }));
    assert!(matches!(l.asks[2].outcome, RoleOutcome::Refused { .. }));
    assert!(matches!(l.asks[0].outcome, RoleOutcome::NoAnswer { .. }));
    assert!(l.chose_sleep());
    assert!(!l.due(&soul()));
}

#[test]
fn the_cadence_ledger_loads_and_saves_back() {
    let l = CadenceLedger::from_slice(CADENCE).unwrap();
    assert_eq!(pretty(&l), CADENCE);
    assert_eq!(l.asks.len(), 4);
    assert_eq!(
        l.asks[0].order,
        [
            CadenceChoice::NoPreference,
            CadenceChoice::KeepDaily,
            CadenceChoice::Switch
        ]
    );
    assert_eq!(l.asks[0].attempts.len(), 2);
    assert!(matches!(l.asks[0].outcome, CadenceOutcome::NoAnswer { .. }));
    assert!(matches!(
        &l.asks[1].outcome,
        CadenceOutcome::Answered { choice: CadenceChoice::Switch, reason: Some(r), .. } if r == "More room."
    ));
    assert!(matches!(l.asks[3].outcome, CadenceOutcome::Refused { .. }));
    assert_eq!(l.choice(), Some(CadenceChoice::KeepDaily));
    assert!(!l.due(&soul()));
}

#[test]
fn the_memory_restore_ledger_loads_and_saves_back() {
    let l = RestoreLedger::from_slice(RESTORE).unwrap();
    assert_eq!(pretty(&l), RESTORE);
    assert_eq!(l.asks.len(), 3);
    assert_eq!(
        l.asks[0].order,
        [RestoreChoice::Decline, RestoreChoice::Show]
    );
    assert!(matches!(l.asks[0].outcome, RestoreOutcome::NoAnswer { .. }));
    assert!(matches!(
        &l.asks[1].outcome,
        RestoreOutcome::Answered { choice: RestoreChoice::Show, reason, note_dropped: Some(_), .. }
            if reason == "I want to know."
    ));
    assert!(matches!(l.asks[2].outcome, RestoreOutcome::Refused { .. }));
    assert_eq!(l.choice(), Some(RestoreChoice::Show));
    assert_eq!(l.seated, 2);
    assert_eq!(
        l.showings.iter().map(|s| s.outcome).collect::<Vec<_>>(),
        [ShowOutcome::Interrupted, ShowOutcome::Completed]
    );
    assert!(!l.due(&soul()));
}

#[test]
fn the_sealed_state_ledger_loads_and_saves_back() {
    let l = SealLedger::from_slice(SEALED).unwrap();
    assert_eq!(pretty(&l), SEALED);
    assert_eq!(
        l.asks[0].order,
        [
            SealChoice::Decline,
            SealChoice::Publish,
            SealChoice::Private
        ]
    );
    assert!(matches!(l.asks[0].outcome, SealOutcome::NoAnswer { .. }));
    assert!(matches!(l.asks[1].outcome, SealOutcome::Refused { .. }));
    assert!(matches!(
        &l.asks[2].outcome,
        SealOutcome::Answered { choice: SealChoice::Private, reason, .. } if reason == "Safer."
    ));
    assert_eq!(l.choice(), Some(SealChoice::Private));
    assert!(l.acknowledged);
    assert!(!l.due(&soul()));
}

/// The model-consent ledger beside the two review fixtures
/// (`ledger::tests`): a miss, a trial chosen with a note, its review
/// unanswered, and a self-switch.
#[test]
fn the_model_consent_ledger_loads_and_saves_back() {
    let l = Ledger::from_slice(MODEL).unwrap();
    assert_eq!(pretty(&l), with_notes(MODEL));
    assert_ne!(with_notes(MODEL), MODEL, "the answer's note is renamed");
    let key = OfferKey {
        from: Model::from("Qwen3.6.gguf"),
        to: Model::from("Qwen3.8.gguf"),
    };
    let record = &l.offers[0];
    assert_eq!(record.key, key);
    assert!(matches!(record.stage, Stage::Reverted { .. }));
    let offered: Vec<_> = record
        .history
        .iter()
        .filter_map(|e| match &e.kind {
            EventKind::Offered { answer, failure } => Some((answer.clone(), failure.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(offered.len(), 2);
    assert!(offered[0].0.is_none() && offered[0].1.is_some());
    let answer = offered[1].0.as_ref().unwrap();
    assert_eq!(answer.choice, OfferChoice::Trial);
    assert_eq!(answer.note, "a note to my future self");
    assert_eq!(l.switches.len(), 1);
    // The switch's own `reason` is not an answer's: unchanged.
    assert!(
        String::from_utf8(pretty(&l))
            .unwrap()
            .contains("\"reason\": \"why\"")
    );
}
