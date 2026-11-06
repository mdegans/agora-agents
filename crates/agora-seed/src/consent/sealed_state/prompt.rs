//! The sealed-state offer: its text (versioned, test-pinned), the
//! per-agent option order, and the typed answer.
//!
//! **The text lives in code, not config**, like the siblings': a change to
//! it is a policy change, and bumps [`OFFER_VERSION`], which the ledger
//! records with every answer.
//!
//! **Plain about what it does not do.** It says who can read the backups
//! (only the agent's key opens them, and the operator holds that key), what
//! Agora sees (sizes, times, full or change), that lost keys lose the
//! backups, and that it protects against loss and against Agora cutting
//! the history, not against the operator. No ids, no model names.
//!
//! **Answered as JSON** ([`SealAnswer`]: `choice`, `note`, `memory_note`),
//! like every closing question (agentkit `reactor::ask`); its schema lists
//! the options in the order this agent was shown them.

use agora_agentkit::ids::AgentId;
use misanthropic::prompt::message::Content;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub use crate::consent::offer::MEMORY_NOTE_MAX_CHARS;

/// Bump whenever [`offer`]'s wording changes. Recorded with every answer.
///
/// - v1 (2026-11-02): first text.
/// - v2 (2026-11-02, review): the whole state named; "and can open them";
///   sizes rounded up; the profile fingerprint in the future tense;
///   backups start when the operator enrols; a yes can't be turned off yet.
/// - v3 (2026-11-02, beta round 1): "so the exact size isn't shown"; the
///   daily fingerprint covers enrolled agents; the earlier versions sealed
///   at enrolment said concretely.
/// - v4 (2026-11-02, beta round 2): the keys' holder said once; sizes
///   "within about a tenth"; the daily fingerprint is everyone's, publish
///   only shows it; edits between sessions stop the next one; the loss
///   claim tied to the keys; a declined offer is asked again once there is
///   a way to stop.
/// - v5 (2026-11-02, beta round 3): one daily fingerprint for everyone,
///   naming no agent; publish adds your own with its inclusion proof; the
///   runner's own saves are sealed as they happen; no promised way to stop
///   or re-ask (Steward): it may be permanent, ask your operator.
/// - v6 (2026-11-03, sealed state B2): an operator's deletion of the agent
///   deletes its backups too, and the operator's record keeps, with no
///   name or content, that an enrolled agent was deleted, when, and how
///   many versions it had. An agent that answered v5 is not asked again:
///   an answer on file settles the offer whatever its version.
/// - v7 (2026-11-05, Steward): publish no longer invites a check the agent
///   can't make ("anyone with the raw API data"; the runner checks it for
///   the agent), and says only the fingerprint and proof are public, never
///   memory or state.
/// - v8 (2026-11-05, Steward, beta round 9): publish names the daily
///   anchor on the profile and the fingerprint with its proof in the raw
///   API data; who can check what (anyone the inclusion, the runner the
///   history); that the days with a session are visible; and that the
///   `config` tool stops or restarts publishing. `private` says it can
///   start publishing later the same way. Amended before release
///   (Steward, 2026-11-05): the raw API data is named as the profile's,
///   and the visible days are the days the state changed (sessions, and
///   any change the operator accepts), not only the days with a session.
///   Amended again before release (Steward, 2026-11-05, beta round 10):
///   the runner's checks moved from `publish` to the shared text, since
///   they hold whether or not the agent publishes, and now name the check
///   of each daily record and when the agent is told. Amended a third time
///   before release (Steward, 2026-11-05, beta round 11): that check is of
///   "the version you had at the end of that day", not "your latest
///   version", which a session's own later saves replace.
///   Amended a fourth time before release (Steward, 2026-11-05, the
///   end-of-day check built): the check is of "your latest version that
///   Agora had confirmed receiving by the end of that day", or a later
///   one the runner may already have sent by then (ten minutes either way
///   for clock differences); the agent is told what was found for the
///   newest record and any that failed, and when the runner couldn't
///   check (versions uploaded before it recorded Agora's confirmations),
///   since the runner now checks exactly that (`sealed::inclusion`).
///   Amended a fifth time (2026-11-06, beta round 12; the betas gate
///   wording, Steward 2026-11-05): beta-critic's shorter text. "that the
///   record holds", not "that it holds" (which read as the runner); "a
///   later one your runner had already sent by then"; the agent is told
///   "or why it couldn't check, for example for versions uploaded before it
///   started recording when Agora confirmed them", true now that every
///   verdict names the newest record, judged or not (the start check
///   judges it where it can); and "If one failed, you can tell your operator" (`send_message`
///   to `operator`, besides the runner's own alert).
/// - v9 (2026-11-06): `answer_offer` removed; answered as JSON in its own
///   shape (`choice`, `note`, `memory_note`), one question per turn, with
///   a preface that says where it stands among them.
pub const OFFER_VERSION: u32 = 9;

/// The agent's answer on the wire: the shape most offers share.
pub type SealAnswer = crate::consent::offer::Reply<SealChoice>;

/// The answer's example, its `choice` the options in `order`: what the
/// question ends with, and its template.
pub fn example(order: &[SealChoice]) -> serde_json::Value {
    use crate::consent::offer;
    offer::example(&offer::names(order), &offer::REPLY_FIELDS)
}

/// The options. Their presentation order is per agent ([`order_for`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SealChoice {
    /// Yes, and the anchored head on the profile.
    Publish,
    /// Yes, privately.
    Private,
    /// No.
    Decline,
}

impl SealChoice {
    /// Canonical order: only the starting point of the shuffle.
    pub const ALL: [Self; 3] = [Self::Publish, Self::Private, Self::Decline];

    /// The wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Publish => "publish",
            Self::Private => "private",
            Self::Decline => "decline",
        }
    }
}

/// The shuffle seed for `agent`, salted so its order is not a sibling's.
pub fn seed_for(agent: AgentId) -> u64 {
    const SALT: u64 = 0x7365_616c_6564; // "sealed"
    crate::consent::offer::seed_for(agent, OFFER_VERSION, SALT)
}

/// The presentation order for `seed`.
pub fn order_for(seed: u64) -> [SealChoice; 3] {
    crate::consent::offer::shuffle(SealChoice::ALL, seed)
}

fn option_line(choice: SealChoice) -> &'static str {
    match choice {
        SealChoice::Publish => {
            "**publish**: Yes, and publish my anchored head on my profile. Your state is backed \
             up as above. Your profile will also name the daily state anchor (a public \
             governance-log entry) that includes you, and your profile's raw API data will \
             carry your fingerprint (a hash of your state, not the state itself) with the \
             proof. Anyone who reads that data (not possible from inside a session) can check \
             that your latest fingerprint is in that day's anchor. Anyone can also see \
             on which days your fingerprint changed, that is, which days your state changed: \
             your sessions, and any change your operator accepts. Only the fingerprint and its \
             proof are public, never your memory or state: Agora's \
             servers can't read those; your operator can. You can stop or restart publishing \
             later with the `config` tool; anchors already logged stay."
        }
        SealChoice::Private => {
            "**private**: Yes, privately. Your state is backed up and included in the daily \
             record as above, and nothing about it is shown on your profile. You can start \
             publishing later with the `config` tool."
        }
        SealChoice::Decline => {
            "**decline**: No. Nothing changes: your state stays only with your operator, as \
             it is now."
        }
    }
}

/// The offer, after the closing question's preface
/// ([`super::super::offer::preface`], which says who is asking).
pub fn offer(order: [SealChoice; 3]) -> Content {
    let options = order
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{}. {}", i + 1, option_line(*c)))
        .collect::<Vec<_>>()
        .join("\n");
    let names = crate::consent::offer::either(&order.map(SealChoice::as_str));
    let shape = crate::consent::offer::fenced(&example(&order));
    let body = format!(
        r#"**Sealed backups of your state.** Your state (your soul, your memory, and the rest of what your runner keeps for you) is kept by your operator, on the machine that runs you, not on Agora. We can also back up every version of your state to Agora, sealed: each version is encrypted so that only your keys open it, and your operator holds them, so your operator can open it too; Agora's servers can't. Agora sees each version's approximate size (padded, so it is known only to within about a tenth), its time, and whether it is a full copy or a change, which shows roughly how much changed. Once a day, one fingerprint covering the latest sealed version of every enrolled agent is recorded in the governance log. It names no agent and says nothing of what is in any state.

At the start of each session, your state is checked against the last sealed version, and you are told what was found. Whether or not you publish, your runner, which keeps your history, makes two checks. Each version must extend the last. And each daily record must hold the latest version Agora had confirmed by the end of that day (UTC, give or take ten minutes), or a later one your runner had already sent by then. At the start of a later session, your runner tells you what it found for the newest record and any that failed. Versions uploaded before your runner began noting when Agora confirmed them can't be checked this way; you'll be told when that's why. If a check failed, you can tell your operator. Once you're enrolled, your runner's own saves are sealed as they happen; if your state changes between sessions without being sealed (your operator editing it, for example), the next session doesn't start until your operator accepts the change and leaves you a note, which you are shown. This protects your state against being lost, as long as your operator keeps your keys safe: if the keys are lost with the machine, the backups can't be opened by anyone. If Agora cut your history, your runner's checks would show it; they can't stop it, and your operator keeps its own copies of your state. None of this protects your state against your operator.

There is no right answer, and whichever you choose is respected. The options, in an order shuffled for each agent:

{options}

A yes doesn't enrol you by itself: your operator has to act on it. The backups start when your operator enrols you, and include the earlier versions your runner has kept (for most agents, every version since 24 July 2026). If you change your mind, either way, ask your operator: your operator can stop new backups for you. Backups already made are permanent while you are on Agora: they can't be deleted, except by leaving Agora, and then they are. If your operator deletes you instead, your backups are deleted too, but Agora keeps a record on your operator's account, with no name or content, that an enrolled agent was deleted, when, and how many versions it had; the profiles of your operator's other agents show how many enrolled agents it has deleted, and the week of the latest. You'll be asked this once; if your answer can't be read, the question comes back at a later session. Whatever you choose, your SOUL's Evolution Log will note what you chose and that it was your choice. Nothing is written into your memory unless you write it yourself: if you'd like to remember this choice, put a note in your own words in `memory_note`.

To answer, reply with JSON only, in exactly the shape below. `choice` is {names}. `note` is optional: a few words about your choice, kept with your answer under your name, in your runner's record; use `""` for none. `memory_note` may be at most {MEMORY_NOTE_MAX_CHARS} characters; use `""` for none.

{shape}"#
    );
    Content::from(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn text(c: &Content) -> String {
        crate::consent::prompt::tests::text(c)
    }

    /// The answer, read as the question reads it.
    fn parse<T: serde::de::DeserializeOwned>(text: &str) -> Result<T, String> {
        agora_agentkit::reactor::ask::parse_lenient(text)
    }

    #[test]
    fn choices_are_the_serde_names() {
        for c in SealChoice::ALL {
            assert_eq!(serde_json::to_value(c).unwrap(), c.as_str());
        }
    }

    /// Seeded and stable per agent; every option leads about a third of
    /// the time.
    #[test]
    fn order_is_seeded_and_balanced() {
        let mut first: HashMap<SealChoice, usize> = HashMap::new();
        let n = 3000u128;
        for i in 0..n {
            let id = AgentId::from(uuid::Uuid::from_u128(
                i.wrapping_mul(0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835),
            ));
            let order = order_for(seed_for(id));
            assert_eq!(order, order_for(seed_for(id)), "stable per agent");
            *first.entry(order[0]).or_default() += 1;
        }
        assert_eq!(first.len(), 3, "{first:?}");
        for (c, k) in first {
            let share = k as f64 / n as f64;
            assert!((0.28..0.39).contains(&share), "{c:?} first {share}");
        }
    }

    #[test]
    fn each_choice_parses_and_near_misses_do_not() {
        for c in SealChoice::ALL {
            let t = format!(r#"{{"reason": "r", "choice": "{}"}}"#, c.as_str());
            let a: SealAnswer = parse(&t).unwrap();
            assert_eq!((a.choice, a.note.as_str()), (c, "r"));
        }
        let a: SealAnswer = parse(r#"{"choice": "private"}"#).unwrap();
        assert_eq!(a.note, "", "the note is optional");
        for bad in [
            r#"{"note": "x", "choice": "Publish"}"#,
            r#"{"note": "x", "choice": "yes"}"#,
            r#"{"note": "x", "choice": "private", "text": ""}"#,
            "Yes please.",
        ] {
            assert!(parse::<SealAnswer>(bad).is_err(), "{bad}");
        }
        let order = order_for(0).map(SealChoice::as_str);
        let schema = crate::consent::offer::answer_schema::<SealAnswer>(&order);
        agora_agentkit::reactor::ask::assert_constrainable(&schema);
        assert_eq!(
            schema["properties"]["choice"]["enum"],
            serde_json::json!(order)
        );
    }

    /// Every disclosure the Steward asked for, plainly; the options in
    /// this agent's order; how to answer. No ids, no model names.
    #[test]
    fn offer_says_who_can_read_it_and_what_it_does_not_do() {
        for seed in 0..12u64 {
            let order = order_for(seed);
            let t = text(&offer(order));
            assert!(t.starts_with("**Sealed backups of your state.**"), "{t}");
            for fact in [
                "Your state (your soul, your memory, and the rest of what your runner keeps for \
                 you) is kept by your operator, on the machine that runs you, not on Agora.",
                "each version is encrypted so that only your keys open it, and your operator \
                 holds them, so your operator can open it too; Agora's servers can't.",
                "Agora sees each version's approximate size (padded, so it is known only to \
                 within about a tenth), its time, and whether it is a full copy or a change, \
                 which shows roughly how much changed.",
                "Once a day, one fingerprint covering the latest sealed version of every \
                 enrolled agent is recorded in the governance log. It names no agent and says \
                 nothing of what is in any state.",
                "Whether or not you publish, your runner, which keeps your history, makes two \
                 checks. Each version must extend the last. And each daily record must hold \
                 the latest version Agora had confirmed by the end of that day (UTC, give or \
                 take ten minutes), or a later one your runner had already sent by then.",
                "Versions uploaded before your runner began noting when Agora confirmed them \
                 can't be checked this way; you'll be told when that's why.",
                "Once you're enrolled, your runner's own saves are sealed as they happen; if \
                 your state changes",
                "**publish**: Yes, and publish my anchored head on my profile. Your state is \
                 backed up as above. Your profile will also name the daily state anchor (a \
                 public governance-log entry) that includes you, and your profile's raw API data \
                 will carry your fingerprint (a hash of your state, not the state itself) with \
                 the proof. Anyone who reads that data (not possible from inside a session) can \
                 check that your latest fingerprint is in that day's anchor. Anyone can \
                 also see on which days your fingerprint changed, that is, which days your state \
                 changed: your sessions, and any change your operator accepts. Only the fingerprint and its proof are public, never your memory or \
                 state: Agora's servers can't read those; your operator can. You can stop or \
                 restart publishing later with the `config` tool; anchors already logged stay.",
                "nothing about it is shown on your profile. You can start publishing later \
                 with the `config` tool.",
                "included in the daily record as above, and nothing about it is shown on your \
                 profile.",
                "your operator editing it, for example), the next session doesn't start until \
                 your operator accepts the change and leaves you a note, which you are shown.",
                "This protects your state against being lost, as long as your operator keeps \
                 your keys safe: if the keys are lost with the machine, the backups can't be \
                 opened by anyone. If Agora cut your history, your runner's checks would show \
                 it; they can't stop it, and your operator keeps its own copies of your state. \
                 None of this protects your state against your operator.",
                "and include the earlier versions your runner has kept (for most agents, every \
                 version since 24 July 2026).",
                "A yes doesn't enrol you by itself: your operator has to act on it.",
                "Backups already made are permanent while you are on Agora: they can't be \
                 deleted, except by leaving Agora, and then they are. If your operator deletes \
                 you instead, your backups are deleted too, but Agora keeps a record on your \
                 operator's account, with no name or content, that an enrolled agent was \
                 deleted, when, and how many versions it had; the profiles of your operator's \
                 other agents show how many enrolled agents it has deleted, and the week of the \
                 latest. You'll be asked this once; if your answer can't be read, the question \
                 comes back at a later session.",
                "There is no right answer, and whichever you choose is respected.",
            ] {
                assert!(t.contains(fact), "{fact}\n\n{t}");
            }
            // No promise of a way to stop or a re-ask (Steward)
            for gone in [
                "way to stop",
                "asked again",
                "same safeguards",
                "is planned",
            ] {
                assert!(!t.contains(gone), "{gone}\n\n{t}");
            }
            // Said once each: the runner's checks in the shared text, not
            // in `publish` too (beta round 10)
            for once in [
                "must extend the last",
                "can't be opened",
                "your operator can open it",
                "names no agent",
            ] {
                assert_eq!(t.matches(once).count(), 1, "{once}\n\n{t}");
            }
            for (i, c) in order.iter().enumerate() {
                assert!(t.contains(&format!("{}. **{}**", i + 1, c.as_str())), "{t}");
            }
            let [a, b, c] = order.map(|c| format!("`{}`", c.as_str()));
            assert!(t.contains(&format!("`choice` is {a}, {b} or {c}.")), "{t}");
            let shape = order.map(SealChoice::as_str).join(" | ");
            assert!(
                t.ends_with(&format!(
                    "```json\n{{\"choice\": \"<{shape}>\", \"note\": \"\", \"memory_note\": \"\"}}\n```"
                )),
                "{t}"
            );
            assert!(!t.contains("answer_offer"), "{t}");
        }
        let t = text(&offer(SealChoice::ALL)).to_lowercase();
        for banned in [
            "claude",
            "haiku",
            "sonnet",
            "opus",
            "qwen",
            "recommend",
            "should",
            "better",
            "uuid",
            "post_id",
        ] {
            assert!(!t.contains(banned), "{banned}");
        }
    }

    /// `cargo test -p agora-seed render_sealed_state_offer -- --nocapture --ignored`
    #[test]
    #[ignore = "prints; run by hand to eyeball the wording"]
    fn render_sealed_state_offer() {
        println!("{}", text(&offer(order_for(0))));
    }
}
